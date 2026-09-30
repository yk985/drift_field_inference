"""Random *scalar* field generators on a periodic grid.

Everything in this project builds the drift out of scalar fields (a potential,
and optionally a stream function), never out of independent vector components.
That is what makes the Helmholtz decomposition exact and free -- see
``dfi/fields/random_field.py``. This module is the machinery for drawing those
scalars.

Two generators, same interface:

``gaussian_random_field``
    Draws a stationary Gaussian random field with a chosen power spectrum, by
    filtering white noise in Fourier space. This is the default. It is a few
    milliseconds at 512^2, exactly periodic, and -- because we band-limit it --
    can be differentiated *spectrally and exactly* on the grid.

``perlin_fbm``
    Classic gradient (Perlin) noise summed over octaves, tileable, for the
    organic blobby look. Available for ``d <= 3``.

Both return a zero-mean, unit-variance array of the grid shape.

Why band-limiting matters
-------------------------
Downstream we interpolate ``U`` and ``grad U`` with quintic B-splines to get the
field at arbitrary walker positions. For a band-limited field the interpolation
error on the drift is ~1e-5 relative and "gradient of the interpolant" agrees
with "interpolant of the gradient", so the ground truth is self-consistent. Without band-limiting, the
field has power at the grid scale, spectral differentiation rings, and the
"exact" drift you are trying to recover is itself uncertain at the percent
level -- which would quietly contaminate every estimator benchmark.
"""
from __future__ import annotations

import numpy as np

from .domain import Box

__all__ = ["gaussian_random_field", "perlin_fbm", "spectral_gradient",
           "spectral_laplacian", "lowpass", "wavenumbers"]


# --------------------------------------------------------------------------
# Fourier helpers
# --------------------------------------------------------------------------

def wavenumbers(box: Box, shape) -> tuple:
    """Angular wavenumber grids for ``rfftn`` output.

    Returns ``(k_list, k_mag)`` where ``k_list[j]`` broadcasts against the
    half-spectrum and ``k_mag`` is ``|k|``.
    """
    shape = box._as_shape(shape)
    h = box.spacing(shape)
    ks = []
    for j in range(box.d):
        if j == box.d - 1:
            f = np.fft.rfftfreq(shape[j], d=h[j])
        else:
            f = np.fft.fftfreq(shape[j], d=h[j])
        bshape = [1] * box.d
        bshape[j] = f.size
        ks.append((2.0 * np.pi * f).reshape(bshape))
    kmag = np.sqrt(sum(k ** 2 for k in ks))
    return ks, kmag


def _amplitude(kmag, spectrum, ell, slope, k_ring, ring_width):
    """sqrt of the power spectral density, as a function of ``|k|``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if spectrum == "gaussian":
            # One dominant scale, extremely smooth. P(k) = exp(-(k*ell)^2 / 2).
            amp = np.exp(-0.25 * (kmag * ell) ** 2)
        elif spectrum == "powerlaw":
            # Multi-scale / fBm-like. Flat below 1/ell, k^-slope above it.
            amp = (1.0 + (kmag * ell) ** 2) ** (-0.25 * slope)
        elif spectrum == "bandpass":
            # Power concentrated on a shell in k-space: cellular, wave-like.
            w = ring_width * k_ring
            amp = np.exp(-0.5 * ((kmag - k_ring) / w) ** 2)
        else:
            raise ValueError(
                f"unknown spectrum {spectrum!r}; expected one of "
                "'gaussian', 'powerlaw', 'bandpass'"
            )
    amp = np.nan_to_num(amp, nan=0.0, posinf=0.0)
    return amp


def _band_limit(kmag, box: Box, shape, cutoff: float):
    """Smooth low-pass at ``cutoff`` times the Nyquist wavenumber.

    A super-Gaussian roll-off: flat in the passband, then a fast but smooth
    decay, so we lose no visible structure and gain a genuinely band-limited
    field (no Gibbs ringing, unlike a hard cutoff).
    """
    if cutoff >= 1.0:
        return 1.0
    h = box.spacing(shape)
    k_nyq = np.pi / np.max(h)
    return np.exp(-np.log(2.0) * (kmag / (cutoff * k_nyq)) ** 8)


def lowpass(values: np.ndarray, box: Box, cutoff: float = 0.25) -> np.ndarray:
    """Band-limit an existing periodic field in place of generating one."""
    shape = values.shape
    _, kmag = wavenumbers(box, shape)
    axes = tuple(range(len(shape)))
    f = np.fft.rfftn(values, axes=axes)
    f *= _band_limit(kmag, box, shape, cutoff)
    return np.fft.irfftn(f, s=shape, axes=axes)


def spectral_gradient(values: np.ndarray, box: Box) -> np.ndarray:
    """Exact gradient of a band-limited periodic field, shape ``(d, *grid)``.

    For a field whose Fourier content lies below Nyquist this is not an
    approximation: differentiation is multiplication by ``i k``, with no
    truncation error at all. That is why the ground-truth drift here is exact
    to machine precision rather than to a finite-difference stencil.
    """
    shape = values.shape
    axes = tuple(range(len(shape)))
    ks, _ = wavenumbers(box, shape)
    f = np.fft.rfftn(values, axes=axes)
    out = np.empty((box.d,) + shape, dtype=float)
    for j in range(box.d):
        out[j] = np.fft.irfftn(1j * ks[j] * f, s=shape, axes=axes)
    return out


def spectral_laplacian(values: np.ndarray, box: Box) -> np.ndarray:
    """Exact Laplacian of a band-limited periodic field."""
    shape = values.shape
    axes = tuple(range(len(shape)))
    _, kmag = wavenumbers(box, shape)
    f = np.fft.rfftn(values, axes=axes)
    return np.fft.irfftn(-(kmag ** 2) * f, s=shape, axes=axes)


# --------------------------------------------------------------------------
# Generator 1: spectral Gaussian random field  (default)
# --------------------------------------------------------------------------

def gaussian_random_field(
    box: Box,
    shape,
    rng: np.random.Generator,
    *,
    spectrum: str = "gaussian",
    correlation_length: float | None = None,
    slope: float = 4.0,
    k_ring: float | None = None,
    ring_width: float = 0.25,
    cutoff: float = 0.25,
) -> np.ndarray:
    """Draw a stationary Gaussian random field on a periodic grid.

    Parameters
    ----------
    box, shape:
        Domain and grid resolution (an int, or a per-dimension tuple).
    rng:
        Source of randomness; pass a seeded generator for reproducibility.
    spectrum:
        ``'gaussian'`` (default), ``'powerlaw'`` (multi-scale / fBm-like) or
        ``'bandpass'`` (cellular, energy on a k-shell).

        The default is Gaussian for a reason worth knowing. With a power-law
        spectrum in 2D the variance of the *second* derivatives goes like
        ``int k^3 P(k) dk``, which for ``slope=4`` diverges logarithmically --
        so the field's curvature, and hence its gradient scale
        ``|b| / |grad b|``, is set by wherever the band limit happens to sit
        rather than by ``correlation_length``. Two bad consequences: the field
        carries real structure at scales no estimator could resolve, and every
        result depends silently on the grid resolution. A Gaussian spectrum
        cuts off exponentially, so all derivatives are governed by
        ``correlation_length`` and the field is resolution-independent. Reach
        for ``'powerlaw'`` when rough multi-scale terrain is the point, and
        then use ``slope >= 6`` in 2D.
    correlation_length:
        Feature size, in domain units. Defaults to a fifth of the shortest box
        side, which gives roughly 5 features across the domain.
    slope:
        Power-law exponent for ``spectrum='powerlaw'``. Larger is smoother:
        ``slope=2`` is rough and craggy, ``slope=6`` is gentle and rolling.
    k_ring, ring_width:
        Shell wavenumber and relative width for ``spectrum='bandpass'``.
        ``k_ring`` defaults to ``2*pi/correlation_length``.
    cutoff:
        Band-limit as a fraction of Nyquist. The default 0.25 is chosen so the
        quintic-spline interpolation of the drift is accurate to ~1e-5
        relative; raising it towards Nyquist buys visibly finer structure at
        the cost of ground-truth accuracy (0.55 costs ~3e-3). It still leaves
        ~n/8 features per axis, which is plenty of structure to look at.

    Returns
    -------
    ndarray of the grid shape, zero mean and unit variance.
    """
    shape = box._as_shape(shape)
    if correlation_length is None:
        correlation_length = float(np.min(box.length)) / 5.0
    if k_ring is None:
        k_ring = 2.0 * np.pi / correlation_length

    _, kmag = wavenumbers(box, shape)
    amp = _amplitude(kmag, spectrum, correlation_length, slope, k_ring, ring_width)
    amp = amp * _band_limit(kmag, box, shape, cutoff)
    amp = np.where(kmag == 0.0, 0.0, amp)  # zero mean

    # Filtering real white noise keeps the spectrum Hermitian for free, so the
    # inverse transform is real without any symmetry bookkeeping.
    axes = tuple(range(len(shape)))
    white = rng.standard_normal(shape)
    field = np.fft.irfftn(np.fft.rfftn(white, axes=axes) * amp, s=shape,
                          axes=axes)

    std = field.std()
    if std < 1e-14:
        raise RuntimeError(
            "generated field is numerically constant -- the spectrum removed "
            "all power. Check correlation_length against the box size and the "
            "grid resolution."
        )
    return (field - field.mean()) / std


# --------------------------------------------------------------------------
# Generator 2: Perlin / fractional Brownian motion
# --------------------------------------------------------------------------

def _fade(t):
    """Perlin's quintic ease curve: zero 1st and 2nd derivatives at 0 and 1.

    The quintic (not the older cubic) is what makes fBm noise C^2, which we
    need because the drift is a *derivative* of this field.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _perlin_octave(shape, res, rng, d):
    """One tileable Perlin octave on a grid of ``shape`` with ``res`` cells."""
    # Random unit gradient at each lattice node; wrap the last node onto the
    # first so the octave tiles seamlessly.
    g = rng.standard_normal(tuple(res) + (d,))
    g /= np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12

    # Continuous lattice coordinate of every output sample.
    coords = [np.linspace(0.0, res[j], shape[j], endpoint=False) for j in range(d)]
    mesh = np.meshgrid(*coords, indexing="ij")
    cell = [np.floor(m).astype(np.intp) for m in mesh]
    frac = [m - c for m, c in zip(mesh, cell)]
    fade = [_fade(f) for f in frac]

    total = np.zeros(shape)
    for corner in range(2 ** d):
        offs = [(corner >> j) & 1 for j in range(d)]
        idx = tuple((cell[j] + offs[j]) % res[j] for j in range(d))
        grad = g[idx]                                    # (*shape, d)
        dist = np.stack([frac[j] - offs[j] for j in range(d)], axis=-1)
        dot = np.sum(grad * dist, axis=-1)
        weight = np.ones(shape)
        for j in range(d):
            weight = weight * (fade[j] if offs[j] else 1.0 - fade[j])
        total += weight * dot
    return total


def perlin_fbm(
    box: Box,
    shape,
    rng: np.random.Generator,
    *,
    base_res: int = 4,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    cutoff: float = 0.25,
) -> np.ndarray:
    """Tileable fractal Perlin noise (``d <= 3``), zero mean and unit variance.

    ``base_res`` sets how many features span the box at the coarsest octave;
    each further octave doubles the frequency (``lacunarity``) and scales the
    amplitude by ``gain``. Small ``gain`` gives a smooth rolling landscape,
    ``gain=0.5`` the classic fBm look.

    The result is low-passed like the spectral generator, so it can be
    differentiated spectrally with the same accuracy.
    """
    shape = box._as_shape(shape)
    d = box.d
    if d > 3:
        raise ValueError(
            f"perlin_fbm supports d <= 3 (got d={d}); the 2^d corner loop and "
            "the n^d grid both blow up. Use the spectral backend "
            "(SpectralDriftField) for higher dimensions."
        )

    total = np.zeros(shape)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        res = [max(1, int(round(base_res * lacunarity ** o))) for _ in range(d)]
        if any(r > s for r, s in zip(res, shape)):
            break  # octave finer than the grid: nothing left to resolve
        total += amp * _perlin_octave(shape, res, rng, d)
        norm += amp
        amp *= gain
    if norm == 0.0:
        raise ValueError("no octave fit in the grid; raise the resolution or "
                         "lower base_res")

    total = lowpass(total, box, cutoff)
    return (total - total.mean()) / total.std()
