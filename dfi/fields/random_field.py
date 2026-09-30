"""Randomly generated drift fields, in two interchangeable backends.

``GridDriftField``  (d = 1, 2, 3)
    The potential lives on a periodic grid, is differentiated *spectrally*
    (exactly, because the field is band-limited), and is evaluated at arbitrary
    walker positions by quintic B-spline interpolation. This is the backend you
    plot and animate.

``SpectralDriftField``  (any d, including 4-10)
    The potential is a finite random-Fourier-feature sum,
    ``U(x) = sum_m a_m cos(w_m . x + phi_m)``, which is a sample from a
    stationary Gaussian random field with a kernel set by how ``w`` is drawn.
    No grid, so no ``n^d`` blow-up; ``U``, ``grad U`` and ``lap U`` are all
    analytic, and it runs on the GPU. This is the backend for the ``d``-sweep
    in the phase diagram.

Both build the drift the same way::

    b(x) = -grad U(x) + omega * A grad U(x)

with ``A`` constant antisymmetric, which keeps the Helmholtz split exact and
the Boltzmann stationary density exact for every ``omega`` (see ``base.py`` for
the two-line proof). ``omega`` reads directly as the rotational-to-gradient
amplitude ratio: ``omega=0`` is equilibrium, ``omega=1`` puts equal power in
the rotational component.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import DriftField, random_antisymmetric
from .domain import Box
from .noise import (gaussian_random_field, perlin_fbm, spectral_gradient,
                    spectral_laplacian)

__all__ = ["GridDriftField", "SpectralDriftField", "random_grid_field",
           "random_spectral_field"]


# --------------------------------------------------------------------------
# Grid backend
# --------------------------------------------------------------------------

class _SplineInterpolator:
    """Quintic B-spline interpolation of a gridded scalar, vectorised.

    Prefiltering is done once at construction, so each query is a single
    ``map_coordinates`` call -- O(1) per point and comfortable with a million
    walkers at a time.

    Order 5 (quintic), not the usual cubic. The accuracy that matters here is
    not on ``U`` but on its *derivative*, and the interpolation error at a
    wavenumber ``k`` grows like ``(k h)^(order+1)``. Measured on this project's
    fields, at a band limit of 0.25 Nyquist: cubic gives ~5e-4 relative error
    on the drift, quintic ~7e-6. Since the whole point is to benchmark
    estimators whose errors are percent-level, the ground truth has to be
    several orders of magnitude tighter than that, and quintic is free (one
    extra prefilter pass at construction).
    """

    def __init__(self, values: np.ndarray, box: Box, order: int = 5):
        self.box = box
        self.shape = values.shape
        self.order = order
        self.mode = "grid-wrap" if box.periodic else "nearest"
        self.coef = ndimage.spline_filter(values, order=order, mode=self.mode,
                                          output=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        coords = self.box.to_index(x, self.shape)
        flat = coords.reshape(self.box.d, -1)
        out = ndimage.map_coordinates(self.coef, flat, order=self.order,
                                      mode=self.mode, prefilter=False)
        return out.reshape(x.shape[:-1])


class GridDriftField(DriftField):
    """Drift built from a potential sampled on a periodic grid.

    Parameters
    ----------
    potential_grid:
        The scalar ``U`` on a grid of shape ``(n,)*d``.
    box:
        Domain. Periodic is strongly preferred here: it matches the FFT that
        generated the field, keeps walkers in the sampled region, and removes
        the empty-tail-bin pathology from the estimator comparison.
    omega:
        Rotational-to-gradient amplitude ratio (0 = equilibrium).
    A:
        Constant antisymmetric matrix; defaults to the canonical rotation.
    """

    def __init__(self, potential_grid, box: Box, *, omega: float = 0.0,
                 A: np.ndarray | None = None, name: str | None = None,
                 rng: np.random.Generator | None = None, meta: dict | None = None):
        super().__init__(box, name)
        self.U_grid = np.ascontiguousarray(potential_grid, dtype=float)
        if self.U_grid.ndim != box.d:
            raise ValueError(
                f"potential grid has {self.U_grid.ndim} axes but the box is "
                f"{box.d}-dimensional")
        self.omega = float(omega)
        rng = rng or np.random.default_rng(0)
        self.A = random_antisymmetric(box.d, rng) if A is None else np.asarray(A, float)
        self.meta = dict(meta or {})

        # Exact derivatives on the grid, then interpolate those. For a
        # band-limited field the two operations commute to ~1e-5 relative, so
        # "gradient of the interpolant" and "interpolant of the gradient"
        # agree and the ground truth is self-consistent.
        self._grad_grid = spectral_gradient(self.U_grid, box)
        self._lap_grid = spectral_laplacian(self.U_grid, box)

        self._U = _SplineInterpolator(self.U_grid, box)
        self._dU = [_SplineInterpolator(self._grad_grid[j], box) for j in range(box.d)]
        self._lap = _SplineInterpolator(self._lap_grid, box)

    # -- evaluation --------------------------------------------------------
    def potential(self, x):
        return self._U(x)

    def grad_potential(self, x):
        x = np.asarray(x, float)
        return np.stack([f(x) for f in self._dU], axis=-1)

    def drift(self, x):
        g = self.grad_potential(x)
        b = -g
        if self.omega != 0.0:
            b = b + self.omega * (g @ self.A.T)
        return b

    def rotational_part(self, x):
        if self.omega == 0.0:
            return np.zeros_like(np.asarray(x, float))
        return self.omega * (self.grad_potential(x) @ self.A.T)

    def divergence(self, x, eps=1e-5):
        # div b = -lap U exactly: the rotational term is divergence-free by
        # construction, so it contributes nothing.
        return -self._lap(x)

    # -- structure ---------------------------------------------------------
    @property
    def is_gradient(self):
        return self.omega == 0.0

    @property
    def has_potential(self):
        return True

    @property
    def boltzmann_exact(self):
        return True  # holds for every omega, by the antisymmetry argument

    def with_omega(self, omega: float) -> "GridDriftField":
        """The same potential, a different rotational strength.

        This is the constructor the identifiability experiment must use.
        ``random_grid_field`` renormalises ``U`` so that ``RMS|b| = 1``, which
        means two independently generated fields with different ``omega`` have
        *different* potentials and therefore different stationary densities --
        which would confound the very effect being measured. Going through
        ``with_omega`` holds ``U`` (hence ``rho_ss``) bit-identical and changes
        only the dynamics, so any difference an estimator sees is attributable
        to the rotational component alone.

        Note that ``RMS|b|`` does grow like ``sqrt(1 + omega^2)``: adding
        circulation genuinely speeds the walkers up. That is physics, and it is
        why errors should be reported relative to each field's own
        :meth:`typical_scale`.
        """
        out = GridDriftField(self.U_grid, self.box, omega=omega, A=self.A,
                             name=self.name, meta=self.meta)
        # Reuse the prefiltered interpolators: same U, so same splines.
        out._grad_grid, out._lap_grid = self._grad_grid, self._lap_grid
        out._U, out._dU, out._lap = self._U, self._dU, self._lap
        return out

    # -- gridded quantities, for plotting ---------------------------------
    def drift_on_grid(self) -> np.ndarray:
        """``b`` on the native grid, shape ``(d, *grid)`` -- no interpolation."""
        g = self._grad_grid
        b = -g
        if self.omega != 0.0:
            b = b + self.omega * np.tensordot(self.A, g, axes=([1], [0]))
        return b

    def rho_ss_on_grid(self, D: float) -> np.ndarray:
        """Normalised stationary density on the native grid.

        Exact (not simulated), because the Boltzmann form is exact here.
        """
        logp = -self.U_grid / D
        p = np.exp(logp - logp.max())
        return p / (p.sum() * np.prod(self.box.spacing(self.U_grid.shape)))

    def entropy_production(self, D: float) -> float:
        """Steady-state entropy production rate ``sigma = <|b_rot|^2> / D``.

        Evaluated by quadrature against the exact stationary density, so it is
        a ground-truth number to compare an estimate against (the extra-credit
        item in the guide).
        """
        if self.omega == 0.0:
            return 0.0
        rho = self.rho_ss_on_grid(D)
        rot = self.omega * np.tensordot(self.A, self._grad_grid, axes=([1], [0]))
        sq = np.sum(rot ** 2, axis=0)
        dv = float(np.prod(self.box.spacing(self.U_grid.shape)))
        return float(np.sum(sq * rho) * dv / D)

    # -- persistence -------------------------------------------------------
    def save(self, path):
        np.savez_compressed(
            path, U=self.U_grid, lo=self.box.lo, hi=self.box.hi,
            periodic=self.box.periodic, omega=self.omega, A=self.A,
            name=self.name, meta=np.array(repr(self.meta)))

    @classmethod
    def load(cls, path) -> "GridDriftField":
        z = np.load(path, allow_pickle=False)
        box = Box(z["lo"], z["hi"], bool(z["periodic"]))
        return cls(z["U"], box, omega=float(z["omega"]), A=z["A"],
                   name=str(z["name"]))

    def summary(self):
        base = super().summary()
        res = "x".join(str(s) for s in self.U_grid.shape)
        return f"{base} | grid {res} | omega={self.omega:g}"


def random_grid_field(
    d: int = 2,
    resolution: int = 256,
    *,
    box: Box | None = None,
    omega: float = 0.0,
    generator: str = "spectral",
    seed: int | None = None,
    drift_scale: float = 1.0,
    A: np.ndarray | None = None,
    **kwargs,
) -> GridDriftField:
    """Draw a random drift field on a grid.

    Parameters
    ----------
    d, resolution:
        Dimension and grid points per axis. ``resolution**d`` cells, so keep
        ``d <= 3``; above that use :func:`random_spectral_field`.
    box:
        Defaults to the periodic cube ``[-1, 1]^d``.
    omega:
        Rotational-to-gradient ratio. ``0`` is equilibrium; sweep it upward for
        the identifiability experiment.
    generator:
        ``'spectral'`` (Gaussian random field, default) or ``'perlin'``.
    drift_scale:
        The potential is rescaled so the RMS drift magnitude equals this. It
        fixes the physical units of ``b``, which makes ``D``, ``dt`` and the
        reported errors comparable across fields with different roughness.
    **kwargs:
        Forwarded to the generator (``spectrum``, ``correlation_length``,
        ``slope``, ``cutoff``, or ``base_res``/``octaves``/``gain``). The
        default ``spectrum='gaussian'`` gives a field whose gradient scale is
        genuinely ``correlation_length``; see :func:`gaussian_random_field`
        for why that matters.
    """
    rng = np.random.default_rng(seed)
    box = box or Box.cube(d, 1.0, periodic=True)
    if box.d != d:
        raise ValueError(f"box is {box.d}-dimensional but d={d} was requested")
    shape = (resolution,) * d

    if generator == "spectral":
        U = gaussian_random_field(box, shape, rng, **kwargs)
    elif generator == "perlin":
        U = perlin_fbm(box, shape, rng, **kwargs)
    else:
        raise ValueError(
            f"unknown generator {generator!r}; expected 'spectral' or 'perlin'")

    # Normalise so RMS|b| == drift_scale. |b|^2 = (1 + omega^2 |A g|^2/|g|^2)|g|^2,
    # and A is normalised so that ratio is ~1, hence the sqrt(1 + omega^2).
    g = spectral_gradient(U, box)
    rms_grad = float(np.sqrt(np.mean(np.sum(g ** 2, axis=0))))
    U = U * (drift_scale / (rms_grad * np.sqrt(1.0 + omega ** 2)))

    meta = {"generator": generator, "seed": seed, "resolution": resolution,
            "drift_scale": drift_scale, **kwargs}
    nice = {"spectral": "grf", "perlin": "perlin"}[generator]
    return GridDriftField(U, box, omega=omega, A=A, rng=rng,
                          name=f"random-{nice}-{d}D", meta=meta)


# --------------------------------------------------------------------------
# Spectral (mesh-free) backend -- any dimension
# --------------------------------------------------------------------------

class SpectralDriftField(DriftField):
    """Random-Fourier-feature potential: analytic, mesh-free, any dimension.

    ``U(x) = sqrt(2/M) sum_m a_m cos(w_m . x + phi_m)``

    Drawing ``w`` from a density makes this a sample from a stationary Gaussian
    random field whose covariance kernel is that density's Fourier transform
    (Bochner's theorem) -- Gaussian ``w`` gives an RBF kernel, Student-t ``w``
    gives Matern. Restricting ``w`` to the reciprocal lattice ``2 pi n / L``
    keeps the field exactly periodic on the torus, so it is the same object as
    the grid backend, just evaluated without a grid.
    """

    def __init__(self, w: np.ndarray, phase: np.ndarray, amp: np.ndarray,
                 box: Box, *, omega: float = 0.0, A: np.ndarray | None = None,
                 name: str | None = None, rng: np.random.Generator | None = None,
                 meta: dict | None = None):
        super().__init__(box, name)
        self.w = np.asarray(w, float)          # (M, d)
        self.phase = np.asarray(phase, float)  # (M,)
        self.amp = np.asarray(amp, float)      # (M,)
        self.omega = float(omega)
        rng = rng or np.random.default_rng(0)
        self.A = random_antisymmetric(box.d, rng) if A is None else np.asarray(A, float)
        self.meta = dict(meta or {})

    @property
    def n_features(self) -> int:
        return self.w.shape[0]

    # -- evaluation --------------------------------------------------------
    def potential(self, x):
        x = np.asarray(x, float)
        theta = x @ self.w.T + self.phase
        return np.cos(theta) @ self.amp

    def grad_potential(self, x):
        x = np.asarray(x, float)
        theta = x @ self.w.T + self.phase
        return (-np.sin(theta) * self.amp) @ self.w

    def laplacian_potential(self, x):
        x = np.asarray(x, float)
        theta = x @ self.w.T + self.phase
        return -(np.cos(theta) * self.amp) @ np.sum(self.w ** 2, axis=1)

    def drift(self, x):
        g = self.grad_potential(x)
        b = -g
        if self.omega != 0.0:
            b = b + self.omega * (g @ self.A.T)
        return b

    def rotational_part(self, x):
        if self.omega == 0.0:
            return np.zeros_like(np.asarray(x, float))
        return self.omega * (self.grad_potential(x) @ self.A.T)

    def divergence(self, x, eps=1e-5):
        return -self.laplacian_potential(x)

    @property
    def is_gradient(self):
        return self.omega == 0.0

    @property
    def has_potential(self):
        return True

    @property
    def boltzmann_exact(self):
        return True

    # -- torch path --------------------------------------------------------
    @property
    def supports_torch(self):
        return True

    def drift_torch(self, x):
        import torch
        w = torch.as_tensor(self.w, dtype=x.dtype, device=x.device)
        ph = torch.as_tensor(self.phase, dtype=x.dtype, device=x.device)
        am = torch.as_tensor(self.amp, dtype=x.dtype, device=x.device)
        theta = x @ w.T + ph
        g = (-torch.sin(theta) * am) @ w
        b = -g
        if self.omega != 0.0:
            A = torch.as_tensor(self.A, dtype=x.dtype, device=x.device)
            b = b + self.omega * (g @ A.T)
        return b

    def with_omega(self, omega: float) -> "SpectralDriftField":
        """The same potential, a different rotational strength.

        See :meth:`GridDriftField.with_omega` -- holding ``U`` fixed while
        sweeping ``omega`` is what makes the identifiability comparison clean.
        """
        return SpectralDriftField(self.w, self.phase, self.amp, self.box,
                                  omega=omega, A=self.A, name=self.name,
                                  meta=self.meta)

    def save(self, path):
        np.savez_compressed(path, w=self.w, phase=self.phase, amp=self.amp,
                            lo=self.box.lo, hi=self.box.hi,
                            periodic=self.box.periodic, omega=self.omega,
                            A=self.A, name=self.name)

    @classmethod
    def load(cls, path) -> "SpectralDriftField":
        z = np.load(path, allow_pickle=False)
        box = Box(z["lo"], z["hi"], bool(z["periodic"]))
        return cls(z["w"], z["phase"], z["amp"], box, omega=float(z["omega"]),
                   A=z["A"], name=str(z["name"]))

    def summary(self):
        return f"{super().summary()} | {self.n_features} modes | omega={self.omega:g}"


def random_spectral_field(
    d: int = 4,
    n_features: int = 256,
    *,
    box: Box | None = None,
    omega: float = 0.0,
    correlation_length: float | None = None,
    kernel: str = "gaussian",
    nu: float = 2.5,
    seed: int | None = None,
    drift_scale: float = 1.0,
    A: np.ndarray | None = None,
) -> SpectralDriftField:
    """Draw a random drift field in any dimension, mesh-free.

    Parameters
    ----------
    d, n_features:
        Dimension and number of Fourier modes ``M``. More modes means a richer
        field; ``M`` around 256-1024 is plenty up to ``d = 10``.
    kernel:
        ``'gaussian'`` (RBF: very smooth) or ``'matern'`` (rougher, controlled
        by ``nu``).
    correlation_length:
        Feature size in domain units; defaults to a fifth of the box side.
    drift_scale:
        RMS drift magnitude, matched to the grid backend's convention so that
        results are comparable across backends.
    """
    rng = np.random.default_rng(seed)
    box = box or Box.cube(d, 1.0, periodic=True)
    if correlation_length is None:
        correlation_length = float(np.min(box.length)) / 5.0

    # Bochner sampling: the spectral density of the kernel is the law of w.
    w = rng.standard_normal((n_features, d)) / correlation_length
    if kernel == "matern":
        # Student-t with 2*nu dof = Gaussian scaled by an inverse chi.
        u = rng.chisquare(2.0 * nu, size=(n_features, 1)) / (2.0 * nu)
        w = w / np.sqrt(u)
    elif kernel != "gaussian":
        raise ValueError(
            f"unknown kernel {kernel!r}; expected 'gaussian' or 'matern'")

    if box.periodic:
        # Snap onto the reciprocal lattice so the field is exactly periodic.
        n = np.round(w * box.length / (2.0 * np.pi))
        dead = np.all(n == 0, axis=1)
        if np.any(dead):
            # A zero mode is a constant: replace it with the lowest live mode
            # rather than dropping features and silently changing M.
            n[dead, 0] = 1.0
        w = 2.0 * np.pi * n / box.length

    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    amp = rng.standard_normal(n_features) * np.sqrt(2.0 / n_features)

    field = SpectralDriftField(w, phase, amp, box, omega=omega, A=A, rng=rng,
                               name=f"random-spectral-{d}D",
                               meta={"kernel": kernel, "seed": seed,
                                     "correlation_length": correlation_length})

    # Match the grid backend's normalisation: RMS|b| == drift_scale.
    probe = box.sample_uniform(min(20000, 2000 * d), rng)
    rms = float(np.sqrt(np.mean(np.sum(field.grad_potential(probe) ** 2, axis=-1))))
    field.amp = field.amp * (drift_scale / (rms * np.sqrt(1.0 + omega ** 2)))
    return field
