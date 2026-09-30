"""Line integral convolution -- the texture that makes a flow field readable.

A quiver plot of a random field is a thicket of arrows; a streamplot picks a
few arbitrary seed points and hides everything between them. LIC instead
smears a field of white noise *along* the flow, so every pixel shows the local
flow direction as a visible grain. The eye reads it as a continuous texture and
picks up stagnation points, saddles and vortices immediately.

The algorithm: for each pixel, walk forward and backward along the streamline
through it for a fixed arc length, averaging the noise you pass over. Pixels on
the same streamline average nearly the same samples and end up correlated;
pixels on neighbouring streamlines do not. The result is grain that is
elongated along the flow.

Arrays here are indexed ``[ix, iy]`` to match the field convention in
``dfi.fields``; :func:`dfi.viz.fields.to_image` transposes for display.
"""
from __future__ import annotations

import numpy as np

try:  # numba turns a ~30 s pure-python loop into ~0.2 s
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a else a[0]

    prange = range


@njit(cache=True, parallel=True, fastmath=True)
def _lic_kernel(vx, vy, noise, n_steps, step_len, periodic):
    nx, ny = noise.shape
    out = np.zeros((nx, ny), dtype=np.float64)
    for i in prange(nx):
        for j in range(ny):
            acc = noise[i, j]
            wsum = 1.0
            for direction in range(2):
                sgn = 1.0 if direction == 0 else -1.0
                px = i + 0.5
                py = j + 0.5
                for s in range(n_steps):
                    # nearest-neighbour velocity lookup (the texture is noise
                    # anyway; bilinear here buys nothing visible and costs 4x)
                    ix = int(px)
                    iy = int(py)
                    if periodic:
                        ix = ix % nx
                        iy = iy % ny
                    else:
                        if ix < 0:
                            ix = 0
                        elif ix >= nx:
                            ix = nx - 1
                        if iy < 0:
                            iy = 0
                        elif iy >= ny:
                            iy = ny - 1
                    ux = vx[ix, iy]
                    uy = vy[ix, iy]
                    mag = np.sqrt(ux * ux + uy * uy)
                    if mag < 1e-12:
                        break
                    px += sgn * step_len * ux / mag
                    py += sgn * step_len * uy / mag
                    if periodic:
                        px = px % nx
                        py = py % ny
                    elif px < 0 or px >= nx or py < 0 or py >= ny:
                        break
                    ix = int(px) % nx if periodic else int(px)
                    iy = int(py) % ny if periodic else int(py)
                    # Triangular weight: samples far along the streamline
                    # contribute less, which keeps the grain crisp instead of
                    # washing out into a uniform blur.
                    w = 1.0 - s / n_steps
                    acc += w * noise[ix, iy]
                    wsum += w
            out[i, j] = acc / wsum
    return out


def line_integral_convolution(vx, vy, *, n_steps: int = 24,
                              step_len: float = 0.7, periodic: bool = True,
                              seed: int = 0, contrast: float = 1.0,
                              noise_scale: int = 1) -> np.ndarray:
    """LIC texture for a 2D vector field, returned in ``[0, 1]``.

    Parameters
    ----------
    vx, vy:
        Field components on a regular grid, indexed ``[ix, iy]``.
    n_steps, step_len:
        Streamline length in pixels is roughly ``n_steps * step_len``. Longer
        gives smoother, more directional streaks; too long and everything
        blurs together.
    noise_scale:
        Size of a noise speckle in pixels. Larger values give a coarser,
        more painterly grain -- useful when the figure will be shown small.
    contrast:
        Post-hoc contrast stretch about the mean.
    """
    vx = np.ascontiguousarray(vx, dtype=np.float64)
    vy = np.ascontiguousarray(vy, dtype=np.float64)
    nx, ny = vx.shape
    rng = np.random.default_rng(seed)

    if noise_scale > 1:
        coarse = rng.random((max(1, nx // noise_scale), max(1, ny // noise_scale)))
        noise = np.kron(coarse, np.ones((noise_scale, noise_scale)))[:nx, :ny]
        if noise.shape != (nx, ny):  # pragma: no cover - odd sizes
            noise = np.resize(noise, (nx, ny))
    else:
        noise = rng.random((nx, ny))
    noise = np.ascontiguousarray(noise, dtype=np.float64)

    tex = _lic_kernel(vx, vy, noise, int(n_steps), float(step_len), bool(periodic))

    # Normalise, then stretch contrast about the mean. LIC output has much
    # lower variance than the input noise (that is what the averaging does),
    # so without this the texture is nearly invisible.
    tex -= tex.mean()
    sd = tex.std()
    if sd > 1e-12:
        tex /= sd
    tex = 0.5 + 0.5 * np.tanh(contrast * tex)
    return np.clip(tex, 0.0, 1.0)


def have_numba() -> bool:
    return _HAVE_NUMBA
