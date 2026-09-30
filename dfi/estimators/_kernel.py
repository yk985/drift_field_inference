"""Numerical machinery behind :class:`~dfi.estimators.local.KernelRegression`.

Kept apart from the estimator so the estimator reads as the statistics it is,
and this file as the arithmetic that makes it fast.

Everything a Gaussian-kernel local regression needs at a query point ``q`` is a
handful of kernel-weighted sums over the samples ``(x_i, y_i)``, with
``u_i = x_i - q`` (minimum image on a torus) and ``w_i = exp(-|u_i|^2 / 2h^2)``:

    S0 = sum w          S1 = sum w u        S2 = sum w u u^T
    T0 = sum w y        T1 = sum w u y^T

Nadaraya-Watson needs ``S0, T0``; local-linear needs all five.

What ``support`` thresholds is a sixth number: the **count of samples within
two bandwidths**, ``#{i : |u_i| <= 2h}``. In 2D that disc has area ``12.6 h^2``
-- almost exactly one bin cell at the same smoothing scale (``12 h^2``) -- so
"at least 8 within 2h" is rung 1's ``min_count`` rule carried over.

Two more natural choices both fail, and it is worth knowing how, because both
failures look like a method doing well. Measured at D = 0.05 on the 2D field,
where the walkers explore about a third of the box:

- the effective sample size ``S0^2 / sum w^2`` measures how *evenly* the
  weight is spread, not how *much* there is. Far from a compact cloud a wide
  kernel weights every member of it about equally, so it stays large at any
  distance: a local-linear fit claimed support over **100%** of the box while
  wrong by more than ``|b|`` across whole regions;
- the weighted count ``S0 = sum w`` is better but still has a Gaussian tail.
  Next to a cloud of ~50k samples it stays above 8 out to ``r ~ 4.2 h``, and the
  same fit claimed **93%**.

A compact neighbourhood has no tail.

Two ways of computing them:

**Lattice + FFT** (``d <= 3``). Bin every sample onto a fine lattice. Each sum
is then a *correlation* of a binned data array with a fixed kernel array --
``S1`` is the counts correlated with ``K(u) u``, ``T0`` the summed targets
correlated with ``K(u)`` -- and a correlation is one product in Fourier space.
On a periodic box this is exact (the lattice *is* the torus), costs ``O(M^d
log M)`` regardless of how many samples there are, and uses every one of them.
The kernels are separable, so their transforms are outer products of 1D
transforms and cost nothing. Binning moves each sample to its cell centre,
which is equivalent to adding a uniform jitter of one cell; with cells no wider
than ``h/3`` that widens the kernel by under 1%.

Measured against the explicit sums on the same data: **0.1-0.3%** apart when
the targets are noise-free, which is the binning error proper. With real
targets they differ by ~6%, and that is *not* a larger error -- the jitter
reweights each sample by ~15%, and the samples carry noise ~25x larger than
``b``, so the two paths see a different draw of the same noise. Their errors
against the truth agree to three decimals (1.390 vs 1.394 on the case tested).

**Chunked direct sums** (``d > 3``). A lattice of ``M^d`` cells is impossible,
so compute the sums explicitly for blocks of queries, on the GPU when there is
one. Cost is ``O(n_query * n_samples * d^2)``, which is why the estimator
compresses the data first (see ``KernelRegression.fit``).
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import map_coordinates

#: Relative ridge on the slope block of the local-linear normal equations. It
#: keeps sparse nodes solvable, and degrades them smoothly towards
#: Nadaraya-Watson rather than towards garbage. At 1e-3 it shrinks a
#: well-conditioned slope by 0.1%, which is far below anything measured here.
RIDGE = 1e-3


# --------------------------------------------------------------------------
# lattice backend
# --------------------------------------------------------------------------

class Lattice:
    """A regular grid of cells over the box, sized for the finest bandwidth.

    On an open box the lattice is padded by four of the widest bandwidths on
    every side and then treated as periodic: the padding is wide enough that
    the Gaussian has decayed to ``e^-8`` before anything can wrap around.
    """

    CAP = {1: 8192, 2: 512, 3: 128}

    def __init__(self, box, h_min: float, h_max: float):
        d = box.d
        self.box = box
        self.d = d
        self.periodic = bool(box.periodic)
        pad = 0.0 if self.periodic else 4.0 * h_max
        self.origin = np.asarray(box.lo, float) - pad
        self.length = np.asarray(box.length, float) + 2.0 * pad
        # Three cells across the narrowest kernel, rounded up to a power of two
        # for the FFT, and capped because M^d is the memory bill.
        want = 3.0 * float(self.length.max()) / h_min
        M = 2 ** int(np.ceil(np.log2(max(want, 2.0))))
        self.M = int(np.clip(M, 32, self.CAP[d]))
        self.shape = (self.M,) * d
        self.cell = self.length / self.M
        #: The narrowest bandwidth this lattice can represent faithfully.
        self.h_floor = 3.0 * float(self.cell.max())

    # -- data onto the lattice ----------------------------------------------
    def flat_index(self, x: np.ndarray) -> np.ndarray:
        i = np.floor((x - self.origin) / self.cell).astype(np.intp)
        if self.periodic:
            i %= self.M
        else:
            np.clip(i, 0, self.M - 1, out=i)
        return np.ravel_multi_index(tuple(i.T), self.shape)

    def bin(self, x: np.ndarray, y: np.ndarray):
        """Per-cell sample count and per-cell summed target, as FFTs."""
        flat = self.flat_index(x)
        n = self.M ** self.d
        cnt = np.bincount(flat, minlength=n).reshape(self.shape).astype(float)
        sy = [np.bincount(flat, weights=y[:, j], minlength=n)
              .reshape(self.shape) for j in range(y.shape[1])]
        fcnt = sfft.fftn(cnt, workers=-1)
        fsy = [sfft.fftn(s, workers=-1) for s in sy]
        return fcnt, fsy

    # -- kernels ---------------------------------------------------------------
    def _axis_offsets(self, a: int) -> np.ndarray:
        """Centre-to-centre offsets along one axis, minimum image."""
        k = np.arange(self.M)
        k = np.where(k < self.M // 2, k, k - self.M)
        return k * self.cell[a]

    def kernel_transforms(self, h: float, local_linear: bool):
        """FFTs of ``K``, ``K u_a`` and ``K u_a u_b``.

        Built from 1D transforms: for a separable kernel the d-dimensional
        transform is the outer product of the per-axis ones, so nothing here
        ever touches an ``M^d`` real-space array.
        """
        d = self.d
        g, gu, guu = [], [], []
        for a in range(d):
            u = self._axis_offsets(a)
            e = np.exp(-0.5 * (u / h) ** 2)
            g.append(sfft.fft(e))
            if local_linear:
                gu.append(sfft.fft(e * u))
                guu.append(sfft.fft(e * u * u))

        def outer(factors):
            out = factors[0]
            for f in factors[1:]:
                out = np.multiply.outer(out, f)
            return out

        out = {"K": outer(g)}
        if local_linear:
            out["Ku"] = [outer([gu[b] if b == a else g[b] for b in range(d)])
                         for a in range(d)]
            out["Kuu"] = {}
            for a in range(d):
                for b in range(a, d):
                    if a == b:
                        fac = [guu[c] if c == a else g[c] for c in range(d)]
                    else:
                        fac = [gu[c] if c in (a, b) else g[c] for c in range(d)]
                    out["Kuu"][(a, b)] = outer(fac)
        return out

    @staticmethod
    def correlate(fdata, fkernel) -> np.ndarray:
        """``sum_c data(c) kernel(c - q)`` at every node ``q``.

        Correlation, not convolution: the kernel is evaluated at the offset of
        the *sample* from the *query*. In Fourier space that is the data
        transform times the conjugate kernel transform. Getting this backwards
        flips the sign of every odd moment -- which leaves Nadaraya-Watson
        untouched and silently breaks local-linear, so it is worth stating.
        """
        return sfft.ifftn(fdata * np.conj(fkernel), workers=-1).real

    # -- solve on the lattice -----------------------------------------------
    def ball_count(self, fcnt, radius: float) -> np.ndarray:
        """Number of samples within ``radius`` of every node.

        The one kernel here that is not separable, so it is built as a real
        ``M^d`` indicator array -- once per fit, not once per bandwidth.
        """
        grids = np.meshgrid(*[self._axis_offsets(a) for a in range(self.d)],
                            indexing="ij")
        ball = (sum(g * g for g in grids) <= radius * radius).astype(float)
        cnt = self.correlate(fcnt, sfft.fftn(ball, workers=-1))
        return np.maximum(np.rint(cnt), 0.0)

    def solve(self, fcnt, fsy, h: float, local_linear: bool):
        """``(b, S0)`` at every node: the estimate and the weighted sum."""
        d = self.d
        kt = self.kernel_transforms(h, local_linear)
        S0 = self.correlate(fcnt, kt["K"])
        T0 = np.stack([self.correlate(f, kt["K"]) for f in fsy], axis=-1)

        # FFT round-off leaves ~1e-12 relative noise in S0 where there is no
        # data at all, occasionally negative. Treat such nodes as empty.
        empty = S0 <= 1e-9 * max(float(S0.max()), 1e-300)
        S0s = np.where(empty, 1.0, S0)

        if not local_linear:
            b = T0 / S0s[..., None]
        else:
            S1 = np.stack([self.correlate(fcnt, k) for k in kt["Ku"]], axis=-1)
            S2 = np.empty(self.shape + (d, d))
            for (a, c), k in kt["Kuu"].items():
                S2[..., a, c] = S2[..., c, a] = self.correlate(fcnt, k)
            T1 = np.empty(self.shape + (d, d))           # [..., a, j]
            for a, k in enumerate(kt["Ku"]):
                for j, f in enumerate(fsy):
                    T1[..., a, j] = self.correlate(f, k)
            b = solve_local_linear(S0s, S1, S2, T0, T1, h)
        b[empty] = 0.0
        return b, np.where(empty, 0.0, S0)

    # -- evaluate the lattice anywhere --------------------------------------
    def interpolate(self, grid: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Linear interpolation of node values at arbitrary points."""
        q = self.box.wrap(x) if self.periodic else self.box.clip(x)
        coords = ((q - self.origin) / self.cell - 0.5).T
        mode = "grid-wrap" if self.periodic else "nearest"
        if grid.ndim == self.d:
            return map_coordinates(grid, coords, order=1, mode=mode)
        return np.stack([map_coordinates(grid[..., j], coords, order=1,
                                         mode=mode)
                         for j in range(grid.shape[-1])], axis=-1)


def solve_local_linear(S0, S1, S2, T0, T1, h):
    """Constant term of the weighted least-squares fit ``y ~ b + B u``.

    Normal equations per point, one right-hand side per output component::

        [ S0    S1^T ] [ b_j   ]   [ T0_j    ]
        [ S1    S2   ] [ B_.j  ] = [ T1_.j   ]

    ``T1[..., a, j] = sum w u_a y_j``. Only the constant term ``b`` is kept --
    the slope ``B`` is the local Jacobian, a by-product here.
    """
    lead = S0.shape
    d = T0.shape[-1]
    A = np.zeros(lead + (d + 1, d + 1))
    A[..., 0, 0] = S0
    A[..., 0, 1:] = S1
    A[..., 1:, 0] = S1
    A[..., 1:, 1:] = S2
    ridge = RIDGE * S0 * h * h
    for a in range(d):
        A[..., 1 + a, 1 + a] += ridge
    R = np.zeros(lead + (d + 1, d))
    R[..., 0, :] = T0
    R[..., 1:, :] = T1
    sol = np.linalg.solve(A.reshape(-1, d + 1, d + 1), R.reshape(-1, d + 1, d))
    return sol[:, 0, :].reshape(lead + (d,))


# --------------------------------------------------------------------------
# direct-sum backend
# --------------------------------------------------------------------------

def _torch_device(device):
    try:
        import torch
    except ImportError:
        return None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else None
    return torch.device(device) if device else None


def direct_sums(x_tr, y_tr, q, hs, *, box, local_linear: bool, device=None,
                budget: float | None = None):
    """Drift estimates at ``q`` for every bandwidth in ``hs``.

    Returns ``b`` of shape ``(len(hs), n_query, d)`` and the number of
    samples within ``2h`` of each query, shape ``(len(hs), n_query)``. Distances are computed once per
    block of queries and reused across bandwidths, which is what makes
    cross-validation over a grid of ``h`` affordable.

    Runs on the GPU through torch when one is available, else in numpy. The
    sums are accumulated in float32 and the normal equations solved in float64
    on the host: summation in float32 costs ~1e-6 relative accuracy, three
    orders below the smallest error this project reports.
    """
    hs = [float(h) for h in np.atleast_1d(hs)]
    n, d = x_tr.shape
    nq = len(q)
    L = np.asarray(box.length, float) if box.periodic else None

    dev = _torch_device(device)
    # Peak memory is a few (block, n, d) arrays: the offsets U, the weighted
    # offsets W*U and one temporary. The d x d moments come out of batched
    # matrix products and are never materialised per sample, so the block
    # size is set by n*d -- not n*d^2, which would shrink blocks to a single
    # query at d = 10 and spend the whole run on host-device round trips.
    if budget is None:
        budget = 4e8 if dev is not None else 8e7       # elements (float32/64)
    chunk = int(max(1, min(nq, budget // max(4 * n * d, 1))))

    if dev is not None:
        import torch
        X = torch.as_tensor(x_tr, dtype=torch.float32, device=dev)
        Y = torch.as_tensor(y_tr, dtype=torch.float32, device=dev)
        Lt = torch.as_tensor(L, dtype=torch.float32, device=dev) \
            if L is not None else None

        def block(qb):
            Q = torch.as_tensor(qb, dtype=torch.float32, device=dev)
            U = X[None, :, :] - Q[:, None, :]                  # (c, n, d)
            if Lt is not None:
                U = U - Lt * torch.round(U / Lt)
            r2 = (U * U).sum(-1)                               # (c, n)
            acc = {}
            for h in hs:
                W = torch.exp(r2 * (-0.5 / (h * h)))
                m = {"S0": W.sum(1), "T0": W @ Y,
                     "N2": (r2 <= 4.0 * h * h).sum(1).to(torch.float32)}
                if local_linear:
                    WU = W[..., None] * U
                    m["S1"] = WU.sum(1)
                    m["S2"] = WU.transpose(1, 2) @ U
                    m["T1"] = WU.transpose(1, 2) @ Y
                    del WU
                for k, v in m.items():
                    acc.setdefault(k, []).append(v)
            # One transfer per moment for all bandwidths, not one per h.
            host = {k: torch.stack(v).double().cpu().numpy()
                    for k, v in acc.items()}
            return [{k: v[i] for k, v in host.items()} for i in range(len(hs))]
    else:
        def block(qb):
            U = x_tr[None, :, :] - qb[:, None, :]
            if L is not None:
                U = U - L * np.round(U / L)
            r2 = np.einsum("cnd,cnd->cn", U, U)
            out = []
            for h in hs:
                W = np.exp(r2 * (-0.5 / (h * h)))
                m = {"S0": W.sum(1), "T0": W @ y_tr,
                     "N2": (r2 <= 4.0 * h * h).sum(1).astype(float)}
                if local_linear:
                    WU = W[..., None] * U
                    m["S1"] = WU.sum(1)
                    m["S2"] = np.einsum("cna,cnb->cab", WU, U)
                    m["T1"] = np.einsum("cna,nj->caj", WU, y_tr)
                out.append(m)
            return out

    b = np.zeros((len(hs), nq, d))
    mass = np.zeros((len(hs), nq))
    for s in range(0, nq, chunk):
        e = min(nq, s + chunk)
        for k, (h, m) in enumerate(zip(hs, block(np.asarray(q[s:e], float)))):
            S0 = m["S0"]
            empty = S0 <= 1e-12
            S0s = np.where(empty, 1.0, S0)
            if local_linear:
                bb = solve_local_linear(S0s, m["S1"], m["S2"], m["T0"],
                                        m["T1"], h)
            else:
                bb = m["T0"] / S0s[:, None]
            bb[empty] = 0.0
            b[k, s:e] = bb
            mass[k, s:e] = m["N2"]
    return b, mass
