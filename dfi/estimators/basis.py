"""Rung 3: Stochastic Force Inference -- project the drift onto a basis.

Reference: A. Frishman & P. Ronceray, "Learning force fields from stochastic
trajectories", Phys. Rev. X **10**, 021009 (2020).

The idea in one line: stop estimating ``b`` pointwise and instead expand it in a
finite basis, ``b_j(x) = sum_a c_{ja} f_a(x)``, then solve for the coefficients
by least squares against the Kramers-Moyal increments. Every increment informs
every coefficient -- instead of dividing the data among bins or kernels, all of
it is used for each unknown.

The estimator
-------------
With ``F`` the ``(n, p)`` matrix of basis functions at the sample points and
``Y = dx/dt`` the ``(n, d)`` targets,

.. math::  C = (F^T F)^{-1} F^T Y,  \\qquad  \\hat b(x) = f(x)\\, C .

Only the sufficient statistics ``G = F^T F``, ``B = F^T Y`` and ``sum |y|^2`` are
ever needed, so they are accumulated in chunks (on the GPU when there is one)
and ``F`` is never held whole. The **empirical** Gram matrix is used, which
weights the basis by the sampled density -- the right normalisation, since it
down-weights functions that are large only where no data lives.

The error estimate
------------------
Each increment carries noise of variance ``sigma^2 = 2D/dt`` per component. The
coefficient covariance is ``sigma^2 G^{-1}``, so the mean squared error of the
fit, averaged over the sampled points, has variance part

.. math::  \\langle |\\hat b - b|^2 \\rangle_{var}
           = \\frac{d\\, p\\, \\sigma^2}{n} = \\frac{2\\, d\\, p\\, D}{n\\, \\Delta t}

-- ``p d`` free coefficients sharing ``n d`` noisy observations. This is
Frishman & Ronceray's ``n_b d / (2 I)`` with information ``I = tau <F^2> / 4D``.
(An earlier draft of this docstring wrote ``p d D / (n dt)``, missing the 2.)
``sigma^2`` is estimated from the fit's own residuals, so the estimate uses no
ground truth. It does not include the bias from the part of ``b`` outside the
span of the basis; plotting the two against sample size shows where the bias
takes over.

Bases
-----
``'fourier'``
    ``[1, cos(k.x), sin(k.x), ...]`` over reciprocal-lattice vectors
    ``k = 2 pi n / L`` with ``|n|^2 <= degree``, one of each ``+-n`` pair. The
    natural basis on the torus -- and an exact one here, since both field
    generators build ``U`` from lattice modes. The cutoff is on the squared
    *Euclidean* norm, matching the isotropic spectrum of the fields: an
    ``|n|_inf`` cutoff would enumerate ``(2 n_max + 1)^d`` modes, mostly in the
    corners of frequency space where the fields have no power.
``'polynomial'``
    Products of Legendre polynomials in coordinates scaled to the data, total
    degree ``<= degree``. Legendre rather than monomials to keep the Gram
    matrix conditioned. Exact for Ornstein-Uhlenbeck at degree 1.
``'rbf'``
    Gaussian bumps on a regular grid of ``n_centers`` per axis (width one
    spacing, minimum image on a torus), plus a constant. Fixed size -- not
    nested, so not cross-validated.
``'auto'`` (default)
    Fourier on a periodic box, polynomial on an open one.

Choosing the size
-----------------
``degree='cv'`` (default): K-fold cross-validation grouped by walker, as in rungs
1 and 2. The bases are **nested** -- each candidate is a leading block of the
largest -- so one pass over the data yields every candidate's Gram matrix, and
each fold's held-out loss follows from sufficient statistics:

.. math::  \\sum_{held-out} |y - f C|^2
           = S - 2\\, \\mathrm{tr}(C^T B) + \\mathrm{tr}(C^T G C) .

Cross-validation over basis size costs one Cholesky per candidate per fold, and
no refits. The largest candidate is capped both by ``max_basis`` and by the
compute budget ``rows * p^2``.

The curse of dimensionality, quantified
---------------------------------------
The number of lattice modes with ``|n|^2 <= L`` grows like ``L^(d/2)``: 49 up to
``L = 16`` in 2D, 131 up to ``L = 3`` in 5D, 1161 up to ``L = 3`` in 10D. The
variance term grows linearly in ``p``, so at fixed data the affordable basis
shrinks as ``d`` grows -- while the fields' own modes move outward (the typical
``|n|^2`` of a mode in the 10D field is about 6). Past some ``d`` the basis can
no longer reach where the field lives, and that is where a method that does not
enumerate a basis should take over.
"""
from __future__ import annotations

from math import comb

import numpy as np

from .base import TrajectoryEstimator, register

__all__ = ["BasisProjection"]


# --------------------------------------------------------------------------
# basis construction
# --------------------------------------------------------------------------

def _lattice_half(d: int, level: int) -> np.ndarray:
    """Integer vectors ``n != 0`` with ``|n|^2 <= level``, one per ``+-n`` pair.

    Sorted by ``|n|^2`` (then lexicographically), so the modes up to any lower
    level are a leading block -- which is what makes the bases nested.
    """
    out = []
    buf = [0] * d

    def rec(k, remaining):
        if k == d:
            out.append(tuple(buf))
            return
        r = int(np.floor(np.sqrt(remaining)))
        for v in range(-r, r + 1):
            buf[k] = v
            rec(k + 1, remaining - v * v)
        buf[k] = 0

    rec(0, level)
    vecs = np.array(out, dtype=np.int64).reshape(-1, d)
    nz = np.any(vecs != 0, axis=1)
    vecs = vecs[nz]
    # keep the representative whose first nonzero component is positive
    first = vecs[np.arange(len(vecs)), np.argmax(vecs != 0, axis=1)]
    vecs = vecs[first > 0]
    order = np.lexsort(tuple(vecs[:, ::-1].T) + (np.sum(vecs ** 2, axis=1),))
    return vecs[order]


def _count_lattice_ball(d: int, level: int) -> int:
    """Number of integer vectors with ``|n|^2 <= level`` (all signs, incl. 0)."""
    # counts[s] = number of vectors in k dims with squared norm exactly s
    counts = np.zeros(level + 1, dtype=object)
    counts[0] = 1
    for _ in range(d):
        new = np.zeros(level + 1, dtype=object)
        for v in range(-int(np.sqrt(level)), int(np.sqrt(level)) + 1):
            s = v * v
            new[s:] += counts[:level + 1 - s]
        counts = new
    return int(sum(counts))


def _multi_indices(d: int, degree: int) -> np.ndarray:
    """Exponent tuples with total degree ``<= degree``, sorted by total degree."""
    out = []
    buf = [0] * d

    def rec(k, remaining):
        if k == d:
            out.append(tuple(buf))
            return
        for v in range(remaining + 1):
            buf[k] = v
            rec(k + 1, remaining - v)
        buf[k] = 0

    rec(0, degree)
    idx = np.array(out, dtype=np.int64).reshape(-1, d)
    order = np.lexsort(tuple(idx[:, ::-1].T) + (idx.sum(1),))
    return idx[order]


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

@register
class BasisProjection(TrajectoryEstimator):
    """Stochastic Force Inference: least-squares projection onto a basis.

    Parameters
    ----------
    basis:
        ``'auto'``, ``'fourier'``, ``'polynomial'`` or ``'rbf'``.
    degree:
        Size of the basis: the maximum ``|n|^2`` for Fourier, the maximum total
        degree for polynomials (ignored for ``'rbf'``). ``'cv'`` chooses it by
        cross-validation over walkers.
    n_centers:
        RBF centres per axis.
    ridge:
        Tikhonov term, relative to the mean diagonal of the Gram matrix. Zero
        by default: least squares on a well-posed basis needs no tuning, and a
        condition number is reported with every fit so ill-posedness is visible
        rather than papered over.
    max_basis:
        Upper bound on the number of basis functions considered.
    """

    key = "sfi"
    label = "Basis projection (SFI)"
    color_slot = 2

    #: Compute budget for accumulating the Gram matrix, in rows * p^2.
    BUDGET = {"cuda": 4e13, "cpu": 2e12}
    #: Beyond this total degree polynomials only amplify noise at the edges of
    #: the data, whatever the cross-validation grid would say.
    MAX_POLY_DEGREE = 16

    def __init__(self, basis: str = "auto", degree="cv", n_centers: int = 6,
                 ridge: float = 0.0, max_basis: int = 2048, cv_folds: int = 3,
                 device=None, seed: int = 0, **params):
        super().__init__(basis=basis, degree=degree, **params)
        if basis not in ("auto", "fourier", "polynomial", "rbf"):
            raise ValueError(f"unknown basis {basis!r}")
        if isinstance(degree, str) and degree != "cv":
            raise ValueError("degree must be an int or 'cv'")
        self._basis_arg = basis
        self._degree_arg = degree
        self.basis = basis
        self.degree = degree
        self.n_centers = int(n_centers)
        self.ridge = float(ridge)
        self.max_basis = int(max_basis)
        self.cv_folds = int(cv_folds)
        self.device = device
        self.seed = int(seed)

        self.box = None
        self.dt = None
        self.coef = None          # (p, d)
        self._gram = None         # (p, p), G / n
        self._apply_inv = None    # applies G^{-1}; for the error band
        self.truncated_ = 0       # directions dropped as unresolvable
        self._modes = None        # fourier: (m, d) wave vectors; poly: exponents
        self._centers = None      # rbf
        self.cv_ = None
        self.resid_var_ = None    # sigma^2 estimated from residuals
        self.n_ = None

    # -- the basis ---------------------------------------------------------
    @property
    def n_basis(self) -> int:
        if self.coef is None:
            raise RuntimeError("fit() first")
        return int(self.coef.shape[0])

    def _levels(self, d: int, cap: int):
        """Candidate sizes ``(level, p)`` for the nested bases, ascending."""
        out = []
        if self.basis == "fourier":
            lvl = 0
            while True:
                p = _count_lattice_ball(d, lvl)       # == 1 + 2 * (#half modes)
                if p > cap:
                    break
                if not out or p > out[-1][1]:          # skip levels adding nothing
                    out.append((lvl, p))
                lvl += 1
        else:                                          # polynomial
            deg = 0
            while comb(d + deg, deg) <= cap and deg <= self.MAX_POLY_DEGREE:
                out.append((deg, comb(d + deg, deg)))
                deg += 1
        # Keep sizes growing by at least ~20% per step. In 2D there are hundreds
        # of Fourier levels below the cap, one lattice shell apart; testing all
        # of them buys nothing, since the CV curve is flat on that scale.
        thinned = []
        for lvl, p in out:
            if not thinned or p >= 1.2 * thinned[-1][1] or (lvl, p) == out[-1]:
                thinned.append((lvl, p))
        return thinned

    def _setup_basis(self, box, top_level):
        d = box.d
        if self.basis == "fourier":
            n = _lattice_half(d, top_level)
            self._modes = 2.0 * np.pi * n / np.asarray(box.length)[None, :]
        elif self.basis == "polynomial":
            self._modes = _multi_indices(d, top_level)
            # Scale to where the data is, not to the box. On an open box the
            # walkers may fill a third of it, and Legendre polynomials scaled
            # to the box are then nearly collinear on the data: the Gram
            # matrix of 1D Ornstein-Uhlenbeck lost positive-definiteness at
            # degree 15 that way.
        else:
            m = self.n_centers
            axes = [np.asarray(box.lo)[i] + (np.arange(m) + 0.5)
                    * np.asarray(box.length)[i] / m for i in range(d)]
            mesh = np.meshgrid(*axes, indexing="ij")
            self._centers = np.stack([g.ravel() for g in mesh], axis=-1)
            self._rbf_width = float(np.min(box.length)) / m

    def _features(self, x, xp=np, p=None):
        """Basis at ``x`` (``(n, d)``), in numpy or torch (``xp``)."""
        if self.basis == "fourier":
            K = self._modes
            if p is not None:
                K = K[: (p - 1) // 2]
            if xp is np:
                ph = x @ K.T
                f = np.empty((x.shape[0], 1 + 2 * K.shape[0]), dtype=x.dtype)
                f[:, 0] = 1.0
                f[:, 1::2] = np.cos(ph)
                f[:, 2::2] = np.sin(ph)
                return f
            import torch
            Kt = torch.as_tensor(K, dtype=x.dtype, device=x.device)
            ph = x @ Kt.T
            f = torch.empty((x.shape[0], 1 + 2 * K.shape[0]), dtype=x.dtype,
                            device=x.device)
            f[:, 0] = 1.0
            f[:, 1::2] = torch.cos(ph)
            f[:, 2::2] = torch.sin(ph)
            return f

        if self.basis == "polynomial":
            E = self._modes if p is None else self._modes[:p]
            if xp is np:
                u = (x - self._poly_center) / self._poly_scale
            else:
                import torch
                u = (x - torch.as_tensor(self._poly_center, dtype=x.dtype,
                                         device=x.device)) / torch.as_tensor(
                    self._poly_scale, dtype=x.dtype, device=x.device)
            top = int(E.max()) if E.size else 0
            # Legendre values P_0..P_top for every coordinate, by recurrence.
            P = [xp.ones_like(u), u]
            for k in range(1, top):
                P.append(((2 * k + 1) * u * P[k] - k * P[k - 1]) / (k + 1))
            f = []
            for e in E:
                col = None
                for i, ei in enumerate(e):
                    if ei:
                        term = P[int(ei)][:, i]
                        col = term if col is None else col * term
                f.append(col if col is not None else P[0][:, 0])
            return (np.stack(f, axis=1) if xp is np
                    else __import__("torch").stack(f, dim=1))

        # rbf
        C = self._centers
        if xp is np:
            diff = x[:, None, :] - C[None, :, :]
            if self.box.periodic:
                L = np.asarray(self.box.length)
                diff -= L * np.round(diff / L)
            r2 = np.sum(diff * diff, axis=-1)
            f = np.exp(-0.5 * r2 / self._rbf_width ** 2)
            return np.concatenate([np.ones((len(x), 1), x.dtype), f], axis=1)
        import torch
        Ct = torch.as_tensor(C, dtype=x.dtype, device=x.device)
        diff = x[:, None, :] - Ct[None, :, :]
        if self.box.periodic:
            L = torch.as_tensor(self.box.length, dtype=x.dtype, device=x.device)
            diff = diff - L * torch.round(diff / L)
        f = torch.exp(-0.5 * (diff * diff).sum(-1) / self._rbf_width ** 2)
        return torch.cat([torch.ones((len(x), 1), dtype=x.dtype,
                                     device=x.device), f], dim=1)

    def features(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted basis: ``(..., d)`` in, ``(..., n_basis)`` out."""
        self._check_fitted()
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        f = self._features(x.reshape(-1, x.shape[-1]), p=self.n_basis)
        return f.reshape(lead + (f.shape[-1],))

    # -- sufficient statistics --------------------------------------------
    def _device(self):
        if self.device == "cpu":
            return None
        try:
            import torch
        except ImportError:
            return None
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda") if torch.cuda.is_available() else None

    def _accumulate(self, x, y, groups, n_groups, p):
        """Per-group ``G = F^T F``, ``B = F^T Y``, ``S = sum |y|^2``, ``n``.

        Accumulated in float32 blocks and summed in float64: a block of 40k
        rows loses ~1e-7 relative precision, and the block totals add without
        further loss.
        """
        dev = self._device()
        d = y.shape[1]
        G = np.zeros((n_groups, p, p))
        B = np.zeros((n_groups, p, d))
        S = np.zeros(n_groups)
        N = np.zeros(n_groups, dtype=np.int64)
        chunk = int(max(2048, min(200_000, 4e8 // max(p * 8, 1))))
        order = np.argsort(groups, kind="stable")
        bounds = np.searchsorted(groups[order], np.arange(n_groups + 1))
        for g in range(n_groups):
            rows = order[bounds[g]:bounds[g + 1]]
            N[g] = len(rows)
            S[g] = float(np.sum(y[rows] ** 2))
            for s in range(0, len(rows), chunk):
                r = rows[s:s + chunk]
                if dev is not None:
                    import torch
                    xt = torch.as_tensor(x[r], dtype=torch.float32, device=dev)
                    yt = torch.as_tensor(y[r], dtype=torch.float32, device=dev)
                    F = self._features(xt, xp=torch, p=p)
                    G[g] += (F.T @ F).double().cpu().numpy()
                    B[g] += (F.T @ yt).double().cpu().numpy()
                else:
                    F = self._features(x[r], p=p)
                    G[g] += F.T @ F
                    B[g] += F.T @ y[r]
        return G, B, S, N

    def _solve(self, G, B):
        """``(C, apply_inverse)`` for ``(G + ridge) C = B``.

        Cholesky when the block is positive definite, which is the normal
        case. Otherwise a truncated eigendecomposition: directions the data
        cannot distinguish (eigenvalues below 1e-10 of the largest) are dropped
        rather than inverted. ``self.truncated_`` records how many.
        """
        from scipy.linalg import cho_factor, cho_solve, LinAlgError
        p = G.shape[0]
        A = G + self.ridge * float(np.trace(G)) / p * np.eye(p)
        try:
            c = cho_factor(A, lower=True, check_finite=False)
            self.truncated_ = 0
            return (cho_solve(c, B, check_finite=False),
                    lambda R: cho_solve(c, R, check_finite=False))
        except (LinAlgError, ValueError):
            lam, V = np.linalg.eigh(A)
            keep = lam > 1e-10 * lam.max()
            self.truncated_ = int((~keep).sum())
            inv = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)

            def apply(R):
                return V @ (inv[:, None] * (V.T @ R)) if R.ndim == 2 \
                    else V @ (inv * (V.T @ R))
            return apply(B), apply

    # -- fit ---------------------------------------------------------------
    def fit(self, traj):
        self.box, self.dt = traj.box, traj.dt
        d = traj.d
        if self._basis_arg == "auto":
            self.basis = "fourier" if self.box.periodic else "polynomial"
        x, y = self.regression_data(traj)
        if self.basis == "polynomial":
            lo, hi = np.percentile(x, [0.5, 99.5], axis=0)
            self._poly_center = 0.5 * (lo + hi)
            self._poly_scale = np.maximum(0.5 * (hi - lo), 1e-12)
        n = len(x)
        dev = "cuda" if self._device() is not None else "cpu"
        cap = int(min(self.max_basis, np.sqrt(self.BUDGET[dev] / max(n, 1))))

        if self.basis == "rbf":
            self._setup_basis(self.box, None)
            p = self._centers.shape[0] + 1
            groups = np.zeros(n, dtype=np.int64)
            G, B, S, N = self._accumulate(x, y, groups, 1, p)
            level = None
            chosen_p = p
            Gt, Bt, St = G[0], B[0], S[0]
        else:
            levels = self._levels(d, cap)
            if not levels:
                raise ValueError(f"no {self.basis} basis fits within {cap} "
                                 f"functions in {d}D")
            if self._degree_arg == "cv":
                top_level, top_p = levels[-1]
            else:
                # A fixed degree is sized directly, not looked up in the CV
                # grid -- that grid is thinned, and may skip the level asked for.
                want = int(self._degree_arg)
                top_p = (_count_lattice_ball(d, want) if self.basis == "fourier"
                         else comb(d + want, want))
                if top_p > cap:
                    raise ValueError(
                        f"degree {want} needs {top_p} basis functions, over the "
                        f"cap of {cap} here")
                top_level = want
                levels = [(want, top_p)]
            self._setup_basis(self.box, top_level)

            walker = np.repeat(np.arange(traj.n_walkers), traj.n_frames - 1)
            k = max(2, min(self.cv_folds, traj.n_walkers))
            perm = np.random.default_rng(self.seed + 17).permutation(traj.n_walkers)
            fold_of_walker = np.empty(traj.n_walkers, dtype=np.int64)
            for g, part in enumerate(np.array_split(perm, k)):
                fold_of_walker[part] = g
            groups = fold_of_walker[walker]
            G, B, S, N = self._accumulate(x, y, groups, k, top_p)
            Gt, Bt, St = G.sum(0), B.sum(0), S.sum()

            if self._degree_arg == "cv":
                losses = []
                for lvl, p in levels:
                    tot = 0.0
                    for g in range(k):
                        Gtr = (Gt - G[g])[:p, :p]
                        Btr = (Bt - B[g])[:p]
                        C, _ = self._solve(Gtr, Btr)
                        rss = (S[g] - 2.0 * np.sum(C * B[g][:p])
                               + np.sum(C * (G[g][:p, :p] @ C)))
                        tot += rss
                    losses.append(tot / N.sum())
                losses = np.asarray(losses)
                i = int(np.argmin(losses))
                level, chosen_p = levels[i]
                self.cv_ = {"level": np.array([l for l, _ in levels]),
                            "n_basis": np.array([p for _, p in levels]),
                            "loss": losses, "level_selected": level,
                            "n_basis_selected": chosen_p}
            else:
                level, chosen_p = top_level, top_p

        # Kept so sub_fit() can produce any smaller nested basis for free.
        self._stats = (Gt, Bt, float(St), int(n))
        p = chosen_p
        Gp, Bp = Gt[:p, :p], Bt[:p]
        self.coef, self._apply_inv = self._solve(Gp, Bp)
        rss = St - 2.0 * np.sum(self.coef * Bp) + np.sum(self.coef * (Gp @ self.coef))
        self.n_ = int(n)
        self.resid_var_ = float(max(rss, 0.0) / (n * d))
        self._gram = Gp / n
        self.degree = level
        self.params["degree"] = level
        self.params["n_basis"] = p
        self.params["basis"] = self.basis
        self._fitted = True
        return self

    def sub_fit(self, p: int) -> "BasisProjection":
        """The same fit restricted to the first ``p`` basis functions.

        Exact, not approximate: the bases are nested, so the smaller basis's
        normal equations are the leading block of the stored ones. Lets an
        analysis sweep the basis size -- measured error and error bound side by
        side -- from a single pass over the data. ``p`` must not exceed the
        largest basis this fit accumulated.
        """
        import copy
        self._check_fitted()
        if self.basis == "rbf":
            raise ValueError("an RBF basis is not nested; refit instead")
        Gt, Bt, St, n = self._stats
        if p > Gt.shape[0]:
            raise ValueError(f"only {Gt.shape[0]} functions were accumulated")
        out = copy.copy(self)
        out.params = dict(self.params)
        Gp, Bp = Gt[:p, :p], Bt[:p]
        out.coef, out._apply_inv = out._solve(Gp, Bp)
        rss = St - 2.0 * np.sum(out.coef * Bp) + np.sum(out.coef * (Gp @ out.coef))
        out.resid_var_ = float(max(rss, 0.0) / (n * Bt.shape[1]))
        out._gram = Gp / n
        out.params["n_basis"] = p
        return out

    # -- predict -----------------------------------------------------------
    def predict(self, x):
        self._check_fitted()
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        out = np.empty((len(flat), self.coef.shape[1]))
        step = int(max(1024, 4e7 // max(self.n_basis, 1)))
        for s in range(0, len(flat), step):
            out[s:s + step] = self._features(flat[s:s + step], p=self.n_basis) @ self.coef
        return out.reshape(lead + (self.coef.shape[1],))

    # -- the theory --------------------------------------------------------
    def error_bound(self, traj=None) -> float:
        """Predicted mean squared error from noise alone: ``d p sigma^2 / n``.

        ``sigma^2`` is the residual variance per component, so no ground truth
        enters. Averaged over the sampled points; excludes projection bias.
        ``traj`` is accepted for interface compatibility and not needed -- the
        statistics were kept from the fit.
        """
        self._check_fitted()
        d = self.coef.shape[1]
        return float(d * self.n_basis * self.resid_var_ / self.n_)

    def condition_number(self) -> float:
        """Condition number of the empirical Gram matrix of the chosen basis."""
        if self._gram is None:
            raise RuntimeError("fit() first")
        return float(np.linalg.cond(self._gram))

    def uncertainty(self, x):
        """Per-component standard error of ``b_hat(x)``, shape ``x.shape[:-1]``.

        ``sqrt(sigma^2 f(x)^T G^{-1} f(x))``, from the coefficient covariance.
        The same for every component (isotropic noise). Covers noise only, so
        it is honest exactly where the basis is not biased.
        """
        self._check_fitted()
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        f = self._features(x.reshape(-1, x.shape[-1]), p=self.n_basis)
        q = np.einsum("np,np->n", f, self._apply_inv(f.T).T)
        return np.sqrt(self.resid_var_ * np.maximum(q, 0.0)).reshape(lead)

    def diagnostics(self) -> dict:
        """Numbers worth a column in a results table."""
        self._check_fitted()
        return {"n_basis": self.n_basis,
                "gram_cond": self.condition_number(),
                "truncated": self.truncated_,
                "bound_mse": self.error_bound(),
                "resid_var": self.resid_var_}
