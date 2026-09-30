"""Rungs 1 and 2 of the ladder: local averaging of Kramers-Moyal increments.

Both estimators here compute the same conditional expectation

.. math::  b(x) \\approx \\frac{1}{\\Delta t}\\,
           E\\big[\\,x(t+\\Delta t) - x(t)\\ \\big|\\ x(t) = x\\,\\big]

and differ only in how they localise it: hard bins, or smooth kernel weights.

Both are implemented. Rung 1 was written with the user; rung 2's numerical
machinery lives in :mod:`dfi.estimators._kernel` so that this file reads as
the statistics.

Sanity path: fit :class:`BinnedKramersMoyal` on a 1D Ornstein-Uhlenbeck run
where ``b(x) = -kx`` is a straight line. If the estimate is not a straight line
of slope ``-k`` through the origin, stop and fix that before touching anything
harder. ``tests/test_dfi.py`` runs exactly that check.
"""
from __future__ import annotations

import numpy as np

from .base import TrajectoryEstimator, register

__all__ = ["BinnedKramersMoyal", "KernelRegression"]


@register
class BinnedKramersMoyal(TrajectoryEstimator):
    """Rung 1. Average the increments falling in each spatial bin.

    Algorithm
    ---------
    1. Lay a regular grid of ``bins`` cells per axis over the box.
    2. Assign every observed ``(x, dx)`` pair to the cell containing ``x``.
       ``np.ravel_multi_index`` on the per-axis bin indices turns a
       ``d``-dimensional cell address into a flat one; ``np.bincount`` with
       ``weights=dx[:, j]`` then sums each component per cell in one pass.
       Resist the urge to loop over cells -- it is 100x slower and the whole
       estimator is three lines this way.
    3. Divide the summed ``dx`` by the count and by ``dt``. That cell's value
       is the drift estimate there.
    4. Cells with fewer than ``min_count`` samples are undefined. Record which,
       and report it from :meth:`support`.
    5. ``predict(x)`` looks up the cell containing ``x``.

    What to expect, and what to say about it
    ----------------------------------------
    - **Variance.** A cell holding ``n`` increments averages noise of size
      ``sqrt(2D/dt)`` down to ``sqrt(2D/(n dt))``. So the error at ``x`` falls
      like ``1/sqrt(n(x))`` and the estimator is worst exactly where the
      density is lowest. Plot the error against local count and you should see
      that power law; if you do not, something is wrong with the binning.
    - **Bias from bin width -- first order, not second.** The cell value is
      the average of ``b`` over the cell, which is accurate to O(``h^2``) *at
      the cell centre*: the linear term cancels by symmetry there. But
      ``predict`` returns that one value everywhere in the cell, so at a query
      a distance ``delta`` from the centre the error is ``grad b . delta`` --
      first order. Averaged over where queries land it is
      ``|grad b| h / sqrt(12)``. Measured with noise-free targets on the 2D
      random field: error vs cell width has log-log slope **0.97** at random
      points and **1.84** at cell centres. This is the defect rung 2 removes
      (a kernel centred on the query is O(``h^2``) everywhere, slope 1.85
      measured), and most of why it wins.
      Together with the variance above, the total error is minimised at an
      intermediate ``bins`` -- the U-shaped curve.
    - **Curse of dimensionality.** ``bins**d`` cells. At ``d=6`` and 32 bins
      that is 10^9 cells for maybe 10^7 samples: essentially every cell is
      empty. This estimator is not merely inaccurate in high ``d``, it is
      undefined, and that is the honest reason the ladder continues.
    - **Discretisation bias.** Independent of all the above, the target itself
      is biased at O(``dt``). Sweep ``dt`` (via
      ``Trajectories.subsample_time``) and the bias should be linear in ``dt``
      with a coefficient related to ``(b . grad) b``.

    Parameters
    ----------
    bins:
        Cells per axis: an ``int``, ``'auto'`` (the ``N^(1/(d+2))`` rule of
        :meth:`auto_bins`) or ``'cv'`` (cross-validated, see :meth:`_cv_bins`).
    min_count:
        Cells with fewer samples are treated as unsupported.
    """

    key = "binned_km"
    label = "Binned Kramers-Moyal"
    color_slot = 0

    #: Refuse to allocate a grid larger than this many cells. See :meth:`fit`.
    MAX_CELLS = 20_000_000

    def __init__(self, bins="auto", min_count: int = 8, **params):
        super().__init__(bins=bins, min_count=min_count, **params)
        # Keep the request as given, so a refit on different data re-resolves
        # 'auto' rather than silently reusing the first fit's bin count.
        self._bins_arg = bins
        self.bins = bins if bins in ("auto", "cv") else int(bins)
        self.cv_ = None
        self.min_count = int(min_count)
        self.box = None
        self.dt = None
        self.b_grid = None       # (bins,)*d + (d,) estimated drift per cell
        self.counts = None       # (bins,)*d sample count per cell
        self._dims = None        # (bins,)*d
        self._b_flat = None      # (bins**d, d) -- same buffer as b_grid
        self._counts_flat = None # (bins**d,)   -- same buffer as counts
        self._msq_flat = None    # (bins**d, d) mean of (dx/dt)^2 per cell

    # -- cell bookkeeping --------------------------------------------------
    def _cell(self, x: np.ndarray) -> np.ndarray:
        """Flat cell label for each point, shape ``x.shape[:-1]``.

        ``fit``, ``predict`` and ``support`` all go through this one helper, so
        the binning convention cannot drift apart between fitting and
        prediction -- which is the failure mode that produces an estimate that
        looks plausible and is shifted by one cell.

        ``np.ravel_multi_index`` wants a *sequence of d index arrays*, not one
        ``(..., d)`` array, which is what ``moveaxis`` + ``tuple`` produce here.
        """
        x = np.asarray(x, float)
        if self.box.periodic:
            # On a torus, a query outside the box is a legitimate coordinate
            # for a point inside it. Fold it in rather than clamping it to the
            # edge cell.
            x = self.box.wrap(x)
        h = self.box.length / self.bins
        idx = ((x - self.box.lo) / h).astype(np.intp)
        # A point exactly on the upper face lands on index == bins, which
        # ravel_multi_index rejects outright. Clipping also makes an
        # out-of-box query on an open domain resolve to the nearest cell.
        np.clip(idx, 0, self.bins - 1, out=idx)
        return np.ravel_multi_index(tuple(np.moveaxis(idx, -1, 0)), self._dims)

    @staticmethod
    def auto_bins(n_transitions: int, d: int) -> int:
        """Bins per axis from the bias-variance balance of a *piecewise-constant* fit.

        The bias at a query point is first order in the cell width ``h`` (see
        the class docstring), so it contributes O(``h^2``) to the squared
        error; the variance in a cell holding ``n_cell = N / bins^d``
        increments is O(``1/n_cell``). Balancing,

            h^2 ~ bins^d / N   with h ~ 1/bins   =>   bins ~ N^{1/(d+2)} .

        An earlier version of this rule used ``N^{1/(d+4)}``, which is the
        balance for an O(``h^2``) bias -- true at cell centres, false where the
        estimate is actually used. The wrong exponent showed up as a constant
        that would not stay put: fitted to the measured optima it drifted from
        0.8 to 3.9 across 1D OU and the 2D random field, and rose steadily
        *within* 1D. With ``1/(d+2)`` the same seven optima imply constants of
        0.28-0.48, and ``0.34`` is their geometric mean.

        Still a starting point. The optimum also depends on the noise level
        ``2D/dt`` and on how fast ``b`` varies, which no rule in ``N`` alone
        can see -- ``bins='cv'`` measures it instead.
        """
        bins = 0.34 * float(n_transitions) ** (1.0 / (d + 2.0))
        return int(np.clip(round(bins), 4, 64))

    # -- fit ---------------------------------------------------------------
    def fit(self, traj):
        """Estimate the drift on a regular grid of cells."""
        self.box = traj.box
        self.dt = traj.dt
        d = self.box.d

        # regression_data gives (x, dx/dt), already flattened to (N*T, d) and
        # already minimum-image corrected -- so a walker wrapping around the
        # torus contributes its true small step rather than a box-sized jump.
        # Note y is ALREADY divided by dt; dividing again is the classic bug.
        x, y = self.regression_data(traj)

        if self._bins_arg == "auto":
            self.bins = self.auto_bins(traj.n_transitions, d)
        elif self._bins_arg == "cv":
            walker = np.repeat(np.arange(traj.n_walkers), traj.n_frames - 1)
            self.bins = self._cv_bins(x, y, walker, d)
        self.params["bins"] = self.bins
        self._set_bins(self.bins, d, traj.n_transitions)
        self._accumulate(x, y)
        self._fitted = True
        return self

    def _set_bins(self, bins: int, d: int, n_transitions: int):
        self.bins = int(bins)
        self._dims = (self.bins,) * d
        n_cells = self.bins ** d
        # bins**d grows explosively. Fail with the reason rather than with an
        # opaque MemoryError -- this limit *is* the result that motivates
        # rungs 3 and 4, so it deserves to be stated rather than crashed into.
        if n_cells > self.MAX_CELLS:
            raise ValueError(
                f"{self.bins} bins in {d}D is {n_cells:.3g} cells, more than "
                f"this estimator will allocate. With {n_transitions:,} "
                "transitions nearly every cell would be empty anyway: binning "
                "is not merely inaccurate at this dimension, it is undefined. "
                "Lower `bins`, or move to BasisProjection / NeuralDrift.")

    def _cv_bins(self, x, y, walker, d, folds: int = 3, points: int = 9):
        """Bin count by K-fold cross-validation, grouped by walker.

        The same procedure :class:`KernelRegression` uses for its bandwidth, so
        the two rungs are compared as methods rather than as tuning rules:
        fit on some walkers, predict the raw targets of the rest, keep the
        count with the lowest squared prediction error. Candidates span a
        factor ~3 either side of :meth:`auto_bins`. The winner is then scaled
        up by ``(K/(K-1))^(1/(d+2))``, because it was tuned on a fraction
        ``(K-1)/K`` of the data and the optimum grows as ``N^(1/(d+2))``.

        A held-out point in a cell no training walker visited is predicted as
        0 -- exactly what ``predict`` would return -- so over-fine grids are
        penalised for their holes, as they should be.

        One bin is an allowed answer. It is the global mean, and in high ``d``
        it can genuinely be the best this estimator can do; a floor of 2 would
        force a 2^d-cell grid on data that cannot fill it and make binning
        look worse than a kernel that is free to widen to the same average.
        """
        n = len(x)
        base = self.auto_bins(n, d)
        cand = np.unique(np.clip(np.round(
            base * 2.0 ** np.linspace(-1.6, 1.6, points)), 1, None).astype(int))
        cand = cand[cand.astype(float) ** d <= self.MAX_CELLS]
        rng = np.random.default_rng(7)
        groups = np.array_split(rng.permutation(int(walker.max()) + 1),
                                max(2, min(folds, int(walker.max()) + 1)))
        loss = np.zeros((len(groups), len(cand)))
        for k, held in enumerate(groups):
            te = np.isin(walker, held)
            for i, b in enumerate(cand):
                self._set_bins(b, d, n)
                self._accumulate(x[~te], y[~te])
                pred = self._b_flat[self._cell(x[te])]
                loss[k, i] = np.mean(np.sum((y[te] - pred) ** 2, axis=-1))
        mean = loss.mean(0)
        best = float(cand[int(np.argmin(mean))])
        frac = 1.0 - 1.0 / len(groups)
        chosen = int(np.clip(round(best * (1.0 / frac) ** (1.0 / (d + 2.0))),
                             1, None))
        self.cv_ = {"bins": cand, "loss": mean, "loss_folds": loss,
                    "bins_min_cv": best, "bins_selected": chosen}
        return chosen

    def _accumulate(self, x, y):
        """Per-cell count, mean target and mean squared target."""
        d = self.box.d
        n_cells = self.bins ** d
        flat = self._cell(x)

        # bincount is the group-by: one vectorised pass per component, no
        # Python loop over cells. `minlength` pins the result to the full grid
        # -- without it the array stops at the highest *occupied* cell, and the
        # reshape below fails whenever the last cells happen to be empty.
        counts = np.bincount(flat, minlength=n_cells)
        sums = np.stack(
            [np.bincount(flat, weights=y[:, j], minlength=n_cells)
             for j in range(d)], axis=-1)                      # (n_cells, d)
        sq = np.stack(
            [np.bincount(flat, weights=y[:, j] ** 2, minlength=n_cells)
             for j in range(d)], axis=-1)

        # Empty cells: numerator is 0, so dividing by 1 leaves 0 without
        # tripping a divide-by-zero warning. Their value is meaningless either
        # way; `support` is what marks them as untrustworthy.
        denom = np.maximum(counts, 1)[:, None]
        self._b_flat = sums / denom
        self._msq_flat = sq / denom
        self._counts_flat = counts

        # The documented shapes. reshape returns a view, so this costs nothing.
        self.b_grid = self._b_flat.reshape(self._dims + (d,))
        self.counts = counts.reshape(self._dims)

    # -- predict -----------------------------------------------------------
    def predict(self, x):
        self._check_fitted()
        return self._b_flat[self._cell(x)]

    def support(self, x):
        """False where the cell containing ``x`` has too few samples."""
        self._check_fitted()
        return self._counts_flat[self._cell(x)] >= self.min_count

    def predict_diffusion(self, x):
        """Per-component ``D(x)`` from the second Kramers-Moyal moment.

        Accumulated in the same pass as the drift. Since ``y = dx/dt`` was the
        regression target, ``E[dx^2]/(2 dt) = E[y^2]\\,dt/2``.

        Strictly this second moment is ``2D\\,dt + (b\\,dt)^2``, so it
        over-estimates ``D`` by a term of order ``dt``; that term is negligible
        at any usable step size. Worth watching regardless: a ``dt`` too coarse
        for the field inflates this above the true ``D`` long before the drift
        estimate visibly breaks, so it is an early warning for free.
        """
        self._check_fitted()
        return 0.5 * self.dt * self._msq_flat[self._cell(x)]


@register
class KernelRegression(TrajectoryEstimator):
    """Rung 2. Smooth kernel weights instead of hard bin edges.

    .. math::  \\hat b(q) = \\frac{\\sum_i K_h(x_i - q)\\, y_i}
                                 {\\sum_i K_h(x_i - q)},
               \\qquad y_i = \\Delta x_i / \\Delta t,
               \\qquad K_h(u) = e^{-|u|^2 / 2h^2}

    That is Nadaraya-Watson (``local_linear=False``): a weighted average of the
    raw targets, the weight falling off with distance from the query. With
    ``local_linear=True`` it instead fits ``y ~ b + B (x - q)`` by weighted
    least squares around each query and keeps the constant ``b``.

    How the kernel is determined
    ----------------------------
    **Shape barely matters; width is everything.** Gaussian, Epanechnikov and
    box kernels give errors within a few percent of each other at their own
    best width. The bandwidth ``h`` is the same bias-variance dial as the bin
    width in rung 1:

    - *variance* -- the average is built from ``n_eff = (sum w)^2 / sum w^2``
      samples, each carrying noise of variance ``2D/dt`` per component, so it
      scales as ``(2D/dt) / (n rho(q) h^d)``;
    - *bias* -- averaging over a neighbourhood of size ``h`` is off by
      ``O(h^2)``. For Nadaraya-Watson the leading term is

      .. math::  h^2 \\Big[\\tfrac12 \\Delta b
                 + (\\nabla b)\\,\\nabla \\log \\rho\\Big]

      and the second piece is specific and nasty here: for a gradient field
      ``grad log rho_ss = b / D``, so the bias grows like ``1/D`` -- worst at
      *low* noise, exactly where the variance is smallest. On 1D OU
      (``b = -kx``) it shrinks the fitted slope by a factor
      ``1 - h^2 k / D = 1 - h^2 / sigma_ss^2``. Local-linear fitting cancels
      that term identically (it is exact for any linear ``b``), which is the
      whole reason it exists.

    **Choosing h.** Three options, via ``bandwidth``:

    ``float``
        Use it.
    ``'silverman'``
        ``h = sigma n^(-1/(d+4))`` from the spread of the positions. A density
        estimation rule: it has the right exponent and the wrong everything
        else, because it cannot see the noise level ``2D/dt`` or the curvature
        of ``b`` -- the two things the regression optimum actually depends on.
        Kept as a reference point.
    ``'cv'`` (default)
        K-fold cross-validation over a log-spaced grid of ``h``, **grouped by
        walker**: fit on some walkers, predict the raw targets ``y`` of the
        others, keep the ``h`` with the smallest mean squared prediction error.
        Held-out ``y`` carries noise independent of the fit, so
        ``E|y - b_hat|^2 = (noise floor) + E|b - b_hat|^2`` -- the floor does
        not depend on ``h``, so minimising prediction error minimises the true
        error without ever seeing the truth. The floor is ~500x the signal at
        typical settings, which is why CV here needs every held-out increment
        rather than a sample of them. Grouping by walker, not by row, matters
        for local-linear: a row's own offset from its time-neighbours *is* its
        increment, so row-level splits leak.

        The selected ``h`` is then shrunk by ``((K-1)/K)^(1/(d+4))``, since it
        was tuned on a fraction ``(K-1)/K`` of the data and the optimum scales
        as ``n^(-1/(d+4))``.

    Backends
    --------
    ``d <= 3``: every sample is binned onto a fine lattice and all kernel sums
    are FFT correlations (see :mod:`dfi.estimators._kernel`). Uses all the
    data, costs the same for any ``n``, and makes a CV grid nearly free.

    ``d > 3``: explicit chunked sums, on the GPU when available. Cost is
    ``n_query * n_samples * d^2``, so the data is **compressed** first by
    merging each walker's increments into blocks of ``m`` consecutive steps.
    The block target ``(x_{t+m} - x_t) / (m dt)`` averages the same noise --
    the increments telescope -- so the information about ``b`` is preserved
    and only the ``O(m dt)`` discretisation bias grows. ``m`` is capped so a
    block's RMS displacement stays under 10% of the box; past that cap, rows
    are subsampled at random, and that *does* throw information away.

    Parameters
    ----------
    bandwidth:
        ``float``, ``'cv'`` or ``'silverman'``.
    local_linear:
        Local-linear (default) or Nadaraya-Watson.
    max_samples:
        Row budget for the ``d > 3`` backend.
    min_count:
        ``support`` is False where fewer than this many observed transitions
        lie within ``2h`` of the query. In 2D that disc is the area of one bin
        cell at the same smoothing scale, so this is rung 1's ``min_count``
        carried over. (Why not the effective sample size, or ``sum w``: see
        :mod:`dfi.estimators._kernel` -- both claimed support far from any
        data.)
    cv_folds, cv_points:
        Folds and number of candidate bandwidths.
    """

    key = "kernel"
    label = "Kernel regression"
    color_slot = 1

    #: Above this dimension a lattice is impossible and direct sums are used.
    MAX_LATTICE_D = 3
    #: Row budget for cross-validation on the direct-sum backend.
    CV_ROWS = 30_000

    def __init__(self, bandwidth="cv", local_linear: bool = True,
                 max_samples: int | None = 200_000, min_count: float = 8.0,
                 cv_folds: int = 3, cv_points: int = 11, device=None,
                 seed: int = 0, **params):
        super().__init__(bandwidth=bandwidth, local_linear=local_linear,
                         **params)
        if isinstance(bandwidth, str) and bandwidth not in ("cv", "silverman"):
            raise ValueError(f"bandwidth must be a float, 'cv' or 'silverman', "
                             f"got {bandwidth!r}")
        self._bw_arg = bandwidth
        self.bandwidth = bandwidth if isinstance(bandwidth, str) \
            else float(bandwidth)
        self.local_linear = bool(local_linear)
        # Instance label, so the two variants stay distinct in every results
        # table and legend -- sweeps group rows by label.
        self.label = ("Kernel, local-linear" if self.local_linear
                      else "Kernel, Nadaraya-Watson")
        self.max_samples = max_samples
        self.min_count = float(min_count)
        self.cv_folds = int(cv_folds)
        self.cv_points = int(cv_points)
        self.device = device
        self.seed = int(seed)

        self.box = None
        self.dt = None
        self.backend_ = None     # 'lattice' or 'direct'
        self.cv_ = None          # dict: the CV curve, for plotting
        self.stride_ = 1         # block length used by the direct backend
        self.n_used_ = None      # rows the final fit is built from
        self._lat = None
        self.b_grid = None       # lattice backend: estimate at every node
        self.mass_grid = None    # transitions within 2h of every node
        self.x_train = None      # direct backend: the (compressed) data
        self.y_train = None
        self._cache = (None, None, None)

    # -- data --------------------------------------------------------------
    def _training_data(self, traj):
        """``(x, y, walker_id)``, compressed for the direct backend."""
        N = traj.n_walkers
        if self.backend_ == "direct" and self.max_samples \
                and traj.n_transitions > self.max_samples:
            want = int(np.ceil(traj.n_transitions / self.max_samples))
            # Cap the block so its RMS displacement stays under 10% of the
            # box. Measured from the increments, not from the D the simulator
            # was given -- an estimator should not be handed the answer.
            dx = traj.increments()
            step = float(np.sqrt(np.mean(np.sum(dx * dx, axis=-1))))
            cap = int(max(1, ((0.1 * float(np.min(traj.box.length))) / step) ** 2))
            self.stride_ = int(max(1, min(want, cap, traj.n_frames - 1)))
            if self.stride_ > 1:
                traj = traj.subsample_time(self.stride_)
        x, y = self.regression_data(traj)
        walker = np.repeat(np.arange(N), traj.n_frames - 1)
        if self.backend_ == "direct" and self.max_samples \
                and len(x) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            keep = np.sort(rng.choice(len(x), self.max_samples, replace=False))
            x, y, walker = x[keep], y[keep], walker[keep]
        return x, y, walker, traj.dt

    def default_bandwidth(self, x: np.ndarray) -> float:
        """Silverman's rule: ``h = sigma * n^(-1/(d+4))``.

        ``sigma`` is the robust spread of the positions (the smaller of the
        standard deviation and ``IQR/1.349``, averaged over axes). A starting
        point and a reference line -- see the class docstring for why it is
        not the answer for regression.
        """
        x = np.asarray(x, float)
        n, d = x.shape
        sd = x.std(axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        sigma = float(np.mean(np.minimum(sd, (q75 - q25) / 1.349)))
        return sigma * n ** (-1.0 / (d + 4.0))

    def _candidates(self, h0: float, floor: float = 0.0) -> np.ndarray:
        """Log-spaced bandwidths from 0.35x to 11x the Silverman value.

        Wide on purpose, and skewed upwards: for regression the optimum sits
        above the density-estimation rule, far above it for local-linear on a
        nearly linear field and in high ``d``. The ceiling is 40% of the box on
        a torus -- past that the minimum-image kernel is truncated so hard it
        stops being a local average -- and 25% on an open box, where the lattice
        padding grows with ``h``.
        """
        L = float(np.min(self.box.length))
        cap = (0.4 if self.box.periodic else 0.25) * L
        hs = h0 * 2.0 ** np.linspace(-1.5, 3.5, self.cv_points)
        hs = np.clip(hs, max(floor, 1e-6), cap)
        return np.unique(np.round(hs, 12))

    def _walker_folds(self, n_walkers: int):
        rng = np.random.default_rng(self.seed + 17)
        k = max(2, min(self.cv_folds, n_walkers))
        return np.array_split(rng.permutation(n_walkers), k)

    def _pick(self, hs, loss, n_train_frac, d):
        """Minimum of the CV curve, refined by a parabola in ``log h``."""
        i = int(np.argmin(loss))
        h = float(hs[i])
        if 0 < i < len(hs) - 1:
            lx = np.log(hs[i - 1:i + 2])
            c2, c1, _ = np.polyfit(lx, loss[i - 1:i + 2], 2)
            if c2 > 0:
                h = float(np.exp(np.clip(-c1 / (2 * c2), lx[0], lx[-1])))
        return h, h * n_train_frac ** (1.0 / (d + 4.0))

    # -- fit ---------------------------------------------------------------
    def fit(self, traj):
        from ._kernel import Lattice

        self.box, d = traj.box, traj.d
        self.backend_ = "lattice" if d <= self.MAX_LATTICE_D else "direct"
        self._cache = (None, None, None)
        x, y, walker, self.dt = self._training_data(traj)
        self.n_used_ = len(x)
        h_pilot = self.default_bandwidth(x)
        self.h_silverman_ = h_pilot

        if isinstance(self._bw_arg, str) and self._bw_arg == "cv":
            hs = self._candidates(h_pilot)
        elif isinstance(self._bw_arg, str):
            hs = np.array([h_pilot])
        else:
            hs = np.array([float(self._bw_arg)])

        if self.backend_ == "lattice":
            self._lat = Lattice(self.box, float(hs.min()), float(hs.max()))
            hs = np.unique(np.maximum(hs, self._lat.h_floor))
            if self._bw_arg == "cv":
                h = self._cv_lattice(x, y, walker, hs, d)
            else:
                h = float(hs[0])
            fcnt, fsy = self._lat.bin(x, y)
            self.b_grid, _ = self._lat.solve(fcnt, fsy, h, self.local_linear)
            self.mass_grid = self._lat.ball_count(fcnt, 2.0 * h)
        else:
            h = (self._cv_direct(x, y, walker, hs, d)
                 if self._bw_arg == "cv" else float(hs[0]))
            self.x_train, self.y_train = x, y

        self.h_ = float(h)
        self.params["bandwidth"] = self.h_
        self._fitted = True
        return self

    def _cv_lattice(self, x, y, walker, hs, d):
        folds = self._walker_folds(int(walker.max()) + 1)
        loss = np.zeros((len(folds), len(hs)))
        for k, held in enumerate(folds):
            te = np.isin(walker, held)
            fcnt, fsy = self._lat.bin(x[~te], y[~te])
            xt, yt = x[te], y[te]
            for i, h in enumerate(hs):
                grid, _ = self._lat.solve(fcnt, fsy, h, self.local_linear)
                pred = self._lat.interpolate(grid, xt)
                loss[k, i] = np.mean(np.sum((yt - pred) ** 2, axis=-1))
        frac = 1.0 - 1.0 / len(folds)
        h_raw, h = self._pick(hs, loss.mean(0), frac, d)
        h = max(h, self._lat.h_floor)
        self.cv_ = {"h": hs, "loss": loss.mean(0), "loss_folds": loss,
                    "h_min_cv": h_raw, "h_selected": h, "n_rows": len(x)}
        return h

    def _cv_direct(self, x, y, walker, hs, d):
        from ._kernel import direct_sums

        rng = np.random.default_rng(self.seed + 29)
        n_cv = min(len(x), self.CV_ROWS)
        sel = rng.choice(len(x), n_cv, replace=False) if n_cv < len(x) \
            else np.arange(len(x))
        xc, yc, wc = x[sel], y[sel], walker[sel]
        folds = self._walker_folds(int(walker.max()) + 1)
        loss = np.zeros((len(folds), len(hs)))
        n_train = 0
        for k, held in enumerate(folds):
            te = np.isin(wc, held)
            if te.all() or not te.any():
                continue
            b, _ = direct_sums(xc[~te], yc[~te], xc[te], hs, box=self.box,
                               local_linear=self.local_linear,
                               device=self.device)
            loss[k] = np.mean(np.sum((yc[te][None] - b) ** 2, axis=-1),
                              axis=1)
            n_train += int((~te).sum())
        n_train /= len(folds)
        # Tuned on n_train rows, applied to len(x): rescale by the rate.
        h_raw, h = self._pick(hs, loss.mean(0), n_train / len(x), d)
        self.cv_ = {"h": hs, "loss": loss.mean(0), "loss_folds": loss,
                    "h_min_cv": h_raw, "h_selected": h, "n_rows": n_cv}
        return h

    # -- predict -----------------------------------------------------------
    def _evaluate(self, x):
        """``(b_hat, count within 2h)`` at ``x``, shapes ``(n, d)`` and ``(n,)``.

        On the direct backend each row is a block of ``stride_`` increments,
        so its weight in the count is ``stride_``: the count stays in units of
        observed transitions, comparable with rung 1's cell counts.
        """
        from ._kernel import direct_sums

        if self.backend_ == "lattice":
            return (self._lat.interpolate(self.b_grid, x),
                    self._lat.interpolate(self.mass_grid, x))
        # Direct sums are the expensive path, and the metrics call predict and
        # support back to back on the same points -- compute both once.
        import hashlib
        key = hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()
        if self._cache[0] == key:
            return self._cache[1], self._cache[2]
        b, mass = direct_sums(self.x_train, self.y_train, x, [self.h_],
                              box=self.box, local_linear=self.local_linear,
                              device=self.device)
        mass = mass[0] * self.stride_
        self._cache = (key, b[0], mass)
        return b[0], mass

    def predict(self, x):
        self._check_fitted()
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        b, _ = self._evaluate(x.reshape(-1, x.shape[-1]))
        return b.reshape(lead + (x.shape[-1],))

    def support(self, x):
        """False where fewer than ``min_count`` transitions lie within ``2h``.

        Deliberately a hard neighbourhood even though the estimate itself uses
        a smooth kernel: the question is not how the answer was weighted but
        whether any data near the query took part in it.
        """
        self._check_fitted()
        x = np.asarray(x, float)
        lead = x.shape[:-1]
        _, mass = self._evaluate(x.reshape(-1, x.shape[-1]))
        return (mass >= self.min_count).reshape(lead)
