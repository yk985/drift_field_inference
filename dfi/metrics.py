"""Measuring how wrong an estimate is -- and *in which direction* it is wrong.

Three decisions are baked in here, each of which changes the numbers a lot and
each of which is easy to get wrong quietly.

**1. Where you evaluate matters more than the metric you use.**
Averaging the error uniformly over the box counts regions the walkers never
visited, where every method is guessing. Averaging under ``rho_ss`` counts each
region in proportion to how much data supports it. The two can differ by an
order of magnitude on the double well. :class:`EvalSet` makes the choice
explicit and records it, so a number always comes with the measure it was taken
under.

**2. Normalise, or the numbers are meaningless across systems.**
All errors here are divided by ``RMS|b|`` under the same measure. So ``1.0``
means "as wrong as predicting zero", and any value near or above 1 means the
method has learned nothing -- which is a judgement a raw MSE cannot express.

**3. Decompose the error along the field's own structure.**
Because these fields are built as ``b = -grad U + omega A grad U`` with the two
terms orthogonal *pointwise*, an error vector can be split exactly into a
gradient-direction part and a rotational-direction part. That is what makes the
identifiability result quantitative: a snapshot method's error lives entirely in
the rotational direction and grows precisely like ``omega``, while its
gradient-direction error stays flat. Two numbers, one figure, argument settled.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np

from .fields.base import DriftField
from .simulate import sample_stationary

__all__ = ["EvalSet", "make_eval_set", "decompose_error", "drift_error",
           "error_vs_density", "compare"]


# --------------------------------------------------------------------------
# Where to evaluate
# --------------------------------------------------------------------------

@dataclass
class EvalSet:
    """Points at which to score an estimate, plus the measure they carry.

    ``weights`` sum to 1 and define the averaging measure. ``local_density`` is
    the sampling density of the *training* data at each point, which is what
    :func:`error_vs_density` bins against.
    """

    x: np.ndarray
    weights: np.ndarray
    measure: str = "stationary"
    local_density: np.ndarray | None = None
    meta: dict = _dc_field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.x)

    def __repr__(self):
        return f"<EvalSet n={self.n} measure={self.measure!r}>"


def make_eval_set(field: DriftField, D: float, *, n: int = 20000,
                  measure: str = "stationary", seed: int = 0,
                  traj=None, density_bins: int = 48,
                  x: np.ndarray | None = None) -> EvalSet:
    """Build the evaluation points.

    ``measure``:

    ``'stationary'``
        Points drawn from ``rho_ss``. The default and the honest one: it scores
        a method where the data actually is, which is where its answer will be
        used.
    ``'uniform'``
        Points uniform on the box. Harsher, and worth reporting alongside --
        the gap between the two *is* the extrapolation penalty, and a method
        with a big gap is one that only works where it was fed.

    Pass ``traj`` to attach the training data's local density to each point, so
    the error can afterwards be resolved against sample count.

    ``x`` supplies the evaluation points directly instead of drawing them.
    Two configurations that differ only in how much data the *estimator* saw
    should be scored at the same points anyway, and for a mesh-free field in
    high ``d`` drawing them means running a burn-in for every one of them --
    the dominant cost of a dimension sweep. :func:`dfi.sweeps.run_sweep` reuses
    the points across such configurations through this argument. The local
    density is still recomputed per ``traj``, since that is what changes.
    """
    rng = np.random.default_rng(seed)
    if x is not None:
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != field.box.d:
            raise ValueError(
                f"eval points have shape {x.shape}, expected (n, {field.box.d})")
    elif measure == "stationary":
        x = sample_stationary(field, n, D, rng=rng)
    elif measure == "uniform":
        x = field.box.sample_uniform(n, rng)
    else:
        raise ValueError(f"unknown measure {measure!r}")
    w = np.full(len(x), 1.0 / len(x))

    dens = None
    if traj is not None:
        dens = _local_density(traj, x, bins=density_bins)
    return EvalSet(x, w, measure, dens, {"D": D, "field": field.name, "n": n})


#: Largest histogram the density diagnostic will allocate, in cells.
_MAX_DENSITY_CELLS = 20_000_000


def _local_density(traj, x, bins: int = 48, *, k: int = 32,
                   max_points: int = 200_000) -> np.ndarray:
    """Training-sample count per unit volume at each evaluation point.

    Two estimators, picked by dimension, because the obvious one does not
    survive the sweep this project is built to run.

    In low ``d`` a histogram is exact, cheap and has no tuning knob worth
    arguing about. But it needs ``bins**d`` cells, and at the default 48 bins
    that is 5e16 cells by ``d = 10`` -- the diagnostic would die of the very
    curse of dimensionality it exists to document, and take the whole sweep
    down with it. So above the cell budget we switch to a **k-nearest-neighbour
    density**: the distance ``r_k`` from the query point to the ``k``-th
    nearest training sample gives ``rho = k / (n * V_d(r_k))``. It is noisier
    than a histogram and biased at small ``k``, but it costs ``n log n``
    regardless of ``d``, which is the only property that matters here.

    The volume constant is computed in log space: ``V_d`` for ``d = 10`` at the
    radii involved underflows a direct evaluation, and only ratios of densities
    are ever used downstream anyway.
    """
    box = traj.box
    d = box.d
    pts = traj.x.reshape(-1, d)

    if bins ** d <= _MAX_DENSITY_CELLS:
        edges = [np.linspace(box.lo[i], box.hi[i], bins + 1) for i in range(d)]
        H, _ = np.histogramdd(pts, bins=edges)
        idx = []
        for i in range(d):
            j = np.clip(np.digitize(x[:, i], edges[i]) - 1, 0, bins - 1)
            idx.append(j)
        cell_vol = float(np.prod(box.length / bins))
        return H[tuple(idx)] / cell_vol

    from scipy.spatial import cKDTree
    from scipy.special import gammaln

    if len(pts) > max_points:          # the kNN radius barely moves past this
        sel = np.random.default_rng(0).choice(len(pts), max_points,
                                              replace=False)
        pts = pts[sel]
    k = int(min(k, len(pts)))

    # cKDTree wants periodic data in [0, L); shifting by `lo` is all that is
    # needed once positions are wrapped. Without boxsize the walkers either
    # side of a face would look like distant neighbours and the density would
    # be understated in a shell around the whole boundary.
    if box.periodic:
        tree_pts = box.wrap(pts) - box.lo
        query = box.wrap(x) - box.lo
        # wrap can return exactly L for a coordinate at the face; nudge inside.
        L = box.length
        tree_pts = np.minimum(tree_pts, L * (1 - 1e-12))
        query = np.minimum(query, L * (1 - 1e-12))
        tree = cKDTree(tree_pts, boxsize=L)
    else:
        tree = cKDTree(pts)
        query = x
    r = tree.query(query, k=k)[0][:, -1]

    log_unit_ball = 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1.0)
    with np.errstate(divide="ignore"):
        log_vol = log_unit_ball + d * np.log(r)
    return np.where(r > 0, np.exp(np.log(k) - np.log(len(pts)) - log_vol), 0.0)


# --------------------------------------------------------------------------
# Error decomposition
# --------------------------------------------------------------------------

def decompose_error(field: DriftField, x: np.ndarray, b_hat: np.ndarray) -> dict:
    """Split the error vector into gradient / rotational / residual directions.

    At each point the true field supplies two orthogonal unit vectors: the
    gradient direction ``grad U / |grad U|`` and the rotational direction
    ``A grad U / |A grad U|``. Any error vector projects onto them, and
    whatever is left over is genuinely off-structure.

    Returns per-point signed components ``e_grad``, ``e_rot`` and the residual
    magnitude ``e_perp``, along with the reference scales they should be
    compared against.

    Reading the output: a method that recovers the potential perfectly but
    misses the circulation entirely -- which is what *every* snapshot method
    must do -- shows ``e_grad ~ 0`` and ``e_rot`` equal to the full rotational
    amplitude. That is a diagnosis, not just an error bar.
    """
    x = np.asarray(x, float)
    b_true = field.drift(x)
    err = np.asarray(b_hat, float) - b_true

    out = {"error": err, "b_true": b_true}
    try:
        g = field.grad_potential(x)
    except NotImplementedError:
        return out

    gn = np.linalg.norm(g, axis=-1, keepdims=True)
    ghat = np.divide(g, gn, out=np.zeros_like(g), where=gn > 1e-12)
    e_grad = np.sum(err * ghat, axis=-1)
    out["e_grad"] = e_grad
    out["scale_grad"] = gn[..., 0]

    rot = field.rotational_part(x)
    rn = np.linalg.norm(rot, axis=-1, keepdims=True)
    if np.any(rn > 1e-12):
        rhat = np.divide(rot, rn, out=np.zeros_like(rot), where=rn > 1e-12)
    else:
        # omega = 0: there is no rotational component, but we still want the
        # direction A grad U so the "did it hallucinate circulation?" question
        # has an answer at omega = 0 too.
        A = getattr(field, "A", None)
        if A is None:
            return out
        alt = g @ np.asarray(A).T
        an = np.linalg.norm(alt, axis=-1, keepdims=True)
        rhat = np.divide(alt, an, out=np.zeros_like(alt), where=an > 1e-12)
    e_rot = np.sum(err * rhat, axis=-1)
    out["e_rot"] = e_rot
    out["scale_rot"] = rn[..., 0]

    perp = err - e_grad[..., None] * ghat - e_rot[..., None] * rhat
    out["e_perp"] = np.linalg.norm(perp, axis=-1)
    return out


# --------------------------------------------------------------------------
# The headline numbers
# --------------------------------------------------------------------------

def drift_error(estimator, field: DriftField, evalset: EvalSet, *,
                use_support: bool = True) -> dict:
    """Score one fitted estimator against the truth.

    Returns a flat dict, ready to become one row of a results table:

    ``nrmse``
        ``RMS|b_hat - b| / RMS|b|`` under the evaluation measure. The headline
        number. 0 is perfect, 1 is as good as predicting zero.
    ``nrmse_grad``, ``nrmse_rot``
        The same, restricted to the gradient and rotational directions and
        normalised by each component's own RMS amplitude. These are the two
        numbers the identifiability figure plots.
    ``cosine``
        Mean alignment of estimated and true drift. Separates "right shape,
        wrong scale" (cosine ~1, nrmse large) from "wrong shape" -- a
        distinction the RMSE alone hides, and one that immediately reveals a
        missing ``1/dt`` or a factor of 2 in ``D``.
    ``scale``
        ``RMS|b_hat| / RMS|b|``. Systematic shrinkage towards zero is the
        signature of over-smoothing, and it shows up here as a value below 1
        while the cosine stays high.
    ``supported_fraction``
        Fraction of evaluation points the estimator claims to cover.
    """
    x, w = evalset.x, evalset.weights
    b_hat = np.asarray(estimator.predict(x), float)
    parts = decompose_error(field, x, b_hat)
    b_true, err = parts["b_true"], parts["error"]

    # Coverage is always *measured*; it is only *applied* when use_support.
    # Scoring on the supported points alone rewards abstention -- a method
    # that answers for 2% of the points is graded on those 2% -- so a
    # comparison between methods should pass use_support=False, where a point
    # a method declines still counts, at whatever it actually predicts there.
    claimed = np.ones(len(x), bool)
    try:
        claimed = np.asarray(estimator.support(x), bool)
    except NotImplementedError:
        pass
    mask = claimed if use_support else np.ones(len(x), bool)
    if not mask.any():
        raise ValueError(
            f"{estimator.describe()} claims support at none of the "
            f"{len(x)} evaluation points")
    ww = w[mask] / w[mask].sum()

    def rms(v):
        return float(np.sqrt(np.sum(ww * np.sum(v[mask] ** 2, axis=-1))))

    def rms1(v):
        return float(np.sqrt(np.sum(ww * v[mask] ** 2)))

    scale = rms(b_true)
    out = {
        "estimator": estimator.key,
        "label": estimator.label,
        "measure": evalset.measure,
        "n_eval": int(mask.sum()),
        "supported_fraction": float(claimed.mean()),
        "scored_on_support": bool(use_support),
        "rms_true": scale,
        "rmse": rms(err),
        "nrmse": rms(err) / scale if scale else np.nan,
        "scale_ratio": rms(b_hat) / scale if scale else np.nan,
    }
    dot = np.sum(b_hat * b_true, axis=-1)
    nn = (np.linalg.norm(b_hat, axis=-1) * np.linalg.norm(b_true, axis=-1))
    cos = np.divide(dot, nn, out=np.zeros_like(dot), where=nn > 1e-15)
    out["cosine"] = float(np.sum(ww * cos[mask]))

    if "e_grad" in parts:
        sg = rms1(parts["scale_grad"])
        out["nrmse_grad"] = rms1(parts["e_grad"]) / sg if sg else np.nan
    if "e_rot" in parts:
        sr = rms1(parts["scale_rot"])
        # At omega = 0 the rotational amplitude is zero, so normalise by the
        # gradient scale instead: the question there is "did it invent
        # circulation that is not present?", and the answer should be ~0.
        denom = sr if sr > 1e-12 else rms1(parts["scale_grad"])
        out["nrmse_rot"] = rms1(parts["e_rot"]) / denom if denom else np.nan
        out["rms_rot_true"] = sr
    if "e_perp" in parts:
        out["nrmse_perp"] = rms1(parts["e_perp"]) / scale if scale else np.nan
    return out


def error_vs_density(estimator, field: DriftField, evalset: EvalSet, *,
                     n_bins: int = 12) -> dict:
    """Error resolved against how much training data was locally available.

    The guide asks for estimator variance against local sample density rather
    than one global number, and this is that plot's data. For a local averaging
    method the normalised error should fall like ``n^-1/2``; seeing that
    exponent come out of the data is a strong check that the estimator is
    behaving as theory says, and a *departure* from it localises the problem
    (a flat tail means bias, not variance, dominates there).

    Returns bin centres, mean error per bin, and counts.
    """
    if evalset.local_density is None:
        raise ValueError(
            "this EvalSet has no local_density; build it with "
            "make_eval_set(..., traj=traj)")
    x = evalset.x
    b_hat = np.asarray(estimator.predict(x), float)
    err = np.linalg.norm(b_hat - field.drift(x), axis=-1)
    scale = float(np.sqrt(np.mean(np.sum(field.drift(x) ** 2, axis=-1))))

    dens = evalset.local_density
    live = dens > 0
    if not live.any():
        raise ValueError("no evaluation point has any training data nearby")
    edges = np.geomspace(dens[live].min(), dens[live].max(), n_bins + 1)
    idx = np.clip(np.digitize(dens, edges) - 1, 0, n_bins - 1)
    centres, means, counts = [], [], []
    for b in range(n_bins):
        m = live & (idx == b)
        if m.sum() < 5:
            continue
        centres.append(float(np.sqrt(edges[b] * edges[b + 1])))
        means.append(float(np.sqrt(np.mean(err[m] ** 2)) / scale))
        counts.append(int(m.sum()))
    return {"density": np.array(centres), "nrmse": np.array(means),
            "counts": np.array(counts), "estimator": estimator.key,
            "label": estimator.label}


def compare(estimators, field: DriftField, evalset: EvalSet, **kw):
    """Score several fitted estimators; returns a pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame([drift_error(e, field, evalset, **kw) for e in estimators])
