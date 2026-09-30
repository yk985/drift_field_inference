"""Euler-Maruyama simulation of ``dx = b(x) dt + sqrt(2D) dW``.

The parameterisation is deliberately in terms of what an *observer* gets:

- ``n_steps`` recorded transitions at spacing ``dt`` (so ``n_steps + 1`` frames),
- integrated internally with ``substeps`` sub-steps of ``dt / substeps``.

Raising ``substeps`` makes the simulated path converge to the exact diffusion
while leaving the observation interval ``dt`` untouched. That separation is the
whole reason the discretisation-bias experiment is meaningful: with
``substeps=1`` an O(dt) error in the estimator is indistinguishable from an
O(dt) error in the *simulator*, and you would be measuring your own integrator.
Simulate with ``substeps`` large, observe at ``dt``, and the bias you plot is
the estimator's.
"""
from __future__ import annotations

import warnings

import numpy as np

from .fields.base import DriftField
from .fields.domain import Box
from .trajectories import Trajectories

__all__ = ["simulate", "sample_stationary", "check_step", "suggest_dt"]


# --------------------------------------------------------------------------
# Step-size diagnostics
# --------------------------------------------------------------------------

def _jacobian_scale(field: DriftField, n: int = 256, eps: float = 1e-4,
                    seed: int = 0) -> tuple:
    """RMS drift magnitude and RMS Jacobian *spectral* norm.

    The spectral norm, not the Frobenius norm: what controls Euler-Maruyama
    accuracy is how much ``b`` changes over a step, ``|J dx| <= ||J||_2 |dx|``.
    Frobenius overestimates that by up to ``sqrt(d)``, which would make the
    suggested step shrink spuriously with dimension.
    """
    rng = np.random.default_rng(seed)
    x = field.box.sample_uniform(n, rng)
    b = field.drift(x)
    J = np.empty((n, field.d, field.d))
    for j in range(field.d):
        s = np.zeros(field.d)
        s[j] = eps
        xp, xm = x + s, x - s
        if field.box.periodic:
            xp, xm = field.box.wrap(xp), field.box.wrap(xm)
        J[:, :, j] = (field.drift(xp) - field.drift(xm)) / (2.0 * eps)
    b_rms = float(np.sqrt(np.mean(np.sum(b ** 2, axis=-1))))
    j_rms = float(np.sqrt(np.mean(np.linalg.norm(J, ord=2, axis=(1, 2)) ** 2)))
    return b_rms, j_rms


def check_step(field: DriftField, dt: float, D: float, *, verbose: bool = False) -> dict:
    """Are ``dt`` and ``D`` sane for this field?

    Three numbers, all of which should be small:

    ``drift_cfl``    ``|b| dt / ell`` -- the deterministic step must be small
                     compared with the scale over which ``b`` varies.
    ``diffusive_cfl``  ``sqrt(2 d D dt) / ell`` -- so must the random step.
                     This is usually the binding constraint and the easiest to
                     get wrong: a walker that diffuses a full gradient scale
                     per step steps straight over the structure of ``b``, and
                     the stationary density comes out visibly wrong even though
                     the run looks perfectly stable.
    ``stability``    ``|grad b| dt`` -- above ~1 Euler-Maruyama is unstable and
                     the trajectory is nonsense, not merely inaccurate.

    ``ell = |b| / |grad b|`` is the field's own gradient scale, which works for
    random fields and closed-form ones alike.
    """
    b_rms, j_rms = _jacobian_scale(field)
    ell = b_rms / j_rms if j_rms > 0 else float(np.min(field.box.length))
    out = {
        "drift_length_scale": ell,
        "drift_cfl": b_rms * dt / ell,
        "diffusive_cfl": np.sqrt(2.0 * field.d * D * dt) / ell,
        "stability": j_rms * dt,
        "rms_drift": b_rms,
    }
    out["ok"] = bool(out["stability"] < 0.1 and out["drift_cfl"] < 0.1
                     and out["diffusive_cfl"] < 0.2)
    if verbose:
        for k, v in out.items():
            print(f"  {k:22s} {v}")
    return out


def suggest_dt(field: DriftField, D: float, *,
               drift_cfl: float = 0.05, diffusive_cfl: float = 0.15,
               stability: float = 0.05) -> float:
    """The largest ``dt`` meeting all three accuracy criteria at once.

    The three budgets are deliberately different sizes. The diffusive step is
    allowed to be a larger fraction of the gradient scale than the drift step,
    because its error is random and averages out along a trajectory, whereas
    the drift step's error is systematic and accumulates. Demanding the same
    tightness of both -- the naive choice -- drives ``dt`` down by orders of
    magnitude and buys nothing.
    """
    b_rms, j_rms = _jacobian_scale(field)
    ell = b_rms / j_rms if j_rms > 0 else float(np.min(field.box.length))
    dt_drift = drift_cfl * ell / max(b_rms, 1e-12)
    dt_diff = (diffusive_cfl * ell) ** 2 / max(2.0 * field.d * D, 1e-12)
    dt_stab = stability / max(j_rms, 1e-12)
    return float(min(dt_drift, dt_diff, dt_stab))


# --------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------

def _initial_positions(field, n_walkers, x0, rng, D):
    box = field.box
    if x0 is None:
        if box.periodic:
            return box.sample_uniform(n_walkers, rng)
        # Open domain: start at the potential minimum region, then rely on
        # burn_in to equilibrate. Starting uniformly in an open box would put
        # walkers where rho_ss is negligible and waste the burn-in.
        return np.tile(box.center, (n_walkers, 1)) + 0.1 * rng.standard_normal(
            (n_walkers, box.d)) * np.min(box.length)
    if isinstance(x0, str):
        if x0 == "stationary":
            return sample_stationary(field, n_walkers, D, rng=rng)
        if x0 == "uniform":
            return box.sample_uniform(n_walkers, rng)
        if x0 == "center":
            return np.tile(box.center, (n_walkers, 1))
        raise ValueError(f"unknown x0 preset {x0!r}")
    x0 = np.asarray(x0, float)
    if x0.ndim == 1:
        # A single point: every walker starts together. This is the setup for
        # the spreading-cloud animation, where you watch the initial delta
        # relax onto rho_ss.
        return np.tile(x0, (n_walkers, 1))
    if x0.shape != (n_walkers, box.d):
        raise ValueError(
            f"x0 has shape {x0.shape}, expected ({n_walkers}, {box.d}) or ({box.d},)")
    return x0.copy()


def _apply_boundary(x, box: Box, boundary: str):
    if boundary == "periodic":
        return box.wrap(x)
    if boundary == "reflecting":
        return box.reflect(x)
    return x


# --------------------------------------------------------------------------
# The integrator
# --------------------------------------------------------------------------

def simulate(
    field: DriftField,
    *,
    n_walkers: int = 256,
    n_steps: int = 1000,
    dt: float = 1e-2,
    D: float = 0.05,
    substeps: int = 1,
    burn_in: float = 0.0,
    x0=None,
    obs_noise: float = 0.0,
    boundary: str | None = None,
    seed: int | None = None,
    backend: str = "numpy",
    device: str | None = None,
    torch_dtype: str = "float32",
    store_dtype=np.float32,
    progress: bool = False,
    check: bool = True,
) -> Trajectories:
    """Simulate ``N`` walkers and record their positions.

    Parameters
    ----------
    field:
        The ground-truth drift field.
    n_walkers, n_steps:
        ``N`` walkers, ``n_steps`` recorded transitions (``n_steps + 1`` frames).
    dt:
        **Observation** interval -- the spacing of recorded frames.
    D:
        Diffusion coefficient. The noise amplitude is ``sqrt(2 D dt)``.
    substeps:
        Euler sub-steps per recorded interval. ``1`` means the recorded path is
        the Euler path; larger values converge the path to the true diffusion
        while keeping the observation interval fixed.
    burn_in:
        Time to run and discard before recording, so the recorded frames come
        from the stationary state. For a confined system, use at least a few
        relaxation times.
    x0:
        ``None`` (uniform on a torus, centered blob otherwise), a single point
        ``(d,)`` (all walkers together -- the spreading-cloud animation), an
        explicit ``(N, d)`` array, or one of ``'stationary'``, ``'uniform'``,
        ``'center'``.
    obs_noise:
        Per-coordinate measurement noise added to the recorded positions. The
        clean path is retained in ``Trajectories.x_true`` for diagnostics.
    boundary:
        ``'periodic'``, ``'reflecting'`` or ``'none'``; defaults to the box.
    backend:
        ``'numpy'`` or ``'torch'``. The torch path needs
        ``field.supports_torch`` and pays off for large ``N`` in high ``d``.
    torch_dtype:
        ``'float32'`` (default) or ``'float64'`` for the GPU integrator.
        float32 is typically several times faster on consumer GPUs, whose
        double-precision throughput is a small fraction of single. The cost is
        a per-step rounding error around ``1e-7`` of the position, which is
        five orders of magnitude below the diffusive step itself -- irrelevant
        next to the Monte Carlo error. Switch to float64 only to rule
        precision out when chasing a discrepancy.
    store_dtype:
        Storage precision for recorded positions. float32 halves memory and is
        far finer than any statistical error here; integration is always
        float64.

    Returns
    -------
    Trajectories
    """
    box = field.box
    if boundary is None:
        boundary = "periodic" if box.periodic else "none"
    if boundary not in ("periodic", "reflecting", "none"):
        raise ValueError(f"unknown boundary {boundary!r}")
    if boundary == "periodic" and not box.periodic:
        raise ValueError("periodic boundary requested but the box is not periodic")
    if substeps < 1:
        raise ValueError("substeps must be >= 1")

    n_bytes = n_walkers * (n_steps + 1) * box.d * np.dtype(store_dtype).itemsize
    if n_bytes > 4e9:
        raise MemoryError(
            f"recording would need {n_bytes/1e9:.1f} GB. Reduce n_walkers or "
            "n_steps, or record a coarser dt.")

    dt_int = dt / substeps
    if check:
        diag = check_step(field, dt_int, D)
        if not diag["ok"]:
            warnings.warn(
                f"integration step may be too coarse for this field: "
                f"stability={diag['stability']:.3g} (want <0.1), "
                f"drift_cfl={diag['drift_cfl']:.3g} (want <0.1), "
                f"diffusive_cfl={diag['diffusive_cfl']:.3g} (want <0.2). "
                f"Try dt={suggest_dt(field, D):.2g} or raise substeps.",
                RuntimeWarning, stacklevel=2)

    rng = np.random.default_rng(seed)
    x = _initial_positions(field, n_walkers, x0, rng, D)

    if backend == "torch":
        if not field.supports_torch:
            raise ValueError(
                f"{type(field).__name__} has no torch path (grid-interpolated "
                "fields are numpy-only); use backend='numpy', or a "
                "SpectralDriftField for the GPU sweeps.")
        out = _run_torch(field, x, n_steps, dt_int, substeps, D, boundary,
                         burn_in, seed, store_dtype, progress, device, torch_dtype)
    elif backend == "numpy":
        out = _run_numpy(field, x, n_steps, dt_int, substeps, D, boundary,
                         burn_in, rng, store_dtype, progress)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    traj = Trajectories(
        x=out, dt=dt, D=D, box=box, dt_integration=dt_int, burn_in=burn_in,
        field_name=getattr(field, "name", ""),
        meta={"substeps": substeps, "boundary": boundary, "seed": seed,
              "backend": backend, "field": field.summary()},
    )
    if obs_noise > 0:
        traj = traj.with_observation_noise(obs_noise, rng)
    return traj


def _progress(it, total, on):
    if not on:
        return it
    try:
        from tqdm.auto import tqdm
        return tqdm(it, total=total, leave=False)
    except ImportError:  # pragma: no cover
        return it


def _run_numpy(field, x, n_steps, dt_int, substeps, D, boundary, burn_in,
               rng, store_dtype, progress):
    box = field.box
    amp = np.sqrt(2.0 * D * dt_int)
    n, d = x.shape

    n_burn = int(round(burn_in / dt_int))
    for _ in _progress(range(n_burn), n_burn, progress and n_burn > 0):
        x = _apply_boundary(
            x + field.drift(x) * dt_int + amp * rng.standard_normal((n, d)),
            box, boundary)

    rec = np.empty((n, n_steps + 1, d), dtype=store_dtype)
    rec[:, 0] = x
    for t in _progress(range(n_steps), n_steps, progress):
        for _ in range(substeps):
            x = _apply_boundary(
                x + field.drift(x) * dt_int + amp * rng.standard_normal((n, d)),
                box, boundary)
        rec[:, t + 1] = x
    return rec


def _run_torch(field, x, n_steps, dt_int, substeps, D, boundary, burn_in,
               seed, store_dtype, progress, device, torch_dtype="float32"):
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtp = {"float32": torch.float32, "float64": torch.float64}[torch_dtype]
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed) if seed is not None else 0)

    box = field.box
    xt = torch.as_tensor(x, dtype=dtp, device=dev)
    lo = torch.as_tensor(box.lo, dtype=dtp, device=dev)
    L = torch.as_tensor(box.length, dtype=dtp, device=dev)
    amp = float(np.sqrt(2.0 * D * dt_int))
    n, d = xt.shape

    def bound(v):
        if boundary == "periodic":
            return lo + torch.remainder(v - lo, L)
        if boundary == "reflecting":
            y = torch.remainder(v - lo, 2.0 * L)
            y = torch.where(y > L, 2.0 * L - y, y)
            return lo + y
        return v

    def step(v):
        noise = torch.randn(n, d, generator=gen, dtype=dtp, device=dev)
        return bound(v + field.drift_torch(v) * dt_int + amp * noise)

    n_burn = int(round(burn_in / dt_int))
    for _ in _progress(range(n_burn), n_burn, progress and n_burn > 0):
        xt = step(xt)

    rec = torch.empty((n, n_steps + 1, d), dtype=torch.float32, device=dev)
    rec[:, 0] = xt.to(torch.float32)
    for t in _progress(range(n_steps), n_steps, progress):
        for _ in range(substeps):
            xt = step(xt)
        rec[:, t + 1] = xt.to(torch.float32)
    return rec.cpu().numpy().astype(store_dtype)


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------

def sample_stationary(field: DriftField, n: int, D: float, *,
                      rng: np.random.Generator | None = None,
                      method: str = "auto", burn_in: float | None = None,
                      dt: float | None = None) -> np.ndarray:
    """Draw ``n`` independent samples from ``rho_ss`` -- the snapshot dataset.

    This is the input to the density-only half of the project: positions with
    no time ordering, from which only the gradient part of the drift can ever
    be recovered.

    ``method='grid'`` samples the exact Boltzmann density on the field's own
    grid (available for :class:`GridDriftField`), which is fast and free of
    equilibration error. ``method='dynamic'`` runs independent walkers past
    equilibrium and takes the final frame, which works for any field.
    ``'auto'`` picks the grid route when it exists.
    """
    rng = rng or np.random.default_rng(0)
    has_grid = hasattr(field, "rho_ss_on_grid") and field.boltzmann_exact
    if method == "auto":
        method = "grid" if has_grid else "dynamic"

    if method == "grid":
        if not has_grid:
            raise ValueError(
                f"{type(field).__name__} has no exact grid density; "
                "use method='dynamic'")
        shape = field.U_grid.shape
        p = field.rho_ss_on_grid(D).ravel()
        p = p / p.sum()
        cells = rng.choice(p.size, size=n, p=p)
        idx = np.stack(np.unravel_index(cells, shape), axis=-1).astype(float)
        # Uniform jitter inside the chosen cell. The cell is far smaller than
        # any feature of rho_ss, so this smooths the density on a scale that is
        # negligible compared with the interpolation error already present.
        idx += rng.uniform(-0.5, 0.5, size=idx.shape)
        h = field.box.spacing(shape)
        x = field.box.lo + idx * h
        return field.box.wrap(x) if field.box.periodic else field.box.clip(x)

    if method == "dynamic":
        dt = dt or suggest_dt(field, D)
        if burn_in is None:
            b_rms, j_rms = _jacobian_scale(field)
            ell = b_rms / j_rms if j_rms > 0 else float(np.min(field.box.length))
            # Long enough to diffuse across a few correlation lengths.
            burn_in = 20.0 * ell ** 2 / max(2.0 * field.d * D, 1e-12)
        # Many independent walkers in a mesh-free field (d > 3): on the GPU
        # when the field supports it -- 500k walkers in 10D is hours in numpy.
        backend = "numpy"
        if getattr(field, "supports_torch", False):
            try:
                import torch
                if torch.cuda.is_available():
                    backend = "torch"
            except ImportError:
                pass
        traj = simulate(field, n_walkers=n, n_steps=1, dt=dt, D=D,
                        burn_in=burn_in, x0="uniform" if field.box.periodic else None,
                        seed=int(rng.integers(1 << 30)), check=False,
                        store_dtype=np.float64, backend=backend)
        return traj.x[:, -1]

    raise ValueError(f"unknown method {method!r}")
