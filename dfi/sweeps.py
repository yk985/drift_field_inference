"""Running many configurations and collecting the errors into one table.

One :class:`SweepConfig` is one experiment: build a field, simulate it, fit
every estimator, score them all on the same evaluation set. :func:`run_sweep`
does that over a grid of configs and returns a tidy DataFrame -- one row per
(config, estimator) -- which is what every figure in ``dfi.viz.sweeps``
consumes.

Design decisions that keep the comparison honest:

- **All estimators see exactly the same data** within a config, and are scored
  on exactly the same evaluation points under the same measure. Anything else
  is not a comparison.
- **Snapshot estimators get a matched budget.** They are handed ``N*T``
  independent samples -- the same number of observations the trajectory
  methods get, not the same number of *walkers*. That is the generous reading,
  and it makes the identifiability result airtight: the snapshot method still
  cannot see circulation even with equal data.
- **Unimplemented estimators are recorded, not fatal.** A method whose ``fit``
  raises ``NotImplementedError`` yields a row with ``status='not_implemented'``
  and the sweep continues, so the harness is usable from day one with only the
  baselines and fills in as you write each rung.
- **Results are cached by config hash**, so a long sweep can be resumed and a
  new estimator can be added without recomputing the trajectories.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
import traceback
from dataclasses import dataclass, asdict, field as _dc_field
from pathlib import Path

import numpy as np

from .fields import random_grid_field, random_spectral_field
from .metrics import drift_error, make_eval_set
from .simulate import sample_stationary, simulate, suggest_dt

__all__ = ["SweepConfig", "sweep_grid", "run_sweep", "build_field",
           "simulate_config", "default_traj_cache"]

#: Bump when anything that changes a simulated path changes -- the simulator,
#: the field generators, the burn-in rule -- so cached trajectories from the old
#: code stop matching instead of being silently reused.
SIM_CACHE_VERSION = 1


@dataclass(frozen=True)
class SweepConfig:
    """One experiment. Everything the result depends on lives here.

    ``dt=None`` asks :func:`dfi.simulate.suggest_dt` for a stable step, which
    is what you want when sweeping ``d`` or ``D`` -- a step that is fine for
    one configuration can be unstable for another, and a sweep that silently
    changes accuracy is worthless.
    """

    d: int = 2
    n_walkers: int = 512
    n_steps: int = 2000
    dt: float | None = None
    D: float = 0.3
    omega: float = 0.0
    obs_noise: float = 0.0
    substeps: int = 1
    seed: int = 0
    correlation_length: float = 0.45
    drift_scale: float = 1.0
    resolution: int = 128          # grid backend (d <= 3)
    n_features: int = 512          # spectral backend (any d)
    backend: str = "auto"          # 'grid', 'spectral', or 'auto'
    burn_in_tau: float = 8.0       # burn-in, in diffusive times across ell
    measure: str = "stationary"
    n_eval: int = 20000
    #: Score only where each estimator claims support (the default), or
    #: everywhere under the measure. Use False when comparing methods: masked
    #: scoring rewards abstention, since a method is then graded only on the
    #: points it chose to answer.
    use_support: bool = True

    @property
    def n_transitions(self) -> int:
        return self.n_walkers * self.n_steps

    def key(self) -> str:
        fields = asdict(self)
        # Fields added after caches existed are left out of the hash while at
        # their default, so every config that predates them keeps its key.
        if fields["use_support"] is True:
            del fields["use_support"]
        blob = json.dumps(fields, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


def sweep_grid(**axes) -> list:
    """Cartesian product of the named axes into a list of configs.

    ``sweep_grid(d=[2, 4, 6], n_walkers=[64, 256], seed=[0, 1])`` gives 12
    configs. Scalars are held fixed. Include a ``seed`` axis whenever you
    intend to draw error bars -- a single seed gives you a number with no
    uncertainty, which is not a result.
    """
    keys = list(axes)
    vals = [v if isinstance(v, (list, tuple, np.ndarray)) else [v]
            for v in axes.values()]
    return [SweepConfig(**dict(zip(keys, combo)))
            for combo in itertools.product(*vals)]


def build_field(cfg: SweepConfig):
    """Construct the ground-truth field for a config.

    Picks the grid backend up to ``d = 3`` (so it can be drawn) and the
    mesh-free spectral backend above, where an ``n^d`` grid is impossible. Both
    share the ``b = -grad U + omega A grad U`` construction, so results are
    comparable across the switch -- but say in the write-up where it happens,
    since it is a change of generative model even though it is not a change of
    structure.

    The field is always built at ``omega = 0`` and then given its circulation
    by :meth:`with_omega`. This is not cosmetic. The constructors normalise
    ``U`` so that ``RMS|b| = drift_scale``, which divides ``U`` by
    ``sqrt(1 + omega^2)`` -- so building directly at each ``omega`` would give
    a *different potential*, and therefore a different ``rho_ss``, at every
    point of an omega sweep. Measured at ``omega = 3``: the density's dynamic
    range collapses from 59 to 3.6, a total-variation distance of 0.25 from the
    ``omega = 0`` density. That would confound the identifiability experiment
    with exactly the effect it is trying to isolate. Going through
    ``with_omega`` holds ``U`` -- and hence ``rho_ss`` -- bit-identical across
    the sweep, so the only thing that changes is the circulation.

    The consequence is that ``RMS|b|`` grows like ``sqrt(1 + omega^2)`` along
    the sweep. That is physics, not drift: circulation genuinely speeds the
    walkers up, and errors are reported relative to each field's own scale.
    """
    backend = cfg.backend
    if backend == "auto":
        backend = "grid" if cfg.d <= 3 else "spectral"
    if backend == "grid":
        field = random_grid_field(
            d=cfg.d, resolution=cfg.resolution, omega=0.0, seed=cfg.seed,
            correlation_length=cfg.correlation_length,
            drift_scale=cfg.drift_scale)
    elif backend == "spectral":
        field = random_spectral_field(
            d=cfg.d, n_features=cfg.n_features, omega=0.0, seed=cfg.seed,
            correlation_length=cfg.correlation_length,
            drift_scale=cfg.drift_scale)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return field if cfg.omega == 0.0 else field.with_omega(cfg.omega)


def default_traj_cache() -> Path:
    """Where cached trajectories live unless told otherwise.

    Outside the project on purpose. A full study caches a gigabyte or two of
    positions, which does not belong in a synced or versioned folder. Override
    with the ``DFI_TRAJ_CACHE`` environment variable.
    """
    import os
    env = os.environ.get("DFI_TRAJ_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "dfi" / "trajectories"


def _sim_key(cfg: SweepConfig, sim_backend: str) -> str:
    """Everything a simulated path depends on -- and nothing else.

    Excludes ``n_eval``, ``measure`` and ``use_support``, which change how a
    fit is *scored*, not what the walkers did. So every scoring variant of a
    config, and every estimator ever run on it, reads the same trajectory.
    """
    f = asdict(cfg)
    for k in ("n_eval", "measure", "use_support"):
        f.pop(k)
    f["_backend"] = sim_backend
    f["_version"] = SIM_CACHE_VERSION
    blob = json.dumps(f, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def simulate_config(cfg: SweepConfig, *, sim_backend: str = "numpy",
                    traj_cache: str | Path | None = None, verbose: bool = False):
    """Build the field and the trajectories for a config: ``(field, traj, dt)``.

    With ``traj_cache`` set, a trajectory simulated before is loaded instead
    of simulated again. That is exact, not approximate: the simulator is
    seeded, so a re-simulation would reproduce the same paths bit for bit --
    the cache only skips the cost. Its real value is for comparisons: every
    estimator run on a config, today or after the next rung is written, is
    fitted to literally the same walkers.
    """
    field = build_field(cfg)
    dt = cfg.dt if cfg.dt is not None else suggest_dt(field, cfg.D)
    path = None
    if traj_cache is not None:
        path = Path(traj_cache) / f"{_sim_key(cfg, sim_backend)}.npz"
        if path.exists():
            from .trajectories import Trajectories
            traj = Trajectories.load(path)
            if traj.x.shape[:2] == (cfg.n_walkers, cfg.n_steps + 1):
                if verbose:
                    print(f"    (trajectories from cache {path.name})")
                return field, traj, dt
    traj = simulate(field, n_walkers=cfg.n_walkers, n_steps=cfg.n_steps,
                    dt=dt, D=cfg.D, substeps=cfg.substeps,
                    burn_in=_burn_in_time(field, cfg), obs_noise=cfg.obs_noise,
                    seed=cfg.seed, backend=sim_backend, check=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        traj.save(tmp, compress=False)
        tmp.replace(path)              # atomic: never leave a half-written file
    return field, traj, dt


def _burn_in_time(field, cfg: SweepConfig) -> float:
    from .simulate import _jacobian_scale
    b, j = _jacobian_scale(field)
    ell = b / j if j > 0 else float(np.min(field.box.length))
    return cfg.burn_in_tau * ell ** 2 / max(2.0 * cfg.d * cfg.D, 1e-12)


def _eval_key(cfg: SweepConfig) -> tuple:
    """Everything the *evaluation points* depend on -- and nothing else.

    Deliberately excludes ``n_walkers``, ``n_steps``, ``dt``, ``substeps`` and
    ``obs_noise``: those change what the estimator is fed, not where it should
    be scored. Two budgets on the same field must be judged at the same points
    or the comparison between them is not a comparison.
    """
    return (cfg.d, cfg.seed, cfg.backend, cfg.resolution, cfg.n_features,
            cfg.correlation_length, cfg.drift_scale, cfg.omega, cfg.D,
            cfg.n_eval, cfg.measure)


def run_one(cfg: SweepConfig, estimator_factories: dict, *,
            sim_backend: str = "numpy", verbose: bool = True,
            eval_cache: dict | None = None,
            traj_cache: str | Path | None = None) -> list:
    """Run a single config against every estimator. Returns a list of rows."""
    t0 = time.time()
    field, traj, dt = simulate_config(cfg, sim_backend=sim_backend,
                                      traj_cache=traj_cache, verbose=verbose)
    t_sim = time.time() - t0

    # For a mesh-free field there is no exact density to sample, so drawing
    # evaluation points means running a burn-in for all of them -- which for
    # d >= 4 costs more than the sweep itself. Reuse them across the configs
    # that share a field and a measure.
    ekey = _eval_key(cfg)
    x_eval = eval_cache.get(ekey) if eval_cache is not None else None
    evalset = make_eval_set(field, cfg.D, n=cfg.n_eval, measure=cfg.measure,
                            seed=cfg.seed + 9999, traj=traj, x=x_eval)
    if eval_cache is not None:
        eval_cache[ekey] = evalset.x

    snapshots = None
    base = dict(asdict(cfg))
    base.update({"dt_actual": dt, "n_transitions": cfg.n_transitions,
                 "sim_seconds": t_sim, "config_key": cfg.key()})

    rows = []
    for name, factory in estimator_factories.items():
        row = dict(base, estimator_name=name)
        t1 = time.time()
        try:
            est = factory()
            if getattr(est, "uses_dynamics", True):
                try:
                    est.fit(traj, field=field)      # OracleDrift wants the field
                except TypeError:
                    est.fit(traj)
            else:
                if snapshots is None:
                    # Matched observation budget, not matched walker count.
                    snapshots = sample_stationary(
                        field, min(cfg.n_transitions, 500_000), cfg.D,
                        rng=np.random.default_rng(cfg.seed + 5))
                    if cfg.obs_noise > 0:
                        # Snapshots are measured positions too: the same
                        # measurement noise the trajectories carry.
                        snapshots = field.box.wrap(
                            snapshots + cfg.obs_noise * np.random.default_rng(
                                cfg.seed + 6).standard_normal(snapshots.shape))
                est.fit(snapshots, cfg.D, box=field.box)
            scores = drift_error(est, field, evalset,
                                 use_support=cfg.use_support)
            # drift_error's `n_eval` counts the points actually scored; left
            # under that name it overwrote the config field of the same name,
            # so a row stopped describing the experiment that produced it.
            scores["n_scored"] = scores.pop("n_eval")
            row.update(scores)
            # Resolved hyperparameters, prefixed so they cannot collide with a
            # config field. Worth recording because several of them are chosen
            # from the data -- BinnedKramersMoyal's bin count, for one -- and a
            # results table that cannot say what was actually run is not a
            # result, it is a number.
            row.update({f"param_{k}": v for k, v in
                        getattr(est, "params", {}).items()})
            # Estimators that can say something about their own fit -- a
            # condition number, a predicted error -- get it recorded beside the
            # measured one, so the two can be plotted against each other.
            if hasattr(est, "diagnostics"):
                row.update({f"diag_{k}": v for k, v in est.diagnostics().items()})
            row["status"] = "ok"
        except NotImplementedError:
            row["status"] = "not_implemented"
        except Exception as exc:                      # noqa: BLE001
            row["status"] = "error"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"    ! {name} failed: {row['error_message']}")
                traceback.print_exc(limit=2)
        row["fit_seconds"] = time.time() - t1
        rows.append(row)

    if verbose:
        ok = [r for r in rows if r.get("status") == "ok"]
        best = min(ok, key=lambda r: r["nrmse"], default=None)
        msg = (f"  d={cfg.d} N={cfg.n_walkers} T={cfg.n_steps} "
               f"omega={cfg.omega:g} sigma={cfg.obs_noise:g} "
               f"[{cfg.n_transitions:,} transitions, {t_sim:.1f}s]")
        if best:
            msg += f"  best: {best['estimator_name']} nrmse={best['nrmse']:.4f}"
        n_missing = sum(r["status"] == "not_implemented" for r in rows)
        if n_missing:
            msg += f"  ({n_missing} not implemented)"
        print(msg)
    return rows


def run_sweep(configs, estimator_factories: dict, *, cache: str | Path | None = None,
              sim_backend: str = "numpy", verbose: bool = True,
              traj_cache: str | Path | None = None):
    """Run a list of configs and return a tidy DataFrame.

    ``cache`` is a CSV path. Configs already present (matched on
    ``config_key`` and ``estimator_name``) are skipped and reloaded, so a sweep
    can be interrupted, extended with new estimators, or resumed on another
    day without recomputing.

    ``traj_cache`` is a directory of saved trajectories (see
    :func:`simulate_config`). The CSV cache skips *estimators already scored*;
    the trajectory cache skips *simulations already run*, so adding a new
    estimator to an old sweep costs only its fits.
    """
    import pandas as pd

    done, rows, eval_cache = None, [], {}
    if cache is not None:
        cache = Path(cache)
        if cache.exists():
            done = pd.read_csv(cache)
            if verbose:
                print(f"  resuming from {cache} ({len(done)} rows)")

    for i, cfg in enumerate(configs, 1):
        k = cfg.key()
        if done is not None and "config_key" in done:
            # Only a *successful* run counts as cached. A row recorded as
            # 'not_implemented' or 'error' must be retried, or every rung you
            # implement would stay permanently cached as missing -- which is
            # exactly the workflow this project is built around.
            rows_k = done[done.config_key == k]
            if "status" in rows_k:
                rows_k = rows_k[rows_k.status == "ok"]
            have = set(rows_k["estimator_name"])
            todo = {n: f for n, f in estimator_factories.items() if n not in have}
            if not todo:
                if verbose:
                    print(f"  [{i}/{len(configs)}] cached  {k}")
                continue
        else:
            todo = estimator_factories
        if verbose:
            print(f"  [{i}/{len(configs)}]", end=" ")
        new = run_one(cfg, todo, sim_backend=sim_backend, verbose=verbose,
                      eval_cache=eval_cache, traj_cache=traj_cache)
        rows.extend(new)
        if cache is not None:
            df_new = pd.DataFrame(new)
            if done is not None and "config_key" in done:
                # Drop the stale rows we have just recomputed, so a retried
                # 'not_implemented' does not survive beside its 'ok' replacement.
                stale = (done.config_key == k) & done.estimator_name.isin(todo)
                done = done[~stale]
            df_all = pd.concat([done, df_new]) if done is not None else df_new
            cache.parent.mkdir(parents=True, exist_ok=True)
            df_all.to_csv(cache, index=False)
            done = df_all

    if done is None:
        return pd.DataFrame(rows)
    # Return the configs that were asked for -- not the whole cache file. A
    # cache accumulates every config ever run against it (a --quick pass and a
    # full pass, say), and handing all of it back makes the caller's groupby
    # silently average across budgets it never requested. That happened: the
    # rung-1 sweep figure once mixed n_steps=1000 and n_steps=4000 rows.
    wanted = {c.key() for c in configs}
    return done[done.config_key.isin(wanted)].reset_index(drop=True)
