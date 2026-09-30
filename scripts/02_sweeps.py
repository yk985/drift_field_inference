"""Run the estimator comparison sweeps and draw the comparison figures.

This works **today**, with only the `ZeroDrift` and `OracleDrift` baselines: it
produces the plots with the null line at 1.0 and the oracle at 0, so you can see
the axes and the layout before writing an estimator. As each rung is
implemented, its curve appears automatically -- nothing here needs editing.

    python scripts/02_sweeps.py --quick                # small, ~1 min
    python scripts/02_sweeps.py --which budget omega   # pick sweeps
    python scripts/02_sweeps.py --all                  # everything, slow

Results are cached in ``outputs/data/sweep_*.csv`` and keyed by config hash, so
runs resume and adding a new estimator does not recompute the trajectories.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfi.estimators import REGISTRY
from dfi.sweeps import SweepConfig, run_sweep, sweep_grid
from dfi.viz import use_style, figure_caption
from dfi.viz.sweeps import (plot_error_curves, plot_identifiability,
                            plot_phase_diagram)

FIG = ROOT / "outputs" / "figures"
DATA = ROOT / "outputs" / "data"

# Every estimator in the registry, plus the two calibration baselines. Methods
# that are not implemented yet are recorded as such and simply do not appear on
# the plots -- no edits needed here as you fill them in.
def all_estimators():
    est = {"zero": lambda: REGISTRY["zero"](),
           "oracle": lambda: REGISTRY["oracle"]()}
    for key in ("binned_km", "kernel", "sfi", "nn", "kde_score", "dsm"):
        est[key] = (lambda k=key: REGISTRY[k]())
    return est


def save(fig, name, caption=None):
    if caption:
        figure_caption(fig, caption)
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def implemented(df):
    """Rows for estimators that actually ran, minus the oracle."""
    ok = df[(df.status == "ok") & (df.estimator_name != "oracle")]
    return ok


def sweep_budget(est, quick):
    """Error against total observed transitions N*T -- the master curve."""
    print("\n[budget] error vs N*T")
    walkers = [32, 128, 512] if quick else [16, 64, 256, 1024, 4096]
    cfgs = sweep_grid(d=2, n_walkers=walkers, n_steps=[500 if quick else 2000],
                      D=0.3, omega=0.0, seed=[0] if quick else [0, 1, 2])
    df = run_sweep(cfgs, est, cache=DATA / "sweep_budget.csv")
    ok = implemented(df)
    if ok.empty:
        print("  (no estimator implemented yet -- skipping figure)")
        return
    agg = (ok.groupby(["estimator_name", "label", "n_transitions"])
             .nrmse.agg(["mean", "min", "max"]).reset_index()
             .rename(columns={"mean": "nrmse"}))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_error_curves(agg, ax, x="n_transitions", y="nrmse",
                      band=("min", "max"),
                      title="Error against total observed transitions")
    save(fig, "20_sweep_budget",
         "2D random field, D=0.3, omega=0. Band spans seeds. The dashed guide "
         "is the Monte Carlo rate n^-1/2; a curve leaving it has hit its bias "
         "floor.")


def sweep_dt(est, quick):
    """Discretisation bias: error against the observation interval."""
    print("\n[dt] error vs sampling interval")
    base = SweepConfig(d=2, n_walkers=256, n_steps=2000, D=0.3)
    from dfi.sweeps import build_field
    from dfi.simulate import suggest_dt
    dt0 = suggest_dt(build_field(base), base.D)
    cfgs = sweep_grid(d=2, n_walkers=256, n_steps=[1000 if quick else 3000],
                      D=0.3, dt=[dt0 * m for m in (1, 2, 4, 8, 16)],
                      substeps=8, seed=0)
    df = run_sweep(cfgs, est, cache=DATA / "sweep_dt.csv")
    ok = implemented(df)
    if ok.empty:
        print("  (no estimator implemented yet -- skipping figure)")
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_error_curves(ok.rename(columns={"dt_actual": "dt"}), ax, x="dt",
                      y="nrmse", slope_guide=1.0,
                      title="Discretisation bias")
    save(fig, "21_sweep_dt",
         "substeps=8 throughout, so the underlying path is near-exact and the "
         "trend is the estimator's O(dt) bias, not the integrator's.")


def sweep_noise(est, quick):
    """Measurement noise: a bias of sigma^2 grad log rho / dt in the target."""
    print("\n[noise] error vs observation noise")
    cfgs = sweep_grid(d=2, n_walkers=256, n_steps=[1000 if quick else 3000],
                      D=0.3, obs_noise=[0.0, 0.002, 0.005, 0.01, 0.02], seed=0)
    df = run_sweep(cfgs, est, cache=DATA / "sweep_noise.csv")
    ok = implemented(df)
    if ok.empty:
        print("  (no estimator implemented yet -- skipping figure)")
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_error_curves(ok, ax, x="obs_noise", y="nrmse", logx=False,
                      slope_guide=None, title="Measurement noise")
    save(fig, "22_sweep_noise",
         "Position noise biases dx/dt by sigma^2 grad log rho / dt -- a term "
         "that does not vanish as dt -> 0, so finer sampling makes this worse.")


def sweep_omega(est, quick):
    """The identifiability experiment. The headline result."""
    print("\n[omega] trajectory vs snapshot recovery")
    cfgs = sweep_grid(d=2, n_walkers=512, n_steps=[1000 if quick else 3000],
                      D=0.3, omega=[0.0, 0.5, 1.0, 2.0, 3.0], seed=0)
    df = run_sweep(cfgs, est, cache=DATA / "sweep_omega.csv")
    ok = implemented(df)
    if ok.empty or "nrmse_rot" not in ok:
        print("  (no estimator implemented yet -- skipping figure)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    plot_error_curves(ok, axes[0], x="omega", y="nrmse", logx=False,
                      slope_guide=None, title="Error against the full drift")
    plot_identifiability(ok, axes[1])
    fig.tight_layout()
    save(fig, "23_identifiability",
         "omega varies with U -- and therefore rho_ss -- held bit-identical. A "
         "snapshot method's rotational-direction error must rise linearly with "
         "omega while its gradient-direction error stays flat: the information "
         "is absent from its input, not merely hard to extract.")


def sweep_dimension(est, quick):
    """The phase diagram: which method wins, as a function of budget and d."""
    print("\n[dimension] phase diagram")
    dims = [2, 3, 4] if quick else [2, 3, 4, 6, 8, 10]
    walkers = [64, 256] if quick else [32, 128, 512, 2048]
    cfgs = sweep_grid(d=dims, n_walkers=walkers,
                      n_steps=[500 if quick else 2000], D=0.3, seed=0)
    df = run_sweep(cfgs, est, cache=DATA / "sweep_dimension.csv",
                   sim_backend="numpy")
    ok = implemented(df)
    have = set(ok.estimator_name)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    if {"sfi", "nn"} <= have:
        plot_phase_diagram(ok, ax, methods=("sfi", "nn"), key="estimator_name")
        save(fig, "24_phase_diagram",
             "Colour is the winning method, opacity its margin. The black line "
             "is the crossover: basis projection wins in low d where its "
             "closed form and analytic error bar are unbeatable; the network "
             "takes over once the basis size outgrows the data.")
    else:
        plt.close(fig)
        missing = {"sfi", "nn"} - have
        print(f"  (phase diagram needs both sfi and nn; missing {missing})")
        if ok.empty:
            return
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        plot_error_curves(ok, ax, x="d", y="nrmse", logx=False,
                          slope_guide=None, title="Error against dimension")
        save(fig, "24_error_vs_dimension")


SWEEPS = {"budget": sweep_budget, "dt": sweep_dt, "noise": sweep_noise,
          "omega": sweep_omega, "dimension": sweep_dimension}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="*", default=["budget", "omega"],
                    choices=list(SWEEPS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    args = ap.parse_args()

    use_style(args.theme)
    DATA.mkdir(parents=True, exist_ok=True)
    which = list(SWEEPS) if args.all else args.which

    print(f"registry: {sorted(REGISTRY)}")
    print("Estimators not yet implemented are recorded and skipped; their "
          "curves appear automatically once you write them.\n")

    for name in which:
        SWEEPS[name](all_estimators(), args.quick)

    print(f"\nDone. Figures in {FIG.relative_to(ROOT)}, "
          f"cached results in {DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
