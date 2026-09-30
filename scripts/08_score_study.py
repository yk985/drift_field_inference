"""Rung 5 -- what snapshots can and cannot tell you about the drift.

    python scripts/08_score_study.py                      # all stages
    python scripts/08_score_study.py --which entropy      # one stage

Writes to ``outputs/score/figures/``. The six per-estimator figures (on gradient
fields, every rung's walkers) come from the shared study script:
``03_local_estimator.py --estimators kde dsm`` -> ``outputs/score_kde``, ``score_dsm``.

``identifiability``
    One field, the rotation ``omega`` swept from 0 to 4 with ``field.with_omega``,
    which leaves ``rho_ss`` bit-identical. Trajectory methods (SFI, the
    local-linear kernel, the amortised CNN) fit the walkers; snapshot methods
    (KDE score, denoising score matching) fit independent samples of the -- same
    -- stationary density. Errors against the full drift, its gradient part and
    its rotational part.
``entropy``
    Entropy production ``sigma = <|b_rot|^2>_rho / D`` estimated from one dataset
    alone: the rotational part is what is left of a trajectory estimate of the
    drift after subtracting ``D`` times a snapshot estimate of the score, both
    from the same walkers. Against the exact value.
``positions``
    Snapshots as an idealisation: the score estimators refitted on the
    positions the walkers actually visited (correlated, not independent),
    against independent snapshots of the same count, and against the trajectory
    methods on the same walkers.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfi.estimators.amortised import AmortizedDrift
from dfi.estimators.basis import BasisProjection
from dfi.estimators.local import KernelRegression
from dfi.estimators.score import DenoisingScoreMatching, KDEScore
from dfi.metrics import drift_error, make_eval_set
from dfi.simulate import sample_stationary
from dfi.sweeps import SweepConfig, default_traj_cache, simulate_config
from dfi.viz import figure_caption, use_style
from dfi.viz import style as st
from dfi.viz.estimates import COARSE, cell_centres, quiver_panel, vectors_at

OUT = ROOT / "outputs" / "score"
FIG = OUT / "figures"
DATA = OUT / "data"
D = 0.3
OMEGAS = (0.0, 0.5, 1.0, 2.0, 4.0)
METHODS = {  # key: (label, colour slot, family)
    "sfi": ("SFI", 2, "trajectory"),
    "ll": ("LL kernel", 1, "trajectory"),
    "amort": ("Amortised CNN", 4, "trajectory"),
    "kde": ("KDE score", 5, "snapshot"),
    "dsm": ("Denoising score matching", 6, "snapshot"),
}


def color(k):
    return st.series_style(METHODS[k][1])["color"]


def save(fig, name, caption):
    width = max(60, int(fig.get_size_inches()[0] * 17))
    figure_caption(fig, "\n".join(textwrap.wrap(caption, width)))
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def run(omega, args, n_walkers=400, n_steps=3000):
    cfg = SweepConfig(d=2, n_walkers=n_walkers, n_steps=n_steps, D=D, omega=omega, seed=0)
    return simulate_config(cfg, traj_cache=args.traj_cache)


def fit(key, traj, field, snaps):
    if key == "sfi":
        return BasisProjection().fit(traj)
    if key == "ll":
        return KernelRegression().fit(traj)
    if key == "amort":
        return AmortizedDrift().fit(traj)
    if key == "kde":
        return KDEScore().fit(snaps, D, box=field.box)
    return DenoisingScoreMatching().fit(snaps, D, box=field.box)


# --------------------------------------------------------------------------
# identifiability
# --------------------------------------------------------------------------

def stage_identifiability(args):
    print("\n[identifiability] rotation swept, rho_ss held fixed")
    DATA.mkdir(parents=True, exist_ok=True)
    rows, fits_at_2 = [], {}
    snap_preds = {}
    for w in OMEGAS:
        field, traj, dt = run(w, args)
        snaps = sample_stationary(field, 500_000, D, rng=np.random.default_rng(5))
        ev = make_eval_set(field, D, n=12000, seed=991)
        for key in METHODS:
            t0 = time.time()
            est = fit(key, traj, field, snaps)
            r = drift_error(est, field, ev, use_support=False)
            rows.append(dict(omega=w, method=key, nrmse=r["nrmse"],
                             nrmse_grad=r["nrmse_grad"], nrmse_rot=r["nrmse_rot"],
                             rms_rot_true=r["rms_rot_true"], rms_true=r["rms_true"],
                             seconds=time.time() - t0))
            if METHODS[key][2] == "snapshot":
                snap_preds.setdefault(key, []).append(est.predict(ev.x))
            if w == 2.0:
                fits_at_2[key] = est
            print(f"  omega={w:<4g} {key:<6} nrmse {r['nrmse']:.3f}  grad "
                  f"{r['nrmse_grad']:.3f}  rot {r['nrmse_rot']:.3f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "identifiability.csv", index=False)
    same = {k: max(float(np.abs(p - v[0]).max()) for p in v) for k, v in snap_preds.items()}
    print("  snapshot predictions, max difference across omega: " + ", ".join(
        f"{k} {v:.1e}" for k, v in same.items()))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.0))
    for key in METHODS:
        g = df[df.method == key].sort_values("omega")
        dash = (4, 2) if METHODS[key][2] == "snapshot" else ()
        for ax, col in zip(axes, ("nrmse", "nrmse_grad", "nrmse_rot")):
            gg = g if col != "nrmse_rot" else g[g.omega > 0]
            ax.plot(gg.omega, gg[col], lw=2, marker="o", markersize=5, color=color(key),
                    dashes=dash, label=METHODS[key][0])
    titles = ("Against the full drift", "Against its gradient part",
              "Against its rotational part")
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel(r"rotation strength  $\omega$")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("normalised error")
    axes[0].axhline(1.0, color=st.active().ink_muted, lw=1, dashes=(3, 2))
    axes[2].axhline(1.0, color=st.active().ink_muted, lw=1, dashes=(3, 2))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    at4 = df[df.omega == 4.0].set_index("method")
    at0 = df[df.omega == 0.0].set_index("method")
    save(fig, "20_identifiability",
         f"One 2D field (the study's, seed 0), D = {D}; omega swept with with_omega, which "
         f"leaves rho_ss bit-identical. Trajectory methods (solid) fit 400 walkers x 3000 "
         f"steps; snapshot methods (dashed) fit 500k independent samples of rho_ss -- the "
         f"same density at every omega, and their predictions differ across omega by at "
         f"most " + ", ".join(f"{v:.0e} ({k})" for k, v in same.items()) + ": identical. "
         f"Left: against the full drift the snapshot methods climb from "
         + ", ".join(f"{at0.nrmse[k]:.2f}" for k in ("kde", "dsm")) + " at omega = 0 to "
         + ", ".join(f"{at4.nrmse[k]:.2f}" for k in ("kde", "dsm")) + " at omega = 4, "
         f"while SFI stays at {at0.nrmse['sfi']:.2f} -> {at4.nrmse['sfi']:.2f}. Middle: on "
         f"the gradient part nothing changes for them. Right: their rotational error is "
         f"1.0 at every omega -- they return no rotation at all, which is not a weakness "
         f"of the methods: the rotation is not in the data. Each component is normalised "
         f"by its own RMS. (The amortised CNN was trained with |omega| <= 1.5.)")

    # maps at omega = 2: rotational parts
    field, traj, dt = run(2.0, args)
    X, Y = cell_centres(field.box, COARSE)
    grad = lambda x: field.gradient_part(x)
    panels = [("true rotational part", lambda x: field.rotational_part(x))]
    for key in ("sfi", "dsm"):
        est = fits_at_2[key]
        panels.append((f"{METHODS[key][0]}: estimate minus true gradient part",
                       lambda x, e=est: e.predict(x) - grad(x)))
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    tx, ty = vectors_at(panels[0][1], X, Y)
    vmax = float(np.percentile(np.hypot(tx, ty), 99))
    for ax, (title, fn) in zip(axes, panels):
        vx, vy = vectors_at(fn, X, Y)
        quiver_panel(ax, field.box, X, Y, vx, vy, vmax=vmax, title=title)
    fig.tight_layout()
    save(fig, "21_rotation_maps",
         f"omega = 2. Left: the rotational part of the drift, omega A grad U -- "
         f"divergence-free and everywhere orthogonal to the gradient, so it leaves rho_ss "
         f"untouched. Middle: what SFI recovers beyond the gradient part, from the "
         f"trajectories. Right: the same for denoising score matching from snapshots -- "
         f"nothing, on one arrow scale. The circulation is visible only in how walkers "
         f"move, never in where they are.")


# --------------------------------------------------------------------------
# entropy production
# --------------------------------------------------------------------------

def stage_entropy(args):
    print("\n[entropy] sigma = <|b - D score|^2> / D from one dataset")
    rows = []
    for w in OMEGAS:
        field, traj, dt = run(w, args)
        true = field.entropy_production(D)
        # positions the walkers visited, as snapshots (subsampled in time)
        pos = traj.x[:, ::6].reshape(-1, 2)
        rng = np.random.default_rng(3)
        pts = pos[rng.choice(len(pos), 50_000, replace=False)]
        sfi = BasisProjection().fit(traj)
        dsm = DenoisingScoreMatching().fit(pos, D, box=field.box)
        b = sfi.predict(pts)
        est_raw = float(np.mean(np.sum((b - dsm.predict(pts)) ** 2, 1)) / D)
        est_oracle_score = float(np.mean(np.sum((b - D * field.score_ss(pts, D)) ** 2, 1)) / D)
        # SFI predicts its own noise-only squared error; the estimate above
        # carries it as a positive bias, so subtract it
        bias = sfi.error_bound() / D
        rows.append(dict(omega=w, true=true, raw=est_raw, debiased=est_raw - bias,
                         oracle_score=est_oracle_score, sfi_bound=bias,
                         n_positions=len(pos)))
        print(f"  omega={w:<4g} true {true:8.3f}   estimate {est_raw:8.3f}   debiased "
              f"{est_raw - bias:8.3f}   (with the exact score {est_oracle_score:8.3f})",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "entropy_production.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    th = st.active()
    ax = axes[0]
    ax.plot(df.omega, df.true, lw=2.5, color=th.ink, label="exact")
    ax.plot(df.omega, df.raw, lw=2, marker="o", color=color("sfi"),
            label="SFI drift − D × DSM score (same walkers)")
    ax.plot(df.omega, df.debiased, lw=2, marker="s", dashes=(4, 2), color=color("dsm"),
            label="the same, minus SFI's own predicted error")
    ax.plot(df.omega, df.oracle_score, lw=1.5, marker="^", dashes=(1, 1.5),
            color=th.ink_muted, label="SFI drift − D × exact score")
    ax.set_xlabel(r"rotation strength  $\omega$")
    ax.set_ylabel(r"entropy production rate  $\sigma$")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("Entropy production from one dataset")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax = axes[1]
    nz = df[df.omega > 0]
    ax.plot(nz.omega, nz.raw / nz.true, lw=2, marker="o", color=color("sfi"), label="estimate")
    ax.plot(nz.omega, nz.debiased / nz.true, lw=2, marker="s", dashes=(4, 2),
            color=color("dsm"), label="debiased")
    ax.axhline(1.0, color=th.ink_muted, lw=1, dashes=(3, 2))
    ax.set_xlabel(r"rotation strength  $\omega$")
    ax.set_ylabel("estimate / exact")
    ax.set_title("Relative accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    z = df[df.omega == 0].iloc[0]
    save(fig, "22_entropy_production",
         f"sigma = <|b_rot|^2>_rho / D, the dissipation that keeps the system out of "
         f"equilibrium, estimated from 400 walkers x 3000 steps and nothing else: the "
         f"drift from the increments (SFI), the score from the positions the same walkers "
         f"visited (denoising score matching), and their difference b - D grad log rho = "
         f"b_rot. At omega = 0 the truth is 0 and the raw estimate is {z.raw:.3f}: every "
         f"estimation error enters squared and positive. SFI predicts its own noise-level "
         f"error without the truth, and subtracting it leaves {z.debiased:.3f}. Relative "
         f"accuracy at omega = 1, 2, 4: " + ", ".join(
             f"{r.raw / r.true:.2f} (debiased {r.debiased / r.true:.2f})"
             for _, r in nz[nz.omega >= 1].iterrows()) + ".")


# --------------------------------------------------------------------------
# positions vs independent snapshots
# --------------------------------------------------------------------------

def stage_positions(args):
    print("\n[positions] snapshots from the walkers themselves")
    rows = []
    for N in (16, 64, 256, 1024):
        cfg = SweepConfig(d=2, n_walkers=N, n_steps=12000, D=D, seed=0)
        field, traj, dt = simulate_config(cfg, traj_cache=args.traj_cache)
        ev = make_eval_set(field, D, n=12000, seed=991, traj=traj)
        n = min(traj.n_transitions, 500_000)
        stride = max(1, traj.n_transitions // n)
        pos = traj.x[:, :-1:stride].reshape(-1, 2)[:n]
        iid = sample_stationary(field, len(pos), D, rng=np.random.default_rng(5))
        for src, snaps in (("walker positions", pos), ("independent", iid)):
            for M, key in ((KDEScore, "kde"), (DenoisingScoreMatching, "dsm")):
                est = M().fit(snaps, D, box=field.box)
                r = drift_error(est, field, ev, use_support=False)
                rows.append(dict(N=N, source=src, method=key, nrmse=r["nrmse"],
                                 n_snapshots=len(snaps), n_transitions=traj.n_transitions))
                print(f"  N={N:<5d} {src:<17} {key}: {r['nrmse']:.3f}", flush=True)
        for key in ("sfi", "ll"):
            est = fit(key, traj, field, None)
            r = drift_error(est, field, ev, use_support=False)
            rows.append(dict(N=N, source="increments", method=key, nrmse=r["nrmse"],
                             n_snapshots=0, n_transitions=traj.n_transitions))
            print(f"  N={N:<5d} increments        {key}: {r['nrmse']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "positions_vs_snapshots.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for key in ("sfi", "ll"):
        g = df[df.method == key]
        ax.plot(g.N, g.nrmse, lw=2, marker="o", color=color(key),
                label=f"{METHODS[key][0]}: increments")
    for key in ("kde", "dsm"):
        for src, dash in (("walker positions", ()), ("independent", (4, 2))):
            g = df[(df.method == key) & (df.source == src)]
            ax.plot(g.N, g.nrmse, lw=2, marker="s", dashes=dash, color=color(key),
                    label=f"{METHODS[key][0]}: {src}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("walkers (12000 steps each)")
    ax.set_ylabel("normalised error")
    ax.set_title("The same walkers: what their increments and their positions each give")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = df.pivot_table(index="N", columns=["method", "source"], values="nrmse")
    save(fig, "23_positions_vs_snapshots",
         f"The study's field and walkers (D = {D}, gradient field). Solid circles: trajectory "
         f"methods on the increments. Squares: score estimators on the positions those "
         f"walkers visited (solid; up to 500k, thinned in time), or on as many independent "
         f"samples of rho_ss (dashed). At 16 walkers: DSM on positions "
         f"{p[('dsm', 'walker positions')][16]:.3f}, on independent snapshots "
         f"{p[('dsm', 'independent')][16]:.3f}, SFI on increments {p[('sfi', 'increments')][16]:.3f}. "
         f"At 1024: {p[('dsm', 'walker positions')][1024]:.3f}, "
         f"{p[('dsm', 'independent')][1024]:.3f}, {p[('sfi', 'increments')][1024]:.3f}. "
         f"Positions of few walkers are correlated and cover rho_ss unevenly, which costs "
         f"the score methods their idealised advantage; with many walkers they converge.")


STAGES = {"identifiability": stage_identifiability, "entropy": stage_entropy,
          "positions": stage_positions}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--traj-cache", type=Path, default=default_traj_cache())
    args = ap.parse_args()
    use_style("light")
    t0 = time.time()
    for s in args.which:
        STAGES[s](args)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
