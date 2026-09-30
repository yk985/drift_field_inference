"""One estimator on its own: what it recovers, and where it stops.

The same study for each estimator, run on *identical* data so the figure sets
can be laid side by side:

    python scripts/03_local_estimator.py                          # binned KM
    python scripts/03_local_estimator.py --estimators nw ll       # both kernels
    python scripts/03_local_estimator.py --estimators nw --which dims
    python scripts/03_local_estimator.py --quick                  # layout check

Each estimator writes its figures to its own tree:

    binned  -> outputs/binned_km/     (Binned Kramers-Moyal, rung 1)
    nw      -> outputs/kernel_nw/     (kernel regression, Nadaraya-Watson)
    ll      -> outputs/kernel_ll/     (kernel regression, local-linear)
    sfi     -> outputs/sfi/           (basis projection, rung 3)
    nn      -> outputs/nn_mlp/        (neural network, MLP, rung 4)
    cnn     -> outputs/nn_cnn/        (neural network, CNN, rung 4; d <= 3)
    amort   -> outputs/cnn_amortised/ (pretrained CNN, rung 4b; 2D only, no bins/dims)
    kde     -> outputs/score_kde/     (KDE score from snapshots, rung 5)
    dsm     -> outputs/score_dsm/     (denoising score matching from snapshots, rung 5)

Snapshot estimators (rung 5) never see the trajectories. They get independent
samples from rho_ss, as many as the trajectories have transitions (capped at
500k, as ``dfi.sweeps.run_one`` does), and the known D.

Stages, in the order they build the argument:

``setup``    the test itself -- true field as a quiver, walkers running on it
``noise``    recovered field against the noise level D
``walkers``  recovered field against how many walkers were watched
``bins``     the smoothing dial (bin count, or bandwidth), and the error resolved
             against local data; for the networks, the training statistics
``dims``     d = 2, 3, 5, 10: where local averaging runs out
``sweeps``   walkers, D and measurement noise as error curves

**Same data, not merely the same settings.** Every simulation goes through
:func:`dfi.sweeps.simulate_config` with a trajectory cache, so all estimators --
now or after the next rung is written -- are fitted to the very same walkers.
Simulations are seeded, so even a cold cache reproduces the paths bit for bit;
the cache only removes the cost. Sweep scores live in one shared table,
``outputs/local_study/data/``, one row per (config, estimator).

Scores here are taken where each estimator claims support, with coverage
reported beside them -- rung 1's convention. ``outputs/kernel/`` compares the
methods under the stricter every-point scoring.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfi.estimators.basis import BasisProjection
from dfi.estimators.local import BinnedKramersMoyal, KernelRegression
from dfi.estimators.amortised import AmortizedDrift
from dfi.estimators.neural import NeuralDrift
from dfi.estimators.score import DenoisingScoreMatching, KDEScore
from dfi.simulate import sample_stationary
from dfi.metrics import drift_error, error_vs_density, make_eval_set
from dfi.sweeps import (SweepConfig, default_traj_cache, run_sweep,
                        simulate_config, sweep_grid)
from dfi.viz import (animate_diffusion, figure_caption, plot_trajectories,
                     use_style)
from dfi.viz import style as st
from dfi.viz.estimates import (COARSE, cell_centres, error_panel,
                                estimate_maps, human, quiver_panel,
                                row_colorbar, vectors_at)
from dfi.viz.sweeps import plot_error_curves, plot_error_vs_density

OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "local_study" / "data"


# --------------------------------------------------------------------------
# the estimators this study knows about
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    cli: str                    # name on the command line
    key: str                    # estimator_name in the shared results table
    folder: str                 # outputs/<folder>/
    title: str
    make: Callable
    kind: str                   # 'binned', 'kernel', 'basis', 'neural', 'amortised', 'snapshot'
    max_d: int = 10             # largest dimension the method can run in

    @property
    def kernel(self) -> bool:
        return self.kind == "kernel"

    @property
    def fig(self) -> Path:
        return OUTPUTS / self.folder / "figures"

    @property
    def anim(self) -> Path:
        return OUTPUTS / self.folder / "animations"

    def knob(self, est) -> str:
        """The smoothing this fit actually used, as a short label."""
        return {"kernel": lambda: f"h = {est.h_:.3f}",
                "binned": lambda: f"{est.bins} bins",
                "basis": lambda: f"{est.n_basis} functions",
                "neural": lambda: _neural_knob(est.params.get("grid"),
                                               est.history.get("refit_steps",
                                                               est.params["best_step"])),
                "amortised": lambda: "pretrained",
                "snapshot": lambda: (f"h = {est.h_:.3f}" if self.cli == "kde"
                                     else f"σ = {est.sigma_:.3f}"),
                }[self.kind]()

    def knob_row(self, row) -> str:
        """The same, from a results-table row."""
        return {"kernel": lambda: f"h = {row.param_bandwidth:.2f}",
                "binned": lambda: f"{int(row.param_bins)} bins",
                "basis": lambda: f"p = {int(row.param_n_basis)}",
                "neural": lambda: _neural_knob(
                    getattr(row, "param_grid", None), int(row.diag_refit_steps)),
                "amortised": lambda: "pretrained",
                "snapshot": lambda: (f"h = {row.param_bandwidth:.2f}" if self.cli == "kde"
                                     else f"σ = {row.param_sigma_eval:.3f}"),
                }[self.kind]()

    @property
    def smoothing(self) -> str:
        return {"kernel": "the bandwidth, chosen by cross-validation over walkers",
                "binned": "the bin count, from the N^(1/(d+2)) rule",
                "basis": "the Fourier basis size, chosen by cross-validation "
                         "over walkers",
                "neural": "the number of training steps, chosen by early "
                          "stopping on held-out walkers, then a refit on all of "
                          "them" + (" (and, for the CNN, the output grid, chosen "
                                    "the same way)" if self.cli == "cnn" else ""),
                "amortised": "none at fit time -- the network was trained once, on "
                             "3000 other simulated fields, and is only applied here",
                "snapshot": ("the bandwidth, by held-out score matching" if self.cli == "kde"
                             else "the noise level the score is read at, by held-out "
                                  "score matching") + "; fitted to independent snapshots "
                             "from rho_ss (as many as there are transitions, capped at "
                             "500k) and the known D, never to the trajectories"
                }[self.kind]


SPECS = {
    "binned": Spec("binned", "binned_km", "binned_km",
                   "Binned Kramers-Moyal", lambda: BinnedKramersMoyal(), "binned"),
    "nw": Spec("nw", "kernel_nw", "kernel_nw", "Kernel regression, Nadaraya-Watson",
               lambda: KernelRegression(local_linear=False), "kernel"),
    "ll": Spec("ll", "kernel_ll", "kernel_ll", "Kernel regression, local-linear",
               lambda: KernelRegression(local_linear=True), "kernel"),
    "sfi": Spec("sfi", "sfi", "sfi", "Basis projection (SFI)",
                lambda: BasisProjection(), "basis"),
    "nn": Spec("nn", "nn_mlp", "nn_mlp", "Neural network, MLP",
               lambda: NeuralDrift(arch="mlp"), "neural"),
    "cnn": Spec("cnn", "nn_cnn", "nn_cnn", "Neural network, CNN",
                lambda: NeuralDrift(arch="cnn"), "neural", max_d=3),
    "amort": Spec("amort", "cnn_amortised", "cnn_amortised", "Amortised CNN",
                  lambda: AmortizedDrift(), "amortised", max_d=2),
    "kde": Spec("kde", "kde_score", "score_kde", "KDE score (snapshots)",
                lambda: KDEScore(), "snapshot"),
    "dsm": Spec("dsm", "dsm", "score_dsm", "Denoising score matching (snapshots)",
                lambda: DenoisingScoreMatching(), "snapshot"),
}


SNAPSHOT_CAP = 500_000


def snapshots_for(traj, field, D, seed=0):
    """Independent rho_ss samples, matched to the transition count (capped)."""
    return sample_stationary(field, min(traj.n_transitions, SNAPSHOT_CAP), D,
                             rng=np.random.default_rng(seed + 5))


def fit_spec(spec, traj, field, D, seed=0):
    """Fit one estimator the way its family consumes data."""
    est = spec.make()
    if spec.kind == "snapshot":
        return est.fit(snapshots_for(traj, field, D, seed), D, box=field.box)
    return est.fit(traj)


def _neural_knob(grid, steps):
    g = grid if grid is not None and np.isfinite(float(grid)) else None
    return (f"{int(g)}² grid, {int(steps)} steps" if g is not None
            else f"{int(steps)} steps")


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def save(fig, spec: Spec, name, caption=None):
    if caption:
        # Wrap to the figure's own width. figure_caption draws a single
        # unwrapped line, and `bbox_inches='tight'` then widens the saved
        # image to contain it -- so a long caption silently pads the figure
        # with whitespace instead of overflowing, which is harder to notice.
        width = max(60, int(fig.get_size_inches()[0] * 17))
        figure_caption(fig, "\n".join(textwrap.wrap(caption, width)))
    spec.fig.mkdir(parents=True, exist_ok=True)
    p = spec.fig / f"{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def case(*, D, n_walkers, n_steps, args, d=2, seed=0):
    """Field and trajectories for one picture-stage run, through the cache.

    Built as a SweepConfig so a picture stage and a sweep that share settings
    share the trajectory too -- and so this study and any other script using
    :func:`simulate_config` see identical walkers.
    """
    cfg = SweepConfig(d=d, n_walkers=n_walkers, n_steps=n_steps, D=D,
                      seed=seed)
    field, traj, dt = simulate_config(cfg, traj_cache=args.traj_cache)
    return field, traj, dt


def score(field, est, D, traj, seed=0):
    ev = make_eval_set(field, D, n=12000, seed=seed + 991, traj=traj)
    return drift_error(est, field, ev)


def agg(df, x):
    ok = df[df.status == "ok"]
    return (ok.groupby([x, "label"]).nrmse.agg(["mean", "min", "max"])
              .reset_index().rename(columns={"mean": "nrmse"}))


# --------------------------------------------------------------------------
# stage 1 -- the test itself
# --------------------------------------------------------------------------

def stage_setup(args, specs):
    print("\n[setup] the test: field, walkers, and one fit")
    D, n_walkers, n_steps = 0.3, 400, 3000 if args.quick else 12000
    field, traj, dt = case(D=D, n_walkers=n_walkers, n_steps=n_steps, args=args)

    for spec in specs:
        est = fit_spec(spec, traj, field, D)
        row = score(field, est, D, traj)
        print(f"  {spec.cli:<6} {spec.knob(est):<12} nrmse {row['nrmse']:.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))
        box = field.box
        (EX, EY, ex, ey, sup), _, _ = estimate_maps(field, est)
        # Truth drawn at the same points as the estimate, so the outer panels
        # differ only by estimation error and not by resolution.
        tx, ty = vectors_at(field.drift, EX, EY)
        vmax = float(np.percentile(np.hypot(tx, ty), 99.0))
        where = (f"at {est.bins} bins/axis" if spec.kind == "binned"
                 else f"on a {EX.shape[0]}x{EX.shape[1]} grid")
        quiver_panel(axes[0], box, EX, EY, tx, ty, vmax=vmax, colorbar=True,
                     title=r"Ground truth  $b(x) = -\nabla U$" f"  ({where})")

        # A short window, not the whole run: fourteen paths of 12000 steps is
        # a hairball that hides both the paths and the field underneath.
        window = min(traj.n_frames, 260)
        plot_trajectories(traj, field, axes[1], n_show=10, background="quiver",
                          n_grid=COARSE, mute_background=0.5,
                          frame_range=(0, window), linewidth=1.2,
                          title=f"{n_walkers} walkers running "
                                f"(10 shown, first {window} steps)")
        for f_ in (axes[1].set_xticks, axes[1].set_yticks):
            f_([])
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")

        quiver_panel(axes[2], box, EX, EY, ex, ey, vmax=vmax, support=sup,
                     colorbar=True,
                     title=f"Recovered, {spec.knob(est)}  "
                           f"(nrmse {row['nrmse']:.3f})")
        fig.tight_layout()
        save(fig, spec, "01_setup",
             f"{spec.title}. 2D random field, D = {D}, dt = {dt:.2e} from "
             f"suggest_dt, {traj.n_transitions:,} observed transitions -- the "
             f"same walkers for every estimator in this study. Smoothing: "
             f"{spec.smoothing}. Arrows share one length scale across panels, "
             f"so a shrunken estimate reads as shrunken. Middle: paths coloured "
             f"by time over the same arrows, muted; only the opening window is "
             f"drawn.")

        if args.animate and spec.kind == "binned":
            spec.anim.mkdir(parents=True, exist_ok=True)
            print("  rendering the walker clip (this is the slow part)")
            animate_diffusion(
                traj, field, n_show=220, trail=26, background="quiver",
                n_grid=COARSE, panels=("density",),
                stride=max(1, n_steps // 300),
                out=str(spec.anim / "walkers.mp4"), fps=30,
                title="Walkers on the field the estimator must recover")


# --------------------------------------------------------------------------
# stages 2 and 3 -- recovery against D, and against N
# --------------------------------------------------------------------------

def _comparison_figure(spec, field, fits, *, hero_value, hero_label, hero_sub,
                       name, caption, col_title):
    """Two rows -- estimate, then error -- one column per configuration."""
    box = field.box
    TX, TY = cell_centres(box, COARSE)
    tx, ty = vectors_at(field.drift, TX, TY)
    vmax = float(np.percentile(np.hypot(tx, ty), 99.0))

    n = len(fits)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.15 * (n + 1), 6.6))
    quiver_panel(axes[0, 0], box, TX, TY, tx, ty, vmax=vmax,
                 title=f"Ground truth  (at {COARSE} bins)")
    st.hero(axes[1, 0], hero_value, hero_label, sub=hero_sub)

    packed = []
    for key, est, row in fits:
        panels, err, cov = estimate_maps(field, est)
        packed.append((key, est, row, panels, err, cov))
    pooled = np.concatenate([e[np.isfinite(e)].ravel()
                             for *_, e, _ in packed])
    # A robust upper limit, capped: one badly-estimated cell in the smallest
    # run can be several times |b|, and letting it set the scale would
    # compress every other panel into the first tenth of the ramp.
    err_max = float(min(np.percentile(pooled, 97), 2.0)) if pooled.size else 1.0

    ims = []
    for j, (key, est, row, (EX, EY, ex, ey, sup), err, cov) in enumerate(
            packed, 1):
        q = quiver_panel(axes[0, j], box, EX, EY, ex, ey, vmax=vmax,
                         support=sup, title=col_title(key, est, row))
        im = error_panel(axes[1, j], box, err, vmax=err_max,
                         title=f"nrmse {row['nrmse']:.3f}\n"
                               f"answers for {cov:.0%} of the box")
        ims.append((q, im))
    fig.tight_layout()
    # One colorbar per row, attached to the whole row: a colorbar on a single
    # axes shrinks only that axes and leaves the columns misaligned.
    row_colorbar(fig, axes[0, 1:], ims[0][0], r"$|b|$")
    row_colorbar(fig, axes[1, 1:], ims[0][1],
                 r"$|\hat b - b| \,/\, \mathrm{RMS}|b|$", extend="max")
    save(fig, spec, name, caption)


def _support_phrase(spec):
    abstain = ("; the estimator declines to answer there and the error map is "
               "blank, so a low nrmse beside a low coverage is a method "
               "answering only the easy questions")
    return {"binned": "Grey marks cells holding fewer than min_count increments"
                      + abstain,
            "kernel": "Grey marks points with fewer than min_count transitions "
                      "within two bandwidths" + abstain,
            "basis": "A basis fit is global and has no support threshold -- it "
                     "answers everywhere, including where no walker went. So "
                     "the nrmse, scored under rho_ss where the walkers are, can "
                     "be excellent while the error map shows large errors in "
                     "the parts of the box they never visited: at low D the "
                     "fitted Fourier series extrapolates across the gaps",
            "neural": "A network is a global fit with no support threshold -- it "
                      "answers everywhere, including where no walker went, with "
                      "whatever its architecture extrapolates to. So the nrmse, "
                      "scored under rho_ss where the walkers are, can be "
                      "excellent while the error map shows large errors in the "
                      "unvisited parts of the box",
            "snapshot": "A snapshot score has no support where no sample fell: grey "
                        "marks points with fewer than 8 samples within 2h (KDE; the "
                        "network answers everywhere)",
            "amortised": "The network answers everywhere, and where no walker went "
                         "its answer is what it learned to expect from the training "
                         "fields -- close to zero drift, shrunk, rather than an "
                         "extrapolation"}[spec.kind]


def stage_noise(args, specs):
    print("\n[noise] recovery against the noise level D")
    Ds = [0.05, 0.15, 0.4, 1.0]
    n_walkers, n_steps = 400, 3000 if args.quick else 12000
    runs = [(D,) + case(D=D, n_walkers=n_walkers, n_steps=n_steps, args=args)
            for D in Ds]
    for spec in specs:
        fits = []
        for D, field, traj, dt in runs:
            t0 = time.time()
            est = fit_spec(spec, traj, field, D)
            row = score(field, est, D, traj)
            fits.append((D, est, row))
            print(f"  {spec.cli:<6} D={D:<5g} {spec.knob(est):<12} "
                  f"nrmse={row['nrmse']:.3f}  "
                  f"covered={row['supported_fraction']:.0%}  "
                  f"[{time.time() - t0:.1f}s]")
        _comparison_figure(
            spec, runs[0][1], fits,
            hero_value=f"{n_walkers * n_steps / 1e6:.1f}M",
            hero_label="observed transitions",
            hero_sub=f"{n_walkers} walkers x {n_steps} steps, held fixed",
            name="02_recovery_vs_noise",
            col_title=lambda D, est, row, s=spec: f"$D = {D:g}$   ({s.knob(est)})",
            caption=f"{spec.title}. Same field, same walkers as every other "
                    f"estimator in this study, same observation budget "
                    f"throughout; only the noise level changes. "
                    f"{_support_phrase(spec)}. dt is re-chosen per D by suggest_dt, so "
                    f"the columns differ in elapsed time, not in data volume.")


def stage_walkers(args, specs):
    print("\n[walkers] recovery against how many walkers were watched")
    Ns = [16, 64, 256, 1024]
    D, n_steps = 0.3, 3000 if args.quick else 12000
    runs = [(N,) + case(D=D, n_walkers=N, n_steps=n_steps, args=args)
            for N in Ns]
    for spec in specs:
        fits = []
        for N, field, traj, dt in runs:
            est = fit_spec(spec, traj, field, D)
            row = score(field, est, D, traj)
            fits.append((N, est, row))
            print(f"  {spec.cli:<6} N={N:<5d} {spec.knob(est):<12} "
                  f"nrmse={row['nrmse']:.3f}  "
                  f"covered={row['supported_fraction']:.0%}")
        how = {"binned": "The bin count is chosen from the sample size, so the "
                         "estimate coarsens as the data thins -- the arrows get "
                         "blockier rather than noisier.",
               "kernel": "The bandwidth is chosen by cross-validation over "
                         "walkers, so the estimate widens as the data thins -- "
                         "smoother, and shrunk towards the local mean, rather "
                         "than noisier.",
               "basis": "The basis size is chosen by cross-validation over "
                        "walkers, so the estimate keeps fewer Fourier modes as "
                        "the data thins -- smoother, missing fine structure "
                        "rather than noisier.",
               "neural": "Training stops when the held-out walkers stop "
                         "improving, which comes sooner when the data is thin -- "
                         "so the estimate is smoother, missing fine structure, "
                         "rather than noisier.",
               "snapshot": "The snapshot count follows the transition count, so "
                           "this column is really a snapshot budget -- up to the "
                           "500k cap, which the two largest runs hit.",
               "amortised": "Nothing is chosen per dataset: the network reads the "
                            "noise level off its inputs and smooths accordingly, as "
                            "it learned to on the training fields."}[spec.kind]
        _comparison_figure(
            spec, runs[0][1], fits,
            hero_value=f"D = {D}",
            hero_label="noise level, held fixed",
            hero_sub=f"{n_steps} steps per walker",
            name="03_recovery_vs_walkers",
            col_title=lambda N, est, row, s=spec: f"$N = {N}$   ({s.knob(est)})",
            caption=f"{spec.title}. {how} That is the bias-variance trade "
                    f"being made for you; figure 04 is the same trade made by "
                    f"hand.")


# --------------------------------------------------------------------------
# stage 4 -- the dial behind both of the above
# --------------------------------------------------------------------------

def stage_bins(args, specs):
    print("\n[bins] the smoothing dial, and error against local data")
    D = 0.3
    budgets = [(64, 3000), (1024, 3000 if args.quick else 12000)]
    runs = [case(D=D, n_walkers=N, n_steps=T, args=args) for N, T in budgets]
    L = 2.0
    for spec in specs:
        if spec.kind == "basis":
            _basis_dial(args, spec, runs, D)
            continue
        if spec.kind == "neural":
            _training_figure(args, spec, runs, D)
            continue
        if spec.kind == "snapshot":
            _snapshot_dial(args, spec, runs, D)
            continue
        if spec.kind == "amortised":
            print(f"  {spec.cli:<6} no smoothing dial to draw; training and the learned "
                  f"smoothing are in scripts/07_amortised_study.py")
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
        th = st.active()
        curves, notes = [], []
        for i, (field, traj, dt) in enumerate(runs):
            ev = make_eval_set(field, D, n=12000, seed=991, traj=traj)
            sty = st.series_style(i)
            lab = f"$N\\,T$ = {traj.n_transitions / 1e3:.0f}k"
            chosen = spec.make().fit(traj)
            if spec.kernel:
                grid = np.asarray(chosen.cv_["h"])
                errs = [drift_error(KernelRegression(
                    bandwidth=float(h), local_linear=chosen.local_linear)
                    .fit(traj), field, ev)["nrmse"] for h in grid]
                x_pick, x_ref = chosen.h_, chosen.h_silverman_
            else:
                grid = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64])
                errs = [drift_error(BinnedKramersMoyal(bins=int(b)).fit(traj),
                                    field, ev)["nrmse"] for b in grid]
                x_pick, x_ref = chosen.bins, None
            axes[0].plot(grid, errs, lw=2, markersize=5, label=lab, **sty)
            j = int(np.argmin(errs))
            axes[0].scatter([grid[j]], [errs[j]], s=90, facecolor="none",
                            edgecolor=sty["color"], linewidths=1.6, zorder=5)
            axes[0].axvline(x_pick, color=sty["color"], lw=1.0, dashes=(2, 3),
                            alpha=0.8)
            if x_ref is not None:
                axes[0].axvline(x_ref, color=sty["color"], lw=1.0,
                                dashes=(1, 1), alpha=0.5)
            pick_err = drift_error(chosen, field, ev)["nrmse"]
            notes.append((traj.n_transitions, grid[j], errs[j], x_pick,
                          pick_err, x_ref))
            print(f"  {spec.cli:<6} NT={traj.n_transitions:>10,}  best "
                  f"{grid[j]:.3g} -> {errs[j]:.3f}   chosen {x_pick:.3g} -> "
                  f"{pick_err:.3f}"
                  + (f"   silverman {x_ref:.3g}" if x_ref is not None else ""))
            c = error_vs_density(chosen, field, ev)
            c["label"] = lab
            curves.append(c)

        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("bandwidth  $h$" if spec.kernel else "bins per axis")
        axes[0].set_ylabel(r"normalised error  $\|\hat b-b\|/\|b\|$")
        axes[0].set_title("Bandwidth is a bias-variance dial" if spec.kernel
                          else "Bin count is a bias-variance dial")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(loc="upper center")
        key_txt = ("dashed: cross-validation's choice\ndotted: Silverman's "
                   "rule\ncircle: measured optimum" if spec.kernel else
                   "dashed: the automatic rule\ncircle: measured optimum")
        axes[0].text(0.03, 0.05, key_txt, transform=axes[0].transAxes,
                     fontsize=8.5, color=th.ink_muted, va="bottom")
        plot_error_vs_density(curves, axes[1],
                              title="Error where the data is thin")
        fig.tight_layout()
        pen = ", ".join(f"{100 * (pe / be - 1):+.0f}% at {nt / 1e3:.0f}k"
                        for nt, _, be, _, pe, _ in notes)
        pen_note = ("; negative means it beat the grid by landing between "
                    "grid points" if any(pe < be for _, _, be, _, pe, _ in notes)
                    else "")
        if spec.kernel:
            dial = (f"Left: too narrow a kernel averages too few increments "
                    f"(variance); too wide and it smooths real structure away "
                    f"(bias, O(h^2)). Cross-validation's choice is off the best "
                    f"point on the grid by {pen}{pen_note}. Silverman's rule, a density "
                    f"formula that cannot see the noise level, sits well left "
                    f"of both.")
        else:
            dial = (f"Left: too few bins and the cell average smooths real "
                    f"structure away (bias -- first order, O(h), because a bin "
                    f"answers with one value for its whole cell); too many and "
                    f"each cell holds too few increments (variance). The rule "
                    f"is off the best point on the grid by {pen}{pen_note}.")
        save(fig, spec,
             "04_bandwidth_and_density" if spec.kernel else "04_bins_and_density",
             f"{spec.title}. 2D random field, D = {D}. {dial} The optimum moves "
             f"as the data grows, which is why the smoothing is chosen from "
             f"the data and not fixed. Right: the same estimate resolved against "
             f"local sample density; the n^-1/2 guide is the rate of a local "
             f"average. The two right-hand curves are not comparable point by "
             f"point, since the larger run also smooths less.")


def _snapshot_dial(args, spec, runs, D):
    """KDE: error against bandwidth, with the score-CV and density-CV choices.
    DSM: error against the noise level the score is read at, and training."""
    th = st.active()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    notes = []
    for i, (field, traj, dt) in enumerate(runs):
        ev = make_eval_set(field, D, n=12000, seed=991, traj=traj)
        sty = st.series_style(i)
        snaps = snapshots_for(traj, field, D)
        lab = f"{len(snaps) / 1e3:.0f}k snapshots"
        chosen = spec.make().fit(snaps, D, box=field.box)
        if spec.cli == "kde":
            grid = np.asarray(chosen.cv_["h"])
            errs = [drift_error(KDEScore(bandwidth=float(h)).fit(snaps, D, box=field.box),
                                field, ev, use_support=False)["nrmse"] for h in grid]
            pick, ref = chosen.h_, chosen.h_density_
        else:
            grid = np.asarray(chosen.history["sigma_grid"])
            errs = []
            for sg in grid:
                chosen.sigma_ = float(sg)
                errs.append(drift_error(chosen, field, ev, use_support=False)["nrmse"])
            chosen.sigma_ = float(grid[int(np.argmin(chosen.history["sigma_sm_loss"]))])
            pick, ref = chosen.sigma_, None
        pick_err = drift_error(chosen, field, ev, use_support=False)["nrmse"]
        j = int(np.argmin(errs))
        axes[0].plot(grid, errs, lw=2, markersize=5, label=lab, **sty)
        axes[0].scatter([grid[j]], [errs[j]], s=90, facecolor="none",
                        edgecolor=sty["color"], linewidths=1.6, zorder=5)
        axes[0].axvline(pick, color=sty["color"], lw=1.0, dashes=(2, 3))
        if ref is not None:
            axes[0].axvline(ref, color=sty["color"], lw=1.0, dashes=(1, 1), alpha=0.6)
        notes.append((len(snaps), grid[j], errs[j], pick, pick_err, ref))
        print(f"  {spec.cli:<6} {len(snaps):>8,} snapshots  best {grid[j]:.3g} -> "
              f"{errs[j]:.3f}   chosen {pick:.3g} -> {pick_err:.3f}"
              + (f"   density-optimal {ref:.3g}" if ref is not None else ""))
        if spec.cli == "dsm":
            h = chosen.history
            axes[1].plot(h["epoch"], h["train"], lw=2, color=sty["color"],
                         label=f"{lab}: train")
            axes[1].plot(h["epoch"], h["val"], lw=1.5, color=sty["color"],
                         dashes=(3, 2), label=f"{lab}: held out")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bandwidth  $h$" if spec.cli == "kde"
                       else r"noise level the score is read at  $\sigma$")
    axes[0].set_ylabel(r"normalised error  $\|\hat b-b\|/\|b\|$")
    axes[0].set_title("Smoothing is a bias-variance dial")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="upper center")
    axes[0].text(0.03, 0.05, ("dashed: held-out score matching's choice\n"
                              "dotted: held-out likelihood's choice\ncircle: measured optimum")
                 if spec.cli == "kde" else
                 "dashed: held-out score matching's choice\ncircle: measured optimum",
                 transform=axes[0].transAxes, fontsize=8.5, color=th.ink_muted, va="bottom")
    if spec.cli == "kde":
        field, traj, dt = runs[-1]
        ev = make_eval_set(field, D, n=12000, seed=991, traj=traj)
        chosen = spec.make().fit(snapshots_for(traj, field, D), D, box=field.box)
        c = error_vs_density(chosen, field, ev)
        c["label"] = "chosen bandwidth"
        plot_error_vs_density([c], axes[1], title="Error where snapshots are thin")
    else:
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel(r"denoising loss  $|\sigma s + \epsilon|^2$")
        axes[1].set_title("Training (the loss floor is not zero: it is the "
                          "denoising variance)", fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    pen = ", ".join(f"{100 * (pe / be - 1):+.0f}% at {n / 1e3:.0f}k"
                    for n, _, be, _, pe, _ in notes)
    if spec.cli == "kde":
        ratio = ", ".join(f"{pk / rf:.2f}x at {n / 1e3:.0f}k" for n, _, _, pk, _, rf in notes)
        body = (f"Left: the KDE's score against its bandwidth. Held-out score matching's "
                f"choice is off the best point by {pen}. The bandwidth that maximises "
                f"held-out likelihood -- the right one for the density itself -- is smaller "
                f"(score-optimal / density-optimal: {ratio}): differentiating a density "
                f"amplifies its noise, so the score wants more smoothing. Right: error "
                f"against local sample density; the tails, where rho_ss is tiny and the "
                f"drift largest, are where the score is worst. The KDE uses at most 100k "
                f"of the snapshots.")
    else:
        body = (f"Left: the score read at different noise levels sigma of one trained "
                f"network -- small sigma is least smoothed but least trained. Held-out "
                f"score matching's choice is off the best point by {pen}. Right: the "
                f"denoising loss per epoch on training and held-out snapshots. It cannot "
                f"reach zero: the best denoiser still leaves the variance of eps given the "
                f"noisy point, so the flat curve is convergence, not failure.")
    save(fig, spec, "04_smoothing_dial",
         f"{spec.title}. 2D random field, D = {D}. {body}")


def _basis_dial(args, spec, runs, D):
    """Error against basis size, with the fit's own error prediction.

    Both panels put the measured error beside the noise-only prediction
    ``sqrt(d p sigma^2 / n) / RMS|b|``, computed from residuals alone. Left:
    across basis sizes at two budgets -- the prediction rises like sqrt(p)
    while the measured error first falls (bias shrinking) and then joins it.
    Right: across budgets for fixed sizes, using nested walker subsets of the
    same run -- the two track until a too-small basis hits its bias floor.
    """
    th = st.active()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    notes = []
    for i, (field, traj, dt) in enumerate(runs):
        ev = make_eval_set(field, D, n=12000, seed=991, traj=traj)
        sty = st.series_style(i)
        lab = f"$N\\,T$ = {traj.n_transitions / 1e3:.0f}k"
        chosen = spec.make().fit(traj)
        sizes = chosen.cv_["n_basis"]
        errs, preds = [], []
        for p_ in sizes:
            sub = chosen.sub_fit(int(p_))
            r = drift_error(sub, field, ev)
            errs.append(r["nrmse"])
            preds.append(np.sqrt(sub.error_bound()) / r["rms_true"])
        errs, preds = np.array(errs), np.array(preds)
        axes[0].plot(sizes, errs, lw=2, markersize=5, label=lab, **sty)
        axes[0].plot(sizes, preds, lw=1.3, color=sty["color"], dashes=(1, 1.5))
        j = int(np.argmin(errs))
        axes[0].scatter([sizes[j]], [errs[j]], s=90, facecolor="none",
                        edgecolor=sty["color"], linewidths=1.6, zorder=5)
        axes[0].axvline(chosen.n_basis, color=sty["color"], lw=1.0,
                        dashes=(2, 3), alpha=0.8)
        pick = drift_error(chosen, field, ev)["nrmse"]
        notes.append((traj.n_transitions, int(sizes[j]), errs[j],
                      chosen.n_basis, pick))
        print(f"  {spec.cli:<6} NT={traj.n_transitions:>10,}  best p={sizes[j]} "
              f"-> {errs[j]:.3f}   CV p={chosen.n_basis} -> {pick:.3f}   "
              f"cond={chosen.condition_number():.1f}")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of basis functions  $p$")
    axes[0].set_ylabel(r"normalised error  $\|\hat b-b\|/\|b\|$")
    axes[0].set_title("Basis size is a bias-variance dial")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="upper center")
    axes[0].text(0.03, 0.05, "solid: measured against the truth\n"
                 "dotted: predicted from the data alone\n"
                 "dashed: cross-validation's choice\ncircle: measured optimum",
                 transform=axes[0].transAxes, fontsize=8.5, color=th.ink_muted,
                 va="bottom")

    # Right: fixed sizes, nested subsets of the big run.
    field, big, _ = runs[-1]
    ev = make_eval_set(field, D, n=12000, seed=991)
    subsets = [16, 64, 256, big.n_walkers]
    # A basis too small to span the field, and one comfortably large enough.
    fixed = [(4, "p = 13"), (82, "p = 261")]
    tracks = {}
    for k, (level, name) in enumerate(fixed):
        sty = st.series_style(k + 2)
        meas, pred, nts = [], [], []
        for N in subsets:
            tr = big.subsample_walkers(N, rng=np.random.default_rng(3)) \
                if N < big.n_walkers else big
            e = BasisProjection(basis="fourier", degree=level).fit(tr)
            r = drift_error(e, field, ev)
            meas.append(r["nrmse"])
            pred.append(np.sqrt(e.error_bound()) / r["rms_true"])
            nts.append(tr.n_transitions)
        axes[1].plot(nts, meas, lw=2, markersize=5, label=f"{name}, measured", **sty)
        axes[1].plot(nts, pred, lw=1.3, color=sty["color"], dashes=(1, 1.5),
                     label=f"{name}, predicted")
        tracks[name] = (np.array(nts), np.array(meas), np.array(pred))
        print(f"  {spec.cli:<6} {name}: measured/predicted "
              + ", ".join(f"{m / q:.2f}" for m, q in zip(meas, pred)))
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"observed transitions  $N\,T$")
    axes[1].set_ylabel("normalised error")
    axes[1].set_title("Measured error against the fit's own prediction")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    small, large = tracks["p = 13"], tracks["p = 261"]

    def story(name, t):
        r = t[1] / t[2]
        if r[-1] > 1.5 * r[0]:
            return (f"With {name} measured/predicted runs {r[0]:.2f} -> "
                    f"{r[-1]:.2f}: it tracks while noise dominates, then "
                    f"departs, because the variance keeps falling and the bias "
                    f"from the modes this basis leaves out does not -- the "
                    f"departure is that bias, measured.")
        return (f"With {name} measured/predicted runs {r[0]:.2f} -> {r[-1]:.2f}: "
                f"the error falls at the predicted rate throughout.")
    gap = ", ".join(f"{100 * (pe / be - 1):+.0f}% at {nt / 1e3:.0f}k"
                    for nt, _, be, _, pe in notes)
    save(fig, spec, "04_basis_size_and_bound",
         f"{spec.title}, Fourier basis. 2D random field, D = {D}. Dotted lines "
         f"are the fit's own error prediction, sqrt(d p sigma^2 / n) / RMS|b|, "
         f"with sigma^2 from its residuals -- no ground truth used. Left: "
         f"a small basis misses modes the field has (bias); a large one "
         f"spends noise on coefficients it does not need (variance, growing as "
         f"sqrt(p)). Once the basis spans the field, measured and predicted "
         f"error coincide -- the prediction is the error. Cross-validation's "
         f"choice is off the best size by {gap}. Right: nested walker subsets "
         f"of the largest run. {story('p = 261', large)} "
         f"{story('p = 13', small)} (Scored on rho_ss; with few "
         f"walkers the sample has not yet covered rho_ss, which pushes the ratio "
         f"above 1 without any bias.)")


def _truth_monitor(field, ev):
    """Ground-truth nrmse as a training monitor -- a diagnostic, never a signal."""
    bt = np.asarray(field.drift(ev.x), float)
    rms = float(np.sqrt(np.mean(np.sum(bt ** 2, -1))))

    def monitor(pred):
        return float(np.sqrt(np.mean(np.sum((pred(ev.x) - bt) ** 2, -1)))) / rms
    return monitor, rms


def _training_figure(args, spec, runs, D):
    """What training a network looks like, at the dial stage's two budgets.

    One row per budget. Left: the loss the optimiser actually minimises,
    relative to the noise floor -- flat, because the drift is ~0.1% of it.
    Middle: the gain over predicting zero on training and on held-out walkers,
    the same loss with the shared noise cancelled; the held-out peak is where
    training stops. Right: the ground-truth error along the way, which training
    never sees, with SFI and the local-linear kernel on the same walkers.
    """
    th = st.active()
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
    notes = []
    for i, (field, traj, dt) in enumerate(runs):
        ev = make_eval_set(field, D, n=12000, seed=991, traj=traj)
        mon, rms = _truth_monitor(field, ev)
        t0 = time.time()
        est = spec.make()
        est.monitor = mon
        est.fit(traj)
        secs = time.time() - t0
        final = drift_error(est, field, ev)["nrmse"]
        refs = {"SFI": drift_error(BasisProjection().fit(traj), field, ev)["nrmse"],
                "LL kernel": drift_error(KernelRegression().fit(traj), field,
                                         ev)["nrmse"]}
        h = est.history
        sel = h["select"]
        floor = h["noise_floor"]
        steps = np.r_[0, sel["step"]]
        g_tr = np.r_[0.0, sel["gain_train"]]
        g_va = np.r_[0.0, sel["gain_val"]]
        best = sel["best_step"]
        nt = (f"{traj.n_transitions / 1e3:,.0f}k" if traj.n_transitions < 1e6
              else f"{traj.n_transitions / 1e6:.1f}M")
        a0, a1, a2 = axes[i]
        c_tr, c_va = st.series_style(0)["color"], st.series_style(1)["color"]

        a0.plot(steps, (sel["y2_train"] - g_tr) / floor, lw=2, color=c_tr,
                label="training walkers")
        a0.plot(steps, (sel["y2_val"] - g_va) / floor, lw=2, color=c_va,
                label="held-out walkers")
        a0.axhline(1.0, color=th.ink_muted, lw=1.2, dashes=(3, 2),
                   label=r"noise floor $2dD/\Delta t$")
        a0.set_ylim(0.97, 1.03)
        a0.set_xlabel("optimisation step")
        a0.set_ylabel("loss / noise floor")
        a0.set_title(f"{nt} transitions: the loss the optimiser sees")
        a0.legend(loc="upper right", fontsize=8)
        a0.grid(True, alpha=0.3)
        a0.text(0.03, 0.05, f"drift power / loss at the start = "
                f"{rms ** 2 / h['mean_y2']:.2%}",
                transform=a0.transAxes, fontsize=9, color=th.ink_secondary)

        if "select_by_grid" in h:
            for M, hg in sorted(h["select_by_grid"].items()):
                if M == h["grid"]:
                    continue
                a1.plot(np.r_[0, hg["step"]], np.r_[0.0, hg["gain_val"]], lw=1,
                        color=c_va, alpha=0.5, dashes=(2, 1.5))
                a1.annotate(f"{M}² grid", (hg["step"][-1], hg["gain_val"][-1]),
                            fontsize=7.5, color=th.ink_muted,
                            textcoords="offset points", xytext=(2, 0),
                            annotation_clip=True)
        a1.plot(steps, g_tr, lw=2, color=c_tr, label="training walkers")
        a1.plot(steps, g_va, lw=2, color=c_va, label="held-out walkers"
                + (f" ({h['grid']}² grid)" if "grid" in h else ""))
        a1.axvline(best, color=c_va, lw=1.0, dashes=(2, 3))
        a1.axhline(rms ** 2, color=th.ink_muted, lw=1.0, dashes=(1, 1.5),
                   label=r"RMS$|b|^2$ (truth, for reference)")
        top = max(1.35 * rms ** 2, 1.15 * float(np.max(g_va)))
        a1.set_ylim(max(float(np.min(g_va)) * 1.1, -rms ** 2) if np.min(g_va) < 0
                    else -0.05 * rms ** 2, top)
        if g_tr.max() > top:
            a1.text(0.97, 0.93, "training gain runs off the top:\nthe fit is "
                    "absorbing noise", transform=a1.transAxes, ha="right",
                    va="top", fontsize=8, color=c_tr)
        a1.set_xlabel("optimisation step")
        a1.set_ylabel(r"gain over predicting zero  "
                      r"$\overline{|y|^2}-\overline{|y-\hat b|^2}$")
        a1.set_title("The same loss with the noise cancelled")
        a1.legend(loc="lower right", fontsize=8)
        a1.grid(True, alpha=0.3)

        a2.plot(steps, np.r_[1.0, sel["monitor"]], lw=2, color=c_va,
                label=f"selection run "
                      f"({traj.n_walkers - h['n_val_walkers']} walkers)")
        if "refit" in h:
            rf = h["refit"]
            a2.plot(np.r_[0, rf["step"]], np.r_[1.0, rf["monitor"]], lw=2,
                    color=st.series_style(4)["color"],
                    label=f"refit on all {traj.n_walkers} walkers")
        a2.axvline(best, color=c_va, lw=1.0, dashes=(2, 3))
        for k, (name, v) in enumerate(refs.items()):
            a2.axhline(v, color=st.series_style(k + 2)["color"], lw=1.2,
                       dashes=(4, 2), label=f"{name}: {v:.3f}")
        a2.set_yscale("log")
        a2.set_xlabel("optimisation step")
        a2.set_ylabel("nrmse against the true field")
        a2.set_title(f"Against the truth (diagnostic only): final {final:.3f}")
        a2.legend(loc="upper right", fontsize=8)
        a2.grid(True, which="both", alpha=0.3)

        i_best = int(np.argmin(sel["monitor"]))
        notes.append(dict(nt=nt, best=best, final=final, refs=refs, secs=secs,
                          oracle=(sel["step"][i_best], sel["monitor"][i_best]),
                          at_best=sel["monitor"][sel["step"].index(best)],
                          refit=h.get("refit_steps"), grid=h.get("grid")))
        n = notes[-1]
        print(f"  {spec.cli:<6} NT={traj.n_transitions:>10,}  stop at {best} "
              f"(truth-best {n['oracle'][0]}: {n['oracle'][1]:.3f}, at stop "
              f"{n['at_best']:.3f})  refit {h.get('refit_steps')} -> {final:.3f}"
              f"   SFI {refs['SFI']:.3f}  LL {refs['LL kernel']:.3f}"
              + (f"  grids {h['grid_cv']}" if "grid_cv" in h else "")
              + f"  [{secs:.0f}s]")
    fig.tight_layout()
    per = ". ".join(
        f"At {n['nt']} it stopped at step {n['best']} (the truth's best step on "
        f"that run was {n['oracle'][0]}: {n['oracle'][1]:.3f} there, "
        f"{n['at_best']:.3f} at the stop), then refit to {n['final']:.3f}"
        + (f" on a {n['grid']}² grid" if n["grid"] else "")
        + f" -- against SFI {n['refs']['SFI']:.3f} and LL "
          f"{n['refs']['LL kernel']:.3f}"
        for n in notes)
    grid_note = (" Faint dashed lines: the held-out gain for the grids the "
                 "selection did not pick." if spec.cli == "cnn" else "")
    save(fig, spec, "04_training",
         f"{spec.title}, training statistics. 2D random field, D = {D}. Left: the "
         f"raw squared loss is pinned to the noise floor 2dD/dt, because each "
         f"target's noise is ~1000x the drift -- a converged fit and an untrained "
         f"one differ by a fraction of a percent, so this curve cannot say whether "
         f"training worked. Middle: the loss difference against predicting zero on "
         f"the same targets, where the shared noise cancels; it rises exactly as "
         f"the error falls. On training walkers it keeps rising once the network "
         f"starts fitting noise; on held-out walkers it peaks, and the dotted "
         f"vertical line is that peak, where training stops.{grid_note} Right: "
         f"ground-truth error along the same runs, never seen by training. "
         f"{per}.")


def _neural_dims_panel(args, spec, ax, top, n_steps):
    """Held-out gain during training, one curve per dimension, top budget."""
    th = st.active()
    for k, (_, r) in enumerate(top.iterrows()):
        cfg = SweepConfig(d=int(r.d), n_walkers=int(r.n_walkers), n_steps=n_steps,
                          D=0.3, seed=0, n_eval=8000)
        field, traj, dt = simulate_config(cfg, traj_cache=args.traj_cache)
        est = spec.make().fit(traj)
        sel = est.history["select"]
        frac = np.r_[0.0, sel["gain_val"]] / r.rms_true ** 2
        sty = st.series_style(k)
        ax.plot(np.r_[0, sel["step"]], frac, lw=2, color=sty["color"],
                label=f"$d$ = {int(r.d)}   (nrmse {r.nrmse:.2f})")
        ax.axvline(sel["best_step"], color=sty["color"], lw=0.9, dashes=(2, 3))
        print(f"  {spec.cli:<6} d={int(r.d)}: best held-out gain "
              f"{max(sel['gain_val']) / r.rms_true ** 2:.3f} of |b|^2 at step "
              f"{sel['best_step']}")
    ax.axhline(1.0, color=th.ink_muted, lw=1, dashes=(3, 2))
    ax.axhline(0.0, color=th.ink_muted, lw=0.8)
    ax.set_ylim(-0.2, 1.1)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(left=0)
    ax.set_xlabel("optimisation step (selection run)")
    ax.set_ylabel(r"held-out gain / RMS$|b|^2$")
    ax.set_title("How much of the field training finds, by dimension")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)


# --------------------------------------------------------------------------
# stage 5 -- dimension
# --------------------------------------------------------------------------

def _dims_table(args, specs):
    import pandas as pd
    dims = [2, 3, 5, 10]
    walkers = [32, 128, 512] if args.quick else [32, 128, 512, 2048]
    n_steps = 800 if args.quick else 3000
    frames = []
    # A grid-backed method (the CNN) cannot run past d = 3, so each group of
    # estimators is swept only over the dimensions it can reach.
    for max_d in sorted({s.max_d for s in specs}):
        group = [s for s in specs if s.max_d == max_d]
        cfgs = sweep_grid(d=[dd for dd in dims if dd <= max_d],
                          n_walkers=walkers, n_steps=[n_steps], D=0.3, seed=0,
                          n_eval=8000)
        frames.append(run_sweep(cfgs, {s.key: s.make for s in group},
                                cache=DATA / "sweep_dimension.csv",
                                traj_cache=args.traj_cache))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["config_key", "estimator_name"], keep="last")
    return df, dims, walkers, n_steps


def stage_dims(args, specs):
    print("\n[dims] d = 2, 3, 5, 10")
    df, dims, walkers, n_steps = _dims_table(args, specs)
    for spec in specs:
        if spec.max_d < 3:
            print(f"  ({spec.cli}: 2D only, no dimension figure)")
            continue
        ok = df[(df.status == "ok") & (df.estimator_name == spec.key)].copy()
        if ok.empty:
            print(f"  ({spec.cli}: nothing ran)")
            continue
        ok["label"] = ok.d.map(lambda v: f"$d$ = {int(v)}")
        spec_dims = [dd for dd in dims if dd <= spec.max_d]
        dts = ok.groupby("d").dt_actual.first()

        fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
        plot_error_curves(ok, axes[0], x="n_transitions", y="nrmse",
                          slope_guide=None,
                          title="Error against budget, by dimension")
        leg = axes[0].get_legend()
        if leg is not None:
            leg.remove()

        top = ok[ok.n_walkers == max(walkers)].sort_values("d")
        sty = st.series_style(0)
        if spec.kind == "neural":
            _neural_dims_panel(args, spec, axes[1], top, n_steps)
        elif spec.kind == "basis" and "diag_bound_mse" in top:
            # A global basis always "covers" everything, so a coverage panel
            # is empty of information. Show instead what separates a good
            # basis fit from a bad one without the truth: measured error
            # against the fit's own noise-only prediction. The gap is bias.
            pred = np.sqrt(top.diag_bound_mse / top.rms_true ** 2)
            axes[1].plot(top.d, top.nrmse, lw=2, markersize=6,
                         label="measured against the truth", **sty)
            axes[1].plot(top.d, pred, lw=1.6, color=sty["color"],
                         dashes=(1, 1.5), marker="o", markersize=4,
                         markerfacecolor="none",
                         label="predicted from the data alone")
            axes[1].axhline(1.0, color=st.active().ink_muted, lw=1,
                            dashes=(3, 2))
            axes[1].set_yscale("log")
            axes[1].set_xticks(dims)
            axes[1].set_xlabel("dimension  $d$")
            axes[1].set_ylabel("normalised error")
            axes[1].set_title("Measured error against the fit's own prediction")
            axes[1].grid(True, which="both", alpha=0.3)
            axes[1].legend(loc="lower right", fontsize=8)
            for (_, r), pv in zip(top.iterrows(), pred):
                axes[1].annotate(spec.knob_row(r), (r.d, r.nrmse),
                                 textcoords="offset points", xytext=(0, 8),
                                 fontsize=7.5, ha="center",
                                 color=st.active().ink_muted)
        else:
            axes[1].plot(top.d, top.supported_fraction, lw=2, markersize=6, **sty)
            axes[1].set_ylim(-0.05, 1.12)
            axes[1].set_xticks(dims)
            axes[1].set_xlabel("dimension  $d$")
            axes[1].set_ylabel("fraction of evaluation points covered")
            axes[1].set_title("Where the estimator will answer")
            axes[1].grid(True, alpha=0.3)
            for _, r in top.iterrows():
                high = r.supported_fraction > 0.5
                label = (spec.knob_row(r) if spec.kind != "binned" else
                         f"{int(r.param_bins)} bins\n"
                         f"{human(int(r.param_bins) ** int(r.d))} cells")
                axes[1].annotate(label, (r.d, r.supported_fraction),
                                 textcoords="offset points",
                                 xytext=(0, -10 if high else 10), fontsize=7.5,
                                 ha="center", va="top" if high else "bottom",
                                 color=st.active().ink_muted)

        ref = top.set_index("d")
        slopes = {}
        for dd in spec_dims:
            g = ok[ok.d == dd].sort_values("n_transitions")
            if len(g) >= 2:
                slopes[dd] = float(np.polyfit(np.log(g.n_transitions),
                                              np.log(g.nrmse), 1)[0])
        slope_txt = ", ".join(f"{v:+.2f} at d = {k}" for k, v in slopes.items())
        top_d = spec_dims[-1]
        print(f"  {spec.cli:<6} " + "  ".join(
            f"d={dd}: {ref.nrmse[dd]:.3f} ({spec.knob_row(ref.loc[dd])}, "
            f"{ref.supported_fraction[dd]:.0%})" for dd in spec_dims if dd in ref.index))
        print(f"  {spec.cli:<6} slopes: {slope_txt}")

        if spec.kind == "snapshot":
            col = "param_bandwidth" if spec.cli == "kde" else "param_sigma_eval"
            span = {int(dd): (g[col].min(), g[col].max()) for dd, g in ok.groupby("d")}
            span_txt = ", ".join(f"{lo:.3f}-{hi:.3f} at d = {dd}"
                                 for dd, (lo, hi) in span.items())
            mech = (f"Snapshots only, as many as there are transitions up to 500k. The "
                    f"{'bandwidth' if spec.cli == 'kde' else 'read-out noise level'} went "
                    f"{span_txt}. The error at d = {top_d} is {ref.nrmse[top_d]:.2f}: "
                    f"a density in {top_d} dimensions has to be resolved before it can "
                    f"be differentiated, and snapshots are no less subject to that than "
                    f"increments are.")
        elif spec.kind == "neural":
            span = {int(dd): (int(g.diag_refit_steps.min()),
                              int(g.diag_refit_steps.max()))
                    for dd, g in ok.groupby("d")}
            span_txt = ", ".join(f"{lo}-{hi} at d = {dd}"
                                 for dd, (lo, hi) in span.items())
            vols = ", ".join(f"{(2.0 / 0.45) ** dd:,.0f} at d = {dd}"
                             for dd in spec_dims)
            mech = (f"Early stopping trained for {span_txt} steps (refit) across "
                    f"the budget. Right: the gain over predicting zero on "
                    f"held-out walkers during training at the largest budget, "
                    f"as a fraction of the drift's power -- how much of the "
                    f"field the network finds before it starts fitting noise. "
                    f"The error at d = {top_d} is {ref.nrmse[top_d]:.2f}. "
                    + ("Only d <= 3: the output is a grid of M^d nodes. "
                       if spec.max_d < 10 else "")
                    + f"The field holds about (L/ell)^d independent correlation "
                      f"volumes -- {vols} -- against a fixed budget.")
        elif spec.kind == "basis":
            span = {int(dd): (int(g.param_n_basis.min()), int(g.param_n_basis.max()))
                    for dd, g in ok.groupby("d")}
            span_txt = ", ".join(f"{lo}-{hi} at d = {dd}"
                                 for dd, (lo, hi) in span.items())
            if "diag_bound_mse" in top:
                pred = np.sqrt(top.diag_bound_mse / top.rms_true ** 2)
                pred_txt = ", ".join(f"{p_:.2f} at d = {int(dd)}"
                                     for dd, p_ in zip(top.d, pred))
            else:
                pred_txt = "n/a"
            mech = (f"Cross-validation chose {span_txt} basis functions across "
                    f"the budget. The Fourier modes the field is made of move "
                    f"outward in lattice norm as d grows, while the number of "
                    f"modes below any cutoff grows like (cutoff)^(d/2), so the "
                    f"affordable basis stops reaching them: the error at "
                    f"d = {top_d} is {ref.nrmse[top_d]:.2f}. The fit's own "
                    f"noise-only error prediction at the top budget -- no ground "
                    f"truth used -- is {pred_txt}; where it sits far below the "
                    f"measured error, what is left is projection bias, the part "
                    f"of b outside the span of the basis.")
        elif spec.kernel:
            span = {int(dd): (g.param_bandwidth.min(), g.param_bandwidth.max())
                    for dd, g in ok.groupby("d")}
            span_txt = ", ".join(f"{lo:.2f}-{hi:.2f} at d = {dd}"
                                 for dd, (lo, hi) in span.items())
            mech = (f"Cross-validation widened the kernel as d grew -- "
                    f"bandwidths {span_txt} across the budget -- until at "
                    f"d = {top_d} it averages over most of the box, and an "
                    f"average over most of the box is close to the global mean: "
                    f"nrmse {ref.nrmse[top_d]:.2f} at {ref.supported_fraction[top_d]:.0%} "
                    f"coverage, against 1.0 for predicting zero. A kernel does "
                    f"not refuse to answer the way a bin grid does; it answers "
                    f"with less and less that is local.")
        else:
            span = {int(dd): (int(g.param_bins.min()), int(g.param_bins.max()))
                    for dd, g in ok.groupby("d")}
            span_txt = ", ".join(f"{lo}" + (f"-{hi}" if hi != lo else "")
                                 + f" at d = {dd}" for dd, (lo, hi) in span.items())
            mech = (f"bins**d cells against a fixed budget, so coverage "
                    f"collapses: the rule used {span_txt} bins per axis. At "
                    f"d = {top_d} the score of {ref.nrmse[top_d]:.2f} is taken "
                    f"only on the {ref.supported_fraction[top_d]:.0%} of "
                    f"evaluation points the grid answers for -- cells a walker "
                    f"happened to linger in, whose averages are the noisiest "
                    f"of all. (outputs/kernel/ scores every method on every "
                    f"point, which removes the reward for abstaining.)")
        fig.tight_layout()
        save(fig, spec, "05_error_vs_dimension",
             f"{spec.title}. D = 0.3, omega = 0, {n_steps} steps per walker; "
             f"budget swept with the walker count; the same walkers as every "
             f"other estimator in this study. Grid-backed field up to d = 3, "
             f"mesh-free above. dt is chosen per dimension and falls from "
             f"{dts.get(2, float('nan')):.1e} to "
             f"{dts.get(top_d, float('nan')):.1e}, which alone inflates a local "
             f"average's error by {np.sqrt(dts[2] / dts[top_d]):.1f}x. "
             f"{mech} Measured slopes d(log nrmse)/d(log NT): {slope_txt}.")


# --------------------------------------------------------------------------
# stage 6 -- the small sweeps
# --------------------------------------------------------------------------

def stage_sweeps(args, specs):
    print("\n[sweeps] walkers, noise level, measurement noise")
    n_steps = 1000 if args.quick else 4000
    seeds = [0] if args.quick else [0, 1, 2]
    est = {s.key: s.make for s in specs}
    kw = dict(traj_cache=args.traj_cache)

    walkers = [16, 32, 64, 128, 256, 512] + ([] if args.quick else [1024])
    df_n = run_sweep(sweep_grid(d=2, n_walkers=walkers, n_steps=[n_steps],
                                D=0.3, seed=seeds), est,
                     cache=DATA / "sweep_walkers.csv", **kw)
    Ds = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
    df_d = run_sweep(sweep_grid(d=2, n_walkers=256, n_steps=[n_steps], D=Ds,
                                seed=seeds), est,
                     cache=DATA / "sweep_D.csv", **kw)
    sig = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
    df_s = run_sweep(sweep_grid(d=2, n_walkers=256, n_steps=[n_steps], D=0.3,
                                obs_noise=sig, seed=seeds), est,
                     cache=DATA / "sweep_obs_noise.csv", **kw)

    for spec in specs:
        dn = df_n[df_n.estimator_name == spec.key]
        dd = df_d[df_d.estimator_name == spec.key]
        ds = df_s[df_s.estimator_name == spec.key]
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.0))
        band = ("min", "max") if len(seeds) > 1 else None
        plot_error_curves(agg(dn, "n_walkers"), axes[0], x="n_walkers",
                          y="nrmse", band=band, direct_labels=False,
                          title="More walkers")
        plot_error_curves(agg(dd, "D"), axes[1], x="D", y="nrmse", logx=True,
                          slope_guide=None, band=band, direct_labels=False,
                          title="Noise level")
        plot_error_curves(agg(ds, "obs_noise"), axes[2], x="obs_noise",
                          y="nrmse", logx=False, slope_guide=None, band=band,
                          direct_labels=False, title="Measurement noise")
        axes[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        pred_note = ""
        if spec.kind == "basis" and "diag_bound_mse" in dn:
            okn = dn[dn.status == "ok"].copy()
            okn["pred"] = np.sqrt(okn.diag_bound_mse / okn.rms_true ** 2)
            gp = okn.groupby("n_walkers")[["pred", "nrmse"]].mean().reset_index()
            axes[0].plot(gp.n_walkers, gp.pred, lw=1.3, dashes=(1, 1.5),
                         color=st.active().ink_secondary)
            axes[0].text(gp.n_walkers.iloc[-1], gp.pred.iloc[-1],
                         "  predicted,\n  no ground truth", fontsize=8,
                         color=st.active().ink_secondary, va="center")
            ratio = gp.nrmse / gp.pred
            pred_note = (f" The dotted line is the fit's own noise-only error "
                         f"prediction; measured/predicted runs "
                         f"{ratio.iloc[0]:.2f} -> {ratio.iloc[-1]:.2f} across "
                         f"the walker axis.")
        dts = dd[dd.status == "ok"].groupby("D").dt_actual.first()
        ad = agg(dd, "D")
        best_D = float(ad.sort_values("nrmse").D.iloc[0])
        an, as_ = agg(dn, "n_walkers"), agg(ds, "obs_noise")
        print(f"  {spec.cli:<6} walkers {an.nrmse.iloc[0]:.3f} -> "
              f"{an.nrmse.iloc[-1]:.3f};  best D = {best_D:g} "
              f"({ad.nrmse.min():.3f});  sigma=0.02 -> {as_.nrmse.iloc[-1]:.3f}")
        fig.tight_layout()
        save(fig, spec, "06_sweeps",
             f"{spec.title}. 2D random field, {n_steps} steps per walker, the "
             f"same walkers as every other estimator in this study. Band spans "
             f"{len(seeds)} seed(s). Left: {an.nrmse.iloc[0]:.2f} with "
             f"{int(an.n_walkers.iloc[0])} walkers to {an.nrmse.iloc[-1]:.2f} "
             f"with {int(an.n_walkers.iloc[-1])}; the n^-1/2 guide is the Monte "
             f"Carlo rate, and falling short of it here partly reflects the "
             f"smoothing being re-chosen at every point. Middle: the lowest "
             f"error is at D = {best_D:g}. Above it, the per-increment noise "
             f"sqrt(2D/dt) grows -- and a stable dt itself shrinks with D, from "
             f"{dts.get(min(Ds), float('nan')):.1e} to "
             f"{dts.get(max(Ds), float('nan')):.1e} across this axis. Below it "
             f"the walkers stop exploring. Right: position noise biases the "
             f"target itself -- noisy positions sit downhill of true ones, so "
             f"every method that averages dx/dt converges to "
             f"b (1 + sigma^2/(D dt)): quadratic in sigma, worse as dt -> 0, "
             f"and out of reach of any smoothing choice.{pred_note}")


STAGES = {"setup": stage_setup, "noise": stage_noise, "walkers": stage_walkers,
          "bins": stage_bins, "dims": stage_dims, "sweeps": stage_sweeps}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimators", nargs="+", default=["binned"],
                    choices=list(SPECS))
    ap.add_argument("--which", nargs="*", default=list(STAGES),
                    choices=list(STAGES))
    ap.add_argument("--quick", action="store_true",
                    help="smaller runs, for checking the layout")
    ap.add_argument("--animate", action="store_true",
                    help="also render the walker mp4 (binned only; needs ffmpeg)")
    ap.add_argument("--traj-cache", type=Path, default=default_traj_cache(),
                    help="directory of cached trajectories (default: "
                         "%(default)s, or $DFI_TRAJ_CACHE)")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    args = ap.parse_args()

    use_style(args.theme)
    specs = [SPECS[e] for e in args.estimators]
    DATA.mkdir(parents=True, exist_ok=True)
    for s in specs:
        s.fig.mkdir(parents=True, exist_ok=True)
    print(f"estimators: {', '.join(s.cli for s in specs)}   trajectory cache: "
          f"{args.traj_cache}")
    t0 = time.time()
    for name in args.which:
        STAGES[name](args, specs)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
