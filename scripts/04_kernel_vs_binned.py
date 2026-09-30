"""Rung 2 against rung 1: how a kernel estimate works, and when it wins.

Writes to ``outputs/kernel/``.

    python scripts/04_kernel_vs_binned.py --quick               # layout check
    python scripts/04_kernel_vs_binned.py --which explain       # one stage
    python scripts/04_kernel_vs_binned.py                       # everything

Stages:

``explain``    what a kernel averages, why Nadaraya-Watson shrinks, and why
               both beat bins -- each claim checked against a closed form
``bandwidth``  how the bandwidth is chosen: cross-validation against the
               ground-truth optimum and against Silverman's rule
``recovery``   the three estimates side by side on the same data, twice:
               a well-sampled field, and a low-noise one the walkers barely
               explore
``sweeps``     walkers, D, measurement noise -- all three methods
``dims``       d = 2, 3, 5, 10 -- all three methods

Every method picks its own smoothing by cross-validation grouped by walker, so
the comparison is between methods and not between tuning rules. And every score
is taken over *all* evaluation points, including ones a method declines to
answer for (it is scored on what it actually predicts there): scoring only on
supported points rewards abstention, and cross-validation -- which predicts
every held-out point -- would then be optimising something other than the score.
Coverage is still measured and reported. Sweep rows are cached in
``outputs/kernel/data/`` by config hash.
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
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfi import ornstein_uhlenbeck, simulate, suggest_dt
from dfi.estimators.local import BinnedKramersMoyal, KernelRegression
from dfi.metrics import drift_error, make_eval_set
from dfi.sweeps import (SweepConfig, _burn_in_time, build_field, run_sweep,
                        sweep_grid)
from dfi.viz import figure_caption, use_style
from dfi.viz import style as st
from dfi.viz.estimates import (COARSE, cell_centres, error_panel,
                                estimate_maps, quiver_panel, row_colorbar,
                                vectors_at)
from dfi.viz.sweeps import plot_error_curves

OUT = ROOT / "outputs" / "kernel"
FIG = OUT / "figures"
DATA = OUT / "data"

BINNED = "Binned Kramers-Moyal"
NW = "Kernel, Nadaraya-Watson"
LL = "Kernel, local-linear"
#: Fixed colour per method across every figure here. Slot 3 belongs to the
#: (unimplemented) neural rung elsewhere in the project; it is borrowed for
#: Nadaraya-Watson only inside this comparison.
SLOTS = {BINNED: 0, LL: 1, NW: 3}
ORDER = (BINNED, NW, LL)


def methods():
    return {"binned_cv": lambda: BinnedKramersMoyal(bins="cv"),
            "kernel_nw": lambda: KernelRegression(local_linear=False),
            "kernel_ll": lambda: KernelRegression(local_linear=True)}


def sty(label):
    return st.series_style(SLOTS[label])


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def save(fig, name, caption=None):
    if caption:
        width = max(60, int(fig.get_size_inches()[0] * 17))
        figure_caption(fig, "\n".join(textwrap.wrap(caption, width)))
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def field2d(seed=0):
    return build_field(SweepConfig(d=2, seed=seed))


def run2d(field, *, D, n_walkers, n_steps, seed=0):
    dt = suggest_dt(field, D)
    return simulate(field, n_walkers=n_walkers, n_steps=n_steps, dt=dt, D=D,
                    seed=seed, check=False,
                    burn_in=_burn_in_time(field, SweepConfig(d=2, D=D)))


def h_equivalent(bins, length):
    """Gaussian bandwidth with the same second moment as a box of one cell.

    A box of width ``w`` has variance ``w^2/12``, so ``h = w / sqrt(12)``.
    It puts bin counts and bandwidths on one axis; it does not make the two
    kernels equivalent, and figure 01(d) shows why not.
    """
    return length / np.asarray(bins, float) / np.sqrt(12.0)


def _style_axis_log(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


# --------------------------------------------------------------------------
# stage 1 -- how it works
# --------------------------------------------------------------------------

def stage_explain(args):
    print("\n[explain] what a kernel averages, and three checks against theory")
    k, D = 2.0, 0.4
    sig = np.sqrt(D / k)
    field = ornstein_uhlenbeck(d=1, k=k)
    n_steps = 2000 if args.quick else 4000
    traj = simulate(field, n_walkers=400, n_steps=n_steps, dt=2e-3, D=D,
                    seed=0, burn_in=5.0, boundary="none", check=False)
    th = st.active()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))

    # (a) the effective weight ------------------------------------------------
    ax = axes[0, 0]
    h_demo, q = 0.25, 1.3 * sig
    xs = np.linspace(-3.2 * sig, 3.2 * sig, 600)
    rho = np.exp(-xs ** 2 / (2 * sig ** 2))
    K = np.exp(-(xs - q) ** 2 / (2 * h_demo ** 2))
    eff = K * rho
    w_box = np.sqrt(12.0) * h_demo
    ax.fill_between(xs, 0, rho / rho.max(), color=th.ink_muted, alpha=0.15,
                    lw=0, label=r"stationary density $\rho_{ss}$")
    ax.plot(xs, K, lw=2, **{**sty(LL), "marker": None},
            label=fr"Gaussian kernel, $h={h_demo}$")
    lo_edge = q - w_box / 2
    ax.plot([lo_edge, lo_edge, lo_edge + w_box, lo_edge + w_box],
            [0, 1, 1, 0], lw=1.6, **{**sty(BINNED), "marker": None},
            label="box of the same variance")
    ax.plot(xs, eff / eff.max(), lw=2.2, color=th.ink, dashes=(4, 2),
            label=r"what is actually averaged: kernel $\times\ \rho$")
    centre = q * sig ** 2 / (sig ** 2 + h_demo ** 2)
    ax.axvline(q, color=th.ink_muted, lw=1)
    ax.axvline(centre, color=th.ink, lw=1, dashes=(1, 2))
    ax.annotate("query $q$", (q + 0.03, 1.08), ha="left", fontsize=9,
                color=th.ink_secondary, annotation_clip=False)
    ax.annotate("centre of mass", (centre - 0.03, 1.08), ha="right",
                fontsize=9, color=th.ink, annotation_clip=False)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("$x$")
    ax.set_ylabel("weight (each curve scaled to 1)")
    ax.set_title("(a) Denser data on one side drags the average towards it")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(False)

    # (b) the three estimates at one smoothing scale ----------------------
    ax = axes[0, 1]
    bins_same = int(round(float(field.box.length[0]) / w_box))
    fits = {BINNED: BinnedKramersMoyal(bins=bins_same, min_count=1).fit(traj),
            NW: KernelRegression(bandwidth=h_demo, local_linear=False).fit(traj),
            LL: KernelRegression(bandwidth=h_demo).fit(traj)}
    xq = np.linspace(-2.5 * sig, 2.5 * sig, 500)[:, None]
    ax.plot(xq[:, 0], -k * xq[:, 0], color=th.ink, lw=1.2, dashes=(3, 2),
            label="truth  $b=-kx$")
    for lab in ORDER:
        s_ = sty(lab)
        ax.plot(xq[:, 0], fits[lab].predict(xq)[:, 0], lw=2,
                color=s_["color"], label=lab)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\hat b(x)$")
    ax.set_title(fr"(b) Same smoothing scale ($h={h_demo}$), three answers")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) the shrinkage law ------------------------------------------------
    ax = axes[1, 0]
    hs = np.array([0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3])
    xr = np.linspace(-sig, sig, 101)[:, None]
    ratio, ll_slope = [], []
    for h in hs:
        s_nw = np.polyfit(xr[:, 0], KernelRegression(
            bandwidth=h, local_linear=False).fit(traj).predict(xr)[:, 0], 1)[0]
        s_ll = np.polyfit(xr[:, 0], KernelRegression(
            bandwidth=h).fit(traj).predict(xr)[:, 0], 1)[0]
        ratio.append(s_nw / s_ll)
        ll_slope.append(s_ll / -k)
    ratio = np.array(ratio)
    hh = np.linspace(0, hs.max() * 1.05, 200)
    ax.plot(hh, 1 / (1 + hh ** 2 / sig ** 2), color=th.ink, lw=1.6,
            label=r"exact: $1/(1+h^2/\sigma_{ss}^2)$")
    ax.plot(hh, 1 - hh ** 2 / sig ** 2, color=th.ink_muted, lw=1.2,
            dashes=(3, 2), label=r"leading order: $1-h^2k/D$")
    ax.plot(hs, ratio, linestyle="none", markersize=8,
            marker=sty(NW)["marker"], color=sty(NW)["color"],
            label="measured: NW slope / LL slope")
    ax.plot(hs, ll_slope, linestyle="none", markersize=7,
            marker=sty(LL)["marker"], color=sty(LL)["color"],
            label="measured: LL slope / true slope")
    ax.axhline(1.0, color=th.ink_muted, lw=0.8)
    ax.set_ylim(0.55, 1.08)
    ax.set_xlabel("bandwidth  $h$")
    ax.set_ylabel("fitted slope, as a fraction")
    ax.set_title("(c) On OU, Nadaraya-Watson shrinks by exactly this much")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    err_law = float(np.max(np.abs(ratio - 1 / (1 + hs ** 2 / sig ** 2))))
    print(f"  NW/LL slope ratio vs exact law: max deviation {err_law:.4f}")

    # (d) bias order, noise-free ---------------------------------------------
    ax = axes[1, 1]
    f2 = field2d()
    tr2 = run2d(f2, D=0.3, n_walkers=256 if args.quick else 512, n_steps=3000)
    x2, _ = tr2.drift_target()
    y2 = f2.drift(x2)

    def clean(cls):
        class Clean(cls):
            def regression_data(self, traj):
                return x2, y2
        return Clean

    q2 = make_eval_set(f2, 0.3, n=20000, seed=5).x
    bt = f2.drift(q2)
    scale = float(np.sqrt(np.mean(np.sum(bt ** 2, -1))))

    def rms(b):
        return float(np.sqrt(np.mean(np.sum((b - bt) ** 2, -1)))) / scale

    L2 = float(f2.box.length[0])
    bins_list = np.array([8, 12, 16, 24, 32, 48])
    e_rand, e_centre = [], []
    for b in bins_list:
        est = clean(BinnedKramersMoyal)(bins=int(b), min_count=1).fit(tr2)
        w = L2 / b
        c = (np.floor((q2 - f2.box.lo) / w) + 0.5) * w + f2.box.lo
        e_rand.append(rms(est.predict(q2)))
        e_centre.append(float(np.sqrt(np.mean(np.sum(
            (est.predict(q2) - f2.drift(c)) ** 2, -1)))) / scale)
    heq = h_equivalent(bins_list, L2)
    kh = np.array([0.012, 0.018, 0.027, 0.04, 0.06, 0.09])
    e_nw = [rms(clean(KernelRegression)(bandwidth=h, local_linear=False)
                .fit(tr2).predict(q2)) for h in kh]
    e_ll = [rms(clean(KernelRegression)(bandwidth=h).fit(tr2).predict(q2))
            for h in kh]
    slopes = {}
    for name, xv, yv in (("binned, anywhere in the cell", heq, e_rand),
                         ("binned, at the cell centre", heq, e_centre),
                         (NW, kh, e_nw), (LL, kh, e_ll)):
        slopes[name] = float(np.polyfit(np.log(xv), np.log(yv), 1)[0])
    ax.plot(heq, e_rand, lw=2, **sty(BINNED),
            label=f"binned, anywhere in the cell  (slope "
                  f"{slopes['binned, anywhere in the cell']:.2f})")
    ax.plot(heq, e_centre, lw=1.4, color=sty(BINNED)["color"],
            dashes=(3, 2), marker="o", markersize=4, markerfacecolor="none",
            label=f"binned, at the cell centre  (slope "
                  f"{slopes['binned, at the cell centre']:.2f})")
    ax.plot(kh, e_nw, lw=2, **sty(NW), label=f"{NW}  (slope {slopes[NW]:.2f})")
    ax.plot(kh, e_ll, lw=2, **sty(LL), label=f"{LL}  (slope {slopes[LL]:.2f})")
    _style_axis_log(ax)
    ax.set_xlabel(r"smoothing scale  $h$  (bins: cell width$/\sqrt{12}$)")
    ax.set_ylabel(r"error with noise-free targets  $\|\hat b-b\|/\|b\|$")
    ax.set_title("(d) Bins are first-order accurate; kernels second")
    ax.legend(loc="upper left", fontsize=8)
    print("  bias slopes: " + ", ".join(f"{k_}: {v:.2f}"
                                        for k_, v in slopes.items()))

    fig.tight_layout(h_pad=2.5)
    save(fig, "01_how_the_kernel_works",
         f"(a)-(c): 1D Ornstein-Uhlenbeck, k = {k}, D = {D}, sigma_ss = "
         f"{sig:.3f}, {traj.n_transitions:,} transitions. (a) The kernel is "
         f"centred on the query, but the samples it weighs are not uniform: "
         f"the product of kernel and density peaks at q sigma^2/(sigma^2+h^2), "
         f"closer to the dense middle. Nadaraya-Watson therefore reports the "
         f"drift from there. (b) At one common smoothing scale: bins give a "
         f"staircase, Nadaraya-Watson a shrunken line, local-linear the right "
         f"line -- it fits a slope inside the window, so the lopsided weights "
         f"no longer move the intercept. (c) The shrinkage predicted in closed "
         f"form, measured as the ratio of the two kernel fits so the shared "
         f"noise cancels; worst deviation {err_law:.4f}. Local-linear's own "
         f"slope scatters by a few percent around 1 -- sampling noise, largest "
         f"at small h where the window holds the fewest samples, and common to "
         f"both fits, which is exactly why the ratio is the clean test. "
         f"(d) 2D random field, "
         f"targets replaced by the exact drift so only bias remains. A bin "
         f"answers with one number for its whole cell, so a query off-centre "
         f"is wrong by grad b times the offset: first order. At the centre, or "
         f"with a kernel centred on the query, the linear term cancels.")


# --------------------------------------------------------------------------
# stage 2 -- choosing the bandwidth
# --------------------------------------------------------------------------

def stage_bandwidth(args):
    print("\n[bandwidth] cross-validation against the truth it never sees")
    field = field2d()
    D = 0.3
    L = float(field.box.length[0])
    budgets = [(128, 3000), (1024, 3000 if not args.quick else 1000)]
    th = st.active()
    fig, axes = plt.subplots(2, len(budgets), figsize=(13.4, 9.4),
                             sharex="col")
    summary = []
    for col, (N, T) in enumerate(budgets):
        traj = run2d(field, D=D, n_walkers=N, n_steps=T)
        ev = make_eval_set(field, D, n=12000, seed=991, traj=traj)
        ms_b = float(np.mean(np.sum(field.drift(ev.x) ** 2, -1)))
        top, bot = axes[0, col], axes[1, col]
        for lab, make in ((BINNED, lambda: BinnedKramersMoyal(bins="cv")),
                          (NW, lambda: KernelRegression(local_linear=False)),
                          (LL, lambda: KernelRegression())):
            t0 = time.time()
            est = make().fit(traj)
            cv = est.cv_
            if lab == BINNED:
                grid = cv["bins"]
                xs = h_equivalent(grid, L)
                truth = [drift_error(BinnedKramersMoyal(bins=int(b)).fit(traj),
                                     field, ev, use_support=False)["nrmse"]
                         for b in grid]
                x_sel = h_equivalent(est.bins, L)
                x_sil = None
            else:
                grid = cv["h"]
                xs = np.asarray(grid)
                truth = [drift_error(KernelRegression(
                    bandwidth=float(h), local_linear=est.local_linear).fit(traj),
                    field, ev, use_support=False)["nrmse"] for h in grid]
                x_sel = est.h_
                x_sil = est.h_silverman_
            truth = np.asarray(truth)
            chosen = drift_error(est, field, ev, use_support=False)["nrmse"]
            s_ = sty(lab)
            top.plot(xs, truth, lw=2, label=lab, **s_)
            top.scatter([x_sel], [chosen], s=140, facecolor="none",
                        edgecolor=s_["color"], linewidths=2, zorder=6)
            # Linear axis: the excess is exactly 0 at the minimum, and on a
            # log axis that zero becomes a bottomless artificial valley.
            excess = (cv["loss"] - cv["loss"].min()) / ms_b
            bot.plot(xs, excess, lw=2, label=lab, **s_)
            bot.axvline(x_sel, color=s_["color"], lw=1, dashes=(2, 2),
                        alpha=0.8)
            row = dict(budget=N * T, method=lab, cv_choice=float(x_sel),
                       cv_nrmse=chosen, best_nrmse=float(truth.min()),
                       best_at=float(xs[int(np.argmin(truth))]))
            if x_sil is not None:
                sil = drift_error(KernelRegression(
                    bandwidth=float(x_sil), local_linear=est.local_linear)
                    .fit(traj), field, ev, use_support=False)["nrmse"]
                top.scatter([x_sil], [sil], s=70, marker="x",
                            color=s_["color"], linewidths=2, zorder=6)
                row.update(silverman=float(x_sil), silverman_nrmse=sil)
            summary.append(row)
            print(f"  NT={N * T:>9,} {lab:<24} CV -> {chosen:.3f}   "
                  f"best on grid {truth.min():.3f}   "
                  + (f"Silverman -> {row['silverman_nrmse']:.3f}   "
                     if x_sil is not None else "")
                  + f"[{time.time() - t0:.1f}s]")
        _style_axis_log(top)
        top.set_title(f"$N\\,T$ = {N * T / 1e3:,.0f}k transitions")
        top.set_ylabel(r"true error  $\|\hat b-b\|/\|b\|$" if col == 0 else "")
        bot.set_xscale("log")
        bot.set_ylim(-0.02, 0.8)
        bot.grid(True, which="both", alpha=0.3)
        bot.set_xlabel(r"smoothing scale  $h$  (bins: cell width$/\sqrt{12}$)")
        bot.set_ylabel("cross-validation loss above its minimum\n"
                       r"(units of mean $|b|^2$)" if col == 0 else "")
        if col == 0:
            top.legend(loc="upper center", fontsize=8)
            top.text(0.02, 0.03, "circle: what CV chose\n"
                     "cross: Silverman's rule",
                     transform=top.transAxes, fontsize=8.5,
                     color=th.ink_muted, va="bottom")
    fig.tight_layout()
    worst = max(r["cv_nrmse"] / r["best_nrmse"] for r in summary)
    sil_pen = [r["silverman_nrmse"] / r["cv_nrmse"] for r in summary
               if "silverman_nrmse" in r]
    save(fig, "02_choosing_the_bandwidth",
         f"2D random field, D = {D}. Top: the true error of each method across "
         f"its own smoothing grid, computed against the ground truth. Bottom: "
         f"what cross-validation sees instead -- mean squared error predicting "
         f"the raw increments of held-out walkers. That loss sits on a noise "
         f"floor hundreds of times larger than the signal, but the floor does "
         f"not depend on h, so its minimum is where the true error's minimum "
         f"is. Across these panels CV's choice is never more than "
         f"{100 * (worst - 1):.0f}% worse than the best point on the grid. "
         f"Silverman's rule, a density-estimation formula that cannot see the "
         f"noise level, lands {100 * (min(sil_pen) - 1):.0f}-"
         f"{100 * (max(sil_pen) - 1):.0f}% worse. The selected h sits slightly "
         f"left of the CV minimum by design: it was tuned on 2/3 of the "
         f"walkers and is rescaled to the full data by the method's own rate, "
         f"n^(-1/(d+4)) for a kernel and n^(-1/(d+2)) for a bin width. Bottom "
         f"panels are clipped above; the steep left arms continue off-axis.")


# --------------------------------------------------------------------------
# stage 3 -- side by side
# --------------------------------------------------------------------------

def _recovery_figure(field, traj, D, name, headline, caption_fn):
    """Truth and three fits, quiver over error map.

    Three numbers per fit, because each answers a different question and any
    one of them alone misleads:

    - nrmse under ``rho_ss`` -- accuracy where the walkers actually are;
    - box coverage -- where the method is willing to answer at all;
    - error on the *shared* region, the part of the box all three support --
      the only like-for-like accuracy comparison, since each method's own
      support covers a different part of the box.
    """
    ev = make_eval_set(field, D, n=12000, seed=991, traj=traj)
    ests = {BINNED: BinnedKramersMoyal(bins="cv").fit(traj),
            NW: KernelRegression(local_linear=False).fit(traj),
            LL: KernelRegression().fit(traj)}
    box = field.box
    TX, TY = cell_centres(box, COARSE)
    tx, ty = vectors_at(field.drift, TX, TY)
    vmax = float(np.percentile(np.hypot(tx, ty), 99.0))

    maps = {lab: estimate_maps(field, ests[lab]) for lab in ORDER}
    shared = np.logical_and.reduce([np.isfinite(maps[lab][1]) for lab in ORDER])
    stats = {}
    for lab in ORDER:
        _, err, cov = maps[lab]
        stats[lab] = {
            "nrmse": drift_error(ests[lab], field, ev, use_support=False)["nrmse"],
            "box": cov,
            "shared": (float(np.sqrt(np.mean(err[shared] ** 2)))
                       if shared.any() else float("nan")),
            "own": float(np.sqrt(np.nanmean(err ** 2))),
            "knob": (f"{ests[lab].bins} bins" if lab == BINNED
                     else f"h = {ests[lab].h_:.3f}"),
        }
        print(f"  {lab:<24} {stats[lab]['knob']:<10} nrmse {stats[lab]['nrmse']:.3f}"
              f"  box {cov:.0%}  error on its own support "
              f"{stats[lab]['own']:.2f}  on shared region "
              f"{stats[lab]['shared']:.2f}")
    stats["shared_fraction"] = float(shared.mean())

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.3))
    quiver_panel(axes[0, 0], box, TX, TY, tx, ty, vmax=vmax,
                 title="Ground truth")
    st.hero(axes[1, 0], headline[0], headline[1], sub=headline[2])
    pooled = np.concatenate([m[1][np.isfinite(m[1])] for m in maps.values()])
    err_max = float(min(np.percentile(pooled, 97), 2.0)) if pooled.size else 1.0
    handles = None
    for j, lab in enumerate(ORDER, 1):
        (EX, EY, ex, ey, sup), err, cov = maps[lab]
        sd = stats[lab]
        q = quiver_panel(axes[0, j], box, EX, EY, ex, ey, vmax=vmax,
                         support=sup, title=f"{lab}\n{sd['knob']}, CV")
        im = error_panel(axes[1, j], box, err, vmax=err_max,
                         title=f"nrmse {sd['nrmse']:.3f} on " r"$\rho_{ss}$"
                               f"\nbox {cov:.0%}  ·  shared {sd['shared']:.2f}")
        handles = handles or (q, im)
    fig.tight_layout()
    row_colorbar(fig, axes[0, 1:], handles[0], r"$|b|$")
    row_colorbar(fig, axes[1, 1:], handles[1],
                 r"$|\hat b - b| \,/\, \mathrm{RMS}|b|$", extend="max")
    save(fig, name, caption_fn(stats))
    return stats


def stage_recovery(args):
    print("\n[recovery] side by side")
    field = field2d()
    T = 1500 if args.quick else 3000
    D = 0.3
    traj = run2d(field, D=D, n_walkers=128, n_steps=T)

    def cap_a(sd):
        return (
            f"Same data for all three ({traj.n_transitions:,} transitions); each "
            f"picks its own smoothing by cross-validation over walkers. Bins are "
            f"drawn at their own cell centres, kernels on a 24x24 grid; arrow "
            f"length is on one scale throughout. Under rho_ss the errors are "
            f"{sd[BINNED]['nrmse']:.3f} / {sd[NW]['nrmse']:.3f} / "
            f"{sd[LL]['nrmse']:.3f}. The binned error map shows its own grid: "
            f"lowest at cell centres, growing towards the edges -- the "
            f"first-order bias of figure 01(d), visible directly. 'box' is the "
            f"share of the box each method answers for; 'shared' is its RMS "
            f"error over the part of the box all three answer for "
            f"({sd['shared_fraction']:.0%} here), uniformly weighted.")

    _recovery_figure(field, traj, D, "03a_recovery_well_sampled",
                     (f"{traj.n_transitions / 1e3:.0f}k", "observed transitions",
                      f"128 walkers x {T} steps, D = {D}"), cap_a)

    print("  --- low noise, poorly explored ---")
    Dl = 0.05
    traj = run2d(field, D=Dl, n_walkers=128, n_steps=T)

    def cap_b(sd):
        return (
            f"Low noise: walkers stay in the basins they start in. Under rho_ss "
            f"-- where the data is -- local-linear looks best by far "
            f"({sd[LL]['nrmse']:.3f} against {sd[NW]['nrmse']:.3f} and "
            f"{sd[BINNED]['nrmse']:.3f}). Across the box the picture is "
            f"different. Cross-validation tunes each bandwidth on held-out "
            f"walkers, so it optimises accuracy where walkers go; local-linear "
            f"can afford a wide kernel there (its bias ignores the steep density "
            f"gradients that force Nadaraya-Watson narrow), and a wide kernel "
            f"reaches data from far away -- hence it answers for "
            f"{sd[LL]['box']:.0%} of the box against {sd[NW]['box']:.0%} and "
            f"{sd[BINNED]['box']:.0%}. Support here means 'at least 8 "
            f"transitions within 2h', and with noise ~4|b| per increment 8 "
            f"samples still leave an error of order |b|: it certifies that data "
            f"was present, not that the answer is accurate. Inside their own "
            f"support the three average {sd[BINNED]['own']:.2f} / "
            f"{sd[NW]['own']:.2f} / {sd[LL]['own']:.2f} over the box, and on the "
            f"region all three share ({sd['shared_fraction']:.0%} of the box) "
            f"{sd[BINNED]['shared']:.2f} / {sd[NW]['shared']:.2f} / "
            f"{sd[LL]['shared']:.2f}. The gap between those numbers and the "
            f"rho_ss column is the extrapolation penalty.")

    _recovery_figure(field, traj, Dl, "03b_recovery_poorly_explored",
                     (f"D = {Dl}", "walkers stay in their basins",
                      f"128 walkers x {T} steps"), cap_b)


# --------------------------------------------------------------------------
# stages 4 and 5 -- sweeps
# --------------------------------------------------------------------------

def _agg(df, x):
    ok = df[df.status == "ok"]
    return (ok.groupby([x, "label"]).nrmse.agg(["mean", "min", "max"])
              .reset_index().rename(columns={"mean": "nrmse"}))


def stage_sweeps(args):
    print("\n[sweeps] walkers, D, measurement noise -- three methods")
    n_steps = 1000 if args.quick else 3000
    seeds = [0] if args.quick else [0, 1]
    est = methods()
    walkers = [16, 64, 256] if args.quick else [16, 32, 64, 128, 256, 512, 1024]
    df_n = run_sweep(sweep_grid(use_support=False, d=2, n_walkers=walkers, n_steps=[n_steps],
                                D=0.3, seed=seeds), est,
                     cache=DATA / "sweep_walkers.csv")
    Ds = [0.05, 0.2, 0.8] if args.quick else [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
    df_d = run_sweep(sweep_grid(use_support=False, d=2, n_walkers=256, n_steps=[n_steps], D=Ds,
                                seed=seeds), est, cache=DATA / "sweep_D.csv")
    sig = [0.0, 0.005, 0.02] if args.quick else \
        [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
    df_s = run_sweep(sweep_grid(use_support=False, d=2, n_walkers=256, n_steps=[n_steps], D=0.3,
                                obs_noise=sig, seed=seeds), est,
                     cache=DATA / "sweep_obs_noise.csv")

    band = ("min", "max") if len(seeds) > 1 else None
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    plot_error_curves(_agg(df_n, "n_walkers"), axes[0], x="n_walkers",
                      y="nrmse", band=band, color_slots=SLOTS,
                      direct_labels=False, title="More walkers")
    plot_error_curves(_agg(df_d, "D"), axes[1], x="D", y="nrmse",
                      slope_guide=None, band=band, color_slots=SLOTS,
                      direct_labels=False, title="Noise level")
    plot_error_curves(_agg(df_s, "obs_noise"), axes[2], x="obs_noise",
                      y="nrmse", logx=False, slope_guide=None, band=band,
                      color_slots=SLOTS, direct_labels=False,
                      title="Measurement noise")
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    for i, ax in enumerate(axes):
        leg = ax.get_legend()
        if leg is not None and i > 0:
            leg.remove()

    # Headline ratios, computed from the table rather than typed in.
    g = _agg(df_n, "n_walkers").pivot(index="n_walkers", columns="label",
                                      values="nrmse").sort_index()
    r_bl = g[BINNED] / g[LL]
    gd = _agg(df_d, "D").pivot(index="D", columns="label",
                               values="nrmse").sort_index()
    r_nl = gd[NW] / gd[LL]
    print(f"  binned / local-linear across the walker sweep: "
          f"{r_bl.iloc[0]:.2f}x at N={g.index[0]} -> {r_bl.iloc[-1]:.2f}x at "
          f"N={g.index[-1]}")
    print("  NW / LL across D: " + ", ".join(
        f"{v:.2f} (D={k:g})" for k, v in r_nl.items()))
    fig.tight_layout()
    save(fig, "04_sweeps",
         f"2D random field, {n_steps} steps per walker; band spans "
         f"{len(seeds)} seed(s). Every method chooses its own smoothing by "
         f"cross-validation at every point. Left: binning's error is "
         f"{r_bl.iloc[0]:.1f}x local-linear's with {g.index[0]} walkers and "
         f"{r_bl.iloc[-1]:.1f}x with {g.index[-1]} -- the gap widens with data, "
         f"because a first-order bias shrinks more slowly than a second-order "
         f"one as the smoothing tightens. Middle: local-linear's lead over "
         f"Nadaraya-Watson is {r_nl.iloc[0]:.2f}x at D = {gd.index[0]:g} and "
         f"{r_nl.iloc[-1]:.2f}x at D = {gd.index[-1]:g}. That is the 1/D "
         f"density-gradient bias of figure 01: decisive at low noise, "
         f"irrelevant once variance dominates. Right: position noise biases "
         f"the target itself -- noisy positions sit downhill of the true ones "
         f"(Tweedie's formula), so every method that averages dx/dt converges "
         f"to b (1 + sigma^2/(D dt)). It is quadratic in sigma, and no smoothing "
         f"choice removes it, which is why the three curves converge.")


def stage_dims(args):
    print("\n[dims] d = 2, 3, 5, 10 -- three methods")
    dims = [2, 3, 5, 10]
    walkers = [32, 128, 512] if args.quick else [32, 128, 512, 2048]
    n_steps = 800 if args.quick else 3000
    cfgs = sweep_grid(use_support=False, d=dims, n_walkers=walkers, n_steps=[n_steps], D=0.3,
                      seed=0, n_eval=8000)
    df = run_sweep(cfgs, methods(), cache=DATA / "sweep_dimension.csv")
    ok = df[df.status == "ok"].copy()
    if ok.empty:
        print("  (nothing ran)")
        return

    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    ymin = max(1e-2, ok.nrmse.min() * 0.7)
    ymax = ok.nrmse.max() * 1.4
    for i, d in enumerate(dims):
        ax = fig.add_subplot(gs[0, i])
        sub = ok[ok.d == d]
        plot_error_curves(sub, ax, x="n_transitions", y="nrmse",
                          slope_guide=None, color_slots=SLOTS,
                          direct_labels=False, title=f"$d$ = {d}",
                          ylabel="normalised error")
        ax.set_ylim(ymin, ymax)
        if i > 0:
            ax.set_ylabel("")       # plot_error_curves treats "" as "default"
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        else:
            ax.legend(loc="lower left", fontsize=8)

    top = ok[ok.n_walkers == max(walkers)]
    ax = fig.add_subplot(gs[1, 0:2])
    for lab in ORDER:
        s = top[top.label == lab].sort_values("d")
        ax.plot(s.d, s.nrmse, lw=2, label=lab, **sty(lab))
    ax.axhline(1.0, color=st.active().ink_muted, lw=1, dashes=(3, 2))
    ax.text(dims[-1], 1.0, "  predicting zero", va="bottom", ha="right",
            fontsize=8, color=st.active().ink_muted)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xlabel("dimension  $d$")
    ax.set_ylabel("normalised error")
    ax.set_title(f"At the largest budget ({int(top.n_transitions.iloc[0]):,} "
                 "transitions)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # What cross-validation did about the dimension. Coverage is no longer
    # informative once every method tunes its own smoothing -- all of them
    # cover everything, by retreating to coarser and coarser averages -- so
    # show the retreat itself: the chosen smoothing scale relative to the box.
    ax = fig.add_subplot(gs[1, 2:4])
    L = 2.0                                  # the default periodic box side
    top = top.copy()
    top["smooth"] = np.where(
        top.label == BINNED,
        h_equivalent(top.param_bins.astype(float).clip(lower=1), L) / L,
        top.param_bandwidth.astype(float) / L)
    for lab in ORDER:
        s_ = top[top.label == lab].sort_values("d")
        ax.plot(s_.d, s_.smooth, lw=2, label=lab, **sty(lab))
    th = st.active()
    one_bin = float(h_equivalent(1, L) / L)
    ax.axhline(one_bin, color=th.ink_muted, lw=1, dashes=(3, 2))
    ax.text(dims[0], one_bin * 1.08, "  one bin: the global mean",
            fontsize=8, color=th.ink_muted, va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xlabel("dimension  $d$")
    ax.set_ylabel("smoothing scale chosen by CV / box side\n"
                  r"(bins: cell width$/\sqrt{12}$)")
    ax.set_title("How far each method has to retreat")
    ax.grid(True, which="both", alpha=0.3)

    piv = top.pivot(index="d", columns="label", values="nrmse")
    smo = top.pivot(index="d", columns="label", values="smooth")
    bins_top = top[top.label == BINNED].set_index("d").param_bins
    for d in dims:
        if d in piv.index:
            print(f"  d={d:<2} " + "  ".join(
                f"{lab}: {piv.loc[d, lab]:.3f} (smoothing {smo.loc[d, lab]:.3f})"
                for lab in ORDER if lab in piv.columns))
    fig.tight_layout()
    lines = "; ".join(
        f"d = {d}: " + ", ".join(f"{lab.split(',')[-1].strip()} "
                                  f"{piv.loc[d, lab]:.2f}"
                                  for lab in ORDER if lab in piv.columns)
        for d in dims if d in piv.index)
    d_hi = dims[-1]
    save(fig, "05_error_vs_dimension",
         f"D = 0.3, {n_steps} steps per walker, one seed. Grid-backed field to "
         f"d = 3, mesh-free above. Every method cross-validates its smoothing "
         f"at every point, and one bin -- the global mean -- is an allowed "
         f"answer. Above d = 3 the kernel uses direct sums on a compressed copy "
         f"of the data (increments merged into short per-walker blocks, then "
         f"subsampled to 200k rows), so its high-d numbers are not built from "
         f"every transition. Error at the largest budget -- {lines}. Bottom "
         f"right is the mechanism: as d grows, cross-validation widens every "
         f"method's neighbourhood until at d = {d_hi} binning uses "
         f"{int(bins_top.get(d_hi, -1))} bin(s) per axis and the kernels sit at "
         f"{smo.loc[d_hi, LL]:.2f} of the box side. An estimate averaged over "
         f"most of the space is close to the global mean, which is why the "
         f"errors converge on 1.0, the score of predicting zero: at this budget "
         f"there is nothing local left to learn. Kernels degrade more gracefully "
         f"than bins on the way there, but neither escapes.")


STAGES = {"explain": stage_explain, "bandwidth": stage_bandwidth,
          "recovery": stage_recovery, "sweeps": stage_sweeps,
          "dims": stage_dims}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", nargs="*", default=list(STAGES),
                    choices=list(STAGES))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    args = ap.parse_args()
    use_style(args.theme)
    for p in (FIG, DATA):
        p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name in args.which:
        STAGES[name](args)
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
