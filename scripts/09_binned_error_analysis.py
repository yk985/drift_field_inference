"""The error budget of rung 1, derived on paper and measured here.

Writes to ``outputs/binned_km/`` (figure ``07_error_budget.png`` and a JSON of
every number) and copies the figure into ``docs/figs/`` for the LaTeX note
``docs/binned_km_error_analysis.tex``.

    python scripts/09_binned_error_analysis.py --quick     # fewer replicas
    python scripts/09_binned_error_analysis.py

The claims under test, for a cell of width ``h`` holding ``n`` increments:

``variance``    Var(b_hat) = 2D / (n dt) = 2D / T_occupation, per component.
                Since n = M rho h^d, this is 2D / (M rho h^d dt): independent of
                dt at fixed total observation time, and growing like h^-d.
``bias``        the cell answers with one value, so at a query a distance delta
                from the centre the error is grad b . delta -- first order in h,
                mean square |grad b|^2 h^2 / 12.
``optimum``     balancing the two gives h* = (A/2B)^(1/3) in 1D with
                A = 2 D L / (M dt) and B = <|grad b|^2>/12.
``obs noise``   x~ = x + eps adds 2 sigma^2 per increment (a *difference* of two
                draws), telescoping to 2 sigma^2 per *visit* inside a cell; and
                -- the effect that actually matters -- biases the estimate to
                b (1 + sigma^2/(D dt)), because the cell assignment selects on
                eps whenever the density has a gradient.

Everything is measured on a 1D periodic field b(x) = -a sin(pi x) on [-1, 1],
where rho_ss = exp((a / pi D) cos(pi x)) / Z is known in closed form. ``R``
independent replicas are simulated side by side so that the variance of a
per-cell estimate is *measured across replicas*, not inferred from one fit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfi.viz import style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "binned_km"

# -- the test system -------------------------------------------------------
A, D, DT = 1.0, 0.3, 1e-3
LO, L = -1.0, 2.0
BIN_LIST = (4, 8, 16, 32, 64, 128)
SIGMAS = (0.0, 0.005, 0.01, 0.02)


def b_of(x):
    return -A * np.sin(np.pi * x)


def grad_b(x):
    return -A * np.pi * np.cos(np.pi * x)


def wrap(x):
    return (x - LO) % L + LO


def rho_moments():
    """``<|grad b|^2>_rho`` for the optimal-width prediction."""
    xf = np.linspace(LO, LO + L, 400001)
    rho = np.exp((A / (np.pi * D)) * np.cos(np.pi * xf))
    rho /= np.trapezoid(rho, xf)
    return float(np.trapezoid(grad_b(xf) ** 2 * rho, xf))


def pass_(sigma, R, W, T, burn, seed, bin_on_true=False):
    """One simulation pass, binned at every width in ``BIN_LIST`` at once.

    Returns, per width: ``b_hat`` of shape ``(R, bins)``, the counts, the number
    of contiguous visit segments (for the telescoping term), and the first two
    moments of the *true* ``b`` inside each cell (for the readout bias).
    """
    rng = np.random.default_rng(seed)
    n = R * W
    x = rng.uniform(LO, LO + L, n)
    amp = np.sqrt(2 * D * DT)
    for _ in range(burn):
        x = wrap(x + b_of(x) * DT + amp * rng.standard_normal(n))

    acc = {nb: {k: np.zeros(R * nb) for k in ("sum", "cnt", "run", "bs", "bsq")}
           for nb in BIN_LIST}
    reps = {nb: np.repeat(np.arange(R), W) * nb for nb in BIN_LIST}
    prev = {nb: np.full(n, -1) for nb in BIN_LIST}
    e_prev = sigma * rng.standard_normal(n)

    for _ in range(T):
        dx = b_of(x) * DT + amp * rng.standard_normal(n)
        e_next = sigma * rng.standard_normal(n)
        dxo = dx + e_next - e_prev
        xb = x if bin_on_true else x + e_prev
        e_prev = e_next
        bx = b_of(x)
        for nb in BIN_LIST:
            h = L / nb
            cell = np.clip(((wrap(xb) - LO) / h).astype(int), 0, nb - 1)
            flat = reps[nb] + cell
            a = acc[nb]
            a["sum"] += np.bincount(flat, weights=dxo, minlength=R * nb)
            a["cnt"] += np.bincount(flat, minlength=R * nb)
            a["run"] += np.bincount(flat[cell != prev[nb]], minlength=R * nb)
            a["bs"] += np.bincount(flat, weights=bx, minlength=R * nb)
            a["bsq"] += np.bincount(flat, weights=bx * bx, minlength=R * nb)
            prev[nb] = cell
        x = wrap(x + dx)

    out = {}
    for nb in BIN_LIST:
        a = acc[nb]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[nb] = dict(
                bhat=(a["sum"] / a["cnt"] / DT).reshape(R, nb),
                cnt=a["cnt"].reshape(R, nb),
                run=a["run"].reshape(R, nb),
                bbar=(a["bs"] / a["cnt"]).reshape(R, nb),
                bsq=(a["bsq"] / a["cnt"]).reshape(R, nb),
            )
    return out


def budget(res, nb):
    """Density-weighted variance, readout bias^2 and total MSE at one width."""
    r = res[nb]
    cnt, bhat = r["cnt"], r["bhat"]
    ok = np.isfinite(bhat).all(axis=0) & (cnt.min(axis=0) > 32)
    w = cnt.mean(axis=0)[ok]
    w = w / w.sum()
    var = bhat[:, ok].var(axis=0, ddof=1)
    # The within-cell spread of the true b *is* the piecewise-constant readout
    # error: predict returns one value for queries spread over the whole cell.
    read = np.maximum(r["bsq"].mean(0)[ok] - r["bbar"].mean(0)[ok] ** 2, 0.0)
    pred_var = 2 * D / (cnt.mean(axis=0)[ok] * DT)
    return dict(n_cells=int(ok.sum()), n_med=float(np.median(cnt.mean(0)[ok])),
                var=float(w @ var), var_pred=float(w @ pred_var),
                read=float(w @ read), total=float(w @ (var + read)),
                ratio_med=float(np.median(var / pred_var)))


def noise_row(r, sigma, D_, dt):
    """Inflation and the two competing variance formulas at one sigma."""
    ok = r["cnt"].min(axis=0) > 32
    n, nr = r["cnt"].mean(0)[ok], r["run"].mean(0)[ok]
    v = r["bhat"][:, ok].var(axis=0, ddof=1)
    diff = 2 * D_ / (n * dt)
    naive = diff + 2 * sigma ** 2 / (n * dt ** 2)
    tele = diff + 2 * sigma ** 2 * nr / (n ** 2 * dt ** 2)
    infl = r["bhat"].mean(0)[ok] / r["bbar"].mean(0)[ok]
    return dict(sigma=sigma, predicted=1 + sigma ** 2 / (D_ * dt),
                inflation=float(np.median(infl)),
                inflation_lo=float(np.percentile(infl, 10)),
                inflation_hi=float(np.percentile(infl, 90)),
                steps_per_visit=float(np.median(n / nr)),
                var_over_diff=float(np.median(v / diff)),
                var_over_naive=float(np.median(v / naive)),
                var_over_tele=float(np.median(v / tele)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    R, W, T, burn = (16, 100, 1500, 1000) if args.quick else (64, 200, 4000, 3000)
    M = W * T
    t0 = time.time()

    gb2 = rho_moments()
    res = {}
    for sg in SIGMAS:
        print(f"  pass sigma={sg:g} ...", flush=True)
        res[sg] = pass_(sg, R, W, T, burn, seed=11 + int(sg * 1000))
    print("  pass sigma=0.02, binning on the true position ...", flush=True)
    res["true"] = pass_(0.02, R, W, T, burn, seed=99, bin_on_true=True)

    # 1. the h budget at sigma = 0
    rows = [dict(bins=nb, h=L / nb, **budget(res[0.0], nb)) for nb in BIN_LIST]
    a_coef = 2 * D * L / (M * DT)          # density-averaged variance = A / h
    b_coef = gb2 / 12.0                    # readout^2 = B h^2
    h_star = (a_coef / (2 * b_coef)) ** (1 / 3)

    # 2. observation noise
    noise = [noise_row(res[sg][16], sg, D, DT) for sg in SIGMAS]
    split = noise_row(res["true"][16], 0.02, D, DT)

    data = dict(system=dict(a=A, D=D, dt=DT, box=[LO, LO + L], replicas=R,
                            walkers=W, steps=T, burn=burn, M=M, grad_b2=gb2),
                h_budget=rows, A=a_coef, B=b_coef, h_star=h_star,
                bins_star=L / h_star, noise=noise, bin_on_true=split)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / "error_analysis.json").write_text(json.dumps(data, indent=2))

    # -- figure ------------------------------------------------------------
    style.use_style("light")
    theme = style.active()
    fig, axg = plt.subplots(2, 2, figsize=(11.0, 9.4))
    axes = axg.ravel()
    c0, c1, c2 = (style.series_color(i) for i in (0, 1, 2))

    ax = axes[0]
    allv, allp = [], []
    for nb in BIN_LIST:
        rr = res[0.0][nb]
        ok = rr["cnt"].min(axis=0) > 32
        if not ok.any():
            continue
        allv.append(rr["bhat"][:, ok].var(axis=0, ddof=1))
        allp.append(2 * D / (rr["cnt"].mean(0)[ok] * DT))
    allv, allp = np.concatenate(allv), np.concatenate(allp)
    lim = [min(allp.min(), allv.min()) * 0.6, max(allp.max(), allv.max()) * 1.6]
    ax.plot(lim, lim, color=theme.ink_muted, lw=1, zorder=1)
    ax.scatter(allp, allv, s=18, color=c0, alpha=0.75, lw=0, zorder=2)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel=r"predicted  $2D/(n\,\Delta t)$",
           ylabel=r"measured  $\mathrm{Var}_R(\hat b)$")
    ax.set_title("every cell, every width\nmedian ratio "
                 f"{np.median(allv / allp):.2f}", loc="left")
    style.despine(ax)

    ax = axes[1]
    hs = np.array([r["h"] for r in rows])
    ax.plot(hs, [r["var"] for r in rows], "o-", color=c0, label="variance (measured)")
    ax.plot(hs, a_coef / hs, "--", color=c0, lw=1, label=r"$2DL/(M\,\Delta t\,h)$")
    ax.plot(hs, [r["read"] for r in rows], "s-", color=c1,
            label=r"readout bias$^2$ (measured)")
    ax.plot(hs, b_coef * hs ** 2, "--", color=c1, lw=1,
            label=r"$\langle|\nabla b|^2\rangle h^2/12$")
    ax.plot(hs, [r["total"] for r in rows], "^-", color=c2, lw=2, label="total MSE")
    ax.axvline(h_star, color=theme.ink_muted, lw=1, ls=":")
    ax.set(xscale="log", yscale="log", xlabel="cell width $h$",
           ylabel="mean square error")
    ax.annotate(f"$h^\\star={h_star:.3f}$\n({L / h_star:.0f} bins)",
                (h_star, ax.get_ylim()[0]), xytext=(5, 6),
                textcoords="offset points", fontsize=8,
                color=theme.ink_secondary, va="bottom")
    ax.set_title("the U curve, term by term", loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    style.despine(ax)

    ax = axes[2]
    sg = np.array([d["sigma"] for d in noise])
    meas = np.array([d["inflation"] for d in noise])
    lo = np.array([d["inflation_lo"] for d in noise])
    hi = np.array([d["inflation_hi"] for d in noise])
    fine = np.linspace(0, sg.max() * 1.05, 100)
    ax.plot(fine, 1 + fine ** 2 / (D * DT), "--", color=theme.ink_muted, lw=1.2,
            label=r"$1+\sigma_{\rm obs}^2/(D\Delta t)$")
    ax.fill_between(sg, lo, hi, color=c1, alpha=0.18, lw=0)
    ax.plot(sg, meas, "o-", color=c1, label=r"measured (bin on $\tilde x$)")
    ax.plot([0.02], [split["inflation"]], "D", color=c2, ms=7,
            label=r"bin on true $x$")
    ax.set(xlabel=r"$\sigma_{\rm obs}$", ylabel=r"$\hat b\,/\,\bar b$")
    ax.set_title("noise biases, it does not just blur", loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    style.despine(ax)

    # (d) the two competing noise-variance formulas, across cell widths.
    # Excess over the diffusion floor: the naive form predicts a constant
    # 1 + sigma^2/(D dt); telescoping predicts 1 + (sigma^2/(D dt)) n_vis/n,
    # which shrinks as the cell gets wide and visits get long.
    ax = axes[3]
    sig = 0.02
    hh, ex_noisy, ex_true, ex_tele = [], [], [], []
    for nb in BIN_LIST:
        rn_, rt_ = res[sig][nb], res["true"][nb]
        okn = rn_["cnt"].min(axis=0) > 32
        okt = rt_["cnt"].min(axis=0) > 32
        if not (okn.any() and okt.any()):
            continue
        hh.append(L / nb)
        n_, v_ = rn_["cnt"].mean(0)[okn], rn_["bhat"][:, okn].var(axis=0, ddof=1)
        ex_noisy.append(np.median(v_ / (2 * D / (n_ * DT))))
        m_, w_ = rt_["cnt"].mean(0)[okt], rt_["bhat"][:, okt].var(axis=0, ddof=1)
        ex_true.append(np.median(w_ / (2 * D / (m_ * DT))))
        ex_tele.append(1 + (sig ** 2 / (D * DT))
                       * np.median(rt_["run"].mean(0)[okt] / m_))
    hh = np.array(hh)
    ax.axhline(1 + sig ** 2 / (D * DT), color=theme.ink_muted, lw=1.2, ls="--",
               label=r"naive $1+\sigma^2/(D\Delta t)$")
    ax.plot(hh, ex_tele, ":", color=c2, lw=1.4,
            label=r"telescoped $1+\frac{\sigma^2}{D\Delta t}\frac{n_{\rm vis}}{n}$")
    ax.plot(hh, ex_true, "D-", color=c2, label=r"measured, bin on true $x$")
    ax.plot(hh, ex_noisy, "o-", color=c1, label=r"measured, bin on $\tilde x$")
    ax.axhline(1.0, color=theme.grid, lw=1, zorder=0)
    ax.set(xscale="log", yscale="log", xlabel="cell width $h$",
           ylabel=r"variance $/$ diffusion floor $2D/(n\Delta t)$")
    ax.set_title(r"where the noise variance goes ($\sigma_{\rm obs}=0.02$)", loc="left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.8)   # headroom so the legend clears the data
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    style.despine(ax)

    style.figure_caption(fig, textwrap.fill("1D periodic field $b=-\\sin\\pi x$, $D=0.3$, "
        f"$\\Delta t=10^{{-3}}$, {R} independent replicas of {W}$\\times${T} steps. "
        "(a) the variance of a cell's estimate is $2D/(n\\Delta t)$ at every width "
        "and every local density. (b) variance $\\propto 1/h$ against a "
        "first-order readout bias $\\propto h^2$, with the predicted optimum. "
        "(c) measurement noise inflates the drift by $1+\\sigma^2/(D\\Delta t)$ "
        "-- an effect of the cell assignment, which vanishes when the cell is "
        "chosen with the true position. (d) with the true position the excess "
        "variance follows the telescoped form at every width; with the noisy one it "
        "sits at 3-4x the diffusion floor, above the naive bound and explained by "
        "neither formula.", 145))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    out = OUT / "figures" / "07_error_budget.png"
    fig.savefig(out, dpi=170)
    shutil.copy(out, ROOT / "docs" / "figs" / "binned_error_budget.png")
    plt.close(fig)

    print(f"\n-> {out}")
    print(f"-> {OUT / 'data' / 'error_analysis.json'}")
    print(f"\nh* = {h_star:.4f} ({L / h_star:.1f} bins); measured optimum at "
          f"{min(rows, key=lambda r: r['total'])['bins']} bins")
    for r in rows:
        print(f"  bins={r['bins']:4d}  n={r['n_med']:8.0f}  var={r['var']:.4f} "
              f"(pred {r['var_pred']:.4f}, ratio {r['ratio_med']:.2f})  "
              f"read^2={r['read']:.4f}  total={r['total']:.4f}")
    for d in noise:
        print(f"  sigma={d['sigma']:.3f}  inflation {d['inflation']:.3f} "
              f"(predicted {d['predicted']:.3f})  steps/visit "
              f"{d['steps_per_visit']:.1f}  var/naive {d['var_over_naive']:.2f}"
              f"  var/telescoped {d['var_over_tele']:.2f}")
    print(f"  bin on true x, sigma=0.02: inflation {split['inflation']:.3f}, "
          f"var/telescoped {split['var_over_tele']:.2f}, "
          f"var/naive {split['var_over_naive']:.2f}")
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
