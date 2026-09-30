"""Study the trained amortised CNN from its saved weights.

    python scripts/07_amortised_study.py                   # all stages
    python scripts/07_amortised_study.py --which kernel    # one stage

Stages, each writing to ``outputs/cnn_amortised/figures/``:

``training``  loss and validation error through training (from the history file)
``test``      100 held-out simulated fields, never seen in training, at three
              budgets: the amortised CNN against SFI, the local-linear kernel and
              (on a subset) the per-dataset MLP, all fitted to the same walkers
``ood``       fields deliberately outside the training distribution: shorter and
              longer correlation, stronger rotation, a stronger drift, noise
              levels beyond the trained range
``generators`` fields built by other generators than training's Gaussian-spectrum
              fields: Perlin noise, power-law and band-pass spectra, sparse wells,
              a single lattice mode, terraces with sharp fronts
``kernel``    what the network learned to do: its equivalent smoothing kernel,
              ``d b_hat(c0) / d mean(c)``, measured on the study's own walkers as
              the budget and the noise change, beside the bandwidth
              cross-validation picks for the local-linear kernel

The figures in ``01_setup`` ... ``06_sweeps`` come from the shared study script
(``03_local_estimator.py --estimators amort``), on the same walkers as rungs 1-4.
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts"))

from dfi.estimators.amortised import (AmortizedDrift, make_features,
                                      splat_groups)
from dfi.estimators.basis import BasisProjection
from dfi.estimators.local import KernelRegression
from dfi.estimators.neural import NeuralDrift
from dfi.metrics import drift_error, make_eval_set
from dfi.sweeps import (SweepConfig, _burn_in_time, build_field,
                        default_traj_cache, simulate_config)
from dfi.simulate import simulate, suggest_dt
from dfi.trajectories import Trajectories
from dfi.viz import figure_caption, use_style
from dfi.viz import style as st

DATA_MODULE = __import__("05_amortised_data")
OUT = ROOT / "outputs" / "cnn_amortised"
FIG = OUT / "figures"
DATA = OUT / "data"
BUDGETS = (1, 4, 16)
METHODS = {"amortised": ("Amortised CNN", 4), "sfi": ("SFI", 2),
           "ll": ("LL kernel", 1), "mlp": ("MLP (per dataset)", 3)}


def save(fig, name, caption):
    width = max(60, int(fig.get_size_inches()[0] * 17))
    figure_caption(fig, "\n".join(textwrap.wrap(caption, width)))
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def color(key):
    return st.series_style(METHODS[key][1])["color"]


def subset(traj, sel):
    return Trajectories(x=traj.x[sel], dt=traj.dt, D=traj.D, box=traj.box)


def fit_score(key, traj, field, ev, amort):
    t0 = time.time()
    if key == "amortised":
        est = amort.fit(traj)
    elif key == "sfi":
        est = BasisProjection().fit(traj)
    elif key == "ll":
        est = KernelRegression().fit(traj)
    else:
        est = NeuralDrift().fit(traj)
    return drift_error(est, field, ev)["nrmse"], time.time() - t0


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def stage_training(args):
    print("\n[training]")
    H = json.loads((DATA / "train_history.json").read_text())
    h = H["history"]
    steps = np.array(h["step"])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8))
    axes[0].plot(steps, h["train_loss"], lw=2, color=st.series_style(0)["color"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimisation step")
    axes[0].set_ylabel("training loss (weighted relative error²)")
    axes[0].set_title("Training loss, mean over the last interval")
    axes[0].grid(True, which="both", alpha=0.3)
    notes = []
    for i, k in enumerate(BUDGETS):
        net = [v[str(k)][0] for v in h["val"]]
        raw = h["val"][0][str(k)][1]
        c = st.series_style(i + 1)["color"]
        axes[1].plot(steps, net, lw=2, color=c, label=f"{k}/16 of each field's walkers")
        axes[1].axhline(raw, color=c, lw=1, dashes=(2, 2))
        notes.append((k, min(net), raw))
    axes[1].axvline(H["best_step"], color=st.active().ink_muted, lw=1, dashes=(1, 2))
    axes[1].set_yscale("log")
    axes[1].set_xlabel("optimisation step")
    axes[1].set_ylabel("validation nrmse under rho_ss")
    axes[1].set_title("200 held-out fields (dashed: the network's shrunk-mean input alone)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    txt = "; ".join(f"{k}/16 of the walkers: {b:.3f} against {r:.2f} for the input alone"
                    for k, b, r in notes)
    print("  " + txt)
    save(fig, "10_training",
         f"Amortised CNN, {H['n_params']:,} parameters, trained on {H['train_fields']} "
         f"simulated fields. Each training sample is a random field at a random budget "
         f"(a random subset of its walker groups) in a random lattice orientation. The "
         f"checkpoint kept is the weight average with the lowest mean validation nrmse, "
         f"at step {H['best_step']} (dotted). Best validation nrmse: {txt}. The dashed "
         f"lines are the per-node shrunk mean the network receives as input -- the gap "
         f"is what the convolutions add.")


# --------------------------------------------------------------------------
# held-out test fields
# --------------------------------------------------------------------------

def stage_test(args):
    print("\n[test] held-out simulated fields")
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / "test_fields.csv"
    rows = pd.read_csv(cache).to_dict("records") if cache.exists() else []
    done = {(r["field"], r["k"], r["method"]) for r in rows}
    amort = AmortizedDrift(checkpoint=args.checkpoint)
    n_fields = 10 if args.quick else args.test_fields
    for i in range(n_fields):
        p = DATA_MODULE.field_params("test", i)
        want = [(k, m) for k in BUDGETS for m in ("amortised", "sfi", "ll")
                if (i, k, m) not in done]
        if i < args.mlp_fields:
            want += [(k, "mlp") for k in (1, 16) if (i, k, "mlp") not in done]
        if not want:
            continue
        field, traj, dt = DATA_MODULE.simulate_field(p)
        groups = (np.random.default_rng(p["seed"] + 1).permutation(traj.n_walkers)
                  % DATA_MODULE.N_GROUPS)
        ev = make_eval_set(field, p["D"], n=8000, seed=p["seed"] + 7)
        for k, m in want:
            sub = subset(traj, groups < k)
            err, secs = fit_score(m, sub, field, ev, amort)
            rows.append(dict(field=i, k=k, method=m, nrmse=err, seconds=secs,
                             n_transitions=sub.n_transitions, **{f"p_{a}": b for a, b in p.items()}))
        pd.DataFrame(rows).to_csv(cache, index=False)
        print(f"  field {i:3d}  D={p['D']:.2f} ell={p['correlation_length']:.2f} "
              f"omega={p['omega']:+.2f}  " + "  ".join(
                  f"{m}@{k}: {r['nrmse']:.3f}" for r in rows[-len(want):]
                  for m, k in [(r['method'], r['k'])]), flush=True)

    df = pd.DataFrame(rows)
    df = df[df.field < n_fields]
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.2))
    th = st.active()
    summary = []
    # The MLP ran on a subset of fields only; a median over other fields is not
    # comparable, so it is compared pairwise in the caption instead.
    for m in ("amortised", "sfi", "ll"):
        g = df[df.method == m].groupby("k")
        if not len(g):
            continue
        med = g.nrmse.median()
        q1, q3 = g.nrmse.quantile(0.25), g.nrmse.quantile(0.75)
        x = g.n_transitions.median()
        axes[0].errorbar(x, med, yerr=[med - q1, q3 - med], lw=2, capsize=3,
                         marker="o", color=color(m), label=METHODS[m][0])
        summary.append((m, med.to_dict()))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("observed transitions (median over fields)")
    axes[0].set_ylabel("nrmse (median, interquartile bar)")
    axes[0].set_title(f"{n_fields} unseen fields, same walkers for every method")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    piv = df.pivot_table(index=["field", "k"], columns="method", values="nrmse")
    meta = df.drop_duplicates(["field", "k"]).set_index(["field", "k"])
    best_classical = piv[["sfi", "ll"]].min(axis=1)
    ratio = piv["amortised"] / best_classical
    for j, k in enumerate(BUDGETS):
        sel = ratio.xs(k, level="k")
        Ds = meta.xs(k, level="k").loc[sel.index, "p_D"]
        axes[1].scatter(Ds, sel, s=16, color=st.series_style(j)["color"],
                        label=f"{k}/16 of the walkers", alpha=0.8)
    axes[1].axhline(1.0, color=th.ink_muted, lw=1, dashes=(3, 2))
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("noise level D of the field")
    axes[1].set_ylabel("amortised / best of SFI and LL")
    axes[1].set_title("Per field: below 1, the pretrained network wins")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    for j, k in enumerate(BUDGETS):
        sel = ratio.xs(k, level="k")
        ell = meta.xs(k, level="k").loc[sel.index, "p_correlation_length"]
        axes[2].scatter(ell, sel, s=16, color=st.series_style(j)["color"], alpha=0.8)
    axes[2].axhline(1.0, color=th.ink_muted, lw=1, dashes=(3, 2))
    axes[2].set_yscale("log")
    axes[2].set_xlabel("correlation length of the field")
    axes[2].set_ylabel("amortised / best of SFI and LL")
    axes[2].set_title("Against field roughness")
    axes[2].grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    wins = {k: float((ratio.xs(k, level="k") < 1).mean()) for k in BUDGETS}
    med_ratio = {k: float(ratio.xs(k, level="k").median()) for k in BUDGETS}
    for m, d in summary:
        print(f"  {m:<10} median nrmse " + "  ".join(f"k={k}: {v:.3f}" for k, v in d.items()))
    print("  amortised beats the better classical fit on " + ", ".join(
        f"{wins[k]:.0%} of fields at k={k} (median ratio {med_ratio[k]:.2f})" for k in BUDGETS))
    secs = df.groupby("method").seconds.median()
    sub = df[df.field.isin(df[df.method == "mlp"].field.unique())]
    paired = sub.pivot_table(index="k", columns="method", values="nrmse", aggfunc="median")
    mlp_note = ""
    if "mlp" in paired:
        mlp_note = (f" On the {sub.field.nunique()} fields where the per-dataset MLP also "
                    f"ran, median nrmse at 1/16 and 16/16 of the walkers: MLP "
                    f"{paired.loc[1, 'mlp']:.3f} and {paired.loc[16, 'mlp']:.3f}, "
                    f"amortised {paired.loc[1, 'amortised']:.3f} and "
                    f"{paired.loc[16, 'amortised']:.3f}.")
        print("  paired with MLP:")
        print(paired.round(3).to_string())
    save(fig, "11_test_fields",
         f"Held-out test fields (seeds never used in training), each simulated with "
         f"the same parameter ranges as training, at 1, 4 and 16 of its 16 walker "
         f"groups. Every method is fitted to the same walkers and scored on the same "
         f"points under rho_ss. The amortised CNN beats the better of SFI and the "
         f"local-linear kernel on " + ", ".join(f"{wins[k]:.0%} of fields at {k}/16"
                                                  for k in BUDGETS)
         + f" (median ratio " + ", ".join(f"{med_ratio[k]:.2f}" for k in BUDGETS)
         + ")." + mlp_note + " Median time per dataset: " + ", ".join(
             f"{METHODS[m][0]} {v:.1f}s" for m, v in secs.items()) + ".")


# --------------------------------------------------------------------------
# out of distribution
# --------------------------------------------------------------------------

OOD = [
    ("in range", dict()),
    ("correlation 0.2", dict(correlation_length=0.2)),
    ("correlation 1.0", dict(correlation_length=1.0)),
    ("rotation ω = 3", dict(omega=3.0)),
    ("drift × 2", dict(drift_scale=2.0)),
    ("D = 0.02", dict(D=0.02)),
    ("D = 3", dict(D=3.0)),
]


def stage_ood(args):
    print("\n[ood] outside the training distribution")
    cache = DATA / "ood.csv"
    rows = pd.read_csv(cache).to_dict("records") if cache.exists() else []
    done = {(r["variant"], r["rep"], r["method"]) for r in rows}
    amort = AmortizedDrift(checkpoint=args.checkpoint)
    reps = 3 if args.quick else args.ood_reps
    for v, (name, change) in enumerate(OOD):
        for rep in range(reps):
            want = [m for m in ("amortised", "sfi", "ll") if (name, rep, m) not in done]
            if not want:
                continue
            base = dict(d=2, n_walkers=512, n_steps=2000, D=0.3, omega=0.0,
                        correlation_length=0.45, seed=400_000 + 100 * v + rep)
            base.update(change)
            cfg = SweepConfig(**base)
            field = build_field(cfg)
            dt = suggest_dt(field, cfg.D)
            traj = simulate(field, n_walkers=cfg.n_walkers, n_steps=cfg.n_steps,
                            dt=dt, D=cfg.D, burn_in=_burn_in_time(field, cfg),
                            seed=cfg.seed, check=False)
            ev = make_eval_set(field, cfg.D, n=8000, seed=cfg.seed + 7)
            for m in want:
                err, secs = fit_score(m, traj, field, ev, amort)
                rows.append(dict(variant=name, rep=rep, method=m, nrmse=err))
            pd.DataFrame(rows).to_csv(cache, index=False)
            print(f"  {name:<16} rep {rep}  " + "  ".join(
                f"{r['method']}: {r['nrmse']:.3f}" for r in rows[-len(want):]), flush=True)

    df = pd.DataFrame(rows)
    df = df[df.rep < reps]
    names = [n for n, _ in OOD]
    fig, ax = plt.subplots(figsize=(13.4, 5.0))
    for j, m in enumerate(("amortised", "sfi", "ll")):
        for i, n in enumerate(names):
            vals = df[(df.variant == n) & (df.method == m)].nrmse.values
            x = i + (j - 1) * 0.22
            ax.scatter(np.full(len(vals), x), vals, s=14, color=color(m), alpha=0.6)
            ax.plot([x - 0.08, x + 0.08], [np.median(vals)] * 2, lw=3, color=color(m),
                    label=METHODS[m][0] if i == 0 else None)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylabel("nrmse")
    ax.set_title("Fields the network was not trained on (dots: fields; bars: median)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    med = df.groupby(["variant", "method"]).nrmse.median().unstack()
    med["ratio"] = med["amortised"] / med[["sfi", "ll"]].min(axis=1)
    print(med.loc[names].round(3).to_string())
    save(fig, "12_out_of_distribution",
         f"512 walkers x 2000 steps per field, {reps} fields per column. Training "
         f"covered correlation lengths 0.3-0.7, |omega| <= 1.5, unit drift scale and "
         f"D in [0.05, 1.5]. Median amortised / best classical: " + ", ".join(
             f"{n} {med.loc[n, 'ratio']:.2f}" for n in names)
         + ". SFI and the kernel have no training distribution to leave; the network does.")


# --------------------------------------------------------------------------
# what it learned: the equivalent kernel
# --------------------------------------------------------------------------

def equivalent_kernel(amort, traj):
    """``K_c = d b_hat_x(c0) / d mean_x(c)`` at the densest node ``c0``.

    The network's output is a nonlinear function of the per-node sums ``Y``; its
    gradient, rescaled by ``N_c``, is the weight a linear smoother would give to
    node ``c``'s mean ``Y_c / N_c``. Summing it gives the smoother's gain (1:
    unbiased locally; below 1: shrinkage).
    """
    import torch
    if amort.net is None:
        amort._load()
    dev = amort._dev
    M = amort._meta["grid"]
    N, Y, S, n = splat_groups(traj, M)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    Nt, Yt = t(N), t(Y).requires_grad_(True)
    c0 = np.unravel_index(int(np.argmax(N[0])), (M, M))
    feats = make_features(Nt, Yt, t(S), t(n), torch.tensor([traj.dt], device=dev))
    out = amort.net(feats)[0, 0, c0[0], c0[1]]
    (g,) = torch.autograd.grad(out, Yt)
    K = (g[0, 0] * Nt[0]).double().cpu().numpy()
    K = np.roll(K, (M // 2 - c0[0], M // 2 - c0[1]), axis=(0, 1))
    h = float(traj.box.length[0]) / M
    ii, jj = np.meshgrid(np.arange(M) - M // 2, np.arange(M) - M // 2, indexing="ij")
    r2 = (ii ** 2 + jj ** 2) * h * h
    A = np.abs(K)
    width = float(np.sqrt((A * r2).sum() / A.sum() / 2.0))   # Gaussian-equivalent h
    return K, width, float(K.sum())


def stage_kernel(args):
    print("\n[kernel] the smoothing the network learned")
    amort = AmortizedDrift(checkpoint=args.checkpoint)
    cases = ([("N", N, 0.3) for N in (16, 64, 256, 1024)]
             + [("D", 400, D) for D in (0.05, 0.15, 0.4, 1.0)])
    res = []
    for kind, N, D in cases:
        cfg = SweepConfig(d=2, n_walkers=N, n_steps=12000, D=D, seed=0)
        field, traj, dt = simulate_config(cfg, traj_cache=args.traj_cache)
        K, width, gain = equivalent_kernel(amort, traj)
        ll = KernelRegression().fit(traj)
        res.append(dict(kind=kind, N=N, D=D, K=K, width=width, gain=gain, h=ll.h_,
                        nt=traj.n_transitions))
        print(f"  N={N:<5d} D={D:<5g}  network width {width:.3f}  gain {gain:.2f}  "
              f"LL CV bandwidth {ll.h_:.3f}", flush=True)

    fig = plt.figure(figsize=(18.0, 8.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.5])
    M = res[0]["K"].shape[0]
    win = 14
    sl = slice(M // 2 - win, M // 2 + win + 1)
    h = 2.0 / M
    ext = [-win * h, win * h, -win * h, win * h]
    for row, kind in enumerate(("N", "D")):
        rr = [r for r in res if r["kind"] == kind]
        for j, r in enumerate(rr):
            ax = fig.add_subplot(gs[row, j])
            Kz = r["K"][sl, sl]
            vmax = float(np.abs(Kz).max())
            ax.imshow(Kz.T, origin="lower", extent=ext, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax)
            circ = plt.Circle((0, 0), 2 * r["h"], fill=False, lw=1,
                              color=st.active().ink_secondary, ls="--")
            ax.add_patch(circ)
            ax.set_title((f"N = {r['N']}" if kind == "N" else f"D = {r['D']:g}")
                         + f"\nwidth {r['width']:.3f}, gain {r['gain']:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(gs[row, 4])
        x = [r["nt"] if kind == "N" else r["D"] for r in rr]
        ax.plot(x, [r["width"] for r in rr], lw=2, marker="o",
                color=color("amortised"), label="network: Gaussian-equivalent width")
        ax.plot(x, [r["h"] for r in rr], lw=2, marker="s", color=color("ll"),
                label="LL kernel: bandwidth chosen by CV")
        ax.set_xscale("log")
        ax.set_xlabel("observed transitions" if kind == "N" else "noise level D")
        ax.set_ylabel("smoothing scale")
        ax.set_title("Smoothing against " + ("budget" if kind == "N" else "noise"))
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    txt = "; ".join(f"{'N=' + str(r['N']) if r['kind'] == 'N' else 'D=' + format(r['D'], 'g')}: "
                    f"{r['width']:.3f} vs {r['h']:.3f}" for r in res)
    save(fig, "13_learned_smoothing",
         f"The network's equivalent kernel on the study's own walkers (field seed 0, "
         f"never trained on): the derivative of its output at the densest node with "
         f"respect to each node's mean increment, times that node's count -- the "
         f"weight a linear smoother would give it. Top: 12000 steps, walkers 16 to "
         f"1024, D = 0.3. Bottom: 400 walkers, D from 0.05 to 1.0. Dashed circles: "
         f"twice the bandwidth the local-linear kernel's cross-validation picks on the "
         f"same data. Gain is the kernel's sum (1 = no shrinkage). Width (network) vs "
         f"bandwidth (LL): {txt}. The network adapts differently from a kernel. Against "
         f"noise it widens, as a kernel would, but more steeply. Against budget it "
         f"hardly changes width at all; what changes is the gain, i.e. how far it "
         f"shrinks towards zero. Its kernels are also elongated and carry negative "
         f"side lobes: the x-drift of a gradient field is correlated further along y "
         f"than along x, and the network learned that covariance from the training "
         f"fields -- closer to a Wiener filter matched to the field statistics than to "
         f"a bandwidth.")


# --------------------------------------------------------------------------
# other generators: fields built nothing like the training fields
# --------------------------------------------------------------------------

def _grid_field(U, box, name):
    """A GridDriftField from any potential, scaled to RMS|b| = 1 like training."""
    from dfi.fields.noise import lowpass, spectral_gradient
    from dfi.fields.random_field import GridDriftField
    U = lowpass(U - U.mean(), box)          # same band limit as the generators
    g = spectral_gradient(U, box)
    U = U / float(np.sqrt(np.mean(np.sum(g ** 2, axis=0))))
    return GridDriftField(U, box, name=name)


def _generator_fields():
    """name -> (description, builder(seed) -> field). All gradient fields, unit
    RMS drift, on the training torus, so only the *shape* of the field differs
    from training."""
    from dfi.fields.domain import Box
    from dfi.fields.random_field import random_grid_field
    box = Box.cube(2, 1.0, periodic=True)
    res = 128
    X, Yg = np.meshgrid(*box.axes(res), indexing="ij")

    def perlin(seed):
        return random_grid_field(d=2, resolution=res, generator="perlin", seed=seed)

    def powerlaw(seed):
        return random_grid_field(d=2, resolution=res, spectrum="powerlaw",
                                 correlation_length=0.45, slope=4.0, seed=seed)

    def bandpass(seed):
        return random_grid_field(d=2, resolution=res, spectrum="bandpass",
                                 k_ring=2 * np.pi / 0.5, seed=seed)

    def wells(seed):
        rng = np.random.default_rng(seed)
        U = np.zeros_like(X)
        for _ in range(6):
            c = rng.uniform(-1, 1, 2)
            dx = X - c[0]
            dx -= 2.0 * np.round(dx / 2.0)
            dy = Yg - c[1]
            dy -= 2.0 * np.round(dy / 2.0)
            U -= rng.uniform(0.5, 1.5) * np.exp(-(dx ** 2 + dy ** 2) / (2 * 0.1 ** 2))
        return _grid_field(U, box, "sparse wells")

    def eggcarton(seed):
        rng = np.random.default_rng(seed)
        n1, n2 = rng.integers(1, 3, 2)
        U = (np.cos(np.pi * n1 * X + rng.uniform(0, 2 * np.pi))
             * np.cos(np.pi * n2 * Yg + rng.uniform(0, 2 * np.pi)))
        return _grid_field(U, box, "egg carton")

    def terraces(seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(1, 3, 2)
        phase = rng.uniform(0, 2 * np.pi)
        ramp = np.sin(np.pi * (n[0] * X + n[1] * Yg) + phase)
        U = np.tanh(4.0 * ramp) + 0.3 * np.cos(np.pi * (X - Yg) * rng.integers(1, 3))
        return _grid_field(U, box, "terraces")

    return {
        "Perlin fBm": ("fractal Perlin noise, 4 octaves", perlin),
        "power-law GRF": ("multi-scale spectrum k^-4, fine structure", powerlaw),
        "band-pass GRF": ("energy on a ring |k| = 2π/0.5: cellular", bandpass),
        "sparse wells": ("6 narrow Gaussian wells on a flat plain", wells),
        "egg carton": ("a single lattice mode", eggcarton),
        "terraces": ("tanh steps: sharp fronts between flat terraces", terraces),
    }


def stage_generators(args):
    print("\n[generators] fields from other generators")
    cache = DATA / "generators.csv"
    rows = pd.read_csv(cache).to_dict("records") if cache.exists() else []
    done = {(r["generator"], r["rep"], r["method"]) for r in rows}
    amort = AmortizedDrift(checkpoint=args.checkpoint)
    gens = _generator_fields()
    reps = 3 if args.quick else args.ood_reps
    D, N, T = 0.3, 512, 2000
    examples = {}
    for g, (name, (desc, build)) in enumerate(gens.items()):
        for rep in range(reps):
            seed = 500_000 + 100 * g + rep
            field = build(seed)
            if rep == 0:
                examples[name] = field
            want = [m for m in ("amortised", "sfi", "ll") if (name, rep, m) not in done]
            if not want:
                continue
            cfg = SweepConfig(d=2, n_walkers=N, n_steps=T, D=D, seed=seed)
            # Rough fields need a far smaller stable integration step (down to
            # 3e-5) than training fields (~1e-3). Integrate finely but observe
            # every ~1e-3, as training did -- otherwise the walkers would barely
            # move in T frames and coverage, not structure, would be tested.
            dt_int = suggest_dt(field, D)
            sub = max(1, int(round(1e-3 / dt_int)))
            traj = simulate(field, n_walkers=N, n_steps=T, dt=dt_int * sub,
                            substeps=sub, D=D, burn_in=_burn_in_time(field, cfg),
                            seed=seed, check=False)
            ev = make_eval_set(field, D, n=8000, seed=seed + 7)
            for m in want:
                err, _ = fit_score(m, traj, field, ev, amort)
                rows.append(dict(generator=name, rep=rep, method=m, nrmse=err))
            pd.DataFrame(rows).to_csv(cache, index=False)
            print(f"  {name:<14} rep {rep}  " + "  ".join(
                f"{r['method']}: {r['nrmse']:.3f}" for r in rows[-len(want):]), flush=True)

    df = pd.DataFrame(rows)
    df = df[df.rep < reps]
    names = list(gens)
    th = st.active()
    fig = plt.figure(figsize=(18.0, 8.4))
    gs = fig.add_gridspec(2, len(names), height_ratios=[1, 1.35])
    for i, n in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        f = examples[n]
        mag = np.hypot(*f.drift_on_grid()[:2])
        ax.imshow(f.U_grid.T, origin="lower", cmap="viridis", extent=[-1, 1, -1, 1])
        ax.set_title(f"{n}\n{gens[n][0]}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    for j, m in enumerate(("amortised", "sfi", "ll")):
        for i, n in enumerate(names):
            vals = df[(df.generator == n) & (df.method == m)].nrmse.values
            x = i + (j - 1) * 0.22
            ax.scatter(np.full(len(vals), x), vals, s=14, color=color(m), alpha=0.6)
            ax.plot([x - 0.08, x + 0.08], [np.median(vals)] * 2, lw=3, color=color(m),
                    label=METHODS[m][0] if i == 0 else None)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylabel("nrmse")
    ax.set_title("Fields from generators the network never saw "
                 "(dots: fields; bars: median)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    med = df.groupby(["generator", "method"]).nrmse.median().unstack()
    med["ratio"] = med["amortised"] / med[["sfi", "ll"]].min(axis=1)
    print(med.loc[names].round(3).to_string())
    save(fig, "14_other_generators",
         f"Top: the potential U of one field per generator. Every field is a gradient "
         f"field scaled to RMS|b| = 1, the training amplitude, so only its structure "
         f"differs from training (which saw only Gaussian-spectrum random fields). "
         f"{N} walkers x {T} steps, D = {D}, {reps} fields per generator. Median "
         f"amortised / best of SFI and LL: " + ", ".join(
             f"{n} {med.loc[n, 'ratio']:.2f}" for n in names) + ".")


STAGES = {"training": stage_training, "test": stage_test, "ood": stage_ood,
          "kernel": stage_kernel, "generators": stage_generators}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--test-fields", type=int, default=100)
    ap.add_argument("--mlp-fields", type=int, default=15)
    ap.add_argument("--ood-reps", type=int, default=8)
    ap.add_argument("--traj-cache", type=Path, default=default_traj_cache())
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="weights to study (default: models/amortised_cnn_2d.pt)")
    args = ap.parse_args()
    use_style("light")
    t0 = time.time()
    for s in args.which:
        STAGES[s](args)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
