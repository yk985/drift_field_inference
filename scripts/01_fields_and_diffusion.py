"""Generate random flow fields, diffuse walkers through them, and draw it all.

This is the demo for the half of the project that does not depend on any
estimator: field generation, simulation, and every static figure. Run it once
and look at ``outputs/figures`` before writing a line of estimator code -- if
the fields and the trajectories are not right, nothing downstream can be.

    python scripts/01_fields_and_diffusion.py            # figures only
    python scripts/01_fields_and_diffusion.py --animate  # + mp4/gif
    python scripts/01_fields_and_diffusion.py --theme dark
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfi import (random_grid_field, ornstein_uhlenbeck, DoubleWell1D,
                 simulate, sample_stationary, suggest_dt, check_step)
from dfi.viz import (use_style, plot_field_card, plot_helmholtz, plot_drift,
                     plot_potential, plot_density, plot_field_1d,
                     plot_trajectories, plot_snapshot, plot_sampling_density,
                     plot_msd, plot_increment_check, plot_trajectories_1d,
                     animate_diffusion, animate_omega_sweep, animate_relaxation,
                     figure_caption)

FIG = ROOT / "outputs" / "figures"
ANI = ROOT / "outputs" / "animations"
DATA = ROOT / "outputs" / "data"


def save(fig, name, caption=None):
    if caption:
        figure_caption(fig, caption)
    for d in (FIG,):
        d.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animate", action="store_true",
                    help="also render mp4/gif animations (slower)")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gif", action="store_true",
                    help="write .gif instead of .mp4")
    args = ap.parse_args()

    use_style(args.theme)
    ext = "gif" if args.gif else "mp4"
    for d in (FIG, ANI, DATA):
        d.mkdir(parents=True, exist_ok=True)

    # D is chosen so rho_ss covers most of the torus. At small D the walkers
    # collapse into one basin and never see the rest of the field, which makes
    # every later "error over the domain" a statement about extrapolation
    # rather than about the estimator. field.concentration(D) reports this.
    D = 0.30
    t_start = time.time()

    # ------------------------------------------------------------------
    print("\n[1] Random 2D field, equilibrium (omega = 0)")
    field = random_grid_field(d=2, resolution=args.resolution, omega=0.0,
                              seed=args.seed, correlation_length=0.45,
                              drift_scale=1.0)
    print(f"    {field.summary()}")
    diag = check_step(field, suggest_dt(field, D), D)
    conc = field.concentration(D)
    print(f"    gradient scale ell = {diag['drift_length_scale']:.3f}, "
          f"RMS|b| = {diag['rms_drift']:.3f}")
    print(f"    density dynamic range = {conc['density_range']:.0f}  "
          f"(walkers cover the domain when this is small)")
    field.save(DATA / "field_omega0.npz")

    fig, _ = plot_field_card(field, D, n=args.resolution, seed=args.seed)
    save(fig, "01_field_card_equilibrium")

    # ------------------------------------------------------------------
    print("\n[2] The same potential with circulation added (omega = 1.5)")
    field_rot = field.with_omega(1.5)
    fig, _ = plot_helmholtz(field_rot, n=args.resolution, seed=args.seed)
    save(fig, "02_helmholtz_decomposition",
         "b = -grad U + omega A grad U with omega = 1.5.  The two components are "
         "orthogonal pointwise and the rotational one is divergence free, so "
         "rho_ss is unchanged by omega.")

    # the identifiability figure, as a static pair
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.6))
    # One shared speed scale across both drift panels. With independent
    # scales the two panels would look equally fast, hiding the fact that
    # circulation genuinely speeds the walkers up by sqrt(1 + omega^2).
    from dfi.viz import field_on_grid
    vx, vy = field_on_grid(field_rot, args.resolution)
    vmax_shared = float(np.percentile(np.hypot(vx, vy), 99.0))
    for j, (f_, w) in enumerate([(field, 0.0), (field_rot, 1.5)]):
        plot_drift(f_, axes[0, j], n=args.resolution, seed=args.seed,
                   vmax=vmax_shared, colorbar=(j == 1),
                   title=rf"Drift field,  $\omega={w:g}$")
        plot_density(f_, D, axes[1, j], n=args.resolution, colorbar=(j == 1),
                     title=rf"Stationary density,  $\omega={w:g}$")
    save(fig, "03_identifiability_static",
         "Both drift panels share one speed scale; both density panels are "
         "bit-identical arrays. Top row: completely different dynamics. "
         "Bottom row: the same distribution. A snapshot-only estimator sees "
         "only the bottom row, so it cannot tell these two systems apart.")

    # ------------------------------------------------------------------
    print("\n[3] Simulating walkers")
    dt = suggest_dt(field, D)
    print(f"    dt = {dt:.2e}")
    t0 = time.time()
    traj = simulate(field, n_walkers=2000, n_steps=4000, dt=dt, D=D,
                    burn_in=10.0, seed=args.seed, progress=False)
    print(f"    {traj!r} in {time.time()-t0:.1f}s")
    traj.save(DATA / "traj_omega0.npz")

    traj_rot = simulate(field_rot, n_walkers=2000, n_steps=4000, dt=dt, D=D,
                        burn_in=10.0, seed=args.seed, progress=False)
    traj_rot.save(DATA / "traj_omega1.5.npz")

    stats = traj.step_statistics()
    print(f"    rms step {stats['rms_step']:.4f} "
          f"(pure diffusion would be {stats['rms_step_pure_diffusion']:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
    plot_trajectories(traj, field, axes[0], n_show=14, frame_range=(0, 700),
                      title=r"Trajectories,  $\omega=0$")
    plot_trajectories(traj_rot, field_rot, axes[1], n_show=14,
                      frame_range=(0, 700), title=r"Trajectories,  $\omega=1.5$")
    fig.tight_layout()
    save(fig, "04_trajectories_over_field", traj.caption()
         + "   |   path colour ramps with time; field muted to sit behind")

    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.0))
    plot_sampling_density(traj, axes[0])
    plot_msd(traj, axes[1])
    plot_increment_check(traj, axes[2])
    fig.tight_layout()
    save(fig, "05_trajectory_diagnostics", traj.caption())

    # ------------------------------------------------------------------
    print("\n[4] Snapshots (no time ordering)")
    snaps = sample_stationary(field, 40000, D, rng=np.random.default_rng(1))
    np.save(DATA / "snapshots_omega0.npy", snaps)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.8))
    plot_snapshot(snaps, field.box, axes[0],
                  title="Snapshot dataset (40k independent samples)")
    plot_density(field, D, axes[1], n=args.resolution,
                 title=r"Exact $\rho_{\rm ss}$")
    fig.tight_layout()
    save(fig, "06_snapshots",
         "This dataset is identical in distribution for every omega. It "
         "contains no information about circulation.")

    # ------------------------------------------------------------------
    print("\n[5] Measurement noise")
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 4.4))
    for ax, sig in zip(axes, (0.0, 0.004, 0.02)):
        t_ = traj.with_observation_noise(sig, np.random.default_rng(0)) if sig else traj
        plot_increment_check(t_, ax, title=rf"$\sigma_{{\rm obs}}={sig:g}$")
    fig.tight_layout()
    save(fig, "07_measurement_noise",
         "Observation noise inflates the increment variance by 2 sigma^2, which "
         "does not vanish as dt -> 0. At fine sampling it swamps the drift.")

    # ------------------------------------------------------------------
    print("\n[6] Closed-form 1D systems")
    ou = ornstein_uhlenbeck(d=1, k=2.0)
    fig, _ = plot_field_1d(ou, D=0.4)
    save(fig, "08_ou_1d", "Ornstein-Uhlenbeck: everything is closed-form. "
                          "Validate every estimator here first.")

    dw = DoubleWell1D(a=1.0, b=1.0)
    fig, _ = plot_field_1d(dw, D=0.08)
    save(fig, "09_double_well_1d",
         f"Double well, D=0.08: barrier {dw.barrier_height:.2f}, "
         f"Kramers rate {dw.kramers_rate(0.08):.2e} per unit time.")

    dw_traj = simulate(dw, n_walkers=200, n_steps=6000, dt=2e-3, D=0.08,
                       burn_in=5.0, boundary="none", seed=3)
    fig, _ = plot_trajectories_1d(dw_traj, dw, n_show=8, D=0.08)
    save(fig, "10_double_well_trajectories",
         "Metastability: long dwells, rare hops. If a run shows no hops, no "
         "estimator can recover the barrier region.")

    # ------------------------------------------------------------------
    if args.animate:
        print("\n[7] Animations")
        t0 = time.time()
        animate_diffusion(traj_rot, field_rot, n_show=220, trail=26, stride=3,
                          frame_range=(0, 1200), panels=("density",), fps=30,
                          out=ANI / f"diffusion_omega1.5.{ext}", dpi=130,
                          seed=args.seed)
        print(f"    diffusion animation in {time.time()-t0:.0f}s")

        t0 = time.time()
        animate_omega_sweep(field, np.concatenate([np.linspace(0, 2.5, 50),
                                                   np.linspace(2.5, 0, 25)]),
                            D, n_grid=200, fps=20,
                            out=ANI / f"omega_sweep.{ext}", dpi=130,
                            seed=args.seed)
        print(f"    omega sweep in {time.time()-t0:.0f}s")

        t0 = time.time()
        # Start every walker at one point so there is something to relax
        # from; this is the run where the 'relax' panel is meaningful.
        relax = simulate(field, n_walkers=6000, n_steps=900, dt=dt, D=D,
                         burn_in=0.0, x0=np.array([0.0, 0.0]), seed=5)
        animate_relaxation(relax, field, D, stride=4, fps=30,
                           out=ANI / f"relaxation.{ext}", dpi=130)
        animate_diffusion(relax, field, n_show=200, trail=22, stride=4,
                          panels=("relax",), fps=30,
                          out=ANI / f"spreading_cloud.{ext}", dpi=130,
                          seed=args.seed)
        print(f"    relaxation animation in {time.time()-t0:.0f}s")

    print(f"\nDone in {time.time()-t_start:.0f}s. "
          f"Figures in {FIG.relative_to(ROOT)}"
          + (f", animations in {ANI.relative_to(ROOT)}" if args.animate else ""))


if __name__ == "__main__":
    main()
