"""Animations: diffusion in a field, and the field itself changing.

Three of them, each making a point that a static figure cannot:

:func:`animate_diffusion`
    Walkers moving through the field, with fading trails and glowing heads.
    Optional side panels show the cloud relaxing onto ``rho_ss`` in real time.
    This is the one that shows *why* the inverse problem is hard: the drift is
    a small systematic bias on top of a much larger random step, and you can
    see the randomness dominating.

:func:`animate_omega_sweep`
    The rotational strength rising from zero while the stationary density sits
    there unchanged. The identifiability gap as a moving picture.

:func:`animate_relaxation`
    A point source spreading into the stationary density, beside the exact
    answer. The clean way to show the simulation is correct.

Writing files: ``.mp4`` needs ffmpeg on PATH, ``.gif`` uses Pillow and needs
nothing. Pick by extension.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from . import style as st
from .fields import plot_drift, to_image, add_colorbar
from .trajectories import trail_segments

__all__ = ["animate_diffusion", "animate_omega_sweep", "animate_relaxation",
           "save_animation"]


def save_animation(anim, path, *, fps: int = 30, dpi: int = 140,
                   progress: bool = True):
    """Write an animation to ``.mp4`` (ffmpeg) or ``.gif`` (Pillow)."""
    path = str(path)
    cb = None
    if progress:
        def cb(i, n):
            if n and (i % max(1, n // 20) == 0 or i == n - 1):
                print(f"\r  writing {path}: {100*(i+1)//max(n,1):3d}%", end="")
    if path.lower().endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=-1,
                              extra_args=["-pix_fmt", "yuv420p",
                                          "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
    anim.save(path, writer=writer, dpi=dpi, progress_callback=cb)
    if progress:
        print()
    return path


def _fading_colors(n_walkers, trail_len, base_colors, min_alpha=0.0,
                   max_alpha=0.9):
    """Per-segment RGBA with alpha ramping up towards the head of each trail."""
    a = np.linspace(min_alpha, max_alpha, trail_len)
    cols = np.repeat(base_colors, trail_len, axis=0)
    alpha = np.tile(a, n_walkers)
    return np.concatenate([cols, alpha[:, None]], axis=1)


def _glow(ax, color, size, layers=(("", 1.0, 1.0),)):
    """Stacked scatter layers making a soft halo around each walker."""
    out = []
    for _, smul, amul in layers:
        out.append(ax.scatter([], [], s=size * smul, c=[color], alpha=amul,
                              linewidths=0, zorder=5))
    return out


def animate_diffusion(traj, field=None, *, n_show: int = 250, trail: int = 28,
                      stride: int = 1, frame_range=None, fps: int = 30,
                      panels=("density",), background: str = "lic",
                      out=None, dpi: int = 140, seed: int = 0,
                      n_grid: int = 220, figsize=None, glow: bool = True,
                      color_by: str = "single", title: str | None = None,
                      D: float | None = None, progress: bool = True,
                      mute_background: float = 0.35, relax_bins: int = 24):
    """Animate walkers diffusing through a drift field.

    Parameters
    ----------
    traj:
        The recorded paths.
    field:
        Drawn as the background. Optional but strongly recommended -- walkers
        on a blank canvas say nothing about the field being inferred.
    n_show, trail:
        How many walkers, and how many past frames of tail each one drags. The
        tail is what turns a cloud of jittering dots into visible flow.
    stride:
        Show every ``stride``-th recorded frame. Use it to cover a long run in
        a short clip.
    panels:
        Extra panels beside the main one. ``'density'`` (default) shows the
        live 2D histogram of walkers; ``'msd'`` grows the MSD curve against the
        free-diffusion line; ``'relax'`` plots the distance from the walker
        distribution to ``rho_ss``. Pass ``()`` for the field alone.

        Note that ``'relax'`` is only informative for a run that starts *out*
        of equilibrium (``x0`` a single point). On a burned-in run the curve
        sits flat on its own finite-sample floor -- which the panel now draws
        explicitly, because a flat line at 0.4 otherwise looks like a failure
        to converge when it is simply the noise in comparing 2000 samples
        against a continuous density.
    color_by:
        ``'single'`` (default) paints every walker one contrasting hue, which
        is right when walker identity carries no meaning. ``'walker'`` assigns
        categorical slots and is capped at the palette size -- past eight,
        recycled hues are decoration, not information.
    out:
        Path to write (``.mp4`` or ``.gif``). Returns the animation either way.
    """
    box = traj.box
    if traj.d < 2:
        raise ValueError("animate_diffusion needs d >= 2; use "
                         "plot_trajectories_1d for 1D runs")
    panels = tuple(panels or ())
    D = traj.D if D is None else D
    th = st.active()

    lo, hi = frame_range or (0, traj.n_frames)
    frames = list(range(lo, hi, stride))
    x = traj.x[:n_show, :, :2]
    n_w = x.shape[0]

    ncols = 1 + len(panels)
    if figsize is None:
        figsize = (6.0 * ncols + 0.6, 6.0)
    fig, axes = plt.subplots(1, ncols, figsize=figsize,
                             gridspec_kw={"width_ratios": [1.25] + [1] * len(panels)})
    axes = np.atleast_1d(axes)
    ax_main = axes[0]

    # -- static background --------------------------------------------------
    if field is not None and background == "lic":
        plot_drift(field, ax_main, n=n_grid, style="lic", seed=seed,
                   colorbar=False, alpha_texture=0.32, title="")
    elif field is not None and background == "density":
        from .fields import plot_density
        plot_density(field, D, ax_main, n=n_grid, colorbar=False, title="")
    elif field is not None and background == "quiver":
        # Arrows rather than texture. Worth having for the binned estimator in
        # particular: it resolves the field on a grid of cells, so a gridded
        # background shows what there is to recover at the scale it can see.
        plot_drift(field, ax_main, n=n_grid, style="quiver", colorbar=False,
                   title="")
    if field is not None and background is not None and mute_background > 0:
        ax_main.add_patch(plt.Rectangle(
            (box.lo[0], box.lo[1]), box.length[0], box.length[1],
            facecolor=th.surface, alpha=mute_background, edgecolor="none",
            zorder=1))
    ax_main.set_xlim(box.lo[0], box.hi[0])
    ax_main.set_ylim(box.lo[1], box.hi[1])
    ax_main.set_aspect("equal")
    ax_main.grid(False)
    ax_main.set_xlabel("$x$")
    ax_main.set_ylabel("$y$")
    ax_main.set_title(title if title is not None
                      else f"{n_w} walkers  |  D = {D:g}")

    # -- trails -------------------------------------------------------------
    if color_by == "walker":
        if n_w > len(th.series):
            raise ValueError(
                f"color_by='walker' with {n_w} walkers exceeds the "
                f"{len(th.series)}-slot palette; use color_by='single'")
        base = np.array([to_rgb(th.series[i]) for i in range(n_w)])
    elif color_by == "single":
        # Orange on the blue field: one contrasting hue, so the walkers read
        # as a layer above the field rather than as part of it.
        base = np.tile(np.array(to_rgb(th.series[1])), (n_w, 1))
    else:
        raise ValueError(f"unknown color_by {color_by!r}")
    lc = LineCollection([], linewidths=1.1, capstyle="round", zorder=3)
    ax_main.add_collection(lc)

    layers = [("halo", 9.0, 0.10), ("mid", 3.2, 0.22), ("core", 1.0, 1.0)] \
        if glow else [("core", 1.0, 1.0)]
    heads = []
    for _, smul, amul in layers:
        heads.append(ax_main.scatter(np.zeros(n_w), np.zeros(n_w), s=18 * smul,
                                     c=base, alpha=amul, linewidths=0, zorder=5))
    clock = ax_main.text(0.02, 0.975, "", transform=ax_main.transAxes,
                         ha="left", va="top", fontsize=11, color=th.ink,
                         weight="semibold")

    # -- side panels --------------------------------------------------------
    art = {}
    rho_exact = None
    if field is not None and getattr(field, "boltzmann_exact", False) \
            and hasattr(field, "rho_ss_on_grid"):
        rho_exact = field.rho_ss_on_grid(D)

    nb = relax_bins
    rng_box = [[box.lo[0], box.hi[0]], [box.lo[1], box.hi[1]]]
    if rho_exact is not None:
        k = rho_exact.shape[0] // nb
        ref = rho_exact[:nb*k, :nb*k].reshape(nb, k, nb, k).mean(axis=(1, 3))
        ref = ref / ref.sum()
    else:
        ref = None

    def hist_of(frame):
        H, _, _ = np.histogram2d(traj.x[:, frame, 0], traj.x[:, frame, 1],
                                 bins=nb, range=rng_box)
        s = H.sum()
        return H / s if s else H

    for k_panel, name in enumerate(panels, start=1):
        ax = axes[k_panel]
        if name == "density":
            im = ax.imshow(hist_of(frames[0]).T, origin="lower", extent=box.extent,
                           cmap=st.cmap_density(), interpolation="bilinear",
                           vmin=0, vmax=None)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_title(f"Walker density ({traj.n_walkers} walkers)")
            art["density"] = im
        elif name == "relax":
            if ref is None:
                raise ValueError("the 'relax' panel needs a field with an "
                                 "exact stationary density")
            ax.set_xlim(traj.times[lo], traj.times[min(hi, traj.n_frames) - 1])
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("$t$")
            ax.set_ylabel(r"TV$\,(\hat\rho_t,\ \rho_{\rm ss})$")
            ax.set_title("Relaxation to the stationary density")
            # The floor a *perfectly* equilibrated sample would still show,
            # from comparing n_walkers draws against a continuous density over
            # nb^2 bins. Without this line, a converged run looks stuck.
            floor = np.sqrt(nb * nb / (2.0 * np.pi * traj.n_walkers))
            ax.axhline(floor, color=th.ink_muted, lw=1.0, dashes=(2, 2))
            ax.text(0.99, floor, " sampling floor ", ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), fontsize=8,
                    color=th.ink_muted)
            ln, = ax.plot([], [], color=th.series[0], lw=2)
            dot, = ax.plot([], [], marker="o", ms=6, color=th.series[0])
            art["relax"] = (ln, dot, [], [])
        elif name == "msd":
            ax.set_xscale("log")
            ax.set_yscale("log")
            t = traj.times[1:]
            free = 2.0 * traj.d * D * t
            ax.plot(t, free, color=th.series[1], lw=1.6, dashes=(4, 1.5),
                    label=r"free $2dDt$")
            ax.set_xlim(t[0], t[-1])
            msd_all = traj.msd()
            ax.set_ylim(min(msd_all[1], free[0]) * 0.5, max(msd_all.max(), free[-1]) * 2)
            ax.set_xlabel("$t$")
            ax.set_ylabel(r"$\langle|x(t)-x(0)|^2\rangle$")
            ax.set_title("Mean squared displacement")
            ax.legend()
            ln, = ax.plot([], [], color=th.series[0], lw=2, label="simulated")
            art["msd"] = (ln, msd_all)
        else:
            raise ValueError(f"unknown panel {name!r}")

    fig.tight_layout()

    # -- frame update -------------------------------------------------------
    def update(f):
        a = max(lo, f - trail)
        window = x[:, a:f + 1]
        out_art = []
        if window.shape[1] >= 2:
            seg, keep = trail_segments(window, box)
            cols = _fading_colors(n_w, window.shape[1] - 1, base)
            cols[~keep, 3] = 0.0
            lc.set_segments(seg)
            lc.set_color(cols)
        else:
            lc.set_segments([])
        out_art.append(lc)

        pos = x[:, f]
        for h in heads:
            h.set_offsets(pos)
            out_art.append(h)
        clock.set_text(f"t = {f * traj.dt:.2f}")
        out_art.append(clock)

        if "density" in art:
            H = hist_of(f)
            im = art["density"]
            im.set_data(H.T)
            im.set_clim(0, max(H.max(), 1e-9))
            out_art.append(im)
        if "relax" in art:
            ln, dot, ts, vs = art["relax"]
            ts.append(f * traj.dt)
            vs.append(0.5 * np.abs(hist_of(f) - ref).sum())
            ln.set_data(ts, vs)
            dot.set_data([ts[-1]], [vs[-1]])
            out_art += [ln, dot]
        if "msd" in art:
            ln, msd_all = art["msd"]
            j = max(2, f - lo + 1)
            ln.set_data(traj.times[1:j], msd_all[1:j])
            out_art.append(ln)
        return out_art

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps,
                         blit=False, repeat=False)
    if out is not None:
        save_animation(anim, out, fps=fps, dpi=dpi, progress=progress)
        plt.close(fig)
    return anim


def animate_omega_sweep(field, omegas, D: float, *, n_grid: int = 220,
                        fps: int = 20, out=None, dpi: int = 140, seed: int = 0,
                        figsize=(12.4, 6.2), progress: bool = True,
                        n_samples: int = 40000):
    """Raise the rotational strength; watch the density refuse to move.

    Left: the drift field, which changes completely. Right: the exact
    stationary density, which is *bit-identical* at every frame because
    ``b_rot = omega A grad U`` is divergence-free and orthogonal to
    ``grad rho_ss``.

    This is the single most useful animation in the project. Any method that
    sees only the right-hand panel -- kernel density estimation of snapshots,
    score matching on snapshots -- cannot distinguish the first frame from the
    last, no matter how much data it is given.
    """
    omegas = np.asarray(omegas, float)
    th = st.active()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    f0 = field.with_omega(float(omegas[0]))
    from .fields import plot_density
    plot_density(f0, D, axes[1], n=n_grid,
                 title=r"Stationary density $\rho_{\rm ss}$  (identical throughout)")

    speeds = []
    for w in omegas:
        from .fields import field_on_grid
        vx, vy = field_on_grid(field.with_omega(float(w)), n_grid)
        speeds.append(np.percentile(np.hypot(vx, vy), 99.0))
    vmax = float(max(speeds))

    # The omega readout goes inside the left panel. A figure-level banner sits
    # in the same horizontal band as the axes titles, and collides with them
    # once tight_layout has run.
    def _stamp(w):
        return axes[0].text(
            0.025, 0.972,
            (r"$\omega = %.2f$" % w) + ("   (equilibrium)" if abs(w) < 1e-9 else ""),
            transform=axes[0].transAxes, ha="left", va="top", fontsize=13,
            weight="semibold", color=th.ink, zorder=10,
            bbox=dict(facecolor=th.surface, edgecolor="none", alpha=0.82,
                      boxstyle="round,pad=0.35"))

    def update(i):
        w = float(omegas[i])
        axes[0].clear()
        plot_drift(field.with_omega(w), axes[0], n=n_grid, style="lic",
                   seed=seed, vmax=vmax, colorbar=False,
                   title=r"Drift field   $b=-\nabla U+\omega A\nabla U$")
        # axes.clear() destroys the text artist, so re-stamp it each frame.
        _stamp(w)
        return []

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=len(omegas),
                         interval=1000 / fps, blit=False, repeat=False)
    if out is not None:
        save_animation(anim, out, fps=fps, dpi=dpi, progress=progress)
        plt.close(fig)
    return anim


def animate_relaxation(traj, field, D: float | None = None, *, bins: int = 64,
                       stride: int = 1, fps: int = 30, out=None, dpi: int = 140,
                       figsize=(12.6, 6.0), progress: bool = True,
                       frame_range=None):
    """A point source spreading into ``rho_ss``, beside the exact answer.

    Run this with ``x0`` a single point. The left panel is the empirical
    density at time ``t``; the right is the exact stationary density. When the
    two stop differing, the run has equilibrated -- which is how you choose
    ``burn_in`` honestly instead of guessing.
    """
    D = traj.D if D is None else D
    box = traj.box
    th = st.active()
    lo, hi = frame_range or (0, traj.n_frames)
    frames = list(range(lo, hi, stride))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    rng_box = [[box.lo[0], box.hi[0]], [box.lo[1], box.hi[1]]]

    def hist_of(f):
        H, _, _ = np.histogram2d(traj.x[:, f, 0], traj.x[:, f, 1], bins=bins,
                                 range=rng_box, density=True)
        return H

    im = axes[0].imshow(hist_of(frames[0]).T, origin="lower", extent=box.extent,
                        cmap=st.cmap_density(), interpolation="bilinear")
    axes[0].set_aspect("equal")
    axes[0].grid(False)
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    add_colorbar(axes[0], im, r"$\hat\rho_t$")

    from .fields import plot_density
    plot_density(field, D, axes[1], n=256, title=r"Exact  $\rho_{\rm ss}$")

    clock = axes[0].text(0.02, 0.975, "", transform=axes[0].transAxes, ha="left",
                         va="top", fontsize=11, color=th.ink, weight="semibold")

    # A fixed color scale, set from the equilibrated end of the run, so the
    # spreading is visible as the cloud dimming and widening rather than being
    # renormalised away frame by frame.
    vmax = float(np.percentile(hist_of(frames[-1]), 99.5)) * 1.6

    def update(f):
        H = hist_of(f)
        im.set_data(H.T)
        im.set_clim(0, vmax)
        axes[0].set_title(f"Walker density at $t = {f*traj.dt:.3f}$")
        clock.set_text(f"t = {f * traj.dt:.3f}")
        return [im, clock]

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps,
                         blit=False, repeat=False)
    if out is not None:
        save_animation(anim, out, fps=fps, dpi=dpi, progress=progress)
        plt.close(fig)
    return anim
