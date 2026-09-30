"""Drawing walker paths, and the diagnostics that say whether they are usable.

Two jobs here, and they are different jobs.

The *beauty* plots (:func:`plot_trajectories`) put paths on top of the field so
you can see the dynamics -- walkers pooling in potential wells, or circulating
when a rotational component is present.

The *diagnostic* plots are the ones that decide whether an estimate means
anything. :func:`plot_sampling_density` is the important one: a Kramers-Moyal
estimate is a local average, so its variance at ``x`` scales like ``1/n(x)``,
and the regions where a method "fails" are usually just the regions nothing
visited. Plotting error without plotting sample density next to it invites the
wrong conclusion.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from . import style as st
from .fields import add_colorbar, plot_drift, plot_density, to_image

__all__ = ["plot_trajectories", "plot_snapshot", "plot_sampling_density",
           "plot_msd", "plot_increment_check", "plot_trajectories_1d",
           "trail_segments"]


def _ax(ax, figsize=(5.6, 5.6)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def trail_segments(x, box, *, max_jump_frac: float = 0.5):
    """Path points -> line segments, with periodic wrap-arounds removed.

    ``x`` has shape ``(n_walkers, n_frames, 2)``. On a torus a walker that
    leaves the right edge reappears on the left, and drawing that as a line
    paints a stripe across the whole figure. Any segment longer than half the
    box is a wrap artefact, so we drop it -- the path simply reappears on the
    other side, which is what actually happened.

    Returns ``(segments, keep)`` where ``segments`` has shape ``(M, 2, 2)``.
    """
    x = np.asarray(x, float)
    seg = np.stack([x[:, :-1], x[:, 1:]], axis=2)          # (N, T, 2, 2)
    seg = seg.reshape(-1, 2, 2)
    delta = seg[:, 1] - seg[:, 0]
    if box.periodic:
        keep = np.all(np.abs(delta) < max_jump_frac * box.length, axis=-1)
    else:
        keep = np.ones(len(seg), bool)
    return seg, keep


def plot_trajectories(traj, field=None, ax=None, *, n_show: int = 16,
                      background: str | None = "lic", frame_range=None,
                      linewidth: float = 1.0, alpha: float = 0.85,
                      color_by: str = "time", show_heads: bool = True,
                      seed: int = 0, n_grid: int = 256, title: str | None = None,
                      D: float | None = None, mute_background: float = 0.4):
    """Walker paths over the field that produced them.

    ``color_by``:

    ``'time'`` (default)
        A single hue ramping along each path, in the *contrasting* family to
        the blue field beneath, so the paths sit clearly on top. Time is a
        magnitude, so a sequential ramp is the right encoding for it.
    ``'walker'``
        The categorical palette, one slot per walker -- identity encoding.
        Only meaningful for a handful of walkers; the palette has eight slots
        and cycling it past that turns identity into decoration. Raises above
        eight rather than silently repeating colors.

    ``mute_background`` washes the field towards the surface color so the
    trajectories are the subject and the field is context. Set it to 0 when the
    field itself is the point.

    Keep ``n_show`` small. Sixty overlapping random walks is a hairball that
    hides both the paths and the field; a dozen shows the same behaviour and
    stays readable.
    """
    ax = _ax(ax)
    box = traj.box
    if background == "lic" and field is not None:
        plot_drift(field, ax, n=n_grid, style="lic", seed=seed, colorbar=False,
                   alpha_texture=0.32, title="")
    elif background == "density" and field is not None:
        plot_density(field, D if D is not None else traj.D, ax, n=n_grid,
                     colorbar=False, title="")
    elif background == "quiver" and field is not None:
        plot_drift(field, ax, n=n_grid, style="quiver", colorbar=False,
                   title="")
    elif background == "potential" and field is not None:
        from .fields import plot_potential
        plot_potential(field, ax, n=n_grid, colorbar=False, title="")

    th = st.active()
    if background is not None and field is not None and mute_background > 0:
        ax.add_patch(plt.Rectangle(
            (box.lo[0], box.lo[1]), box.length[0], box.length[1],
            facecolor=th.surface, alpha=mute_background, edgecolor="none",
            zorder=1))

    lo, hi = frame_range or (0, traj.n_frames)
    x = traj.x[:n_show, lo:hi, :2]
    seg, keep = trail_segments(x, box)

    n_w, n_seg = x.shape[0], x.shape[1] - 1
    if color_by == "walker":
        if n_w > len(th.series):
            raise ValueError(
                f"color_by='walker' with {n_w} walkers exceeds the "
                f"{len(th.series)}-slot categorical palette; lower n_show or "
                "use color_by='time'")
        idx = np.repeat(np.arange(n_w), n_seg)
        cols = np.array([to_rgb(th.series[i]) for i in idx])
        colors = np.concatenate([cols, np.full((len(cols), 1), alpha)], axis=1)
    elif color_by == "time":
        t = np.tile(np.linspace(0.0, 1.0, n_seg), n_w)
        # Orange ramp over the blue field: one contrasting hue, so the paths
        # read as a separate layer rather than blending into the background.
        colors = st.CMAP_SEQ2(0.35 + 0.6 * t)
        colors[:, 3] = alpha
    else:
        raise ValueError(f"unknown color_by {color_by!r}")

    ax.add_collection(LineCollection(seg[keep], colors=colors[keep],
                                     linewidths=linewidth, capstyle="round",
                                     zorder=2))
    if show_heads:
        head = x[:, -1]
        ax.scatter(head[:, 0], head[:, 1], s=22, c=th.surface,
                   edgecolors=th.ink, linewidths=1.0, zorder=4)

    ax.set_xlim(box.lo[0], box.hi[0])
    ax.set_ylim(box.lo[1], box.hi[1])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title if title is not None
                 else f"{min(n_show, traj.n_walkers)} walkers, "
                      f"t = {(hi - lo - 1) * traj.dt:g}")
    return ax


def plot_snapshot(samples, box, ax=None, *, s: float = 3.0, alpha: float = 0.35,
                  title: str = "Snapshot samples", color=None):
    """Scatter of positions with no time ordering -- the snapshot dataset.

    Worth drawing next to a trajectory plot of the *same* system: the two look
    almost identical, which is the visual form of the identifiability argument.
    """
    ax = _ax(ax)
    th = st.active()
    ax.scatter(samples[..., 0].ravel(), samples[..., 1].ravel(), s=s,
               c=color or th.series[0], alpha=alpha, linewidths=0, zorder=2)
    ax.set_xlim(box.lo[0], box.hi[0])
    ax.set_ylim(box.lo[1], box.hi[1])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    return ax


def plot_sampling_density(traj, ax=None, *, bins: int = 96, log: bool = True,
                          title: str = "Where the data actually is",
                          contour_frac: float | None = 0.02):
    """Histogram of visited positions -- the denominator of every local estimate.

    The optional contour marks the region holding all but ``contour_frac`` of
    the samples. Outside it, a binned estimator is averaging a handful of
    increments and its error is dominated by variance, not by any property of
    the method. Report errors inside this contour, or report them as a function
    of local density -- but do not average over the empty region and call it a
    method comparison.
    """
    ax = _ax(ax)
    box = traj.box
    H, xe, ye = np.histogram2d(
        traj.x[..., 0].ravel(), traj.x[..., 1].ravel(), bins=bins,
        range=[[box.lo[0], box.hi[0]], [box.lo[1], box.hi[1]]])
    img = np.log10(H + 1.0) if log else H
    im = ax.imshow(img.T, origin="lower", extent=box.extent,
                   cmap=st.cmap_density(), interpolation="nearest")
    if contour_frac:
        flat = np.sort(H.ravel())[::-1]
        csum = np.cumsum(flat)
        thresh = flat[np.searchsorted(csum, (1.0 - contour_frac) * csum[-1])]
        xc = 0.5 * (xe[1:] + xe[:-1])
        yc = 0.5 * (ye[1:] + ye[:-1])
        ax.contour(xc, yc, H.T, levels=[thresh], colors=[st.active().series[1]],
                   linewidths=1.2)
        ax.plot([], [], color=st.active().series[1], lw=1.2,
                label=f"{100*(1-contour_frac):g}% of samples")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(box.lo[0], box.hi[0])
    ax.set_ylim(box.lo[1], box.hi[1])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    add_colorbar(ax, im, r"$\log_{10}$ counts" if log else "counts")
    return ax


def plot_msd(traj, ax=None, *, field=None, title="Mean squared displacement"):
    """MSD against the free-diffusion line ``2 d D t``.

    A cheap, strong check: at short times the walkers cannot know about the
    drift, so the MSD *must* follow the free line. If it does not, ``D`` and
    ``dt`` are not what you think. Departure at long times is the drift and the
    boundary doing their work.
    """
    ax = _ax(ax, (5.6, 4.0))
    th = st.active()
    t, msd = traj.times, traj.msd()
    free = 2.0 * traj.d * traj.D * t
    ax.loglog(t[1:], msd[1:], color=th.series[0], lw=2, label="simulated")
    ax.loglog(t[1:], free[1:], color=th.series[1], lw=1.6, dashes=(4, 1.5),
              label=r"free diffusion $2dDt$")
    if traj.box.periodic:
        sat = np.sum(traj.box.length ** 2) / 6.0
        ax.axhline(sat, color=th.ink_muted, lw=1.0, dashes=(1, 2))
        ax.text(t[1], sat * 1.1, "torus saturation", fontsize=8,
                color=th.ink_muted, va="bottom")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\langle |x(t)-x(0)|^2\rangle$")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_increment_check(traj, ax=None, *, field=None,
                         title="Increment distribution"):
    """Are the recorded increments consistent with the ``D`` we asked for?

    Histogram of ``dx`` against ``N(b dt, 2 D dt)``. Two failure modes show up
    here immediately: a variance mismatch (wrong ``D``, or measurement noise
    inflating it by ``2 sigma^2``) and heavy tails (``dt`` too coarse for the
    field, so the Euler step is jumping across structure).
    """
    ax = _ax(ax, (5.6, 4.0))
    th = st.active()
    dx = traj.increments().ravel()
    sd = np.sqrt(2.0 * traj.D * traj.dt)
    ax.hist(dx / sd, bins=120, density=True, color=th.series[0], alpha=0.75,
            edgecolor="none", label="observed")
    z = np.linspace(-5, 5, 400)
    ax.plot(z, np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi), color=th.series[1],
            lw=2, label=r"$\mathcal{N}(0,1)$")
    obs = dx.std() / sd
    ax.set_xlabel(r"increment / $\sqrt{2D\,\Delta t}$")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    extra = ""
    if traj.obs_noise:
        pred = np.sqrt(1.0 + 2.0 * traj.obs_noise ** 2 / (2 * traj.D * traj.dt))
        extra = f"\npredicted with $\\sigma_{{obs}}$: {pred:.3f}"
    ax.text(0.02, 0.96, f"observed sd: {obs:.3f}{extra}", transform=ax.transAxes,
            va="top", fontsize=8.5, color=th.ink_secondary)
    return ax


def plot_trajectories_1d(traj, field=None, *, n_show: int = 12,
                         figsize=(11.0, 4.0), axes=None, D: float | None = None):
    """1D paths as ``x`` against ``t``, beside the density they sample.

    The classic double-well picture: flat stretches in one well, rare hops. If
    the run shows no hops, the estimator has no information about the barrier
    and no method will recover it -- read that off this plot before blaming an
    estimator.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={"width_ratios": [3, 1]})
    else:
        fig = axes[0].figure
    th = st.active()
    ax = axes[0]
    for i in range(min(n_show, traj.n_walkers)):
        ax.plot(traj.times, traj.x[i, :, 0],
                color=th.series[i % len(th.series)], lw=0.9, alpha=0.8)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title(f"{min(n_show, traj.n_walkers)} of {traj.n_walkers} walkers")

    ax = axes[1]
    ax.hist(traj.x[..., 0].ravel(), bins=90, density=True, orientation="horizontal",
            color=th.series[0], alpha=0.75, edgecolor="none", label="sampled")
    if field is not None and field.boltzmann_exact:
        Dv = D if D is not None else traj.D
        xs = np.linspace(traj.x.min(), traj.x.max(), 400)
        U = field.potential(xs[:, None])
        p = np.exp(-(U - U.min()) / Dv)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        p = p / trapz(p, xs)
        ax.plot(p, xs, color=th.series[1], lw=2, label=r"$e^{-U/D}$")
        ax.legend(fontsize=8)
    ax.set_ylim(axes[0].get_ylim())
    ax.set_xlabel("density")
    ax.set_yticklabels([])
    ax.set_title("Occupancy")
    fig.tight_layout()
    return fig, axes
