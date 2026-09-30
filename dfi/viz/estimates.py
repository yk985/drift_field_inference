"""Drawing a *fitted estimate* against the field it is meant to recover.

Quiver panels with a shared arrow scale, error heatmaps, and the bookkeeping
that keeps several fits comparable side by side. Used by the per-rung study
scripts; nothing here is specific to one estimator.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize

from . import style as st
from .fields import add_colorbar, to_image

__all__ = ["cell_centres", "vectors_at", "estimate_arrows", "quiver_panel",
           "error_panel", "row_colorbar", "estimate_maps", "human"]

#: Arrows per axis for a reference panel, and the density that fixes the arrow
#: length scale. Estimates with a native grid are drawn at their own
#: resolution instead; only arrow *length* is held to this common scale.
COARSE = 24
#: Resolution of error heatmaps.
FINE = 160


def cell_centres(box, n):
    """``n x n`` cell-centre coordinates as ``(X, Y)``.

    Centres, not ``box.axes``: on a periodic box those are the lower *edges*
    of the cells, and an arrow drawn at the edge of the cell it describes sits
    half a cell away from the value it is reporting.
    """
    h = box.length / n
    ax_ = [box.lo[i] + (np.arange(n) + 0.5) * h[i] for i in range(2)]
    return np.meshgrid(ax_[0], ax_[1], indexing="ij")


def vectors_at(fn, X, Y):
    """Evaluate a vector-valued callable at ``(X, Y)`` -> ``(vx, vy)``."""
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    v = np.asarray(fn(pts), float).reshape(X.shape + (2,))
    return v[..., 0], v[..., 1]


def estimate_arrows(est, box, max_arrows: int = 26, n: int = COARSE):
    """Arrows for a fitted estimate: ``(X, Y, vx, vy, supported)``.

    A binned estimate is sampled at *its own* cell centres. Its bin count is
    chosen from the data and differs between fits, and resampling it onto a
    common grid would alias one resolution against another -- blocky artefacts
    that belong to the picture rather than to the estimator. Anything without a
    native grid (a kernel estimate is defined everywhere) is drawn on an
    ``n x n`` grid.
    """
    bins = getattr(est, "bins", None)
    if isinstance(bins, (int, np.integer)):
        stride = max(1, int(np.ceil(bins / max_arrows)))
        h = box.length / bins
        idx = np.arange(0, bins, stride)
        ax_ = [box.lo[i] + (idx + 0.5) * h[i] for i in range(2)]
        X, Y = np.meshgrid(ax_[0], ax_[1], indexing="ij")
    else:
        X, Y = cell_centres(box, n)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    v = np.asarray(est.predict(pts), float).reshape(X.shape + (2,))
    sup = np.asarray(est.support(pts)).reshape(X.shape)
    return X, Y, v[..., 0], v[..., 1], sup


def quiver_panel(ax, box, X, Y, vx, vy, *, vmax, title="", support=None,
                 colorbar=False, n_ref: int = COARSE):
    """Arrows on a shared scale, with unsupported cells marked.

    ``scale`` is pinned to ``vmax`` rather than left to matplotlib's per-call
    autoscaling, because autoscaled arrows make a field that has been shrunk
    towards zero look identical to one that has not -- which is exactly the
    failure mode (over-smoothing) these panels exist to show.
    """
    th = st.active()
    if support is not None and not np.all(support):
        bad = ~np.asarray(support, bool)
        ax.scatter(X[bad], Y[bad], s=46, marker="s", color=th.ink_muted,
                   alpha=0.16, linewidths=0, zorder=0)
    q = ax.quiver(X, Y, vx, vy, np.hypot(vx, vy), cmap=st.cmap_density(),
                  norm=Normalize(0.0, vmax), pivot="mid", width=0.007,
                  scale=vmax * n_ref * 0.85, scale_units="width", zorder=2)
    ax.set_xlim(box.lo[0], box.hi[0])
    ax.set_ylim(box.lo[1], box.hi[1])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    if colorbar:
        add_colorbar(ax, q, r"$|b|$")
    return q


def error_panel(ax, box, err, *, vmax, title="", colorbar=False):
    """Relative error as a heatmap; unsupported cells left blank."""
    im = ax.imshow(to_image(err), origin="lower", extent=box.extent,
                   cmap=st.CMAP_SEQ2, vmin=0.0, vmax=vmax,
                   interpolation="nearest", zorder=0)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    if colorbar:
        add_colorbar(ax, im, r"$|\hat b - b| \,/\, \mathrm{RMS}|b|$")
    return im


def row_colorbar(fig, axs, mappable, label, **kw):
    """One recessive colorbar spanning a whole row of panels.

    Attached to the row rather than to its last panel: a colorbar on a single
    axes shrinks only that axes, which leaves the columns misaligned.
    """
    th = st.active()
    cb = fig.colorbar(mappable, ax=list(axs), pad=0.015, fraction=0.022, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2, color=th.ink_muted,
                      labelcolor=th.ink_secondary)
    cb.set_label(label, fontsize=9, color=th.ink_secondary)
    return cb


def estimate_maps(field, est, *, fine: int = FINE):
    """Arrows, an error map, and box coverage for one fit.

    Returns ``(arrows, err, box_coverage)``. Two coverage questions can
    disagree wildly: the metrics' ``supported_fraction`` is taken over points
    drawn from ``rho_ss`` ("can it answer where the walkers are?"), this one
    over the box ("can it answer anywhere?"). At small ``D`` the first can be
    100% while the second is a third, and reporting only the first makes a
    badly incomplete estimate look total.
    """
    box = field.box
    arrows = estimate_arrows(est, box)
    pts = box.grid_points((fine, fine))
    b_true = np.asarray(field.drift(pts), float)
    b_hat = np.asarray(est.predict(pts), float)
    scale = float(np.sqrt(np.mean(np.sum(b_true ** 2, axis=-1))))
    err = np.linalg.norm(b_hat - b_true, axis=-1).reshape(fine, fine) / scale
    sup = np.asarray(est.support(pts)).reshape(fine, fine)
    return arrows, np.where(sup, err, np.nan), float(sup.mean())


def human(n: int) -> str:
    """Counts a reader can compare at a glance: 484, 3.4k, 9.8M."""
    for lim, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= lim:
            return f"{n / lim:.1f}{suf}"
    return str(int(n))
