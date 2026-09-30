"""Drawing drift fields, potentials and stationary densities.

Conventions used throughout, so panels can be overlaid safely:

- field arrays are indexed ``[ix, iy]`` (matching ``dfi.fields``),
- :func:`to_image` transposes them for ``imshow(origin='lower')``,
- every 2D axes gets ``extent=box.extent`` so data coordinates *are* physical
  coordinates, and trajectories can be plotted straight on top.

Color follows the job, not the mood: speed and density are magnitudes and get
the one-hue sequential ramp; the potential is signed and gets the diverging
ramp with a neutral midpoint at zero.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from . import style as st
from .lic import line_integral_convolution

__all__ = ["to_image", "field_on_grid", "plot_drift", "plot_potential",
           "plot_density", "plot_helmholtz", "plot_field_card", "plot_field_1d",
           "add_colorbar"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def to_image(a: np.ndarray) -> np.ndarray:
    """``[ix, iy]`` field array -> array ready for ``imshow(origin='lower')``."""
    return np.asarray(a).T


def _ax(ax, figsize=(5.2, 5.2)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def add_colorbar(ax, mappable, label: str = "", *, pad: float = 0.02,
                 fraction: float = 0.045):
    """A thin, recessive colorbar. The data is the subject; this is a key."""
    th = st.active()
    cb = ax.figure.colorbar(mappable, ax=ax, pad=pad, fraction=fraction)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2, color=th.ink_muted,
                      labelcolor=th.ink_secondary)
    if label:
        cb.set_label(label, fontsize=9, color=th.ink_secondary)
    return cb


def _square(ax, box):
    ax.set_xlim(box.lo[0], box.hi[0])
    ax.set_ylim(box.lo[1], box.hi[1])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def field_on_grid(field, n: int = 256, which: str = "drift"):
    """Evaluate a 2D field on an ``n x n`` grid.

    Returns ``(vx, vy)`` indexed ``[ix, iy]``. ``which`` selects the full drift,
    its gradient part, or its rotational part -- the three panels of a
    Helmholtz figure.
    """
    if field.d != 2:
        raise ValueError(f"field_on_grid needs d == 2, got d = {field.d}")
    pts = field.box.grid_points((n, n))
    if which == "drift":
        v = field.drift(pts)
    elif which == "gradient":
        v = field.gradient_part(pts)
    elif which == "rotational":
        v = field.rotational_part(pts)
    else:
        raise ValueError(f"unknown component {which!r}")
    v = v.reshape(n, n, 2)
    return v[..., 0], v[..., 1]


def _scalar_on_grid(field, n, fn):
    pts = field.box.grid_points((n, n))
    return np.asarray(fn(pts)).reshape(n, n)


# --------------------------------------------------------------------------
# the drift field
# --------------------------------------------------------------------------

def plot_drift(field, ax=None, *, n: int = 256, style: str = "lic",
               color_by: str = "speed", arrows: int = 0, seed: int = 0,
               vmax: float | None = None, title: str | None = None,
               colorbar: bool = True, which: str = "drift",
               lic_kwargs: dict | None = None, alpha_texture: float = 0.45,
               stream_density: float = 1.2):
    """Draw a 2D drift field.

    ``style``:

    ``'lic'``
        Line-integral-convolution texture tinted by speed. Shows the direction
        at *every* pixel, so saddles and stagnation points are visible without
        hunting. The default, and the one to use for a headline figure.
    ``'stream'``
        Matplotlib streamlines over a speed heatmap. Cleaner for slides;
        streamline seeding is arbitrary, so fine structure can hide between
        lines.
    ``'quiver'``
        Plain arrows. Only legible on a coarse grid.

    ``arrows`` overlays that many arrows per axis on top of any style, which
    is worth doing because a texture shows orientation but not *sign*.
    """
    ax = _ax(ax)
    box = field.box
    vx, vy = field_on_grid(field, n, which)
    speed = np.hypot(vx, vy)
    if vmax is None:
        vmax = float(np.percentile(speed, 99.0)) or 1.0
    norm = Normalize(0.0, vmax)
    cmap = st.cmap_density()

    if style == "lic":
        # Grain size and streak length are set relative to the grid so a
        # figure looks the same at any resolution. noise_scale > 1 gives a
        # coarser speckle, which reads far better once the figure is scaled
        # down into a report than single-pixel noise does.
        kw = dict(n_steps=max(14, n // 8), step_len=0.8,
                  periodic=box.periodic, seed=seed, contrast=1.15,
                  noise_scale=max(2, n // 110))
        kw.update(lic_kwargs or {})
        tex = line_integral_convolution(vx, vy, **kw)
        if color_by == "speed":
            rgb = cmap(norm(to_image(speed)))[..., :3]
        else:
            base = np.array(plt.matplotlib.colors.to_rgb(st.active().series[0]))
            rgb = np.broadcast_to(base, to_image(speed).shape + (3,)).copy()
        # Multiply the texture in as a shading term, keeping enough of the
        # base color that the sequential ramp still reads as a magnitude.
        shade = (1.0 - alpha_texture) + alpha_texture * to_image(tex)[..., None]
        ax.imshow(np.clip(rgb * shade, 0, 1), origin="lower", extent=box.extent,
                  interpolation="bilinear", zorder=0)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    elif style == "stream":
        im = ax.imshow(to_image(speed), origin="lower", extent=box.extent,
                       cmap=cmap, norm=norm, interpolation="bilinear", zorder=0)
        xs, ys = box.axes((n, n))
        ax.streamplot(xs, ys, to_image(vx), to_image(vy),
                      density=stream_density, linewidth=0.7,
                      color=st.active().ink, arrowsize=0.7,
                      broken_streamlines=False)
        sm = im
    elif style == "quiver":
        m = max(1, n // 28)
        xs, ys = box.axes((n, n))
        X, Y = np.meshgrid(xs[::m], ys[::m], indexing="ij")
        q = ax.quiver(X, Y, vx[::m, ::m], vy[::m, ::m], speed[::m, ::m],
                      cmap=cmap, norm=norm, pivot="mid", width=0.004)
        sm = q
    else:
        raise ValueError(f"unknown style {style!r}")

    if arrows:
        m = max(1, n // arrows)
        xs, ys = box.axes((n, n))
        X, Y = np.meshgrid(xs[::m], ys[::m], indexing="ij")
        ax.quiver(X, Y, vx[::m, ::m], vy[::m, ::m], pivot="mid",
                  color=st.active().ink, alpha=0.5, width=0.003,
                  scale=None, zorder=3)

    _square(ax, box)
    if colorbar:
        add_colorbar(ax, sm, r"$|b|$")
    ax.set_title(title if title is not None else _default_title(field, which))
    return ax


def _default_title(field, which):
    return {"drift": "Drift field  $b(x)$",
            "gradient": r"Gradient part  $-\nabla U$",
            "rotational": r"Rotational part  $b_{\rm rot}$"}[which]


def plot_potential(field, ax=None, *, n: int = 256, contours: int = 9,
                   title: str = "Potential  $U(x)$", colorbar: bool = True):
    """Potential heatmap with contours.

    ``U`` is signed and zero-mean here, so it gets the diverging ramp with a
    neutral midpoint -- the gray band marks ``U = 0``, and blue/red read as
    "valley" and "hill" without a legend.
    """
    ax = _ax(ax)
    U = _scalar_on_grid(field, n, field.potential)
    lim = float(np.abs(U).max()) or 1.0
    im = ax.imshow(to_image(U), origin="lower", extent=field.box.extent,
                   cmap=st.cmap_div(), norm=TwoSlopeNorm(0.0, -lim, lim),
                   interpolation="bilinear")
    if contours:
        xs, ys = field.box.axes((n, n))
        ax.contour(xs, ys, to_image(U), levels=contours,
                   colors=st.active().ink, linewidths=0.4, alpha=0.35)
    _square(ax, field.box)
    if colorbar:
        add_colorbar(ax, im, r"$U$")
    ax.set_title(title)
    return ax


def plot_density(field, D: float, ax=None, *, n: int = 256, log: bool = False,
                 title: str | None = None, colorbar: bool = True,
                 samples: np.ndarray | None = None, bins: int = 96):
    """Stationary density.

    Draws the exact ``rho_ss`` when the field has one; pass ``samples`` to draw
    an empirical histogram instead (for the side-by-side that shows a
    simulation actually reached the right distribution).
    """
    ax = _ax(ax)
    box = field.box
    if samples is not None:
        H, _, _ = np.histogram2d(
            samples[..., 0].ravel(), samples[..., 1].ravel(), bins=bins,
            range=[[box.lo[0], box.hi[0]], [box.lo[1], box.hi[1]]], density=True)
        img, lbl = H.T, r"empirical $\rho$"
    else:
        if hasattr(field, "rho_ss_on_grid"):
            rho = field.rho_ss_on_grid(D)
            if rho.shape != (n, n):
                rho = _scalar_on_grid(field, n, lambda p: np.exp(field.log_rho_ss(p, D)))
        else:
            rho = _scalar_on_grid(field, n, lambda p: np.exp(field.log_rho_ss(p, D)))
        img, lbl = to_image(rho), r"$\rho_{\rm ss}$"
    if log:
        img = np.log10(np.maximum(img, img[img > 0].min() if np.any(img > 0) else 1e-12))
        lbl = r"$\log_{10}\,\rho$"
    im = ax.imshow(img, origin="lower", extent=box.extent,
                   cmap=st.cmap_density(), interpolation="bilinear")
    _square(ax, box)
    if colorbar:
        add_colorbar(ax, im, lbl)
    ax.set_title(title if title is not None
                 else f"Stationary density   $D={D:g}$")
    return ax


def plot_helmholtz(field, *, n: int = 256, figsize=(13.2, 4.6), seed: int = 0,
                   suptitle: str | None = None):
    """The decomposition ``b = -grad U + b_rot`` as three panels.

    All three share one color scale, so the relative size of the two parts is
    readable directly off the image -- which is the whole point when you are
    arguing about what a snapshot-only method can and cannot see.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    parts = ["drift", "gradient", "rotational"]
    vmax = 0.0
    for w in parts:
        vx, vy = field_on_grid(field, n, w)
        vmax = max(vmax, float(np.percentile(np.hypot(vx, vy), 99.0)))
    for ax, w in zip(axes, parts):
        plot_drift(field, ax, n=n, style="lic", vmax=vmax, seed=seed,
                   which=w, colorbar=(w == "rotational"))
        if w != "drift":
            ax.set_ylabel("")
    th = st.active()
    if field.is_gradient:
        axes[2].text(0.5, 0.5, "identically zero\n(equilibrium)",
                     transform=axes[2].transAxes, ha="center", va="center",
                     color=th.ink_secondary, fontsize=10)
    fig.suptitle(suptitle or f"Helmholtz decomposition  --  {field.name}",
                 x=0.007, ha="left", fontsize=12, weight="semibold",
                 color=th.ink)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, axes


def plot_field_card(field, D: float, *, n: int = 256, figsize=(11.0, 9.4),
                    seed: int = 0):
    """A four-panel identity card for a field: drift, potential, density, spectrum.

    Worth generating once per experiment and keeping next to the results -- it
    is the answer to "what did you actually run this on".
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_drift(field, axes[0, 0], n=n, seed=seed, arrows=18)
    plot_potential(field, axes[0, 1], n=n)
    plot_density(field, D, axes[1, 0], n=n)

    ax = axes[1, 1]
    th = st.active()
    U = _scalar_on_grid(field, n, field.potential)
    P = np.abs(np.fft.rfftn(U)) ** 2
    from ..fields.noise import wavenumbers
    _, kmag = wavenumbers(field.box, (n, n))
    kb = np.linspace(0, kmag.max(), 60)
    which = np.digitize(kmag.ravel(), kb)
    prof = np.array([P.ravel()[which == i].mean() if np.any(which == i) else np.nan
                     for i in range(1, len(kb))])
    kc = 0.5 * (kb[1:] + kb[:-1])
    good = np.isfinite(prof) & (prof > 0)
    ax.loglog(kc[good], prof[good] / np.nanmax(prof), color=th.series[0], lw=2)
    ax.set_xlabel(r"wavenumber $|k|$")
    ax.set_ylabel("normalised power")
    ax.set_title("Potential power spectrum")
    ax.grid(True, which="both", alpha=0.35)

    st.figure_caption(fig, f"{field.summary()}   |   D = {D:g}", y=0.005)
    fig.tight_layout()
    return fig, axes


# --------------------------------------------------------------------------
# 1D
# --------------------------------------------------------------------------

def plot_field_1d(field, D: float | None = None, *, n: int = 600,
                  figsize=(12.4, 3.4), axes=None):
    """Drift, potential and stationary density for a 1D system.

    Three panels rather than two with a twin y-axis: ``U`` and ``rho`` have
    unrelated units and unrelated scales, and overlaying them on two y-scales
    lets the reader infer a crossing point that means nothing. Separate panels
    on a shared x-axis show the same relationship honestly -- the density peaks
    where the potential dips, and you can see it without decoding two axes.

    In 1D every drift is a gradient (there is no divergence-free field on a
    line), so there is no rotational component to draw. Worth saying in the
    caption: it makes the identifiability point before the 2D figure lands.
    """
    if field.d != 1:
        raise ValueError(f"plot_field_1d needs d == 1, got {field.d}")
    show_rho = D is not None and field.boltzmann_exact
    n_panels = 3 if show_rho else 2
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharex=True)
    else:
        fig = axes[0].figure
    th = st.active()
    x = np.linspace(field.box.lo[0], field.box.hi[0], n)
    xc = x[:, None]

    ax = axes[0]
    ax.plot(x, field.drift(xc)[:, 0], color=th.series[0], lw=2)
    ax.axhline(0, color=th.ink_muted, lw=0.8, alpha=0.6)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$b(x)$")
    ax.set_title("Drift")

    U = field.potential(xc)
    ax = axes[1]
    ax.plot(x, U, color=th.series[1], lw=2)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$U(x)$")
    ax.set_title("Potential")

    if show_rho:
        p = np.exp(-(U - U.min()) / D)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        p = p / trapz(p, x)
        ax = axes[2]
        ax.fill_between(x, 0, p, color=th.series[0], alpha=0.18, lw=0)
        ax.plot(x, p, color=th.series[0], lw=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\rho_{\rm ss}(x)$")
        ax.set_title(r"Stationary density   $\rho\propto e^{-U/D}$"
                     + f"   $(D={D:g})$")
    fig.tight_layout()
    return fig, axes
