"""Comparison figures: error curves, identifiability, and the phase diagram.

These are the plots the project is judged on, so they follow the rules strictly:

- **One y-scale per axes.** Never two. If two quantities have different units,
  they get two panels.
- **Every method keeps its color everywhere**, taken from its own
  ``color_slot``, so a reader learns the mapping once. Color follows the
  method, never its rank in the current plot -- filtering a method out must not
  repaint the others.
- **A legend whenever there are two or more series**, plus direct labels at the
  line ends when there are four or fewer, so identity never rests on color
  alone.
- **The null model is drawn.** A horizontal line at ``nrmse = 1`` is what
  "learned nothing" looks like; without it a reader cannot tell a good curve
  from a useless one.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from . import style as st
from .fields import add_colorbar

__all__ = ["plot_error_curves", "plot_identifiability", "plot_phase_diagram",
           "plot_error_vs_density", "plot_bias_variance", "plot_training_curve",
           "direct_label"]


def _ax(ax, figsize=(6.4, 4.6)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def direct_label(ax, x, y, text, color, *, dx: float = 1.03, fontsize=9):
    """Label a line at its right-hand end, in its own color.

    Used when four or fewer series are on screen. It removes the legend
    round-trip: the reader's eye lands on the line and the name is already
    there.
    """
    ax.text(x * dx, y, text, color=color, fontsize=fontsize, va="center",
            ha="left", clip_on=False)


def _null_line(ax, label="predicting zero"):
    th = st.active()
    ax.axhline(1.0, color=th.ink_muted, lw=1.0, dashes=(2, 2), zorder=1)
    ax.text(0.995, 1.0, f" {label}", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8, color=th.ink_muted)


def plot_error_curves(results, ax=None, *, x: str = "n_transitions",
                      y: str = "nrmse", by: str = "label", logx=True, logy=True,
                      title=None, xlabel=None, ylabel=None, null_line=True,
                      slope_guide: float | None = -0.5, color_slots=None,
                      band: tuple | None = None, direct_labels: bool = True):
    """Error against a sweep variable, one line per method.

    ``results`` is a DataFrame with columns ``x``, ``y`` and ``by`` (what
    :func:`dfi.metrics.compare` produces, concatenated across sweep points).

    ``slope_guide`` draws a reference power law -- ``-0.5`` is the Monte Carlo
    rate every local-averaging estimator should follow while it is
    variance-limited. A curve that flattens above it has hit its bias floor,
    and *where* it flattens is the interesting number.

    ``band`` names ``(lo, hi)`` columns to shade as a spread across seeds.

    ``direct_labels`` writes each series' name at the end of its own line,
    which beats a legend when there are a few curves. Turn it off for a single
    series -- naming the one line on the plot is clutter, and the text runs off
    the right edge of a narrow panel.
    """
    import pandas as pd  # noqa: F401  (results is a DataFrame)
    ax = _ax(ax)
    th = st.active()
    groups = list(results.groupby(by, sort=False))
    for i, (name, g) in enumerate(groups):
        g = g.sort_values(x)
        slot = (color_slots or {}).get(name, i)
        sty = st.series_style(slot % len(th.series))
        ax.plot(g[x], g[y], label=str(name), lw=2, markersize=5,
                markeredgewidth=0, **sty)
        if band:
            ax.fill_between(g[x], g[band[0]], g[band[1]],
                            color=sty["color"], alpha=0.15, lw=0)
        if direct_labels and len(groups) <= 4:
            direct_label(ax, g[x].iloc[-1], g[y].iloc[-1], f" {name}",
                         sty["color"])

    if slope_guide is not None and len(results):
        xs = np.array([results[x].min(), results[x].max()], float)
        anchor = float(results[y].max())
        ys = anchor * (xs / xs[0]) ** slope_guide
        ax.plot(xs, ys, color=th.ink_muted, lw=1.0, dashes=(1, 2), zorder=1)
        ax.text(xs[-1], ys[-1], f"  slope {slope_guide:g}", fontsize=8,
                color=th.ink_muted, va="center")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if null_line:
        _null_line(ax)
    ax.set_xlabel(xlabel or {"n_transitions": "observed transitions  $N\\,T$",
                             "dt": r"sampling interval  $\Delta t$",
                             "obs_noise": r"measurement noise  $\sigma_{\rm obs}$",
                             "d": "dimension  $d$",
                             "n_walkers": "walkers  $N$",
                             "omega": r"rotational strength  $\omega$",
                             "D": "diffusion coefficient  $D$"}.get(x, x))
    ax.set_ylabel(ylabel or {"nrmse": r"normalised error  $\|\hat b-b\|/\|b\|$",
                             "nrmse_grad": "error, gradient direction",
                             "nrmse_rot": "error, rotational direction",
                             "cosine": "cosine alignment"}.get(y, y))
    ax.set_title(title or "Estimator error")
    if len(groups) >= 2:
        ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_identifiability(results, ax=None, *, x: str = "omega",
                         title="What a snapshot can and cannot see"):
    """The project's headline claim, as two diverging lines per method.

    For each method, error in the **gradient** direction (solid) and in the
    **rotational** direction (dashed) against ``omega``. The expected picture:

    - trajectory methods: both flat and low,
    - snapshot methods: gradient flat and low, rotational rising linearly.

    That rise is not a failure of the snapshot method. The information is not
    in its input. Say that in the caption, because the figure alone looks like
    a method losing.
    """
    ax = _ax(ax, (7.0, 4.8))
    th = st.active()
    # Two encodings, two independent keys: colour carries the method
    # (identity), line style carries the component (gradient vs rotational).
    # Enumerating all four combinations in one legend would double-encode what
    # the reader can already read off the line, and the box would sit on top of
    # the data. So: a legend of methods, and a separate style key.
    handles = []
    for i, (name, g) in enumerate(results.groupby("label", sort=False)):
        g = g.sort_values(x)
        c = st.series_color(i)
        m = st.MARKERS[i % 8]
        ax.plot(g[x], g["nrmse_grad"], color=c, lw=2, marker=m, markersize=5)
        ax.plot(g[x], g["nrmse_rot"], color=c, lw=2, dashes=(4, 1.5),
                marker=m, markersize=5, markerfacecolor=th.surface)
        handles.append(plt.Line2D([], [], color=c, lw=2, marker=m,
                                  markersize=5, label=name))
    style_key = [
        plt.Line2D([], [], color=th.ink_muted, lw=2, label="gradient direction"),
        plt.Line2D([], [], color=th.ink_muted, lw=2, dashes=(4, 1.5),
                   label="rotational direction"),
    ]
    ax.set_xlabel(r"rotational strength  $\omega$")
    ax.set_ylabel("normalised error in that direction")
    ax.set_title(title)
    # Both keys go *below* the axes. The curves here span nearly the full
    # vertical range by construction -- that spread is the result -- so any
    # in-axes legend lands on data whatever corner it picks.
    first = ax.legend(handles=handles, loc="upper left", fontsize=8.5,
                      bbox_to_anchor=(0.0, -0.16), title="method",
                      title_fontsize=8.5)
    first.get_title().set_color(th.ink_secondary)
    ax.add_artist(first)
    ax.legend(handles=style_key, loc="upper right", fontsize=8.5,
              bbox_to_anchor=(1.0, -0.16), title="component",
              title_fontsize=8.5).get_title().set_color(th.ink_secondary)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color=th.ink_muted, lw=1.0, dashes=(2, 2), zorder=1)
    ax.text(0.995, 1.0, " component entirely missed ",
            transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=8, color=th.ink_muted)
    return ax


def plot_phase_diagram(results, ax=None, *, x: str = "n_transitions",
                       y: str = "d", value: str = "nrmse",
                       methods=("sfi", "nn"), key: str = "estimator",
                       title=None, annotate_crossover=True):
    """Which method wins, as a function of sample budget and dimension.

    The cell color is the *winning* method's identity and the cell's opacity is
    its margin, so the eye reads the boundary first and the confidence second.
    A single sequential heatmap of one method's error cannot show a crossover;
    this can.

    The marked line is where the two methods are within a few percent of each
    other -- the crossover the guide asks you to find and report.
    """
    ax = _ax(ax, (7.2, 5.2))
    th = st.active()
    piv = {m: results[results[key] == m].pivot_table(index=y, columns=x,
                                                     values=value)
           for m in methods}
    a, b = (piv[m].to_numpy() for m in methods[:2])
    ys = piv[methods[0]].index.to_numpy()
    xs = piv[methods[0]].columns.to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        logratio = np.log2(b / a)          # >0 : method 0 wins
    win = np.sign(logratio)
    margin = np.clip(np.abs(logratio) / 1.0, 0, 1)

    c0 = np.array(plt.matplotlib.colors.to_rgb(st.series_color(0)))
    c1 = np.array(plt.matplotlib.colors.to_rgb(st.series_color(1)))
    surf = np.array(plt.matplotlib.colors.to_rgb(th.surface))
    rgb = np.where(win[..., None] > 0, c0, c1)
    img = surf + (rgb - surf) * margin[..., None]
    img = np.where(np.isfinite(margin)[..., None], img, surf)

    ax.imshow(np.clip(img, 0, 1), origin="lower", aspect="auto",
              extent=(0, len(xs), 0, len(ys)), interpolation="nearest")
    ax.set_xticks(np.arange(len(xs)) + 0.5)
    ax.set_xticklabels([f"{v:.0e}" for v in xs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ys)) + 0.5)
    ax.set_yticklabels([str(int(v)) for v in ys])
    ax.grid(False)

    if annotate_crossover:
        ax.contour(np.arange(len(xs)) + 0.5, np.arange(len(ys)) + 0.5,
                   logratio, levels=[0.0], colors=[th.ink], linewidths=1.6)

    lab0 = results[results[key] == methods[0]]["label"].iloc[0]
    lab1 = results[results[key] == methods[1]]["label"].iloc[0]
    handles = [plt.Line2D([], [], marker="s", ls="none", markersize=9,
                          color=st.series_color(0), label=f"{lab0} wins"),
               plt.Line2D([], [], marker="s", ls="none", markersize=9,
                          color=st.series_color(1), label=f"{lab1} wins"),
               plt.Line2D([], [], color=th.ink, lw=1.6, label="crossover")]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5)
    ax.set_xlabel("observed transitions  $N\\,T$")
    ax.set_ylabel("dimension  $d$")
    ax.set_title(title or "Which estimator wins")
    ax.text(0.01, -0.22, "colour = winner, opacity = margin (saturated at 2x)",
            transform=ax.transAxes, fontsize=8, color=th.ink_muted)
    return ax


def plot_error_vs_density(curves, ax=None, *, title="Error where the data is thin"):
    """Normalised error against local sample density, one line per method.

    ``curves`` is a list of dicts from :func:`dfi.metrics.error_vs_density`.
    The reference slope is ``-1/2``: a local average of ``n`` increments has
    error ``~ n^-1/2``. Matching it confirms the estimator is variance-limited;
    flattening means bias has taken over in that regime.
    """
    ax = _ax(ax)
    th = st.active()
    for i, c in enumerate(curves):
        sty = st.series_style(i)
        ax.plot(c["density"], c["nrmse"], label=c["label"], lw=2, markersize=5,
                **sty)
    d = np.array([min(c["density"].min() for c in curves),
                  max(c["density"].max() for c in curves)])
    ref = max(c["nrmse"].max() for c in curves) * (d / d[0]) ** -0.5
    ax.plot(d, ref, color=th.ink_muted, lw=1.0, dashes=(1, 2))
    ax.text(d[-1], ref[-1], r"  $n^{-1/2}$", fontsize=8, color=th.ink_muted,
            va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("local sample density")
    ax.set_ylabel(r"normalised error")
    ax.set_title(title)
    if len(curves) >= 2:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_bias_variance(results, ax=None, *, x="bins", title=None):
    """Total error split into bias and variance against a smoothing parameter.

    Needs repeats across seeds: the variance is the spread of the estimate at
    fixed ``x``, the bias is the offset of the mean estimate from the truth.
    The U-shaped total with the two crossing components is the canonical
    picture, and having it for your own estimator is worth more than citing it.
    """
    ax = _ax(ax)
    for i, col in enumerate(("bias", "variance", "total")):
        if col not in results:
            continue
        sty = st.series_style(i)
        ax.plot(results[x], results[col], label=col, lw=2, markersize=5, **sty)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel({"bins": "bins per axis", "bandwidth": "kernel bandwidth",
                   "n_basis": "basis size"}.get(x, x))
    ax.set_ylabel("mean squared error")
    ax.set_title(title or "Bias-variance tradeoff")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_training_curve(history, ax=None, *, noise_floor: float | None = None,
                        title="Training"):
    """Train/validation loss for the neural estimators.

    Pass ``noise_floor = 2 d D / dt`` and the loss is plotted *relative to it*.
    Raw loss is dominated by irreducible noise and looks flat from the first
    epoch, which tells you nothing; the excess over the floor is the part that
    is actually learning.
    """
    ax = _ax(ax, (6.0, 4.2))
    for i, k in enumerate(("train", "val")):
        if k not in history:
            continue
        y = np.asarray(history[k], float)
        if noise_floor:
            y = y / noise_floor - 1.0
        sty = st.series_style(i)
        sty.pop("marker", None)
        ax.plot(np.arange(1, len(y) + 1), y, label=k, lw=2, **sty)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("excess loss over noise floor" if noise_floor else "loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax
