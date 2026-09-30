"""Design tokens and matplotlib theming for the whole project.

Every figure in this project gets its colors from here, so that a drift field, a
trajectory bundle and a phase diagram all read as one system.

Color is assigned by *job*, never by taste:

- categorical (identity: which estimator?)  -> ``SERIES``, in fixed slot order
- sequential  (magnitude: |b|, density)     -> ``CMAP_SEQ``  (one hue, light->dark)
- diverging   (polarity: signed error)      -> ``CMAP_DIV``  (blue<->red, gray mid)
- status      (state: pass/fail)            -> ``STATUS``    (reserved, never a series)

The palette is a validated instance: the eight categorical slots clear the
colorblind-separation gates in slot order, and the first three clear them for
all-pairs forms (scatter, small multiples). Past three series, secondary
encoding (marker shape + direct labels) does the work, which is why
``series_style`` hands you a marker along with the color.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

# Categorical slots, fixed order. Never cycle past slot 8 -- fold into "Other".
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

# Secondary encoding, paired with the slots above so identity is never
# carried by hue alone (matters for print, CVD, and thin lines).
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = [(None, None), (None, None), (4, 1.5), (1, 1.5),
          (6, 1.5, 1, 1.5), (3, 1, 3, 1), (5, 2), (1, 1)]

# Single-hue sequential ramp (blue 100 -> 700).
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
ORANGE_RAMP = ["#fbe0d3", "#f8c7ae", "#f5ae89", "#f19365", "#ee7c47", "#eb6834",
               "#d95926", "#bd4c1f", "#9d3f19", "#7d3214", "#5e250f"]

# Diverging pair: blue <-> red with a *neutral gray* midpoint (never a hue).
DIV_ANCHORS_LIGHT = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                     "#f6bdbc", "#e34948", "#8f2222"]
DIV_ANCHORS_DARK = ["#104281", "#3987e5", "#9ec5f4", "#383835",
                    "#f0a3a2", "#e66767", "#a12b2b"]

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}


@dataclass(frozen=True)
class Theme:
    """One resolved set of surface/ink/series tokens."""

    name: str
    surface: str
    surface_alt: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    series: list = _dc_field(default_factory=list)
    div_anchors: list = _dc_field(default_factory=list)


LIGHT = Theme(
    name="light", surface="#fcfcfb", surface_alt="#f0efec", ink="#0b0b0b",
    ink_secondary="#52514e", ink_muted="#8a8983", grid="#e3e2dd",
    series=SERIES_LIGHT, div_anchors=DIV_ANCHORS_LIGHT,
)
DARK = Theme(
    name="dark", surface="#1a1a19", surface_alt="#232322", ink="#ffffff",
    ink_secondary="#c3c2b7", ink_muted="#8a8983", grid="#333331",
    series=SERIES_DARK, div_anchors=DIV_ANCHORS_DARK,
)
THEMES = {"light": LIGHT, "dark": DARK}

_ACTIVE = {"theme": LIGHT}


def active() -> Theme:
    """The theme currently installed by :func:`use_style`."""
    return _ACTIVE["theme"]


# --------------------------------------------------------------------------
# Colormaps
# --------------------------------------------------------------------------

def _ramp(name, colors):
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def _fade_to(name, colors, surface):
    """Sequential ramp whose zero end melts into the chart surface.

    Used for densities and |b| heatmaps, where "near zero" should read as
    empty space rather than as a color.
    """
    return LinearSegmentedColormap.from_list(name, [surface] + list(colors), N=256)


CMAP_SEQ = _ramp("dfi_seq", BLUE_RAMP)
CMAP_SEQ_R = _ramp("dfi_seq_r", BLUE_RAMP[::-1])
CMAP_SEQ2 = _ramp("dfi_seq2", ORANGE_RAMP)
CMAP_DIV = _ramp("dfi_div", DIV_ANCHORS_LIGHT)
CMAP_DIV_DARK = _ramp("dfi_div_dark", DIV_ANCHORS_DARK)
CMAP_DENSITY_LIGHT = _fade_to("dfi_density_light", BLUE_RAMP[2:], LIGHT.surface)
CMAP_DENSITY_DARK = _fade_to("dfi_density_dark", BLUE_RAMP[2:], DARK.surface)

for _cm in (CMAP_SEQ, CMAP_SEQ_R, CMAP_SEQ2, CMAP_DIV, CMAP_DIV_DARK,
            CMAP_DENSITY_LIGHT, CMAP_DENSITY_DARK):
    try:
        mpl.colormaps.register(_cm, force=True)
    except Exception:  # pragma: no cover - older matplotlib
        pass


def cmap_density(theme: Theme | None = None):
    """Sequential ramp for densities, matched to the active surface."""
    theme = theme or active()
    return CMAP_DENSITY_DARK if theme.name == "dark" else CMAP_DENSITY_LIGHT


def cmap_div(theme: Theme | None = None):
    """Diverging ramp for signed quantities, matched to the active surface."""
    theme = theme or active()
    return CMAP_DIV_DARK if theme.name == "dark" else CMAP_DIV


def series_style(i: int, theme: Theme | None = None) -> dict:
    """Color + marker + dash for categorical slot ``i`` (identity encoding).

    Returns kwargs that can be splatted straight into ``ax.plot``. The marker
    and dash are the secondary channel: they keep series distinguishable in
    grayscale, in print, and for colorblind readers.
    """
    theme = theme or active()
    n = len(theme.series)
    if i >= n:
        raise IndexError(
            f"categorical slot {i} exceeds the {n}-slot palette; fold the extra "
            "series into 'Other' or facet into small multiples instead of "
            "cycling hues"
        )
    return {"color": theme.series[i], "marker": MARKERS[i], "dashes": DASHES[i]}


def series_color(i: int, theme: Theme | None = None) -> str:
    theme = theme or active()
    return theme.series[i % len(theme.series)]


def alpha(color, a: float):
    """``color`` at opacity ``a`` as an RGBA tuple."""
    return to_rgba(color, a)


# --------------------------------------------------------------------------
# rcParams
# --------------------------------------------------------------------------

def use_style(theme: str = "light", *, scale: float = 1.0) -> Theme:
    """Install the project matplotlib theme. Call once per script.

    ``scale`` multiplies every font size, for slides (1.3) vs. paper (1.0).
    """
    th = THEMES[theme]
    _ACTIVE["theme"] = th
    fs = 10 * scale
    mpl.rcParams.update({
        "figure.facecolor": th.surface,
        "figure.edgecolor": th.surface,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.facecolor": th.surface,
        "savefig.bbox": "tight",
        "axes.facecolor": th.surface,
        "axes.edgecolor": th.grid,
        "axes.labelcolor": th.ink_secondary,
        "axes.titlecolor": th.ink,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": fs * 1.15,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": fs,
        "axes.prop_cycle": mpl.cycler(color=th.series),
        "grid.color": th.grid,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": th.ink_muted,
        "ytick.color": th.ink_muted,
        "xtick.labelcolor": th.ink_secondary,
        "ytick.labelcolor": th.ink_secondary,
        "xtick.labelsize": fs * 0.9,
        "ytick.labelsize": fs * 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": fs * 0.9,
        "legend.labelcolor": th.ink_secondary,
        "legend.handlelength": 1.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "image.cmap": "dfi_seq",
        "font.size": fs,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "mathtext.fontset": "dejavusans",
        "animation.embed_limit": 200,
    })
    return th


def figure_caption(fig, text: str, *, y: float = -0.01):
    """One-line caption in secondary ink, bottom-left.

    Use it to state the regime (D, dt, N) so a figure stays readable when it is
    detached from the script that made it.
    """
    th = active()
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8.5,
             color=th.ink_muted, transform=fig.transFigure)


def hero(ax, value: str, label: str, *, sub: str = ""):
    """A stat tile: one big number, a label, an optional sub-line.

    Used when the answer is a single scalar (a coverage fraction, say) and a
    chart would just be an evasion.
    """
    th = active()
    ax.axis("off")
    ax.text(0.0, 0.62, value, fontsize=30, weight="semibold", color=th.ink,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(0.0, 0.28, label, fontsize=10, color=th.ink_secondary,
            ha="left", va="center", transform=ax.transAxes)
    if sub:
        ax.text(0.0, 0.10, sub, fontsize=8.5, color=th.ink_muted,
                ha="left", va="center", transform=ax.transAxes)


def despine(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
