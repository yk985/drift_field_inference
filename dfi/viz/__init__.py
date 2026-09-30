"""Plotting and animation, all sharing one design system (see style.py)."""
from . import style
from .style import use_style, active, series_style, series_color, figure_caption
from .fields import (plot_drift, plot_potential, plot_density, plot_helmholtz,
                     plot_field_card, plot_field_1d, field_on_grid, to_image,
                     add_colorbar)
from .trajectories import (plot_trajectories, plot_snapshot, plot_sampling_density,
                          plot_msd, plot_increment_check, plot_trajectories_1d)
from .animate import (animate_diffusion, animate_omega_sweep,
                      animate_relaxation, save_animation)
from .lic import line_integral_convolution

__all__ = [
    "style", "use_style", "active", "series_style", "series_color",
    "figure_caption", "plot_drift", "plot_potential", "plot_density",
    "plot_helmholtz", "plot_field_card", "plot_field_1d", "field_on_grid",
    "to_image", "add_colorbar", "plot_trajectories", "plot_snapshot",
    "plot_sampling_density", "plot_msd", "plot_increment_check",
    "plot_trajectories_1d", "animate_diffusion", "animate_omega_sweep",
    "animate_relaxation", "save_animation", "line_integral_convolution",
]
