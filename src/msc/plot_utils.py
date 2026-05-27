"""Shared matplotlib/seaborn setup for paper-quality figures."""

import matplotlib.pyplot as plt
import seaborn as sns

PLOTTING_STYLE: dict[str, bool | str | int | list[str]] = {
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def set_plotting_style() -> None:
    """Apply consistent paper-quality styling to all subsequent plots.

    Calls sns.set_theme first (whitegrid base), then overrides font and
    axis settings via rcParams so they take precedence over seaborn defaults.
    """
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(PLOTTING_STYLE)
