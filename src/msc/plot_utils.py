"""Shared matplotlib/seaborn setup for paper-quality figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

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


def load_thumb(
    path: Path, ax: plt.Axes, dpi: int = 300, max_size: int = 512
) -> np.ndarray:
    """Load an image resized to the pixel budget of the given axes.

    Args:
        path:
          Path to the image file.
        ax:
          The axes the image will be displayed in.
        dpi (optional):
          Output DPI. Defaults to 300.
        max_size (optional):
          Hard cap on either dimension in pixels. Defaults to 512.

    Returns:
        Image as a uint8 numpy array sized to fit the axes pixel budget.
    """
    fig = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    bbox = ax.get_position()
    ax_w_px = min(int(bbox.width * fig_w * dpi), max_size)
    ax_h_px = min(int(bbox.height * fig_h * dpi), max_size)
    img = Image.open(path).convert("RGB")
    img.thumbnail((ax_w_px, ax_h_px), Image.LANCZOS)
    return np.array(img)


def set_plotting_style() -> None:
    """Apply consistent paper-quality styling to all subsequent plots.

    Calls sns.set_theme first (whitegrid base), then overrides font and
    axis settings via rcParams so they take precedence over seaborn defaults.
    """
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(PLOTTING_STYLE)
