"""Stitch denoising-step snapshots into a two-row strip figure."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

from msc.constants import PROJECT_ROOT
from msc.plot_utils import set_plotting_style

BASE = PROJECT_ROOT / "output_data/viz_stable_diffusion_details"
TOP_STEPS = [1, 200, 400]
BOT_STEPS = [600, 800, 1000]
MAX_SIZE = 512
DPI = 300


def stitch(
    steps: list[int], fig: plt.Figure, ax: plt.Axes
) -> tuple[np.ndarray, int, int]:
    """Load, resize to axes pixel budget, and stitch images side by side.

    Returns:
        Tuple of (stitched image array, per-image width, per-image height).
    """
    fig_w, fig_h = fig.get_size_inches()
    bbox = ax.get_position()
    per_w = min(int(bbox.width * fig_w * DPI / len(steps)), MAX_SIZE)
    per_h = min(int(bbox.height * fig_h * DPI), MAX_SIZE)

    imgs = []
    for s in steps:
        img = Image.open(BASE / f"step_{s:04d}.png").convert("RGB")
        img.thumbnail((per_w, per_h), Image.LANCZOS)
        imgs.append(img)

    w, h = imgs[0].size
    canvas = Image.new("RGB", (w * len(imgs), h), "white")
    for i, img in enumerate(imgs):
        canvas.paste(img, (i * w, 0))
    return np.array(canvas), w, h


def label_axis(ax: plt.Axes, steps: list[int], w: int, top: bool) -> None:
    """Configure tick labels and spine visibility for a strip axis."""
    ax.set_yticks([])
    ax.set_xticks([w * (i + 0.5) for i in range(len(steps))])
    ax.set_xticklabels([str(s) for s in steps])
    ax.tick_params(axis="x", length=0, pad=4)
    if top:
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    """Build and save the two-row denoising strip figure."""
    set_plotting_style()

    n = len(TOP_STEPS)
    cell = 3.0
    fig = plt.figure(figsize=(n * cell, 2 * cell + 0.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 1, figure=fig, hspace=-0.135)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    top_arr, w_top, _ = stitch(TOP_STEPS, fig, ax_top)
    bot_arr, w_bot, _ = stitch(BOT_STEPS, fig, ax_bot)

    ax_top.imshow(top_arr)
    label_axis(ax_top, TOP_STEPS, w_top, top=True)

    ax_bot.imshow(bot_arr)
    label_axis(ax_bot, BOT_STEPS, w_bot, top=False)

    out = BASE / "strength-strip.pdf"
    plt.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
