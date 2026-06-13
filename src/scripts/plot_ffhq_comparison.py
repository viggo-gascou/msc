"""Plot a single-source PyFeat vs LibreFace vs FineFace comparison figure for FFHQ.

Three rows (one per model), five columns (source + four expressions).
Row labels on the left, expression labels on top.

Expected folder structure for hardcoded images::

    comparisons/04278/
        source.png
        fineface_smile.png
        fineface_surprise.png
        fineface_anger.png
        fineface_sadness.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from msc.constants import FIGURES_DIR

_COMPARISONS_DIR = Path("comparisons/04278")
_PYFEAT_DIR = Path("eval/sd21_pyfeat_pretrain_ffhq_source")
_LIBREFACE_DIR = Path("eval/sd21_libreface_pretrain_ffhq_source")
_EMOTION_ORDER = ["smile", "surprise", "anger", "sadness"]
_IMG_SIZE = 256


def main() -> None:
    """Generate PyFeat vs LibreFace vs FineFace comparison figure for a FFHQ source."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    source_path = str(_COMPARISONS_DIR / "source.png")

    py_meta = pd.read_parquet(_PYFEAT_DIR / "metadata.parquet")
    lib_meta = pd.read_parquet(_LIBREFACE_DIR / "metadata.parquet")

    py_id = py_meta[py_meta["source_path"] == source_path]["sample_id"].iloc[0]
    lib_id = lib_meta[lib_meta["source_path"] == source_path]["sample_id"].iloc[0]

    has_fineface = all(
        (_COMPARISONS_DIR / f"fineface_{expr}.png").exists() for expr in _EMOTION_ORDER
    )
    n_rows = 3 if has_fineface else 2
    col_labels = ["Source"] + [e.capitalize() for e in _EMOTION_ORDER]

    fig, axes = plt.subplots(n_rows, 5, figsize=(5 * 1.6, n_rows * 1.6))

    for col_idx, lbl in enumerate(col_labels):
        axes[0, col_idx].set_title(lbl, fontsize=7, pad=3)

    for row_idx, (meta, sample_id, row_lbl) in enumerate(
        [(py_meta, py_id, "PyFeat (Ours)"), (lib_meta, lib_id, "LibreFace (Ours)")]
    ):
        _show(axes[row_idx, 0], _load_img(source_path))
        _row_label(axes[row_idx, 0], row_lbl)
        for col_idx, expr in enumerate(_EMOTION_ORDER, start=1):
            row = meta[
                (meta["sample_id"] == sample_id) & (meta["config_name"] == expr)
            ].iloc[0]
            _show(axes[row_idx, col_idx], _load_img(row["generated_path"]))

    if has_fineface:
        _show(axes[2, 0], _load_img(source_path))
        _row_label(axes[2, 0], "FineFace")
        for col_idx, expr in enumerate(_EMOTION_ORDER, start=1):
            _show(
                axes[2, col_idx],
                _load_img(str(_COMPARISONS_DIR / f"fineface_{expr}.png")),
            )

    fig.tight_layout(pad=0.3)

    out = args.out or FIGURES_DIR / "qualitative_ffhq_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved to {out}")


def _row_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08, 0.5, label, transform=ax.transAxes, fontsize=7, va="center", ha="right"
    )


def _load_img(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img.resize((_IMG_SIZE, _IMG_SIZE), Image.LANCZOS))


def _show(ax: plt.Axes, img: np.ndarray) -> None:
    ax.imshow(img)
    ax.axis("off")


if __name__ == "__main__":
    main()
