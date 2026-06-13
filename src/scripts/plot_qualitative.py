"""Generate a qualitative figure from one or more eval run directories.

For emotion mode, each row is a source image and the columns show the generated
expressions. For paired mode, columns are source, generated, and target.

When multiple run directories are given, their outputs are shown side-by-side for
the same samples, grouped by expression.

Requires save_source_images.py to have been run first to populate source_path
(and target_path) in metadata.parquet.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from msc.constants import FIGURES_DIR

_EMOTION_ORDER = ["smile", "surprise", "anger", "sadness"]
_IMG_SIZE = 256
_RUN_LABELS: dict[str, str] = {"pyfeat": "PyFeat", "libreface": "LibreFace"}


def main() -> None:
    """Generate qualitative figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--n", type=int, default=5, help="Number of samples to show.")
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="Explicit sample_ids to show (overrides --n).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = [_load_run(d) for d in args.run_dirs]
    eval_mode = runs[0]["eval_mode"]

    if len(runs) > 1:
        # Match across runs by source_path since dataset lengths differ between
        # detectors.
        shared_paths = set(runs[0]["meta"]["source_path"].dropna().unique())
        for run in runs[1:]:
            shared_paths &= set(run["meta"]["source_path"].dropna().unique())
        shared_paths_sorted = sorted(shared_paths)
        if len(shared_paths_sorted) < args.n:
            print(
                f"Warning: only {len(shared_paths_sorted)} shared source images across "
                "runs"
            )
        rng = np.random.default_rng(args.seed)
        source_paths = sorted(
            rng.choice(
                shared_paths_sorted,
                size=min(args.n, len(shared_paths_sorted)),
                replace=False,
            ).tolist()
        )
        fig = (
            _plot_emotion_by_path(runs, source_paths)
            if eval_mode == "emotion"
            else _plot_paired_by_path(runs, source_paths)
        )
    else:
        all_ids = sorted(runs[0]["meta"]["sample_id"].unique().tolist())
        if args.samples is not None:
            sample_ids = [s for s in args.samples if s in all_ids]
        else:
            rng = np.random.default_rng(args.seed)
            sample_ids = sorted(
                rng.choice(
                    all_ids, size=min(args.n, len(all_ids)), replace=False
                ).tolist()
            )
        fig = (
            _plot_emotion(runs[0], sample_ids)
            if eval_mode == "emotion"
            else _plot_paired(runs[0], sample_ids)
        )

    out = args.out or FIGURES_DIR / f"qualitative_{eval_mode}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved to {out}")


def _load_run(run_dir: Path) -> dict:
    """Load metadata and infer run label from directory name.

    Returns:
        The loaded run as a dictionary.
    """
    meta = pd.read_parquet(run_dir / "metadata.parquet")
    label = run_dir.name
    for key, val in _RUN_LABELS.items():
        if key in run_dir.name:
            label = val
            break
    eval_mode = "paired" if "paired" in meta["config_name"].values else "emotion"
    return {"dir": run_dir, "meta": meta, "label": label, "eval_mode": eval_mode}


def _load_img(path: str) -> np.ndarray:
    """Load and resize an image to _IMG_SIZE, returning an HWC uint8 array.

    Returns:
        The loaded image as an HWC uint8 array.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img.resize((_IMG_SIZE, _IMG_SIZE), Image.LANCZOS))


def _plot_emotion(run: dict, sample_ids: list[int]) -> plt.Figure:
    """Single-run emotion grid: rows = samples, cols = [source | expressions].

    Returns:
        The generated figure.
    """
    n_cols = 1 + len(_EMOTION_ORDER)
    n_rows = len(sample_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_labels = ["Source"] + [e.capitalize() for e in _EMOTION_ORDER]
    for col_idx, lbl in enumerate(col_labels):
        axes[0, col_idx].set_title(lbl, fontsize=7, pad=3)
    for row_idx, sample_id in enumerate(sample_ids):
        src_row = run["meta"][run["meta"]["sample_id"] == sample_id].iloc[0]
        _show(axes[row_idx, 0], _load_img(src_row["source_path"]))
        for col_idx, expr in enumerate(_EMOTION_ORDER, start=1):
            gen_row = run["meta"][
                (run["meta"]["sample_id"] == sample_id)
                & (run["meta"]["config_name"] == expr)
            ].iloc[0]
            _show(axes[row_idx, col_idx], _load_img(gen_row["generated_path"]))
    fig.tight_layout(pad=0.3)
    return fig


def _plot_emotion_by_path(runs: list[dict], source_paths: list[str]) -> plt.Figure:
    """Multi-run emotion grid matched on source_path.

    Returns:
        The generated figure.
    """
    n_runs = len(runs)
    n_cols = 1 + n_runs * len(_EMOTION_ORDER)
    n_rows = len(source_paths)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_labels = ["Source"]
    for expr in _EMOTION_ORDER:
        for run in runs:
            col_labels.append(f"{expr.capitalize()}\n{run['label']}")
    for col_idx, lbl in enumerate(col_labels):
        axes[0, col_idx].set_title(lbl, fontsize=7, pad=3)
    for row_idx, src_path in enumerate(source_paths):
        _show(axes[row_idx, 0], _load_img(src_path))
        col_idx = 1
        for expr in _EMOTION_ORDER:
            for run in runs:
                gen_row = run["meta"][
                    (run["meta"]["source_path"] == src_path)
                    & (run["meta"]["config_name"] == expr)
                ].iloc[0]
                _show(axes[row_idx, col_idx], _load_img(gen_row["generated_path"]))
                col_idx += 1
    fig.tight_layout(pad=0.3)
    return fig


def _plot_paired(run: dict, sample_ids: list[int]) -> plt.Figure:
    """Single-run paired grid: rows = samples, cols = [source | generated | target].

    Returns:
        The generated figure.
    """
    has_target = "target_path" in run["meta"].columns
    n_cols = 2 + (1 if has_target else 0)
    n_rows = len(sample_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_labels = ["Source", "Generated"] + (["Target"] if has_target else [])
    for col_idx, lbl in enumerate(col_labels):
        axes[0, col_idx].set_title(lbl, fontsize=7, pad=3)
    for row_idx, sample_id in enumerate(sample_ids):
        row = run["meta"][run["meta"]["sample_id"] == sample_id].iloc[0]
        _show(axes[row_idx, 0], _load_img(row["source_path"]))
        _show(axes[row_idx, 1], _load_img(row["generated_path"]))
        if has_target:
            _show(axes[row_idx, 2], _load_img(row["target_path"]))
    fig.tight_layout(pad=0.3)
    return fig


def _plot_paired_by_path(runs: list[dict], source_paths: list[str]) -> plt.Figure:
    """Multi-run paired grid matched on source_path.

    Returns:
        The generated figure.
    """
    has_target = "target_path" in runs[0]["meta"].columns
    n_cols = 1 + len(runs) + (1 if has_target else 0)
    n_rows = len(source_paths)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_labels = (
        ["Source"]
        + [f"Generated\n{r['label']}" for r in runs]
        + (["Target"] if has_target else [])
    )
    for col_idx, lbl in enumerate(col_labels):
        axes[0, col_idx].set_title(lbl, fontsize=7, pad=3)
    for row_idx, src_path in enumerate(source_paths):
        _show(axes[row_idx, 0], _load_img(src_path))
        for run_idx, run in enumerate(runs):
            gen_row = run["meta"][run["meta"]["source_path"] == src_path].iloc[0]
            _show(axes[row_idx, 1 + run_idx], _load_img(gen_row["generated_path"]))
        if has_target:
            tgt_path = runs[0]["meta"][runs[0]["meta"]["source_path"] == src_path].iloc[
                0
            ]["target_path"]
            _show(axes[row_idx, -1], _load_img(tgt_path))
    fig.tight_layout(pad=0.3)
    return fig


def _show(ax: plt.Axes, img: np.ndarray) -> None:
    ax.imshow(img)
    ax.axis("off")


if __name__ == "__main__":
    main()
