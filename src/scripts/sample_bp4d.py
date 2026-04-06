"""Create a small sample of BP4D data for local development and testing.

Copies 2 raw images per subject from T1, and writes subset HDF5 files for
the preprocessed crops and embeddings, so the full dataset is not needed
to work on the data pipeline locally.

Output structure:
    Sample/
    ├── Sequences/
    │   └── {subject}/T1/   (2 raw JPEGs per subject)
    ├── Preprocessed/T1.h5  (2 faces per subject)
    └── Embeddings/T1.h5    (2 embeddings per subject)

Usage:
    uv run python src/scripts/sample_bp4d.py
    uv run python src/scripts/sample_bp4d.py --task T2 --n 5
"""

import argparse
import shutil

import h5py
import pandas as pd
from loguru import logger
from tqdm import tqdm

from msc.constants import (
    BP4D_EMBEDDINGS_DIR,
    BP4D_INDEX_PATH,
    BP4D_PREPROCESSED_DIR,
    BP4D_SEQUENCES_DIR,
    DATA_DIR,
)
from msc.data.bp4d import load_index, resolve_frame_path

SAMPLE_DIR = DATA_DIR / "BP4D/Sample"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Create a small BP4D sample dataset.")
    p.add_argument(
        "--task", type=str, default="T1", help="Task to sample (default: T1)"
    )
    p.add_argument("--n", type=int, default=2, help="Frames per subject (default: 2)")
    return p.parse_args()


def main() -> None:
    """Create a small BP4D sample dataset."""
    args = parse_args()
    task = args.task
    n = args.n

    index = load_index()
    task_index = index[index["task"] == task]
    subjects = sorted(task_index["subject"].unique())
    sampled_rows: list[pd.DataFrame] = []

    logger.info(f"Sampling {n} frames from {len(subjects)} subjects in {task}")

    seq_out = SAMPLE_DIR / "Sequences"
    pre_out = SAMPLE_DIR / "Preprocessed"
    emb_out = SAMPLE_DIR / "Embeddings"
    for d in (seq_out, pre_out, emb_out):
        d.mkdir(parents=True, exist_ok=True)

    pre_src = BP4D_PREPROCESSED_DIR / f"{task}.h5"
    emb_src = BP4D_EMBEDDINGS_DIR / f"{task}.h5"

    if not pre_src.exists():
        logger.error(f"Preprocessed file not found: {pre_src}")
        return
    if not emb_src.exists():
        logger.error(f"Embeddings file not found: {emb_src}")
        return

    with (
        h5py.File(pre_src, "r") as pre_f,
        h5py.File(emb_src, "r") as emb_f,
        h5py.File(pre_out / f"{task}.h5", "w") as pre_dst,
        h5py.File(emb_out / f"{task}.h5", "w") as emb_dst,
    ):
        for subject in tqdm(subjects, desc="Subjects"):
            if subject not in pre_f or subject not in emb_f:
                logger.warning(f"{subject} missing from HDF5 — skipping")
                continue

            frames = pre_f[subject]["indices"][:n]
            count = len(frames)

            # Collect matching index rows for the sampled frames (AU frame = img + 1)
            sampled_rows.append(
                task_index[
                    (task_index["subject"] == subject)
                    & (task_index["frame"].isin(frames + 1))
                ]
            )

            # Subset HDF5
            pre_grp = pre_dst.create_group(subject)
            pre_grp.create_dataset(
                "faces", data=pre_f[subject]["faces"][:count], compression="lzf"
            )
            pre_grp.create_dataset("indices", data=frames)

            emb_grp = emb_dst.create_group(subject)
            emb_grp.create_dataset(
                "arcface", data=emb_f[subject]["arcface"][:count], compression="lzf"
            )
            emb_grp.create_dataset(
                "adaface", data=emb_f[subject]["adaface"][:count], compression="lzf"
            )

            # Copy raw JPEGs
            img_out_dir = seq_out / subject / task
            img_out_dir.mkdir(parents=True, exist_ok=True)
            for img_frame in frames:
                path = resolve_frame_path(
                    BP4D_SEQUENCES_DIR, subject, task, int(img_frame)
                )
                if path is None:
                    logger.warning(f"No image for {subject}/{task}/{img_frame}")
                    continue
                shutil.copy2(path, img_out_dir / path.name)

    # Write filtered index containing only the sampled frames
    sample_index = pd.concat(sampled_rows, ignore_index=True)
    sample_index_path = SAMPLE_DIR / BP4D_INDEX_PATH.name
    sample_index.to_parquet(sample_index_path, index=False)
    logger.info(
        f"Sample index written to {sample_index_path} ({len(sample_index)} rows)"
    )

    logger.info(f"Sample written to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
