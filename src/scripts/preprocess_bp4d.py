"""Preprocess BP4D sequences into aligned face tensors.

Only processes frames that have AU coding, as listed in bp4d_index.parquet
(built by build_bp4d_index.py). Saves one HDF5 file per task with all subjects
grouped inside:

    Preprocessed/T1.h5
    ├── F001/
    │   ├── faces    (N, 3, 112, 112) float32
    │   └── indices  (N,) int32  — 0-based image frame numbers
    ├── F002/
    │   └── ...
    └── M001/
        └── ...

indices are 0-based image frame numbers (AU frame number - 1).
Frames where no face is detected are skipped and logged to <output_dir>/failed.txt.

Usage:
    uv run python src/scripts/preprocess_bp4d.py
    uv run python src/scripts/preprocess_bp4d.py --force
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from msc.constants import DATA_DIR
from msc.face_embeddings.preprocessor import FacePreprocessor

INPUT_DIR = Path.home() / "projects/semedit/data/BP4D/Sequences"
OUTPUT_DIR = DATA_DIR / "BP4D/Preprocessed"
INDEX_PATH = DATA_DIR / "BP4D/bp4d_index.parquet"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Preprocess BP4D face sequences.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-process tasks that already have a saved .h5 file",
    )
    p.add_argument(
        "--task", type=str, default=None, help="Only process a single task (e.g. T1)"
    )
    return p.parse_args()


def coded_frame_paths(
    index: pd.DataFrame, root: Path
) -> dict[str, dict[str, list[Path]]]:
    """Return {task: {subject: [img_path, ...]}} for all coded frames in the index.

    AU coding uses 1-based frame numbers; image filenames are 0-based (frame N-1).
    """
    tasks: dict[str, dict[str, list[Path]]] = defaultdict(dict)
    for (subject, task), group in index.groupby(["subject", "task"]):
        paths = []
        for au_frame in group["frame"]:
            img_frame = int(au_frame) - 1  # AU 1-based → image 0-based
            img_path = root / subject / task / f"{img_frame:04d}.jpg"
            # annoyingly some have 3-digit or 2-digit frame numbers instead of 4 :(
            if not img_path.exists():
                img_path = root / subject / task / f"{img_frame:03d}.jpg"
            if not img_path.exists():
                img_path = root / subject / task / f"{img_frame:02d}.jpg"
            if img_path.exists():
                paths.append(img_path)
        if paths:
            tasks[task][subject] = sorted(paths)
    return dict(tasks)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return

    if not INDEX_PATH.exists():
        logger.error(f"Index not found: {INDEX_PATH} — run build_bp4d_index.py first")
        return

    index = pd.read_parquet(INDEX_PATH)
    tasks = coded_frame_paths(index, INPUT_DIR)
    if not tasks:
        logger.error("No coded frames found")
        return

    if args.task:
        if args.task not in tasks:
            logger.error(f"Task {args.task} not found in index")
            return
        tasks = {args.task: tasks[args.task]}

    total_frames = sum(
        len(frames) for subjects in tasks.values() for frames in subjects.values()
    )
    logger.info(f"Found {len(tasks)} tasks, {total_frames} coded frames total")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    preprocessor = FacePreprocessor()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    tasks_processed = tasks_skipped = 0

    for task, subjects in tqdm(tasks.items(), desc="Tasks", unit="task"):
        out_path = OUTPUT_DIR / f"{task}.h5"

        if not args.force and out_path.exists():
            tasks_skipped += 1
            continue

        with h5py.File(out_path, "w") as hf:
            for subject, frames in tqdm(
                subjects.items(), desc=task, unit="subject", leave=False
            ):
                faces: list[np.ndarray] = []
                indices: list[int] = []
                for img_path in tqdm(frames, desc=subject, unit="frame", leave=False):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Could not read {img_path}")
                        failed.append(str(img_path))
                        continue

                    try:
                        faces.append(preprocessor.preprocess(img).numpy())
                        indices.append(int(img_path.stem))
                    except ValueError:
                        failed.append(str(img_path))

                if not faces:
                    logger.warning(f"No faces detected in {subject}/{task} — skipping")
                    continue

                grp = hf.create_group(subject)
                grp.create_dataset("faces", data=np.stack(faces), compression="lzf")
                grp.create_dataset("indices", data=np.array(indices, dtype=np.int32))

        tasks_processed += 1

    logger.info(
        f"Done — tasks processed: {tasks_processed}  skipped (exists): {tasks_skipped} "
        f"failed frames: {len(failed)}"
    )

    if failed:
        fail_log = OUTPUT_DIR / "failed.txt"
        fail_log.write_text("\n".join(failed) + "\n")
        logger.info(f"Failed frames written to {fail_log}")


if __name__ == "__main__":
    main()
