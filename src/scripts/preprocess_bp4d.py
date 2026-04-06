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
"""

import argparse

import cv2
import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm

from msc.constants import BP4D_PREPROCESSED_DIR, BP4D_SEQUENCES_DIR
from msc.data.bp4d import coded_frame_paths, load_index
from msc.face_embeddings.preprocessor import FacePreprocessor


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


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not BP4D_SEQUENCES_DIR.exists():
        logger.error(f"Input directory not found: {BP4D_SEQUENCES_DIR}")
        return

    try:
        index = load_index()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    tasks = coded_frame_paths(index, BP4D_SEQUENCES_DIR)
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
    logger.info(f"Output directory: {BP4D_PREPROCESSED_DIR}")

    preprocessor = FacePreprocessor()

    failed: list[str] = []
    tasks_processed = tasks_skipped = 0

    for task, subjects in tqdm(tasks.items(), desc="Tasks", unit="task"):
        out_path = BP4D_PREPROCESSED_DIR / f"{task}.h5"

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
        fail_log = BP4D_PREPROCESSED_DIR / "failed.txt"
        fail_log.write_text("\n".join(failed) + "\n")
        logger.info(f"Failed frames written to {fail_log}")


if __name__ == "__main__":
    main()
