# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "libreface",
#   "loguru",
#   "pandas",
#   "pyarrow",
#   "torch",
#   "tqdm",
#   "opencv-python-headless==4.10.0.84",
#   "numpy<2",
# ]
# ///
"""Run LibreFace AU detection + intensity estimation on BP4D, one parquet per task.

Dependencies are declared inline — just run with uv, no project setup needed:
    uv run libreface_bp4d.py --sequences-dir /path/to/BP4D/Sequences
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import libreface
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

HPC_SEQUENCES_DIR = Path("/home/semedit/data/BP4D/Sequences")
DEFAULT_SEQUENCES_DIR = (
    HPC_SEQUENCES_DIR
    if HPC_SEQUENCES_DIR.exists()
    else Path("data/BP4D/Sample/Sequences")
)
DEFAULT_OUT_DIR = Path("data/BP4D/libreface")


def main() -> None:
    """Parse arguments and run LibreFace AU extraction on BP4D."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=DEFAULT_SEQUENCES_DIR,
        help=f"BP4D Sequences directory (default: {DEFAULT_SEQUENCES_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for per-task parquet files (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="LibreFace inference batch size (default: 32).",
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Process a single task only (e.g. T1)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Images per accumulation chunk before concatenating (default: 500).",
    )
    args = parser.parse_args()
    run(
        sequences_dir=args.sequences_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        task_filter=args.task,
    )


def run(
    sequences_dir: Path,
    output_dir: Path,
    batch_size: int,
    chunk_size: int,
    task_filter: str | None,
) -> None:
    """Run LibreFace on BP4D sequences and write one parquet per task.

    Args:
        sequences_dir:
          Root BP4D Sequences directory (Subject/Task/*.jpg structure).
        output_dir:
          Directory to write per-task parquet files.
        batch_size:
          Number of images per LibreFace forward pass.
        chunk_size:
          Number of images per accumulation chunk before concatenating results.
        task_filter:
          If set, only process this task (e.g. 'T1').
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")
    logger.info(f"Sequences: {sequences_dir}")
    logger.info(f"Output:    {output_dir}")

    by_task = discover_tasks(sequences_dir=sequences_dir)
    if task_filter is not None:
        by_task = {k: v for k, v in by_task.items() if k == task_filter}
        if not by_task:
            logger.error(f"Task '{task_filter}' not found in {sequences_dir}")
            return

    logger.info(f"Found {len(by_task)} tasks: {sorted(by_task)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for task, entries in sorted(by_task.items()):
        process_task(
            task=task,
            entries=entries,
            output_dir=output_dir,
            batch_size=batch_size,
            chunk_size=chunk_size,
            device=device,
        )

    logger.info("All tasks complete.")


def discover_tasks(sequences_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """Return {task: [(subject, image_path), ...]} for all BP4D sequences.

    Args:
        sequences_dir:
          Root BP4D Sequences directory.

    Returns:
        Dict mapping task name to list of (subject, image_path) pairs.
    """
    by_task: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for subject_dir in sorted(sequences_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        for task_dir in sorted(subject_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for img in sorted(task_dir.glob("*.jpg")):
                by_task[task_dir.name].append((subject_dir.name, img))
    return dict(by_task)


def process_task(
    task: str,
    entries: list[tuple[str, Path]],
    output_dir: Path,
    batch_size: int,
    chunk_size: int,
    device: str,
) -> None:
    """Run LibreFace on one BP4D task and write a parquet file.

    Args:
        task:
          Task name (e.g. 'T1').
        entries:
          List of (subject, image_path) pairs for this task.
        output_dir:
          Directory to write the output parquet.
        batch_size:
          Number of images per LibreFace forward pass.
        chunk_size:
          Number of images per accumulation chunk before concatenating results.
        device:
          PyTorch device string.
    """
    out_path = output_dir / f"libreface_bp4d_{task}.parquet"
    subjects = [s for s, _ in entries]
    image_paths = [str(p) for _, p in entries]
    n = len(image_paths)
    chunks = [image_paths[i : i + chunk_size] for i in range(0, n, chunk_size)]
    subject_chunks = [subjects[i : i + chunk_size] for i in range(0, n, chunk_size)]

    logger.info(f"[{task}] {n} frames / {len(set(subjects))} subjects -> {out_path}")
    t0 = time.perf_counter()
    rows = []

    for chunk_imgs, chunk_subjects in tqdm(
        zip(chunks, subject_chunks), total=len(chunks), desc=task, unit="chunk"
    ):
        det_df, intensity_df = libreface.get_au_intensities_and_detect_aus_video(
            chunk_imgs, device=device, batch_size=batch_size, num_workers=0
        )
        rows.append(
            pd.concat(
                [
                    pd.Series(chunk_subjects, name="subject"),
                    pd.Series(chunk_imgs, name="image"),
                    det_df.reset_index(drop=True),
                    intensity_df.reset_index(drop=True),
                ],
                axis=1,
            )
        )

    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"  Done: {elapsed:.1f}s  ({elapsed / n * 1000:.0f}ms/img)"
        f"  ->  {len(combined)} rows"
    )


if __name__ == "__main__":
    main()
