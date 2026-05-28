# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "py-feat @ git+https://github.com/cosanlab/py-feat.git@c4f6364299ea2258ae1e73ed73c95750a18bff3e",
#   "pandas",
#   "pyarrow",
#   "torch",
#   "torchvision",
#   "pillow",
#   "tqdm",
#   "loguru",
# ]
# ///
"""Run py-feat AU detection on BP4D coded frames, one output parquet per task.

Run with:
    .venv-pyfeat/bin/python src/scripts/pyfeat_bp4d.py --task T1
"""

import argparse
import time
import typing as t
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from feat import Detector
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

TASKS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

DEFAULT_SEQUENCES_DIR = Path("/home/semedit/data/BP4D/Sequences")
DEFAULT_INDEX_PATH = Path("data/BP4D/bp4d_index.parquet")
DEFAULT_OUT_DIR = Path("data/BP4D/pyfeat")

_to_tensor = transforms.ToTensor()


def resolve_frame_path(
    root: Path, subject: str, task: str, img_frame: int
) -> Path | None:
    """Resolve the image path, handling BP4D's mixed 2/3/4-digit padding.

    Args:
        root:
            Root sequences directory.
        subject:
            Subject ID (e.g. 'F001').
        task:
            Task ID (e.g. 'T1').
        img_frame:
            Frame number (1-indexed as stored in the index).

    Returns:
        Resolved path if found, None if the frame is missing on disk.
    """
    for digits in (4, 3, 2):
        path = root / subject / task / f"{img_frame:0{digits}d}.jpg"
        if path.exists():
            return path
    return None


class BP4DImageDataset(Dataset):
    """Torch Dataset wrapping BP4D frame paths with subject/task metadata."""

    def __init__(self, records: list[tuple[Path, str, str, int]]) -> None:
        """Initialise with a list of (image_path, subject, task, au_frame) tuples."""
        self.records = records

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, t.Any]:
        """Load and return a single frame as a uint8 tensor with metadata.

        Returns:
            Dict with Image tensor and subject/task/frame metadata fields.
        """
        path, subject, task, au_frame = self.records[idx]
        img = Image.open(path).convert("RGB")
        tensor = (_to_tensor(img) * 255).byte()  # (C, H, W) uint8
        return {
            "Image": tensor,
            "FileName": str(path),
            "Frame": idx,
            "Scale": 1.0,
            "Padding": {"Left": 0, "Top": 0, "Right": 0, "Bottom": 0},
            "Subject": subject,
            "Task": task,
            "AUFrame": au_frame,
        }


def collate_fn(batch: list[dict[str, t.Any]]) -> dict[str, t.Any]:
    """Collate a list of sample dicts into a batched dict.

    Returns:
        Batched dict with a stacked Image tensor and lists for all metadata fields.
    """
    return {
        "Image": torch.stack([b["Image"] for b in batch]),
        "FileName": [b["FileName"] for b in batch],
        "Frame": [b["Frame"] for b in batch],
        "Scale": [b["Scale"] for b in batch],
        "Padding": [b["Padding"] for b in batch],
        "Subject": [b["Subject"] for b in batch],
        "Task": [b["Task"] for b in batch],
        "AUFrame": [b["AUFrame"] for b in batch],
    }


def main() -> None:
    """Parse arguments and delegate to run()."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences-dir", type=Path, default=DEFAULT_SEQUENCES_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--face-threshold", type=float, default=0.9)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    run(
        sequences_dir=args.sequences_dir,
        index_path=args.index,
        output_dir=args.output_dir,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        face_threshold=args.face_threshold,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
    )


def run(
    sequences_dir: Path,
    index_path: Path,
    output_dir: Path,
    task: str,
    batch_size: int,
    num_workers: int,
    face_threshold: float,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> None:
    """Run py-feat AU detection on one BP4D task and write results to parquet.

    Args:
        sequences_dir:
            Root directory containing BP4D frame JPEGs.
        index_path:
            Path to the BP4D index parquet with subject/task/frame columns.
        output_dir:
            Directory to write the output parquet file.
        task:
            Task ID to process (e.g. 'T1').
        batch_size:
            Number of frames per detection batch.
        num_workers:
            DataLoader worker processes.
        face_threshold:
            Minimum detection confidence to accept a face.
        shard_idx (optional):
            Zero-based shard index for parallel runs. Defaults to 0.
        num_shards (optional):
            Total number of shards. Defaults to 1.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    index = pd.read_parquet(index_path, columns=["subject", "task", "frame"])
    index = index[index["task"] == task].reset_index(drop=True)
    logger.info(f"Task {task}: {len(index)} coded frames")

    records: list[tuple[Path, str, str, int]] = []
    n_missing = 0
    for row in index.itertuples(index=False):
        path = resolve_frame_path(sequences_dir, row.subject, row.task, row.frame - 1)
        if path is None:
            n_missing += 1
        else:
            records.append((path, row.subject, row.task, row.frame))
    if n_missing:
        logger.warning(f"{n_missing} frames not found on disk -- skipped")

    records = records[shard_idx::num_shards]
    logger.info(f"{len(records)} frames to process (shard {shard_idx}/{num_shards})")

    stem = (
        f"bp4d_pyfeat_{task}"
        if num_shards == 1
        else f"bp4d_pyfeat_{task}_shard{shard_idx}"
    )
    output_path = output_dir / f"{stem}.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings(
        "ignore", message=".*serialized model", category=UserWarning, module=".*skops"
    )
    detector = Detector(
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        device=device,
    )

    dataset = BP4DImageDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    n = len(records)
    logger.info(f"{n} frames -> {output_path}")

    rows = []
    n_no_face = 0
    frame_counter = 0
    t0 = time.perf_counter()

    for batch in tqdm(loader, desc=f"pyfeat-bp4d-{task}", unit="batch"):
        faces_data = detector.detect_faces(
            batch["Image"], face_size=112, face_detection_threshold=face_threshold
        )
        batch_results = detector.forward(faces_data)

        file_names, frame_ids = [], []
        for i, face in enumerate(faces_data):
            n_faces = len(face["scores"])
            frame_ids.append(np.repeat(frame_counter + i, n_faces))
            file_names.append(np.repeat(batch["FileName"][i], n_faces))
        batch_results["input"] = np.concatenate(file_names)
        batch_results["frame"] = np.concatenate(frame_ids)
        batch_results["Identity"] = 0

        # Keep largest face per frame, skip frames with no detection
        for i, frame_idx in enumerate(
            range(frame_counter, frame_counter + len(batch["FileName"]))
        ):
            frame_results = batch_results[batch_results["frame"] == frame_idx]
            if frame_results.isna().any().any():
                n_no_face += 1
                continue
            largest_idx = frame_results.assign(
                face_area=frame_results["FaceRectWidth"]
                * frame_results["FaceRectHeight"]
            )["face_area"].idxmax()
            row = frame_results.loc[[largest_idx]]
            au_df = row.aus.copy()
            au_df.insert(0, "image", batch["FileName"][i])
            au_df.insert(0, "au_frame", batch["AUFrame"][i])
            au_df.insert(0, "task", batch["Task"][i])
            au_df.insert(0, "subject", batch["Subject"][i])
            rows.append(au_df.reset_index(drop=True))

        frame_counter += len(batch["FileName"])

    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(output_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Done: {elapsed:.1f}s ({elapsed / n * 1000:.0f}ms/img)"
        f" -> {len(combined)} rows, {n_no_face} frames skipped (no face)"
    )


if __name__ == "__main__":
    main()
