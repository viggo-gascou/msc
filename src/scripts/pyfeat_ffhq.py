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
r"""Run py-feat AU detection on age-filtered FFHQ images.

Run with:
    .venv-pyfeat/bin/python src/scripts/pyfeat_ffhq.py \
        --index data/FFHQ/ffhq_index.parquet
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

DEFAULT_IMAGES_DIR = Path("/home/semedit/data/FFHQ/images1024x1024")
DEFAULT_INDEX_PATH = Path("data/FFHQ/ffhq_index.parquet")
DEFAULT_OUT_PATH = Path("data/FFHQ/ffhq_pyfeat.parquet")

_to_tensor = transforms.ToTensor()


class ImagePathDataset(Dataset):
    """Torch Dataset wrapping a list of image paths."""

    def __init__(self, paths: list[Path]) -> None:
        """Initialise with a list of image paths."""
        self.paths = paths

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, t.Any]:
        """Load and return a single image as a uint8 tensor with metadata.

        Returns:
            Dict with Image tensor and path/frame metadata fields.
        """
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = (_to_tensor(img) * 255).byte()  # (C, H, W) uint8
        return {
            "Image": tensor,
            "FileName": str(path),
            "Frame": idx,
            "Scale": 1.0,
            "Padding": {"Left": 0, "Top": 0, "Right": 0, "Bottom": 0},
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
    }


def main() -> None:
    """Parse arguments and delegate to run()."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--face-threshold", type=float, default=0.9)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    run(
        images_dir=args.images_dir,
        index_path=args.index,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        face_threshold=args.face_threshold,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
    )


def run(
    images_dir: Path,
    index_path: Path,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    face_threshold: float,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> None:
    """Run py-feat AU detection on FFHQ and write results to parquet.

    Args:
        images_dir:
            Directory containing FFHQ images.
        index_path:
            Path to the FFHQ index parquet with an image_number column.
        output_path:
            Path to write the output parquet file.
        batch_size:
            Number of images per detection batch.
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

    all_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    allowed = set(pd.read_parquet(index_path)["image_number"])
    image_paths = [p for p in all_paths if p.name in allowed]
    image_paths = image_paths[shard_idx::num_shards]
    logger.info(
        f"Index filtered {len(all_paths)} -> {len(image_paths)} images "
        f"(shard {shard_idx}/{num_shards})"
    )

    if num_shards > 1:
        output_path = output_path.with_stem(f"{output_path.stem}_shard{shard_idx}")

    warnings.filterwarnings(
        "ignore", message=".*serialized model", category=UserWarning, module=".*skops"
    )
    detector = Detector(
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        device=device,
    )

    dataset = ImagePathDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    n = len(image_paths)
    logger.info(f"{n} images -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_no_face = 0
    frame_counter = 0
    t0 = time.perf_counter()

    for batch in tqdm(loader, desc="pyfeat-ffhq", unit="batch"):
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
        for frame_idx in range(frame_counter, frame_counter + len(batch["FileName"])):
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
            au_df.insert(0, "image", batch["FileName"][frame_idx - frame_counter])
            rows.append(au_df.reset_index(drop=True))

        frame_counter += len(batch["FileName"])

    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(output_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Done: {elapsed:.1f}s ({elapsed / n * 1000:.0f}ms/img)"
        f" -> {len(combined)} rows, {n_no_face} images skipped (no face)"
    )


if __name__ == "__main__":
    main()
