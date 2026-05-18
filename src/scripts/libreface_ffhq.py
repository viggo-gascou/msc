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
"""Run LibreFace AU detection + intensity estimation on age-filtered FFHQ images.

Dependencies are declared inline — just run with uv, no project setup needed:
    uv run libreface_ffhq.py --images-dir /path/to/images --index ffhq_index.parquet
"""

import argparse
import time
from pathlib import Path

import libreface
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

DEFAULT_IMAGES_DIR = Path("/home/semedit/data/FFHQ/images1024x1024")
DEFAULT_OUT_PATH = Path("data/FFHQ/ffhq_libreface.parquet")


def main() -> None:
    """Parse arguments and run LibreFace AU extraction on FFHQ."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory of FFHQ images.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Parquet index with 'image_number' column for age filtering.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUT_PATH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="LibreFace inference batch size (default: 32).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Images per accumulation chunk before concatenating (default: 500).",
    )
    args = parser.parse_args()
    run(
        images_dir=args.images_dir,
        index_path=args.index,
        output_path=args.output,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )


def run(
    images_dir: Path,
    index_path: Path,
    output_path: Path,
    batch_size: int,
    chunk_size: int,
) -> None:
    """Run LibreFace on age-filtered FFHQ images and save to parquet.

    Args:
        images_dir:
          Directory containing FFHQ images.
        index_path:
          Parquet index with 'image_number' column.
        output_path:
          Destination parquet file.
        batch_size:
          Number of images per LibreFace forward pass.
        chunk_size:
          Number of images per accumulation chunk before concatenating results.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    all_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    allowed = set(pd.read_parquet(index_path)["image_number"])
    image_paths = [str(p) for p in all_paths if p.name in allowed]
    logger.info(f"Index filtered {len(all_paths)} -> {len(image_paths)} images")

    n = len(image_paths)
    chunks = [image_paths[i : i + chunk_size] for i in range(0, n, chunk_size)]
    logger.info(f"{n} images in {len(chunks)} chunks -> {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = []

    for chunk in tqdm(chunks, desc="libreface-ffhq", unit="chunk"):
        det_df, intensity_df = libreface.get_au_intensities_and_detect_aus_video(
            chunk, device=device, batch_size=batch_size, num_workers=0
        )
        rows.append(
            pd.concat(
                [
                    pd.Series(chunk, name="image"),
                    det_df.reset_index(drop=True),
                    intensity_df.reset_index(drop=True),
                ],
                axis=1,
            )
        )

    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(output_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Done: {elapsed:.1f}s  ({elapsed / n * 1000:.0f}ms/img)"
        f"  ->  {len(combined)} rows  ->  {output_path}"
    )


if __name__ == "__main__":
    main()
