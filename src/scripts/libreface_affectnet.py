# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "libreface",
#   "pandas",
#   "torch",
#   "tqdm",
# ]
# ///
"""Run LibreFace AU detection + intensity on all AffectNet+ images, per split."""

import time
from pathlib import Path

import pandas as pd
import torch
import libreface
from tqdm import tqdm

DATA_DIR = Path("data/AffectNet+")

SPLITS = {
    "human_train": DATA_DIR / "human_annotated" / "train_set" / "images",
    "human_val":   DATA_DIR / "human_annotated" / "validation_set" / "images",
    "no_human":    DATA_DIR / "no_human_annotated" / "images",
}

OUT_PATHS = {
    "human_train": Path("libreface_human_train.csv"),
    "human_val":   Path("libreface_human_val.csv"),
    "no_human":    Path("libreface_no_human.csv"),
}

CHUNK_SIZE = 500

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}\n")

for split_name, images_dir in SPLITS.items():
    image_paths = sorted(str(p) for p in images_dir.iterdir() if p.is_file())
    if not image_paths:
        print(f"[{split_name}] No images found in {images_dir}, skipping.")
        continue

    out_path = OUT_PATHS[split_name]
    n = len(image_paths)
    chunks = [image_paths[i : i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]

    print(f"[{split_name}] {n} images in {len(chunks)} chunks → {out_path}")
    t0 = time.perf_counter()
    rows = []

    for chunk in tqdm(chunks, desc=split_name, unit="chunk"):
        det_df, intensity_df = libreface.get_au_intensities_and_detect_aus_video(
            chunk,
            device=device,
            batch_size=32,
            num_workers=0,
        )
        chunk_df = pd.concat(
            [
                pd.Series(chunk, name="image"),
                det_df.reset_index(drop=True),
                intensity_df.reset_index(drop=True),
            ],
            axis=1,
        )
        rows.append(chunk_df)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"  Done: {elapsed:.1f}s  ({elapsed / n * 1000:.0f}ms/img)  →  {len(combined)} rows")
    print()

print("All splits complete.")
