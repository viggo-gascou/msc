# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "libreface",
#   "pandas",
#   "pyarrow",
#   "torch",
#   "tqdm",
# ]
# ///
"""Run LibreFace AU detection + intensity on adult FFHQ images (filtered by index)."""

import time
from pathlib import Path

import pandas as pd
import torch
import libreface
from tqdm import tqdm

IMAGES_DIR = Path("data/FFHQ/images1024x1024")
INDEX_PATH = Path("data/FFHQ/ffhq_index.parquet")
OUT_PATH = Path("data/FFHQ/libreface_ffhq.csv")
CHUNK_SIZE = 500

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

all_paths = sorted(
    p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

index = pd.read_parquet(INDEX_PATH)
before = len(all_paths)
image_paths = [
    str(p) for p in all_paths if str(p.relative_to(IMAGES_DIR)) in index["image_number"]
]
print(f"Index filtered {before} → {len(image_paths)} images (adults only)")

n = len(image_paths)
chunks = [image_paths[i : i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]

print(f"{n} images in {len(chunks)} chunks → {OUT_PATH}\n")

t0 = time.perf_counter()
rows = []

for chunk in tqdm(chunks, desc="ffhq", unit="chunk"):
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
combined.to_csv(OUT_PATH, index=False)

elapsed = time.perf_counter() - t0
print(f"\nDone: {elapsed:.1f}s  ({elapsed / n * 1000:.0f}ms/img)  →  {len(combined)} rows")
