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
"""Run LibreFace AU detection + intensity on BP4D, saving one parquet per task."""

import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import libreface
from tqdm import tqdm

HPC_DATA_DIR = Path("/home/semedit/data")
if HPC_DATA_DIR.exists():
    SEQUENCES_DIR = HPC_DATA_DIR / "BP4D" / "Sequences"
    OUT_DIR = HPC_DATA_DIR / "BP4D" / "libreface"
else:
    SEQUENCES_DIR = Path("data/BP4D/Sample/Sequences")
    OUT_DIR = Path("data/BP4D/libreface")

CHUNK_SIZE = 500

OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Sequences: {SEQUENCES_DIR}")
print(f"Output:    {OUT_DIR}\n")

# Group image paths by task: {task: [(subject, path), ...]}
by_task: dict[str, list[tuple[str, Path]]] = defaultdict(list)
for subject_dir in sorted(SEQUENCES_DIR.iterdir()):
    if not subject_dir.is_dir():
        continue
    for task_dir in sorted(subject_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for img in sorted(task_dir.glob("*.jpg")):
            by_task[task_dir.name].append((subject_dir.name, img))

print(f"Found {len(by_task)} tasks: {sorted(by_task)}\n")

for task, entries in sorted(by_task.items()):
    out_path = OUT_DIR / f"libreface_bp4d_{task}.parquet"
    subjects = [s for s, _ in entries]
    image_paths = [str(p) for _, p in entries]
    n = len(image_paths)
    chunks = [image_paths[i : i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]
    subject_chunks = [subjects[i : i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]

    print(f"[{task}] {n} frames across {len(set(subjects))} subjects → {out_path}")
    t0 = time.perf_counter()
    rows = []

    for chunk_imgs, chunk_subjects in tqdm(
        zip(chunks, subject_chunks), total=len(chunks), desc=task, unit="chunk"
    ):
        det_df, intensity_df = libreface.get_au_intensities_and_detect_aus_video(
            chunk_imgs,
            device=device,
            batch_size=32,
            num_workers=0,
        )
        chunk_df = pd.concat(
            [
                pd.Series(chunk_subjects, name="subject"),
                pd.Series(chunk_imgs, name="image"),
                det_df.reset_index(drop=True),
                intensity_df.reset_index(drop=True),
            ],
            axis=1,
        )
        rows.append(chunk_df)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"  Done: {elapsed:.1f}s  ({elapsed / n * 1000:.0f}ms/img)  →  {len(combined)} rows\n")

print("All tasks complete.")
