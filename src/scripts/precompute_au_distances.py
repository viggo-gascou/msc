"""Precompute pairwise normalised L1 AU distances for all BP4D sequences.

For each (subject, task) sequence, computes the N×N matrix of pairwise AU
distances between all coded frames and saves the result to an HDF5 file.
The dataset can load individual sequence matrices at init time to use AU
distance instead of (or alongside) temporal distance for target frame sampling.

Distances are computed on normalised AU values (intensity AUs divided by 5,
occurrence AUs left at 0/1) so all AUs contribute on the same scale.

Output: BP4D_DATA_DIR/au_distances.h5
  Keys:  "{subject}/{task}"  (e.g. "F001/T1")
  Value: float32 dataset of shape (N, N) — symmetric, zeros on diagonal.
         Row/column order matches the frame order in the index parquet
         (sorted ascending by frame number within each sequence).
"""

import h5py
import numpy as np
import pandas as pd

from msc.au_adapter import AU_SCALE
from msc.constants import (
    BP4D_AU_COLUMN_MAP,
    BP4D_AU_COLUMNS,
    BP4D_DATA_DIR,
    BP4D_TEST_INDEX_PATH,
    BP4D_TRAIN_INDEX_PATH,
    BP4D_VAL_INDEX_PATH,
)
from msc.data.bp4d import load_index

OUTPUT_PATH = BP4D_DATA_DIR / "au_distances.h5"


def compute_sequence_distances(au_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise L1 distances for a sequence of normalised AU vectors.

    Args:
        au_matrix:
            Array of shape (N, num_aus) with normalised AU values.

    Returns:
        Symmetric float32 distance matrix of shape (N, N).
    """
    # Broadcast: (N, 1, D) - (1, N, D) → (N, N, D) → sum → (N, N)
    diff = au_matrix[:, None, :] - au_matrix[None, :, :]
    return np.abs(diff).sum(axis=-1).astype(np.float32)


def main() -> None:
    """Compute and save AU distance matrices for all sequences."""
    index = pd.concat(
        [
            load_index(path=BP4D_TRAIN_INDEX_PATH),
            load_index(path=BP4D_VAL_INDEX_PATH),
            load_index(path=BP4D_TEST_INDEX_PATH),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["subject", "task", "frame"])

    au_cols = [BP4D_AU_COLUMN_MAP[au] for au in BP4D_AU_COLUMNS]

    all_distances: list[float] = []

    groups = index.groupby(["subject", "task"])
    print(f"Computing distances for {len(groups)} sequences...")

    with h5py.File(OUTPUT_PATH, "w") as f:
        for (subject, task), group in groups:
            group = group.sort_values("frame")
            aus = group[au_cols].to_numpy(dtype=np.float32, na_value=0.0)
            aus = aus * np.array(AU_SCALE, dtype=np.float32)

            dist = compute_sequence_distances(aus)
            f.require_group(subject).create_dataset(task, data=dist, compression="gzip")

            # Collect upper-triangle values (excluding diagonal) for stats
            n = len(group)
            if n > 1:
                triu = dist[np.triu_indices(n, k=1)]
                all_distances.extend(triu.tolist())

    print(f"Saved {len(groups)} distance matrices to {OUTPUT_PATH}")

    distances = np.array(all_distances, dtype=np.float32)
    print(f"\nAU distance distribution across {len(distances):,} pairs:")
    for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{p:>3}: {np.percentile(distances, p):.4f}")


main()
