"""Inspect AU change as a function of frame distance in the BP4D dataset."""

import numpy as np
import pandas as pd

from msc.constants import BP4D_TRAIN_INDEX_PATH, BP4D_AU_COLUMNS, BP4D_AU_COLUMN_MAP

N_SAMPLES = 5000
DISTANCES = [5, 10, 25, 50, 100]


def main() -> None:
    index = pd.read_parquet(BP4D_TRAIN_INDEX_PATH)
    au_cols = [BP4D_AU_COLUMN_MAP[au] for au in BP4D_AU_COLUMNS]

    rng = np.random.default_rng(seed=42)

    print(f"{'Min dist':>10} {'Mean |ΔAU|':>12} {'Median |ΔAU|':>14} {'% any change':>14}")
    print("-" * 54)

    for min_dist in DISTANCES:
        deltas: list[float] = []
        any_change: list[bool] = []

        groups = index.groupby(["subject", "task"])
        attempts = 0

        while len(deltas) < N_SAMPLES and attempts < N_SAMPLES * 20:
            attempts += 1
            key, group = list(groups)[rng.integers(len(groups))]
            group = group.reset_index(drop=True)

            if len(group) < min_dist + 1:
                continue

            src_idx = rng.integers(len(group))
            candidates = group[
                (group["frame"] - group.iloc[src_idx]["frame"]).abs() >= min_dist
            ]
            if candidates.empty:
                continue

            tgt_idx = rng.integers(len(candidates))
            src = group.iloc[src_idx][au_cols].fillna(0).values.astype(float)
            tgt = candidates.iloc[tgt_idx][au_cols].fillna(0).values.astype(float)
            delta = np.abs(tgt - src)
            deltas.append(delta.mean())
            any_change.append(bool((delta > 0).any()))

        print(
            f"{min_dist:>10} {np.mean(deltas):>12.4f} {np.median(deltas):>14.4f}"
            f" {100 * np.mean(any_change):>13.1f}%"
        )


if __name__ == "__main__":
    main()
