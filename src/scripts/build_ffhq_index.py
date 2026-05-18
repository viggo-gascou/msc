"""Build a filtered index of FFHQ images by age group and confidence."""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from msc.constants import FFHQ_INDEX_PATH, RANDOM_SEED

TARGET_GROUPS = {"20-29", "30-39", "40-49", "50-69", "70-120"}
DEFAULT_CONFIDENCE = 0.6


def main() -> None:
    """Filter FFHQ aging labels by age group and confidence and save a parquet index."""
    parser = argparse.ArgumentParser(
        description="Filter FFHQ aging labels by age group and confidence."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/FFHQ/ffhq_aging_labels.csv"),
        help="Path to the ffhq_aging_labels.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FFHQ_INDEX_PATH,
        help=f"Output parquet path (default: {FFHQ_INDEX_PATH}).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Minimum age_group_confidence to include (default: 0.7).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    logger.info(f"Loaded {len(df)} rows from {args.labels}")

    mask = df["age_group"].isin(TARGET_GROUPS) & (
        df["age_group_confidence"] >= args.min_confidence
    )
    filtered = df[mask].reset_index(drop=True)
    logger.info(
        f"Kept {len(filtered)} / {len(df)} images"
        f" (confidence >= {args.min_confidence:.2f})"
    )
    for group in sorted(TARGET_GROUPS):
        n = int((filtered["age_group"] == group).sum())
        logger.info(f"  {group}: {n}")

    filtered = filtered.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    n = len(filtered)
    train_end = int(n * 0.90)
    val_end = train_end + int(n * 0.05)
    splits = (
        ["train"] * train_end
        + ["val"] * (val_end - train_end)
        + ["test"] * (n - val_end)
    )
    filtered["split"] = splits

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.output, index=False)
    logger.info(f"Saved index to {args.output}")


if __name__ == "__main__":
    main()
