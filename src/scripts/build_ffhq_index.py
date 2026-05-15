"""Build a filtered index of FFHQ images by age group and confidence."""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_GROUPS = {"20-29", "30-39", "40-49", "50-69", "70-120"}
DEFAULT_CONFIDENCE = 0.7


def make_index() -> None:
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
        default=Path("data/FFHQ/ffhq_index.parquet"),
        help="Output path for the filtered parquet index.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Minimum age_group_confidence to include. Defaults to 0.7.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.labels)
    logger.info("Loaded %d rows from %s", len(df), args.labels)

    mask = df["age_group"].isin(TARGET_GROUPS) & (
        df["age_group_confidence"] >= args.min_confidence
    )
    filtered = df[mask].reset_index(drop=True)

    logger.info(
        "Kept %d / %d images (confidence >= %.2f)",
        len(filtered),
        len(df),
        args.min_confidence,
    )
    for group in sorted(TARGET_GROUPS):
        n = (filtered["age_group"] == group).sum()
        logger.info("  %s: %d", group, n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.output, index=False)
    logger.info("Saved index to %s", args.output)


if __name__ == "__main__":
    make_index()
