"""Build a Parquet index of coded frames and AU labels from BP4D AUCoding files.

Reads AU_OCC and AU_INT CSVs and writes a single Parquet file where each row
is one coded frame:

    subject | task | frame | AU1 | AU2 | AU4 | ... | AU6_int | AU10_int | ...

AU occurrence columns are named AU{n} (e.g. AU6), intensity columns AU{n}_int.
Values are 0/1 for occurrence, 0-5 for intensity, null for missing (9).

Note: frame numbers are 1-based (as in the AU CSV files). Image filenames are
0-based, so image frame N = AU frame N+1.

Usage:
    uv run python src/scripts/build_bp4d_index.py
"""

from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from msc.constants import DATA_DIR

AU_CODING_DIR = Path.home() / "projects/semedit/data/BP4D/AUCoding"
OUTPUT_PATH = DATA_DIR / "BP4D/bp4d_index.parquet"

INTENSITY_AUS = [6, 10, 12, 14, 17]


def parse_subject_task(stem: str) -> tuple[str, str]:
    """Parse 'F001_T1' → ('F001', 'T1')."""
    subject, task = stem.split("_")
    return subject, task


def load_occurrences(path: Path) -> pd.DataFrame:
    """Load an AU_OCC CSV.

    Returns a DataFrame with frame numbers as index and AU numbers as columns.
    Values are 0 (absent), 1 (present), or 9 (missing).
    """
    df = pd.read_csv(path, header=0, index_col=0)
    df.columns = df.columns.astype(int)
    df.index = df.index.astype(int)
    return df


def load_intensity(path: Path) -> pd.Series:
    """Load an AU_INT CSV.

    Returns a Series indexed by frame number, with intensity values.
    Value 9 is replaced with NaN (missing).
    """
    df = pd.read_csv(path, header=None, names=["frame", "intensity"], index_col=0)
    s = df["intensity"].astype(float)
    s[s == 9] = float("nan")
    return s


def main() -> None:
    occ_dir = AU_CODING_DIR / "AU_OCC"
    int_dir = AU_CODING_DIR / "AU_INT"

    occ_files = sorted(occ_dir.glob("[FM]*T[0-9].csv"))
    if not occ_files:
        logger.error(f"No occurrence CSV files found in {occ_dir}")
        return

    logger.info(f"Found {len(occ_files)} occurrence files")

    rows: list[pd.DataFrame] = []

    for occ_path in tqdm(occ_files, desc="Building index", unit="seq"):
        subject, task = parse_subject_task(occ_path.stem)
        df = load_occurrences(occ_path)

        # Drop columns that are entirely 9 (never coded), replace 9 with NaN
        coded_cols = [col for col in df.columns if (df[col] != 9).any()]
        df = df[coded_cols].astype(float)
        df[df == 9] = float("nan")

        # Rename columns to AU{n}
        df.columns = [f"AU{c}" for c in df.columns]
        df.index.name = "frame"
        df = df.reset_index()

        # Add intensity columns
        for au in INTENSITY_AUS:
            int_path = int_dir / f"AU{au:02d}" / f"{occ_path.stem}_AU{au:02d}.csv"
            if int_path.exists():
                s = load_intensity(int_path)
                s.index.name = "frame"
                df = df.merge(
                    s.rename(f"AU{au}_int").reset_index(), on="frame", how="left"
                )

        df.insert(0, "task", task)
        df.insert(0, "subject", subject)
        rows.append(df)

    index = pd.concat(rows, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(OUTPUT_PATH, index=False)

    logger.info(f"Wrote {len(index)} coded frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
