"""FFHQ dataset utilities — index loading and HDF5 stem filtering."""

from pathlib import Path

import h5py
import pandas as pd
from loguru import logger

from ...constants import FFHQ_EMBEDDINGS_PATH, FFHQ_INDEX_PATH, PYFEAT_FFHQ_PATH


def load_ffhq_df() -> pd.DataFrame:
    """Load the merged FFHQ dataset index with AU data and split assignments.

    Reads the LibreFace parquet (AU values), merges the split column from the
    age-filtered index, then filters to images whose ArcFace embeddings are
    present in the HDF5 file.

    Returns:
        DataFrame with AU columns, image_number, and split column, restricted
        to images with valid embeddings.

    Raises:
        FileNotFoundError:
          If the LibreFace parquet, age index, or embeddings HDF5 are missing.
    """
    for path, hint in [
        (PYFEAT_FFHQ_PATH, "run pyfeat_ffhq.py first"),
        (FFHQ_INDEX_PATH, "run build_ffhq_index.py first"),
        (FFHQ_EMBEDDINGS_PATH, "run precompute_embeddings.py --dataset ffhq first"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — {hint}")

    libreface = pd.read_parquet(PYFEAT_FFHQ_PATH)
    libreface["image_number"] = libreface["image"].apply(lambda p: Path(p).name)

    index = pd.read_parquet(FFHQ_INDEX_PATH, columns=["image_number", "split"])
    df = libreface.merge(index, on="image_number", how="inner")

    with h5py.File(FFHQ_EMBEDDINGS_PATH, "r") as f:
        valid_stems = set(f.keys())

    before = len(df)
    mask = df["image_number"].apply(lambda n: Path(n).stem).isin(valid_stems)
    df = pd.DataFrame(df.loc[mask]).reset_index(drop=True)
    logger.info(
        f"FFHQ index: {before} -> {len(df)} images (filtered to valid embeddings)"
    )
    return df
