"""Download FFHQ metadata and data files."""

import os
import sys
from pathlib import Path
from zipfile import ZipFile

import gdown
from loguru import logger

from msc.constants import DATA_DIR
from msc.log_utils import LOG_FORMAT

logger.remove()
logger.add(sys.stderr, format=LOG_FORMAT, colorize=True)

JSON_METADATA_FILE_ID = "16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA"
ZIP_DATA_FILE_ID = "1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv"
OUTPUT_DIR = Path("ffhq_data")
FORCE_REDOWNLOAD = bool(os.getenv("FORCE_REDOWNLOAD"))


def main() -> None:
    """Entry point: download and extract FFHQ data."""
    metadata_dir = DATA_DIR / OUTPUT_DIR / "metadata"
    data_dir = DATA_DIR / OUTPUT_DIR / "data"
    for d in (metadata_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    json_path = download_json(metadata_dir=metadata_dir)
    zip_path = download_zip(data_dir=data_dir)
    extract_zip(zip_path=zip_path, data_dir=data_dir)

    logger.info("Done.")
    logger.info(f"JSON: {json_path}")
    logger.info(f"Data directory: {data_dir}")


def download_json(metadata_dir: Path) -> Path:
    """Download the FFHQ dataset JSON metadata file.

    Args:
        metadata_dir:
          Directory where the JSON file will be saved.

    Returns:
        Path to the downloaded JSON file.

    Raises:
        RuntimeError:
          If the download fails.
    """
    out_path = metadata_dir / "ffhq-dataset-v2.json"
    if out_path.exists() and not FORCE_REDOWNLOAD:
        logger.info(f"{out_path} already exists, skipping.")
        return out_path
    logger.info(f"Downloading JSON to {out_path} ...")
    result = gdown.download(
        id=JSON_METADATA_FILE_ID, output=str(out_path), quiet=False, format="json"
    )
    if not result:
        raise RuntimeError("Failed to download JSON metadata file.")
    return Path(result)


def download_zip(data_dir: Path) -> Path:
    """Download the FFHQ ZIP archive.

    Args:
        data_dir:
          Directory where the ZIP file will be downloaded.

    Returns:
        Path to the ZIP archive.

    Raises:
        RuntimeError:
          If the download fails or the result is not a ZIP file.
    """
    out_path = data_dir / "ffhq.zip"
    if out_path.exists() and not FORCE_REDOWNLOAD:
        logger.info(f"{out_path} already exists, skipping.")
        return out_path
    logger.info(f"Downloading ZIP to {out_path} ...")
    downloaded = gdown.download(id=ZIP_DATA_FILE_ID, output=str(out_path), quiet=False)
    if not downloaded:
        raise RuntimeError("ZIP download failed.")
    return Path(downloaded)


def extract_zip(zip_path: Path, data_dir: Path) -> None:
    """Extract a ZIP archive into the given directory.

    Args:
        zip_path:
          Path to the ZIP file to extract.
        data_dir:
          Destination directory for extracted contents.
    """
    logger.info(f"Extracting {zip_path} into {data_dir} ...")
    with ZipFile(file=zip_path, mode="r") as zf:
        zf.extractall(path=data_dir)
    logger.info("Extraction complete.")


if __name__ == "__main__":
    main()
