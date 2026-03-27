# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gdown>=5.2.1",
# ]
# ///
"""Simple FFHQ metadata and archive downloader using gdown.

This script downloads:
1) The FFHQ JSON metadata file into ./ffhq_data/metadata
2) A ZIP archive from the provided Google Drive folder into ./ffhq_data/data

It then extracts the ZIP archive into ./ffhq_data/data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import gdown

from msc.constants import DATA_DIR

JSON_URL = "https://drive.google.com/file/d/16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA"
ZIP_FOLDER_URL = "https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS"


def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    metadata_dir = DATA_DIR / root / "metadata"
    data_dir = DATA_DIR / root / "data"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir, data_dir


def download_json(metadata_dir: Path) -> Path:
    out_path = metadata_dir / "ffhq-dataset-v2.json"
    print(f"Downloading JSON to {out_path} ...")
    result = gdown.download(url=JSON_URL, output=str(out_path), quiet=False, fuzzy=True)
    if not result:
        raise RuntimeError("Failed to download JSON metadata file.")
    return Path(result)


def download_zip_from_folder(data_dir: Path) -> Path:
    print(f"Downloading files from folder link into {data_dir} ...")
    downloaded = gdown.download_folder(
        url=ZIP_FOLDER_URL,
        output=str(data_dir),
        quiet=False,
        use_cookies=False,
    )
    if not downloaded:
        raise RuntimeError("No files were downloaded from the folder link.")

    zip_files = [Path(p) for p in downloaded if str(p).lower().endswith(".zip")]
    if not zip_files:
        raise RuntimeError("Folder download did not contain a .zip file.")
    if len(zip_files) > 1:
        # Pick the largest ZIP if there are multiple candidates.
        zip_files.sort(key=lambda p: p.stat().st_size if p.exists() else -1, reverse=True)

    zip_path = zip_files[0]
    print(f"Selected ZIP archive: {zip_path}")
    return zip_path


def extract_zip(zip_path: Path, data_dir: Path) -> None:
    print(f"Extracting {zip_path} into {data_dir} ...")
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FFHQ JSON and ZIP with gdown")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ffhq_data"),
        help="Root output directory where metadata/ and data/ will be created (default: ./ffhq_data)",
    )
    parser.add_argument(
        "--remove-zip",
        action="store_true",
        help="Delete the downloaded ZIP file after successful extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    metadata_dir, data_dir = _ensure_dirs(root)

    json_path = download_json(metadata_dir)
    zip_path = download_zip_from_folder(data_dir)
    extract_zip(zip_path, data_dir)

    if args.remove_zip:
        print(f"Removing ZIP archive: {zip_path}")
        zip_path.unlink(missing_ok=True)

    print("Done.")
    print(f"JSON: {json_path}")
    print(f"Data directory: {data_dir}")


if __name__ == "__main__":
    main()
