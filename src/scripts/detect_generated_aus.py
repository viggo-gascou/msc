r"""Detect AUs on generated images. Runs in the py-feat or LibreFace environment.

This script is intentionally standalone -- it does not import from msc.
Run it from the environment that has py-feat or LibreFace installed.

Usage (py-feat env)::

    python src/scripts/detect_generated_aus.py \
        --images-dir eval/bp4d_pyfeat/generated \
        --detector pyfeat \
        --output eval/bp4d_pyfeat/au_detections.parquet

Usage (LibreFace env)::

    python src/scripts/detect_generated_aus.py \
        --images-dir eval/bp4d_libreface/generated \
        --detector libreface \
        --output eval/bp4d_libreface/au_detections.parquet
"""

import argparse
import re
import typing as t
from pathlib import Path

import pandas as pd

# py-feat AU columns (20 AUs in order)
_PYFEAT_AU_COLUMNS: list[str] = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]

# LibreFace AU columns (17 AUs, intensity/occurrence columns)
_LIBREFACE_AU_COLUMN_MAP: dict[str, str] = {
    "AU01": "au_1_intensity",
    "AU02": "au_2_intensity",
    "AU04": "au_4_intensity",
    "AU05": "au_5_intensity",
    "AU06": "au_6_intensity",
    "AU07": "au_7",
    "AU09": "au_9_intensity",
    "AU10": "au_10",
    "AU12": "au_12_intensity",
    "AU14": "au_14",
    "AU15": "au_15_intensity",
    "AU17": "au_17_intensity",
    "AU20": "au_20_intensity",
    "AU23": "au_23",
    "AU24": "au_24",
    "AU25": "au_25_intensity",
    "AU26": "au_26_intensity",
}
_LIBREFACE_AU_COLUMNS: list[str] = list(_LIBREFACE_AU_COLUMN_MAP.keys())
_LIBREFACE_AU_SCALE: dict[str, float] = {
    au: (1.0 / 5.0 if col.endswith("_intensity") else 1.0)
    for au, col in _LIBREFACE_AU_COLUMN_MAP.items()
}


def main() -> None:
    """Detect AUs on generated images and save results to parquet.

    Raises:
        FileNotFoundError:
            If no sample_*.png files are found in the images directory.
    """
    parser = argparse.ArgumentParser(
        description="Detect AUs on generated images (py-feat or LibreFace)."
    )
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument(
        "--detector", type=str, required=True, choices=["pyfeat", "libreface"]
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_paths = sorted(images_dir.glob("sample_*.png"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No sample_*.png files found in {images_dir}")

    print(f"Detecting AUs on {len(image_paths)} images with {args.detector}...")

    if args.detector == "pyfeat":
        rows = _detect_pyfeat(image_paths=image_paths)
    else:
        rows = _detect_libreface(image_paths=image_paths)

    detections = pd.DataFrame(rows)
    detections.to_parquet(args.output, index=False)
    print(f"Saved {len(detections)} rows to {args.output}")

    au_cols = [c for c in detections.columns if c.startswith("detected_")]
    missing = detections[au_cols].isna().all(axis=1).sum()
    if missing > 0:
        print(
            f"Warning: {missing}/{len(detections)} images had no face detected"
            " (NaN AUs)."
        )


def _detect_pyfeat(image_paths: list[Path]) -> list[dict[str, t.Any]]:
    """Run py-feat detection on a list of image paths.

    Args:
        image_paths:
            Sorted list of generated image paths.

    Returns:
        List of dicts with sample_id and detected AU columns.
    """
    from feat import Detector  # type: ignore[import]

    detector = Detector(
        au_model="xgb",
        identity_model=None,
        emotion_model=None,
        facepose_model=None,
    )

    rows: list[dict[str, t.Any]] = []
    for path in image_paths:
        sample_id = _parse_sample_id(path)
        row: dict[str, t.Any] = {"sample_id": sample_id}

        try:
            result = detector.detect_image(str(path))
        except Exception as e:
            print(f"  Warning: {path.name}: {e}")
            result = None

        if result is not None and len(result) > 0:
            au_row = result.aus.iloc[0]
            for au in _PYFEAT_AU_COLUMNS:
                col = au.upper()
                row[f"detected_{au}"] = (
                    float(au_row[col]) if col in au_row else float("nan")
                )
        else:
            if result is not None:
                print(f"  Warning: {path.name}: no face detected")
            for au in _PYFEAT_AU_COLUMNS:
                row[f"detected_{au}"] = float("nan")

        rows.append(row)

    return rows


def _detect_libreface(image_paths: list[Path]) -> list[dict[str, t.Any]]:
    """Run LibreFace detection on a list of image paths.

    Args:
        image_paths:
            Sorted list of generated image paths.

    Returns:
        List of dicts with sample_id and detected AU columns (normalised to [0,1]).
    """
    import cv2  # type: ignore[import]
    from libreface import get_facial_attributes  # type: ignore[import]

    rows: list[dict[str, t.Any]] = []
    for path in image_paths:
        sample_id = _parse_sample_id(path)
        row: dict[str, t.Any] = {"sample_id": sample_id}
        try:
            img = cv2.imread(str(path))
            result = get_facial_attributes(img)
            for au, raw_col in _LIBREFACE_AU_COLUMN_MAP.items():
                val = result.get(raw_col, float("nan"))
                row[f"detected_{au}"] = float(val) * _LIBREFACE_AU_SCALE[au]
            for au in _PYFEAT_AU_COLUMNS:
                if f"detected_{au}" not in row:
                    row[f"detected_{au}"] = float("nan")
        except Exception as e:
            print(f"  Warning: {path.name}: {e}")
            for au in _PYFEAT_AU_COLUMNS:
                row[f"detected_{au}"] = float("nan")
        rows.append(row)

    return rows


def _parse_sample_id(path: Path) -> int:
    """Extract the integer sample ID from a filename like sample_0042.png.

    Args:
        path:
            Image path with filename matching sample_NNNN.png.

    Returns:
        Integer sample ID.

    Raises:
        ValueError:
            If the filename does not match the expected pattern.
    """
    match = re.search(r"sample_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse sample ID from {path.name}")
    return int(match.group(1))


if __name__ == "__main__":
    main()
