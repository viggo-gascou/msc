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
from loguru import logger

# py-feat AU columns (20 AUs in order)
_PYFEAT_AU_COLUMNS: list[str] = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU09",
    "AU10",
    "AU11",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU24",
    "AU25",
    "AU26",
    "AU28",
    "AU43",
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

    logger.info(f"Detecting AUs on {len(image_paths)} images with {args.detector}...")

    if args.detector == "pyfeat":
        rows = _detect_pyfeat(image_paths=image_paths)
    else:
        rows = _detect_libreface(image_paths=image_paths)

    detections = pd.DataFrame(rows)
    detections.to_parquet(args.output, index=False)
    logger.info(f"Saved {len(detections)} rows to {args.output}")

    au_cols = [c for c in detections.columns if c.startswith("detected_")]
    missing = detections[au_cols].isna().all(axis=1).sum()
    if missing > 0:
        logger.warning(
            f"{missing}/{len(detections)} images had no face detected (NaN AUs)."
        )


def _detect_pyfeat(image_paths: list[Path]) -> list[dict[str, t.Any]]:
    """Run py-feat detection on a list of image paths.

    Args:
        image_paths:
            Sorted list of generated image paths.

    Returns:
        List of dicts with sample_id and detected AU columns.
    """
    import warnings

    import torch
    from feat import Detector  # type: ignore[import]
    from PIL import Image
    from torchvision import transforms  # type: ignore[import]

    to_tensor = transforms.ToTensor()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    warnings.filterwarnings(
        "ignore", message=".*serialized model", category=UserWarning, module=".*skops"
    )
    detector = Detector(
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model=None,
        identity_model=None,
        device=device,
    )

    rows: list[dict[str, t.Any]] = []
    for path in image_paths:
        sample_id, config_name = _parse_sample_meta(path)
        row: dict[str, t.Any] = {"sample_id": sample_id, "config_name": config_name}

        try:
            img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
            tensor = (to_tensor(img) * 255).byte().unsqueeze(0)  # (1, C, H, W) uint8
            faces_data = detector.detect_faces(
                tensor, face_size=112, face_detection_threshold=0.3
            )
            result = detector.forward(faces_data)
            result["frame"] = 0  # scalar broadcast — works for 1 or N face rows
            frame_result = result[result["frame"] == 0]
            au_cols = [c for c in frame_result.columns if c.startswith("AU")]
            if len(frame_result) == 0 or frame_result[au_cols].isna().all(axis=1).all():
                raise ValueError("no face detected")
            largest_idx = frame_result.assign(
                face_area=frame_result["FaceRectWidth"] * frame_result["FaceRectHeight"]
            )["face_area"].idxmax()
            au_row = frame_result.loc[[largest_idx]].aus.iloc[0]
            for au in _PYFEAT_AU_COLUMNS:
                col = au.upper()
                row[f"detected_{au}"] = (
                    float(au_row[col]) if col in au_row else float("nan")
                )
        except Exception as e:
            logger.warning(f"{path.name}: {e}")
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
    import torch  # type: ignore[import]
    import libreface  # type: ignore[import]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    str_paths = [str(p) for p in image_paths]

    det_df, intensity_df = libreface.get_au_intensities_and_detect_aus_video(
        str_paths, device=device, batch_size=32, num_workers=0
    )
    combined = pd.concat(
        [det_df.reset_index(drop=True), intensity_df.reset_index(drop=True)], axis=1
    )

    rows: list[dict[str, t.Any]] = []
    for i, path in enumerate(image_paths):
        sample_id, config_name = _parse_sample_meta(path)
        row: dict[str, t.Any] = {"sample_id": sample_id, "config_name": config_name}
        try:
            result_row = combined.iloc[i]
            for au, raw_col in _LIBREFACE_AU_COLUMN_MAP.items():
                val = result_row[raw_col] if raw_col in result_row.index else float("nan")
                row[f"detected_{au}"] = float(val) * _LIBREFACE_AU_SCALE[au]
        except Exception as e:
            logger.warning(f"{path.name}: {e}")
            for au in _LIBREFACE_AU_COLUMN_MAP:
                row[f"detected_{au}"] = float("nan")
        rows.append(row)

    return rows


def _parse_sample_meta(path: Path) -> tuple[int, str]:
    """Extract sample ID and config name from a filename like sample_0042_smile.png.

    Args:
        path:
            Image path with filename matching sample_NNNN_<config>.png.

    Returns:
        Tuple of (sample_id, config_name).

    Raises:
        ValueError:
            If the filename does not match the expected pattern.
    """
    match = re.fullmatch(r"sample_(\d+)_([a-z_]+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse sample metadata from {path.name}")
    return int(match.group(1)), match.group(2)


if __name__ == "__main__":
    main()
