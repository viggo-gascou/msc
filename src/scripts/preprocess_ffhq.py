"""Detect and align faces from FFHQ images, storing 112x112 crops to HDF5.

Reads image names from the LibreFace parquet, runs SCRFD detection at
1024x1024, aligns each face to the ArcFace canonical template with
norm_crop, and writes the result to:

    FFHQ/ffhq_preprocessed.h5
    ├── 00000  (3, 112, 112) float32  [-1, 1]
    ├── 00001  (3, 112, 112) float32  [-1, 1]
    ...

Run this before precompute_embeddings.py --dataset ffhq.
"""

import argparse

import cv2
import h5py
import numpy as np
import pandas as pd
from insightface.utils import face_align
from loguru import logger
from tqdm import tqdm

from msc.constants import FFHQ_IMAGES_DIR, FFHQ_PREPROCESSED_PATH, LIBREFACE_FFHQ_PATH
from msc.enums import ONNXProvider
from msc.face_embeddings.arcface import ArcFaceEmbedding


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Preprocess FFHQ faces (detect + align).")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing HDF5 file instead of resuming.",
    )
    p.add_argument(
        "--det-size",
        type=int,
        default=512,
        help="Square detection input size in pixels (default: 512).",
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not LIBREFACE_FFHQ_PATH.exists():
        logger.error(f"LibreFace parquet not found: {LIBREFACE_FFHQ_PATH}")
        return

    arcface = ArcFaceEmbedding(
        providers=ONNXProvider.CUDA, ctx_id=0, det_size=(args.det_size, args.det_size)
    )
    det_model = arcface.app.det_model

    image_names = (
        pd.read_parquet(LIBREFACE_FFHQ_PATH, columns=["image"])["image"]
        .apply(lambda p: str(p).split("/")[-1])
        .tolist()
    )
    logger.info(f"Processing {len(image_names)} images -> {FFHQ_PREPROCESSED_PATH}")

    FFHQ_PREPROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.force else "a"
    saved = skipped = missing = no_face = 0

    with h5py.File(FFHQ_PREPROCESSED_PATH, mode) as f:
        for name in tqdm(image_names, desc="preprocess-ffhq"):
            stem = name.split(".")[0]
            if stem in f:
                skipped += 1
                continue
            img_path = FFHQ_IMAGES_DIR / name
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                missing += 1
                continue
            bgr = cv2.imread(str(img_path))
            _, kpss = det_model.detect(bgr, max_num=1)
            if kpss is None or len(kpss) == 0:
                logger.warning(f"No face detected: {name}")
                no_face += 1
                continue
            aimg = face_align.norm_crop(bgr, landmark=kpss[0])
            crop = (aimg.transpose(2, 0, 1).astype(np.float32) - 127.5) / 127.5
            f.create_dataset(stem, data=crop, compression="lzf")
            saved += 1

    logger.info(
        f"Done — saved: {saved}  skipped: {skipped}"
        f"  missing: {missing}  no_face: {no_face}"
    )


if __name__ == "__main__":
    main()
