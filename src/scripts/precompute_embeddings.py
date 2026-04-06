"""Precompute ArcFace and AdaFace embeddings for all preprocessed BP4D faces.

Reads aligned face tensors from the preprocessed HDF5 files produced by
preprocess_bp4d.py and writes per-frame embeddings to a parallel set of HDF5
files:

    Embeddings/T1.h5
    ├── F001/
    │   ├── arcface  (N, 512) float32
    │   └── adaface  (N, 512) float32
    ├── F002/
    │   └── ...

ArcFace embeddings are used as the IP-Adapter-FaceID conditioning signal.
AdaFace embeddings are precomputed for the reference image only — during
training the generated image is embedded on-the-fly so gradients can flow.
"""

import argparse

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from msc.constants import BP4D_EMBEDDINGS_DIR, BP4D_PREPROCESSED_DIR
from msc.enums import ONNXProvider
from msc.face_embeddings.adaface import load_adaface
from msc.face_embeddings.arcface import ArcFaceEmbedding


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Precompute face embeddings for BP4D.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-process tasks that already have a saved .h5 file",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for AdaFace inference (default: 32)",
    )
    p.add_argument(
        "--task", type=str, default=None, help="Only process a single task (e.g. T8)"
    )
    return p.parse_args()


def embed_adaface_batched(
    model: torch.nn.Module, faces: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    """Run AdaFace on a subject's faces in batches.

    Args:
        model:
          AdaFaceEmbedding in eval mode.
        faces:
          Array of shape (N, 3, 112, 112) float32 in [-1, 1].
        batch_size:
          Number of faces per forward pass.
        device:
          PyTorch device string.

    Returns:
      Embeddings of shape (N, 512) float32.
    """
    embeddings = []
    tensor = torch.from_numpy(faces).to(device)
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i : i + batch_size]
            embeddings.append(model(batch).cpu().float().numpy())
    return np.concatenate(embeddings, axis=0)


def embed_arcface_batched(
    rec_model: object, faces: np.ndarray, batch_size: int
) -> np.ndarray:
    """Run ArcFace recognition on a subject's faces in batches.

    Bypasses InsightFace detection by calling forward() directly on
    already-aligned 112x112 crops stored in the preprocessed HDF5.

    Faces are stored in [-1, 1] — we denormalise to [0, 255] so that
    forward()'s internal (x - 127.5) / 127.5 sees the correct values.

    Args:
        rec_model:
          InsightFace ArcFaceONNX recognition model from app.models['recognition'].
        faces:
          Array of shape (N, 3, 112, 112) float32 in [-1, 1].
        batch_size:
          Number of faces per forward pass.

    Returns:
      L2-normalised embeddings of shape (N, 512) float32.
    """
    embeddings = []
    for i in range(0, len(faces), batch_size):
        batch = faces[i : i + batch_size] * 127.5 + 127.5  # [-1, 1] → [0, 255]
        embs = rec_model.forward(batch)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        embeddings.append(embs)
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not BP4D_PREPROCESSED_DIR.exists():
        logger.error(f"Preprocessed directory not found: {BP4D_PREPROCESSED_DIR}")
        return

    h5_paths = (
        [BP4D_PREPROCESSED_DIR / f"{args.task}.h5"]
        if args.task
        else sorted(BP4D_PREPROCESSED_DIR.glob("*.h5"))
    )
    if not h5_paths:
        logger.error(f"No .h5 files found in {BP4D_PREPROCESSED_DIR}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading AdaFace model...")
    adaface = load_adaface(device=device)

    logger.info("Loading ArcFace model...")
    arcface = ArcFaceEmbedding(
        providers=ONNXProvider.CUDA if device == "cuda" else ONNXProvider.CPU,
        ctx_id=0 if device == "cuda" else -1,
    )
    rec_model = arcface.app.models["recognition"]

    tasks_processed = tasks_skipped = 0

    for h5_path in tqdm(h5_paths, desc="Tasks", unit="task"):
        task = h5_path.stem
        out_path = BP4D_EMBEDDINGS_DIR / h5_path.name

        if not args.force and out_path.exists():
            tasks_skipped += 1
            continue

        with h5py.File(h5_path, "r") as src, h5py.File(out_path, "w") as dst:
            for subject in tqdm(src.keys(), desc=task, unit="subject", leave=False):
                faces = src[subject]["faces"][:]  # (N, 3, 112, 112) float32

                adaface_embs = embed_adaface_batched(
                    model=adaface,
                    faces=faces,
                    batch_size=args.batch_size,
                    device=device,
                )

                arcface_embs = embed_arcface_batched(
                    rec_model=rec_model, faces=faces, batch_size=args.batch_size
                )

                grp = dst.create_group(subject)
                grp.create_dataset("arcface", data=arcface_embs, compression="lzf")
                grp.create_dataset("adaface", data=adaface_embs, compression="lzf")

        tasks_processed += 1

    logger.info(
        f"Done — tasks processed: {tasks_processed}  skipped (exists): {tasks_skipped}"
    )


if __name__ == "__main__":
    main()
