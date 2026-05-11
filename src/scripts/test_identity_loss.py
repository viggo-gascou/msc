"""Compare ArcFace and AdaFace embeddings on test images."""

import argparse

import torch
from loguru import logger

from msc.constants import DATA_DIR
from msc.enums import ONNXProvider
from msc.face_embeddings import AdaFaceEmbedding, ArcFaceEmbedding, load_adaface
from msc.torch_utils import load_image_as_tensor


def similarity_stats(embeddings: torch.Tensor) -> dict[str, float]:
    """Compute similarity statistics for a batch of embeddings.

    Args:
        embeddings: Tensor of shape (N, D) containing N embeddings of dimension D.

    Returns:
        Dictionary with keys "max", "min", "mean", "std" and corresponding values.
    """
    sim_matrix = embeddings @ embeddings.T
    mask = ~torch.eye(len(embeddings), dtype=torch.bool)
    cross = sim_matrix[mask]
    return {
        "max": cross.max().item(),
        "min": cross.min().item(),
        "mean": cross.mean().item(),
        "std": cross.std().item(),
    }


def main() -> None:
    """Compare ArcFace and AdaFace pairwise similarities on test images."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det-size", type=int, nargs=2, default=[640, 640], metavar=("H", "W")
    )
    args = parser.parse_args()
    det_size = tuple(args.det_size)

    test_dir = DATA_DIR / "test"
    image_paths = sorted(
        test_dir.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1])
    )
    logger.info(f"Loading {len(image_paths)} images from {test_dir}")

    images = torch.stack(
        [load_image_as_tensor(p, dtype=torch.float32, scale=True) for p in image_paths]
    )

    # --- ArcFace ---
    logger.info("Extracting ArcFace embeddings (InsightFace buffalo_l)...")
    arcface = ArcFaceEmbedding(providers=[ONNXProvider.CPU], det_size=det_size)
    arc_embs = arcface.embed_batch(imgs=images)

    no_face = (arc_embs.norm(dim=1) == 0).sum().item()
    if no_face:
        logger.warning(f"ArcFace: no face detected in {no_face}/{len(images)} images")

    valid_arc = arc_embs[arc_embs.norm(dim=1) > 0]
    arc_stats = similarity_stats(embeddings=valid_arc)
    logger.info(f"ArcFace similarities ({len(valid_arc)} valid):")
    for k, v in arc_stats.items():
        logger.info(f"  {k}: {v:.4f}")

    # --- AdaFace ---
    logger.info("Loading AdaFace model...")
    adaface: AdaFaceEmbedding = load_adaface(device="cpu", det_size=det_size)

    logger.info("Extracting AdaFace embeddings (IR-101)...")
    ada_embs = adaface.embed_batch(imgs=images)

    valid_ada = ada_embs[arc_embs.norm(dim=1) > 0]
    ada_stats = similarity_stats(embeddings=valid_ada)
    logger.info(f"AdaFace similarities ({len(valid_ada)} valid):")
    for k, v in ada_stats.items():
        logger.info(f"  {k}: {v:.4f}")

    # --- Comparison ---
    logger.info("Embedding space agreement (cosine sim between ArcFace and AdaFace):")
    agreement = torch.nn.functional.cosine_similarity(valid_arc, valid_ada, dim=1)
    logger.info(f"  mean: {agreement.mean():.4f}  std: {agreement.std():.4f}")


if __name__ == "__main__":
    main()
