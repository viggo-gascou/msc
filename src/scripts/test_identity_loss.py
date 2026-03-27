"""Identity Loss Test Script."""

import torch
from loguru import logger
from omegaconf import DictConfig

from msc.cli import cli
from msc.constants import DATA_DIR
from msc.losses import IdentityLoss
from msc.torch_utils import load_image_as_tensor


@cli()
def main(cfg: DictConfig) -> None:
    """Main entry point.

    Args:
        cfg: Configuration as OmegaConf DictConfig
    """
    TEST_DATA_DIR = DATA_DIR / "test"
    image_paths = sorted(
        TEST_DATA_DIR.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1])
    )

    logger.info(f"Loading {len(image_paths)} images from {TEST_DATA_DIR}")
    images = torch.stack(
        [load_image_as_tensor(p, dtype=torch.float32, scale=True) for p in image_paths]
    )  # (N, C, H, W)

    identity_loss = IdentityLoss(
        model_name=cfg.identity.model,
        ctx_id=cfg.identity.ctx_id,
        providers=cfg.identity.providers,
        det_size=tuple(cfg.identity.det_size),
    )

    logger.info("Extracting embeddings...")
    with torch.no_grad():
        embeddings = identity_loss._get_embeddings(images)  # (N, 512)

    # Check how many images had no face detected (zero vectors)
    norms = embeddings.norm(dim=1)
    no_face_mask = norms == 0
    no_face = no_face_mask.sum().item()
    if no_face:
        logger.warning(f"No face detected in {no_face}/{len(image_paths)} images")

    # Only keep embeddings where a face was detected
    valid_embeddings = embeddings[~no_face_mask]
    logger.info(f"Computing similarities on {len(valid_embeddings)} valid embeddings")

    # Pairwise cosine similarity matrix — fast via matmul since embeddings are normalised
    sim_matrix = valid_embeddings @ valid_embeddings.T  # (N, N)

    # Mask out the diagonal (self-similarity is always 1.0)
    mask = ~torch.eye(len(valid_embeddings), dtype=torch.bool)
    cross_sims = sim_matrix[mask]

    logger.info(f"Pairwise cosine similarities ({len(cross_sims)} pairs):")
    logger.info(f"  Max:  {cross_sims.max():.4f}")
    logger.info(f"  Min:  {cross_sims.min():.4f}")
    logger.info(f"  Mean: {cross_sims.mean():.4f}")
    logger.info(f"  Std:  {cross_sims.std():.4f}")
    logger.info(f"  Shape: {sim_matrix.shape}")
    logger.info(f"  Shape (single): {embeddings[0].shape}")

    # Demonstrate the actual loss — pick the first two images with valid embeddings
    valid_indices = (~no_face_mask).nonzero(as_tuple=True)[0]
    if len(valid_indices) >= 2:
        pair = images[valid_indices[:2]]
        loss = identity_loss(pair[:1], pair[1:])
        logger.info(f"  Identity loss (two different images): {loss.item():.4f}")

        same = identity_loss(pair[:1], pair[:1])
        logger.info(f"  Identity loss (same image vs itself): {same.item():.4f}")


if __name__ == "__main__":
    main()
