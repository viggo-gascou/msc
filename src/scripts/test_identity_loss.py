"""Identity Loss Test Script."""

import itertools

import torch
from loguru import logger
from omegaconf import DictConfig
from torch.nn.functional import cosine_similarity

from msc.cli import cli
from msc.constants import DATA_DIR
from msc.losses import IdentityLoss


@cli()
def main(cfg: DictConfig) -> None:
    """Main entry point.

    Args:
        cfg: Configuration as OmegaConf DictConfig
    """
    identity_loss = IdentityLoss(
        model_name=cfg.identity.model, detector_backend=cfg.identity.backend
    )
    TEST_DATA_DIR = DATA_DIR / "test"
    image_paths = (
        sorted(
            [path for path in TEST_DATA_DIR.glob("*.png")],
            key=lambda x: int(x.stem.split("_")[-1]),
        )
    ) * 10
    results = identity_loss.embedding(
        list(map(str, image_paths)), enforce_detection=False
    )
    # compute cosine similarities between all embeddings
    sims = []
    for result_1, result_2 in itertools.product(results, results):
        sims.append(
            cosine_similarity(
                torch.Tensor(result_1[0]["embedding"]),
                torch.Tensor(result_2[0]["embedding"]),
                dim=0,
            )
        )
    logger.info(f"Max cosine similarity: {max(sims)}")
    logger.info(f"Min cosine similarity: {min(sims)}")
    logger.info(f"Avg cosine similarity: {sum(sims) / len(sims)}")


if __name__ == "__main__":
    main()
