"""Abstract base class for face embedding models."""

import abc
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ..enums import IdentityModel, ONNXProvider


def load_insightface(
    model_name: IdentityModel = IdentityModel.BUFFALO_L,
    providers: list[ONNXProvider] | ONNXProvider = [
        ONNXProvider.CUDA,
        ONNXProvider.CPU,
    ],
    root: Path | None = None,
    ctx_id: int = 0,
    det_size: tuple[int, int] = (640, 640),
) -> object:
    """Load and prepare an InsightFace FaceAnalysis model silently.

    Suppresses the verbose stdout output InsightFace prints during model
    loading and provider initialisation.

    Args:
        model_name:
          InsightFace model to use. Defaults to buffalo_l.
        providers:
          ONNX Runtime execution providers, tried in order.
        root:
          Root directory for cached model weights.
        ctx_id:
          GPU context ID. Use -1 for CPU.
        det_size:
          Detection input resolution.

    Returns:
      Prepared FaceAnalysis instance.
    """
    from insightface.app import FaceAnalysis

    from ..constants import CACHE_DIR

    if root is None:
        root = Path(CACHE_DIR, "insightface")

    # Suppress verbose print statements from InsightFace
    old_stdout = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            provider_strings = [
                p.value if isinstance(p, ONNXProvider) else p for p in providers
            ]
            app = FaceAnalysis(
                name=model_name.value, providers=provider_strings, root=root
            )
            app.prepare(ctx_id=ctx_id, det_size=det_size)
        finally:
            sys.stdout = old_stdout
    return app


class FaceEmbedding(abc.ABC):
    """Abstract base class for face embedding extractors.

    Defines a common interface for both differentiable (AdaFace) and
    non-differentiable (ArcFace) embedding models.
    """

    @abc.abstractmethod
    def embed(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract a normalised embedding from a single image.

        Args:
            img:
              BGR uint8 numpy array or float RGB tensor of shape (C, H, W).

        Returns:
          Normalised embedding of shape (512,).
        """

    def embed_batch(
        self, imgs: list[np.ndarray | torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Extract normalised embeddings for a batch of images.

        Args:
            imgs:
              List of BGR numpy arrays or RGB tensors, or a batched float
              tensor of shape (B, C, H, W).

        Returns:
          Embeddings of shape (B, 512).
        """
        if isinstance(imgs, torch.Tensor):
            imgs = list(imgs)
        return torch.stack([self.embed(img=img) for img in imgs])
