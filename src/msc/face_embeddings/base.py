"""Abstract base class for face embedding models."""

import abc

import numpy as np
import torch


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
