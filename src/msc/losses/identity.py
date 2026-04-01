"""Identity loss using AdaFace embeddings."""

import typing as t

import torch
import torch.nn.functional as F

from ..face_embeddings import AdaFaceEmbedding
from .base import CustomLoss


class IdentityLoss(CustomLoss):
    """Cosine identity loss based on AdaFace face embeddings.

    Compares face identity of two batches of pre-processed face crops by
    extracting differentiable AdaFace embeddings and computing cosine distance.

    Input tensors must be pre-aligned 112x112 face crops normalised to [-1, 1],
    shaped (B, 3, 112, 112).
    """

    def __init__(
        self,
        embedder: AdaFaceEmbedding,
        weight: t.Optional[torch.Tensor] = None,
    ) -> None:
        """Initialise IdentityLoss.

        Args:
            embedder:
              AdaFaceEmbedding instance used to extract face embeddings.
            weight:
              Scalar weight tensor passed to the base class, defaults to None.
        """
        super().__init__(weight=weight)
        self.embedder = embedder

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean cosine identity loss over a batch.

        Args:
            input:
              Predicted face crops, shape (B, 3, 112, 112) in [-1, 1].
            target:
              Ground-truth face crops, shape (B, 3, 112, 112) in [-1, 1].

        Returns:
          Scalar loss in [0, 1], where 0 means identical identities
          and 1 means orthogonal embeddings.
        """
        emb_input = self.embedder(input)
        emb_target = self.embedder(target)

        similarity = F.cosine_similarity(emb_input, emb_target, dim=1)
        loss = (1.0 - similarity) / 2.0

        if self.weight is not None:
            loss = loss * self.weight

        return loss.mean()
