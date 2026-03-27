"""Identity Loss using face embeddings."""

import typing as t
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis

from ..config import DEFAULT_ONNX_PROVIDERS
from ..constants import CACHE_DIR
from ..enums import IdentityModel, ONNXProvider
from ..torch_utils import tensor_to_bgr
from .base import CustomLoss


class IdentityLoss(CustomLoss):
    """Cosine identity loss based on frozen ArcFace face embeddings.

    Compares the face identity of two batches of images by extracting
    ArcFace embeddings via InsightFace and computing cosine distance.

    Input tensors are expected to be float in [0, 1] or [-1, 1], shaped
    (B, C, H, W) with RGB channel order.
    """

    def __init__(
        self,
        model_name: IdentityModel = IdentityModel.BUFFALO_L,
        providers: list[ONNXProvider] | ONNXProvider = DEFAULT_ONNX_PROVIDERS,
        ctx_id: int = -1,
        det_size: tuple[int, int] = (640, 640),
        weight: t.Optional[torch.Tensor] = None,
    ) -> None:
        """Initialise the identity loss.

        Args:
            model_name (optional):
                InsightFace model to use. Defaults to `buffalo_l`.
            providers (optional):
                ONNX Runtime execution providers, tried in order.
                Defaults to `[CUDA, CoreML, CPU]`.
            ctx_id (optional):
                GPU device index passed to `FaceAnalysis.prepare`.
                Defaults to `-1` (CPU/MPS). Use `0` on a CUDA machine.
            det_size (optional):
                Detection input resolution. Defaults to `(640, 640)`.
            weight (optional):
                Scalar weight tensor passed to the base class, defaults to `None`.
        """
        super().__init__(weight=weight)

        provider_strings = [
            p.value if isinstance(p, ONNXProvider) else p for p in providers
        ]

        self.app = FaceAnalysis(
            name=model_name.value,
            providers=provider_strings,
            root=Path(CACHE_DIR, "insightface"),
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def _get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract normalised ArcFace embeddings for a batch of images.

        Args:
            images:
                Float tensor of shape ``(B, C, H, W)``.

        Returns:
            Float tensor of shape ``(B, 512)`` on the same device as
            ``images``. Images where no face is detected get a zero
            embedding vector.
        """
        device = images.device
        batch_embeddings: list[np.ndarray] = []

        for img_tensor in images:
            bgr = tensor_to_bgr(img_tensor)
            faces = self.app.get(bgr)

            if faces:
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                batch_embeddings.append(face.normed_embedding)
            else:
                batch_embeddings.append(np.zeros(512, dtype=np.float32))

        return torch.from_numpy(np.stack(batch_embeddings)).to(device)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean cosine identity loss over a batch.

        Args:
            input:
                Predicted images, shape ``(B, C, H, W)``.
            target:
                Ground-truth images, shape ``(B, C, H, W)``.

        Returns:
            Scalar loss in ``[0, 1]``, where ``0`` means identical
            identities and ``1`` means orthogonal embeddings.
        """
        with torch.no_grad():
            emb_input = self._get_embeddings(input)
            emb_target = self._get_embeddings(target)

        similarity = F.cosine_similarity(emb_input, emb_target, dim=1)
        loss = (1.0 - similarity) / 2.0

        if self.weight is not None:
            loss = loss * self.weight

        return loss.mean()
