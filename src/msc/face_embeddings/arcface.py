"""ArcFace embedding extractor using InsightFace buffalo_l."""

from pathlib import Path

import numpy as np
import torch

from ..constants import CACHE_DIR
from ..enums import IdentityModel, ONNXProvider
from ..torch_utils import tensor_to_bgr
from .base import FaceEmbedding


class ArcFaceEmbedding(FaceEmbedding):
    """InsightFace ArcFace embedding extractor.

    Wraps InsightFace buffalo_l to extract normed ArcFace embeddings.
    Non-differentiable (ONNX-based). Used for offline preprocessing and
    as the conditioning signal for the IP adapter, which was trained on
    these exact embeddings.
    """

    def __init__(
        self,
        model_name: IdentityModel = IdentityModel.BUFFALO_L,
        providers: list[ONNXProvider] | ONNXProvider = ONNXProvider.CPU,
        ctx_id: int = -1,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        """Initialise ArcFaceEmbedding.

        Args:
            model_name:
              InsightFace model to use. Defaults to buffalo_l.
            providers:
              ONNX Runtime execution providers, tried in order.
            ctx_id:
              InsightFace GPU context ID. Use -1 for CPU.
            det_size:
              Detection input resolution.
        """
        from insightface.app import FaceAnalysis

        provider_strings = [
            p.value if isinstance(p, ONNXProvider) else p
            for p in (providers if isinstance(providers, list) else [providers])
        ]
        self.app = FaceAnalysis(
            name=model_name.value,
            providers=provider_strings,
            root=Path(CACHE_DIR, "insightface"),
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def embed(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract a normalised ArcFace embedding from a single image.

        Returns a zero vector if no face is detected.

        Args:
            img:
              BGR uint8 numpy array or float RGB tensor of shape (C, H, W).

        Returns:
          Normalised embedding of shape (512,).
        """
        bgr = tensor_to_bgr(image=img) if isinstance(img, torch.Tensor) else img
        faces = self.app.get(bgr)
        if not faces:
            return torch.zeros(512)
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return torch.from_numpy(face.normed_embedding)
