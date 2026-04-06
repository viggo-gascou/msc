"""Face detection and alignment to the ArcFace canonical template."""

import cv2
import numpy as np
import torch
from insightface.utils import face_align

from .base import load_insightface


class FacePreprocessor:
    """Detect and align faces to the ArcFace canonical 112x112 template.

    Uses InsightFace RetinaFace for detection and norm_crop for the
    similarity transform, matching the preprocessing assumed during
    ArcFace and AdaFace training.
    """

    def __init__(self, ctx_id: int = 0, det_size: tuple[int, int] = (640, 640)) -> None:
        """Initialise FacePreprocessor.

        Args:
            ctx_id:
              InsightFace GPU context ID. Use -1 for CPU.
            det_size:
              Detection input resolution. Should match input image size for
              best detection results.
        """
        self.app = load_insightface(ctx_id=ctx_id, det_size=det_size)

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Detect, align, and normalise a single BGR image.

        Picks the largest detected face when multiple are present.

        Args:
            img:
              BGR uint8 image as returned by OpenCV.

        Returns:
          Aligned face tensor of shape (3, 112, 112) in [-1, 1].

        Raises:
            ValueError:
              If no face is detected in the image.
        """
        faces = self.app.get(img=img)
        if not faces:
            raise ValueError("No face detected")
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        aligned = face_align.norm_crop(img=img, landmark=face.kps, image_size=112)
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(aligned).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5
