"""Utility functions for working with PyTorch tensors and images."""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from torchvision.io import decode_image


def load_image_as_tensor(
    path: str | Path, dtype: torch.dtype = torch.uint16, scale: bool = False
) -> torch.Tensor:
    """Load an image from a given path as a tensor.

    Args:
        path: Path to the image file.
        dtype: Data type of the tensor.
        scale: Whether to scale the tensor to the range [0, 1].

    Returns:
        The loaded image as a tensor.
    """
    if isinstance(path, Path):
        path = str(path)
    return F.to_dtype(decode_image(path), dtype=dtype, scale=scale)


def tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    """Convert a single (C, H, W) float tensor to a BGR uint8 array.

    Handles both [0, 1] and [-1, 1] input ranges.

    Args:
        image:
            Float tensor of shape `(C, H, W)`.

    Returns:
        BGR uint8 array of shape `(H, W, 3)`.
    """
    img = image.detach().cpu().float()
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    img = (img * 255).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).numpy()  # (H, W, C) RGB
    return img[:, :, ::-1].copy()  # RGB -> BGR
