"""AdaFace differentiable embedding model based on IR-101.

Architecture sourced from:
https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/raw/main/models/iresnet/model.py
Weights loaded from model.safetensors at the same repo root.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    Sequential,
)

from ..torch_utils import tensor_to_bgr
from .base import FaceEmbedding
from .preprocessor import FacePreprocessor


def load_adaface(device: str = "cuda") -> "AdaFaceEmbedding":
    """Download weights and return a ready-to-use AdaFaceEmbedding.

    Args:
        device:
          PyTorch device string to load the model onto.

    Returns:
      AdaFaceEmbedding with pretrained weights loaded and set to eval mode.
    """
    filename = hf_hub_download(
        repo_id="minchul/cvlface_adaface_ir101_webface12m",
        filename="model.safetensors",
    )
    state = load_safetensors(filename=filename)
    state_clean = {k.removeprefix("model.net."): v for k, v in state.items()}

    backbone = IR101()
    backbone.load_state_dict(state_clean, strict=True)
    backbone.eval()

    model = AdaFaceEmbedding(backbone=backbone).to(device=device)
    model.eval()
    return model


class AdaFaceEmbedding(FaceEmbedding, nn.Module):
    """Differentiable AdaFace embedding model based on IR-101.

    Extends both FaceEmbedding and nn.Module. The forward method accepts
    preprocessed 112x112 crops and is differentiable, making it suitable
    for use as an identity loss during training. The embed method handles
    raw images by detecting and aligning on the fly (no gradients).
    """

    def __init__(self, backbone: "IR101") -> None:
        """Initialise AdaFaceEmbedding.

        Args:
            backbone:
              IR101 backbone to wrap.
        """
        super().__init__()
        self.backbone = backbone
        self._preprocessor: FacePreprocessor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised embeddings from preprocessed face crops.

        Args:
            x:
              Batch of aligned face images of shape (B, 3, 112, 112)
              normalised to [-1, 1].

        Returns:
          L2-normalised embeddings of shape (B, 512).
        """
        return F.normalize(self.backbone(x), dim=-1)

    def embed(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Detect, align, and embed a single raw image without gradients.

        Initialises a FacePreprocessor on first call.

        Args:
            img:
              BGR uint8 numpy array or float RGB tensor of shape (C, H, W).

        Returns:
          L2-normalised embedding of shape (512,).
        """
        if self._preprocessor is None:
            self._preprocessor = FacePreprocessor()
        bgr = tensor_to_bgr(image=img) if isinstance(img, torch.Tensor) else img
        crop = self._preprocessor.preprocess(img=bgr).unsqueeze(0)
        with torch.no_grad():
            return self.forward(x=crop).squeeze(0)


class IR101(Module):
    """IR-101 backbone for face recognition.

    Architecture mirrors the upstream iresnet model.py exactly so that
    weights load cleanly from the safetensors checkpoint.
    """

    def __init__(self, output_dim: int = 512) -> None:
        """Initialise IR101.

        Args:
            output_dim:
              Dimension of the output embedding.
        """
        super().__init__()
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        self.body = Sequential(
            BasicBlockIR(in_channel=64, depth=64, stride=2),
            BasicBlockIR(in_channel=64, depth=64, stride=1),
            BasicBlockIR(in_channel=64, depth=64, stride=1),
            BasicBlockIR(in_channel=64, depth=128, stride=2),
            *[BasicBlockIR(in_channel=128, depth=128, stride=1) for _ in range(12)],
            BasicBlockIR(in_channel=128, depth=256, stride=2),
            *[BasicBlockIR(in_channel=256, depth=256, stride=1) for _ in range(29)],
            BasicBlockIR(in_channel=256, depth=512, stride=2),
            BasicBlockIR(in_channel=512, depth=512, stride=1),
            BasicBlockIR(in_channel=512, depth=512, stride=1),
        )
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(0.4),
            Flatten(),
            Linear(512 * 7 * 7, output_dim),
            BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the backbone.

        Args:
            x:
              Input tensor of shape (B, 3, 112, 112).

        Returns:
          Feature tensor of shape (B, output_dim).
        """
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)


class BasicBlockIR(Module):
    """IR residual block with pre-activation BatchNorm.

    Uses MaxPool shortcut when spatial dimensions are unchanged to avoid
    introducing extra parameters on the identity path.
    """

    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        """Initialise BasicBlockIR.

        Args:
            in_channel:
              Number of input channels.
            depth:
              Number of output channels.
            stride:
              Stride applied to the second conv and the shortcut.
        """
        super().__init__()
        self.shortcut_layer: MaxPool2d | Sequential
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block.

        Args:
            x:
              Input feature map.

        Returns:
          Output feature map with residual added.
        """
        return self.res_layer(x) + self.shortcut_layer(x)
