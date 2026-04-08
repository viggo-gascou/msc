"""Configuration for the project."""

from dataclasses import dataclass, field

from .enums import LogLevel, ONNXProvider

DEFAULT_ONNX_PROVIDERS = [ONNXProvider.CUDA, ONNXProvider.COREML, ONNXProvider.CPU]


@dataclass
class Params:
    """Configuration for the project."""

    unet_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    """The UNet model id to use"""

    learning_rate: float = 0.001
    """Learning rate for optimizer"""

    batch_size: int = 32
    """Number of samples per batch"""

    epochs: int = 10
    """Number of training epochs"""

    patience: int = 5
    """Early stopping patience"""

    early_stopping: bool = True
    """Whether to use early stopping"""


@dataclass
class IPAdapter:
    """Configuration for the IP adapter."""

    repo: str = "h94/IP-Adapter-FaceID"
    """The IP adapter repo to use"""

    weight_id: str = "ip-adapter-faceid_sd15.bin"
    """The IP adapter weight id to use"""


@dataclass
class IdentityLoss:
    """Configuration for the identity loss."""

    weight: float = 1.0
    """Weight for identity loss."""

    ctx_id: int = -1
    """GPU device index (-1 for CPU)."""

    det_size: tuple[int, int] = (640, 640)
    """Face detection input resolution."""


@dataclass
class Args:
    """Top level arguments for the project."""

    identity: IdentityLoss
    """Identity loss configuration"""

    parameters: Params
    """Project parameters"""

    ip_adapter: IPAdapter = field(default_factory=IPAdapter)
    """IP-Adapter configuration"""

    log_level: LogLevel = LogLevel.INFO
    """Logging level"""
