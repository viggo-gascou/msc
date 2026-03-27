"""Configuration for the project."""

from dataclasses import dataclass, field

from .enums import IdentityModel, LogLevel, ONNXProvider

DEFAULT_ONNX_PROVIDERS = [ONNXProvider.CUDA, ONNXProvider.COREML, ONNXProvider.CPU]


@dataclass
class Params:
    """Configuration for the project."""

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
class IdentityLoss:
    """Configuration for the identity loss."""

    weight: float = 1.0
    """Weight for identity loss."""

    model: IdentityModel = IdentityModel.BUFFALO_L
    """InsightFace model pack to use."""

    providers: list[ONNXProvider] = field(
        default_factory=lambda: DEFAULT_ONNX_PROVIDERS
    )
    """ONNX Runtime execution providers, tried in order."""

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

    log_level: LogLevel = LogLevel.INFO
    """Logging level"""
