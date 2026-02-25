"""Configuration for the project."""

from dataclasses import dataclass

from .enums import IdentityBackend, IdentityModel, LogLevel


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

    backend: IdentityBackend = IdentityBackend.OPENCV
    """Backend for face detection."""

    model: IdentityModel = IdentityModel.VGG_FACE
    """Model for face recognition."""


@dataclass
class Args:
    """Top level arguments for the project."""

    identity: IdentityLoss
    """Identity loss configuration"""

    parameters: Params
    """Project parameters"""

    log_level: LogLevel = LogLevel.INFO
    """Logging level"""
