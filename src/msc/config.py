"""Configuration for the project."""

from dataclasses import dataclass, field

from .enums import LogLevel, ONNXProvider

DEFAULT_ONNX_PROVIDERS = [ONNXProvider.CUDA, ONNXProvider.COREML, ONNXProvider.CPU]


@dataclass
class DataloaderParams:
    """Configuration for the dataloader."""

    num_workers: int = 2
    """Number of workers for data loading"""


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

    augmentation_proba: float = 0.5
    """Probability of applying augmentations"""

    checkpoint_dir: str = "checkpoints"
    """Directory to save training checkpoints"""

    checkpoint_every: int = 1
    """Save a checkpoint every N epochs"""

    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps before an optimizer update."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    dataloader: DataloaderParams = field(default_factory=DataloaderParams)
    """Configuration for the dataloader"""


@dataclass
class IPAdapter:
    """Configuration for the IP adapter."""

    repo: str = "h94/IP-Adapter-FaceID"
    """The IP adapter repo to use."""

    weight_id: str = "ip-adapter-faceid-portrait-v11_sd15.bin"
    """The IP adapter weights file to use."""


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
class WandB:
    """Configuration for Weights & Biases logging."""

    enabled: bool = True
    """Whether to log to W&B"""

    project: str = "semedit"
    """W&B project name"""

    entity: str = "msc-semedit"
    """W&B entity (team or username)."""

    run_name: str | None = None
    """Optional run name. None lets W&B auto-generate one."""


@dataclass
class Args:
    """Top level arguments for the project."""

    identity: IdentityLoss = field(default_factory=IdentityLoss)
    """Identity loss configuration"""

    parameters: Params = field(default_factory=Params)
    """Project parameters"""

    ip_adapter: IPAdapter = field(default_factory=IPAdapter)
    """IP-Adapter configuration"""

    wandb: WandB = field(default_factory=WandB)
    """Weights & Biases configuration"""

    log_level: LogLevel = LogLevel.INFO
    """Logging level"""
