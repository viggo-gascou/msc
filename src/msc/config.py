"""Configuration for the project."""

from dataclasses import dataclass

from .enums import LogLevel, ONNXProvider

DEFAULT_ONNX_PROVIDERS = [ONNXProvider.CUDA, ONNXProvider.COREML, ONNXProvider.CPU]


@dataclass
class DataloaderParams:
    """Configuration for the dataloader."""

    batch_size: int = 32
    """Number of samples per batch"""

    num_workers: int = 2
    """Number of workers for data loading"""


@dataclass
class OptimizerParams:
    """Configuration for the AdamW optimizer."""

    learning_rate: float = 1e-4
    """Learning rate"""

    weight_decay: float = 0.01
    """Weight decay (L2 regularisation)"""

    adam_beta1: float = 0.9
    """AdamW beta1 (first moment decay)"""

    adam_beta2: float = 0.999
    """AdamW beta2 (second moment decay)"""

    adam_eps: float = 1e-8
    """AdamW epsilon for numerical stability"""


@dataclass
class Params:
    """Configuration for the project."""

    optimizer: OptimizerParams
    """Optimizer configuration"""

    dataloader: DataloaderParams
    """Dataloader configuration"""

    unet_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    """The UNet model id to use"""

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

    gradient_checkpointing: bool = False
    """Enable gradient checkpointing on the UNet to reduce VRAM usage."""


@dataclass
class IPAdapter:
    """Configuration for the IP adapter."""

    repo: str = "h94/IP-Adapter-FaceID"
    """The IP adapter repo to use."""

    weight_id: str = "ip-adapter-faceid-portrait-v11_sd15.bin"
    """The IP adapter weights file to use."""


@dataclass
class WandBConfig:
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

    parameters: Params
    """Project parameters"""

    ip_adapter: IPAdapter
    """IP-Adapter configuration"""

    wandb: WandBConfig
    """Weights & Biases configuration"""

    log_level: LogLevel = LogLevel.INFO
    """Logging level"""
