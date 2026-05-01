"""Configuration for the project."""

from dataclasses import dataclass
from typing import Annotated

import tyro

from .enums import LogLevel, ONNXProvider

DEFAULT_ONNX_PROVIDERS = [ONNXProvider.CUDA, ONNXProvider.COREML, ONNXProvider.CPU]


@dataclass
class DataloaderParams:
    """Configuration for the dataloader.

    Attributes:
        batch_size:
            Number of samples per batch. Defaults to 32.
        num_workers:
            Number of workers for data loading. Defaults to 2.
    """

    batch_size: int = 32
    num_workers: int = 2


@dataclass
class OptimizerParams:
    """Configuration for the AdamW optimizer.

    Attributes:
        learning_rate:
            Learning rate. Defaults to 1e-4.
        weight_decay:
            Weight decay (L2 regularisation). Defaults to 0.01.
        adam_beta1:
            AdamW beta1 (first moment decay). Defaults to 0.9.
        adam_beta2:
            AdamW beta2 (second moment decay). Defaults to 0.999.
        adam_eps:
            AdamW epsilon for numerical stability. Defaults to 1e-8.
    """

    learning_rate: Annotated[float, tyro.conf.arg(name="lr")] = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


@dataclass
class LoraParams:
    """Configuration for LoRA fine-tuning of the UNet.

    Attributes:
        enabled:
            Whether to apply LoRA to the UNet attention layers. Defaults to False.
        rank:
            LoRA rank (r). Higher rank = more capacity. Defaults to 32.
        alpha:
            LoRA scaling factor (lora_alpha). Defaults to None, which uses
            ``rank`` as the alpha (effective scale = 1.0).
        dropout:
            Dropout probability applied to LoRA layers. Defaults to 0.0.
    """

    enabled: bool = False
    rank: int = 32
    alpha: int | None = None
    dropout: float = 0.0


@dataclass
class Params:
    """Configuration for the project.

    Attributes:
        optimizer:
            Optimizer configuration.
        dataloader:
            Dataloader configuration.
        lora:
            LoRA fine-tuning configuration for the UNet.
        unet_model:
            The UNet model id to use.
            Defaults to 'stable-diffusion-v1-5/stable-diffusion-v1-5'.
        vae_model:
            Optional VAE model id to override the one bundled with the pipeline.
        epochs:
            Number of training epochs. Defaults to 10.
        patience:
            Early stopping patience. Defaults to 5.
        early_stopping:
            Whether to use early stopping. Defaults to True.
        augmentation_proba:
            Probability of applying augmentations. Defaults to 0.5.
        checkpoint_dir:
            Directory to save model checkpoints. Defaults to 'checkpoints'.
        checkpoint_every:
            Save a checkpoint every N epochs. Defaults to 1.
        gradient_accumulation_steps:
            Number of gradient accumulation steps before an optimizer update.
            Defaults to 1.
        max_grad_norm:
            Maximum gradient norm for clipping. Defaults to 1.0.
        x0_loss_weight:
            Weight for the x0 auxiliary loss (predicted clean image vs target latents).
            Defaults to 0.5.
        snr_gamma:
            Min-SNR loss weighting gamma (Section 3.4 of arxiv.org/abs/2303.09556).
            Set to a negative value to disable. Defaults to 5.0.
        cfg_dropout_prob:
            Probability of zeroing AU conditioning per sample during training
            (enables CFG guidance at inference via au_scale). Set to 0 to disable.
            Defaults to 0.1.
        gradient_checkpointing:
            Enable gradient checkpointing on the UNet to trade compute for VRAM.
            Defaults to False.
        max_frame_distance:
            Minimum temporal distance (frames) between source and target at the
            start of training. Anneals linearly to min_frame_distance over all
            epochs. Defaults to 50.
        min_frame_distance:
            Minimum temporal distance (frames) between source and target at the
            end of training. Defaults to 5.
        reconstruction:
            If True, use a reconstruction objective (denoise source back to
            itself conditioned on source AUs) instead of the transfer objective
            (denoise source toward a different target frame). Disables the frame
            distance curriculum. Defaults to False.
        resume_from:
            Path to a checkpoint .pt file to resume training from. When set,
            training resumes from that checkpoint's epoch and optimizer state.
            Defaults to None (start from scratch).
    """

    optimizer: OptimizerParams
    dataloader: DataloaderParams
    lora: LoraParams
    unet_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    vae_model: str | None = None
    epochs: int = 50
    patience: int = 5
    early_stopping: bool = True
    augmentation_proba: float = 0.5
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    x0_loss_weight: float = 0.5
    snr_gamma: float = 5.0
    cfg_dropout_prob: float = 0.1
    gradient_checkpointing: bool = False
    max_frame_distance: int = 50
    min_frame_distance: int = 5
    reconstruction: bool = False
    resume_from: str | None = None


@dataclass
class IPAdapterConfig:
    """Configuration for the IP adapter.

    Attributes:
        repo:
            The IP adapter repo to use. Defaults to 'h94/IP-Adapter-FaceID'.
        weight_id:
            The IP adapter weights file to use.
            Defaults to 'ip-adapter-faceid-portrait-v11_sd15.bin'.
    """

    repo: str = "h94/IP-Adapter-FaceID"
    weight_id: str = "ip-adapter-faceid-portrait-v11_sd15.bin"


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging.

    Attributes:
        enabled:
            Whether to log to W&B. Defaults to True.
        project:
            W&B project name. Defaults to 'semedit'.
        entity:
            W&B entity (team or username). Defaults to 'msc-semedit'.
        run_name:
            Optional run name. None lets W&B auto-generate one. Defaults to None.
    """

    enabled: bool = True
    project: str = "semedit"
    entity: str = "msc-semedit"
    run_name: str | None = None


@dataclass
class Args:
    """Top level arguments for the project.

    Attributes:
        parameters:
            Project parameters
        ip_adapter:
            IP-Adapter configuration
        wandb:
            Weights & Biases configuration
        log_level:
            Logging level
    """

    parameters: Params
    ip_adapter: IPAdapterConfig
    wandb: WandBConfig
    log_level: LogLevel = LogLevel.INFO
