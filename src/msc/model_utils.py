"""Model utilities."""

import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inference_pipeline import AUIPAdapterPipeline

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from .au_adapter import (
    MLPProjModel,
    SourceConditionedUNet,
    load_au_adapter,
    load_ip_adapter_weights,
    setup_unet_processors,
)


def expand_unet_in_channels(unet: UNet2DConditionModel) -> None:
    """Expand UNet conv_in from 4 to 8 input channels for source conditioning.

    The first 4 channels keep pretrained weights (noisy target latents).
    The last 4 channels are zero-initialised (clean source latents), so the
    model starts from the pretrained checkpoint and gradually learns to use
    the source conditioning signal.

    Args:
        unet: UNet2DConditionModel to modify in-place.
    """
    old = unet.conv_in
    new_conv = nn.Conv2d(
        8, old.out_channels, kernel_size=old.kernel_size, padding=old.padding
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4].copy_(old.weight)
        new_conv.bias.copy_(old.bias)
    unet.conv_in = new_conv
    unet.config["in_channels"] = 8


def load_model(
    params: DictConfig, ip_cfg: DictConfig
) -> tuple[
    SourceConditionedUNet,
    AutoencoderKL,
    CLIPTextModel,
    CLIPTokenizer,
    t.Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler],
    nn.ModuleDict,
    MLPProjModel,
]:
    """Load the pipeline and set up unified AU+IP-Adapter attention processors.

    Args:
        params:
            Model parameters.
        ip_cfg:
            IP adapter configuration.

    Returns:
        A tuple of (unet, vae, text_encoder, tokenizer, scheduler, au_procs,
        face_proj).

    Raises:
        ValueError: If the pipeline fails to load.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(params.unet_model)
    if pipeline is None:
        raise ValueError("Failed to load pipeline")

    unet: UNet2DConditionModel = pipeline.unet

    # Expand conv_in to 8 channels for source image conditioning
    expand_unet_in_channels(unet)

    vae: AutoencoderKL = pipeline.vae
    text_encoder: CLIPTextModel = pipeline.text_encoder
    tokenizer: CLIPTokenizer = pipeline.tokenizer
    scheduler: t.Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler] = (
        pipeline.scheduler
    )

    # Replace all cross-attention processors with unified AU+IP-Adapter ones
    au_procs = setup_unet_processors(unet)

    # Load pretrained IP-Adapter weights; get back the face projector
    ip_adapter_path = hf_hub_download(repo_id=ip_cfg.repo, filename=ip_cfg.weight_id)
    face_proj = load_ip_adapter_weights(unet, ip_adapter_path)

    wrapped_unet = SourceConditionedUNet(unet)

    return wrapped_unet, vae, text_encoder, tokenizer, scheduler, au_procs, face_proj


def load_inference_pipeline(
    params: DictConfig, ip_cfg: DictConfig, au_ckpt_path: str, device: str = "cuda"
) -> "AUIPAdapterPipeline":
    """Load a trained AUIPAdapterPipeline ready for inference.

    Loads the base Stable Diffusion weights, installs AU+IP-Adapter processors,
    and restores trained AU adapter weights from a checkpoint.

    Args:
        params:
            Model parameters (`unet_model`).
        ip_cfg:
            IP-Adapter config (`repo`, `weight_id`).
        au_ckpt_path:
            Path to the AU adapter safetensors checkpoint produced by
            `save_au_adapter`.
        device:
            Target device string (e.g. `"cuda"` or `"cpu"`). Weights are
            loaded in bfloat16 on CUDA and float32 on CPU.

    Returns:
        `AUIPAdapterPipeline` with all weights loaded.

    Raises:
        ValueError: If the base pipeline fails to load.
    """
    from .enums import ONNXProvider
    from .face_embeddings.arcface import ArcFaceEmbedding
    from .inference_pipeline import AUIPAdapterPipeline

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        params.unet_model,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    )
    if pipeline is None:
        raise ValueError("Failed to load pipeline")

    ip_adapter_path = hf_hub_download(repo_id=ip_cfg.repo, filename=ip_cfg.weight_id)

    # Expand conv_in to 8 channels for source image conditioning
    expand_unet_in_channels(pipeline.unet)

    # load_au_adapter sets up processors, loads IP-Adapter weights, and restores
    # AU adapter weights — but doesn't return face_proj.
    au_encoder, identity_adapter, _ = load_au_adapter(
        au_ckpt_path, pipeline.unet, ip_adapter_path
    )
    # Load face_proj from the IP-Adapter checkpoint (idempotent, weights already
    # in the UNet from load_au_adapter above).
    face_proj = load_ip_adapter_weights(pipeline.unet, ip_adapter_path)

    # Wrap UNet for source latent concatenation
    pipeline.unet = SourceConditionedUNet(pipeline.unet)

    ctx_id = 0 if device == "cuda" else -1
    arcface_extractor = ArcFaceEmbedding(
        providers=[ONNXProvider.CUDA, ONNXProvider.CPU], ctx_id=ctx_id
    )

    auip_pipeline = AUIPAdapterPipeline.from_pipeline(
        pipeline, au_encoder, identity_adapter, face_proj, arcface_extractor
    )
    auip_pipeline.to(device)
    return auip_pipeline


def freeze_model_layers(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    au_procs: nn.ModuleDict,
) -> tuple[UNet2DConditionModel, AutoencoderKL, CLIPTextModel, nn.ModuleDict]:
    """Freeze the layers of the given models.

    Args:
        unet:
            The UNet model.
        vae:
            The VAE model.
        text_encoder:
            The text encoder model.
        au_procs:
            The AU projection modules.

    Returns:
        The frozen models.
    """
    for model in [unet, vae, text_encoder]:
        for p in model.parameters():
            p.requires_grad = False
    if unet.encoder_hid_proj is not None:
        for p in unet.encoder_hid_proj.parameters():
            p.requires_grad = False

    # Unfreeze AU projection weights only
    for proc in au_procs.values():
        for name, p in proc.named_parameters():
            if "to_k_au" in name or "to_v_au" in name:
                p.requires_grad = True

    # Unfreeze conv_in so it can learn to process 8-channel input
    for p in unet.conv_in.parameters():
        p.requires_grad = True

    return unet, vae, text_encoder, au_procs
