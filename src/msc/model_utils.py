"""Model utilities."""

import typing as t

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from .au_adapter import load_ip_adapter_weights, setup_unet_processors


def load_model(
    params: DictConfig, ip_cfg: DictConfig
) -> tuple[
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    CLIPTextModel,
    CLIPTokenizer,
    t.Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler],
    nn.ModuleDict,
]:
    """Load the pipeline and set up unified AU+IP-Adapter attention processors.

    Args:
        params:
            Model parameters.
        ip_cfg:
            IP adapter configuration.

    Returns:
        A tuple of (pipeline, unet, vae, text_encoder, tokenizer, scheduler,
        au_procs).

    Raises:
        ValueError: If the pipeline fails to load.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(params.unet_model)
    if pipeline is None:
        raise ValueError("Failed to load pipeline")

    unet: UNet2DConditionModel = pipeline.unet
    vae: AutoencoderKL = pipeline.vae
    text_encoder: CLIPTextModel = pipeline.text_encoder
    tokenizer: CLIPTokenizer = pipeline.tokenizer
    scheduler: t.Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler] = (
        pipeline.scheduler
    )

    # Replace all cross-attention processors with unified AU+IP-Adapter ones
    au_procs = setup_unet_processors(unet)

    # Load pretrained IP-Adapter weights into the processors
    ip_adapter_path = hf_hub_download(repo_id=ip_cfg.repo, filename=ip_cfg.weight_id)
    load_ip_adapter_weights(unet, ip_adapter_path)

    return pipeline, unet, vae, text_encoder, tokenizer, scheduler, au_procs


def freeze_model_layers(
    unet: UNet2DConditionModel, vae: AutoencoderKL, text_encoder: CLIPTextModel
) -> tuple[UNet2DConditionModel, AutoencoderKL, CLIPTextModel]:
    """Freeze the layers of the given models.

    Args:
        unet:
            The UNet model.
        vae:
            The VAE model.
        text_encoder:
            The text encoder model.

    Returns:
        The frozen models.
    """
    for model in [unet, vae, text_encoder]:
        for p in model.parameters():
            p.requires_grad = False
    if unet.encoder_hid_proj is not None:
        for p in unet.encoder_hid_proj.parameters():
            p.requires_grad = False
    return unet, vae, text_encoder
