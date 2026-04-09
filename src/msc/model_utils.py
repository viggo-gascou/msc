"""Model utilities."""

import typing as t

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from omegaconf import DictConfig
from transformers import CLIPTextModel, CLIPTokenizer


def load_model(
    params: DictConfig, ip_cfg: DictConfig
) -> tuple[
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    CLIPTextModel,
    CLIPTokenizer,
    t.Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler],
]:
    """Load the model and return the pipeline and its components.

    Args:
        params:
            Model parameters.
        ip_cfg:
            IP adapter configuration.

    Returns:
        A tuple containing the pipeline and its components.

    Raises:
        ValueError: If the pipeline fails to load.

    """
    pipeline = StableDiffusionPipeline.from_pretrained(params.unet_model)
    if pipeline is None:
        raise ValueError("Failed to load pipeline")
    pipeline.load_ip_adapter(
        ip_cfg.repo,
        subfolder="",
        weight_name=ip_cfg.weight_id,
        image_encoder_folder=None,
    )
    pipeline.set_ip_adapter_scale(scale=0.6)

    unet: UNet2DConditionModel = pipeline.unet
    vae: AutoencoderKL = pipeline.vae
    text_encoder: CLIPTextModel = pipeline.text_encoder
    tokenizer: CLIPTokenizer = pipeline.tokenizer
    scheduler: t.Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler] = (
        pipeline.scheduler
    )

    return pipeline, unet, vae, text_encoder, tokenizer, scheduler


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
