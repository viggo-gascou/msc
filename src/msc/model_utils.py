"""Model utilities."""

import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inference_pipeline import AUIPAdapterPipeline

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from .au_adapter import (
    MLPProjModel,
    load_au_adapter,
    load_ip_adapter_weights,
    setup_unet_processors,
)
from .config import IPAdapterConfig, LoraParams, Params


def load_model(
    params: Params, ip_cfg: IPAdapterConfig
) -> tuple[
    UNet2DConditionModel,
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

    if params.vae_model is not None:
        pipeline.vae = AutoencoderKL.from_pretrained(params.vae_model)

    unet: UNet2DConditionModel = pipeline.unet
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

    return unet, vae, text_encoder, tokenizer, scheduler, au_procs, face_proj


def load_inference_pipeline(
    params: Params,
    ip_cfg: IPAdapterConfig,
    au_ckpt_path: str,
    device: str = "cuda",
    load_arcface: bool = True,
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
        load_arcface:
            If False, skip loading the ArcFace extractor (useful when
            arcface embeddings are supplied directly at inference time).

    Returns:
        `AUIPAdapterPipeline` with all weights loaded.

    Raises:
        ValueError: If the base pipeline fails to load.
    """
    from .inference_pipeline import AUIPAdapterPipeline

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        params.unet_model,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    )
    if pipeline is None:
        raise ValueError("Failed to load pipeline")

    if params.vae_model is not None:
        pipeline.vae = AutoencoderKL.from_pretrained(
            params.vae_model, torch_dtype=dtype
        )

    # EulerDiscreteScheduler supports custom timestep lists needed for img2img;
    # DDIMScheduler does not.
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    ip_adapter_path = hf_hub_download(repo_id=ip_cfg.repo, filename=ip_cfg.weight_id)

    # load_au_adapter sets up processors, loads IP-Adapter weights, and restores
    # AU adapter weights — but doesn't return face_proj.
    au_encoder, identity_adapter, _ = load_au_adapter(
        au_ckpt_path, pipeline.unet, ip_adapter_path
    )
    # Load face_proj from the IP-Adapter checkpoint (idempotent, weights already
    # in the UNet from load_au_adapter above).
    face_proj = load_ip_adapter_weights(pipeline.unet, ip_adapter_path)

    arcface_extractor = None
    if load_arcface:
        from .enums import ONNXProvider
        from .face_embeddings.arcface import ArcFaceEmbedding

        ctx_id = 0 if device == "cuda" else -1
        arcface_extractor = ArcFaceEmbedding(
            providers=[ONNXProvider.CUDA, ONNXProvider.CPU], ctx_id=ctx_id
        )

    auip_pipeline = AUIPAdapterPipeline.from_pipeline(
        pipeline, au_encoder, identity_adapter, face_proj, arcface_extractor
    )
    auip_pipeline.to(device)
    # diffusers' .to() only moves registered pipeline components; move the
    # custom modules explicitly so all tensors land on the same device/dtype.
    auip_pipeline.au_encoder = auip_pipeline.au_encoder.to(device=device, dtype=dtype)
    auip_pipeline.identity_adapter = auip_pipeline.identity_adapter.to(
        device=device, dtype=dtype
    )
    auip_pipeline.face_proj = auip_pipeline.face_proj.to(device=device, dtype=dtype)
    return auip_pipeline


def freeze_model_layers(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    au_procs: nn.ModuleDict,
    lora_cfg: LoraParams,
) -> tuple[UNet2DConditionModel, AutoencoderKL, CLIPTextModel, nn.ModuleDict]:
    """Freeze the layers of the given models, optionally adding LoRA to the UNet.

    Args:
        unet:
            The UNet model.
        vae:
            The VAE model.
        text_encoder:
            The text encoder model.
        au_procs:
            The AU projection modules.
        lora_cfg:
            LoRA configuration. If ``lora_cfg.enabled`` is True, LoRA adapters
            are injected into the UNet attention layers before freezing.

    Returns:
        The frozen models.
    """
    for model in [unet, vae, text_encoder]:
        for p in model.parameters():
            p.requires_grad = False
    if unet.encoder_hid_proj is not None:
        for p in unet.encoder_hid_proj.parameters():
            p.requires_grad = False

    if lora_cfg.enabled:
        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha if lora_cfg.alpha is not None else lora_cfg.rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=lora_cfg.dropout,
        )
        unet.add_adapter(adapter_config=lora_config, adapter_name="unet")

    # Unfreeze AU projection weights only
    for proc in au_procs.values():
        for name, p in proc.named_parameters():
            if "to_k_au" in name or "to_v_au" in name:
                p.requires_grad = True

    return unet, vae, text_encoder, au_procs
