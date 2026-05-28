"""AUAdapterPipeline: Stable Diffusion with AU expression + identity conditioning."""

from __future__ import annotations

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipelineOutput,
)
from PIL import Image

from .au_adapter import AUEncoder, IdentityAdapter
from .constants import PYFEAT_AU_COLUMNS, PYFEAT_AU_NAME_TO_IDX, PYFEAT_AU_SCALE
from .face_embeddings.arcface import ArcFaceEmbedding
from .torch_utils import tensor_to_bgr


class AUAdapterPipeline(StableDiffusionImg2ImgPipeline):
    """StableDiffusionPipeline extended with AU expression and identity conditioning.

    Use `from_pipeline` to construct an instance from an existing
    `StableDiffusionPipeline` that already has `AUAttnProcessor` s
    installed in its UNet.

    At inference the pipeline:

    1. Extracts an ArcFace embedding from the input image (or uses a
       pre-extracted embedding if provided directly).
    2. Encodes AU values via `encode_aus` — produces uncond‖cond tokens
       for classifier-free guidance.
    3. Personalises the conditional AU tokens with `IdentityAdapter` so the
       expression encoding is subject-specific.
    4. Delegates the full denoising loop to the parent `StableDiffusionPipeline`
       via `cross_attention_kwargs`.
    """

    @classmethod
    def from_pipeline(
        cls,
        pipeline: StableDiffusionImg2ImgPipeline | StableDiffusionPipeline,
        au_encoder: AUEncoder,
        identity_adapter: IdentityAdapter,
        arcface_extractor: ArcFaceEmbedding | None = None,
    ) -> AUAdapterPipeline:
        """Create the class from a prepared pipeline and AU adapter components.

        Args:
            pipeline:
                Base `StableDiffusionImg2ImgPipeline` with `AUAttnProcessor` s
                already installed in its UNet (via `setup_unet_processors`).
            au_encoder: Trained AUEncoder.
            identity_adapter: Trained IdentityAdapter.
            arcface_extractor:
                Optional `ArcFaceEmbedding` used to extract embeddings from raw
                images at call time. If ``None``, ``arcface_embeds`` must be
                supplied directly to ``__call__``.

        Returns:
            AUAdapterPipeline ready for inference.
        """
        instance = cls(**pipeline.components)
        instance.au_encoder = au_encoder
        instance.identity_adapter = identity_adapter
        instance.arcface_extractor = arcface_extractor
        return instance

    def encode_aus(
        self, aus: torch.Tensor | dict[str, float], arcface_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Encode AU values for classifier-free guidance.

        Encodes both unconditional (zero AUs, zero identity) and conditional AU
        tokens, then personalises both halves with ``IdentityAdapter`` — the
        unconditional half receives a zero ArcFace embedding to match the CFG
        dropout behaviour seen during training.

        Args:
            aus:
                Either a full AU tensor of shape (B, num_aus), or a dict mapping
                AU names (e.g. ``{"AU12": 1.0, "AU06": 2.0}``) to intensities.
                Unspecified AUs default to 0 (neutral/absent).
            arcface_embeds: ArcFace embeddings of shape (B, 512).

        Returns:
            AU tokens of shape (2B, num_au_tokens, dim) — uncond first, cond second.
        """
        dtype = next(self.au_encoder.parameters()).dtype
        if isinstance(aus, dict):
            values = [0.0] * len(PYFEAT_AU_COLUMNS)
            for au_name, val in aus.items():
                if (idx := PYFEAT_AU_NAME_TO_IDX.get(au_name)) is not None:
                    values[idx] = val
            aus = (
                torch.tensor(values, dtype=torch.float32)
                * torch.tensor(PYFEAT_AU_SCALE, dtype=torch.float32)
            ).unsqueeze(0)
        aus = aus.to(device=arcface_embeds.device, dtype=dtype)
        arcface_embeds = arcface_embeds.to(dtype=dtype)
        au_tokens = self.au_encoder.encode(aus)
        uncond_au, cond_au = au_tokens.chunk(2, dim=0)
        uncond_au = self.identity_adapter(uncond_au, torch.zeros_like(arcface_embeds))
        cond_au = self.identity_adapter(cond_au, arcface_embeds)
        return torch.cat([uncond_au, cond_au], dim=0)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        aus: torch.Tensor | dict[str, float],
        image: np.ndarray | torch.Tensor | Image.Image | None = None,
        arcface_embeds: torch.Tensor | None = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.5,
        negative_prompt: str | list[str] | None = None,
        au_scale: float = 1.0,
        **kwargs,
    ) -> StableDiffusionPipelineOutput:
        """Generate images conditioned on AU values and an identity embedding.

        Either ``image`` or ``arcface_embeds`` must be provided. If ``image``
        is given, the ArcFace embedding is extracted on the fly using the
        extractor passed to ``from_pipeline``.

        Args:
            prompt: Text prompt(s).
            aus: AU intensity tensor of shape (B, 23).
            image:
                Input face image to extract identity from. Accepts a BGR uint8
                numpy array, an RGB float tensor of shape (C, H, W), or a PIL
                Image. Mutually exclusive with ``arcface_embeds``.
            arcface_embeds:
                Pre-extracted ArcFace embedding of shape (B, 512). Takes
                precedence over ``image`` if both are supplied.
            height: Output image height in pixels.
            width: Output image width in pixels.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale applied to text conditioning.
            strength: Img2img strength — how much to noise the source image before
                denoising. 0.0 = no change, 1.0 = pure generation from noise. For AU
                editing, 0.4–0.6 is recommended.
            negative_prompt: Optional negative prompt(s).
            au_scale: Expression conditioning strength.
            **kwargs: Forwarded to `StableDiffusionPipeline.__call__`.

        Returns:
            `StableDiffusionPipelineOutput` containing generated images.

        Raises:
            ValueError: If neither ``image`` nor ``arcface_embeds`` is provided,
                or if ``image`` is provided but no extractor was set.
        """
        device = self._execution_device

        if arcface_embeds is None:
            if image is None:
                raise ValueError("Either `image` or `arcface_embeds` must be provided.")
            if self.arcface_extractor is None:
                raise ValueError(
                    "No arcface_extractor set. Pass one to `from_pipeline` or "
                    "provide `arcface_embeds` directly."
                )
            if isinstance(image, Image.Image):
                pil_image = image
                image = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
            elif isinstance(image, torch.Tensor):
                pil_image = Image.fromarray(
                    (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                image = tensor_to_bgr(image)
            else:
                pil_image = Image.fromarray(image[:, :, ::-1])
            arcface_embeds = self.arcface_extractor.embed(image).unsqueeze(0).to(device)
        else:
            arcface_embeds = arcface_embeds.to(device)
            if image is not None:
                if isinstance(image, Image.Image):
                    pil_image = image
                elif isinstance(image, torch.Tensor):
                    pil_image = Image.fromarray(
                        (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    )
                else:
                    pil_image = Image.fromarray(image[:, :, ::-1])
            else:
                pil_image = None

        au_tokens = self.encode_aus(aus, arcface_embeds)

        if pil_image is None:
            raise ValueError("`image` is required for img2img inference.")

        return super().__call__(
            prompt=prompt,
            image=pil_image,
            strength=strength,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={"au_embedding": au_tokens, "au_scale": au_scale},
            **kwargs,
        )
