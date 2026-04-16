"""MSCPipeline: Stable Diffusion with AU expression + FaceID identity conditioning."""

from __future__ import annotations

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
)
from PIL import Image

from .au_adapter import AUEncoder, IdentityAdapter, MLPProjModel
from .constants import BP4D_AU_COLUMNS
from .face_embeddings.arcface import ArcFaceEmbedding
from .torch_utils import tensor_to_bgr


class AUIPAdapterPipeline(StableDiffusionPipeline):
    """StableDiffusionPipeline extended with AU expression and FaceID conditioning.

    Use `from_pipeline` to construct an instance from an existing
    `StableDiffusionPipeline` that already has `AUIPAttnProcessor` s
    installed in its UNet.

    At inference the pipeline:

    1. Extracts an ArcFace embedding from the input image (or uses a
       pre-extracted embedding if provided directly).
    2. Encodes AU values via `encode_aus` — produces uncond‖cond tokens
       for classifier-free guidance.
    3. Personalises the conditional AU tokens with `IdentityAdapter` so the
       expression encoding is subject-specific.
    4. Projects the ArcFace embedding to IP-Adapter face tokens via
       `encode_image`.
    5. Delegates the full denoising loop to the parent `StableDiffusionPipeline`
       via `cross_attention_kwargs`.
    """

    @classmethod
    def from_pipeline(
        cls,
        pipeline: StableDiffusionPipeline,
        au_encoder: AUEncoder,
        identity_adapter: IdentityAdapter,
        face_proj: MLPProjModel,
        arcface_extractor: ArcFaceEmbedding | None = None,
    ) -> AUIPAdapterPipeline:
        """Create the class from a prepared pipeline and AU adapter components.

        Args:
            pipeline:
                Base `StableDiffusionPipeline` with `AUIPAttnProcessor` s
                already installed in its UNet (via `setup_unet_processors`).
            au_encoder: Trained AUEncoder.
            identity_adapter: Trained IdentityAdapter.
            face_proj: Pretrained face projector loaded from IP-Adapter checkpoint.
            arcface_extractor:
                Optional `ArcFaceEmbedding` used to extract embeddings from raw
                images at call time. If ``None``, ``arcface_embeds`` must be
                supplied directly to ``__call__``.

        Returns:
            AUIPAdapterPipeline ready for inference.
        """
        instance = cls(**pipeline.components)
        instance.au_encoder = au_encoder
        instance.identity_adapter = identity_adapter
        instance.face_proj = face_proj
        instance.arcface_extractor = arcface_extractor
        return instance

    def encode_identity(
        self, arcface_embeds: torch.Tensor, do_cfg: bool = True
    ) -> torch.Tensor:
        """Project ArcFace embeddings to IP-Adapter face tokens.

        For CFG the uncond pass projects zeros through ``face_proj`` so the
        attention processor sees a neutral identity for the unconditional branch.

        Args:
            arcface_embeds: ArcFace embeddings of shape (B, 512).
            do_cfg: Whether to prepend uncond tokens for classifier-free guidance.

        Returns:
            Face tokens of shape (B, num_tokens, dim) without CFG or
            (2B, num_tokens, dim) with CFG — uncond first, cond second.
        """
        id_tokens = self.face_proj(arcface_embeds)
        if do_cfg:
            uncond_id_tokens = self.face_proj(torch.zeros_like(arcface_embeds))
            id_tokens = torch.cat([uncond_id_tokens, id_tokens], dim=0)
        return id_tokens

    def encode_aus(
        self,
        aus: torch.Tensor | dict[str, float],
        arcface_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Encode AU values for classifier-free guidance.

        Encodes both unconditional (zero AUs) and conditional AU tokens, then
        personalises the conditional half with the subject's identity via
        ``IdentityAdapter``.

        Args:
            aus:
                Either a full AU tensor of shape (B, num_aus), or a dict mapping
                AU names (e.g. ``{"AU12": 1.0, "AU06": 2.0}``) to intensities.
                Unspecified AUs default to 0 (neutral/absent).
            arcface_embeds: ArcFace embeddings of shape (B, 512).

        Returns:
            AU tokens of shape (2B, num_au_tokens, dim) — uncond first, cond second.
        """
        if isinstance(aus, dict):
            values = [aus.get(col, 0.0) for col in BP4D_AU_COLUMNS]
            aus = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        aus = aus.to(arcface_embeds.device)
        au_tokens = self.au_encoder.encode(aus)
        uncond_au, cond_au = au_tokens.chunk(2, dim=0)
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
        negative_prompt: str | list[str] | None = None,
        au_scale: float = 1.0,
        id_scale: float = 0.6,
        **kwargs,
    ) -> StableDiffusionPipelineOutput:
        """Generate images conditioned on AU values and a FaceID embedding.

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
            negative_prompt: Optional negative prompt(s).
            au_scale: Expression conditioning strength.
            id_scale: Identity conditioning strength.
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
                image = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB→BGR
            elif isinstance(image, torch.Tensor):
                image = tensor_to_bgr(image)
            arcface_embeds = self.arcface_extractor.embed(image).unsqueeze(0).to(device)
        else:
            arcface_embeds = arcface_embeds.to(device)

        do_cfg = guidance_scale > 1.0

        au_tokens = self.encode_aus(aus, arcface_embeds)
        id_tokens = self.encode_identity(arcface_embeds, do_cfg=do_cfg)

        return super().__call__(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={
                "id_embedding": id_tokens,
                "id_scale": id_scale,
                "au_embedding": au_tokens,
                "au_scale": au_scale,
            },
            **kwargs,
        )
