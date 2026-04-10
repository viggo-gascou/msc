"""MSCPipeline: Stable Diffusion with AU expression + FaceID identity conditioning."""

from __future__ import annotations

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
)

from .au_adapter import AUEncoder, IdentityAdapter, MLPProjModel


class AUIPAdapterPipeline(StableDiffusionPipeline):
    """StableDiffusionPipeline extended with AU expression and FaceID conditioning.

    Use `from_pipeline` to construct an instance from an existing
    `StableDiffusionPipeline` that already has `AUIPAttnProcessor` s
    installed in its UNet.

    At inference the pipeline:

    1. Encodes AU values via `AUEncoder.encode` — produces uncond‖cond tokens
       for classifier-free guidance.
    2. Personalises the conditional AU tokens with `IdentityAdapter` so the
       expression encoding is subject-specific.
    3. Projects the ArcFace embedding to IP-Adapter face tokens via
       `MLPProjModel`.
    4. Delegates the full denoising loop to the parent `StableDiffusionPipeline`
       via `cross_attention_kwargs`.
    """

    @classmethod
    def from_pipeline(
        cls,
        pipeline: StableDiffusionPipeline,
        au_encoder: AUEncoder,
        identity_adapter: IdentityAdapter,
        face_proj: MLPProjModel,
    ) -> AUIPAdapterPipeline:
        """Create the class from a prepared pipeline and AU adapter components.

        Args:
            pipeline:
                Base `StableDiffusionPipeline` with `AUIPAttnProcessor` s
                already installed in its UNet (via `setup_unet_processors`).
            au_encoder: Trained AUEncoder.
            identity_adapter: Trained IdentityAdapter.
            face_proj: Pretrained face projector loaded from IP-Adapter checkpoint.

        Returns:
            AUIPAdapterPipeline ready for inference.
        """
        instance = cls(**pipeline.components)
        instance.au_encoder = au_encoder
        instance.identity_adapter = identity_adapter
        instance.face_proj = face_proj
        return instance

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        aus: torch.Tensor,
        arcface_embeds: torch.Tensor,
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

        Args:
            prompt: Text prompt(s).
            aus: AU intensity tensor of shape (B, 23).
            arcface_embeds: ArcFace identity embedding of shape (B, 512).
            height: Output image height in pixels.
            width: Output image width in pixels.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale applied to text conditioning.
            negative_prompt: Optional negative prompt(s).
            au_scale: Expression conditioning strength (applied inside the
                attention processor).
            id_scale: Identity conditioning strength.
            **kwargs: Forwarded to `StableDiffusionPipeline.__call__`.

        Returns:
            `StableDiffusionPipelineOutput` containing generated images.
        """
        # Encode AUs for CFG: (2B, num_au_tokens, 768) — uncond first, cond second.
        # AUIPAttnProcessor interpolates at attention time:
        #   output = uncond + au_scale * (cond - uncond)
        au_tokens = self.au_encoder.encode(aus)

        # Personalise only the conditional half with subject identity
        uncond_au, cond_au = au_tokens.chunk(2, dim=0)
        cond_au = self.identity_adapter(cond_au, arcface_embeds)
        au_tokens = torch.cat([uncond_au, cond_au], dim=0)

        # Project ArcFace to IP-Adapter face tokens: (B, num_id_tokens, 768)
        id_tokens = self.face_proj(arcface_embeds)

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
