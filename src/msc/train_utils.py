"""Training, validation, and evaluation loops for the AU adapter."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.training_utils import compute_snr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .au_adapter import AUEncoder, IdentityAdapter, prefixed
from .config import Params


def forward_batch(
    batch: dict,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler | LMSDiscreteScheduler | PNDMScheduler,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    device: torch.device,
    params: Params,
) -> torch.Tensor:
    """Shared forward pass for train, val, and test.

    Args:
        batch: BP4DSample batch from the dataloader.
        unet: UNet2DConditionModel with AUAttnProcessors.
        vae: VAE encoder/decoder.
        text_encoder: CLIP text encoder.
        tokenizer: CLIP tokenizer.
        scheduler: Noise scheduler.
        au_encoder: AU conditioning encoder.
        identity_adapter: Identity-conditioned AU token adapter.
        device: Target device.
        params: Training parameters.

    Returns:
        Scalar x0 loss between the predicted clean image (recovered from noisy
        source) and the target latents, optionally weighted by Min-SNR.
    """
    vae_dtype = next(vae.parameters()).dtype
    src_pixels = batch["image"].to(device=device, dtype=vae_dtype)
    arcface_embeds = batch["arcface"].to(device=device, dtype=vae_dtype)

    if params.reconstruction:
        # Reconstruct source from itself; condition on source AUs so the model
        # learns to associate AU values with the appearance they produce.
        au_values = batch["aus"].to(device)
    else:
        au_values = batch["target_aus"].to(device)

    # Replace NaN AU values (uncoded frames) with 0 — "AU not active".
    # NaN would propagate through au_encoder and poison the entire forward pass.
    au_values = au_values.nan_to_num(0.0)

    # Per-sample CFG dropout: zero out AU values with probability cfg_dropout_prob
    # so the model learns both conditional and unconditional distributions.
    # Only applied during training — val/test always evaluate full conditioning.
    if params.cfg_dropout_prob > 0 and unet.training:
        keep = torch.rand(au_values.shape[0], device=device) >= params.cfg_dropout_prob
        au_values = au_values * keep.to(dtype=au_values.dtype).unsqueeze(1)
        arcface_embeds = arcface_embeds * keep.to(dtype=arcface_embeds.dtype).unsqueeze(
            1
        )

    au_tokens = au_encoder(au_values)
    au_tokens = identity_adapter(au_tokens, arcface_embeds)

    src_latents = (
        vae.encode(src_pixels).latent_dist.sample() * vae.config.scaling_factor
    )
    noise = torch.randn_like(src_latents)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (src_latents.shape[0],), device=device
    ).long()
    noisy_src = scheduler.add_noise(
        original_samples=src_latents, noise=noise, timesteps=timesteps
    )

    ids = tokenizer(
        [""] * src_latents.shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    cond = text_encoder(ids).last_hidden_state

    pred_eps = unet(  # type: ignore[not-callable]
        sample=noisy_src,
        timestep=timesteps,
        encoder_hidden_states=cond,
        cross_attention_kwargs={"au_embedding": au_tokens, "au_scale": 1.0},
    ).sample

    # Recover the implied clean image from the noisy source and predicted noise.
    # Comparing against target latents directly supervises expression transfer —
    # the model must steer denoising from source toward the target expression.
    if params.snr_gamma < 0:
        snr_weights = None
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is
        # slightly changed. This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(scheduler, timesteps)
        snr_weights = torch.stack(
            [snr, params.snr_gamma * torch.ones_like(timesteps)], dim=1
        ).min(dim=1)[0]

        # we are either predicting the noise, epsilon (used in SD 1.5), or
        # "velocity" (used in SD 2.x)
        if scheduler.config.prediction_type == "epsilon":
            snr_weights = snr_weights / snr
        elif scheduler.config.prediction_type == "v_prediction":
            snr_weights = snr_weights / (snr + 1)

    alpha_t = (
        scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
        .to(pred_eps)
        .view(-1, 1, 1, 1)
    )
    pred_x0 = (noisy_src - (1 - alpha_t).sqrt() * pred_eps) / alpha_t.sqrt()
    if params.reconstruction:
        target_latents = src_latents
    else:
        tgt_pixels = batch["target_image"].to(device=device, dtype=vae_dtype)
        target_latents = (
            vae.encode(tgt_pixels).latent_dist.sample() * vae.config.scaling_factor
        )
    x0_loss = F.mse_loss(pred_x0.float(), target_latents.float(), reduction="none")
    if snr_weights is not None:
        return (x0_loss.mean(dim=list(range(1, pred_x0.ndim))) * snr_weights).mean()
    return x0_loss.mean()


def train_one_epoch(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler | LMSDiscreteScheduler | PNDMScheduler,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    accelerator: Accelerator,
    device: torch.device,
    params: Params,
) -> float:
    """Train all trainable components for one epoch.

    Args:
        unet: UNet with frozen base weights and trainable AU projections.
        vae: Frozen VAE.
        text_encoder: Frozen text encoder.
        tokenizer: CLIP tokenizer.
        scheduler: Noise scheduler.
        au_encoder: Trainable AU encoder.
        identity_adapter: Trainable identity adapter.
        optimizer: AdamW optimizer.
        loader: Training dataloader.
        accelerator: Accelerate wrapper.
        device: Target device.
        params: Training parameters.

    Returns:
        Average training loss for the epoch.
    """
    unet.train()
    au_encoder.train()
    identity_adapter.train()
    epoch_loss = 0.0
    trainable_params = [p for pg in optimizer.param_groups for p in pg["params"]]
    disable_tqdm = not accelerator.is_main_process

    for batch in tqdm(
        loader, desc="Training", unit="batch", leave=False, disable=disable_tqdm
    ):
        with accelerator.accumulate(unet, au_encoder, identity_adapter):
            loss = forward_batch(
                batch,
                unet,
                vae,
                text_encoder,
                tokenizer,
                scheduler,
                au_encoder,
                identity_adapter,
                device,
                params,
            )
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, params.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        gathered_loss = accelerator.gather(loss.detach()).mean()
        epoch_loss += gathered_loss.item()
        if accelerator.sync_gradients:
            accelerator.log({"train/batch_loss": gathered_loss.item()})

    return epoch_loss / len(loader)


def validate(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler | LMSDiscreteScheduler | PNDMScheduler,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    loader: DataLoader,
    accelerator: Accelerator,
    device: torch.device,
    params: Params,
) -> float:
    """Evaluate loss on the validation set.

    Args:
        unet: UNet model.
        vae: VAE encoder.
        text_encoder: Text encoder.
        tokenizer: CLIP tokenizer.
        scheduler: Noise scheduler.
        au_encoder: AU encoder.
        identity_adapter: Identity adapter.
        loader: Validation dataloader.
        accelerator: Accelerator.
        device: Target device
        params: Training parameters.

    Returns:
        Average validation loss.
    """
    unet.eval()
    au_encoder.eval()
    identity_adapter.eval()
    epoch_loss = 0.0
    disable_tqdm = not accelerator.is_main_process

    with torch.no_grad():
        for batch in tqdm(
            loader, desc="Validation", unit="batch", leave=False, disable=disable_tqdm
        ):
            loss = forward_batch(
                batch,
                unet,
                vae,
                text_encoder,
                tokenizer,
                scheduler,
                au_encoder,
                identity_adapter,
                device,
                params,
            )
            epoch_loss += loss.item()
            accelerator.log({"val/batch_loss": loss.item()})

    return epoch_loss / len(loader)


def evaluate(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler | LMSDiscreteScheduler | PNDMScheduler,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    loader: DataLoader,
    accelerator: Accelerator,
    device: torch.device,
    params: Params,
) -> float:
    """Evaluate loss on the test set.

    Args:
        unet: UNet model.
        vae: VAE encoder.
        text_encoder: Text encoder.
        tokenizer: CLIP tokenizer.
        scheduler: Noise scheduler.
        au_encoder: AU encoder.
        identity_adapter: Identity adapter.
        loader: Test dataloader.
        accelerator: Accelerator.
        device: Target device.
        params: Training parameters.

    Returns:
        Average test loss.
    """
    unet.eval()
    au_encoder.eval()
    identity_adapter.eval()
    epoch_loss = 0.0
    disable_tqdm = not accelerator.is_main_process

    with torch.no_grad():
        for batch in tqdm(
            loader, desc="Testing", unit="batch", leave=False, disable=disable_tqdm
        ):
            loss = forward_batch(
                batch,
                unet,
                vae,
                text_encoder,
                tokenizer,
                scheduler,
                au_encoder,
                identity_adapter,
                device,
                params,
            )
            epoch_loss += loss.item()
            accelerator.log({"test/batch_loss": loss.item()})

    return epoch_loss / len(loader)


def save_checkpoint(
    path: str,
    unet: UNet2DConditionModel,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    au_procs: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    patience_counter: int,
) -> None:
    """Save a resumable training checkpoint.

    Stores model weights (excluding pretrained IP-Adapter projections),
    optimizer state, and training metadata in a single file via torch.save.

    Args:
        path: Destination path for the checkpoint file.
        unet: UNet model (LoRA weights extracted if present).
        au_encoder: AU encoder module.
        identity_adapter: Identity adapter module.
        au_procs: AU attention processors (ModuleDict).
        optimizer: Current optimizer.
        epoch: Last completed epoch (0-indexed).
        best_val_loss: Best validation loss seen so far.
        patience_counter: Current early-stopping patience count.
    """
    lora_sd = {k: v for k, v in unet.state_dict().items() if "lora_" in k}
    flat: dict[str, torch.Tensor] = {}
    flat.update(prefixed(au_encoder.state_dict(), "au_encoder"))
    flat.update(prefixed(identity_adapter.state_dict(), "identity_adapter"))
    flat.update(prefixed(au_procs.state_dict(), "au_procs"))
    flat.update(prefixed(lora_sd, "lora"))

    torch.save(
        {
            "model": flat,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        },
        path,
    )


def load_checkpoint(
    path: str,
    unet: UNet2DConditionModel,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    au_procs: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, int]:
    """Load a training checkpoint in-place.

    Restores model weights and optimizer state into existing objects and
    returns the saved training metadata.

    Args:
        path: Path to the checkpoint file produced by save_checkpoint.
        unet: UNet model (LoRA weights restored if present in checkpoint).
        au_encoder: AU encoder to restore weights into.
        identity_adapter: Identity adapter to restore weights into.
        au_procs: AU processors to restore weights into.
        optimizer: Optimizer to restore state into.

    Returns:
        Tuple of (epoch, best_val_loss, patience_counter).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    flat: dict[str, torch.Tensor] = checkpoint["model"]

    def _strip(prefix: str) -> dict[str, torch.Tensor]:
        n = len(prefix) + 1
        return {
            k[n:].removeprefix("module."): v
            for k, v in flat.items()
            if k.startswith(prefix + ".")
        }

    au_encoder.load_state_dict(_strip("au_encoder"))
    identity_adapter.load_state_dict(_strip("identity_adapter"))
    au_procs.load_state_dict(_strip("au_procs"), strict=False)
    lora_sd = _strip("lora")
    if lora_sd:
        unet.load_state_dict(lora_sd, strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])

    return (
        checkpoint["epoch"],
        checkpoint["best_val_loss"],
        checkpoint["patience_counter"],
    )
