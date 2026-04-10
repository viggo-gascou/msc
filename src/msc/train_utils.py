"""Training, validation, and evaluation loops for the AU adapter."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .au_adapter import AUEncoder, IdentityAdapter, MLPProjModel, prefixed


def forward_batch(
    batch: dict,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler | LMSDiscreteScheduler | PNDMScheduler,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    face_proj: MLPProjModel,
    device: torch.device,
) -> torch.Tensor:
    """Shared forward pass for train, val, and test.

    Args:
        batch: BP4DSample batch from the dataloader.
        unet: UNet2DConditionModel with AUIPAttnProcessors.
        vae: VAE encoder/decoder.
        text_encoder: CLIP text encoder.
        tokenizer: CLIP tokenizer.
        scheduler: Noise scheduler.
        au_encoder: AU conditioning encoder.
        identity_adapter: Identity-conditioned AU token adapter.
        face_proj: Pretrained MLP projecting ArcFace → IP-Adapter face tokens.
        device: Target device.

    Returns:
        Scalar MSE loss between predicted and actual noise.
    """
    vae_dtype = next(vae.parameters()).dtype
    proj_dtype = next(face_proj.parameters()).dtype
    pixel_values = batch["image"].to(device=device, dtype=vae_dtype)
    arcface_embeds = batch["arcface"].to(device=device, dtype=proj_dtype)
    au_values = batch["aus"].to(device)

    au_tokens = au_encoder(au_values)
    au_tokens = identity_adapter(au_tokens, arcface_embeds)
    id_tokens = face_proj(arcface_embeds)

    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
    ).long()
    noisy = scheduler.add_noise(
        original_samples=latents, noise=noise, timesteps=timesteps
    )

    ids = tokenizer(
        [""] * latents.shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    cond = text_encoder(ids).last_hidden_state

    pred = unet(  # type: ignore[not-callable]
        sample=noisy,
        timestep=timesteps,
        encoder_hidden_states=cond,
        cross_attention_kwargs={
            "id_embedding": id_tokens,
            "id_scale": 1.0,
            "au_embedding": au_tokens,
            "au_scale": 1.0,
        },
    ).sample
    return F.mse_loss(input=pred, target=noise)


def train_one_epoch(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler | LMSDiscreteScheduler | PNDMScheduler,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    face_proj: MLPProjModel,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    accelerator: Accelerator,
    device: torch.device,
    max_grad_norm: float = 1.0,
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
        face_proj: Frozen face projector (ArcFace → IP-Adapter tokens).
        optimizer: AdamW optimizer.
        loader: Training dataloader.
        accelerator: Accelerate wrapper.
        device: Target device.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Average training loss for the epoch.
    """
    unet.train()
    au_encoder.train()
    identity_adapter.train()
    epoch_loss = 0.0
    trainable_params = [p for pg in optimizer.param_groups for p in pg["params"]]

    for batch in tqdm(loader, desc="Training", unit="batch", leave=False):
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
                face_proj,
                device,
            )
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
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
    face_proj: MLPProjModel,
    loader: DataLoader,
    device: torch.device,
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
        face_proj: Face projector.
        loader: Validation dataloader.
        device: Target device.

    Returns:
        Average validation loss.
    """
    unet.eval()
    au_encoder.eval()
    identity_adapter.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", unit="batch", leave=False):
            loss = forward_batch(
                batch,
                unet,
                vae,
                text_encoder,
                tokenizer,
                scheduler,
                au_encoder,
                identity_adapter,
                face_proj,
                device,
            )
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDIMScheduler | LMSDiscreteScheduler | PNDMScheduler,
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    face_proj: MLPProjModel,
    loader: DataLoader,
    device: torch.device,
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
        face_proj: Face projector.
        loader: Test dataloader.
        device: Target device.

    Returns:
        Average test loss.
    """
    unet.eval()
    au_encoder.eval()
    identity_adapter.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", unit="batch", leave=False):
            loss = forward_batch(
                batch,
                unet,
                vae,
                text_encoder,
                tokenizer,
                scheduler,
                au_encoder,
                identity_adapter,
                face_proj,
                device,
            )
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def save_checkpoint(
    path: str,
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
        au_encoder: AU encoder module.
        identity_adapter: Identity adapter module.
        au_procs: AU attention processors (ModuleDict).
        optimizer: Current optimizer.
        epoch: Last completed epoch (0-indexed).
        best_val_loss: Best validation loss seen so far.
        patience_counter: Current early-stopping patience count.
    """
    procs_sd = {
        k: v
        for k, v in au_procs.state_dict().items()
        if not k.endswith(("to_k_ip.weight", "to_v_ip.weight"))
    }
    flat: dict[str, torch.Tensor] = {}
    flat.update(prefixed(au_encoder.state_dict(), "au_encoder"))
    flat.update(prefixed(identity_adapter.state_dict(), "identity_adapter"))
    flat.update(prefixed(procs_sd, "au_procs"))

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
        return {k[n:]: v for k, v in flat.items() if k.startswith(prefix + ".")}

    au_encoder.load_state_dict(_strip("au_encoder"))
    identity_adapter.load_state_dict(_strip("identity_adapter"))
    au_procs.load_state_dict(_strip("au_procs"), strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])

    return (
        checkpoint["epoch"],
        checkpoint["best_val_loss"],
        checkpoint["patience_counter"],
    )
