"""Minimal FaceID IP-Adapter training loop on BP4D."""

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from msc.cli import cli
from msc.constants import RANDOM_SEED
from msc.data import get_dataloaders
from msc.train_utils import freeze_model_layers, load_model


@cli()
def train(cfg: DictConfig) -> None:
    """Train a model on BP4D."""
    set_seed(RANDOM_SEED)
    accelerator = Accelerator()
    device = accelerator.device

    params = cfg.parameters
    ip_cfg = cfg.ip_adapter

    pipeline, unet, vae, text_encoder, tokenizer, scheduler = load_model(params, ip_cfg)
    unet, vae, text_encoder = freeze_model_layers(unet, vae, text_encoder)

    train_loader, val_loader, test_loader = get_dataloaders(params)

    # TODO: add optimizer
    unet, vae, text_encoder = accelerator.prepare(unet, vae, text_encoder)
    train_loader, val_loader, test_loader = accelerator.prepare(
        train_loader, val_loader, test_loader
    )

    for epoch in tqdm(range(params.epochs), desc="Epochs", unit="epoch"):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training", unit="batch", leave=False):
            pixel_values = batch["image"].to(device)
            faceid_embeds = batch["arcface"].to(device)

            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
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

            pred = unet(
                sample=noisy,
                timestep=timesteps,
                encoder_hidden_states=cond,
                added_cond_kwargs={"image_embeds": [faceid_embeds.unsqueeze(1)]},
            ).sample
            loss = F.mse_loss(input=pred, target=noise)
            epoch_loss += loss.item()

            # TODO: enable backward pass once trainable params are added
            # accelerator.backward(loss)

        logger.info(f"epoch {epoch + 1} | loss {epoch_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    train()
