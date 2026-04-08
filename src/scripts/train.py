"""Minimal FaceID IP-Adapter training loop on BP4D."""

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from msc.cli import cli
from msc.data.dataset import BP4DDataset, get_transforms


@cli()
def train(cfg: DictConfig) -> None:
    """Train a model on BP4D."""
    accelerator = Accelerator()
    device = accelerator.device

    params = cfg.parameters
    ip_cfg = cfg.ip_adapter

    pipeline = StableDiffusionPipeline.from_pretrained(params.unet_model)
    pipeline.load_ip_adapter(
        ip_cfg.repo,
        subfolder="",
        weight_name=ip_cfg.weight_id,
        image_encoder_folder=None,
    )
    pipeline.set_ip_adapter_scale(scale=0.6)

    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer: CLIPTokenizer = pipeline.tokenizer
    scheduler = pipeline.scheduler

    # Freeze everything
    for model in [unet, vae, text_encoder]:
        for p in model.parameters():
            p.requires_grad = False
    if unet.encoder_hid_proj is not None:
        for p in unet.encoder_hid_proj.parameters():
            p.requires_grad = False

    train_tf, val_tf = get_transforms()
    dataset = BP4DDataset(transform=train_tf)
    loader = DataLoader(
        dataset, batch_size=params.batch_size, shuffle=True, num_workers=4
    )

    unet, loader = accelerator.prepare(unet, loader)
    for model in [vae, text_encoder]:
        model.to(device)

    for epoch in range(params.epochs):
        epoch_loss = 0.0
        for batch in loader:
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

            accelerator.backward(loss)

        print(f"epoch {epoch + 1} | loss {epoch_loss / len(loader):.4f}")


if __name__ == "__main__":
    train()
