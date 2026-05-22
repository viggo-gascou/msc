"""AU adapter training loop on BP4D/FFHQ."""

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTokenizer

from msc.au_adapter import AUEncoder, IdentityAdapter, load_au_adapter, save_au_adapter
from msc.cli import cli
from msc.config import Args
from msc.constants import RANDOM_SEED
from msc.data import get_dataloaders
from msc.log_utils import setup_file_logging
from msc.model_utils import freeze_model_layers, load_model
from msc.train_utils import (
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate,
)


@cli()
def train(cfg: Args) -> None:
    """Train AU adapter on BP4D/FFHQ."""
    load_dotenv()
    set_seed(RANDOM_SEED)

    params = cfg.parameters
    opt_cfg = params.optimizer
    wandb_cfg = cfg.wandb

    accelerator = Accelerator(
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        log_with="wandb" if wandb_cfg.enabled else None,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        run_dir = setup_file_logging(cfg)
        logger.info(f"Run directory: {run_dir.absolute()}")

    if wandb_cfg.enabled:
        accelerator.init_trackers(
            project_name=wandb_cfg.project,
            init_kwargs={
                "wandb": {
                    "entity": wandb_cfg.entity,
                    "name": wandb_cfg.run_name or None,
                    "config": asdict(params),
                }
            },
        )

    unet, vae, text_encoder, tokenizer, scheduler, au_procs = load_model(params)
    unet, vae, text_encoder, au_procs = freeze_model_layers(
        unet,
        vae,
        text_encoder,
        au_procs,
        params.lora,
        skip_lora_init=params.adapter_weights is not None,
    )
    if params.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Cast frozen components to the accelerator's dtype; trainable parts stay
    # in fp32 and are handled by Accelerate's mixed precision autocast.
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        accelerator.mixed_precision, torch.float32
    )
    for model in [vae, text_encoder]:
        model.to(device, dtype=weight_dtype)

    au_encoder = AUEncoder(num_tokens=params.au_num_tokens)
    identity_adapter = IdentityAdapter()

    optimizer = torch.optim.AdamW(
        params=(
            list(au_encoder.parameters())
            + list(identity_adapter.parameters())
            + [p for p in unet.parameters() if p.requires_grad]
        ),
        lr=opt_cfg.learning_rate,
        betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
        eps=opt_cfg.adam_eps,
        weight_decay=opt_cfg.weight_decay,
    )

    if params.dataset == "ffhq":
        params.reconstruction = True
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=params.dataset,
        dataloader_params=params.dataloader,
        augmentation_proba=params.augmentation_proba,
    )
    if params.dataset == "bp4d" and not params.reconstruction:
        # Fix val/test at the end-of-curriculum distance so they are comparable
        # to late-stage training difficulty, not the easier random-pair baseline.
        val_loader.dataset.set_min_au_distance(params.min_au_distance)
        test_loader.dataset.set_min_au_distance(params.min_au_distance)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if params.resume_from is not None:
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            params.resume_from, unet, au_encoder, identity_adapter, au_procs, optimizer
        )
        start_epoch += 1  # resume from the next epoch
        if accelerator.is_main_process:
            logger.info(
                f"Resumed from {params.resume_from}: epoch {start_epoch}, "
                f"best val loss {best_val_loss:.4f}"
            )
    elif params.adapter_weights is not None:
        au_encoder, identity_adapter, au_procs = load_au_adapter(
            params.adapter_weights, unet
        )
        if accelerator.is_main_process:
            logger.info(
                f"Loaded pre-trained adapter weights from {params.adapter_weights}"
            )

    unet, au_encoder, identity_adapter, optimizer, train_loader = accelerator.prepare(
        unet, au_encoder, identity_adapter, optimizer, train_loader
    )

    tokenizer: CLIPTokenizer = tokenizer

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    checkpoint_dir = Path(params.checkpoint_dir) / (
        f"{slurm_job_id}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(OmegaConf.structured(cfg), checkpoint_dir / "config.yaml")
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"

    for epoch in tqdm(
        range(start_epoch, params.epochs),
        desc="Epochs",
        unit="epoch",
        disable=not accelerator.is_main_process,
    ):
        if not params.reconstruction:
            au_dist = params.min_au_distance + (
                params.max_au_distance - params.min_au_distance
            ) * epoch / max(params.epochs - 1, 1)
            train_loader.dataset.set_min_au_distance(au_dist)

        train_loss = train_one_epoch(
            unet,
            vae,
            text_encoder,
            tokenizer,
            scheduler,
            au_encoder,
            identity_adapter,
            optimizer,
            train_loader,
            accelerator,
            device,
            params=params,
        )
        val_loss = validate(
            unet,
            vae,
            text_encoder,
            tokenizer,
            scheduler,
            au_encoder,
            identity_adapter,
            val_loader,
            accelerator,
            device,
            params=params,
        )
        if accelerator.is_main_process:
            logger.info(
                f"epoch {epoch + 1} | train {train_loss:.4f} | val {val_loss:.4f}"
            )
        accelerator.log(
            {"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch + 1}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if accelerator.is_main_process:
                save_au_adapter(
                    accelerator.unwrap_model(model=unet),
                    accelerator.unwrap_model(model=au_encoder),
                    accelerator.unwrap_model(model=identity_adapter),
                    au_procs,
                    checkpoint_dir / "best_au_adapter.safetensors",
                )
                logger.info(f"Saved best model (val loss {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if (
                params.early_stopping
                and patience_counter >= params.patience
                and accelerator.is_main_process
            ):
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

        if (epoch + 1) % params.checkpoint_every == 0 and accelerator.is_main_process:
            save_checkpoint(
                str(checkpoint_path),
                accelerator.unwrap_model(model=unet),
                accelerator.unwrap_model(model=au_encoder),
                accelerator.unwrap_model(model=identity_adapter),
                au_procs,
                optimizer,
                epoch,
                best_val_loss,
                patience_counter,
            )
            logger.info(f"Saved checkpoint at epoch {epoch + 1}")

    test_loss = evaluate(
        unet,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        au_encoder,
        identity_adapter,
        test_loader,
        accelerator,
        device,
        params=params,
    )
    if accelerator.is_main_process:
        logger.info(f"test loss: {test_loss:.4f}")
    accelerator.log({"test/loss": test_loss})
    accelerator.end_training()


if __name__ == "__main__":
    train()
