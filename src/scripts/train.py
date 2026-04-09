"""FaceID IP-Adapter + AU adapter training loop on BP4D."""

from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import CLIPTokenizer

from msc.au_adapter import AUEncoder, IdentityAdapter, save_au_adapter
from msc.cli import cli
from msc.constants import RANDOM_SEED
from msc.data.dataset import get_dataloaders
from msc.model_utils import freeze_model_layers, load_model
from msc.train_utils import (
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate,
)


@cli()
def train(cfg: DictConfig) -> None:
    """Train AU adapter on BP4D with FaceID IP-Adapter conditioning."""
    load_dotenv()
    set_seed(RANDOM_SEED)

    params = cfg.parameters
    ip_cfg = cfg.ip_adapter
    wandb_cfg = cfg.wandb

    accelerator = Accelerator(log_with="wandb" if wandb_cfg.enabled else None)
    device = accelerator.device

    if wandb_cfg.enabled:
        accelerator.init_trackers(
            project_name=wandb_cfg.project,
            config={
                "learning_rate": params.learning_rate,
                "batch_size": params.batch_size,
                "epochs": params.epochs,
                "augmentation_proba": params.augmentation_proba,
                "early_stopping": params.early_stopping,
                "patience": params.patience,
                "unet_model": params.unet_model,
                "ip_adapter_repo": ip_cfg.repo,
                "ip_adapter_weight_id": ip_cfg.weight_id,
            },
            init_kwargs={
                "wandb": {
                    "entity": wandb_cfg.entity,
                    "name": wandb_cfg.run_name or None,
                }
            },
        )

    pipeline, unet, vae, text_encoder, tokenizer, scheduler, au_procs = load_model(
        params, ip_cfg
    )
    unet, vae, text_encoder = freeze_model_layers(unet, vae, text_encoder)

    # Unfreeze AU projection weights only
    for proc in au_procs.values():
        for name, p in proc.named_parameters():
            if "to_k_au" in name or "to_v_au" in name:
                p.requires_grad = True

    au_encoder = AUEncoder()
    identity_adapter = IdentityAdapter()

    optimizer = torch.optim.AdamW(
        params=(
            list(au_encoder.parameters())
            + list(identity_adapter.parameters())
            + [
                p
                for proc in au_procs.values()
                for p in proc.parameters()
                if p.requires_grad
            ]
        ),
        lr=params.learning_rate,
    )

    train_loader, val_loader, test_loader = get_dataloaders(params)

    unet, au_encoder, identity_adapter, optimizer, train_loader = accelerator.prepare(
        unet, au_encoder, identity_adapter, optimizer, train_loader
    )
    for model in [vae, text_encoder]:
        model.to(device)

    tokenizer: CLIPTokenizer = tokenizer

    # Resume from checkpoint if one exists
    checkpoint_dir = Path(params.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if checkpoint_path.exists():
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            str(checkpoint_path), au_encoder, identity_adapter, au_procs, optimizer
        )
        start_epoch += 1  # resume from the next epoch
        logger.info(
            f"Resumed from checkpoint: epoch {start_epoch}, "
            f"best val loss {best_val_loss:.4f}"
        )

    for epoch in tqdm(range(start_epoch, params.epochs), desc="Epochs", unit="epoch"):
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
            device,
        )
        logger.info(f"epoch {epoch + 1} | train {train_loss:.4f} | val {val_loss:.4f}")
        accelerator.log(
            {"train/loss": train_loss, "val/loss": val_loss}, step=epoch + 1
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_au_adapter(
                au_encoder, identity_adapter, au_procs, "best_au_adapter.safetensors"
            )
            logger.info(f"Saved best model (val loss {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if params.early_stopping and patience_counter >= params.patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

        if (epoch + 1) % params.checkpoint_every == 0:
            save_checkpoint(
                str(checkpoint_path),
                au_encoder,
                identity_adapter,
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
        device,
    )
    logger.info(f"test loss: {test_loss:.4f}")
    accelerator.log({"test/loss": test_loss})
    accelerator.end_training()


if __name__ == "__main__":
    train()
