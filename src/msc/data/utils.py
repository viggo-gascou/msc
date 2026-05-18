"""Shared data utilities — transforms and unified dataloader factory."""

import typing as t

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ..config import DataloaderParams
from ..constants import BP4D_TEST_INDEX_PATH, BP4D_TRAIN_INDEX_PATH, BP4D_VAL_INDEX_PATH
from .bp4d.dataset import BP4DDataset
from .ffhq.dataset import FFHQDataset
from .ffhq.utils import load_ffhq_df


def get_transforms(
    augmentation_proba: float = 0.0, resolution: int = 512
) -> tuple[v2.Compose, v2.Compose]:
    """Return train and val/test transforms.

    Both resize and center-crop to the target resolution, convert to float
    [0, 1], and normalise to [-1, 1].

    Args:
        augmentation_proba:
          Probability of applying data augmentations.
        resolution (optional):
          Target image resolution (width and height). Defaults to 512.

    Returns:
        Tuple of (train_transforms, val_transforms).
    """
    train_transforms = v2.Compose(
        [
            v2.Resize(resolution),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    val_transforms = v2.Compose(
        [
            v2.Resize(resolution),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return train_transforms, val_transforms


def get_dataloaders(
    dataset: t.Literal["bp4d", "ffhq"],
    dataloader_params: DataloaderParams,
    augmentation_proba: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for the given dataset.

    Args:
        dataset:
          Which dataset to load ('bp4d' or 'ffhq').
        dataloader_params:
          Batch size and num_workers config.
        augmentation_proba:
          Probability of applying data augmentations.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_tf, val_tf = get_transforms(augmentation_proba=augmentation_proba)
    pin_memory = dataloader_params.pin_memory and dataloader_params.num_workers > 0
    persistent_workers = (
        dataloader_params.persistent_workers and dataloader_params.num_workers > 0
    )

    if dataset == "ffhq":
        df = load_ffhq_df()
        train_ds = FFHQDataset(split="train", transform=train_tf, df=df)
        val_ds = FFHQDataset(split="val", transform=val_tf, df=df)
        test_ds = FFHQDataset(split="test", transform=val_tf, df=df)
    else:
        train_ds = BP4DDataset(index_path=BP4D_TRAIN_INDEX_PATH, transform=train_tf)
        val_ds = BP4DDataset(index_path=BP4D_VAL_INDEX_PATH, transform=val_tf)
        test_ds = BP4DDataset(index_path=BP4D_TEST_INDEX_PATH, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=dataloader_params.batch_size,
        shuffle=True,
        num_workers=dataloader_params.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dataloader_params.batch_size,
        num_workers=dataloader_params.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=dataloader_params.batch_size,
        num_workers=dataloader_params.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader
