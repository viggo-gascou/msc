"""PyTorch Dataset for BP4D face sequences."""

from collections import defaultdict
from pathlib import Path
from typing import Callable, NotRequired, TypedDict

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2
from torchvision.tv_tensors import Image as TVImage

from ..au_adapter import AU_SCALE
from ..config import DataloaderParams
from ..constants import (
    BP4D_AU_COLUMN_MAP,
    BP4D_AU_COLUMNS,
    BP4D_EMBEDDINGS_DIR,
    BP4D_PREPROCESSED_DIR,
    BP4D_SEQUENCES_DIR,
    BP4D_TEST_INDEX_PATH,
    BP4D_TRAIN_INDEX_PATH,
    BP4D_VAL_INDEX_PATH,
)
from .bp4d import load_index, resolve_frame_path


def get_transforms(
    augmentation_proba: float, resolution: int = 512
) -> tuple[v2.Compose, v2.Compose]:
    """Return train and val/test transforms for BP4D images.

    Both resize and center-crop to the target resolution, convert to float
    [0, 1], and normalise to [-1, 1].

    Args:
        augmentation_proba: Probability of applying data augmentations
        resolution: Target image resolution (width and height)

    Returns:
        (train_transforms, val_transforms)
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
    dataloader_params: DataloaderParams, augmentation_proba: float
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation, and test dataloaders for BP4D.

    Args:
        dataloader_params: Dataloader configuration (batch size, num workers).
        augmentation_proba: Probability of applying data augmentations.

    Returns:
        A tuple of train, validation, and test dataloaders.
    """
    train_transforms, val_test_transforms = get_transforms(augmentation_proba)

    train_ds = BP4DDataset(index_path=BP4D_TRAIN_INDEX_PATH, transform=train_transforms)
    val_ds = BP4DDataset(index_path=BP4D_VAL_INDEX_PATH, transform=val_test_transforms)
    test_ds = BP4DDataset(
        index_path=BP4D_TEST_INDEX_PATH, transform=val_test_transforms
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=dataloader_params.batch_size,
        shuffle=True,
        num_workers=dataloader_params.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dataloader_params.batch_size,
        num_workers=dataloader_params.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=dataloader_params.batch_size,
        num_workers=dataloader_params.num_workers,
    )
    return train_loader, val_loader, test_loader


class BP4DSample(TypedDict):
    """A single BP4D sample."""

    subject: str
    task: str
    frame: int

    # Raw image loaded from disk, shape (3, H, W) uint8 RGB
    image: TVImage

    # Identity embeddings, shape (512,)
    arcface: torch.Tensor
    adaface: torch.Tensor

    # AU occurrence labels, shape (len(BP4D_AU_COLUMNS),), float32, NaN for missing
    aus: torch.Tensor

    # Target frame from the same subject/task
    target_image: TVImage
    target_aus: torch.Tensor

    # Aligned 112x112 face crop in [-1, 1], shape (3, 112, 112) — optional,
    # only needed when computing AdaFace loss on-the-fly during training
    face: NotRequired[torch.Tensor | None]


class BP4DDataset(VisionDataset):
    """BP4D frame-level dataset.

    Each item is one coded frame for one subject/task. The raw image is always
    loaded from disk as the primary output.
    The aligned face crop is optional — only needed when computing AdaFace loss
    on the generated image on-the-fly during training.

    HDF5 files are opened lazily per worker to be compatible with
    torch DataLoader multiprocessing.

    Args:
        tasks:
          Which task IDs to include (e.g. ['T1', 'T2']). None = all tasks.
        subjects:
          Which subject IDs to include (e.g. ['F001', 'M001']). None = all.
        load_face:
          If True, also load the aligned 112x112 crop from the preprocessed
          HDF5. Off by default — only enable when the training loop needs it.
        transform:
          Transform applied to the raw image tensor.
        target_transform:
          Transform applied to the AU label tensor.
        sequences_dir:
          Override for the raw sequences directory (default: BP4D_SEQUENCES_DIR).
        preprocessed_dir:
          Override for the preprocessed HDF5 directory.
        embeddings_dir:
          Override for the embeddings HDF5 directory.
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        subjects: list[str] | None = None,
        load_face: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        sequences_dir: Path = BP4D_SEQUENCES_DIR,
        preprocessed_dir: Path = BP4D_PREPROCESSED_DIR,
        embeddings_dir: Path = BP4D_EMBEDDINGS_DIR,
        index_path: Path | None = None,
    ) -> None:
        """BP4D frame-level dataset.

        Args:
            tasks:
                List of tasks to include. If None, all tasks are included.
            subjects:
                List of subjects to include. If None, all subjects are included.
            load_face:
                Whether to load face images.
            transform:
                Optional transform to apply to the image.
            target_transform:
                Optional transform to apply to the AUs.
            sequences_dir:
                Path to the BP4D sequences directory.
            preprocessed_dir:
                Path to the preprocessed HDF5 directory.
            embeddings_dir:
                Path to the embeddings HDF5 directory.
            index_path:
                Override for the index parquet path. Defaults to BP4D_INDEX_PATH.
        """
        super().__init__(
            root=sequences_dir, transform=transform, target_transform=target_transform
        )
        self.load_face = load_face
        self.preprocessed_dir = preprocessed_dir
        self.embeddings_dir = embeddings_dir

        index = load_index(path=index_path)

        if tasks is not None:
            index = index[index["task"].isin(tasks)]
        if subjects is not None:
            index = index[index["subject"].isin(subjects)]

        self.index = index.reset_index(drop=True)

        # Each entry is (dataset_idx, 0-based frame number) to allow fast
        # distance filtering without iloc lookups in __getitem__.
        self.seq_index: defaultdict[tuple[str, str], list[tuple[int, int]]] = (
            defaultdict(list)
        )
        for i, row in enumerate(self.index.itertuples(index=False)):
            self.seq_index[(str(row.subject), str(row.task))].append(
                (i, int(row.frame) - 1)
            )

        # Minimum temporal distance between source and target frames for curriculum
        # learning. Set to 0 to disable filtering.
        self.min_target_distance: int = 0

        # HDF5 handles — opened lazily in _open_h5 to support multiprocessing
        self.preprocessed: dict[str, h5py.File] = {}
        self.embeddings: dict[str, h5py.File] = {}

    def set_min_target_distance(self, distance: int) -> None:
        """Set the minimum temporal distance for target frame sampling.

        Args:
            distance:
                Minimum absolute frame difference between source and target.
                Set to 0 to disable filtering.
        """
        self.min_target_distance = distance

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.index)

    def __getitem__(self, index: int) -> BP4DSample:
        """Return the sample at the given index.

        Args:
            index: Index of the sample to return.

        Returns:
            The sample at the given index.
        """
        row = self.index.iloc[index]
        subject: str = row["subject"]
        task: str = row["task"]
        au_frame: int = int(row["frame"])
        img_frame = au_frame - 1  # AU 1-based → image 0-based

        pre_f = self.open_h5(
            cache=self.preprocessed, path=self.preprocessed_dir / f"{task}.h5"
        )
        emb_f = self.open_h5(
            cache=self.embeddings, path=self.embeddings_dir / f"{task}.h5"
        )

        # Locate this frame in the HDF5 arrays via the stored indices dataset
        indices: np.ndarray = pre_f[subject]["indices"][:]
        pos = int(np.searchsorted(indices, img_frame))

        arcface = torch.from_numpy(emb_f[subject]["arcface"][pos])
        adaface = torch.from_numpy(emb_f[subject]["adaface"][pos])

        aus = torch.tensor(
            [row.get(BP4D_AU_COLUMN_MAP[col], float("nan")) for col in BP4D_AU_COLUMNS],
            dtype=torch.float32,
        ) * torch.tensor(AU_SCALE)

        image = self.load_raw(subject, task, img_frame)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            aus = self.target_transform(aus)

        # Sample a target frame from the same subject/task, respecting the
        # minimum temporal distance for curriculum learning.
        candidates = self.seq_index[(subject, task)]
        if self.min_target_distance > 0:
            valid = [
                (idx, f)
                for idx, f in candidates
                if abs(f - img_frame) >= self.min_target_distance
            ]
            if not valid:
                valid = candidates
        else:
            valid = candidates
        target_idx, target_img_frame = valid[
            int(torch.randint(len(valid), (1,)).item())
        ]
        target_row = self.index.iloc[target_idx]
        target_image = self.load_raw(
            subject=subject, task=task, img_frame=target_img_frame
        )
        if self.transform is not None:
            target_image = self.transform(target_image)
        target_aus = torch.tensor(
            [target_row.get(col, float("nan")) for col in BP4D_AU_COLUMNS],
            dtype=torch.float32,
        ) * torch.tensor(AU_SCALE)

        face = (
            torch.from_numpy(pre_f[subject]["faces"][pos]) if self.load_face else None
        )

        sample: BP4DSample = {
            "subject": subject,
            "task": task,
            "frame": img_frame,
            "image": image,
            "arcface": arcface,
            "adaface": adaface,
            "aus": aus,
            "target_image": target_image,
            "target_aus": target_aus,
        }
        if face is not None:
            sample["face"] = face
        return sample

    def open_h5(self, cache: dict[str, h5py.File], path: Path) -> h5py.File:
        """Open an HDF5 file for reading, caching it in memory.

        Args:
            cache: A dictionary mapping file paths to h5py.File objects.
            path: The path to the HDF5 file.

        Returns:
            The opened h5py.File object.
        """
        key = str(path)
        if key not in cache:
            cache[key] = h5py.File(path, "r")
        return cache[key]

    def load_raw(self, subject: str, task: str, img_frame: int) -> TVImage:
        """Load a raw JPEG from disk as a uint8 RGB TVImage of shape (3, H, W).

        Args:
            subject: Subject ID.
            task: Task ID.
            img_frame: Frame number.

        Returns:
            The loaded image as a TVImage.

        Raises:
            FileNotFoundError: If no image is found for the given subject/task/frame.
        """
        path = resolve_frame_path(
            root=Path(self.root), subject=subject, task=task, img_frame=img_frame
        )
        if path is None:
            raise FileNotFoundError(
                f"No image found for {self.root}/{subject}/{task}/{img_frame}"
            )
        return TVImage(decode_image(str(path), mode=ImageReadMode.RGB))
