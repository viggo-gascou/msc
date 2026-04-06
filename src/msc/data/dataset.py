"""PyTorch Dataset for BP4D face sequences."""

from pathlib import Path
from typing import Callable, NotRequired, TypedDict

import h5py
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.tv_tensors import Image as TVImage

from ..constants import (
    BP4D_AU_COLUMNS,
    BP4D_EMBEDDINGS_DIR,
    BP4D_PREPROCESSED_DIR,
    BP4D_SEQUENCES_DIR,
)
from .bp4d import load_index, resolve_frame_path


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

    # Target frame from the same subject/task — ground truth expression to generate
    target_image: TVImage
    # AU labels for the target frame — conditioning signal for the diffusion model
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

        index = load_index(index_path)

        if tasks is not None:
            index = index[index["task"].isin(tasks)]
        if subjects is not None:
            index = index[index["subject"].isin(subjects)]

        self.index = index.reset_index(drop=True)

        # {(subject, task): [row indices]} for fast target frame sampling
        self.seq_index: dict[tuple[str, str], list[int]] = {}
        for i, row in enumerate(self.index.itertuples(index=False)):
            key = (row.subject, row.task)
            self.seq_index.setdefault(key, []).append(i)

        # HDF5 handles — opened lazily in _open_h5 to support multiprocessing
        self.preprocessed: dict[str, h5py.File] = {}
        self.embeddings: dict[str, h5py.File] = {}

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

        pre_f = self.open_h5(self.preprocessed, self.preprocessed_dir / f"{task}.h5")
        emb_f = self.open_h5(self.embeddings, self.embeddings_dir / f"{task}.h5")

        # Locate this frame in the HDF5 arrays via the stored indices dataset
        indices: np.ndarray = pre_f[subject]["indices"][:]
        pos = int(np.searchsorted(indices, img_frame))

        arcface = torch.from_numpy(emb_f[subject]["arcface"][pos])
        adaface = torch.from_numpy(emb_f[subject]["adaface"][pos])

        aus = torch.tensor(
            [row.get(col, float("nan")) for col in BP4D_AU_COLUMNS], dtype=torch.float32
        )

        image = self.load_raw(subject, task, img_frame)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            aus = self.target_transform(aus)

        # Sample a target frame from the same subject/task
        candidates = self.seq_index[(subject, task)]
        target_idx = candidates[int(torch.randint(len(candidates), (1,)).item())]
        target_row = self.index.iloc[target_idx]
        target_img_frame = int(target_row["frame"]) - 1
        target_image = self.load_raw(subject, task, target_img_frame)
        if self.transform is not None:
            target_image = self.transform(target_image)
        target_aus = torch.tensor(
            [target_row.get(col, float("nan")) for col in BP4D_AU_COLUMNS],
            dtype=torch.float32,
        )
        if self.target_transform is not None:
            target_aus = self.target_transform(target_aus)

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
        path = resolve_frame_path(Path(self.root), subject, task, img_frame)
        if path is None:
            raise FileNotFoundError(f"No image found for {subject}/{task}/{img_frame}")
        return TVImage(decode_image(str(path), mode=ImageReadMode.RGB))
