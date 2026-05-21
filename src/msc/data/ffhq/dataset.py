"""PyTorch Dataset for FFHQ self-pair pre-training."""

import collections.abc as c
import typing as t
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.tv_tensors import Image as TVImage

from ...constants import (
    FFHQ_CAPTIONS_PATH,
    FFHQ_EMBEDDINGS_PATH,
    FFHQ_IMAGES_DIR,
    LIBREFACE_AU_COLUMNS,
    LIBREFACE_AU_SCALE,
)
from .utils import load_ffhq_df


class FFHQSample(t.TypedDict):
    """A single FFHQ sample."""

    image: TVImage
    arcface: torch.Tensor
    aus: torch.Tensor
    target_image: TVImage
    target_aus: torch.Tensor
    target_caption: str


class FFHQDataset(Dataset):
    """FFHQ frame-level dataset for self-pair pre-training.

    Each item is a self-pair: source and target are the same image, with
    identical AU vectors. This trains the reconstruction objective before
    fine-tuning on BP4D expression pairs.

    HDF5 is opened lazily per DataLoader worker to support multiprocessing.

    Args:
        split:
          Dataset split to use ('train', 'val', or 'test').
        transform:
          Transform applied to the image tensor.
        df:
          Pre-loaded index DataFrame (from load_ffhq_df). If None, loads from
          disk — pass a shared df to avoid redundant reads across splits.
    """

    def __init__(
        self,
        split: t.Literal["train", "val", "test"],
        transform: c.Callable | None = None,
        df: pd.DataFrame | None = None,
        captions_path: Path | None = None,
    ) -> None:
        """Initialise the FFHQ dataset for a given split.

        Args:
            split:
              Dataset split to load ('train', 'val', or 'test').
            transform (optional):
              Transform applied to the image tensor. Defaults to None.
            df (optional):
              Pre-loaded index DataFrame from load_ffhq_df(). Pass a shared
              instance across splits to avoid redundant disk reads. Defaults to None.
            captions_path (optional):
              Path to the captions parquet. Defaults to FFHQ_CAPTIONS_PATH.
              If the file does not exist, captions will be empty strings.
        """
        super().__init__()
        self.transform = transform
        full_df = df if df is not None else load_ffhq_df()
        self.df = full_df[full_df["split"] == split].reset_index(drop=True)
        self.h5: h5py.File | None = None

        cap_path = captions_path or FFHQ_CAPTIONS_PATH
        if cap_path.exists():
            caps = pd.read_parquet(cap_path, columns=["image_number", "caption"])
            self.df = self.df.merge(caps, on="image_number", how="left")
            self.df["caption"] = self.df["caption"].fillna("")
        else:
            self.df["caption"] = ""

    def open_h5(self) -> h5py.File:
        """Open the ArcFace embeddings HDF5 file, caching the handle per worker.

        Returns:
            The open h5py.File handle.
        """
        if self.h5 is None:
            self.h5 = h5py.File(FFHQ_EMBEDDINGS_PATH, "r")
        return self.h5

    def set_min_au_distance(self, distance: float) -> None:
        """No-op — FFHQ uses self-pairs so AU distance filtering does not apply.

        Args:
            distance:
              Ignored.
        """

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> FFHQSample:
        """Return the sample at the given index.

        Args:
            index:
              Index of the sample to return.

        Returns:
            Self-pair sample with image, arcface embedding, and AU vector.
        """
        row = self.df.iloc[index]
        stem = Path(row["image_number"]).stem

        img_path = FFHQ_IMAGES_DIR / row["image_number"]
        image = TVImage(decode_image(str(img_path), mode=ImageReadMode.RGB))
        if self.transform is not None:
            image = self.transform(image)

        arcface = torch.from_numpy(self.open_h5()[stem][:].copy())

        aus = torch.tensor(
            [float(row[col]) for col in LIBREFACE_AU_COLUMNS], dtype=torch.float32
        ) * torch.tensor(LIBREFACE_AU_SCALE)

        caption: str = str(row["caption"])

        sample: FFHQSample = {
            "image": image,
            "arcface": arcface,
            "aus": aus,
            "target_image": image,
            "target_aus": aus,
            "target_caption": caption,
        }
        return sample
