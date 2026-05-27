"""PyTorch Dataset for BP4D face sequences."""

import collections.abc as c
import typing as t
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.tv_tensors import Image as TVImage

from ...constants import (
    BP4D_CAPTIONS_PATH,
    BP4D_EMBEDDINGS_DIR,
    BP4D_PREPROCESSED_DIR,
    BP4D_SEQUENCES_DIR,
    PYFEAT_AU_COLUMNS,
    PYFEAT_AU_SCALE,
)
from .utils import load_index, load_pyfeat_aus, resolve_frame_path


class BP4DSample(t.TypedDict):
    """A single BP4D sample."""

    subject: str
    task: str
    frame: int

    image: TVImage
    arcface: torch.Tensor
    adaface: torch.Tensor
    aus: torch.Tensor
    caption: str

    target_image: TVImage
    target_aus: torch.Tensor
    target_caption: str

    face: t.NotRequired[torch.Tensor | None]


class BP4DDataset(VisionDataset):
    """BP4D frame-level dataset.

    Each item is one coded frame for one subject/task. The raw image is always
    loaded from disk as the primary output. The aligned face crop is optional —
    only needed when computing AdaFace loss on the generated image on-the-fly.

    HDF5 files are opened lazily per worker to be compatible with
    torch DataLoader multiprocessing.

    Args:
        tasks:
          Which task IDs to include (e.g. ['T1', 'T2']). None = all tasks.
        subjects:
          Which subject IDs to include (e.g. ['F001', 'M001']). None = all.
        load_face:
          If True, also load the aligned 112x112 crop from the preprocessed HDF5.
        transform:
          Transform applied to the raw image tensor.
        target_transform:
          Transform applied to the AU label tensor.
        sequences_dir:
          Override for the raw sequences directory.
        preprocessed_dir:
          Override for the preprocessed HDF5 directory.
        embeddings_dir:
          Override for the embeddings HDF5 directory.
        index_path:
          Override for the index parquet path.
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        subjects: list[str] | None = None,
        load_face: bool = False,
        transform: c.Callable | None = None,
        target_transform: c.Callable | None = None,
        sequences_dir: Path = BP4D_SEQUENCES_DIR,
        preprocessed_dir: Path = BP4D_PREPROCESSED_DIR,
        embeddings_dir: Path = BP4D_EMBEDDINGS_DIR,
        index_path: Path | None = None,
        captions_path: Path | None = None,
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
            captions_path (optional):
                Path to the captions parquet. Defaults to BP4D_CAPTIONS_PATH.
                If the file does not exist, captions will be empty strings.
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
        self._lf_aus: dict[str, dict[tuple[str, int], np.ndarray]] = (
            load_pyfeat_aus()
        )
        valid = {
            (task, subj, frame)
            for task, task_dict in self._lf_aus.items()
            for subj, frame in task_dict
        }
        before = len(self.index)
        self.index = self.index[
            self.index.apply(
                lambda r: (str(r["task"]), str(r["subject"]), int(r["frame"])) in valid,
                axis=1,
            )
        ].reset_index(drop=True)
        dropped = before - len(self.index)
        if dropped:
            logger.warning(
                f"BP4DDataset: dropped {dropped}/{before} frames"
                " with no py-feat detection"
            )

        self.seq_index: defaultdict[
            tuple[str, str], list[tuple[int, int, np.ndarray]]
        ] = defaultdict(list)
        for i, row in enumerate(self.index.itertuples(index=False)):
            au_vec = np.nan_to_num(
                self._lookup_aus(str(row.task), str(row.subject), int(row.frame)),
                nan=0.0,
            )
            self.seq_index[(str(row.subject), str(row.task))].append(
                (i, int(row.frame) - 1, au_vec)
            )

        self.subject_index: defaultdict[str, list[tuple[int, int, np.ndarray]]] = (
            defaultdict(list)
        )
        for (subject, _task), entries in self.seq_index.items():
            self.subject_index[subject].extend(entries)

        self.min_au_distance: float = 0.0

        cap_path = captions_path or BP4D_CAPTIONS_PATH
        self.caption_map: dict[tuple[str, str], str] = {}
        if cap_path.exists():
            caps = pd.read_parquet(cap_path, columns=["subject", "task", "caption"])
            mode_caps = (
                caps.groupby(["subject", "task"])["caption"]
                .agg(lambda x: x.mode().iloc[0])
                .reset_index()
            )
            for row in mode_caps.itertuples(index=False):
                self.caption_map[(str(row.subject), str(row.task))] = str(row.caption)

        self.preprocessed: dict[str, h5py.File] = {}
        self.embeddings: dict[str, h5py.File] = {}

    def set_min_au_distance(self, distance: float) -> None:
        """Set the minimum AU L1 distance for target frame sampling.

        Args:
            distance:
              Minimum AU L1 distance between source and target normalised AU
              vectors. Set to 0 to disable filtering.
        """
        self.min_au_distance = distance

    def _lookup_aus(self, task: str, subject: str, frame: int) -> np.ndarray:
        task_dict = self._lf_aus.get(task)
        if task_dict is not None:
            aus = task_dict.get((subject, frame))
            if aus is not None:
                return aus
        return np.full(len(PYFEAT_AU_COLUMNS), float("nan"), dtype=np.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.index)

    def __getitem__(self, index: int) -> BP4DSample:
        """Return the sample at the given index.

        Args:
            index:
              Index of the sample to return.

        Returns:
            The sample at the given index.
        """
        row = self.index.iloc[index]
        subject: str = row["subject"]
        task: str = row["task"]
        au_frame: int = int(row["frame"])
        img_frame = au_frame - 1

        pre_f = self.open_h5(
            cache=self.preprocessed, path=self.preprocessed_dir / f"{task}.h5"
        )
        emb_f = self.open_h5(
            cache=self.embeddings, path=self.embeddings_dir / f"{task}.h5"
        )

        indices: np.ndarray = pre_f[subject]["indices"][:]
        pos = int(np.searchsorted(indices, img_frame))

        arcface = torch.from_numpy(emb_f[subject]["arcface"][pos])
        adaface = torch.from_numpy(emb_f[subject]["adaface"][pos])

        aus = torch.from_numpy(
            self._lookup_aus(task, subject, au_frame)
        ) * torch.tensor(PYFEAT_AU_SCALE)

        image = self.load_raw(subject=subject, task=task, img_frame=img_frame)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            aus = self.target_transform(aus)

        candidates = self.subject_index[subject]
        if self.min_au_distance > 0:
            src_aus = np.nan_to_num(self._lookup_aus(task, subject, au_frame), nan=0.0)
            valid = [
                (idx, f, aus_vec)
                for idx, f, aus_vec in candidates
                if float(np.abs(src_aus - aus_vec).sum()) >= self.min_au_distance
            ]
            if not valid:
                valid = candidates
        else:
            valid = candidates

        target_idx, target_img_frame, _ = valid[
            int(torch.randint(len(valid), (1,)).item())
        ]
        target_row = self.index.iloc[target_idx]
        target_task = str(target_row["task"])
        target_image = self.load_raw(
            subject=subject, task=target_task, img_frame=target_img_frame
        )
        if self.transform is not None:
            target_image = self.transform(target_image)
        target_aus = torch.from_numpy(
            self._lookup_aus(target_task, subject, int(target_row["frame"]))
        ) * torch.tensor(PYFEAT_AU_SCALE)

        face = (
            torch.from_numpy(pre_f[subject]["faces"][pos]) if self.load_face else None
        )

        caption = self.caption_map.get((subject, task), "")
        target_caption = self.caption_map.get((subject, target_task), "")

        sample: BP4DSample = {
            "subject": subject,
            "task": task,
            "frame": img_frame,
            "image": image,
            "arcface": arcface,
            "adaface": adaface,
            "aus": aus,
            "caption": caption,
            "target_image": target_image,
            "target_aus": target_aus,
            "target_caption": target_caption,
        }
        if face is not None:
            sample["face"] = face
        return sample

    def open_h5(self, cache: dict[str, h5py.File], path: Path) -> h5py.File:
        """Open an HDF5 file for reading, caching it per worker.

        Args:
            cache:
              Dict mapping file path strings to open h5py.File handles.
            path:
              Path to the HDF5 file.

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
            subject:
              Subject ID.
            task:
              Task ID.
            img_frame:
              0-based frame number.

        Returns:
            The loaded image as a TVImage.

        Raises:
            FileNotFoundError:
              If no image is found for the given subject/task/frame.
        """
        path = resolve_frame_path(
            root=Path(self.root), subject=subject, task=task, img_frame=img_frame
        )
        if path is None:
            raise FileNotFoundError(
                f"No image found for {self.root}/{subject}/{task}/{img_frame}"
            )
        return TVImage(decode_image(str(path), mode=ImageReadMode.RGB))
