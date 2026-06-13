"""Update eval metadata with source (and target) image paths.

Reads args.json from one or more eval run directories, recreates the same dataset
and sample indices, and adds source_path (and target_path for paired mode) columns
to each run's metadata.parquet. No images are copied or decoded.

For paired mode, the target is recovered deterministically by matching the stored
requested_aus against the subject's AU vectors in the dataset index.

Run this on the cluster where the datasets are available.

Usage::

    uv run src/scripts/save_source_images.py eval/sd21_pyfeat_pretrain_bp4d_source
    uv run src/scripts/save_source_images.py eval/sd21_*
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from msc.constants import BP4D_SEQUENCES_DIR, BP4D_TEST_INDEX_PATH, FFHQ_IMAGES_DIR
from msc.data import BP4DDataset, FFHQDataset
from msc.data.bp4d.utils import resolve_frame_path


def main() -> None:
    """Add source/target image paths to eval metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    for eval_dir in args.eval_dirs:
        logger.info(f"Processing {eval_dir}")
        _process_run(eval_dir)


def _process_run(eval_dir: Path) -> None:
    """Add source_path (and target_path) to metadata for one eval run."""
    args_path = eval_dir / "args.json"
    if not args_path.exists():
        logger.warning(f"No args.json in {eval_dir}, skipping")
        return

    run_args = json.loads(args_path.read_text())
    eval_mode: str = run_args.get("eval_mode", "emotion")
    # args.json may have the wrong dataset value (saved as "bp4d" for all runs);
    # infer from directory name which is reliably named.
    if "ffhq" in eval_dir.name:
        dataset = "ffhq"
    elif "bp4d" in eval_dir.name:
        dataset = "bp4d"
    else:
        dataset = run_args["dataset"]
    min_au_distance: float = run_args.get("min_au_distance", 2.0)

    metadata_path = eval_dir / "metadata.parquet"
    if not metadata_path.exists():
        logger.warning(f"No metadata.parquet in {eval_dir}, skipping")
        return

    metadata = pd.read_parquet(metadata_path)
    if "source_path" in metadata.columns:
        logger.info(f"{eval_dir}: overwriting existing source_path")

    is_bp4d = dataset == "bp4d"
    num_aus = len(metadata["requested_aus"].iloc[0])
    detector: str = "libreface" if num_aus == 17 else "pyfeat"
    ds = _build_dataset(
        is_bp4d=is_bp4d,
        eval_mode=eval_mode,
        min_au_distance=min_au_distance,
        detector=detector,
    )

    is_paired = eval_mode == "paired" and is_bp4d
    samples = metadata.drop_duplicates("sample_id").set_index("sample_id")

    # Precompute per-subject scaled AU matrices once for fast target lookup.
    subject_aus: dict[str, np.ndarray] = {}
    if is_paired and isinstance(ds, BP4DDataset):
        au_scale = np.array(ds.au_scale, dtype=np.float32)
        for subject, candidates in ds.subject_index.items():
            subject_aus[subject] = np.stack(
                [np.nan_to_num(c[2], nan=0.0) * au_scale for c in candidates]
            )

    source_paths: dict[int, str] = {}
    target_paths: dict[int, str] = {}

    for sample_id, meta_row in tqdm(
        samples.iterrows(), total=len(samples), desc=eval_dir.name
    ):
        dataset_idx = int(meta_row["dataset_idx"])
        if is_bp4d:
            assert isinstance(ds, BP4DDataset)
            row = ds.index.iloc[dataset_idx]
            src_path = resolve_frame_path(
                root=BP4D_SEQUENCES_DIR,
                subject=str(row["subject"]),
                task=str(row["task"]),
                img_frame=int(row["frame"]) - 1,
            )
        else:
            assert isinstance(ds, FFHQDataset)
            row = ds.df.iloc[dataset_idx]
            src_path = FFHQ_IMAGES_DIR / str(row["image_number"])

        if src_path is not None:
            source_paths[int(sample_id)] = str(src_path)

        if is_paired and isinstance(ds, BP4DDataset):
            subject = str(meta_row["subject"])
            requested_aus = np.array(meta_row["requested_aus"], dtype=np.float32)
            aus_matrix = subject_aus[subject]
            best = int(np.argmin(np.abs(aus_matrix - requested_aus).sum(axis=1)))
            candidates = ds.subject_index[subject]
            best_idx, best_frame = candidates[best][0], candidates[best][1]
            tgt_row = ds.index.iloc[best_idx]
            tgt_path = resolve_frame_path(
                root=BP4D_SEQUENCES_DIR,
                subject=subject,
                task=str(tgt_row["task"]),
                img_frame=best_frame,
            )
            if tgt_path is not None:
                target_paths[int(sample_id)] = str(tgt_path)

    metadata["source_path"] = metadata["sample_id"].map(source_paths)
    if target_paths:
        metadata["target_path"] = metadata["sample_id"].map(target_paths)

    metadata.to_parquet(metadata_path)
    logger.info(
        f"{eval_dir}: updated metadata with {len(source_paths)} source"
        + (f" + {len(target_paths)} target" if target_paths else "")
        + " paths"
    )


def _build_dataset(
    is_bp4d: bool, eval_mode: str, min_au_distance: float, detector: str = "pyfeat"
) -> BP4DDataset | FFHQDataset:
    """Build dataset index only — no image loading."""
    if is_bp4d:
        ds: BP4DDataset | FFHQDataset = BP4DDataset(
            index_path=BP4D_TEST_INDEX_PATH, detector=detector
        )
        if eval_mode == "paired":
            ds.set_min_au_distance(min_au_distance)
    else:
        ds = FFHQDataset(split="test", detector=detector)

    return ds


if __name__ == "__main__":
    main()
