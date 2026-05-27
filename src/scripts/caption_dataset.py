"""Generate BLIP-2 captions for BP4D or FFHQ frames and save to parquet."""

import argparse
import typing as t
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from msc.constants import (
    BP4D_DATA_DIR,
    BP4D_INDEX_PATH,
    BP4D_SEQUENCES_DIR,
    FFHQ_DATA_DIR,
    FFHQ_IMAGES_DIR,
    FFHQ_INDEX_PATH,
)

BP4D_CAPTIONS_PATH = BP4D_DATA_DIR / "bp4d_captions.parquet"
FFHQ_CAPTIONS_PATH = FFHQ_DATA_DIR / "ffhq_captions.parquet"


class DatasetConfig(t.TypedDict):
    """DatasetConfig."""

    index_path: Path
    output_path: Path
    id_columns: list[str]


DATASETS: dict[str, DatasetConfig] = {
    "bp4d": {
        "index_path": BP4D_INDEX_PATH,
        "output_path": BP4D_CAPTIONS_PATH,
        "id_columns": ["subject", "task", "frame"],
    },
    "ffhq": {
        "index_path": FFHQ_INDEX_PATH,
        "output_path": FFHQ_CAPTIONS_PATH,
        "id_columns": ["image_number"],
    },
}


def caption_dataset(
    dataset: str,
    batch_size: int,
    max_new_tokens: int,
    model_name: str,
    save_every: int,
    index_path: Path,
    output_path: Path,
) -> None:
    """Generate BLIP-2 captions for all frames in a dataset.

    Supports resumption: if the output file already exists, already-captioned
    frames are skipped.

    Args:
        dataset:
          Dataset to caption ('bp4d' or 'ffhq').
        batch_size:
          Number of images to process per forward pass.
        max_new_tokens:
          Maximum number of tokens to generate per caption.
        model_name:
          HuggingFace model ID for the BLIP-2 model.
        save_every:
          Flush results to disk every this many processed frames.
        index_path:
          Path to the dataset index parquet.
        output_path:
          Path to write the output captions parquet.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    id_cols = DATASETS[dataset]["id_columns"]

    df = pd.read_parquet(index_path)[id_cols].drop_duplicates()
    logger.info(f"Loaded {len(df)} frames from {index_path}")

    existing: pd.DataFrame
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        done_keys = set(map(tuple, existing[id_cols].itertuples(index=False)))
        mask = df.apply(lambda r: tuple(r[c] for c in id_cols) not in done_keys, axis=1)
        df = df[mask].reset_index(drop=True)
        logger.info(f"Resuming: {len(done_keys)} done, {len(df)} remaining")
    else:
        existing = pd.DataFrame(columns=[*id_cols, "caption"])

    if df.empty:
        logger.info("All frames already captioned.")
        return

    logger.info(f"Loading {model_name}...")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    records: list[dict[str, object]] = []
    rows = df.to_dict("records")

    for i in tqdm(range(0, len(rows), batch_size), desc="Captioning"):
        batch = rows[i : i + batch_size]

        images: list[Image.Image] = []
        valid: list[dict[str, object]] = []
        for row in batch:
            path = _resolve_image(dataset=dataset, row=row)
            if path is None:
                logger.warning(f"Image not found: {row}")
                continue
            images.append(Image.open(path).convert("RGB"))
            valid.append(row)

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(
                images=images, text=["a photo of"] * len(images), return_tensors="pt"
            ).to(device, torch.float16)

            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for row, caption in zip(valid, captions):
            records.append({**{c: row[c] for c in id_cols}, "caption": caption.strip()})

        if save_every > 0 and len(records) >= save_every:
            existing = _flush(existing=existing, records=records, path=output_path)
            records = []

    existing = _flush(existing=existing, records=records, path=output_path)
    logger.info(f"Done. {len(existing)} captions saved to {output_path}")


def _resolve_image(dataset: str, row: dict[str, object]) -> Path | None:
    """Resolve the image path for a given dataset row.

    Args:
        dataset: Dataset name ('bp4d' or 'ffhq').
        row: Index row dict.

    Returns:
        Path to the image, or None if not found.
    """
    if dataset == "ffhq":
        p = FFHQ_IMAGES_DIR / str(row["image_number"])
        return p if p.exists() else None

    base = int(row["frame"]) - 1  # type: ignore[arg-type]
    for digits in (4, 3, 2):
        p = (
            BP4D_SEQUENCES_DIR
            / str(row["subject"])
            / str(row["task"])
            / f"{str(base).zfill(digits)}.jpg"
        )
        if p.exists():
            return p
    return None


def _flush(
    existing: pd.DataFrame, records: list[dict[str, object]], path: Path
) -> pd.DataFrame:
    """Append new records to the existing dataframe and write to disk.

    Args:
        existing: Already-saved captions dataframe.
        records: New caption records to append.
        path: Output parquet path.

    Returns:
        Updated combined dataframe.
    """
    if not records:
        return existing
    combined = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    logger.info(f"Flushed {len(records)} records ({len(combined)} total)")
    return combined


def main() -> None:
    """Parse arguments and run captioning."""
    parser = argparse.ArgumentParser(
        description="Generate BLIP-2 captions for BP4D or FFHQ frames."
    )
    parser.add_argument("--dataset", type=str, choices=list(DATASETS), default="bp4d")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Flush to disk every N processed frames.",
    )
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    caption_dataset(
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        model_name=args.model,
        save_every=args.save_every,
        index_path=args.index_path or cfg["index_path"],
        output_path=args.output_path or cfg["output_path"],
    )


if __name__ == "__main__":
    main()
