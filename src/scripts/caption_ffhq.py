# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers",
#   "torch",
#   "Pillow",
#   "pandas",
#   "pyarrow",
#   "tqdm",
# ]
# ///
"""Generate BLIP-2 captions for FFHQ images, writing image_id + caption to parquet."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

HPC_DATA_DIR = Path("/home/semedit/data/FFHQ")
LOCAL_DATA_DIR = Path("data/FFHQ")
DATA_DIR = HPC_DATA_DIR if HPC_DATA_DIR.exists() else LOCAL_DATA_DIR

DEFAULT_IMAGES_DIR = DATA_DIR / "images1024x1024"
DEFAULT_LIBREFACE = DATA_DIR / "ffhq_libreface.parquet"
DEFAULT_OUT = DATA_DIR / "ffhq_captions.parquet"
SAVE_EVERY = 100  # batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--libreface", type=Path, default=DEFAULT_LIBREFACE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}  |  Model: {args.model}")

    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    allowed = set(
        pd.read_parquet(args.libreface, columns=["image"])["image"]
        .apply(lambda p: str(p).split("/")[-1].split(".")[0])
    )
    print(f"LibreFace index: {len(allowed)} images")

    images = sorted(p for p in args.images_dir.rglob("*.png") if p.stem in allowed)
    print(f"Matched {len(images)} images in {args.images_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing parquet
    done: set[str] = set()
    if args.out.exists():
        done = set(pd.read_parquet(args.out, columns=["image_id"])["image_id"])
        print(f"Resuming — {len(done)} already done, {len(images) - len(done)} remaining")
    images = [p for p in images if p.stem not in done]

    rows: list[dict] = []
    printed_example = False

    for batch_idx, i in enumerate(tqdm(range(0, len(images), args.batch_size), desc="captioning", unit="batch")):
        batch_paths = images[i : i + args.batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(batch_imgs, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inputs)
        captions = processor.batch_decode(out, skip_special_tokens=True)

        for path, caption in zip(batch_paths, captions):
            rows.append({"image_id": path.stem, "caption": caption.strip()})

        if not printed_example:
            print(f"\nExample — {batch_paths[0].name}: {captions[0].strip()}\n")
            printed_example = True

        if (batch_idx + 1) % SAVE_EVERY == 0:
            _flush(rows, args.out)
            rows = []

    if rows:
        _flush(rows, args.out)

    total = len(done) + len(images)
    print(f"Done — {total} captions saved to {args.out}")


def _flush(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    main()
