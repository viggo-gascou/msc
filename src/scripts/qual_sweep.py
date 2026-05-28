"""Qualitative sweep over AU configurations, au_scale, and guidance_scale.

For each sampled image (from BP4D and FFHQ test splits) and each predefined AU
configuration, generates images for every (au_scale, guidance_scale) combination.

Output layout::

    <output_dir>/
      bp4d/
        {i:02d}_{subject}_{task}/
          source.png
          {config}/
            au{au_scale:.2f}_gs{guidance_scale:.1f}.png
          {config}_grid.png
      ffhq/
        {i:02d}_{image_id}/
          source.png
          {config}/
            au{au_scale:.2f}_gs{guidance_scale:.1f}.png
          {config}_grid.png
"""

import argparse
import typing as t
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from msc.constants import BP4D_TEST_INDEX_PATH, PYFEAT_AU_NAME_TO_IDX, RANDOM_SEED
from msc.data import BP4DDataset, FFHQDataset, get_transforms
from msc.model_utils import load_inference_pipeline

if t.TYPE_CHECKING:
    from msc.inference_pipeline import AUAdapterPipeline

# Predefined expression configurations.
# Each dict overrides specific AUs on top of the source AU values; unspecified
# AUs stay at their source values. "neutral" leaves the source AUs unchanged.
AU_CONFIGS: dict[str, dict[str, float]] = {
    "neutral": {},
    "smile": {"AU06": 0.8, "AU12": 0.8},
    "surprise": {"AU01": 0.7, "AU02": 0.7, "AU05": 0.8, "AU26": 0.6},
    "disgust": {"AU09": 0.8, "AU15": 0.6},
    "fear": {
        "AU01": 0.7,
        "AU02": 0.5,
        "AU04": 0.6,
        "AU05": 0.8,
        "AU07": 0.6,
        "AU20": 0.7,
        "AU26": 0.5,
    },
    "anger": {"AU04": 0.8, "AU05": 0.5, "AU07": 0.7, "AU23": 0.6},
    "sadness": {"AU01": 0.7, "AU04": 0.6, "AU15": 0.7},
    "contempt": {"AU12": 0.6, "AU14": 0.7},
}

AU_SCALES: list[float] = [0.5, 0.75, 1.0, 1.5]
GUIDANCE_SCALES: list[float] = [1.0, 3.0, 7.5, 10.0]


def main() -> None:
    """Run the qualitative AU sweep."""
    parser = argparse.ArgumentParser(
        description="Qualitative sweep: AU configs x au_scale x guidance_scale."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory containing best_au_adapter.safetensors and config.yaml.",
    )
    parser.add_argument("--au-adapter-path", type=str)
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--output-dir", type=str, default="qual_sweep")
    parser.add_argument(
        "--dataset", type=str, default="both", choices=["bp4d", "ffhq", "both"]
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    au_adapter_path, config_path = resolve_model_paths(args)
    cfg = OmegaConf.load(config_path)
    pipeline = load_inference_pipeline(
        cfg.parameters, au_adapter_path, device=args.device, load_arcface=False
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("bp4d", "both"):
        run_bp4d(
            pipeline=pipeline,
            output_dir=output_dir,
            num_samples=args.num_samples,
            strength=args.strength,
            num_steps=args.num_steps,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )

    if args.dataset in ("ffhq", "both"):
        run_ffhq(
            pipeline=pipeline,
            output_dir=output_dir,
            num_samples=args.num_samples,
            strength=args.strength,
            num_steps=args.num_steps,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )

    logger.info(f"Done. Results in {output_dir.resolve()}")


def resolve_model_paths(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve adapter and config paths from --model-dir or explicit args.

    Args:
        args:
            Parsed argument namespace.

    Returns:
        Tuple of (au_adapter_path, config_path).

    Raises:
        ValueError:
            If neither --model-dir nor both explicit paths are provided.
    """
    if args.model_dir is not None:
        model_dir = Path(args.model_dir)
        return (
            str(args.au_adapter_path or model_dir / "best_au_adapter.safetensors"),
            str(args.config_path or model_dir / "config.yaml"),
        )
    if args.au_adapter_path is None or args.config_path is None:
        raise ValueError(
            "Provide --model-dir or both --au-adapter-path and --config-path."
        )
    return args.au_adapter_path, args.config_path


def run_bp4d(
    pipeline: "AUAdapterPipeline",
    output_dir: Path,
    num_samples: int,
    strength: float,
    num_steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Run the qualitative AU sweep on the BP4D test set."""
    _, val_tf = get_transforms()
    ds = BP4DDataset(index_path=BP4D_TEST_INDEX_PATH, transform=val_tf)
    indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[
        :num_samples
    ].tolist()

    bp4d_dir = output_dir / "bp4d"
    bp4d_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        sample = ds[idx]
        source_pil = tensor_to_pil(tensor=sample["image"])
        sample_dir = bp4d_dir / f"{i:02d}_{sample['subject']}_{sample['task']}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"BP4D sample {i + 1}/{num_samples}: {sample['subject']} {sample['task']}"
        )
        sweep_sample(
            pipeline=pipeline,
            source_pil=source_pil,
            source_aus=sample["aus"],
            arcface=sample["arcface"],
            caption=sample.get("target_caption", ""),
            sample_dir=sample_dir,
            strength=strength,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
        )


def run_ffhq(
    pipeline: "AUAdapterPipeline",
    output_dir: Path,
    num_samples: int,
    strength: float,
    num_steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Run the qualitative AU sweep on the FFHQ test set."""
    _, val_tf = get_transforms()
    ds = FFHQDataset(split="test", transform=val_tf)
    indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[
        :num_samples
    ].tolist()

    ffhq_dir = output_dir / "ffhq"
    ffhq_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        sample = ds[idx]
        source_pil = tensor_to_pil(tensor=sample["image"])
        image_id = Path(ds.df.iloc[idx]["image_number"]).stem
        sample_dir = ffhq_dir / f"{i:02d}_{image_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FFHQ sample {i + 1}/{num_samples}: {image_id}")
        sweep_sample(
            pipeline=pipeline,
            source_pil=source_pil,
            source_aus=sample["aus"],
            arcface=sample["arcface"],
            caption=sample.get("target_caption", ""),
            sample_dir=sample_dir,
            strength=strength,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
        )


def sweep_sample(
    pipeline: "AUAdapterPipeline",
    source_pil: Image.Image,
    source_aus: torch.Tensor,
    arcface: torch.Tensor,
    caption: str,
    sample_dir: Path,
    strength: float,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Run the full AU config x au_scale x guidance_scale sweep for one sample."""
    source_pil.save(sample_dir / "source.png")

    for config_name, au_overrides in AU_CONFIGS.items():
        config_dir = sample_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Start from source AUs (NaN -> 0 as in training) and override target AUs.
        target_aus = source_aus.clone().nan_to_num(0.0)
        for au_name, val in au_overrides.items():
            target_aus[PYFEAT_AU_NAME_TO_IDX[au_name]] = val

        # rows: au_scale, cols: guidance_scale; first col is source
        grid_rows: list[list[Image.Image]] = []
        rowlabels: list[str] = []
        collabels = ["source"] + [f"gs={gs}" for gs in GUIDANCE_SCALES]

        for au_scale in AU_SCALES:
            row: list[Image.Image] = [source_pil.resize((256, 256))]
            for gs in GUIDANCE_SCALES:
                gen = generate(
                    pipeline=pipeline,
                    source_pil=source_pil,
                    arcface=arcface,
                    aus=target_aus,
                    caption=caption,
                    au_scale=au_scale,
                    guidance_scale=gs,
                    strength=strength,
                    num_steps=num_steps,
                    device=device,
                    dtype=dtype,
                )
                gen.save(config_dir / f"au{au_scale:.2f}_gs{gs:.1f}.png")
                row.append(gen)

            grid_rows.append(row)
            rowlabels.append(f"au={au_scale:.2f}")

        grid = make_grid(rows=grid_rows, rowlabels=rowlabels, collabels=collabels)
        grid.save(sample_dir / f"{config_name}_grid.png")
        logger.info(f"  [{config_name}] grid saved")


def generate(
    pipeline: "AUAdapterPipeline",
    source_pil: Image.Image,
    arcface: torch.Tensor,
    aus: torch.Tensor,
    caption: str,
    au_scale: float,
    guidance_scale: float,
    strength: float,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Image.Image:
    """Generate an image using the AUAdapter pipeline.

    Returns:
        Image.Image: The generated image.
    """
    arcface_in = arcface.unsqueeze(0).to(device=device, dtype=dtype)
    aus_in = aus.unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        out = pipeline(
            prompt=caption,
            aus=aus_in,
            image=source_pil,
            arcface_embeds=arcface_in,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            au_scale=au_scale,
        )
    return out.images[0]


def make_grid(
    rows: list[list[Image.Image]],
    rowlabels: list[str],
    collabels: list[str],
    cell_size: int = 256,
) -> Image.Image:
    """Make a grid of images with optional row and column labels.

    Returns:
        Image.Image: The grid image.
    """
    n_rows = len(rows)
    n_cols = len(rows[0])
    grid = Image.new("RGB", (cell_size * n_cols, cell_size * n_rows))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            img = img.resize((cell_size, cell_size))
            if r == 0 and collabels:
                img = label(img=img, text=collabels[c])
            elif c == 0 and rowlabels:
                img = label(img=img, text=rowlabels[r])
            grid.paste(img, (c * cell_size, r * cell_size))
    return grid


def label(img: Image.Image, text: str, font_size: int = 14) -> Image.Image:
    """Label an image with the given text using a bold font.

    Returns:
        Image.Image: The labeled image.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, out.width, font_size + 4], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return out


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalised (mean=0.5, std=0.5) float tensor to a PIL image.

    Returns:
        RGB PIL image.
    """
    arr = (
        (tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).astype("uint8")
    return Image.fromarray(arr)


if __name__ == "__main__":
    main()
