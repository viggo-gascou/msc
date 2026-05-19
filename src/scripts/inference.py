"""Inference script for the AU IP Adapter model."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

from msc.constants import BP4D_AU_COLUMN_MAP, BP4D_AU_COLUMNS, BP4D_SEQUENCES_DIR
from msc.model_utils import load_inference_pipeline


def inference() -> None:
    """Run a single AU-conditioned inference example and save the output image.

    Raises:
        ValueError: If the subject or task is not valid.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="F001")
    parser.add_argument("--task", type=str, default="T1")
    parser.add_argument(
        "--prompt",
        type=str,
        default="portrait photo, frontal face, natural lighting, realistic skin",
    )
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--au-scale", type=float, default=0.6)
    parser.add_argument("--id-scale", type=float, default=0.9)
    parser.add_argument(
        "--aus-json",
        type=str,
        default='{"AU12": 1.0, "AU06": 2.0}',
        help='JSON dict of AU values, e.g. \'{"AU12": 1.5, "AU06": 2.0}\'',
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="inference",
        help="Prefix for saved image files.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to config.yaml used for loading inference components.",
    )
    parser.add_argument(
        "--au-adapter-path",
        type=str,
        default="checkpoints/best_au_adapter.safetensors",
        help="Path to AU adapter safetensors checkpoint.",
    )
    args = parser.parse_args()

    try:
        aus = json.loads(args.aus_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --aus-json value: {args.aus_json}") from exc
    if not isinstance(aus, dict):
        raise ValueError("--aus-json must decode to a JSON object/dict")

    cfg = OmegaConf.load(args.config_path)

    pipeline = load_inference_pipeline(
        cfg.parameters,
        args.au_adapter_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    df = pd.read_parquet("data/BP4D/Sample/bp4d_index.parquet")

    subject = args.subject
    task = args.task
    subset = df[(df["subject"] == subject) & (df["task"] == task)].copy()
    row = subset.iloc[0]

    source_aus = {
        au: float(row.get(BP4D_AU_COLUMN_MAP[au], 0.0)) for au in BP4D_AU_COLUMNS
    }
    requested_aus = {au: float(aus.get(au, 0.0)) for au in BP4D_AU_COLUMNS}

    au_cols = [BP4D_AU_COLUMN_MAP[au] for au in BP4D_AU_COLUMNS]
    target_matrix = subset[au_cols].fillna(0.0).to_numpy(dtype=float)
    requested_vec = [requested_aus[au] for au in BP4D_AU_COLUMNS]
    diffs = ((target_matrix - requested_vec) ** 2).sum(axis=1)
    target_row = subset.iloc[int(diffs.argmin())]

    # AU frame numbers are 1-based, image filenames are 0-based.
    source_image = Image.open(
        BP4D_SEQUENCES_DIR / subject / task / f"{int(row['frame']) - 1:04d}.jpg"
    )

    output = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        aus=aus,
        image=source_image,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        au_scale=args.au_scale,
        id_scale=args.id_scale,
    )
    generated = output.images[0]

    out_dir = Path("output_data/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    single_out = out_dir / f"{args.output_prefix}_{subject}.png"
    generated.save(single_out)

    comparison = Image.new("RGB", (source_image.width * 2, source_image.height))
    comparison.paste(source_image.convert("RGB"), (0, 0))
    comparison.paste(generated.convert("RGB"), (source_image.width, 0))
    comparison_out = out_dir / f"{args.output_prefix}_{subject}_comparison.png"
    comparison.save(comparison_out)

    target_aus = {
        au: float(target_row.get(BP4D_AU_COLUMN_MAP[au], 0.0)) for au in BP4D_AU_COLUMNS
    }
    print("Saved:")
    print(f"  generated:  {single_out}")
    print(f"  comparison: {comparison_out} (input | generated | target)")
    print("AU scores:")
    print("  source:")
    print("   ", source_aus)
    print("  requested:")
    print("   ", requested_aus)
    print("  target (closest in dataset):")
    print("   ", target_aus)


if __name__ == "__main__":
    inference()


# uv run inference \
#   --output-prefix test_steps \
#   --config-path decreased_lr/checkpoints/config.yaml \
#   --au-adapter-path decreased_lr/best_au_adapter.safetensors \
#   --num-inference-steps 40 \
#   --guidance-scale 6 \
#   --strength 0.25 \
#   --au-scale 1.0 \
#   --id-scale 0.9 \
#   --prompt "frontal portrait photo, realistic skin, natural lighting" \
#   --negative-prompt "blurry, noisy, artifacts, distorted face" \
#   --aus-json '{"AU06":4.0,"AU12":4.0,"AU17":2.5}'
