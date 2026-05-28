# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "diffusers>=0.37.0",
#   "transformers",
#   "torch",
#   "accelerate",
#   "Pillow",
#   "matplotlib",
#   "numpy",
# ]
# ///
"""Text-to-image with intermediate step snapshots every 50 steps."""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

SNAPSHOT_EVERY = 50


def decode_latents(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> Image.Image:
    """Decode a latent tensor to a PIL image without gradient tracking.

    Returns:
        Decoded PIL image.
    """
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return pipe.numpy_to_pil(image)[0]


def main() -> None:
    """Run text-to-image generation and save intermediate step snapshots."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, default="a photo of a mountain lake at sunset"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, cartoon, painting, illustration, deformed, ugly",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--model", type=str, default="SG161222/Realistic_Vision_V4.0_noVAE"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("output_data/viz_stable_diffusion_details")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(
        f"Device: {device}  |  Steps: {args.steps}"
        f"  |  Snapshots every: {SNAPSHOT_EVERY}"
    )
    print(f"Prompt: {args.prompt}\n")

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, steps_offset=0)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(desc="generating")

    snapshots = []

    def on_step_end(
        pipe: StableDiffusionPipeline,
        step: int,
        timestep: int,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if (step + 1) % SNAPSHOT_EVERY == 0 or step == 0:
            img = decode_latents(pipe, callback_kwargs["latents"])
            path = args.out_dir / f"step_{step + 1:04d}.png"
            img.save(path)
            snapshots.append((step + 1, img))
            print(f"  Saved step {step + 1:4d} → {path}")
        return callback_kwargs

    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        generator=generator,
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    final_path = args.out_dir / "final.png"
    result.images[0].save(final_path)
    print(f"\nFinal image → {final_path}")

    # Paper-quality denoising grid
    # Rows = 200-step blocks, columns = 50-step offsets within each block.
    # Absolute step for any cell = row_start (y-label) + column_offset (x-label).
    steps_per_row = 200 // SNAPSHOT_EVERY  # 4 cols when SNAPSHOT_EVERY=50
    grid_snaps = [(s, img) for s, img in snapshots if s % SNAPSHOT_EVERY == 0]
    n_rows = (len(grid_snaps) + steps_per_row - 1) // steps_per_row

    cell = 1.7  # inches per image cell
    fig, axes = plt.subplots(
        n_rows,
        steps_per_row,
        figsize=(steps_per_row * cell, n_rows * cell),
        dpi=300,
        squeeze=False,
        gridspec_kw={"hspace": 0.05, "wspace": 0.04},
    )
    fig.patch.set_facecolor("white")

    for idx, (step, img) in enumerate(grid_snaps):
        row, col = divmod(idx, steps_per_row)
        ax = axes[row, col]
        ax.imshow(np.array(img))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(len(grid_snaps), n_rows * steps_per_row):
        row, col = divmod(idx, steps_per_row)
        axes[row, col].set_visible(False)

    # Column headers: offset within each 200-step block (+50, +100, +150, +200)
    for col in range(steps_per_row):
        axes[0, col].set_title(
            f"+{(col + 1) * SNAPSHOT_EVERY}", fontsize=9, pad=5, color="#444444"
        )

    # Row labels (left): starting denoising step of each block (0, 200, 400, ...)
    for row in range(n_rows):
        axes[row, 0].set_ylabel(
            str(row * steps_per_row * SNAPSHOT_EVERY),
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
            color="#444444",
        )

    fig.supxlabel("Step offset within block", fontsize=10)
    fig.supylabel("Denoising step", fontsize=10)

    grid_path = args.out_dir / "grid.png"
    plt.savefig(grid_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Grid ({len(grid_snaps)} snapshots) → {grid_path}")


if __name__ == "__main__":
    main()
