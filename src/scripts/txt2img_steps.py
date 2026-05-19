# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "diffusers>=0.37.0",
#   "transformers",
#   "torch",
#   "accelerate",
#   "Pillow",
# ]
# ///
"""Text-to-image with intermediate step snapshots every 50 steps."""

import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

SNAPSHOT_EVERY = 10


def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return pipe.numpy_to_pil(image)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of a mountain lake at sunset")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, cartoon, painting, illustration, deformed, ugly")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model", type=str, default="SG161222/Realistic_Vision_V4.0_noVAE")
    parser.add_argument("--out-dir", type=Path, default=Path("output_data/viz_stable_diffusion_details"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Device: {device}  |  Steps: {args.steps}  |  Snapshots every: {SNAPSHOT_EVERY}")
    print(f"Prompt: {args.prompt}\n")

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, steps_offset=0)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(desc="generating")

    snapshots = []

    def on_step_end(pipe, step, timestep, callback_kwargs):
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

    # Save a grid of all snapshots
    cols = 5
    rows = (len(snapshots) + cols - 1) // cols
    w, h = snapshots[0][1].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, (step, img) in enumerate(snapshots):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    grid_path = args.out_dir / "grid.png"
    grid.save(grid_path)
    print(f"Grid ({len(snapshots)} snapshots) → {grid_path}")


if __name__ == "__main__":
    main()
