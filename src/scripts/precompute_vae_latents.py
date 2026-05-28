"""Precompute and cache VAE latent encodings for all BP4D frames.

Encodes every unique (subject, task, frame) across the train/val/test splits
and writes the latents to an HDF5 file keyed by subject → task → frame number.
Resumable: frames already present in the file are skipped.

HDF5 structure:
    f[subject][task][str(frame)] = float32 array of shape (4, 64, 64)

Usage:
    uv run src/scripts/precompute_vae_latents.py
    uv run src/scripts/precompute_vae_latents.py --vae-model path/to/vae
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.io import ImageReadMode, decode_image
from tqdm import tqdm

from msc.constants import (
    BP4D_SEQUENCES_DIR,
    BP4D_TEST_INDEX_PATH,
    BP4D_TRAIN_INDEX_PATH,
    BP4D_VAE_LATENTS_PATH,
    BP4D_VAL_INDEX_PATH,
)
from msc.data.bp4d import load_index, resolve_frame_path
from msc.data.dataset import get_transforms


def _encode_latent(
    vae: torch.nn.Module, img: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Encode a (3, H, W) float32 image in [-1, 1] to a VAE latent.

    Args:
        vae:
            Frozen VAE encoder.
        img:
            Image tensor of shape (3, H, W) in [-1, 1].
        device:
            Target device.

    Returns:
        Float32 numpy array of shape (4, H/8, W/8).
    """
    vae_dtype = next(vae.parameters()).dtype
    x = img.unsqueeze(0).to(device=device, dtype=vae_dtype)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    return latent.squeeze(0).float().cpu().numpy()


def main() -> None:
    """Encode all BP4D frames and write latents to HDF5.

    Raises:
        ValueError:
            If the output file already exists and was built with a different VAE.
    """
    parser = argparse.ArgumentParser(
        description="Precompute VAE latent encodings for all BP4D frames."
    )
    parser.add_argument(
        "--unet-model", type=str, default="Manojb/stable-diffusion-2-1-base"
    )
    parser.add_argument("--vae-model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-path", type=str, default=str(BP4D_VAE_LATENTS_PATH))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    print("Loading VAE...")
    if args.vae_model is not None:
        vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=dtype)
    else:
        vae = AutoencoderKL.from_pretrained(
            args.unet_model, subfolder="vae", torch_dtype=dtype
        )
    vae = vae.to(device=device).eval()

    _, val_transform = get_transforms(augmentation_proba=0.0)

    # Collect all unique (subject, task, frame) across all splits
    frames: set[tuple[str, str, int]] = set()
    index_paths = (BP4D_TRAIN_INDEX_PATH, BP4D_VAL_INDEX_PATH, BP4D_TEST_INDEX_PATH)
    for index_path in index_paths:
        index = load_index(path=index_path)
        for _, row in index.iterrows():
            frames.add((str(row["subject"]), str(row["task"]), int(row["frame"])))

    print(f"Found {len(frames):,} unique frames across all splits.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    encoded = 0
    missing = 0

    vae_id = args.vae_model if args.vae_model is not None else f"{args.unet_model}/vae"

    with h5py.File(output_path, "a") as f:
        existing_vae = f.attrs.get("vae_model", None)
        if existing_vae is None:
            f.attrs["vae_model"] = vae_id
        elif existing_vae != vae_id:
            raise ValueError(
                f"Cache was built with VAE '{existing_vae}' but you requested "
                f"'{vae_id}'. Delete the cache file and rerun to rebuild."
            )

        for subject, task, frame in tqdm(
            sorted(frames), desc="Encoding frames", unit="frame"
        ):
            key = str(frame)
            # Skip if already cached
            if subject in f and task in f[subject] and key in f[subject][task]:
                skipped += 1
                continue

            path = resolve_frame_path(
                root=BP4D_SEQUENCES_DIR, subject=subject, task=task, img_frame=frame - 1
            )
            if path is None:
                missing += 1
                continue

            img = val_transform(decode_image(str(path), mode=ImageReadMode.RGB))
            latent = _encode_latent(vae=vae, img=img, device=device)

            grp = f.require_group(f"{subject}/{task}")
            grp.create_dataset(key, data=latent, compression="gzip", compression_opts=1)
            encoded += 1

    print(f"\nDone: {encoded} encoded, {skipped} skipped, {missing} missing.")
    print(f"Saved to {output_path}")


main()
