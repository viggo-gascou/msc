r"""Quantitative evaluation: AU MSE, distribution shift, latent distance, ArcFace cosim.

Two-phase workflow
------------------
Phase 1 (project env) -- generate images and compute non-AU metrics::

    uv run src/scripts/eval_quantitative.py \
        --au-adapter-path checkpoints/.../best_au_adapter.safetensors \
        --config-path checkpoints/.../config.yaml \
        --dataset bp4d \
        --output-dir eval/bp4d_pyfeat

Phase 2 (pyfeat/libreface env) -- detect AUs on generated images::

    python src/scripts/detect_generated_aus.py \
        --images-dir eval/bp4d_pyfeat/generated \
        --detector pyfeat \
        --output eval/bp4d_pyfeat/au_detections.parquet

Phase 3 (project env) -- compute AU metrics::

    uv run src/scripts/eval_quantitative.py \
        --output-dir eval/bp4d_pyfeat \
        --au-detections eval/bp4d_pyfeat/au_detections.parquet \
        --metrics-only
"""

import argparse
import json
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from msc.constants import BP4D_TEST_INDEX_PATH, PYFEAT_AU_COLUMNS, RANDOM_SEED
from msc.data import BP4DDataset, FFHQDataset, get_transforms
from msc.model_utils import load_inference_pipeline

if t.TYPE_CHECKING:
    from msc.inference_pipeline import AUAdapterPipeline


def main() -> None:
    """Run quantitative evaluation.

    Raises:
        ValueError:
            If required arguments are missing for the requested mode.
    """
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation of AU adapter models."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory containing best_au_adapter.safetensors and config.yaml.",
    )
    parser.add_argument("--au-adapter-path", type=str)
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--dataset", type=str, default="bp4d", choices=["bp4d", "ffhq"])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--au-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--au-detections",
        type=str,
        default=None,
        help="Path to AU detections parquet (from detect_generated_aus.py).",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip generation; load existing metadata and compute AU metrics only.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    if args.metrics_only:
        metadata = pd.read_parquet(output_dir / "metadata.parquet")
        if args.au_detections is None:
            raise ValueError("--au-detections is required with --metrics-only")
        detections = pd.read_parquet(args.au_detections)
        _print_au_metrics(metadata=metadata, detections=detections)
        return

    au_adapter_path, config_path = _resolve_model_paths(args)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    cfg = OmegaConf.load(config_path)
    pipeline = load_inference_pipeline(
        cfg.parameters, au_adapter_path, device=args.device, load_arcface=True
    )

    gen_dir = output_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    metadata = _run_eval(
        pipeline=pipeline,
        dataset=args.dataset,
        num_samples=args.num_samples,
        strength=args.strength,
        au_scale=args.au_scale,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        gen_dir=gen_dir,
        seed=args.seed,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        device=device,
        dtype=dtype,
    )
    shard_suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""
    metadata_path = output_dir / f"metadata{shard_suffix}.parquet"
    metadata.to_parquet(metadata_path, index=False)
    logger.info(f"Metadata saved to {metadata_path}")

    _print_non_au_summary(metadata=metadata, dataset=args.dataset)

    if args.au_detections is not None:
        detections = pd.read_parquet(args.au_detections)
        _print_au_metrics(metadata=metadata, detections=detections)


def _resolve_model_paths(args: argparse.Namespace) -> tuple[str, str]:
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


def _run_eval(
    pipeline: "AUAdapterPipeline",
    dataset: str,
    num_samples: int,
    strength: float,
    au_scale: float,
    guidance_scale: float,
    num_steps: int,
    gen_dir: Path,
    seed: int,
    shard_idx: int,
    num_shards: int,
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    """Generate images and compute ArcFace + latent metrics for all samples.

    Args:
        pipeline:
            Loaded AUAdapterPipeline with arcface_extractor set.
        dataset:
            Dataset to evaluate on ('bp4d' or 'ffhq').
        num_samples:
            Number of test samples to evaluate.
        strength:
            img2img noise strength.
        au_scale:
            AU conditioning scale.
        guidance_scale:
            Classifier-free guidance scale.
        num_steps:
            Number of denoising steps.
        gen_dir:
            Directory to save generated images.
        seed:
            Random seed for sample selection.
        shard_idx:
            Zero-based shard index for parallel runs.
        num_shards:
            Total number of shards.
        device:
            Target device.
        dtype:
            Model dtype.

    Returns:
        DataFrame with one row per sample containing all computed metrics.
    """
    _, val_tf = get_transforms()
    is_bp4d = dataset == "bp4d"

    if is_bp4d:
        ds: BP4DDataset | FFHQDataset = BP4DDataset(
            index_path=BP4D_TEST_INDEX_PATH, transform=val_tf
        )
    else:
        ds = FFHQDataset(split="test", transform=val_tf)

    all_indices = torch.randperm(
        len(ds), generator=torch.Generator().manual_seed(seed)
    )[:num_samples].tolist()
    indices = [
        (i, idx) for i, idx in enumerate(all_indices) if i % num_shards == shard_idx
    ]

    rows: list[dict[str, t.Any]] = []
    for i, idx in tqdm(indices, desc="Evaluating", unit="sample"):
        sample = ds[idx]
        row = _evaluate_sample(
            pipeline=pipeline,
            sample=sample,
            sample_id=i,
            dataset_idx=idx,
            is_bp4d=is_bp4d,
            ds=ds,
            strength=strength,
            au_scale=au_scale,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            gen_dir=gen_dir,
            device=device,
            dtype=dtype,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _evaluate_sample(
    pipeline: "AUAdapterPipeline",
    sample: dict[str, t.Any],
    sample_id: int,
    dataset_idx: int,
    is_bp4d: bool,
    ds: "BP4DDataset | FFHQDataset",
    strength: float,
    au_scale: float,
    guidance_scale: float,
    num_steps: int,
    gen_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, t.Any]:
    source_pil = _tensor_to_pil(tensor=sample["image"])
    arcface = sample["arcface"].unsqueeze(0).to(device=device, dtype=dtype)

    # BP4D: transfer target expression; FFHQ: reconstruct from source AUs.
    requested_aus = (sample["target_aus"] if is_bp4d else sample["aus"]).nan_to_num(0.0)
    aus_in = requested_aus.unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        out = pipeline(
            prompt=sample.get("target_caption", ""),
            aus=aus_in,
            image=source_pil,
            arcface_embeds=arcface,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            au_scale=au_scale,
        )
    gen_pil: Image.Image = out.images[0]

    gen_path = gen_dir / f"sample_{sample_id:04d}.png"
    gen_pil.save(gen_path)

    arcface_cosim: float | None = None
    if pipeline.arcface_extractor is not None:
        gen_bgr = np.array(gen_pil.convert("RGB"))[:, :, ::-1]
        gen_arcface = pipeline.arcface_extractor.embed(gen_bgr).to(device)
        src_arcface = sample["arcface"].to(device)
        arcface_cosim = F.cosine_similarity(
            src_arcface.unsqueeze(0), gen_arcface.unsqueeze(0), dim=1
        ).item()

    latent_d_src_tgt: float | None = None
    latent_d_pred_tgt: float | None = None
    latent_d_src_pred: float | None = None
    if is_bp4d:
        z_src = _encode_latent(vae=pipeline.vae, img=sample["image"], device=device)
        z_tgt = _encode_latent(
            vae=pipeline.vae, img=sample["target_image"], device=device
        )
        gen_tensor = _pil_to_normalized_tensor(gen_pil)
        z_pred = _encode_latent(vae=pipeline.vae, img=gen_tensor, device=device)
        latent_d_src_tgt = _rmse(z_src, z_tgt)
        latent_d_pred_tgt = _rmse(z_pred, z_tgt)
        latent_d_src_pred = _rmse(z_src, z_pred)

    row: dict[str, t.Any] = {
        "sample_id": sample_id,
        "dataset_idx": dataset_idx,
        "generated_path": str(gen_path),
        "requested_aus": requested_aus.tolist(),
        "source_aus": sample["aus"].nan_to_num(0.0).tolist(),
        "arcface_cosim": arcface_cosim,
        "latent_d_src_tgt": latent_d_src_tgt,
        "latent_d_pred_tgt": latent_d_pred_tgt,
        "latent_d_src_pred": latent_d_src_pred,
    }
    if is_bp4d:
        row["subject"] = sample["subject"]
        row["task"] = sample["task"]
    else:
        row["image_id"] = Path(  # type: ignore[union-attr]
            ds.df.iloc[dataset_idx]["image_number"]
        ).stem

    return row


def _print_au_metrics(metadata: pd.DataFrame, detections: pd.DataFrame) -> None:
    """Compute and print AU MSE and distribution shift metrics.

    Args:
        metadata:
            Per-sample metadata from the generation phase.
        detections:
            Per-sample AU detections from detect_generated_aus.py.
    """
    merged = metadata.merge(detections, on="sample_id", how="inner")
    n = len(merged)
    logger.info(f"Computing AU metrics over {n} samples with detections.")

    requested = np.stack(merged["requested_aus"].tolist())
    source = np.stack(merged["source_aus"].tolist())
    detected = merged[[f"detected_{au}" for au in PYFEAT_AU_COLUMNS]].values

    au_mse = float(np.mean((detected - requested) ** 2))
    au_mae = float(np.mean(np.abs(detected - requested)))

    w_src_req = float(
        np.mean(
            [
                wasserstein_distance(source[:, k], requested[:, k])
                for k in range(len(PYFEAT_AU_COLUMNS))
            ]
        )
    )
    w_det_req = float(
        np.mean(
            [
                wasserstein_distance(detected[:, k], requested[:, k])
                for k in range(len(PYFEAT_AU_COLUMNS))
            ]
        )
    )
    improvement = w_det_req / max(w_src_req, 1e-8)

    logger.info("AU metrics:")
    logger.info(f"  MSE (detected vs requested):                {au_mse:.4f}")
    logger.info(f"  MAE (detected vs requested):                {au_mae:.4f}")
    logger.info(f"  Wasserstein source->requested (baseline):   {w_src_req:.4f}")
    logger.info(f"  Wasserstein detected->requested:            {w_det_req:.4f}")
    logger.info(f"  Improvement ratio (< 1 = better):          {improvement:.4f}")


def _print_non_au_summary(metadata: pd.DataFrame, dataset: str) -> None:
    """Print ArcFace and latent distance summary statistics.

    Args:
        metadata:
            Per-sample metadata from the generation phase.
        dataset:
            Dataset name ('bp4d' or 'ffhq').
    """
    cosim = metadata["arcface_cosim"].dropna()
    if len(cosim) > 0:
        logger.info(
            f"ArcFace cosine similarity:  mean={cosim.mean():.4f}  "
            f"median={cosim.median():.4f}  std={cosim.std():.4f}"
        )

    if dataset == "bp4d":
        d_src_tgt = metadata["latent_d_src_tgt"].dropna().values
        d_pred_tgt = metadata["latent_d_pred_tgt"].dropna().values
        valid = d_src_tgt >= max(float(np.percentile(d_src_tgt, 10)), 1e-3)
        ratio = d_pred_tgt[valid] / d_src_tgt[valid]
        improved = float((ratio < 1).mean() * 100)
        logger.info(
            f"Latent distance:  mean d(src,tgt)={d_src_tgt.mean():.4f}  "
            f"mean d(pred,tgt)={d_pred_tgt.mean():.4f}  "
            f"improved={improved:.1f}%  "
            f"median ratio={float(np.median(ratio)):.4f}"
        )


def _encode_latent(
    vae: torch.nn.Module, img: torch.Tensor, device: torch.device
) -> torch.Tensor:
    vae_dtype = next(vae.parameters()).dtype
    x = img.unsqueeze(0).to(device=device, dtype=vae_dtype)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    return latent.squeeze(0).float()


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).pow(2).mean().sqrt().item()


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (
        (tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).astype("uint8")
    return Image.fromarray(arr)


def _pil_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


if __name__ == "__main__":
    main()
