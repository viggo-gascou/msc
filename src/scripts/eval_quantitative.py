r"""Quantitative evaluation: AU MSE, distribution shift, ArcFace cosim.

Two-phase workflow
------------------
Phase 1 (project env) -- generate images and compute non-AU metrics::

    uv run src/scripts/eval_quantitative.py \
        --model-dir models/sd21_pyfeat_pretrain \
        --dataset bp4d \
        --output-dir eval/sd21_pyfeat_pretrain_bp4d

Phase 2 (pyfeat/libreface env) -- detect AUs on generated images::

    python src/scripts/detect_generated_aus.py \
        --images-dir eval/sd21_pyfeat_pretrain_bp4d/generated \
        --detector pyfeat \
        --output eval/sd21_pyfeat_pretrain_bp4d/au_detections.parquet

Phase 3 (project env) -- compute AU metrics::

    uv run src/scripts/eval_quantitative.py \
        --output-dir eval/sd21_pyfeat_pretrain_bp4d \
        --au-detections eval/sd21_pyfeat_pretrain_bp4d/au_detections.parquet \
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

from msc.constants import (
    BP4D_TEST_INDEX_PATH,
    LIBREFACE_AU_NAME_TO_IDX,
    PYFEAT_AU_NAME_TO_IDX,
    RANDOM_SEED,
)
from msc.data import BP4DDataset, FFHQDataset, get_transforms
from msc.model_utils import load_inference_pipeline

if t.TYPE_CHECKING:
    from msc.inference_pipeline import AUAdapterPipeline

# Fixed emotion AU configs used as conditioning targets during eval.
# Keys are AU names; values are intensities in [0, 1].
# All unspecified AUs are set to 0 (neutral).
AU_CONFIGS: dict[str, dict[str, float]] = {
    "smile": {"AU06": 0.8, "AU12": 0.8},
    "surprise": {"AU01": 0.8, "AU02": 0.8, "AU05": 0.8, "AU26": 0.8},
    "anger": {"AU04": 0.8, "AU05": 0.8, "AU07": 0.8, "AU23": 0.8},
    "sadness": {"AU01": 0.8, "AU04": 0.8, "AU15": 0.8, "AU17": 0.8},
}


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
        "--eval-mode",
        type=str,
        default="emotion",
        choices=["emotion", "paired"],
        help=(
            "'emotion': generate one image per source per AU_CONFIGS emotion. "
            "'paired': generate one image per source conditioned on a random "
            "same-subject target frame (BP4D only). Default: emotion."
        ),
    )
    parser.add_argument(
        "--au-baseline",
        type=str,
        default="source",
        choices=["zero", "source"],
        help=(
            "emotion mode only. How to set unspecified AUs: 'source' uses the source "
            "image AUs as base (default), 'zero' sets all unspecified AUs to 0."
        ),
    )
    parser.add_argument(
        "--min-au-distance",
        type=float,
        default=2.0,
        help=(
            "paired mode only. Minimum L1 AU distance between source and target frame. "
            "Default: 2.0."
        ),
    )
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
    parser.add_argument(
        "--au-configs-json",
        type=str,
        default=None,
        help=(
            "JSON override for the AU_CONFIGS dict used in emotion mode, e.g. "
            "'{\"au05_au12\": {\"AU05\": 1.0, \"AU12\": 1.0}}'. "
            "Defaults to the built-in four-emotion configs."
        ),
    )
    args = parser.parse_args()

    if args.au_configs_json is not None:
        try:
            au_configs: dict[str, dict[str, float]] = json.loads(args.au_configs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --au-configs-json: {args.au_configs_json}") from exc
    else:
        au_configs = AU_CONFIGS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        (output_dir / "args.json").write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8"
        )

    if args.metrics_only:
        metadata = pd.read_parquet(output_dir / "metadata.parquet")
        if args.au_detections is None:
            raise ValueError("--au-detections is required with --metrics-only")
        detections = pd.read_parquet(args.au_detections)
        _print_non_au_summary(metadata=metadata)
        _print_au_metrics(
            metadata=metadata, detections=detections, eval_mode=args.eval_mode,
            au_configs=au_configs,
        )
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
        eval_mode=args.eval_mode,
        au_baseline=args.au_baseline,
        min_au_distance=args.min_au_distance,
        au_configs=au_configs,
    )
    shard_suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""
    metadata_path = output_dir / f"metadata{shard_suffix}.parquet"
    metadata.to_parquet(metadata_path, index=False)
    logger.info(f"Metadata saved to {metadata_path}")

    _print_non_au_summary(metadata=metadata)

    if args.au_detections is not None:
        detections = pd.read_parquet(args.au_detections)
        _print_au_metrics(metadata=metadata, detections=detections, au_configs=au_configs)


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
    eval_mode: str = "emotion",
    au_baseline: str = "source",
    min_au_distance: float = 2.0,
    au_configs: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Generate images and compute ArcFace + latent metrics for all samples.

    In 'emotion' mode, each source image is generated once per AU config.
    In 'paired' mode, each source image is generated once conditioned on a
    randomly sampled same-subject target frame (BP4D only).

    Args:
        pipeline:
            Loaded AUAdapterPipeline with arcface_extractor set.
        dataset:
            Dataset to evaluate on ('bp4d' or 'ffhq').
        num_samples:
            Number of source images to evaluate.
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
        eval_mode (optional):
            'emotion' uses fixed AU_CONFIGS targets; 'paired' uses random
            same-subject target frames. Defaults to 'emotion'.
        au_baseline (optional):
            emotion mode only. 'source' uses source AUs as base, 'zero' uses
            all zeros. Defaults to 'source'.
        min_au_distance (optional):
            paired mode only. Minimum L1 AU distance between source and target.
            Defaults to 2.0.

    Returns:
        DataFrame with one row per generated image.
    """
    _, val_tf = get_transforms()
    is_bp4d = dataset == "bp4d"
    detector = "libreface" if pipeline.au_encoder.num_aus == 17 else "pyfeat"

    if is_bp4d:
        ds: BP4DDataset | FFHQDataset = BP4DDataset(
            index_path=BP4D_TEST_INDEX_PATH, transform=val_tf, detector=detector
        )
        if eval_mode == "paired":
            ds.set_min_au_distance(min_au_distance)
    else:
        ds = FFHQDataset(split="test", transform=val_tf, detector=detector)

    au_name_to_idx = (
        LIBREFACE_AU_NAME_TO_IDX if detector == "libreface" else PYFEAT_AU_NAME_TO_IDX
    )

    all_indices = torch.randperm(
        len(ds), generator=torch.Generator().manual_seed(seed)
    )[:num_samples].tolist()
    source_indices = [
        (i, idx) for i, idx in enumerate(all_indices) if i % num_shards == shard_idx
    ]

    rows: list[dict[str, t.Any]] = []
    for i, idx in tqdm(source_indices, desc="Evaluating", unit="source"):
        sample = ds[idx]

        if eval_mode == "paired":
            aus_tensor = sample["target_aus"].nan_to_num(0.0)
            row = _evaluate_sample(
                pipeline=pipeline,
                sample=sample,
                sample_id=i,
                dataset_idx=idx,
                config_name="paired",
                requested_aus=aus_tensor,
                is_bp4d=is_bp4d,
                ds=ds,
                strength=strength,
                au_scale=au_scale,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                gen_dir=gen_dir,
                device=device,
                dtype=dtype,
                paired_latents=True,
            )
            rows.append(row)
        else:
            for config_name, overrides in (au_configs or AU_CONFIGS).items():
                if au_baseline == "zero":
                    aus_tensor = torch.zeros(
                        pipeline.au_encoder.num_aus, dtype=torch.float32
                    )
                else:
                    aus_tensor = sample["aus"].nan_to_num(0.0).clone()
                for au_name, intensity in overrides.items():
                    if au_name in au_name_to_idx:
                        aus_tensor[au_name_to_idx[au_name]] = intensity
                row = _evaluate_sample(
                    pipeline=pipeline,
                    sample=sample,
                    sample_id=i,
                    dataset_idx=idx,
                    config_name=config_name,
                    requested_aus=aus_tensor,
                    is_bp4d=is_bp4d,
                    ds=ds,
                    strength=strength,
                    au_scale=au_scale,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    gen_dir=gen_dir,
                    device=device,
                    dtype=dtype,
                    paired_latents=False,
                )
                rows.append(row)

    return pd.DataFrame(rows)


def _evaluate_sample(
    pipeline: "AUAdapterPipeline",
    sample: dict[str, t.Any],
    sample_id: int,
    dataset_idx: int,
    config_name: str,
    requested_aus: torch.Tensor,
    is_bp4d: bool,
    ds: "BP4DDataset | FFHQDataset",
    strength: float,
    au_scale: float,
    guidance_scale: float,
    num_steps: int,
    gen_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    paired_latents: bool = False,
) -> dict[str, t.Any]:
    source_pil = _tensor_to_pil(tensor=sample["image"])
    arcface = sample["arcface"].unsqueeze(0).to(device=device, dtype=dtype)
    aus_in = requested_aus.unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        out = pipeline(
            prompt=sample["target_caption"],
            aus=aus_in,
            image=source_pil,
            arcface_embeds=arcface,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            au_scale=au_scale,
        )
    gen_pil: Image.Image = out.images[0]

    gen_path = gen_dir / f"sample_{sample_id:04d}_{config_name}.png"
    gen_pil.save(gen_path)

    arcface_cosim: float | None = None
    if pipeline.arcface_extractor is not None:
        gen_bgr = np.array(gen_pil.convert("RGB"))[:, :, ::-1]
        gen_arcface = pipeline.arcface_extractor.embed(gen_bgr).to(device)
        src_arcface = sample["arcface"].to(device)
        arcface_cosim = F.cosine_similarity(
            src_arcface.unsqueeze(0), gen_arcface.unsqueeze(0), dim=1
        ).item()

    z_src = _encode_latent(vae=pipeline.vae, img=sample["image"], device=device)
    gen_tensor = _pil_to_normalized_tensor(gen_pil)
    z_pred = _encode_latent(vae=pipeline.vae, img=gen_tensor, device=device)
    latent_d_src_pred = _rmse(z_src, z_pred)

    latent_d_src_tgt: float | None = None
    latent_d_pred_tgt: float | None = None
    if paired_latents:
        z_tgt = _encode_latent(
            vae=pipeline.vae, img=sample["target_image"], device=device
        )
        latent_d_src_tgt = _rmse(z_src, z_tgt)
        latent_d_pred_tgt = _rmse(z_pred, z_tgt)

    row: dict[str, t.Any] = {
        "sample_id": sample_id,
        "dataset_idx": dataset_idx,
        "config_name": config_name,
        "generated_path": str(gen_path),
        "requested_aus": requested_aus.tolist(),
        "source_aus": sample["aus"].nan_to_num(0.0).tolist(),
        "arcface_cosim": arcface_cosim,
        "latent_d_src_pred": latent_d_src_pred,
        "latent_d_src_tgt": latent_d_src_tgt,
        "latent_d_pred_tgt": latent_d_pred_tgt,
    }
    if is_bp4d:
        row["subject"] = sample["subject"]
        row["task"] = sample["task"]
    else:
        row["image_id"] = Path(  # type: ignore[union-attr]
            ds.df.iloc[dataset_idx]["image_number"]
        ).stem

    return row


def _print_au_metrics(
    metadata: pd.DataFrame,
    detections: pd.DataFrame,
    eval_mode: str = "emotion",
    au_configs: dict[str, dict[str, float]] | None = None,
) -> None:
    """Compute and print AU MSE and Wasserstein metrics.

    Args:
        metadata:
            Per-sample metadata from the generation phase.
        detections:
            Per-sample AU detections from detect_generated_aus.py.
        eval_mode (optional):
            'emotion' prints per-config metrics vs fixed AU_CONFIGS targets.
            'paired' prints overall AU accuracy vs the requested target AUs.
            Defaults to 'emotion'.
    """
    merged = metadata.merge(detections, on=["sample_id", "config_name"], how="inner")
    logger.info(f"Computing AU metrics over {len(merged)} samples with detections.")

    au_cols = [c for c in detections.columns if c.startswith("detected_")]
    au_names = [c.removeprefix("detected_") for c in au_cols]

    if eval_mode == "paired":
        detected = merged[au_cols].values
        source = np.stack(merged["source_aus"].tolist())
        requested = np.stack(merged["requested_aus"].tolist())
        n = min(detected.shape[1], requested.shape[1])
        au_mae = float(np.nanmean(np.abs(detected[:, :n] - requested[:, :n])))
        au_mse = float(np.nanmean((detected[:, :n] - requested[:, :n]) ** 2))
        src_det_mae = float(np.nanmean(np.abs(detected[:, :n] - source[:, :n])))
        logger.info(f"  [paired]  n={len(merged)}")
        logger.info(f"    MAE  detected vs requested AUs: {au_mae:.4f}")
        logger.info(f"    MSE  detected vs requested AUs: {au_mse:.4f}")
        logger.info(f"    MAE  src->det (all AUs):        {src_det_mae:.4f}")
        return

    per_config: list[dict[str, float]] = []

    for config_name, group in merged.groupby("config_name"):
        detected = group[au_cols].values
        source = np.stack(group["source_aus"].tolist())
        config_aus = (au_configs or AU_CONFIGS)[str(config_name)]

        # Build (N, num_active) arrays using only the AUs present in this config
        # and in the detection output, comparing against the known fixed targets.
        active_cols = [
            i
            for i, name in enumerate(au_names)
            if name.upper() in {k.upper() for k in config_aus}
        ]
        target_vals = np.array(
            [config_aus[au_names[i].upper()] for i in active_cols], dtype=np.float32
        )

        inactive_cols = [i for i in range(len(au_names)) if i not in active_cols]

        det_active = detected[:, active_cols]  # (N, num_active)
        target_row = np.broadcast_to(target_vals, det_active.shape)
        det_inactive = detected[:, inactive_cols]  # (N, num_inactive), should be ~0

        au_mse = float(np.mean((det_active - target_row) ** 2))
        au_mae = float(np.mean(np.abs(det_active - target_row)))
        w_det_req = float(
            np.mean(
                [
                    wasserstein_distance(
                        det_active[:, j], np.full(len(group), target_vals[j])
                    )
                    for j in range(len(active_cols))
                ]
            )
        )
        inactive_mse = float(np.nanmean(det_inactive**2))
        inactive_mae = float(np.nanmean(np.abs(det_inactive)))

        n_det = min(detected.shape[1], source.shape[1])
        src_det_mae = float(np.nanmean(np.abs(detected[:, :n_det] - source[:, :n_det])))
        src_det_mse = float(np.nanmean((detected[:, :n_det] - source[:, :n_det]) ** 2))

        per_config.append(
            {
                "au_mse": au_mse,
                "au_mae": au_mae,
                "w_det_req": w_det_req,
                "inactive_mse": inactive_mse,
                "inactive_mae": inactive_mae,
                "src_det_mae": src_det_mae,
                "src_det_mse": src_det_mse,
            }
        )

        tgt_mean = target_vals.mean()
        logger.info(f"  [{config_name}]  n={len(group)}")
        logger.info(f"    MSE  active AUs (target={tgt_mean:.2f}): {au_mse:.4f}")
        logger.info(f"    MAE  active AUs:                         {au_mae:.4f}")
        logger.info(f"    Wasserstein det->req (active):           {w_det_req:.4f}")
        logger.info(f"    MSE  inactive AUs (target=0):            {inactive_mse:.4f}")
        logger.info(f"    MAE  inactive AUs:                       {inactive_mae:.4f}")
        logger.info(f"    MAE  src->det (all AUs):                 {src_det_mae:.4f}")
        logger.info(f"    MSE  src->det (all AUs):                 {src_det_mse:.4f}")

    if per_config:
        avg = {k: float(np.mean([d[k] for d in per_config])) for k in per_config[0]}
        logger.info("  [average across all configs]")
        logger.info(f"    MSE  active AUs:               {avg['au_mse']:.4f}")
        logger.info(f"    MAE  active AUs:               {avg['au_mae']:.4f}")
        logger.info(f"    Wasserstein det->req (active): {avg['w_det_req']:.4f}")
        logger.info(f"    MSE  inactive AUs (target=0):  {avg['inactive_mse']:.4f}")
        logger.info(f"    MAE  inactive AUs:             {avg['inactive_mae']:.4f}")
        logger.info(f"    MAE  src->det (all AUs):       {avg['src_det_mae']:.4f}")
        logger.info(f"    MSE  src->det (all AUs):       {avg['src_det_mse']:.4f}")


def _print_non_au_summary(metadata: pd.DataFrame) -> None:
    """Print ArcFace cosim and latent distance summary statistics.

    Args:
        metadata:
            Per-sample metadata from the generation phase.
    """
    cosim = metadata["arcface_cosim"].dropna()
    if len(cosim) > 0:
        logger.info(
            f"ArcFace cosine similarity:  mean={cosim.mean():.4f}  "
            f"median={cosim.median():.4f}  std={cosim.std():.4f}"
        )

    d_src_pred = metadata["latent_d_src_pred"].dropna()
    if len(d_src_pred) > 0:
        logger.info(
            f"Latent d(src, pred):  mean={d_src_pred.mean():.4f}  "
            f"median={d_src_pred.median():.4f}"
        )

    if "latent_d_src_tgt" in metadata.columns:
        d_src_tgt = metadata["latent_d_src_tgt"].dropna()
        d_pred_tgt = metadata["latent_d_pred_tgt"].dropna()
        if len(d_src_tgt) > 0:
            ratio = (d_pred_tgt / d_src_tgt).dropna()
            logger.info(
                f"Latent d(src, tgt):   mean={d_src_tgt.mean():.4f}  "
                f"median={d_src_tgt.median():.4f}"
            )
            logger.info(
                f"Latent d(pred, tgt):  mean={d_pred_tgt.mean():.4f}  "
                f"median={d_pred_tgt.median():.4f}"
            )
            logger.info(
                f"Ratio d(pred,tgt)/d(src,tgt):  "
                f"mean={ratio.mean():.4f}  median={ratio.median():.4f}  "
                f"(< 1 means pred is closer to target than source)"
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
