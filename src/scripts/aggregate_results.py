"""Aggregate evaluation metrics from all eval/ directories into a single CSV.

Usage::

    uv run src/scripts/aggregate_results.py
    uv run src/scripts/aggregate_results.py --eval-dir eval/ --output results.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wasserstein_distance

AU_CONFIGS: dict[str, dict[str, float]] = {
    "smile": {"AU06": 0.8, "AU12": 0.8},
    "surprise": {"AU01": 0.8, "AU02": 0.8, "AU05": 0.8, "AU26": 0.8},
    "anger": {"AU04": 0.8, "AU05": 0.8, "AU07": 0.8, "AU23": 0.8},
    "sadness": {"AU01": 0.6, "AU04": 0.7, "AU15": 0.6, "AU17": 0.5},
}


def main() -> None:
    """Aggregate metrics from all eval directories."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, default="eval")
    parser.add_argument("--output", type=str, default="results.parquet")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    rows = []

    for d in sorted(eval_dir.iterdir()):
        metadata_path = d / "metadata.parquet"
        detections_path = d / "au_detections.parquet"
        args_path = d / "args.json"

        if not (metadata_path.exists() and detections_path.exists()):
            continue

        eval_mode = "emotion"
        if args_path.exists():
            saved = json.loads(args_path.read_text(encoding="utf-8"))
            eval_mode = saved.get("eval_mode", "emotion")

        logger.info(f"Processing {d.name} (mode={eval_mode})")

        metadata = pd.read_parquet(metadata_path)
        detections = pd.read_parquet(detections_path)

        row: dict[str, object] = {"run": d.name, "eval_mode": eval_mode}

        cosim = metadata["arcface_cosim"].dropna()
        if len(cosim):
            row["arcface_mean"] = cosim.mean()
            row["arcface_median"] = cosim.median()
            row["arcface_std"] = cosim.std()

        d_src_pred = metadata["latent_d_src_pred"].dropna()
        if len(d_src_pred):
            row["latent_d_src_pred_mean"] = d_src_pred.mean()
            row["latent_d_src_pred_median"] = d_src_pred.median()

        if "latent_d_src_tgt" in metadata.columns:
            d_src_tgt = metadata["latent_d_src_tgt"].dropna()
            d_pred_tgt = metadata["latent_d_pred_tgt"].dropna()
            if len(d_src_tgt):
                ratio = d_pred_tgt / d_src_tgt
                row["latent_d_src_tgt_mean"] = d_src_tgt.mean()
                row["latent_d_pred_tgt_mean"] = d_pred_tgt.mean()
                row["latent_ratio_mean"] = ratio.mean()
                row["latent_ratio_median"] = ratio.median()

        merged = metadata.merge(
            detections, on=["sample_id", "config_name"], how="inner"
        )
        au_cols = [c for c in detections.columns if c.startswith("detected_")]
        au_names = [c.removeprefix("detected_") for c in au_cols]

        if eval_mode == "paired":
            detected = merged[au_cols].values
            source = np.stack(merged["source_aus"].tolist())
            requested = np.stack(merged["requested_aus"].tolist())
            n = min(detected.shape[1], requested.shape[1])
            row["au_mae"] = float(
                np.nanmean(np.abs(detected[:, :n] - requested[:, :n]))
            )
            row["au_mse"] = float(np.nanmean((detected[:, :n] - requested[:, :n]) ** 2))
            row["src_det_mae"] = float(
                np.nanmean(np.abs(detected[:, :n] - source[:, :n]))
            )
        else:
            per_config: list[dict[str, object]] = []
            for config_name, group in merged.groupby("config_name"):
                detected = group[au_cols].values
                source = np.stack(group["source_aus"].tolist())
                config_aus = AU_CONFIGS[str(config_name)]

                active_cols = [
                    i
                    for i, name in enumerate(au_names)
                    if name.upper() in {k.upper() for k in config_aus}
                ]
                target_vals = np.array(
                    [config_aus[au_names[i].upper()] for i in active_cols],
                    dtype=np.float32,
                )
                inactive_cols = [
                    i for i in range(len(au_names)) if i not in active_cols
                ]

                det_active = detected[:, active_cols]
                target_row = np.broadcast_to(target_vals, det_active.shape)
                det_inactive = detected[:, inactive_cols]
                n = min(detected.shape[1], source.shape[1])

                emotion_metrics: dict[str, object] = {
                    "au_mse": float(np.mean((det_active - target_row) ** 2)),
                    "au_mae": float(np.mean(np.abs(det_active - target_row))),
                    "wasserstein": float(
                        np.mean(
                            [
                                wasserstein_distance(
                                    det_active[:, j],
                                    np.full(len(group), target_vals[j]),
                                )
                                for j in range(len(active_cols))
                            ]
                        )
                    ),
                    "inactive_mse": float(np.nanmean(det_inactive**2)),
                    "inactive_mae": float(np.nanmean(np.abs(det_inactive))),
                    "src_det_mae": float(
                        np.nanmean(np.abs(detected[:, :n] - source[:, :n]))
                    ),
                    "src_det_mse": float(
                        np.nanmean((detected[:, :n] - source[:, :n]) ** 2)
                    ),
                }
                per_config.append(emotion_metrics)

                # Per-emotion row (for appendix table), with per-emotion arcface
                emotion_row: dict[str, object] = {
                    **row,
                    "emotion": str(config_name),
                    **emotion_metrics,
                }
                config_meta = metadata[metadata["config_name"] == config_name]
                config_cosim = config_meta["arcface_cosim"].dropna()
                if len(config_cosim):
                    emotion_row["arcface_mean"] = config_cosim.mean()
                    emotion_row["arcface_median"] = config_cosim.median()
                    emotion_row["arcface_std"] = config_cosim.std()
                rows.append(emotion_row)

            if per_config:
                for k in per_config[0]:
                    row[k] = float(np.mean([d[k] for d in per_config]))

        rows.append(row)

    df = pd.DataFrame(rows)
    output = Path(args.output)
    df.to_parquet(output, index=False)
    logger.info(f"Saved {len(df)} rows to {output}")
    print(df.to_string())


if __name__ == "__main__":
    main()
