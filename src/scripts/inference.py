"""Inference script for the AU IP Adapter model."""

import h5py
import pandas as pd
import torch
from omegaconf import OmegaConf

from msc.constants import BP4D_AU_COLUMNS
from msc.model_utils import load_inference_pipeline

cfg = OmegaConf.load("checkpoints/config.yaml")

pipeline = load_inference_pipeline(
    cfg.parameters,
    cfg.ip_adapter,
    "checkpoints/best_au_adapter.safetensors",
    device="cpu",
)

df = pd.read_parquet("data/BP4D/Sample/bp4d_index.parquet")

subject = "F001"
row = df[df["subject"] == subject].iloc[0]

with h5py.File("data/BP4D/Sample/Embeddings/T1.h5", "r") as f:
    arcface = torch.from_numpy(f[subject]["arcface"][0]).unsqueeze(0)  # (1, 512)

aus = torch.tensor(
    [row.get(col, 0.0) for col in BP4D_AU_COLUMNS], dtype=torch.float32
).unsqueeze(0)  # (1, 23)

output = pipeline(prompt="", aus={"AU12": 1.0, "AU06": 2.0}, arcface_embeds=arcface)
output.images[0].save(f"inference_{subject}.png")
