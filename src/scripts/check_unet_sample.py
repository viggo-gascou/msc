"""Quick script to inspect the output of unet.sample."""

import torch
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet"
)
unet.eval()

batch_size = 1
latent_channels = unet.config.in_channels  # 4
height = width = 64  # 512 / 8

noisy_latents = torch.randn(batch_size, latent_channels, height, width)
timesteps = torch.tensor([500])
encoder_hidden_states = torch.randn(batch_size, 77, 768)

with torch.no_grad():
    output = unet(noisy_latents, timesteps, encoder_hidden_states)

print("type:", type(output))
print("fields:", [k for k in output.keys()])
print("sample shape:", output.sample.shape)
print("sample dtype:", output.sample.dtype)
print("sample min/max:", output.sample.min().item(), output.sample.max().item())
