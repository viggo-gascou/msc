from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

base = Path(__file__).parents[2] / "output_data/viz_stable_diffusion_details"
top_steps = [1, 200, 400]
bot_steps = [600, 800, 1000]

def load_strip(steps):
    imgs = [Image.open(base / f"step_{s:04d}.png") for s in steps]
    w, h = imgs[0].size
    strip = Image.new("RGB", (w * len(imgs), h), "white")
    for i, img in enumerate(imgs):
        strip.paste(img, (i * w, 0))
    return np.array(strip), w, h

top_arr, w, h = load_strip(top_steps)
bot_arr, _, _ = load_strip(bot_steps)

cell = 3.0
n = len(top_steps)
fig = plt.figure(figsize=(n * cell, 2 * cell + 0.5), dpi=300)
fig.patch.set_facecolor("white")
gs = GridSpec(2, 1, figure=fig, hspace=-0.135)

# Top strip — labels on top
ax_top = fig.add_subplot(gs[0])
ax_top.imshow(top_arr)
ax_top.set_yticks([])
ax_top.set_xticks([w * (i + 0.5) for i in range(n)])
ax_top.set_xticklabels([str(s) for s in top_steps], fontsize=15, color="#333333")
ax_top.xaxis.set_label_position("top")
ax_top.xaxis.tick_top()
ax_top.tick_params(axis="x", length=0, pad=4)
for spine in ax_top.spines.values():
    spine.set_visible(False)

# Bottom strip — labels on bottom
ax_bot = fig.add_subplot(gs[1])
ax_bot.imshow(bot_arr)
ax_bot.set_yticks([])
ax_bot.set_xticks([w * (i + 0.5) for i in range(n)])
ax_bot.set_xticklabels([str(s) for s in bot_steps], fontsize=15, color="#333333")
ax_bot.tick_params(axis="x", length=0, pad=4)
for spine in ax_bot.spines.values():
    spine.set_visible(False)

out = base / "strength-strip.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out}")
