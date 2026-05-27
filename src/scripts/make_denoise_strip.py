from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

base = Path(__file__).parents[2] / "output_data/viz_stable_diffusion_details"
steps = [1, 200, 400, 600, 800, 1000]

imgs = [Image.open(base / f"step_{s:04d}.png") for s in steps]
w, h = imgs[0].size

# Stitch images with PIL — guaranteed zero gap
strip = Image.new("RGB", (w * len(steps), h), "white")
for i, img in enumerate(imgs):
    strip.paste(img, (i * w, 0))

# Single axes so matplotlib handles only the labels
cell = 3.0
fig, ax = plt.subplots(figsize=(len(steps) * cell, cell + 0.3), dpi=300)
fig.patch.set_facecolor("white")

ax.imshow(np.array(strip))
ax.set_yticks([])
ax.set_xticks([w * (i + 0.5) for i in range(len(steps))])
ax.set_xticklabels([str(s) for s in steps], fontsize=10, color="#333333")
ax.tick_params(axis="x", length=0, pad=4)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0.06)

out = base / "strip.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out}")
