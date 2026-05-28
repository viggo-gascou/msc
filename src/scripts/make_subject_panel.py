"""Create a subject panel figure showing diverse BP4D participants."""

from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from msc.constants import BP4D_PREPROCESSED_DIR

SUBJECTS = ["F001", "F006", "F012", "M001", "M008", "M015"]
TASK = "T1"
OUT_PATH = Path("figures/bp4d_subjects.png")
GAP = 4


def main() -> None:
    """Load one frame per subject from HDF5 and save a side-by-side panel."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    h5_path = BP4D_PREPROCESSED_DIR / f"{TASK}.h5"
    faces: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as f:
        for subject in SUBJECTS:
            if subject not in f:
                print(f"Warning: {subject} not in HDF5, skipping")
                continue
            face = f[subject]["faces"][0]  # (3, H, W), float32 in [-1, 1]
            faces.append(face)

    n = len(faces)
    h, w = faces[0].shape[1], faces[0].shape[2]

    canvas = Image.new("RGB", (n * w + (n - 1) * GAP, h), color=(255, 255, 255))

    for i, face in enumerate(faces):
        pixels = ((face + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(pixels.transpose(1, 2, 0))
        canvas.paste(img, (i * (w + GAP), 0))

    canvas.save(OUT_PATH)
    print(f"Saved panel to {OUT_PATH} ({canvas.width}x{canvas.height}px, {n} subjects)")


if __name__ == "__main__":
    main()
