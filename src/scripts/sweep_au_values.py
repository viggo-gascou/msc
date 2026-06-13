"""Sweep a single AU across out-of-distribution values to probe model behaviour."""

import argparse
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from msc.constants import LIBREFACE_AU_NAME_TO_IDX, PYFEAT_AU_NAME_TO_IDX
from msc.data import FFHQDataset, get_transforms
from msc.model_utils import load_inference_pipeline

_AU_VALUES: list[float] = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
_CELL_SIZE: int = 256

# FFHQ image IDs present in both the PyFeat and LibreFace pretrain eval runs.
_OVERLAPPING_IMAGE_IDS: list[str] = [
    "01401",
    "01524",
    "02277",
    "02455",
    "02528",
    "02728",
    "03575",
    "03929",
    "04278",
    "06047",
    "06242",
    "06358",
    "06915",
    "08607",
    "09172",
    "09725",
    "09855",
    "09864",
    "09948",
    "10630",
    "10842",
    "11152",
    "11488",
    "13869",
    "13984",
    "15165",
    "16199",
    "17078",
    "17780",
    "19546",
    "19554",
    "20558",
    "20587",
    "21070",
    "21237",
    "21609",
    "21940",
    "24469",
    "25008",
    "25241",
    "26544",
    "26563",
    "27051",
    "28902",
    "29369",
    "31477",
    "31501",
    "31512",
    "31763",
    "32282",
    "33541",
    "34272",
    "36102",
    "36591",
    "37051",
    "37216",
    "37537",
    "37775",
    "38338",
    "38941",
    "39576",
    "39900",
    "42549",
    "42686",
    "44162",
    "44275",
    "44974",
    "45923",
    "46270",
    "47245",
    "47402",
    "47452",
    "47966",
    "49085",
    "50269",
    "51951",
    "52078",
    "53020",
    "53704",
    "54256",
    "54442",
    "54736",
    "55503",
    "56298",
    "57655",
    "60651",
    "60661",
    "60984",
    "66467",
    "66894",
    "67049",
    "67313",
    "68988",
    "69271",
    "69528",
]


def main() -> None:
    """Run AU value sweep and save one grid image per source sample.

    Raises:
        ValueError: If specified AU names or image IDs are not found in the dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--au-adapter-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--au-names", type=str, nargs="+", default=["AU05", "AU12"])
    parser.add_argument(
        "--detector", type=str, default="pyfeat", choices=["pyfeat", "libreface"]
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--image-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific FFHQ image IDs to use (e.g. 04278 01401). Overrides --num-samples.",
    )
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--au-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/au_ood_sweep")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    au_adapter_path, config_path = _resolve_model_paths(args)
    cfg = OmegaConf.load(config_path)
    pipeline = load_inference_pipeline(
        cfg.parameters, au_adapter_path, device=args.device, load_arcface=False
    )

    au_name_to_idx = (
        LIBREFACE_AU_NAME_TO_IDX
        if args.detector == "libreface"
        else PYFEAT_AU_NAME_TO_IDX
    )
    for au_name in args.au_names:
        if au_name not in au_name_to_idx:
            raise ValueError(f"{au_name} not found for detector {args.detector}")

    _, val_tf = get_transforms()
    ds = FFHQDataset(split="test", transform=val_tf, detector=args.detector)

    image_ids = args.image_ids if args.image_ids is not None else _OVERLAPPING_IMAGE_IDS
    image_ids = image_ids[: args.num_samples]
    id_to_idx = {Path(row["image_number"]).stem: i for i, row in ds.df.iterrows()}
    missing = [iid for iid in image_ids if iid not in id_to_idx]
    if missing:
        raise ValueError(f"Image IDs not found in dataset: {missing}")
    indices = [id_to_idx[iid] for iid in image_ids]

    for au_name in args.au_names:
        au_idx = au_name_to_idx[au_name]
        out_dir = Path(args.output_dir) / au_name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Sweeping {au_name}")

        for i, idx in enumerate(indices):
            sample = ds[idx]
            source_pil = _tensor_to_pil(sample["image"])
            arcface = sample["arcface"].unsqueeze(0).to(device=device, dtype=dtype)

            col_images: list[Image.Image] = [
                source_pil.resize((_CELL_SIZE, _CELL_SIZE))
            ]
            col_labels: list[str] = ["source"]

            for val in _AU_VALUES:
                aus = sample["aus"].nan_to_num(0.0).clone()
                aus[au_idx] = val
                aus_in = aus.unsqueeze(0).to(device=device, dtype=dtype)

                with torch.no_grad():
                    out = pipeline(
                        prompt="",
                        aus=aus_in,
                        image=source_pil,
                        arcface_embeds=arcface,
                        num_inference_steps=args.num_steps,
                        strength=args.strength,
                        guidance_scale=args.guidance_scale,
                        au_scale=args.au_scale,
                    )
                col_images.append(out.images[0].resize((_CELL_SIZE, _CELL_SIZE)))
                col_labels.append(f"{au_name}={val:+.1f}")

            grid = _make_row_grid(images=col_images, labels=col_labels)
            out_path = out_dir / f"sample_{i:02d}.png"
            grid.save(out_path)
            logger.info(f"Saved {out_path}")


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


def _make_row_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    """Arrange images in a single row with labels on top.

    Args:
        images: The list of images to arrange.
        labels: The list of labels corresponding to the images.

    Returns:
        The arranged grid image.
    """
    label_h = 24
    cell_w, cell_h = images[0].size
    grid = Image.new(
        "RGB", (cell_w * len(images), cell_h + label_h), color=(20, 20, 20)
    )
    for i, (img, label) in enumerate(zip(images, labels)):
        grid.paste(img, (i * cell_w, label_h))
        _draw_label(grid, label=label, x=i * cell_w, y=0, w=cell_w, h=label_h)
    return grid


def _draw_label(img: Image.Image, label: str, x: int, y: int, w: int, h: int) -> None:
    """Draw a label bar at the given position.

    Args:
        img: The image to draw on.
        label: The label text to display.
        x: The x-coordinate of the top-left corner.
        y: The y-coordinate of the top-left corner.
        w: The width of the label bar.
        h: The height of the label bar.
    """
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], fill=(30, 30, 30))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", h - 6
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((x + 4, y + 2), label, fill=(220, 220, 220), font=font)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalised [-1, 1] CHW tensor to a PIL image.

    Args:
        tensor: The input tensor to convert.

    Returns:
        The converted PIL image.
    """
    arr = (
        (tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).astype("uint8")
    return Image.fromarray(arr)


if __name__ == "__main__":
    main()
