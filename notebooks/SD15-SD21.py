import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt

    from msc.constants import FIGURES_DIR
    from msc.plot_utils import load_thumb, set_plotting_style

    set_plotting_style()

    BASE_DIR = Path("SD21vSD15").resolve()
    REFERENCE_IMG = BASE_DIR / "00002.png"
    SD_15_IMG = BASE_DIR / "sd_15_pretrain_str_02_gd_10_lf_00002.png"
    SD_21_IMG = BASE_DIR / "sd_21_pretrain_str_02_gd_10_py_00002.png"

    titles = ["Source", "SD 1.5", "SD 2.1"]
    paths = [REFERENCE_IMG, SD_15_IMG, SD_21_IMG]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, path, title in zip(axes, paths, titles):
        ax.imshow(load_thumb(path, ax))
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sd21_vs_sd15.pdf")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
