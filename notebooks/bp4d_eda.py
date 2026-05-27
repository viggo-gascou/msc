import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def imports():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from msc.plot_utils import set_plotting_style
    from msc.constants import (
        BP4D_INDEX_PATH,
        BP4D_AU_INTENSITY_COLUMNS,
        BP4D_AU_OCCURRENCE_COLUMNS,
        LIBREFACE_BP4D_DIR,
        LIBREFACE_AU_COLUMN_MAP,
        LIBREFACE_FFHQ_PATH,
        PYFEAT_BP4D_DIR,
        PYFEAT_AU_COLUMNS,
        FIGURES_DIR,
    )

    set_plotting_style()
    return (
        BP4D_AU_INTENSITY_COLUMNS,
        BP4D_AU_OCCURRENCE_COLUMNS,
        FIGURES_DIR,
        LIBREFACE_AU_COLUMN_MAP,
        LIBREFACE_BP4D_DIR,
        LIBREFACE_FFHQ_PATH,
        PYFEAT_AU_COLUMNS,
        PYFEAT_BP4D_DIR,
        pd,
        plt,
        sns,
    )


@app.cell
def data_loading(LIBREFACE_AU_COLUMN_MAP, LIBREFACE_BP4D_DIR, pd):
    tasks = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    dfs = []
    for task in tasks:
        df = pd.read_parquet(LIBREFACE_BP4D_DIR / f"libreface_bp4d_{task}.parquet")
        df["task"] = task
        dfs.append(df)
    df_libreface = pd.concat(dfs, ignore_index=True)

    col_rename = {v: k for k, v in LIBREFACE_AU_COLUMN_MAP.items()}
    df_libreface = df_libreface[["task", *LIBREFACE_AU_COLUMN_MAP.values()]].rename(columns=col_rename)

    au_occurrence_cols = sorted([k for k, v in LIBREFACE_AU_COLUMN_MAP.items() if not v.endswith("_intensity")])
    au_intensity_cols = sorted([k for k, v in LIBREFACE_AU_COLUMN_MAP.items() if v.endswith("_intensity")])
    return au_intensity_cols, au_occurrence_cols, df_libreface


@app.cell
def histogram_helper(plt, sns):
    def plot_au_histograms(df, cols, title, fname, FIGURES_DIR, normalize=False, bins=20):
        n = len(cols)
        ncols = min(n, 6)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
        axes = axes.flatten() if n > 1 else [axes]
        data = df[cols] / 5 if normalize else df[cols]
        for ax, col in zip(axes, cols):
            sns.histplot(data[col], ax=ax, bins=bins, color="steelblue")
            ax.set_title(col)
            ax.set_xlabel("")
            ax.set_ylabel("")
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(title, y=1.01)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / fname)
        plt.show()

    return (plot_au_histograms,)


@app.cell
def au_distribution(
    FIGURES_DIR,
    au_intensity_cols,
    au_occurrence_cols,
    df_libreface,
    plot_au_histograms,
):
    plot_au_histograms(df_libreface, au_occurrence_cols, "BP4D LibreFace Binary AUs", "bp4d_libreface_occ_distribution.pdf", FIGURES_DIR)
    plot_au_histograms(df_libreface, au_intensity_cols, "BP4D LibreFace Intensity AUs (normalised)", "bp4d_libreface_int_distribution.pdf", FIGURES_DIR, normalize=True)
    return


@app.cell
def bp4d_data_loading(BP4D_AU_INTENSITY_COLUMNS, pd):
    df_bp4d = pd.read_parquet("data/BP4D/bp4d_index.parquet")
    bp4d_int_cols = [f"{au}_int" for au in BP4D_AU_INTENSITY_COLUMNS]
    return bp4d_int_cols, df_bp4d


@app.cell
def bp4d_distribution(
    BP4D_AU_OCCURRENCE_COLUMNS,
    FIGURES_DIR,
    bp4d_int_cols,
    df_bp4d,
    plot_au_histograms,
):
    plot_au_histograms(df_bp4d, BP4D_AU_OCCURRENCE_COLUMNS, "BP4D Ground Truth Occurrence AUs", "bp4d_gt_occ_distribution.pdf", FIGURES_DIR)
    plot_au_histograms(df_bp4d, bp4d_int_cols, "BP4D Ground Truth Intensity AUs (normalised)", "bp4d_gt_int_distribution.pdf", FIGURES_DIR, normalize=True)
    return


@app.cell
def ffhq_libreface_loading(LIBREFACE_AU_COLUMN_MAP, LIBREFACE_FFHQ_PATH, pd):
    col_rename_ffhq = {v: k for k, v in LIBREFACE_AU_COLUMN_MAP.items()}
    df_ffhq = pd.read_parquet(LIBREFACE_FFHQ_PATH)[list(LIBREFACE_AU_COLUMN_MAP.values())].rename(columns=col_rename_ffhq)
    return (df_ffhq,)


@app.cell
def ffhq_distribution(
    FIGURES_DIR,
    au_intensity_cols,
    au_occurrence_cols,
    df_ffhq,
    plot_au_histograms,
):
    plot_au_histograms(df_ffhq, au_occurrence_cols, "FFHQ LibreFace Binary AUs", "ffhq_libreface_occ_distribution.pdf", FIGURES_DIR)
    plot_au_histograms(df_ffhq, au_intensity_cols, "FFHQ LibreFace Intensity AUs (normalised)", "ffhq_libreface_int_distribution.pdf", FIGURES_DIR, normalize=True)
    return


@app.cell
def pyfeat_bp4d_loading(PYFEAT_AU_COLUMNS, PYFEAT_BP4D_DIR, pd):
    _tasks = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    _dfs = []
    for _task in _tasks:
        _df = pd.read_parquet(PYFEAT_BP4D_DIR / f"bp4d_pyfeat_{_task}.parquet")
        _df["task"] = _task
        _dfs.append(_df)
    df_pyfeat = pd.concat(_dfs, ignore_index=True)[["subject", "task", "au_frame", *PYFEAT_AU_COLUMNS]]
    return (df_pyfeat,)


@app.cell
def pyfeat_vs_gt_comparison(
    BP4D_AU_INTENSITY_COLUMNS,
    BP4D_AU_OCCURRENCE_COLUMNS,
    FIGURES_DIR,
    PYFEAT_AU_COLUMNS,
    df_bp4d,
    df_pyfeat,
    plt,
    sns,
):
    # AUs present in both GT and py-feat
    gt_all = set(BP4D_AU_OCCURRENCE_COLUMNS) | set(BP4D_AU_INTENSITY_COLUMNS)
    shared_aus = sorted(gt_all & set(PYFEAT_AU_COLUMNS))

    # Restrict py-feat to only the coded frames present in the GT index
    coded_frames = df_bp4d[["subject", "task", "frame"]].rename(columns={"frame": "au_frame"})
    df_pyfeat_coded = df_pyfeat.merge(coded_frames, on=["subject", "task", "au_frame"], how="inner")

    ncols = 2
    nrows = len(shared_aus)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, nrows * 2.2))

    for i, au in enumerate(shared_aus):
        # GT column: intensity AUs use the _int column, occurrence AUs use plain name
        gt_col = f"{au}_int" if au in BP4D_AU_INTENSITY_COLUMNS else au
        gt_vals = df_bp4d[gt_col].dropna()
        if au in BP4D_AU_INTENSITY_COLUMNS:
            gt_vals = gt_vals / 5.0  # normalise to [0, 1]

        pyfeat_vals = df_pyfeat_coded[au].dropna()

        sns.histplot(gt_vals, ax=axes[i, 0], bins=20, color="steelblue")
        axes[i, 0].set_title(f"{au} — GT")
        axes[i, 0].set_xlabel("")
        axes[i, 0].set_ylabel("")

        sns.histplot(pyfeat_vals, ax=axes[i, 1], bins=20, color="darkorange")
        axes[i, 1].set_title(f"{au} — py-feat")
        axes[i, 1].set_xlabel("")
        axes[i, 1].set_ylabel("")

    fig.suptitle("BP4D: Ground Truth vs py-feat AU distributions", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bp4d_gt_vs_pyfeat.pdf")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
