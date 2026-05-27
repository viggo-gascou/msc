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
    from msc.constants import FFHQ_DATA_DIR, FFHQ_IMAGES_DIR, FFHQ_INDEX_PATH, FFHQ_CAPTIONS_PATH, FFHQ_EMBEDDINGS_PATH, FFHQ_PREPROCESSED_PATH, LIBREFACE_FFHQ_PATH, FIGURES_DIR, TABLES_DIR
    set_plotting_style()
    return (
        FFHQ_DATA_DIR,
        FFHQ_INDEX_PATH,
        FIGURES_DIR,
        TABLES_DIR,
        pd,
        plt,
        sns,
    )


@app.cell
def data_loading(FFHQ_DATA_DIR, FFHQ_INDEX_PATH, pd):
    df_raw_index = pd.read_csv(FFHQ_DATA_DIR / "ffhq_aging_labels.csv")
    df_index = pd.read_parquet(FFHQ_INDEX_PATH)
    age_group_order = sorted(df_index["age_group"].unique(), key= lambda x: int(x.split("-")[0]))
    df_index["age_group"] = pd.Categorical(df_index["age_group"], age_group_order)
    raw_age_group_order = sorted(df_raw_index["age_group"].unique(), key= lambda x: int(x.split("-")[0]))
    df_raw_index["age_group"] = pd.Categorical(df_raw_index["age_group"], raw_age_group_order)
    df_raw_index
    return age_group_order, df_index, df_raw_index, raw_age_group_order


@app.cell
def _(FIGURES_DIR, df_raw_index, plt, sns):
    def plot_age_group_dist(df, fname):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x="age_group", discrete=True, alpha=1, shrink=.8)
        ax.set_xlabel("Age Group")
        ax.set_ylim(0, 20_000)
        ax.tick_params(axis="x")
        plt.savefig(FIGURES_DIR / fname)
        plt.show()

    plot_age_group_dist(df_raw_index, "raw_age_group_distribution.pdf")
    return (plot_age_group_dist,)


@app.cell
def _(df_index, plot_age_group_dist):
    plot_age_group_dist(df_index, "age_group_distribution.pdf")
    return


@app.cell
def _(
    TABLES_DIR,
    age_group_order,
    df_index,
    df_raw_index,
    pd,
    raw_age_group_order,
):
    all_groups = sorted(set(raw_age_group_order) | set(age_group_order), key=lambda x: int(x.split("-")[0]))
    raw_counts = df_raw_index["age_group"].value_counts()
    filtered_counts = df_index["age_group"].value_counts()
    df_comparison = pd.DataFrame({
        "Age Group": all_groups,
        "Count (Before)": [raw_counts.get(g) for g in all_groups],
        "Count (After)": [filtered_counts.get(g) for g in all_groups],
    })
    df_comparison["% (Before)"] = (df_comparison["Count (Before)"] / len(df_raw_index)) * 100
    df_comparison["% (After)"] = (df_comparison["Count (After)"] / len(df_index)) * 100
    df_comparison.to_latex(
        TABLES_DIR / "age_group_comparison.tex",
        index=False,
        column_format="ccccc",
        formatters={
            "Count (Before)": "{:,.0f}".format,
            "Count (After)": "{:,.0f}".format,
            "% (Before)": r"{:.2f}\%".format,
            "% (After)": r"{:.2f}\%".format,
        },
        escape=True,
        na_rep="--",
    )
    df_comparison
    return


if __name__ == "__main__":
    app.run()
