"""Generate LaTeX results tables from aggregated evaluation metrics.

Usage::

    uv run src/scripts/make_results_table.py
    uv run src/scripts/make_results_table.py --results results.parquet
"""

import argparse
import re

import pandas as pd

from msc.constants import TABLES_DIR


def main() -> None:
    """Generate LaTeX results tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.results)
    df = _parse_run_names(df)
    df = df[df["stage"] != "finetune"]

    emotion = df[df["eval_mode"] == "emotion"]
    averaged = emotion[emotion["emotion"].isna()]
    per_config = emotion[emotion["emotion"].notna()]
    _make_global_best_table(averaged)
    for dataset in ["bp4d", "ffhq"]:
        _make_emotion_table(averaged[averaged["dataset"] == dataset], dataset=dataset)
        _make_emotion_per_config_table(
            per_config[per_config["dataset"] == dataset], dataset=dataset
        )
    _make_paired_table(df[df["eval_mode"] == "paired"])


def _format_with_bold(
    df: pd.DataFrame, metrics: list[tuple[str, bool]], decimals: int = 3
) -> pd.DataFrame:
    """Format metric columns as strings, wrapping the best value per group in bold.

    Groups by all index levels except the first (Detector) and bolds the
    winning value in each group for each metric.

    Returns:
        The formatted DataFrame with bolded best values.
    """
    out = df.copy().astype(object)
    for col, higher_is_better in metrics:
        bold_idxs: set = set()
        group_levels = list(range(1, df.index.nlevels))
        for _, group in df[col].groupby(level=group_levels):
            if len(group) < 2:
                continue
            best = group.idxmax() if higher_is_better else group.idxmin()
            bold_idxs.add(best)
        for idx in df.index:
            val = df.loc[idx, col]
            text = f"{val:.{decimals}f}" if pd.notna(val) else "--"
            out.loc[idx, col] = f"\\textbf{{{text}}}" if idx in bold_idxs else text
    return out


def _strip_table_env(latex: str) -> str:
    """Remove the outer table environment, leaving only the tabular block.

    Returns:
        The stripped LaTeX string with the table environment removed.
    """
    lines = latex.split("\n")
    skip_prefixes = (r"\begin{table}", r"\end{table}", r"\caption", r"\label")
    kept = [
        ln for ln in lines if not any(ln.strip().startswith(p) for p in skip_prefixes)
    ]
    return "\n".join(kept).strip()


def _insert_group_midrules(latex: str) -> str:
    """Insert cmidrule between groups, with width reflecting hierarchy depth.

    The column span narrows at deeper index levels so that outer group boundaries
    are visually heavier than inner ones:
    - level 0 (Detector)  -> cmidrule{2-N}
    - level 1 (Stage)     -> cmidrule{3-N}
    - level 2 (Dataset)   -> cmidrule{4-N}
    - leaf (Baseline)     -> cmidrule{5-N}  (data columns only)
    Column count is inferred from the tabular column format string.

    Returns:
        The modified LaTeX string with group midrules inserted.
    """
    m = re.search(r"\\begin\{tabular\}\{([^}]+)\}", latex)
    ncols = len(m.group(1)) if m else 0

    lines = latex.split("\n")
    result = []
    past_header = False
    is_first_data_row = True

    for line in lines:
        if not past_header:
            result.append(line)
            if line.strip() == r"\midrule":
                past_header = True
            continue

        if not line.strip().endswith("\\\\"):
            result.append(line)
            continue

        parts = line.split("&")
        effective_level: int | None = None
        for i, part in enumerate(parts):
            if r"\multirow" in part:
                effective_level = i
                break
        if effective_level is None:
            effective_level = 0
            for part in parts:
                if part.strip() == "":
                    effective_level += 1
                else:
                    break

        if not is_first_data_row:
            start_col = effective_level + 1
            if start_col <= ncols:
                result.append(f"\\cmidrule{{{start_col}-{ncols}}}")

        result.append(line)
        is_first_data_row = False

    return "\n".join(result)


def _parse_run_names(df: pd.DataFrame) -> pd.DataFrame:
    """Parse run name into detector, stage, dataset, baseline columns.

    Returns:
        The DataFrame with parsed run name columns added.
    """
    pattern = re.compile(
        r"sd21_(?P<detector>pyfeat|libreface)_(?P<stage>\w+?)_(?P<dataset>bp4d|ffhq)_?(?P<baseline>source|zero|paired)?"
    )
    parsed = df["run"].str.extract(pattern)
    return pd.concat([df, parsed], axis=1)


def _make_global_best_table(df: pd.DataFrame) -> None:
    """Produce an averaged table: mean across BP4D and FFHQ."""
    arcface_col = "ArcFace $\\uparrow$"
    active_col = "AU MAE (active) $\\downarrow$"
    inactive_col = "AU MAE (inactive) $\\downarrow$"

    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Detector": r["detector"].capitalize(),
                "Stage": r["stage"].capitalize(),
                "Baseline": r["baseline"].capitalize(),
                arcface_col: r["arcface_mean"],
                active_col: r["au_mae"],
                inactive_col: r["inactive_mae"],
            }
        )

    out = (
        pd.DataFrame(rows)
        .groupby(["Detector", "Stage", "Baseline"])
        .mean(numeric_only=True)
        .sort_index()
    )

    metrics = [(arcface_col, True), (active_col, False), (inactive_col, False)]

    latex = _format_with_bold(out, metrics).style.to_latex(
        multirow_align="t",
        hrules=True,
        sparse_index=True,
        column_format="lllccc",
        caption=(
            "Emotion-mode results averaged across BP4D and FFHQ eval datasets. "
            "AU MAE (active) measures how closely generated expressions match the "
            "target \\acp{AU}. AU MAE (inactive) measures leakage into non-target "
            "\\acp{AU}."
        ),
        label="tab:results-emotion-avg",
        position="htb",
    )
    latex = _strip_table_env(_insert_group_midrules(latex))

    path = TABLES_DIR / "results_emotion_avg.tex"
    path.write_text(latex, encoding="utf-8")
    print(latex)


def _make_emotion_table(df: pd.DataFrame, *, dataset: str) -> None:
    """Produce emotion-mode results table for a single eval dataset."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Detector": r["detector"].capitalize(),
                "Stage": r["stage"].capitalize(),
                "Baseline": r["baseline"].capitalize(),
                "ArcFace $\\uparrow$": r["arcface_mean"],
                "AU MAE (active) $\\downarrow$": r["au_mae"],
                "AU MAE (inactive) $\\downarrow$": r["inactive_mae"],
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(["Detector", "Stage", "Baseline"])
        .set_index(["Detector", "Stage", "Baseline"])
    )

    arcface_col = "ArcFace $\\uparrow$"
    active_col = "AU MAE (active) $\\downarrow$"
    inactive_col = "AU MAE (inactive) $\\downarrow$"
    metrics = [(arcface_col, True), (active_col, False), (inactive_col, False)]

    latex = _format_with_bold(out, metrics).style.to_latex(
        multirow_align="t",
        hrules=True,
        sparse_index=True,
        column_format="lllccc",
        caption=(
            f"Emotion-mode quantitative results on {dataset.upper()} averaged across "
            "smile, surprise, anger, and sadness. "
            "AU MAE (active) measures how closely the generated expressions match the "
            "target \\acp{AU}. AU MAE (inactive) measures leakage into non-target "
            "\\acp{AU}."
        ),
        label=f"tab:results-emotion-{dataset}",
        position="htb",
    )
    latex = _insert_group_midrules(latex)

    path = TABLES_DIR / f"results_emotion_{dataset}.tex"
    path.write_text(latex, encoding="utf-8")
    print(latex)


def _make_emotion_per_config_table(df: pd.DataFrame, *, dataset: str) -> None:
    """Produce per-emotion breakdown table for the appendix for a single dataset."""
    emotion_order = ["smile", "surprise", "anger", "sadness"]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Detector": r["detector"].capitalize(),
                "Stage": r["stage"].capitalize(),
                "Baseline": r["baseline"].capitalize(),
                "Emotion": str(r["emotion"]).capitalize(),
                "ArcFace $\\uparrow$": r["arcface_mean"],
                "AU MAE (active) $\\downarrow$": r["au_mae"],
                "AU MAE (inactive) $\\downarrow$": r["inactive_mae"],
            }
        )

    out = (
        pd.DataFrame(rows)
        .assign(
            Emotion=lambda d: pd.Categorical(
                d["Emotion"], [e.capitalize() for e in emotion_order]
            )
        )
        .sort_values(["Detector", "Stage", "Baseline", "Emotion"])
        .set_index(["Detector", "Stage", "Baseline", "Emotion"])
    )

    arcface_col = "ArcFace $\\uparrow$"
    active_col = "AU MAE (active) $\\downarrow$"
    inactive_col = "AU MAE (inactive) $\\downarrow$"
    metrics = [(arcface_col, True), (active_col, False), (inactive_col, False)]

    latex = _format_with_bold(out, metrics).style.to_latex(
        multirow_align="t",
        hrules=True,
        sparse_index=True,
        column_format="llllccc",
        caption=(
            f"Per-emotion quantitative results on {dataset.upper()}. "
            "AU MAE (active) measures how closely the generated expressions match "
            "the target \\acp{AU}. "
            "AU MAE (inactive) measures leakage into non-target \\acp{AU}."
        ),
        label=f"tab:results-emotion-per-config-{dataset}",
        position="htb",
    )
    latex = _insert_group_midrules(latex)

    path = TABLES_DIR / f"results_emotion_per_config_{dataset}.tex"
    path.write_text(latex, encoding="utf-8")
    print(latex)


def _make_paired_table(df: pd.DataFrame) -> None:
    """Produce paired-mode results table."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Detector": r["detector"].capitalize(),
                "Stage": r["stage"].capitalize(),
                "ArcFace $\\uparrow$": r["arcface_mean"],
                "AU MAE $\\downarrow$": r["au_mae"],
                "$d(\\hat{z}, z_\\mathrm{tgt}) / d(z_\\mathrm{src}, z_\\mathrm{tgt})$ "
                "\\downarrow$": r["latent_ratio_mean"],
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(["Detector", "Stage"])
        .set_index(["Detector", "Stage"])
    )

    arcface_col = "ArcFace $\\uparrow$"
    mae_col = "AU MAE $\\downarrow$"
    ratio_col = (
        "$d(\\hat{z}, z_\\mathrm{tgt}) / d(z_\\mathrm{src}, z_\\mathrm{tgt})$ "
        "$\\downarrow$"
    )
    metrics = [(arcface_col, True), (mae_col, False), (ratio_col, False)]

    latex = _format_with_bold(out, metrics).style.to_latex(
        multirow_align="t",
        hrules=True,
        sparse_index=True,
        column_format="llccc",
        caption=(
            "Paired-mode quantitative results. "
            "AU MAE measures accuracy against target \\acp{AU} from BP4D. "
            "The latent ratio measures how close the generated image is to the target "
            "relative to the source; values below 1 indicate the prediction is "
            "closer to the target than the source is."
        ),
        label="tab:results-paired",
        position="htb",
    )
    latex = _strip_table_env(_insert_group_midrules(latex))

    path = TABLES_DIR / "results_paired.tex"
    path.write_text(latex, encoding="utf-8")
    print(latex)


if __name__ == "__main__":
    main()
