#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scripts.plot_style import apply_shared_style, save_png_pdf


def save_barplot(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = sns.barplot(data=df, x=x, y=y, hue=hue, palette="deep")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Test UAR")
    plt.xticks(rotation=20, ha="right")
    save_png_pdf(plt.gcf(), out_path.with_suffix(""))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    overview = pd.read_csv(reports / "final_baseline_stage1_stage2_overview.csv")

    # Table 1: in-corpus
    in_df = overview[overview["training_setup"] == "in_corpus"].copy()
    in_df["domain"] = in_df["target_domain"]
    in_df = in_df[["experiment", "domain", "model_variant", "test_uar", "test_wa", "test_macro_f1"]]
    in_df = in_df.sort_values(["domain", "model_variant"])
    in_round = in_df.copy()
    in_round[["test_uar", "test_wa", "test_macro_f1"]] = in_round[
        ["test_uar", "test_wa", "test_macro_f1"]
    ].round(3)
    in_round.to_csv(reports / "paper_table_in_corpus.csv", index=False)

    # Table 2: cross-corpus baselines
    cross_base = overview[
        (overview["family"] == "baseline")
        & (overview["training_setup"].isin(["cross_single_source", "cross_multi_source"]))
    ].copy()
    cross_base = cross_base[
        [
            "experiment",
            "training_setup",
            "source_domain",
            "target_domain",
            "model_variant",
            "test_uar",
            "test_wa",
            "test_macro_f1",
        ]
    ]
    cross_base = cross_base.sort_values(["source_domain", "target_domain", "model_variant"])
    cross_round = cross_base.copy()
    cross_round[["test_uar", "test_wa", "test_macro_f1"]] = cross_round[
        ["test_uar", "test_wa", "test_macro_f1"]
    ].round(3)
    cross_round.to_csv(reports / "paper_table_cross_corpus.csv", index=False)

    # Table 3: stage1/stage2 chunk-focused
    focus = overview[
        overview["model_variant"].isin(
            ["chunk", "chunk_domain_utt", "chunk_domain_both", "chunk_domain_both_weakchunk"]
        )
    ].copy()
    focus = focus[focus["training_setup"].str.contains("cross_")]
    focus = focus[
        [
            "experiment",
            "family",
            "training_setup",
            "source_domain",
            "target_domain",
            "model_variant",
            "test_uar",
            "test_wa",
            "test_macro_f1",
            "gain_vs_plain_chunk",
            "gain_vs_stage1_chunk_domain_utt",
        ]
    ]
    focus = focus.sort_values(["source_domain", "target_domain", "family", "model_variant"])
    focus_round = focus.copy()
    focus_round[
        [
            "test_uar",
            "test_wa",
            "test_macro_f1",
            "gain_vs_plain_chunk",
            "gain_vs_stage1_chunk_domain_utt",
        ]
    ] = focus_round[
        [
            "test_uar",
            "test_wa",
            "test_macro_f1",
            "gain_vs_plain_chunk",
            "gain_vs_stage1_chunk_domain_utt",
        ]
    ].round(3)
    focus_round.to_csv(reports / "paper_table_stage1_stage2.csv", index=False)

    # Figures
    apply_shared_style()

    in_plot = overview[overview["training_setup"] == "in_corpus"].copy()
    in_plot["group"] = in_plot["target_domain"]
    save_barplot(
        in_plot,
        x="group",
        y="test_uar",
        hue="model_variant",
        title="In-Corpus Performance (Test UAR)",
        out_path=reports / "fig_in_corpus_barplot.png",
    )

    cb_plot = overview[
        (overview["family"] == "baseline")
        & (overview["training_setup"].isin(["cross_single_source", "cross_multi_source"]))
    ].copy()
    cb_plot["group"] = cb_plot["source_domain"] + " -> " + cb_plot["target_domain"]
    save_barplot(
        cb_plot,
        x="group",
        y="test_uar",
        hue="model_variant",
        title="Cross-Corpus Baseline Performance (Test UAR)",
        out_path=reports / "fig_cross_corpus_barplot.png",
    )

    s_plot = overview[
        overview["model_variant"].isin(
            ["chunk", "chunk_domain_utt", "chunk_domain_both", "chunk_domain_both_weakchunk"]
        )
    ].copy()
    s_plot = s_plot[s_plot["training_setup"].str.contains("cross_")]
    s_plot["group"] = s_plot["source_domain"] + " -> " + s_plot["target_domain"]
    save_barplot(
        s_plot,
        x="group",
        y="test_uar",
        hue="model_variant",
        title="Chunk Baseline vs Stage1/Stage2 Variants (Test UAR)",
        out_path=reports / "fig_stage1_stage2_barplot.png",
    )

    # Markdown summary with highlighted best values
    in_best = in_round.loc[in_round.groupby("domain")["test_uar"].idxmax()]
    cb_tmp = cross_round.copy()
    cb_tmp["direction"] = cb_tmp["source_domain"] + " -> " + cb_tmp["target_domain"]
    cb_best = cb_tmp.loc[cb_tmp.groupby("direction")["test_uar"].idxmax()][
        ["direction", "experiment", "test_uar"]
    ]

    ss = overview[
        overview["model_variant"].isin(
            ["chunk", "chunk_domain_utt", "chunk_domain_both", "chunk_domain_both_weakchunk"]
        )
    ].copy()
    ss = ss[ss["training_setup"].str.contains("cross_")]
    ss["direction"] = ss["source_domain"] + " -> " + ss["target_domain"]
    ss_best = ss.loc[ss.groupby("direction")["test_uar"].idxmax()][
        ["direction", "experiment", "model_variant", "test_uar"]
    ]

    lines: list[str] = []
    lines.append("## Paper Results Summary")
    lines.append("")
    lines.append("### In-Corpus")
    for _, row in in_best.iterrows():
        lines.append(
            f"- {row['domain']}: **{row['experiment']} ({row['test_uar']:.3f})** is best test UAR."
        )
    lines.append(
        "- Interpretation: in-corpus performance is strongest overall; IEMOCAP favors attention pooling while podcast is tightly clustered."
    )
    lines.append("")
    lines.append("### Cross-Corpus Baseline")
    for _, row in cb_best.iterrows():
        lines.append(
            f"- {row['direction']}: **{row['experiment']} ({row['test_uar']:.3f})** is best baseline."
        )
    lines.append(
        "- Interpretation: cross-corpus transfer is asymmetric; natural -> acted remains easier than acted -> natural."
    )
    lines.append("")
    lines.append("### Stage 1 / Stage 2")
    for _, row in ss_best.iterrows():
        lines.append(
            f"- {row['direction']}: best chunk-family variant is **{row['experiment']} ({row['test_uar']:.3f})**."
        )
    lines.append(
        "- Interpretation: Stage 1 helps selectively, while Stage 2 default can over-regularize single-source transfer; multi-source acted -> natural shows modest Stage 2 gains."
    )
    lines.append("")
    lines.append("### Assets")
    lines.append("- `reports/paper_table_in_corpus.csv`")
    lines.append("- `reports/paper_table_cross_corpus.csv`")
    lines.append("- `reports/paper_table_stage1_stage2.csv`")
    lines.append("- `reports/fig_in_corpus_barplot.png`")
    lines.append("- `reports/fig_cross_corpus_barplot.png`")
    lines.append("- `reports/fig_stage1_stage2_barplot.png`")

    (reports / "paper_results_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
