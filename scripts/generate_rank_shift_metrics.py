#!/usr/bin/env python3
"""Generate rank-shift (bump) chart across UAR, Macro-F1, and robustness."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_style import apply_shared_style, save_png_pdf


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _short_label(exp: str) -> str:
    mapping = {
        "iemocap_to_podcast_chunk": "i2p-chunk",
        "iemocap_to_podcast_chunk_domain_utt": "i2p-s1",
        "podcast_to_iemocap_chunk": "p2i-chunk",
        "podcast_to_iemocap_chunk_domain_utt": "p2i-s1",
    }
    return mapping.get(exp, exp)


def _family(exp: str) -> str:
    if exp.startswith("iemocap_to_podcast"):
        return "IEMOCAP→PODCAST"
    if exp.startswith("podcast_to_iemocap"):
        return "PODCAST→IEMOCAP"
    return "Other"


def main() -> None:
    apply_shared_style()
    ms_path = REPORTS / "multiseed_core_results.csv"
    if not ms_path.exists():
        raise FileNotFoundError(f"Missing required CSV: {ms_path}")
    df = pd.read_csv(ms_path).copy()

    needed = {"experiment", "mean_test_uar", "mean_test_macro_f1", "std_test_uar"}
    miss = sorted(needed - set(df.columns))
    if miss:
        raise ValueError(f"Missing columns in multiseed CSV: {miss}")

    df["label"] = df["experiment"].map(_short_label)
    df["family"] = df["experiment"].map(_family)

    # Rank convention: smaller rank number is better.
    df["rank_uar"] = df["mean_test_uar"].rank(ascending=False, method="min").astype(int)
    df["rank_macro_f1"] = df["mean_test_macro_f1"].rank(ascending=False, method="min").astype(int)
    df["rank_robustness"] = df["std_test_uar"].rank(ascending=True, method="min").astype(int)

    long_rows = []
    metric_order = ["UAR", "Macro-F1", "Robustness"]
    for _, r in df.iterrows():
        long_rows.extend(
            [
                {"label": r["label"], "family": r["family"], "metric": "UAR", "rank": int(r["rank_uar"])},
                {"label": r["label"], "family": r["family"], "metric": "Macro-F1", "rank": int(r["rank_macro_f1"])},
                {"label": r["label"], "family": r["family"], "metric": "Robustness", "rank": int(r["rank_robustness"])},
            ]
        )
    d = pd.DataFrame(long_rows)
    d["metric"] = pd.Categorical(d["metric"], categories=metric_order, ordered=True)
    d = d.sort_values(["label", "metric"]).reset_index(drop=True)
    d.to_csv(REPORTS / "plotdata_rank_shift_metrics.csv", index=False)

    color_map = {"IEMOCAP→PODCAST": "#1f77b4", "PODCAST→IEMOCAP": "#d62728", "Other": "#7f7f7f"}
    x_map = {m: i for i, m in enumerate(metric_order)}

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    for label in sorted(d["label"].unique()):
        dd = d[d["label"] == label].sort_values("metric")
        xs = np.array([x_map[m] for m in dd["metric"]], dtype=float)
        ys = dd["rank"].values.astype(float)
        fam = dd["family"].iloc[0]
        c = color_map.get(fam, "#7f7f7f")
        is_top_any = bool((ys == 1).any())
        lw = 3.0 if is_top_any else 1.8
        ms = 9 if is_top_any else 6
        alpha = 0.95 if is_top_any else 0.75
        z = 4 if is_top_any else 2
        ax.plot(xs, ys, marker="o", linewidth=lw, markersize=ms, color=c, alpha=alpha, zorder=z)

        # annotate left and right endpoints
        ax.text(
            xs[0] - 0.08,
            ys[0],
            label,
            ha="right",
            va="center",
            fontsize=10,
            color=c,
            fontweight="bold" if is_top_any else "normal",
        )
        ax.text(
            xs[-1] + 0.08,
            ys[-1],
            label,
            ha="left",
            va="center",
            fontsize=10,
            color=c,
            fontweight="bold" if is_top_any else "normal",
        )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(metric_order, rotation=0)
    ax.set_xlim(-0.35, 2.35)
    max_rank = int(d["rank"].max())
    ax.set_yticks(np.arange(1, max_rank + 1))
    ax.set_ylim(max_rank + 0.4, 0.6)  # rank 1 at top
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Rank Shift Across Metrics")
    ax.grid(axis="y", alpha=0.22)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=color_map["IEMOCAP→PODCAST"], lw=2.5, label="IEMOCAP→PODCAST"),
        Line2D([0], [0], color=color_map["PODCAST→IEMOCAP"], lw=2.5, label="PODCAST→IEMOCAP"),
    ]
    ax.legend(handles=legend_handles, frameon=True, title="", loc="lower center")

    save_png_pdf(fig, REPORTS / "fig_rank_shift_metrics")

    rank_table = df[
        [
            "label",
            "mean_test_uar",
            "rank_uar",
            "mean_test_macro_f1",
            "rank_macro_f1",
            "std_test_uar",
            "rank_robustness",
        ]
    ].sort_values("rank_uar")
    rank_table.to_csv(REPORTS / "plotdata_rank_shift_metrics_table.csv", index=False)
    print("Generated:")
    print("- reports/fig_rank_shift_metrics.png")
    print("- reports/fig_rank_shift_metrics.pdf")
    print("- reports/plotdata_rank_shift_metrics.csv")
    print("- reports/plotdata_rank_shift_metrics_table.csv")


if __name__ == "__main__":
    main()

