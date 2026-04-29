#!/usr/bin/env python3
"""Generate stage-contribution waterfall figure from existing overview results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_style import FIGSIZE, PUB_LABEL, apply_shared_style, save_png_pdf


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _get_uar(overview: pd.DataFrame, exp: str) -> float:
    row = overview.loc[overview["experiment"] == exp, "test_uar"]
    if row.empty:
        raise ValueError(f"Missing experiment in overview: {exp}")
    return float(row.iloc[0])


def _waterfall_one(
    ax: plt.Axes,
    title: str,
    baseline: float,
    stage1: float,
    stage2: float,
    y_min: float,
    y_max: float,
) -> dict[str, float]:
    d1 = stage1 - baseline
    d2 = stage2 - stage1

    labels = ["Base", "ΔS1", "ΔS2", "Final"]
    x = np.arange(4)
    pos_color = "#2ca02c"
    neg_color = "#d62728"
    base_color = "#95a5a6"
    final_color = "#34495e"

    # Base bar
    ax.bar(x[0], baseline, width=0.64, color=base_color, edgecolor="black", linewidth=0.7, zorder=3)

    # Delta stage1 floating bar
    s1_bottom = baseline if d1 >= 0 else stage1
    ax.bar(
        x[1],
        abs(d1),
        bottom=s1_bottom,
        width=0.64,
        color=pos_color if d1 >= 0 else neg_color,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

    # Delta stage2 floating bar
    s2_bottom = stage1 if d2 >= 0 else stage2
    ax.bar(
        x[2],
        abs(d2),
        bottom=s2_bottom,
        width=0.64,
        color=pos_color if d2 >= 0 else neg_color,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

    # Final bar
    ax.bar(x[3], stage2, width=0.64, color=final_color, edgecolor="black", linewidth=0.7, zorder=3)

    # Connectors
    ax.plot([x[0] + 0.32, x[1] - 0.32], [baseline, baseline], color="black", linewidth=1.0, alpha=0.8, zorder=4)
    ax.plot([x[1] + 0.32, x[2] - 0.32], [stage1, stage1], color="black", linewidth=1.0, alpha=0.8, zorder=4)
    ax.plot([x[2] + 0.32, x[3] - 0.32], [stage2, stage2], color="black", linewidth=1.0, alpha=0.8, zorder=4)

    # Labels
    y_span = y_max - y_min
    up = 0.012 * y_span

    bbox_kw = {"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 0.2}
    ax.text(
        x[0],
        baseline + up,
        f"{baseline:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=bbox_kw,
    )
    ax.text(
        x[1],
        max(s1_bottom + 0.5 * abs(d1), y_min + 0.10 * y_span),
        f"{d1:+.3f}",
        ha="center",
        va="center",
        fontsize=10,
        color=pos_color if d1 >= 0 else neg_color,
        fontweight="bold",
        clip_on=True,
        bbox=bbox_kw,
    )
    ax.text(
        x[2],
        max(s2_bottom + 0.5 * abs(d2), y_min + 0.10 * y_span),
        f"{d2:+.3f}",
        ha="center",
        va="center",
        fontsize=10,
        color=pos_color if d2 >= 0 else neg_color,
        fontweight="bold",
        clip_on=True,
        bbox=bbox_kw,
    )
    ax.text(
        x[3],
        stage2 + up,
        f"{stage2:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox=bbox_kw,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_ylabel("Test UAR")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # In-panel key to avoid global legend overlap.
    ax.text(
        0.02,
        0.98,
        "Gray=Base, Green=+Δ, Red=-Δ, Blue=Final",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 0.15},
    )

    return {"baseline": baseline, "stage1": stage1, "stage2": stage2, "delta_s1": d1, "delta_s2": d2}


def main() -> None:
    apply_shared_style()
    overview_path = REPORTS / "final_baseline_stage1_stage2_overview.csv"
    if not overview_path.exists():
        raise FileNotFoundError(f"Missing file: {overview_path}")
    overview = pd.read_csv(overview_path)

    specs = [
        (
            PUB_LABEL["iemocap_to_podcast"],
            "iemocap_to_podcast_chunk",
            "iemocap_to_podcast_chunk_domain_utt",
            "iemocap_to_podcast_chunk_domain_both",
        ),
        (
            PUB_LABEL["podcast_to_iemocap"],
            "podcast_to_iemocap_chunk",
            "podcast_to_iemocap_chunk_domain_utt",
            "podcast_to_iemocap_chunk_domain_both",
        ),
        (
            PUB_LABEL["iemocap_plus_cremad_to_podcast"],
            "iemocap_plus_cremad_to_podcast_chunk",
            "iemocap_plus_cremad_to_podcast_chunk_domain_utt",
            "iemocap_plus_cremad_to_podcast_chunk_domain_both",
        ),
    ]

    vals = []
    triplets = []
    for name, e0, e1, e2 in specs:
        b = _get_uar(overview, e0)
        s1 = _get_uar(overview, e1)
        s2 = _get_uar(overview, e2)
        triplets.append((name, b, s1, s2))
        vals.extend([b, s1, s2])
    g_min = min(vals) - 0.012
    g_max = max(vals) + 0.016

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9), constrained_layout=True, sharey=False)
    rows = []
    for ax, (name, b, s1, s2) in zip(axes, triplets):
        out = _waterfall_one(ax, name, b, s1, s2, g_min, g_max)
        rows.append({"direction": name, **out})

    # Intentionally no global suptitle/legend to prevent top overlap.
    save_png_pdf(fig, REPORTS / "fig_stage_contribution_waterfall")

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS / "plotdata_stage_contribution_waterfall.csv", index=False)
    print("\nStage deltas summary (UAR):")
    print(summary[["direction", "baseline", "stage1", "stage2", "delta_s1", "delta_s2"]].to_string(index=False))


if __name__ == "__main__":
    main()

