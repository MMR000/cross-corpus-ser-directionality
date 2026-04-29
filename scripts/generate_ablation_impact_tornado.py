#!/usr/bin/env python3
"""Generate ranked ablation impact tornado/lollipop chart."""

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


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src = pd.read_csv(REPORTS / "source_fraction_ablation.csv")
    tgt = pd.read_csv(REPORTS / "uda_target_fraction_ablation.csv")
    s2 = pd.read_csv(REPORTS / "stage2_ablation_iemocap_to_podcast.csv")
    return src, tgt, s2


def _build_source_family(src: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    ref_chunk = float(src[(src["model_variant"] == "chunk") & (src["source_fraction"] == 1.0)]["test_uar"].iloc[0])
    ref_s1 = float(
        src[(src["model_variant"] == "chunk_domain_utt") & (src["source_fraction"] == 1.0)]["test_uar"].iloc[0]
    )
    for _, r in src.iterrows():
        mv = str(r["model_variant"])
        frac = float(r["source_fraction"])
        if mv == "chunk":
            delta = float(r["test_uar"] - ref_chunk)
            label = f"SRC chunk f={frac:.2f}"
            ref = "chunk@1.00"
        else:
            delta = float(r["test_uar"] - ref_s1)
            label = f"SRC s1 f={frac:.2f}"
            ref = "chunk_domain_utt@1.00"
        rows.append(
            {
                "family": "Source Fraction",
                "item_label": label,
                "delta_uar": delta,
                "reference": ref,
            }
        )
    return rows


def _build_target_family(tgt: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, r in tgt.iterrows():
        frac = float(r["target_fraction"])
        delta = float(r["gain_vs_plain_chunk"])
        rows.append(
            {
                "family": "Target Fraction",
                "item_label": f"TGT s1 f={frac:.2f}",
                "delta_uar": delta,
                "reference": "plain_chunk",
            }
        )
    return rows


def _build_stage2_family(s2: pd.DataFrame) -> list[dict]:
    keep = {
        "chunk_domain_both_default": "S2 default",
        "chunk_only_w01": "S2 chunk w0.1",
        "chunk_only_w03": "S2 chunk w0.3",
        "chunk_only_w05": "S2 chunk w0.5",
        "both_weakchunk": "S2 both w0.1",
        "both_midchunk": "S2 both w0.3",
    }
    rows: list[dict] = []
    for _, r in s2.iterrows():
        v = str(r["ablation_variant"])
        if v not in keep:
            continue
        rows.append(
            {
                "family": "Stage2 Ablation",
                "item_label": keep[v],
                "delta_uar": float(r["absolute_gain_vs_chunk_domain_utt"]),
                "reference": "stage1_chunk_domain_utt",
            }
        )
    return rows


def main() -> None:
    apply_shared_style()
    src, tgt, s2 = _load_inputs()
    rows = _build_source_family(src) + _build_target_family(tgt) + _build_stage2_family(s2)
    d = pd.DataFrame(rows)
    d = d.sort_values("delta_uar", ascending=True).reset_index(drop=True)
    d.to_csv(REPORTS / "plotdata_ablation_impact_tornado.csv", index=False)

    family_colors = {
        "Source Fraction": "#1f77b4",
        "Target Fraction": "#2ca02c",
        "Stage2 Ablation": "#9467bd",
    }
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(10.8, 7.4), constrained_layout=True)
    ax.axvline(0.0, color="black", linewidth=1.0, zorder=1)
    for i, r in d.iterrows():
        c = family_colors[r["family"]]
        x = float(r["delta_uar"])
        ax.hlines(y=i, xmin=0, xmax=x, color=c, linewidth=2.2, alpha=0.9, zorder=2)
        ax.plot(x, i, "o", color=c, markersize=7.5, zorder=3, markeredgecolor="black", markeredgewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(d["item_label"].tolist(), rotation=0)
    ax.set_xlabel("ΔUAR (relative to family reference)")
    ax.set_title("Ablation Impact Tornado (Ranked)")
    ax.grid(axis="x", alpha=0.25, zorder=0)

    # Annotate strongest positive/negative impacts
    i_min = int(d["delta_uar"].idxmin())
    i_max = int(d["delta_uar"].idxmax())
    for idx, tag in [(i_min, "Most Negative"), (i_max, "Most Positive")]:
        x = float(d.loc[idx, "delta_uar"])
        yy = int(idx)
        ax.annotate(
            f"{tag}: {x:+.3f}",
            xy=(x, yy),
            xytext=(8 if x >= 0 else -8, -10),
            textcoords="offset points",
            ha="left" if x >= 0 else "right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 0.25},
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
        )

    # Family separators and right-side family tags
    fam_counts = d["family"].value_counts()
    running = 0
    for fam in d["family"].drop_duplicates():
        n = int(fam_counts[fam])
        center = running + (n - 1) / 2
        ax.text(
            1.01,
            center / max(len(d) - 1, 1),
            fam,
            transform=ax.transAxes,
            rotation=0,
            ha="left",
            va="center",
            fontsize=10,
            color=family_colors[fam],
            fontweight="bold",
        )
        running += n

    save_png_pdf(fig, REPORTS / "fig_ablation_impact_tornado")

    print("Generated:")
    print("- reports/fig_ablation_impact_tornado.png")
    print("- reports/fig_ablation_impact_tornado.pdf")
    print("- reports/plotdata_ablation_impact_tornado.csv")
    print("")
    print("Family references:")
    print("- Source Fraction: chunk@1.00 (for chunk) and chunk_domain_utt@1.00 (for s1 rows)")
    print("- Target Fraction: plain_chunk")
    print("- Stage2 Ablation: stage1_chunk_domain_utt")


if __name__ == "__main__":
    main()

