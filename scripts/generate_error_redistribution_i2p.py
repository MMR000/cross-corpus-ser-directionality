#!/usr/bin/env python3
"""Create publication-quality error-redistribution figure for IEMOCAP->PODCAST.

Caption-ready summary:
This figure compares Baseline Chunk and Stage1 Domain-Utt using normalized
confusion matrices on IEMOCAP->PODCAST. Panel A reports per-class diagonal
recall deltas (Stage1 - Baseline). Panel B visualizes off-diagonal confusion
redistribution as directed flows: red edges indicate increased confusion mass
in Stage1, blue edges indicate reduced confusion mass, and edge width scales
with absolute change magnitude.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.data.datasets import ID_TO_EMOTION
from scripts.plot_style import apply_shared_style, save_png_pdf


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"
REPORTS = ROOT / "reports"


def _load_norm_confusion(path: Path) -> np.ndarray:
    cm = pd.read_csv(path, header=None).values.astype(float)
    row_sum = np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
    return cm / row_sum


def _top_changed_offdiag(delta: np.ndarray, labels: list[str], k: int = 6) -> list[dict]:
    n = delta.shape[0]
    rows: list[dict] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(delta[i, j])
            rows.append(
                {
                    "true": labels[i],
                    "pred": labels[j],
                    "delta": d,
                    "abs_delta": abs(d),
                }
            )
    rows_sorted = sorted(rows, key=lambda r: r["abs_delta"], reverse=True)
    return rows_sorted[:k]


def _plot_panel_a(ax: plt.Axes, delta_diag: np.ndarray, labels: list[str]) -> None:
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in delta_diag]
    x = np.arange(len(labels))
    ax.bar(x, delta_diag, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Recall Delta (Stage1 - Baseline)")
    ax.set_title("A. Diagonal Recall Change")
    for xi, yi in zip(x, delta_diag):
        va = "bottom" if yi >= 0 else "top"
        offset = 0.003 if yi >= 0 else -0.003
        ax.text(xi, yi + offset, f"{yi:+.3f}", ha="center", va=va, fontsize=10)


def _plot_panel_b(ax: plt.Axes, delta: np.ndarray, labels: list[str], top_k: int = 6) -> list[dict]:
    n = len(labels)
    G = nx.DiGraph()
    for lbl in labels:
        G.add_node(lbl)

    top_edges = _top_changed_offdiag(delta, labels, k=top_k)
    for row in top_edges:
        if row["abs_delta"] > 0:
            G.add_edge(row["true"], row["pred"], delta=row["delta"], abs_delta=row["abs_delta"])

    # Fixed layout gives stable, cleaner publication placement.
    pos = {
        "happy": np.array([0.0, 1.05]),
        "angry": np.array([1.15, 0.0]),
        "neutral": np.array([0.0, -1.05]),
        "sad": np.array([-1.15, 0.0]),
    }

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="#f8f9fa",
        edgecolors="black",
        linewidths=1.1,
        node_size=1800,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)

    # Edge-label offsets reduce overlap around the right side (angry node).
    label_offset_map = {
        ("happy", "angry"): np.array([0.06, 0.10]),
        ("sad", "angry"): np.array([0.02, -0.08]),
        ("neutral", "angry"): np.array([0.05, -0.12]),
        ("angry", "happy"): np.array([0.05, 0.07]),
        ("angry", "neutral"): np.array([0.04, -0.07]),
        ("happy", "neutral"): np.array([-0.06, -0.02]),
        ("neutral", "happy"): np.array([-0.06, 0.02]),
        ("sad", "happy"): np.array([-0.03, 0.08]),
        ("happy", "sad"): np.array([-0.03, 0.08]),
        ("sad", "neutral"): np.array([-0.03, -0.08]),
        ("neutral", "sad"): np.array([-0.03, -0.08]),
    }

    legend_rows: list[str] = []
    for idx, (u, v, attrs) in enumerate(G.edges(data=True), start=1):
        d = float(attrs["delta"])
        mag = float(attrs["abs_delta"])
        color = "#d62728" if d > 0 else "#1f77b4"
        width = 1.4 + 22.0 * mag
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=color,
            arrowstyle="-|>",
            arrowsize=18,
            connectionstyle="arc3,rad=0.16",
            alpha=0.9,
            ax=ax,
        )

        xm = 0.56 * pos[u][0] + 0.44 * pos[v][0]
        ym = 0.56 * pos[u][1] + 0.44 * pos[v][1]
        off = label_offset_map.get((u, v), np.array([0.0, 0.0]))
        ax.text(
            xm + off[0],
            ym + off[1],
            f"[{idx}]",
            fontsize=10,
            color="black",
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.4},
        )
        legend_rows.append(f"[{idx}] {u}→{v}: {d:+.3f}")

    ax.set_title("B. Off-diagonal Error Redistribution")
    ax.set_axis_off()
    ax.text(
        0.02,
        0.02,
        "\n".join(legend_rows),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#999999", "pad": 0.45},
    )
    return top_edges


def main() -> None:
    apply_shared_style()
    labels = [ID_TO_EMOTION[i] for i in sorted(ID_TO_EMOTION.keys())]

    baseline_path = EXP / "iemocap_to_podcast_chunk" / "confusion_test.csv"
    stage1_path = EXP / "iemocap_to_podcast_chunk_domain_utt" / "confusion_test.csv"
    if not baseline_path.exists() or not stage1_path.exists():
        raise FileNotFoundError(
            "Required confusion matrices are missing:\n"
            f"- {baseline_path}\n"
            f"- {stage1_path}"
        )

    cm_base = _load_norm_confusion(baseline_path)
    cm_s1 = _load_norm_confusion(stage1_path)
    delta = cm_s1 - cm_base
    delta_diag = np.diag(delta)

    pd.DataFrame(cm_base, index=labels, columns=labels).to_csv(
        REPORTS / "plotdata_i2p_confusion_baseline_norm.csv"
    )
    pd.DataFrame(cm_s1, index=labels, columns=labels).to_csv(
        REPORTS / "plotdata_i2p_confusion_stage1_norm.csv"
    )
    pd.DataFrame(delta, index=labels, columns=labels).to_csv(
        REPORTS / "plotdata_i2p_confusion_delta_stage1_minus_baseline.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    _plot_panel_a(axes[0], delta_diag, labels)
    top_edges = _plot_panel_b(axes[1], delta, labels, top_k=6)
    fig.suptitle("IEMOCAP→PODCAST: Error Redistribution (Stage1 - Baseline)", y=1.02)
    save_png_pdf(fig, REPORTS / "fig_error_redistribution_i2p")

    lines = [
        "## Error Redistribution Note (IEMOCAP→PODCAST)",
        "",
        "Strongest off-diagonal confusion redistributions (Stage1 - Baseline):",
    ]
    for r in top_edges:
        direction = "increase" if r["delta"] > 0 else "decrease"
        lines.append(
            f"- {r['true']} → {r['pred']}: {r['delta']:+.3f} ({direction} in confusion mass)"
        )
    lines.append("")
    lines.append(
        "Interpretation tip: red edges indicate where Stage1 confuses more than baseline; "
        "blue edges indicate reduced confusion."
    )
    (REPORTS / "fig_error_redistribution_i2p_note.md").write_text("\n".join(lines), encoding="utf-8")

    print("Generated:")
    print("- reports/fig_error_redistribution_i2p.png")
    print("- reports/fig_error_redistribution_i2p.pdf")
    print("- reports/fig_error_redistribution_i2p_note.md")
    print("- reports/plotdata_i2p_confusion_delta_stage1_minus_baseline.csv")


if __name__ == "__main__":
    main()

