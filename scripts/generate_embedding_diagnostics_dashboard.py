#!/usr/bin/env python3
"""Generate embedding diagnostics dashboard from 2D plot coordinates only.

Important interpretation note:
All diagnostics in this figure are computed on 2D projected coordinates
(t-SNE/UMAP-like plotted points), so they are visualization-space proxies.
They should not be interpreted as exact properties of the original
high-dimensional representation space.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.plot_style import apply_shared_style, save_png_pdf


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _resolve_input(primary: str, fallback: Path) -> Path:
    p = Path(primary)
    if p.exists():
        return p
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Input not found:\n- {primary}\n- {fallback}")


def _draw_domain_ellipse(ax: plt.Axes, pts: np.ndarray, color: str, label: str) -> None:
    if pts.shape[0] < 3:
        return
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # ~95% contour for 2D Gaussian: sqrt(chi2.ppf(0.95, df=2)) ~= 2.4477
    scale = 2.4477
    width, height = 2 * scale * np.sqrt(np.clip(vals, 1e-12, None))
    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=2.0,
        linestyle="--",
        label=label,
        alpha=0.95,
    )
    ax.add_patch(ell)


def _point_in_ellipse(xy: np.ndarray, mean: np.ndarray, cov: np.ndarray, chi2_thr: float = 5.991) -> np.ndarray:
    inv_cov = np.linalg.pinv(cov)
    centered = xy - mean
    q = np.einsum("ni,ij,nj->n", centered, inv_cov, centered)
    return q <= chi2_thr


def _compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    xy = df[["x", "y"]].values
    doms = sorted(df["domain"].unique().tolist())
    emos = sorted(df["emotion"].unique().tolist())
    if len(doms) != 2:
        raise ValueError(f"Expected exactly 2 domains, got {doms}")

    d0, d1 = doms
    p0 = df[df["domain"] == d0][["x", "y"]].values
    p1 = df[df["domain"] == d1][["x", "y"]].values
    c0 = p0.mean(axis=0)
    c1 = p1.mean(axis=0)

    centroid_dist = float(np.linalg.norm(c0 - c1))

    cov0 = np.cov(p0.T)
    cov1 = np.cov(p1.T)
    in0 = _point_in_ellipse(xy, c0, cov0)
    in1 = _point_in_ellipse(xy, c1, cov1)
    overlap_proxy = float(np.mean(in0 & in1))

    dom_centroids = {d0: c0, d1: c1}
    own_d = np.array(
        [np.linalg.norm(xy[i] - dom_centroids[df.iloc[i]["domain"]]) for i in range(len(df))],
        dtype=float,
    )
    other_d = np.array(
        [
            np.linalg.norm(xy[i] - dom_centroids[d1 if df.iloc[i]["domain"] == d0 else d0])
            for i in range(len(df))
        ],
        dtype=float,
    )
    domain_sep_proxy = float(np.mean(other_d > own_d))

    emo_centroids = {e: df[df["emotion"] == e][["x", "y"]].values.mean(axis=0) for e in emos}
    within = []
    for e in emos:
        pe = df[df["emotion"] == e][["x", "y"]].values
        ce = emo_centroids[e]
        within.append(float(np.mean(np.linalg.norm(pe - ce, axis=1))))
    within_compactness = float(np.mean(within))

    pair_d = []
    for i in range(len(emos)):
        for j in range(i + 1, len(emos)):
            pair_d.append(float(np.linalg.norm(emo_centroids[emos[i]] - emo_centroids[emos[j]])))
    between_sep = float(np.mean(pair_d))

    emotion_sep_proxy = float(between_sep / max(within_compactness, 1e-12))
    return {
        "source_target_centroid_distance": centroid_dist,
        "source_target_overlap_proxy": overlap_proxy,
        "within_emotion_compactness": within_compactness,
        "between_emotion_centroid_separation": between_sep,
        "domain_separability_proxy": domain_sep_proxy,
        "emotion_separability_proxy": emotion_sep_proxy,
    }


def _centroid_distance_matrix(df: pd.DataFrame, emotions: list[str]) -> np.ndarray:
    centers = {e: df[df["emotion"] == e][["x", "y"]].values.mean(axis=0) for e in emotions}
    m = np.zeros((len(emotions), len(emotions)), dtype=float)
    for i, ei in enumerate(emotions):
        for j, ej in enumerate(emotions):
            m[i, j] = np.linalg.norm(centers[ei] - centers[ej])
    return m


def _panel_embedding(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="emotion",
        style="domain",
        s=12,
        alpha=0.32,
        ax=ax,
        legend=True,
    )
    domain_colors = {"source": "#1f77b4", "target": "#d62728"}
    for d in sorted(df["domain"].unique()):
        pts = df[df["domain"] == d][["x", "y"]].values
        _draw_domain_ellipse(ax, pts, color=domain_colors.get(d, "black"), label=f"{d} 95% ellipse")
    ax.set_title(title)
    ax.set_xlabel("Dim-1 (2D projection)")
    ax.set_ylabel("Dim-2 (2D projection)")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(handles=handles, labels=labels, loc="best", fontsize=9, frameon=True, title="")


def main() -> None:
    apply_shared_style()
    baseline_path = _resolve_input(
        "/mnt/data/plotdata_embedding_chunk_baseline_v2_points.csv",
        REPORTS / "plotdata_embedding_chunk_baseline_v2_points.csv",
    )
    stage1_path = _resolve_input(
        "/mnt/data/plotdata_embedding_chunk_domain_utt_v2_points.csv",
        REPORTS / "plotdata_embedding_chunk_domain_utt_v2_points.csv",
    )
    d0 = pd.read_csv(baseline_path)
    d1 = pd.read_csv(stage1_path)

    metrics0 = _compute_metrics(d0)
    metrics1 = _compute_metrics(d1)
    metric_order = [
        "source_target_centroid_distance",
        "source_target_overlap_proxy",
        "within_emotion_compactness",
        "between_emotion_centroid_separation",
        "domain_separability_proxy",
        "emotion_separability_proxy",
    ]
    short_label = {
        "source_target_centroid_distance": "Dom Centroid Dist",
        "source_target_overlap_proxy": "Dom Overlap Proxy",
        "within_emotion_compactness": "Within Emo Compact",
        "between_emotion_centroid_separation": "Between Emo Sep",
        "domain_separability_proxy": "Domain Sep Proxy",
        "emotion_separability_proxy": "Emotion Sep Proxy",
    }

    metric_df = pd.DataFrame(
        {
            "metric": metric_order,
            "Baseline Chunk": [metrics0[m] for m in metric_order],
            "Stage1 Domain-Utt": [metrics1[m] for m in metric_order],
        }
    )
    metric_df.to_csv(REPORTS / "plotdata_embedding_diagnostics_metrics.csv", index=False)

    emotions = sorted(d0["emotion"].unique().tolist())
    dist0 = _centroid_distance_matrix(d0, emotions)
    dist1 = _centroid_distance_matrix(d1, emotions)
    delta_dist = dist1 - dist0
    pd.DataFrame(delta_dist, index=emotions, columns=emotions).to_csv(
        REPORTS / "plotdata_embedding_centroid_distance_delta_stage1_minus_baseline.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), constrained_layout=True)
    _panel_embedding(axes[0, 0], d0, "A. Baseline: 2D Embedding + Domain 95% Ellipses")
    _panel_embedding(axes[0, 1], d1, "B. Stage1: 2D Embedding + Domain 95% Ellipses")

    # Panel C: quantitative diagnostics bar chart
    cax = axes[1, 0]
    x = np.arange(len(metric_order))
    w = 0.36
    y0 = np.array([metrics0[m] for m in metric_order], dtype=float)
    y1 = np.array([metrics1[m] for m in metric_order], dtype=float)
    cax.bar(x - w / 2, y0, width=w, label="Baseline", color="#4c78a8")
    cax.bar(x + w / 2, y1, width=w, label="Stage1", color="#e45756")
    cax.set_xticks(x)
    cax.set_xticklabels([short_label[m] for m in metric_order], rotation=25, ha="right")
    cax.set_ylabel("2D Proxy Value")
    cax.set_title("C. Quantitative Diagnostics (2D Proxies)")
    cax.legend(frameon=True, title="")

    # Panel D: centroid-distance matrix delta
    dax = axes[1, 1]
    sns.heatmap(
        delta_dist,
        cmap="RdBu_r",
        center=0.0,
        annot=True,
        fmt=".2f",
        xticklabels=emotions,
        yticklabels=emotions,
        cbar_kws={"label": "Stage1 - Baseline", "pad": 0.03},
        ax=dax,
    )
    dax.set_title("D. Emotion Centroid Distance Delta")
    dax.set_xlabel("Emotion")
    dax.set_ylabel("Emotion")
    dax.tick_params(axis="x", rotation=20)
    dax.tick_params(axis="y", rotation=0)

    fig.suptitle("Embedding Diagnostics Dashboard (IEMOCAP→PODCAST, 2D Projection Space)", y=1.01)
    save_png_pdf(fig, REPORTS / "fig_embedding_diagnostics_dashboard")

    note = [
        "# Embedding Diagnostics Dashboard Metrics",
        "",
        "All metrics below are computed on 2D projected plotting coordinates only.",
        "They are visualization-space diagnostics, not direct claims about the original high-dimensional embeddings.",
        "",
        "## Metric definitions",
        "",
        "- **source_target_centroid_distance**: Euclidean distance between source and target domain centroids in 2D.",
        "- **source_target_overlap_proxy**: Fraction of all points that lie inside both domain 95% Gaussian ellipses (overlap proxy).",
        "- **within_emotion_compactness**: Mean Euclidean distance from each point to its emotion centroid (lower means tighter clusters).",
        "- **between_emotion_centroid_separation**: Mean pairwise Euclidean distance among emotion centroids (higher means better separation).",
        "- **domain_separability_proxy**: Fraction of points closer to their own domain centroid than to the other domain centroid.",
        "- **emotion_separability_proxy**: Ratio = between_emotion_centroid_separation / within_emotion_compactness.",
        "",
        "## Reading guidance",
        "",
        "- For compactness, lower is better.",
        "- For separation/separability, higher is better.",
        "- Interpret trends as 2D visualization diagnostics only.",
    ]
    (REPORTS / "fig_embedding_diagnostics_dashboard_metrics.md").write_text("\n".join(note), encoding="utf-8")

    print("Generated:")
    print("- reports/fig_embedding_diagnostics_dashboard.png")
    print("- reports/fig_embedding_diagnostics_dashboard.pdf")
    print("- reports/fig_embedding_diagnostics_dashboard_metrics.md")
    print("- reports/plotdata_embedding_diagnostics_metrics.csv")
    print("- reports/plotdata_embedding_centroid_distance_delta_stage1_minus_baseline.csv")


if __name__ == "__main__":
    main()

