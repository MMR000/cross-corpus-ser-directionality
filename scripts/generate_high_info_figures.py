#!/usr/bin/env python3
"""Generate high-information paper figures from existing experiment outputs only.

Outputs:
- slope/dumbbell comparison (Stage progression)
- class-wise delta heatmap
- paired normalized confusion matrices
- improved embedding plots (with centroids) + plotting-point CSVs
- gain-vs-baseline fraction ablation plots

All figures are saved as both PNG and PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from src.data.datasets import ID_TO_EMOTION, build_loader
from src.models import build_model
from src.training.utils import device_from_config, load_yaml, safe_torch_load
from scripts.plot_style import FIGSIZE, PUB_LABEL, apply_shared_style, save_png_pdf


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs" / "phase2"
EXP = ROOT / "exp"


def _require(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(f"- {m}" for m in missing))


def _normalized_confusion_from_csv(path: Path) -> np.ndarray:
    cm = pd.read_csv(path, header=None).values.astype(float)
    rs = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, np.clip(rs, 1e-12, None))


def make_slope_stage_comparison(overview: pd.DataFrame) -> None:
    rows = []
    direction_label = {
        "iemocap_to_podcast": PUB_LABEL["iemocap_to_podcast"],
        "podcast_to_iemocap": PUB_LABEL["podcast_to_iemocap"],
        "iemocap_plus_cremad_to_podcast": PUB_LABEL["iemocap_plus_cremad_to_podcast"],
    }
    stage_label = {
        "baseline_chunk": PUB_LABEL["baseline"],
        "stage1_domain_utt": PUB_LABEL["stage1"],
        "stage2_domain_both": PUB_LABEL["stage2"],
    }
    specs = [
        ("iemocap_to_podcast", "iemocap_to_podcast_chunk", "iemocap_to_podcast_chunk_domain_utt", "iemocap_to_podcast_chunk_domain_both"),
        ("podcast_to_iemocap", "podcast_to_iemocap_chunk", "podcast_to_iemocap_chunk_domain_utt", "podcast_to_iemocap_chunk_domain_both"),
        ("iemocap_plus_cremad_to_podcast", "iemocap_plus_cremad_to_podcast_chunk", "iemocap_plus_cremad_to_podcast_chunk_domain_utt", "iemocap_plus_cremad_to_podcast_chunk_domain_both"),
    ]
    for direction, b, s1, s2 in specs:
        for stage_name, exp_name in [("baseline_chunk", b), ("stage1_domain_utt", s1), ("stage2_domain_both", s2)]:
            uar = float(overview.loc[overview["experiment"] == exp_name, "test_uar"].iloc[0])
            rows.append({"direction": direction, "stage": stage_name, "test_uar": uar})
    d = pd.DataFrame(rows)
    d["direction_pub"] = d["direction"].map(direction_label)
    d["stage_pub"] = d["stage"].map(stage_label)
    d.to_csv(REPORTS / "plotdata_slope_stage_comparison.csv", index=False)

    stage_order = ["baseline_chunk", "stage1_domain_utt", "stage2_domain_both"]
    x_map = {s: i for i, s in enumerate(stage_order)}
    fig, ax = plt.subplots(figsize=FIGSIZE["slope"], constrained_layout=True)
    palette = sns.color_palette("colorblind", 3)
    dir_order = ["iemocap_to_podcast", "podcast_to_iemocap", "iemocap_plus_cremad_to_podcast"]
    for i, direction in enumerate(dir_order):
        sub = d[d["direction"] == direction].set_index("stage").loc[stage_order].reset_index()
        xs = [x_map[s] for s in sub["stage"]]
        ys = sub["test_uar"].values
        ax.plot(xs, ys, marker="o", linewidth=2.2, color=palette[i], label=direction_label[direction])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.002, f"{y:.3f}", ha="center", va="bottom", fontsize=8, color=palette[i])
    ax.set_xticks(range(len(stage_order)))
    ax.set_xticklabels([stage_label[s] for s in stage_order], rotation=0)
    ax.set_ylabel("Test UAR")
    ax.set_title("Stage Progression")
    ax.legend(title="", frameon=True, loc="best")
    save_png_pdf(fig, REPORTS / "fig_slope_stage_comparison")


def make_classwise_delta_heatmap(classwise: pd.DataFrame) -> None:
    d = classwise.copy()
    d["p2i_stage1_minus_baseline"] = d["p2i_stage1_recall"] - d["p2i_chunk_recall"]
    heat = d[
        [
            "emotion",
            "improve_stage1_i2p",
            "hurt_stage2_vs_stage1_i2p",
            "p2i_stage1_minus_baseline",
            "hurt_stage2_vs_stage1_p2i",
        ]
    ].rename(
        columns={
            "improve_stage1_i2p": "i2p S1-B",
            "hurt_stage2_vs_stage1_i2p": "i2p S2-S1",
            "p2i_stage1_minus_baseline": "p2i S1-B",
            "hurt_stage2_vs_stage1_p2i": "p2i S2-S1",
        }
    )
    heat = heat.set_index("emotion").loc[["angry", "happy", "sad", "neutral"]]
    heat.to_csv(REPORTS / "plotdata_classwise_delta_heatmap.csv")

    fig, ax = plt.subplots(figsize=FIGSIZE["heatmap"], constrained_layout=True)
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.5,
        cbar_kws={"label": "Recall Delta", "pad": 0.04},
        ax=ax,
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Recall Delta", labelpad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("Emotion")
    ax.set_title("Class-wise Recall Delta Heatmap")
    save_png_pdf(fig, REPORTS / "fig_classwise_delta_heatmap")


def make_paired_confusion_i2p() -> None:
    base = _normalized_confusion_from_csv(EXP / "iemocap_to_podcast_chunk" / "confusion_test.csv")
    s1 = _normalized_confusion_from_csv(EXP / "iemocap_to_podcast_chunk_domain_utt" / "confusion_test.csv")
    labels = [ID_TO_EMOTION[i] for i in sorted(ID_TO_EMOTION.keys())]
    pd.DataFrame(base, index=labels, columns=labels).to_csv(REPORTS / "plotdata_i2p_confusion_baseline_norm.csv")
    pd.DataFrame(s1, index=labels, columns=labels).to_csv(REPORTS / "plotdata_i2p_confusion_stage1_norm.csv")

    fig, axes = plt.subplots(
        1, 2, figsize=FIGSIZE["paired_confusion"], sharex=True, sharey=True, constrained_layout=True
    )
    ims = []
    for ax, cm, title in [
        (axes[0], base, f"{PUB_LABEL['iemocap_to_podcast']} {PUB_LABEL['baseline']}"),
        (axes[1], s1, f"{PUB_LABEL['iemocap_to_podcast']} {PUB_LABEL['stage1']}"),
    ]:
        hm = sns.heatmap(
            cm,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            annot=True,
            fmt=".2f",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ims.append(hm.collections[0])
        ax.set_title(title)
        ax.set_xlabel("Predicted Emotion")
        ax.set_ylabel("True Emotion")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)
    cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("Normalized Recall", labelpad=12)
    fig.suptitle("Paired Normalized Confusion", y=1.02)
    save_png_pdf(fig, REPORTS / "fig_i2p_confusion_paired")


@torch.no_grad()
def _collect_embedding_points(config_path: Path, exp_name: str, max_per_domain: int = 900) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    device = device_from_config(cfg)
    ckpt = safe_torch_load(EXP / exp_name / "best.ckpt", map_location=device)
    saved_cfg = ckpt.get("config")
    if isinstance(saved_cfg, dict) and isinstance(saved_cfg.get("model"), dict):
        cfg["model"] = {**cfg.get("model", {}), **saved_cfg["model"]}
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    data_cfg = cfg["data"]
    src_loader = build_loader(
        manifest_path=data_cfg["train_manifest"],
        batch_size=16,
        shuffle=False,
        num_workers=2,
        path_column=data_cfg.get("path_column", "processed_wav_path"),
        label_column=data_cfg.get("label_column", "emotion"),
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        max_duration_sec=data_cfg.get("max_duration_sec"),
        metadata_path=data_cfg.get("metadata_manifest"),
        domain_column=None,
        domain_id_map=None,
    )
    tgt_manifest = cfg.get("uda", {}).get("target_dev_manifest", data_cfg["test_manifest"])
    tgt_loader = build_loader(
        manifest_path=tgt_manifest,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        path_column=data_cfg.get("path_column", "processed_wav_path"),
        label_column=data_cfg.get("label_column", "emotion"),
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        max_duration_sec=data_cfg.get("max_duration_sec"),
        metadata_path=data_cfg.get("metadata_manifest"),
        domain_column=None,
        domain_id_map=None,
    )
    rows = []
    for loader, domain in [(src_loader, "source"), (tgt_loader, "target")]:
        c = 0
        for batch in loader:
            out = model(batch["waveforms"].to(device), batch["lengths"].to(device), domain_ids=None)
            emb = out["pooled"].detach().cpu().numpy()
            labels = batch["labels"].numpy()
            for i in range(emb.shape[0]):
                rows.append({"domain": domain, "emotion": ID_TO_EMOTION[int(labels[i])], "emb": emb[i]})
                c += 1
                if c >= max_per_domain:
                    break
            if c >= max_per_domain:
                break
    mat = np.stack([r["emb"] for r in rows], axis=0)
    z = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(mat)
    data = []
    for i, r in enumerate(rows):
        data.append({"x": float(z[i, 0]), "y": float(z[i, 1]), "domain": r["domain"], "emotion": r["emotion"]})
    return pd.DataFrame(data)


def _add_centroids(df: pd.DataFrame, by: str) -> pd.DataFrame:
    cent = df.groupby(by, as_index=False)[["x", "y"]].mean()
    cent["kind"] = by
    return cent


def _plot_embedding_with_centroids(df: pd.DataFrame, title: str, out_base: Path, csv_base: Path) -> None:
    df.to_csv(csv_base.with_suffix(".csv"), index=False)
    emo_cent = _add_centroids(df, "emotion")
    dom_cent = _add_centroids(df, "domain")
    emo_cent.to_csv(csv_base.with_name(csv_base.stem + "_emotion_centroids.csv"), index=False)
    dom_cent.to_csv(csv_base.with_name(csv_base.stem + "_domain_centroids.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["embedding_pair"], constrained_layout=True)
    # panel A emotion
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="emotion",
        style="emotion",
        s=11,
        alpha=0.30,
        ax=axes[0],
        legend=True,
    )
    axes[0].scatter(emo_cent["x"], emo_cent["y"], marker="X", s=110, c="black", label="Emotion centroid")
    axes[0].set_title("A. Colored by Emotion")
    axes[0].set_xlabel("t-SNE-1")
    axes[0].set_ylabel("t-SNE-2")
    # panel B domain
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="domain",
        style="domain",
        s=11,
        alpha=0.30,
        ax=axes[1],
        legend=True,
    )
    axes[1].scatter(dom_cent["x"], dom_cent["y"], marker="X", s=120, c="black", label="Domain centroid")
    axes[1].set_title("B. Colored/Marked by Domain")
    axes[1].set_xlabel("t-SNE-1")
    axes[1].set_ylabel("t-SNE-2")
    fig.suptitle(title, y=1.03)
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("")
            leg.set_bbox_to_anchor((1.02, 1.0))
            leg._loc = 2
    save_png_pdf(fig, out_base)


def make_embedding_v2() -> None:
    d_base = _collect_embedding_points(CONFIGS / "iemocap_to_podcast_chunk.yaml", "iemocap_to_podcast_chunk")
    d_s1 = _collect_embedding_points(
        CONFIGS / "iemocap_to_podcast_chunk_domain_utt.yaml", "iemocap_to_podcast_chunk_domain_utt"
    )
    _plot_embedding_with_centroids(
        d_base,
        f"{PUB_LABEL['iemocap_to_podcast']} {PUB_LABEL['baseline']} Embedding",
        REPORTS / "fig_embedding_chunk_baseline_v2",
        REPORTS / "plotdata_embedding_chunk_baseline_v2_points",
    )
    _plot_embedding_with_centroids(
        d_s1,
        f"{PUB_LABEL['iemocap_to_podcast']} {PUB_LABEL['stage1']} Embedding",
        REPORTS / "fig_embedding_chunk_domain_utt_v2",
        REPORTS / "plotdata_embedding_chunk_domain_utt_v2_points",
    )


def make_fraction_gain_plots(tgt_df: pd.DataFrame, src_df: pd.DataFrame, overview: pd.DataFrame) -> None:
    # target fraction: delta vs plain chunk
    plain = float(overview.loc[overview["experiment"] == "iemocap_to_podcast_chunk", "test_uar"].iloc[0])
    d_t = tgt_df.sort_values("target_fraction").copy()
    d_t["delta_vs_plain_chunk"] = d_t["test_uar"] - plain
    d_t.to_csv(REPORTS / "plotdata_target_fraction_gain.csv", index=False)
    fig, ax = plt.subplots(figsize=FIGSIZE["line_ablation"], constrained_layout=True)
    ax.plot(d_t["target_fraction"], d_t["delta_vs_plain_chunk"], marker="o", linewidth=2.1)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("Target Unlabeled Fraction")
    ax.set_ylabel("Delta UAR vs Plain Chunk")
    ax.set_title("Target Fraction Gain vs Baseline")
    save_png_pdf(fig, REPORTS / "fig_target_fraction_ablation")

    # source fraction: delta vs plain chunk and vs stage1_full where relevant
    stage1_full = float(overview.loc[overview["experiment"] == "iemocap_to_podcast_chunk_domain_utt", "test_uar"].iloc[0])
    d_s = src_df.copy()
    d_s["delta_vs_plain_chunk"] = d_s["test_uar"] - plain
    d_s["delta_vs_stage1_full"] = np.where(
        d_s["model_variant"] == "chunk_domain_utt",
        d_s["test_uar"] - stage1_full,
        np.nan,
    )
    d_s.to_csv(REPORTS / "plotdata_source_fraction_gain.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharex=True, constrained_layout=True)
    for variant, label in [("chunk", "Chunk"), ("chunk_domain_utt", "Chunk + Domain-Utt")]:
        dd = d_s[d_s["model_variant"] == variant].sort_values("source_fraction")
        axes[0].plot(dd["source_fraction"], dd["delta_vs_plain_chunk"], marker="o", linewidth=2, label=label)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Delta vs Plain Chunk")
    axes[0].set_xlabel("Source Labeled Fraction")
    axes[0].set_ylabel("Delta UAR")
    axes[0].set_xticks([0.25, 0.50, 1.00])
    axes[0].legend(frameon=True, loc="best")

    dd = d_s[d_s["model_variant"] == "chunk_domain_utt"].sort_values("source_fraction")
    axes[1].plot(dd["source_fraction"], dd["delta_vs_stage1_full"], marker="o", linewidth=2, color="tab:green")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Chunk+Domain-Utt Delta vs Stage1 (1.00)")
    axes[1].set_xlabel("Source Labeled Fraction")
    axes[1].set_ylabel("Delta UAR")
    axes[1].set_xticks([0.25, 0.50, 1.00])

    fig.suptitle("Source Fraction Gain Plots", y=1.02)
    save_png_pdf(fig, REPORTS / "fig_source_fraction_ablation")


def fig_stage2_ablation_i2p(ab_df: pd.DataFrame) -> None:
    keep = [
        ("iemocap_to_podcast_chunk", PUB_LABEL["baseline"]),
        ("iemocap_to_podcast_chunk_domain_utt", PUB_LABEL["stage1"]),
        ("iemocap_to_podcast_chunk_domain_both", PUB_LABEL["stage2"]),
        ("iemocap_to_podcast_chunk_domain_both_both_weakchunk", "both_weakchunk"),
        ("iemocap_to_podcast_chunk_domain_both_both_midchunk", "both_midchunk"),
        ("iemocap_to_podcast_chunk_domain_both_chunk_only_w01", "chunk_only_w01"),
        ("iemocap_to_podcast_chunk_domain_both_chunk_only_w03", "chunk_only_w03"),
        ("iemocap_to_podcast_chunk_domain_both_chunk_only_w05", "chunk_only_w05"),
    ]
    map_name = {k: v for k, v in keep}
    d = ab_df[ab_df["experiment"].isin(map_name.keys())].copy()
    d["label"] = d["experiment"].map(map_name)
    order = [
        PUB_LABEL["baseline"],
        PUB_LABEL["stage1"],
        PUB_LABEL["stage2"],
        "both_weakchunk",
        "both_midchunk",
        "chunk_only_w01",
        "chunk_only_w03",
        "chunk_only_w05",
    ]
    stage1_ref = float(d.loc[d["label"] == PUB_LABEL["stage1"], "test_uar"].iloc[0])
    d["delta_vs_stage1"] = d["test_uar"] - stage1_ref
    d = d.set_index("label").reindex(order).reset_index()
    d.to_csv(REPORTS / "plotdata_stage2_ablation_i2p.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    sns.barplot(
        data=d,
        y="label",
        x="delta_vs_stage1",
        hue="label",
        order=order,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Delta UAR vs Stage 1")
    ax.set_ylabel("Variant")
    ax.set_title(f"{PUB_LABEL['stage2']} Ablation vs {PUB_LABEL['stage1']} ({PUB_LABEL['iemocap_to_podcast']})")
    save_png_pdf(fig, REPORTS / "fig_stage2_ablation_i2p")


def fig_multiseed(ms_df: pd.DataFrame) -> None:
    d = ms_df.copy()
    d["label"] = d["experiment"].map(
        {
            "iemocap_to_podcast_chunk": f"{PUB_LABEL['iemocap_to_podcast']} {PUB_LABEL['baseline']}",
            "iemocap_to_podcast_chunk_domain_utt": f"{PUB_LABEL['iemocap_to_podcast']} {PUB_LABEL['stage1']}",
            "podcast_to_iemocap_chunk": f"{PUB_LABEL['podcast_to_iemocap']} {PUB_LABEL['baseline']}",
            "podcast_to_iemocap_chunk_domain_utt": f"{PUB_LABEL['podcast_to_iemocap']} {PUB_LABEL['stage1']}",
        }
    )
    order = [
        f"{PUB_LABEL['iemocap_to_podcast']} {PUB_LABEL['baseline']}",
        f"{PUB_LABEL['iemocap_to_podcast']} {PUB_LABEL['stage1']}",
        f"{PUB_LABEL['podcast_to_iemocap']} {PUB_LABEL['baseline']}",
        f"{PUB_LABEL['podcast_to_iemocap']} {PUB_LABEL['stage1']}",
    ]
    dd = d.set_index("label").loc[order].reset_index()
    dd.to_csv(REPORTS / "plotdata_multiseed_robustness.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    x = np.arange(len(dd))
    ax.errorbar(
        x,
        dd["mean_test_uar"],
        yerr=dd["std_test_uar"],
        fmt="o",
        markersize=6,
        capsize=4,
        linewidth=1.6,
        color="black",
    )
    ax.scatter(x, dd["mean_test_uar"], s=38, c=sns.color_palette("colorblind", 4), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(dd["label"])
    ax.set_xlabel("Model / Direction")
    ax.set_ylabel("Mean Test UAR")
    ax.set_title("Multi-seed Robustness (mean ± std)")
    for t in ax.get_xticklabels():
        t.set_rotation(20)
        t.set_ha("right")
    save_png_pdf(fig, REPORTS / "fig_multiseed_robustness")


def write_caption_manifest() -> None:
    cap_lines = [
        "## Paper Figure Captions (High-information v2)",
        "",
        "### fig_slope_stage_comparison.png / .pdf",
        "Slope chart of stage progression (baseline chunk, Stage 1, Stage 2) for three transfer directions. The x-axis denotes stage and the y-axis denotes test UAR; each line corresponds to one transfer direction.",
        "",
        "### fig_classwise_delta_heatmap.png / .pdf",
        "Class-wise recall delta heatmap with rows as emotion classes and columns as directional stage deltas: i2p Stage1-baseline, i2p Stage2-Stage1, p2i Stage1-baseline, and p2i Stage2-Stage1. Cells are numerically annotated.",
        "",
        "### fig_i2p_confusion_paired.png / .pdf",
        "Paired normalized confusion matrices for IEMOCAP→PODCAST under baseline chunk and Stage 1 (chunk_domain_utt). Axes denote true and predicted emotion labels.",
        "",
        "### fig_embedding_chunk_baseline_v2.png / .pdf",
        "Two-panel t-SNE embedding visualization for IEMOCAP→PODCAST chunk baseline. Panel A uses emotion colors/markers; Panel B uses domain colors/markers. Transparent points show samples; X markers show centroids.",
        "",
        "### fig_embedding_chunk_domain_utt_v2.png / .pdf",
        "Two-panel t-SNE embedding visualization for IEMOCAP→PODCAST chunk_domain_utt with the same styling as baseline. Transparent points show samples; X markers show centroids.",
        "",
        "### fig_target_fraction_ablation.png / .pdf",
        "Target-fraction ablation plotted as delta UAR relative to plain chunk baseline. X-axis is target unlabeled fraction; y-axis is gain/loss versus baseline.",
        "",
        "### fig_source_fraction_ablation.png / .pdf",
        "Source-fraction ablation plotted as gain curves. Left panel shows delta UAR vs plain chunk for chunk and chunk_domain_utt; right panel shows chunk_domain_utt delta UAR vs Stage1 full-source reference.",
        "",
        "### fig_stage2_ablation_i2p.png / .pdf",
        "Horizontal delta plot for IEMOCAP→PODCAST Stage 2 ablations, reporting each variant as ΔUAR relative to Stage 1.",
        "",
        "### fig_multiseed_robustness.png / .pdf",
        "Multi-seed robustness errorbar point plot showing mean UAR with standard-deviation bars for i2p_chunk, i2p_chunk_domain_utt, p2i_chunk, and p2i_chunk_domain_utt.",
    ]
    (REPORTS / "paper_figure_captions.md").write_text("\n".join(cap_lines), encoding="utf-8")

    rows = [
        ("fig_slope_stage_comparison.png", "core", "Stage progression slope chart", "main", "main"),
        ("fig_classwise_delta_heatmap.png", "classwise", "Class-wise stage delta heatmap", "main", "main"),
        ("fig_i2p_confusion_paired.png", "classwise", "Paired normalized confusion matrices (i2p)", "main", "main"),
        ("fig_embedding_chunk_baseline_v2.png", "embedding", "Embedding plot baseline with centroids", "main", "main"),
        ("fig_embedding_chunk_domain_utt_v2.png", "embedding", "Embedding plot stage1 with centroids", "main", "main"),
        ("fig_target_fraction_ablation.png", "ablation", "Target fraction gain plot", "supplementary", "supplementary"),
        ("fig_source_fraction_ablation.png", "ablation", "Source fraction gain plot", "supplementary", "supplementary"),
        ("fig_stage2_ablation_i2p.png", "ablation", "Stage2 ablation comparison", "supplementary", "supplementary"),
        ("fig_multiseed_robustness.png", "robustness", "Mean/std multi-seed UAR", "supplementary", "supplementary"),
    ]
    pd.DataFrame(
        rows,
        columns=["figure_file", "figure_group", "description", "main_or_supplementary", "recommended_use"],
    ).to_csv(REPORTS / "paper_figures_manifest.csv", index=False)


def main() -> None:
    apply_shared_style()
    _require(
        [
            REPORTS / "final_baseline_stage1_stage2_overview.csv",
            REPORTS / "classwise_cross_corpus_summary.csv",
            REPORTS / "uda_target_fraction_ablation.csv",
            REPORTS / "source_fraction_ablation.csv",
            REPORTS / "stage2_ablation_iemocap_to_podcast.csv",
            REPORTS / "multiseed_core_results.csv",
            CONFIGS / "iemocap_to_podcast_chunk.yaml",
            CONFIGS / "iemocap_to_podcast_chunk_domain_utt.yaml",
            EXP / "iemocap_to_podcast_chunk" / "best.ckpt",
            EXP / "iemocap_to_podcast_chunk_domain_utt" / "best.ckpt",
            EXP / "iemocap_to_podcast_chunk" / "confusion_test.csv",
            EXP / "iemocap_to_podcast_chunk_domain_utt" / "confusion_test.csv",
        ]
    )

    overview = pd.read_csv(REPORTS / "final_baseline_stage1_stage2_overview.csv")
    classwise = pd.read_csv(REPORTS / "classwise_cross_corpus_summary.csv")
    tgt_df = pd.read_csv(REPORTS / "uda_target_fraction_ablation.csv")
    src_df = pd.read_csv(REPORTS / "source_fraction_ablation.csv")
    s2_df = pd.read_csv(REPORTS / "stage2_ablation_iemocap_to_podcast.csv")
    ms_df = pd.read_csv(REPORTS / "multiseed_core_results.csv")

    make_slope_stage_comparison(overview)
    make_classwise_delta_heatmap(classwise)
    make_paired_confusion_i2p()
    make_embedding_v2()
    make_fraction_gain_plots(tgt_df, src_df, overview)
    fig_stage2_ablation_i2p(s2_df)
    fig_multiseed(ms_df)
    write_caption_manifest()

    outputs = [
        "fig_slope_stage_comparison.png/.pdf",
        "fig_classwise_delta_heatmap.png/.pdf",
        "fig_i2p_confusion_paired.png/.pdf",
        "fig_embedding_chunk_baseline_v2.png/.pdf",
        "fig_embedding_chunk_domain_utt_v2.png/.pdf",
        "fig_target_fraction_ablation.png/.pdf",
        "fig_source_fraction_ablation.png/.pdf",
        "fig_stage2_ablation_i2p.png/.pdf",
        "fig_multiseed_robustness.png/.pdf",
        "paper_figure_captions.md",
        "paper_figures_manifest.csv",
        "plotdata_slope_stage_comparison.csv",
        "plotdata_classwise_delta_heatmap.csv",
        "plotdata_i2p_confusion_baseline_norm.csv",
        "plotdata_i2p_confusion_stage1_norm.csv",
        "plotdata_embedding_chunk_baseline_v2_points.csv",
        "plotdata_embedding_chunk_baseline_v2_points_emotion_centroids.csv",
        "plotdata_embedding_chunk_baseline_v2_points_domain_centroids.csv",
        "plotdata_embedding_chunk_domain_utt_v2_points.csv",
        "plotdata_embedding_chunk_domain_utt_v2_points_emotion_centroids.csv",
        "plotdata_embedding_chunk_domain_utt_v2_points_domain_centroids.csv",
        "plotdata_target_fraction_gain.csv",
        "plotdata_source_fraction_gain.csv",
    ]
    print("Generation complete. Outputs in reports/:")
    for o in outputs:
        print(f"- {o}")


if __name__ == "__main__":
    main()
