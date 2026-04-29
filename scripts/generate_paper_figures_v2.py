#!/usr/bin/env python3
"""Generate paper-ready figure assets from existing result files only."""

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


def _ensure_inputs(required: Iterable[Path]) -> None:
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        msg = ["Missing required input file(s):"] + [f"- {m}" for m in missing]
        raise FileNotFoundError("\n".join(msg))


def _setup_style() -> None:
    apply_shared_style()


def _save(fig: plt.Figure, path: Path) -> None:
    # backward-compatible single-path helper now writes both png/pdf
    save_png_pdf(fig, path.with_suffix(""))


def fig_in_corpus_v2(df: pd.DataFrame) -> Path:
    plot_df = df[df["training_setup"] == "in_corpus"].copy()
    plot_df["corpus"] = plot_df["target_domain"].str.upper()
    plot_df["model"] = plot_df["model_variant"].replace(
        {"meanpool": "MeanPool", "attnpool": "AttnPool", "chunk": "Chunk"}
    )
    order = ["IEMOCAP", "PODCAST"]
    hue_order = ["MeanPool", "AttnPool", "Chunk"]
    fig, ax = plt.subplots(figsize=FIGSIZE["line_ablation"], constrained_layout=True)
    sns.barplot(
        data=plot_df,
        x="corpus",
        y="test_uar",
        hue="model",
        order=order,
        hue_order=hue_order,
        palette="colorblind",
        ax=ax,
    )
    ax.set_xlabel("Corpus")
    ax.set_ylabel("Test UAR")
    ax.set_title("In-Corpus Performance")
    ax.legend(title="Model", frameon=True)
    out = REPORTS / "fig_in_corpus_barplot_v2.png"
    _save(fig, out)
    return out


def fig_cross_corpus_v2(df: pd.DataFrame) -> Path:
    plot_df = df[
        (df["family"] == "baseline")
        & (df["training_setup"].isin(["cross_single_source", "cross_multi_source"]))
    ].copy()
    plot_df["direction"] = plot_df["source_domain"] + " \u2192 " + plot_df["target_domain"]
    plot_df["model"] = plot_df["model_variant"].replace(
        {"meanpool": "MeanPool", "attnpool": "AttnPool", "chunk": "Chunk"}
    )
    order = ["iemocap \u2192 podcast", "podcast \u2192 iemocap", "iemocap+crema_d \u2192 podcast"]
    hue_order = ["MeanPool", "AttnPool", "Chunk"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    sns.barplot(
        data=plot_df,
        x="direction",
        y="test_uar",
        hue="model",
        order=order,
        hue_order=hue_order,
        palette="colorblind",
        ax=ax,
    )
    ax.set_xlabel("Transfer Direction")
    ax.set_ylabel("Test UAR")
    ax.set_title("Cross-Corpus Baseline Performance")
    for t in ax.get_xticklabels():
        t.set_rotation(15)
        t.set_ha("right")
    ax.legend(title="Model", frameon=True)
    out = REPORTS / "fig_cross_corpus_barplot_v2.png"
    _save(fig, out)
    return out


def fig_stage12_v2(df: pd.DataFrame) -> Path:
    keep = [
        ("iemocap_to_podcast_chunk", "Chunk"),
        ("iemocap_to_podcast_chunk_domain_utt", "Stage1: Domain-Utt"),
        ("iemocap_to_podcast_chunk_domain_both", "Stage2: Domain-Both"),
        ("iemocap_to_podcast_chunk_domain_both_both_weakchunk", "Stage2: Both-WeakChunk"),
        ("podcast_to_iemocap_chunk", "Chunk"),
        ("podcast_to_iemocap_chunk_domain_utt", "Stage1: Domain-Utt"),
        ("podcast_to_iemocap_chunk_domain_both", "Stage2: Domain-Both"),
        ("iemocap_plus_cremad_to_podcast_chunk", "Chunk"),
        ("iemocap_plus_cremad_to_podcast_chunk_domain_utt", "Stage1: Domain-Utt"),
        ("iemocap_plus_cremad_to_podcast_chunk_domain_both", "Stage2: Domain-Both"),
    ]
    map_label = {k: v for k, v in keep}
    plot_df = df[df["experiment"].isin(map_label.keys())].copy()
    plot_df["variant"] = plot_df["experiment"].map(map_label)
    plot_df["direction"] = plot_df["source_domain"] + " \u2192 " + plot_df["target_domain"]
    order = ["iemocap \u2192 podcast", "podcast \u2192 iemocap", "iemocap+crema_d \u2192 podcast"]
    hue_order = ["Chunk", "Stage1: Domain-Utt", "Stage2: Domain-Both", "Stage2: Both-WeakChunk"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    sns.barplot(
        data=plot_df,
        x="direction",
        y="test_uar",
        hue="variant",
        order=order,
        hue_order=hue_order,
        palette="deep",
        ax=ax,
    )
    ax.set_xlabel("Transfer Direction")
    ax.set_ylabel("Test UAR")
    ax.set_title("Chunk Baseline vs Stage 1/Stage 2 Variants")
    for t in ax.get_xticklabels():
        t.set_rotation(15)
        t.set_ha("right")
    ax.legend(title="Variant", frameon=True)
    out = REPORTS / "fig_stage1_stage2_barplot_v2.png"
    _save(fig, out)
    return out


def _classwise_long(class_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_chunk_recall", f"{prefix}_stage1_recall", f"{prefix}_stage2_recall"]
    out = class_df[["emotion"] + cols].copy()
    out = out.melt(id_vars="emotion", value_vars=cols, var_name="variant", value_name="recall")
    mapping = {
        f"{prefix}_chunk_recall": "Chunk",
        f"{prefix}_stage1_recall": "Stage1: Domain-Utt",
        f"{prefix}_stage2_recall": "Stage2: Domain-Both",
    }
    out["variant"] = out["variant"].map(mapping)
    return out


def fig_classwise_i2p(class_df: pd.DataFrame) -> Path:
    plot_df = _classwise_long(class_df, "i2p")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    sns.barplot(
        data=plot_df,
        x="emotion",
        y="recall",
        hue="variant",
        order=["angry", "happy", "sad", "neutral"],
        hue_order=["Chunk", "Stage1: Domain-Utt", "Stage2: Domain-Both"],
        palette="deep",
        ax=ax,
    )
    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.0)
    ax.set_title("Class-wise Recall: IEMOCAP \u2192 PODCAST")
    ax.legend(title="Variant", frameon=True)
    out = REPORTS / "fig_classwise_i2p.png"
    _save(fig, out)
    return out


def fig_classwise_p2i(class_df: pd.DataFrame) -> Path:
    plot_df = _classwise_long(class_df, "p2i")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    sns.barplot(
        data=plot_df,
        x="emotion",
        y="recall",
        hue="variant",
        order=["angry", "happy", "sad", "neutral"],
        hue_order=["Chunk", "Stage1: Domain-Utt", "Stage2: Domain-Both"],
        palette="deep",
        ax=ax,
    )
    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.0)
    ax.set_title("Class-wise Recall: PODCAST \u2192 IEMOCAP")
    ax.legend(title="Variant", frameon=True)
    out = REPORTS / "fig_classwise_p2i.png"
    _save(fig, out)
    return out


def fig_classwise_delta_i2p(class_df: pd.DataFrame) -> Path:
    d = class_df[["emotion", "improve_stage1_i2p", "hurt_stage2_vs_stage1_i2p"]].copy()
    d = d.melt(
        id_vars="emotion",
        value_vars=["improve_stage1_i2p", "hurt_stage2_vs_stage1_i2p"],
        var_name="delta_type",
        value_name="delta_recall",
    )
    d["delta_type"] = d["delta_type"].map(
        {
            "improve_stage1_i2p": "Stage1 - Baseline",
            "hurt_stage2_vs_stage1_i2p": "Stage2 - Stage1",
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    sns.barplot(
        data=d,
        x="emotion",
        y="delta_recall",
        hue="delta_type",
        order=["angry", "happy", "sad", "neutral"],
        hue_order=["Stage1 - Baseline", "Stage2 - Stage1"],
        palette="Set2",
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Recall Delta")
    ax.set_title("Recall Deltas: IEMOCAP \u2192 PODCAST")
    ax.legend(title="Delta", frameon=True)
    out = REPORTS / "fig_classwise_delta_i2p.png"
    _save(fig, out)
    return out


def fig_classwise_delta_p2i(class_df: pd.DataFrame) -> Path:
    d = class_df[["emotion", "p2i_stage1_recall", "hurt_stage2_vs_stage1_p2i"]].copy()
    d["stage1_minus_baseline"] = class_df["p2i_stage1_recall"] - class_df["p2i_chunk_recall"]
    d["stage2_minus_stage1"] = class_df["hurt_stage2_vs_stage1_p2i"]
    d = d[["emotion", "stage1_minus_baseline", "stage2_minus_stage1"]].melt(
        id_vars="emotion",
        value_vars=["stage1_minus_baseline", "stage2_minus_stage1"],
        var_name="delta_type",
        value_name="delta_recall",
    )
    d["delta_type"] = d["delta_type"].map(
        {
            "stage1_minus_baseline": "Stage1 - Baseline",
            "stage2_minus_stage1": "Stage2 - Stage1",
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    sns.barplot(
        data=d,
        x="emotion",
        y="delta_recall",
        hue="delta_type",
        order=["angry", "happy", "sad", "neutral"],
        hue_order=["Stage1 - Baseline", "Stage2 - Stage1"],
        palette="Set2",
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Recall Delta")
    ax.set_title("Recall Deltas: PODCAST \u2192 IEMOCAP")
    ax.legend(title="Delta", frameon=True)
    out = REPORTS / "fig_classwise_delta_p2i.png"
    _save(fig, out)
    return out


def fig_target_fraction(tgt_df: pd.DataFrame, overview_df: pd.DataFrame) -> Path:
    d = tgt_df.sort_values("target_fraction").copy()
    plain_chunk = float(
        overview_df.loc[overview_df["experiment"] == "iemocap_to_podcast_chunk", "test_uar"].iloc[0]
    )
    stage1_full = float(
        overview_df.loc[overview_df["experiment"] == "iemocap_to_podcast_chunk_domain_utt", "test_uar"].iloc[0]
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(d["target_fraction"], d["test_uar"], marker="o", linewidth=2, label="Stage1 (fractional target)")
    ax.axhline(plain_chunk, linestyle="--", linewidth=1.5, color="gray", label="Plain chunk baseline")
    ax.axhline(stage1_full, linestyle=":", linewidth=1.8, color="black", label="Stage1 full target (1.00)")
    ax.set_xticks([0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("Target Unlabeled Fraction")
    ax.set_ylabel("Test UAR")
    ax.set_title("Target-Unlabeled Fraction Ablation (IEMOCAP \u2192 PODCAST)")
    ax.legend(frameon=True, loc="best")
    out = REPORTS / "fig_target_fraction_ablation.png"
    _save(fig, out)
    return out


def fig_source_fraction(src_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for variant, label in [("chunk", "Chunk"), ("chunk_domain_utt", "Chunk + Domain-Utt")]:
        d = src_df[src_df["model_variant"] == variant].sort_values("source_fraction")
        ax.plot(d["source_fraction"], d["test_uar"], marker="o", linewidth=2, label=label)
    ax.set_xticks([0.25, 0.50, 1.00])
    ax.set_xlabel("Source Labeled Fraction")
    ax.set_ylabel("Test UAR")
    ax.set_title("Source-Fraction Ablation (IEMOCAP \u2192 PODCAST)")
    ax.legend(frameon=True, loc="best")
    out = REPORTS / "fig_source_fraction_ablation.png"
    _save(fig, out)
    return out


def fig_stage2_ablation_i2p(ab_df: pd.DataFrame) -> Path:
    keep = [
        ("iemocap_to_podcast_chunk", "Plain chunk"),
        ("iemocap_to_podcast_chunk_domain_utt", "Stage1: Domain-Utt"),
        ("iemocap_to_podcast_chunk_domain_both", "Stage2 default"),
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
        "Plain chunk",
        "Stage1: Domain-Utt",
        "Stage2 default",
        "both_weakchunk",
        "both_midchunk",
        "chunk_only_w01",
        "chunk_only_w03",
        "chunk_only_w05",
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sns.barplot(
        data=d,
        y="label",
        x="test_uar",
        order=order,
        palette="viridis",
        ax=ax,
    )
    ax.set_xlabel("Test UAR")
    ax.set_ylabel("Variant")
    ax.set_title("Stage 2 Ablation (IEMOCAP \u2192 PODCAST)")
    out = REPORTS / "fig_stage2_ablation_i2p.png"
    _save(fig, out)
    return out


def fig_multiseed(ms_df: pd.DataFrame) -> Path:
    d = ms_df.copy()
    d["label"] = d["experiment"].map(
        {
            "iemocap_to_podcast_chunk": "i2p: chunk",
            "iemocap_to_podcast_chunk_domain_utt": "i2p: chunk_domain_utt",
            "podcast_to_iemocap_chunk": "p2i: chunk",
            "podcast_to_iemocap_chunk_domain_utt": "p2i: chunk_domain_utt",
        }
    )
    order = ["i2p: chunk", "i2p: chunk_domain_utt", "p2i: chunk", "p2i: chunk_domain_utt"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(
        d.set_index("label").loc[order].index,
        d.set_index("label").loc[order, "mean_test_uar"],
        yerr=d.set_index("label").loc[order, "std_test_uar"],
        capsize=4,
        color=sns.color_palette("colorblind", 4),
    )
    ax.set_ylabel("Mean Test UAR")
    ax.set_xlabel("Model / Direction")
    ax.set_title("Multi-seed Robustness (mean \u00b1 std)")
    for t in ax.get_xticklabels():
        t.set_rotation(15)
        t.set_ha("right")
    out = REPORTS / "fig_multiseed_robustness.png"
    _save(fig, out)
    return out


@torch.no_grad()
def _collect_embeddings(config_path: Path, exp_name: str, max_samples_per_domain: int = 800) -> pd.DataFrame:
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
    for loader, domain_name in [(src_loader, "source"), (tgt_loader, "target")]:
        count = 0
        for batch in loader:
            wf = batch["waveforms"].to(device)
            ln = batch["lengths"].to(device)
            out = model(wf, ln, domain_ids=None)
            emb = out["pooled"].detach().cpu().numpy()
            labels = batch["labels"].numpy()
            for i in range(emb.shape[0]):
                rows.append({"domain": domain_name, "emotion": ID_TO_EMOTION[int(labels[i])], "embedding": emb[i]})
                count += 1
                if count >= max_samples_per_domain:
                    break
            if count >= max_samples_per_domain:
                break
    mat = np.stack([r["embedding"] for r in rows], axis=0)
    z = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(mat)
    out_rows = []
    for i, r in enumerate(rows):
        out_rows.append({"x": float(z[i, 0]), "y": float(z[i, 1]), "domain": r["domain"], "emotion": r["emotion"]})
    return pd.DataFrame(out_rows)


def _plot_embedding_panel(df: pd.DataFrame, title: str, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    # A: emotion
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="emotion",
        style="emotion",
        s=14,
        alpha=0.6,
        ax=axes[0],
        legend=True,
    )
    axes[0].set_title("A. Colored by Emotion")
    axes[0].set_xlabel("t-SNE-1")
    axes[0].set_ylabel("t-SNE-2")
    # B: domain
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="domain",
        style="domain",
        s=14,
        alpha=0.6,
        ax=axes[1],
        legend=True,
    )
    axes[1].set_title("B. Colored/Marked by Domain")
    axes[1].set_xlabel("t-SNE-1")
    axes[1].set_ylabel("t-SNE-2")
    fig.suptitle(title, y=1.02)
    _save(fig, out_path)
    return out_path


def fig_embedding_v2() -> tuple[Path, Path]:
    d1 = _collect_embeddings(CONFIGS / "iemocap_to_podcast_chunk.yaml", "iemocap_to_podcast_chunk")
    d2 = _collect_embeddings(
        CONFIGS / "iemocap_to_podcast_chunk_domain_utt.yaml", "iemocap_to_podcast_chunk_domain_utt"
    )
    out1 = _plot_embedding_panel(
        d1, "IEMOCAP \u2192 PODCAST: Chunk Baseline Embedding", REPORTS / "fig_embedding_chunk_baseline_v2.png"
    )
    out2 = _plot_embedding_panel(
        d2,
        "IEMOCAP \u2192 PODCAST: Chunk + Domain-Utt Embedding",
        REPORTS / "fig_embedding_chunk_domain_utt_v2.png",
    )
    return out1, out2


def write_captions_and_manifest(generated: list[Path]) -> tuple[Path, Path]:
    caption_lines = [
        "## Paper Figure Captions",
        "",
        "### fig_in_corpus_barplot_v2.png",
        "Grouped bar chart of in-corpus test UAR for IEMOCAP and PODCAST across MeanPool, AttnPool, and Chunk models. The x-axis shows corpus, the y-axis shows UAR, and colors indicate model type.",
        "",
        "### fig_cross_corpus_barplot_v2.png",
        "Grouped bar chart of cross-corpus baseline test UAR across IEMOCAP\u2192PODCAST, PODCAST\u2192IEMOCAP, and IEMOCAP+CREMA-D\u2192PODCAST. The x-axis shows transfer direction, the y-axis shows UAR, and colors indicate pooling model.",
        "",
        "### fig_stage1_stage2_barplot_v2.png",
        "Grouped bar chart comparing chunk-family variants (Chunk baseline, Stage1 Domain-Utt, Stage2 Domain-Both, and weak-chunk Stage2 where available) across transfer directions. The x-axis shows transfer direction and the y-axis shows UAR.",
        "",
        "### fig_classwise_i2p.png",
        "Class-wise recall comparison for IEMOCAP\u2192PODCAST across Chunk baseline, Stage1 Domain-Utt, and Stage2 Domain-Both. Emotion classes are on the x-axis and recall is on the y-axis.",
        "",
        "### fig_classwise_p2i.png",
        "Class-wise recall comparison for PODCAST\u2192IEMOCAP across Chunk baseline, Stage1 Domain-Utt, and Stage2 Domain-Both. Emotion classes are on the x-axis and recall is on the y-axis.",
        "",
        "### fig_classwise_delta_i2p.png",
        "Recall deltas per emotion for IEMOCAP\u2192PODCAST, showing Stage1 minus baseline and Stage2 minus Stage1. Positive values indicate recall gain; negative values indicate recall loss.",
        "",
        "### fig_classwise_delta_p2i.png",
        "Recall deltas per emotion for PODCAST\u2192IEMOCAP, showing Stage1 minus baseline and Stage2 minus Stage1. Positive values indicate recall gain; negative values indicate recall loss.",
        "",
        "### fig_target_fraction_ablation.png",
        "Line plot of IEMOCAP\u2192PODCAST Stage1 performance versus target unlabeled fraction (0.25, 0.50, 0.75, 1.00). Horizontal reference lines mark plain chunk baseline and full-target Stage1.",
        "",
        "### fig_source_fraction_ablation.png",
        "Line plot of IEMOCAP\u2192PODCAST UAR versus source labeled fraction for Chunk and Chunk+Domain-Utt. The x-axis shows source fraction and the y-axis shows UAR.",
        "",
        "### fig_stage2_ablation_i2p.png",
        "Horizontal comparison of IEMOCAP\u2192PODCAST UAR across Stage2 ablation variants, including chunk-only and combined-loss settings, alongside plain chunk and Stage1 references.",
        "",
        "### fig_multiseed_robustness.png",
        "Mean test UAR with standard-deviation error bars for four core models (i2p chunk, i2p chunk_domain_utt, p2i chunk, p2i chunk_domain_utt) across seeds.",
        "",
        "### fig_embedding_chunk_baseline_v2.png",
        "Two-panel t-SNE visualization for IEMOCAP\u2192PODCAST chunk baseline embeddings: Panel A colored by emotion class; Panel B colored/marked by source vs target domain.",
        "",
        "### fig_embedding_chunk_domain_utt_v2.png",
        "Two-panel t-SNE visualization for IEMOCAP\u2192PODCAST chunk_domain_utt embeddings: Panel A colored by emotion class; Panel B colored/marked by source vs target domain.",
    ]
    caption_path = REPORTS / "paper_figure_captions.md"
    caption_path.write_text("\n".join(caption_lines), encoding="utf-8")

    manifest_rows = [
        ("fig_in_corpus_barplot_v2.png", "core", "In-corpus grouped UAR bar chart", "main", "main"),
        ("fig_cross_corpus_barplot_v2.png", "core", "Cross-corpus baseline grouped UAR bar chart", "main", "main"),
        ("fig_stage1_stage2_barplot_v2.png", "core", "Chunk-family Stage1/Stage2 comparison", "main", "main"),
        ("fig_classwise_i2p.png", "classwise", "I2P class-wise recall comparison", "supplementary", "supplementary"),
        ("fig_classwise_p2i.png", "classwise", "P2I class-wise recall comparison", "supplementary", "supplementary"),
        ("fig_classwise_delta_i2p.png", "classwise", "I2P class-wise recall deltas", "supplementary", "supplementary"),
        ("fig_classwise_delta_p2i.png", "classwise", "P2I class-wise recall deltas", "supplementary", "supplementary"),
        ("fig_target_fraction_ablation.png", "ablation", "Target fraction ablation (Stage1 i2p)", "supplementary", "supplementary"),
        ("fig_source_fraction_ablation.png", "ablation", "Source fraction ablation (i2p)", "supplementary", "supplementary"),
        ("fig_stage2_ablation_i2p.png", "ablation", "Stage2 ablation comparison (i2p)", "supplementary", "supplementary"),
        ("fig_multiseed_robustness.png", "robustness", "Multi-seed mean/std UAR", "main", "main"),
        ("fig_embedding_chunk_baseline_v2.png", "embedding", "Embedding visualization (chunk baseline)", "supplementary", "supplementary"),
        ("fig_embedding_chunk_domain_utt_v2.png", "embedding", "Embedding visualization (chunk_domain_utt)", "supplementary", "supplementary"),
    ]
    manifest = pd.DataFrame(
        manifest_rows,
        columns=["figure_file", "figure_group", "description", "main_or_supplementary", "recommended_use"],
    )
    manifest_path = REPORTS / "paper_figures_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return caption_path, manifest_path


def main() -> None:
    _setup_style()
    required = [
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
    ]
    _ensure_inputs(required)

    overview = pd.read_csv(REPORTS / "final_baseline_stage1_stage2_overview.csv")
    class_df = pd.read_csv(REPORTS / "classwise_cross_corpus_summary.csv")
    tgt_df = pd.read_csv(REPORTS / "uda_target_fraction_ablation.csv")
    src_df = pd.read_csv(REPORTS / "source_fraction_ablation.csv")
    s2_df = pd.read_csv(REPORTS / "stage2_ablation_iemocap_to_podcast.csv")
    ms_df = pd.read_csv(REPORTS / "multiseed_core_results.csv")

    generated: list[Path] = []
    generated.append(fig_in_corpus_v2(overview))
    generated.append(fig_cross_corpus_v2(overview))
    generated.append(fig_stage12_v2(overview))
    generated.append(fig_classwise_i2p(class_df))
    generated.append(fig_classwise_p2i(class_df))
    generated.append(fig_classwise_delta_i2p(class_df))
    generated.append(fig_classwise_delta_p2i(class_df))
    generated.append(fig_target_fraction(tgt_df, overview))
    generated.append(fig_source_fraction(src_df))
    generated.append(fig_stage2_ablation_i2p(s2_df))
    generated.append(fig_multiseed(ms_df))
    emb1, emb2 = fig_embedding_v2()
    generated.extend([emb1, emb2])
    cap, man = write_captions_and_manifest(generated)
    generated.extend([cap, man])

    print("Generated files:")
    for p in generated:
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
