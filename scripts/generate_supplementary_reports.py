#!/usr/bin/env python3
"""Generate requested supplementary result reports from existing exp outputs."""

from __future__ import annotations

from pathlib import Path

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
from src.training.utils import load_yaml, safe_torch_load, device_from_config


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXP = ROOT / "exp"
CONFIGS = ROOT / "configs" / "phase2"


def _read_metric(exp_name: str) -> dict:
    row = pd.read_csv(EXP / exp_name / "final_metrics.csv").iloc[0]
    return {
        "experiment": exp_name,
        "test_uar": float(row["test_uar"]),
        "test_wa": float(row["test_wa"]),
        "test_macro_f1": float(row["test_macro_f1"]),
    }


def generate_multiseed_core() -> None:
    core = [
        "iemocap_to_podcast_chunk",
        "iemocap_to_podcast_chunk_domain_utt",
        "podcast_to_iemocap_chunk",
        "podcast_to_iemocap_chunk_domain_utt",
    ]
    rows = []
    for base in core:
        exps = [base, f"{base}_seed7", f"{base}_seed13"]
        vals = [_read_metric(e) for e in exps]
        df = pd.DataFrame(vals)
        rows.append(
            {
                "experiment": base,
                "seeds": "42,7,13",
                "mean_test_uar": df["test_uar"].mean(),
                "std_test_uar": df["test_uar"].std(ddof=1),
                "mean_test_wa": df["test_wa"].mean(),
                "std_test_wa": df["test_wa"].std(ddof=1),
                "mean_test_macro_f1": df["test_macro_f1"].mean(),
                "std_test_macro_f1": df["test_macro_f1"].std(ddof=1),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "multiseed_core_results.csv", index=False)
    out_r = out.copy()
    for c in out_r.columns:
        if c.startswith("mean_") or c.startswith("std_"):
            out_r[c] = out_r[c].round(3)
    out_r.to_csv(REPORTS / "multiseed_core_results_rounded.csv", index=False)

    md = [
        "## Multi-seed Core Results",
        "",
        "Mean and standard deviation across seeds {42, 7, 13}.",
        "",
    ]
    for _, r in out.iterrows():
        md.append(
            f"- `{r['experiment']}`: UAR {r['mean_test_uar']:.3f}±{r['std_test_uar']:.3f}, "
            f"WA {r['mean_test_wa']:.3f}±{r['std_test_wa']:.3f}, "
            f"Macro-F1 {r['mean_test_macro_f1']:.3f}±{r['std_test_macro_f1']:.3f}"
        )
    (REPORTS / "multiseed_core_summary.md").write_text("\n".join(md), encoding="utf-8")


def _classwise_recall_from_confusion(csv_path: Path) -> pd.DataFrame:
    cm = pd.read_csv(csv_path, header=None).values.astype(float)
    rs = cm.sum(axis=1, keepdims=True)
    recall = np.divide(cm.diagonal(), np.clip(rs.squeeze(), 1e-12, None))
    return pd.DataFrame(
        {
            "class_id": list(range(len(recall))),
            "emotion": [ID_TO_EMOTION[i] for i in range(len(recall))],
            "recall": recall,
        }
    )


def generate_classwise_cross_summary() -> None:
    runs = {
        "i2p_chunk": "iemocap_to_podcast_chunk",
        "i2p_s1": "iemocap_to_podcast_chunk_domain_utt",
        "i2p_s2": "iemocap_to_podcast_chunk_domain_both",
        "p2i_chunk": "podcast_to_iemocap_chunk",
        "p2i_s1": "podcast_to_iemocap_chunk_domain_utt",
        "p2i_s2": "podcast_to_iemocap_chunk_domain_both",
        "iemocap_in": "iemocap_in_corpus_chunk",
    }
    rec = {}
    for k, exp_name in runs.items():
        rec[k] = _classwise_recall_from_confusion(EXP / exp_name / "confusion_test.csv")

    out = rec["i2p_chunk"][["emotion", "recall"]].rename(columns={"recall": "i2p_chunk_recall"})
    out = out.merge(rec["i2p_s1"][["emotion", "recall"]].rename(columns={"recall": "i2p_stage1_recall"}), on="emotion")
    out = out.merge(rec["i2p_s2"][["emotion", "recall"]].rename(columns={"recall": "i2p_stage2_recall"}), on="emotion")
    out = out.merge(rec["p2i_chunk"][["emotion", "recall"]].rename(columns={"recall": "p2i_chunk_recall"}), on="emotion")
    out = out.merge(rec["p2i_s1"][["emotion", "recall"]].rename(columns={"recall": "p2i_stage1_recall"}), on="emotion")
    out = out.merge(rec["p2i_s2"][["emotion", "recall"]].rename(columns={"recall": "p2i_stage2_recall"}), on="emotion")
    out = out.merge(rec["iemocap_in"][["emotion", "recall"]].rename(columns={"recall": "iemocap_in_corpus_recall"}), on="emotion")
    out["degrade_acted_to_natural_vs_in_corpus"] = out["i2p_chunk_recall"] - out["iemocap_in_corpus_recall"]
    out["improve_stage1_i2p"] = out["i2p_stage1_recall"] - out["i2p_chunk_recall"]
    out["hurt_stage2_vs_stage1_i2p"] = out["i2p_stage2_recall"] - out["i2p_stage1_recall"]
    out["hurt_stage2_vs_stage1_p2i"] = out["p2i_stage2_recall"] - out["p2i_stage1_recall"]
    out.to_csv(REPORTS / "classwise_cross_corpus_summary.csv", index=False)

    most_degraded = out.sort_values("degrade_acted_to_natural_vs_in_corpus").iloc[0]
    improved = out.sort_values("improve_stage1_i2p", ascending=False).iloc[0]
    hurt_i2p = out.sort_values("hurt_stage2_vs_stage1_i2p").iloc[0]
    hurt_p2i = out.sort_values("hurt_stage2_vs_stage1_p2i").iloc[0]
    md = [
        "## Class-wise Cross-Corpus Summary",
        "",
        f"- Most degraded emotion in acted->natural (vs iemocap in-corpus chunk): **{most_degraded['emotion']}** ({most_degraded['degrade_acted_to_natural_vs_in_corpus']:.3f}).",
        f"- Largest class gain under Stage 1 (i2p): **{improved['emotion']}** ({improved['improve_stage1_i2p']:.3f}).",
        f"- Most hurt class by Stage 2 vs Stage 1 (i2p): **{hurt_i2p['emotion']}** ({hurt_i2p['hurt_stage2_vs_stage1_i2p']:.3f}).",
        f"- Most hurt class by Stage 2 vs Stage 1 (p2i): **{hurt_p2i['emotion']}** ({hurt_p2i['hurt_stage2_vs_stage1_p2i']:.3f}).",
    ]
    (REPORTS / "classwise_cross_corpus_summary.md").write_text("\n".join(md), encoding="utf-8")


def generate_fraction_ablations() -> None:
    # target unlabeled fractions (domain_utt i2p)
    tgt_rows = []
    exp_map = {
        0.25: "iemocap_to_podcast_chunk_domain_utt_tgtfrac25",
        0.50: "iemocap_to_podcast_chunk_domain_utt_tgtfrac50",
        0.75: "iemocap_to_podcast_chunk_domain_utt_tgtfrac75",
        1.00: "iemocap_to_podcast_chunk_domain_utt",
    }
    base = _read_metric("iemocap_to_podcast_chunk")
    for frac, exp_name in exp_map.items():
        m = _read_metric(exp_name)
        tgt_rows.append(
            {
                "target_fraction": frac,
                "experiment": exp_name,
                "test_uar": m["test_uar"],
                "test_wa": m["test_wa"],
                "test_macro_f1": m["test_macro_f1"],
                "gain_vs_plain_chunk": m["test_uar"] - base["test_uar"],
            }
        )
    tgt_df = pd.DataFrame(tgt_rows).sort_values("target_fraction")
    tgt_df.to_csv(REPORTS / "uda_target_fraction_ablation.csv", index=False)
    best_tgt = tgt_df.loc[tgt_df["test_uar"].idxmax()]
    (REPORTS / "uda_target_fraction_ablation.md").write_text(
        "\n".join(
            [
                "## UDA Target Fraction Ablation (i2p chunk_domain_utt)",
                "",
                f"- Best target-unlabeled fraction by UAR: **{best_tgt['target_fraction']:.2f}** "
                f"({best_tgt['test_uar']:.3f}).",
                "- Full table is in `reports/uda_target_fraction_ablation.csv`.",
            ]
        ),
        encoding="utf-8",
    )

    src_rows = []
    exp_map_src = [
        ("chunk", 0.25, "iemocap_to_podcast_chunk_srcfrac25"),
        ("chunk", 0.50, "iemocap_to_podcast_chunk_srcfrac50"),
        ("chunk", 1.00, "iemocap_to_podcast_chunk"),
        ("chunk_domain_utt", 0.25, "iemocap_to_podcast_chunk_domain_utt_srcfrac25"),
        ("chunk_domain_utt", 0.50, "iemocap_to_podcast_chunk_domain_utt_srcfrac50"),
        ("chunk_domain_utt", 1.00, "iemocap_to_podcast_chunk_domain_utt"),
    ]
    for variant, frac, exp_name in exp_map_src:
        m = _read_metric(exp_name)
        src_rows.append(
            {
                "model_variant": variant,
                "source_fraction": frac,
                "experiment": exp_name,
                "test_uar": m["test_uar"],
                "test_wa": m["test_wa"],
                "test_macro_f1": m["test_macro_f1"],
            }
        )
    src_df = pd.DataFrame(src_rows).sort_values(["model_variant", "source_fraction"])
    src_df.to_csv(REPORTS / "source_fraction_ablation.csv", index=False)
    (REPORTS / "source_fraction_ablation.md").write_text(
        "\n".join(
            [
                "## Source Fraction Ablation (i2p)",
                "",
                "- Compares plain chunk and chunk_domain_utt at source labeled fractions 0.25 / 0.50 / 1.00.",
                "- See `reports/source_fraction_ablation.csv` for full metrics.",
            ]
        ),
        encoding="utf-8",
    )


@torch.no_grad()
def _collect_embeddings(config_path: Path, exp_name: str, max_samples_per_domain: int = 800) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    device = device_from_config(cfg)
    ckpt = safe_torch_load(EXP / exp_name / "best.ckpt", map_location=device)
    saved_cfg = ckpt.get("config")
    if isinstance(saved_cfg, dict) and isinstance(saved_cfg.get("model"), dict):
        cfg["model"] = {**cfg.get("model", {}), **saved_cfg["model"]}
    model = build_model(cfg).to(device)
    # Backward compatibility: older Stage1 checkpoints used `domain_classifier.*`
    # while current code uses `domain_classifier_utt.*`/`domain_classifier_chunk.*`.
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
            y = batch["labels"].numpy()
            for i in range(emb.shape[0]):
                rows.append(
                    {
                        "domain": domain_name,
                        "emotion": ID_TO_EMOTION[int(y[i])],
                        "embedding": emb[i],
                    }
                )
                count += 1
                if count >= max_samples_per_domain:
                    break
            if count >= max_samples_per_domain:
                break
    # flatten
    mat = np.stack([r["embedding"] for r in rows], axis=0)
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    z = tsne.fit_transform(mat)
    out_rows = []
    for i, r in enumerate(rows):
        out_rows.append(
            {
                "x": float(z[i, 0]),
                "y": float(z[i, 1]),
                "domain": r["domain"],
                "emotion": r["emotion"],
            }
        )
    return pd.DataFrame(out_rows)


def generate_embedding_figures() -> None:
    sns.set_theme(style="whitegrid")
    specs = [
        (
            CONFIGS / "iemocap_to_podcast_chunk.yaml",
            "iemocap_to_podcast_chunk",
            REPORTS / "fig_embedding_chunk_baseline.png",
            "Chunk Baseline Embedding (t-SNE)",
        ),
        (
            CONFIGS / "iemocap_to_podcast_chunk_domain_utt.yaml",
            "iemocap_to_podcast_chunk_domain_utt",
            REPORTS / "fig_embedding_chunk_domain_utt.png",
            "Chunk + Domain-Utt Embedding (t-SNE)",
        ),
    ]
    md = ["## Embedding Visualization Summary", ""]
    for cfg_path, exp_name, out_png, title in specs:
        df = _collect_embeddings(cfg_path, exp_name)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.scatterplot(data=df, x="x", y="y", hue="emotion", style="emotion", s=16, ax=axes[0], legend=True)
        axes[0].set_title(f"{title} - Color by Emotion")
        sns.scatterplot(data=df, x="x", y="y", hue="domain", style="domain", s=16, ax=axes[1], legend=True)
        axes[1].set_title(f"{title} - Mark Domain")
        for ax in axes:
            ax.set_xlabel("t-SNE-1")
            ax.set_ylabel("t-SNE-2")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        md.append(f"- Generated `{out_png.relative_to(ROOT)}` from `{exp_name}`.")
    (REPORTS / "embedding_visualization_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    generate_multiseed_core()
    generate_classwise_cross_summary()
    generate_fraction_ablations()
    generate_embedding_figures()
    print("Supplementary reports generated.")


if __name__ == "__main__":
    main()
