#!/usr/bin/env python3
"""Generate paper-ready tabular artifacts from existing manifests and results only."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from src.data.datasets import EMOTION_TO_ID, load_manifest
from src.training.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXP = ROOT / "exp"
CONFIGS = ROOT / "configs" / "phase2"
DATA = ROOT / "data"
EMOTIONS = list(EMOTION_TO_ID.keys())
CORE_MULTI_SEED = [
    "iemocap_to_podcast_chunk",
    "iemocap_to_podcast_chunk_domain_utt",
    "podcast_to_iemocap_chunk",
    "podcast_to_iemocap_chunk_domain_utt",
]


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(3)
    return out


def _style_label(corpus: str) -> str:
    return {
        "iemocap": "acted",
        "crema_d": "acted",
        "podcast": "naturalistic",
        "iemocap+crema_d": "multi-source",
    }.get(corpus, "UNKNOWN")


def _duration_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    total_col = None
    avg_col = None
    dur_col = "processed_duration_sec" if "processed_duration_sec" in df.columns else (
        "duration_sec" if "duration_sec" in df.columns else None
    )
    if dur_col is not None:
        total_col = dur_col
        avg_col = dur_col
    return total_col, avg_col


def _basic_stats(df: pd.DataFrame, corpus: str, split_name: str | None = None) -> dict:
    row = {
        "corpus": corpus,
        "style_label": _style_label(corpus),
        "num_utterances_after_filtering": int(len(df)),
        "num_speakers": int(df["speaker_id"].nunique()) if "speaker_id" in df.columns else None,
    }
    for emo in EMOTIONS:
        row[f"count_{emo}"] = int((df["emotion"] == emo).sum()) if "emotion" in df.columns else None
    total_col, avg_col = _duration_cols(df)
    if total_col is not None:
        row["total_duration_sec"] = float(df[total_col].sum())
        row["avg_utterance_duration_sec"] = float(df[avg_col].mean())
    if split_name is not None:
        row["split"] = split_name
    return row


def generate_dataset_statistics() -> list[Path]:
    meta = load_manifest(DATA / "manifests" / "all_metadata_processed.csv")
    by_corpus_rows = []
    for corpus in ["iemocap", "podcast", "crema_d"]:
        sub = meta[meta["dataset"] == corpus].copy()
        by_corpus_rows.append(_basic_stats(sub, corpus))
    # Explicit multi-source corpus aggregate used in paper.
    ms = meta[meta["dataset"].isin(["iemocap", "crema_d"])].copy()
    by_corpus_rows.append(_basic_stats(ms, "iemocap+crema_d"))
    main_df = pd.DataFrame(by_corpus_rows)

    split_rows = []
    split_patterns = [
        ("iemocap", DATA / "splits" / "protocol_a_in_corpus" / "iemocap_train.csv", "train"),
        ("iemocap", DATA / "splits" / "protocol_a_in_corpus" / "iemocap_dev.csv", "validation"),
        ("iemocap", DATA / "splits" / "protocol_a_in_corpus" / "iemocap_test.csv", "test"),
        ("podcast", DATA / "splits" / "protocol_a_in_corpus" / "podcast_train.csv", "train"),
        ("podcast", DATA / "splits" / "protocol_a_in_corpus" / "podcast_dev.csv", "validation"),
        ("podcast", DATA / "splits" / "protocol_a_in_corpus" / "podcast_test.csv", "test"),
        ("crema_d", DATA / "splits" / "protocol_a_in_corpus" / "crema_d_train.csv", "train"),
        ("crema_d", DATA / "splits" / "protocol_a_in_corpus" / "crema_d_dev.csv", "validation"),
        ("crema_d", DATA / "splits" / "protocol_a_in_corpus" / "crema_d_test.csv", "test"),
        ("iemocap+crema_d", DATA / "splits" / "protocol_c_multi_source" / "iemocap_plus_crema_d_to_podcast_train.csv", "train"),
        ("iemocap+crema_d", DATA / "splits" / "protocol_c_multi_source" / "iemocap_plus_crema_d_to_podcast_test.csv", "test"),
    ]
    for corpus, path, split_name in split_patterns:
        if not path.exists():
            continue
        df = load_manifest(path)
        split_rows.append(_basic_stats(df, corpus, split_name))
    split_df = pd.DataFrame(split_rows)

    # merge split counts into main_df where available
    counts = split_df.pivot_table(index="corpus", columns="split", values="num_utterances_after_filtering", aggfunc="sum")
    counts.columns = [f"{c}_count" for c in counts.columns]
    main_df = main_df.merge(counts.reset_index(), on="corpus", how="left")

    main_path = REPORTS / "paper_dataset_statistics.csv"
    rounded_path = REPORTS / "paper_dataset_statistics_rounded.csv"
    by_split_path = REPORTS / "paper_dataset_statistics_by_split.csv"
    main_df.to_csv(main_path, index=False)
    _round_df(main_df).to_csv(rounded_path, index=False)
    split_df.to_csv(by_split_path, index=False)

    have_duration = "total_duration_sec" in main_df.columns and main_df["total_duration_sec"].notna().any()
    lines = [
        "## Paper Dataset Statistics Summary",
        "",
        "Artifacts are computed from harmonized processed metadata and split CSVs only.",
        f"Duration fields {'were' if have_duration else 'were not'} available/computable from metadata.",
    ]
    if not have_duration:
        lines.append("Duration columns were omitted because no reliable duration field was available.")
    lines.extend([
        "",
        "Included corpora/statistical groups:",
        "- `iemocap` (acted)",
        "- `podcast` (naturalistic)",
        "- `crema_d` (acted)",
        "- `iemocap+crema_d` (multi-source aggregate used for source-side reporting)",
    ])
    summary_path = REPORTS / "paper_dataset_statistics_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return [main_path, rounded_path, by_split_path, summary_path]


def _load_result_row(exp_name: str) -> dict:
    final_path = EXP / exp_name / "final_metrics.csv"
    summary_path = EXP / exp_name / "summary.csv"
    if final_path.exists():
        df = pd.read_csv(final_path)
    elif summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        raise FileNotFoundError(f"Missing result file for {exp_name}")
    row = df.iloc[0].to_dict()
    row["experiment"] = exp_name
    return row


def generate_in_corpus_tables() -> list[Path]:
    overview = pd.read_csv(REPORTS / "final_baseline_stage1_stage2_overview.csv")
    in_corpus = overview[overview["training_setup"] == "in_corpus"].copy()
    rows = []
    for exp_name in in_corpus["experiment"].tolist():
        row = _load_result_row(exp_name)
        cfg = load_yaml(CONFIGS / f"{exp_name}.yaml")
        rows.append(
            {
                "experiment": exp_name,
                "corpus": in_corpus.loc[in_corpus["experiment"] == exp_name, "target_domain"].iloc[0],
                "pooling_type": in_corpus.loc[in_corpus["experiment"] == exp_name, "model_variant"].iloc[0],
                "test_uar": row.get("test_uar"),
                "test_wa": row.get("test_wa"),
                "test_macro_f1": row.get("test_macro_f1"),
                "best_dev_uar": row.get("best_dev_uar"),
                "num_test_samples": row.get("test_num_samples"),
                "seed": cfg.get("seed", "UNKNOWN"),
            }
        )
    df = pd.DataFrame(rows).sort_values(["corpus", "pooling_type"]).reset_index(drop=True)
    best = df.loc[df.groupby("corpus")["test_uar"].idxmax()].sort_values("corpus").reset_index(drop=True)
    p1 = REPORTS / "paper_in_corpus_full_metrics.csv"
    p2 = REPORTS / "paper_in_corpus_full_metrics_rounded.csv"
    p3 = REPORTS / "paper_in_corpus_best_by_corpus.csv"
    df.to_csv(p1, index=False)
    _round_df(df).to_csv(p2, index=False)
    _round_df(best).to_csv(p3, index=False)
    return [p1, p2, p3]


def _seed_candidates(base: str) -> list[tuple[str, int]]:
    return [(base, 42), (f"{base}_seed7", 7), (f"{base}_seed13", 13)]


def generate_multiseed_tables() -> list[Path]:
    rows = []
    deltas = []
    missing_lines = ["## Paper Multiseed Summary", "", "Missing or incomplete seed runs:"]
    for exp_name in CORE_MULTI_SEED:
        found_runs = []
        found_seeds = []
        missing_seeds = []
        for candidate, seed in _seed_candidates(exp_name):
            path = EXP / candidate / "final_metrics.csv"
            if path.exists():
                df = pd.read_csv(path)
                row = df.iloc[0].to_dict()
                row["resolved_experiment"] = candidate
                row["seed"] = seed
                found_runs.append(row)
                found_seeds.append(seed)
            else:
                missing_seeds.append(seed)
        if not found_runs:
            missing_lines.append(f"- `{exp_name}`: no seed runs found")
            continue
        rdf = pd.DataFrame(found_runs)
        rows.append(
            {
                "experiment": exp_name,
                "mean_uar": float(rdf["test_uar"].mean()),
                "std_uar": float(rdf["test_uar"].std(ddof=0)),
                "mean_wa": float(rdf["test_wa"].mean()),
                "std_wa": float(rdf["test_wa"].std(ddof=0)),
                "mean_macro_f1": float(rdf["test_macro_f1"].mean()),
                "std_macro_f1": float(rdf["test_macro_f1"].std(ddof=0)),
                "seed_list_found": ",".join(str(s) for s in found_seeds),
                "num_successful_runs_found": int(len(found_runs)),
            }
        )
        if missing_seeds:
            missing_lines.append(f"- `{exp_name}` missing seeds: {', '.join(str(s) for s in missing_seeds)}")
    summary_df = pd.DataFrame(rows)

    # pairwise deltas within each direction
    pairs = [
        ("iemocap_to_podcast_chunk_domain_utt", "iemocap_to_podcast_chunk"),
        ("podcast_to_iemocap_chunk_domain_utt", "podcast_to_iemocap_chunk"),
    ]
    for a, b in pairs:
        if a in summary_df["experiment"].values and b in summary_df["experiment"].values:
            ra = summary_df[summary_df["experiment"] == a].iloc[0]
            rb = summary_df[summary_df["experiment"] == b].iloc[0]
            deltas.append(
                {
                    "comparison": f"{a} - {b}",
                    "delta_mean_uar": float(ra["mean_uar"] - rb["mean_uar"]),
                    "delta_mean_wa": float(ra["mean_wa"] - rb["mean_wa"]),
                    "delta_mean_macro_f1": float(ra["mean_macro_f1"] - rb["mean_macro_f1"]),
                }
            )
    delta_df = pd.DataFrame(deltas)

    p1 = REPORTS / "paper_multiseed_summary.csv"
    p2 = REPORTS / "paper_multiseed_summary_rounded.csv"
    p3 = REPORTS / "paper_multiseed_pairwise_deltas.csv"
    p4 = REPORTS / "paper_multiseed_summary.md"
    summary_df.to_csv(p1, index=False)
    _round_df(summary_df).to_csv(p2, index=False)
    _round_df(delta_df).to_csv(p3, index=False)
    if len(missing_lines) == 3:
        missing_lines.append("- none")
    p4.write_text("\n".join(missing_lines), encoding="utf-8")
    return [p1, p2, p3, p4]


def generate_stage2_ablation_tables() -> list[Path]:
    s2 = pd.read_csv(REPORTS / "stage2_ablation_iemocap_to_podcast.csv").copy()
    s2["source_domain"] = "iemocap"
    s2["target_domain"] = "podcast"
    s2["gain_vs_plain_chunk"] = s2["absolute_gain_vs_plain_chunk"]
    s2["gain_vs_stage1_domain_utt"] = s2["absolute_gain_vs_chunk_domain_utt"]
    s2["rank_by_uar"] = s2["test_uar"].rank(ascending=False, method="min").astype(int)
    cols = [
        "experiment",
        "source_domain",
        "target_domain",
        "utterance_domain_loss_weight",
        "chunk_domain_loss_weight",
        "test_uar",
        "test_wa",
        "test_macro_f1",
        "gain_vs_plain_chunk",
        "gain_vs_stage1_domain_utt",
        "rank_by_uar",
    ]
    out = s2[cols].sort_values(["rank_by_uar", "experiment"]).reset_index(drop=True)
    best = out.nsmallest(1, "rank_by_uar")
    summary_lines = [
        "## Paper Stage 2 Ablation Summary",
        "",
        f"Best setting by test UAR: `{best.iloc[0]['experiment']}` with UAR={best.iloc[0]['test_uar']:.3f}.",
        "All rows are derived from existing Stage 2 ablation results only.",
    ]
    p1 = REPORTS / "paper_stage2_ablation_full.csv"
    p2 = REPORTS / "paper_stage2_ablation_full_rounded.csv"
    p3 = REPORTS / "paper_stage2_ablation_best_setting.csv"
    p4 = REPORTS / "paper_stage2_ablation_summary.md"
    out.to_csv(p1, index=False)
    _round_df(out).to_csv(p2, index=False)
    _round_df(best).to_csv(p3, index=False)
    p4.write_text("\n".join(summary_lines), encoding="utf-8")
    return [p1, p2, p3, p4]


def main() -> None:
    outputs = []
    outputs.extend(generate_dataset_statistics())
    outputs.extend(generate_in_corpus_tables())
    outputs.extend(generate_multiseed_tables())
    outputs.extend(generate_stage2_ablation_tables())
    print("Generated:")
    for path in outputs:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

