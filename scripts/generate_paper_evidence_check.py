#!/usr/bin/env python3
"""Verify that major paper-cited experiments have usable result artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXP = ROOT / "exp"


EXPECTED_GROUPS = {
    "in_corpus_runs": [
        "iemocap_in_corpus_meanpool",
        "iemocap_in_corpus_attnpool",
        "iemocap_in_corpus_chunk",
        "podcast_in_corpus_meanpool",
        "podcast_in_corpus_attnpool",
        "podcast_in_corpus_chunk",
    ],
    "cross_corpus_baselines": [
        "iemocap_to_podcast_meanpool",
        "iemocap_to_podcast_attnpool",
        "iemocap_to_podcast_chunk",
        "podcast_to_iemocap_meanpool",
        "podcast_to_iemocap_attnpool",
        "podcast_to_iemocap_chunk",
        "iemocap_plus_cremad_to_podcast_meanpool",
        "iemocap_plus_cremad_to_podcast_attnpool",
        "iemocap_plus_cremad_to_podcast_chunk",
    ],
    "stage1_runs": [
        "iemocap_to_podcast_chunk_domain_utt",
        "podcast_to_iemocap_chunk_domain_utt",
        "iemocap_plus_cremad_to_podcast_chunk_domain_utt",
    ],
    "stage2_default_runs": [
        "iemocap_to_podcast_chunk_domain_both",
        "podcast_to_iemocap_chunk_domain_both",
        "iemocap_plus_cremad_to_podcast_chunk_domain_both",
    ],
    "stage2_ablation_runs": [
        "iemocap_to_podcast_chunk_domain_both_chunk_only_w01",
        "iemocap_to_podcast_chunk_domain_both_chunk_only_w03",
        "iemocap_to_podcast_chunk_domain_both_chunk_only_w05",
        "iemocap_to_podcast_chunk_domain_both_both_weakchunk",
        "iemocap_to_podcast_chunk_domain_both_both_midchunk",
    ],
    "multi_seed_runs": [
        "iemocap_to_podcast_chunk",
        "iemocap_to_podcast_chunk_seed7",
        "iemocap_to_podcast_chunk_seed13",
        "iemocap_to_podcast_chunk_domain_utt",
        "iemocap_to_podcast_chunk_domain_utt_seed7",
        "iemocap_to_podcast_chunk_domain_utt_seed13",
        "podcast_to_iemocap_chunk",
        "podcast_to_iemocap_chunk_seed7",
        "podcast_to_iemocap_chunk_seed13",
        "podcast_to_iemocap_chunk_domain_utt",
        "podcast_to_iemocap_chunk_domain_utt_seed7",
        "podcast_to_iemocap_chunk_domain_utt_seed13",
    ],
    "fraction_ablations": [
        "iemocap_to_podcast_chunk_srcfrac25",
        "iemocap_to_podcast_chunk_srcfrac50",
        "iemocap_to_podcast_chunk_domain_utt_srcfrac25",
        "iemocap_to_podcast_chunk_domain_utt_srcfrac50",
        "iemocap_to_podcast_chunk_domain_utt_tgtfrac25",
        "iemocap_to_podcast_chunk_domain_utt_tgtfrac50",
        "iemocap_to_podcast_chunk_domain_utt_tgtfrac75",
    ],
}


def _check_one(exp_name: str, group: str) -> dict:
    exp_dir = EXP / exp_name
    summary_exists = (exp_dir / "summary.csv").exists()
    final_exists = (exp_dir / "final_metrics.csv").exists()
    analysis_exists = (exp_dir / "analysis").exists()
    found = exp_dir.exists()
    usable = bool(found and (summary_exists or final_exists))
    return {
        "experiment": exp_name,
        "group": group,
        "found": found,
        "summary_csv_exists": summary_exists,
        "final_metrics_csv_exists": final_exists,
        "analysis_folder_exists": analysis_exists,
        "usable_for_paper_table": "yes" if usable else "no",
    }


def main() -> None:
    rows = []
    for group, names in EXPECTED_GROUPS.items():
        for name in names:
            rows.append(_check_one(name, group))
    df = pd.DataFrame(rows).sort_values(["group", "experiment"]).reset_index(drop=True)
    csv_path = REPORTS / "paper_evidence_check.csv"
    df.to_csv(csv_path, index=False)

    missing = df[df["usable_for_paper_table"] == "no"]
    lines = [
        "# Paper Evidence Check",
        "",
        "This report verifies whether expected experiment result artifacts are present for post-hoc paper aggregation.",
        "",
        f"Total expected runs checked: {len(df)}",
        f"Usable runs found: {int((df['usable_for_paper_table'] == 'yes').sum())}",
        f"Missing / unusable runs: {len(missing)}",
        "",
        "## Missing or unusable runs",
    ]
    if missing.empty:
        lines.append("- none")
    else:
        for _, r in missing.iterrows():
            lines.append(
                f"- `{r['experiment']}` ({r['group']}): "
                f"summary={r['summary_csv_exists']}, final_metrics={r['final_metrics_csv_exists']}, analysis={r['analysis_folder_exists']}"
            )
    md_path = REPORTS / "paper_evidence_check.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("Generated:")
    print(f"- {csv_path.relative_to(ROOT)}")
    print(f"- {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

