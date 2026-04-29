"""Emotion normalization utilities for 4-class SER experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Canonical 4-class mapping for main experiments.
EMOTION_MAP = {
    "angry": "angry",
    "anger": "angry",
    "happy": "happy",
    "happiness": "happy",
    "excited": "happy",
    "sad": "sad",
    "sadness": "sad",
    "neutral": "neutral",
}

MAIN_EMOTIONS = {"angry", "happy", "sad", "neutral"}


def _clean_label(label: str) -> str:
    return str(label).strip().lower().replace("-", " ").replace("_", " ")


def normalize_emotion(label: str) -> Optional[str]:
    """Map a raw label into one of 4 target emotions, else return None."""
    if label is None:
        return None
    clean = _clean_label(label)
    if not clean:
        return None
    return EMOTION_MAP.get(clean)


def is_valid_main_emotion(label: str) -> bool:
    """Check whether a raw label can be mapped to the 4 target classes."""
    return normalize_emotion(label) in MAIN_EMOTIONS


@dataclass
class LabelMappingSummary:
    dataset: str
    total_samples: int
    dropped_samples: int
    raw_counts: Counter
    mapped_counts: Counter


def summarize_label_mapping(
    dataset_name: str, raw_labels: Iterable[str]
) -> LabelMappingSummary:
    """Build a label mapping summary from an iterable of raw labels."""
    raw_counter: Counter = Counter()
    mapped_counter: Counter = Counter()
    dropped = 0
    total = 0
    for label in raw_labels:
        total += 1
        clean = _clean_label(label)
        raw_counter[clean] += 1
        mapped = normalize_emotion(clean)
        if mapped is None:
            dropped += 1
        else:
            mapped_counter[mapped] += 1
    return LabelMappingSummary(
        dataset=dataset_name,
        total_samples=total,
        dropped_samples=dropped,
        raw_counts=raw_counter,
        mapped_counts=mapped_counter,
    )


def save_label_mapping_reports(
    summaries: list[LabelMappingSummary], csv_path: Path, txt_path: Path
) -> None:
    """Write label mapping summaries to CSV and TXT reports."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = ["Label Mapping Summary\n", "=" * 60]
    for summary in summaries:
        lines.append(f"\nDataset: {summary.dataset}")
        lines.append(f"Total samples: {summary.total_samples}")
        lines.append(f"Dropped (unmapped): {summary.dropped_samples}")
        lines.append("Mapped distribution:")
        for emo in sorted(MAIN_EMOTIONS):
            lines.append(f"  - {emo}: {summary.mapped_counts.get(emo, 0)}")
        lines.append("Top raw labels:")
        for label, cnt in summary.raw_counts.most_common(20):
            lines.append(f"  - {label}: {cnt}")

        for label, cnt in summary.raw_counts.items():
            rows.append(
                {
                    "dataset": summary.dataset,
                    "raw_label": label,
                    "raw_count": cnt,
                    "mapped_label": normalize_emotion(label),
                    "mapped_count": summary.mapped_counts.get(
                        normalize_emotion(label), 0
                    )
                    if normalize_emotion(label) is not None
                    else 0,
                    "is_dropped": normalize_emotion(label) is None,
                }
            )

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    txt_path.write_text("\n".join(lines), encoding="utf-8")
