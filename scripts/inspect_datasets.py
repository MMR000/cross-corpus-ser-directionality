#!/usr/bin/env python3
"""Inspect Hugging Face SER datasets and save schema report."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from pprint import pformat
from typing import Any

from src.data.manifest_utils import (
    DATASET_SPECS,
    detect_audio_candidates,
    detect_label_candidates,
    infer_schema,
    list_split_names,
    load_hf_dataset_safe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("inspect_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dataset schemas and example samples.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/dataset_schema_report.txt"),
        help="Path to schema report text file.",
    )
    parser.add_argument(
        "--max-sample-chars",
        type=int,
        default=2000,
        help="Maximum characters to print per sample preview.",
    )
    return parser.parse_args()


def sample_preview(sample: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(sample, default=str, indent=2)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text


def main() -> None:
    args = parse_args()
    report_lines: list[str] = ["Dataset Schema Inspection Report", "=" * 80]

    for spec in DATASET_SPECS:
        report_lines.append(f"\nDataset: {spec.hf_name} ({spec.short_name})")
        ds = load_hf_dataset_safe(spec.hf_name)
        if ds is None:
            msg = f"WARNING: Could not load {spec.hf_name}. Skipping."
            LOGGER.warning(msg)
            report_lines.append(msg)
            continue

        split_names = list_split_names(ds)
        report_lines.append(f"Splits: {split_names}")
        LOGGER.info("%s splits: %s", spec.hf_name, split_names)

        for split in split_names:
            split_ds = ds[split]
            report_lines.append(f"\n  Split: {split} | Num rows: {len(split_ds)}")
            columns = list(split_ds.column_names)
            report_lines.append(f"  Columns: {columns}")

            if len(split_ds) == 0:
                report_lines.append("  Empty split; no sample preview.")
                continue

            sample = split_ds[0]
            schema = infer_schema(sample, columns)
            audio_candidates = detect_audio_candidates(columns)
            label_candidates = detect_label_candidates(columns)

            report_lines.append(f"  Inferred schema: {pformat(schema)}")
            report_lines.append(f"  Audio-like columns by name: {audio_candidates}")
            report_lines.append(f"  Label-like columns by name: {label_candidates}")
            report_lines.append("  Sample preview:")
            report_lines.append(sample_preview(sample, args.max_sample_chars))

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report_lines), encoding="utf-8")
    LOGGER.info("Schema report saved to %s", args.report_path)
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
