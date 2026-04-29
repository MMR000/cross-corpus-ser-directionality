#!/usr/bin/env python3
"""Prepare datasets: normalize labels, export raw audio, and build manifests."""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.data.emotion_mapping import save_label_mapping_reports, summarize_label_mapping
from src.data.emotion_mapping import normalize_emotion
from src.data.manifest_utils import (
    DATASET_SPECS,
    choose_split_mode,
    compute_duration_seconds,
    ensure_parent,
    extract_audio_sample,
    infer_speaker_id,
    infer_schema,
    load_hf_dataset_safe,
    sample_id_from_fields,
    safe_string,
    write_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("prepare_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SER datasets and manifests.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Root directory of ser_project.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap for debugging. If unset, process full split.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help=(
            "Optional cap per dataset after label mapping. "
            "When set without --max-samples-per-class, a balanced per-class cap is derived."
        ),
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional cap per dataset per mapped class for stratified subsampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite exported wavs if they already exist.",
    )
    return parser.parse_args()


def export_audio(audio_obj: Any, output_path: Path, overwrite: bool) -> tuple[bool, int, float]:
    """Export audio object to wav and return (ok, sampling_rate, duration_sec)."""
    if output_path.exists() and not overwrite:
        try:
            info = sf.info(output_path)
            return True, int(info.samplerate), float(info.duration)
        except Exception:
            pass

    try:
        array, sr = extract_audio_sample(audio_obj)
    except Exception:
        return False, 0, 0.0

    ensure_parent(output_path)
    sf.write(output_path, array, sr)
    duration = compute_duration_seconds(len(array), sr)
    return True, sr, duration


def pick_value(sample: dict[str, Any], key: Optional[str]) -> str:
    if key is None:
        return ""
    return safe_string(sample.get(key))


def main() -> None:
    args = parse_args()
    root = args.project_root
    data_root = root / "data"
    manifests_dir = data_root / "manifests"
    raw_dir = data_root / "raw"
    reports_dir = root / "reports"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    label_summary_inputs: dict[str, list[str]] = defaultdict(list)

    missing_audio_field = 0
    audio_decode_failure = 0
    unmapped_label = 0
    subsampling_drop = 0
    valid_samples = 0
    all_durations: list[float] = []
    dataset_kept_counts: Counter = Counter()
    dataset_class_counts: dict[str, Counter] = defaultdict(Counter)

    # If only dataset-level cap is given, derive balanced per-class cap.
    derived_per_class_cap: Optional[int] = None
    if args.max_samples_per_dataset is not None and args.max_samples_per_class is None:
        derived_per_class_cap = int(np.ceil(args.max_samples_per_dataset / 4))
    per_class_cap = args.max_samples_per_class if args.max_samples_per_class is not None else derived_per_class_cap

    for spec in DATASET_SPECS:
        LOGGER.info("Loading dataset %s", spec.hf_name)
        ds = load_hf_dataset_safe(spec.hf_name)
        if ds is None:
            LOGGER.warning("Skipping %s due to load failure.", spec.hf_name)
            continue

        dataset_rows: list[dict[str, Any]] = []
        for split_name, split_ds in ds.items():
            if len(split_ds) == 0:
                continue

            sample0 = split_ds[0]
            columns = list(split_ds.column_names)
            schema = infer_schema(sample0, columns)
            LOGGER.info(
                "%s/%s inferred schema: audio=%s label=%s speaker=%s text=%s",
                spec.short_name,
                split_name,
                schema["audio_col"],
                schema["label_col"],
                schema["speaker_col"],
                schema["text_col"],
            )

            limit = args.max_samples_per_split or len(split_ds)
            iterator = range(min(limit, len(split_ds)))
            for idx in tqdm(iterator, desc=f"{spec.short_name}:{split_name}", leave=False):
                sample = split_ds[idx]
                if not schema["audio_col"] or schema["audio_col"] not in sample:
                    missing_audio_field += 1
                    continue
                audio_value = sample.get(schema["audio_col"])
                if audio_value is None:
                    missing_audio_field += 1
                    continue

                emotion_raw = pick_value(sample, schema["label_col"])
                label_summary_inputs[spec.short_name].append(emotion_raw)
                emotion = normalize_emotion(emotion_raw)
                if emotion is None:
                    unmapped_label += 1
                    continue

                dataset_name = spec.short_name

                # Stratified subsampling by mapped emotion label.
                if per_class_cap is not None and dataset_class_counts[dataset_name][emotion] >= per_class_cap:
                    subsampling_drop += 1
                    continue
                if (
                    args.max_samples_per_dataset is not None
                    and dataset_kept_counts[dataset_name] >= args.max_samples_per_dataset
                ):
                    subsampling_drop += 1
                    continue

                speaker_id = infer_speaker_id(dataset_name, sample, schema["speaker_col"])
                text = pick_value(sample, schema["text_col"])
                sample_id = sample_id_from_fields(dataset_name, split_name, idx, speaker_id, text)

                rel_path = Path(dataset_name) / split_name / f"{sample_id}.wav"
                wav_path = raw_dir / rel_path
                ok, sr, duration_sec = export_audio(audio_value, wav_path, args.overwrite)
                if not ok:
                    audio_decode_failure += 1
                    continue

                row = {
                    "sample_id": sample_id,
                    "wav_path": str(wav_path.resolve()),
                    "dataset": dataset_name,
                    "original_split": split_name,
                    "speaker_id": speaker_id,
                    "emotion_raw": emotion_raw,
                    "emotion": emotion,
                    "text": text,
                    "sampling_rate": sr,
                    "duration_sec": duration_sec,
                }
                dataset_rows.append(row)
                all_rows.append(row)
                valid_samples += 1
                all_durations.append(duration_sec)
                dataset_kept_counts[dataset_name] += 1
                dataset_class_counts[dataset_name][emotion] += 1

        if dataset_rows:
            write_manifest(dataset_rows, manifests_dir / f"{spec.short_name}_metadata.csv")
            LOGGER.info(
                "Saved %d rows to %s",
                len(dataset_rows),
                manifests_dir / f"{spec.short_name}_metadata.csv",
            )

    write_manifest(all_rows, manifests_dir / "all_metadata.csv")
    LOGGER.info("Saved unified manifest: %s", manifests_dir / "all_metadata.csv")

    label_summaries = [
        summarize_label_mapping(dataset_name, labels)
        for dataset_name, labels in sorted(label_summary_inputs.items())
    ]
    save_label_mapping_reports(
        label_summaries,
        reports_dir / "label_mapping_report.csv",
        reports_dir / "label_mapping_report.txt",
    )

    avg_duration = float(np.mean(all_durations)) if all_durations else 0.0
    prep_summary = [
        "Data Preparation Summary",
        "=" * 80,
        f"Valid samples: {valid_samples}",
        f"Dropped due to missing audio field: {missing_audio_field}",
        f"Dropped due to audio decode failure: {audio_decode_failure}",
        f"Dropped due to unmapped label: {unmapped_label}",
        f"Dropped due to subsampling caps: {subsampling_drop}",
        f"Average duration (sec): {avg_duration:.3f}",
        "",
        "Subsampling parameters:",
        f"  - max_samples_per_dataset: {args.max_samples_per_dataset}",
        f"  - max_samples_per_class: {args.max_samples_per_class}",
        f"  - derived_per_class_cap: {derived_per_class_cap}",
        "",
        "Per-dataset counts:",
    ]
    if all_rows:
        counts = Counter(row["dataset"] for row in all_rows)
        for name, count in sorted(counts.items()):
            prep_summary.append(f"  - {name}: {count}")
            ds_df = pd.DataFrame([row for row in all_rows if row["dataset"] == name])
            split_mode, coverage = choose_split_mode(ds_df["speaker_id"])
            prep_summary.append(
                "    speaker coverage: "
                f"known={int(coverage['known'])}/{int(coverage['total'])}, "
                f"unknown_ratio={coverage['unknown_ratio']:.3f}, "
                f"unique_known={int(coverage['unique_known_speakers'])}"
            )
            prep_summary.append(f"    recommended split mode: {split_mode}")
    (reports_dir / "data_preparation_summary.txt").write_text(
        "\n".join(prep_summary), encoding="utf-8"
    )
    LOGGER.info("Saved preparation summary to %s", reports_dir / "data_preparation_summary.txt")

    # Helpful overview CSV for quick quality checks.
    if all_rows:
        pd.DataFrame(all_rows).groupby(["dataset", "emotion"]).size().reset_index(name="count").to_csv(
            reports_dir / "prepared_label_distribution.csv", index=False
        )


if __name__ == "__main__":
    main()
