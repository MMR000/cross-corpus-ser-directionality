#!/usr/bin/env python3
"""Preprocess exported wavs to 16 kHz mono and generate stats."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("preprocess_audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SER audio files.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Root directory of ser_project.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sampling rate.",
    )
    parser.add_argument("--min-duration", type=float, default=1.0, help="Minimum duration in seconds.")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=12.0,
        help="Maximum duration in seconds. Use <= 0 to disable max filter.",
    )
    parser.add_argument(
        "--peak-norm",
        type=float,
        default=0.98,
        help="Peak normalization limit to avoid clipping.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files.")
    return parser.parse_args()


def duration_bucket(duration_sec: float) -> str:
    if duration_sec < 2.0:
        return "short"
    if duration_sec <= 5.0:
        return "medium"
    return "long"


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(path))
    return waveform, sr


def preprocess_waveform(
    waveform: torch.Tensor, sr: int, target_sr: int, peak_norm: float
) -> tuple[torch.Tensor, int]:
    # Convert to mono by averaging channels.
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    # Safe peak normalization.
    peak = torch.max(torch.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
        waveform = waveform * float(peak_norm)
    return waveform, sr


def main() -> None:
    args = parse_args()
    root = args.project_root
    manifest_path = root / "data/manifests/all_metadata.csv"
    out_manifest_path = root / "data/manifests/all_metadata_processed.csv"
    processed_root = root / "data/processed"
    reports_dir = root / "reports"
    processed_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    output_rows: list[dict] = []

    for row in tqdm(df.to_dict(orient="records"), desc="preprocess", leave=False):
        wav_path = Path(row["wav_path"])
        dataset = str(row["dataset"])
        split = str(row["original_split"])
        sample_id = str(row["sample_id"])

        if not wav_path.exists():
            row["processed_wav_path"] = ""
            row["duration_bucket"] = "unknown"
            row["is_valid_length"] = False
            output_rows.append(row)
            continue

        rel_out = Path(dataset) / split / f"{sample_id}.wav"
        out_path = processed_root / rel_out
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if out_path.exists() and not args.overwrite:
                info = sf.info(out_path)
                proc_sr = int(info.samplerate)
                proc_duration = float(info.duration)
            else:
                waveform, sr = load_audio(wav_path)
                proc_wave, proc_sr = preprocess_waveform(waveform, sr, args.target_sr, args.peak_norm)
                torchaudio.save(str(out_path), proc_wave, proc_sr)
                proc_duration = float(proc_wave.size(-1) / proc_sr)

            valid = proc_duration >= args.min_duration and (
                args.max_duration <= 0 or proc_duration <= args.max_duration
            )
            row["processed_wav_path"] = str(out_path.resolve())
            row["processed_sampling_rate"] = proc_sr
            row["processed_duration_sec"] = proc_duration
            row["duration_bucket"] = duration_bucket(proc_duration)
            row["is_valid_length"] = bool(valid)
        except Exception as exc:
            LOGGER.warning("Failed preprocessing %s: %s", wav_path, exc)
            row["processed_wav_path"] = ""
            row["duration_bucket"] = "unknown"
            row["is_valid_length"] = False

        output_rows.append(row)

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_manifest_path, index=False)
    LOGGER.info("Processed manifest saved: %s", out_manifest_path)

    valid_df = out_df[out_df["is_valid_length"] == True].copy()  # noqa: E712
    if valid_df.empty:
        LOGGER.warning("No valid samples after preprocessing filters.")
        return

    stats_rows: list[dict] = []
    grouped = valid_df.groupby("dataset")
    for dataset, ds_df in grouped:
        counts_by_class = ds_df["emotion"].value_counts().to_dict()
        counts_by_duration = ds_df["duration_bucket"].value_counts().to_dict()
        stats_rows.append(
            {
                "dataset": dataset,
                "total_samples": int(len(ds_df)),
                "num_speakers": int(ds_df["speaker_id"].fillna("").nunique()),
                "mean_duration_sec": float(ds_df["processed_duration_sec"].mean()),
                "std_duration_sec": float(ds_df["processed_duration_sec"].std(ddof=0)),
                "short_count": int(counts_by_duration.get("short", 0)),
                "medium_count": int(counts_by_duration.get("medium", 0)),
                "long_count": int(counts_by_duration.get("long", 0)),
                "angry_count": int(counts_by_class.get("angry", 0)),
                "happy_count": int(counts_by_class.get("happy", 0)),
                "sad_count": int(counts_by_class.get("sad", 0)),
                "neutral_count": int(counts_by_class.get("neutral", 0)),
            }
        )

    stats_df = pd.DataFrame(stats_rows).sort_values("dataset")
    stats_csv = reports_dir / "dataset_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)

    # Build readable text report including histogram-like bins.
    hist_bins = [0, 1, 2, 3, 5, 8, 12, 20]
    hist_labels = ["0-1", "1-2", "2-3", "3-5", "5-8", "8-12", "12-20+"]
    text_lines = ["Dataset Statistics", "=" * 80]
    for dataset, ds_df in grouped:
        durations = ds_df["processed_duration_sec"].to_numpy()
        hist_counts, _ = np.histogram(durations, bins=hist_bins)
        text_lines.append(f"\nDataset: {dataset}")
        text_lines.append(f"Total samples: {len(ds_df)}")
        text_lines.append(f"Speakers: {ds_df['speaker_id'].fillna('').nunique()}")
        text_lines.append(
            f"Duration mean/std: {ds_df['processed_duration_sec'].mean():.3f} / "
            f"{ds_df['processed_duration_sec'].std(ddof=0):.3f}"
        )
        text_lines.append("Duration histogram:")
        for label, count in zip(hist_labels, hist_counts.tolist()):
            text_lines.append(f"  - {label}s: {count}")
        text_lines.append("Class distribution:")
        for emo, cnt in ds_df["emotion"].value_counts().to_dict().items():
            text_lines.append(f"  - {emo}: {cnt}")

    stats_txt = reports_dir / "dataset_statistics.txt"
    stats_txt.write_text("\n".join(text_lines), encoding="utf-8")
    LOGGER.info("Statistics reports saved: %s and %s", stats_csv, stats_txt)


if __name__ == "__main__":
    main()
