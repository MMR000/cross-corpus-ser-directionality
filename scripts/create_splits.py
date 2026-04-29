#!/usr/bin/env python3
"""Create in-corpus and cross-corpus split manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.data.manifest_utils import choose_split_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("create_splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SER experiment split files.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def _check_ratios(train: float, dev: float, test: float) -> None:
    total = train + dev + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def _split_with_groups(
    df: pd.DataFrame, seed: int, train_ratio: float, dev_ratio: float, test_ratio: float
) -> Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Speaker-independent split when enough speaker diversity exists."""
    speakers = df["speaker_id"].fillna("").astype(str).str.strip()
    split_mode, _ = choose_split_mode(speakers)
    if split_mode != "speaker_aware":
        return None

    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_dev_idx, test_idx = next(gss.split(df, groups=speakers))
    train_dev_df = df.iloc[train_dev_idx]
    test_df = df.iloc[test_idx]

    rem_total = train_ratio + dev_ratio
    dev_within_train_dev = dev_ratio / rem_total
    gss2 = GroupShuffleSplit(n_splits=1, test_size=dev_within_train_dev, random_state=seed + 1)
    train_idx, dev_idx = next(gss2.split(train_dev_df, groups=train_dev_df["speaker_id"].fillna("").astype(str)))
    train_df = train_dev_df.iloc[train_idx]
    dev_df = train_dev_df.iloc[dev_idx]
    return train_df, dev_df, test_df


def _split_stratified(
    df: pd.DataFrame, seed: int, train_ratio: float, dev_ratio: float, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = df["emotion"].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels if labels.nunique() > 1 else None,
    )
    rem_total = train_ratio + dev_ratio
    dev_within_train = dev_ratio / rem_total
    train_labels = train_df["emotion"].astype(str)
    train_df, dev_df = train_test_split(
        train_df,
        test_size=dev_within_train,
        random_state=seed + 1,
        stratify=train_labels if train_labels.nunique() > 1 else None,
    )
    return train_df, dev_df, test_df


def split_in_corpus(
    df: pd.DataFrame, seed: int, train_ratio: float, dev_ratio: float, test_ratio: float
) -> tuple[pd.DataFrame, str, dict[str, float]]:
    split_mode, coverage = choose_split_mode(df["speaker_id"])
    grouped = _split_with_groups(df, seed, train_ratio, dev_ratio, test_ratio)
    if grouped is None:
        train_df, dev_df, test_df = _split_stratified(df, seed, train_ratio, dev_ratio, test_ratio)
        split_mode = "stratified_fallback"
    else:
        train_df, dev_df, test_df = grouped
        split_mode = "speaker_aware"

    train_df = train_df.copy()
    dev_df = dev_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    dev_df["split"] = "dev"
    test_df["split"] = "test"
    merged = pd.concat([train_df, dev_df, test_df], axis=0).reset_index(drop=True)
    return merged, split_mode, coverage


def save_protocol_a(in_corpus_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, ds_df in in_corpus_df.groupby("dataset"):
        for split_name in ("train", "dev", "test"):
            split_df = ds_df[ds_df["split"] == split_name]
            split_df.to_csv(out_dir / f"{dataset}_{split_name}.csv", index=False)


def save_protocol_b(in_corpus_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("iemocap", "podcast"),
        ("podcast", "iemocap"),
        ("iemocap", "crema_d"),
        ("crema_d", "podcast"),
    ]
    for source, target in pairs:
        source_train = in_corpus_df[(in_corpus_df["dataset"] == source) & (in_corpus_df["split"].isin(["train", "dev"]))]
        target_test = in_corpus_df[(in_corpus_df["dataset"] == target) & (in_corpus_df["split"] == "test")]
        source_train.to_csv(out_dir / f"{source}_to_{target}_train.csv", index=False)
        target_test.to_csv(out_dir / f"{source}_to_{target}_test.csv", index=False)


def save_protocol_c(in_corpus_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    triplets = [
        (["iemocap", "crema_d"], "podcast"),
        (["podcast", "crema_d"], "iemocap"),
        (["iemocap", "podcast"], "crema_d"),
    ]
    for sources, target in triplets:
        source_name = "_plus_".join(sources)
        source_train = in_corpus_df[
            (in_corpus_df["dataset"].isin(sources)) & (in_corpus_df["split"].isin(["train", "dev"]))
        ]
        target_test = in_corpus_df[(in_corpus_df["dataset"] == target) & (in_corpus_df["split"] == "test")]
        source_train.to_csv(out_dir / f"{source_name}_to_{target}_train.csv", index=False)
        target_test.to_csv(out_dir / f"{source_name}_to_{target}_test.csv", index=False)


def main() -> None:
    args = parse_args()
    _check_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)
    root = args.project_root
    manifest_path = root / "data/manifests/all_metadata_processed.csv"
    split_root = root / "data/splits"
    split_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing processed manifest: {manifest_path}")

    df = pd.read_csv(manifest_path)
    valid_df = df[df["is_valid_length"] == True].copy()  # noqa: E712
    if valid_df.empty:
        raise RuntimeError("No valid samples found in processed manifest.")

    in_corpus_parts = []
    dataset_split_modes: dict[str, str] = {}
    dataset_coverage: dict[str, dict[str, float]] = {}
    for dataset, ds_df in valid_df.groupby("dataset"):
        split_df, split_mode, coverage = split_in_corpus(
            ds_df, args.seed, args.train_ratio, args.dev_ratio, args.test_ratio
        )
        split_df.to_csv(split_root / f"in_corpus_{dataset}_all.csv", index=False)
        in_corpus_parts.append(split_df)
        dataset_split_modes[dataset] = split_mode
        dataset_coverage[dataset] = coverage
        LOGGER.info(
            "Protocol A in-corpus split created for %s (%d rows, mode=%s)",
            dataset,
            len(split_df),
            split_mode,
        )

    in_corpus_df = pd.concat(in_corpus_parts, axis=0).reset_index(drop=True)
    in_corpus_df.to_csv(split_root / "in_corpus_all.csv", index=False)

    save_protocol_a(in_corpus_df, split_root / "protocol_a_in_corpus")
    save_protocol_b(in_corpus_df, split_root / "protocol_b_one_to_one")
    save_protocol_c(in_corpus_df, split_root / "protocol_c_multi_source")

    summary_lines = [
        "Split Creation Summary",
        "=" * 80,
        f"Total valid rows: {len(in_corpus_df)}",
        "",
    ]
    for dataset, ds_df in in_corpus_df.groupby("dataset"):
        summary_lines.append(f"Dataset: {dataset}")
        coverage = dataset_coverage.get(dataset, {})
        mode = dataset_split_modes.get(dataset, "stratified_fallback")
        summary_lines.append(
            "  - speaker coverage: "
            f"known={int(coverage.get('known', 0))}/{int(coverage.get('total', 0))}, "
            f"unknown_ratio={coverage.get('unknown_ratio', 1.0):.3f}, "
            f"unique_known={int(coverage.get('unique_known_speakers', 0))}"
        )
        summary_lines.append(f"  - split_mode: {mode}")
        summary_lines.append(f"  - train: {len(ds_df[ds_df['split'] == 'train'])}")
        summary_lines.append(f"  - dev:   {len(ds_df[ds_df['split'] == 'dev'])}")
        summary_lines.append(f"  - test:  {len(ds_df[ds_df['split'] == 'test'])}")
    (root / "reports/split_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    LOGGER.info("Split files generated under %s", split_root)


if __name__ == "__main__":
    main()
