"""Dataset and dataloader utilities for Phase 2 SER experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

EMOTION_TO_ID = {"angry": 0, "happy": 1, "sad": 2, "neutral": 3}
ID_TO_EMOTION = {v: k for k, v in EMOTION_TO_ID.items()}


@dataclass
class SplitConfig:
    train_manifest: str
    dev_manifest: Optional[str]
    test_manifest: Optional[str]


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "is_valid_length" in df.columns:
        # Works for bool-like strings and booleans.
        mask = df["is_valid_length"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[mask].copy()
    return df.reset_index(drop=True)


def enrich_with_metadata(split_df: pd.DataFrame, metadata_path: Optional[str | Path]) -> pd.DataFrame:
    """Merge split CSV with unified processed metadata (by sample_id) when provided."""
    if metadata_path is None:
        return split_df
    if "sample_id" not in split_df.columns:
        return split_df
    metadata_df = load_manifest(metadata_path)
    if "sample_id" not in metadata_df.columns:
        return split_df

    split_keep = split_df.copy()
    merged = split_keep.merge(
        metadata_df,
        on="sample_id",
        how="left",
        suffixes=("", "_meta"),
    )
    # Prefer split-specific columns; fill missing with metadata columns when available.
    for col in ["processed_wav_path", "wav_path", "emotion", "dataset", "speaker_id"]:
        meta_col = f"{col}_meta"
        if col in merged.columns and meta_col in merged.columns:
            merged[col] = merged[col].fillna(merged[meta_col])
        elif col not in merged.columns and meta_col in merged.columns:
            merged[col] = merged[meta_col]
    drop_cols = [c for c in merged.columns if c.endswith("_meta")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def resolve_path_column(df: pd.DataFrame, preferred_column: str) -> str:
    if preferred_column in df.columns:
        return preferred_column
    if "processed_wav_path" in df.columns:
        return "processed_wav_path"
    if "wav_path" in df.columns:
        return "wav_path"
    raise KeyError("No audio path column found in manifest.")


class SERManifestDataset(Dataset):
    """Audio dataset backed by split or manifest CSV files."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        path_column: str = "processed_wav_path",
        label_column: str = "emotion",
        target_sr: int = 16000,
        max_duration_sec: Optional[float] = None,
        domain_column: Optional[str] = None,
        domain_id_map: Optional[dict[str, int]] = None,
    ) -> None:
        self.df = manifest_df.copy().reset_index(drop=True)
        self.path_column = resolve_path_column(self.df, path_column)
        self.label_column = label_column
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec
        self.domain_column = domain_column
        self.domain_id_map = domain_id_map

        if label_column not in self.df.columns:
            raise KeyError(f"Missing label column '{label_column}' in manifest.")

        self.df = self.df[self.df[self.label_column].isin(EMOTION_TO_ID.keys())].reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No valid rows remain after filtering for target emotion labels.")

        if self.domain_id_map is not None:
            col = self.domain_column or "dataset"
            if col not in self.df.columns:
                raise KeyError(f"Domain column '{col}' not found in manifest for domain-adaptation training.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = str(row[self.path_column])
        waveform, sr = torchaudio.load(path)
        # Convert to mono.
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, self.target_sr).squeeze(0)
            sr = self.target_sr

        if self.max_duration_sec and self.max_duration_sec > 0:
            max_len = int(self.max_duration_sec * sr)
            if waveform.numel() > max_len:
                waveform = waveform[:max_len]

        label_name = str(row[self.label_column])
        label_id = EMOTION_TO_ID[label_name]
        item: dict[str, Any] = {
            "waveform": waveform,
            "length": waveform.numel(),
            "label_id": label_id,
            "label_name": label_name,
            "sample_id": str(row.get("sample_id", idx)),
            "dataset": str(row.get("dataset", "")),
            "path": path,
        }
        if self.domain_id_map is not None:
            col = self.domain_column or "dataset"
            key = str(row[col])
            if key not in self.domain_id_map:
                raise KeyError(
                    f"Domain value {key!r} not in domain_id_map keys {sorted(self.domain_id_map.keys())!r}."
                )
            item["domain_id"] = int(self.domain_id_map[key])
        return item


def ser_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    waveforms = torch.zeros((len(batch), max_len), dtype=torch.float32)
    labels = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)

    sample_ids: list[str] = []
    datasets: list[str] = []
    paths: list[str] = []
    for i, item in enumerate(batch):
        wav = item["waveform"].float()
        waveforms[i, : wav.numel()] = wav
        sample_ids.append(item["sample_id"])
        datasets.append(item["dataset"])
        paths.append(item["path"])

    out: dict[str, Any] = {
        "waveforms": waveforms,
        "lengths": lengths,
        "labels": labels,
        "sample_ids": sample_ids,
        "datasets": datasets,
        "paths": paths,
    }
    if batch and "domain_id" in batch[0]:
        out["domain_ids"] = torch.tensor([int(item["domain_id"]) for item in batch], dtype=torch.long)
    return out


def build_loader(
    manifest_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    path_column: str,
    label_column: str,
    sample_rate: int = 16000,
    max_duration_sec: Optional[float] = None,
    metadata_path: Optional[str | Path] = None,
    domain_column: Optional[str] = None,
    domain_id_map: Optional[dict[str, int]] = None,
) -> DataLoader:
    df = load_manifest(manifest_path)
    df = enrich_with_metadata(df, metadata_path)
    dataset = SERManifestDataset(
        manifest_df=df,
        path_column=path_column,
        label_column=label_column,
        target_sr=sample_rate,
        max_duration_sec=max_duration_sec,
        domain_column=domain_column,
        domain_id_map=domain_id_map,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=ser_collate_fn,
    )


def build_domain_id_map_from_train_df(
    train_df: pd.DataFrame,
    domain_column: str,
    explicit_classes: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    """Build stable string->id mapping for domain labels (sorted keys when auto)."""
    if explicit_classes is not None:
        return {str(k): int(v) for k, v in explicit_classes.items()}
    if domain_column not in train_df.columns:
        raise KeyError(f"domain_column {domain_column!r} missing from train manifest.")
    uniq = sorted(train_df[domain_column].dropna().astype(str).unique().tolist())
    return {name: i for i, name in enumerate(uniq)}


def split_train_dev_from_combined(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use split column from combined train/dev manifest when available."""
    if "split" not in df.columns:
        raise KeyError("Combined manifest does not include 'split' column.")
    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()
    if train_df.empty or dev_df.empty:
        raise ValueError("Combined manifest must contain both train and dev rows.")
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)
