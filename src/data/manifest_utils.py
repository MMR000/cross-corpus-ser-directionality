"""Shared utilities for dataset inspection and manifest creation."""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import DatasetDict, load_dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    hf_name: str
    short_name: str


DATASET_SPECS = [
    DatasetSpec(hf_name="AbstractTTS/IEMOCAP", short_name="iemocap"),
    DatasetSpec(hf_name="AbstractTTS/PODCAST", short_name="podcast"),
    DatasetSpec(hf_name="AbstractTTS/CREMA-D", short_name="crema_d"),
]


LABEL_CANDIDATE_PATTERNS = [
    r".*emotion.*",
    r".*label.*",
    r".*sentiment.*",
    r".*category.*",
    r".*class.*",
]
SPEAKER_CANDIDATE_PATTERNS = [r".*speaker.*", r".*spk.*", r".*actor.*", r".*session.*"]
TEXT_CANDIDATE_PATTERNS = [
    r"^text$",
    r".*transcript.*",
    r".*sentence.*",
    r".*utterance.*",
    r".*content.*",
]


def load_hf_dataset_safe(name: str) -> Optional[DatasetDict]:
    """Load a Hugging Face dataset with robust warning-based fallback."""
    try:
        ds = load_dataset(name)
        return ds  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to load dataset '%s': %s", name, exc)
        return None


def normalize_text_for_match(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _pattern_match(value: str, patterns: list[str]) -> bool:
    lowered = normalize_text_for_match(value)
    return any(re.match(pat, lowered) for pat in patterns)


def _is_audio_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {str(k).lower() for k in value.keys()}
    has_wave = "array" in keys or "path" in keys or "bytes" in keys
    has_sr = "sampling_rate" in keys or "sr" in keys
    return has_wave and has_sr


def detect_audio_field(sample: dict[str, Any], columns: list[str]) -> Optional[str]:
    """Detect an audio column, including nested dict structures."""
    for col in columns:
        value = sample.get(col)
        if _is_audio_dict(value):
            return col
        if isinstance(value, dict):
            for nested_value in value.values():
                if _is_audio_dict(nested_value):
                    return col
    # fallback: by column name hint
    for col in columns:
        if "audio" in normalize_text_for_match(col) or "wav" in normalize_text_for_match(col):
            return col
    return None


def _find_candidate_column(
    columns: list[str], patterns: list[str], forbidden: set[str]
) -> Optional[str]:
    for col in columns:
        if col in forbidden:
            continue
        if _pattern_match(col, patterns):
            return col
    return None


def infer_schema(
    sample: dict[str, Any], columns: list[str]
) -> dict[str, Optional[str]]:
    """Infer canonical column names from a sample and split columns."""
    audio_col = detect_audio_field(sample, columns)
    used = {audio_col} if audio_col else set()
    label_col = _find_candidate_column(columns, LABEL_CANDIDATE_PATTERNS, used)
    if label_col:
        used.add(label_col)
    speaker_col = _find_candidate_column(columns, SPEAKER_CANDIDATE_PATTERNS, used)
    if speaker_col:
        used.add(speaker_col)
    text_col = _find_candidate_column(columns, TEXT_CANDIDATE_PATTERNS, used)
    return {
        "audio_col": audio_col,
        "label_col": label_col,
        "speaker_col": speaker_col,
        "text_col": text_col,
    }


def extract_audio_payload(value: Any) -> Optional[dict[str, Any]]:
    """Extract the first audio payload dict from value or nested dict."""
    if _is_audio_dict(value):
        return value
    if isinstance(value, dict):
        for nested in value.values():
            if _is_audio_dict(nested):
                return nested
    return None


def _to_numpy_audio_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    # torch tensor or tensor-like object
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            return value.detach().cpu().numpy().astype(np.float32)
        except Exception:
            return None
    if hasattr(value, "numpy"):
        try:
            return value.numpy().astype(np.float32)
        except Exception:
            return None
    return None


def _ensure_mono_1d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # Handle both [channels, time] and [time, channels].
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return arr.mean(axis=0).astype(np.float32)
        if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
            return arr.mean(axis=1).astype(np.float32)
        return arr.mean(axis=1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def extract_audio_sample(audio_obj: Any) -> tuple[np.ndarray, int]:
    """
    Extract (audio_array, sampling_rate) from HF audio representations.

    Supports:
      - decoded dicts with array/sampling_rate
      - datasets AudioDecoder objects (e.g., torchcodec AudioDecoder)
      - dicts with path or bytes
      - array/tensor-like with sampling_rate attribute
    """
    try:
        if audio_obj is None:
            raise ValueError("audio object is None")

        # Case 1: decoded dict-like payload
        if isinstance(audio_obj, dict):
            array = _to_numpy_audio_array(audio_obj.get("array"))
            sr = int(audio_obj.get("sampling_rate", audio_obj.get("sr", 0)) or 0)
            if array is not None and sr > 0:
                return _ensure_mono_1d(array), sr

            path = audio_obj.get("path")
            if path:
                loaded, loaded_sr = sf.read(str(path), always_2d=False)
                return _ensure_mono_1d(np.asarray(loaded, dtype=np.float32)), int(loaded_sr)

            audio_bytes = audio_obj.get("bytes")
            if audio_bytes is not None:
                loaded, loaded_sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
                return _ensure_mono_1d(np.asarray(loaded, dtype=np.float32)), int(loaded_sr)

            # Sometimes nested under another key.
            nested = extract_audio_payload(audio_obj)
            if nested is not None and nested is not audio_obj:
                return extract_audio_sample(nested)

        # Case 2: HF AudioDecoder object with get_all_samples().
        if hasattr(audio_obj, "get_all_samples"):
            decoded = audio_obj.get_all_samples()
            sr = int(
                getattr(decoded, "sample_rate", 0)
                or getattr(decoded, "sampling_rate", 0)
                or getattr(audio_obj, "sample_rate", 0)
                or getattr(audio_obj, "sampling_rate", 0)
            )
            data = getattr(decoded, "data", None)
            if data is None:
                data = getattr(decoded, "array", None)
            array = _to_numpy_audio_array(data)
            if array is not None and sr > 0:
                return _ensure_mono_1d(array), sr

        # Case 3: generic object with array + sampling_rate attributes.
        if hasattr(audio_obj, "array"):
            array = _to_numpy_audio_array(getattr(audio_obj, "array"))
            sr = int(
                getattr(audio_obj, "sampling_rate", 0)
                or getattr(audio_obj, "sample_rate", 0)
                or 0
            )
            if array is not None and sr > 0:
                return _ensure_mono_1d(array), sr

        # Case 4: direct path-like string.
        if isinstance(audio_obj, str):
            loaded, loaded_sr = sf.read(audio_obj, always_2d=False)
            return _ensure_mono_1d(np.asarray(loaded, dtype=np.float32)), int(loaded_sr)

        raise ValueError(f"Unsupported audio object type: {type(audio_obj)}")
    except Exception as exc:
        LOGGER.warning(
            "Audio extraction failed for object type %s: %s",
            type(audio_obj).__name__,
            exc,
        )
        raise ValueError(f"audio extraction failed: {exc}") from exc


def safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def sample_id_from_fields(
    dataset: str, split: str, index: int, speaker_id: str, text: str
) -> str:
    """Generate deterministic sample ids for exported waveforms."""
    stable = f"{dataset}|{split}|{index}|{speaker_id}|{text}"
    suffix = hashlib.md5(stable.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}_{split}_{index:08d}_{suffix}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_duration_seconds(num_samples: int, sr: int) -> float:
    if sr <= 0:
        return 0.0
    return float(num_samples) / float(sr)


def write_manifest(rows: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "original_split"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


def detect_label_candidates(columns: list[str]) -> list[str]:
    return [c for c in columns if _pattern_match(c, LABEL_CANDIDATE_PATTERNS)]


def detect_audio_candidates(columns: list[str]) -> list[str]:
    cands = []
    for c in columns:
        c_norm = normalize_text_for_match(c)
        if "audio" in c_norm or "wav" in c_norm:
            cands.append(c)
    return cands


def list_split_names(dataset_dict: DatasetDict) -> list[str]:
    return list(dataset_dict.keys())


def maybe_to_numpy_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value)
    return None


def _is_unknown_speaker(value: str) -> bool:
    normalized = normalize_text_for_match(str(value))
    return normalized in {"", "unknown", "none", "nan", "null", "n_a"}


def infer_iemocap_speaker_id_from_file(file_value: Any) -> Optional[str]:
    """Infer IEMOCAP speaker id from file name like Ses01F_impro01_F000.wav."""
    if file_value is None:
        return None
    file_name = Path(str(file_value)).name
    match = re.match(r"^(Ses\d{2}[FM])_", file_name)
    if match:
        return match.group(1)
    return None


def infer_cremad_speaker_id_from_file(file_value: Any) -> Optional[str]:
    """Infer CREMA-D speaker id from file name like 1001_DFA_ANG_XX.wav."""
    if file_value is None:
        return None
    file_name = Path(str(file_value)).name
    match = re.match(r"^(\d+)_", file_name)
    if match:
        return match.group(1)
    return None


def infer_speaker_id(
    dataset_name: str,
    sample: dict[str, Any],
    inferred_speaker_col: Optional[str],
) -> str:
    """Infer speaker ids with dataset-specific rules and robust fallback."""
    dataset = normalize_text_for_match(dataset_name)

    # Dataset-specific rule: IEMOCAP speaker id from 'file' field.
    if dataset == "iemocap":
        file_value = sample.get("file")
        inferred = infer_iemocap_speaker_id_from_file(file_value)
        if inferred is not None:
            return inferred

    # Dataset-specific rule: CREMA-D speaker id from 'file' field.
    if dataset == "crema_d":
        file_value = sample.get("file")
        inferred = infer_cremad_speaker_id_from_file(file_value)
        if inferred is not None:
            return inferred

    # Dataset-specific rule: PODCAST has no reliable speaker id for now.
    if dataset == "podcast":
        return "unknown"

    # Generic schema-based fallback if present.
    if inferred_speaker_col:
        value = safe_string(sample.get(inferred_speaker_col)).strip()
        if not _is_unknown_speaker(value):
            return value

    return "unknown"


def choose_split_mode(
    speaker_series: pd.Series,
    min_unique_speakers: int = 3,
    unknown_ratio_threshold: float = 0.5,
) -> tuple[str, dict[str, float]]:
    """
    Decide split mode from speaker-id coverage.

    Returns:
        mode: "speaker_aware" or "stratified_fallback"
        coverage: dict with total/known/unknown/unknown_ratio/unique_known_speakers
    """
    speakers = speaker_series.fillna("").astype(str).str.strip()
    is_known = ~speakers.apply(_is_unknown_speaker)
    total = int(len(speakers))
    known = int(is_known.sum())
    unknown = int(total - known)
    unknown_ratio = float(unknown / total) if total > 0 else 1.0
    unique_known = int(speakers[is_known].nunique())

    mode = "speaker_aware"
    if total == 0 or unique_known < min_unique_speakers or unknown_ratio > unknown_ratio_threshold:
        mode = "stratified_fallback"

    coverage = {
        "total": float(total),
        "known": float(known),
        "unknown": float(unknown),
        "unknown_ratio": float(unknown_ratio),
        "unique_known_speakers": float(unique_known),
    }
    return mode, coverage
