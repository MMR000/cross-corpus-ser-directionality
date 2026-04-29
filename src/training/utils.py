"""Training helper utilities."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def device_from_config(config: dict) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "auto"))
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str | Path, map_location: torch.device | str | None = None):
    """
    Load checkpoints using safest available torch.load mode.

    Tries weights_only=True first (safer). Falls back for older torch versions
    or legacy checkpoints when needed.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        logging.getLogger(__name__).warning(
            "torch.load(weights_only=...) unsupported; falling back to default load for %s", path
        )
        return torch.load(path, map_location=map_location)
    except Exception:
        logging.getLogger(__name__).warning(
            "weights_only checkpoint load failed; trying legacy load for %s", path
        )
        return torch.load(path, map_location=map_location)
