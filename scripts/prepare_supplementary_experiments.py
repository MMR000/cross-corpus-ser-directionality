#!/usr/bin/env python3
"""Prepare config/manifests for supplementary non-architectural experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.training.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "phase2"
SUPP_DIR = ROOT / "data" / "splits" / "supplementary"


def _save_yaml(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _sample_source_fraction(base_manifest: str, fraction: float, out_path: Path) -> None:
    df = pd.read_csv(ROOT / base_manifest)
    if "split" in df.columns:
        train = df[df["split"] == "train"].copy()
        dev = df[df["split"] == "dev"].copy()
        n = max(1, int(round(len(train) * fraction)))
        train_sub = train.sample(n=n, random_state=42).sort_index()
        out = pd.concat([train_sub, dev], ignore_index=True)
    else:
        n = max(1, int(round(len(df) * fraction)))
        out = df.sample(n=n, random_state=42).sort_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def _sample_target_fraction(manifest: str, fraction: float, out_path: Path) -> None:
    df = pd.read_csv(ROOT / manifest)
    n = max(1, int(round(len(df) * fraction)))
    sub = df.sample(n=n, random_state=42).sort_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)


def main() -> None:
    generated_configs: list[str] = []

    # -------- 1) Multi-seed core runs --------
    multi_seed_specs = [
        "iemocap_to_podcast_chunk",
        "iemocap_to_podcast_chunk_domain_utt",
        "podcast_to_iemocap_chunk",
        "podcast_to_iemocap_chunk_domain_utt",
    ]
    for base_name in multi_seed_specs:
        base_cfg = load_yaml(CONFIG_DIR / f"{base_name}.yaml")
        for seed in [7, 13]:
            cfg = dict(base_cfg)
            cfg["seed"] = seed
            cfg["experiment"] = dict(base_cfg["experiment"])
            cfg["experiment"]["name"] = f"{base_name}_seed{seed}"
            out = CONFIG_DIR / f"{cfg['experiment']['name']}.yaml"
            _save_yaml(cfg, out)
            generated_configs.append(str(out.relative_to(ROOT)))

    # -------- 2) UDA target fraction (i2p domain_utt only) --------
    base_uda = load_yaml(CONFIG_DIR / "iemocap_to_podcast_chunk_domain_utt.yaml")
    for frac in [0.25, 0.50, 0.75]:
        pct = int(frac * 100)
        cfg = dict(base_uda)
        cfg["experiment"] = dict(base_uda["experiment"])
        cfg["experiment"]["name"] = f"iemocap_to_podcast_chunk_domain_utt_tgtfrac{pct}"
        cfg["data"] = dict(base_uda["data"])
        cfg["uda"] = dict(base_uda["uda"])
        tr_out = SUPP_DIR / f"podcast_train_frac{pct}.csv"
        dv_out = SUPP_DIR / f"podcast_dev_frac{pct}.csv"
        _sample_target_fraction(base_uda["uda"]["target_train_manifest"], frac, tr_out)
        _sample_target_fraction(base_uda["uda"]["target_dev_manifest"], frac, dv_out)
        cfg["uda"]["target_train_manifest"] = str(tr_out.relative_to(ROOT))
        cfg["uda"]["target_dev_manifest"] = str(dv_out.relative_to(ROOT))
        out = CONFIG_DIR / f"{cfg['experiment']['name']}.yaml"
        _save_yaml(cfg, out)
        generated_configs.append(str(out.relative_to(ROOT)))

    # -------- 3) Source fraction ablation (i2p chunk + i2p domain_utt) --------
    for base_name in ["iemocap_to_podcast_chunk", "iemocap_to_podcast_chunk_domain_utt"]:
        base_cfg = load_yaml(CONFIG_DIR / f"{base_name}.yaml")
        for frac in [0.25, 0.50]:
            pct = int(frac * 100)
            cfg = dict(base_cfg)
            cfg["experiment"] = dict(base_cfg["experiment"])
            cfg["data"] = dict(base_cfg["data"])
            cfg["experiment"]["name"] = f"{base_name}_srcfrac{pct}"
            mf_out = SUPP_DIR / f"iemocap_to_podcast_train_srcfrac{pct}.csv"
            _sample_source_fraction(base_cfg["data"]["train_manifest"], frac, mf_out)
            cfg["data"]["train_manifest"] = str(mf_out.relative_to(ROOT))
            out = CONFIG_DIR / f"{cfg['experiment']['name']}.yaml"
            _save_yaml(cfg, out)
            generated_configs.append(str(out.relative_to(ROOT)))

    print("Prepared configs:")
    for p in generated_configs:
        print(p)


if __name__ == "__main__":
    main()
