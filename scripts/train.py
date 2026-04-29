#!/usr/bin/env python3
"""Train Phase 2 SER baselines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import torch

from src.data.datasets import (
    build_domain_id_map_from_train_df,
    build_loader,
    enrich_with_metadata,
    load_manifest,
    split_train_dev_from_combined,
)
from src.models import build_model
from src.training.trainer import SERTrainer, TrainerConfig
from src.training.utils import device_from_config, dump_json, ensure_dir, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SER baseline model from YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ser_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def _train_manifest_df_for_domain_mapping(config: dict) -> "pd.DataFrame":
    """Training rows used to define domain-id mapping (respects combined train/dev CSV)."""
    data_cfg = config["data"]
    train_path = data_cfg["train_manifest"]
    metadata_path = data_cfg.get("metadata_manifest")
    df = load_manifest(train_path)
    df = enrich_with_metadata(df, metadata_path)
    if data_cfg.get("dev_manifest"):
        return df
    train_df, _ = split_train_dev_from_combined(df)
    return train_df


def _load_manifest_with_metadata(path: str, metadata_path: str | None) -> "pd.DataFrame":
    df = load_manifest(path)
    return enrich_with_metadata(df, metadata_path)


def _build_target_unlabeled_loader(
    config: dict,
    domain_column: str,
    domain_id_map: dict[str, int],
) -> tuple["torch.utils.data.DataLoader", int]:
    """Build target unlabeled loader from target train/dev manifests for UDA."""
    uda_cfg = config.get("uda", {})
    target_train_manifest = uda_cfg.get("target_train_manifest")
    if not target_train_manifest:
        raise ValueError("UDA is enabled but 'uda.target_train_manifest' is missing.")
    metadata_path = config["data"].get("metadata_manifest")
    target_df = _load_manifest_with_metadata(str(target_train_manifest), metadata_path)
    target_dev_manifest = uda_cfg.get("target_dev_manifest")
    if target_dev_manifest:
        target_dev_df = _load_manifest_with_metadata(str(target_dev_manifest), metadata_path)
        target_df = pd.concat([target_df, target_dev_df], axis=0, ignore_index=True)

    temp_dir = Path(config["experiment"].get("output_dir", "exp")) / config["experiment"]["name"] / "tmp_manifests"
    ensure_dir(temp_dir)
    target_tmp = temp_dir / "target_unlabeled_train_dev.csv"
    target_df.to_csv(target_tmp, index=False)
    loader = build_loader(
        manifest_path=target_tmp,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 2)),
        path_column=str(config["data"].get("path_column", "processed_wav_path")),
        label_column=str(config["data"].get("label_column", "emotion")),
        sample_rate=int(config["data"].get("sample_rate", 16000)),
        max_duration_sec=config["data"].get("max_duration_sec"),
        metadata_path=config["data"].get("metadata_manifest"),
        domain_column=domain_column,
        domain_id_map=domain_id_map,
    )
    return loader, int(len(target_df))


def _build_train_and_dev_loaders(
    config: dict,
    domain_column: str | None = None,
    train_domain_id_map: dict[str, int] | None = None,
):
    data_cfg = config["data"]
    train_manifest = data_cfg["train_manifest"]
    dev_manifest = data_cfg.get("dev_manifest")
    metadata_path = data_cfg.get("metadata_manifest")
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 2))
    path_col = str(data_cfg.get("path_column", "processed_wav_path"))
    label_col = str(data_cfg.get("label_column", "emotion"))
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    max_duration = data_cfg.get("max_duration_sec")

    if dev_manifest:
        train_loader = build_loader(
            manifest_path=train_manifest,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            path_column=path_col,
            label_column=label_col,
            sample_rate=sample_rate,
            max_duration_sec=max_duration,
            metadata_path=metadata_path,
            domain_column=domain_column,
            domain_id_map=train_domain_id_map,
        )
        dev_loader = build_loader(
            manifest_path=dev_manifest,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            path_column=path_col,
            label_column=label_col,
            sample_rate=sample_rate,
            max_duration_sec=max_duration,
            metadata_path=metadata_path,
            domain_column=None,
            domain_id_map=None,
        )
        return train_loader, dev_loader

    # Fallback for one-to-one files that include train+dev rows in one CSV.
    train_df = load_manifest(train_manifest)
    train_df, dev_df = split_train_dev_from_combined(train_df)
    temp_dir = Path(config["experiment"].get("output_dir", "exp")) / config["experiment"]["name"] / "tmp_manifests"
    ensure_dir(temp_dir)
    train_tmp = temp_dir / "train.csv"
    dev_tmp = temp_dir / "dev.csv"
    train_df.to_csv(train_tmp, index=False)
    dev_df.to_csv(dev_tmp, index=False)

    train_loader = build_loader(
        manifest_path=train_tmp,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        path_column=path_col,
        label_column=label_col,
        sample_rate=sample_rate,
        max_duration_sec=max_duration,
        metadata_path=metadata_path,
        domain_column=domain_column,
        domain_id_map=train_domain_id_map,
    )
    dev_loader = build_loader(
        manifest_path=dev_tmp,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        path_column=path_col,
        label_column=label_col,
        sample_rate=sample_rate,
        max_duration_sec=max_duration,
        metadata_path=metadata_path,
        domain_column=None,
        domain_id_map=None,
    )
    return train_loader, dev_loader


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))

    exp_name = config["experiment"]["name"]
    exp_root = Path(config["experiment"].get("output_dir", "exp")) / exp_name
    ensure_dir(exp_root)
    logger = setup_logging(exp_root / "train.log")
    device = device_from_config(config)
    cuda_available = torch.cuda.is_available()
    logger.info("torch.cuda.is_available(): %s", cuda_available)
    logger.info("Selected device: %s", device)
    if device.type == "cuda" and cuda_available:
        gpu_name = torch.cuda.get_device_name(device.index or 0)
        logger.info("Using GPU: %s", gpu_name)
    else:
        logger.warning("Training is running on CPU. This may be slow.")

    model_name = str(config["model"].get("name", "")).lower()
    domain_model_names = {
        "chunk_attention_domain_utt",
        "chunk_domain_utt",
        "chunk_attention_domain_both",
        "chunk_domain_both",
    }
    uda_cfg = config.get("uda", {})
    uda_enabled = bool(uda_cfg.get("enabled", False))
    domain_column: str | None = None
    domain_id_map: dict[str, int] | None = None
    source_train_count = 0
    target_unlabeled_count = 0
    target_unlabeled_loader = None
    unique_domain_count = 0
    if model_name in domain_model_names:
        domain_column = str(config["data"].get("domain_column", "dataset"))
        train_dom_df = _train_manifest_df_for_domain_mapping(config)
        if uda_enabled:
            target_train_df = _load_manifest_with_metadata(
                str(uda_cfg["target_train_manifest"]),
                config["data"].get("metadata_manifest"),
            )
            target_dev_manifest = uda_cfg.get("target_dev_manifest")
            if target_dev_manifest:
                target_dev_df = _load_manifest_with_metadata(
                    str(target_dev_manifest),
                    config["data"].get("metadata_manifest"),
                )
                target_train_df = pd.concat([target_train_df, target_dev_df], axis=0, ignore_index=True)
            mapping_df = pd.concat([train_dom_df, target_train_df], axis=0, ignore_index=True)
        else:
            mapping_df = train_dom_df
        explicit = config["data"].get("domain_classes")
        domain_id_map = build_domain_id_map_from_train_df(
            mapping_df,
            domain_column,
            explicit if isinstance(explicit, dict) else None,
        )
        config["data"]["domain_column"] = domain_column
        config["data"]["domain_id_map"] = domain_id_map
        config["model"]["num_domains"] = len(domain_id_map)
        unique_domain_count = int(len(domain_id_map))
        if len(domain_id_map) < 2:
            logger.warning(
                "Training data has only one domain value (%s); utterance domain-adversarial loss is disabled.",
                ", ".join(sorted(domain_id_map.keys())),
            )

    train_loader, dev_loader = _build_train_and_dev_loaders(
        config,
        domain_column=domain_column,
        train_domain_id_map=domain_id_map,
    )
    source_train_count = len(train_loader.dataset)

    if model_name in domain_model_names and uda_enabled:
        target_unlabeled_loader, target_unlabeled_count = _build_target_unlabeled_loader(
            config,
            domain_column=domain_column or "dataset",
            domain_id_map=domain_id_map or {},
        )

    test_loader = None
    test_manifest = config["data"].get("test_manifest")
    if test_manifest:
        test_loader = build_loader(
            manifest_path=test_manifest,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["data"].get("num_workers", 2)),
            path_column=str(config["data"].get("path_column", "processed_wav_path")),
            label_column=str(config["data"].get("label_column", "emotion")),
            sample_rate=int(config["data"].get("sample_rate", 16000)),
            max_duration_sec=config["data"].get("max_duration_sec"),
            metadata_path=config["data"].get("metadata_manifest"),
            domain_column=None,
            domain_id_map=None,
        )

    model = build_model(config).to(device)
    utt_dom_w = float(config["model"].get("utterance_domain_loss_weight", config["model"].get("domain_loss_weight", 1.0)))
    chunk_dom_w = float(config["model"].get("chunk_domain_loss_weight", 0.0))
    if int(config["model"].get("num_domains", 0)) < 2:
        utt_dom_w = 0.0
        chunk_dom_w = 0.0
    utterance_domain_loss_active = utt_dom_w > 0.0
    chunk_domain_loss_active = chunk_dom_w > 0.0
    domain_loss_active = utterance_domain_loss_active or chunk_domain_loss_active

    source_domain_name = str(uda_cfg.get("source_domain", "n/a"))
    target_domain_name = str(uda_cfg.get("target_domain", "n/a"))
    logger.info("Source domain: %s", source_domain_name)
    logger.info("Target domain: %s", target_domain_name)
    logger.info("Source labeled train samples: %d", source_train_count)
    logger.info("Target unlabeled samples: %d", target_unlabeled_count)
    logger.info("Utterance domain loss active: %s", utterance_domain_loss_active)
    logger.info("Chunk domain loss active: %s", chunk_domain_loss_active)
    logger.info("Unique domains in domain-classifier training: %d", unique_domain_count)
    config.setdefault("runtime_info", {})
    config["runtime_info"].update(
        {
            "source_train_samples": source_train_count,
            "target_unlabeled_samples": target_unlabeled_count,
            "domain_loss_active": domain_loss_active,
            "utterance_domain_loss_active": utterance_domain_loss_active,
            "chunk_domain_loss_active": chunk_domain_loss_active,
            "source_domain": source_domain_name,
            "target_domain": target_domain_name,
            "unique_domain_count": unique_domain_count,
        }
    )
    dump_json(config, exp_root / "resolved_config.json")

    trainer = SERTrainer(
        model=model,
        device=device,
        output_dir=exp_root,
        config=TrainerConfig(
            epochs=int(config["train"]["epochs"]),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"].get("weight_decay", 0.0)),
            grad_clip=config["train"].get("grad_clip"),
            utterance_domain_loss_weight=utt_dom_w,
            chunk_domain_loss_weight=chunk_dom_w,
        ),
        logger=logger,
    )
    summary = trainer.fit(
        train_loader,
        dev_loader,
        test_loader,
        config,
        target_unlabeled_loader=target_unlabeled_loader,
    )
    summary["source_train_samples"] = source_train_count
    summary["target_unlabeled_samples"] = target_unlabeled_count
    summary["domain_loss_active"] = domain_loss_active
    summary["utterance_domain_loss_active"] = utterance_domain_loss_active
    summary["chunk_domain_loss_active"] = chunk_domain_loss_active
    summary["source_domain"] = source_domain_name
    summary["target_domain"] = target_domain_name
    summary["num_domains"] = unique_domain_count
    stage1_valid = bool(
        model_name in domain_model_names
        and domain_loss_active
        and unique_domain_count >= 2
        and ((not uda_enabled) or target_unlabeled_count > 0)
    )
    summary["stage1_valid_domain_adaptation"] = stage1_valid
    summary["unique_domain_count"] = unique_domain_count
    if model_name in domain_model_names and not stage1_valid:
        summary["stage1_validity_note"] = "invalid_source_only_single_domain_domain_loss_disabled"
    elif model_name in domain_model_names:
        summary["stage1_validity_note"] = "valid_domain_adversarial_setup"
    pd.DataFrame([summary]).to_csv(exp_root / "final_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(exp_root / "summary.csv", index=False)
    logger.info(
        "Final summary | best_epoch=%s | best_dev_uar=%.4f | checkpoint=%s | test_uar=%s | test_wa=%s | test_macro_f1=%s",
        summary.get("best_epoch"),
        float(summary.get("best_dev_uar", 0.0)),
        summary.get("checkpoint"),
        summary.get("test_uar", "n/a"),
        summary.get("test_wa", "n/a"),
        summary.get("test_macro_f1", "n/a"),
    )
    print(f"Training complete: {exp_name}")
    print(summary)


if __name__ == "__main__":
    main()
