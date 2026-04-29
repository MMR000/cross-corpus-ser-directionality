#!/usr/bin/env python3
"""Evaluate trained Phase 2 checkpoints on one or more split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from src.data.datasets import ID_TO_EMOTION, build_loader
from src.models import build_model
from src.training.metrics import (
    compute_classification_metrics,
    export_analysis_figures,
    export_confusion_matrix,
)
from src.training.utils import device_from_config, ensure_dir, load_yaml, safe_torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SER checkpoint with YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to exp/<name>/best.ckpt",
    )
    return parser.parse_args()


@torch.no_grad()
def run_eval(model: torch.nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        waveforms = batch["waveforms"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        logits = model(waveforms, lengths, domain_ids=None)["logits"]
        preds = torch.argmax(logits, dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    metrics = compute_classification_metrics(y_true_np, y_pred_np)
    metrics["num_samples"] = int(len(y_true))
    metrics["y_true"] = y_true_np
    metrics["y_pred"] = y_pred_np
    return metrics


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    exp_name = config["experiment"]["name"]
    exp_root = Path(config["experiment"].get("output_dir", "exp")) / exp_name
    ensure_dir(exp_root)
    device = device_from_config(config)

    ckpt_path = args.checkpoint or (exp_root / "best.ckpt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = safe_torch_load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("config")
    if isinstance(saved_cfg, dict) and isinstance(saved_cfg.get("model"), dict):
        merged_model = {**config.get("model", {}), **saved_cfg["model"]}
        config = {**config, "model": merged_model}

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_sets = config.get("evaluation", {}).get("sets", [])
    if not eval_sets:
        # Default to test split from training config.
        test_manifest = config["data"].get("test_manifest")
        if not test_manifest:
            raise ValueError("No evaluation.sets provided and no data.test_manifest found.")
        eval_sets = [{"name": "test", "manifest": test_manifest}]

    rows = []
    label_ids = sorted(ID_TO_EMOTION.keys())
    label_names = [ID_TO_EMOTION[i] for i in label_ids]
    for entry in eval_sets:
        name = str(entry["name"])
        manifest = str(entry["manifest"])
        loader = build_loader(
            manifest_path=manifest,
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
        metrics = run_eval(model, loader, device)
        export_confusion_matrix(
            y_true=metrics["y_true"],
            y_pred=metrics["y_pred"],
            labels=label_ids,
            label_names=label_names,
            output_prefix=exp_root / f"confusion_eval_{name}",
        )
        export_analysis_figures(
            y_true=metrics["y_true"],
            y_pred=metrics["y_pred"],
            labels=label_ids,
            label_names=label_names,
            output_dir=exp_root / "analysis",
        )
        rows.append(
            {
                "eval_set": name,
                "manifest": manifest,
                "uar": metrics["uar"],
                "wa": metrics["wa"],
                "macro_f1": metrics["macro_f1"],
                "num_samples": metrics["num_samples"],
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(exp_root / "evaluation_results.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
