#!/usr/bin/env python3
"""Generate implementation-details and validation-protocol audit artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.training.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs" / "phase2"


def _infer_pooling(model_name: str) -> str:
    name = model_name.lower()
    if "mean" in name:
        return "mean"
    if "attention" in name and "chunk" not in name:
        return "attention"
    if "chunk" in name:
        return "chunk_attention"
    return "UNKNOWN"


def _infer_discriminator(model_name: str) -> str:
    name = model_name.lower()
    if "domain_utt" in name:
        return "utterance_domain_classifier"
    if "domain_both" in name:
        return "utterance_and_chunk_domain_classifier"
    return "none"


def generate_implementation_details() -> list[Path]:
    rows = []
    for cfg_path in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_yaml(cfg_path)
        feat = cfg.get("features", {})
        model = cfg.get("model", {})
        train = cfg.get("train", {})
        data = cfg.get("data", {})
        rows.append(
            {
                "config_file": cfg_path.name,
                "experiment_name": cfg.get("experiment", {}).get("name", cfg_path.stem),
                "model_name": model.get("name", "UNKNOWN"),
                "encoder_backbone_name": feat.get("type", "UNKNOWN"),
                "wav2vec2_variant": feat.get("model_name", "UNKNOWN"),
                "frozen_vs_finetuned": "frozen" if feat.get("freeze", True) else "fine_tuned",
                "chunk_settings": f"size={model.get('chunk_size_sec', 1.0)},overlap={model.get('chunk_overlap_sec', 0.5)}"
                if "chunk" in str(model.get("name", "")).lower()
                else "not_applicable",
                "chunk_count_or_chunk_length": model.get("chunk_size_sec", "UNKNOWN")
                if "chunk" in str(model.get("name", "")).lower()
                else "not_applicable",
                "pooling_type": _infer_pooling(str(model.get("name", "UNKNOWN"))),
                "discriminator_type": _infer_discriminator(str(model.get("name", "UNKNOWN"))),
                "hidden_dimensions": feat.get("hidden_dim", "UNKNOWN"),
                "optimizer": "AdamW",  # trainer default
                "learning_rate": train.get("lr", "UNKNOWN"),
                "batch_size": train.get("batch_size", "UNKNOWN"),
                "num_epochs": train.get("epochs", "UNKNOWN"),
                "seed": cfg.get("seed", "UNKNOWN"),
                "lambda_or_domain_loss_weights": model.get("domain_loss_weight", model.get("utterance_domain_loss_weight", "UNKNOWN")),
                "utterance_domain_loss_weight": model.get("utterance_domain_loss_weight", model.get("domain_loss_weight", 0.0)),
                "chunk_domain_loss_weight": model.get("chunk_domain_loss_weight", 0.0),
                "grl_alpha": model.get("domain_grl_alpha", 1.0),
                "chunk_grl_alpha": model.get("chunk_domain_grl_alpha", model.get("domain_grl_alpha", 1.0)),
                "early_stopping_setting": "UNKNOWN (fixed epochs; best checkpoint by dev UAR)",
                "validation_setting": "explicit dev_manifest" if data.get("dev_manifest") else "train CSV split column fallback",
                "max_duration_sec": data.get("max_duration_sec", "UNKNOWN"),
                "dropout": model.get("dropout", 0.2),
                "num_classes": model.get("num_classes", 4),
            }
        )
    df = pd.DataFrame(rows).sort_values("experiment_name").reset_index(drop=True)
    p1 = REPORTS / "paper_implementation_details.csv"
    p2 = REPORTS / "paper_implementation_details.md"
    df.to_csv(p1, index=False)
    lines = [
        "# Paper Implementation Details",
        "",
        "Values are parsed from the YAML configs used for experiments, with selected code-level defaults filled in when not explicit.",
        "",
        "Code-derived defaults used:",
        "- optimizer = `AdamW` from `src/training/trainer.py` (`SERTrainer.__init__`)",
        "- dropout default = `0.2` in model classes",
        "- num_classes default = `4` in model classes",
        "- wav2vec2 freeze default = `True` in `src/features/audio_features.py` (`build_frontend` / `Wav2Vec2Frontend`)",
        "- GRL alpha default = `1.0` when not explicit",
        "",
        f"Total configs audited: {len(df)}",
    ]
    p2.write_text("\n".join(lines), encoding="utf-8")
    return [p1, p2]


def generate_validation_protocol_audit() -> list[Path]:
    lines = [
        "# Paper Validation Protocol Audit",
        "",
        "## Early stopping / model selection split",
        "- Dev-set evaluation is performed in `src/training/trainer.py` within `SERTrainer.fit()` via `evaluate_loader(dev_loader, 'dev')`.",
        "- Best checkpoint selection uses `val_uar` only: `if val_metrics['uar'] > best_uar:` in `SERTrainer.fit()`.",
        "- The checkpoint saved is `best.ckpt` via `_save_checkpoint()` in `src/training/trainer.py`.",
        "",
        "## What data forms the dev loader",
        "- Dev loader construction is implemented in `scripts/train.py` within `_build_train_and_dev_loaders()`.",
        "- If `data.dev_manifest` is present, that file is used directly as the dev split.",
        "- Otherwise, the code falls back to splitting the combined train CSV using `split_train_dev_from_combined()` from `src/data/datasets.py`, which expects a `split` column with `train` and `dev` rows.",
        "",
        "## Cross-corpus UDA settings",
        "- In single-source UDA, the target unlabeled loader is built in `scripts/train.py` within `_build_target_unlabeled_loader()`.",
        "- That target loader concatenates `uda.target_train_manifest` and, if present, `uda.target_dev_manifest` for domain-classifier training only.",
        "- Emotion supervision in UDA is source-only: `src_out['logits']` is compared with `src_labels` in `SERTrainer._step_train_uda()`.",
        "- Domain loss uses both source and target batches in `SERTrainer._step_train_uda()` by concatenating domain logits/labels.",
        "",
        "## Are target labels used for model selection?",
        "- The dev loader used for checkpoint selection is the `dev_loader` created by `_build_train_and_dev_loaders()`.",
        "- For the paper's cross-corpus UDA configs, `data.dev_manifest` is empty, so validation comes from the source-side combined train/dev CSV split, not from target labels.",
        "- `uda.target_dev_manifest` is included in the unlabeled domain-training pool, but no target emotion labels are consumed in `_step_train_uda()`.",
        "- Therefore the effective protocol is source-validation model selection with target-unlabeled domain adaptation, not target-label-assisted checkpoint selection.",
        "",
        "## Checkpoint metric",
        "- Checkpoint selection metric: `val_uar`.",
        "- Implemented in `src/training/trainer.py` (`SERTrainer.fit`).",
        "",
        "## Protocol classification",
        "- In-corpus: standard dev-set model selection on the corpus dev split.",
        "- Cross-corpus UDA: source-validation-only checkpoint selection, with target unlabeled train/dev data used for domain loss but not emotion validation.",
        "",
        "## Exact code locations",
        "- `scripts/train.py` -> `_build_train_and_dev_loaders()`",
        "- `scripts/train.py` -> `_build_target_unlabeled_loader()`",
        "- `src/training/trainer.py` -> `SERTrainer._step_train_uda()`",
        "- `src/training/trainer.py` -> `SERTrainer.evaluate_loader()`",
        "- `src/training/trainer.py` -> `SERTrainer.fit()`",
        "- `src/data/datasets.py` -> `split_train_dev_from_combined()`",
    ]
    p = REPORTS / "paper_validation_protocol_audit.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return [p]


def main() -> None:
    outputs = []
    outputs.extend(generate_implementation_details())
    outputs.extend(generate_validation_protocol_audit())
    print("Generated:")
    for path in outputs:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

