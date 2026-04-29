# Paper Validation Protocol Audit

## Early stopping / model selection split
- Dev-set evaluation is performed in `src/training/trainer.py` within `SERTrainer.fit()` via `evaluate_loader(dev_loader, 'dev')`.
- Best checkpoint selection uses `val_uar` only: `if val_metrics['uar'] > best_uar:` in `SERTrainer.fit()`.
- The checkpoint saved is `best.ckpt` via `_save_checkpoint()` in `src/training/trainer.py`.

## What data forms the dev loader
- Dev loader construction is implemented in `scripts/train.py` within `_build_train_and_dev_loaders()`.
- If `data.dev_manifest` is present, that file is used directly as the dev split.
- Otherwise, the code falls back to splitting the combined train CSV using `split_train_dev_from_combined()` from `src/data/datasets.py`, which expects a `split` column with `train` and `dev` rows.

## Cross-corpus UDA settings
- In single-source UDA, the target unlabeled loader is built in `scripts/train.py` within `_build_target_unlabeled_loader()`.
- That target loader concatenates `uda.target_train_manifest` and, if present, `uda.target_dev_manifest` for domain-classifier training only.
- Emotion supervision in UDA is source-only: `src_out['logits']` is compared with `src_labels` in `SERTrainer._step_train_uda()`.
- Domain loss uses both source and target batches in `SERTrainer._step_train_uda()` by concatenating domain logits/labels.

## Are target labels used for model selection?
- The dev loader used for checkpoint selection is the `dev_loader` created by `_build_train_and_dev_loaders()`.
- For the paper's cross-corpus UDA configs, `data.dev_manifest` is empty, so validation comes from the source-side combined train/dev CSV split, not from target labels.
- `uda.target_dev_manifest` is included in the unlabeled domain-training pool, but no target emotion labels are consumed in `_step_train_uda()`.
- Therefore the effective protocol is source-validation model selection with target-unlabeled domain adaptation, not target-label-assisted checkpoint selection.

## Checkpoint metric
- Checkpoint selection metric: `val_uar`.
- Implemented in `src/training/trainer.py` (`SERTrainer.fit`).

## Protocol classification
- In-corpus: standard dev-set model selection on the corpus dev split.
- Cross-corpus UDA: source-validation-only checkpoint selection, with target unlabeled train/dev data used for domain loss but not emotion validation.

## Exact code locations
- `scripts/train.py` -> `_build_train_and_dev_loaders()`
- `scripts/train.py` -> `_build_target_unlabeled_loader()`
- `src/training/trainer.py` -> `SERTrainer._step_train_uda()`
- `src/training/trainer.py` -> `SERTrainer.evaluate_loader()`
- `src/training/trainer.py` -> `SERTrainer.fit()`
- `src/data/datasets.py` -> `split_train_dev_from_combined()`