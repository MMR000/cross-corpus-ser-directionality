# Structured Latent Affect Modeling for Cross-Corpus SER

This repository contains the experimental pipeline for the paper:

**Structured Latent Affect Modeling for Cross-Corpus Speech Emotion Recognition via Dynamic Chunk-Level Temporal Learning**

The current implementation includes the full Phase 1 data pipeline scaffold:
- dataset schema inspection
- 4-class emotion normalization
- unified metadata + raw audio export
- audio preprocessing to 16 kHz mono
- in-corpus and cross-corpus split generation

## Project Structure

```text
ser_project/
  configs/
  data/
    raw/
    processed/
    manifests/
    splits/
  exp/
  reports/
  scripts/
    inspect_datasets.py
    prepare_datasets.py
    preprocess_audio.py
    create_splits.py
  src/
    data/
      emotion_mapping.py
      manifest_utils.py
```

## Environment Setup

```bash
cd ser_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets Used

- `AbstractTTS/IEMOCAP`
- `AbstractTTS/PODCAST`
- `AbstractTTS/CREMA-D`

If a dataset fails to load, scripts continue with warnings and process available datasets.

## Phase 1 Commands

Run from `ser_project/`:

```bash
PYTHONPATH=. python scripts/inspect_datasets.py --report-path reports/dataset_schema_report.txt
```

```bash
PYTHONPATH=. python scripts/prepare_datasets.py --project-root .
```

```bash
PYTHONPATH=. python scripts/preprocess_audio.py --project-root . --target-sr 16000 --min-duration 1.0 --max-duration 12.0
```

```bash
PYTHONPATH=. python scripts/create_splits.py --project-root . --seed 42
```

## Generated Outputs

### Reports
- `reports/dataset_schema_report.txt`
- `reports/label_mapping_report.csv`
- `reports/label_mapping_report.txt`
- `reports/data_preparation_summary.txt`
- `reports/dataset_statistics.csv`
- `reports/dataset_statistics.txt`
- `reports/split_summary.txt`

### Manifests
- `data/manifests/iemocap_metadata.csv`
- `data/manifests/podcast_metadata.csv`
- `data/manifests/crema_d_metadata.csv`
- `data/manifests/all_metadata.csv`
- `data/manifests/all_metadata_processed.csv`

### Split Files
- `data/splits/protocol_a_in_corpus/*`
- `data/splits/protocol_b_one_to_one/*`
- `data/splits/protocol_c_multi_source/*`

## Re-runnable Behavior

- Existing exported audio is reused unless `--overwrite` is set.
- Scripts create parent directories automatically.
- Split generation is deterministic given `--seed`.

## Next Implementation Phases

- feature extraction (`log-Mel`, `wav2vec2`/`XLS-R`)
- baseline models (A/B/C groups)
- proposed structured latent affect model
- training, evaluation, stress, ablation, and analysis suites
- YAML config system for reproducible experiments

## Phase 2: Baseline Training and Evaluation

Implemented in this phase:
- dataset loaders from `data/manifests/all_metadata_processed.csv` + `data/splits/...`
- feature frontends: `log-Mel`, `wav2vec2`
- models:
  - `wav2vec2 + mean pooling`
  - `wav2vec2 + attention pooling`
  - `chunk-level attention` (no domain adaptation)
- shared trainer with:
  - UAR / WA / Macro F1
  - checkpoint saving (`best.ckpt`)
  - CSV logs (`train_log.csv`, `final_metrics.csv`)
  - confusion matrices (`.csv` and `.png`)
- evaluation script for in-corpus and one-to-one cross-corpus protocols

### Core scripts

```bash
PYTHONPATH=. python scripts/train.py --config <config_path>
PYTHONPATH=. python scripts/evaluate.py --config <config_path>
```

### Phase 2 Configs

All configs are under `configs/phase2/`.

### Runnable training commands (12 experiments)

```bash
PYTHONPATH=. python scripts/train.py --config configs/phase2/iemocap_in_corpus_meanpool.yaml
PYTHONPATH=. python scripts/train.py --config configs/phase2/iemocap_in_corpus_attnpool.yaml
PYTHONPATH=. python scripts/train.py --config configs/phase2/iemocap_in_corpus_chunk.yaml

PYTHONPATH=. python scripts/train.py --config configs/phase2/podcast_in_corpus_meanpool.yaml
PYTHONPATH=. python scripts/train.py --config configs/phase2/podcast_in_corpus_attnpool.yaml
PYTHONPATH=. python scripts/train.py --config configs/phase2/podcast_in_corpus_chunk.yaml

PYTHONPATH=. python scripts/train.py --config configs/phase2/iemocap_to_podcast_meanpool.yaml
PYTHONPATH=. python scripts/train.py --config configs/phase2/iemocap_to_podcast_attnpool.yaml
PYTHONPATH=. python scripts/train.py --config configs/phase2/iemocap_to_podcast_chunk.yaml

PYTHONPATH=. python scripts/train.py --config configs/phase2/podcast_to_iemocap_meanpool.yaml
PYTHONPATH=. python scripts/train.py --config configs/phase2/podcast_to_iemocap_attnpool.yaml
PYTHONPATH=. python scripts/train.py --config configs/phase2/podcast_to_iemocap_chunk.yaml
```

### Optional explicit evaluation commands

```bash
PYTHONPATH=. python scripts/evaluate.py --config configs/phase2/iemocap_in_corpus_meanpool.yaml
PYTHONPATH=. python scripts/evaluate.py --config configs/phase2/iemocap_to_podcast_meanpool.yaml
```
