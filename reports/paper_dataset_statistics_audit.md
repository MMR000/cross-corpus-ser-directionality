# Dataset Statistics Audit

## Source of truth
- Primary: `/home/mmr/emo_paper/ser_project/data/manifests/all_metadata_processed.csv` (harmonized labels in `emotion`, corpus via `dataset`, split via `original_split`)
- Cross-check: `/home/mmr/emo_paper/ser_project/reports/paper_dataset_statistics.csv`
- Cross-check: `/home/mmr/emo_paper/ser_project/reports/paper_dataset_statistics_by_split.csv`

## Included corpora (current paper setting)
- IEMOCAP
- PODCAST (includes MSP-PODCAST naming)
- CREMA-D

## Disagreement check: per-corpus existing report vs recomputed
- `IEMOCAP`: DISAGREE -> angry_count: recomputed=1269 existing=1226; happy_count: recomputed=2632 existing=2546; sad_count: recomputed=1250 existing=1149; neutral_count: recomputed=1726 existing=1685; total_count: recomputed=6877 existing=6606
- `PODCAST`: match
- `CREMA-D`: match

## Disagreement check: split-wise existing report vs recomputed
- `IEMOCAP` `train`: DISAGREE -> angry_count: recomputed=1269 existing=790; happy_count: recomputed=2632 existing=1490; sad_count: recomputed=1250 existing=571; neutral_count: recomputed=1726 existing=997
- `PODCAST` `train`: DISAGREE -> angry_count: recomputed=8000 existing=5600; happy_count: recomputed=8000 existing=5600; sad_count: recomputed=8000 existing=5600; neutral_count: recomputed=8000 existing=5600
- `CREMA-D` `train`: DISAGREE -> angry_count: recomputed=1271 existing=879; happy_count: recomputed=1271 existing=879; sad_count: recomputed=1271 existing=879; neutral_count: recomputed=1087 existing=752
