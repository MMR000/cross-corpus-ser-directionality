## Stage 1 Final Summary

This final Stage 1 report is generated from existing `exp/` outputs only (no retraining).

- Stage 1 (utterance-level domain adversarial learning) helps slightly for acted -> natural:
  - `iemocap_to_podcast`: UAR `0.3615 -> 0.3688` (`+0.0073`, `+2.02%`)
  - `iemocap_plus_cremad_to_podcast`: UAR `0.3465 -> 0.3502` (`+0.0037`, `+1.08%`)
- Stage 1 hurts for natural -> acted:
  - `podcast_to_iemocap`: UAR `0.4262 -> 0.4090` (`-0.0172`, `-4.04%`)
- Stage 1 is not sufficient as a universal solution across cross-corpus directions.

Artifacts:

- `reports/stage1_final_comparison.csv`
- `reports/stage1_final_comparison_rounded.csv`
