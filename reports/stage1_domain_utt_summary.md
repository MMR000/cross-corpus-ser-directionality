## Stage 1 Domain-Utt Comparison

This report compares plain chunk baselines against Stage 1 utterance-level domain-adversarial variants using existing `exp/` outputs only (no retraining).

- `iemocap_to_podcast`: `chunk_domain_utt` improves test UAR from `0.3615` to `0.3688` (`+0.0073` absolute).
- `podcast_to_iemocap`: `chunk_domain_utt` decreases test UAR from `0.4262` to `0.4090` (`-0.0172` absolute).
- `iemocap+crema_d_to_podcast`: `chunk_domain_utt` improves test UAR from `0.3465` to `0.3502` (`+0.0037` absolute).

Files:

- `reports/stage1_domain_utt_comparison.csv` (full precision)
- `reports/stage1_domain_utt_comparison_rounded.csv` (rounded for paper/report use)
