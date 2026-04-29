## Final Baseline + Stage1 + Stage2 Overview

Consolidated from existing `exp/*/final_metrics.csv` outputs (no retraining).

- Best in-corpus model: `iemocap_in_corpus_attnpool` with test UAR `0.5165`.
- Best cross-corpus baseline (among meanpool/attnpool/chunk and multi-source baselines): `podcast_to_iemocap_chunk` with test UAR `0.4262`.
- Multi-source helps for acted->natural? No (`iemocap_plus_cremad_to_podcast_chunk=0.3465` vs `iemocap_to_podcast_chunk=0.3615`).
- Stage 1 helps? Mixed: yes for `iemocap->podcast` (`0.3688` vs chunk `0.3615`), no for `podcast->iemocap` (`0.4090` vs chunk `0.4262`).
- Stage 2 helps? Not universally: hurts for `iemocap->podcast` default (`0.3504`) and `podcast->iemocap` (`0.3950`), but helps for `iemocap+crema_d->podcast` (`0.3556` vs Stage1 `0.3502`).
- Final recommendation for `iemocap_to_podcast`: use Stage 1 `chunk_domain_utt` as default (`0.3688`); if Stage 2 must be used, prefer weak chunk weighting (`both_weakchunk=0.3508`) over default Stage 2 (`0.3504`).
- Final recommendation for `podcast_to_iemocap`: keep plain chunk baseline (`0.4262`); avoid current Stage 1/2 adversarial variants for this direction.
