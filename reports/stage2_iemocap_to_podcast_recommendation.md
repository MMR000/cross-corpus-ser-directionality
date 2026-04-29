## Stage 2 Recommendation (IEMOCAP -> PODCAST)

Based on `reports/stage2_ablation_iemocap_to_podcast.csv`, none of the tested Stage 2 chunk-domain variants surpass the Stage 1 utterance-only setup for this direction.

Reference performance (test UAR):

- Plain chunk: `0.3615`
- Chunk + domain_utt (Stage 1): `0.3688`  **(best overall)**
- Chunk + domain_both default (utt=1.0, chunk=1.0): `0.3504`

Best within Stage 2 ablation variants:

- `both_weakchunk` (utt=1.0, chunk=0.1): `0.3508`  
  - vs Stage 2 default: `+0.0004`
  - vs plain chunk: `-0.0106`
  - vs Stage 1 domain_utt: `-0.0179`

Practical recommendation:

1. Use **Stage 1 (`chunk + domain_utt`)** as the primary setting for `iemocap_to_podcast`.
2. Treat Stage 2 as **not recommended** for this direction under current design.
3. If Stage 2 must be retained for completeness, use `both_weakchunk` (utt=1.0, chunk=0.1) as the least harmful Stage 2 option.
