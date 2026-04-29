## Stage 2 Ablation (IEMOCAP -> PODCAST)

This ablation isolates chunk-level vs combined utterance+chunk domain adaptation for the hardest direction using existing Stage 2 runs.

Reference points:

- Plain chunk: `test_uar = 0.3615`
- Chunk + domain_utt (Stage 1): `test_uar = 0.3688`
- Chunk + domain_both default (Stage 2, utt=1.0/chunk=1.0): `test_uar = 0.3504`

Findings:

- **Chunk-only is consistently harmful**: all chunk-only variants (`w=0.1/0.3/0.5`) are around `0.3429-0.3433` UAR, below both plain chunk and Stage 1 domain_utt.
- **Reducing chunk weight in the combined setup helps only marginally**:
  - `both_weakchunk (utt=1.0, chunk=0.1)` reaches `0.3508`, slightly above Stage 2 default (`+0.0004`) but still below plain chunk (`-0.0106`) and Stage 1 (`-0.0179`).
  - `both_midchunk (utt=1.0, chunk=0.3)` is effectively identical to Stage 2 default (`0.3504`).
- **Conclusion for this direction**: degradation is primarily tied to introducing chunk-level domain alignment itself; weakening chunk weight only slightly mitigates it and does not recover Stage 1 or plain-chunk performance.

Artifacts:

- `reports/stage2_ablation_iemocap_to_podcast.csv`
- `reports/stage2_ablation_iemocap_to_podcast_rounded.csv`
