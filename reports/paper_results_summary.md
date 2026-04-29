## Paper Results Summary

### In-Corpus
- iemocap: **iemocap_in_corpus_attnpool (0.517)** is best test UAR.
- podcast: **podcast_in_corpus_attnpool (0.510)** is best test UAR.
- Interpretation: in-corpus performance is strongest overall; IEMOCAP favors attention pooling while podcast is tightly clustered.

### Cross-Corpus Baseline
- iemocap -> podcast: **iemocap_to_podcast_meanpool (0.363)** is best baseline.
- iemocap+crema_d -> podcast: **iemocap_plus_cremad_to_podcast_meanpool (0.358)** is best baseline.
- podcast -> iemocap: **podcast_to_iemocap_chunk (0.426)** is best baseline.
- Interpretation: cross-corpus transfer is asymmetric; natural -> acted remains easier than acted -> natural.

### Stage 1 / Stage 2
- iemocap -> podcast: best chunk-family variant is **iemocap_to_podcast_chunk_domain_utt (0.369)**.
- iemocap+crema_d -> podcast: best chunk-family variant is **iemocap_plus_cremad_to_podcast_chunk_domain_both (0.356)**.
- podcast -> iemocap: best chunk-family variant is **podcast_to_iemocap_chunk (0.426)**.
- Interpretation: Stage 1 helps selectively, while Stage 2 default can over-regularize single-source transfer; multi-source acted -> natural shows modest Stage 2 gains.

### Assets
- `reports/paper_table_in_corpus.csv`
- `reports/paper_table_cross_corpus.csv`
- `reports/paper_table_stage1_stage2.csv`
- `reports/fig_in_corpus_barplot.png`
- `reports/fig_cross_corpus_barplot.png`
- `reports/fig_stage1_stage2_barplot.png`