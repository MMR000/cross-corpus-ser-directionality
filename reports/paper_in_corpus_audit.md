# In-Corpus Results Audit

## Recognized in-corpus runs
- `iemocap_in_corpus_attnpool` | corpus=IEMOCAP | pooling=AttnPool | source=summary.csv | path=`/home/mmr/emo_paper/ser_project/exp/iemocap_in_corpus_attnpool/summary.csv`
- `iemocap_in_corpus_chunk` | corpus=IEMOCAP | pooling=Chunk | source=summary.csv | path=`/home/mmr/emo_paper/ser_project/exp/iemocap_in_corpus_chunk/summary.csv`
- `iemocap_in_corpus_meanpool` | corpus=IEMOCAP | pooling=MeanPool | source=summary.csv | path=`/home/mmr/emo_paper/ser_project/exp/iemocap_in_corpus_meanpool/summary.csv`
- `podcast_in_corpus_attnpool` | corpus=PODCAST | pooling=AttnPool | source=summary.csv | path=`/home/mmr/emo_paper/ser_project/exp/podcast_in_corpus_attnpool/summary.csv`
- `podcast_in_corpus_chunk` | corpus=PODCAST | pooling=Chunk | source=summary.csv | path=`/home/mmr/emo_paper/ser_project/exp/podcast_in_corpus_chunk/summary.csv`
- `podcast_in_corpus_meanpool` | corpus=PODCAST | pooling=MeanPool | source=summary.csv | path=`/home/mmr/emo_paper/ser_project/exp/podcast_in_corpus_meanpool/summary.csv`

## Excluded runs and reasons
- `iemocap_plus_cremad_to_podcast_attnpool`: source != target (not in-corpus run)
- `iemocap_plus_cremad_to_podcast_chunk`: source != target (not in-corpus run)
- `iemocap_plus_cremad_to_podcast_chunk_domain_both`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_plus_cremad_to_podcast_chunk_domain_utt`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_plus_cremad_to_podcast_meanpool`: source != target (not in-corpus run)
- `iemocap_to_podcast_attnpool`: source != target (not in-corpus run)
- `iemocap_to_podcast_chunk`: source != target (not in-corpus run)
- `iemocap_to_podcast_chunk_domain_both`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_both_both_midchunk`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_both_both_weakchunk`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_both_chunk_only_w01`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_both_chunk_only_w03`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_both_chunk_only_w05`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_utt`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_utt_seed13`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_utt_seed7`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_utt_srcfrac25`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_utt_srcfrac50`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_utt_tgtfrac25`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_utt_tgtfrac50`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_domain_utt_tgtfrac75`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_seed13`: source != target (not in-corpus run); pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_seed7`: source != target (not in-corpus run); pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_srcfrac25`: source != target (not in-corpus run); pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_chunk_srcfrac50`: source != target (not in-corpus run); pooling type not one of MeanPool/AttnPool/Chunk
- `iemocap_to_podcast_meanpool`: source != target (not in-corpus run)
- `podcast_to_iemocap_attnpool`: source != target (not in-corpus run)
- `podcast_to_iemocap_chunk`: source != target (not in-corpus run)
- `podcast_to_iemocap_chunk_domain_both`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `podcast_to_iemocap_chunk_domain_utt`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `podcast_to_iemocap_chunk_domain_utt_seed13`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `podcast_to_iemocap_chunk_domain_utt_seed7`: source != target (not in-corpus run); adaptation/stage variant excluded; pooling type not one of MeanPool/AttnPool/Chunk
- `podcast_to_iemocap_chunk_seed13`: source != target (not in-corpus run); pooling type not one of MeanPool/AttnPool/Chunk
- `podcast_to_iemocap_chunk_seed7`: source != target (not in-corpus run); pooling type not one of MeanPool/AttnPool/Chunk
- `podcast_to_iemocap_meanpool`: source != target (not in-corpus run)

## Coverage check (MeanPool / AttnPool / Chunk)
- `IEMOCAP`: complete coverage
- `PODCAST`: complete coverage
- `CREMA-D`: missing MeanPool, AttnPool, Chunk

## Extracted metric source paths
- `iemocap_in_corpus_attnpool` -> `/home/mmr/emo_paper/ser_project/exp/iemocap_in_corpus_attnpool/summary.csv` (summary.csv)
- `iemocap_in_corpus_chunk` -> `/home/mmr/emo_paper/ser_project/exp/iemocap_in_corpus_chunk/summary.csv` (summary.csv)
- `iemocap_in_corpus_meanpool` -> `/home/mmr/emo_paper/ser_project/exp/iemocap_in_corpus_meanpool/summary.csv` (summary.csv)
- `podcast_in_corpus_attnpool` -> `/home/mmr/emo_paper/ser_project/exp/podcast_in_corpus_attnpool/summary.csv` (summary.csv)
- `podcast_in_corpus_chunk` -> `/home/mmr/emo_paper/ser_project/exp/podcast_in_corpus_chunk/summary.csv` (summary.csv)
- `podcast_in_corpus_meanpool` -> `/home/mmr/emo_paper/ser_project/exp/podcast_in_corpus_meanpool/summary.csv` (summary.csv)
