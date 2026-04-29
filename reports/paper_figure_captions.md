## Paper Figure Captions (High-information v2)

### fig_slope_stage_comparison.png / .pdf
Slope chart of stage progression (baseline chunk, Stage 1, Stage 2) for three transfer directions. The x-axis denotes stage and the y-axis denotes test UAR; each line corresponds to one transfer direction.

### fig_classwise_delta_heatmap.png / .pdf
Class-wise recall delta heatmap with rows as emotion classes and columns as directional stage deltas: i2p Stage1-baseline, i2p Stage2-Stage1, p2i Stage1-baseline, and p2i Stage2-Stage1. Cells are numerically annotated.

### fig_i2p_confusion_paired.png / .pdf
Paired normalized confusion matrices for IEMOCAP→PODCAST under baseline chunk and Stage 1 (chunk_domain_utt). Axes denote true and predicted emotion labels.

### fig_embedding_chunk_baseline_v2.png / .pdf
Two-panel t-SNE embedding visualization for IEMOCAP→PODCAST chunk baseline. Panel A uses emotion colors/markers; Panel B uses domain colors/markers. Transparent points show samples; X markers show centroids.

### fig_embedding_chunk_domain_utt_v2.png / .pdf
Two-panel t-SNE embedding visualization for IEMOCAP→PODCAST chunk_domain_utt with the same styling as baseline. Transparent points show samples; X markers show centroids.

### fig_target_fraction_ablation.png / .pdf
Target-fraction ablation plotted as delta UAR relative to plain chunk baseline. X-axis is target unlabeled fraction; y-axis is gain/loss versus baseline.

### fig_source_fraction_ablation.png / .pdf
Source-fraction ablation plotted as gain curves. Left panel shows delta UAR vs plain chunk for chunk and chunk_domain_utt; right panel shows chunk_domain_utt delta UAR vs Stage1 full-source reference.

### fig_stage2_ablation_i2p.png / .pdf
Horizontal delta plot for IEMOCAP→PODCAST Stage 2 ablations, reporting each variant as ΔUAR relative to Stage 1.

### fig_multiseed_robustness.png / .pdf
Multi-seed robustness errorbar point plot showing mean UAR with standard-deviation bars for i2p_chunk, i2p_chunk_domain_utt, p2i_chunk, and p2i_chunk_domain_utt.