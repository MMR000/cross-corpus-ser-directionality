## Figure Layout Fixes (v2, Shared Style)

All figure scripts were refactored to use a shared plotting style in `scripts/plot_style.py`:

- `figure.dpi=160`, `savefig.dpi=300`
- `axes.titlesize=18`, `axes.labelsize=15`
- `xtick.labelsize=12`, `ytick.labelsize=12`
- `legend.fontsize=11`
- `constrained_layout=True` where possible
- save with `bbox_inches="tight"` and export both PNG/PDF

### fig_slope_stage_comparison
- Replaced script-like labels with publication labels:
  - `IEMOCAP→PODCAST`
  - `PODCAST→IEMOCAP`
  - `IEMOCAP+CREMA-D→PODCAST`
  - `Baseline`, `Stage 1`, `Stage 2`
- Reduced title verbosity and kept x-ticks at 0°.

### fig_classwise_delta_heatmap
- Increased size to heatmap target size.
- Shortened column labels to:
  - `i2p S1-B`, `i2p S2-S1`, `p2i S1-B`, `p2i S2-S1`
- Set x-tick rotation to ~25° (not 90°), y-ticks to 0°.
- Increased colorbar label padding for readability.
- Kept per-cell numeric annotation.

### fig_i2p_confusion_paired
- Applied paired-confusion target size with constrained layout.
- Kept y-ticks horizontal and x-ticks low rotation (~20°).
- Added shared colorbar with extra label padding.
- Shortened panel titles.

### fig_target_fraction_ablation / fig_source_fraction_ablation
- Kept gain-oriented y-axis definitions (delta plots).
- Reduced title verbosity and improved spacing.
- Ensured non-vertical x-tick labels and readable legends.

### fig_multiseed_robustness
- Replaced bar chart with errorbar point plot (mean ± std).
- Kept direction/model labels readable with low rotation (~20°).

### fig_stage2_ablation_i2p
- Replaced raw-UAR bar chart with delta-vs-Stage1 chart.
- Added zero reference line and shorter title.

### fig_embedding_chunk_baseline_v2 / fig_embedding_chunk_domain_utt_v2
- Kept only black `X` centroids.
- Removed in-plot centroid text labels.
- Reduced point size and increased transparency (`s≈11`, `alpha≈0.30`).
- Kept legend-based interpretation only.
- Preserved reproducibility exports:
  - point CSVs
  - emotion/domain centroid CSVs
