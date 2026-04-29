# Embedding Diagnostics Dashboard Metrics

All metrics below are computed on 2D projected plotting coordinates only.
They are visualization-space diagnostics, not direct claims about the original high-dimensional embeddings.

## Metric definitions

- **source_target_centroid_distance**: Euclidean distance between source and target domain centroids in 2D.
- **source_target_overlap_proxy**: Fraction of all points that lie inside both domain 95% Gaussian ellipses (overlap proxy).
- **within_emotion_compactness**: Mean Euclidean distance from each point to its emotion centroid (lower means tighter clusters).
- **between_emotion_centroid_separation**: Mean pairwise Euclidean distance among emotion centroids (higher means better separation).
- **domain_separability_proxy**: Fraction of points closer to their own domain centroid than to the other domain centroid.
- **emotion_separability_proxy**: Ratio = between_emotion_centroid_separation / within_emotion_compactness.

## Reading guidance

- For compactness, lower is better.
- For separation/separability, higher is better.
- Interpret trends as 2D visualization diagnostics only.