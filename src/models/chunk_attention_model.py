"""Chunk-level attention model without domain adaptation."""

from __future__ import annotations

import torch
from torch import nn

from src.features.audio_features import build_frontend
from src.models.pooling import AttentionPool1D


class ChunkAttentionModel(nn.Module):
    """
    Dynamic chunk-level model:
      1) Extract frame-level features.
      2) Build overlapping chunks.
      3) Mean-pool each chunk to chunk embeddings.
      4) Attention-pool chunk embeddings to utterance embedding.
      5) Classify emotion from utterance embedding.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.frontend = build_frontend(config)
        hidden = int(self.frontend.output_dim)  # type: ignore[attr-defined]
        self.chunk_size_sec = float(config["model"].get("chunk_size_sec", 1.0))
        self.chunk_overlap_sec = float(config["model"].get("chunk_overlap_sec", 0.5))
        self.num_classes = int(config["model"].get("num_classes", 4))
        dropout = float(config["model"].get("dropout", 0.2))

        self.chunk_pool = AttentionPool1D(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_classes),
        )

    def _chunk_indices(self, num_frames: int, frame_hz: float) -> list[tuple[int, int]]:
        chunk_frames = max(1, int(round(self.chunk_size_sec * frame_hz)))
        step_sec = max(1e-3, self.chunk_size_sec - self.chunk_overlap_sec)
        step_frames = max(1, int(round(step_sec * frame_hz)))
        if num_frames <= chunk_frames:
            return [(0, max(1, num_frames))]
        spans: list[tuple[int, int]] = []
        start = 0
        while start < num_frames:
            end = min(num_frames, start + chunk_frames)
            spans.append((start, end))
            if end >= num_frames:
                break
            start += step_frames
        return spans

    def _build_chunk_embeddings(
        self, features: torch.Tensor, lengths: torch.Tensor, frame_hz: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_chunks: list[torch.Tensor] = []
        chunk_lengths: list[int] = []
        for i in range(features.size(0)):
            valid_t = int(lengths[i].item())
            seq = features[i, :valid_t]
            spans = self._chunk_indices(valid_t, frame_hz)
            chunk_embs = []
            for s, e in spans:
                chunk_emb = seq[s:e].mean(dim=0)
                chunk_embs.append(chunk_emb)
            stacked = torch.stack(chunk_embs, dim=0)
            batch_chunks.append(stacked)
            chunk_lengths.append(stacked.size(0))

        max_chunks = max(chunk_lengths)
        dim = features.size(-1)
        padded = torch.zeros((features.size(0), max_chunks, dim), device=features.device)
        for i, ch in enumerate(batch_chunks):
            padded[i, : ch.size(0)] = ch
        return padded, torch.tensor(chunk_lengths, dtype=torch.long, device=features.device)

    def encode_pooled(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return utterance embedding, chunk embeddings, attention weights, and chunk lengths."""
        feats = self.frontend(waveforms, lengths)
        chunk_embeddings, chunk_lengths = self._build_chunk_embeddings(
            feats.features, feats.lengths, feats.frame_hz
        )
        utt_embedding, chunk_attn = self.chunk_pool(chunk_embeddings, chunk_lengths)
        return utt_embedding, chunk_embeddings, chunk_attn, chunk_lengths

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del domain_ids  # Baseline ignores domain labels; kept for a uniform call signature.
        utt_embedding, _, chunk_attn, chunk_lengths = self.encode_pooled(waveforms, lengths)
        logits = self.classifier(utt_embedding)
        return {
            "logits": logits,
            "pooled": utt_embedding,
            "chunk_attention": chunk_attn,
            "chunk_lengths": chunk_lengths,
        }
