"""Utterance-level pooling baselines on top of shared feature frontends."""

from __future__ import annotations

import torch
from torch import nn

from src.features.audio_features import build_frontend
from src.models.pooling import AttentionPool1D, masked_mean


class Wav2Vec2MeanPoolingModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.frontend = build_frontend(config)
        hidden = int(self.frontend.output_dim)  # type: ignore[attr-defined]
        num_classes = int(config["model"].get("num_classes", 4))
        dropout = float(config["model"].get("dropout", 0.2))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del domain_ids
        feats = self.frontend(waveforms, lengths)
        pooled = masked_mean(feats.features, feats.lengths)
        logits = self.classifier(pooled)
        return {"logits": logits, "pooled": pooled}


class Wav2Vec2AttentionPoolingModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.frontend = build_frontend(config)
        hidden = int(self.frontend.output_dim)  # type: ignore[attr-defined]
        num_classes = int(config["model"].get("num_classes", 4))
        dropout = float(config["model"].get("dropout", 0.2))
        self.pool = AttentionPool1D(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del domain_ids
        feats = self.frontend(waveforms, lengths)
        pooled, attn = self.pool(feats.features, feats.lengths)
        logits = self.classifier(pooled)
        return {"logits": logits, "pooled": pooled, "attention": attn}
