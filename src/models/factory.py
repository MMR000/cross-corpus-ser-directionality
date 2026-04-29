"""Model factory for Phase 2 experiments."""

from __future__ import annotations

from torch import nn

from src.models.chunk_attention_domain_utt import ChunkAttentionDomainUttModel
from src.models.chunk_attention_model import ChunkAttentionModel
from src.models.wav2vec_models import Wav2Vec2AttentionPoolingModel, Wav2Vec2MeanPoolingModel


def build_model(config: dict) -> nn.Module:
    name = str(config["model"]["name"]).lower()
    if name in {"wav2vec2_mean_pool", "mean_pool"}:
        return Wav2Vec2MeanPoolingModel(config)
    if name in {"wav2vec2_attention_pool", "attention_pool"}:
        return Wav2Vec2AttentionPoolingModel(config)
    if name in {"chunk_attention", "chunk_level_attention"}:
        return ChunkAttentionModel(config)
    if name in {
        "chunk_attention_domain_utt",
        "chunk_domain_utt",
        "chunk_attention_domain_both",
        "chunk_domain_both",
    }:
        return ChunkAttentionDomainUttModel(config)
    raise ValueError(f"Unsupported model name: {name}")
