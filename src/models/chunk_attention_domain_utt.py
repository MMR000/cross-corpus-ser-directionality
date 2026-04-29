"""Chunk attention model with utterance/chunk domain-adversarial heads (DANN-style)."""

from __future__ import annotations

import torch
from torch import nn

from src.models.chunk_attention_model import ChunkAttentionModel
from src.models.grad_reverse import grad_reverse


class ChunkAttentionDomainUttModel(ChunkAttentionModel):
    """
    Same encoder as ``ChunkAttentionModel``; adds domain classifiers on:
    - utterance embedding (pooled)
    - chunk embeddings (optional, Stage 2)
    Classifiers are used during training when ``domain_ids`` is provided.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.num_domains = int(config["model"].get("num_domains", 0))
        self.utt_domain_grl_alpha = float(config["model"].get("domain_grl_alpha", 1.0))
        self.chunk_domain_grl_alpha = float(
            config["model"].get("chunk_domain_grl_alpha", self.utt_domain_grl_alpha)
        )
        dropout = float(config["model"].get("dropout", 0.2))
        hidden = int(self.frontend.output_dim)  # type: ignore[attr-defined]

        if self.num_domains >= 2:
            self.domain_classifier_utt = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.num_domains),
            )
            self.domain_classifier_chunk = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.num_domains),
            )
        else:
            self.domain_classifier_utt = None
            self.domain_classifier_chunk = None

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> dict:
        pooled, chunk_embeddings, chunk_attn, chunk_lengths = self.encode_pooled(waveforms, lengths)
        logits = self.classifier(pooled)
        out: dict = {
            "logits": logits,
            "pooled": pooled,
            "chunk_embeddings": chunk_embeddings,
            "chunk_attention": chunk_attn,
            "chunk_lengths": chunk_lengths,
        }
        if (
            self.training
            and domain_ids is not None
            and self.domain_classifier_utt is not None
        ):
            utt_rev = grad_reverse(pooled, self.utt_domain_grl_alpha)
            out["domain_logits"] = self.domain_classifier_utt(utt_rev)

            if self.domain_classifier_chunk is not None:
                chunk_rev = grad_reverse(chunk_embeddings, self.chunk_domain_grl_alpha)
                # [B, max_chunks, D] -> [B, max_chunks, num_domains]
                chunk_domain_logits = self.domain_classifier_chunk(chunk_rev)
                chunk_mask = (
                    torch.arange(chunk_domain_logits.size(1), device=chunk_domain_logits.device)
                    .unsqueeze(0)
                    .lt(chunk_lengths.unsqueeze(1))
                )
                out["chunk_domain_logits"] = chunk_domain_logits[chunk_mask]
                out["chunk_domain_targets"] = (
                    domain_ids.unsqueeze(1).expand(-1, chunk_domain_logits.size(1))[chunk_mask]
                )
        return out
