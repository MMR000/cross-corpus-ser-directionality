"""Pooling layers for utterance and chunk aggregation."""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    max_t = sequence.size(1)
    idx = torch.arange(max_t, device=sequence.device).unsqueeze(0)
    mask = (idx < lengths.unsqueeze(1)).float().unsqueeze(-1)
    summed = (sequence * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


class AttentionPool1D(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(input_dim, 1)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_t = sequence.size(1)
        idx = torch.arange(max_t, device=sequence.device).unsqueeze(0)
        valid = idx < lengths.unsqueeze(1)
        scores = self.scorer(sequence).squeeze(-1)
        scores = scores.masked_fill(~valid, -1e9)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), sequence).squeeze(1)
        return pooled, attn
