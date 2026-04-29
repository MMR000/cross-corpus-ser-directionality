"""Gradient reversal for domain-adversarial training."""

from __future__ import annotations

import torch
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Reverse gradients flowing through this tensor by factor ``alpha``."""
    return _GradientReversalFunction.apply(x, alpha)
