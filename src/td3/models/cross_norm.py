"""
CrossNorm module for TD3 networks.

CrossNorm swaps per-sample normalization statistics across the batch to reduce
covariate shift for off-policy training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CrossNorm1d(nn.Module):
    """Cross-normalization"""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.view(1, -1)
        elif x.dim() > 2:
            # Flatten to (batch, features) for indicator stacks or sequence inputs.
            x = x.view(x.size(0), -1)

        if x.dim() != 2:
            raise ValueError("CrossNorm1d expects a 2D tensor of shape (batch, features)")
        if x.size(1) != self.num_features:
            raise ValueError(
                f"CrossNorm1d expected {self.num_features} features, got {x.size(1)}"
            )

        # Per-sample statistics across features (LayerNorm-style).
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        if self.training and x.size(0) > 1:
            # Swap normalization stats across samples to reduce covariate shift.
            perm = torch.randperm(x.size(0), device=x.device)
            mean = mean[perm]
            std = std[perm]

        x_norm = (x - mean) / std
        # Learnable affine transform, same shape as features.
        return x_norm * self.weight + self.bias

