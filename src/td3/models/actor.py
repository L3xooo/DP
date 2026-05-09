"""
Actor network module for TD3 reinforcement learning agent.

Author: Peter Likavec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from td3.models.cross_norm import CrossNorm1d


class Actor(nn.Module):
    """
    Actor network for TD3, mapping states to action probabilities.

    The network consists of three fully connected layers with ReLU activations.
    Used as part of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        dropout_rate: float = 0.0,
        normalization: str | None = "cross",
    ):
        """
        Actor with optional dropout and normalization layers.

        Args:
            input_dim: Dimension of flattened input state.
            action_dim: Number of action logits produced.
            hidden_size: Width of hidden layers.
            dropout_rate: Dropout probability applied after hidden layers during training.
            normalization: One of {None, 'batch', 'layer', 'cross'}; adds BatchNorm1d,
                           LayerNorm, or CrossNorm after fully-connected layers which
                           stabilizes activations across changing input distributions
                           (helps generalization across regimes).
        """
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_dim)

        self.normalization = normalization
        if normalization == "batch":
            self.norm1 = nn.BatchNorm1d(hidden_size)
            self.norm2 = nn.BatchNorm1d(hidden_size)
        elif normalization == "layer":
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
        elif normalization == "cross":
            self.norm1 = CrossNorm1d(hidden_size)
            self.norm2 = CrossNorm1d(hidden_size)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate and dropout_rate > 0.0 else nn.Identity()

    def forward(self, state: Tensor) -> Tensor:
        """
        Perform a forward pass through the Actor network.

        Args:
            state: Input state tensor of shape

        Returns:
             Action probabilities of shape (batch_size, action_dim), produced by applying
             a sigmoid activation to the output layer.
        """
        if state.dim() > 2:
            x = state.view(state.size(0), -1)
        else:
            x = state

        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        return torch.sigmoid(self.out(x))
