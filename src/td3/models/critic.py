"""
Critic network module for TD3 reinforcement learning agent.

Author: Peter Likavec
"""

import torch.nn as nn
import torch

from td3.models.cross_norm import CrossNorm1d


class Critic(nn.Module):
    """
    Critic network that estimates the Q-value for a given state-action pair.

    The network consists of three fully connected layers with ReLU activations,
    batch normalization for improved stability, and optional dropout for regularization.
    Used as part of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        dropout_rate: float = 0.0,
        normalization: str | None = "cross",
    ):
        super(Critic, self).__init__()
        # Combine state and action as input to the critic
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Normalization layers: prefer LayerNorm for small-batch RL scenarios
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

        # Optional dropout for critic hidden representations
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate and dropout_rate > 0.0 else nn.Identity()


    def forward(self, state, action):
        """
        Perform a forward pass through the critic network.

        Args:
            state (torch.Tensor): Current environment state of shape (batch_size, state_dim).
            action (torch.Tensor): Action taken by the agent of shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Estimated Q-value of shape (batch_size, 1).
        """
        x = torch.cat([state, action], dim=1)

        x = self.fc1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        q_value = self.fc3(x)
        return q_value
