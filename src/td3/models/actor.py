"""
Actor network module for TD3 reinforcement learning agent.

Author: Peter Likavec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Actor(nn.Module):
    """
    Actor network for TD3, mapping states to action probabilities.

    The network consists of three fully connected layers with ReLU activations.
    Used as part of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 256):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_dim)

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # vystupna vrstva zabezpeci, ze su hodnoty od 0 do 1
        return torch.sigmoid(self.out(x))
