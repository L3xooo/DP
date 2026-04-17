"""
Critic network module for TD3 reinforcement learning agent.

Author: Peter Likavec
"""

import torch.nn as nn
import torch


class Critic(nn.Module):
    """
    Critic network that estimates the Q-value for a given state-action pair.

    The network consists of three fully connected layers with ReLU activations.
    Used as part of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """
        Perform a forward pass through the critic network.

        Args:
            state (torch.Tensor): Current environment state of shape (batch_size, state_dim).
            action (torch.Tensor): Action taken by the agent of shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Estimated Q-value of shape (batch_size, 1).
        """
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
