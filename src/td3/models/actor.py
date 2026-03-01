import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from entmax import entmax15, entmax_bisect


def add_logit_noise(logits: torch.Tensor, noise_std: float, noise_clip: float) -> torch.Tensor:
    if noise_std and noise_std > 0:
        noise = torch.randn_like(logits) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        return logits + noise
    return logits


def logits_to_weights(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return entmax_bisect(logits, dim=-1, alpha=1.6)

class Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 256):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_dim)

    def forward(self, state: Tensor) -> Tensor:
        if state.dim() > 2:
            x = state.view(state.size(0), -1)
        else:
            x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))
