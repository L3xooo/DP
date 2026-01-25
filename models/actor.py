import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.logger import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

def add_logit_noise(logits: torch.Tensor, noise_std: float, noise_clip: float) -> torch.Tensor:
    if noise_std and noise_std > 0:
        noise = torch.randn_like(logits) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        return logits + noise
    return logits

def logits_to_weights(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    t = max(1e-6, float(temperature))
    return torch.softmax(logits / t, dim=-1)

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

        # raw logits (unbounded output)
        # return self.out(x)
        # squashed to (0, 1)
        return torch.sigmoid(self.out(x))