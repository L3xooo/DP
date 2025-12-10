# python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def logits_to_probs(scores: Tensor, temperature: float = 1.0) -> Tensor:
    eps = 1e-9
    t = max(1e-6, float(temperature))
    scaled = scores / t
    scaled = torch.clamp(scaled, min=eps)
    probs = scaled / (scaled.sum(dim=-1, keepdim=True) + eps)
    return probs

class Actor(nn.Module):
    """
    Actor that accepts either:
      - a flattened state tensor of shape (batch, input_dim) OR
      - a 2D per-step tensor of shape (batch, num_assets, feature_dim)
    The module flattens inputs as needed and runs a simple MLP to produce positive logits.
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 256):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_dim)
        self.softplus = nn.Softplus()

    def forward(self, state: Tensor) -> Tensor:
        # Accept state shaped (batch, ...) and flatten from dim=1 onward
        if state.dim() > 2:
            x = state.view(state.size(0), -1)
        else:
            x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.softplus(self.out(x)) + 1e-9
        return logits
