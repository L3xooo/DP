"""
Utilities for processing logits in the TD3 algorithm, including adding noise and converting logits to weights.

Author: Peter Likavec
"""

import torch
from entmax import entmax_bisect


def add_logit_noise(logits: torch.Tensor, noise_std: float, noise_clip: float) -> torch.Tensor:
    """
    Add Gaussian noise to the logits with clipping to ensure the noise does not exceed a specified range.
    Args:
        logits: The input logits to which noise will be added.
        noise_std: The standard deviation of the Gaussian noise to be added.
        noise_clip: The maximum absolute value for the noise.

    Returns:
        Logits with added noise, where the noise is clipped to the range [-noise_clip, noise_clip].
    """
    if noise_std and noise_std > 0:
        noise = torch.randn_like(logits) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        return logits + noise
    return logits


def logits_to_weights(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalize the logits to produce a valid long probability distribution over actions using the entmax transformation.
    Args:
        logits: The input logits to be transformed into weights.

    Returns:
        A tensor of the same shape as logits, where each slice is valid distribution obtained
        by applying the entmax transformation with alpha=1.6 along the last dimension.
    """
    return entmax_bisect(logits, dim=-1, alpha=1.6)
