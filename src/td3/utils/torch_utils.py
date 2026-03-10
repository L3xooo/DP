"""
Model export and inspection utilities for the TD3 training pipeline.

Provides helpers to count trainable parameters and export Actor and Critic
models to ONNX format for inference and deployment.

Author: Peter Likavec
"""

from pathlib import Path

import torch

from td3.models.actor import Actor
from td3.models.critic import Critic


def count_params(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_actor(
    input_dim: int = 280,
    action_dim: int = 11,
    hidden_size: int = 512,
    output_dir: str = "../../../model_exports",
) -> None:
    """Exports the Actor model to ONNX format."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "actor.onnx"

    model = Actor(input_dim=input_dim, action_dim=action_dim, hidden_size=hidden_size)
    model.eval()

    dummy = torch.randn(1, input_dim)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["state"],
        output_names=["action_raw"],
        opset_version=17,
    )

    print(f"Saved {output_path}")


def export_critic(
    state_dim: int = 280,
    action_dim: int = 11,
    hidden_size: int = 512,
    output_dir: str = "../../../model_exports",
) -> None:
    """Exports the Critic model to ONNX format."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "critic.onnx"

    model = Critic(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size)
    model.eval()

    dummy_state = torch.randn(1, state_dim)
    dummy_action = torch.randn(1, action_dim)

    torch.onnx.export(
        model,
        (dummy_state, dummy_action),
        str(output_path),
        input_names=["state", "action"],
        output_names=["q_value"],
        opset_version=17,
    )

    print(f"Saved {output_path}")


if __name__ == "__main__":
    export_actor()
    export_critic()
