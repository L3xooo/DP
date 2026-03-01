import torch

from td3.models.actor import Actor


model = Actor(input_dim=280, action_dim=11, hidden_size=512)
model.eval()

dummy = torch.randn(1, 280)  # (batch, input_dim)

torch.onnx.export(
    model,
    dummy,
    "actor.onnx",
    input_names=["state"],
    output_names=["action_raw"],
    opset_version=17,
)

print("Saved actor.onnx")

# import netron
# netron.start("actor.onnx")

import torch
from td3.models.critic import Critic

model = Critic(state_dim=280, action_dim=11, hidden_size=512)
model.eval()

dummy_state = torch.randn(1, 280)
dummy_action = torch.randn(1, 11)

torch.onnx.export(
    model,
    (dummy_state, dummy_action),  # tuple inputs
    "critic.onnx",
    input_names=["state", "action"],
    output_names=["q_value"],
    opset_version=17,
)

print("Saved critic.onnx")
