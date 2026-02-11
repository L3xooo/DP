import os
import torch
from torchviz import make_dot
from torchinfo import summary

from td3.config.ticker_config import ALL_TICKERS
from td3.models.actor import Actor
from td3.models.critic import Critic


def visualize_actor(state_dim: int, action_dim: int, hidden_size: int = 512, device: str = "cpu"):
    actor = Actor(input_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size).to(device)

    x = torch.randn(1, state_dim, device=device)

    # Forward pre summary (textový popis)
    print("=== ACTOR SUMMARY ===")
    summary(actor, input_size=(1, state_dim))

    # Forward pre graf
    y = actor(x)
    dot = make_dot(y, params=dict(actor.named_parameters()))
    os.makedirs("model_viz", exist_ok=True)
    actor_path = os.path.join("model_viz", "actor_graph")
    dot.format = "png"
    dot.render(actor_path, cleanup=True)
    print(f"Actor graph saved to {actor_path}.png")

def visualize_critic(state_dim: int, action_dim: int, hidden_size: int = 512, device: str = "cpu"):
    critic = Critic(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size).to(device)

    # Dummy vstup: batch_size = 1
    state = torch.randn(1, state_dim, device=device)
    action = torch.randn(1, action_dim, device=device)

    print("=== CRITIC SUMMARY ===")
    summary(critic, input_size=[(1, state_dim), (1, action_dim)])

    y = critic(state, action)
    dot = make_dot(y, params=dict(critic.named_parameters()))
    os.makedirs("model_viz", exist_ok=True)
    critic_path = os.path.join("model_viz", "critic_graph")
    dot.format = "png"
    dot.render(critic_path, cleanup=True)
    print(f"Critic graph saved to {critic_path}.png")

if __name__ == "__main__":
    state_dim = 280
    action_dim = 11

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    #
    # visualize_actor(state_dim, action_dim, hidden_size=512, device=device)
    # visualize_critic(state_dim, action_dim, hidden_size=512, device=device)

    print(ALL_TICKERS.split())
