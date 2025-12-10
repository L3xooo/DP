# python
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from models.actor import Actor, logits_to_probs
from models.critic import Critic
from utils.logger import WithLogger, log_numpy, log_stock_value
from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

@WithLogger()
class TD3:
    """
    TD3 that expects a flattened state vector (1D per timestep) or batches thereof.
    Create with state_dim = env.observation_space.shape[0].
    """
    def __init__(self, state_dim, action_dim, hidden_size=512, gamma=0.99, tau=0.005, lr=1e-3):
        self.total_it = 0
        self.policy_noise = 0.2
        self.noise_clip = 0.5

        # state_dim should be the flattened observation size (int)
        self.actor = Actor(input_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size)
        self.critic1 = Critic(state_dim, action_dim, hidden_size)
        self.critic2 = Critic(state_dim, action_dim, hidden_size)

        self.target_actor = Actor(input_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_size)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_size)

        self.mu = np.zeros(action_dim)
        # self.noise = OrnsteinUhlenbeckActionNoise(mu=self.mu, sigma=0.2, theta=0.15)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, noise_scale=0.2, temperature=1.0) -> np.ndarray:
        # state can be 1D (single timestep) or already batched
        if not isinstance(state, (np.ndarray,)):
            state = np.array(state, dtype=np.float32)
        else:
            state = state.astype(np.float32)

        # ensure batch dimension
        if state.ndim == 1:
            state = state.reshape(1, -1)

        state_t = torch.tensor(state, dtype=torch.float32).to(next(self.actor.parameters()).device)
        with torch.no_grad():
            # self.logger.info("[select_action] - State: %s", np.round(state_t.cpu().numpy(), 2))
            logits = self.actor(state_t).squeeze(0)
            # self.logger.info("[select_action] - Raw logits: %s", np.round(logits.cpu().numpy(), 2))
            if noise_scale and noise_scale > 0:
                noise = torch.randn_like(logits) * noise_scale
                noise = noise.clamp(-3 * noise_scale, 3 * noise_scale)
                # self.logger.info("[select_action] - Raw Noise: %s", np.round(noise.cpu().numpy(), 2))
                logits = logits + noise
            # self.logger.info("[select_action] - Noisy Logits: %s", np.round(logits.cpu().numpy(), 2))
            logits = logits.clamp(-50.0, 50.0)
            probs_t = logits_to_probs(logits.unsqueeze(0), temperature=temperature)
            probs = np.round(probs_t.cpu().numpy().reshape(-1), 2)

        probs = np.clip(probs, 0.0, 1.0)
        s = probs.sum()
        if s <= 0 or not np.isfinite(s):
            probs = np.ones_like(probs) / probs.size
        else:
            probs = probs / s

        log_numpy(self.logger, probs, "Action")
        return probs

    # python
    def update(self, replay_buffer, batch_size, temperature=1.0):
        if replay_buffer.size() < batch_size:
            return

        self.total_it += 1

        states, actions, rewards, dones, next_states = replay_buffer.sample_batch(batch_size)

        device = next(self.actor.parameters()).device

        # to tensor on correct device
        states = torch.tensor(states, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)

        # Flatten any per-asset / per-feature dims so networks receive (batch, input_dim)
        if states.dim() > 2:
            states = states.reshape(states.size(0), -1)
        if next_states.dim() > 2:
            next_states = next_states.reshape(next_states.size(0), -1)

        with torch.no_grad():
            next_logits = self.target_actor(next_states)
            noise = (torch.randn_like(next_logits) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_logits = (next_logits + noise).clamp(-50.0, 50.0)
            next_actions = logits_to_probs(next_logits, temperature=temperature)

            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = nn.MSELoss()(q1, target_q)
        critic2_loss = nn.MSELoss()(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.total_it % 2 == 0:
            actor_logits = self.actor(states).clamp(-50.0, 50.0)
            actor_actions = logits_to_probs(actor_logits, temperature=temperature)
            actor_loss = -self.critic1(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target_network(self.target_actor, self.actor)
            self._update_target_network(self.target_critic1, self.critic1)
            self._update_target_network(self.target_critic2, self.critic2)

    def _update_target_network(self, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)