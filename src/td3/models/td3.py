import torch
from torch import optim
import torch.nn as nn
import numpy as np

from td3.metrics.metrics import StepMetrics
from td3.models.actor import Actor, add_logit_noise, logits_to_weights
from td3.models.critic import Critic
import os

from td3.utils.logger import WithLogger


@WithLogger()
class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_size=512,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        noise_init=0.3,
        noise_final=0.05,
        noise_anneal_episodes=500,
        learning_starts=1000,
        device=None,
    ):
        self.device = device
        self.total_it = 0
        self.policy_noise = 0.2
        self.noise_clip = 0.5

        self._expl_noise_init = float(noise_init)
        self._expl_noise_final = float(noise_final)
        self._expl_noise_anneal = int(noise_anneal_episodes)
        self.current_episode = 0

        self.actor = Actor(input_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size).to(
            self.device
        )
        self.critic1 = Critic(state_dim, action_dim, hidden_size).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_size).to(self.device)

        self.target_actor = Actor(
            input_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size
        ).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_size).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_size).to(self.device)

        self.mu = np.zeros(action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau

    def set_episode(self, episode_index: int):
        self.current_episode = int(episode_index)

        if self._expl_noise_anneal <= 0:
            self.policy_noise = self._expl_noise_final
            return

        progress = min(self.current_episode / float(self._expl_noise_anneal), 1.0)
        self.policy_noise = float(
            self._expl_noise_init + (self._expl_noise_final - self._expl_noise_init) * progress
        )

    @torch.no_grad()
    def select_action(
        self,
        state,
        noise_std=None,
        noise_clip=None,
        use_noise: bool = True,
    ):
        # state can be 1D (single timestep) or already batched
        if not isinstance(state, (np.ndarray,)):
            state = np.array(state, dtype=np.float32)
        else:
            state = state.astype(np.float32)

        if state.ndim == 1:
            state = state.reshape(1, -1)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = self.actor(state_t)

        if not use_noise:
            noisy_logits = logits
        else:
            if noise_std is None:
                noise_std = float(self.policy_noise)
            if noise_clip is None:
                noise_clip = float(self.noise_clip)

            noisy_logits = add_logit_noise(logits, noise_std, noise_clip)

        weights = logits_to_weights(noisy_logits)
        action = weights.squeeze(0).cpu().numpy()
        return action, noisy_logits

    def update(self, replay_buffer, batch_size) -> StepMetrics:
        if replay_buffer.size() < batch_size:
            return StepMetrics()

        self.total_it += 1
        states, actions, rewards, dones, next_states = replay_buffer.sample_batch(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Flatten any per-asset / per-feature dims so networks receive (batch, input_dim)
        if states.dim() > 2:
            states = states.reshape(states.size(0), -1)
        if next_states.dim() > 2:
            next_states = next_states.reshape(next_states.size(0), -1)

        with torch.no_grad():
            next_logits = self.target_actor(next_states)

            next_logits_noisy = add_logit_noise(next_logits, self.policy_noise, self.noise_clip)

            next_actions = logits_to_weights(next_logits_noisy)

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

        critic1_loss_val = float(critic1_loss.detach().cpu().item())
        critic2_loss_val = float(critic2_loss.detach().cpu().item())
        actor_loss_val = None

        q1_mean = float(q1.detach().mean().cpu().item())
        q2_mean = float(q2.detach().mean().cpu().item())

        if self.total_it % 2 == 0:
            actor_logits = self.actor(states)
            actor_actions = logits_to_weights(actor_logits)

            q1_pi = self.critic1(states, actor_actions)

            actor_loss = -q1_pi.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target_network(self.target_actor, self.actor)
            self._update_target_network(self.target_critic1, self.critic1)
            self._update_target_network(self.target_critic2, self.critic2)

            actor_loss_val = float(actor_loss.detach().cpu().item())

        return StepMetrics(
            actor_loss=actor_loss_val,
            critic1_loss=critic1_loss_val,
            critic2_loss=critic2_loss_val,
            q1_mean=q1_mean,
            q2_mean=q2_mean,
        )

    def _update_target_network(self, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, directory, filename='td3_model.pth'):
        path = os.path.join(directory, filename)
        torch.save(
            {
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
            },
            path,
        )

    def load_model(self, filename='td3_model.pth'):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
