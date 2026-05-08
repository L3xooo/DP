"""
Metrics tracking dataclasses for the TD3 training pipeline.

Defines a four-level hierarchy: StepMetrics → EpisodeMetrics → RunMetrics → ExperimentMetrics,
used to collect, aggregate, and report training statistics across steps, episodes, and runs.

Author: Peter Likavec
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from td3.utils.logs.logger import WithLogger


@dataclass
class StepMetrics:
    """Metrics collected at a single environment step."""

    reward: float = 0.0
    portfolio_value: float = 0.0
    weights: Optional[List[float]] = None

    critic1_loss: Optional[float] = None
    critic2_loss: Optional[float] = None
    actor_loss: Optional[float] = None

    q1_mean: Optional[float] = None
    q2_mean: Optional[float] = None
    target_q_mean: Optional[float] = None

    def set_basic(
        self, reward: float, portfolio_value: float, weights: List[float]
    ) -> "StepMetrics":
        """
        Set the core environment metrics for this step.
        Args:
            reward: Reward received at this step.
            portfolio_value: Portfolio value after this step.
            weights: Asset allocation weights after this step.

        Returns:
            StepMetrics: Self, for method chaining.
        """
        self.reward = float(reward)
        self.portfolio_value = float(portfolio_value)
        self.weights = weights
        return self


@WithLogger()
@dataclass
class EpisodeMetrics:
    """
    Aggregated metrics for a single training episode.

    Accumulates StepMetrics across all steps and computes episode-level
    summaries for rewards, portfolio value, losses, and Q-values.
    """

    steps: List[StepMetrics] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    exploration_noise: Optional[float] = None
    final_portfolio_value: float = 0.0
    final_weights: List[List[float]] = field(default_factory=list)
    actor_loss: Optional[float] = None
    actor_loss_all_steps: List[float] = field(default_factory=list)
    critic1_loss: Optional[float] = None
    all_critic1_loss: List[float] = field(default_factory=list)
    critic2_loss: Optional[float] = None
    all_critic2_loss: List[float] = field(default_factory=list)

    q1_mean: Optional[float] = None
    q2_mean: Optional[float] = None
    target_q_mean: Optional[float] = None

    def update(self, metrics: StepMetrics) -> None:
        """
        Append a step's metrics and update running episode state.
        Args:
            metrics: StepMetrics instance from the latest environment step.
        """
        self.steps.append(metrics)
        self.rewards.append(metrics.reward)
        self.total_reward += metrics.reward
        self.final_portfolio_value = metrics.portfolio_value
        self.final_weights.append(metrics.weights)

    def aggregate(self) -> None:
        """
        Compute episode-level summaries from accumulated step metrics.

        Calculates means for actor loss, critic losses, and Q-values,
        then computes total reward and final portfolio value. Clears
        the step buffer after aggregation.
        """
        actor_losses = [s.actor_loss for s in self.steps if s.actor_loss is not None]
        self.actor_loss_all_steps = actor_losses
        self.actor_loss = float(np.mean(actor_losses)) if actor_losses else None

        c1 = [s.critic1_loss for s in self.steps if s.critic1_loss is not None]
        c2 = [s.critic2_loss for s in self.steps if s.critic2_loss is not None]
        self.all_critic1_loss = c1
        self.all_critic2_loss = c2
        if c1:
            self.critic1_loss = float(np.mean(c1))
        if c2:
            self.critic2_loss = float(np.mean(c2))

        q1s = [s.q1_mean for s in self.steps if s.q1_mean is not None]
        q2s = [s.q2_mean for s in self.steps if s.q2_mean is not None]
        tqs = [s.target_q_mean for s in self.steps if s.target_q_mean is not None]

        self.q1_mean = float(np.mean(q1s)) if q1s else None
        self.q2_mean = float(np.mean(q2s)) if q2s else None
        self.target_q_mean = float(np.mean(tqs)) if tqs else None

        self.total_reward = float(sum(self.rewards))
        self.final_portfolio_value = float(self.steps[-1].portfolio_value) if self.steps else 0.0
        self.steps = []

    def __str__(self):
        return f"EpisodeMetrics(total_reward={self.total_reward})"


@dataclass
class RunMetrics:
    """Metrics for a single training run, composed of episodes or steps."""

    run_id: str = None
    episodes: List[EpisodeMetrics | StepMetrics] = field(default_factory=list)

    def start_step(self) -> StepMetrics:
        """
        Create and register a new StepMetrics for the current step.
        Returns:
            StepMetrics: A fresh step metrics instance appended to this run.
        """
        step = StepMetrics()
        self.episodes.append(step)
        return step

    def start_episode(self) -> EpisodeMetrics:
        """
        Create and register a new EpisodeMetrics for the current episode.
        Returns:
            EpisodeMetrics: A fresh episode metrics instance appended to this run.
        """
        ep = EpisodeMetrics()
        self.episodes.append(ep)
        return ep

    def __len__(self) -> int:
        """
        Return the number of recorded episodes or steps in this run.
        Returns:
            int: Length of the episodes list.
        """
        return len(self.episodes)


@dataclass
class ExperimentMetrics:
    """Top-level container for metrics across all runs in an experiment."""

    runs: List[RunMetrics] = field(default_factory=list)

    def start_run(self, run_id: str) -> RunMetrics:
        """
        Create and register a new RunMetrics for the given run ID.
        Args:
            run_id: Identifier suffix; stored as 'run_<run_id>'.
        Returns:
            RunMetrics: A fresh run metrics instance appended to this experiment.
        """
        r = RunMetrics(run_id="run_" + run_id)
        self.runs.append(r)
        return r
