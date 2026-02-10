from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pytest_bdd import steps


@dataclass
class StepMetrics:
    reward: float = 0.0
    portfolio_value: float = 0.0
    weights: Optional[List[float]] = None

    def set_basic(self, reward: float, portfolio_value: float, weights: List[float]) -> "StepMetrics":
        self.reward = float(reward)
        self.portfolio_value = float(portfolio_value)
        self.weights = weights
        return self

@dataclass
class EpisodeMetrics:
    steps: List[StepMetrics] = field(default_factory=list)
    total_reward: float = 0.0
    final_portfolio_value: float = 0.0
    final_weights: List[List[float]] = field(default_factory=list)

    def update(self, metrics: StepMetrics) -> None:
        self.steps.append(metrics)
        self.total_reward = metrics.reward
        self.final_portfolio_value = metrics.portfolio_value
        self.final_weights.append(metrics.weights)

    def aggregate(self) -> None:
        self.total_reward = sum(step.reward for step in self.steps)
        self.final_portfolio_value = self.steps[-1].portfolio_value if self.steps else 0.0
        self.steps = []

    def __str__(self):
        return f"EpisodeMetrics(total_reward={self.total_reward})"

@dataclass
class RunMetrics:
    run_id: str = None
    episodes: List[EpisodeMetrics | StepMetrics] = field(default_factory=list)

    def start_step(self) -> StepMetrics:
        step = StepMetrics()
        self.episodes.append(step)
        return step

    def start_episode(self) -> EpisodeMetrics:
        ep = EpisodeMetrics()
        self.episodes.append(ep)
        return ep

    def __len__(self) -> int:
        return len(self.episodes)

@dataclass
class ExperimentMetrics:
    # config_name: str
    runs: List[RunMetrics] = field(default_factory=list)

    def start_run(self, run_id: str) -> RunMetrics:
        r = RunMetrics(run_id="run_" + run_id)
        self.runs.append(r)
        return r