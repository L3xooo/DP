from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RunMetrics:
    # Metrics from the last step of each episode
    rewards: List[float] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)

    # Average of values after each step for each episode
    actor_losses: List[Optional[float]] = field(default_factory=list)
    critic_values: Dict[str, List[float]] = field(default_factory=lambda: {"c1": [], "c2": []})

    # Metrics from each step for each episode
    weights: List[Any] = field(default_factory=list)
