from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RunMetrics:
    rewards: List[float] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    critic_values: Dict[str, List[float]] = field(
        default_factory=lambda: {"c1": [], "c2": []}
    )
    weights: List[Any] = field(default_factory=list)
