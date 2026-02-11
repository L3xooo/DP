from dataclasses import dataclass, field, asdict
from typing import Optional, List
from pathlib import Path
import json
import torch

from td3.config.ticker_config import TickerConfig
from td3.utils.logger import WithLogger


@WithLogger()
@dataclass(frozen=True)
class AppConfig:
    iterations: int = 5
    number_of_episodes: int = 50
    batch_size: int = 128
    learning_start_episode: int | None = 100
    replay_buffer_size: int = 100_000

    ticker_config: TickerConfig = TickerConfig("10_TICKERS")

    data_dir: str = "../indicators"
    start_date: str = "2016-05-01"
    end_date: str = "2019-01-18"
    initial_cash: float = 10000.0
    temperature: float = 1.0

    filter_out: List[str] = field(
        default_factory=lambda: [
            "obv",
            "volume_base",
            "open",
            "high",
            "low",
            "unix",
        ]
    )

    hidden_size: int = 512
    lr: float = 3e-4
    noise_init: float = 0.3
    noise_final: float = 0.05
    noise_anneal_episodes: int = 500
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is not None:
            return
        if torch.cuda.is_available():
            chosen = "cuda"
        # elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        #     chosen = "mps"
        else:
            chosen = "cpu"
        print("Device chosen:", chosen)
        object.__setattr__(self, "device", chosen)

    def to_json(self, out_dir: str | Path) -> None:
        d = asdict(self)

        d["ticker_config"] = {
            "name": self.ticker_config.name,
            "tickers": self.ticker_config.tickers,
        }

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "config.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "AppConfig":
        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with path.open("r", encoding="utf-8") as f:
            d: dict[str, Any] = json.load(f)

        # rebuild ticker_config
        tc = d.get("ticker_config")
        if isinstance(tc, dict):
            d["ticker_config"] = TickerConfig(tc.get("name", "10_TICKERS"))
        else:
            d["ticker_config"] = TickerConfig("10_TICKERS")

        return cls(**d)