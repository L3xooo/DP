"""
Configuration module for the TD3 training pipeline.

Defines AppConfig, the central dataclass holding all hyperparameters,
environment settings, and model options required to run a training session.

Author: Peter Likavec
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any
from pathlib import Path
import json
import torch

from td3.config.ticker_config import TickerConfig
from td3.utils.logger import WithLogger


@WithLogger()
@dataclass(frozen=True)
class AppConfig:
    """
    Central configuration for a TD3 training run.

    Holds all hyperparameters and settings including training loop controls,
    replay buffer size, network architecture, noise schedule, data filtering,
    and the target ticker universe.
    """

    iterations: int = 3
    number_of_episodes: int = 300
    batch_size: int = 512
    replay_buffer_size: int = 100_000

    ticker_config: TickerConfig = TickerConfig("10_TICKERS")

    data_dir: str = "../indicators"
    start_date: str = "2016-05-01"
    end_date: str = "2019-01-18"
    initial_cash: float = 10000.0

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

    filter_in: List[str] = field(
        default_factory=lambda: [
            'adx',
            'aroon',
            'aroon_down',
            'aroon_up',
            'atr',
            'bb_bbh',
            'bb_bbl',
            'bb_bbm',
            'cmf',
            'ema20',
            'ema50',
            'kch_high',
            'kch_low',
            'kch_mid',
            'macd_diff',
            'mfi',
            'pocket_pivot',
            'rel_close',
            'rel_high',
            'rel_low',
            'rel_open',
            'roc',
            'rsi',
            'sma',
            'squeeze',
            'tsi',
            'vwap',
        ]
    )

    hidden_size: int = 512
    lr: float = 3e-4
    noise_init: float = 0.3
    noise_final: float = 0.05
    noise_anneal_episodes: int = 500
    device: Optional[str] = None

    def __post_init__(self):
        """
        Set the compute device after initialization.

        Defaults to CUDA if available, otherwise CPU.
        Skips detection if device was explicitly set.
        """
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
        """
        Serialize the config to a JSON file.
        Args:
            out_dir: Directory path where config.json will be written.
                     Created if it does not exist.
        Returns:
            None
        """
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
        """
        Load an AppConfig from a JSON file or directory.
        Args:
            path: Path to a config.json file or a directory containing one.

        Returns:
            AppConfig: A new instance populated from the JSON data.
        """

        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with path.open("r", encoding="utf-8") as f:
            d: dict[str, Any] = json.load(f)

        tc = d.get("ticker_config")
        if isinstance(tc, dict):
            d["ticker_config"] = TickerConfig(tc.get("name", "10_TICKERS"))
        else:
            d["ticker_config"] = TickerConfig("10_TICKERS")

        return cls(**d)
