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
from td3.utils.date_utils import check_if_later_date
from td3.utils.logs.logger import WithLogger


@WithLogger()
@dataclass(frozen=True)
class AppConfig:
    """
    Central configuration for a TD3 training run.

    Holds all hyperparameters and settings including training loop controls,
    replay buffer size, network architecture, noise schedule, data filtering,
    and the target ticker universe.
    """

    iterations: int = 1
    number_of_episodes: int = 100
    batch_size: int = 256
    replay_buffer_size: int = 60_000
    data_dir: str = "../indicators"
    # 1027 steps per episode
    start_date: str = "2016-05-01"
    end_date: str = "2020-05-30"
    initial_cash: float = 10000.0
    hidden_size: int = 256
    lr: float = 3e-5
    noise_init: float = 0.4
    noise_final: float = 0.08
    device: Optional[str] = None
    dropout_rate: float = 0.0
    normalization: Optional[str] = "cross"
    lookback_window: int = 20
    ticker_config: TickerConfig = TickerConfig("10_TICKERS")

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

    def __post_init__(self):
        """
        Set the compute device after initialization.
        Defaults to CUDA if available, otherwise CPU. Skips detection if device was explicitly set.
        """
        if self.lookback_window < 1:
            raise ValueError("lookback_window must be >= 1")
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
            out_dir: Directory path where config.json will be written. Created if it does not exist.
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
    def load_from_train_config(
            cls,
            path: str | Path,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
    ) -> "AppConfig":
        """
        Load AppConfig from a config.json file, allowing overrides for certain fields.

        Args:
            path: Path to the config.json file or directory containing it.
            start_date: Optional override for the start date of the training data. Must be earlier than end_date if provided.
            end_date: Optional override for the end date of the training data. Must be later than start_date if provided.

        Returns:
            An instance of AppConfig with values loaded from the file and overrides applied.
        """
        cls.logger.info("Loading AppConfig from %s", path)
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"

        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path}")

        with path.open("r", encoding="utf-8") as f:
            d: dict[str, Any] = json.load(f)

        tc = d.get("ticker_config")
        if not isinstance(tc, dict):
            raise ValueError("config.json must contain object 'ticker_config'")

        name = tc.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("ticker_config.name must be a non-empty string")

        if name == "RANDOM_TICKERS":
            tickers = tc.get("tickers")
            if (
                not isinstance(tickers, list)
                or not tickers
                or any(not isinstance(t, str) or not t.strip() for t in tickers)
            ):
                raise ValueError(
                    "ticker_config.tickers must be a non-empty list of non-empty strings "
                    "when name == 'RANDOM_TICKERS'"
                )
            d["ticker_config"] = TickerConfig(name=name, _random_tickers=tickers)
        else:
            d["ticker_config"] = TickerConfig(name=name)

        if start_date is None or end_date is None:
            raise TypeError("start_date and end_date must be provided")

        # Config end date, test start date
        check_if_later_date(sooner_date=d.get("end_date"), later_date=start_date)
        d["start_date"] = start_date

        # Test start date and test end date
        check_if_later_date(sooner_date=start_date, later_date=end_date)
        d["end_date"] = end_date

        d["number_of_episodes"] = 1
        return cls(**d)
