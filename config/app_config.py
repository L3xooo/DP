from dataclasses import dataclass
from typing import Optional
from ticker_config import TickerConfigName, get_ticker_config

@dataclass(frozen=True)
class AppConfig:
    number_of_episodes: int = 500
    batch_size: int = 64
    learning_start_episode: int = 100
    replay_buffer_size: int = 1_000_000
    ticker_preset: TickerConfigName = "10_TICKERS"
    data_dir: str = "../indicators"
    start_date: str = "2019-01-21"
    end_date: str = "2023-12-24"
    initial_cash: float = 10000
    temperature: float = 1.0

    hidden_size: int = 512
    lr: float = 3e-4
    noise_init: float = 0.3
    noise_final: float = 0.05
    noise_anneal_episodes: int = 500
    device: Optional[str] = None

def get_tickers(cfg: AppConfig):
    return get_ticker_config(cfg.ticker_preset).tickers