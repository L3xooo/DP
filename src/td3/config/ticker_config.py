from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

TickerConfigName = Literal["10_TICKERS", "30_TICKERS", "ANOTHER_10_TICKERS"]

TICKER_PRESETS: Dict[TickerConfigName, List[str]] = {
    "ANOTHER_10_TICKERS": ["BRK.B", "UNH", "V", "MA", "AVGO", "LLY", "JPM", "XOM", "COST", "HD"],
    "10_TICKERS": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "JNJ", "XOM"],
    "30_TICKERS": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "IBM",
        "AMZN", "HD", "MCD", "NKE", "SBUX", "COST",
        "JPM", "BAC", "WFC", "GS", "MS",
        "JNJ", "PFE", "MRK", "ABBV", "UNH",
        "XOM", "CVX", "COP",
        "CAT", "BA", "GE",
        "VZ", "T",
    ],
}

@dataclass(frozen=True)
class TickerConfig:
    """Ticker universe preset chosen by name."""
    name: TickerConfigName

    @property
    def tickers(self) -> List[str]:
        return TICKER_PRESETS[self.name]

    @property
    def tickers_with_cash(self) -> List[str]:
        return ["Cash", *self.tickers]

    def __len__(self) -> int:
        return len(self.tickers)