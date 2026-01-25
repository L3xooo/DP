from dataclasses import dataclass
from typing import Dict, List, Literal

TickerConfigName = Literal["10_TICKERS", "30_TICKERS"]

@dataclass(frozen=True)
class TickerConfig:
    name: str
    tickers: List[str]


TICKER_CONFIGS: Dict[str, TickerConfig] = {
    "10_TICKERS": TickerConfig(
        name="10_TICKERS",
        tickers=[
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "AMZN",  # Amazon
            "GOOGL", # Alphabet (Google)
            "META",  # Meta Platforms
            "TSLA",  # Tesla
            "NVDA",  # Nvidia
            "JPM",   # JPMorgan Chase
            "JNJ",   # Johnson & Johnson
            "XOM"]),
    "30_TICKERS": TickerConfig(
        name="30_TICKERS",
        tickers=[
            # Technology
            "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "IBM",
            # Consumer / Retail
            "AMZN", "HD", "MCD", "NKE", "SBUX", "COST",
            # Financials
            "JPM", "BAC", "WFC", "GS", "MS",
            # Healthcare
            "JNJ", "PFE", "MRK", "ABBV", "UNH",
            # Energy
            "XOM", "CVX", "COP",
            # Industrials
            "CAT", "BA", "GE",
            # Communications
            "VZ", "T",
        ],
    ),
}

def get_tickers_with_cash(name: TickerConfigName) -> list[str]:
    return ["Cash", *TICKER_CONFIGS[name].tickers]

def get_ticker_config(name: TickerConfigName) -> TickerConfig:
    return TICKER_CONFIGS[name]