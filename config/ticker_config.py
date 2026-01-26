from dataclasses import dataclass
from typing import Dict, List, Literal

TickerConfigName = Literal["10_TICKERS", "30_TICKERS", "ANOTHER_10_TICKERS"]

@dataclass(frozen=True)
class TickerConfig:
    name: str
    tickers: List[str]

TICKER_CONFIGS: Dict[str, TickerConfig] = {
    "ANOTHER_10_TICKERS": TickerConfig(
        name="ANOTHER_10_TICKERS",
        tickers=["BRK.B", "UNH", "V", "MA", "AVGO", "LLY", "JPM", "XOM", "COST", "HD"],
    ),
    "10_TICKERS": TickerConfig(
        name="10_TICKERS",
        tickers=["AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","JPM","JNJ","XOM"],
    ),
    "30_TICKERS": TickerConfig(
        name="30_TICKERS",
        tickers=[
            "AAPL","MSFT","GOOGL","META","NVDA","AMD","INTC","IBM",
            "AMZN","HD","MCD","NKE","SBUX","COST",
            "JPM","BAC","WFC","GS","MS",
            "JNJ","PFE","MRK","ABBV","UNH",
            "XOM","CVX","COP",
            "CAT","BA","GE",
            "VZ","T",
        ],
    ),
}

def get_tickers_with_cash(name: TickerConfigName) -> List[str]:
    return ["Cash"] + TICKER_CONFIGS[name].tickers

def get_ticker_config(name: TickerConfigName) -> TickerConfig:
    return TICKER_CONFIGS[name]