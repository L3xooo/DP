"""
Alpaca trading client wrapper.

Usage:
- Configure environment variables (recommended):
    ALPACA_API_KEY_ID
    ALPACA_API_SECRET_KEY
    ALPACA_PAPER=true|false   # default true

Optional:
    ALPACA_BASE_URL           # override endpoint; e.g. https://paper-api.alpaca.markets

Example:
    from src.alpaca.alpaca import AlpacaClient
    client = AlpacaClient()  # reads keys from env, uses paper by default
    acct = client.get_account()
    print(acct.status)

    # Place a market order
    order = client.submit_order(
        symbol="AAPL", qty=1, side="buy", type_="market", time_in_force="day"
    )
    print(order.id)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Literal, Any

# Lazily import the SDK to avoid hard dependency at import time if not installed
try:
    from alpaca_trade_api import REST
    from alpaca_trade_api.entity import Account, Position, Order
except Exception:  # pragma: no cover - handled at runtime
    REST = None  # type: ignore
    Account = Position = Order = Any  # type: ignore


PaperBaseURL = "https://paper-api.alpaca.markets"
LiveBaseURL = "https://api.alpaca.markets"

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "opg", "ioc", "fok"]


@dataclass
class AlpacaSettings:
    api_key_id: Optional[str] = None
    api_secret_key: Optional[str] = None
    paper: bool = True
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AlpacaSettings":
        return cls(
            api_key_id=os.getenv("ALPACA_API_KEY_ID"),
            api_secret_key=os.getenv("ALPACA_API_SECRET_KEY"),
            paper=_to_bool(os.getenv("ALPACA_PAPER", "true")),
            base_url=os.getenv("ALPACA_BASE_URL"),
        )

    def resolve_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return PaperBaseURL if self.paper else LiveBaseURL


class AlpacaClient:
    """Thin wrapper around alpaca-trade-api REST client with helpful defaults."""

    def __init__(self, settings: Optional[AlpacaSettings] = None) -> None:
        self.settings = settings or AlpacaSettings.from_env()

        # Validate keys
        if not self.settings.api_key_id or not self.settings.api_secret_key:
            raise RuntimeError(
                "Alpaca API credentials missing. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY."
            )

        if REST is None:
            raise RuntimeError(
                "alpaca-trade-api not installed. Add 'alpaca-trade-api' to requirements and install."
            )

        self._rest = REST(
            self.settings.api_key_id,
            self.settings.api_secret_key,
            base_url=self.settings.resolve_base_url(),
        )

    # Account and positions
    def get_account(self) -> Account:
        return self._rest.get_account()

    def get_positions(self) -> list[Position]:
        return list(self._rest.list_positions())

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            return self._rest.get_position(symbol)
        except Exception:
            return None

    # Orders
    def list_orders(self, status: Literal["open", "closed", "all"] = "open", limit: int = 50) -> list[Order]:
        return list(self._rest.list_orders(status=status, limit=limit))

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: Side,
        type_: OrderType = "market",
        time_in_force: TimeInForce = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> Order:
        if qty <= 0:
            raise ValueError("qty must be positive")
        if type_ in ("limit", "stop_limit") and not limit_price:
            raise ValueError("limit_price required for limit or stop_limit orders")
        if type_ in ("stop", "stop_limit") and not stop_price:
            raise ValueError("stop_price required for stop or stop_limit orders")

        return self._rest.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type_,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=client_order_id,
            extended_hours=extended_hours,
        )

    def cancel_order(self, order_id: str) -> None:
        self._rest.cancel_order(order_id)

    def get_order(self, order_id: str) -> Order:
        return self._rest.get_order(order_id)

    # Market data (basic)
    def get_last_trade(self, symbol: str) -> Any:
        # Uses v2 data; requires proper subscription level for live.
        return self._rest.get_last_trade(symbol)


def _to_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}
