"""
DynamicPortfolio algorithm module.

Defines DynamicPortfolio, a QuantConnect algorithm that dynamically allocates
weights to a set of equities based on a CSV file. Positions are rebalanced
daily according to the weights defined per date.

Author: Peter Likavec
"""

from AlgorithmImports import *
import csv, io
from datetime import datetime


class DynamicPortfolio(QCAlgorithm):

    def initialize(self):
        self.set_security_initializer(lambda security: security.set_fee_model(ConstantFeeModel(0, "USD")))

        self.set_start_date(2021, 6, 1)
        self.set_end_date(2024, 12, 30)
        self.set_cash(10000)

        key = "weights_td3_model_run_0.pth.csv"

        if not self.object_store.contains_key(key):
            self.debug(f"ObjectStore key not found: {key}")
            return

        csv_text = self.object_store.read(key)

        self.weights_by_date = {}     # date -> [Cash + 10 weights]
        self.weights_headers = None   # ["Cash","AAPL",...,"XOM"]

        reader = csv.reader(io.StringIO(csv_text))
        for row in reader:
            if not row:
                continue

            if row[0].strip().lower() == "date":
                self.weights_headers = [x.strip() for x in row[1:]]  # drop "date"
                continue

            if len(row) != 12:
                self.debug(f"Skipping bad row len={len(row)}: {row[:3]}...")
                continue

            d = datetime.strptime(row[0].strip(), "%Y-%m-%d").date()
            weights = [float(x) for x in row[1:]]  # cash + 10 assets
            self.weights_by_date[d] = weights

        self.debug(f"Loaded {key}: days={len(self.weights_by_date)} | headers={self.weights_headers}")

        self.tickers = sorted(self.weights_headers[1:])

        for t in self.tickers:
            self.add_equity(t, Resolution.DAILY)

        self.last_rebalance_date = None

    def on_data(self, data: Slice):
        today = self.time.date()

        # once per day
        if self.last_rebalance_date == today:
            return
        self.last_rebalance_date = today

        row = self.weights_by_date.get(today)
        if row is None:
            self.debug(f"{today} | No weights")
            return

        if not self.weights_headers:
            self.debug(f"{today} | Missing CSV headers, cannot map weights safely")
            return

        weights_map = dict(zip(self.weights_headers, row))

        weights_str = ", ".join(f"{name}={float(val):.4f}" for name, val in weights_map.items())

        self.debug(f"Portfolio Value = {self.portfolio.total_portfolio_value}")

        tick_line = " | ".join(
            f"{t}: w={max(0.0, min(float(weights_map.get(t, 0.0)), 1.0)):.6f}, "
            f"close={(data[self.symbol(t)].close if (data.contains_key(self.symbol(t)) and data[self.symbol(t)]) else float('nan')):.2f}"
            if (data.contains_key(self.symbol(t)) and data[self.symbol(t)])
            else f"{t}: w={max(0.0, min(float(weights_map.get(t, 0.0)), 1.0)):.6f}, close=N/A"
            for t in self.tickers
        )
        self.debug(f"{today} | {tick_line}")

        for t in self.tickers:
            original_w = float(weights_map.get(t, 0.0))
            w = max(0.0, min(original_w, 1.0))

            self.set_holdings(t, w)