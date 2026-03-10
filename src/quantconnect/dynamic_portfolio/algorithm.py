from AlgorithmImports import *
import csv, io
from datetime import datetime


class DynamicPortfolio(QCAlgorithm):
    def __init__(self):
        super().__init__()
        self.last_rebalance_date = None
        self.weights_headers = None
        self.weights_by_date = None
        self.tickers = None

    def initialize(self):
        self.set_security_initializer(
            lambda security: security.set_fee_model(ConstantFeeModel(0, "USD"))
        )

        self.set_start_date(2019, 1, 21)
        self.set_end_date(2023, 12, 24)
        self.set_cash(10000)

        self.tickers = [
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "NVDA",
            "JPM",
            "JNJ",
            "XOM",
        ]
        for t in self.tickers:
            self.add_equity(t, Resolution.DAILY)

        key = "weights.csv"
        if not self.object_store.contains_key(key):
            self.debug(f"ObjectStore key not found: {key}")
            return

        csv_text = self.object_store.read(key)

        self.weights_by_date = {}
        self.weights_headers = None

        reader = csv.reader(io.StringIO(csv_text))
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "date":
                self.weights_headers = [x.strip() for x in row[1:]]
                continue
            if len(row) != 12:
                self.debug(f"Skipping bad row len={len(row)}: {row[:3]}...")
                continue

            d = datetime.strptime(row[0].strip(), "%Y-%m-%d").date()
            weights = [float(x) for x in row[1:]]
            self.weights_by_date[d] = weights

        self.debug(
            f"Loaded {key}: days={len(self.weights_by_date)} | headers={self.weights_headers}"
        )

        self.last_rebalance_date = None

    def on_data(self, data: Slice):
        today = self.time.date()
        if self.last_rebalance_date == today:
            return

        self.last_rebalance_date = today

        row = self.weights_by_date.get(today)

        if row is None or not self.weights_headers:
            return

        weights_map = dict(zip(self.weights_headers, row))

        for t in self.tickers:
            w = max(0.0, min(float(weights_map.get(t, 0.0)), 1.0))
            self.set_holdings(t, w)
