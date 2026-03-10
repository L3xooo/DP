from AlgorithmImports import *


class BenchmarkPortfolio(QCAlgorithm):
    def __init__(self):
        super().__init__()
        self.invested_once = None
        self.equal_weight = None
        self.tickers = None

    def initialize(self):
        self.set_security_initializer(lambda s: s.set_fee_model(ConstantFeeModel(0, "USD")))

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

        self.invested_once = False
        self.equal_weight = 1.0 / (len(self.tickers) + 1)

    def on_data(self, data: Slice):
        if self.invested_once:
            return
        if any(
            not data.contains_key(self.symbol(t)) or data[self.symbol(t)] is None
            for t in self.tickers
        ):
            return
        for t in self.tickers:
            self.set_holdings(t, self.equal_weight)

        self.invested_once = True
