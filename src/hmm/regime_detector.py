"""
HMM-based regime detector.

Fits a 2-state Gaussian HMM on daily returns and provides
regime probability estimates for each trading day.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:
    """Wraps HMM fitting and regime probability computation."""

    def __init__(self, n_components=2, n_iter=10000, random_state=42):
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            tol=1e-4,
        )
        self.bull_regime = None

    def fit(self, close_prices: pd.DataFrame):
        """
        Fit the HMM on daily log-returns.

        Args:
            close_prices: DataFrame of shape (T_days, N_tickers) with daily close prices,
                          indexed by datetime.
        """
        # Compute daily log-returns directly
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

        avg_daily = log_returns.mean(axis=1)
        self.daily_index = avg_daily.index
        returns = avg_daily.values.reshape(-1, 1)

        self.model.fit(returns)

        # Identify bull regime (higher mean)
        if self.model.means_[0, 0] > self.model.means_[1, 0]:
            self.bull_regime = 0
        else:
            self.bull_regime = 1

        # Compute daily regime probabilities
        daily_probs = self.model.predict_proba(returns)[:, self.bull_regime]

        # Shift by 1 day (paper uses ĥ_{t-1})
        self.daily_prob_series = pd.Series(daily_probs, index=self.daily_index)
        self.daily_prob_shifted = self.daily_prob_series.shift(1)
        self.daily_prob_shifted.iloc[0] = 1.0

    def get_regime_probs(self, day_index: pd.DatetimeIndex) -> np.ndarray:
        """
        Return regime probabilities aligned to the given daily index.

        Args:
            day_index: DatetimeIndex of trading days.

        Returns:
            Array of shape (T_days,) with P(bull) for each day.
        """
        probs = day_index.map(self.daily_prob_shifted).values.astype(np.float32)
        return np.nan_to_num(probs, nan=1.0)