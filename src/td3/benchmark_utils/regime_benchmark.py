"""
Benchmark regime classification using Hidden Markov Model (HMM).

Provides a principled statistical benchmark for evaluating regime detection systems.
The HMM is fitted on the full return series (full-sample fitting, no lookahead),
representing the "oracle" regime labels that an optimal statistical model would assign.

This is acceptable for benchmarking because:
1. We're evaluating classification quality, not online prediction
2. HMM shows the gap between causal (your system) and optimal (HMM)
3. Results clearly document the lookahead nature of the benchmark

Author: Devin (Evaluation Framework)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HMMBenchmarkConfig:
    """Configuration for HMM benchmark."""
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42
    verbose: bool = False


def compute_benchmark_regimes(
    prices: np.ndarray,
    config: Optional[HMMBenchmarkConfig] = None,
) -> np.ndarray:
    """
    Compute benchmark regime labels using a 3-state Gaussian HMM fitted on returns.

    This function:
    1. Computes log returns from prices
    2. Fits a 3-state GaussianHMM to the full return series (full-sample fitting)
    3. Extracts the hidden state sequence
    4. Maps states to regime labels {0: Bullish, 1: Neutral, 2: Bearish}
       based on mean return ordering (lowest return = Bearish, highest = Bullish)

    Args:
        prices: Array of shape (T, N) or (T,) containing asset prices.
                If (T, N), uses cross-sectional mean of prices.
                If (T,), uses prices directly (e.g., index prices).
        config: HMMBenchmarkConfig instance. If None, uses defaults.

    Returns:
        Array of shape (T,) with integer regime labels {0, 1, 2}.
        - 0: Bullish (low volatility, high returns)
        - 1: Neutral (moderate volatility, moderate returns)
        - 2: Bearish (high volatility, low returns)

    Notes:
        - This function uses full-sample HMM fitting, which introduces lookahead bias.
        - This is acceptable for benchmarking because we're evaluating classification
          quality against an "oracle" model, not online prediction capability.
        - The regime mapping is determined by mean return of each HMM state.
    """
    if config is None:
        config = HMMBenchmarkConfig()

    # Ensure prices are 1D (T,)
    if prices.ndim == 2:
        prices = np.mean(prices, axis=1)
    elif prices.ndim != 1:
        raise ValueError(f"prices must be 1D or 2D, got shape {prices.shape}")

    # Compute log returns
    log_returns = np.log(prices[1:] / prices[:-1] + 1e-12)

    # Import HMM here to avoid hard dependency if not used
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError(
            "hmmlearn is required for HMM benchmark. "
            "Install with: pip install hmmlearn"
        )

    # Fit HMM on returns (reshape to (T, 1) for hmmlearn)
    X = log_returns.reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        random_state=config.random_state,
        verbose=config.verbose,
    )
    hmm.fit(X)

    # Extract hidden state sequence
    hidden_states = hmm.predict(X)  # Shape (T-1,)

    # Map HMM states to regime labels based on mean return
    # State with lowest mean return -> Bearish (2)
    # State with highest mean return -> Bullish (0)
    state_means = hmm.means_.flatten()
    state_order = np.argsort(state_means)  # Ascending order: low, medium, high
    state_to_regime = {
        state_order[0]: 2,  # Lowest return -> Bearish
        state_order[1]: 1,  # Medium return -> Neutral
        state_order[2]: 0,  # Highest return -> Bullish
    }

    # Apply mapping
    regimes = np.array([state_to_regime[s] for s in hidden_states], dtype=int)

    # Pad first element (lost in return calculation) with initial regime
    # Use the regime at t=1 as the regime for t=0
    regimes = np.concatenate([[regimes[0]], regimes])

    return regimes


def compute_benchmark_regimes_causal(
    prices: np.ndarray,
    window: int = 252,
    config: Optional[HMMBenchmarkConfig] = None,
) -> np.ndarray:
    """
    Compute benchmark regime labels using a rolling HMM (causal, no lookahead).

    This is an alternative to the full-sample HMM that respects causality:
    - At each time step t, fit HMM only on data up to t
    - Use the predicted state at t as the regime label
    - Requires at least `window` observations before producing labels

    This is slower (O(T²) instead of O(T)) but eliminates lookahead bias.
    Useful for comparing against your causal system.

    Args:
        prices: Array of shape (T, N) or (T,) containing asset prices.
        window: Minimum number of observations required before fitting HMM.
        config: HMMBenchmarkConfig instance. If None, uses defaults.

    Returns:
        Array of shape (T,) with integer regime labels {0, 1, 2}.
        First `window` elements are set to 1 (Neutral) due to insufficient data.
    """
    if config is None:
        config = HMMBenchmarkConfig()

    # Ensure prices are 1D (T,)
    if prices.ndim == 2:
        prices = np.mean(prices, axis=1)
    elif prices.ndim != 1:
        raise ValueError(f"prices must be 1D or 2D, got shape {prices.shape}")

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError(
            "hmmlearn is required for HMM benchmark. "
            "Install with: pip install hmmlearn"
        )

    T = len(prices)
    regimes = np.ones(T, dtype=int)  # Default to Neutral

    # Compute log returns
    log_returns = np.log(prices[1:] / prices[:-1] + 1e-12)

    # Rolling HMM fit
    for t in range(window, T):
        # Fit HMM on data up to time t
        X = log_returns[:t].reshape(-1, 1)
        hmm = GaussianHMM(
            n_components=config.n_states,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            random_state=config.random_state,
            verbose=False,
        )
        hmm.fit(X)

        # Predict state at time t
        X_t = log_returns[t].reshape(1, 1)
        hidden_state = hmm.predict(X_t)[0]

        # Map HMM state to regime based on mean return
        state_means = hmm.means_.flatten()
        state_order = np.argsort(state_means)
        state_to_regime = {
            state_order[0]: 2,  # Lowest return -> Bearish
            state_order[1]: 1,  # Medium return -> Neutral
            state_order[2]: 0,  # Highest return -> Bullish
        }
        regimes[t] = state_to_regime[hidden_state]

    return regimes


def compute_benchmark_regimes_vix_percentile(
    prices: np.ndarray,
    vol_window: int = 20,
    norm_window: int = 252,
    low_percentile: float = 33.33,
    high_percentile: float = 66.67,
) -> np.ndarray:
    """
    Compute benchmark regime labels using VIX-percentile classification (causal).

    This is a simpler, causal alternative to HMM:
    - Compute realized volatility (rolling std of log returns)
    - Normalize to 0-100 scale using trailing window percentiles
    - Classify: vol < 33rd percentile = Bullish, 33-67 = Neutral, > 67 = Bearish

    This is fully causal (no lookahead) and directly comparable to your system.

    Args:
        prices: Array of shape (T, N) or (T,) containing asset prices.
        vol_window: Rolling window for volatility calculation.
        norm_window: Trailing window for percentile normalization.
        low_percentile: Percentile threshold for Bullish/Neutral boundary.
        high_percentile: Percentile threshold for Neutral/Bearish boundary.

    Returns:
        Array of shape (T,) with integer regime labels {0, 1, 2}.
    """
    # Ensure prices are 1D (T,)
    if prices.ndim == 2:
        prices = np.mean(prices, axis=1)
    elif prices.ndim != 1:
        raise ValueError(f"prices must be 1D or 2D, got shape {prices.shape}")

    # Compute log returns
    log_returns = np.log(prices[1:] / prices[:-1] + 1e-12)

    # Compute rolling volatility
    vol = pd.Series(log_returns).rolling(window=vol_window, min_periods=2).std().values
    vol = np.concatenate([[0], vol])  # Pad first element

    # Compute rolling percentiles (causal, trailing window)
    vol_series = pd.Series(vol)
    low_pctl = vol_series.rolling(window=norm_window, min_periods=1).quantile(low_percentile / 100).values
    high_pctl = vol_series.rolling(window=norm_window, min_periods=1).quantile(high_percentile / 100).values

    # Classify based on percentiles
    regimes = np.ones(len(prices), dtype=int)  # Default to Neutral
    regimes[vol <= low_pctl] = 0  # Bullish
    regimes[vol >= high_pctl] = 2  # Bearish

    return regimes
