"""
Regime awareness utilities for portfolio management.

Provides functions to calculate market regime indicators including VIX-like measures,
realized volatility, and regime classification to enrich the state space.

Author: Peter Likavec
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_realized_volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate realized volatility using rolling standard deviation of log returns.
    
    Args:
        prices: Array of prices (T, N) where T is time steps and N is assets
        window: Rolling window size for volatility calculation
        
    Returns:
        Array of realized volatility values (T, N)
    """
    #+ 1e-12 adds tiny value to prevent division by zero
    # np.log() converts to log returns (logarithmic percentage changes)

    log_returns = np.log(prices[1:] / prices[:-1] + 1e-12)
    df_returns = pd.DataFrame(log_returns)
    rolling_std = df_returns.rolling(window=window, min_periods=2).std().values

    vol = np.zeros_like(prices)
    vol[1:] = np.nan_to_num(rolling_std, nan=0.0)
    return vol


def calculate_market_regime_indicator(prices: np.ndarray,
                                       window: int = 20,
                                       bars_per_day: int = 1,
                                       norm_days: int = 252,
                                       lower_pct: float = 5.0,
                                       upper_pct: float = 95.0) -> np.ndarray:
    """
    Calculate a market-wide regime indicator similar to VIX.

    This computes the cross-sectional average of realized volatility
    across all assets, serving as a market fear gauge.

    Workflow:
        1. Compute per-bar realized volatility and average across assets.
        2. Aggregate to daily volatility (last bar of each day).
        3. Normalize using a rolling percentile window over strictly past days.
        4. Broadcast day T-1's indicator to day T's bars (lag-1 shift).

    The lag-1 shift ensures no look-ahead bias: every intraday bar only
    sees information available at the prior day's close.  The normalization
    window is in daily units (default 252 ≈ 1 trading year).

    Args:
        prices: Array of prices (T, N) where T is time steps and N is assets
        window: Rolling window size for volatility calculation (in bars)
        bars_per_day: Number of bars per trading day (1 for daily data,
                      13 for 30-min intraday bars)
        norm_days: Lookback in trading days for percentile normalization
                   (default 252 ≈ 1 year)
        lower_pct: Lower percentile for normalization (default 5.0)
        upper_pct: Upper percentile for normalization (default 95.0)

    Returns:
        Array of regime indicator values (T,) in range [0, 100],
        one value per bar (constant within each trading day).
    """
    vol = calculate_realized_volatility(prices, window)
    market_vol = np.mean(vol, axis=1)  # (T,)

    print(f"[REGIME DEBUG] Using DAILY rolling percentile normalization "
          f"(bars_per_day={bars_per_day}, norm_days={norm_days}, "
          f"pct={lower_pct}-{upper_pct})")

    # ── Step 1: Aggregate to daily ───────────────────────────────────
    # Take the last bar of each day as the daily volatility reading
    T = len(market_vol)
    day_indices = list(range(bars_per_day - 1, T, bars_per_day))
    if day_indices[-1] != T - 1:
        day_indices.append(T - 1)  # include final partial day
    daily_vol = market_vol[day_indices]  # (num_days,)

    # ── Step 2: Rolling percentile normalization on daily data ───────
    # daily_normalized[d] represents day d-1's vol normalised against
    # days [start, d).  This builds in a 1-day lag so that day d's bars
    # only ever see information available at the close of day d-1.
    num_days = len(daily_vol)
    daily_normalized = np.zeros(num_days)

    for d in range(1, num_days):
        start = max(0, d - norm_days)
        history = daily_vol[start:d]  # strictly past: [start, d)

        p_low = np.percentile(history, lower_pct)
        p_high = np.percentile(history, upper_pct)
        range_vol = p_high - p_low + 1e-12

        daily_normalized[d] = ((daily_vol[d - 1] - p_low) / range_vol) * 100

    daily_normalized = np.clip(daily_normalized, 0, 100)

    # ── Step 3: Broadcast to bars (no extra shift) ───────────────────
    # The 1-day lag is already in daily_normalized, so day i's bars
    # receive daily_normalized[i] directly.
    normalized = np.zeros(T)
    for i, day_end in enumerate(day_indices):
        day_start = 0 if i == 0 else day_indices[i - 1] + 1
        normalized[day_start:day_end + 1] = daily_normalized[i]

    return normalized


def classify_regime(regime_indicator: np.ndarray,
                   low_threshold: float = 30.0,
                   high_threshold: float = 70.0) -> np.ndarray:
    """
    Classify market regime based on regime indicator.
    
    Args:
        regime_indicator: Array of regime indicator values
        low_threshold: Threshold below which regime is considered bullish
        high_threshold: Threshold above which regime is considered bearish
        
    Returns:
        Array of regime classifications:
        - 0: Bullish (low volatility, favorable conditions)
        - 1: Neutral (moderate volatility)
        - 2: Bearish (high volatility, unfavorable conditions)
    """
    regimes = np.zeros_like(regime_indicator, dtype=int)
    
    regimes[regime_indicator < low_threshold] = 0  # Bullish
    regimes[(regime_indicator >= low_threshold) & (regime_indicator < high_threshold)] = 1  # Neutral
    regimes[regime_indicator >= high_threshold] = 2  # Bearish
    
    return regimes


def classify_regime_step2(regime_indicator: np.ndarray, current_regime: int, 
                          low_thresh: float = 30.0, high_thresh: float = 70.0, 
                          buffer: float = 5.0) -> int:
    """
    Applies hysteresis buffer to classify the regime for a single time step,
    preventing state-flickering and stabilization noise inside the MDP loop.
    
    Args:
        regime_indicator: Array of regime indicator values (most recent value at end)
        current_regime: Current regime state (0: Bullish, 1: Neutral, 2: Bearish)
        low_thresh: Lower threshold for regime classification (default: 30.0)
        high_thresh: Upper threshold for regime classification (default: 70.0)
        buffer: Buffer zone around thresholds to prevent flickering (default: 5.0)
        
    Returns:
        Integer representing the new regime state:
        - 0: Bullish (low volatility, favorable conditions)
        - 1: Neutral (moderate volatility)
        - 2: Bearish (high volatility, unfavorable conditions)
    """
    # Grab the most recent indicator value calculated for the current day
    val = regime_indicator[-1] 
    
    if current_regime == 0:    # Currently Bullish
        if val > (low_thresh + buffer): 
            return 1           # Switch up to Neutral (requires breaking past 35)
    elif current_regime == 1:  # Currently Neutral
        if val < (low_thresh - buffer): 
            return 0           # Switch down to Bullish (requires dropping below 25)
        if val > (high_thresh + buffer): 
            return 2           # Switch up to Bearish (requires breaking past 75)
    elif current_regime == 2:  # Currently Bearish
        if val < (high_thresh - buffer): 
            return 1           # Switch down to Neutral (requires dropping below 65)
        
    return current_regime      # Maintain current regime state if within the buffer zones


def add_regime_features(features: np.ndarray,
                       prices: np.ndarray,
                       window: int = 20,
                       bars_per_day: int = 1) -> np.ndarray:
    """
    Add regime awareness features to the existing feature array.
    
    Args:
        features: Existing feature array (T, N, F)
        prices: Price array (T, N)
        window: Rolling window for volatility calculation
        bars_per_day: Number of bars per trading day (1 for daily data,
                      13 for 30-min intraday bars)
        
    Returns:
        Augmented feature array (T, N, F+3) with additional regime features:
        - realized_volatility: Asset-specific volatility
        - market_regime: Market-wide regime indicator (broadcast to all assets)
        - regime_classification: Discrete regime class (0, 1, 2)
    """
    T, N, F = features.shape
    
    # Calculate regime indicators
    realized_vol = calculate_realized_volatility(prices, window)
    market_regime = calculate_market_regime_indicator(prices, window, bars_per_day=bars_per_day)
    regime_class = classify_regime(market_regime)
    
    # Reshape for broadcasting across assets
    market_regime_broadcast = market_regime.reshape(-1, 1).repeat(N, axis=1)
    regime_class_broadcast = regime_class.reshape(-1, 1).repeat(N, axis=1)
    
    # Stack new features
    new_features = np.dstack([
        features,
        realized_vol,
        market_regime_broadcast,
        regime_class_broadcast
    ])
    
    return new_features