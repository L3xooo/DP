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
                                       window: int = 20) -> np.ndarray:
    """
    Calculate a market-wide regime indicator similar to VIX.
    
    This computes the cross-sectional average of realized volatility
    across all assets, serving as a market fear gauge.
    
    Args:
        prices: Array of prices (T, N) where T is time steps and N is assets
        window: Rolling window size for volatility calculation
        
    Returns:
        Array of regime indicator values (T,)
    """
    vol = calculate_realized_volatility(prices, window)
    market_vol = np.mean(vol, axis=1)

    normalized = np.zeros_like(market_vol)
    # Step-by-step rolling expansion of min/max bounds
    for t in range(1, len(market_vol)):
        history = market_vol[:t+1]
        min_vol = np.percentile(history, 5)
        max_vol = np.percentile(history, 95)
        range_vol = max_vol - min_vol + 1e-12
        normalized[t] = ((market_vol[t] - min_vol) / range_vol) * 100

    return np.clip(normalized, 0, 100)


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


def add_regime_features(features: np.ndarray,
                       prices: np.ndarray,
                       window: int = 20) -> np.ndarray:
    """
    Add regime awareness features to the existing feature array.
    
    Args:
        features: Existing feature array (T, N, F)
        prices: Price array (T, N)
        window: Rolling window for volatility calculation
        
    Returns:
        Augmented feature array (T, N, F+3) with additional regime features:
        - realized_volatility: Asset-specific volatility
        - market_regime: Market-wide regime indicator (broadcast to all assets)
        - regime_classification: Discrete regime class (0, 1, 2)
    """
    T, N, F = features.shape
    
    # Calculate regime indicators
    realized_vol = calculate_realized_volatility(prices, window)
    market_regime = calculate_market_regime_indicator(prices, window)
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
