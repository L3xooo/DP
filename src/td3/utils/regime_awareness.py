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
                                       norm_window: int = 252) -> np.ndarray:
    """
    Calculate a market-wide regime indicator similar to VIX.

    This computes the cross-sectional average of realized volatility
    across all assets, serving as a market fear gauge.

    Args:
        prices: Array of prices (T, N) where T is time steps and N is assets
        window: Rolling window size for volatility calculation
        norm_window: Trailing lookback window (in time steps) used to bound the
            min/max normalization. Using a trailing window instead of the full
            expanding history keeps the 0-100 scale comparably calibrated
            throughout the series (e.g. "70" means roughly the same thing in
            year 1 as in year 5), rather than drifting as new all-time extremes
            get set. Normalization remains fully causal (no lookahead).

    Returns:
        Array of regime indicator values (T,)
    """
    vol = calculate_realized_volatility(prices, window)
    market_vol = np.mean(vol, axis=1)

    # Rolling (trailing) min/max normalization, causal and O(T) via pandas.
    # min_periods=1 makes it behave like an expanding window until enough
    # history accumulates to fill norm_window, then it becomes a true
    # trailing window.
    market_vol_series = pd.Series(market_vol)
    running_min = market_vol_series.rolling(window=norm_window, min_periods=1).min().values
    running_max = market_vol_series.rolling(window=norm_window, min_periods=1).max().values

    range_vol = running_max - running_min + 1e-12
    normalized = ((market_vol - running_min) / range_vol) * 100
    normalized[0] = 0

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


def classify_regime_step2(regime_indicator: np.ndarray, current_regime: int, 
                          low_thresh: float = 30.0, high_thresh: float = 70.0, 
                          buffer: float = 3.0) -> int:
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
    
    # Direct transition thresholds (lower than normal to allow direct Bullish<->Bearish switches)
    direct_high_thresh = high_thresh + buffer - 5.0  # 70 (was 75, now more achievable)
    direct_low_thresh = low_thresh - buffer + 5.0    # 30 (was 25, now more achievable)
    
    if current_regime == 0:    # Currently Bullish
        if val > direct_high_thresh:
            return 2           # Direct switch to Bearish (requires breaking past 70)
        elif val > (low_thresh + buffer): 
            return 1           # Switch to Neutral (requires breaking past 35)
    elif current_regime == 1:  # Currently Neutral
        if val < (low_thresh - buffer): 
            return 0           # Switch down to Bullish (requires dropping below 25)
        elif val > (high_thresh + buffer): 
            return 2           # Switch up to Bearish (requires breaking past 75)
    elif current_regime == 2:  # Currently Bearish
        if val < direct_low_thresh:
            return 0           # Direct switch to Bullish (requires dropping below 30)
        elif val < (high_thresh - buffer): 
            return 1           # Switch to Neutral (requires dropping below 65)
        
    return current_regime      # Maintain current regime state if within the buffer zones


def calculate_yang_zhang_volatility(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Yang-Zhang volatility estimator using OHLC prices.

    Decomposes price movement into overnight gaps, open-to-close drift,
    and intraday range (Rogers-Satchell), then blends them into a single
    variance estimate. Reacts faster than close-to-close vol because a
    single bar with a wide high-low range immediately inflates the signal.

    Args:
        open_prices:  (T, N) array of open prices.
        high_prices:  (T, N) array of high prices.
        low_prices:   (T, N) array of low prices.
        close_prices: (T, N) array of close prices.
        window: Rolling window size in bars.

    Returns:
        (T, N) array of Yang-Zhang volatility values.
    """
    eps = 1e-12

    log_ho = np.log(high_prices[1:] / (open_prices[1:] + eps) + eps)
    log_lo = np.log(low_prices[1:] / (open_prices[1:] + eps) + eps)
    log_co = np.log(close_prices[1:] / (open_prices[1:] + eps) + eps)

    # Overnight: open relative to previous close
    log_oc = np.log(open_prices[1:] / (close_prices[:-1] + eps) + eps)
    # Close-to-close
    log_cc = np.log(close_prices[1:] / (close_prices[:-1] + eps) + eps)

    # Rogers-Satchell: captures intraday range regardless of direction
    rs_elements = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    # Rolling stats via pandas (handles NaN edges cleanly)
    T_minus1, N = log_oc.shape
    var_overnight = np.zeros((T_minus1, N))
    var_oc = np.zeros((T_minus1, N))
    var_rs = np.zeros((T_minus1, N))

    for col in range(N):
        var_overnight[:, col] = (
            pd.Series(log_oc[:, col]).rolling(window=window, min_periods=2).var().values
        )
        var_oc[:, col] = (
            pd.Series(log_cc[:, col]).rolling(window=window, min_periods=2).var().values
        )
        var_rs[:, col] = (
            pd.Series(rs_elements[:, col]).rolling(window=window, min_periods=2).mean().values
        )

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_variance = var_overnight + k * var_oc + (1 - k) * var_rs

    yz_vol = np.sqrt(np.clip(yz_variance, 0, None))

    # Pad first row with zeros to match original shape (T, N)
    result = np.zeros_like(close_prices)
    result[1:] = np.nan_to_num(yz_vol, nan=0.0)
    return result


def apply_volume_boost(
    close_prices: np.ndarray,
    volume: np.ndarray,
    base_vol: np.ndarray,
    vol_zscore_window: int = 50,
) -> np.ndarray:
    """
    Amplify a volatility signal on high-volume down bars.

    On bars where: (1) the return is negative AND (2) volume is unusually
    high (positive z-score relative to recent history), the base vol is
    multiplied by (1 + volume_z_score). Otherwise the multiplier is 1.0.

    This makes the signal spike immediately on heavy selling — volume
    surges can precede the worst of a decline.

    Args:
        close_prices: (T, N) close prices for computing bar returns.
        volume:       (T, N) volume per bar.
        base_vol:     (T, N) any volatility estimate to boost.
        vol_zscore_window: Lookback for volume z-score (in bars).

    Returns:
        (T, N) array of volume-boosted stress values.
    """
    T, N = close_prices.shape
    boosted = base_vol.copy()

    for col in range(N):
        returns = np.zeros(T)
        returns[1:] = np.log(close_prices[1:, col] / (close_prices[:-1, col] + 1e-12) + 1e-12)

        vol_series = pd.Series(volume[:, col])
        rolling_mean = vol_series.rolling(window=vol_zscore_window, min_periods=2).mean().values
        rolling_std = vol_series.rolling(window=vol_zscore_window, min_periods=2).std().values

        z_score = (volume[:, col] - rolling_mean) / (rolling_std + 1e-12)
        volume_shock = np.clip(z_score, 0, None)  # Only positive shocks
        is_down = (returns < 0).astype(float)

        multiplier = 1.0 + (volume_shock * is_down)
        boosted[:, col] = base_vol[:, col] * multiplier

    return boosted


def calculate_market_stress_indicator(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    volume: np.ndarray,
    vol_window: int = 20,
    vol_zscore_window: int = 50,
    norm_window: int = 252,
) -> np.ndarray:
    """
    Market-wide stress indicator using Yang-Zhang vol + volume boosting.

    Pipeline: OHLCV → YZ vol per asset → volume boost per asset →
    cross-sectional mean → trailing min-max normalization to 0-100.

    Args:
        open_prices:  (T, N) open prices.
        high_prices:  (T, N) high prices.
        low_prices:   (T, N) low prices.
        close_prices: (T, N) close prices.
        volume:       (T, N) volume.
        vol_window:       Rolling window for YZ vol (in bars).
        vol_zscore_window: Lookback for volume z-score (in bars).
        norm_window:  Trailing window for min-max normalization (in bars).

    Returns:
        (T,) array of stress indicator values scaled 0-100.
    """
    # Step 1: Yang-Zhang vol per asset
    yz_vol = calculate_yang_zhang_volatility(
        open_prices, high_prices, low_prices, close_prices, window=vol_window
    )

    # Step 2: Volume-boosted stress per asset
    boosted = apply_volume_boost(close_prices, volume, yz_vol, vol_zscore_window)

    # Step 3: Cross-sectional mean → single market-wide signal
    market_stress = np.mean(boosted, axis=1)

    # Step 4: Trailing min-max normalization (causal, no lookahead)
    stress_series = pd.Series(market_stress)
    running_min = stress_series.rolling(window=norm_window, min_periods=1).min().values
    running_max = stress_series.rolling(window=norm_window, min_periods=1).max().values

    range_stress = running_max - running_min + 1e-12
    normalized = ((market_stress - running_min) / range_stress) * 100
    normalized[0] = 0

    return np.clip(normalized, 0, 100)


def add_regime_features(features: np.ndarray,
                       prices: np.ndarray,
                       window: int = 20,
                       norm_window: int = 252) -> np.ndarray:
    """
    Add regime awareness features to the existing feature array.
    
    Args:
        features: Existing feature array (T, N, F)
        prices: Price array (T, N)
        window: Rolling window for volatility calculation
        norm_window: Trailing lookback window used to normalize the market
            regime indicator (see calculate_market_regime_indicator)
        
    Returns:
        Augmented feature array (T, N, F+3) with additional regime features:
        - realized_volatility: Asset-specific volatility
        - market_regime: Market-wide regime indicator (broadcast to all assets)
        - regime_classification: Discrete regime class (0, 1, 2)
    """
    T, N, F = features.shape
    
    # Calculate regime indicators
    realized_vol = calculate_realized_volatility(prices, window)
    market_regime = calculate_market_regime_indicator(prices, window, norm_window)
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