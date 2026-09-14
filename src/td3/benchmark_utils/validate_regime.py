"""
Regime Indicator Validation — Step 1: Lead-Lag Analysis

Compares three volatility signals against forward max drawdowns:
  A) Original close-to-close realized vol
  B) Yang-Zhang vol alone (no volume boost)
  C) Yang-Zhang vol + volume boost

Tests whether each signal leads, coincides with, or lags real forward
drawdowns via cross-correlation at multiple horizons.

Usage:
    python validate_regime.py

Outputs:
    validation/lead_lag_crosscorrelation.png  — side-by-side correlograms
    validation/lead_lag_timeseries.png        — 4-panel time series overlay

Author: Peter Likavec
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Mock torch so we can import app_config without a working torch install
if 'torch' not in sys.modules:
    import types
    _torch = types.ModuleType('torch')
    _torch.device = lambda *a, **kw: None
    _torch.cuda = types.ModuleType('torch.cuda')
    _torch.cuda.is_available = lambda: False
    sys.modules['torch'] = _torch
    sys.modules['torch.cuda'] = _torch.cuda

from td3.config.app_config import AppConfig
from td3.data.data_processor import DataProcessor
from td3.utils.regime_awareness import (
    calculate_realized_volatility,
    calculate_market_regime_indicator,
    calculate_yang_zhang_volatility,
    apply_volume_boost,
    calculate_market_stress_indicator,
)

OUTPUT_DIR = "validation"

# ─── Data frequency scaling ─────────────────────────────────────────
# The data is 30-minute bars. A standard US equity trading day has
# 6.5 hours = 13 bars of 30 minutes each.
BARS_PER_DAY = 13

BURN_IN = 252 * BARS_PER_DAY        # 1 year of trading days in bars (3,276 bars)
MAX_LAG = 20 * BARS_PER_DAY         # ±20 trading days in bars (260 bars)
HORIZONS_DAYS = [5, 10, 20]         # Forward drawdown horizons in trading days
HORIZONS = [h * BARS_PER_DAY for h in HORIZONS_DAYS]

# Volatility and normalization windows scaled to bars
VOL_WINDOW = 20 * BARS_PER_DAY        # 20 trading days (260 bars)
VOL_ZSCORE_WINDOW = 50 * BARS_PER_DAY # 50 trading days for volume z-score (650 bars)
NORM_WINDOW = 252 * BARS_PER_DAY      # 1 trading year  (3,276 bars)


# ─────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def compute_forward_max_drawdown(prices_1d: np.ndarray, horizon: int) -> np.ndarray:
    """
    For each bar t, compute the maximum drawdown over [t, t + horizon].

    Maximum drawdown = largest peak-to-trough decline within the window,
    measured as a fraction of the peak (0 = no drawdown, 1 = total loss).

    Bars where the forward window extends beyond available data are set
    to NaN so they are excluded from correlation calculations.

    Args:
        prices_1d: 1-D array of prices (e.g. equal-weighted avg).
        horizon:   Number of forward bars in the drawdown window.

    Returns:
        1-D array (same length as prices_1d) of forward MDD values.
    """
    T = len(prices_1d)
    mdd = np.full(T, np.nan)

    for t in range(T - horizon):
        window = prices_1d[t : t + horizon + 1]
        running_max = np.maximum.accumulate(window)
        drawdowns = (running_max - window) / (running_max + 1e-12)
        mdd[t] = np.max(drawdowns)

    return mdd


def compute_cross_correlation(
    signal: np.ndarray,
    target: np.ndarray,
    max_lag: int = 20,
) -> tuple:
    """
    Pearson cross-correlation between signal and target at integer lags.

    Convention:
        lag > 0 : signal[t] vs target[t + lag]  → signal LEADS target
        lag = 0 : coincident
        lag < 0 : signal[t] vs target[t + lag]  → signal LAGS target

    We want peak correlation at positive lag — that means the volatility
    signal rises BEFORE the drawdown happens.

    Args:
        signal: 1-D array (the regime indicator / raw vol).
        target: 1-D array (forward max drawdown).
        max_lag: Maximum lag in both directions.

    Returns:
        (lags, correlations) — both 1-D arrays of length 2*max_lag + 1.
    """
    T = min(len(signal), len(target))
    signal = signal[:T]
    target = target[:T]

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.full(len(lags), np.nan)

    for i, lag in enumerate(lags):
        if lag >= 0:
            s = signal[: T - lag]
            tgt = target[lag : T]
        else:
            s = signal[-lag : T]
            tgt = target[: T + lag]

        mask = ~(np.isnan(s) | np.isnan(tgt))
        if mask.sum() < 30:
            continue

        correlations[i] = np.corrcoef(s[mask], tgt[mask])[0, 1]

    return lags, correlations


# ─────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────

def plot_crosscorrelation_comparison(
    all_results: list,
    all_titles: list,
    save_path: str,
) -> None:
    """
    Side-by-side correlograms for N signals.
    One row per horizon, one column per signal.
    """
    horizons = list(all_results[0].keys())
    n_rows = len(horizons)
    n_cols = len(all_results)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    for row, h in enumerate(horizons):
        h_days = h // BARS_PER_DAY
        for col, (results, title) in enumerate(zip(all_results, all_titles)):
            ax = axes[row, col]
            lags_bars, corrs = results[h]
            lags_days = lags_bars / BARS_PER_DAY

            colors = [
                "green" if l > 0 else "red" if l < 0 else "steelblue"
                for l in lags_bars
            ]
            ax.bar(lags_days, corrs, width=1.0 / BARS_PER_DAY, color=colors,
                   alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

            peak_idx = np.nanargmax(np.abs(corrs))
            peak_lag_days = lags_bars[peak_idx] / BARS_PER_DAY
            peak_r = corrs[peak_idx]

            ax.set_title(
                f"{title}  |  {h_days}d MDD  |  "
                f"Peak lag = {peak_lag_days:+.1f}d  (r = {peak_r:.3f})",
                fontsize=10,
            )
            ax.set_xlabel("Lag in trading days (positive = signal leads)")
            ax.set_ylabel("Pearson r")
            ax.set_xlim(-MAX_LAG / BARS_PER_DAY - 0.5, MAX_LAG / BARS_PER_DAY + 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_timeseries_comparison(
    dates,
    eq_price: np.ndarray,
    indicator_original: np.ndarray,
    indicator_yz_only: np.ndarray,
    indicator_yz_boosted: np.ndarray,
    forward_mdd_10d: np.ndarray,
    save_path: str,
) -> None:
    """Five-panel time series: price, original, YZ-only, YZ+boost, forward MDD."""
    date_index = pd.to_datetime(dates)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Panel 1 — Equal-weighted price
    ax1.plot(date_index, eq_price, color="steelblue", linewidth=1)
    ax1.set_ylabel("Eq-Weighted Avg Price")
    ax1.set_title("Price / Original / YZ Only / YZ+Volume Boost / Forward 10d MDD")
    ax1.grid(True, alpha=0.3)

    # Panel 2 — Original regime indicator
    ax2.plot(date_index, indicator_original, color="darkorange", linewidth=1)
    ax2.axhline(y=30, color="green", linestyle="--", alpha=0.7)
    ax2.axhline(y=70, color="red", linestyle="--", alpha=0.7)
    ax2.fill_between(date_index, 0, 30, alpha=0.08, color="green")
    ax2.fill_between(date_index, 70, 100, alpha=0.08, color="red")
    ax2.set_ylabel("Original (0-100)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — YZ only (no volume boost)
    ax3.plot(date_index, indicator_yz_only, color="teal", linewidth=1)
    ax3.axhline(y=30, color="green", linestyle="--", alpha=0.7)
    ax3.axhline(y=70, color="red", linestyle="--", alpha=0.7)
    ax3.fill_between(date_index, 0, 30, alpha=0.08, color="green")
    ax3.fill_between(date_index, 70, 100, alpha=0.08, color="red")
    ax3.set_ylabel("YZ Only (0-100)")
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.3)

    # Panel 4 — YZ + volume boosted indicator
    ax4.plot(date_index, indicator_yz_boosted, color="purple", linewidth=1)
    ax4.axhline(y=30, color="green", linestyle="--", alpha=0.7)
    ax4.axhline(y=70, color="red", linestyle="--", alpha=0.7)
    ax4.fill_between(date_index, 0, 30, alpha=0.08, color="green")
    ax4.fill_between(date_index, 70, 100, alpha=0.08, color="red")
    ax4.set_ylabel("YZ+Vol Boost (0-100)")
    ax4.set_ylim(-5, 105)
    ax4.grid(True, alpha=0.3)

    # Panel 5 — Forward 10-day max drawdown
    ax5.fill_between(date_index, 0, forward_mdd_10d, color="crimson", alpha=0.4)
    ax5.plot(date_index, forward_mdd_10d, color="crimson", linewidth=0.8)
    ax5.set_ylabel("Forward 10d MDD")
    ax5.set_xlabel("Date")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def run_cross_correlation(signal_name, raw_signal, forward_mdds):
    """Run cross-correlation for one signal against all horizons. Returns results dict."""
    print(f"\n{'─' * 70}")
    print(f"{signal_name:^70}")
    print(f"{'─' * 70}")

    cc_results = {}
    for h, h_days in zip(HORIZONS, HORIZONS_DAYS):
        signal_trimmed = raw_signal[BURN_IN:]
        target_trimmed = forward_mdds[h][BURN_IN:]

        lags, corrs = compute_cross_correlation(signal_trimmed, target_trimmed, MAX_LAG)
        cc_results[h] = (lags, corrs)

        peak_idx = np.nanargmax(np.abs(corrs))
        peak_lag_bars = lags[peak_idx]
        peak_lag_days = peak_lag_bars / BARS_PER_DAY
        peak_r = corrs[peak_idx]

        if peak_lag_bars > 0:
            verdict = f"LEADS by {peak_lag_days:.1f} day(s)"
        elif peak_lag_bars == 0:
            verdict = "COINCIDENT"
        else:
            verdict = f"LAGS by {abs(peak_lag_days):.1f} day(s)"

        print(f"  Horizon {h_days:2d}d  ->  peak |r| = {abs(peak_r):.4f}  at lag = {peak_lag_days:+6.1f} days  ->  {verdict}")

    print(f"{'─' * 70}")
    return cc_results


def print_verdict(signal_name, cc_results):
    """Print PASS/MARGINAL/FAIL for the 10-day horizon."""
    h_10d = 10 * BARS_PER_DAY
    lags, corrs = cc_results[h_10d]
    peak_idx = np.nanargmax(np.abs(corrs))
    lag_bars = lags[peak_idx]
    lag_days = lag_bars / BARS_PER_DAY

    print(f"\n  [{signal_name}]")
    if lag_bars > 0:
        print(f"  PASS: Signal leads forward 10d drawdowns by {lag_days:.1f} day(s).")
    elif lag_bars == 0:
        print(f"  MARGINAL: Signal is coincident with forward 10d drawdowns.")
    else:
        print(f"  FAIL: Signal lags forward 10d drawdowns by {abs(lag_days):.1f} day(s).")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load full panel (all columns including OHLCV) ─────────────
    app_config = AppConfig()
    dp = DataProcessor(data_dir=app_config.data_dir, parquet_dir=app_config.parquet_dir)

    panel = dp.load_panel(
        tickers=app_config.ticker_config.tickers,
        start=app_config.start_date,
        end=app_config.end_date,
    )

    tickers = sorted(panel.columns.get_level_values(0).unique().tolist())
    dates = panel.index.strftime('%Y-%m-%d').tolist()
    N = len(tickers)
    T = len(panel)

    # Build (T, N) arrays for each OHLCV column
    close_prices = np.column_stack([panel[(t, 'close')].values for t in tickers]).astype(np.float64)
    open_prices  = np.column_stack([panel[(t, 'open')].values  for t in tickers]).astype(np.float64)
    high_prices  = np.column_stack([panel[(t, 'high')].values  for t in tickers]).astype(np.float64)
    low_prices   = np.column_stack([panel[(t, 'low')].values   for t in tickers]).astype(np.float64)
    volume       = np.column_stack([panel[(t, 'volume_base')].values for t in tickers]).astype(np.float64)

    print(f"Loaded {T} bars ({T // BARS_PER_DAY} trading days), {N} assets: {tickers}")
    print(f"Data frequency: 30-min bars ({BARS_PER_DAY} bars/day)")
    print(f"Date range: {dates[0]}  ->  {dates[-1]}")
    print(f"Burn-in exclusion: first {BURN_IN} bars ({BURN_IN // BARS_PER_DAY} trading days)")

    # ── 2. Compute SIGNAL A: Original close-to-close vol ─────────────
    vol_original = calculate_realized_volatility(close_prices, window=VOL_WINDOW)
    raw_vol_original = np.mean(vol_original, axis=1)  # (T,)

    indicator_original = calculate_market_regime_indicator(
        close_prices, window=VOL_WINDOW, norm_window=NORM_WINDOW
    )

    # ── 3. Compute SIGNAL B: Yang-Zhang only (no volume boost) ──────
    yz_vol = calculate_yang_zhang_volatility(
        open_prices, high_prices, low_prices, close_prices, window=VOL_WINDOW
    )
    raw_vol_yz_only = np.mean(yz_vol, axis=1)  # (T,)

    # Normalize YZ-only to 0-100 the same way as the original indicator
    yz_only_series = pd.Series(raw_vol_yz_only)
    yz_only_min = yz_only_series.rolling(window=NORM_WINDOW, min_periods=1).min().values
    yz_only_max = yz_only_series.rolling(window=NORM_WINDOW, min_periods=1).max().values
    yz_only_range = yz_only_max - yz_only_min + 1e-12
    indicator_yz_only = np.clip(((raw_vol_yz_only - yz_only_min) / yz_only_range) * 100, 0, 100)
    indicator_yz_only[0] = 0

    # ── 4. Compute SIGNAL C: Yang-Zhang + volume boost ───────────────
    yz_boosted = apply_volume_boost(
        close_prices, volume, yz_vol, vol_zscore_window=VOL_ZSCORE_WINDOW
    )
    raw_vol_yz_boosted = np.mean(yz_boosted, axis=1)  # (T,)

    indicator_yz_boosted = calculate_market_stress_indicator(
        open_prices, high_prices, low_prices, close_prices, volume,
        vol_window=VOL_WINDOW, vol_zscore_window=VOL_ZSCORE_WINDOW, norm_window=NORM_WINDOW,
    )

    # ── 5. Compute TARGET: forward max drawdown ──────────────────────
    eq_price = np.mean(close_prices, axis=1)

    forward_mdds = {}
    for h, h_days in zip(HORIZONS, HORIZONS_DAYS):
        forward_mdds[h] = compute_forward_max_drawdown(eq_price, h)
        valid = np.sum(~np.isnan(forward_mdds[h][BURN_IN:]))
        print(f"  Forward {h_days:2d}d MDD ({h} bars): {valid} valid observations (after burn-in)")

    # ── 6. Cross-correlation for all three signals ───────────────────
    cc_original = run_cross_correlation(
        "SIGNAL A: Close-to-Close Realized Vol (original)", raw_vol_original, forward_mdds
    )
    cc_yz_only = run_cross_correlation(
        "SIGNAL B: Yang-Zhang Vol Only (no volume boost)", raw_vol_yz_only, forward_mdds
    )
    cc_yz_boosted = run_cross_correlation(
        "SIGNAL C: Yang-Zhang Vol + Volume Boost", raw_vol_yz_boosted, forward_mdds
    )

    # ── 7. Verdicts ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"{'VERDICTS (based on 10-day horizon)':^70}")
    print(f"{'=' * 70}")
    print_verdict("Close-to-Close (original)", cc_original)
    print_verdict("Yang-Zhang Only", cc_yz_only)
    print_verdict("Yang-Zhang + Volume Boost", cc_yz_boosted)
    print()

    # ── 8. Save plots ────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_crosscorrelation_comparison(
        [cc_original, cc_yz_only, cc_yz_boosted],
        ["Close-to-Close (original)", "Yang-Zhang Only", "YZ + Volume Boost"],
        save_path=os.path.join(OUTPUT_DIR, "lead_lag_crosscorrelation.png"),
    )
    print(f"Saved: {OUTPUT_DIR}/lead_lag_crosscorrelation.png")

    h_10d = 10 * BARS_PER_DAY
    plot_timeseries_comparison(
        dates, eq_price,
        indicator_original, indicator_yz_only, indicator_yz_boosted,
        forward_mdds[h_10d],
        save_path=os.path.join(OUTPUT_DIR, "lead_lag_timeseries.png"),
    )
    print(f"Saved: {OUTPUT_DIR}/lead_lag_timeseries.png")


if __name__ == "__main__":
    main()
