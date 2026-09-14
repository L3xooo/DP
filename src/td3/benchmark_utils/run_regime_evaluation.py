"""
End-to-end regime classification evaluation script.

This script:
1. Loads price data using DataProcessor
2. Generates predicted regime labels using your system's logic
3. Generates benchmark (HMM) regime labels
4. Aligns them temporally
5. Runs evaluate_regimes() and prints full report
6. Generates all visualizations
7. Optionally computes daily portfolio returns for trading-impact analysis

Usage:
    python run_regime_evaluation.py [--output_dir ./results] [--include_trading_impact]

Author: Devin (Evaluation Framework)
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import argparse
import sys

from td3.config.app_config import AppConfig
from td3.data.data_processor import DataProcessor
from td3.utils.regime_awareness import calculate_market_regime_indicator
from td3.benchmark_utils.regime_benchmark import compute_benchmark_regimes, HMMBenchmarkConfig
from td3.benchmark_utils.regime_benchmark import compute_benchmark_regimes_vix_percentile
from td3.benchmark_utils.regime_evaluation import evaluate_regimes, print_regime_evaluation
from td3.benchmark_utils.regime_evaluation_plots import generate_evaluation_dashboard


def generate_predicted_regimes(
    prices: np.ndarray,
    config: AppConfig,
) -> np.ndarray:
    """
    Generate predicted regime labels using your system's logic.

    This mirrors the logic in PortfolioEnv._precompute_regimes().

    Args:
        prices: Array of shape (T, N) or (T,) containing asset prices.
        config: AppConfig instance with regime parameters.

    Returns:
        Array of shape (T,) with regime labels {0, 1, 2}.
    """
    # Ensure prices are 2D (T, N)
    if prices.ndim == 1:
        prices = prices.reshape(-1, 1)
    elif prices.ndim != 2:
        raise ValueError(f"prices must be 1D or 2D, got shape {prices.shape}")

    T = prices.shape[0]
    regimes = np.ones(T, dtype=int)  # Default to Neutral (1)

    if T <= 20:
        return regimes

    # Calculate market regime indicator
    market_indicator = calculate_market_regime_indicator(
        prices,
        window=config.regime_volatility_window,
        norm_window=config.regime_norm_window,
    )

    # Apply hysteresis-based classification in single pass (O(T))
    # Matches logic in regime_awareness.py::classify_regime_step2()
    current_regime = 1  # Start as Neutral
    low_thresh = config.regime_low_threshold
    high_thresh = config.regime_high_threshold
    buffer = 5.0

    # Direct transition thresholds (lower than normal to allow direct Bullish<->Bearish switches)
    direct_high_thresh = high_thresh + buffer - 5.0  # 70 (was 75, now more achievable)
    direct_low_thresh = low_thresh - buffer + 5.0    # 30 (was 25, now more achievable)

    for t in range(len(market_indicator)):
        if t >= 20:  # Only classify when we have enough data
            val = market_indicator[t]

            if current_regime == 0:  # Currently Bullish
                if val > direct_high_thresh:
                    current_regime = 2  # Direct switch to Bearish (requires breaking past 70)
                elif val > (low_thresh + buffer):
                    current_regime = 1  # Switch to Neutral (requires breaking past 35)
            elif current_regime == 1:  # Currently Neutral
                if val < (low_thresh - buffer):
                    current_regime = 0  # Switch to Bullish (requires dropping below 25)
                elif val > (high_thresh + buffer):
                    current_regime = 2  # Switch to Bearish (requires breaking past 75)
            elif current_regime == 2:  # Currently Bearish
                if val < direct_low_thresh:
                    current_regime = 0  # Direct switch to Bullish (requires dropping below 30)
                elif val < (high_thresh - buffer):
                    current_regime = 1  # Switch to Neutral (requires dropping below 65)

        regimes[t] = current_regime

    return regimes


def compute_daily_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute daily log returns from prices.

    Args:
        prices: Array of shape (T, N) or (T,) containing prices.

    Returns:
        Array of shape (T,) with daily log returns.
    """
    if prices.ndim == 2:
        prices = np.mean(prices, axis=1)
    elif prices.ndim != 1:
        raise ValueError(f"prices must be 1D or 2D, got shape {prices.shape}")

    log_returns = np.log(prices[1:] / prices[:-1] + 1e-12)
    return np.concatenate([[0], log_returns])  # Pad first element


def run_evaluation(
    config: AppConfig,
    output_dir: Optional[str] = None,
    include_trading_impact: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Run end-to-end regime evaluation.

    Args:
        config: AppConfig instance.
        output_dir: Directory to save results and plots.
        include_trading_impact: Whether to compute trading impact metrics.

    Returns:
        (y_true, y_pred, evaluation_result_dict)
    """
    print("\n" + "=" * 80)
    print("REGIME CLASSIFICATION EVALUATION")
    print("=" * 80)

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n[1/5] Loading price data...")
    processor = DataProcessor(
        data_dir=config.data_dir,
        parquet_dir=config.parquet_dir,
    )

    try:
        panel = processor.load_panel(
            tickers=config.ticker_config.tickers,
            filter_cols=config.filter_out,
            start=config.start_date,
            end=config.end_date,
        )
        print(f"  ✓ Loaded {len(panel)} time steps, {len(config.ticker_config.tickers)} tickers")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        raise

    # Extract prices (use close prices or first price column)
    # Assuming prices are in the first level of MultiIndex columns
    price_cols = [col for col in panel.columns if col[1] == "close"]
    if not price_cols:
        # Fallback: use first column per ticker
        price_cols = [(ticker, panel.columns.get_level_values(1)[0])
                     for ticker in config.ticker_config.tickers]

    prices_df = panel[[col for col in panel.columns if col[1] == "close"]]
    prices = prices_df.values  # Shape (T, N)
    dates = panel.index

    print(f"  ✓ Extracted prices: shape {prices.shape}")

    # ========================================================================
    # 2. GENERATE PREDICTED REGIMES (YOUR SYSTEM)
    # ========================================================================
    print("\n[2/5] Generating predicted regimes (your system)...")
    y_pred = generate_predicted_regimes(prices, config)
    print(f"  ✓ Generated {len(y_pred)} regime labels")
    print(f"    Bullish: {np.sum(y_pred == 0)} ({100 * np.sum(y_pred == 0) / len(y_pred):.1f}%)")
    print(f"    Neutral: {np.sum(y_pred == 1)} ({100 * np.sum(y_pred == 1) / len(y_pred):.1f}%)")
    print(f"    Bearish: {np.sum(y_pred == 2)} ({100 * np.sum(y_pred == 2) / len(y_pred):.1f}%)")

    # ========================================================================
    # 3. GENERATE BENCHMARK REGIMES (HMM)
    # ========================================================================
    print("\n[3/5] Generating benchmark regimes (HMM)...")
    try:
        #hmm_config = HMMBenchmarkConfig(verbose=False)
        #y_true = compute_benchmark_regimes(prices, config=hmm_config)
        y_true = compute_benchmark_regimes_vix_percentile(prices)
        print(f"  ✓ Generated {len(y_true)} regime labels")
        print(f"    Bullish: {np.sum(y_true == 0)} ({100 * np.sum(y_true == 0) / len(y_true):.1f}%)")
        print(f"    Neutral: {np.sum(y_true == 1)} ({100 * np.sum(y_true == 1) / len(y_true):.1f}%)")
        print(f"    Bearish: {np.sum(y_true == 2)} ({100 * np.sum(y_true == 2) / len(y_true):.1f}%)")
    except ImportError as e:
        print(f"  ✗ HMM benchmark requires hmmlearn: {e}")
        print("    Install with: pip install hmmlearn")
        raise

    # ========================================================================
    # 4. COMPUTE DAILY RETURNS (FOR TRADING IMPACT)
    # ========================================================================
    daily_returns = None
    if include_trading_impact:
        print("\n[4/5] Computing daily returns...")
        daily_returns = compute_daily_returns(prices)
        print(f"  ✓ Computed {len(daily_returns)} daily returns")
        print(f"    Mean: {np.mean(daily_returns):.6f}")
        print(f"    Std: {np.std(daily_returns):.6f}")

    # ========================================================================
    # 5. RUN EVALUATION
    # ========================================================================
    print("\n[5/5] Running comprehensive evaluation...")
    result = evaluate_regimes(
        y_true=y_true,
        y_pred=y_pred,
        daily_returns=daily_returns,
        transition_tolerance=3,
        lag_tolerance=10,
    )
    print("  ✓ Evaluation complete")

    # ========================================================================
    # PRINT REPORT
    # ========================================================================
    print_regime_evaluation(result)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save label sequences
        labels_df = pd.DataFrame({
            "date": dates,
            "y_true": y_true,
            "y_pred": y_pred,
            "match": y_true == y_pred,
        })
        labels_path = output_dir / "regime_labels.csv"
        labels_df.to_csv(labels_path, index=False)
        print(f"\n✓ Saved regime labels: {labels_path}")

        # Save evaluation metrics as JSON
        import json
        metrics_dict = {
            "classification": {
                "accuracy": float(result.classification.accuracy),
                "macro_f1": float(result.classification.macro_f1),
                "weighted_f1": float(result.classification.weighted_f1),
                "cohens_kappa": float(result.classification.cohens_kappa),
            },
            "transitions": {
                "n_true_transitions": int(result.transitions.n_true_transitions),
                "n_pred_transitions": int(result.transitions.n_pred_transitions),
                "n_matched_transitions": int(result.transitions.n_matched_transitions),
                "transition_f1": float(result.transitions.transition_f1),
            },
            "lag": {
                "mean_lag": float(result.lag.mean_lag),
                "median_lag": float(result.lag.median_lag),
                "std_lag": float(result.lag.std_lag),
            },
            "temporal_stability": {
                "transition_rate_ratio": float(result.temporal_stability.transition_rate_ratio),
                "flickering_score": float(result.temporal_stability.flickering_score),
            },
        }
        if result.trading_impact is not None:
            metrics_dict["trading_impact"] = {
                "regime_error_cost": float(result.trading_impact.regime_error_cost),
                "correct_sharpe": float(result.trading_impact.correct_sharpe),
                "incorrect_sharpe": float(result.trading_impact.incorrect_sharpe),
            }

        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"✓ Saved evaluation metrics: {metrics_path}")

        # Generate plots
        print(f"\nGenerating visualizations...")
        generate_evaluation_dashboard(result, output_dir=output_dir)

    return y_true, y_pred, result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run regime classification evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./regime_evaluation_results",
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--include_trading_impact",
        action="store_true",
        default=True,
        help="Include trading impact analysis",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2016-05-01",
        help="Start date for data",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2020-05-30",
        help="End date for data",
    )

    args = parser.parse_args()

    # Create config
    config = AppConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        enable_regime_awareness=True,
    )

    # Run evaluation
    try:
        y_true, y_pred, result = run_evaluation(
            config=config,
            output_dir=args.output_dir,
            include_trading_impact=args.include_trading_impact,
        )
        print("\n" + "=" * 80)
        print("✓ EVALUATION COMPLETE")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
