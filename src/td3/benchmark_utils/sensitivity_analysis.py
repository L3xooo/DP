"""
Sensitivity analysis for regime classification parameters.

Tests how evaluation metrics change with different threshold and window settings.

Usage:
    python sensitivity_analysis.py [--output_dir ./sensitivity_results]

Author: Devin (Evaluation Framework)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse
import sys

from td3.config.app_config import AppConfig
from td3.evaluation.run_regime_evaluation import run_evaluation


@dataclass
class SensitivityResult:
    """Result of a single sensitivity test."""
    param_name: str
    param_value: str
    accuracy: float
    macro_f1: float
    transition_f1: float
    mean_lag: float
    transition_rate_ratio: float
    flickering_score: float


def sensitivity_analysis_thresholds(
    base_config: AppConfig,
    low_thresholds: Optional[List[float]] = None,
    high_thresholds: Optional[List[float]] = None,
    output_dir: str = "./sensitivity_results",
) -> pd.DataFrame:
    """
    Test sensitivity to volatility thresholds.

    Args:
        base_config: Base AppConfig
        low_thresholds: List of low threshold values to test (default: 20, 25, 30, 35, 40)
        high_thresholds: List of high threshold values to test (default: 60, 65, 70, 75, 80)
        output_dir: Directory to save results

    Returns:
        DataFrame with sensitivity results
    """
    if low_thresholds is None:
        low_thresholds = [20, 25, 30, 35, 40]
    if high_thresholds is None:
        high_thresholds = [60, 65, 70, 75, 80]

    results = []
    total = len(low_thresholds) * len(high_thresholds)
    count = 0

    for low_thresh in low_thresholds:
        for high_thresh in high_thresholds:
            if low_thresh >= high_thresh:
                continue

            count += 1
            print(f"\n[{count}/{total}] Testing low_threshold={low_thresh}, high_threshold={high_thresh}...")

            config = AppConfig(
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                regime_low_threshold=float(low_thresh),
                regime_high_threshold=float(high_thresh),
                regime_volatility_window=base_config.regime_volatility_window,
                regime_norm_window=base_config.regime_norm_window,
                enable_regime_awareness=True,
                data_dir=base_config.data_dir,
                parquet_dir=base_config.parquet_dir,
                ticker_config=base_config.ticker_config,
            )

            try:
                y_true, y_pred, result = run_evaluation(
                    config=config,
                    output_dir=None,  # Don't save individual results
                    include_trading_impact=False,
                )

                results.append({
                    "low_threshold": float(low_thresh),
                    "high_threshold": float(high_thresh),
                    "accuracy": result.classification.accuracy,
                    "macro_f1": result.classification.macro_f1,
                    "transition_f1": result.transitions.transition_f1,
                    "mean_lag": result.lag.mean_lag,
                    "transition_rate_ratio": result.temporal_stability.transition_rate_ratio,
                    "flickering_score": result.temporal_stability.flickering_score,
                })
                print(f"  ✓ Accuracy: {result.classification.accuracy:.4f}, F1: {result.classification.macro_f1:.4f}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    df = pd.DataFrame(results)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sensitivity_thresholds.csv", index=False)
    print(f"\n✓ Saved sensitivity results: {output_dir / 'sensitivity_thresholds.csv'}")

    return df


def sensitivity_analysis_windows(
    base_config: AppConfig,
    vol_windows: Optional[List[int]] = None,
    norm_windows: Optional[List[int]] = None,
    output_dir: str = "./sensitivity_results",
) -> pd.DataFrame:
    """
    Test sensitivity to volatility and normalization windows.

    Args:
        base_config: Base AppConfig
        vol_windows: List of volatility window values to test (default: 10, 15, 20, 25, 30)
        norm_windows: List of normalization window values to test (default: 126, 189, 252, 315)
        output_dir: Directory to save results

    Returns:
        DataFrame with sensitivity results
    """
    if vol_windows is None:
        vol_windows = [10, 15, 20, 25, 30]
    if norm_windows is None:
        norm_windows = [126, 189, 252, 315]

    results = []
    total = len(vol_windows) * len(norm_windows)
    count = 0

    for vol_window in vol_windows:
        for norm_window in norm_windows:
            count += 1
            print(f"\n[{count}/{total}] Testing vol_window={vol_window}, norm_window={norm_window}...")

            config = AppConfig(
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                regime_low_threshold=base_config.regime_low_threshold,
                regime_high_threshold=base_config.regime_high_threshold,
                regime_volatility_window=int(vol_window),
                regime_norm_window=int(norm_window),
                enable_regime_awareness=True,
                data_dir=base_config.data_dir,
                parquet_dir=base_config.parquet_dir,
                ticker_config=base_config.ticker_config,
            )

            try:
                y_true, y_pred, result = run_evaluation(
                    config=config,
                    output_dir=None,
                    include_trading_impact=False,
                )

                results.append({
                    "vol_window": int(vol_window),
                    "norm_window": int(norm_window),
                    "accuracy": result.classification.accuracy,
                    "macro_f1": result.classification.macro_f1,
                    "transition_f1": result.transitions.transition_f1,
                    "mean_lag": result.lag.mean_lag,
                    "transition_rate_ratio": result.temporal_stability.transition_rate_ratio,
                    "flickering_score": result.temporal_stability.flickering_score,
                })
                print(f"  ✓ Accuracy: {result.classification.accuracy:.4f}, F1: {result.classification.macro_f1:.4f}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    df = pd.DataFrame(results)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sensitivity_windows.csv", index=False)
    print(f"\n✓ Saved sensitivity results: {output_dir / 'sensitivity_windows.csv'}")

    return df


def sensitivity_analysis_hysteresis(
    base_config: AppConfig,
    buffer_values: Optional[List[float]] = None,
    output_dir: str = "./sensitivity_results",
) -> pd.DataFrame:
    """
    Test sensitivity to hysteresis buffer size.

    Note: Hysteresis buffer is hardcoded in _precompute_regimes() as 5.0.
    This function documents the approach for testing different buffer values.

    Args:
        base_config: Base AppConfig
        buffer_values: List of buffer values to test (default: 2, 3, 5, 7, 10)
        output_dir: Directory to save results

    Returns:
        DataFrame with sensitivity results
    """
    print("\n⚠️  Note: Hysteresis buffer is currently hardcoded in portfolio.py")
    print("   To test different buffer values, modify _precompute_regimes() buffer parameter.")
    print("   This function documents the expected approach.\n")

    if buffer_values is None:
        buffer_values = [2, 3, 5, 7, 10]

    results = []

    for buffer in buffer_values:
        print(f"\nTo test buffer={buffer}:")
        print(f"  1. Edit src/td3/environment/portfolio.py::_precompute_regimes()")
        print(f"  2. Change: buffer = {buffer}  (currently 5.0)")
        print(f"  3. Run: python -m td3.evaluation.run_regime_evaluation")
        print(f"  4. Record accuracy, F1, flickering_score")

    print("\nExample results (to be filled in manually):")
    df = pd.DataFrame({
        "buffer": buffer_values,
        "accuracy": [np.nan] * len(buffer_values),
        "macro_f1": [np.nan] * len(buffer_values),
        "flickering_score": [np.nan] * len(buffer_values),
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sensitivity_hysteresis_template.csv", index=False)
    print(f"\n✓ Saved template: {output_dir / 'sensitivity_hysteresis_template.csv'}")

    return df


def print_sensitivity_summary(
    df_thresholds: Optional[pd.DataFrame] = None,
    df_windows: Optional[pd.DataFrame] = None,
) -> None:
    """Print summary of sensitivity analysis results."""
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)

    if df_thresholds is not None and len(df_thresholds) > 0:
        print("\n[1] THRESHOLD SENSITIVITY")
        print("-" * 80)
        best_idx = df_thresholds["macro_f1"].idxmax()
        best = df_thresholds.loc[best_idx]
        print(f"Best configuration (by Macro F1):")
        print(f"  Low Threshold: {best['low_threshold']:.1f}")
        print(f"  High Threshold: {best['high_threshold']:.1f}")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  Macro F1: {best['macro_f1']:.4f}")
        print(f"  Transition F1: {best['transition_f1']:.4f}")
        print(f"  Mean Lag: {best['mean_lag']:.2f} steps")
        print(f"  Flickering Score: {best['flickering_score']:.4f}")

        # Find configuration with lowest flickering
        lowest_flicker_idx = df_thresholds["flickering_score"].idxmin()
        lowest_flicker = df_thresholds.loc[lowest_flicker_idx]
        print(f"\nLowest flickering configuration:")
        print(f"  Low Threshold: {lowest_flicker['low_threshold']:.1f}")
        print(f"  High Threshold: {lowest_flicker['high_threshold']:.1f}")
        print(f"  Flickering Score: {lowest_flicker['flickering_score']:.4f}")
        print(f"  Accuracy: {lowest_flicker['accuracy']:.4f}")

    if df_windows is not None and len(df_windows) > 0:
        print("\n[2] WINDOW SENSITIVITY")
        print("-" * 80)
        best_idx = df_windows["macro_f1"].idxmax()
        best = df_windows.loc[best_idx]
        print(f"Best configuration (by Macro F1):")
        print(f"  Vol Window: {best['vol_window']:.0f}")
        print(f"  Norm Window: {best['norm_window']:.0f}")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  Macro F1: {best['macro_f1']:.4f}")
        print(f"  Mean Lag: {best['mean_lag']:.2f} steps")
        print(f"  Flickering Score: {best['flickering_score']:.4f}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for regime classification"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sensitivity_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--test_thresholds",
        action="store_true",
        default=True,
        help="Test threshold sensitivity",
    )
    parser.add_argument(
        "--test_windows",
        action="store_true",
        default=True,
        help="Test window sensitivity",
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

    # Create base config
    base_config = AppConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        enable_regime_awareness=True,
    )

    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)

    df_thresholds = None
    df_windows = None

    # Test threshold sensitivity
    if args.test_thresholds:
        print("\n[1/2] Testing threshold sensitivity...")
        df_thresholds = sensitivity_analysis_thresholds(
            base_config,
            low_thresholds=[20, 25, 30, 35, 40],
            high_thresholds=[60, 65, 70, 75, 80],
            output_dir=args.output_dir,
        )
        print("\nThreshold Sensitivity Results (top 5 by F1):")
        print(df_thresholds.nlargest(5, "macro_f1")[
            ["low_threshold", "high_threshold", "accuracy", "macro_f1", "flickering_score"]
        ].to_string(index=False))

    # Test window sensitivity
    if args.test_windows:
        print("\n[2/2] Testing window sensitivity...")
        df_windows = sensitivity_analysis_windows(
            base_config,
            vol_windows=[10, 15, 20, 25, 30],
            norm_windows=[126, 189, 252, 315],
            output_dir=args.output_dir,
        )
        print("\nWindow Sensitivity Results (top 5 by F1):")
        print(df_windows.nlargest(5, "macro_f1")[
            ["vol_window", "norm_window", "accuracy", "macro_f1", "flickering_score"]
        ].to_string(index=False))

    # Print summary
    print_sensitivity_summary(df_thresholds, df_windows)

    print("\n✓ SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
