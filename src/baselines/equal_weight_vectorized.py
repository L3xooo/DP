"""
Baseline 1B: Vectorized Equal-Weight Daily Rebalanced (1/N).

Pure NumPy/Pandas implementation — no PortfolioEnv loop.
Computes portfolio value, turnover, and transaction costs in a single
vectorized pass over the price matrix.

Runs both gross (c=0) and net (c=0.001) variants and reports both.

Usage:
    python baselines/equal_weight_vectorized.py

Author: Teodora Simon
"""

import sys
import os
import json

# Resolve src/ relative to this file so the script works from any CWD
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from td3.config.app_config import AppConfig
from td3.data.data_processor import DataProcessor
from td3.utils.file_utils import create_experiment_directories, RunType
from td3.utils.graph_utils import plot_price_history
from td3.utils.logs.logger import LoggerFactory
from metrics import compute_all_metrics, print_metrics

logger = LoggerFactory.create_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  Vectorized core
# ═════════════════════════════════════════════════════════════════════

def run_equal_weight_baseline(
    P: np.ndarray,
    V_init: float = 10_000.0,
    c: float = 0.001,
) -> dict:
    """
    Vectorized 1/N daily-rebalanced portfolio.

    Parameters
    ----------
    P : ndarray, shape (T, N)
        Daily close prices.  P[t, i] = price of asset i on day t.
    V_init : float
        Initial portfolio value.
    c : float
        Proportional transaction-cost coefficient (one-way).

    Returns
    -------
    dict with keys
        V          (T,)     net portfolio value series
        r_ew       (T-1,)   gross daily EW return ratios
        tau        (T-1,)   daily turnover (L1 norm of weight drift)
        costs      (T-1,)   dollar transaction costs per day
        W_drift    (T-1, N) pre-rebalance drifted weights
        log_ret    (T-1,)   daily log returns (net of costs)
    """
    T, N = P.shape
    assert np.all(P > 0), "All prices must be strictly positive"
    w_star = 1.0 / N

    # ── Step 1-4: fully vectorized ───────────────────────────────────
    R = P[1:] / P[:-1]                              # daily price return ratios (T-1, N)
    r_ew = R.mean(axis=1)                            # (T-1,)
    R_sum = R.sum(axis=1, keepdims=True)             # (T-1, 1)
    W_drift = R / R_sum                              # (T-1, N)
    tau = np.abs(w_star - W_drift).sum(axis=1)       # (T-1,)

    # ── Step 5: net portfolio value via cumprod ──────────────────────
    net_factors = r_ew * (1.0 - c * tau)             # (T-1,)
    V = np.empty(T)
    V[0] = V_init
    V[1:] = V_init * np.cumprod(net_factors)

    # ── Derived series for reporting ─────────────────────────────────
    V_pre = np.empty(T - 1)
    V_pre[0] = V[0] * r_ew[0]      # day 0: start value * gross return
    V_pre[1:] = V[1:-1] * r_ew[1:] # day t: yesterday's net value * today's gross return
    costs = c * tau * V_pre

    log_ret = np.log(net_factors)

    return dict(
        V=V, r_ew=r_ew, tau=tau, costs=costs,
        W_drift=W_drift, log_ret=log_ret,
    )


def main() -> None:

    TRANSACTION_COST = 0.001  

    app_config = AppConfig(
        enable_regime_awareness=False,
        enable_hmm_regime=True,
        start_date="2017-05-01",
        end_date="2021-05-30",
    )

    # ── Load data ────────────────────────────────────────────────────
    dp = DataProcessor(
        data_dir=app_config.data_dir,
        parquet_dir=None,
    )
    data_3d_features, data_3d_prices, tickers, features, all_dates, hmm_regime_probs = (
        dp.load_data(app_config, add_regime=app_config.enable_regime_awareness)
    )

    P = data_3d_prices.squeeze(-1).astype(np.float64)  # (T, N)
    T, N = P.shape

    logger.info("Tickers (%d): %s", N, tickers)
    logger.info("Price matrix shape: (%d, %d)", T, N)
    logger.info("Date range: %s  to  %s", all_dates[0], all_dates[-1])

    # ── Output directories ───────────────────────────────────────────
    experiment_dir, plots_dir, models_dir, weights_dir, rewards_dir = (
        create_experiment_directories(run_type=RunType.TEST)
    )
    app_config.to_json(experiment_dir)
    plot_price_history(
        data_3d_prices, tickers, all_dates,
        save_path=os.path.join(plots_dir, "price_history.png"),
    )

    # ── Run vectorized baseline (gross and net) ──────────────────────
    res_net   = run_equal_weight_baseline(P, V_init=10_000.0, c=TRANSACTION_COST)
    res_gross = run_equal_weight_baseline(P, V_init=10_000.0, c=0.0)

    V_net   = res_net["V"]
    V_gross = res_gross["V"]

    # ── Export drifted weights CSV ───────────────────────────────────
    csv_path = os.path.join(weights_dir, "drifted_weights_ew_vectorized.csv")
    header = "Date," + ",".join(tickers) + ",Turnover\n"
    with open(csv_path, "w") as f:
        f.write(header)
        W_drift = res_net["W_drift"]  # (T-1, N)
        tau = res_net["tau"]
        for t in range(W_drift.shape[0]):
            w_str = ",".join(f"{w:.6f}" for w in W_drift[t])
            f.write(f"{all_dates[t + 1]},{w_str},{tau[t]:.6f}\n")

    # ── Chart 1: Portfolio value — gross vs net ──────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(all_dates, V_gross, color="steelblue", linewidth=1.5,
            label="Gross (c=0)")
    ax.plot(all_dates, V_net, color="darkorange", linewidth=1.5,
            label=f"Net (c={TRANSACTION_COST})")
    ax.axhline(10_000, color="gray", linestyle="--", alpha=0.5, label="Start (10 000)")
    ax.set_title("1/N Vectorized Baseline — Portfolio Value", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "baseline_portfolio_value.png"), dpi=150)
    plt.close(fig)

    # ── Chart 2: Cumulative log return ───────────────────────────────
    cum_log_net   = np.cumsum(res_net["log_ret"])
    cum_log_gross = np.cumsum(res_gross["log_ret"])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cum_log_gross, color="steelblue", linewidth=1.5, label="Gross")
    ax.plot(cum_log_net, color="darkorange", linewidth=1.5, label="Net")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("1/N Vectorized Baseline — Cumulative Log Return", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Log Return")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "baseline_cumulative_reward.png"), dpi=150)
    plt.close(fig)

    # ── Chart 3: Daily turnover ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(res_net["tau"])), res_net["tau"], width=1.0,
           color="slategray", alpha=0.6)
    ax.set_title("Daily Turnover", fontsize=13, fontweight="bold")
    ax.set_ylabel("Turnover (L1)")
    ax.set_xlabel("Day")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "baseline_daily_turnover.png"), dpi=150)
    plt.close(fig)

    # ── Sanity check ────────────────────────────────────────────────
    total_log_net   = np.log(V_net[-1] / V_net[0])
    total_log_gross = np.log(V_gross[-1] / V_gross[0])
    sanity_diff_net   = abs(cum_log_net[-1] - total_log_net)
    sanity_diff_gross = abs(cum_log_gross[-1] - total_log_gross)

    if sanity_diff_net > 1e-8:
        logger.warning("SANITY CHECK FAILED (net) — investigate numerical drift.")
    else:
        logger.info("SANITY CHECK PASSED (net)  |diff| = %.2e", sanity_diff_net)
    if sanity_diff_gross > 1e-8:
        logger.warning("SANITY CHECK FAILED (gross) — investigate numerical drift.")
    else:
        logger.info("SANITY CHECK PASSED (gross) |diff| = %.2e", sanity_diff_gross)

    # ── Unified metrics ──────────────────────────────────────────────
    BARS_PER_DAY = 1

    metrics_gross = compute_all_metrics(
        V_gross, bars_per_day=BARS_PER_DAY,
        tau=res_gross["tau"], costs=res_gross["costs"],
    )
    metrics_net = compute_all_metrics(
        V_net, bars_per_day=BARS_PER_DAY,
        tau=res_net["tau"], costs=res_net["costs"],
    )

    print_metrics(metrics_gross, label="1/N EQUAL-WEIGHT BASELINE — GROSS (c=0)")
    print_metrics(metrics_net, label=f"1/N EQUAL-WEIGHT BASELINE — NET (c={TRANSACTION_COST})")

    # ── Export metrics to JSON ───────────────────────────────────────
    metrics_json = {
        "gross": {k: float(v) for k, v in metrics_gross.items()},
        "net":   {k: float(v) for k, v in metrics_net.items()},
    }
    json_path = os.path.join(experiment_dir, "ew_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info("Metrics JSON: %s", json_path)

    logger.info("Weights CSV : %s", csv_path)
    logger.info("Plots dir   : %s", plots_dir)


if __name__ == "__main__":
    main()
