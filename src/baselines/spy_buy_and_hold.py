"""
Baseline 2: S&P 500 Buy-and-Hold (SPY).

100% allocation to SPY, zero rebalancing, zero turnover.
Portfolio value is simply V_t = V_0 * P_SPY[t] / P_SPY[0].

Downloads SPY daily data from Alpaca Markets API, caches it
locally as a CSV file, and computes the buy-and-hold baseline.

Usage:
    python baselines/spy_buy_and_hold.py

Author: Teodora Simon
"""

import sys
import os

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from td3.utils.file_utils import create_experiment_directories, RunType
from td3.utils.logs.logger import LoggerFactory
from metrics import compute_all_metrics, print_metrics, drawdown_series

logger = LoggerFactory.create_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  Configuration
# ═════════════════════════════════════════════════════════════════════

# Alpaca API credentials — set via environment variables or fill in directly
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "PKF3UEGWJR3PEEBCCNHEGM55EG")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "9mGYz5K7yCdvoRYNZuLJbHcNLABFG5xuhCAzXd672ptY")

# Local cache path (downloaded once, reused on subsequent runs)
SPY_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "spy_daily.csv"
)

START_DATE = "2021-06-01"
END_DATE   = "2024-12-30"
V_INIT     = 10_000.0
BARS_PER_DAY = 1


# ═════════════════════════════════════════════════════════════════════
#  Data download & loading
# ═════════════════════════════════════════════════════════════════════

def download_spy_from_alpaca(
    api_key: str,
    secret_key: str,
    start_date: str,
    end_date: str,
    cache_path: str,
) -> pd.DataFrame:
    """
    Download SPY daily bars from Alpaca and cache as CSV.

    Parameters
    ----------
    api_key : str
        Alpaca API key.
    secret_key : str
        Alpaca secret key.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    cache_path : str
        Path to save/load the CSV cache.

    Returns
    -------
    pd.DataFrame with columns: date, close (at minimum).
    """


    logger.info("Downloading SPY daily bars from Alpaca (%s to %s)...",
                start_date, end_date)

    client = StockHistoricalDataClient(api_key, secret_key)

    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start_date, tz="America/New_York"),
        end=pd.Timestamp(end_date, tz="America/New_York"),
    )

    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    # Normalise columns to match project convention
    df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    # Cache locally
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Cached %d daily bars to %s", len(df), cache_path)

    return df


def load_spy_prices(
    start_date: str,
    end_date: str,
    api_key: str = "",
    secret_key: str = "",
    cache_path: str = "",
) -> tuple[np.ndarray, list[str]]:
    """
    Load SPY close prices — from cache if available, otherwise download.

    Returns
    -------
    P : ndarray, shape (T,)
        Close prices.
    dates : list[str]
        Date strings for each bar.
    """
    # Try loading from cache first
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading SPY data from cache: %s", cache_path)
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
    else:
        if not api_key or not secret_key:
            raise ValueError(
                "No cached SPY data found and Alpaca credentials not set.\n"
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables,\n"
                "or place a pre-downloaded parquet at: " + cache_path
            )
        df = download_spy_from_alpaca(
            api_key, secret_key, start_date, end_date, cache_path
        )

    df = df.sort_values("date")
    mask = (df["date"] >= pd.to_datetime(start_date)) & (
            df["date"] <= pd.to_datetime(end_date))
    df = df.loc[mask].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"No SPY data found between {start_date} and {end_date}"
        )

    P = df["close"].values.astype(np.float64)
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

    return P, dates


def run_spy_buy_and_hold(
    P: np.ndarray,
    V_init: float = 10_000.0,
) -> dict:
    """
    Buy-and-hold on a single asset.

    V_t = V_0 * P[t] / P[0]

    Parameters
    ----------
    P : ndarray, shape (T,)
        Close prices of SPY.
    V_init : float
        Initial portfolio value.

    Returns
    -------
    dict with keys
        V        (T,)   portfolio value series
        log_ret  (T-1,) bar-level log returns
    """
    assert np.all(P > 0), "All prices must be strictly positive"

    V = V_init * P / P[0]
    log_ret = np.log(P[1:] / P[:-1])

    return dict(V=V, log_ret=log_ret)


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load SPY prices ──────────────────────────────────────────────
    P_spy, all_dates = load_spy_prices(
        start_date=START_DATE,
        end_date=END_DATE,
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        cache_path=SPY_CACHE_PATH,
    )
    T = len(P_spy)

    logger.info("SPY daily bars: %d trading days", T)
    logger.info("Date range: %s  to  %s", all_dates[0], all_dates[-1])

    # ── Run buy-and-hold ─────────────────────────────────────────────
    res = run_spy_buy_and_hold(P_spy, V_init=V_INIT)
    V = res["V"]

    # ── Output directories ───────────────────────────────────────────
    experiment_dir, plots_dir, models_dir, weights_dir, rewards_dir = (
        create_experiment_directories(run_type=RunType.TEST)
    )

    # ── Sanity check ─────────────────────────────────────────────────
    cum_log = np.cumsum(res["log_ret"])
    total_log = np.log(V[-1] / V[0])
    sanity_diff = abs(cum_log[-1] - total_log)

    if sanity_diff > 1e-8:
        logger.warning("SANITY CHECK FAILED — |diff| = %.2e", sanity_diff)
    else:
        logger.info("SANITY CHECK PASSED  |diff| = %.2e", sanity_diff)

    # ── Chart 1: Portfolio value ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(all_dates, V, color="steelblue", linewidth=1.5, label="SPY B&H")
    ax.axhline(V_INIT, color="gray", linestyle="--", alpha=0.5,
               label=f"Start ({V_INIT:,.0f})")
    ax.set_title("SPY Buy & Hold — Portfolio Value", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "spy_portfolio_value.png"), dpi=150)
    plt.close(fig)

    # ── Chart 2: Cumulative log return ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cum_log, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("SPY Buy & Hold — Cumulative Log Return", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Log Return")
    ax.set_xlabel("Step")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "spy_cumulative_reward.png"), dpi=150)
    plt.close(fig)

    # ── Chart 3: Drawdown ────────────────────────────────────────────
    dd = drawdown_series(V)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(range(len(dd)), 0, -dd * 100, color="crimson", alpha=0.4)
    ax.set_title("SPY Buy & Hold — Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Bar")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "spy_drawdown.png"), dpi=150)
    plt.close(fig)

    # ── Metrics ──────────────────────────────────────────────────────
    metrics = compute_all_metrics(V, bars_per_day=BARS_PER_DAY)
    print_metrics(metrics, label="SPY BUY & HOLD BASELINE")

    logger.info("Plots dir : %s", plots_dir)


if __name__ == "__main__":
    main()
