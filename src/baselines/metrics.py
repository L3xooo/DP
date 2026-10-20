"""
Unified evaluation metrics for portfolio baselines.

All functions take a portfolio value series V of shape (T,) as the primary
input.  This makes them strategy-agnostic: any baseline or agent that
produces a value series can be evaluated with the same code.

Conventions:
    - bars_per_day : number of intraday bars in one trading day (default 13
      for 30-min bars).  Used to convert bar-level returns to daily returns
      before annualisation.
    - trading_days_per_year : 252 (standard US equity assumption).
    - R_f : annualised risk-free rate (default 0.0).

Author: Teodora Simon
"""

import numpy as np
from typing import Dict, Optional


# ═════════════════════════════════════════════════════════════════════
#  Core helpers
# ═════════════════════════════════════════════════════════════════════

def _bar_log_returns(V: np.ndarray) -> np.ndarray:
    """Log returns between consecutive bars: ln(V[t+1] / V[t])."""
    return np.log(V[1:] / V[:-1])


def _bar_simple_returns(V: np.ndarray) -> np.ndarray:
    """Simple returns between consecutive bars: V[t+1]/V[t] - 1."""
    return V[1:] / V[:-1] - 1.0


# ═════════════════════════════════════════════════════════════════════
#  Drawdown & Volatility Metrics
# ═════════════════════════════════════════════════════════════════════

def cagr(
    V: np.ndarray,
    bars_per_day: int = 1,
    trading_days_per_year: int = 252,
) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (V_T / V_0) ^ (trading_days_per_year / T_days) - 1

    where T_days = (T - 1) / bars_per_day is the number of trading days
    spanned by the value series.
    """
    T = len(V)
    T_days = (T - 1) / bars_per_day
    if T_days <= 0:
        return 0.0
    return float((V[-1] / V[0]) ** (trading_days_per_year / T_days) - 1.0)


def maximum_drawdown(V: np.ndarray) -> float:
    """
    Maximum Drawdown (MDD).

    MDD = max_t [ (peak_t - V_t) / peak_t ]

    where peak_t = max_{tau <= t} V_tau.
    Returns a positive number (e.g. 0.266 means 26.6% drawdown).
    """
    running_max = np.maximum.accumulate(V)
    drawdowns = (running_max - V) / running_max
    return float(np.max(drawdowns))


def drawdown_series(V: np.ndarray) -> np.ndarray:
    """
    Full drawdown series (for plotting).

    Returns an array of shape (T,) where each element is the fractional
    drawdown from the running peak at that point.
    """
    running_max = np.maximum.accumulate(V)
    return (running_max - V) / running_max


def annualized_volatility(
    V: np.ndarray,
    bars_per_day: int = 1,
    trading_days_per_year: int = 252,
) -> float:
    """
    Annualized volatility of simple returns.

    sigma_ann = std(R_bar) * sqrt(bars_per_day * trading_days_per_year)

    where R_bar are bar-level simple returns.
    """
    R = _bar_simple_returns(V)
    bars_per_year = bars_per_day * trading_days_per_year
    return float(np.std(R, ddof=1) * np.sqrt(bars_per_year))


def downside_deviation(
    V: np.ndarray,
    R_f: float = 0.0,
    bars_per_day: int = 1,
    trading_days_per_year: int = 252,
) -> float:
    """
    Annualized downside deviation (used in Sortino ratio).

    Only considers returns below the risk-free rate (scaled to bar level).
    """
    R = _bar_simple_returns(V)
    bars_per_year = bars_per_day * trading_days_per_year
    rf_per_bar = (1.0 + R_f) ** (1.0 / bars_per_year) - 1.0
    diff = R - rf_per_bar
    downside = diff[diff < 0]
    if len(downside) == 0:
        return 0.0
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(bars_per_year))


# ═════════════════════════════════════════════════════════════════════
#  Risk-Adjusted Return Metrics
# ═════════════════════════════════════════════════════════════════════

def sharpe_ratio(
    V: np.ndarray,
    R_f: float = 0.0,
    bars_per_day: int = 1,
    trading_days_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe Ratio.

    Sharpe = (CAGR - R_f) / sigma_ann
    """
    ann_vol = annualized_volatility(V, bars_per_day, trading_days_per_year)
    if ann_vol < 1e-12:
        return 0.0
    ann_ret = cagr(V, bars_per_day, trading_days_per_year)
    return float((ann_ret - R_f) / ann_vol)


def sortino_ratio(
    V: np.ndarray,
    R_f: float = 0.0,
    bars_per_day: int = 1,
    trading_days_per_year: int = 252,
) -> float:
    """
    Annualized Sortino Ratio.

    Sortino = (CAGR - R_f) / sigma_D
    """
    dd = downside_deviation(V, R_f, bars_per_day, trading_days_per_year)
    if dd < 1e-12:
        return 0.0
    ann_ret = cagr(V, bars_per_day, trading_days_per_year)
    return float((ann_ret - R_f) / dd)


def calmar_ratio(
    V: np.ndarray,
    bars_per_day: int = 1,
    trading_days_per_year: int = 252,
) -> float:
    """
    Calmar Ratio.

    Calmar = CAGR / MDD
    """
    mdd = maximum_drawdown(V)
    if mdd < 1e-12:
        return 0.0
    ann_ret = cagr(V, bars_per_day, trading_days_per_year)
    return float(ann_ret / mdd)


# ═════════════════════════════════════════════════════════════════════
#  Trading Efficiency Metrics
# ═════════════════════════════════════════════════════════════════════

def average_turnover(tau: np.ndarray) -> float:
    """Average turnover per bar."""
    return float(np.mean(tau))


def total_fee_drag(costs: np.ndarray) -> float:
    """Total dollar transaction costs over the backtest."""
    return float(np.sum(costs))


def win_loss_ratio(V: np.ndarray) -> Dict[str, float]:
    """
    Win / Loss statistics based on bar-level simple returns.

    Returns dict with:
        win_rate   : fraction of bars with positive return
        loss_rate  : fraction of bars with negative return
        win_loss   : avg_win / avg_loss (absolute ratio)
        avg_win    : mean of positive returns
        avg_loss   : mean of negative returns (as positive number)
    """
    R = _bar_simple_returns(V)
    wins = R[R > 0]
    losses = R[R < 0]
    n = len(R)

    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0
    ratio = avg_win / avg_loss if avg_loss > 1e-12 else 0.0

    return dict(
        win_rate=float(len(wins) / n) if n > 0 else 0.0,
        loss_rate=float(len(losses) / n) if n > 0 else 0.0,
        win_loss=float(ratio),
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


# ═════════════════════════════════════════════════════════════════════
#  Unified report
# ═════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    V: np.ndarray,
    R_f: float = 0.0,
    bars_per_day: int = 1,
    trading_days_per_year: int = 252,
    tau: Optional[np.ndarray] = None,
    costs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all metrics from a portfolio value series.

    Parameters
    ----------
    V : ndarray, shape (T,)
        Portfolio value at each bar.
    R_f : float
        Annualised risk-free rate.
    bars_per_day : int
        Number of intraday bars per trading day.
    trading_days_per_year : int
        Trading days in a year (252 for US equities).
    tau : ndarray, shape (T-1,), optional
        Per-bar turnover.  If None, trading metrics are omitted.
    costs : ndarray, shape (T-1,), optional
        Per-bar dollar costs.  If None, fee drag is omitted.

    Returns
    -------
    Dict[str, float] with all metric values.
    """
    bpd = bars_per_day
    tdy = trading_days_per_year

    metrics = {
        "cagr": cagr(V, bpd, tdy),
        "annualized_volatility": annualized_volatility(V, bpd, tdy),
        "max_drawdown": maximum_drawdown(V),
        "sharpe_ratio": sharpe_ratio(V, R_f, bpd, tdy),
        "sortino_ratio": sortino_ratio(V, R_f, bpd, tdy),
        "calmar_ratio": calmar_ratio(V, bpd, tdy),
        "total_return_pct": (V[-1] / V[0] - 1.0) * 100,
        "total_log_return": float(np.log(V[-1] / V[0])),
    }

    wl = win_loss_ratio(V)
    metrics["win_rate"] = wl["win_rate"]
    metrics["loss_rate"] = wl["loss_rate"]
    metrics["win_loss_ratio"] = wl["win_loss"]
    metrics["avg_win"] = wl["avg_win"]
    metrics["avg_loss"] = wl["avg_loss"]

    if tau is not None:
        metrics["avg_turnover"] = average_turnover(tau)
        metrics["max_turnover"] = float(np.max(tau))

    if costs is not None:
        metrics["total_fee_drag"] = total_fee_drag(costs)

    return metrics


def print_metrics(
    metrics: Dict[str, float],
    label: str = "PORTFOLIO",
) -> None:
    """Pretty-print a metrics dict to the console."""
    print("=" * 65)
    print(f"  {label}")
    print("-" * 65)
    print(f"  CAGR                   : {metrics['cagr']:.4%}")
    print(f"  Annualized Volatility  : {metrics['annualized_volatility']:.4%}")
    print(f"  Max Drawdown           : {metrics['max_drawdown']:.4%}")
    print("-" * 65)
    print(f"  Sharpe Ratio           : {metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio          : {metrics['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio           : {metrics['calmar_ratio']:.4f}")
    print("-" * 65)
    print(f"  Total Return           : {metrics['total_return_pct']:.2f}%")
    print(f"  Total Log Return       : {metrics['total_log_return']:.6f}")
    print("-" * 65)
    print(f"  Win Rate               : {metrics['win_rate']:.4%}")
    print(f"  Loss Rate              : {metrics['loss_rate']:.4%}")
    print(f"  Win/Loss Ratio         : {metrics['win_loss_ratio']:.4f}")
    print(f"  Avg Win                : {metrics['avg_win']:.6f}")
    print(f"  Avg Loss               : {metrics['avg_loss']:.6f}")

    if "avg_turnover" in metrics:
        print("-" * 65)
        print(f"  Avg Turnover           : {metrics['avg_turnover']:.6f}")
        print(f"  Max Turnover           : {metrics['max_turnover']:.6f}")

    if "total_fee_drag" in metrics:
        print(f"  Total Fee Drag ($)     : {metrics['total_fee_drag']:.2f}")

    print("=" * 65)
