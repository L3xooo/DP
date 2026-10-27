"""
Compute Sharpe, Sortino, Calmar from weight CSVs + daily price files.

Run on the remote where both price files and weight CSVs are available.
"""
import csv
import numpy as np
import os
import glob

# ── CONFIGURE THESE PATHS ──────────────────────────────────────────
PRICE_DIR = "/home/xsimont/DP/indicators/indicators"  # folder containing AAPL/, AMZN/, ... each with normalized.csv
RESULTS_BASE = "/data/teodora_portfolio/test"

SCENARIOS = {
    "baseline":    os.path.join(RESULTS_BASE, "baseline_2026-10-16", "weights"),
    "regime_only": os.path.join(RESULTS_BASE, "regime_2026-10-16",   "weights"),
    "hmm_only":    os.path.join(RESULTS_BASE, "hmm_2026-10-16",     "weights"),
    "both":        os.path.join(RESULTS_BASE, "both_2026-10-16",     "weights"),
}
# ────────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "TSLA", "XOM"]
RISK_FREE_RATE = 0.0


def load_prices(ticker):
    """Load daily close prices from indicators/{ticker}/normalized.csv"""
    path = os.path.join(PRICE_DIR, ticker, "normalized.csv")
    prices = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            prices[row["date"]] = float(row["close"])
    return prices


def load_weights(path):
    """Load weight CSV, return (dates, weight_matrix)."""
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    dates = [r["Date"] for r in rows]
    W = np.array([[float(r["Cash"])] + [float(r[t]) for t in TICKERS] for r in rows])
    return dates, W


def compute_metrics(daily_returns):
    """Compute Sharpe, Sortino, Calmar from daily return series."""
    T = len(daily_returns)
    num_years = T / 252.0

    V = np.cumprod(1 + daily_returns)
    total_return = V[-1] - 1.0
    cagr = (1 + total_return) ** (1 / num_years) - 1

    ann_vol = np.std(daily_returns, ddof=1) * np.sqrt(252)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

    downside = daily_returns[daily_returns < 0]
    dd_std = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 1 else 1e-12
    sortino = (cagr - RISK_FREE_RATE) / dd_std

    running_max = np.maximum.accumulate(V)
    drawdowns = (V - running_max) / running_max
    mdd = abs(np.min(drawdowns))
    calmar = cagr / mdd if mdd > 0 else 0.0

    return {
        "CAGR": cagr, "Ann. Vol": ann_vol, "Sharpe": sharpe,
        "Sortino": sortino, "MDD": mdd, "Calmar": calmar,
    }


# ── Load prices ─────────────────────────────────────────────────────
all_prices = {t: load_prices(t) for t in TICKERS}

# ── Process each scenario, each weight file ─────────────────────────
# CSV header
print("Scenario,Model,CAGR,Ann_Vol,Sharpe,Sortino,MDD,Calmar")

for scenario_name, weights_dir in SCENARIOS.items():
    weight_files = sorted(glob.glob(os.path.join(weights_dir, "*.csv")))

    if not weight_files:
        continue

    all_sharpes = []
    all_cagrs = []
    all_sortinos = []
    all_calmars = []
    all_vols = []
    all_mdds = []

    for wf in weight_files:
        model_label = os.path.basename(wf).replace("weights_", "").replace(".csv", "")
        dates, W = load_weights(wf)

        daily_returns = []
        for i in range(1, len(dates)):
            d_prev, d_curr = dates[i - 1], dates[i]
            asset_rets = []
            for t in TICKERS:
                p0 = all_prices[t].get(d_prev)
                p1 = all_prices[t].get(d_curr)
                asset_rets.append((p1 / p0 - 1) if (p0 and p1 and p0 > 0) else 0.0)

            port_ret = W[i - 1, 0] * 0.0 + np.dot(W[i - 1, 1:], asset_rets)
            daily_returns.append(port_ret)

        daily_returns = np.array(daily_returns)
        m = compute_metrics(daily_returns)

        all_cagrs.append(m["CAGR"])
        all_vols.append(m["Ann. Vol"])
        all_sharpes.append(m["Sharpe"])
        all_sortinos.append(m["Sortino"])
        all_mdds.append(m["MDD"])
        all_calmars.append(m["Calmar"])

        print(f"{scenario_name},{model_label},{m['CAGR']:.6f},{m['Ann. Vol']:.6f},{m['Sharpe']:.6f},{m['Sortino']:.6f},{m['MDD']:.6f},{m['Calmar']:.6f}")

    # Summary rows
    print(f"{scenario_name},MEDIAN,{np.median(all_cagrs):.6f},{np.median(all_vols):.6f},{np.median(all_sharpes):.6f},{np.median(all_sortinos):.6f},{np.median(all_mdds):.6f},{np.median(all_calmars):.6f}")
    print(f"{scenario_name},Q1_25,{np.percentile(all_cagrs,25):.6f},{np.percentile(all_vols,25):.6f},{np.percentile(all_sharpes,25):.6f},{np.percentile(all_sortinos,25):.6f},{np.percentile(all_mdds,25):.6f},{np.percentile(all_calmars,25):.6f}")
    print(f"{scenario_name},Q3_75,{np.percentile(all_cagrs,75):.6f},{np.percentile(all_vols,75):.6f},{np.percentile(all_sharpes,75):.6f},{np.percentile(all_sortinos,75):.6f},{np.percentile(all_mdds,75):.6f},{np.percentile(all_calmars,75):.6f}")
