"""
HMM Regime Detection Prototype — Step 2 + Step 3

Loads close prices for 10 tickers, computes equally-weighted average
daily log-returns, fits a 2-state Gaussian HMM, and validates the results.

Based on: Bauman et al., "Deep Reinforcement Learning for Goal-Based
Investing Under Regime-Switching" (Section 3)
"""

import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# ============================================================
# Step 2: Prepare HMM Training Data
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "indicators")
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "XOM"]
TRAIN_START = "2016-05-01"
TRAIN_END = "2020-05-30"

# --- Load close prices for all tickers ---
close_frames = []
for ticker in TICKERS:
    path = os.path.join(DATA_DIR, ticker, "normalized.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df.loc[TRAIN_START:TRAIN_END, ["close"]]
    df.columns = [ticker]
    close_frames.append(df)

close_prices = pd.concat(close_frames, axis=1, join="inner")
print(f"Close prices shape: {close_prices.shape}")
print(f"Date range: {close_prices.index[0]} to {close_prices.index[-1]}")

# --- Compute 30-min log-returns ---
log_returns_30min = np.log(close_prices / close_prices.shift(1))
log_returns_30min = log_returns_30min.dropna()

# --- Aggregate to daily returns for HMM fitting ---
# Sum of log-returns within a day = log of cumulative daily return.
# The HMM needs daily-scale data to detect multi-day regime shifts,
# not bar-level volatility spikes (see paper: monthly frequency).
daily_returns = log_returns_30min.resample("D").sum()
daily_returns = daily_returns.loc[(daily_returns != 0).any(axis=1)]  # drop non-trading days

# --- Equally-weighted average across tickers ---
avg_daily_returns = daily_returns.mean(axis=1)

# --- Reshape for hmmlearn (n_samples, n_features) ---
returns_train = avg_daily_returns.values.reshape(-1, 1)
print(f"returns_train shape: {returns_train.shape}")
print(f"Mean daily return: {np.mean(returns_train):.8f}")
print(f"Std daily return:  {np.std(returns_train):.8f}")

# ============================================================
# Step 3: Fit the 2-State Gaussian HMM
# ============================================================

model = GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=10000,
    random_state=42,
    tol=1e-4,
)

model.fit(returns_train)

# --- Inspect learned parameters ---
print("\n=== HMM Parameters ===")
print(f"Converged: {model.monitor_.converged}")
print(f"Iterations used: {model.monitor_.iter}")

for i in range(2):
    mean = model.means_[i, 0]
    var = model.covars_[i, 0, 0]
    std = np.sqrt(var)
    print(f"\nRegime {i}:")
    print(f"  Mean:  {mean:.8f}")
    print(f"  Std:   {std:.8f}")

print(f"\nTransition matrix:\n{model.transmat_}")
print(f"\nStart probabilities: {model.startprob_}")

# ============================================================
# Validation
# ============================================================

# --- Identify bull vs bear ---
if model.means_[0, 0] > model.means_[1, 0]:
    bull_regime, bear_regime = 0, 1
else:
    bull_regime, bear_regime = 1, 0

bull_std = np.sqrt(model.covars_[bull_regime, 0, 0])
bear_std = np.sqrt(model.covars_[bear_regime, 0, 0])

print(f"\nBull regime: {bull_regime} (mean={model.means_[bull_regime, 0]:.8f}, std={bull_std:.8f})")
print(f"Bear regime: {bear_regime} (mean={model.means_[bear_regime, 0]:.8f}, std={bear_std:.8f})")

# --- Assertions ---
assert bear_std > bull_std, "Bear regime should have higher volatility than bull"
print("PASS: Bear volatility > Bull volatility")

assert model.transmat_[0, 0] > 0.5, "Regimes should be persistent"
assert model.transmat_[1, 1] > 0.5, "Regimes should be persistent"
print(f"P(stay in bull): {model.transmat_[bull_regime, bull_regime]:.4f}")
print(f"P(stay in bear): {model.transmat_[bear_regime, bear_regime]:.4f}")
print("PASS: Regimes are persistent")

# --- Decode regimes and plot ---
hidden_states = model.predict(returns_train)
regime_probs = model.predict_proba(returns_train)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

colors = ['green' if s == bull_regime else 'red' for s in hidden_states]
axes[0].scatter(range(len(returns_train)), returns_train.flatten(),
                c=colors, s=0.5, alpha=0.5)
axes[0].set_title("Returns Colored by HMM Regime (Green=Bull, Red=Bear)")
axes[0].set_ylabel("Log Return")

axes[1].plot(regime_probs[:, bull_regime], color='green', linewidth=0.5)
axes[1].set_title("P(Bull Regime) Over Time")
axes[1].set_ylabel("Probability")
axes[1].set_ylim(0, 1)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("hmm_step3_validation.png", dpi=150)
plt.show()
print("Validation plot saved to hmm_step3_validation.png")