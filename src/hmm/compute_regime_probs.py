"""
Step 4: Compute regime probabilities online for daily data.

Based on: Bauman et al., Section 4.1 — state feature ĥ_{t-1}
"""

import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# ============================================================
# Step 2+3: Prepare data and fit HMM (reused from fit_hmm.py)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "indicators")
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "XOM"]
TRAIN_START = "2016-05-01"
TRAIN_END = "2020-05-30"

# Load daily close prices
close_frames = []
for ticker in TICKERS:
    path = os.path.join(DATA_DIR, ticker, "normalized.csv")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df.loc[TRAIN_START:TRAIN_END, ["close"]]
    df.columns = [ticker]
    close_frames.append(df)

close_prices = pd.concat(close_frames, axis=1, join="inner")

# Compute daily log-returns
daily_returns = np.log(close_prices / close_prices.shift(1))
daily_returns = daily_returns.dropna()

avg_daily_returns = daily_returns.mean(axis=1)
returns_train = avg_daily_returns.values.reshape(-1, 1)

# Fit HMM
model = GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=10000,
    random_state=42,
    tol=1e-4,
)
model.fit(returns_train)

# Identify bull regime
if model.means_[0, 0] > model.means_[1, 0]:
    bull_regime = 0
else:
    bull_regime = 1

# ============================================================
# Step 4: Compute regime probabilities
# ============================================================

# --- 4a: Compute daily regime probabilities (online/filtered) ---
# predict_proba uses the forward algorithm — causal, no look-ahead
daily_regime_probs = model.predict_proba(returns_train)
# Extract P(bull) for each day
daily_bull_prob = daily_regime_probs[:, bull_regime]

# Create a Series indexed by date
daily_prob_series = pd.Series(
    daily_bull_prob,
    index=avg_daily_returns.index,
    name="hmm_regime_prob"
)

# --- 4b: Shift by 1 day (paper uses ĥ_{t-1}, not ĥ_t) ---
# At day t, the agent uses yesterday's regime probability
daily_prob_shifted = daily_prob_series.shift(1)
daily_prob_shifted.iloc[0] = 1.0  # assume bull on first day (no prior data)

regime_probs = daily_prob_shifted.values

print(f"Daily regime probs shape: {daily_bull_prob.shape}")
print(f"Shifted regime probs shape: {regime_probs.shape}")
print(f"Sample values (first 20 days): {regime_probs[:20]}")
print(f"NaN count: {np.isnan(regime_probs).sum()}")

# ============================================================
# Validation
# ============================================================

# --- Check 1: All values between 0 and 1 ---
assert np.all(regime_probs >= 0) and np.all(regime_probs <= 1), \
    "Regime probabilities must be in [0, 1]"
print("PASS: All probabilities in [0, 1]")

# --- Check 2: Shape matches daily data ---
assert len(regime_probs) == len(avg_daily_returns), \
    f"Shape mismatch: {len(regime_probs)} vs {len(avg_daily_returns)}"
print(f"PASS: Shape matches ({len(regime_probs)} days)")

# --- Plot ---
fig, ax = plt.subplots(1, 1, figsize=(14, 5))

ax.plot(daily_prob_shifted.index, daily_prob_shifted.values,
        color='green', linewidth=0.8)
ax.set_title("Daily P(Bull) — shifted by 1 day (ĥ_{t-1})")
ax.set_ylabel("Probability")
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("hmm_step4_validation.png", dpi=150)
plt.show()
print("Validation plot saved to hmm_step4_validation.png")