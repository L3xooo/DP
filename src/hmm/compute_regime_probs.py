"""
Step 4: Compute regime probabilities online and broadcast to 30-min bars.

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

# Load close prices
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

# 30-min log-returns → aggregate to daily
log_returns_30min = np.log(close_prices / close_prices.shift(1))
log_returns_30min = log_returns_30min.dropna()

daily_returns = log_returns_30min.resample("D").sum()
daily_returns = daily_returns.loc[(daily_returns != 0).any(axis=1)]
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

# --- 4c: Broadcast daily probability to 30-min bars ---
# Each 30-min bar gets the regime probability of its trading day
bar_dates = log_returns_30min.index  # 30-min timestamps
bar_days = bar_dates.normalize()     # extract just the date part

regime_probs_30min = bar_days.map(daily_prob_shifted).values

# Handle any bars whose day is missing (e.g., first day)
regime_probs_30min = np.nan_to_num(regime_probs_30min, nan=1.0)

print(f"Daily regime probs shape: {daily_bull_prob.shape}")
print(f"30-min regime probs shape: {regime_probs_30min.shape}")
print(f"Sample values (first 20 bars): {regime_probs_30min[:20]}")
print(f"NaN count: {np.isnan(regime_probs_30min).sum()}")

# ============================================================
# Validation
# ============================================================

# --- Check 1: All values between 0 and 1 ---
assert np.all(regime_probs_30min >= 0) and np.all(regime_probs_30min <= 1), \
    "Regime probabilities must be in [0, 1]"
print("PASS: All probabilities in [0, 1]")

# --- Check 2: Bars on the same day have the same probability ---
test_day = bar_days[100]
same_day_mask = bar_days == test_day
unique_vals = np.unique(regime_probs_30min[same_day_mask])
assert len(unique_vals) == 1, f"Bars on {test_day} have different probs: {unique_vals}"
print(f"PASS: All bars on {test_day} share the same probability ({unique_vals[0]:.4f})")

# --- Check 3: Shape matches the 30-min data ---
assert len(regime_probs_30min) == len(log_returns_30min), \
    f"Shape mismatch: {len(regime_probs_30min)} vs {len(log_returns_30min)}"
print(f"PASS: Shape matches ({len(regime_probs_30min)} bars)")

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Daily regime probability
axes[0].plot(daily_prob_shifted.index, daily_prob_shifted.values,
             color='green', linewidth=0.8)
axes[0].set_title("Daily P(Bull) — shifted by 1 day (ĥ_{t-1})")
axes[0].set_ylabel("Probability")
axes[0].set_ylim(0, 1)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Broadcasted to 30-min bars (zoom into a 2-week window to verify)
zoom_start = 5000
zoom_end = 5200
axes[1].plot(range(zoom_start, zoom_end),
             regime_probs_30min[zoom_start:zoom_end],
             color='green', linewidth=0.8, drawstyle='steps-post')
axes[1].set_title("P(Bull) Broadcasted to 30-min Bars (zoomed 200-bar window)")
axes[1].set_ylabel("Probability")
axes[1].set_ylim(0, 1)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("hmm_step4_validation.png", dpi=150)
plt.show()
print("Validation plot saved to hmm_step4_validation.png")