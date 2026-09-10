import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, 'src')

# --- Configuration (matches your app_config.py) ---

DATA_DIR = "indicators"                # or "indicators" for parquet
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "XOM"]
TRAIN_START = "2016-05-01"
TRAIN_END = "2020-05-30"

# --- Step 1: Load close prices for all tickers ---
close_frames = []
for ticker in TICKERS:
    path = os.path.join(DATA_DIR, ticker, "normalized.parquet")
    print(path)
    df = pd.read_parquet(path)
    df["date"]=pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    # Filter to training period
    df = df.loc[TRAIN_START:TRAIN_END, ["close"]]
    df.columns = [ticker]
    close_frames.append(df)

prices_df = pd.concat(close_frames, axis=1)

print(prices_df.head(20))

# # Merge all tickers on date (inner join = only dates all 10 have data)
# close_prices = pd.concat(close_frames, axis=1, join="inner")
# print(f"Close prices shape: {close_prices.shape}")  # expect ~(1008, 10)
# print(f"Date range: {close_prices.index[0]} to {close_prices.index[-1]}")



# # --- Step 2: Compute daily log-returns per ticker ---
# log_returns = np.log(close_prices / close_prices.shift(1))
# log_returns = log_returns.dropna()   # drop first row (NaN)
#
# # --- Step 3: Compute equally-weighted average return ---
# avg_returns = log_returns.mean(axis=1)  # shape: (T-1,)
#
# # --- Step 4: Reshape for hmmlearn ---
# returns_train = avg_returns.values.reshape(-1, 1)  # shape: (T-1, 1)
# print(f"returns_train shape: {returns_train.shape}")
#
#
# # --- Sanity checks ---
# # 1. Shape: should be roughly (1000, 1)
# assert returns_train.shape[1] == 1
# assert 10_000 < returns_train.shape[0] < 16_000, f"Unexpected length: {returns_train.shape[0]}"
#
# # 2. No NaN or Inf
# assert not np.any(np.isnan(returns_train)), "NaN found in returns"
# assert not np.any(np.isinf(returns_train)), "Inf found in returns"
#
# # 3. Statistical sanity
# mean_ret = np.mean(returns_train)
# std_ret = np.std(returns_train)
# print(f"Mean daily return: {mean_ret:.6f}")   # expect small positive ~0.0001 to 0.001
# print(f"Std daily return:  {std_ret:.6f}")     # expect ~0.005 to 0.02
#
# # 4. Visual inspection
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(2, 1, figsize=(12, 8))
#
# # Return series — look for volatility clusters (especially early 2020 COVID crash)
# axes[0].plot(avg_returns.index, returns_train.flatten(), linewidth=0.5)
# axes[0].set_title("Equally-Weighted Average Daily Log-Returns (Training Period)")
# axes[0].set_ylabel("Log Return")
# axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
#
# # Histogram — should be roughly bell-shaped, potentially fat-tailed
# axes[1].hist(returns_train, bins=50, edgecolor='black', alpha=0.7)
# axes[1].set_title("Distribution of Daily Log-Returns")
# axes[1].set_xlabel("Log Return")
# axes[1].set_ylabel("Frequency")
#
# plt.tight_layout()
# plt.savefig("hmm_step2_validation.png", dpi=150)
# plt.show()
# print("Validation plots saved.")