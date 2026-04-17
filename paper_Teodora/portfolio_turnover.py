import pandas as pd
import numpy as np

def compute_turnover(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Remove non-weight columns if needed (e.g., time column)
    # Keep only numeric columns
    weights = df.select_dtypes(include=[np.number])

    # Convert to numpy array for easier math
    W = weights.values  # shape: (T, N)

    # Compute absolute differences between consecutive rows
    abs_diff = np.abs(W[1:] - W[:-1])  # shape: (T-1, N)

    # Sum across assets, then average across time
    turnover = abs_diff.sum(axis=1).mean()

    return turnover

# Example usage:
csv_file = "/home/teodora/Desktop/workspace/DP/simulations/train/run_2026-03-10_20-53-40/weights/run_0/episode_1_weights.csv"
turnover_value = compute_turnover(csv_file)
print("Portfolio Turnover:", turnover_value)