import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List

def plot_episode_weights(
    all_weights: List[List[float]],
    tickers: List[str],
    save_dir: str = "returns",
    episode: int = 1,
    filename: str | None = None,
    title: str | None = None,
    dpi: int = 200,
) -> str | None:
    """Plot portfolio weights as a stacked bar chart.

    Args:
        all_weights: List of weight arrays for each time step.
        tickers: List of asset ticker symbols.
        save_dir: Directory to save the plot.
        episode: Episode number for default filename/title.
        filename: Custom filename for the plot.
        title: Custom title for the plot.
        dpi: Resolution of the saved image.

    Returns:
        Path to the saved plot file, or None if no weights provided.
    """
    if len(all_weights) == 0:
        return None

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        filename = f"episode_{episode}_weights.jpg"

    if title is None:
        title = f"Portfolio Weights Over Time - Episode {episode}"

    W = np.asarray(all_weights, dtype=float)
    T, N = W.shape
    x = np.arange(T)

    W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(T)

    for i in range(N):
        plt.bar(x, W[:, i], bottom=bottom, label=tickers[i])
        bottom += W[:, i]

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Weight")
    plt.ylim(0, 1)
    plt.legend(title="Assets", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, format="jpg")
    plt.close()

    return filename


def main():
    """Main function to run portfolio simulation and analysis."""
    # Load data
    weights = pd.read_csv(
        r"/home/teodora/Desktop/workspace/DP/portfolio_utils_results/data/weights_td3_model_run_1.csv",
        parse_dates=["Date"]
    ).set_index("Date")

    prices = pd.read_csv(
        r"/home/teodora/Desktop/workspace/DP/portfolio_utils_results/data/weights_td3_model_run_1_prices.csv",
        parse_dates=["date"]
    ).set_index("date")

    capital = 10000
    cash = capital
    positions = {col: 0 for col in prices.columns if col != "Cash"}

    portfolio_values = []
    positions_history = []
    dates = []

    # what this fixes?
    # aligns trade timing with QC
    # fixes 1-day shift
    # matches when positions actually change

    # t -> signal (weights known), t+1 -> execution (order filled), t+2 -> evaluation (portfolio valued)
    # Day t     → compute weights
    # Day t+1   → trades execute
    # Day t+2   → portfolio P&L realized

    for t in range(len(prices) - 2):  # shift one more step, why -2?
        date = prices.index[t]
        price_signal = prices.iloc[t]
        price_execution = prices.iloc[t + 1]  # price next
        price_next = prices.iloc[t + 2]

        weights_t = weights.iloc[t]

        print(f"\nDate: {date}")
        print("weights sum:", weights_t.sum())
        print("cash weight:", weights_t['Cash'])
        print("invested weight:", weights_t.drop('Cash').sum())

        # compute total value using current holdings at signal time
        # portfolio value at t (before rebalancing)
        holdings_value = sum(positions[a] * price_signal[a] for a in positions)
        total_value = cash + holdings_value # portfolio value at t

        # rebalance USING execution price
        # positions -> exclude cash automatically
        for asset in positions:
            target_value = weights_t[asset] * total_value
            current_value = positions[asset] * price_execution[asset]

            delta_value = target_value - current_value
            delta_shares = int(delta_value / price_execution[asset])
            # updating positions and cash
            positions[asset] += delta_shares
            cash -= delta_shares * price_execution[asset]

        # store snapshot after rebalance
        positions_history.append(positions.copy())
        dates.append(prices.index[t + 1])  # execution day

        # valuation, You compute portfolio value after prices move again
        holdings_value_next = sum(positions[a] * price_next[a] for a in positions)
        total_value_next = cash + holdings_value_next

        portfolio_values.append(total_value_next)

    positions_df = pd.DataFrame(positions_history, index=dates)
    # positions_df.to_csv("returns/quantities_qc_logic_2_using_round.csv")
    # portfolio_values.to_csv("/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/quantities_qc_logic.csv")

    positions_df["PortfolioValue"] = portfolio_values
    positions_df["Returns"] = positions_df["PortfolioValue"].pct_change()  # z tehoto vlastne pocitame secko druhe
    positions_df.to_csv("returns/positions_with_value.csv")

    # Calculate statistics
    mean_return = positions_df["Returns"].mean()
    std_return = positions_df["Returns"].std()
    cumulative_return = (positions_df["PortfolioValue"].iloc[-1] / positions_df["PortfolioValue"].iloc[0]) - 1

    # Sharpe ratio (assuming daily returns, annualized)
    risk_free_rate = 0.0  # can be adjusted to current risk-free rate
    sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0
    sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)  # annualize for daily data

    # Maximum drawdown
    cumulative_returns = (1 + positions_df["Returns"]).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    print(f"\nPortfolio Statistics:")
    print(f"Mean Return (daily): {mean_return:.6f}")
    print(f"Volatility (std dev, daily): {std_return:.6f}")
    print(f"Cumulative Return: {cumulative_return:.6f}")
    print(f"Sharpe Ratio (daily): {sharpe_ratio:.6f}")
    print(f"Sharpe Ratio (annualized): {sharpe_ratio_annualized:.6f}")
    print(f"Maximum Drawdown: {max_drawdown:.6f}")

    print(positions_df.head())

    # Plotting
    # Plot weights using the new function
    all_weights = weights.values.tolist()
    tickers = weights.columns.tolist()
    plot_episode_weights(
        all_weights,
        tickers,
        filename="/home/teodora/Desktop/workspace/DP/portfolio_utils_results/plots/weights_plot.jpg",
        title="Portfolio Weights Over Time"
    )

    # Plot portfolio value and returns
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Portfolio Value over time
    positions_df["PortfolioValue"].plot(
        ax=axes[0],
        title="Portfolio Value Over Time",
        color="blue"
    )
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Daily Returns
    positions_df["Returns"].plot(
        ax=axes[1],
        title="Daily Returns",
        color="green"
    )
    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Return")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/teodora/Desktop/workspace/DP/portfolio_utils_results/plots/portfolio_plots.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()