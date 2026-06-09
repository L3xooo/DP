# import pandas as pd
#
#
# weights = pd.read_csv("simulations/test/run_2026-04-20_19-38-13/weights/weights_td3_model_run_0_output.csv", parse_dates=["Date"]).set_index("Date")
# prices = pd.read_csv("simulations/test/run_2026-04-20_19-38-13/weights/weights_td3_model_run_0_prices.csv", parse_dates=["date"]).set_index("date")
#
# date_to_select = "2024-03-01"
# ticker="AAPL"
# w = weights.loc[date_to_select][ticker]
# p_1 = prices.shift(1).loc[date_to_select][ticker]
# p = prices.loc[date_to_select][ticker]
#
# # print(w)
# # print(p_1, p)
#
# returns = prices.pct_change().fillna(0)
# print(returns.loc[date_to_select][ticker])
#
# portfolio_returns = (w * returns.loc[date_to_select][ticker])
# print(portfolio_returns)

import pandas as pd
import matplotlib.pyplot as plt

weights_dir = "../../simulations/test/run_2026-04-22_20-12-34/"
initial_capital = 10000

# Load weights
weights = pd.read_csv(
    weights_dir + "weights/weights_td3_model_run_0.csv",
    parse_dates=["Date"]
).set_index("Date")

if "row_sum" in weights.columns:
    weights = weights.drop(columns=["row_sum"])

# Load prices
prices = pd.read_csv(
    weights_dir + "weights/weights_td3_model_run_0_prices.csv",
    parse_dates=["date"]
).set_index("date")

# Align dates
common_index = prices.index.intersection(weights.index)
weights = weights.loc[common_index]
prices = prices.loc[common_index]

# Daily asset returns
returns = prices.pct_change().fillna(0)

# Portfolio return per day
portfolio_returns = (weights * returns).sum(axis=1)

# Cumulative product
cum_prod = (1 + portfolio_returns).cumprod()

# Portfolio value
portfolio_value = cum_prod * initial_capital

# Drawdown
running_max = portfolio_value.cummax()
drawdown = (portfolio_value - running_max) / running_max

# Print summary
final_value = portfolio_value.iloc[-1]
profit = final_value - initial_capital

print("Initial capital:", initial_capital)
print("Final portfolio value:", round(final_value, 2))
print("Total profit:", round(profit, 2))

# Save daily asset returns
returns.to_csv(weights_dir + "returns/daily_asset_returns_run0.csv")

# Build combined DataFrame
combined = returns.copy()
combined["portfolio_return"] = portfolio_returns
combined["cum_prod"] = cum_prod
combined["portfolio_value"] = portfolio_value

# Save combined CSV
combined.to_csv(weights_dir + "returns/returns_with_portfolio_run0.csv")

# ---------------------------------------------------------
# 1. Portfolio value plot
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(portfolio_value.index, portfolio_value.values, label="Portfolio Value", linewidth=2)
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(weights_dir + "plots/portfolio_value_plot_run0.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 2. Daily portfolio returns plot
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(portfolio_returns.index, portfolio_returns.values, label="Daily Portfolio Return", linewidth=1.5)
plt.axhline(0, color="black", linewidth=1)
plt.title("Daily Portfolio Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(weights_dir + "plots/daily_portfolio_returns_run0.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 3. Drawdown plot
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(drawdown.index, drawdown.values, label="Drawdown", color="red", linewidth=2)
plt.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
plt.title("Portfolio Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(weights_dir + "plots/drawdown_plot_run0.png", dpi=200)
plt.close()
