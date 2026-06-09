import pandas as pd
import matplotlib.pyplot as plt

initial_capital = 10000
capital = initial_capital

# načítanie dát
weights = pd.read_csv(
    r"/simulations/test/run_2026-04-30_12-35-17/weights/weights_td3_model_run_1.csv",
    parse_dates=["Date"]
).set_index("Date")

prices = pd.read_csv(
    r"/simulations/test/run_2026-04-30_12-35-17/weights/weights_td3_model_run_1_prices.csv",
    parse_dates=["date"]
).set_index("date")

# zosúladenie dátumov
common_index = prices.index.intersection(weights.index)
common_index = common_index.sort_values()

weights = weights.loc[common_index]
prices = prices.loc[common_index]

print(common_index)

prices = prices.sort_index()

# ručný výpočet denných výnosov
returns = prices.copy()
for col in prices.columns:
    returns[col] = (prices[col] - prices[col].shift(1)) / prices[col].shift(1)

# 2/29 = nulový výnos
returns.iloc[0] = 0

# uloženie do CSV
returns.to_csv(
    r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/daily_asset_returns_run0.csv"
)

portfolio_returns = weights * returns

# 2–3. súčet vážených výnosov
portfolio_returns["sucet"] = portfolio_returns.sum(axis=1)

# kontajnery na výsledky
capital_series = []
profit_series = []

# 4–6. iterácia po dňoch
for date in portfolio_returns.index:
    daily_return = portfolio_returns.loc[date, "sucet"]
    daily_profit = capital * daily_return
    capital += daily_profit

    profit_series.append(daily_profit)
    capital_series.append(capital)

# uloženie výsledkov
portfolio_returns["zisk"] = profit_series
portfolio_returns["vynos_portfolia"] = capital_series

portfolio_returns.to_csv(
    r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/portfolio_returns_run0.csv"
)
print(portfolio_returns[["sucet", "zisk", "vynos_portfolia"]])

plt.figure(figsize=(10, 5))
plt.plot(portfolio_returns["vynos_portfolia"].index, portfolio_returns["vynos_portfolia"].values, label="Portfolio Value", linewidth=2)

plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/plots/portfolio_value_plot_run0.png", dpi=200)
plt.close()

