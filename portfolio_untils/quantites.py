# zla logika


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

quantity = (weights * initial_capital) / prices
quantity = quantity.drop(columns=["Cash", "row_sum"])
print(quantity)
quantity.to_csv("/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/quantities.csv")