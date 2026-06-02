import pandas as pd
import matplotlib.pyplot as plt

# načítanie dát
weights = pd.read_csv(
    r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/weights/weights_td3_model_run_1.csv",
    parse_dates=["Date"]
).set_index("Date")
weights = weights.drop(columns=["row_sum"])

prices = pd.read_csv(
    r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/weights/weights_td3_model_run_1_prices.csv",
    parse_dates=["date"]
).set_index("date")

capital = 10000
cash = capital
positions = {col: 0 for col in prices.columns if col != "Cash"}

portfolio_values = []
positions_history = []
dates = []

# for t in range(len(prices) - 1):
#     date = prices.index[t]
#     price_today = prices.iloc[t]
#     price_next = prices.iloc[t+1]
#     w = weights.iloc[t]
#
#     holdings_value = sum(positions[a] * price_today[a] for a in positions)
#     total_value = cash + holdings_value
#
#     for asset in positions:
#         target_value = w[asset] * total_value
#         current_value = positions[asset] * price_next[asset]
#
#         delta_value = target_value - current_value
#         delta_shares = int(delta_value / price_today[asset])
#
#         positions[asset] += delta_shares # you rebalances fully every day
#         cash -= delta_shares * price_today[asset]
#
#     positions_history.append(positions.copy())
#     dates.append(date)
#
#     holdings_value_next = sum(positions[a] * price_next[a] for a in positions)
#     total_value_next = cash + holdings_value_next
#
#     portfolio_values.append(total_value_next)

# what this fixes?
# aligns trade timing with QC
# fixes 1-day shift
# matches when positions actually change


for t in range(len(prices) - 2):   # shift one more step, why -2?
    date = prices.index[t]
    price_signal = prices.iloc[t]
    price_execution = prices.iloc[t + 1] # price next
    price_next = prices.iloc[t + 2]

    w = weights.iloc[t]

    print(f'\nDate: {date}')
    print('weights sum:', w.sum())
    print('cash weight:', w['Cash'])
    print('invested weight:', w.drop('Cash').sum())


    # compute total value using current holdings at signal time
    holdings_value = sum(positions[a] * price_signal[a] for a in positions)
    total_value = cash + holdings_value

    # rebalance USING execution price
    # positions -> exclude cash automatically
    for asset in positions:
        target_value = w[asset] * total_value
        current_value = positions[asset] * price_execution[asset]

        delta_value = target_value - current_value
        delta_shares = int(delta_value / price_execution[asset])

        positions[asset] += delta_shares
        cash -= delta_shares * price_execution[asset]

    # store snapshot after rebalance#
    positions_history.append(positions.copy())
    dates.append(prices.index[t + 1])  # execution day

    # valuation
    holdings_value_next = sum(positions[a] * price_next[a] for a in positions)
    total_value_next = cash + holdings_value_next

    portfolio_values.append(total_value_next)

positions_df = pd.DataFrame(positions_history, index=dates)
positions_df.to_csv("/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/quantities_qc_logic_2.csv")
# portfolio_values.to_csv("/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/quantities_qc_logic.csv")
print (positions_df.head())