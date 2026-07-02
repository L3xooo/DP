import sys
sys.path.insert(0, 'src')

import pandas as pd
from td3.data.data_processor import DataProcessor
from td3.config.app_config import AppConfig
from td3.metrics.metrics import ExperimentMetrics
from td3.utils.file_utils import create_experiment_directories, create_directory

app_config = AppConfig()
'''
df = pd.read_csv("/simulations/test/run_2026-04-30_12-35-17/weights/weights_td3_model_run_1.csv")

for col in df.columns:
    if col != "Date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["row_sum"] = df.drop(columns=["Date", "Cash"]).sum(axis=1)

df.to_csv("/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/weights/weights_td3_model_run_1.csv", index=False)
'''

dp = DataProcessor(data_dir=app_config.data_dir, parquet_dir=app_config.parquet_dir)
df = dp.load_panel(
        tickers=app_config.ticker_config.tickers,
        start="2021-06-1",
        end="2024-12-30")
# Extract close prices
df_prices = df.loc[:, (slice(None), ["close"])]

# Flatten MultiIndex columns → keep only ticker names
df_prices.columns = df_prices.columns.get_level_values(0)

# Now reset index safely
df_prices = df_prices.reset_index()
df_prices.to_csv("prices.csv", index=False)
print(df_prices.head())


# nacitaj vahy
#weights = pd.read_csv("/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-20_19-38-13/weights/weights_td3_model_run_0_output1.csv", parse_dates=["Date"]).set_index("Date")
# print(weights.head())

# nacitaj ceny
#prices = pd.read_csv("/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-20_19-38-13/weights/weights_td3_model_run_0_prices.csv", parse_dates=["date"]).set_index("date")

# returns = prices.pct_change().fillna(0)
# print(returns.head())