import pandas as pd

# Read the Parquet file
df = pd.read_parquet("/indicators/AAPL/normalized.parquet")

# Display the first few rows
print(df.head(100))
df.head(100).to_csv("/home/xsimont/DP/indicators/data.csv")