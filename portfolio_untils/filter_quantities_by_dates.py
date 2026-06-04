import pandas as pd

# Read the quantities file
quantities = pd.read_csv(r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/quantities_qc_logic.csv", parse_dates=["Unnamed: 0"])
quantities = quantities.rename(columns={"Unnamed: 0": "Date"})
quantities = quantities.set_index("Date")

# Read the daily positions test file to get the dates
daily_positions = pd.read_csv(r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/daily_positions_test.csv", parse_dates=["Date"])
dates_to_keep = daily_positions["Date"].unique()

print(f"Total dates in quantities_qc_logic_2.csv: {len(quantities)}")
print(f"Dates to keep from daily_positions_test.csv: {len(dates_to_keep)}")

# Filter quantities to only include dates that exist in daily_positions_test.csv
filtered_quantities = quantities.loc[quantities.index.isin(dates_to_keep)]

print(f"Filtered quantities count: {len(filtered_quantities)}")

# Save to new CSV file
output_file = r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/quantities_qc_logic_2_filtered_using_round.csv"
filtered_quantities.to_csv(output_file)
print(f"Filtered file saved to {output_file}")

# Display first few rows
print("\nFirst 10 rows of filtered output:")
print(filtered_quantities.head(10))
