# reconstruct daily positions from QC trade events
# taking into account both entry and exit times

import pandas as pd
from datetime import datetime, timedelta

trades = pd.read_csv(r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/Muscular Yellow Koala_trades.csv")

print(f"Processing first {len(trades)} records")

# Parse date columns and remove timezone info
trades['Entry Time'] = pd.to_datetime(trades['Entry Time']).dt.tz_localize(None)
trades['Exit Time'] = pd.to_datetime(trades['Exit Time']).dt.tz_localize(None)

# Get all unique tickers
tickers = sorted(trades['Symbols'].unique())
print(f"Tickers found: {tickers}")

# Get all unique dates from Entry Time and Exit Time columns
entry_dates = trades['Entry Time'].dt.date.unique()
exit_dates = trades['Exit Time'].dt.date.unique()
all_dates = sorted(set(entry_dates) | set(exit_dates))
date_range = pd.to_datetime(all_dates)
print(f"Found {len(date_range)} unique dates from trades")

# Initialize dataframe for daily positions
daily_positions = pd.DataFrame(index=date_range, columns=tickers)
daily_positions.index.name = 'Date'
daily_positions = daily_positions.fillna(0)

# Calculate daily positions for each ticker
for ticker in tickers:
    print(f"Processing {ticker}...")
    ticker_trades = trades[trades['Symbols'] == ticker]
    
    for date in date_range:
        # Convert to datetime at end of day (23:59:59) for proper comparison
        date_dt = pd.Timestamp(date).replace(hour=23, minute=59, second=59)
        
        # Add quantity for trades that have entered on or before this date
        entered = ticker_trades[ticker_trades['Entry Time'] <= date_dt]
        # Subtract quantity for trades that have exited on or before this date
        exited = ticker_trades[ticker_trades['Exit Time'] <= date_dt]
        
        total_quantity = entered['Quantity'].sum() - exited['Quantity'].sum()
        daily_positions.loc[date, ticker] = int(total_quantity)

# Reorder columns to match quantities_qc_logic.csv format
column_order = ['AAPL', 'AMZN', 'GOOGL', 'JNJ', 'JPM', 'META', 'MSFT', 'NVDA', 'TSLA', 'XOM']
# Only include columns that exist in our data
column_order = [col for col in column_order if col in daily_positions.columns]
# Add any additional tickers not in the standard order
additional_tickers = [col for col in daily_positions.columns if col not in column_order]
final_columns = column_order + additional_tickers

daily_positions = daily_positions[final_columns]

# Save to CSV
output_file = r"/home/teodora/Desktop/workspace/DP/simulations/test/run_2026-04-30_12-35-17/returns/daily_positions_test.csv"
daily_positions.to_csv(output_file)
print(f"Daily positions saved to {output_file}")

# Display first few rows
print("\nFirst 10 rows of output:")
print(daily_positions.head(10))
