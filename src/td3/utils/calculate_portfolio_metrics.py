"""
Calculate comprehensive portfolio performance metrics.
"""
import csv
import os
import numpy as np

def load_csv_simple(filepath):
    """Load CSV file without pandas."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def calculate_portfolio_metrics(weights_csv, prices_csv=None, initial_cash=10000.0):
    """
    Calculate comprehensive portfolio metrics from weights and price data.
    
    Args:
        weights_csv: Path to weights CSV file
        prices_csv: Path to prices CSV file (if available)
        initial_cash: Starting portfolio value
    
    Returns:
        Dictionary containing all calculated metrics
    """
    # Load weights data
    weights_data = load_csv_simple(weights_csv)
    
    # Extract asset columns (exclude Date and Cash)
    if weights_data:
        asset_cols = [col for col in weights_data[0].keys() if col not in ['Date', 'Cash']]
    else:
        return {}
    
    # For this implementation, we'll calculate metrics based on weight changes
    # In a full implementation, you'd need price data to calculate actual returns
    
    metrics = {}
    
    # Extract weight data as numpy arrays
    weights_matrix = []
    for row in weights_data:
        weight_row = [float(row[col]) for col in asset_cols]
        weights_matrix.append(weight_row)
    weights_matrix = np.array(weights_matrix)
    
    # Calculate cash allocation over time
    cash_allocations = np.array([float(row['Cash']) for row in weights_data])
    
    # Calculate daily changes in weights (proxy for turnover)
    weight_changes = np.diff(weights_matrix, axis=0)
    daily_turnover = np.sum(np.abs(weight_changes), axis=1)
    
    # 1. Daily Returns (using weight changes as proxy)
    # Note: Without actual price data, we can't calculate true returns
    # This is a simplified version using weight stability as proxy
    avg_daily_turnover = np.mean(daily_turnover)
    metrics['average_daily_turnover'] = avg_daily_turnover
    
    # 2. Portfolio Stability Metrics
    metrics['weight_volatility'] = np.std(weights_matrix)
    metrics['avg_cash_allocation'] = np.mean(cash_allocations)
    metrics['cash_volatility'] = np.std(cash_allocations)
    
    # 3. Diversification Metrics
    # Number of assets with >5% allocation on average
    avg_allocations = np.mean(weights_matrix, axis=0)
    diversified_assets = np.sum(avg_allocations > 0.05)
    metrics['diversified_asset_count'] = diversified_assets
    metrics['herfindahl_index'] = np.sum(avg_allocations ** 2)  # Concentration measure
    
    # 4. Risk Metrics (based on weight volatility)
    metrics['portfolio_risk_score'] = np.mean(metrics['weight_volatility'])
    
    # 5. Maximum Position Size
    metrics['max_single_position'] = np.max(weights_matrix)
    metrics['min_single_position'] = np.min(weights_matrix)
    
    # 6. Trading Activity
    metrics['total_trading_activity'] = np.sum(daily_turnover)
    metrics['avg_daily_trading'] = avg_daily_turnover
    
    # 7. Time-based metrics
    metrics['num_trading_days'] = len(weights_data)
    metrics['data_frequency'] = 'daily'
    
    return metrics

def calculate_returns_based_metrics(weights_csv, prices_csv, initial_cash=10000.0, risk_free_rate=0.02):
    """
    Calculate return-based metrics using both weights and price data.
    
    Args:
        weights_csv: Path to weights CSV file
        prices_csv: Path to prices CSV file  
        initial_cash: Starting portfolio value
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
    
    Returns:
        Dictionary containing return-based metrics
    """
    # Load data
    weights_data = load_csv_simple(weights_csv)
    prices_data = load_csv_simple(prices_csv)
    
    if not weights_data or not prices_data:
        print("Error: Could not load data files")
        return {}
    
    # Get asset columns
    asset_cols = [col for col in weights_data[0].keys() if col not in ['Date', 'Cash']]
    price_cols = [col for col in prices_data[0].keys() if col.lower() != 'date']
    
    # Ensure we have matching data
    if len(weights_data) != len(prices_data):
        print(f"Warning: Data length mismatch - weights: {len(weights_data)}, prices: {len(prices_data)}")
        min_length = min(len(weights_data), len(prices_data))
        weights_data = weights_data[:min_length]
        prices_data = prices_data[:min_length]
    
    # Extract weights and prices
    weights_matrix = []
    for row in weights_data:
        weight_row = [float(row[col]) for col in asset_cols]
        weights_matrix.append(weight_row)
    weights_matrix = np.array(weights_matrix)
    
    prices_matrix = []
    for row in prices_data:
        price_row = [float(row[col]) for col in price_cols]
        prices_matrix.append(price_row)
    prices_matrix = np.array(prices_matrix)
    
    # Calculate portfolio value over time
    portfolio_values = []
    cash_allocations = np.array([float(row['Cash']) for row in weights_data])
    
    for i in range(len(weights_matrix)):
        # Portfolio value = cash + sum(weight * price for each asset)
        asset_values = np.sum(weights_matrix[i] * prices_matrix[i])
        portfolio_value = cash_allocations[i] * initial_cash + asset_values * initial_cash
        portfolio_values.append(portfolio_value)
    
    portfolio_values = np.array(portfolio_values)
    
    # Calculate metrics
    metrics = {}
    
    # 1. Daily Returns
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    metrics['daily_returns_mean'] = np.mean(daily_returns)
    metrics['daily_returns_std'] = np.std(daily_returns)
    metrics['daily_returns'] = daily_returns.tolist()
    
    # 2. Total Cumulative Return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    metrics['total_cumulative_return'] = total_return
    metrics['final_portfolio_value'] = portfolio_values[-1]
    metrics['initial_portfolio_value'] = portfolio_values[0]
    
    # 3. Annualized Return
    num_years = len(portfolio_values) / 252  # Assuming 252 trading days per year
    if num_years > 0:
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        metrics['annualized_return'] = annualized_return
    else:
        metrics['annualized_return'] = 0.0
    
    # 4. Annualized Volatility
    daily_vol = np.std(daily_returns)
    annualized_volatility = daily_vol * np.sqrt(252)
    metrics['annualized_volatility'] = annualized_volatility
    
    # 5. Sharpe Ratio
    if annualized_volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        metrics['sharpe_ratio'] = sharpe_ratio
    else:
        metrics['sharpe_ratio'] = 0.0
    
    # 6. Maximum Drawdown
    cumulative_returns = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    metrics['maximum_drawdown'] = max_drawdown
    
    # 7. Sortino Ratio (using downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        if downside_deviation > 0:
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
            metrics['sortino_ratio'] = sortino_ratio
        else:
            metrics['sortino_ratio'] = 0.0
    else:
        metrics['sortino_ratio'] = float('inf')  # No downside risk
    
    # Additional metrics
    metrics['portfolio_values'] = portfolio_values.tolist()
    metrics['num_trading_days'] = len(portfolio_values)
    
    return metrics

def compare_portfolio_metrics(old_weights_csv, new_weights_csv, old_prices_csv=None, new_prices_csv=None):
    """
    Compare portfolio metrics between two approaches.
    
    Args:
        old_weights_csv: Path to old approach weights CSV
        new_weights_csv: Path to new approach weights CSV
        old_prices_csv: Path to old approach prices CSV (optional)
        new_prices_csv: Path to new approach prices CSV (optional)
    
    Returns:
        Comparison results
    """
    print("=" * 70)
    print("PORTFOLIO METRICS COMPARISON")
    print("=" * 70)
    
    # Check if price data is available
    if old_prices_csv and new_prices_csv and os.path.exists(old_prices_csv) and os.path.exists(new_prices_csv):
        print("\nCalculating full return-based metrics...")
        old_metrics = calculate_returns_based_metrics(old_weights_csv, old_prices_csv)
        new_metrics = calculate_returns_based_metrics(new_weights_csv, new_prices_csv)
        
        if old_metrics and new_metrics:
            print_full_metrics_comparison(old_metrics, new_metrics)
    else:
        print("\nPrice data not available, calculating weight-based metrics...")
        old_metrics = calculate_portfolio_metrics(old_weights_csv)
        new_metrics = calculate_portfolio_metrics(new_weights_csv)
        
        if old_metrics and new_metrics:
            print_weight_metrics_comparison(old_metrics, new_metrics)

def print_full_metrics_comparison(old_metrics, new_metrics):
    """Print comparison of return-based metrics."""
    key_metrics = [
        'total_cumulative_return',
        'annualized_return', 
        'annualized_volatility',
        'sharpe_ratio',
        'maximum_drawdown',
        'sortino_ratio'
    ]
    
    print("\n" + "=" * 70)
    print("RETURN-BASED METRICS COMPARISON")
    print("=" * 70)
    
    print(f"{'Metric':<25} {'Without Opt':<15} {'With Opt':<15} {'Change':<15}")
    print("-" * 70)
    
    for metric in key_metrics:
        if metric in old_metrics and metric in new_metrics:
            old_val = old_metrics[metric]
            new_val = new_metrics[metric]
            
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                change = new_val - old_val
                print(f"{metric:<25} {old_val:<15.4f} {new_val:<15.4f} {change:+.4f}")
            else:
                print(f"{metric:<25} {str(old_val):<15} {str(new_val):<15} N/A")
    
    print(f"\nPortfolio Value Growth:")
    print(f"  Initial: ${old_metrics['initial_portfolio_value']:,.2f}")
    print(f"  Final (Old): ${old_metrics['final_portfolio_value']:,.2f}")
    print(f"  Final (New): ${new_metrics['final_portfolio_value']:,.2f}")

def print_weight_metrics_comparison(old_metrics, new_metrics):
    """Print comparison of weight-based metrics."""
    key_metrics = [
        'average_daily_turnover',
        'weight_volatility',
        'avg_cash_allocation',
        'diversified_asset_count',
        'max_single_position'
    ]
    
    print("\n" + "=" * 70)
    print("WEIGHT-BASED METRICS COMPARISON")
    print("=" * 70)
    
    print(f"{'Metric':<30} {'Without Opt':<15} {'With Opt':<15} {'Change':<15}")
    print("-" * 70)
    
    for metric in key_metrics:
        if metric in old_metrics and metric in new_metrics:
            old_val = old_metrics[metric]
            new_val = new_metrics[metric]
            change = new_val - old_val
            print(f"{metric:<30} {old_val:<15.4f} {new_val:<15.4f} {change:+.4f}")

def main():
    """Main function to run portfolio metrics comparison."""
    new_weights = r"/data/teodora_portfolio/test/run_2026-06-16_08-30-38_True_test/weights/weights_td3_model_run_0.pth.csv"
    old_weights = r"/data/teodora_portfolio/test/run_2026-06-16_08-47-11_False_test/weights/weights_td3_model_run_0.pth.csv"
    
    # Check if we have price data available
    # You would need to provide paths to price data files
    old_prices = r"prices.csv"  # Add path if available
    new_prices = r"prices.csv"   # Add path if available
    
    compare_portfolio_metrics(old_weights, new_weights, old_prices, new_prices)

if __name__ == "__main__":
    main()
