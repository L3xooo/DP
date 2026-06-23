"""
Create professional visualization plots for regime awareness portfolio comparison.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_csv_simple(filepath):
    """Load CSV file without pandas."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def calculate_returns_based_metrics(weights_csv, prices_csv, initial_cash=10000.0, risk_free_rate=0.02):
    """Calculate return-based metrics using both weights and price data."""
    weights_data = load_csv_simple(weights_csv)
    prices_data = load_csv_simple(prices_csv)
    
    if not weights_data or not prices_data:
        print("Error: Could not load data files")
        return {}
    
    asset_cols = [col for col in weights_data[0].keys() if col not in ['Date', 'Cash']]
    price_cols = [col for col in prices_data[0].keys() if col.lower() != 'date']
    
    if len(weights_data) != len(prices_data):
        min_length = min(len(weights_data), len(prices_data))
        weights_data = weights_data[:min_length]
        prices_data = prices_data[:min_length]
    
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
    
    portfolio_values = []
    cash_allocations = np.array([float(row['Cash']) for row in weights_data])
    
    for i in range(len(weights_matrix)):
        asset_values = np.sum(weights_matrix[i] * prices_matrix[i])
        portfolio_value = cash_allocations[i] * initial_cash + asset_values * initial_cash
        portfolio_values.append(portfolio_value)
    
    portfolio_values = np.array(portfolio_values)
    
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    cumulative_returns = np.cumprod(1 + daily_returns)
    
    return {
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
        'cumulative_returns': cumulative_returns,
        'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
        'annualized_return': (1 + (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1,
        'volatility': np.std(daily_returns) * np.sqrt(252),
        'sharpe_ratio': ((1 + (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1 - risk_free_rate) / (np.std(daily_returns) * np.sqrt(252)),
        'max_drawdown': np.min((np.cumprod(1 + daily_returns) - np.maximum.accumulate(np.cumprod(1 + daily_returns))) / np.maximum.accumulate(np.cumprod(1 + daily_returns)))
    }

def create_comparison_plots(old_weights, new_weights, prices_csv):
    """Create comprehensive comparison plots."""
    # Calculate metrics for both approaches
    old_metrics = calculate_returns_based_metrics(old_weights, prices_csv)
    new_metrics = calculate_returns_based_metrics(new_weights, prices_csv)
    
    if not old_metrics or not new_metrics:
        print("Error: Could not calculate metrics")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Portfolio Value Over Time
    ax1 = plt.subplot(2, 3, 1)
    days = np.arange(len(old_metrics['portfolio_values']))
    ax1.plot(days, old_metrics['portfolio_values'], label='Without Regime Awareness', linewidth=2, color='#E74C3C')
    ax1.plot(days, new_metrics['portfolio_values'], label='With Regime Awareness', linewidth=2, color='#2ECC71')
    ax1.set_xlabel('Trading Days', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Portfolio Value Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=9)
    
    # 2. Cumulative Returns
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(days[1:], old_metrics['cumulative_returns'], label='Without Regime Awareness', linewidth=2, color='#E74C3C')
    ax2.plot(days[1:], new_metrics['cumulative_returns'], label='With Regime Awareness', linewidth=2, color='#2ECC71')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Trading Days', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Returns', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=9)
    
    # 3. Daily Returns Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(old_metrics['daily_returns'], bins=50, alpha=0.6, label='Without Regime Awareness', color='#E74C3C', edgecolor='black', linewidth=0.5)
    ax3.hist(new_metrics['daily_returns'], bins=50, alpha=0.6, label='With Regime Awareness', color='#2ECC71', edgecolor='black', linewidth=0.5)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Daily Returns', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Daily Returns Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='both', labelsize=9)
    
    # 4. Key Metrics Comparison (Bar Chart)
    ax4 = plt.subplot(2, 3, 4)
    metrics = ['Total Return\n(%)', 'Annualized\nReturn (%)', 'Volatility\n(%)', 'Sharpe\nRatio', 'Max\nDrawdown (%)']
    old_values = [
        old_metrics['total_return'] * 100,
        old_metrics['annualized_return'] * 100,
        old_metrics['volatility'] * 100,
        old_metrics['sharpe_ratio'],
        old_metrics['max_drawdown'] * 100
    ]
    new_values = [
        new_metrics['total_return'] * 100,
        new_metrics['annualized_return'] * 100,
        new_metrics['volatility'] * 100,
        new_metrics['sharpe_ratio'],
        new_metrics['max_drawdown'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax4.bar(x - width/2, old_values, width, label='Without Regime Awareness', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax4.bar(x + width/2, new_values, width, label='With Regime Awareness', color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax4.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Values', fontsize=11, fontweight='bold')
    ax4.set_title('Key Performance Metrics Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='both', labelsize=9)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold')
    
    # 5. Drawdown Analysis
    ax5 = plt.subplot(2, 3, 5)
    old_dd = (old_metrics['cumulative_returns'] - np.maximum.accumulate(old_metrics['cumulative_returns'])) / np.maximum.accumulate(old_metrics['cumulative_returns'])
    new_dd = (new_metrics['cumulative_returns'] - np.maximum.accumulate(new_metrics['cumulative_returns'])) / np.maximum.accumulate(new_metrics['cumulative_returns'])
    ax5.fill_between(days[1:], old_dd, 0, alpha=0.4, label='Without Regime Awareness', color='#E74C3C')
    ax5.fill_between(days[1:], new_dd, 0, alpha=0.4, label='With Regime Awareness', color='#2ECC71')
    ax5.set_xlabel('Trading Days', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Drawdown', fontsize=11, fontweight='bold')
    ax5.set_title('Drawdown Analysis', fontsize=13, fontweight='bold')
    ax5.legend(loc='lower left', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='both', labelsize=9)
    
    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
    REGIME AWARENESS PERFORMANCE SUMMARY
    
    Without Regime Awareness:
    • Total Return: {old_metrics['total_return']*100:.2f}%
    • Annualized Return: {old_metrics['annualized_return']*100:.2f}%
    • Volatility: {old_metrics['volatility']*100:.2f}%
    • Sharpe Ratio: {old_metrics['sharpe_ratio']:.4f}
    • Max Drawdown: {old_metrics['max_drawdown']*100:.2f}%
    • Final Portfolio: ${old_metrics['portfolio_values'][-1]:,.0f}
    
    With Regime Awareness:
    • Total Return: {new_metrics['total_return']*100:.2f}%
    • Annualized Return: {new_metrics['annualized_return']*100:.2f}%
    • Volatility: {new_metrics['volatility']*100:.2f}%
    • Sharpe Ratio: {new_metrics['sharpe_ratio']:.4f}
    • Max Drawdown: {new_metrics['max_drawdown']*100:.2f}%
    • Final Portfolio: ${new_metrics['portfolio_values'][-1]:,.0f}
    
    IMPROVEMENT:
    • Return: +{(new_metrics['total_return'] - old_metrics['total_return'])*100:+.2f}%
    • Volatility: {(new_metrics['volatility'] - old_metrics['volatility'])*100:+.2f}%
    • Sharpe: {new_metrics['sharpe_ratio'] - old_metrics['sharpe_ratio']:+.4f}
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Regime Awareness Impact on Portfolio Performance', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    # Save the figure
    output_path = 'regime_awareness_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

def main():
    """Main function to create comparison plots."""
    new_weights = r"/data/teodora_portfolio/test/run_2026-06-16_08-30-38_True_test/weights/weights_td3_model_run_0.pth.csv"
    old_weights = r"/data/teodora_portfolio/test/run_2026-06-16_08-47-11_False_test/weights/weights_td3_model_run_0.pth.csv"
    prices_csv = r"prices.csv"
    
    create_comparison_plots(old_weights, new_weights, prices_csv)

if __name__ == "__main__":
    main()
