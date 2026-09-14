"""
Visualization module for regime classification evaluation.

Generates publication-ready plots for regime evaluation results:
1. Regime Timeline: colored bar chart comparing true vs predicted
2. Confusion Matrix Heatmap: row-normalized with annotations
3. Transition Detection Plot: scatter marking true and predicted transitions
4. Lag Histogram: signed lag distribution
5. Regime-Conditional Performance: grouped bar charts
6. Correct vs Incorrect Performance: side-by-side comparison

Uses consistent colors: Bullish=#2ECC71, Neutral=#F39C12, Bearish=#E74C3C

Author: Devin (Evaluation Framework)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from td3.benchmark_utils.regime_evaluation import RegimeEvaluationResult


# Color scheme
COLORS = {
    0: "#2ECC71",  # Bullish - green
    1: "#F39C12",  # Neutral - orange
    2: "#E74C3C",  # Bearish - red
}

REGIME_NAMES = {0: "Bullish", 1: "Neutral", 2: "Bearish"}


def plot_regime_timeline(result: RegimeEvaluationResult, figsize: tuple = (16, 6)) -> plt.Figure:
    """
    Plot regime timeline: two-row colored bar chart (true top, predicted bottom).

    Shows regimes over time with disagreement shading.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    T = len(result.y_true)
    time_steps = np.arange(T)

    # Plot true regimes
    for regime in range(3):
        mask = result.y_true == regime
        ax1.scatter(time_steps[mask], np.ones(np.sum(mask)), c=COLORS[regime], s=1, label=REGIME_NAMES[regime])
    ax1.set_ylim(0.5, 1.5)
    ax1.set_yticks([])
    ax1.set_title("True Regimes", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", ncol=3)
    ax1.grid(True, alpha=0.3, axis="x")

    # Plot predicted regimes
    for regime in range(3):
        mask = result.y_pred == regime
        ax2.scatter(time_steps[mask], np.ones(np.sum(mask)), c=COLORS[regime], s=1, label=REGIME_NAMES[regime])
    ax2.set_ylim(0.5, 1.5)
    ax2.set_yticks([])
    ax2.set_title("Predicted Regimes", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", ncol=3)
    ax2.grid(True, alpha=0.3, axis="x")

    # Shade disagreement regions
    disagreement = result.y_true != result.y_pred
    for ax in [ax1, ax2]:
        for i in range(T):
            if disagreement[i]:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="red")

    ax2.set_xlabel("Time Step", fontsize=11)
    fig.suptitle("Regime Classification Timeline", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(result: RegimeEvaluationResult, figsize: tuple = (8, 7)) -> plt.Figure:
    """
    Plot confusion matrix heatmap: row-normalized with counts and percentages.
    """
    fig, ax = plt.subplots(figsize=figsize)

    cm = result.classification.confusion_matrix
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Create heatmap
    im = ax.imshow(cm_normalized, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    # Add text annotations
    for i in range(3):
        for j in range(3):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            text = ax.text(j, i, f"{count}\n({pct:.1%})", ha="center", va="center",
                          color="white" if pct > 0.5 else "black", fontsize=11, fontweight="bold")

    # Labels
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([REGIME_NAMES[i] for i in range(3)])
    ax.set_yticklabels([REGIME_NAMES[i] for i in range(3)])
    ax.set_xlabel("Predicted Regime", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Regime", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix (Row-Normalized)", fontsize=14, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fraction", fontsize=11)

    plt.tight_layout()
    return fig


def plot_transition_detection(result: RegimeEvaluationResult, figsize: tuple = (14, 6)) -> plt.Figure:
    """
    Plot transition detection: scatter marking true and predicted transitions.

    Color-coded by matched/unmatched.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract transitions
    true_transitions = []
    for t in range(1, len(result.y_true)):
        if result.y_true[t] != result.y_true[t - 1]:
            true_transitions.append((t, result.y_true[t - 1], result.y_true[t]))

    pred_transitions = []
    for t in range(1, len(result.y_pred)):
        if result.y_pred[t] != result.y_pred[t - 1]:
            pred_transitions.append((t, result.y_pred[t - 1], result.y_pred[t]))

    # Find matched transitions
    matched_pred_indices = set()
    for true_idx, (true_t, true_from, true_to) in enumerate(true_transitions):
        for pred_idx, (pred_t, pred_from, pred_to) in enumerate(pred_transitions):
            if (pred_from == true_from and pred_to == true_to and
                abs(pred_t - true_t) <= 3 and pred_idx not in matched_pred_indices):
                matched_pred_indices.add(pred_idx)
                break

    # Plot true transitions
    true_times = [t for t, _, _ in true_transitions]
    ax.scatter(true_times, np.ones(len(true_times)), marker="^", s=200, c="green",
              label="True Transitions", zorder=3, edgecolors="darkgreen", linewidth=2)

    # Plot predicted transitions (matched vs unmatched)
    matched_times = [pred_transitions[i][0] for i in matched_pred_indices]
    unmatched_times = [pred_transitions[i][0] for i in range(len(pred_transitions)) if i not in matched_pred_indices]

    if matched_times:
        ax.scatter(matched_times, np.ones(len(matched_times)), marker="v", s=200, c="blue",
                  label="Matched Predictions", zorder=3, edgecolors="darkblue", linewidth=2)
    if unmatched_times:
        ax.scatter(unmatched_times, np.ones(len(unmatched_times)), marker="v", s=200, c="red",
                  label="Unmatched Predictions", zorder=3, edgecolors="darkred", linewidth=2)

    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold")
    ax.set_title("Transition Detection (True vs Predicted)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_lag_histogram(result: RegimeEvaluationResult, figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot lag histogram: signed lag distribution with mean and zero reference.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract lags from matched transitions
    lags = []
    true_transitions = []
    for t in range(1, len(result.y_true)):
        if result.y_true[t] != result.y_true[t - 1]:
            true_transitions.append((t, result.y_true[t - 1], result.y_true[t]))

    pred_transitions = []
    for t in range(1, len(result.y_pred)):
        if result.y_pred[t] != result.y_pred[t - 1]:
            pred_transitions.append((t, result.y_pred[t - 1], result.y_pred[t]))

    for true_idx, (true_t, true_from, true_to) in enumerate(true_transitions):
        for pred_idx, (pred_t, pred_from, pred_to) in enumerate(pred_transitions):
            if (pred_from == true_from and pred_to == true_to and
                abs(pred_t - true_t) <= 10):
                lag = pred_t - true_t
                lags.append(lag)
                break

    if lags:
        ax.hist(lags, bins=20, color="#3498DB", edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(lags), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(lags):.2f}")
        ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="Zero (No Lag)")
        ax.set_xlabel("Lag (steps)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title("Prediction Lag Distribution\n(Positive = Late, Negative = Early)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No matched transitions to analyze lag", ha="center", va="center",
               transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    return fig


def plot_regime_conditional_performance(result: RegimeEvaluationResult, figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot regime-conditional performance: grouped bar charts of Sharpe/return/vol per regime.
    """
    if result.trading_impact is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trading impact data available", ha="center", va="center",
               transform=ax.transAxes, fontsize=12)
        return fig

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    regimes = [0, 1, 2]
    x = np.arange(len(regimes))
    width = 0.25

    # Accuracy
    accuracies = [result.trading_impact.per_regime_metrics[r]["accuracy"] for r in regimes]
    axes[0].bar(x, accuracies, width, color=[COLORS[r] for r in regimes], edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_title("Classification Accuracy by Regime", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([REGIME_NAMES[r] for r in regimes])
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Mean Return
    mean_returns = [result.trading_impact.per_regime_metrics[r]["mean_return"] for r in regimes]
    axes[1].bar(x, mean_returns, width, color=[COLORS[r] for r in regimes], edgecolor="black", linewidth=1.5)
    axes[1].set_ylabel("Mean Daily Return", fontsize=11, fontweight="bold")
    axes[1].set_title("Mean Return by Regime", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([REGIME_NAMES[r] for r in regimes])
    axes[1].grid(True, alpha=0.3, axis="y")

    # Sharpe Ratio
    sharpes = [result.trading_impact.per_regime_metrics[r]["sharpe"] for r in regimes]
    axes[2].bar(x, sharpes, width, color=[COLORS[r] for r in regimes], edgecolor="black", linewidth=1.5)
    axes[2].set_ylabel("Sharpe Ratio", fontsize=11, fontweight="bold")
    axes[2].set_title("Sharpe Ratio by Regime", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([REGIME_NAMES[r] for r in regimes])
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Regime-Conditional Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_correct_vs_incorrect_performance(result: RegimeEvaluationResult, figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot correct vs incorrect performance: side-by-side bar comparison.
    """
    if result.trading_impact is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trading impact data available", ha="center", va="center",
               transform=ax.transAxes, fontsize=12)
        return fig

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    x = np.array([0, 1])
    width = 0.35

    # Mean Return
    mean_returns = [result.trading_impact.correct_mean_return, result.trading_impact.incorrect_mean_return]
    axes[0].bar(x, mean_returns, width, color=["#2ECC71", "#E74C3C"], edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("Mean Daily Return", fontsize=11, fontweight="bold")
    axes[0].set_title("Mean Return", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Correct", "Incorrect"])
    axes[0].grid(True, alpha=0.3, axis="y")

    # Volatility
    volatilities = [result.trading_impact.correct_volatility, result.trading_impact.incorrect_volatility]
    axes[1].bar(x, volatilities, width, color=["#2ECC71", "#E74C3C"], edgecolor="black", linewidth=1.5)
    axes[1].set_ylabel("Volatility", fontsize=11, fontweight="bold")
    axes[1].set_title("Volatility", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Correct", "Incorrect"])
    axes[1].grid(True, alpha=0.3, axis="y")

    # Sharpe Ratio
    sharpes = [result.trading_impact.correct_sharpe, result.trading_impact.incorrect_sharpe]
    axes[2].bar(x, sharpes, width, color=["#2ECC71", "#E74C3C"], edgecolor="black", linewidth=1.5)
    axes[2].set_ylabel("Sharpe Ratio", fontsize=11, fontweight="bold")
    axes[2].set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["Correct", "Incorrect"])
    axes[2].grid(True, alpha=0.3, axis="y")

    # Max Drawdown
    max_dds = [result.trading_impact.correct_max_drawdown, result.trading_impact.incorrect_max_drawdown]
    axes[3].bar(x, max_dds, width, color=["#2ECC71", "#E74C3C"], edgecolor="black", linewidth=1.5)
    axes[3].set_ylabel("Max Drawdown", fontsize=11, fontweight="bold")
    axes[3].set_title("Max Drawdown", fontsize=12, fontweight="bold")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(["Correct", "Incorrect"])
    axes[3].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Correct vs Incorrect Predictions Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def generate_evaluation_dashboard(
    result: RegimeEvaluationResult,
    output_dir: Optional[str] = None,
) -> None:
    """
    Generate all 6 evaluation plots and save to output directory.

    Args:
        result: RegimeEvaluationResult from evaluate_regimes()
        output_dir: Directory to save plots. If None, plots are displayed.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("01_regime_timeline.png", plot_regime_timeline(result)),
        ("02_confusion_matrix.png", plot_confusion_matrix(result)),
        ("03_transition_detection.png", plot_transition_detection(result)),
        ("04_lag_histogram.png", plot_lag_histogram(result)),
        ("05_regime_conditional_performance.png", plot_regime_conditional_performance(result)),
        ("06_correct_vs_incorrect_performance.png", plot_correct_vs_incorrect_performance(result)),
    ]

    for filename, fig in plots:
        if output_dir is not None:
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved: {filepath}")
        else:
            plt.show()
        plt.close(fig)

    print(f"\n✓ Dashboard generated with {len(plots)} plots")
