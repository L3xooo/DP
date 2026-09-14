"""
Comprehensive regime classification evaluation framework.

Evaluates regime detection systems across 5 dimensions:
1. Classification Quality (confusion matrix, accuracy, precision, recall, F1, kappa)
2. Transition Detection Accuracy (change-point detection, direction-aware matching)
3. Prediction Lag (signed lag analysis, per-transition-type breakdown)
4. Temporal Stability (transition rates, flickering, regime duration)
5. Trading Impact (performance analysis by regime and error type)

All implementations use numpy only (no sklearn dependency).

Author: Devin (Evaluation Framework)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
from collections import defaultdict


# ============================================================================
# DATACLASSES FOR STRUCTURED RESULTS
# ============================================================================

@dataclass
class ClassificationMetrics:
    """Classification quality metrics."""
    accuracy: float
    confusion_matrix: np.ndarray  # (3, 3) matrix
    per_class_precision: np.ndarray  # (3,)
    per_class_recall: np.ndarray  # (3,)
    per_class_f1: np.ndarray  # (3,)
    macro_f1: float
    weighted_f1: float
    cohens_kappa: float
    class_support: np.ndarray  # (3,) - count of each class in y_true


@dataclass
class TransitionMetrics:
    """Transition detection accuracy metrics."""
    n_true_transitions: int
    n_pred_transitions: int
    n_matched_transitions: int
    transition_precision: float  # matched / predicted
    transition_recall: float  # matched / true
    transition_f1: float
    per_transition_type_metrics: Dict[Tuple[int, int], Dict]  # (from, to) -> {precision, recall, f1, count}


@dataclass
class LagMetrics:
    """Prediction lag analysis."""
    mean_lag: float  # positive = late, negative = early
    median_lag: float
    std_lag: float
    max_lag: float
    min_lag: float
    sign_bias: float  # fraction of positive lags (late predictions)
    per_transition_type_lag: Dict[Tuple[int, int], Dict]  # (from, to) -> {mean, median, std, count}


@dataclass
class TemporalStabilityMetrics:
    """Temporal stability and flickering metrics."""
    true_transition_rate: float  # transitions per time step
    pred_transition_rate: float
    transition_rate_ratio: float  # pred / true (>1 = flickering, <1 = sluggish)
    true_avg_regime_duration: float  # average run length
    pred_avg_regime_duration: float
    per_regime_avg_duration_true: np.ndarray  # (3,)
    per_regime_avg_duration_pred: np.ndarray  # (3,)
    flickering_score: float  # fraction of predicted runs with length <= 2


@dataclass
class TradingImpactMetrics:
    """Trading impact analysis (requires daily returns)."""
    correct_mean_return: float
    incorrect_mean_return: float
    correct_cumulative_return: float
    incorrect_cumulative_return: float
    correct_volatility: float
    incorrect_volatility: float
    correct_sharpe: float
    incorrect_sharpe: float
    correct_max_drawdown: float
    incorrect_max_drawdown: float
    regime_error_cost: float  # mean_return(correct) - mean_return(incorrect)
    per_regime_metrics: Dict[int, Dict]  # regime -> {accuracy, mean_return, volatility, sharpe, max_drawdown}
    per_error_type_metrics: Dict[Tuple[int, int], Dict]  # (true, pred) -> {mean_return, volatility, sharpe, max_drawdown}


@dataclass
class RegimeEvaluationResult:
    """Complete evaluation result."""
    classification: ClassificationMetrics
    transitions: TransitionMetrics
    lag: LagMetrics
    temporal_stability: TemporalStabilityMetrics
    trading_impact: Optional[TradingImpactMetrics] = None
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


def _find_transitions(regimes: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Find regime transitions (change-points).

    Returns:
        List of (time_index, from_regime, to_regime) tuples.
    """
    transitions = []
    for t in range(1, len(regimes)):
        if regimes[t] != regimes[t - 1]:
            transitions.append((t, regimes[t - 1], regimes[t]))
    return transitions


def _match_transitions(
    true_transitions: List[Tuple[int, int, int]],
    pred_transitions: List[Tuple[int, int, int]],
    tolerance: int = 3,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Match predicted transitions to true transitions within tolerance window.

    Args:
        true_transitions: List of (time_index, from, to) for true regimes.
        pred_transitions: List of (time_index, from, to) for predicted regimes.
        tolerance: Maximum time distance for matching (default 3 steps).

    Returns:
        (matched_pairs, n_matched) where matched_pairs is list of (true_idx, pred_idx)
        and n_matched is count of matched transitions.
    """
    matched_pairs = []
    matched_pred_indices = set()

    for true_idx, (true_t, true_from, true_to) in enumerate(true_transitions):
        best_pred_idx = None
        best_distance = float('inf')

        for pred_idx, (pred_t, pred_from, pred_to) in enumerate(pred_transitions):
            if pred_idx in matched_pred_indices:
                continue

            # Check if direction matches and time is within tolerance
            if (pred_from == true_from and pred_to == true_to and
                abs(pred_t - true_t) <= tolerance):
                distance = abs(pred_t - true_t)
                if distance < best_distance:
                    best_distance = distance
                    best_pred_idx = pred_idx

        if best_pred_idx is not None:
            matched_pairs.append((true_idx, best_pred_idx))
            matched_pred_indices.add(best_pred_idx)

    return matched_pairs, len(matched_pairs)


def _compute_run_lengths(regimes: np.ndarray) -> List[int]:
    """Compute lengths of regime runs (consecutive same regime)."""
    run_lengths = []
    current_run = 1
    for t in range(1, len(regimes)):
        if regimes[t] == regimes[t - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    return run_lengths


def _compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe ratio from daily returns."""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0.0
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from daily returns."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_regimes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    daily_returns: Optional[np.ndarray] = None,
    transition_tolerance: int = 3,
    lag_tolerance: int = 10,
) -> RegimeEvaluationResult:
    """
    Comprehensive regime classification evaluation.

    Args:
        y_true: Array of shape (T,) with true regime labels {0, 1, 2}.
        y_pred: Array of shape (T,) with predicted regime labels {0, 1, 2}.
        daily_returns: Optional array of shape (T,) with daily portfolio returns.
        transition_tolerance: Tolerance window for matching transitions (default 3 steps).
        lag_tolerance: Tolerance for lag analysis (default 10 steps).

    Returns:
        RegimeEvaluationResult with all 5 evaluation dimensions.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"
    assert np.all((y_true >= 0) & (y_true <= 2)), "y_true must contain {0, 1, 2}"
    assert np.all((y_pred >= 0) & (y_pred <= 2)), "y_pred must contain {0, 1, 2}"

    # ========================================================================
    # 1. CLASSIFICATION QUALITY
    # ========================================================================
    cm = _compute_confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)

    # Per-class metrics
    per_class_precision = np.zeros(3)
    per_class_recall = np.zeros(3)
    per_class_f1 = np.zeros(3)
    class_support = np.zeros(3, dtype=int)

    for i in range(3):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        class_support[i] = np.sum(y_true == i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_precision[i] = precision
        per_class_recall[i] = recall
        per_class_f1[i] = f1

    # Macro and weighted F1
    macro_f1 = np.mean(per_class_f1)
    weighted_f1 = np.sum(per_class_f1 * class_support) / np.sum(class_support)

    # Cohen's kappa
    po = accuracy
    pe = np.sum((class_support / len(y_true)) ** 2)
    cohens_kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    classification = ClassificationMetrics(
        accuracy=accuracy,
        confusion_matrix=cm,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        cohens_kappa=cohens_kappa,
        class_support=class_support,
    )

    # ========================================================================
    # 2. TRANSITION DETECTION ACCURACY
    # ========================================================================
    true_transitions = _find_transitions(y_true)
    pred_transitions = _find_transitions(y_pred)
    matched_pairs, n_matched = _match_transitions(
        true_transitions, pred_transitions, tolerance=transition_tolerance
    )

    transition_precision = n_matched / len(pred_transitions) if len(pred_transitions) > 0 else 0.0
    transition_recall = n_matched / len(true_transitions) if len(true_transitions) > 0 else 0.0
    transition_f1 = (
        2 * transition_precision * transition_recall / (transition_precision + transition_recall)
        if (transition_precision + transition_recall) > 0 else 0.0
    )

    # Per-transition-type breakdown
    per_transition_type_metrics = defaultdict(lambda: {"count": 0, "matched": 0})
    for true_idx, (true_t, true_from, true_to) in enumerate(true_transitions):
        key = (true_from, true_to)
        per_transition_type_metrics[key]["count"] += 1
        if any(true_idx == t_idx for t_idx, _ in matched_pairs):
            per_transition_type_metrics[key]["matched"] += 1

    for key in per_transition_type_metrics:
        count = per_transition_type_metrics[key]["count"]
        matched = per_transition_type_metrics[key]["matched"]
        per_transition_type_metrics[key]["recall"] = matched / count if count > 0 else 0.0

    transitions = TransitionMetrics(
        n_true_transitions=len(true_transitions),
        n_pred_transitions=len(pred_transitions),
        n_matched_transitions=n_matched,
        transition_precision=transition_precision,
        transition_recall=transition_recall,
        transition_f1=transition_f1,
        per_transition_type_metrics=dict(per_transition_type_metrics),
    )

    # ========================================================================
    # 3. PREDICTION LAG
    # ========================================================================
    lags = []
    per_transition_type_lag = defaultdict(list)

    for true_idx, pred_idx in matched_pairs:
        true_t, true_from, true_to = true_transitions[true_idx]
        pred_t, pred_from, pred_to = pred_transitions[pred_idx]
        lag = pred_t - true_t  # positive = late, negative = early
        lags.append(lag)
        per_transition_type_lag[(true_from, true_to)].append(lag)

    if lags:
        mean_lag = np.mean(lags)
        median_lag = np.median(lags)
        std_lag = np.std(lags)
        max_lag = np.max(lags)
        min_lag = np.min(lags)
        sign_bias = np.sum(np.array(lags) > 0) / len(lags)
    else:
        mean_lag = median_lag = std_lag = max_lag = min_lag = sign_bias = 0.0

    per_transition_type_lag_dict = {}
    for key, lag_list in per_transition_type_lag.items():
        if lag_list:
            per_transition_type_lag_dict[key] = {
                "mean": np.mean(lag_list),
                "median": np.median(lag_list),
                "std": np.std(lag_list),
                "count": len(lag_list),
            }

    lag = LagMetrics(
        mean_lag=mean_lag,
        median_lag=median_lag,
        std_lag=std_lag,
        max_lag=max_lag,
        min_lag=min_lag,
        sign_bias=sign_bias,
        per_transition_type_lag=per_transition_type_lag_dict,
    )

    # ========================================================================
    # 4. TEMPORAL STABILITY
    # ========================================================================
    true_run_lengths = _compute_run_lengths(y_true)
    pred_run_lengths = _compute_run_lengths(y_pred)

    true_transition_rate = len(true_transitions) / len(y_true)
    pred_transition_rate = len(pred_transitions) / len(y_pred)
    transition_rate_ratio = pred_transition_rate / true_transition_rate if true_transition_rate > 0 else 1.0

    true_avg_regime_duration = np.mean(true_run_lengths)
    pred_avg_regime_duration = np.mean(pred_run_lengths)

    # Per-regime average duration
    per_regime_avg_duration_true = np.zeros(3)
    per_regime_avg_duration_pred = np.zeros(3)
    regime_run_count_true = [0] * 3
    regime_run_count_pred = [0] * 3

    current_regime = y_true[0]
    current_run = 1
    for t in range(1, len(y_true)):
        if y_true[t] == current_regime:
            current_run += 1
        else:
            per_regime_avg_duration_true[current_regime] += current_run
            regime_run_count_true[current_regime] += 1
            current_regime = y_true[t]
            current_run = 1
    per_regime_avg_duration_true[current_regime] += current_run
    regime_run_count_true[current_regime] += 1

    for i in range(3):
        if regime_run_count_true[i] > 0:
            per_regime_avg_duration_true[i] /= regime_run_count_true[i]

    current_regime = y_pred[0]
    current_run = 1
    for t in range(1, len(y_pred)):
        if y_pred[t] == current_regime:
            current_run += 1
        else:
            per_regime_avg_duration_pred[current_regime] += current_run
            regime_run_count_pred[current_regime] += 1
            current_regime = y_pred[t]
            current_run = 1
    per_regime_avg_duration_pred[current_regime] += current_run
    regime_run_count_pred[current_regime] += 1

    for i in range(3):
        if regime_run_count_pred[i] > 0:
            per_regime_avg_duration_pred[i] /= regime_run_count_pred[i]

    # Flickering score: fraction of runs with length <= 2
    flickering_score = np.sum(np.array(pred_run_lengths) <= 2) / len(pred_run_lengths) if pred_run_lengths else 0.0

    temporal_stability = TemporalStabilityMetrics(
        true_transition_rate=true_transition_rate,
        pred_transition_rate=pred_transition_rate,
        transition_rate_ratio=transition_rate_ratio,
        true_avg_regime_duration=true_avg_regime_duration,
        pred_avg_regime_duration=pred_avg_regime_duration,
        per_regime_avg_duration_true=per_regime_avg_duration_true,
        per_regime_avg_duration_pred=per_regime_avg_duration_pred,
        flickering_score=flickering_score,
    )

    # ========================================================================
    # 5. TRADING IMPACT (optional)
    # ========================================================================
    trading_impact = None
    if daily_returns is not None:
        assert len(daily_returns) == len(y_true), "daily_returns must have same length as y_true"

        # Segment by correct vs incorrect
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask

        correct_returns = daily_returns[correct_mask]
        incorrect_returns = daily_returns[incorrect_mask]

        correct_mean_return = np.mean(correct_returns) if len(correct_returns) > 0 else 0.0
        incorrect_mean_return = np.mean(incorrect_returns) if len(incorrect_returns) > 0 else 0.0
        correct_cumulative_return = np.sum(correct_returns) if len(correct_returns) > 0 else 0.0
        incorrect_cumulative_return = np.sum(incorrect_returns) if len(incorrect_returns) > 0 else 0.0
        correct_volatility = np.std(correct_returns) if len(correct_returns) > 0 else 0.0
        incorrect_volatility = np.std(incorrect_returns) if len(incorrect_returns) > 0 else 0.0
        correct_sharpe = _compute_sharpe_ratio(correct_returns) if len(correct_returns) > 0 else 0.0
        incorrect_sharpe = _compute_sharpe_ratio(incorrect_returns) if len(incorrect_returns) > 0 else 0.0
        correct_max_drawdown = _compute_max_drawdown(correct_returns) if len(correct_returns) > 0 else 0.0
        incorrect_max_drawdown = _compute_max_drawdown(incorrect_returns) if len(incorrect_returns) > 0 else 0.0

        regime_error_cost = correct_mean_return - incorrect_mean_return

        # Per-regime metrics
        per_regime_metrics = {}
        for regime in range(3):
            regime_mask = y_true == regime
            regime_correct_mask = regime_mask & correct_mask
            regime_returns = daily_returns[regime_mask]
            regime_correct_returns = daily_returns[regime_correct_mask]

            accuracy_in_regime = np.sum(regime_correct_mask) / np.sum(regime_mask) if np.sum(regime_mask) > 0 else 0.0
            mean_return = np.mean(regime_returns) if len(regime_returns) > 0 else 0.0
            volatility = np.std(regime_returns) if len(regime_returns) > 0 else 0.0
            sharpe = _compute_sharpe_ratio(regime_returns) if len(regime_returns) > 0 else 0.0
            max_drawdown = _compute_max_drawdown(regime_returns) if len(regime_returns) > 0 else 0.0

            per_regime_metrics[regime] = {
                "accuracy": accuracy_in_regime,
                "mean_return": mean_return,
                "volatility": volatility,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
            }

        # Per-error-type metrics (true_regime, pred_regime)
        per_error_type_metrics = {}
        for true_regime in range(3):
            for pred_regime in range(3):
                error_mask = (y_true == true_regime) & (y_pred == pred_regime)
                error_returns = daily_returns[error_mask]

                if len(error_returns) > 0:
                    per_error_type_metrics[(true_regime, pred_regime)] = {
                        "mean_return": np.mean(error_returns),
                        "volatility": np.std(error_returns),
                        "sharpe": _compute_sharpe_ratio(error_returns),
                        "max_drawdown": _compute_max_drawdown(error_returns),
                        "count": len(error_returns),
                    }

        trading_impact = TradingImpactMetrics(
            correct_mean_return=correct_mean_return,
            incorrect_mean_return=incorrect_mean_return,
            correct_cumulative_return=correct_cumulative_return,
            incorrect_cumulative_return=incorrect_cumulative_return,
            correct_volatility=correct_volatility,
            incorrect_volatility=incorrect_volatility,
            correct_sharpe=correct_sharpe,
            incorrect_sharpe=incorrect_sharpe,
            correct_max_drawdown=correct_max_drawdown,
            incorrect_max_drawdown=incorrect_max_drawdown,
            regime_error_cost=regime_error_cost,
            per_regime_metrics=per_regime_metrics,
            per_error_type_metrics=per_error_type_metrics,
        )

    return RegimeEvaluationResult(
        classification=classification,
        transitions=transitions,
        lag=lag,
        temporal_stability=temporal_stability,
        trading_impact=trading_impact,
        y_true=y_true,
        y_pred=y_pred,
    )


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_regime_evaluation(result: RegimeEvaluationResult) -> None:
    """Print formatted evaluation report to console."""
    regime_names = {0: "Bullish", 1: "Neutral", 2: "Bearish"}

    print("\n" + "=" * 80)
    print("REGIME CLASSIFICATION EVALUATION REPORT")
    print("=" * 80)

    # Classification Quality
    print("\n1. CLASSIFICATION QUALITY")
    print("-" * 80)
    print(f"Overall Accuracy: {result.classification.accuracy:.4f}")
    print(f"Cohen's Kappa: {result.classification.cohens_kappa:.4f}")
    print(f"Macro-averaged F1: {result.classification.macro_f1:.4f}")
    print(f"Weighted-averaged F1: {result.classification.weighted_f1:.4f}")

    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Bullish  Neutral  Bearish")
    for i in range(3):
        print(f"True {regime_names[i]:8s}", end="")
        for j in range(3):
            print(f"  {result.classification.confusion_matrix[i, j]:6d}", end="")
        print()

    print("\nPer-Class Metrics:")
    print(f"{'Regime':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    for i in range(3):
        print(f"{regime_names[i]:<12} {result.classification.per_class_precision[i]:<12.4f} "
              f"{result.classification.per_class_recall[i]:<12.4f} "
              f"{result.classification.per_class_f1[i]:<12.4f} "
              f"{result.classification.class_support[i]:<12d}")

    # Transition Detection
    print("\n2. TRANSITION DETECTION ACCURACY")
    print("-" * 80)
    print(f"True Transitions: {result.transitions.n_true_transitions}")
    print(f"Predicted Transitions: {result.transitions.n_pred_transitions}")
    print(f"Matched Transitions: {result.transitions.n_matched_transitions}")
    print(f"Transition Precision: {result.transitions.transition_precision:.4f}")
    print(f"Transition Recall: {result.transitions.transition_recall:.4f}")
    print(f"Transition F1: {result.transitions.transition_f1:.4f}")

    if result.transitions.per_transition_type_metrics:
        print("\nPer-Transition-Type Recall:")
        for (from_regime, to_regime), metrics in sorted(result.transitions.per_transition_type_metrics.items()):
            print(f"  {regime_names[from_regime]} -> {regime_names[to_regime]}: "
                  f"recall={metrics['recall']:.4f} (n={metrics['count']})")

    # Lag Analysis
    print("\n3. PREDICTION LAG")
    print("-" * 80)
    print(f"Mean Lag: {result.lag.mean_lag:.2f} steps")
    print(f"Median Lag: {result.lag.median_lag:.2f} steps")
    print(f"Std Lag: {result.lag.std_lag:.2f} steps")
    print(f"Max Lag: {result.lag.max_lag:.2f} steps")
    print(f"Min Lag: {result.lag.min_lag:.2f} steps")
    print(f"Sign Bias (fraction late): {result.lag.sign_bias:.4f}")
    print("  (Positive lag = prediction is late, Negative = prediction is early)")

    if result.lag.per_transition_type_lag:
        print("\nPer-Transition-Type Lag:")
        for (from_regime, to_regime), lag_stats in sorted(result.lag.per_transition_type_lag.items()):
            print(f"  {regime_names[from_regime]} -> {regime_names[to_regime]}: "
                  f"mean={lag_stats['mean']:.2f}, median={lag_stats['median']:.2f}, "
                  f"std={lag_stats['std']:.2f} (n={lag_stats['count']})")

    # Temporal Stability
    print("\n4. TEMPORAL STABILITY")
    print("-" * 80)
    print(f"True Transition Rate: {result.temporal_stability.true_transition_rate:.4f} per step")
    print(f"Predicted Transition Rate: {result.temporal_stability.pred_transition_rate:.4f} per step")
    print(f"Transition Rate Ratio (pred/true): {result.temporal_stability.transition_rate_ratio:.4f}")
    if result.temporal_stability.transition_rate_ratio > 1.1:
        print("  ⚠️  Flickering detected (too many transitions)")
    elif result.temporal_stability.transition_rate_ratio < 0.9:
        print("  ⚠️  Sluggish detected (too few transitions)")
    else:
        print("  ✓ Transition rate is well-balanced")

    print(f"\nTrue Avg Regime Duration: {result.temporal_stability.true_avg_regime_duration:.2f} steps")
    print(f"Predicted Avg Regime Duration: {result.temporal_stability.pred_avg_regime_duration:.2f} steps")

    print("\nPer-Regime Average Duration:")
    print(f"{'Regime':<12} {'True':<12} {'Predicted':<12}")
    for i in range(3):
        print(f"{regime_names[i]:<12} {result.temporal_stability.per_regime_avg_duration_true[i]:<12.2f} "
              f"{result.temporal_stability.per_regime_avg_duration_pred[i]:<12.2f}")

    print(f"\nFlickering Score: {result.temporal_stability.flickering_score:.4f}")
    print("  (Fraction of predicted runs with length <= 2)")

    # Trading Impact
    if result.trading_impact is not None:
        print("\n5. TRADING IMPACT")
        print("-" * 80)
        print(f"Correct Predictions Mean Return: {result.trading_impact.correct_mean_return:.6f}")
        print(f"Incorrect Predictions Mean Return: {result.trading_impact.incorrect_mean_return:.6f}")
        print(f"Regime Error Cost: {result.trading_impact.regime_error_cost:.6f}")

        print(f"\nCorrect Predictions Cumulative Return: {result.trading_impact.correct_cumulative_return:.6f}")
        print(f"Incorrect Predictions Cumulative Return: {result.trading_impact.incorrect_cumulative_return:.6f}")

        print(f"\nCorrect Predictions Volatility: {result.trading_impact.correct_volatility:.6f}")
        print(f"Incorrect Predictions Volatility: {result.trading_impact.incorrect_volatility:.6f}")

        print(f"\nCorrect Predictions Sharpe Ratio: {result.trading_impact.correct_sharpe:.4f}")
        print(f"Incorrect Predictions Sharpe Ratio: {result.trading_impact.incorrect_sharpe:.4f}")

        print(f"\nCorrect Predictions Max Drawdown: {result.trading_impact.correct_max_drawdown:.6f}")
        print(f"Incorrect Predictions Max Drawdown: {result.trading_impact.incorrect_max_drawdown:.6f}")

        print("\nPer-Regime Performance:")
        print(f"{'Regime':<12} {'Accuracy':<12} {'Mean Return':<15} {'Volatility':<12} {'Sharpe':<12} {'Max DD':<12}")
        for regime in range(3):
            metrics = result.trading_impact.per_regime_metrics[regime]
            print(f"{regime_names[regime]:<12} {metrics['accuracy']:<12.4f} "
                  f"{metrics['mean_return']:<15.6f} {metrics['volatility']:<12.6f} "
                  f"{metrics['sharpe']:<12.4f} {metrics['max_drawdown']:<12.6f}")

    print("\n" + "=" * 80)
