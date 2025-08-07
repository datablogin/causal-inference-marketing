"""Enhanced evaluation metrics for heterogeneous treatment effect estimation.

This module provides specialized metrics for evaluating CATE estimators including
PEHE (Precision in Estimation of Heterogeneous Effect), policy value metrics,
and overlap-weighted evaluation procedures.

These metrics are essential for proper evaluation of HTE methods, especially
when true treatment effects are unknown or vary across subpopulations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

__all__ = [
    "pehe_score",
    "policy_value",
    "overlap_weighted_mse",
    "ate_preservation_score",
    "cate_r2_score",
    "rank_weighted_ate",
    "qini_score",
    "calibration_score",
    "HTEEvaluator",
]


def pehe_score(
    y_true: NDArray[Any], y_pred: NDArray[Any], squared: bool = True
) -> float:
    """Compute Precision in Estimation of Heterogeneous Effect (PEHE).

    PEHE measures the mean squared error between predicted and true
    individual treatment effects. Lower values indicate better performance.

    Args:
        y_true: True individual treatment effects
        y_pred: Predicted individual treatment effects
        squared: If True, return PEHE. If False, return sqrt(PEHE)

    Returns:
        PEHE score (lower is better)

    References:
        Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference.
        Journal of Computational and Graphical Statistics, 20(1), 217-240.
    """
    mse = mean_squared_error(y_true, y_pred)
    return mse if squared else np.sqrt(mse)


def policy_value(
    outcomes: NDArray[Any],
    treatments: NDArray[Any],
    cate_estimates: NDArray[Any],
    policy_threshold: float = 0.0,
) -> float:
    """Compute expected outcome under learned treatment assignment policy.

    Evaluates the value of a policy that treats individuals when their
    predicted CATE exceeds a threshold. This provides an economic
    interpretation of model performance.

    Args:
        outcomes: Observed outcomes
        treatments: Actual treatment assignments
        cate_estimates: Predicted individual treatment effects
        policy_threshold: Threshold for treatment assignment (default 0)

    Returns:
        Expected outcome under the learned policy

    Note:
        This uses inverse propensity weighting for unbiased estimation.
        Requires overlap in treatment assignment.
    """
    # Estimate propensity scores (simplified approach)
    propensity = np.mean(treatments)

    # Policy assigns treatment when CATE > threshold
    policy_treatment = (cate_estimates > policy_threshold).astype(int)

    # Compute policy value using IPW
    # V(π) = E[Y(1) * π(X) + Y(0) * (1-π(X))]
    # Using IPW: sum over i of [Y_i * I(T_i = π(X_i)) / P(T_i | X_i)]

    weights = np.where(
        treatments == policy_treatment,
        1.0 / np.where(policy_treatment == 1, propensity, 1 - propensity),
        0.0,
    )

    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.mean(weights)
        policy_value_est = np.mean(weights * outcomes)
    else:
        # Fallback if no aligned observations
        policy_value_est = np.mean(outcomes)

    return float(policy_value_est)


def overlap_weighted_mse(
    y_true: NDArray[Any], y_pred: NDArray[Any], propensity_scores: NDArray[Any]
) -> float:
    """Compute overlap-weighted mean squared error.

    Weights observations by their propensity score overlap to focus
    evaluation on regions with good covariate balance.

    Args:
        y_true: True treatment effects
        y_pred: Predicted treatment effects
        propensity_scores: Propensity scores for observations

    Returns:
        Overlap-weighted MSE
    """
    # Overlap weights emphasize observations near e(X) = 0.5
    overlap_weights = 4 * propensity_scores * (1 - propensity_scores)

    # Compute weighted MSE
    squared_errors = (y_true - y_pred) ** 2
    weighted_mse = np.average(squared_errors, weights=overlap_weights)

    return float(weighted_mse)


def ate_preservation_score(true_ate: float, cate_estimates: NDArray[Any]) -> float:
    """Measure how well CATE estimates preserve the true ATE.

    Args:
        true_ate: True average treatment effect
        cate_estimates: Individual treatment effect estimates

    Returns:
        Absolute difference between true ATE and mean CATE
    """
    estimated_ate = np.mean(cate_estimates)
    return float(np.abs(true_ate - estimated_ate))


def cate_r2_score(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
    """Compute R² score for CATE predictions.

    Measures how much of the treatment effect heterogeneity
    is explained by the model.

    Args:
        y_true: True individual treatment effects
        y_pred: Predicted individual treatment effects

    Returns:
        R² score for treatment effects
    """
    return float(r2_score(y_true, y_pred))


def rank_weighted_ate(
    outcomes: NDArray[Any],
    treatments: NDArray[Any],
    cate_estimates: NDArray[Any],
    top_k_fraction: float = 0.1,
) -> float:
    """Compute ATE among top-k individuals ranked by predicted CATE.

    This metric evaluates whether the model correctly identifies
    individuals who benefit most from treatment.

    Args:
        outcomes: Observed outcomes
        treatments: Treatment assignments
        cate_estimates: Predicted treatment effects
        top_k_fraction: Fraction of top-ranked individuals to consider

    Returns:
        ATE among top-k ranked individuals
    """
    n = len(cate_estimates)
    k = int(n * top_k_fraction)

    # Get indices of top-k individuals
    top_k_indices = np.argsort(cate_estimates)[-k:]

    # Compute ATE among top-k
    y_top_k = outcomes[top_k_indices]
    t_top_k = treatments[top_k_indices]

    treated_mask = t_top_k == 1
    control_mask = t_top_k == 0

    if not (np.sum(treated_mask) > 0 and np.sum(control_mask) > 0):
        return 0.0  # Cannot compute ATE without both groups

    ate_top_k = np.mean(y_top_k[treated_mask]) - np.mean(y_top_k[control_mask])

    return float(ate_top_k)


def qini_score(
    outcomes: NDArray[Any],
    treatments: NDArray[Any],
    cate_estimates: NDArray[Any],
    n_bins: int = 10,
) -> float:
    """Compute Qini score for treatment effect ranking.

    The Qini score measures the cumulative gain from treating
    individuals in order of predicted treatment effect benefit.

    Args:
        outcomes: Observed outcomes
        treatments: Treatment assignments
        cate_estimates: Predicted treatment effects
        n_bins: Number of bins for Qini curve computation

    Returns:
        Qini score (area under Qini curve)

    References:
        Radcliffe, N. J., & Surry, P. D. (2011). Real-world uplift modelling
        with significance-based uplift trees. Portrait Technical Report TR-2011-1.
    """
    n = len(cate_estimates)

    # Sort by predicted CATE (descending)
    sort_idx = np.argsort(-cate_estimates)
    sorted_outcomes = outcomes[sort_idx]
    sorted_treatments = treatments[sort_idx]

    # Compute cumulative gains
    bin_size = n // n_bins
    qini_values = []

    for i in range(1, n_bins + 1):
        end_idx = min(i * bin_size, n)

        # Outcomes and treatments in first i bins
        y_bin = sorted_outcomes[:end_idx]
        t_bin = sorted_treatments[:end_idx]

        # Compute gain (treated - control outcomes)
        treated_mask = t_bin == 1
        control_mask = t_bin == 0

        if np.sum(treated_mask) > 0 and np.sum(control_mask) > 0:
            gain = np.sum(y_bin[treated_mask]) - np.sum(y_bin[control_mask]) * np.sum(
                treated_mask
            ) / np.sum(control_mask)
        else:
            gain = 0.0

        qini_values.append(gain)

    # Area under Qini curve (trapezoidal rule)
    qini_auc = np.trapz(qini_values, dx=1.0 / n_bins)

    return float(qini_auc)


def calibration_score(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    confidence_intervals: NDArray[Any] | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute calibration score for CATE predictions.

    Measures whether predicted treatment effects are well-calibrated
    by checking if confidence intervals contain true effects at
    the expected rate.

    Args:
        y_true: True treatment effects
        y_pred: Predicted treatment effects
        confidence_intervals: Confidence intervals (n_samples, 2)
        n_bins: Number of bins for calibration analysis

    Returns:
        Dictionary with calibration metrics
    """
    results = {}

    # Binned calibration for point predictions
    bin_boundaries = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_boundaries) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means_pred = []
    bin_means_true = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_means_pred.append(np.mean(y_pred[mask]))
            bin_means_true.append(np.mean(y_true[mask]))

    if len(bin_means_pred) > 0:
        calibration_error = np.mean(
            np.abs(np.array(bin_means_pred) - np.array(bin_means_true))
        )
        results["calibration_error"] = float(calibration_error)
    else:
        results["calibration_error"] = np.nan

    # Coverage probability for confidence intervals
    if confidence_intervals is not None:
        in_interval = (y_true >= confidence_intervals[:, 0]) & (
            y_true <= confidence_intervals[:, 1]
        )
        coverage = np.mean(in_interval)
        results["coverage_probability"] = float(coverage)

        # Interval width
        interval_width = confidence_intervals[:, 1] - confidence_intervals[:, 0]
        results["mean_interval_width"] = float(np.mean(interval_width))

    return results


class HTEEvaluator:
    """Comprehensive evaluator for heterogeneous treatment effect estimators.

    Provides a unified interface for computing multiple evaluation metrics
    and generating diagnostic plots for CATE estimation performance.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 100,
        random_state: int | None = None,
    ):
        """Initialize HTE evaluator.

        Args:
            confidence_level: Confidence level for intervals
            n_bootstrap: Number of bootstrap samples for CI estimation
            random_state: Random seed
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def evaluate(
        self,
        y_true: NDArray[Any],
        y_pred: NDArray[Any],
        outcomes: NDArray[Any] | None = None,
        treatments: NDArray[Any] | None = None,
        propensity_scores: NDArray[Any] | None = None,
        confidence_intervals: NDArray[Any] | None = None,
        true_ate: float | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive evaluation metrics.

        Args:
            y_true: True individual treatment effects
            y_pred: Predicted individual treatment effects
            outcomes: Observed outcomes (for policy value)
            treatments: Treatment assignments (for policy value)
            propensity_scores: Propensity scores (for overlap weighting)
            confidence_intervals: Prediction confidence intervals
            true_ate: True average treatment effect

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Core CATE evaluation metrics
        metrics["pehe"] = pehe_score(y_true, y_pred)
        metrics["pehe_sqrt"] = pehe_score(y_true, y_pred, squared=False)
        metrics["cate_r2"] = cate_r2_score(y_true, y_pred)
        metrics["cate_mse"] = mean_squared_error(y_true, y_pred)

        # ATE preservation
        if true_ate is not None:
            metrics["ate_preservation"] = ate_preservation_score(true_ate, y_pred)

        # Overlap-weighted evaluation
        if propensity_scores is not None:
            metrics["overlap_weighted_mse"] = overlap_weighted_mse(
                y_true, y_pred, propensity_scores
            )

        # Policy-based evaluation
        if outcomes is not None and treatments is not None:
            metrics["policy_value"] = policy_value(outcomes, treatments, y_pred)
            metrics["rank_weighted_ate"] = rank_weighted_ate(
                outcomes, treatments, y_pred
            )
            metrics["qini_score"] = qini_score(outcomes, treatments, y_pred)

        # Calibration analysis
        calibration_metrics = calibration_score(y_true, y_pred, confidence_intervals)
        metrics.update(calibration_metrics)

        # Distributional metrics
        metrics["cate_variance_ratio"] = float(
            np.var(y_pred) / np.var(y_true) if np.var(y_true) > 0 else np.nan
        )

        # Rank correlation
        rank_corr, _ = stats.spearmanr(y_true, y_pred)
        metrics["rank_correlation"] = (
            float(rank_corr) if not np.isnan(rank_corr) else 0.0
        )

        return metrics

    def bootstrap_ci(
        self,
        y_true: NDArray[Any],
        y_pred: NDArray[Any],
        metric_func: Any,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for a metric.

        Args:
            y_true: True values
            y_pred: Predicted values
            metric_func: Function computing the metric
            alpha: Significance level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        rng = np.random.RandomState(self.random_state)
        n = len(y_true)

        bootstrap_scores = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            score = metric_func(y_true[idx], y_pred[idx])
            bootstrap_scores.append(score)

        lower = np.percentile(bootstrap_scores, (alpha / 2) * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

        return float(lower), float(upper)

    def plot_cate_scatter(
        self,
        y_true: NDArray[Any],
        y_pred: NDArray[Any],
        confidence_intervals: NDArray[Any] | None = None,
        ax: Any | None = None,
    ) -> Any:
        """Create scatter plot of true vs predicted CATE.

        Args:
            y_true: True treatment effects
            y_pred: Predicted treatment effects
            confidence_intervals: Optional confidence intervals
            ax: Matplotlib axis

        Returns:
            Matplotlib axis object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=30)

        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            label="Perfect prediction",
            linewidth=2,
        )

        # Error bars if confidence intervals provided
        if confidence_intervals is not None:
            errors = np.abs(confidence_intervals.T - y_pred)
            ax.errorbar(
                y_true, y_pred, yerr=errors, fmt="none", alpha=0.3, color="gray"
            )

        ax.set_xlabel("True Treatment Effect")
        ax.set_ylabel("Predicted Treatment Effect")
        ax.set_title("CATE Predictions vs Truth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add R² to plot
        r2 = cate_r2_score(y_true, y_pred)
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        return ax

