"""Overlap and positivity diagnostics for causal inference.

This module implements tools to assess the overlap assumption (positivity),
which requires that all units have a non-zero probability of receiving
each treatment level conditional on covariates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ..core.base import CovariateData, TreatmentData


@dataclass
class OverlapResults:
    """Results from overlap and positivity assessment."""

    overall_positivity_met: bool
    min_propensity_score: float
    max_propensity_score: float
    propensity_scores: NDArray[Any]
    violations: list[dict[str, Any]]
    common_support_range: tuple[float, float]
    units_in_common_support: int
    total_units: int
    propensity_model_auc: float | None
    extreme_weights_count: int
    recommendation: str
    calibration_results: dict[str, Any] | None = None
    trimming_recommendations: dict[str, dict[str, Any]] | None = None
    roc_curve_data: dict[str, Any] | None = None


def calculate_propensity_scores(
    treatment: TreatmentData,
    covariates: CovariateData,
    model_type: str = "logistic",
    cross_validate: bool = True,
) -> NDArray[Any]:
    """Calculate propensity scores using specified model.

    Args:
        treatment: Treatment assignment data
        covariates: Covariate data
        model_type: Type of model ("logistic" or "random_forest")
        cross_validate: Whether to use cross-validation to avoid overfitting

    Returns:
        Array of propensity scores
    """
    if not isinstance(covariates.values, pd.DataFrame):
        raise ValueError(
            "Covariates must be a DataFrame for propensity score calculation"
        )

    # Prepare data
    X = covariates.values.fillna(covariates.values.mean())  # Simple imputation
    y = np.asarray(treatment.values)

    # Check for degenerate cases
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        # All units have same treatment - return constant probabilities
        if unique_classes[0] == 1:
            return np.ones(len(y)) * 0.99  # All treated
        else:
            return np.ones(len(y)) * 0.01  # All control

    # Select model
    if model_type == "logistic":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Fit and predict
    if cross_validate:
        # Use cross-validation to get out-of-fold predictions
        probas = cross_val_predict(model, X, y, cv=5, method="predict_proba")
        if probas.shape[1] == 2:
            propensity_scores = probas[:, 1]  # Probability of treatment
        else:
            propensity_scores = probas[:, 0]  # Single class case
    else:
        model.fit(X, y)
        probas = model.predict_proba(X)
        if probas.shape[1] == 2:
            propensity_scores = probas[:, 1]
        else:
            propensity_scores = probas[:, 0]

    return np.asarray(propensity_scores)


def check_common_support(
    propensity_scores: NDArray[Any],
    treatment_values: NDArray[Any],
    overlap_threshold: float = 0.1,
) -> tuple[tuple[float, float], int]:
    """Check common support region for propensity scores.

    Args:
        propensity_scores: Propensity score values
        treatment_values: Treatment assignment values
        overlap_threshold: Minimum overlap required at boundaries

    Returns:
        Tuple of (common_support_range, units_in_support)
    """
    treatment_vals = np.asarray(treatment_values)

    # Split propensity scores by treatment group
    treated_ps = propensity_scores[treatment_vals == 1]
    control_ps = propensity_scores[treatment_vals == 0]

    if len(treated_ps) == 0 or len(control_ps) == 0:
        return (0.0, 1.0), 0

    # Find overlapping region
    min_treated = np.min(treated_ps)
    max_treated = np.max(treated_ps)
    min_control = np.min(control_ps)
    max_control = np.max(control_ps)

    # Common support is the intersection
    common_min = max(min_treated, min_control)
    common_max = min(max_treated, max_control)

    # Count units in common support
    in_support = (propensity_scores >= common_min) & (propensity_scores <= common_max)
    units_in_support = int(np.sum(in_support))

    return (float(common_min), float(common_max)), units_in_support


class OverlapDiagnostics:
    """Comprehensive overlap and positivity assessment tools."""

    def __init__(
        self,
        min_propensity: float = 0.01,
        max_propensity: float = 0.99,
        extreme_weight_threshold: float = 10.0,
        overlap_bins: int = 20,
    ):
        """Initialize overlap diagnostics.

        Args:
            min_propensity: Minimum acceptable propensity score
            max_propensity: Maximum acceptable propensity score
            extreme_weight_threshold: Threshold for flagging extreme weights
            overlap_bins: Number of bins for overlap visualization
        """
        self.min_propensity = min_propensity
        self.max_propensity = max_propensity
        self.extreme_weight_threshold = extreme_weight_threshold
        self.overlap_bins = overlap_bins

    def assess_overlap(
        self,
        treatment: TreatmentData,
        covariates: CovariateData,
        propensity_model: str = "logistic",
    ) -> OverlapResults:
        """Comprehensive overlap and positivity assessment.

        Args:
            treatment: Treatment assignment data
            covariates: Covariate data
            propensity_model: Model for propensity score estimation

        Returns:
            OverlapResults object with comprehensive assessment
        """
        # Calculate propensity scores
        propensity_scores = calculate_propensity_scores(
            treatment, covariates, model_type=propensity_model
        )

        # Basic statistics
        min_ps = float(np.min(propensity_scores))
        max_ps = float(np.max(propensity_scores))
        treatment_vals = np.asarray(treatment.values)

        # Check positivity violations
        violations = []
        positivity_met = True

        if min_ps < self.min_propensity:
            positivity_met = False
            violations.append(
                {
                    "type": "low_propensity",
                    "description": f"Minimum propensity score ({min_ps:.4f}) below threshold ({self.min_propensity})",
                    "severity": "high" if min_ps < 0.001 else "medium",
                    "affected_units": int(
                        np.sum(propensity_scores < self.min_propensity)
                    ),
                }
            )

        if max_ps > self.max_propensity:
            positivity_met = False
            violations.append(
                {
                    "type": "high_propensity",
                    "description": f"Maximum propensity score ({max_ps:.4f}) above threshold ({self.max_propensity})",
                    "severity": "high" if max_ps > 0.999 else "medium",
                    "affected_units": int(
                        np.sum(propensity_scores > self.max_propensity)
                    ),
                }
            )

        # Check common support
        common_support_range, units_in_support = check_common_support(
            propensity_scores, treatment_vals
        )

        if units_in_support < 0.8 * len(treatment_vals):
            violations.append(
                {
                    "type": "limited_common_support",
                    "description": f"Only {units_in_support}/{len(treatment_vals)} units in common support",
                    "severity": "medium",
                    "affected_units": len(treatment_vals) - units_in_support,
                }
            )

        # Check for extreme inverse probability weights
        # IPW weights = 1/P(T=t|X) for each unit
        weights_treated = 1.0 / propensity_scores[treatment_vals == 1]
        weights_control = 1.0 / (1.0 - propensity_scores[treatment_vals == 0])
        all_weights = np.concatenate([weights_treated, weights_control])

        extreme_weights = np.sum(all_weights > self.extreme_weight_threshold)

        if extreme_weights > 0:
            violations.append(
                {
                    "type": "extreme_weights",
                    "description": f"{extreme_weights} units have IPW weights > {self.extreme_weight_threshold}",
                    "severity": "medium"
                    if extreme_weights < 0.05 * len(treatment_vals)
                    else "high",
                    "affected_units": int(extreme_weights),
                }
            )

        # Calculate model performance (if applicable)
        try:
            from sklearn.metrics import roc_auc_score

            # Check if we can calculate AUC (need at least 2 classes)
            if len(np.unique(treatment_vals)) >= 2:
                auc_score = roc_auc_score(treatment_vals, propensity_scores)
            else:
                auc_score = None  # Cannot calculate AUC with single class
        except ImportError:
            auc_score = None

        # Calculate calibration metrics
        calibration_results = calculate_calibration_metrics(
            treatment_vals, propensity_scores
        )

        # Generate trimming recommendations
        trimming_recommendations = suggest_trimming_thresholds(
            propensity_scores, treatment_vals
        )

        # ROC curve data (if AUC was calculated)
        roc_curve_data = None
        if auc_score is not None:
            try:
                from sklearn.metrics import roc_curve

                fpr, tpr, _ = roc_curve(treatment_vals, propensity_scores)
                roc_curve_data = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": auc_score,
                }
            except ImportError:
                pass

        # Generate recommendation
        recommendation = self._generate_recommendation(
            violations, positivity_met, auc_score
        )

        return OverlapResults(
            overall_positivity_met=positivity_met,
            min_propensity_score=min_ps,
            max_propensity_score=max_ps,
            propensity_scores=propensity_scores,
            violations=violations,
            common_support_range=common_support_range,
            units_in_common_support=units_in_support,
            total_units=len(treatment_vals),
            propensity_model_auc=auc_score,
            extreme_weights_count=int(extreme_weights),
            recommendation=recommendation,
            calibration_results=calibration_results,
            trimming_recommendations=trimming_recommendations,
            roc_curve_data=roc_curve_data,
        )

    def _generate_recommendation(
        self,
        violations: list[dict[str, Any]],
        positivity_met: bool,
        auc_score: float | None,
    ) -> str:
        """Generate recommendation based on overlap assessment."""
        if positivity_met and len(violations) == 0:
            return "✅ Overlap assumption satisfied. Proceed with causal analysis."

        recommendations = []

        # Check for severe violations
        severe_violations = [v for v in violations if v.get("severity") == "high"]
        if severe_violations:
            recommendations.append("⚠️ Severe overlap violations detected.")
            recommendations.append(
                "Consider: trimming extreme propensity scores, matching, or subclassification."
            )

        # Check propensity model performance
        if auc_score is not None and auc_score > 0.8:
            recommendations.append(
                "⚠️ High propensity model AUC suggests strong confounding."
            )
            recommendations.append(
                "Consider: more flexible models or additional covariates."
            )

        # General recommendations based on violation types
        violation_types = {v["type"] for v in violations}

        if "extreme_weights" in violation_types:
            recommendations.append(
                "Consider: weight trimming or alternative estimators (matching, stratification)."
            )

        if "limited_common_support" in violation_types:
            recommendations.append(
                "Consider: restricting analysis to common support region."
            )

        if not recommendations:
            recommendations.append(
                "Minor overlap issues detected. Consider robustness checks."
            )

        return " ".join(recommendations)

    def print_overlap_summary(self, overlap_results: OverlapResults) -> None:
        """Print a summary of overlap assessment results."""
        print("=== Overlap and Positivity Assessment ===")
        print(
            f"Propensity score range: [{overlap_results.min_propensity_score:.4f}, "
            f"{overlap_results.max_propensity_score:.4f}]"
        )
        print(
            f"Common support range: [{overlap_results.common_support_range[0]:.4f}, "
            f"{overlap_results.common_support_range[1]:.4f}]"
        )
        print(
            f"Units in common support: {overlap_results.units_in_common_support}/"
            f"{overlap_results.total_units} ({overlap_results.units_in_common_support / overlap_results.total_units:.1%})"
        )

        if overlap_results.propensity_model_auc is not None:
            print(f"Propensity model AUC: {overlap_results.propensity_model_auc:.3f}")

        print(
            f"Extreme weights (>{self.extreme_weight_threshold}): {overlap_results.extreme_weights_count}"
        )
        print()

        if overlap_results.overall_positivity_met:
            print("✅ Overall positivity: SATISFIED")
        else:
            print("❌ Overall positivity: VIOLATED")

        if overlap_results.violations:
            print(f"\nViolations detected ({len(overlap_results.violations)}):")
            for i, violation in enumerate(overlap_results.violations, 1):
                severity_icon = "🔴" if violation["severity"] == "high" else "🟡"
                print(f"  {i}. {severity_icon} {violation['description']}")
                print(f"     Affected units: {violation['affected_units']}")

        print(f"\nRecommendation: {overlap_results.recommendation}")

    def create_overlap_table(self, overlap_results: OverlapResults) -> pd.DataFrame:
        """Create a formatted overlap assessment table."""
        data = {
            "Metric": [
                "Min Propensity Score",
                "Max Propensity Score",
                "Common Support Coverage",
                "Extreme Weights",
                "Positivity Satisfied",
            ],
            "Value": [
                f"{overlap_results.min_propensity_score:.4f}",
                f"{overlap_results.max_propensity_score:.4f}",
                f"{overlap_results.units_in_common_support}/{overlap_results.total_units} ({overlap_results.units_in_common_support / overlap_results.total_units:.1%})",
                f"{overlap_results.extreme_weights_count}",
                "Yes" if overlap_results.overall_positivity_met else "No",
            ],
            "Status": [
                "✅"
                if overlap_results.min_propensity_score >= self.min_propensity
                else "❌",
                "✅"
                if overlap_results.max_propensity_score <= self.max_propensity
                else "❌",
                "✅"
                if overlap_results.units_in_common_support
                >= 0.8 * overlap_results.total_units
                else "⚠️",
                "✅" if overlap_results.extreme_weights_count == 0 else "⚠️",
                "✅" if overlap_results.overall_positivity_met else "❌",
            ],
        }

        return pd.DataFrame(data)


def assess_positivity(
    treatment: TreatmentData,
    covariates: CovariateData,
    min_propensity: float = 0.01,
    max_propensity: float = 0.99,
    verbose: bool = True,
) -> OverlapResults:
    """Convenience function to assess positivity assumption.

    Args:
        treatment: Treatment assignment data
        covariates: Covariate data
        min_propensity: Minimum acceptable propensity score
        max_propensity: Maximum acceptable propensity score
        verbose: Whether to print results

    Returns:
        OverlapResults object
    """
    diagnostics = OverlapDiagnostics(
        min_propensity=min_propensity,
        max_propensity=max_propensity,
    )
    results = diagnostics.assess_overlap(treatment, covariates)

    if verbose:
        diagnostics.print_overlap_summary(results)

    return results


def calculate_propensity_overlap(
    treatment: TreatmentData,
    covariates: CovariateData,
    model_type: str = "logistic",
) -> dict[str, Any]:
    """Calculate propensity score overlap metrics.

    Args:
        treatment: Treatment assignment data
        covariates: Covariate data
        model_type: Model for propensity score estimation

    Returns:
        Dictionary with overlap metrics
    """
    propensity_scores = calculate_propensity_scores(treatment, covariates, model_type)
    treatment_vals = np.asarray(treatment.values)

    # Split by treatment group
    treated_ps = propensity_scores[treatment_vals == 1]
    control_ps = propensity_scores[treatment_vals == 0]

    # Calculate overlap metrics
    common_support_range, units_in_support = check_common_support(
        propensity_scores, treatment_vals
    )

    return {
        "propensity_scores": propensity_scores,
        "treated_ps_mean": float(np.mean(treated_ps)),
        "treated_ps_std": float(np.std(treated_ps)),
        "control_ps_mean": float(np.mean(control_ps)),
        "control_ps_std": float(np.std(control_ps)),
        "common_support_range": common_support_range,
        "common_support_coverage": units_in_support / len(treatment_vals),
        "min_propensity": float(np.min(propensity_scores)),
        "max_propensity": float(np.max(propensity_scores)),
    }


def calculate_calibration_metrics(
    true_treatment: NDArray[Any] | pd.Series,
    predicted_probabilities: NDArray[Any] | pd.Series,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calculate propensity score calibration metrics.

    Args:
        true_treatment: Actual treatment assignments (0/1)
        predicted_probabilities: Predicted propensity scores
        n_bins: Number of bins for calibration assessment

    Returns:
        Dictionary with calibration metrics
    """
    try:
        true_treatment = np.asarray(true_treatment)
        predicted_probabilities = np.asarray(predicted_probabilities)

        # Remove any NaN values
        mask = ~(np.isnan(true_treatment) | np.isnan(predicted_probabilities))
        y_true = true_treatment[mask]
        y_prob = predicted_probabilities[mask]

        if len(y_true) == 0:
            return {"error": "No valid data for calibration assessment"}

        # Calculate Brier score
        brier_score = np.mean((y_prob - y_true) ** 2)

        # Bin predictions for calibration curve
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        fraction_of_positives = []
        mean_predicted_value = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

            if np.sum(in_bin) == 0:
                continue

            # Calculate fraction of positives and mean prediction in bin
            fraction_of_positives.append(np.mean(y_true[in_bin]))
            mean_predicted_value.append(np.mean(y_prob[in_bin]))

        # Calculate mean absolute calibration error (MACE)
        if len(fraction_of_positives) > 0:
            mace = np.mean(
                np.abs(np.array(fraction_of_positives) - np.array(mean_predicted_value))
            )
        else:
            mace = np.nan

        # Simple calibration slope (correlation between predicted and actual)
        if len(y_true) > 1:
            calibration_slope = np.corrcoef(y_prob, y_true)[0, 1]
        else:
            calibration_slope = np.nan

        # Determine if well calibrated (MACE < 0.1 and Brier score reasonable)
        well_calibrated = not np.isnan(mace) and mace < 0.1 and brier_score < 0.25

        return {
            "fraction_of_positives": fraction_of_positives,
            "mean_predicted_value": mean_predicted_value,
            "brier_score": float(brier_score),
            "calibration_slope": float(calibration_slope)
            if not np.isnan(calibration_slope)
            else np.nan,
            "mean_absolute_calibration_error": float(mace)
            if not np.isnan(mace)
            else np.nan,
            "well_calibrated": well_calibrated,
            "n_bins_used": len(fraction_of_positives),
        }

    except Exception as e:
        return {"error": str(e)}


def suggest_trimming_thresholds(
    propensity_scores: NDArray[Any] | pd.Series,
    treatment: NDArray[Any] | pd.Series,
    percentiles: list[float] = [1.0, 2.5, 5.0, 10.0],
) -> dict[str, dict[str, Any]]:
    """Suggest propensity score trimming thresholds.

    Args:
        propensity_scores: Array of propensity scores
        treatment: Treatment assignment values
        percentiles: Percentiles for trimming (symmetric from both tails)

    Returns:
        Dictionary with trimming recommendations for each percentile
    """
    try:
        propensity_scores = np.asarray(propensity_scores)
        treatment = np.asarray(treatment)

        # Remove NaN values
        mask = ~(np.isnan(propensity_scores) | np.isnan(treatment))
        ps_clean = propensity_scores[mask]
        treatment_clean = treatment[mask]

        if len(ps_clean) == 0:
            return {"error": "No valid data for trimming analysis"}

        recommendations = {}

        for pct in percentiles:
            # Calculate symmetric percentile bounds
            lower_pct = pct / 2
            upper_pct = 100 - (pct / 2)

            lower_bound = np.percentile(ps_clean, lower_pct)
            upper_bound = np.percentile(ps_clean, upper_pct)

            # Find units within bounds
            within_bounds = (ps_clean >= lower_bound) & (ps_clean <= upper_bound)
            treatment_within = treatment_clean[within_bounds]

            # Count remaining units by treatment group
            n_treated_remaining = np.sum(treatment_within == 1)
            n_control_remaining = np.sum(treatment_within == 0)
            n_total_remaining = len(treatment_within)

            # Calculate trimming statistics
            n_trimmed = len(ps_clean) - n_total_remaining
            pct_trimmed = (n_trimmed / len(ps_clean)) * 100

            recommendations[f"trim_{pct:.1f}pct"] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "n_trimmed": int(n_trimmed),
                "pct_trimmed": float(pct_trimmed),
                "n_treated_remaining": int(n_treated_remaining),
                "n_control_remaining": int(n_control_remaining),
                "n_total_remaining": int(n_total_remaining),
                "treated_control_ratio": float(
                    n_treated_remaining / n_control_remaining
                )
                if n_control_remaining > 0
                else np.inf,
            }

        return recommendations

    except Exception as e:
        return {"error": str(e)}
