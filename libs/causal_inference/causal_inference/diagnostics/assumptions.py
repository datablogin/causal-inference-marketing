"""Assumption checking and confounding detection for causal inference.

This module implements tools to detect potential confounding and assess
the exchangeability assumption critical for causal inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ..core.base import CovariateData, OutcomeData, TreatmentData


def _ensure_dataframe(covariates: CovariateData) -> pd.DataFrame:
    """Convert covariate data to DataFrame format.

    Args:
        covariates: Covariate data that may be array or DataFrame

    Returns:
        DataFrame representation of covariates

    Raises:
        ValueError: If covariates cannot be converted to DataFrame
    """
    if isinstance(covariates.values, pd.DataFrame):
        return covariates.values
    elif isinstance(covariates.values, np.ndarray):
        # Use provided names or generate default names
        if covariates.names and len(covariates.names) == covariates.values.shape[1]:
            column_names = covariates.names
        else:
            column_names = [f"X{i + 1}" for i in range(covariates.values.shape[1])]

        return pd.DataFrame(covariates.values, columns=column_names)
    else:
        raise ValueError(
            f"Covariates must be DataFrame or numpy array, got {type(covariates.values)}"
        )


@dataclass
class AssumptionResults:
    """Results from causal inference assumption checking."""

    exchangeability_likely: bool
    confounding_detected: bool
    confounding_strength: dict[str, float]
    backdoor_criterion_met: bool | None
    collider_bias_risk: bool
    selection_bias_indicators: list[str]
    outcome_predictors: dict[str, float]
    treatment_predictors: dict[str, float]
    assumptions_summary: dict[str, bool]
    recommendations: list[str]


def detect_confounding_associations(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Detect associations that may indicate confounding.

    A variable is a potential confounder if it's associated with both
    treatment and outcome.

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        covariates: Covariate data
        alpha: Significance level for tests

    Returns:
        Dictionary with association results for each covariate
    """
    # Convert to DataFrame if needed
    cov_df = _ensure_dataframe(covariates)

    results = {}
    treatment_vals = np.asarray(treatment.values)
    outcome_vals = np.asarray(outcome.values)

    for covariate_name in cov_df.columns:
        covariate_vals = np.asarray(cov_df[covariate_name])

        # Remove missing values
        mask = ~(
            np.isnan(covariate_vals) | np.isnan(treatment_vals) | np.isnan(outcome_vals)
        )
        cov_clean = covariate_vals[mask]
        treat_clean = treatment_vals[mask]
        outcome_clean = outcome_vals[mask]

        if len(cov_clean) < 10:
            continue

        # Test association with treatment
        try:
            if treatment.treatment_type == "binary":
                # Point-biserial correlation for binary treatment
                treat_corr, treat_p = stats.pointbiserialr(treat_clean, cov_clean)
            else:
                # Pearson correlation for continuous treatment
                treat_corr, treat_p = stats.pearsonr(treat_clean, cov_clean)
        except (ValueError, ZeroDivisionError):
            treat_corr, treat_p = np.nan, np.nan

        # Test association with outcome
        try:
            outcome_corr, outcome_p = stats.pearsonr(cov_clean, outcome_clean)
        except (ValueError, ZeroDivisionError):
            outcome_corr, outcome_p = np.nan, np.nan

        # Determine confounding status
        is_confounder = (
            not np.isnan(treat_corr)
            and not np.isnan(outcome_corr)
            and treat_p < alpha
            and outcome_p < alpha
            and abs(treat_corr) > 0.1
            and abs(outcome_corr) > 0.1
        )

        results[covariate_name] = {
            "treatment_correlation": float(treat_corr)
            if not np.isnan(treat_corr)
            else None,
            "treatment_p_value": float(treat_p) if not np.isnan(treat_p) else None,
            "outcome_correlation": float(outcome_corr)
            if not np.isnan(outcome_corr)
            else None,
            "outcome_p_value": float(outcome_p) if not np.isnan(outcome_p) else None,
            "is_potential_confounder": is_confounder,
            "confounding_strength": abs(treat_corr) * abs(outcome_corr)
            if not (np.isnan(treat_corr) or np.isnan(outcome_corr))
            else 0.0,
        }

    return results


def assess_outcome_prediction(
    outcome: OutcomeData,
    covariates: CovariateData,
    include_interactions: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """Assess how well covariates predict the outcome.

    This helps identify important confounders and assess model specification.

    Args:
        outcome: Outcome data
        covariates: Covariate data
        include_interactions: Whether to include interaction terms
        random_state: Random seed for RandomForest

    Returns:
        Dictionary with prediction results
    """
    # Convert to DataFrame if needed
    X = _ensure_dataframe(covariates)

    # Performance warning for large datasets
    n_obs, n_features = X.shape
    if n_obs > 50000:
        print(
            f"⚠️  Performance warning: Large dataset ({n_obs:,} observations). "
            f"Consider sampling for faster computation."
        )
    if n_features > 100:
        print(
            f"⚠️  Performance warning: Many features ({n_features} features). "
            f"RandomForest computation may be slow."
        )

    # Prepare data
    X = X.fillna(X.mean())
    y = np.asarray(outcome.values)

    # Remove missing outcomes
    mask = ~np.isnan(y)
    X_clean = X.loc[mask]
    y_clean = y[mask]

    if len(y_clean) < 10:
        return {"r2_score": 0.0, "feature_importance": {}}

    # Add interactions if requested
    if include_interactions and X_clean.shape[1] <= 10:  # Limit to avoid explosion
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    X_clean[f"{col1}_x_{col2}"] = X_clean[col1] * X_clean[col2]

    # Fit models
    try:
        # Linear model
        lr = LinearRegression()
        lr.fit(X_clean, y_clean)
        lr_r2 = r2_score(y_clean, lr.predict(X_clean))

        # Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf.fit(X_clean, y_clean)
        rf_r2 = r2_score(y_clean, rf.predict(X_clean))

        # Feature importance
        feature_importance = dict(zip(X_clean.columns, rf.feature_importances_))

        return {
            "linear_r2": float(lr_r2),
            "rf_r2": float(rf_r2),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "most_important_features": sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    except (ValueError, RuntimeError) as e:
        return {"error": str(e), "r2_score": 0.0, "feature_importance": {}}
    except Exception as e:
        # Re-raise unexpected errors rather than silently handling them
        raise RuntimeError(f"Unexpected error in outcome prediction: {e}") from e


def check_collider_bias_risk(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
) -> dict[str, Any]:
    """Check for potential collider bias in covariate adjustment.

    A collider is a variable that is caused by both treatment and outcome.
    Conditioning on colliders can introduce bias.

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        covariates: Covariate data

    Returns:
        Dictionary with collider bias assessment
    """
    if not isinstance(covariates.values, pd.DataFrame):
        return {"collider_risk": False, "potential_colliders": []}

    potential_colliders = []
    treatment_vals = np.asarray(treatment.values)
    outcome_vals = np.asarray(outcome.values)

    for covariate_name in covariates.values.columns:
        covariate_vals = np.asarray(covariates.values[covariate_name])

        # Remove missing values
        mask = ~(
            np.isnan(covariate_vals) | np.isnan(treatment_vals) | np.isnan(outcome_vals)
        )
        cov_clean = covariate_vals[mask]
        treat_clean = treatment_vals[mask]
        outcome_clean = outcome_vals[mask]

        if len(cov_clean) < 10:
            continue

        # Test if covariate is predicted by both treatment and outcome
        try:
            # Predict covariate from treatment and outcome
            X = np.column_stack([treat_clean, outcome_clean])
            lr = LinearRegression()
            lr.fit(X, cov_clean)
            r2 = r2_score(cov_clean, lr.predict(X))

            # If R² is high, covariate might be a collider
            if r2 > 0.2:  # Arbitrary threshold
                potential_colliders.append(
                    {
                        "variable": covariate_name,
                        "r2_from_treatment_outcome": float(r2),
                        "risk_level": "high" if r2 > 0.4 else "medium",
                    }
                )

        except Exception:
            continue

    return {
        "collider_risk": len(potential_colliders) > 0,
        "potential_colliders": potential_colliders,
        "recommendation": "Consider excluding potential colliders from adjustment set"
        if potential_colliders
        else "No colliders detected",
    }


class AssumptionChecker:
    """Comprehensive assumption checking for causal inference."""

    def __init__(
        self,
        alpha: float = 0.05,
        confounding_threshold: float = 0.1,
        prediction_threshold: float = 0.1,
    ):
        """Initialize assumption checker.

        Args:
            alpha: Significance level for statistical tests
            confounding_threshold: Threshold for declaring confounding
            prediction_threshold: Minimum R² for prediction importance
        """
        self.alpha = alpha
        self.confounding_threshold = confounding_threshold
        self.prediction_threshold = prediction_threshold

    def check_all_assumptions(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData,
    ) -> AssumptionResults:
        """Comprehensive assumption checking for causal inference.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            covariates: Covariate data

        Returns:
            AssumptionResults with comprehensive assessment
        """
        # Detect confounding
        confounding_results = detect_confounding_associations(
            treatment, outcome, covariates, self.alpha
        )

        # Assess outcome prediction
        outcome_prediction = assess_outcome_prediction(outcome, covariates)

        # Check for collider bias
        collider_results = check_collider_bias_risk(treatment, outcome, covariates)

        # Summarize confounding
        confounders = [
            name
            for name, results in confounding_results.items()
            if results["is_potential_confounder"]
        ]

        confounding_strength = {
            name: results["confounding_strength"]
            for name, results in confounding_results.items()
        }

        # Assess exchangeability
        strong_confounders = [
            name
            for name, strength in confounding_strength.items()
            if strength > self.confounding_threshold
        ]

        exchangeability_likely = len(strong_confounders) == 0
        confounding_detected = len(confounders) > 0

        # Check backdoor criterion (simplified)
        # In practice, this requires causal graph analysis
        backdoor_criterion_met = None  # Cannot determine without causal graph

        # Selection bias indicators
        selection_bias_indicators = []
        if confounding_detected:
            selection_bias_indicators.append("Measured confounders detected")

        # Extract important predictors
        outcome_predictors: dict[str, float] = outcome_prediction.get("feature_importance", {})
        treatment_predictors: dict[str, float] = {}  # Would need separate analysis

        # Generate recommendations
        recommendations = self._generate_recommendations(
            confounding_detected, strong_confounders, collider_results
        )

        # Assumptions summary
        assumptions_summary = {
            "exchangeability": exchangeability_likely,
            "positivity": True,  # Placeholder - requires separate overlap analysis
            "consistency": True,  # Placeholder - difficult to test directly
            "no_interference": True,  # Placeholder - requires study design knowledge
        }

        return AssumptionResults(
            exchangeability_likely=exchangeability_likely,
            confounding_detected=confounding_detected,
            confounding_strength=confounding_strength,
            backdoor_criterion_met=backdoor_criterion_met,
            collider_bias_risk=collider_results["collider_risk"],
            selection_bias_indicators=selection_bias_indicators,
            outcome_predictors=outcome_predictors,
            treatment_predictors=treatment_predictors,
            assumptions_summary=assumptions_summary,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        confounding_detected: bool,
        strong_confounders: list[str],
        collider_results: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on assumption checking."""
        recommendations = []

        if confounding_detected:
            recommendations.append(
                "Measured confounders detected - ensure proper adjustment"
            )
            if strong_confounders:
                recommendations.append(
                    f"Strong confounders identified: {', '.join(strong_confounders)}"
                )

        if collider_results["collider_risk"]:
            recommendations.append(
                "Potential colliders detected - review adjustment set"
            )
            collider_names = [
                c["variable"] for c in collider_results["potential_colliders"]
            ]
            recommendations.append(f"Consider excluding: {', '.join(collider_names)}")

        if not confounding_detected and not collider_results["collider_risk"]:
            recommendations.append("No major assumption violations detected")

        # General recommendations
        recommendations.extend(
            [
                "Consider sensitivity analysis for unmeasured confounding",
                "Verify exchangeability assumption with domain knowledge",
                "Consider alternative identification strategies if assumptions questionable",
            ]
        )

        return recommendations

    def print_assumption_summary(self, results: AssumptionResults) -> None:
        """Print summary of assumption checking results."""
        print("=== Causal Inference Assumptions Assessment ===")
        print()

        # Exchangeability
        status = "✅ LIKELY" if results.exchangeability_likely else "❌ QUESTIONABLE"
        print(f"Exchangeability (No Unobserved Confounding): {status}")

        if results.confounding_detected:
            print(f"Measured confounders detected: {len(results.confounding_strength)}")
            strong_confounders = [
                name
                for name, strength in results.confounding_strength.items()
                if strength > self.confounding_threshold
            ]
            if strong_confounders:
                print(f"Strong confounders: {', '.join(strong_confounders)}")
        else:
            print("No measured confounders detected")

        # Collider bias
        if results.collider_bias_risk:
            print("⚠️ Potential collider bias detected")

        # Selection bias
        if results.selection_bias_indicators:
            print(
                f"Selection bias indicators: {', '.join(results.selection_bias_indicators)}"
            )

        print()
        print("Key Outcome Predictors:")
        if results.outcome_predictors:
            sorted_predictors = sorted(
                results.outcome_predictors.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for name, importance in sorted_predictors:
                print(f"  - {name}: {importance:.3f}")

        print()
        print("Recommendations:")
        for i, rec in enumerate(results.recommendations, 1):
            print(f"  {i}. {rec}")


def check_confounding_detection(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convenience function for confounding detection.

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        covariates: Covariate data
        alpha: Significance level
        verbose: Whether to print results

    Returns:
        Dictionary with confounding detection results
    """
    results = detect_confounding_associations(treatment, outcome, covariates, alpha)

    confounders = [
        name for name, res in results.items() if res["is_potential_confounder"]
    ]

    if verbose:
        print("=== Confounding Detection ===")
        if confounders:
            print(f"Potential confounders detected: {len(confounders)}")
            for name in confounders:
                strength = results[name]["confounding_strength"]
                print(f"  - {name}: strength = {strength:.3f}")
        else:
            print("No potential confounders detected")

    return results


def check_exchangeability(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    verbose: bool = True,
) -> bool:
    """Check exchangeability assumption (simplified assessment).

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        covariates: Covariate data
        verbose: Whether to print results

    Returns:
        Boolean indicating if exchangeability likely holds
    """
    checker = AssumptionChecker()
    results = checker.check_all_assumptions(treatment, outcome, covariates)

    if verbose:
        checker.print_assumption_summary(results)

    return results.exchangeability_likely


def run_all_assumption_checks(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    verbose: bool = True,
) -> AssumptionResults:
    """Run comprehensive assumption checking.

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        covariates: Covariate data
        verbose: Whether to print results

    Returns:
        AssumptionResults with comprehensive assessment
    """
    checker = AssumptionChecker()
    results = checker.check_all_assumptions(treatment, outcome, covariates)

    if verbose:
        checker.print_assumption_summary(results)

    return results
