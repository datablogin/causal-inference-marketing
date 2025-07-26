"""Model specification tests for causal inference.

This module implements diagnostic tests to assess model specification
and functional form assumptions in causal inference estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from ..core.base import CovariateData, OutcomeData, TreatmentData


@dataclass
class SpecificationResults:
    """Results from model specification tests."""

    linearity_test_results: dict[str, Any]
    interaction_test_results: dict[str, Any]
    functional_form_results: dict[str, Any]
    heteroskedasticity_test: dict[str, Any]
    specification_passed: bool
    problematic_variables: list[str]
    recommendations: list[str]


def linearity_tests(
    outcome: OutcomeData,
    covariates: CovariateData,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Test linearity assumption for continuous covariates.

    Uses polynomial regression and component+residual plots approach
    to test if linear functional form is appropriate.

    Args:
        outcome: Outcome data
        covariates: Covariate data
        alpha: Significance level for tests

    Returns:
        Dictionary with linearity test results
    """
    if not isinstance(covariates.values, pd.DataFrame):
        raise ValueError("Covariates must be a DataFrame")

    y = np.asarray(outcome.values)
    X = covariates.values.fillna(covariates.values.mean())

    # Remove missing outcomes
    mask = ~np.isnan(y)
    y_clean = y[mask]
    X_clean = X.loc[mask]

    linearity_results = {}

    # Test each continuous covariate
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        x_var = X_clean[col].values

        if len(np.unique(x_var)) < 5:  # Skip if not truly continuous
            continue

        # Fit linear model
        X_linear = x_var.reshape(-1, 1)
        linear_model = LinearRegression()
        linear_model.fit(X_linear, y_clean)
        linear_pred = linear_model.predict(X_linear)
        linear_mse = mean_squared_error(y_clean, linear_pred)

        # Fit polynomial model (degree 2)
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(X_linear)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y_clean)
        poly_pred = poly_model.predict(X_poly)
        poly_mse = mean_squared_error(y_clean, poly_pred)

        # F-test for improved fit
        n = len(y_clean)
        p_linear = 1  # Number of parameters in linear model
        p_poly = 2  # Number of parameters in polynomial model

        f_stat = ((linear_mse - poly_mse) / (p_poly - p_linear)) / (
            poly_mse / (n - p_poly)
        )
        f_p_value = 1 - stats.f.cdf(f_stat, p_poly - p_linear, n - p_poly)

        # Linearity test result
        linearity_ok = f_p_value > alpha  # Fail to reject linear form

        linearity_results[col] = {
            "linear_mse": float(linear_mse),
            "polynomial_mse": float(poly_mse),
            "f_statistic": float(f_stat),
            "p_value": float(f_p_value),
            "linearity_assumption_met": linearity_ok,
            "improvement_ratio": float((linear_mse - poly_mse) / linear_mse),
        }

    # Overall assessment
    problematic_vars = [
        var
        for var, result in linearity_results.items()
        if not result["linearity_assumption_met"]
    ]

    return {
        "individual_tests": linearity_results,
        "problematic_variables": problematic_vars,
        "overall_linearity_ok": len(problematic_vars) == 0,
        "recommendation": "Consider polynomial terms or splines for problematic variables"
        if problematic_vars
        else "Linear functional forms appear appropriate",
    }


def interaction_tests(
    outcome: OutcomeData,
    treatment: TreatmentData,
    covariates: CovariateData,
    alpha: float = 0.05,
    max_interactions: int = 5,
) -> dict[str, Any]:
    """Test for significant treatment-covariate interactions.

    Args:
        outcome: Outcome data
        treatment: Treatment data
        covariates: Covariate data
        alpha: Significance level
        max_interactions: Maximum number of interactions to test

    Returns:
        Dictionary with interaction test results
    """
    if not isinstance(covariates.values, pd.DataFrame):
        raise ValueError("Covariates must be a DataFrame")

    y = np.asarray(outcome.values)
    t = np.asarray(treatment.values)
    X = covariates.values.fillna(covariates.values.mean())

    # Remove missing values
    mask = ~(np.isnan(y) | np.isnan(t))
    y_clean = y[mask]
    t_clean = t[mask]
    X_clean = X.loc[mask]

    # Prepare data
    n = len(y_clean)
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns[:max_interactions]

    interaction_results = {}

    for col in numeric_cols:
        x_var = X_clean[col].values

        # Create interaction term
        interaction_term = t_clean * x_var

        # Fit model without interaction
        X_base = np.column_stack([t_clean, x_var])
        base_model = LinearRegression()
        base_model.fit(X_base, y_clean)
        base_pred = base_model.predict(X_base)
        base_mse = mean_squared_error(y_clean, base_pred)

        # Fit model with interaction
        X_interaction = np.column_stack([t_clean, x_var, interaction_term])
        interaction_model = LinearRegression()
        interaction_model.fit(X_interaction, y_clean)
        interaction_pred = interaction_model.predict(X_interaction)
        interaction_mse = mean_squared_error(y_clean, interaction_pred)

        # F-test for interaction significance
        p_base = 2  # treatment + covariate
        p_interaction = 3  # treatment + covariate + interaction

        f_stat = ((base_mse - interaction_mse) / (p_interaction - p_base)) / (
            interaction_mse / (n - p_interaction)
        )
        f_p_value = 1 - stats.f.cdf(f_stat, p_interaction - p_base, n - p_interaction)

        # Extract interaction coefficient
        interaction_coef = interaction_model.coef_[
            -1
        ]  # Last coefficient is interaction

        interaction_results[col] = {
            "interaction_coefficient": float(interaction_coef),
            "f_statistic": float(f_stat),
            "p_value": float(f_p_value),
            "significant": f_p_value < alpha,
            "base_mse": float(base_mse),
            "interaction_mse": float(interaction_mse),
        }

    # Find significant interactions
    significant_interactions = [
        var for var, result in interaction_results.items() if result["significant"]
    ]

    return {
        "individual_tests": interaction_results,
        "significant_interactions": significant_interactions,
        "heterogeneous_treatment_effects": len(significant_interactions) > 0,
        "recommendation": "Consider heterogeneous treatment effect models"
        if significant_interactions
        else "Homogeneous treatment effects assumption appears reasonable",
    }


def functional_form_tests(
    outcome: OutcomeData,
    covariates: CovariateData,
    comparison_models: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Test functional form specification using model comparison.

    Args:
        outcome: Outcome data
        covariates: Covariate data
        comparison_models: List of models to compare ('linear', 'polynomial', 'random_forest')
        random_state: Random seed for RandomForest

    Returns:
        Dictionary with functional form test results
    """
    if comparison_models is None:
        comparison_models = ["linear", "polynomial", "random_forest"]

    if not isinstance(covariates.values, pd.DataFrame):
        raise ValueError("Covariates must be a DataFrame")

    # Performance warning for large datasets
    n_obs, n_features = covariates.values.shape
    if n_obs > 50000 and "random_forest" in comparison_models:
        print(
            f"⚠️  Performance warning: Large dataset ({n_obs:,} observations) with RandomForest. "
            f"Consider using n_estimators=50 or sampling for faster computation."
        )

    y = np.asarray(outcome.values)
    X = covariates.values.fillna(covariates.values.mean())

    # Remove missing outcomes
    mask = ~np.isnan(y)
    y_clean = y[mask]
    X_clean = X.loc[mask]

    model_results = {}

    # Linear model
    if "linear" in comparison_models:
        linear_model = LinearRegression()
        linear_model.fit(X_clean, y_clean)
        linear_pred = linear_model.predict(X_clean)
        linear_r2 = r2_score(y_clean, linear_pred)
        linear_mse = mean_squared_error(y_clean, linear_pred)

        model_results["linear"] = {
            "r2_score": float(linear_r2),
            "mse": float(linear_mse),
            "aic": _calculate_aic(linear_mse, len(y_clean), X_clean.shape[1]),
        }

    # Polynomial model
    if "polynomial" in comparison_models and X_clean.shape[1] <= 5:  # Avoid explosion
        poly_features = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        X_poly = poly_features.fit_transform(X_clean)

        if X_poly.shape[1] < len(y_clean):  # Ensure we have enough observations
            poly_model = LinearRegression()
            poly_model.fit(X_poly, y_clean)
            poly_pred = poly_model.predict(X_poly)
            poly_r2 = r2_score(y_clean, poly_pred)
            poly_mse = mean_squared_error(y_clean, poly_pred)

            model_results["polynomial"] = {
                "r2_score": float(poly_r2),
                "mse": float(poly_mse),
                "aic": _calculate_aic(poly_mse, len(y_clean), X_poly.shape[1]),
            }

    # Random Forest model
    if "random_forest" in comparison_models:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf_model.fit(X_clean, y_clean)
        rf_pred = rf_model.predict(X_clean)
        rf_r2 = r2_score(y_clean, rf_pred)
        rf_mse = mean_squared_error(y_clean, rf_pred)

        model_results["random_forest"] = {
            "r2_score": float(rf_r2),
            "mse": float(rf_mse),
            "aic": float('inf'),  # Use infinity instead of None for AIC
        }

    # Find best model
    best_model = max(model_results.keys(), key=lambda x: model_results[x]["r2_score"])

    # Assess if linear model is adequate
    linear_adequate = True
    if "linear" in model_results and best_model != "linear":
        linear_r2 = model_results["linear"]["r2_score"]
        best_r2 = model_results[best_model]["r2_score"]

        # If other models substantially outperform linear
        if best_r2 - linear_r2 > 0.1:  # 10% improvement threshold
            linear_adequate = False

    return {
        "model_comparisons": model_results,
        "best_model": best_model,
        "linear_model_adequate": linear_adequate,
        "recommendation": "Linear functional form appears adequate"
        if linear_adequate
        else f"Consider {best_model} or more flexible functional forms",
    }


def _calculate_aic(mse: float, n: int, p: int) -> float:
    """Calculate AIC for model comparison."""
    log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - n / 2
    aic = 2 * p - 2 * log_likelihood
    return float(aic)


def test_heteroskedasticity(
    residuals: NDArray[Any],
    fitted_values: NDArray[Any],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Test for heteroskedasticity using Breusch-Pagan test.

    Args:
        residuals: Model residuals
        fitted_values: Model fitted values
        alpha: Significance level

    Returns:
        Dictionary with heteroskedasticity test results
    """
    n = len(residuals)

    # Breusch-Pagan test: regress squared residuals on fitted values
    squared_residuals = residuals**2

    # Linear regression of squared residuals on fitted values
    X = fitted_values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, squared_residuals)

    # Calculate test statistic
    ss_explained = np.sum((model.predict(X) - np.mean(squared_residuals)) ** 2)
    ss_total = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)

    r_squared = ss_explained / ss_total if ss_total > 0 else 0
    lm_statistic = n * r_squared

    # Chi-square test (1 degree of freedom)
    p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)

    return {
        "lm_statistic": float(lm_statistic),
        "p_value": float(p_value),
        "heteroskedasticity_detected": p_value < alpha,
        "recommendation": "Consider robust standard errors"
        if p_value < alpha
        else "Homoskedasticity assumption appears met",
    }


class ModelSpecificationTests:
    """Comprehensive model specification testing."""

    def __init__(
        self,
        alpha: float = 0.05,
        linearity_threshold: float = 0.1,
        interaction_threshold: float = 0.05,
    ):
        """Initialize specification tests.

        Args:
            alpha: Significance level for tests
            linearity_threshold: Threshold for linearity concern
            interaction_threshold: Threshold for interaction significance
        """
        self.alpha = alpha
        self.linearity_threshold = linearity_threshold
        self.interaction_threshold = interaction_threshold

    def comprehensive_specification_tests(
        self,
        outcome: OutcomeData,
        treatment: TreatmentData,
        covariates: CovariateData,
    ) -> SpecificationResults:
        """Run comprehensive model specification tests.

        Args:
            outcome: Outcome data
            treatment: Treatment data
            covariates: Covariate data

        Returns:
            SpecificationResults with comprehensive assessment
        """
        # Linearity tests
        linearity_results = linearity_tests(outcome, covariates, self.alpha)

        # Interaction tests
        interaction_results = interaction_tests(
            outcome, treatment, covariates, self.interaction_threshold
        )

        # Functional form tests
        functional_form_results = functional_form_tests(outcome, covariates)

        # Heteroskedasticity test (fit simple model first)
        y = np.asarray(outcome.values)
        X = (
            covariates.values.fillna(covariates.values.mean())
            if isinstance(covariates.values, pd.DataFrame)
            else covariates.values
        )

        mask = ~np.isnan(y)
        y_clean = y[mask]
        X_clean = X.loc[mask] if isinstance(X, pd.DataFrame) else X[mask]

        # Fit model for residuals
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        fitted = model.predict(X_clean)
        residuals = y_clean - fitted

        heteroskedasticity_results = test_heteroskedasticity(
            residuals, fitted, self.alpha
        )

        # Assess overall specification
        problematic_variables = []

        # Add linearity problems
        problematic_variables.extend(linearity_results["problematic_variables"])

        # Add interaction concerns
        if interaction_results["heterogeneous_treatment_effects"]:
            problematic_variables.extend(
                interaction_results["significant_interactions"]
            )

        # Overall assessment
        specification_passed = (
            linearity_results["overall_linearity_ok"]
            and functional_form_results["linear_model_adequate"]
            and not heteroskedasticity_results["heteroskedasticity_detected"]
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            linearity_results,
            interaction_results,
            functional_form_results,
            heteroskedasticity_results,
        )

        return SpecificationResults(
            linearity_test_results=linearity_results,
            interaction_test_results=interaction_results,
            functional_form_results=functional_form_results,
            heteroskedasticity_test=heteroskedasticity_results,
            specification_passed=specification_passed,
            problematic_variables=list(set(problematic_variables)),
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        linearity_results: dict[str, Any],
        interaction_results: dict[str, Any],
        functional_form_results: dict[str, Any],
        heteroskedasticity_results: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on specification tests."""
        recommendations = []

        # Linearity recommendations
        if not linearity_results["overall_linearity_ok"]:
            recommendations.append(
                "Consider polynomial or spline terms for non-linear variables"
            )
            recommendations.append(
                f"Variables with linearity issues: {', '.join(linearity_results['problematic_variables'])}"
            )

        # Interaction recommendations
        if interaction_results["heterogeneous_treatment_effects"]:
            recommendations.append(
                "Treatment effect heterogeneity detected - consider subgroup analysis"
            )
            recommendations.append(
                f"Significant interactions: {', '.join(interaction_results['significant_interactions'])}"
            )

        # Functional form recommendations
        if not functional_form_results["linear_model_adequate"]:
            recommendations.append(functional_form_results["recommendation"])

        # Heteroskedasticity recommendations
        if heteroskedasticity_results["heteroskedasticity_detected"]:
            recommendations.append(heteroskedasticity_results["recommendation"])

        # General recommendations
        if not recommendations:
            recommendations.append("✅ Model specification appears adequate")
        else:
            recommendations.append(
                "Consider model diagnostics and alternative specifications"
            )

        return recommendations

    def print_specification_summary(self, results: SpecificationResults) -> None:
        """Print summary of specification test results."""
        print("=== Model Specification Tests ===")
        print()

        status = "✅ PASSED" if results.specification_passed else "❌ ISSUES DETECTED"
        print(f"Overall specification: {status}")
        print()

        # Linearity
        print("Linearity Tests:")
        if results.linearity_test_results["overall_linearity_ok"]:
            print("  ✅ Linear functional forms appear appropriate")
        else:
            print(
                f"  ❌ Linearity issues detected: {', '.join(results.linearity_test_results['problematic_variables'])}"
            )

        # Interactions
        print("Interaction Tests:")
        if results.interaction_test_results["heterogeneous_treatment_effects"]:
            print(
                f"  ⚠️ Treatment effect heterogeneity: {', '.join(results.interaction_test_results['significant_interactions'])}"
            )
        else:
            print("  ✅ Homogeneous treatment effects assumption reasonable")

        # Functional form
        print("Functional Form:")
        if results.functional_form_results["linear_model_adequate"]:
            print("  ✅ Linear model adequate")
        else:
            print(f"  ⚠️ {results.functional_form_results['recommendation']}")

        # Heteroskedasticity
        print("Heteroskedasticity:")
        if results.heteroskedasticity_test["heteroskedasticity_detected"]:
            print("  ⚠️ Heteroskedasticity detected")
        else:
            print("  ✅ Homoskedasticity assumption met")

        print()
        print("Recommendations:")
        for i, rec in enumerate(results.recommendations, 1):
            print(f"  {i}. {rec}")


# Convenience functions
def test_model_specification(
    outcome: OutcomeData,
    treatment: TreatmentData,
    covariates: CovariateData,
    verbose: bool = True,
) -> SpecificationResults:
    """Convenience function for comprehensive specification testing."""
    tester = ModelSpecificationTests()
    results = tester.comprehensive_specification_tests(outcome, treatment, covariates)

    if verbose:
        tester.print_specification_summary(results)

    return results
