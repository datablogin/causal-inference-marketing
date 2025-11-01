"""Negative control analysis for assumption testing.

This module provides functions to test causal inference assumptions using
negative control outcomes and exposures that should theoretically have
no causal relationship with the main treatment/outcome.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..core.base import CovariateData, OutcomeData, TreatmentData


def negative_control(
    treatment: Union[TreatmentData, NDArray[Any], pd.Series],
    outcome: Union[OutcomeData, NDArray[Any], pd.Series],
    negative_control_outcome: Optional[Union[NDArray[Any], pd.Series]] = None,
    negative_control_exposure: Optional[Union[NDArray[Any], pd.Series]] = None,
    covariates: Optional[Union[CovariateData, NDArray[Any], pd.DataFrame]] = None,
    alpha: float = 0.05,
    effect_threshold: float = 0.1,
    bootstrap_samples: int = 1000,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Perform negative control analysis to test causal inference assumptions.

    Negative controls are outcomes or exposures that should theoretically have
    no causal relationship with the main treatment or outcome. Finding
    significant associations suggests potential bias from unmeasured confounding,
    selection bias, or other threats to validity.

    Args:
        treatment: Main treatment variable
        outcome: Main outcome variable
        negative_control_outcome: Outcome that should not be affected by treatment
        negative_control_exposure: Exposure that should not affect main outcome
        covariates: Covariates to adjust for in models
        alpha: Significance level for hypothesis tests
        effect_threshold: Threshold for considering effects meaningful
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        random_state: Random state for reproducibility

    Returns:
        Dictionary containing:
            - negative_outcome_results: Results of treatment → negative outcome test
            - negative_exposure_results: Results of negative exposure → outcome test
            - overall_assessment: Summary of assumption violations
            - interpretation: Interpretation and recommendations
            - bias_indicators: Specific indicators of potential bias

    Raises:
        ValueError: If insufficient data or invalid negative controls provided

    Example:
        >>> import numpy as np
        >>> from causal_inference.sensitivity import negative_control
        >>>
        >>> # Simulate main variables
        >>> n = 1000
        >>> treatment = np.random.binomial(1, 0.5, n)
        >>> outcome = 2 * treatment + np.random.normal(0, 1, n)
        >>>
        >>> # Negative controls (should show no effect if assumptions hold)
        >>> neg_outcome = np.random.normal(0, 1, n)  # Unrelated to treatment
        >>> neg_exposure = np.random.binomial(1, 0.5, n)  # Unrelated to outcome
        >>>
        >>> results = negative_control(
        >>>     treatment=treatment,
        >>>     outcome=outcome,
        >>>     negative_control_outcome=neg_outcome,
        >>>     negative_control_exposure=neg_exposure
        >>> )
        >>> print(f"Assumption violations: {results['overall_assessment']}")

    Notes:
        Negative control outcomes should be:
        - Caused by the same confounders as the main outcome
        - NOT caused by the treatment
        - Examples: Past outcomes, related health conditions, etc.

        Negative control exposures should be:
        - Associated with the same confounders as the main treatment
        - NOT causally related to the main outcome
        - Examples: Future treatments, unrelated exposures, etc.
    """
    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Convert inputs
    if hasattr(treatment, "values"):
        t = np.asarray(treatment.values).flatten()
    else:
        t = np.asarray(treatment).flatten()

    if hasattr(outcome, "values"):
        y = np.asarray(outcome.values).flatten()
    else:
        y = np.asarray(outcome).flatten()

    n = len(t)
    if len(y) != n:
        raise ValueError("Treatment and outcome must have same length")

    # Handle covariates
    X = None
    if covariates is not None:
        if hasattr(covariates, "values"):
            X = np.asarray(covariates.values)
        else:
            X = np.asarray(covariates)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) != n:
            raise ValueError("Covariates must have same length as treatment/outcome")

    results = {}

    # Test 1: Treatment → Negative Control Outcome
    if negative_control_outcome is not None:
        neg_outcome = np.asarray(negative_control_outcome).flatten()
        if len(neg_outcome) != n:
            raise ValueError(
                "Negative control outcome must have same length as treatment"
            )

        neg_outcome_results = _test_treatment_negative_outcome(
            t, neg_outcome, X, alpha, effect_threshold, bootstrap_samples
        )
        results["negative_outcome_results"] = neg_outcome_results
    else:
        results["negative_outcome_results"] = None

    # Test 2: Negative Control Exposure → Main Outcome
    if negative_control_exposure is not None:
        neg_exposure = np.asarray(negative_control_exposure).flatten()
        if len(neg_exposure) != n:
            raise ValueError(
                "Negative control exposure must have same length as outcome"
            )

        neg_exposure_results = _test_negative_exposure_outcome(
            neg_exposure, y, X, alpha, effect_threshold, bootstrap_samples
        )
        results["negative_exposure_results"] = neg_exposure_results
    else:
        results["negative_exposure_results"] = None

    # Overall assessment
    violations = []
    bias_indicators = []

    if results["negative_outcome_results"]:
        neg_out = results["negative_outcome_results"]
        if neg_out["significant_effect"]:
            violations.append("Treatment affects negative control outcome")
            bias_indicators.append(
                f"Treatment → negative outcome effect: {neg_out['effect_estimate']:.3f} (p={neg_out['p_value']:.3f})"
            )

    if results["negative_exposure_results"]:
        neg_exp = results["negative_exposure_results"]
        if neg_exp["significant_effect"]:
            violations.append("Negative control exposure affects outcome")
            bias_indicators.append(
                f"Negative exposure → outcome effect: {neg_exp['effect_estimate']:.3f} (p={neg_exp['p_value']:.3f})"
            )

    # Generate interpretation
    if len(violations) == 0:
        overall_assessment = "No assumption violations detected"
        interpretation = (
            "Negative controls support the validity of causal inference assumptions. "
            "No evidence of substantial unmeasured confounding or selection bias."
        )
    elif len(violations) == 1:
        overall_assessment = f"1 assumption violation: {violations[0]}"
        interpretation = (
            "Some evidence of assumption violations. Consider additional controls, "
            "alternative identification strategies, or sensitivity analyses."
        )
    else:
        overall_assessment = f"{len(violations)} assumption violations detected"
        interpretation = (
            "Multiple assumption violations suggest substantial bias. "
            "Strong caution warranted - consider whether observational approach is appropriate."
        )

    results.update(
        {
            "overall_assessment": overall_assessment,
            "interpretation": interpretation,
            "bias_indicators": bias_indicators,
            "n_violations": len(violations),
        }
    )

    return results


def _test_treatment_negative_outcome(
    treatment: NDArray[Any],
    negative_outcome: NDArray[Any],
    covariates: Optional[NDArray[Any]],
    alpha: float,
    effect_threshold: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Test whether treatment affects negative control outcome."""
    # Determine if negative outcome is binary or continuous
    unique_values = np.unique(negative_outcome)
    is_binary = len(unique_values) == 2 and set(unique_values) == {0, 1}

    if covariates is not None:
        X = np.column_stack([treatment, covariates])
    else:
        X = treatment.reshape(-1, 1)

    if is_binary:
        # Logistic regression for binary outcome
        try:
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, negative_outcome)
            coef = model.coef_[0][0]  # Treatment coefficient

            # Wald test for significance

            predictions = model.predict_proba(X)
            # Simplified p-value calculation
            z_score = abs(coef) / (np.sqrt(np.var(predictions[:, 1])) + 1e-10)
            p_value = 2 * (1 - stats.norm.cdf(z_score))

        except (ValueError, np.linalg.LinAlgError):
            # Fallback to simple comparison
            treated_rate = np.mean(negative_outcome[treatment == 1])
            control_rate = np.mean(negative_outcome[treatment == 0])
            coef = treated_rate - control_rate

            # Chi-square test
            contingency = pd.crosstab(treatment, negative_outcome)
            chi2, p_value = stats.chi2_contingency(contingency)[:2]

    else:
        # Linear regression for continuous outcome
        model = LinearRegression()
        model.fit(X, negative_outcome)
        coef = model.coef_[0]  # Treatment coefficient

        # T-test for significance using proper standard error
        predictions = model.predict(X)
        residuals = negative_outcome - predictions
        mse = np.var(residuals)

        # Proper standard error calculation for OLS
        X_design = (
            np.column_stack([np.ones(len(X)), X])
            if X.shape[1] == 1
            else np.column_stack([np.ones(len(X)), X])
        )
        try:
            xtx_inv = np.linalg.inv(X_design.T @ X_design)
            se_coef = np.sqrt(
                mse * xtx_inv[1, 1]
            )  # SE for first coefficient (treatment)
        except np.linalg.LinAlgError:
            # Fallback to simplified calculation
            se_coef = np.sqrt(mse / np.var(treatment) / len(X))

        t_stat = coef / (se_coef + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(X) - X.shape[1] - 1))
        p_value = min(p_value, 1.0)  # Ensure p-value ≤ 1

    # Bootstrap confidence interval
    bootstrap_effects = []
    for _ in range(bootstrap_samples):
        idx = np.random.choice(len(treatment), len(treatment), replace=True)
        treatment[idx]
        y_boot = negative_outcome[idx]
        X_boot = X[idx]

        try:
            if is_binary:
                model_boot = LogisticRegression(random_state=None, max_iter=1000)
                model_boot.fit(X_boot, y_boot)
                bootstrap_effects.append(model_boot.coef_[0][0])
            else:
                model_boot = LinearRegression()
                model_boot.fit(X_boot, y_boot)
                bootstrap_effects.append(model_boot.coef_[0])
        except Exception:
            continue

    if len(bootstrap_effects) > 10:
        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)
    else:
        ci_lower = ci_upper = coef

    significant_effect = p_value < alpha and abs(coef) > effect_threshold

    return {
        "effect_estimate": float(coef),
        "p_value": float(p_value),
        "confidence_interval": [float(ci_lower), float(ci_upper)],
        "significant_effect": significant_effect,
        "outcome_type": "binary" if is_binary else "continuous",
        "interpretation": "Significant treatment effect on negative outcome - possible confounding"
        if significant_effect
        else "No significant treatment effect on negative outcome",
    }


def _test_negative_exposure_outcome(
    negative_exposure: NDArray[Any],
    outcome: NDArray[Any],
    covariates: Optional[NDArray[Any]],
    alpha: float,
    effect_threshold: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Test whether negative control exposure affects main outcome."""
    # Determine if outcome is binary or continuous
    unique_values = np.unique(outcome)
    is_binary = len(unique_values) == 2 and set(unique_values) == {0, 1}

    if covariates is not None:
        X = np.column_stack([negative_exposure, covariates])
    else:
        X = negative_exposure.reshape(-1, 1)

    if is_binary:
        # Logistic regression for binary outcome
        try:
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, outcome)
            coef = model.coef_[0][0]  # Exposure coefficient

            # Simplified p-value calculation
            predictions = model.predict_proba(X)
            z_score = abs(coef) / (np.sqrt(np.var(predictions[:, 1])) + 1e-10)
            p_value = 2 * (1 - stats.norm.cdf(z_score))

        except (ValueError, np.linalg.LinAlgError):
            # Fallback to simple comparison
            exposed_rate = np.mean(outcome[negative_exposure == 1])
            unexposed_rate = np.mean(outcome[negative_exposure == 0])
            coef = exposed_rate - unexposed_rate

            # Chi-square test
            contingency = pd.crosstab(negative_exposure, outcome)
            chi2, p_value = stats.chi2_contingency(contingency)[:2]

    else:
        # Linear regression for continuous outcome
        model = LinearRegression()
        model.fit(X, outcome)
        coef = model.coef_[0]  # Exposure coefficient

        # T-test for significance
        predictions = model.predict(X)
        residuals = outcome - predictions
        se = np.sqrt(np.var(residuals) / len(X))
        t_stat = coef / (se + 1e-10)
        p_value = 2 * (1 - stats.t.sf(abs(t_stat), len(X) - X.shape[1]))

    # Bootstrap confidence interval
    bootstrap_effects = []
    for _ in range(bootstrap_samples):
        idx = np.random.choice(
            len(negative_exposure), len(negative_exposure), replace=True
        )
        negative_exposure[idx]
        y_boot = outcome[idx]
        X_boot = X[idx]

        try:
            if is_binary:
                model_boot = LogisticRegression(random_state=None, max_iter=1000)
                model_boot.fit(X_boot, y_boot)
                bootstrap_effects.append(model_boot.coef_[0][0])
            else:
                model_boot = LinearRegression()
                model_boot.fit(X_boot, y_boot)
                bootstrap_effects.append(model_boot.coef_[0])
        except Exception:
            continue

    if len(bootstrap_effects) > 10:
        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)
    else:
        ci_lower = ci_upper = coef

    significant_effect = p_value < alpha and abs(coef) > effect_threshold

    return {
        "effect_estimate": float(coef),
        "p_value": float(p_value),
        "confidence_interval": [float(ci_lower), float(ci_upper)],
        "significant_effect": significant_effect,
        "outcome_type": "binary" if is_binary else "continuous",
        "interpretation": "Significant negative exposure effect - possible confounding"
        if significant_effect
        else "No significant negative exposure effect",
    }
