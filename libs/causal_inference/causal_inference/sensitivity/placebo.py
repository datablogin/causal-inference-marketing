"""Placebo tests for causal inference assumption validation.

This module provides placebo testing functions that use dummy treatments
and outcomes to test whether the analysis would detect effects where none
should exist, helping to validate identification assumptions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from ..core.base import CovariateData, OutcomeData, TreatmentData


def placebo_test(
    treatment: TreatmentData | NDArray[Any] | pd.Series,
    outcome: OutcomeData | NDArray[Any] | pd.Series,
    covariates: CovariateData | NDArray[Any] | pd.DataFrame | None = None,
    estimator: Callable | None = None,
    placebo_type: str = "random_treatment",
    n_placebo_tests: int = 100,
    alpha: float = 0.05,
    effect_threshold: float = 0.1,
    random_state: int | None = None,
    **estimator_kwargs: Any,
) -> dict[str, Any]:
    """Perform placebo tests to validate causal inference assumptions.

    Placebo tests use dummy treatments or outcomes to test whether the
    analysis methodology would incorrectly detect causal effects where
    none should exist. This helps validate identification assumptions
    and detect specification errors.

    Args:
        treatment: Original treatment variable
        outcome: Original outcome variable
        covariates: Covariates to include in models
        estimator: Causal inference estimator function to use
        placebo_type: Type of placebo test to perform:
            - 'random_treatment': Randomized treatment assignment
            - 'future_outcome': Use future/lagged outcome as placebo
            - 'past_treatment': Use pre-treatment period assignment
            - 'dummy_outcome': Random outcome unrelated to treatment
        n_placebo_tests: Number of placebo tests to run
        alpha: Significance level for detecting false positives
        effect_threshold: Minimum effect size to consider meaningful
        random_state: Random state for reproducibility
        **estimator_kwargs: Additional arguments for estimator

    Returns:
        Dictionary containing:
            - placebo_effects: List of estimated effects from placebo tests
            - p_values: List of p-values from placebo tests
            - false_positive_rate: Proportion of significant placebo effects
            - mean_placebo_effect: Average placebo effect size
            - placebo_effect_distribution: Summary statistics of placebo effects
            - passes_placebo_test: Whether analysis passes placebo validation
            - interpretation: Interpretation and recommendations
            - individual_results: Detailed results from each placebo test

    Raises:
        ValueError: If inputs are invalid or estimator fails

    Example:
        >>> import numpy as np
        >>> from causal_inference.sensitivity import placebo_test
        >>> from causal_inference.estimators import GComputationEstimator
        >>>
        >>> # Simulate data with true treatment effect
        >>> n = 1000
        >>> X = np.random.normal(0, 1, (n, 2))
        >>> T = np.random.binomial(1, 0.5, n)
        >>> Y = 2 * T + X.sum(axis=1) + np.random.normal(0, 1, n)
        >>>
        >>> # Placebo test with random treatment assignment
        >>> results = placebo_test(
        >>>     treatment=T,
        >>>     outcome=Y,
        >>>     covariates=X,
        >>>     placebo_type="random_treatment",
        >>>     n_placebo_tests=50
        >>> )
        >>> print(f"False positive rate: {results['false_positive_rate']:.3f}")
        >>> print(f"Passes placebo test: {results['passes_placebo_test']}")

    Notes:
        Different placebo types test different assumptions:

        - Random treatment: Tests whether analysis finds effects with purely
          random treatment assignment (should find no effects)
        - Future outcome: Tests whether treatment predicts future outcomes
          that occurred before treatment (should find no effects)
        - Past treatment: Tests whether past treatment assignments predict
          current outcomes after controlling for confounders
        - Dummy outcome: Tests whether treatment predicts unrelated outcomes

        A good analysis should pass placebo tests by showing:
        - Low false positive rate (< 5-10%)
        - Small average placebo effect sizes
        - Placebo effects distributed around zero
    """
    # Validate placebo type early
    valid_placebo_types = {
        "random_treatment",
        "future_outcome",
        "past_treatment",
        "dummy_outcome",
    }
    if placebo_type not in valid_placebo_types:
        raise ValueError(
            f"Unknown placebo_type: {placebo_type}. Must be one of: {sorted(valid_placebo_types)}"
        )

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Convert inputs
    if hasattr(treatment, "values"):
        t_original = np.asarray(treatment.values).flatten()
    else:
        t_original = np.asarray(treatment).flatten()

    if hasattr(outcome, "values"):
        y_original = np.asarray(outcome.values).flatten()
    else:
        y_original = np.asarray(outcome).flatten()

    n = len(t_original)
    if len(y_original) != n:
        raise ValueError("Treatment and outcome must have same length")

    # Handle covariates
    X_original = None
    if covariates is not None:
        if hasattr(covariates, "values"):
            X_original = np.asarray(covariates.values)
        else:
            X_original = np.asarray(covariates)
        if X_original.ndim == 1:
            X_original = X_original.reshape(-1, 1)
        if len(X_original) != n:
            raise ValueError("Covariates must have same length as treatment/outcome")

    # Default estimator (simple difference in means)
    if estimator is None:
        estimator = _simple_difference_estimator

    # Run placebo tests
    placebo_effects = []
    p_values = []
    individual_results = []

    for i in range(n_placebo_tests):
        try:
            # Generate placebo treatment/outcome based on type
            if placebo_type == "random_treatment":
                t_placebo, y_placebo = _generate_random_treatment_placebo(
                    t_original, y_original
                )
            elif placebo_type == "future_outcome":
                t_placebo, y_placebo = _generate_future_outcome_placebo(
                    t_original, y_original
                )
            elif placebo_type == "past_treatment":
                t_placebo, y_placebo = _generate_past_treatment_placebo(
                    t_original, y_original
                )
            elif placebo_type == "dummy_outcome":
                t_placebo, y_placebo = _generate_dummy_outcome_placebo(
                    t_original, y_original
                )
            else:
                raise ValueError(f"Unknown placebo_type: {placebo_type}")

            # Estimate placebo effect
            if X_original is not None:
                result = estimator(t_placebo, y_placebo, X_original, **estimator_kwargs)
            else:
                result = estimator(t_placebo, y_placebo, None, **estimator_kwargs)

            if isinstance(result, dict):
                effect = result.get("effect", result.get("ate", 0))
                p_val = result.get("p_value", result.get("pvalue", 1.0))
            elif hasattr(result, "ate"):
                effect = result.ate
                p_val = getattr(result, "p_value", 1.0)
            else:
                effect = float(result)
                p_val = 1.0  # Conservative if no p-value available

            placebo_effects.append(effect)
            p_values.append(p_val)
            individual_results.append(
                {
                    "test_id": i,
                    "effect": effect,
                    "p_value": p_val,
                    "significant": p_val < alpha and abs(effect) > effect_threshold,
                }
            )

        except Exception as e:
            # Handle failed placebo tests
            individual_results.append(
                {
                    "test_id": i,
                    "effect": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "error": str(e),
                }
            )

    # Calculate summary statistics
    valid_effects = [e for e in placebo_effects if not np.isnan(e)]
    valid_p_values = [p for p in p_values if not np.isnan(p)]

    if len(valid_effects) == 0:
        raise ValueError("All placebo tests failed - check estimator and data")

    false_positive_rate = np.mean(
        [
            p < alpha and abs(e) > effect_threshold
            for e, p in zip(valid_effects, valid_p_values)
        ]
    )

    mean_placebo_effect = np.mean(valid_effects)

    placebo_distribution = {
        "mean": float(np.mean(valid_effects)),
        "std": float(np.std(valid_effects)),
        "median": float(np.median(valid_effects)),
        "q25": float(np.percentile(valid_effects, 25)),
        "q75": float(np.percentile(valid_effects, 75)),
        "min": float(np.min(valid_effects)),
        "max": float(np.max(valid_effects)),
    }

    # Assess whether analysis passes placebo test
    passes_test = (
        false_positive_rate <= 0.1  # Low false positive rate
        and abs(mean_placebo_effect) <= effect_threshold  # Small average effect
        and placebo_distribution["std"] <= 2 * effect_threshold  # Reasonable variation
    )

    # Generate interpretation
    if passes_test:
        interpretation = (
            f"✅ Analysis passes placebo test. "
            f"False positive rate: {false_positive_rate:.1%}, "
            f"mean placebo effect: {mean_placebo_effect:.3f}. "
            f"Results suggest identification assumptions are reasonable."
        )
    elif false_positive_rate > 0.2:
        interpretation = (
            f"❌ High false positive rate ({false_positive_rate:.1%}). "
            f"Analysis may be detecting spurious effects. "
            f"Review identification assumptions and model specification."
        )
    elif abs(mean_placebo_effect) > effect_threshold:
        interpretation = (
            f"⚠️ Large average placebo effect ({mean_placebo_effect:.3f}). "
            f"Possible systematic bias in estimation procedure. "
            f"Consider alternative specifications or estimators."
        )
    else:
        interpretation = (
            f"⚠️ Placebo test shows mixed results. "
            f"False positive rate: {false_positive_rate:.1%}, "
            f"mean effect: {mean_placebo_effect:.3f}. "
            f"Interpret main results with caution."
        )

    return {
        "placebo_type": placebo_type,
        "n_tests_run": len(valid_effects),
        "n_tests_requested": n_placebo_tests,
        "placebo_effects": valid_effects,
        "p_values": valid_p_values,
        "false_positive_rate": float(false_positive_rate),
        "mean_placebo_effect": float(mean_placebo_effect),
        "placebo_effect_distribution": placebo_distribution,
        "passes_placebo_test": passes_test,
        "interpretation": interpretation,
        "individual_results": individual_results,
    }


def _simple_difference_estimator(
    treatment: NDArray[Any],
    outcome: NDArray[Any],
    covariates: NDArray[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Simple difference in means estimator for placebo tests."""
    treated_outcomes = outcome[treatment == 1]
    control_outcomes = outcome[treatment == 0]

    if len(treated_outcomes) == 0 or len(control_outcomes) == 0:
        return {"effect": 0.0, "p_value": 1.0}

    effect = np.mean(treated_outcomes) - np.mean(control_outcomes)

    # T-test for significance
    t_stat, p_val = stats.ttest_ind(treated_outcomes, control_outcomes)

    return {"effect": float(effect), "p_value": float(p_val)}


def _generate_random_treatment_placebo(
    treatment: NDArray[Any], outcome: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate placebo with randomized treatment assignment."""
    n = len(treatment)

    # Preserve treatment probability but randomize assignment
    treatment_prob = np.mean(treatment)
    placebo_treatment = np.random.binomial(1, treatment_prob, n)

    return placebo_treatment, outcome


def _generate_future_outcome_placebo(
    treatment: NDArray[Any], outcome: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate placebo using permuted future outcomes."""
    # Simulate "future" outcomes by permuting current outcomes
    placebo_outcome = np.random.permutation(outcome)

    return treatment, placebo_outcome


def _generate_past_treatment_placebo(
    treatment: NDArray[Any], outcome: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate placebo using permuted past treatment."""
    # Simulate "past" treatment by permuting current treatment
    placebo_treatment = np.random.permutation(treatment)

    return placebo_treatment, outcome


def _generate_dummy_outcome_placebo(
    treatment: NDArray[Any], outcome: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate placebo with dummy outcome unrelated to treatment."""
    # Generate random outcome with similar distribution to original
    outcome_mean = np.mean(outcome)
    outcome_std = np.std(outcome)

    # Check if outcome is binary
    unique_vals = np.unique(outcome)
    if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
        # Binary outcome - preserve prevalence
        prevalence = np.mean(outcome)
        placebo_outcome = np.random.binomial(1, prevalence, len(outcome))
    else:
        # Continuous outcome - preserve distribution
        placebo_outcome = np.random.normal(outcome_mean, outcome_std, len(outcome))

    return treatment, placebo_outcome
