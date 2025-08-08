"""Synthetic data generation for Double Machine Learning testing.

This module implements the synthetic data generator specified in the DoubleML
framework for testing statistical properties of DML estimators.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["generate_synthetic_dml_data"]


def generate_synthetic_dml_data(
    n: int = 1000,
    n_features: int = 5,
    true_ate: float = 2.0,
    confounding_strength: float = 1.0,
    noise_level: float = 1.0,
    seed: int = 42,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.integer[Any]],
    NDArray[np.floating[Any]],
    float,
]:
    """Generate synthetic data for Double Machine Learning testing.

    This function creates synthetic data where the true Average Treatment Effect (ATE)
    is known, enabling validation of DML estimators' bias, coverage rates, and other
    statistical properties as specified in Chernozhukov et al. (2018).

    The data generation process follows the DoubleML specification:
    - Generate covariates X with configurable dimensionality
    - Create realistic propensity score function e(X) = 1/(1 + exp(-X[0]))
    - Generate treatment assignment D ~ Bernoulli(e(X))
    - Create outcome Y = true_ate * D + confounding_function(X) + noise

    Args:
        n: Number of observations to generate
        n_features: Number of covariate features
        true_ate: True average treatment effect (target parameter)
        confounding_strength: Strength of confounding relationships between X and Y
        noise_level: Standard deviation of noise term in outcome equation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, D, Y, true_ate) where:
        - X: Covariate matrix of shape (n, n_features)
        - D: Binary treatment vector of shape (n,)
        - Y: Continuous outcome vector of shape (n,)
        - true_ate: The true average treatment effect used for validation

    Example:
        >>> X, D, Y, true_ate = generate_synthetic_dml_data(n=1000, true_ate=2.0)
        >>> print(f"Generated {len(X)} observations with true ATE = {true_ate}")
        Generated 1000 observations with true ATE = 2.0

    References:
        Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
        Newey, W., & Robins, J. (2018). Double/debiased machine learning for
        treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate covariates X from standard normal distribution
    # This provides a realistic baseline for testing ML methods
    X = np.random.normal(size=(n, n_features))

    # Generate propensity scores e(X) = P(D=1|X)
    # Use first covariate as main driver of treatment selection
    # This creates realistic confounding for testing DML robustness
    propensity_logits = X[:, 0]  # Simple linear propensity function
    if n_features > 1:
        # Add nonlinear terms for more realistic confounding
        propensity_logits += 0.5 * X[:, 1]
        if n_features > 2:
            propensity_logits += 0.3 * X[:, 2]

    # Convert to probabilities using logistic function
    e = 1 / (1 + np.exp(-propensity_logits))

    # Generate binary treatment assignment D ~ Bernoulli(e(X))
    D = np.random.binomial(1, e)

    # Generate outcome Y with confounding and treatment effect
    # Confounding function: linear combination of covariates
    confounding_function = confounding_strength * X[:, 0]
    if n_features > 1:
        confounding_function += 0.8 * confounding_strength * X[:, 1]
        if n_features > 2:
            # Add interaction effects for more realistic relationships
            confounding_function += 0.5 * confounding_strength * X[:, 2]
            if n_features > 3:
                confounding_function += 0.3 * confounding_strength * X[:, 0] * X[:, 3]

    # Treatment effect (homogeneous for simplicity)
    treatment_effect = true_ate * D

    # Noise term
    noise = np.random.normal(0, noise_level, n)

    # Final outcome equation
    Y = confounding_function + treatment_effect + noise

    return X, D, Y, true_ate


def generate_heterogeneous_ate_data(
    n: int = 1000,
    n_features: int = 5,
    base_ate: float = 2.0,
    heterogeneity_strength: float = 1.0,
    noise_level: float = 1.0,
    seed: int = 42,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.integer[Any]],
    NDArray[np.floating[Any]],
    float,
]:
    """Generate synthetic data with heterogeneous treatment effects.

    This variant creates data where the treatment effect varies across individuals
    based on their covariates, which is useful for testing CATE estimation methods.

    Args:
        n: Number of observations
        n_features: Number of covariate features
        base_ate: Base average treatment effect
        heterogeneity_strength: How much treatment effects vary across individuals
        noise_level: Standard deviation of noise term
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, D, Y, average_ate) where average_ate is the population average
    """
    # Set random seed
    np.random.seed(seed)

    # Generate covariates
    X = np.random.normal(size=(n, n_features))

    # Generate propensity scores
    propensity_logits = 0.5 * X[:, 0]
    if n_features > 1:
        propensity_logits += 0.3 * X[:, 1]

    e = 1 / (1 + np.exp(-propensity_logits))
    D = np.random.binomial(1, e)

    # Heterogeneous treatment effects based on covariates
    # τ(x) = base_ate + heterogeneity_strength * f(x)
    if n_features >= 2:
        individual_effects = base_ate + heterogeneity_strength * (
            0.5 * X[:, 0] + 0.3 * X[:, 1]
        )
        if n_features >= 3:
            individual_effects += heterogeneity_strength * 0.2 * X[:, 2]
    else:
        individual_effects = base_ate + heterogeneity_strength * 0.5 * X[:, 0]

    # Confounding function (same as homogeneous case)
    confounding_function = X[:, 0]
    if n_features > 1:
        confounding_function += 0.8 * X[:, 1]

    # Treatment effects (now heterogeneous)
    treatment_effects = individual_effects * D

    # Noise
    noise = np.random.normal(0, noise_level, n)

    # Outcome
    Y = confounding_function + treatment_effects + noise

    # Return population average treatment effect
    population_ate = np.mean(individual_effects)

    return X, D, Y, population_ate


def generate_instrumental_variable_data(
    n: int = 1000,
    n_features: int = 5,
    true_ate: float = 2.0,
    instrument_strength: float = 1.0,
    confounding_strength: float = 1.0,
    noise_level: float = 1.0,
    seed: int = 42,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.integer[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    float,
]:
    """Generate synthetic data with instrumental variable for IV-DML testing.

    Args:
        n: Number of observations
        n_features: Number of covariate features
        true_ate: True average treatment effect
        instrument_strength: Strength of instrument (first-stage F-statistic related)
        confounding_strength: Strength of unobserved confounding
        noise_level: Standard deviation of noise terms
        seed: Random seed

    Returns:
        Tuple of (X, D, Y, Z, true_ate) where Z is the instrumental variable
    """
    np.random.seed(seed)

    # Generate covariates and unobserved confounder
    X = np.random.normal(size=(n, n_features))
    U = np.random.normal(size=n)  # Unobserved confounder

    # Generate instrument Z (should be correlated with D but not with outcome directly)
    Z = np.random.normal(size=n)

    # First stage: D = f(Z, X, U) + error
    # Instrument affects treatment, plus confounding through U
    treatment_logits = (
        instrument_strength * Z + 0.5 * X[:, 0] + confounding_strength * U
    )
    if n_features > 1:
        treatment_logits += 0.3 * X[:, 1]

    # Binary treatment
    treatment_probs = 1 / (1 + np.exp(-treatment_logits))
    D = np.random.binomial(1, treatment_probs)

    # Outcome equation: Y = true_ate * D + confounders + noise
    # Important: Z should NOT appear directly in outcome equation
    confounding_function = 0.8 * X[:, 0] + confounding_strength * U
    if n_features > 1:
        confounding_function += 0.5 * X[:, 1]

    treatment_effect = true_ate * D
    noise = np.random.normal(0, noise_level, n)

    Y = confounding_function + treatment_effect + noise

    return X, D, Y, Z, true_ate


def generate_time_series_dml_data(
    n_time_periods: int = 100,
    n_units: int = 50,
    true_ate: float = 2.0,
    time_trend_strength: float = 0.5,
    unit_effects_strength: float = 1.0,
    noise_level: float = 1.0,
    seed: int = 42,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.integer[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.integer[Any]],
    NDArray[np.integer[Any]],
    float,
]:
    """Generate panel/time series data for dynamic DML testing.

    Args:
        n_time_periods: Number of time periods
        n_units: Number of cross-sectional units
        true_ate: True average treatment effect
        time_trend_strength: Strength of time trends
        unit_effects_strength: Strength of unit fixed effects
        noise_level: Standard deviation of noise
        seed: Random seed

    Returns:
        Tuple of (X, D, Y, unit_ids, time_ids, true_ate) for panel DML
    """
    np.random.seed(seed)

    n_obs = n_time_periods * n_units

    # Create panel structure
    unit_ids = np.repeat(np.arange(n_units), n_time_periods)
    time_ids = np.tile(np.arange(n_time_periods), n_units)

    # Generate time-varying covariates
    X = np.random.normal(size=(n_obs, 3))

    # Add time trends
    time_trend = time_trend_strength * (time_ids / n_time_periods)
    X[:, 0] += time_trend

    # Add unit fixed effects
    unit_effects = np.random.normal(0, unit_effects_strength, n_units)
    X[:, 1] += unit_effects[unit_ids]

    # Treatment assignment (depends on lagged outcomes and covariates)
    treatment_logits = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * time_trend
    treatment_probs = 1 / (1 + np.exp(-treatment_logits))
    D = np.random.binomial(1, treatment_probs)

    # Outcome with unit and time effects
    Y = (
        true_ate * D
        + 0.8 * X[:, 0]  # Time-varying covariate effect
        + unit_effects[unit_ids]  # Unit fixed effects
        + 0.5 * time_trend  # Time trend
        + np.random.normal(0, noise_level, n_obs)
    )

    return X, D, Y, unit_ids, time_ids, true_ate
