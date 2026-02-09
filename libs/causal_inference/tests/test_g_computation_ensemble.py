"""Tests for G-computation ensemble functionality."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.g_computation import GComputationEstimator


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known treatment effect."""
    np.random.seed(42)
    n = 500

    # Covariates
    X = np.random.randn(n, 3)

    # Treatment with confounding
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = np.random.binomial(1, propensity)

    # Outcome with treatment effect = 2.0
    true_ate = 2.0
    outcome = (
        2.0 * treatment
        + X[:, 0]
        + 0.5 * X[:, 1]
        + 0.3 * X[:, 2]
        + np.random.randn(n) * 0.5
    )

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=X, names=["X1", "X2", "X3"]),
        "true_ate": true_ate,
    }


def test_ensemble_basic(synthetic_data):
    """Test that ensemble G-computation runs without errors."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_variance_penalty=0.1,
        random_state=42,
        verbose=True,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Check that models were fitted
    assert len(estimator.ensemble_models_fitted) > 0
    assert estimator.ensemble_weights is not None

    # Estimate ATE
    effect = estimator.estimate_ate()
    assert effect.ate is not None
    assert abs(effect.ate - synthetic_data["true_ate"]) < 1.0  # Reasonable estimate


def test_ensemble_weights_valid(synthetic_data):
    """Test that ensemble weights sum to 1 and are non-negative."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_variance_penalty=0.1,
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Check weights sum to 1
    assert abs(np.sum(estimator.ensemble_weights) - 1.0) < 1e-6

    # Check weights are non-negative
    assert np.all(estimator.ensemble_weights >= 0)


def test_ensemble_vs_single_model(synthetic_data):
    """Test that both ensemble and single model work and recover true effect."""
    # Single model
    estimator_single = GComputationEstimator(
        model_type="linear", use_ensemble=False, random_state=42
    )
    estimator_single.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )
    effect_single = estimator_single.estimate_ate()

    # Ensemble
    estimator_ensemble = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_variance_penalty=0.1,
        random_state=42,
    )
    estimator_ensemble.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )
    effect_ensemble = estimator_ensemble.estimate_ate()

    # Both should recover true effect reasonably
    assert abs(effect_single.ate - synthetic_data["true_ate"]) < 1.0
    assert abs(effect_ensemble.ate - synthetic_data["true_ate"]) < 1.0


def test_ensemble_optimization_diagnostics(synthetic_data):
    """Test that ensemble optimization diagnostics are stored."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_variance_penalty=0.1,
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Check diagnostics
    diag = estimator.get_optimization_diagnostics()
    assert diag is not None
    assert "ensemble_success" in diag
    assert "ensemble_objective" in diag
    assert "ensemble_weights" in diag


def test_ensemble_fallback_to_single_model(synthetic_data):
    """Test that ensemble falls back to single model if only one model fits."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear"],  # Only one model
        ensemble_variance_penalty=0.1,
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Should fall back to single model
    assert not estimator.use_ensemble
    assert estimator.outcome_model is not None


def test_ensemble_with_bootstrap(synthetic_data):
    """Test that ensemble works with bootstrap confidence intervals."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge"],
        ensemble_variance_penalty=0.1,
        bootstrap_samples=50,  # Small for speed
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    effect = estimator.estimate_ate()

    # Should have bootstrap CIs
    assert effect.ate_ci_lower is not None
    assert effect.ate_ci_upper is not None
    assert effect.ate_ci_lower < effect.ate < effect.ate_ci_upper

