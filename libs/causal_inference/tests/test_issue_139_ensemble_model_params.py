"""Tests for ensemble model-specific parameters feature (Issue #139)."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.g_computation import GComputationEstimator


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    n = 200

    X = np.random.randn(n, 3)
    treatment = np.random.binomial(1, 0.5, n)
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
        "true_ate": 2.0,
    }


def test_ensemble_model_specific_params(synthetic_data):
    """Test that ensemble_model_params correctly sets model-specific parameters."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_model_params={
            "ridge": {"alpha": 0.5},
            "random_forest": {"n_estimators": 25, "max_depth": 5},
        },
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Verify Ridge has custom alpha
    ridge_model = estimator.ensemble_models_fitted["ridge"]
    assert ridge_model.alpha == 0.5, "Ridge should have alpha=0.5"

    # Verify Random Forest has custom parameters
    rf_model = estimator.ensemble_models_fitted["random_forest"]
    assert rf_model.n_estimators == 25, "RF should have 25 estimators"
    assert rf_model.max_depth == 5, "RF should have max_depth=5"

    # Verify ATE is reasonable
    effect = estimator.estimate_ate()
    assert abs(effect.ate - synthetic_data["true_ate"]) < 0.5


def test_ensemble_model_params_precedence(synthetic_data):
    """Test that ensemble_model_params takes precedence over model_params."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["ridge", "random_forest"],
        model_params={},  # No global params
        ensemble_model_params={
            "ridge": {"alpha": 1.0},  # Ridge-specific
            "random_forest": {"n_estimators": 30, "max_depth": 3},  # RF-specific
        },
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Ridge should use ensemble_model_params
    ridge_model = estimator.ensemble_models_fitted["ridge"]
    assert ridge_model.alpha == 1.0, "Ridge should use ensemble_model_params alpha"

    # RF should have params from ensemble_model_params
    rf_model = estimator.ensemble_models_fitted["random_forest"]
    assert (
        rf_model.n_estimators == 30
    ), "RF should use ensemble_model_params n_estimators"
    assert (
        rf_model.max_depth == 3
    ), "RF should have max_depth from ensemble_model_params"


def test_ensemble_model_params_fallback(synthetic_data):
    """Test that models without specific params fall back to model_params."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        model_params={},  # No default params
        ensemble_model_params={
            "ridge": {"alpha": 0.3},  # Only ridge has custom params
        },
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Ridge has custom params
    ridge_model = estimator.ensemble_models_fitted["ridge"]
    assert ridge_model.alpha == 0.3

    # RF should use defaults (no custom params provided)
    rf_model = estimator.ensemble_models_fitted["random_forest"]
    assert rf_model.n_estimators == 100, "RF should use default n_estimators=100"


def test_backward_compatibility_model_params(synthetic_data):
    """Test that existing code using model_params still works for all compatible models."""
    # This test demonstrates backward compatibility: model_params applies to all models
    # when ensemble_model_params is not specified
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge"],  # Models that don't need params
        model_params={},  # Empty params - all models use defaults
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Both models should be fitted
    assert "linear" in estimator.ensemble_models_fitted
    assert "ridge" in estimator.ensemble_models_fitted

    # Ridge should use default alpha (1.0)
    ridge_model = estimator.ensemble_models_fitted["ridge"]
    assert ridge_model.alpha == 1.0, "Ridge should use default alpha"

    # Should produce reasonable ATE
    effect = estimator.estimate_ate()
    assert abs(effect.ate - synthetic_data["true_ate"]) < 0.6


def test_ensemble_model_params_with_bootstrap(synthetic_data):
    """Test that ensemble_model_params works with bootstrap."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_model_params={
            "ridge": {"alpha": 0.5},
            "random_forest": {"n_estimators": 25},
        },
        bootstrap_samples=50,  # Small number for fast test
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Verify models have correct params
    assert estimator.ensemble_models_fitted["ridge"].alpha == 0.5
    assert estimator.ensemble_models_fitted["random_forest"].n_estimators == 25

    # Estimate with bootstrap
    effect = estimator.estimate_ate()

    # Should have bootstrap CIs
    assert effect.ate_ci_lower is not None
    assert effect.ate_ci_upper is not None
    assert effect.ate_ci_lower < effect.ate < effect.ate_ci_upper
