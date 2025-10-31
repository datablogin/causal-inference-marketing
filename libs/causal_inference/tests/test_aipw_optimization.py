"""Tests for AIPW estimator with component balance optimization."""

import numpy as np
import pytest

from causal_inference.core.base import (
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.aipw import AIPWEstimator


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known treatment effect."""
    np.random.seed(42)
    n = 1000

    # Covariates
    X = np.random.randn(n, 3)

    # Treatment with confounding
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = np.random.binomial(1, propensity)

    # Outcome with treatment effect = 2.0
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


def test_component_optimization(synthetic_data):
    """Test AIPW component balance optimization."""
    # Standard AIPW
    estimator_standard = AIPWEstimator(cross_fitting=True, n_folds=3, random_state=42)
    estimator_standard.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Optimized AIPW
    estimator_optimized = AIPWEstimator(
        cross_fitting=True,
        n_folds=3,
        optimize_component_balance=True,
        component_variance_penalty=0.5,
        influence_function_se=False,  # Required with optimization - use bootstrap instead
        bootstrap_samples=100,  # Reduced for faster tests
        random_state=42,
        verbose=True,
    )
    estimator_optimized.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Estimate ATE (this triggers optimization)
    effect_opt = estimator_optimized.estimate_ate()

    # Check optimization diagnostics
    opt_diag = estimator_optimized.get_optimization_diagnostics()
    assert opt_diag is not None
    assert "optimal_g_computation_weight" in opt_diag
    assert 0.3 <= opt_diag["optimal_g_computation_weight"] <= 0.7

    # Check variance reduction
    assert (
        opt_diag["optimized_estimator_variance"]
        <= opt_diag["standard_estimator_variance"]
    )

    # Both should recover true effect reasonably well
    effect_std = estimator_standard.estimate_ate()

    # Standard AIPW should be close to true effect
    assert abs(effect_std.ate - synthetic_data["true_ate"]) < 0.5

    # Optimized AIPW may have slightly different bias-variance tradeoff
    # but should still be reasonable
    assert abs(effect_opt.ate - synthetic_data["true_ate"]) < 1.0


def test_component_weights_valid(synthetic_data):
    """Test that component weights are in valid range [0.3, 0.7]."""
    estimator = AIPWEstimator(
        cross_fitting=True,
        n_folds=3,
        optimize_component_balance=True,
        component_variance_penalty=0.5,
        influence_function_se=False,
        bootstrap_samples=100,
        random_state=42,
    )
    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Trigger optimization
    estimator.estimate_ate()

    opt_diag = estimator.get_optimization_diagnostics()
    assert opt_diag is not None

    # Check that weights are in valid range
    g_comp_weight = opt_diag["optimal_g_computation_weight"]
    ipw_weight = opt_diag["optimal_ipw_weight"]

    assert 0.3 <= g_comp_weight <= 0.7
    assert 0.3 <= ipw_weight <= 0.7

    # Check that weights sum to approximately 1
    assert abs((g_comp_weight + ipw_weight) - 1.0) < 1e-6


def test_backward_compatibility(synthetic_data):
    """Test that AIPW without optimization still works."""
    estimator = AIPWEstimator(cross_fitting=True, n_folds=3, random_state=42)
    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    effect = estimator.estimate_ate()
    assert abs(effect.ate - synthetic_data["true_ate"]) < 0.5

    # Should not have optimization diagnostics
    opt_diag = estimator.get_optimization_diagnostics()
    assert opt_diag == {} or opt_diag is None


def test_optimization_with_no_cross_fitting(synthetic_data):
    """Test component optimization without cross-fitting."""
    estimator = AIPWEstimator(
        cross_fitting=False,
        optimize_component_balance=True,
        component_variance_penalty=0.5,
        influence_function_se=False,
        bootstrap_samples=100,
        random_state=42,
        verbose=True,
    )
    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    effect = estimator.estimate_ate()
    opt_diag = estimator.get_optimization_diagnostics()

    assert opt_diag is not None
    assert "optimal_g_computation_weight" in opt_diag
    # Component optimization may adjust bias-variance tradeoff
    assert abs(effect.ate - synthetic_data["true_ate"]) < 1.0


def test_optimization_reduces_variance(synthetic_data):
    """Test that optimization reduces variance compared to standard AIPW."""
    # Just test that the optimization reports variance reduction in diagnostics
    estimator = AIPWEstimator(
        cross_fitting=True,
        n_folds=3,
        optimize_component_balance=True,
        component_variance_penalty=0.5,
        influence_function_se=False,
        bootstrap_samples=100,
        random_state=42,
    )
    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )
    estimator.estimate_ate()

    opt_diag = estimator.get_optimization_diagnostics()
    assert opt_diag is not None

    # The optimization objective includes variance reduction
    # So optimized_estimator_variance should be <= standard_estimator_variance
    assert (
        opt_diag["optimized_estimator_variance"]
        <= opt_diag["standard_estimator_variance"]
    )


def test_influence_function_with_optimization_raises_error(synthetic_data):
    """Test that using influence_function_se with optimization raises error."""
    # Should raise ValueError at initialization
    with pytest.raises(ValueError, match="Cannot use influence_function_se=True"):
        AIPWEstimator(
            cross_fitting=True,
            n_folds=3,
            optimize_component_balance=True,
            influence_function_se=True,  # Invalid combination
            random_state=42,
        )


def test_negative_variance_penalty_raises_error():
    """Test that negative component_variance_penalty raises error."""
    with pytest.raises(
        ValueError, match="component_variance_penalty must be non-negative"
    ):
        AIPWEstimator(
            optimize_component_balance=True,
            component_variance_penalty=-0.5,  # Invalid - must be >= 0
        )
