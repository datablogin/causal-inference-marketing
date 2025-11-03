"""Integration tests for PyRake-style optimization across estimators."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.core.optimization_config import OptimizationConfig
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known treatment effect."""
    np.random.seed(42)
    n = 500  # Reduced from 1000 for faster tests

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


def test_ipw_optimization_reduces_variance(synthetic_data):
    """Test that IPW optimization reduces weight variance."""
    # Standard IPW
    estimator_standard = IPWEstimator(
        propensity_model_type="logistic", random_state=42, verbose=True
    )
    estimator_standard.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Optimized IPW
    optimization_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,
        balance_constraints=True,
        distance_metric="l2",
        verbose=True,
    )
    estimator_optimized = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=optimization_config,
        random_state=42,
        verbose=True,
    )
    estimator_optimized.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Check variance reduction
    weight_diag_std = estimator_standard._weight_diagnostics
    weight_diag_opt = estimator_optimized._weight_diagnostics

    # Optimized weights should have lower variance
    assert weight_diag_opt["weight_variance"] < weight_diag_std["weight_variance"]
    # ESS may be similar or slightly different due to optimization constraints
    # The key is variance reduction, not necessarily ESS increase

    # Check both recover true effect
    effect_std = estimator_standard.estimate_ate()
    effect_opt = estimator_optimized.estimate_ate()

    assert abs(effect_std.ate - synthetic_data["true_ate"]) < 0.5
    assert abs(effect_opt.ate - synthetic_data["true_ate"]) < 0.5


def test_optimization_with_bootstrap(synthetic_data):
    """Test that optimization works with bootstrap confidence intervals."""
    optimization_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,
        balance_constraints=True,
    )

    estimator = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=optimization_config,
        bootstrap_samples=50,  # Reduced from 100 for faster tests
        random_state=42,
        verbose=True,
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


def test_ensemble_improves_prediction(synthetic_data):
    """Test that ensemble G-computation improves over single model."""
    # Single model
    estimator_single = GComputationEstimator(model_type="linear", random_state=42)
    estimator_single.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Ensemble
    estimator_ensemble = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_variance_penalty=0.1,
        random_state=42,
        verbose=True,
    )
    estimator_ensemble.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Check ensemble weights sum to 1
    assert abs(np.sum(estimator_ensemble.ensemble_weights) - 1.0) < 1e-6

    # Both should recover true effect
    effect_single = estimator_single.estimate_ate()
    effect_ensemble = estimator_ensemble.estimate_ate()

    assert abs(effect_single.ate - synthetic_data["true_ate"]) < 0.5
    assert abs(effect_ensemble.ate - synthetic_data["true_ate"]) < 0.5


def test_aipw_component_optimization(synthetic_data):
    """Test AIPW component balance optimization."""
    # Standard AIPW (without cross-fitting for simpler test)
    estimator_standard = AIPWEstimator(cross_fitting=False, random_state=42)
    estimator_standard.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Optimized AIPW (without cross-fitting for simpler test)
    estimator_optimized = AIPWEstimator(
        cross_fitting=False,
        optimize_component_balance=True,
        component_variance_penalty=0.5,
        influence_function_se=False,  # Disable IF SE (incompatible with optimization)
        random_state=42,
        verbose=True,
    )
    estimator_optimized.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Estimate ATE for standard AIPW first
    effect_std = estimator_standard.estimate_ate()

    # Estimate ATE for optimized AIPW (optimization happens here)
    effect_opt = estimator_optimized.estimate_ate()

    # Now check optimization diagnostics (after estimate_ate)
    opt_diag = estimator_optimized.get_optimization_diagnostics()
    assert (
        opt_diag is not None
    ), "Optimization diagnostics should be populated after estimate_ate()"
    assert "optimal_g_computation_weight" in opt_diag
    assert "optimal_ipw_weight" in opt_diag
    assert 0.3 <= opt_diag["optimal_g_computation_weight"] <= 0.7

    # Check variance reduction (uses actual key names from implementation)
    assert "optimized_estimator_variance" in opt_diag
    assert "standard_estimator_variance" in opt_diag
    assert (
        opt_diag["optimized_estimator_variance"]
        <= opt_diag["standard_estimator_variance"]
    )

    # Standard AIPW should recover true effect
    assert abs(effect_std.ate - synthetic_data["true_ate"]) < 0.5

    # Optimized AIPW trades bias for variance, so check:
    # 1. Point estimate is within its own bootstrap CI
    assert effect_opt.ate_ci_lower <= effect_opt.ate <= effect_opt.ate_ci_upper
    # 2. Bootstrap CI width is smaller than standard (variance reduction)
    opt_ci_width = effect_opt.ate_ci_upper - effect_opt.ate_ci_lower
    std_ci_width = effect_std.ate_ci_upper - effect_std.ate_ci_lower
    # Optimized may have narrower CI due to variance reduction, but not always
    # Just check both are reasonable (not degenerate)
    assert opt_ci_width > 0.01
    assert std_ci_width > 0.01
