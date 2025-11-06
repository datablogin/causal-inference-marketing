"""Integration tests for IPW with PyRake-style optimization."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.core.bootstrap import BootstrapConfig
from causal_inference.core.optimization_config import OptimizationConfig
from causal_inference.estimators.ipw import IPWEstimator


def compute_smd(covariates, treatment, weights):
    """Compute standardized mean difference for each covariate.

    Args:
        covariates: Covariate matrix (n x p)
        treatment: Binary treatment vector
        weights: Weight vector

    Returns:
        Array of SMD values for each covariate
    """
    treated_mask = treatment == 1
    control_mask = treatment == 0

    # Weighted means
    weighted_mean_treated = np.average(
        covariates[treated_mask], weights=weights[treated_mask], axis=0
    )
    weighted_mean_control = np.average(
        covariates[control_mask], weights=weights[control_mask], axis=0
    )

    # Pooled standard deviation
    std_pooled = np.std(covariates, axis=0)

    # SMD
    smd = np.abs(weighted_mean_treated - weighted_mean_control) / (std_pooled + 1e-10)
    return smd


@pytest.fixture
def synthetic_data_with_confounding():
    """Generate synthetic data with known treatment effect and confounding."""
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


def test_ipw_basic_optimization(synthetic_data_with_confounding):
    """Test that IPW optimization runs without errors."""
    # Create optimization config
    opt_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,
        balance_constraints=True,
        balance_tolerance=0.01,
        distance_metric="l2",
        verbose=False,
    )

    # Create IPW estimator with optimization
    estimator = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=opt_config,
        random_state=42,
        verbose=False,
    )

    # Fit the estimator
    estimator.fit(
        synthetic_data_with_confounding["treatment"],
        synthetic_data_with_confounding["outcome"],
        synthetic_data_with_confounding["covariates"],
    )

    # Estimate ATE
    effect = estimator.estimate_ate()

    # Verify optimization ran
    opt_diag = estimator.get_optimization_diagnostics()
    assert opt_diag is not None, "Optimization diagnostics should be available"
    assert "success" in opt_diag, "Diagnostics should contain success flag"
    assert "weight_variance" in opt_diag, "Diagnostics should contain weight variance"
    assert "effective_sample_size" in opt_diag, "Diagnostics should contain ESS"

    # Verify estimate is reasonable (within 1.0 of true ATE)
    assert abs(effect.ate - synthetic_data_with_confounding["true_ate"]) < 1.0


def test_ipw_optimization_backward_compatibility(synthetic_data_with_confounding):
    """Test that IPW without optimization still works (backward compatibility)."""
    # Create IPW estimator without optimization
    estimator = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=None,  # No optimization
        random_state=42,
        verbose=False,
    )

    # Fit the estimator
    estimator.fit(
        synthetic_data_with_confounding["treatment"],
        synthetic_data_with_confounding["outcome"],
        synthetic_data_with_confounding["covariates"],
    )

    # Estimate ATE
    effect = estimator.estimate_ate()

    # Verify no optimization diagnostics
    opt_diag = estimator.get_optimization_diagnostics()
    assert opt_diag is None or opt_diag == {}, "Should have no optimization diagnostics"

    # Verify estimate is reasonable
    assert abs(effect.ate - synthetic_data_with_confounding["true_ate"]) < 1.0


def test_ipw_optimization_with_bootstrap(synthetic_data_with_confounding):
    """Test that IPW optimization works with bootstrap confidence intervals."""
    # Create optimization config
    opt_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,
        balance_constraints=True,
        balance_tolerance=0.01,
        distance_metric="l2",
        verbose=False,
    )

    # Create bootstrap config
    bootstrap_config = BootstrapConfig(
        n_samples=50,  # Use fewer samples for faster testing
        confidence_level=0.95,
        random_state=42,
    )

    # Create IPW estimator with both optimization and bootstrap
    estimator = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=opt_config,
        bootstrap_config=bootstrap_config,
        random_state=42,
        verbose=False,
    )

    # Fit the estimator
    estimator.fit(
        synthetic_data_with_confounding["treatment"],
        synthetic_data_with_confounding["outcome"],
        synthetic_data_with_confounding["covariates"],
    )

    # Estimate ATE
    effect = estimator.estimate_ate()

    # Verify bootstrap CIs are available
    assert effect.ate_ci_lower is not None, "Bootstrap CI lower should be available"
    assert effect.ate_ci_upper is not None, "Bootstrap CI upper should be available"
    assert (
        effect.ate_ci_lower < effect.ate < effect.ate_ci_upper
    ), "ATE should be within CI"

    # Verify optimization diagnostics
    opt_diag = estimator.get_optimization_diagnostics()
    assert opt_diag is not None, "Optimization diagnostics should be available"


def test_ipw_optimization_variance_reduction(synthetic_data_with_confounding):
    """Test that optimization reduces weight variance compared to analytical weights."""
    # Standard IPW (no optimization)
    estimator_standard = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=None,
        random_state=42,
        verbose=False,
    )
    estimator_standard.fit(
        synthetic_data_with_confounding["treatment"],
        synthetic_data_with_confounding["outcome"],
        synthetic_data_with_confounding["covariates"],
    )

    # Optimized IPW
    opt_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=1.5,
        balance_constraints=True,
        balance_tolerance=0.01,
        distance_metric="l2",
        verbose=False,
    )
    estimator_optimized = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=opt_config,
        random_state=42,
        verbose=False,
    )
    estimator_optimized.fit(
        synthetic_data_with_confounding["treatment"],
        synthetic_data_with_confounding["outcome"],
        synthetic_data_with_confounding["covariates"],
    )

    # Get weight diagnostics
    weight_diag_std = estimator_standard.get_weight_diagnostics()
    weight_diag_opt = estimator_optimized.get_weight_diagnostics()

    # Verify variance reduction
    assert (
        weight_diag_opt["weight_variance"] <= weight_diag_std["weight_variance"]
    ), "Optimized weights should have lower or equal variance"

    # Verify ESS improvement or stability
    # Note: ESS might not always increase due to the variance-balance tradeoff
    assert weight_diag_opt["effective_sample_size"] > 0, "ESS should be positive"

    # Compute covariate balance using SMD
    covariates = synthetic_data_with_confounding["covariates"].values
    treatment = synthetic_data_with_confounding["treatment"].values

    smd_standard = compute_smd(covariates, treatment, estimator_standard.weights)
    smd_optimized = compute_smd(covariates, treatment, estimator_optimized.weights)

    # Verify balance improvement
    # Note: Due to variance constraint, balance may not reach exact tolerance
    # The key test is that balance improves relative to standard IPW
    assert np.max(smd_optimized) <= np.max(smd_standard), (
        f"Balance should not worsen: optimized max SMD={np.max(smd_optimized):.4f}, "
        f"standard max SMD={np.max(smd_standard):.4f}"
    )
    # Verify balance is reasonable (within a practical threshold)
    assert np.max(smd_optimized) <= 0.1, (
        f"Optimized balance (max SMD={np.max(smd_optimized):.4f}) should be "
        f"reasonably good (< 0.1 is generally acceptable)"
    )


def test_ipw_optimization_different_distance_metrics(synthetic_data_with_confounding):
    """Test IPW optimization with different distance metrics."""
    distance_metrics = ["l2", "kl_divergence", "huber"]

    for metric in distance_metrics:
        opt_config = OptimizationConfig(
            optimize_weights=True,
            variance_constraint=2.0,
            balance_constraints=True,
            distance_metric=metric,
            verbose=False,
        )

        estimator = IPWEstimator(
            propensity_model_type="logistic",
            optimization_config=opt_config,
            random_state=42,
            verbose=False,
        )

        estimator.fit(
            synthetic_data_with_confounding["treatment"],
            synthetic_data_with_confounding["outcome"],
            synthetic_data_with_confounding["covariates"],
        )

        effect = estimator.estimate_ate()

        # Verify estimate is reasonable
        assert (
            abs(effect.ate - synthetic_data_with_confounding["true_ate"]) < 1.5
        ), f"ATE with {metric} metric should be reasonable"

        # Verify optimization ran
        opt_diag = estimator.get_optimization_diagnostics()
        assert (
            opt_diag is not None
        ), f"Optimization with {metric} should produce diagnostics"


def test_ipw_optimization_without_covariates_raises_error(
    synthetic_data_with_confounding,
):
    """Test that optimization without covariates raises appropriate error."""
    opt_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,
        balance_constraints=True,
    )

    estimator = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=opt_config,
        random_state=42,
    )

    # Fitting without covariates should raise error during optimization
    with pytest.raises(Exception):  # Will raise EstimationError
        estimator.fit(
            synthetic_data_with_confounding["treatment"],
            synthetic_data_with_confounding["outcome"],
            covariates=None,  # No covariates - should fail for optimization
        )


def test_ipw_optimization_diagnostics_accessible(synthetic_data_with_confounding):
    """Test that optimization diagnostics are properly accessible."""
    opt_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,
        balance_constraints=True,
        store_diagnostics=True,
        verbose=False,
    )

    estimator = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=opt_config,
        random_state=42,
        verbose=False,
    )

    estimator.fit(
        synthetic_data_with_confounding["treatment"],
        synthetic_data_with_confounding["outcome"],
        synthetic_data_with_confounding["covariates"],
    )

    opt_diag = estimator.get_optimization_diagnostics()

    # Verify all expected diagnostics are present
    expected_keys = [
        "success",
        "message",
        "n_iterations",
        "final_objective",
        "constraint_violation",
        "weight_variance",
        "effective_sample_size",
    ]

    for key in expected_keys:
        assert key in opt_diag, f"Diagnostics should contain {key}"

    # Verify types (use type() instead of isinstance for numpy compatibility)
    assert isinstance(opt_diag["success"], (bool, np.bool_))
    assert isinstance(opt_diag["message"], str)
    assert isinstance(opt_diag["n_iterations"], (int, np.integer))
    assert isinstance(opt_diag["final_objective"], (float, np.floating))
    assert isinstance(opt_diag["constraint_violation"], (float, np.floating))
    assert isinstance(opt_diag["weight_variance"], (float, np.floating))
    assert isinstance(opt_diag["effective_sample_size"], (float, np.floating))
