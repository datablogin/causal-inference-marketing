"""Integration tests for IPW with PyRake-style optimization."""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from causal_inference.core.base import (
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from causal_inference.core.bootstrap import BootstrapConfig
from causal_inference.core.optimization_config import OptimizationConfig
from causal_inference.estimators.ipw import IPWEstimator

# SMD threshold for acceptable covariate balance
# Values < 0.1 indicate excellent balance, < 0.15-0.2 is acceptable in many contexts
ACCEPTABLE_SMD_THRESHOLD = 0.15


def compute_smd(
    covariates: NDArray[Any], treatment: NDArray[Any], weights: NDArray[Any]
) -> NDArray[Any]:
    """Compute standardized mean difference for each covariate.

    SMD is a scale-free measure of covariate balance commonly used in
    causal inference. Values < 0.1 indicate excellent balance, < 0.2
    is acceptable in many contexts.

    Args:
        covariates: Covariate matrix (n x p)
        treatment: Binary treatment vector (0/1)
        weights: Weight vector (e.g., IPW weights)

    Returns:
        Array of SMD values for each covariate, one per column

    Raises:
        ValueError: If inputs have mismatched lengths or empty treatment groups

    References:
        Austin, P.C., & Stuart, E.A. (2015). Moving towards best practice
        when using inverse probability of treatment weighting (IPTW).

    Note:
        TODO: Consider moving this to causal_inference.diagnostics.balance
        for reusability across the codebase.
    """
    # Validate inputs
    if len(treatment) != len(covariates) or len(weights) != len(covariates):
        raise ValueError("Treatment, weights, and covariates must have same length")

    treated_mask = treatment == 1
    control_mask = treatment == 0

    # Check for empty groups
    if not np.any(treated_mask) or not np.any(control_mask):
        raise ValueError("Both treatment groups must have observations")

    # Weighted means
    weighted_mean_treated = np.average(
        covariates[treated_mask], weights=weights[treated_mask], axis=0
    )
    weighted_mean_control = np.average(
        covariates[control_mask], weights=weights[control_mask], axis=0
    )

    # Pooled standard deviation across all observations
    # Note: This uses the overall population SD rather than group-specific pooled SD.
    # This approach is valid and commonly used, especially when sample sizes differ.
    # Alternative: np.sqrt((std_treated**2 + std_control**2) / 2)
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
    assert np.max(smd_optimized) <= ACCEPTABLE_SMD_THRESHOLD, (
        f"Optimized balance (max SMD={np.max(smd_optimized):.4f}) should be "
        f"within acceptable range (< {ACCEPTABLE_SMD_THRESHOLD})"
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


def test_ipw_optimization_single_covariate():
    """Test optimization with single covariate (1D array handling)."""
    np.random.seed(42)
    n = 500

    # Generate data with only ONE covariate
    X = np.random.randn(n)  # 1D array

    # Treatment with confounding
    propensity = 1 / (1 + np.exp(-X))
    treatment = np.random.binomial(1, propensity)

    # Outcome with treatment effect = 2.0
    true_ate = 2.0
    outcome = 2.0 * treatment + X + np.random.randn(n) * 0.5

    # Create data objects - CovariateData should handle 1D array
    treatment_data = TreatmentData(values=treatment, treatment_type="binary")
    outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
    covariate_data = CovariateData(values=X, names=["X1"])

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

    # Fit the estimator - should handle 1D array properly
    estimator.fit(treatment_data, outcome_data, covariate_data)

    # Estimate ATE
    effect = estimator.estimate_ate()

    # Verify optimization ran successfully
    opt_diag = estimator.get_optimization_diagnostics()
    assert opt_diag is not None, "Optimization diagnostics should be available"
    assert "success" in opt_diag, "Diagnostics should contain success flag"

    # Verify estimate is reasonable (within 1.0 of true ATE)
    # Note: Tolerance of 1.0 (50% of true ATE=2.0) is appropriate for edge case testing
    # with a single covariate. The primary goal is verifying proper 1D array handling
    # and optimization completion, not achieving optimal statistical accuracy.
    assert (
        abs(effect.ate - true_ate) < 1.0
    ), f"ATE estimate {effect.ate} should be close to true ATE {true_ate}"

    # Verify weights are computed
    assert estimator.weights is not None, "Weights should be computed"
    assert len(estimator.weights) == n, "Weights length should match sample size"

    # Verify covariate balance is achieved after optimization
    # Reshape X for SMD computation (compute_smd expects 2D array)
    X_2d = X.reshape(-1, 1)
    smd = compute_smd(X_2d, treatment, estimator.weights)
    assert np.max(smd) <= ACCEPTABLE_SMD_THRESHOLD, (
        f"Single covariate should achieve acceptable balance: "
        f"SMD={np.max(smd):.4f}, threshold={ACCEPTABLE_SMD_THRESHOLD}"
    )


def test_ipw_optimization_convergence_failure():
    """Test graceful handling of optimization failure.

    This test validates that the estimator handles pathological cases gracefully
    by either: (1) completing successfully with a suboptimal solution, or
    (2) raising an appropriate EstimationError (not crashing with unexpected errors).

    Both outcomes are acceptable for edge case handling.
    """
    np.random.seed(42)
    n = 100

    # Create pathological data that causes convergence issues
    # Use perfectly separable data
    X = np.random.randn(n, 2)
    # Make treatment perfectly predictable from covariates
    treatment = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Outcome - simple relationship
    outcome = 2.0 * treatment + X[:, 0] + np.random.randn(n) * 0.1

    treatment_data = TreatmentData(values=treatment, treatment_type="binary")
    outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
    covariate_data = CovariateData(values=X, names=["X1", "X2"])

    # Create optimization config with very tight constraints
    opt_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=0.5,  # Very tight constraint
        balance_constraints=True,
        balance_tolerance=0.001,  # Very tight tolerance
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

    # Test graceful handling: should either succeed or raise EstimationError
    # (not crash with unexpected errors like AttributeError, TypeError, etc.)
    try:
        estimator.fit(treatment_data, outcome_data, covariate_data)

        # Estimate ATE - should work even if optimization didn't converge
        effect = estimator.estimate_ate()

        # Verify we got a result (Outcome 1: Success path)
        assert (
            effect.ate is not None
        ), "Should get ATE estimate even if optimization struggled"

        # Check diagnostics - may or may not have succeeded
        opt_diag = estimator.get_optimization_diagnostics()
        if opt_diag is not None:
            # If optimization ran, check that diagnostics are reasonable
            assert "success" in opt_diag, "Diagnostics should contain success flag"
            # Note: success might be False, which is acceptable for graceful handling

    except EstimationError:
        # Outcome 2: Proper error path - this is acceptable for pathological cases
        # The key is that we get a meaningful EstimationError, not an unexpected crash
        pass
    except Exception as e:
        # Outcome 3: Unexpected error - this is a test failure
        pytest.fail(
            f"Should either succeed or raise EstimationError, got {type(e).__name__}: {e}"
        )


def test_ipw_optimization_extreme_propensity_scores():
    """Test optimization with extreme propensity scores."""
    np.random.seed(42)
    n = 500

    # Covariates
    X = np.random.randn(n, 3)

    # Create treatment with extreme propensity scores
    # Use large coefficients to push propensities to extremes
    propensity_logit = 3.0 * X[:, 0] + 2.0 * X[:, 1]
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = np.random.binomial(1, propensity)

    # Outcome (no true ATE needed for this test - we're testing error handling)
    outcome = 2.0 * treatment + X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

    treatment_data = TreatmentData(values=treatment, treatment_type="binary")
    outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
    covariate_data = CovariateData(values=X, names=["X1", "X2", "X3"])

    # Create optimization config
    opt_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,
        balance_constraints=True,
        balance_tolerance=0.05,
        distance_metric="l2",
        verbose=False,
    )

    # Create IPW estimator with optimization and overlap checking
    estimator = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=opt_config,
        check_overlap=True,
        overlap_threshold=0.1,
        random_state=42,
        verbose=False,
    )

    # Fit the estimator - should handle extreme propensity scores
    estimator.fit(treatment_data, outcome_data, covariate_data)

    # Check overlap diagnostics - should have warnings
    overlap_diag = estimator.get_overlap_diagnostics()
    assert overlap_diag is not None, "Overlap diagnostics should be available"

    # May or may not satisfy overlap depending on data generation
    # The key is that we get diagnostics and warnings
    if not overlap_diag["overlap_satisfied"]:
        assert (
            len(overlap_diag["violations"]) > 0
        ), "Should have overlap violations listed"

    # Estimate ATE - should work despite extreme propensity scores
    effect = estimator.estimate_ate()
    assert effect.ate is not None, "Should get ATE estimate"

    # Verify weights are bounded (clipped to prevent extreme values)
    weight_diag = estimator.get_weight_diagnostics()
    assert weight_diag is not None, "Weight diagnostics should be available"

    # Weights should be bounded even with extreme propensity scores
    # Note: The threshold of 1000 comes from the IPW estimator's propensity score
    # clipping mechanism. In _compute_weights (ipw.py:541), propensity scores are
    # bounded to [0.001, 0.999], giving maximum theoretical weights of:
    # - For treated: 1/0.001 = 1000 (weight = 1/PS)
    # - For control: 1/(1-0.999) = 1000 (weight = 1/(1-PS))
    assert (
        weight_diag["max_weight"] < 1000
    ), "Weights should be bounded to prevent extreme values"


def test_ipw_optimization_empty_groups():
    """Test handling of edge case with no variation in treatment."""
    np.random.seed(42)
    n = 100

    # All treated (no control units)
    treatment = np.ones(n, dtype=int)

    # Should raise ValueError when creating TreatmentData with only one treatment level
    # This is caught at the data validation layer (even before estimation)
    with pytest.raises(ValueError) as exc_info:
        TreatmentData(values=treatment, treatment_type="binary")

    # Verify error message is informative
    error_message = str(exc_info.value)
    assert (
        "2 unique values" in error_message.lower()
    ), f"Error message should mention need for 2 unique values: {error_message}"
    assert (
        "treated and control" in error_message.lower()
    ), f"Error message should mention treated and control units: {error_message}"
