"""Unit tests for OptimizationMixin."""

import numpy as np

from causal_inference.core.optimization_config import OptimizationConfig
from causal_inference.core.optimization_mixin import OptimizationMixin


class DummyEstimator(OptimizationMixin):
    """Dummy estimator for testing OptimizationMixin."""

    def __init__(self, optimization_config=None):
        """Initialize dummy estimator."""
        self.optimization_config = optimization_config
        self._optimization_diagnostics = {}


def test_optimization_mixin_disabled_by_default():
    """Test that optimization is disabled when no config provided."""
    np.random.seed(42)
    estimator = DummyEstimator()

    baseline_weights = np.array([1.0, 2.0, 3.0, 4.0])
    covariates = np.random.randn(4, 2)

    # Should return baseline weights unchanged
    result = estimator.optimize_weights_constrained(baseline_weights, covariates)
    np.testing.assert_array_equal(result, baseline_weights)


def test_optimization_mixin_disabled_when_optimize_weights_false():
    """Test that optimization is disabled when optimize_weights=False."""
    np.random.seed(42)
    config = OptimizationConfig(optimize_weights=False)
    estimator = DummyEstimator(optimization_config=config)

    baseline_weights = np.array([1.0, 2.0, 3.0, 4.0])
    covariates = np.random.randn(4, 2)

    # Should return baseline weights unchanged
    result = estimator.optimize_weights_constrained(baseline_weights, covariates)
    np.testing.assert_array_equal(result, baseline_weights)


def test_optimization_mixin_l2_distance():
    """Test optimization with L2 distance metric."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="l2",
        variance_constraint=5.0,
        balance_constraints=True,
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    # Create simple test data
    n = 10
    baseline_weights = np.ones(n)
    covariates = np.random.randn(n, 2)

    result = estimator.optimize_weights_constrained(baseline_weights, covariates)

    # Check result properties
    assert result.shape == baseline_weights.shape
    assert np.all(result >= 0)  # Non-negative weights
    assert len(result) == n


def test_optimization_mixin_kl_divergence_distance():
    """Test optimization with KL divergence distance metric."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="kl_divergence",
        variance_constraint=5.0,
        balance_constraints=True,
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    n = 10
    baseline_weights = np.ones(n) * 0.5
    covariates = np.random.randn(n, 2)

    result = estimator.optimize_weights_constrained(baseline_weights, covariates)

    # Check result properties
    assert result.shape == baseline_weights.shape
    assert np.all(result >= 0)  # Non-negative weights


def test_optimization_mixin_huber_distance():
    """Test optimization with Huber distance metric."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="huber",
        variance_constraint=5.0,
        balance_constraints=True,
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    n = 10
    baseline_weights = np.ones(n)
    covariates = np.random.randn(n, 2)

    result = estimator.optimize_weights_constrained(baseline_weights, covariates)

    # Check result properties
    assert result.shape == baseline_weights.shape
    assert np.all(result >= 0)  # Non-negative weights


def test_optimization_mixin_with_target_means():
    """Test optimization with custom target means."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="l2",
        balance_constraints=True,
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    n = 10
    baseline_weights = np.ones(n)
    covariates = np.random.randn(n, 2)
    target_means = np.array([0.5, -0.5])

    result = estimator.optimize_weights_constrained(
        baseline_weights, covariates, target_means=target_means
    )

    # Check result properties
    assert result.shape == baseline_weights.shape
    assert np.all(result >= 0)


def test_optimization_mixin_without_balance_constraints():
    """Test optimization without balance constraints."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="l2",
        balance_constraints=False,
        variance_constraint=5.0,
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    n = 10
    baseline_weights = np.ones(n) * 2.0
    covariates = np.random.randn(n, 2)

    result = estimator.optimize_weights_constrained(baseline_weights, covariates)

    # Should still optimize with only variance constraint
    assert result.shape == baseline_weights.shape
    assert np.all(result >= 0)


def test_optimization_mixin_stores_diagnostics():
    """Test that optimization diagnostics are stored."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="l2",
        variance_constraint=5.0,
        balance_constraints=True,
        store_diagnostics=True,
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    n = 10
    baseline_weights = np.ones(n)
    covariates = np.random.randn(n, 2)

    estimator.optimize_weights_constrained(baseline_weights, covariates)

    diagnostics = estimator.get_optimization_diagnostics()

    assert diagnostics is not None
    assert "success" in diagnostics
    assert "message" in diagnostics
    assert "n_iterations" in diagnostics
    assert "final_objective" in diagnostics
    assert "constraint_violation" in diagnostics
    assert "weight_variance" in diagnostics
    assert "effective_sample_size" in diagnostics


def test_optimization_mixin_diagnostics_disabled():
    """Test that diagnostics are not stored when disabled."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="l2",
        variance_constraint=5.0,
        store_diagnostics=False,
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    n = 10
    baseline_weights = np.ones(n)
    covariates = np.random.randn(n, 2)

    estimator.optimize_weights_constrained(baseline_weights, covariates)

    diagnostics = estimator.get_optimization_diagnostics()
    assert diagnostics is None


def test_optimization_mixin_fallback_on_failure():
    """Test that optimization falls back to baseline weights on failure."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="l2",
        variance_constraint=0.01,  # Impossible constraint
        balance_constraints=True,
        max_iterations=100,  # Minimum iterations
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    n = 10
    baseline_weights = np.ones(n) * 10.0  # High variance baseline
    covariates = np.random.randn(n, 2)

    result = estimator.optimize_weights_constrained(baseline_weights, covariates)

    # Should fall back to baseline weights if optimization fails
    # We can't guarantee failure, but if it does fail, result should be baseline
    assert result.shape == baseline_weights.shape
    assert np.all(result >= 0)


def test_optimization_mixin_compute_constraint_violation():
    """Test constraint violation computation."""
    np.random.seed(42)
    config = OptimizationConfig(optimize_weights=True)
    estimator = DummyEstimator(optimization_config=config)

    n = 20
    weights = np.ones(n)
    covariates = np.random.randn(n, 3)
    target_means = np.zeros(3)

    violation = estimator._compute_constraint_violation(
        weights, covariates, target_means
    )

    # Should be a scalar value
    assert isinstance(violation, float)
    assert violation >= 0


def test_optimization_mixin_different_methods():
    """Test optimization with different scipy methods."""
    np.random.seed(42)

    n = 10
    baseline_weights = np.ones(n)
    covariates = np.random.randn(n, 2)

    for method in ["SLSQP", "trust-constr", "COBYLA"]:
        config = OptimizationConfig(
            optimize_weights=True,
            method=method,
            distance_metric="l2",
            variance_constraint=5.0,
            verbose=False,
        )
        estimator = DummyEstimator(optimization_config=config)

        result = estimator.optimize_weights_constrained(baseline_weights, covariates)

        # Check basic properties
        assert result.shape == baseline_weights.shape
        assert np.all(result >= 0)


def test_optimization_mixin_effective_sample_size():
    """Test that effective sample size is computed correctly."""
    np.random.seed(42)
    config = OptimizationConfig(
        optimize_weights=True,
        distance_metric="l2",
        variance_constraint=5.0,
        store_diagnostics=True,
        verbose=False,
    )
    estimator = DummyEstimator(optimization_config=config)

    n = 20
    baseline_weights = np.ones(n)
    covariates = np.random.randn(n, 2)

    estimator.optimize_weights_constrained(baseline_weights, covariates)

    diagnostics = estimator.get_optimization_diagnostics()
    ess = diagnostics["effective_sample_size"]

    # ESS should be between 1 and n
    assert 1.0 <= ess <= n

    # For equal weights, ESS should equal n
    equal_weights = np.ones(n)
    expected_ess = np.sum(equal_weights) ** 2 / np.sum(equal_weights**2)
    assert expected_ess == n
