"""Unit tests for OptimizationConfig."""

import pytest
from pydantic import ValidationError

from causal_inference.core.optimization_config import OptimizationConfig


def test_optimization_config_defaults():
    """Test that OptimizationConfig has correct default values."""
    config = OptimizationConfig()

    assert config.optimize_weights is False
    assert config.method == "SLSQP"
    assert config.max_iterations == 1000
    assert config.variance_constraint is None
    assert config.balance_constraints is True
    assert config.balance_tolerance == 0.01
    assert config.distance_metric == "l2"
    assert config.verbose is False
    assert config.store_diagnostics is True
    assert config.convergence_tolerance == 1e-6


def test_optimization_config_custom_values():
    """Test OptimizationConfig with custom values."""
    config = OptimizationConfig(
        optimize_weights=True,
        method="trust-constr",
        max_iterations=500,
        variance_constraint=2.0,
        balance_constraints=False,
        balance_tolerance=0.05,
        distance_metric="kl_divergence",
        verbose=True,
        store_diagnostics=False,
        convergence_tolerance=1e-8,
    )

    assert config.optimize_weights is True
    assert config.method == "trust-constr"
    assert config.max_iterations == 500
    assert config.variance_constraint == 2.0
    assert config.balance_constraints is False
    assert config.balance_tolerance == 0.05
    assert config.distance_metric == "kl_divergence"
    assert config.verbose is True
    assert config.store_diagnostics is False
    assert config.convergence_tolerance == 1e-8


def test_optimization_config_method_validation():
    """Test that invalid optimization methods raise errors."""
    with pytest.raises(ValidationError):
        OptimizationConfig(method="invalid_method")


def test_optimization_config_distance_metric_validation():
    """Test that invalid distance metrics raise errors."""
    with pytest.raises(ValidationError):
        OptimizationConfig(distance_metric="invalid_metric")


def test_optimization_config_max_iterations_validation():
    """Test max_iterations bounds validation."""
    # Too low
    with pytest.raises(ValueError):
        OptimizationConfig(max_iterations=50)

    # Too high
    with pytest.raises(ValueError):
        OptimizationConfig(max_iterations=20000)

    # Within bounds
    config = OptimizationConfig(max_iterations=500)
    assert config.max_iterations == 500


def test_optimization_config_balance_tolerance_validation():
    """Test balance_tolerance bounds validation."""
    # Negative
    with pytest.raises(ValueError):
        OptimizationConfig(balance_tolerance=-0.01)

    # Too high
    with pytest.raises(ValueError):
        OptimizationConfig(balance_tolerance=1.5)

    # Within bounds
    config = OptimizationConfig(balance_tolerance=0.1)
    assert config.balance_tolerance == 0.1


def test_optimization_config_convergence_tolerance_validation():
    """Test convergence_tolerance validation."""
    # Must be positive
    with pytest.raises(ValueError):
        OptimizationConfig(convergence_tolerance=0.0)

    with pytest.raises(ValueError):
        OptimizationConfig(convergence_tolerance=-1e-6)

    # Valid value
    config = OptimizationConfig(convergence_tolerance=1e-4)
    assert config.convergence_tolerance == 1e-4


def test_optimization_config_all_methods():
    """Test all valid optimization methods."""
    for method in ["SLSQP", "trust-constr"]:
        config = OptimizationConfig(method=method)
        assert config.method == method


def test_optimization_config_all_distance_metrics():
    """Test all valid distance metrics."""
    for metric in ["l2", "kl_divergence", "huber"]:
        config = OptimizationConfig(distance_metric=metric)
        assert config.distance_metric == metric


def test_optimization_config_variance_constraint_optional():
    """Test that variance_constraint can be None or a float."""
    config_none = OptimizationConfig(variance_constraint=None)
    assert config_none.variance_constraint is None

    config_value = OptimizationConfig(variance_constraint=3.5)
    assert config_value.variance_constraint == 3.5
