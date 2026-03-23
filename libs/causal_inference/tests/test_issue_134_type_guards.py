"""Tests for Issue #134: Replace runtime assertions with proper type guards."""

from __future__ import annotations

import inspect

import numpy as np

from causal_inference.core.optimization_config import OptimizationConfig
from causal_inference.core.optimization_mixin import OptimizationMixin


class ConcreteOptimizer(OptimizationMixin):
    """Concrete class using the mixin, for testing purposes."""

    def __init__(
        self,
        optimization_config: OptimizationConfig | None = None,
    ) -> None:
        super().__init__(optimization_config=optimization_config)


class TestNoAssertions:
    """Verify that assert statements have been removed from optimize_weights_constrained."""

    def test_optimization_mixin_no_assertions(self) -> None:
        """No assert statements should remain in optimize_weights_constrained."""
        source = inspect.getsource(OptimizationMixin.optimize_weights_constrained)
        # Split into lines and check each non-comment line for assert
        for line in source.splitlines():
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # No line should start with 'assert '
            assert not stripped.startswith(
                "assert "
            ), f"Found assert statement in optimize_weights_constrained: {stripped}"


class TestOptimizationStillWorks:
    """Verify that optimization produces correct results after refactoring."""

    def test_optimization_still_works(self) -> None:
        """Optimization with a valid config should produce reasonable weights."""
        config = OptimizationConfig(
            optimize_weights=True,
            method="SLSQP",
            balance_constraints=True,
            store_diagnostics=True,
        )
        optimizer = ConcreteOptimizer(optimization_config=config)

        np.random.seed(42)
        n = 100
        baseline_weights = np.ones(n)
        covariates = np.random.randn(n, 3)

        result = optimizer.optimize_weights_constrained(
            baseline_weights=baseline_weights,
            covariates=covariates,
        )

        # Should return an array of the same length
        assert len(result) == n
        # Weights should be non-negative (due to bounds)
        assert np.all(result >= -1e-10)
        # Should be a numpy array
        assert isinstance(result, np.ndarray)


class TestEarlyReturnPaths:
    """Verify early return behavior when config is None or optimization disabled."""

    def test_optimization_none_config_returns_baseline(self) -> None:
        """When optimization_config is None, baseline weights are returned."""
        optimizer = ConcreteOptimizer(optimization_config=None)
        baseline = np.array([1.0, 2.0, 3.0])
        covariates = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = optimizer.optimize_weights_constrained(
            baseline_weights=baseline,
            covariates=covariates,
        )

        np.testing.assert_array_equal(result, baseline)

    def test_optimization_disabled_returns_baseline(self) -> None:
        """When optimize_weights=False, baseline weights are returned."""
        config = OptimizationConfig(optimize_weights=False)
        optimizer = ConcreteOptimizer(optimization_config=config)
        baseline = np.array([1.0, 2.0, 3.0])
        covariates = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = optimizer.optimize_weights_constrained(
            baseline_weights=baseline,
            covariates=covariates,
        )

        np.testing.assert_array_equal(result, baseline)
