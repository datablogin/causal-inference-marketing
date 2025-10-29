"""Mixin providing PyRake-style constrained optimization capabilities."""

import warnings
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .optimization_config import OptimizationConfig


class OptimizationMixin:
    """Mixin providing PyRake-style constrained optimization capabilities.

    This mixin follows the pattern established by BootstrapMixin:
    - Cooperative __init__ using super()
    - Configuration object for settings
    - Stores optimization results for diagnostics
    - Provides fallback to non-optimized methods on failure
    """

    optimization_config: Union[OptimizationConfig, None]
    _optimization_diagnostics: dict[str, Any]

    def __init__(
        self,
        *args: Any,
        optimization_config: Union[OptimizationConfig, None] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize optimization mixin with configuration.

        Args:
            optimization_config: Configuration for optimization
            *args: Positional arguments for parent class
            **kwargs: Keyword arguments for parent class
        """
        # Pass optimization_config through kwargs for super() call
        if "optimization_config" not in kwargs:
            kwargs["optimization_config"] = optimization_config
        super().__init__(*args, **kwargs)

        # Ensure valid OptimizationConfig
        if optimization_config is not None:
            self.optimization_config = optimization_config
        elif (
            not hasattr(self, "optimization_config") or self.optimization_config is None
        ):
            self.optimization_config = None  # No optimization by default

        self._optimization_diagnostics = {}

    def optimize_weights_constrained(
        self,
        baseline_weights: NDArray[Any],
        covariates: NDArray[Any],
        target_means: Union[NDArray[Any], None] = None,
        variance_constraint: Union[float, None] = None,
    ) -> NDArray[Any]:
        """Optimize weights using PyRake-style constrained optimization.

        Args:
            baseline_weights: Initial/baseline weights (e.g., 1/e(X) for IPW)
            covariates: Covariate matrix (n_obs x n_covariates)
            target_means: Target covariate means (default: observed means)
            variance_constraint: Maximum weight variance (default: from config)

        Returns:
            Optimized weights array
        """
        # Check if optimization is enabled
        if (
            not self.optimization_config
            or not self.optimization_config.optimize_weights
        ):
            return baseline_weights

        n_obs = len(baseline_weights)

        # Set target means to observed means if not provided
        if target_means is None:
            target_means = np.mean(covariates, axis=0)

        # Set variance constraint from config if not provided
        if variance_constraint is None:
            variance_constraint = self.optimization_config.variance_constraint

        # Define objective function
        def objective(w: NDArray[Any]) -> float:
            """Distance from baseline weights."""
            metric = self.optimization_config.distance_metric

            if metric == "l2":
                return float(np.sum((w - baseline_weights) ** 2))
            elif metric == "kl_divergence":
                # KL divergence: sum(w * log(w / baseline))
                eps = 1e-10
                return float(np.sum(w * np.log((w + eps) / (baseline_weights + eps))))
            elif metric == "huber":
                diff = w - baseline_weights
                delta = 1.0
                huber = np.where(
                    np.abs(diff) <= delta,
                    0.5 * diff**2,
                    delta * (np.abs(diff) - 0.5 * delta),
                )
                return float(np.sum(huber))
            else:
                raise ValueError(f"Unknown distance metric: {metric}")

        # Define constraints
        constraints = []

        # Covariate balance constraint
        if self.optimization_config.balance_constraints:

            def balance_constraint(w: NDArray[Any]) -> NDArray[Any]:
                """Constraint: (1/n) X^T w = μ"""
                weighted_means = (covariates.T @ w) / n_obs
                return weighted_means - target_means

            constraints.append({"type": "eq", "fun": balance_constraint})

        # Variance constraint
        if variance_constraint is not None:

            def variance_constraint_func(w: NDArray[Any]) -> float:
                """Constraint: (1/n) ||w||^2 ≤ φ"""
                return float(variance_constraint - np.sum(w**2) / n_obs)

            constraints.append({"type": "ineq", "fun": variance_constraint_func})

        # Non-negativity bounds
        bounds = [(0, None) for _ in range(n_obs)]

        # Optimize
        try:
            result = minimize(
                objective,
                baseline_weights,
                method=self.optimization_config.method,
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.optimization_config.max_iterations,
                    "ftol": self.optimization_config.convergence_tolerance,
                    "disp": self.optimization_config.verbose,
                },
            )

            # Store diagnostics
            if self.optimization_config.store_diagnostics:
                self._optimization_diagnostics = {
                    "success": result.success,
                    "message": result.message,
                    "n_iterations": result.nit,
                    "final_objective": result.fun,
                    "constraint_violation": self._compute_constraint_violation(
                        result.x, covariates, target_means
                    ),
                    "weight_variance": float(np.var(result.x)),
                    "effective_sample_size": float(
                        np.sum(result.x) ** 2 / np.sum(result.x**2)
                    ),
                }

            if not result.success:
                warnings.warn(
                    f"Weight optimization did not converge: {result.message}. "
                    f"Falling back to baseline weights."
                )
                return baseline_weights

            return result.x

        except Exception as e:
            warnings.warn(
                f"Weight optimization failed with error: {str(e)}. "
                f"Falling back to baseline weights."
            )
            return baseline_weights

    def _compute_constraint_violation(
        self,
        weights: NDArray[Any],
        covariates: NDArray[Any],
        target_means: NDArray[Any],
    ) -> float:
        """Compute standardized mean difference for covariate balance.

        Args:
            weights: Weight vector
            covariates: Covariate matrix
            target_means: Target covariate means

        Returns:
            Maximum standardized mean difference across covariates
        """
        weighted_means = (covariates.T @ weights) / len(weights)
        smd = np.abs(weighted_means - target_means) / (
            np.std(covariates, axis=0) + 1e-10
        )
        return float(np.max(smd))

    def get_optimization_diagnostics(self) -> Union[dict[str, Any], None]:
        """Get optimization diagnostics.

        Returns:
            Dictionary of optimization diagnostics or None if not available
        """
        return (
            self._optimization_diagnostics if self._optimization_diagnostics else None
        )
