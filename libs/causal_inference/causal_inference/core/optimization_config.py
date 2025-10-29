"""Configuration for PyRake-style constrained optimization."""

from typing import Literal, Union

from pydantic import BaseModel, Field


class OptimizationConfig(BaseModel):
    """Configuration for PyRake-style constrained optimization.

    Attributes:
        optimize_weights: Enable weight optimization vs analytical computation
        method: Scipy optimization method (SLSQP, trust-constr)
        max_iterations: Maximum optimization iterations
        variance_constraint: Maximum allowed weight variance (φ in PyRake)
        balance_constraints: Enforce covariate balance constraints
        balance_tolerance: Tolerance for covariate balance (SMD units)
        distance_metric: Distance metric for weight optimization
        verbose: Print optimization progress
        store_diagnostics: Store detailed optimization diagnostics
        convergence_tolerance: Convergence tolerance for optimization
    """

    # General optimization settings
    optimize_weights: bool = Field(
        default=False,
        description="Enable weight optimization (vs analytical computation)",
    )

    method: Literal["SLSQP", "trust-constr"] = Field(
        default="SLSQP", description="Scipy optimization method"
    )

    max_iterations: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum optimization iterations",
    )

    # PyRake-style constraints
    variance_constraint: Union[float, None] = Field(
        default=None,
        gt=0.0,
        description="Maximum allowed weight variance (φ in PyRake)",
    )

    balance_constraints: bool = Field(
        default=True, description="Enforce covariate balance constraints"
    )

    balance_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Tolerance for covariate balance (SMD units)",
    )

    # Distance metrics
    distance_metric: Literal["l2", "kl_divergence", "huber"] = Field(
        default="l2", description="Distance metric for weight optimization"
    )

    # Computational settings
    verbose: bool = Field(default=False, description="Print optimization progress")

    store_diagnostics: bool = Field(
        default=True, description="Store detailed optimization diagnostics"
    )

    convergence_tolerance: float = Field(
        default=1e-6, gt=0.0, description="Convergence tolerance for optimization"
    )
