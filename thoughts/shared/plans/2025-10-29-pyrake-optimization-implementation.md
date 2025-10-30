# PyRake-Style Optimization Framework Implementation Plan

## Overview

This plan implements PyRake-style constrained optimization across IPW, G-computation, and AIPW estimators in the CausalInferenceTools library. PyRake optimizes balancing weights under constraints that simultaneously address bias reduction, variance minimization, and predictive power preservation.

## Current State Analysis

### What Exists Now

1. **Mixin Architecture Pattern** (`libs/causal_inference/causal_inference/core/bootstrap.py:326-1250`)
   - `BootstrapMixin` provides template for adding reusable functionality
   - Cooperative `__init__` with `super()` calls
   - Configuration object pattern (`BootstrapConfig`)
   - Abstract methods for subclass implementation

2. **Optimization Pattern Demonstrated** (`libs/causal_inference/causal_inference/estimators/synthetic_control.py:406-463`)
   - Uses `scipy.optimize.minimize` with SLSQP method
   - Equality constraints: `{"type": "eq", "fun": lambda w: np.sum(w) - 1}`
   - Bounds: `[(0, 1) for _ in range(n_control)]`
   - Stores optimization diagnostics
   - Error handling for failed optimization

3. **IPW Weight Computation** (`libs/causal_inference/causal_inference/estimators/ipw.py:500-563`)
   - Analytical weight computation from propensity scores
   - Optional stabilization and truncation
   - Weight diagnostics (ESS, variance, ratio)
   - No optimization layer currently

4. **G-Computation** (`libs/causal_inference/causal_inference/estimators/g_computation.py:127-169`)
   - Rule-based model selection
   - Single model fitting (no ensemble)
   - Performance metrics computed but not used for optimization

5. **AIPW** (`libs/causal_inference/causal_inference/estimators/aipw.py:745-816`)
   - Fixed doubly-robust formula
   - Component diagnostics available
   - No component balance optimization

### Key Discoveries

- **Existing Infrastructure**: Synthetic Control already demonstrates the exact optimization pattern needed (scipy.optimize.minimize with constraints)
- **Extension Points**: BaseEstimator's mixin architecture provides clean integration path
- **Configuration Pattern**: BootstrapConfig demonstrates complete pattern for adding new configs
- **Diagnostics Available**: Weight diagnostics (ESS, variance) provide natural optimization targets

## Desired End State

After implementation:

1. **OptimizationConfig** class provides PyRake-style configuration with variance constraints, balance constraints, and distance metrics
2. **OptimizationMixin** provides reusable optimization methods across all estimators
3. **IPW** supports constrained weight optimization minimizing distance from propensity-based weights while enforcing covariate balance
4. **G-Computation** supports ensemble model weighting with variance penalties
5. **AIPW** supports component balance optimization
6. All optimizations are **optional** and **backward compatible**

### Verification Criteria

- `make ci` passes all tests, type checking, and linting
- New OptimizationConfig validates parameters correctly
- IPW optimization reduces weight variance while maintaining covariate balance
- Optimization diagnostics are accessible via estimator methods
- Bootstrap integration works (optimization runs on each sample)
- Legacy API continues to work without OptimizationConfig

## What We're NOT Doing

- Not modifying existing analytical weight computation (remains as default/fallback)
- Not breaking backward compatibility (all optimization is opt-in)
- Not implementing nested optimization (disable in bootstrap samples)
- Not implementing all distance metrics initially (start with L2, add others later)
- Not optimizing estimators beyond IPW, G-computation, AIPW in this plan
- Not requiring optimization (remains optional feature)

## Implementation Approach

Follow the established mixin pattern demonstrated by BootstrapMixin:
1. Create configuration class with Pydantic validation
2. Create mixin with optimization methods
3. Integrate into estimators via multiple inheritance
4. Add abstract factory method for bootstrap compatibility
5. Provide comprehensive tests demonstrating improvements

## Phase 1: Core Infrastructure

### Overview
Create the foundational optimization configuration and mixin classes following established patterns from BootstrapMixin and BootstrapConfig.

### Changes Required

#### 1. OptimizationConfig Class
**File**: `libs/causal_inference/causal_inference/core/optimization_config.py`
**Changes**: Create new file with PyRake configuration

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union
import numpy as np
from numpy.typing import NDArray

class OptimizationConfig(BaseModel):
    """Configuration for PyRake-style constrained optimization.

    Attributes:
        optimize_weights: Enable weight optimization vs analytical computation
        method: Scipy optimization method (SLSQP, trust-constr, COBYLA)
        max_iterations: Maximum optimization iterations
        variance_constraint: Maximum allowed weight variance (φ in PyRake)
        balance_constraints: Enforce covariate balance constraints
        balance_tolerance: Tolerance for covariate balance (SMD units)
        distance_metric: Distance metric for weight optimization
        verbose: Print optimization progress
        store_diagnostics: Store detailed optimization diagnostics
    """

    # General optimization settings
    optimize_weights: bool = Field(
        default=False,
        description="Enable weight optimization (vs analytical computation)"
    )

    method: Literal["SLSQP", "trust-constr", "COBYLA"] = Field(
        default="SLSQP",
        description="Scipy optimization method"
    )

    max_iterations: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum optimization iterations"
    )

    # PyRake-style constraints
    variance_constraint: Union[float, None] = Field(
        default=None,
        description="Maximum allowed weight variance (φ in PyRake)"
    )

    balance_constraints: bool = Field(
        default=True,
        description="Enforce covariate balance constraints"
    )

    balance_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Tolerance for covariate balance (SMD units)"
    )

    # Distance metrics
    distance_metric: Literal["l2", "kl_divergence", "huber"] = Field(
        default="l2",
        description="Distance metric for weight optimization"
    )

    # Computational settings
    verbose: bool = Field(
        default=False,
        description="Print optimization progress"
    )

    store_diagnostics: bool = Field(
        default=True,
        description="Store detailed optimization diagnostics"
    )

    convergence_tolerance: float = Field(
        default=1e-6,
        gt=0.0,
        description="Convergence tolerance for optimization"
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate optimization method."""
        allowed_methods = {"SLSQP", "trust-constr", "COBYLA"}
        if v not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        return v

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric."""
        allowed_metrics = {"l2", "kl_divergence", "huber"}
        if v not in allowed_metrics:
            raise ValueError(f"distance_metric must be one of {allowed_metrics}")
        return v
```

#### 2. OptimizationMixin Class
**File**: `libs/causal_inference/causal_inference/core/optimization_mixin.py`
**Changes**: Create new file with optimization methods

```python
from typing import Any, Union
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
import warnings

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
        **kwargs: Any
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
        elif not hasattr(self, "optimization_config") or self.optimization_config is None:
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
        if not self.optimization_config or not self.optimization_config.optimize_weights:
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
                    delta * (np.abs(diff) - 0.5 * delta)
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

            constraints.append({
                "type": "eq",
                "fun": balance_constraint
            })

        # Variance constraint
        if variance_constraint is not None:
            def variance_constraint_func(w: NDArray[Any]) -> float:
                """Constraint: (1/n) ||w||^2 ≤ φ"""
                return float(variance_constraint - np.sum(w**2) / n_obs)

            constraints.append({
                "type": "ineq",
                "fun": variance_constraint_func
            })

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
                    "disp": self.optimization_config.verbose
                }
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
                        np.sum(result.x)**2 / np.sum(result.x**2)
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
        target_means: NDArray[Any]
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
        smd = np.abs(weighted_means - target_means) / (np.std(covariates, axis=0) + 1e-10)
        return float(np.max(smd))

    def get_optimization_diagnostics(self) -> Union[dict[str, Any], None]:
        """Get optimization diagnostics.

        Returns:
            Dictionary of optimization diagnostics or None if not available
        """
        return self._optimization_diagnostics if self._optimization_diagnostics else None
```

#### 3. Update __init__.py Exports
**File**: `libs/causal_inference/causal_inference/core/__init__.py`
**Changes**: Add new classes to exports

```python
# Add these imports
from .optimization_config import OptimizationConfig
from .optimization_mixin import OptimizationMixin

# Add to __all__
__all__ = [
    # ... existing exports
    "OptimizationConfig",
    "OptimizationMixin",
]
```

#### 4. Update BaseEstimator __init__
**File**: `libs/causal_inference/causal_inference/core/base.py`
**Changes**: Add optimization_config parameter (lines 638-656)

```python
def __init__(
    self,
    random_state: Union[int, None] = None,
    verbose: bool = False,
    bootstrap_config: Union[Any, None] = None,
    optimization_config: Union[Any, None] = None,  # NEW
) -> None:
    """Initialize the base estimator.

    Args:
        random_state: Random seed for reproducible results
        verbose: Whether to print verbose output during estimation
        bootstrap_config: Configuration for bootstrap confidence intervals
        optimization_config: Configuration for optimization strategies  # NEW
    """
    self.random_state = random_state
    self.verbose = verbose
    self.is_fitted = False

    # Bootstrap configuration (will be set by BootstrapMixin if used)
    self.bootstrap_config = bootstrap_config

    # Optimization configuration (will be set by OptimizationMixin if used)  # NEW
    self.optimization_config = optimization_config  # NEW

    # Data containers
    self.treatment_data: Union[TreatmentData, None] = None
    self.outcome_data: Union[OutcomeData, None] = None
    self.covariate_data: Union[CovariateData, None] = None

    # Results cache
    self._causal_effect: Union[CausalEffect, None] = None

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)
```

### Success Criteria

#### Automated Verification:
- [x] OptimizationConfig validates correctly: `pytest libs/causal_inference/tests/test_optimization_config.py -v`
- [x] OptimizationMixin objective functions work: `pytest libs/causal_inference/tests/test_optimization_mixin.py -v`
- [x] Type checking passes: `make typecheck`
- [x] Linting passes: `make lint`
- [x] All unit tests pass: `make test`

#### Manual Verification:
- [ ] OptimizationConfig can be instantiated with various parameter combinations
- [ ] Distance metrics (l2, kl_divergence, huber) compute correctly
- [ ] Constraint functions (balance, variance) work as expected
- [ ] Optimization diagnostics are stored and accessible

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to Phase 2.

---

## Phase 2: IPW Integration

### Overview
Integrate OptimizationMixin into IPWEstimator to enable PyRake-style weight optimization as an alternative to analytical weight computation.

### Changes Required

#### 1. IPWEstimator Class Definition
**File**: `libs/causal_inference/causal_inference/estimators/ipw.py`
**Changes**: Add OptimizationMixin to inheritance (line 115)

```python
# Change from:
class IPWEstimator(BootstrapMixin, BaseEstimator):

# To:
class IPWEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):
```

#### 2. IPWEstimator __init__
**File**: `libs/causal_inference/causal_inference/estimators/ipw.py`
**Changes**: Add optimization_config parameter (lines 136-180)

```python
def __init__(
    self,
    propensity_model_type: str = "logistic",
    propensity_model_params: dict[str, Any] | None = None,
    weight_truncation: str | None = None,
    truncation_threshold: float = 0.01,
    stabilized_weights: bool = False,
    bootstrap_config: Any | None = None,
    optimization_config: Any | None = None,  # NEW
    check_overlap: bool = True,
    overlap_threshold: float = 0.1,
    # Legacy parameters for backward compatibility
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    verbose: bool = False,
) -> None:
    """Initialize the IPW estimator.

    Args:
        propensity_model_type: Model type ('logistic', 'random_forest')
        propensity_model_params: Parameters to pass to the sklearn model
        weight_truncation: Truncation method ('percentile', 'threshold', None)
        truncation_threshold: Threshold for truncation (0.01 = 1st/99th percentile)
        stabilized_weights: Whether to use stabilized weights
        bootstrap_config: Configuration for bootstrap confidence intervals
        optimization_config: Configuration for weight optimization  # NEW
        check_overlap: Whether to check overlap assumption
        overlap_threshold: Minimum propensity score for overlap check
        bootstrap_samples: Legacy parameter - number of bootstrap samples
        confidence_level: Legacy parameter - confidence level
        random_state: Random seed for reproducibility
        verbose: Whether to print verbose output
    """
    # Create bootstrap config if not provided (for backward compatibility)
    if bootstrap_config is None:
        bootstrap_config = BootstrapConfig(
            n_samples=bootstrap_samples,
            confidence_level=confidence_level,
            random_state=random_state,
        )

    # Call super with both configs
    super().__init__(
        bootstrap_config=bootstrap_config,
        optimization_config=optimization_config,  # NEW
        random_state=random_state,
        verbose=verbose,
    )

    # Store IPW-specific configuration
    self.propensity_model_type = propensity_model_type
    self.propensity_model_params = propensity_model_params or {}
    self.weight_truncation = weight_truncation
    self.truncation_threshold = truncation_threshold
    self.stabilized_weights = stabilized_weights
    self.check_overlap = check_overlap
    self.overlap_threshold = overlap_threshold

    # Initialize storage
    self.propensity_model: Any = None
    self.propensity_scores: NDArray[Any] | None = None
    self.weights: NDArray[Any] | None = None
    self._weight_diagnostics: dict[str, float] = {}
    self._overlap_diagnostics: dict[str, Any] = {}
```

#### 3. Optimize Weights in _fit_implementation
**File**: `libs/causal_inference/causal_inference/estimators/ipw.py`
**Changes**: Add optimization after analytical weight computation (after line 621)

```python
def _fit_implementation(
    self,
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData | None = None,
) -> None:
    """Fit the IPW estimator with optional weight optimization."""

    # Existing: fit propensity model and estimate scores
    self._fit_propensity_model(treatment, covariates)
    self.propensity_scores = self._estimate_propensity_scores()

    # Check overlap if enabled
    if self.check_overlap:
        self._overlap_diagnostics = self._check_overlap(self.propensity_scores)

    # Compute baseline weights analytically
    baseline_weights = self._compute_weights(treatment, self.propensity_scores)

    # NEW: Optimize weights if configured
    if self.optimization_config and self.optimization_config.optimize_weights:
        if covariates is None:
            raise ValueError(
                "Covariates required for weight optimization. "
                "Cannot optimize without covariate balance constraints."
            )

        # Get covariate array
        if hasattr(covariates.values, 'values'):
            covariate_array = covariates.values.values
        else:
            covariate_array = covariates.values

        # Optimize weights with PyRake-style constraints
        self.weights = self.optimize_weights_constrained(
            baseline_weights=baseline_weights,
            covariates=covariate_array,
            variance_constraint=self.optimization_config.variance_constraint,
        )

        if self.verbose:
            opt_diag = self.get_optimization_diagnostics()
            if opt_diag:
                print("\n=== Weight Optimization Results ===")
                print(f"Optimization converged: {opt_diag['success']}")
                print(f"Final objective value: {opt_diag['final_objective']:.6f}")
                print(f"Iterations: {opt_diag['n_iterations']}")
                print(f"Max covariate imbalance (SMD): {opt_diag['constraint_violation']:.4f}")
                print(f"Weight variance: {opt_diag['weight_variance']:.4f}")
                print(f"Effective sample size: {opt_diag['effective_sample_size']:.1f}")
    else:
        # Use baseline analytical weights
        self.weights = baseline_weights

    # Compute weight diagnostics
    self._weight_diagnostics = self._compute_weight_diagnostics(self.weights)

    if self.verbose:
        print("\n=== Weight Diagnostics ===")
        for key, value in self._weight_diagnostics.items():
            print(f"{key}: {value:.4f}")
```

#### 4. Update _create_bootstrap_estimator
**File**: `libs/causal_inference/causal_inference/estimators/ipw.py`
**Changes**: Disable optimization in bootstrap samples (lines 200-222)

```python
def _create_bootstrap_estimator(
    self, random_state: int | None = None
) -> IPWEstimator:
    """Create a new estimator instance for bootstrap sampling.

    Note: Optimization is disabled in bootstrap samples to avoid
    nested optimization and reduce computational cost.
    """
    return IPWEstimator(
        propensity_model_type=self.propensity_model_type,
        propensity_model_params=self.propensity_model_params,
        weight_truncation=self.weight_truncation,
        truncation_threshold=self.truncation_threshold,
        stabilized_weights=self.stabilized_weights,
        bootstrap_config=BootstrapConfig(n_samples=0),  # No nested bootstrap
        optimization_config=None,  # NEW: Disable optimization in bootstrap
        check_overlap=False,  # Skip overlap checks in bootstrap
        overlap_threshold=self.overlap_threshold,
        random_state=random_state,
        verbose=False,  # Reduce verbosity in bootstrap
    )
```

#### 5. Add Import
**File**: `libs/causal_inference/causal_inference/estimators/ipw.py`
**Changes**: Add imports at top of file

```python
from ..core.optimization_config import OptimizationConfig
from ..core.optimization_mixin import OptimizationMixin
```

### Success Criteria

#### Automated Verification:
- [x] IPW with optimization runs without errors: `pytest libs/causal_inference/tests/test_ipw_optimization.py::test_ipw_basic_optimization -v`
- [x] IPW without optimization still works (backward compatibility): `pytest libs/causal_inference/tests/test_ipw.py -v`
- [x] Bootstrap with optimization works: `pytest libs/causal_inference/tests/test_ipw_optimization.py::test_ipw_optimization_with_bootstrap -v`
- [x] Type checking passes: `make typecheck`
- [x] Linting passes: `make lint`
- [x] All tests pass: `make test`

#### Manual Verification:
- [ ] Optimized weights have lower variance than analytical weights
- [ ] Optimized weights maintain better covariate balance
- [ ] Effective sample size improves with optimization
- [ ] Optimization diagnostics show successful convergence
- [ ] ATE estimates remain unbiased (compare optimized vs analytical on synthetic data)
- [ ] Confidence intervals are tighter with optimization (lower variance)

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to Phase 3.

---

## Phase 3: G-Computation Integration

### Overview
Extend OptimizationMixin into G-Computation estimator for ensemble model weighting with variance penalties.

### Changes Required

#### 1. GComputationEstimator Class Definition
**File**: `libs/causal_inference/causal_inference/estimators/g_computation.py`
**Changes**: Add OptimizationMixin to inheritance (line 36)

```python
# Change from:
class GComputationEstimator(BootstrapMixin, BaseEstimator):

# To:
class GComputationEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):
```

#### 2. GComputationEstimator __init__
**File**: `libs/causal_inference/causal_inference/estimators/g_computation.py`
**Changes**: Add optimization_config parameter and ensemble settings

```python
def __init__(
    self,
    model_type: str = "auto",
    model_params: dict[str, Any] | None = None,
    bootstrap_config: Any | None = None,
    optimization_config: Any | None = None,  # NEW
    # Legacy parameters for backward compatibility
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    verbose: bool = False,
    # Large dataset optimization parameters
    chunk_size: int = 10000,
    memory_efficient: bool = True,
    large_dataset_threshold: int = 100000,
    # NEW: Ensemble settings
    use_ensemble: bool = False,
    ensemble_models: list[str] | None = None,
    ensemble_variance_penalty: float = 0.1,
) -> None:
    """Initialize the G-computation estimator.

    Args:
        model_type: Model type ('auto', 'linear', 'logistic', 'random_forest')
        model_params: Parameters to pass to the sklearn model
        bootstrap_config: Configuration for bootstrap confidence intervals
        optimization_config: Configuration for optimization strategies
        bootstrap_samples: Legacy parameter - number of bootstrap samples
        confidence_level: Legacy parameter - confidence level
        random_state: Random seed for reproducibility
        verbose: Whether to print verbose output
        chunk_size: Size of chunks for large dataset processing
        memory_efficient: Use memory-efficient mode for large datasets
        large_dataset_threshold: Threshold for large dataset optimizations
        use_ensemble: Use ensemble of models instead of single model
        ensemble_models: List of model types for ensemble (if use_ensemble=True)
        ensemble_variance_penalty: Penalty on ensemble weight variance
    """
    # Create bootstrap config if not provided (for backward compatibility)
    if bootstrap_config is None:
        bootstrap_config = BootstrapConfig(
            n_samples=bootstrap_samples,
            confidence_level=confidence_level,
            random_state=random_state,
        )

    super().__init__(
        bootstrap_config=bootstrap_config,
        optimization_config=optimization_config,  # NEW
        random_state=random_state,
        verbose=verbose,
    )

    self.model_type = model_type
    self.model_params = model_params or {}
    self.chunk_size = chunk_size
    self.memory_efficient = memory_efficient
    self.large_dataset_threshold = large_dataset_threshold

    # NEW: Ensemble settings
    self.use_ensemble = use_ensemble
    self.ensemble_models = ensemble_models or ["linear", "ridge", "random_forest"]
    self.ensemble_variance_penalty = ensemble_variance_penalty

    # Storage
    self.outcome_model: Any = None
    self.ensemble_models_fitted: dict[str, Any] = {}  # NEW
    self.ensemble_weights: NDArray[Any] | None = None  # NEW
```

#### 3. Add Ensemble Fitting Method
**File**: `libs/causal_inference/causal_inference/estimators/g_computation.py`
**Changes**: Add new method after _select_model

```python
def _fit_ensemble_models(
    self,
    X: pd.DataFrame,
    y: NDArray[Any],
    outcome_type: str,
) -> dict[str, Any]:
    """Fit multiple models for ensemble.

    Args:
        X: Feature matrix
        y: Outcome vector
        outcome_type: Type of outcome ('continuous', 'binary')

    Returns:
        Dictionary of fitted models
    """
    from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    models = {}

    for model_name in self.ensemble_models:
        if outcome_type == "continuous":
            if model_name == "linear":
                model = LinearRegression()
            elif model_name == "ridge":
                model = Ridge(alpha=1.0)
            elif model_name == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state
                )
            else:
                continue
        elif outcome_type == "binary":
            if model_name in ["linear", "ridge"]:
                model = LogisticRegression(max_iter=1000)
            elif model_name == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state
                )
            else:
                continue
        else:
            continue

        try:
            model.fit(X, y)
            models[model_name] = model
            if self.verbose:
                print(f"Fitted {model_name} model")
        except Exception as e:
            if self.verbose:
                print(f"Failed to fit {model_name}: {e}")

    return models

def _optimize_ensemble_weights(
    self,
    models: dict[str, Any],
    X: pd.DataFrame,
    y: NDArray[Any],
) -> NDArray[Any]:
    """Optimize ensemble weights with variance penalty.

    Args:
        models: Dictionary of fitted models
        X: Feature matrix
        y: Outcome vector

    Returns:
        Optimized ensemble weights
    """
    n_models = len(models)
    model_names = list(models.keys())

    # Get predictions from each model
    predictions = np.column_stack([
        models[name].predict(X) for name in model_names
    ])

    def objective(weights: NDArray[Any]) -> float:
        """MSE with variance penalty."""
        ensemble_pred = predictions @ weights
        mse = float(np.mean((y - ensemble_pred) ** 2))

        # Variance penalty (encourage diverse weights)
        variance_penalty = self.ensemble_variance_penalty * float(np.var(weights))

        return mse + variance_penalty

    # Constraints: weights sum to 1, all non-negative
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]
    bounds = [(0, 1) for _ in range(n_models)]

    # Initial guess: equal weights
    initial_weights = np.ones(n_models) / n_models

    from scipy.optimize import minimize
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    if not result.success and self.verbose:
        print(f"Ensemble optimization warning: {result.message}")

    # Store diagnostics
    self._optimization_diagnostics = {
        "ensemble_success": result.success,
        "ensemble_objective": result.fun,
        "ensemble_weights": {name: float(w) for name, w in zip(model_names, result.x)},
    }

    return result.x
```

#### 4. Update _fit_implementation
**File**: `libs/causal_inference/causal_inference/estimators/g_computation.py`
**Changes**: Add ensemble path in fitting logic

```python
def _fit_implementation(
    self,
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData | None = None,
) -> None:
    """Fit the G-computation estimator with optional ensemble."""

    # Prepare features
    features = self._prepare_features(treatment, covariates)

    # NEW: Ensemble path
    if self.use_ensemble:
        self.ensemble_models_fitted = self._fit_ensemble_models(
            features,
            outcome.values.values if hasattr(outcome.values, 'values') else outcome.values,
            outcome.outcome_type
        )

        if len(self.ensemble_models_fitted) > 1:
            self.ensemble_weights = self._optimize_ensemble_weights(
                self.ensemble_models_fitted,
                features,
                outcome.values.values if hasattr(outcome.values, 'values') else outcome.values,
            )

            if self.verbose:
                print("\n=== Ensemble Weights ===")
                for name, weight in zip(
                    self.ensemble_models_fitted.keys(),
                    self.ensemble_weights
                ):
                    print(f"{name}: {weight:.4f}")
        else:
            # Fall back to single model if ensemble failed
            self.use_ensemble = False
            self.outcome_model = self._select_model(outcome.outcome_type)
            self.outcome_model.fit(features, outcome.values)
    else:
        # Existing single model path
        self.outcome_model = self._select_model(outcome.outcome_type)
        self.outcome_model.fit(features, outcome.values)
```

#### 5. Update Prediction Methods
**File**: `libs/causal_inference/causal_inference/estimators/g_computation.py`
**Changes**: Add ensemble prediction in _predict_counterfactuals_regular

```python
def _predict_counterfactuals_regular(
    self,
    treatment_value: float | int,
    covariates: CovariateData | None = None,
) -> NDArray[Any]:
    """Predict counterfactual outcomes with optional ensemble."""

    # Prepare features
    features = self._prepare_counterfactual_features(treatment_value, covariates)

    # NEW: Ensemble prediction
    if self.use_ensemble and self.ensemble_models_fitted:
        predictions = np.column_stack([
            model.predict(features)
            for model in self.ensemble_models_fitted.values()
        ])
        return predictions @ self.ensemble_weights
    else:
        # Existing single model prediction
        return self.outcome_model.predict(features)
```

#### 6. Add Imports
**File**: `libs/causal_inference/causal_inference/estimators/g_computation.py`
**Changes**: Add imports at top

```python
from ..core.optimization_config import OptimizationConfig
from ..core.optimization_mixin import OptimizationMixin
```

### Success Criteria

#### Automated Verification:
- [ ] G-computation with ensemble runs: `pytest libs/causal_inference/tests/test_g_computation_ensemble.py::test_ensemble_basic -v`
- [ ] G-computation without ensemble still works: `pytest libs/causal_inference/tests/test_g_computation.py -v`
- [ ] Ensemble weights sum to 1: `pytest libs/causal_inference/tests/test_g_computation_ensemble.py::test_ensemble_weights_valid -v`
- [ ] Type checking passes: `make typecheck`
- [ ] Linting passes: `make lint`
- [ ] All tests pass: `make test`

#### Manual Verification:
- [ ] Ensemble predictions are weighted combinations of individual models
- [ ] Ensemble weights are diverse (not all weight on one model)
- [ ] Ensemble MSE is lower than or equal to best single model
- [ ] Optimization diagnostics show ensemble weight distribution
- [ ] ATE estimates from ensemble are reasonable

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to Phase 4.

---

## Phase 4: AIPW Integration

### Overview
Add component balance optimization to AIPW estimator to optimize the weighting between G-computation and IPW components.

### Changes Required

#### 1. AIPWEstimator Class Definition
**File**: `libs/causal_inference/causal_inference/estimators/aipw.py`
**Changes**: Add OptimizationMixin to inheritance (line 115)

```python
# Change from:
class AIPWEstimator(BootstrapMixin, BaseEstimator):

# To:
class AIPWEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):
```

#### 2. AIPWEstimator __init__
**File**: `libs/causal_inference/causal_inference/estimators/aipw.py`
**Changes**: Add optimization_config parameter

```python
def __init__(
    self,
    outcome_model_type: str = "auto",
    outcome_model_params: dict[str, Any] | None = None,
    propensity_model_type: str = "logistic",
    propensity_model_params: dict[str, Any] | None = None,
    cross_fitting: bool = True,
    n_folds: int = 5,
    stratify_folds: bool = True,
    influence_function_se: bool = True,
    weight_truncation: str | None = None,
    truncation_threshold: float = 0.01,
    stabilized_weights: bool = False,
    bootstrap_config: Any | None = None,
    optimization_config: Any | None = None,  # NEW
    # NEW: Component optimization settings
    optimize_component_balance: bool = False,
    component_variance_penalty: float = 0.5,
    # Legacy parameters
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    verbose: bool = False,
) -> None:
    """Initialize the AIPW estimator.

    Args:
        outcome_model_type: Model type for outcome model
        outcome_model_params: Parameters for outcome model
        propensity_model_type: Model type for propensity model
        propensity_model_params: Parameters for propensity model
        cross_fitting: Use cross-fitting to reduce bias
        n_folds: Number of folds for cross-fitting
        stratify_folds: Stratify folds by treatment
        influence_function_se: Use influence function for standard errors
        weight_truncation: Truncation method for weights
        truncation_threshold: Threshold for weight truncation
        stabilized_weights: Use stabilized weights
        bootstrap_config: Configuration for bootstrap
        optimization_config: Configuration for optimization
        optimize_component_balance: Optimize G-computation vs IPW balance
        component_variance_penalty: Penalty for deviating from 50/50 balance
        bootstrap_samples: Legacy - number of bootstrap samples
        confidence_level: Legacy - confidence level
        random_state: Random seed
        verbose: Verbose output
    """
    # Create bootstrap config if needed
    if bootstrap_config is None:
        bootstrap_config = BootstrapConfig(
            n_samples=bootstrap_samples,
            confidence_level=confidence_level,
            random_state=random_state,
        )

    super().__init__(
        bootstrap_config=bootstrap_config,
        optimization_config=optimization_config,  # NEW
        random_state=random_state,
        verbose=verbose,
    )

    # Store all configuration
    self.outcome_model_type = outcome_model_type
    self.outcome_model_params = outcome_model_params or {}
    self.propensity_model_type = propensity_model_type
    self.propensity_model_params = propensity_model_params or {}
    self.cross_fitting = cross_fitting
    self.n_folds = n_folds
    self.stratify_folds = stratify_folds
    self.influence_function_se = influence_function_se
    self.weight_truncation = weight_truncation
    self.truncation_threshold = truncation_threshold
    self.stabilized_weights = stabilized_weights

    # NEW: Component optimization settings
    self.optimize_component_balance = optimize_component_balance
    self.component_variance_penalty = component_variance_penalty

    # Storage
    self.g_computation_estimator: Any = None
    self.ipw_estimator: Any = None
    self._aipw_components: dict[str, Any] = {}
    self._component_diagnostics: dict[str, Any] = {}
```

#### 3. Add Component Optimization Method
**File**: `libs/causal_inference/causal_inference/estimators/aipw.py`
**Changes**: Add new method for optimizing component balance

```python
def _optimize_component_balance(
    self,
    g_comp_components: NDArray[Any],
    ipw_components: NDArray[Any],
) -> float:
    """Optimize balance between G-computation and IPW components.

    Args:
        g_comp_components: G-computation component values (μ₁ - μ₀)
        ipw_components: IPW correction component values

    Returns:
        Optimal weight for G-computation component (alpha)
    """
    from scipy.optimize import minimize_scalar

    def objective(alpha: float) -> float:
        """Weighted AIPW variance with balance penalty."""
        # alpha is weight on G-computation (0 to 1)
        # (1 - alpha) is implicit weight on IPW

        combined = alpha * g_comp_components + (1 - alpha) * ipw_components

        # Variance of estimate
        estimate_variance = float(np.var(combined))

        # Penalty for extreme weights (force meaningful contribution from both)
        balance_penalty = self.component_variance_penalty * (alpha - 0.5) ** 2

        return estimate_variance + balance_penalty

    result = minimize_scalar(
        objective,
        bounds=(0.3, 0.7),  # Ensure both components contribute
        method="bounded"
    )

    optimal_alpha = float(result.x)

    # Store diagnostics
    self._optimization_diagnostics = {
        "optimal_g_computation_weight": optimal_alpha,
        "optimal_ipw_weight": 1 - optimal_alpha,
        "optimized_variance": float(np.var(
            optimal_alpha * g_comp_components + (1 - optimal_alpha) * ipw_components
        )),
        "fixed_variance": float(np.var(g_comp_components + ipw_components)),
    }

    if self.verbose:
        print("\n=== Component Balance Optimization ===")
        print(f"Optimal G-computation weight: {optimal_alpha:.4f}")
        print(f"Optimal IPW weight: {1 - optimal_alpha:.4f}")
        print(f"Variance reduction: {self._optimization_diagnostics['fixed_variance'] - self._optimization_diagnostics['optimized_variance']:.6f}")

    return optimal_alpha
```

#### 4. Update ATE Estimation
**File**: `libs/causal_inference/causal_inference/estimators/aipw.py`
**Changes**: Add optimization in _compute_aipw_estimate

```python
def _compute_aipw_estimate(
    self,
    treatment_data: TreatmentData,
    outcome_data: OutcomeData,
) -> float:
    """Compute AIPW estimate with optional component optimization."""

    treatment_values = treatment_data.values
    outcome_values = outcome_data.values

    # Get components
    g_comp_treated = self.g_computation_estimator._predict_counterfactuals_regular(
        1, self.covariate_data
    )
    g_comp_control = self.g_computation_estimator._predict_counterfactuals_regular(
        0, self.covariate_data
    )

    g_comp_component = g_comp_treated - g_comp_control

    # IPW correction terms
    treated_mask = treatment_values == 1
    control_mask = treatment_values == 0

    weights = self.ipw_estimator.weights

    ipw_correction = np.zeros_like(outcome_values)
    ipw_correction[treated_mask] = (
        weights[treated_mask] * (outcome_values[treated_mask] - g_comp_treated[treated_mask])
    )
    ipw_correction[control_mask] = (
        -weights[control_mask] * (outcome_values[control_mask] - g_comp_control[control_mask])
    )

    # Store components for diagnostics
    self._aipw_components = {
        "g_computation": g_comp_component,
        "ipw_correction": ipw_correction,
    }

    # NEW: Optimize component balance if enabled
    if self.optimize_component_balance:
        optimal_alpha = self._optimize_component_balance(
            g_comp_component,
            ipw_correction
        )

        # Use optimized weights
        aipw_estimate = float(np.mean(
            optimal_alpha * g_comp_component + (1 - optimal_alpha) * ipw_correction
        ))
    else:
        # Standard AIPW formula (equal weights)
        aipw_estimate = float(np.mean(g_comp_component + ipw_correction))

    return aipw_estimate
```

#### 5. Add Imports
**File**: `libs/causal_inference/causal_inference/estimators/aipw.py`
**Changes**: Add imports at top

```python
from ..core.optimization_config import OptimizationConfig
from ..core.optimization_mixin import OptimizationMixin
```

### Success Criteria

#### Automated Verification:
- [ ] AIPW with component optimization runs: `pytest libs/causal_inference/tests/test_aipw_optimization.py::test_component_optimization -v`
- [ ] AIPW without optimization still works: `pytest libs/causal_inference/tests/test_aipw.py -v`
- [ ] Component weights are in valid range [0.3, 0.7]: `pytest libs/causal_inference/tests/test_aipw_optimization.py::test_component_weights_valid -v`
- [ ] Type checking passes: `make typecheck`
- [ ] Linting passes: `make lint`
- [ ] All tests pass: `make test`

#### Manual Verification:
- [ ] Component optimization reduces variance compared to standard AIPW
- [ ] Both G-computation and IPW components contribute meaningfully
- [ ] Optimization diagnostics show variance reduction
- [ ] ATE estimates remain unbiased
- [ ] Component balance adapts to data characteristics

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to Phase 5.

---

## Phase 5: Testing & Documentation

### Overview
Create comprehensive tests demonstrating optimization improvements and provide user-facing documentation with examples.

### Changes Required

#### 1. Comprehensive Integration Tests
**File**: `libs/causal_inference/tests/test_optimization_integration.py`
**Changes**: Create new test file

```python
"""Integration tests for PyRake-style optimization across estimators."""
import pytest
import numpy as np
from causal_inference.core.optimization_config import OptimizationConfig
from causal_inference.estimators.ipw import IPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.core.base import TreatmentData, OutcomeData, CovariateData


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
    true_ate = 2.0
    outcome = (
        2.0 * treatment +
        X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] +
        np.random.randn(n) * 0.5
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
        propensity_model_type="logistic",
        random_state=42,
        verbose=True
    )
    estimator_standard.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"]
    )

    # Optimized IPW
    optimization_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,
        balance_constraints=True,
        distance_metric="l2",
        verbose=True
    )
    estimator_optimized = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=optimization_config,
        random_state=42,
        verbose=True
    )
    estimator_optimized.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"]
    )

    # Check variance reduction
    weight_diag_std = estimator_standard._weight_diagnostics
    weight_diag_opt = estimator_optimized._weight_diagnostics

    assert weight_diag_opt["weight_variance"] < weight_diag_std["weight_variance"]
    assert weight_diag_opt["effective_sample_size"] > weight_diag_std["effective_sample_size"]

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
        bootstrap_samples=100,
        random_state=42,
        verbose=True
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"]
    )

    effect = estimator.estimate_ate()

    # Should have bootstrap CIs
    assert effect.ate_ci_lower is not None
    assert effect.ate_ci_upper is not None
    assert effect.ate_ci_lower < effect.ate < effect.ate_ci_upper


def test_ensemble_improves_prediction(synthetic_data):
    """Test that ensemble G-computation improves over single model."""
    # Single model
    estimator_single = GComputationEstimator(
        model_type="linear",
        random_state=42
    )
    estimator_single.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"]
    )

    # Ensemble
    estimator_ensemble = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_variance_penalty=0.1,
        random_state=42,
        verbose=True
    )
    estimator_ensemble.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"]
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
    # Standard AIPW
    estimator_standard = AIPWEstimator(
        cross_fitting=True,
        n_folds=3,
        random_state=42
    )
    estimator_standard.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"]
    )

    # Optimized AIPW
    estimator_optimized = AIPWEstimator(
        cross_fitting=True,
        n_folds=3,
        optimize_component_balance=True,
        component_variance_penalty=0.5,
        random_state=42,
        verbose=True
    )
    estimator_optimized.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"]
    )

    # Check optimization diagnostics
    opt_diag = estimator_optimized.get_optimization_diagnostics()
    assert opt_diag is not None
    assert "optimal_g_computation_weight" in opt_diag
    assert 0.3 <= opt_diag["optimal_g_computation_weight"] <= 0.7

    # Check variance reduction
    assert opt_diag["optimized_variance"] <= opt_diag["fixed_variance"]

    # Both should recover true effect
    effect_std = estimator_standard.estimate_ate()
    effect_opt = estimator_optimized.estimate_ate()

    assert abs(effect_std.ate - synthetic_data["true_ate"]) < 0.5
    assert abs(effect_opt.ate - synthetic_data["true_ate"]) < 0.5
```

#### 2. Example Notebook
**File**: `examples/pyrake_optimization_examples.ipynb`
**Changes**: Create new Jupyter notebook

```markdown
# PyRake-Style Optimization Examples

This notebook demonstrates the use of PyRake-style constrained optimization
across IPW, G-computation, and AIPW estimators.

## Setup

[Import libraries and generate synthetic data]

## IPW with Weight Optimization

[Example showing variance reduction with optimized weights]

## G-Computation with Ensemble Models

[Example showing ensemble model weighting]

## AIPW with Component Balance Optimization

[Example showing component balance optimization]

## Bias-Variance Tradeoff Exploration

[Example showing how to vary variance constraint to explore tradeoff]
```

#### 3. User Documentation
**File**: `docs/optimization.md`
**Changes**: Create new documentation file

```markdown
# Optimization Framework

The CausalInferenceTools library supports PyRake-style constrained optimization
for improving estimator efficiency while maintaining unbiasedness.

## Overview

PyRake optimization simultaneously addresses:
- **Bias Reduction**: Covariate balance constraints
- **Variance Minimization**: Weight variance constraints
- **Predictive Power**: Minimal distance from baseline weights

## Configuration

Use `OptimizationConfig` to control optimization behavior:

```python
from causal_inference.core import OptimizationConfig

config = OptimizationConfig(
    optimize_weights=True,
    variance_constraint=2.0,
    balance_constraints=True,
    balance_tolerance=0.01,
    distance_metric="l2",
    verbose=True
)
```

## IPW Optimization

[Documentation and examples]

## G-Computation Ensemble

[Documentation and examples]

## AIPW Component Optimization

[Documentation and examples]

## Best Practices

[Guidelines for choosing parameters]
```

#### 4. API Reference Updates
**File**: `docs/api/optimization.rst`
**Changes**: Create API documentation

```rst
Optimization API
================

.. automodule:: causal_inference.core.optimization_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: causal_inference.core.optimization_mixin
   :members:
   :undoc-members:
   :show-inheritance:
```

### Success Criteria

#### Automated Verification:
- [ ] All integration tests pass: `pytest libs/causal_inference/tests/test_optimization_integration.py -v`
- [ ] Example notebook runs without errors: `make test-notebooks`
- [ ] Documentation builds: `make docs`
- [ ] Full CI pipeline passes: `make ci`
- [ ] Test coverage remains above 80%: `make test-cov`

#### Manual Verification:
- [ ] Documentation is clear and complete
- [ ] Examples demonstrate all key features
- [ ] API reference is accurate
- [ ] Best practices guide is helpful
- [ ] Users can understand when to use optimization

**Implementation Note**: After completing this phase and all automated verification passes, the implementation is complete. Request final review and testing from the human.

---

## Testing Strategy

### Unit Tests
- OptimizationConfig validation (Phase 1)
- OptimizationMixin methods (Phase 1)
- Distance metrics (l2, KL divergence, Huber) (Phase 1)
- Constraint functions (balance, variance) (Phase 1)
- IPW weight optimization (Phase 2)
- Ensemble model fitting and weighting (Phase 3)
- Component balance optimization (Phase 4)

### Integration Tests
- IPW optimization with bootstrap (Phase 2)
- G-computation ensemble with bootstrap (Phase 3)
- AIPW component optimization with cross-fitting (Phase 4)
- Variance reduction verification (Phase 5)
- Bias preservation verification (Phase 5)
- Backward compatibility verification (Phase 5)

### Manual Testing Steps
1. Generate synthetic data with known treatment effect
2. Run standard estimator and record ATE, variance, diagnostics
3. Run optimized estimator and verify:
   - ATE remains close to true value (unbiased)
   - Variance decreases (efficiency improvement)
   - Diagnostics show successful optimization
4. Vary optimization parameters and observe tradeoffs
5. Test on real-world dataset and verify reasonable results

## Performance Considerations

- Optimization adds computational cost (scipy.optimize calls)
- Use `optimize_weights=False` by default (opt-in)
- Disable optimization in bootstrap samples to reduce cost
- Cache optimization results when possible
- Use efficient distance metrics (L2 is fastest)
- Consider using fewer covariates for balance constraints if performance is critical

## Migration Notes

### For Existing Users
- No changes required - optimization is optional
- All existing code continues to work without modification
- Can gradually adopt optimization by adding `OptimizationConfig`

### For New Features
- Consider adding optimization support to new estimators
- Follow the mixin pattern established here
- Ensure backward compatibility maintained

## References

- Original research: `thoughts/shared/research/2025-10-29-pyrake-optimization-extensibility.md`
- PyRake repository: https://github.com/rwilson4/PyRake
- Synthetic Control pattern: `libs/causal_inference/causal_inference/estimators/synthetic_control.py:406-463`
- Bootstrap pattern: `libs/causal_inference/causal_inference/core/bootstrap.py:326-1250`
