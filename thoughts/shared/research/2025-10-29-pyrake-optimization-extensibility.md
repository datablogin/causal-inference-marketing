---
date: 2025-10-29 17:05:18 UTC
researcher: Claude Code
git_commit: 97e847ea62af3f4b137b116ea463658ab9d7f70c
branch: issue-97-implement-interference-spillover-detection-tools
repository: CausalInferenceTools
topic: "PyRake Optimization Framework Extensibility Across Causal Inference Methods"
tags: [research, codebase, optimization, ipw, g-computation, aipw, constrained-optimization, pyrake]
status: complete
last_updated: 2025-10-29
last_updated_by: Claude Code
---

# Research: PyRake Optimization Framework Extensibility Across Causal Inference Methods

**Date**: 2025-10-29 17:05:18 UTC
**Researcher**: Claude Code
**Git Commit**: 97e847ea62af3f4b137b116ea463658ab9d7f70c
**Branch**: issue-97-implement-interference-spillover-detection-tools
**Repository**: CausalInferenceTools

## Research Question

Can the PyRake optimization approach (combining optimization of error with predictive power for IPW) be extended across the multiple causal inference methods implemented in this codebase (G-computation, IPW, AIPW)?

## Summary

**YES - PyRake-style constrained optimization is highly extensible across all three primary causal inference estimators in this codebase.** The PyRake framework's core concept—explicitly optimizing weights under constraints that balance bias reduction, variance minimization, and covariate balance—can be naturally integrated into:

1. **IPW Estimator**: Replace analytical weight computation with constrained optimization that minimizes distance from baseline propensity weights while satisfying covariate balance and variance constraints
2. **G-Computation**: Add optimization layer for adaptive model selection or ensemble weighting that balances prediction error with model complexity
3. **AIPW**: Optimize the balance between outcome model and propensity score components rather than using the fixed doubly-robust formula

The codebase architecture already demonstrates this pattern in the Synthetic Control estimator (lines 406-463 in `synthetic_control.py`) using `scipy.optimize.minimize` with constraints. The `BaseEstimator` template method pattern and mixin architecture provide clear extension points for adding optimization capabilities without disrupting existing functionality.

## Detailed Findings

### PyRake Methodology Overview

**Source**: https://github.com/rwilson4/PyRake

PyRake implements a constrained optimization framework for calculating balancing weights that simultaneously addresses:

1. **Bias Reduction**: Ensures weighted sample matches population covariate distributions
2. **Variance Minimization**: Limits variance inflation from extreme weights
3. **Predictive Power**: Maintains fidelity to baseline propensity score predictions

**Core Optimization Problem**:
```
minimize D(w, v)
subject to:
  (1/M) X^T w = μ        [Covariate balance constraint]
  (1/M) ||w||_2^2 ≤ φ    [Variance constraint]
  w ≥ 0                   [Non-negativity]
```

Where:
- `D(w, v)` is a distance metric between calculated weights `w` and baseline weights `v`
- `X` is the covariate matrix
- `μ` is the target population mean
- `φ` is the variance constraint parameter

**Key Innovation**: PyRake allows explicit exploration of the bias-variance tradeoff by varying `φ`, enabling dramatic variance reduction with minimal deviation from baseline weights.

### Current IPW Implementation

**Location**: `libs/causal_inference/causal_inference/estimators/ipw.py`

**Current Approach - Analytical Weight Computation**:

The IPW estimator currently computes weights analytically from propensity scores without any optimization:

1. **Propensity Score Estimation** (lines 375-399):
   - Delegates to sklearn models (LogisticRegression or RandomForestClassifier)
   - Uses sklearn's built-in optimization (logistic loss minimization for LR)
   - No custom optimization objective

2. **Weight Computation** (lines 500-563):
   - Analytical formula: `w_treated = 1/e(X)`, `w_control = 1/(1-e(X))`
   - Optional stabilization: multiply by marginal treatment probabilities
   - No optimization loop—weights are deterministic given propensity scores

3. **Weight Truncation** (lines 457-498):
   - Rule-based: percentile truncation or threshold truncation
   - No optimization to find "best" truncation point
   - Fixed thresholds chosen by user

4. **No Explicit Objectives**:
   - No loss function for weight quality
   - No covariate balance constraints
   - No variance penalty
   - Diagnostics (effective sample size, weight variance) computed post-hoc but not optimized

**Gap Relative to PyRake**:
- PyRake optimizes weights directly with explicit variance and balance constraints
- Current IPW trusts propensity score predictions completely (no constraint checking during weight computation)
- Weight truncation is ad-hoc rather than part of unified optimization

### Current G-Computation Implementation

**Location**: `libs/causal_inference/causal_inference/estimators/g_computation.py`

**Current Approach - Sklearn Model Delegation**:

1. **Model Selection** (lines 127-169):
   - Rule-based auto-selection: continuous → LinearRegression, binary → LogisticRegression
   - User can specify model type and hyperparameters
   - No adaptive selection based on data characteristics

2. **Model Fitting** (lines 255-338):
   - Delegates to sklearn's `.fit()` method
   - sklearn optimizes its own loss functions (MSE for linear, log-loss for logistic)
   - No custom optimization objective
   - No explicit bias-variance tradeoff mechanism

3. **Performance Metrics** (lines 308-315):
   - MSE and log-loss computed for diagnostics
   - Not used as optimization objectives
   - No model selection based on these metrics

4. **No Weight Optimization**:
   - Predictions are unweighted averages of counterfactual outcomes
   - No mechanism to balance prediction error against model complexity
   - Bootstrap provides CIs but not optimization guidance

**Gap Relative to PyRake**:
- PyRake formulation could optimize model selection or ensemble weights
- Could balance prediction accuracy with model simplicity
- Could add constraints on model complexity or computational budget

### Current AIPW Implementation

**Location**: `libs/causal_inference/causal_inference/estimators/aipw.py`

**Current Approach - Fixed Doubly-Robust Formula**:

1. **Component Creation** (lines 219-249):
   - Creates independent G-computation and IPW estimators
   - No coordination during fitting
   - Each uses standard methods

2. **Doubly-Robust Combination** (lines 745-816):
   - Fixed mathematical formula: `τ = E[μ₁ - μ₀ + (T/e(X))(Y - μ₁) - ((1-T)/(1-e(X)))(Y - μ₀)]`
   - G-computation component: `μ₁ - μ₀`
   - IPW correction: weighted residuals
   - No optimization of component balance

3. **Component Diagnostics** (lines 1055-1130):
   - Computes separate G-computation and IPW ATEs for comparison
   - Measures relative contribution of each component
   - Reports component balance metrics
   - **BUT**: These are diagnostic only, not used for optimization

4. **Cross-Fitting** (lines 372-667):
   - Reduces overfitting bias through sample splitting
   - Still uses fixed doubly-robust formula within each fold

**Gap Relative to PyRake**:
- Could optimize weight given to G-computation vs IPW components
- Could add constraint that each component contributes meaningfully
- Could minimize variance while maintaining bias correction properties
- Component balance diagnostics suggest natural optimization targets

### BaseEstimator Architecture and Extension Points

**Location**: `libs/causal_inference/causal_inference/core/base.py`

The architecture provides multiple extension points for optimization:

#### 1. Configuration Pattern (lines 638-656)

Current bootstrap integration:
```python
def __init__(self, bootstrap_config: BootstrapConfig | None = None, ...):
    self.bootstrap_config = bootstrap_config
```

**Extension Pattern**: Add `OptimizationConfig` similar to `BootstrapConfig`:
```python
def __init__(self,
             bootstrap_config: BootstrapConfig | None = None,
             optimization_config: OptimizationConfig | None = None,
             ...):
    self.bootstrap_config = bootstrap_config
    self.optimization_config = optimization_config
```

#### 2. Mixin Architecture

Current pattern:
```python
class IPWEstimator(BootstrapMixin, BaseEstimator):  # Line 115 in ipw.py
```

**Extension Pattern**: Add `OptimizationMixin`:
```python
class IPWEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):
```

The mixin could provide:
- `optimize_weights()` method
- `compute_optimization_diagnostics()` method
- `_create_optimization_problem()` abstract method for subclass customization

#### 3. Abstract Method Hooks (lines 670-696)

Current abstract methods:
- `_fit_implementation()`: Core estimation logic
- `_estimate_ate_implementation()`: ATE computation

**Extension Pattern**: Add optimization within `_fit_implementation()`:
```python
def _fit_implementation(self, treatment, outcome, covariates):
    # Existing fitting logic
    self._fit_propensity_model(treatment, covariates)
    self.propensity_scores = self._estimate_propensity_scores()

    # NEW: Optimization layer
    if self.optimization_config and self.optimization_config.optimize_weights:
        self.weights = self._optimize_weights_constrained(
            baseline_weights=self._compute_baseline_weights(),
            covariates=covariates,
            constraints=self.optimization_config.constraints
        )
    else:
        # Existing analytical computation
        self.weights = self._compute_weights(treatment, self.propensity_scores)
```

#### 4. Feature Preparation Hooks (g_computation.py lines 171-253)

Existing pattern:
```python
def _prepare_features(self, treatment, covariates) -> pd.DataFrame:
def _prepare_features_efficient(self, treatment, covariates, n_samples) -> pd.DataFrame:
```

**Extension Pattern**: Add optimization-aware feature selection:
```python
def _prepare_features_optimized(self, treatment, covariates, optimization_config):
    features = self._prepare_features(treatment, covariates)

    if optimization_config.feature_selection:
        features = self._select_features_constrained(
            features=features,
            target=self.outcome_data.values,
            max_features=optimization_config.max_features
        )

    return features
```

#### 5. Prediction Strategy Hooks (g_computation.py lines 340-485)

Current routing logic:
```python
def _predict_counterfactuals(self, treatment_value, covariates):
    if self.memory_efficient and n_obs >= self.large_dataset_threshold:
        return self._predict_counterfactuals_chunked(...)
    else:
        return self._predict_counterfactuals_regular(...)
```

**Extension Pattern**: Add ensemble prediction with optimized weights:
```python
def _predict_counterfactuals(self, treatment_value, covariates):
    if self.optimization_config and self.optimization_config.ensemble_optimization:
        return self._predict_counterfactuals_ensemble_optimized(...)
    elif self.memory_efficient and n_obs >= self.large_dataset_threshold:
        return self._predict_counterfactuals_chunked(...)
    else:
        return self._predict_counterfactuals_regular(...)
```

### Existing Optimization Patterns in Codebase

The codebase already demonstrates sophisticated constrained optimization in several places:

#### 1. Synthetic Control - Direct PyRake Analog

**Location**: `libs/causal_inference/causal_inference/estimators/synthetic_control.py:406-463`

```python
def _optimize_weights(self, treated_pre, control_pre):
    """Optimize weights for synthetic control."""

    def objective(weights):
        """Minimize squared prediction error."""
        synthetic_pre = np.dot(weights, control_pre)
        mse = np.mean((treated_pre - synthetic_pre) ** 2)

        # L2 penalty on weights
        if self.weight_penalty > 0:
            mse += self.weight_penalty * np.sum(weights**2)

        return mse

    # Constraints: weights sum to 1 and non-negative
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_control)]

    # Optimize using scipy
    result = minimize(
        objective,
        initial_weights,
        method=self.optimization_method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    return result.x
```

**Key Parallels to PyRake**:
- ✅ Uses `scipy.optimize.minimize` with constraints
- ✅ Equality constraint (weights sum to 1) - analogous to covariate balance
- ✅ Box constraints (0 ≤ w ≤ 1) - analogous to non-negativity
- ✅ Regularization term (L2 penalty) - analogous to variance constraint
- ✅ Objective minimizes prediction error

**This demonstrates the codebase already has the infrastructure and patterns for PyRake-style optimization!**

#### 2. Policy Optimization with CVXPY

**Location**: `libs/causal_inference/causal_inference/policy/optimization.py:227-314`

```python
def _optimize_ilp(self, uplifts, costs, budget, max_treatment_rate, fairness_constraints):
    """Integer Linear Programming optimization."""

    # Decision variables
    x = cp.Variable(n_individuals, boolean=True)

    # Objective: maximize total uplift
    objective = cp.Maximize(uplifts @ x)

    # Constraints
    constraints = []
    if budget is not None:
        constraints.append(costs @ x <= budget)
    if max_treatment_rate is not None:
        constraints.append(cp.sum(x) <= int(max_treatment_rate * n_individuals))

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=self.solver, verbose=self.verbose)
```

**Demonstrates**:
- Constrained optimization with multiple constraint types
- CVXPY integration for convex problems
- Graceful fallback to simpler methods on failure

#### 3. G-Estimation with Custom Objective

**Location**: `libs/causal_inference/causal_inference/estimators/g_estimation.py:334-449`

```python
def _objective_function(self, params, outcome, treatment, propensity_scores, covariates):
    """Objective: squared difference in IPW-weighted means of adjusted outcomes."""

    # Apply structural model
    adjusted_outcome = self._apply_structural_model(outcome, treatment, parameters, covariates)

    # IPW-weighted means
    treated_mean = np.average(adjusted_outcome[treated_mask], weights=treated_weights)
    control_mean = np.average(adjusted_outcome[control_mask], weights=control_weights)

    # Minimize squared difference
    return (treated_mean - control_mean) ** 2
```

**Demonstrates**:
- Custom objective functions beyond sklearn's built-in losses
- Integration of propensity scores into optimization
- Grid search and gradient-based methods

#### 4. Score-Based DAG Discovery with L-BFGS-B

**Location**: `libs/causal_inference/causal_inference/discovery/score_based.py:662-731`

```python
def optimize_dag(x, w_init, alpha, rho):
    def objective(w_vec):
        w_mat = w_vec.reshape(w.shape)

        # Likelihood term
        residual = x - x @ w_mat
        likelihood = 0.5 / x.shape[0] * np.sum(residual**2)

        # Regularization
        l1_reg = lambda_l1 * np.sum(np.abs(w_mat))
        l2_reg = lambda_l2 * np.sum(w_mat**2)

        # Acyclicity constraint (augmented Lagrangian)
        h = self._h_func(w_mat)
        augmented_lagrangian = alpha * h + 0.5 * rho * h**2

        return likelihood + l1_reg + l2_reg + augmented_lagrangian

    def gradient(w_vec):
        # Analytical gradient computation
        ...

    result = optimize.minimize(
        objective, w_init, method="L-BFGS-B", jac=gradient
    )
```

**Demonstrates**:
- Composite objectives with multiple terms
- Augmented Lagrangian for constraint handling
- Analytical gradient computation for efficiency

## Extensibility Analysis by Estimator

### IPW Estimator - Highest Extensibility

**Current Gap**: Largest gap between current implementation and PyRake philosophy
- ✅ Already has propensity scores (baseline weights)
- ✅ Already has covariate matrix
- ✅ Already computes weight diagnostics
- ❌ No weight optimization
- ❌ No balance constraints
- ❌ No explicit variance penalty

**PyRake Extension Pattern**:

```python
class IPWEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):

    def __init__(self,
                 optimization_config: OptimizationConfig | None = None,
                 ...):
        self.optimization_config = optimization_config

    def _fit_implementation(self, treatment, outcome, covariates):
        # Existing: fit propensity model
        self._fit_propensity_model(treatment, covariates)
        self.propensity_scores = self._estimate_propensity_scores()

        # NEW: Optimize weights with constraints
        if self.optimization_config and self.optimization_config.optimize_weights:
            self.weights = self._optimize_weights_pyrake_style(
                baseline_weights=1.0 / self.propensity_scores,  # v in PyRake
                covariates=covariates,
                target_means=self.optimization_config.target_means,
                variance_constraint=self.optimization_config.variance_penalty,
                distance_metric=self.optimization_config.distance_metric
            )
        else:
            # Existing analytical computation
            self.weights = self._compute_weights(treatment, self.propensity_scores)

    def _optimize_weights_pyrake_style(self, baseline_weights, covariates,
                                       target_means, variance_constraint,
                                       distance_metric="l2"):
        """Optimize weights using PyRake-style constrained optimization."""
        n_obs = len(baseline_weights)
        X = covariates.values

        def objective(w):
            """Distance from baseline weights."""
            if distance_metric == "l2":
                return np.sum((w - baseline_weights) ** 2)
            elif distance_metric == "kl":
                return np.sum(w * np.log(w / baseline_weights))
            elif distance_metric == "huber":
                diff = w - baseline_weights
                delta = 1.0
                return np.sum(np.where(
                    np.abs(diff) <= delta,
                    0.5 * diff**2,
                    delta * (np.abs(diff) - 0.5 * delta)
                ))

        def covariate_balance_constraint(w):
            """Constraint: (1/n) X^T w = μ"""
            weighted_means = (X.T @ w) / n_obs
            return weighted_means - target_means

        def variance_constraint_func(w):
            """Constraint: (1/n) ||w||^2 ≤ φ"""
            return variance_constraint - np.sum(w**2) / n_obs

        constraints = [
            {"type": "eq", "fun": covariate_balance_constraint},
            {"type": "ineq", "fun": variance_constraint_func}
        ]

        bounds = [(0, None) for _ in range(n_obs)]  # Non-negativity

        result = minimize(
            objective,
            baseline_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )

        if not result.success:
            warnings.warn(f"Weight optimization failed: {result.message}")
            return baseline_weights  # Fall back to analytical weights

        return result.x
```

**Benefits**:
1. **Explicit Bias-Variance Tradeoff**: User can vary `variance_constraint` to explore tradeoff
2. **Guaranteed Balance**: Covariate balance enforced as hard constraint
3. **Flexible Distance Metrics**: L2, KL divergence, or Huber loss
4. **Backwards Compatible**: Falls back to analytical weights if optimization disabled
5. **Diagnostic Rich**: Optimization diagnostics (convergence, iterations, objective value) available

**Integration Points**:
- Lines 500-563 in `ipw.py` - replace `_compute_weights()` with conditional optimization
- Lines 565-586 in `ipw.py` - extend `_compute_weight_diagnostics()` with optimization diagnostics
- Lines 224-248 in `ipw.py` - add optimization parameters to model configuration

### G-Computation Estimator - Moderate Extensibility

**Current Gap**: Less direct mapping but still significant potential
- ✅ Already has multiple model types (linear, logistic, random forest)
- ✅ Already has performance metrics (MSE, log-loss)
- ❌ No model selection optimization
- ❌ No ensemble weighting
- ❌ No explicit complexity penalty

**PyRake Extension Pattern 1 - Ensemble Weighting**:

```python
class GComputationEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):

    def _fit_implementation(self, treatment, outcome, covariates):
        if self.optimization_config and self.optimization_config.use_ensemble:
            # Fit multiple models
            self.models = self._fit_model_ensemble(treatment, outcome, covariates)

            # Optimize ensemble weights using PyRake-style constraints
            self.ensemble_weights = self._optimize_ensemble_weights(
                models=self.models,
                outcome=outcome,
                variance_constraint=self.optimization_config.ensemble_variance_penalty
            )
        else:
            # Existing single model fitting
            self.outcome_model = self._select_model(outcome.outcome_type)
            self.outcome_model.fit(X, y)

    def _optimize_ensemble_weights(self, models, outcome, variance_constraint):
        """Optimize ensemble weights with variance penalty."""
        n_models = len(models)

        # Get predictions from each model
        predictions = np.column_stack([
            model.predict(X) for model in models
        ])

        def objective(weights):
            """MSE with variance penalty."""
            ensemble_pred = predictions @ weights
            mse = np.mean((outcome.values - ensemble_pred) ** 2)

            # Variance penalty (encourage diverse weights)
            variance_penalty = variance_constraint * np.var(weights)

            return mse + variance_penalty

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(
            objective,
            np.ones(n_models) / n_models,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        return result.x
```

**PyRake Extension Pattern 2 - Adaptive Model Selection**:

```python
def _select_model_optimized(self, outcome_type, X, y):
    """Select model using cross-validated performance with complexity penalty."""

    candidate_models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=5),
    }

    best_score = float("inf")
    best_model = None

    for name, model in candidate_models.items():
        # Cross-validated performance
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
        mse = -cv_scores.mean()

        # Complexity penalty (PyRake-style variance penalty)
        complexity_penalty = self._compute_model_complexity(model)

        # Combined objective
        objective = mse + self.optimization_config.complexity_weight * complexity_penalty

        if objective < best_score:
            best_score = objective
            best_model = model

    return best_model
```

**Benefits**:
1. **Model Selection**: Automatic selection with explicit complexity tradeoff
2. **Ensemble Diversity**: Variance penalty encourages diverse model weights
3. **Performance Optimization**: Directly minimizes prediction error
4. **Backwards Compatible**: Single model path still available

### AIPW Estimator - Moderate to High Extensibility

**Current Gap**: Fixed doubly-robust formula, but diagnostics already measure component balance
- ✅ Already has component diagnostics (lines 1055-1130)
- ✅ Already computes separate G-computation and IPW estimates
- ✅ Already measures component contributions
- ❌ No optimization of component weights
- ❌ Fixed 50/50 implicit weighting in doubly-robust formula

**PyRake Extension Pattern - Component Weight Optimization**:

```python
class AIPWEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):

    def _estimate_ate_implementation(self):
        # Existing: compute AIPW components
        aipw_ate_fixed = self._compute_aipw_estimate(self.treatment_data, self.outcome_data)

        # NEW: Optimize component balance
        if self.optimization_config and self.optimization_config.optimize_component_balance:
            aipw_ate = self._compute_aipw_optimized(
                self.treatment_data,
                self.outcome_data,
                variance_penalty=self.optimization_config.component_variance_penalty
            )
        else:
            aipw_ate = aipw_ate_fixed

        ...

    def _compute_aipw_optimized(self, treatment, outcome, variance_penalty):
        """Optimize balance between G-computation and IPW components."""

        # Get component-level estimates for each observation
        g_comp_components = self._aipw_components["g_computation"]  # μ₁ - μ₀
        ipw_components = self._aipw_components["ipw_correction"]    # IPW residuals

        # Optimize weights on components
        def objective(alpha):
            """Weighted AIPW with variance penalty."""
            # alpha is weight on G-computation (0 to 1)
            # (1 - alpha) is implicit weight on IPW

            combined = alpha * g_comp_components + (1 - alpha) * ipw_components

            # Variance of estimate
            estimate_variance = np.var(combined)

            # Penalty for extreme weights (force meaningful contribution from both)
            balance_penalty = variance_penalty * (alpha - 0.5) ** 2

            return estimate_variance + balance_penalty

        result = minimize_scalar(
            objective,
            bounds=(0.3, 0.7),  # Ensure both components contribute
            method="bounded"
        )

        optimal_alpha = result.x

        # Compute optimized AIPW
        combined_components = (
            optimal_alpha * g_comp_components +
            (1 - optimal_alpha) * ipw_components
        )

        # Store diagnostics
        self._optimization_diagnostics = {
            "optimal_g_computation_weight": optimal_alpha,
            "optimal_ipw_weight": 1 - optimal_alpha,
            "optimized_variance": np.var(combined_components),
            "fixed_variance": np.var(g_comp_components + ipw_components),
            "variance_reduction": (
                np.var(g_comp_components + ipw_components) - np.var(combined_components)
            ),
        }

        return np.mean(combined_components)
```

**Alternative Pattern - Doubly Robust with Constraints**:

```python
def _compute_aipw_with_balance_constraints(self, treatment, outcome, covariates):
    """AIPW with explicit covariate balance constraints on combined weights."""

    # This more directly mirrors PyRake by treating the full AIPW weight as optimization target
    n_obs = len(treatment.values)

    # Baseline AIPW weights (from standard formula)
    baseline_weights = self._compute_aipw_weights_analytical()

    def objective(w):
        """Distance from baseline AIPW weights."""
        return np.sum((w - baseline_weights) ** 2)

    def balance_constraint(w):
        """Covariate balance constraint."""
        X = covariates.values
        weighted_means = (X.T @ w) / n_obs
        target_means = np.mean(X, axis=0)
        return weighted_means - target_means

    def variance_constraint(w):
        """Variance constraint."""
        return self.optimization_config.variance_threshold - np.var(w)

    constraints = [
        {"type": "eq", "fun": balance_constraint},
        {"type": "ineq", "fun": variance_constraint}
    ]

    result = minimize(
        objective,
        baseline_weights,
        method="SLSQP",
        constraints=constraints
    )

    return result.x
```

**Benefits**:
1. **Adaptive Component Balance**: Weights components based on data rather than fixed formula
2. **Variance Reduction**: Explicit penalty on estimate variance
3. **Diagnostics**: Quantifies variance reduction from optimization
4. **Doubly Robust Maintained**: Still combines both components, just with optimized weights
5. **Balance Constraints**: Can add PyRake-style covariate balance on combined weights

## Architectural Integration Strategy

### Step 1: Create Optimization Configuration

Similar to `BootstrapConfig` (bootstrap.py lines 101-193), create `OptimizationConfig`:

```python
# In libs/causal_inference/causal_inference/core/optimization_config.py

from pydantic import BaseModel, Field
from typing import Literal

class OptimizationConfig(BaseModel):
    """Configuration for PyRake-style constrained optimization."""

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
    variance_constraint: float | None = Field(
        default=None,
        description="Maximum allowed weight variance (φ in PyRake)"
    )

    balance_constraints: bool = Field(
        default=True,
        description="Enforce covariate balance constraints"
    )

    balance_tolerance: float = Field(
        default=0.01,
        description="Tolerance for covariate balance (SMD units)"
    )

    # Distance metrics
    distance_metric: Literal["l2", "kl_divergence", "huber"] = Field(
        default="l2",
        description="Distance metric for weight optimization"
    )

    # Ensemble settings (for G-computation)
    use_ensemble: bool = Field(
        default=False,
        description="Use ensemble of models instead of single model"
    )

    ensemble_variance_penalty: float = Field(
        default=0.1,
        description="Penalty on ensemble weight variance"
    )

    # Component balance settings (for AIPW)
    optimize_component_balance: bool = Field(
        default=False,
        description="Optimize G-computation vs IPW balance in AIPW"
    )

    component_variance_penalty: float = Field(
        default=0.5,
        description="Penalty for deviating from 50/50 component balance"
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
```

### Step 2: Create Optimization Mixin

Similar to `BootstrapMixin` (bootstrap.py lines 326-582), create `OptimizationMixin`:

```python
# In libs/causal_inference/causal_inference/core/optimization_mixin.py

from typing import Any
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from numpy.typing import NDArray

class OptimizationMixin:
    """Mixin providing PyRake-style constrained optimization capabilities."""

    optimization_config: OptimizationConfig
    _optimization_diagnostics: dict[str, Any]

    def optimize_weights_constrained(
        self,
        baseline_weights: NDArray[Any],
        covariates: NDArray[Any],
        target_means: NDArray[Any] | None = None,
        variance_constraint: float | None = None,
    ) -> NDArray[Any]:
        """Optimize weights using PyRake-style constrained optimization.

        Args:
            baseline_weights: Initial/baseline weights (e.g., 1/e(X) for IPW)
            covariates: Covariate matrix
            target_means: Target covariate means (default: observed means)
            variance_constraint: Maximum weight variance (default: from config)

        Returns:
            Optimized weights
        """
        if not self.optimization_config or not self.optimization_config.optimize_weights:
            return baseline_weights

        n_obs = len(baseline_weights)

        if target_means is None:
            target_means = np.mean(covariates, axis=0)

        if variance_constraint is None:
            variance_constraint = self.optimization_config.variance_constraint

        def objective(w: NDArray[Any]) -> float:
            """Distance from baseline weights."""
            metric = self.optimization_config.distance_metric

            if metric == "l2":
                return float(np.sum((w - baseline_weights) ** 2))
            elif metric == "kl_divergence":
                # KL divergence: sum(w * log(w / baseline))
                # Add small epsilon to avoid log(0)
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

        constraints = []

        # Covariate balance constraint
        if self.optimization_config.balance_constraints:
            def balance_constraint(w: NDArray[Any]) -> NDArray[Any]:
                weighted_means = (covariates.T @ w) / n_obs
                return weighted_means - target_means

            constraints.append({
                "type": "eq",
                "fun": balance_constraint
            })

        # Variance constraint
        if variance_constraint is not None:
            def variance_constraint_func(w: NDArray[Any]) -> float:
                return float(variance_constraint - np.sum(w**2) / n_obs)

            constraints.append({
                "type": "ineq",
                "fun": variance_constraint_func
            })

        # Non-negativity bounds
        bounds = [(0, None) for _ in range(n_obs)]

        # Optimize
        result = minimize(
            objective,
            baseline_weights,
            method=self.optimization_config.method,
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self.optimization_config.max_iterations,
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
                "constraint_violation": self._compute_constraint_violation(result.x, covariates, target_means),
                "weight_variance": float(np.var(result.x)),
                "effective_sample_size": float(np.sum(result.x)**2 / np.sum(result.x**2)),
            }

        if not result.success:
            import warnings
            warnings.warn(
                f"Weight optimization did not converge: {result.message}. "
                f"Falling back to baseline weights."
            )
            return baseline_weights

        return result.x

    def _compute_constraint_violation(
        self,
        weights: NDArray[Any],
        covariates: NDArray[Any],
        target_means: NDArray[Any]
    ) -> float:
        """Compute standardized mean difference for covariate balance."""
        weighted_means = (covariates.T @ weights) / len(weights)
        smd = np.abs(weighted_means - target_means) / (np.std(covariates, axis=0) + 1e-10)
        return float(np.max(smd))  # Maximum SMD across covariates

    def get_optimization_diagnostics(self) -> dict[str, Any] | None:
        """Get optimization diagnostics."""
        return getattr(self, "_optimization_diagnostics", None)
```

### Step 3: Integrate into Estimators

#### IPW Estimator Integration

```python
# Modify libs/causal_inference/causal_inference/estimators/ipw.py

class IPWEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):
    """Inverse Probability Weighting estimator with optional PyRake-style optimization."""

    def __init__(
        self,
        propensity_model_type: str = "logistic",
        propensity_model_params: dict[str, Any] | None = None,
        weight_truncation: str | None = None,
        truncation_threshold: float = 0.01,
        stabilized_weights: bool = False,
        bootstrap_config: Any | None = None,
        optimization_config: OptimizationConfig | None = None,  # NEW
        ...
    ):
        super().__init__(
            bootstrap_config=bootstrap_config,
            optimization_config=optimization_config,  # NEW
            random_state=random_state,
            verbose=verbose,
        )

        self.optimization_config = optimization_config  # NEW
        ...

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

        # Check overlap
        if self.check_overlap:
            self._overlap_diagnostics = self._check_overlap(self.propensity_scores)

        # NEW: Optimize weights if configured
        if self.optimization_config and self.optimization_config.optimize_weights:
            # Compute baseline weights
            baseline_weights = self._compute_baseline_weights(treatment, self.propensity_scores)

            # Optimize with PyRake-style constraints
            self.weights = self.optimize_weights_constrained(
                baseline_weights=baseline_weights,
                covariates=covariates.values if isinstance(covariates.values, np.ndarray) else covariates.values.values,
                variance_constraint=self.optimization_config.variance_constraint,
            )

            if self.verbose:
                opt_diag = self.get_optimization_diagnostics()
                if opt_diag:
                    print(f"Weight optimization converged: {opt_diag['success']}")
                    print(f"Final objective: {opt_diag['final_objective']:.6f}")
                    print(f"Max covariate imbalance (SMD): {opt_diag['constraint_violation']:.4f}")
                    print(f"Effective sample size: {opt_diag['effective_sample_size']:.1f}")
        else:
            # Existing: analytical weight computation
            self.weights = self._compute_weights(treatment, self.propensity_scores)

        # Compute weight diagnostics
        self._weight_diagnostics = self._compute_weight_diagnostics(self.weights)

    def _compute_baseline_weights(
        self,
        treatment: TreatmentData,
        propensity_scores: NDArray[Any]
    ) -> NDArray[Any]:
        """Compute baseline IPW weights for optimization starting point."""
        # This is the current analytical computation
        return self._compute_weights(treatment, propensity_scores)
```

### Step 4: Testing and Validation

Create comprehensive tests demonstrating PyRake-style optimization:

```python
# In libs/causal_inference/tests/test_optimization_integration.py

import pytest
import numpy as np
from causal_inference.core.optimization_config import OptimizationConfig
from causal_inference.estimators.ipw import IPWEstimator
from causal_inference.core.base import TreatmentData, OutcomeData, CovariateData

def test_ipw_with_pyrake_optimization():
    """Test IPW with PyRake-style constrained weight optimization."""

    # Generate synthetic data with known DGP
    np.random.seed(42)
    n = 1000

    # Covariates
    X = np.random.randn(n, 3)

    # Propensity score with strong confounding
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = np.random.binomial(1, propensity)

    # Outcome with treatment effect = 2.0
    outcome = 2.0 * treatment + X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

    # Prepare data
    treatment_data = TreatmentData(values=treatment, treatment_type="binary")
    outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
    covariate_data = CovariateData(values=X, names=["X1", "X2", "X3"])

    # Fit with optimization
    optimization_config = OptimizationConfig(
        optimize_weights=True,
        variance_constraint=2.0,  # Constrain variance
        balance_constraints=True,
        balance_tolerance=0.1,
        distance_metric="l2",
        verbose=True
    )

    estimator_optimized = IPWEstimator(
        propensity_model_type="logistic",
        optimization_config=optimization_config,
        bootstrap_samples=100,
        random_state=42,
        verbose=True
    )

    estimator_optimized.fit(treatment_data, outcome_data, covariate_data)
    effect_optimized = estimator_optimized.estimate_ate()

    # Fit without optimization (standard IPW)
    estimator_standard = IPWEstimator(
        propensity_model_type="logistic",
        bootstrap_samples=100,
        random_state=42,
        verbose=True
    )

    estimator_standard.fit(treatment_data, outcome_data, covariate_data)
    effect_standard = estimator_standard.estimate_ate()

    # Check optimization diagnostics
    opt_diag = estimator_optimized.get_optimization_diagnostics()
    assert opt_diag is not None
    assert opt_diag["success"]
    assert opt_diag["constraint_violation"] < 0.1  # Good covariate balance

    # Check weight diagnostics
    weight_diag_opt = estimator_optimized.get_weight_diagnostics()
    weight_diag_std = estimator_standard.get_weight_diagnostics()

    # Optimized weights should have lower variance
    assert weight_diag_opt["weight_variance"] < weight_diag_std["weight_variance"]

    # Optimized weights should have higher effective sample size
    assert weight_diag_opt["effective_sample_size"] > weight_diag_std["effective_sample_size"]

    # Both should recover true effect (2.0) within CI
    assert 1.5 < effect_optimized.ate < 2.5
    assert 1.5 < effect_standard.ate < 2.5

    # Optimized estimate should have tighter CI (lower variance)
    ci_width_opt = effect_optimized.ate_ci_upper - effect_optimized.ate_ci_lower
    ci_width_std = effect_standard.ate_ci_upper - effect_standard.ate_ci_lower

    assert ci_width_opt < ci_width_std

    print("\n=== Optimization Results ===")
    print(f"Standard IPW ATE: {effect_standard.ate:.3f} [{effect_standard.ate_ci_lower:.3f}, {effect_standard.ate_ci_upper:.3f}]")
    print(f"Optimized IPW ATE: {effect_optimized.ate:.3f} [{effect_optimized.ate_ci_lower:.3f}, {effect_optimized.ate_ci_upper:.3f}]")
    print(f"\nVariance reduction: {(1 - weight_diag_opt['weight_variance'] / weight_diag_std['weight_variance']) * 100:.1f}%")
    print(f"ESS improvement: {(weight_diag_opt['effective_sample_size'] / weight_diag_std['effective_sample_size'] - 1) * 100:.1f}%")
    print(f"CI width reduction: {(1 - ci_width_opt / ci_width_std) * 100:.1f}%")
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
1. Create `OptimizationConfig` class with PyRake parameters
2. Create `OptimizationMixin` with constrained optimization methods
3. Add integration tests demonstrating patterns
4. Document optimization API

### Phase 2: IPW Integration (Weeks 3-4)
1. Add `optimization_config` parameter to IPW estimator
2. Implement `_optimize_weights_constrained()` method
3. Add optimization diagnostics to weight diagnostics
4. Create comprehensive tests with synthetic data
5. Add examples showing bias-variance tradeoff exploration

### Phase 3: G-Computation Integration (Weeks 5-6)
1. Implement ensemble model fitting
2. Add `_optimize_ensemble_weights()` method
3. Implement adaptive model selection with complexity penalty
4. Test on multiple outcome types
5. Document ensemble usage patterns

### Phase 4: AIPW Integration (Weeks 7-8)
1. Implement component balance optimization
2. Add `_compute_aipw_optimized()` method
3. Extend component diagnostics with optimization metrics
4. Test variance reduction vs standard AIPW
5. Create visualization of component balance

### Phase 5: Documentation and Validation (Weeks 9-10)
1. Write comprehensive user guide
2. Create tutorial notebooks
3. Benchmark against standard methods
4. Add case studies with real data
5. Update API documentation

## Code References

### PyRake Repository
- Main repository: https://github.com/rwilson4/PyRake
- README: https://github.com/rwilson4/PyRake/blob/master/README.md

### IPW Estimator
- `libs/causal_inference/causal_inference/estimators/ipw.py:375-399` - Propensity score estimation
- `libs/causal_inference/causal_inference/estimators/ipw.py:500-563` - Weight computation
- `libs/causal_inference/causal_inference/estimators/ipw.py:457-498` - Weight truncation
- `libs/causal_inference/causal_inference/estimators/ipw.py:565-586` - Weight diagnostics

### G-Computation Estimator
- `libs/causal_inference/causal_inference/estimators/g_computation.py:127-169` - Model selection
- `libs/causal_inference/causal_inference/estimators/g_computation.py:255-338` - Model fitting
- `libs/causal_inference/causal_inference/estimators/g_computation.py:340-485` - Counterfactual prediction

### AIPW Estimator
- `libs/causal_inference/causal_inference/estimators/aipw.py:219-249` - Component estimators
- `libs/causal_inference/causal_inference/estimators/aipw.py:745-816` - Doubly robust formula
- `libs/causal_inference/causal_inference/estimators/aipw.py:1055-1130` - Component diagnostics
- `libs/causal_inference/causal_inference/estimators/aipw.py:372-667` - Cross-fitting

### BaseEstimator
- `libs/causal_inference/causal_inference/core/base.py:670-696` - Abstract methods
- `libs/causal_inference/causal_inference/core/base.py:698-806` - Template methods
- `libs/causal_inference/causal_inference/core/base.py:638-656` - Bootstrap integration

### Existing Optimization Patterns
- `libs/causal_inference/causal_inference/estimators/synthetic_control.py:406-463` - Constrained weight optimization (PyRake analog)
- `libs/causal_inference/causal_inference/policy/optimization.py:227-314` - CVXPY ILP optimization
- `libs/causal_inference/causal_inference/estimators/g_estimation.py:334-449` - Custom objective optimization
- `libs/causal_inference/causal_inference/discovery/score_based.py:662-731` - L-BFGS-B with gradient

## Related Research

This research builds on several threads in the codebase and external methodologies:

1. **Constrained Optimization for Causal Inference**: The Synthetic Control estimator already demonstrates this pattern successfully
2. **Doubly Robust Methods**: AIPW provides natural target for component balance optimization
3. **Survey Weighting Methods**: PyRake's roots in survey methodology translate naturally to causal inference
4. **Variance Reduction Techniques**: Explicit variance constraints complement existing bootstrap methods

## Key Findings Summary

1. ✅ **PyRake is highly extensible** across IPW, G-computation, and AIPW estimators
2. ✅ **Architecture supports integration** through mixin pattern and configuration objects
3. ✅ **Existing optimization patterns** in codebase demonstrate feasibility (Synthetic Control)
4. ✅ **Clear extension points** identified in BaseEstimator and each estimator
5. ✅ **Benefits are substantial**: Variance reduction, covariate balance guarantees, explicit bias-variance tradeoffs
6. ✅ **Backwards compatible**: Can be added as optional feature without breaking existing API
7. ✅ **Implementation path is clear**: Config → Mixin → Estimator integration → Testing

## Conclusion

The PyRake optimization framework combining error minimization with predictive power is **highly extensible** across the causal inference methods in this codebase. The IPW estimator provides the most direct analog (optimizing weights with balance and variance constraints), but both G-computation (ensemble/model selection optimization) and AIPW (component balance optimization) offer natural integration points.

The codebase architecture—with its template method pattern, mixin composition, and configuration-based feature flags—is well-suited for this extension. The Synthetic Control estimator already demonstrates successful constrained weight optimization using the exact same scipy.optimize patterns that would be used for PyRake integration.

Implementation would follow the established pattern: create `OptimizationConfig` and `OptimizationMixin` (mirroring `BootstrapConfig` and `BootstrapMixin`), then integrate into estimators through existing extension hooks. The approach is backwards compatible and can be rolled out incrementally, starting with IPW as the clearest use case.
