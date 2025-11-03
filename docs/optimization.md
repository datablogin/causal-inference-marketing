# Optimization Framework

The CausalInferenceTools library supports PyRake-style constrained optimization for improving estimator efficiency while maintaining unbiasedness.

## Overview

PyRake optimization simultaneously addresses three key objectives:

- **Bias Reduction**: Maintains covariate balance through equality constraints
- **Variance Minimization**: Reduces weight variance via inequality constraints
- **Predictive Power**: Minimizes distance from baseline weights (propensity-based or model-based)

This framework is available across multiple estimators (IPW, G-computation, AIPW) and is completely optional—all estimators work without optimization using their standard analytical methods.

## Why Use Optimization?

Causal inference estimators often face a bias-variance tradeoff:
- **Standard methods** (e.g., analytical IPW weights) are unbiased but can have high variance
- **Constrained optimization** can reduce variance while maintaining unbiasedness through careful constraint design

Benefits of optimization:
1. **Lower variance** in treatment effect estimates
2. **Tighter confidence intervals** with better coverage
3. **Improved effective sample size** through better weight distribution
4. **Better covariate balance** through explicit balance constraints

## Configuration

Use `OptimizationConfig` to control optimization behavior:

```python
from causal_inference.core import OptimizationConfig

config = OptimizationConfig(
    optimize_weights=True,              # Enable optimization
    variance_constraint=2.0,            # Max weight variance (φ in PyRake)
    balance_constraints=True,           # Enforce covariate balance
    balance_tolerance=0.01,             # Balance tolerance (SMD units)
    distance_metric="l2",               # Distance metric (l2, kl_divergence, huber)
    method="SLSQP",                     # Scipy optimizer (SLSQP, trust-constr, COBYLA)
    max_iterations=1000,                # Max optimization iterations
    convergence_tolerance=1e-6,         # Convergence tolerance
    verbose=True,                       # Print optimization progress
    store_diagnostics=True              # Store detailed diagnostics
)
```

### Key Parameters

**optimize_weights** (bool, default=False)
- Master switch to enable/disable optimization
- When False, estimators use standard analytical methods
- Must be True to use any optimization features

**variance_constraint** (float or None, default=None)
- Maximum allowed weight variance (inequality constraint)
- Lower values force more uniform weights (lower variance, potentially higher bias)
- Higher values allow more extreme weights (higher variance, potentially lower bias)
- None = no variance constraint
- Recommended range: 1.0 to 3.0 for typical applications

**balance_constraints** (bool, default=True)
- Enforce covariate balance through equality constraints
- Ensures weighted covariate means match target means
- Critical for maintaining unbiasedness
- Should generally be True when using optimization

**balance_tolerance** (float, default=0.01)
- Tolerance for covariate balance in standardized mean difference (SMD) units
- Smaller values enforce stricter balance (may be harder to achieve)
- Larger values allow more imbalance (may introduce bias)
- Range: 0.0 to 1.0

**distance_metric** (str, default="l2")
- Metric for measuring distance from baseline weights
- Options:
  - `"l2"`: Euclidean distance (fastest, most common)
  - `"kl_divergence"`: KL divergence (information-theoretic)
  - `"huber"`: Huber loss (robust to outliers)

**method** (str, default="SLSQP")
- Scipy optimization method
- Options:
  - `"SLSQP"`: Sequential Least Squares Programming (handles equality and inequality constraints)
  - `"trust-constr"`: Trust region constrained (more robust for difficult problems)
  - `"COBYLA"`: Constrained Optimization BY Linear Approximation (derivative-free)

## IPW Optimization

IPW optimization minimizes distance from propensity-based weights while enforcing covariate balance and variance constraints.

### Basic Usage

```python
from causal_inference.core import OptimizationConfig
from causal_inference.estimators import IPWEstimator

# Create optimization config
optimization_config = OptimizationConfig(
    optimize_weights=True,
    variance_constraint=2.0,
    balance_constraints=True,
    distance_metric="l2"
)

# Create IPW estimator with optimization
estimator = IPWEstimator(
    propensity_model_type="logistic",
    optimization_config=optimization_config,
    random_state=42
)

# Fit and estimate (optimization happens automatically during fit)
estimator.fit(treatment_data, outcome_data, covariate_data)
effect = estimator.estimate_ate()

print(f"ATE: {effect.ate:.4f}")
print(f"95% CI: [{effect.ate_ci_lower:.4f}, {effect.ate_ci_upper:.4f}]")
```

### How It Works

1. **Baseline Weights**: Compute analytical IPW weights: w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))
2. **Optimization**: Minimize ||w - w_baseline||² subject to:
   - Balance: (1/n) X^T w = μ (covariate means match target)
   - Variance: (1/n) ||w||² ≤ φ (weight variance bounded)
   - Non-negativity: w_i ≥ 0 (all weights positive)
3. **Fallback**: If optimization fails, fall back to baseline weights

### Diagnostics

Access optimization diagnostics after fitting:

```python
# Get weight diagnostics
weight_diag = estimator._weight_diagnostics
print(f"Weight variance: {weight_diag['weight_variance']:.4f}")
print(f"Effective sample size: {weight_diag['effective_sample_size']:.1f}")

# Get optimization diagnostics
opt_diag = estimator.get_optimization_diagnostics()
if opt_diag:
    print(f"Optimization converged: {opt_diag['success']}")
    print(f"Iterations: {opt_diag['n_iterations']}")
    print(f"Final objective: {opt_diag['final_objective']:.6f}")
    print(f"Max covariate imbalance (SMD): {opt_diag['constraint_violation']:.4f}")
    print(f"Weight variance: {opt_diag['weight_variance']:.4f}")
    print(f"Effective sample size: {opt_diag['effective_sample_size']:.1f}")
```

### Bootstrap Integration

Optimization works seamlessly with bootstrap confidence intervals:

```python
estimator = IPWEstimator(
    propensity_model_type="logistic",
    optimization_config=optimization_config,
    bootstrap_samples=1000,  # Enable bootstrap
    random_state=42
)

estimator.fit(treatment_data, outcome_data, covariate_data)
effect = estimator.estimate_ate()

# Bootstrap CIs automatically account for optimization
print(f"ATE: {effect.ate:.4f}")
print(f"Bootstrap 95% CI: [{effect.ate_ci_lower:.4f}, {effect.ate_ci_upper:.4f}]")
```

**Note**: Optimization is automatically disabled in bootstrap samples to avoid nested optimization and reduce computational cost.

## G-Computation Ensemble

G-Computation supports ensemble model weighting with variance penalties to combine multiple outcome models.

### Basic Usage

```python
from causal_inference.estimators import GComputationEstimator

# Create ensemble estimator
estimator = GComputationEstimator(
    use_ensemble=True,
    ensemble_models=["linear", "ridge", "random_forest"],
    ensemble_variance_penalty=0.1,
    random_state=42,
    verbose=True
)

estimator.fit(treatment_data, outcome_data, covariate_data)
effect = estimator.estimate_ate()

print(f"ATE: {effect.ate:.4f}")

# Check ensemble weights
if estimator.ensemble_weights is not None:
    for name, weight in zip(estimator.ensemble_models_fitted.keys(),
                           estimator.ensemble_weights):
        print(f"{name}: {weight:.4f}")
```

### How It Works

1. **Fit Models**: Fit multiple outcome models (linear, ridge, random forest, etc.)
2. **Optimize Weights**: Minimize MSE + variance_penalty * Var(weights) subject to:
   - Sum to 1: Σw_i = 1
   - Non-negativity: w_i ≥ 0
3. **Ensemble Prediction**: Weighted combination of individual model predictions

### Parameters

**use_ensemble** (bool, default=False)
- Enable ensemble mode
- When False, uses single outcome model (standard G-computation)

**ensemble_models** (list of str, default=["linear", "ridge", "random_forest"])
- List of model types to include in ensemble
- Available models:
  - For continuous outcomes: "linear", "ridge", "random_forest"
  - For binary outcomes: "logistic" (used for linear/ridge), "random_forest"

**ensemble_variance_penalty** (float, default=0.1)
- Penalty on ensemble weight variance
- Higher values encourage more diverse weights (less reliance on single model)
- Lower values allow concentration on best-performing model
- Range: 0.0 to 1.0

### Diagnostics

```python
# Get optimization diagnostics
opt_diag = estimator.get_optimization_diagnostics()
if opt_diag:
    print(f"Ensemble optimization success: {opt_diag['ensemble_success']}")
    print(f"Final objective: {opt_diag['ensemble_objective']:.6f}")
    print("Ensemble weights:")
    for name, weight in opt_diag['ensemble_weights'].items():
        print(f"  {name}: {weight:.4f}")
```

## AIPW Component Optimization

AIPW supports component balance optimization to optimize the weighting between G-computation and IPW components.

### Basic Usage

```python
from causal_inference.estimators import AIPWEstimator

# Create AIPW estimator with component optimization
estimator = AIPWEstimator(
    cross_fitting=True,
    n_folds=5,
    optimize_component_balance=True,
    component_variance_penalty=0.5,
    random_state=42,
    verbose=True
)

estimator.fit(treatment_data, outcome_data, covariate_data)
effect = estimator.estimate_ate()

print(f"ATE: {effect.ate:.4f}")
```

> **Important**: Unlike IPW optimization (which happens during `fit()`), AIPW component
> optimization happens during `estimate_ate()`. This means `get_optimization_diagnostics()`
> will return `None` until after you call `estimate_ate()`.
>
> **Correct order:**
> 1. `estimator.fit(...)` - Fits models, no optimization yet
> 2. `effect = estimator.estimate_ate()` - Optimization happens here
> 3. `diagnostics = estimator.get_optimization_diagnostics()` - Now available
>
> This design is intentional because component optimization requires both G-computation
> and IPW components, which are computed during the estimation step.

### How It Works

Standard AIPW uses equal weights (50/50) for G-computation and IPW components:
- ATE = E[μ₁(X) - μ₀(X)] + E[w(X) * (Y - μ(X))]

Component optimization finds optimal weights α (for G-comp) and (1-α) (for IPW):
- Minimize: Var(α * G-comp + (1-α) * IPW) + penalty * (α - 0.5)²
- Subject to: 0.3 ≤ α ≤ 0.7 (both components contribute meaningfully)

### Parameters

**optimize_component_balance** (bool, default=False)
- Enable component balance optimization
- When False, uses standard AIPW with equal weights

**component_variance_penalty** (float, default=0.5)
- Penalty for deviating from 50/50 balance
- Higher values keep weights closer to 0.5 (more conservative)
- Lower values allow more extreme weights (more adaptive)
- Range: 0.0 to 1.0

### Diagnostics

```python
# Get optimization diagnostics
opt_diag = estimator.get_optimization_diagnostics()
if opt_diag:
    print(f"Optimal G-computation weight: {opt_diag['optimal_g_computation_weight']:.4f}")
    print(f"Optimal IPW weight: {opt_diag['optimal_ipw_weight']:.4f}")
    print(f"Fixed variance (50/50): {opt_diag['fixed_variance']:.6f}")
    print(f"Optimized variance: {opt_diag['optimized_variance']:.6f}")
    print(f"Variance reduction: {opt_diag['fixed_variance'] - opt_diag['optimized_variance']:.6f}")
```

## Best Practices

### When to Use Optimization

**Good Use Cases:**
- High weight variance in standard IPW (ESS << n)
- Strong confounding requiring aggressive weighting
- Multiple reasonable outcome models (G-computation ensemble)
- Variance reduction is priority

**Not Recommended:**
- Very small samples (n < 100) - optimization may be unstable
- Perfect overlap (propensity scores all near 0.5) - little room for improvement
- Computational constraints - optimization adds overhead
- When analytical methods already work well

### Choosing Parameters

**Variance Constraint (IPW)**
- Start with 2.0 and adjust based on results
- Lower (1.0-1.5) for aggressive variance reduction
- Higher (2.5-3.0) to stay closer to analytical weights
- Monitor covariate balance and bias

**Distance Metric**
- Default to "l2" (simplest, fastest)
- Try "kl_divergence" for theoretical properties
- Use "huber" if you suspect outlier weights

**Ensemble Variance Penalty (G-computation)**
- Default to 0.1 for balanced ensemble
- Lower (0.0-0.05) to focus on best model
- Higher (0.2-0.5) to encourage diversity

**Component Variance Penalty (AIPW)**
- Default to 0.5 for moderate adaptation
- Lower (0.1-0.3) for more aggressive optimization
- Higher (0.7-1.0) to stay closer to 50/50

### Validation Workflow

1. **Fit without optimization** - establish baseline
2. **Enable optimization** - compare results
3. **Check diagnostics** - verify convergence and balance
4. **Vary parameters** - explore sensitivity
5. **Validate on holdout** - if available

### Interpreting Results

**Successful Optimization:**
- Convergence achieved (success=True)
- Lower weight variance than baseline
- Similar or tighter confidence intervals
- Covariate balance maintained (SMD < tolerance)
- ATE estimate remains reasonable

**Warning Signs:**
- Convergence failed repeatedly
- ATE estimate changes dramatically from baseline
- Poor covariate balance (SMD >> tolerance)
- Unrealistic weights (many near zero or very large)

## Performance Considerations

- Optimization adds computational cost (scipy.optimize overhead)
- Typical overhead: 2-5x vs analytical methods for IPW
- Use `optimize_weights=False` by default (opt-in only when beneficial)
- Optimization is disabled in bootstrap samples automatically
- For large datasets (n > 10,000), consider subsampling for optimization parameter tuning

## Backward Compatibility

All optimization features are **completely optional**:
- Existing code works without modification
- Default behavior unchanged (`optimize_weights=False`)
- No breaking changes to existing APIs
- Can gradually adopt optimization where beneficial

## Examples

See the [example notebook](../examples/pyrake_optimization_examples.ipynb) for complete working examples including:
- IPW weight variance reduction
- G-computation ensemble weighting
- AIPW component balance optimization
- Bias-variance tradeoff exploration

## References

- **PyRake Repository**: https://github.com/rwilson4/PyRake
- **Research Notes**: `thoughts/shared/research/2025-10-29-pyrake-optimization-extensibility.md`
- **Implementation Plan**: `thoughts/shared/plans/2025-10-29-pyrake-optimization-implementation.md`

## Troubleshooting

**Optimization not converging:**
- Increase `max_iterations`
- Relax `balance_tolerance`
- Try different `method` (SLSQP → trust-constr)
- Reduce `variance_constraint` if set too tightly

**High variance despite optimization:**
- Decrease `variance_constraint` (more aggressive)
- Check overlap diagnostics (may need truncation)
- Consider using stabilized weights
- Verify covariate balance is maintained

**Unexpected ATE estimates:**
- Compare with analytical baseline
- Check optimization diagnostics for convergence
- Verify covariate balance (constraint_violation)
- Try different distance metrics
- Disable optimization and validate analytical estimate first

**Performance issues:**
- Use L2 distance metric (fastest)
- Reduce `max_iterations` if acceptable
- Consider approximate methods for very large datasets
- Profile to identify bottlenecks
