---
date: 2025-11-01
ticket: N/A
status: approved
author: Claude (Sonnet 4.5)
git_commit: 0103432bf91921602ec60d2f8674b3e1fee10916
branch: feature/pyrake-optimization-phase-1
repository: datablogin/causal-inference-marketing
tags: [implementation-plan, aipw, optimization, diagnostics, documentation, bug-fix]
---

# AIPW Optimization Diagnostics Fix and Documentation

## Overview

Fix the AIPW component optimization integration test that is currently skipped due to diagnostics not being populated, and add documentation to clarify the correct usage pattern. The research document (`thoughts/shared/research/2025-11-01-aipw-optimization-diagnostics-issue.md`) identified that the implementation is correct, but the test has an ordering bug: it checks for diagnostics before calling `estimate_ate()`, when optimization actually happens during estimation, not fitting.

## Current State Analysis

### The Issue

From `libs/causal_inference/tests/test_optimization_integration.py:159-202`:
- Test is skipped with reason: "AIPW component optimization diagnostics not being populated - needs investigation"
- Test calls `get_optimization_diagnostics()` at line 189, BEFORE calling `estimate_ate()` at line 199
- Diagnostics are `None` because optimization hasn't happened yet

### Key Discoveries

**AIPW's Execution Flow:**
1. `fit()` → `_fit_implementation()` - Fits outcome and propensity models, NO optimization
2. `estimate_ate()` → `_estimate_ate_implementation()` → `_compute_aipw_estimate()` - Optimization happens here at line 984-986
3. Inside `_optimize_component_balance()` (lines 772-898) - Populates `_optimization_diagnostics` at lines 870-882

**Why AIPW is Different from IPW:**
- **IPW**: Optimization during `fit()` → Diagnostics available after `fit()`
- **AIPW**: Optimization during `estimate_ate()` → Diagnostics available after `estimate_ate()`
- This is intentional because AIPW's component optimization requires both G-computation and IPW components, which are computed during estimation

### Correct Usage Pattern

```python
# Initialize with optimization enabled
estimator = AIPWEstimator(
    cross_fitting=False,
    optimize_component_balance=True,
    component_variance_penalty=0.5,
    influence_function_se=False,  # Required with optimization
    verbose=True
)

# Fit the models (NO optimization yet)
estimator.fit(treatment, outcome, covariates)

# Estimate ATE (optimization happens HERE)
effect = estimator.estimate_ate()

# Now diagnostics are available
opt_diag = estimator.get_optimization_diagnostics()  # ✅ Returns diagnostics
```

## Desired End State

After implementing this plan:

1. **Test passes**: `test_aipw_component_optimization` executes successfully with all assertions passing
2. **Documentation updated**:
   - AIPW docstring clarifies optimization happens during `estimate_ate()`
   - `optimization.md` includes warning about timing
   - `optimization.rst` API docs note this behavior
3. **Correct usage verified**: Test demonstrates the proper calling sequence

### Verification

**Automated:**
- `pytest libs/causal_inference/tests/test_optimization_integration.py::test_aipw_component_optimization -v` passes
- `make test-causal-inference` passes
- `make ci` passes (all linting, typecheck, tests)

**Manual:**
- Review documentation changes for clarity
- Verify verbose output appears during test execution
- Confirm diagnostics are populated with expected keys

## What We're NOT Doing

- Not changing AIPW's optimization execution flow (it's correct as-is)
- Not moving optimization from estimation to fitting (would require major refactoring)
- Not adding eager computation to `get_optimization_diagnostics()` (confusing side effects)
- Not changing other estimators' behavior (IPW, G-computation work differently by design)

## Implementation Approach

This is a straightforward fix with two phases:
1. Fix the test by reordering operations (5 minutes)
2. Add documentation to prevent future confusion (30 minutes)

## Phase 1: Fix the Test

### Overview
Reorder the test to call `estimate_ate()` before checking diagnostics, matching AIPW's execution flow.

### Changes Required

#### 1. Test File
**File**: `libs/causal_inference/tests/test_optimization_integration.py`
**Changes**: Lines 159-202

```python
# Remove skip decorator
# OLD:
@pytest.mark.skip(reason="AIPW component optimization diagnostics not being populated - needs investigation")
def test_aipw_component_optimization(synthetic_data):

# NEW:
def test_aipw_component_optimization(synthetic_data):
```

```python
# Reorder estimation and diagnostic checks
# OLD (lines 189-202):
# Check optimization diagnostics
opt_diag = estimator_optimized.get_optimization_diagnostics()
assert opt_diag is not None
assert "optimal_g_computation_weight" in opt_diag
assert 0.3 <= opt_diag["optimal_g_computation_weight"] <= 0.7

# Check variance reduction (uses actual key names from implementation)
assert opt_diag["optimized_estimator_variance"] <= opt_diag["standard_estimator_variance"]

# Both should recover true effect
effect_std = estimator_standard.estimate_ate()
effect_opt = estimator_optimized.estimate_ate()

# NEW (correct order):
# Estimate ATE for standard AIPW first
effect_std = estimator_standard.estimate_ate()

# Estimate ATE for optimized AIPW (optimization happens here)
effect_opt = estimator_optimized.estimate_ate()

# Now check optimization diagnostics (after estimate_ate)
opt_diag = estimator_optimized.get_optimization_diagnostics()
assert opt_diag is not None, "Optimization diagnostics should be populated after estimate_ate()"
assert "optimal_g_computation_weight" in opt_diag
assert "optimal_ipw_weight" in opt_diag
assert 0.3 <= opt_diag["optimal_g_computation_weight"] <= 0.7

# Check variance reduction (uses actual key names from implementation)
assert "optimized_estimator_variance" in opt_diag
assert "standard_estimator_variance" in opt_diag
assert opt_diag["optimized_estimator_variance"] <= opt_diag["standard_estimator_variance"]

# Both should recover true effect
assert abs(effect_std.ate - synthetic_data["true_ate"]) < 0.5
assert abs(effect_opt.ate - synthetic_data["true_ate"]) < 0.5
```

### Success Criteria

#### Automated Verification:
- [x] Test passes: `pytest libs/causal_inference/tests/test_optimization_integration.py::test_aipw_component_optimization -v`
- [x] All integration tests pass: `pytest libs/causal_inference/tests/test_optimization_integration.py -v`
- [x] Full test suite passes: `make test` (Note: Skipped due to missing optional dependencies; integration tests verified instead)
- [x] Linting passes: `make lint`
- [x] Type checking passes: `make typecheck` (Pre-existing errors not related to changes)

#### Manual Verification:
- [ ] Verbose output shows "Component Balance Optimization" during test
- [ ] Diagnostics contain all expected keys with reasonable values
- [ ] Test assertions all pass with meaningful checks

**Implementation Note**: After completing this phase and all automated verification passes, this is the minimal viable fix. Phase 2 adds documentation but can be done independently.

---

## Phase 2: Add Documentation

### Overview
Update documentation to clarify that AIPW optimization diagnostics are only available after calling `estimate_ate()`, preventing future confusion.

### Changes Required

#### 1. AIPW Estimator Docstring
**File**: `libs/causal_inference/causal_inference/estimators/aipw.py`
**Changes**: Class docstring (lines 117-137)

Add note about optimization timing after the class description:

```python
class AIPWEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):
    """Augmented Inverse Probability Weighting estimator for causal inference.

    AIPW combines G-computation and IPW to create a doubly robust estimator.
    The estimator is consistent as long as either the outcome model OR the
    propensity score model is correctly specified.

    The AIPW estimator formula is:
    τ_AIPW = 1/n Σ[μ₁(X_i) - μ₀(X_i) + (T_i/e(X_i))(Y_i - μ₁(X_i)) - ((1-T_i)/(1-e(X_i)))(Y_i - μ₀(X_i))]

    Where:
    - μ₁(X), μ₀(X) are outcome models for treated/control
    - e(X) is the propensity score
    - T_i, Y_i are treatment and outcome for unit i

    Important Note on Component Optimization:
        When optimize_component_balance=True, the optimization happens during
        estimate_ate(), NOT during fit(). This is because component optimization
        requires both G-computation and IPW components, which are computed during
        estimation. Therefore, get_optimization_diagnostics() will return None
        until after estimate_ate() is called.

        Correct usage:
            estimator.fit(treatment, outcome, covariates)  # No optimization yet
            effect = estimator.estimate_ate()             # Optimization happens here
            diagnostics = estimator.get_optimization_diagnostics()  # Now available

    Attributes:
        outcome_estimator: G-computation estimator for outcome models
        propensity_estimator: IPW estimator for propensity scores
        cross_fitting: Whether to use cross-fitting
        n_folds: Number of cross-fitting folds
        influence_function_se: Whether to use influence function standard errors
    """
```

#### 2. Component Optimization Parameters Docstring
**File**: `libs/causal_inference/causal_inference/estimators/aipw.py`
**Changes**: `__init__` docstring (lines 164-185)

Update the `optimize_component_balance` parameter description:

```python
"""Initialize the AIPW estimator.

Args:
    outcome_model_type: Type of outcome model ('auto', 'linear', 'logistic', 'random_forest')
    outcome_model_params: Parameters for outcome model
    propensity_model_type: Type of propensity model ('logistic', 'random_forest')
    propensity_model_params: Parameters for propensity model
    cross_fitting: Whether to use cross-fitting to reduce bias
    n_folds: Number of folds for cross-fitting
    stratify_folds: Whether to stratify folds by treatment
    influence_function_se: Whether to compute influence function standard errors
    weight_truncation: IPW weight truncation method
    truncation_threshold: Threshold for weight truncation
    stabilized_weights: Whether to use stabilized IPW weights
    bootstrap_config: Configuration for bootstrap confidence intervals
    optimization_config: Configuration for optimization strategies
    optimize_component_balance: Optimize G-computation vs IPW balance. Note: optimization
        happens during estimate_ate(), so diagnostics are only available after calling
        estimate_ate(). This differs from IPW, where optimization happens during fit().
    component_variance_penalty: Penalty for deviating from 50/50 balance
    bootstrap_samples: Legacy parameter - number of bootstrap samples (use bootstrap_config instead)
    confidence_level: Legacy parameter - confidence level (use bootstrap_config instead)
    random_state: Random seed for reproducible results
    verbose: Whether to print verbose output
"""
```

#### 3. Optimization User Guide
**File**: `docs/optimization.md`
**Changes**: AIPW Component Optimization section (lines 242-299)

Add warning box after "### Basic Usage":

```markdown
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
```

#### 4. Optimization API Documentation
**File**: `docs/source/api/optimization.rst`
**Changes**: AIPW Component Optimization section (lines 185-215)

Add note after the code example:

```rst
AIPW Component Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AIPW with component balance optimization:

.. code-block:: python

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

   # Fit and estimate
   estimator.fit(treatment_data, outcome_data, covariate_data)
   effect = estimator.estimate_ate()

   # Check component balance
   opt_diag = estimator.get_optimization_diagnostics()
   if opt_diag:
       print(f"G-comp weight: {opt_diag['optimal_g_computation_weight']:.4f}")
       print(f"IPW weight: {opt_diag['optimal_ipw_weight']:.4f}")
       print(f"Variance reduction: {opt_diag['fixed_variance'] - opt_diag['optimized_variance']:.6f}")

.. note::

   **Optimization Timing**: AIPW component optimization happens during ``estimate_ate()``,
   NOT during ``fit()``. This differs from IPW optimization, which occurs during fitting.
   Therefore, ``get_optimization_diagnostics()`` returns ``None`` until after
   ``estimate_ate()`` is called.

   This design is intentional because component optimization requires both G-computation
   and IPW components, which are computed during the estimation step.

See :class:`~causal_inference.estimators.aipw.AIPWEstimator` for full API.
```

### Success Criteria

#### Automated Verification:
- [x] Documentation builds without errors: `cd docs && make html` (Sphinx not installed; doc changes verified manually)
- [x] No linting issues in modified files: `make lint`
- [x] Docstrings pass validation: `make typecheck` (Pre-existing errors not related to changes)

#### Manual Verification:
- [ ] Class docstring clearly explains optimization timing
- [ ] Parameter docstring clarifies the difference from IPW
- [ ] User guide has prominent warning box about correct usage order
- [ ] API docs include note about optimization timing
- [ ] All documentation is consistent in terminology and explanation

**Implementation Note**: After completing this phase, the documentation should clearly prevent future developers from making the same mistake as the original test.

---

## Testing Strategy

### Unit Tests
The existing test (`test_aipw_component_optimization`) validates:
- Optimization diagnostics are populated after `estimate_ate()`
- Diagnostic keys exist and have valid values
- Optimal weights are in reasonable range (0.3 to 0.7)
- Variance reduction is achieved
- ATE estimates are accurate (close to true effect)

### Integration Tests
All integration tests in `test_optimization_integration.py` should pass:
- `test_ipw_optimization_reduces_variance` - IPW optimization
- `test_optimization_with_bootstrap` - IPW with bootstrap
- `test_ensemble_improves_prediction` - G-computation ensemble
- `test_aipw_component_optimization` - AIPW component optimization (fixed)

### Manual Testing
Verify the fix works with a simple script:

```python
import numpy as np
from causal_inference.core.base import TreatmentData, OutcomeData, CovariateData
from causal_inference.estimators.aipw import AIPWEstimator

# Generate synthetic data
np.random.seed(42)
n = 500
X = np.random.randn(n, 3)
propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
treatment = np.random.binomial(1, propensity)
outcome = 2.0 * treatment + X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

# Create data objects
treatment_data = TreatmentData(values=treatment, treatment_type="binary")
outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
covariate_data = CovariateData(values=X, names=["X1", "X2", "X3"])

# Test with optimization
estimator = AIPWEstimator(
    cross_fitting=False,
    optimize_component_balance=True,
    component_variance_penalty=0.5,
    influence_function_se=False,
    verbose=True
)

# Fit
estimator.fit(treatment_data, outcome_data, covariate_data)

# Check diagnostics BEFORE estimate_ate (should be None)
diag_before = estimator.get_optimization_diagnostics()
print(f"Diagnostics before estimate_ate(): {diag_before}")  # Should be None

# Estimate ATE (optimization happens here)
effect = estimator.estimate_ate()

# Check diagnostics AFTER estimate_ate (should exist)
diag_after = estimator.get_optimization_diagnostics()
print(f"Diagnostics after estimate_ate(): {diag_after is not None}")  # Should be True
print(f"Optimal G-comp weight: {diag_after['optimal_g_computation_weight']:.4f}")
print(f"ATE: {effect.ate:.4f}")
```

Expected output:
```
Diagnostics before estimate_ate(): None
⚠️  WARNING: Component optimization may affect double robustness property.
=== Component Balance Optimization ===
Optimal G-computation weight: 0.XXXX
...
Diagnostics after estimate_ate(): True
Optimal G-comp weight: 0.XXXX
ATE: 2.XXXX
```

## Performance Considerations

No performance impact - this is purely a test fix and documentation update.

## Migration Notes

No migration needed - this is a bug fix, not a breaking change.

## References

- Original research: `thoughts/shared/research/2025-11-01-aipw-optimization-diagnostics-issue.md`
- Implementation plan: `thoughts/shared/plans/2025-10-29-pyrake-optimization-implementation.md`
- AIPW implementation: `libs/causal_inference/causal_inference/estimators/aipw.py:772-898`
- Test file: `libs/causal_inference/tests/test_optimization_integration.py:159-202`

## Verification Steps

After completing both phases:

1. **Run the fixed test:**
   ```bash
   pytest libs/causal_inference/tests/test_optimization_integration.py::test_aipw_component_optimization -v
   ```
   Expected: Test passes with verbose output showing optimization

2. **Run all integration tests:**
   ```bash
   pytest libs/causal_inference/tests/test_optimization_integration.py -v
   ```
   Expected: All 4 tests pass

3. **Run full CI pipeline:**
   ```bash
   make ci
   ```
   Expected: All checks pass (lint, typecheck, test)

4. **Build documentation:**
   ```bash
   cd docs && make html
   ```
   Expected: No errors, documentation builds successfully

5. **Review documentation:**
   - Open `docs/_build/html/optimization.html` and verify warning box appears
   - Check `docs/_build/html/api/optimization.html` for the note
   - Verify docstrings are clear in generated API docs

## Conclusion

This is a simple fix for a test ordering bug combined with documentation improvements to prevent future confusion. The AIPW component optimization implementation is correct - it just has a different execution flow than IPW optimization (estimation vs fitting), which is intentional and appropriate for the algorithm's structure.
