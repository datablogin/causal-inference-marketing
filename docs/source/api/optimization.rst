Optimization API
================

PyRake-style constrained optimization framework for improving estimator efficiency.

Overview
--------

The optimization framework provides optional constrained optimization across multiple estimators (IPW, G-computation, AIPW). It simultaneously addresses bias reduction, variance minimization, and predictive power through carefully designed constraints.

For detailed usage and examples, see the :doc:`/optimization` guide.

Configuration
-------------

OptimizationConfig
~~~~~~~~~~~~~~~~~~

.. autoclass:: causal_inference.core.optimization_config.OptimizationConfig
   :members:
   :undoc-members:
   :show-inheritance:

   Configuration for PyRake-style constrained optimization.

   **Key Parameters:**

   * ``optimize_weights`` (bool): Enable/disable optimization (default: False)
   * ``variance_constraint`` (float or None): Maximum allowed weight variance
   * ``balance_constraints`` (bool): Enforce covariate balance (default: True)
   * ``balance_tolerance`` (float): Tolerance for covariate balance in SMD units
   * ``distance_metric`` (str): Distance metric ('l2', 'kl_divergence', 'huber')
   * ``method`` (str): Scipy optimization method ('SLSQP', 'trust-constr', 'COBYLA')
   * ``max_iterations`` (int): Maximum optimization iterations
   * ``convergence_tolerance`` (float): Convergence tolerance
   * ``verbose`` (bool): Print optimization progress
   * ``store_diagnostics`` (bool): Store detailed diagnostics

   **Example:**

   .. code-block:: python

      from causal_inference.core import OptimizationConfig

      config = OptimizationConfig(
          optimize_weights=True,
          variance_constraint=2.0,
          balance_constraints=True,
          balance_tolerance=0.01,
          distance_metric="l2",
          verbose=True
      )

Mixin
-----

OptimizationMixin
~~~~~~~~~~~~~~~~~

.. autoclass:: causal_inference.core.optimization_mixin.OptimizationMixin
   :members:
   :undoc-members:
   :show-inheritance:

   Mixin providing PyRake-style constrained optimization capabilities.

   This mixin follows the pattern established by :class:`~causal_inference.core.bootstrap.BootstrapMixin`:

   * Cooperative ``__init__`` using ``super()``
   * Configuration object for settings
   * Stores optimization results for diagnostics
   * Provides fallback to non-optimized methods on failure

   **Key Methods:**

   .. automethod:: optimize_weights_constrained

      Optimize weights using PyRake-style constrained optimization.

      Minimizes distance from baseline weights subject to:

      * **Covariate balance**: Weighted covariate means match target means
      * **Variance constraint**: Weight variance bounded by specified limit
      * **Non-negativity**: All weights must be non-negative

      **Parameters:**

      * ``baseline_weights`` (NDArray): Initial/baseline weights (e.g., 1/e(X) for IPW)
      * ``covariates`` (NDArray): Covariate matrix (n_obs x n_covariates)
      * ``target_means`` (NDArray or None): Target covariate means (default: observed means)
      * ``variance_constraint`` (float or None): Maximum weight variance (default: from config)

      **Returns:**

      * ``NDArray``: Optimized weights array

   .. automethod:: get_optimization_diagnostics

      Get optimization diagnostics.

      **Returns:**

      * ``dict`` or ``None``: Dictionary of optimization diagnostics including:

        * ``success`` (bool): Whether optimization converged
        * ``message`` (str): Optimization status message
        * ``n_iterations`` (int): Number of iterations performed
        * ``final_objective`` (float): Final objective function value
        * ``constraint_violation`` (float): Maximum covariate imbalance (SMD)
        * ``weight_variance`` (float): Variance of optimized weights
        * ``effective_sample_size`` (float): Effective sample size

Usage in Estimators
-------------------

IPW Optimization
~~~~~~~~~~~~~~~~

IPW estimator with weight optimization:

.. code-block:: python

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

   # Fit and estimate
   estimator.fit(treatment_data, outcome_data, covariate_data)
   effect = estimator.estimate_ate()

   # Access diagnostics
   opt_diag = estimator.get_optimization_diagnostics()
   if opt_diag:
       print(f"Converged: {opt_diag['success']}")
       print(f"Weight variance: {opt_diag['weight_variance']:.4f}")
       print(f"ESS: {opt_diag['effective_sample_size']:.1f}")

See :class:`~causal_inference.estimators.ipw.IPWEstimator` for full API.

G-Computation Ensemble
~~~~~~~~~~~~~~~~~~~~~~

G-computation with ensemble model weighting:

.. code-block:: python

   from causal_inference.estimators import GComputationEstimator

   # Create ensemble estimator
   estimator = GComputationEstimator(
       use_ensemble=True,
       ensemble_models=["linear", "ridge", "random_forest"],
       ensemble_variance_penalty=0.1,
       random_state=42,
       verbose=True
   )

   # Fit and estimate
   estimator.fit(treatment_data, outcome_data, covariate_data)
   effect = estimator.estimate_ate()

   # Check ensemble weights
   opt_diag = estimator.get_optimization_diagnostics()
   if opt_diag:
       print("Ensemble weights:")
       for name, weight in opt_diag['ensemble_weights'].items():
           print(f"  {name}: {weight:.4f}")

See :class:`~causal_inference.estimators.g_computation.GComputationEstimator` for full API.

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

Distance Metrics
----------------

The optimization framework supports multiple distance metrics for measuring deviation from baseline weights:

L2 Distance
~~~~~~~~~~~

Euclidean distance (default):

.. math::

   d(w, w_{baseline}) = \sum_i (w_i - w_{baseline,i})^2

* **Pros**: Simple, fast, convex
* **Cons**: Sensitive to outliers
* **Use when**: Default choice for most applications

KL Divergence
~~~~~~~~~~~~~

Kullback-Leibler divergence:

.. math::

   d(w, w_{baseline}) = \sum_i w_i \log\left(\frac{w_i}{w_{baseline,i}}\right)

* **Pros**: Information-theoretic interpretation, preserves relative weights
* **Cons**: Slower, more complex
* **Use when**: Theoretical properties are important

Huber Loss
~~~~~~~~~~

Robust loss function:

.. math::

   d(w, w_{baseline}) = \sum_i \begin{cases}
   \frac{1}{2}(w_i - w_{baseline,i})^2 & \text{if } |w_i - w_{baseline,i}| \leq \delta \\
   \delta(|w_i - w_{baseline,i}| - \frac{1}{2}\delta) & \text{otherwise}
   \end{cases}

* **Pros**: Robust to outliers, combines L1/L2 properties
* **Cons**: Requires tuning δ parameter
* **Use when**: Suspect outlier weights

Constraints
-----------

The optimization framework supports multiple constraint types:

Covariate Balance
~~~~~~~~~~~~~~~~~

Equality constraint ensuring weighted covariate means match target means:

.. math::

   \frac{1}{n} X^T w = \mu_{target}

This is the key constraint for maintaining unbiasedness.

Variance Constraint
~~~~~~~~~~~~~~~~~~~

Inequality constraint bounding weight variance:

.. math::

   \frac{1}{n} \|w\|^2 \leq \phi

Lower variance leads to more efficient estimates (tighter confidence intervals).

Non-negativity
~~~~~~~~~~~~~~

Box constraint ensuring all weights are non-negative:

.. math::

   w_i \geq 0 \quad \forall i

This is automatically enforced by the optimization framework.

Diagnostics
-----------

Optimization diagnostics provide detailed information about convergence and quality:

**Convergence Diagnostics:**

* ``success`` (bool): Whether optimization converged
* ``message`` (str): Optimization status message
* ``n_iterations`` (int): Number of iterations performed

**Quality Metrics:**

* ``final_objective`` (float): Final objective function value
* ``constraint_violation`` (float): Maximum standardized mean difference (SMD)
* ``weight_variance`` (float): Variance of optimized weights
* ``effective_sample_size`` (float): ESS = (Σw)² / Σw²

**Interpretation:**

* ``success=True``: Optimization converged successfully
* ``constraint_violation < balance_tolerance``: Covariate balance achieved
* Lower ``weight_variance``: More efficient estimates
* Higher ``effective_sample_size``: Better precision

Bootstrap Integration
---------------------

Optimization works seamlessly with bootstrap confidence intervals:

.. code-block:: python

   from causal_inference.core import OptimizationConfig, BootstrapConfig
   from causal_inference.estimators import IPWEstimator

   # Create configs
   optimization_config = OptimizationConfig(
       optimize_weights=True,
       variance_constraint=2.0
   )

   bootstrap_config = BootstrapConfig(
       n_samples=1000,
       confidence_level=0.95,
       random_state=42
   )

   # Create estimator with both
   estimator = IPWEstimator(
       propensity_model_type="logistic",
       optimization_config=optimization_config,
       bootstrap_config=bootstrap_config
   )

   # Bootstrap CIs automatically account for optimization
   estimator.fit(treatment_data, outcome_data, covariate_data)
   effect = estimator.estimate_ate()

   print(f"ATE: {effect.ate:.4f}")
   print(f"95% CI: [{effect.ate_ci_lower:.4f}, {effect.ate_ci_upper:.4f}]")

**Important**: Optimization is automatically disabled in bootstrap samples to avoid nested optimization and reduce computational cost.

Performance Considerations
--------------------------

**Computational Cost:**

* Optimization adds 2-5x overhead vs analytical methods for IPW
* Cost scales with sample size and number of covariates
* L2 distance metric is fastest
* SLSQP method generally performs best

**Optimization:**

* Use ``optimize_weights=False`` by default (opt-in only)
* Optimization disabled in bootstrap samples automatically
* Consider subsampling for parameter tuning on large datasets (n > 10,000)
* Profile to identify bottlenecks if performance is critical

**Convergence:**

* Most problems converge in < 100 iterations
* Increase ``max_iterations`` if needed
* Try different ``method`` if convergence issues
* Relax ``balance_tolerance`` if constraints too strict

Backward Compatibility
----------------------

All optimization features are **completely optional**:

* Existing code works without modification
* Default behavior unchanged (``optimize_weights=False``)
* No breaking changes to existing APIs
* Can gradually adopt optimization where beneficial

See Also
--------

* :doc:`/optimization` - Detailed user guide with examples
* :class:`~causal_inference.estimators.ipw.IPWEstimator` - IPW estimator
* :class:`~causal_inference.estimators.g_computation.GComputationEstimator` - G-computation estimator
* :class:`~causal_inference.estimators.aipw.AIPWEstimator` - AIPW estimator
* :class:`~causal_inference.core.bootstrap.BootstrapMixin` - Bootstrap integration

References
----------

* **PyRake Repository**: https://github.com/rwilson4/PyRake
* **Research Notes**: ``thoughts/shared/research/2025-10-29-pyrake-optimization-extensibility.md``
* **Implementation Plan**: ``thoughts/shared/plans/2025-10-29-pyrake-optimization-implementation.md``
