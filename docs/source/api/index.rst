API Reference
=============

Complete reference documentation for all public APIs.

Core Data Models
================

.. toctree::
   :maxdepth: 2

   data_models

These Pydantic models define the structure and validation for causal inference data:

- :class:`~causal_inference.core.base.TreatmentData` - Treatment/intervention assignments
- :class:`~causal_inference.core.base.OutcomeData` - Outcome variables to analyze
- :class:`~causal_inference.core.base.CovariateData` - Confounding variables
- :class:`~causal_inference.core.base.CausalEffect` - Results container

Estimators
==========

.. toctree::
   :maxdepth: 2

   estimators

All causal inference estimators inherit from :class:`~causal_inference.core.base.BaseEstimator`:

**Identification Methods:**
- :class:`~causal_inference.estimators.g_computation.GComputation` - Standardization/G-formula
- :class:`~causal_inference.estimators.ipw.IPW` - Inverse Probability Weighting
- :class:`~causal_inference.estimators.aipw.AIPW` - Augmented IPW (Doubly Robust)

**Machine Learning Methods:**
- :class:`~causal_inference.estimators.doubly_robust_ml.DoublyRobustML` - Double Machine Learning
- :class:`~causal_inference.estimators.meta_learners.TLearner` - T-Learner
- :class:`~causal_inference.estimators.meta_learners.SLearner` - S-Learner
- :class:`~causal_inference.estimators.meta_learners.XLearner` - X-Learner
- :class:`~causal_inference.estimators.causal_forest.CausalForest` - Causal Random Forest

**Quasi-Experimental Methods:**
- :class:`~causal_inference.estimators.regression_discontinuity.RegressionDiscontinuity` - RDD
- :class:`~causal_inference.estimators.synthetic_control.SyntheticControl` - Synthetic Control
- :class:`~causal_inference.estimators.difference_in_differences.DifferenceInDifferences` - DiD
- :class:`~causal_inference.estimators.iv.InstrumentalVariables` - IV/2SLS

**Survival Analysis:**
- :class:`~causal_inference.estimators.survival_g_computation.SurvivalGComputation` - Survival G-formula
- :class:`~causal_inference.estimators.survival_ipw.SurvivalIPW` - Survival IPW
- :class:`~causal_inference.estimators.survival_aipw.SurvivalAIPW` - Survival AIPW

Optimization
============

.. toctree::
   :maxdepth: 2

   optimization

PyRake-style optimization framework for improving estimator efficiency:

**Weight Optimization:**
- IPW weight optimization to reduce variance while maintaining balance
- Configurable distance metrics (L2, KL divergence, Chi-squared)
- Automatic fallback to analytical weights if optimization fails

**Model Ensemble Optimization:**
- G-computation ensemble weighting across multiple outcome models
- Minimize cross-validated prediction error
- Support for diverse model types (linear, logistic, random forest)

**Component Balance Optimization:**
- AIPW optimization of G-computation vs IPW component weighting
- Data-adaptive balance between outcome modeling and propensity weighting
- Variance reduction while maintaining double robustness

Diagnostics
===========

.. toctree::
   :maxdepth: 2

   diagnostics

Tools for validating assumptions and evaluating causal models:

**Assumption Testing:**
- :mod:`~causal_inference.diagnostics.assumptions` - Test identifying assumptions
- :mod:`~causal_inference.diagnostics.overlap` - Check positivity/common support
- :mod:`~causal_inference.diagnostics.balance` - Assess covariate balance

**Model Evaluation:**
- :mod:`~causal_inference.diagnostics.validation_suite` - Comprehensive model validation
- :mod:`~causal_inference.diagnostics.sensitivity` - Sensitivity to unmeasured confounding
- :mod:`~causal_inference.diagnostics.falsification` - Placebo/negative control tests

**Visualization:**
- :mod:`~causal_inference.diagnostics.visualization` - Diagnostic plots and charts
- :mod:`~causal_inference.diagnostics.reporting` - Automated diagnostic reports

Utilities
=========

.. toctree::
   :maxdepth: 2

   utils

Helper functions and utilities:

**Data Processing:**
- :mod:`~causal_inference.data.synthetic` - Generate synthetic datasets
- :mod:`~causal_inference.data.validation` - Data validation utilities
- :mod:`~causal_inference.data.missing_data` - Handle missing data

**Statistical Tools:**
- :mod:`~causal_inference.core.bootstrap` - Bootstrap confidence intervals
- :mod:`~causal_inference.utils.validation` - Input validation helpers
- :mod:`~causal_inference.utils.benchmarking` - Performance benchmarking

**Discovery Methods:**
- :mod:`~causal_inference.discovery.constraint_based` - PC Algorithm
- :mod:`~causal_inference.discovery.score_based` - GES Algorithm
- :mod:`~causal_inference.discovery.benchmarks` - Discovery benchmarks

Machine Learning Integration
============================

Advanced ML capabilities:

- :mod:`~causal_inference.ml.super_learner` - Ensemble learning
- :mod:`~causal_inference.ml.cross_fitting` - Cross-fitting utilities
- :mod:`~causal_inference.evaluation.hte_metrics` - Heterogeneous treatment effects

Web API
=======

FastAPI service endpoints:

.. automodule:: services.causal_api.routes.attribution
   :members:

.. automodule:: services.causal_api.routes.health
   :members:

Configuration
=============

Settings and configuration:

.. automodule:: shared.config.causal_config
   :members:
   :show-inheritance:

Exception Classes
=================

Custom exceptions for error handling:

.. autoexception:: causal_inference.core.base.CausalInferenceError
.. autoexception:: causal_inference.core.base.AssumptionViolationError
.. autoexception:: causal_inference.core.base.DataValidationError
.. autoexception:: causal_inference.core.base.EstimationError

Quick Reference
===============

**Common Usage Patterns:**

.. code-block:: python

   from causal_inference.estimators import GComputation
   from causal_inference.core import TreatmentData, OutcomeData, CovariateData

   # Basic estimation
   estimator = GComputation()
   effect = estimator.estimate_ate(treatment, outcome, covariates)

   # With diagnostics
   diagnostics = estimator.diagnose(treatment, outcome, covariates)

   # Heterogeneous effects
   hte = estimator.estimate_cate(treatment, outcome, covariates)

**Key Classes:**

- :class:`~causal_inference.core.base.BaseEstimator` - Base class for all estimators
- :class:`~causal_inference.core.base.CausalEffect` - Standard result format
- :class:`~causal_inference.diagnostics.validation_suite.ValidationSuite` - Diagnostic tools

**Key Methods:**

- :meth:`~causal_inference.core.base.BaseEstimator.estimate_ate` - Average Treatment Effect
- :meth:`~causal_inference.core.base.BaseEstimator.estimate_cate` - Conditional ATE
- :meth:`~causal_inference.core.base.BaseEstimator.diagnose` - Run diagnostics
- :meth:`~causal_inference.core.base.BaseEstimator.fit` - Fit the estimator

Module Index
============

.. note::
   Full API reference for all modules will be available once the library is fully integrated.
   For now, see the individual sections above for available functionality.

.. Raw autosummary disabled temporarily due to build issues
.. .. autosummary::
..    :toctree: _autosummary
..    :template: module.rst
..    :recursive:
..
..    causal_inference.core
..    causal_inference.estimators
..    causal_inference.diagnostics
..    causal_inference.data
..    causal_inference.utils
..    causal_inference.ml
..    causal_inference.discovery
