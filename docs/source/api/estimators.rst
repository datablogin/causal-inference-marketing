Estimators
==========

Causal inference estimators for various identification strategies.

Base Estimator
--------------

All estimators inherit from the abstract base class:

.. autoclass:: causal_inference.core.base.BaseEstimator
   :members:
   :show-inheritance:

**Key Methods:**

- ``estimate_ate()`` - Estimate Average Treatment Effect
- ``estimate_cate()`` - Estimate Conditional Average Treatment Effect
- ``fit()`` - Fit the estimator to data
- ``diagnose()`` - Run diagnostic tests

G-Computation (Standardization)
-------------------------------

.. autoclass:: causal_inference.estimators.g_computation.GComputation
   :members:
   :show-inheritance:

**When to Use:**
- When you can model the outcome well
- For binary, categorical, or continuous treatments
- When you have good domain knowledge of outcome predictors

**Assumptions:**
- No unmeasured confounding
- Correct outcome model specification
- Positivity (overlap)

**Example:**

.. code-block:: python

   from causal_inference.estimators import GComputation
   from sklearn.ensemble import RandomForestRegressor

   # Default (uses logistic/linear regression)
   estimator = GComputation()
   effect = estimator.estimate_ate(treatment, outcome, covariates)

   # With custom ML model
   estimator = GComputation(model=RandomForestRegressor(n_estimators=100))
   effect = estimator.estimate_ate(treatment, outcome, covariates)

Inverse Probability Weighting (IPW)
-----------------------------------

.. autoclass:: causal_inference.estimators.ipw.IPW
   :members:
   :show-inheritance:

**When to Use:**
- When you can model treatment assignment (propensity) well
- For creating "pseudo-randomized" samples
- When outcome modeling is challenging

**Assumptions:**
- No unmeasured confounding
- Correct propensity score model specification
- Strong positivity (overlap)

**Example:**

.. code-block:: python

   from causal_inference.estimators import IPW
   from sklearn.ensemble import GradientBoostingClassifier

   # Default (uses logistic regression)
   estimator = IPW()
   effect = estimator.estimate_ate(treatment, outcome, covariates)

   # With custom propensity model
   estimator = IPW(
       propensity_model=GradientBoostingClassifier(),
       stabilized=True  # Use stabilized weights
   )
   effect = estimator.estimate_ate(treatment, outcome, covariates)

Augmented IPW (AIPW) - Doubly Robust
------------------------------------

.. autoclass:: causal_inference.estimators.aipw.AIPW
   :members:
   :show-inheritance:

**When to Use:**
- When you want protection against model misspecification
- Gold standard for observational causal inference
- When both outcome and treatment models are reasonable

**Advantages:**
- Doubly robust: consistent if either model is correct
- Generally most reliable estimator
- Good finite-sample properties

**Example:**

.. code-block:: python

   from causal_inference.estimators import AIPW
   from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

   # Default models
   estimator = AIPW()
   effect = estimator.estimate_ate(treatment, outcome, covariates)

   # Custom models for both outcome and propensity
   estimator = AIPW(
       outcome_model=RandomForestRegressor(),
       propensity_model=RandomForestClassifier(),
       stabilized=True
   )
   effect = estimator.estimate_ate(treatment, outcome, covariates)

Double Machine Learning (DML)
-----------------------------

.. autoclass:: causal_inference.estimators.doubly_robust_ml.DoublyRobustML
   :members:
   :show-inheritance:

**When to Use:**
- High-dimensional data (many covariates)
- When you want to use modern ML methods
- For robust inference with complex models

**Key Features:**
- Cross-fitting to avoid overfitting bias
- Works with any scikit-learn model
- Provides valid confidence intervals with ML models

**Example:**

.. code-block:: python

   from causal_inference.estimators import DoublyRobustML
   from lightgbm import LGBMRegressor, LGBMClassifier

   estimator = DoublyRobustML(
       outcome_model=LGBMRegressor(),
       propensity_model=LGBMClassifier(),
       n_folds=5,  # Cross-fitting folds
       random_state=42
   )
   effect = estimator.estimate_ate(treatment, outcome, covariates)

Meta-Learners
-------------

T-Learner
^^^^^^^^^

.. autoclass:: causal_inference.estimators.meta_learners.TLearner
   :members:
   :show-inheritance:

Separate models for treated and control groups:

.. code-block:: python

   from causal_inference.estimators.meta_learners import TLearner

   estimator = TLearner(base_model=RandomForestRegressor())
   cate = estimator.estimate_cate(treatment, outcome, covariates)

S-Learner
^^^^^^^^^

.. autoclass:: causal_inference.estimators.meta_learners.SLearner
   :members:
   :show-inheritance:

Single model including treatment as a feature:

.. code-block:: python

   from causal_inference.estimators.meta_learners import SLearner

   estimator = SLearner(base_model=RandomForestRegressor())
   cate = estimator.estimate_cate(treatment, outcome, covariates)

X-Learner
^^^^^^^^^

.. autoclass:: causal_inference.estimators.meta_learners.XLearner
   :members:
   :show-inheritance:

Cross-learner with propensity weighting:

.. code-block:: python

   from causal_inference.estimators.meta_learners import XLearner

   estimator = XLearner(
       outcome_model=RandomForestRegressor(),
       propensity_model=RandomForestClassifier()
   )
   cate = estimator.estimate_cate(treatment, outcome, covariates)

Causal Forest
-------------

.. autoclass:: causal_inference.estimators.causal_forest.CausalForest
   :members:
   :show-inheritance:

Tree-based method for heterogeneous treatment effects:

.. code-block:: python

   from causal_inference.estimators import CausalForest

   estimator = CausalForest(
       n_estimators=1000,
       min_samples_leaf=5,
       honest=True  # Honest splitting
   )

   cate = estimator.estimate_cate(treatment, outcome, covariates)
   # Get treatment effect for each individual
   individual_effects = cate.individual_effects

Quasi-Experimental Methods
--------------------------

Regression Discontinuity (RDD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: causal_inference.estimators.regression_discontinuity.RegressionDiscontinuity
   :members:
   :show-inheritance:

For sharp or fuzzy discontinuity designs:

.. code-block:: python

   from causal_inference.estimators import RegressionDiscontinuity

   # Sharp RDD
   estimator = RegressionDiscontinuity(
       cutoff=0.0,
       design_type="sharp",
       bandwidth="optimal"
   )
   effect = estimator.estimate_ate(treatment, outcome, running_variable)

   # Fuzzy RDD
   estimator = RegressionDiscontinuity(
       cutoff=0.0,
       design_type="fuzzy",
       bandwidth=2.0
   )
   effect = estimator.estimate_ate(treatment, outcome, running_variable)

Synthetic Control
^^^^^^^^^^^^^^^^

.. autoclass:: causal_inference.estimators.synthetic_control.SyntheticControl
   :members:
   :show-inheritance:

For panel data with one treated unit:

.. code-block:: python

   from causal_inference.estimators import SyntheticControl

   estimator = SyntheticControl(
       treatment_period=20,  # When treatment started
       method="synthetic_control"
   )

   # Requires panel data format
   effect = estimator.estimate_ate(
       panel_data=panel_df,
       unit_col="unit_id",
       time_col="time",
       outcome_col="outcome",
       treated_unit="treated_unit_id"
   )

Difference-in-Differences (DiD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: causal_inference.estimators.difference_in_differences.DifferenceInDifferences
   :members:
   :show-inheritance:

For parallel trends designs:

.. code-block:: python

   from causal_inference.estimators import DifferenceInDifferences

   estimator = DifferenceInDifferences(
       treatment_period=pd.Timestamp("2023-01-01"),
       method="two_way_fixed_effects"
   )

   effect = estimator.estimate_ate(
       panel_data=panel_df,
       unit_col="unit_id",
       time_col="date",
       outcome_col="outcome",
       treatment_col="treated"
   )

Instrumental Variables (IV)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: causal_inference.estimators.iv.InstrumentalVariables
   :members:
   :show-inheritance:

For handling endogeneity with valid instruments:

.. code-block:: python

   from causal_inference.estimators import InstrumentalVariables

   estimator = InstrumentalVariables(method="2sls")

   effect = estimator.estimate_ate(
       treatment=treatment,
       outcome=outcome,
       instruments=instruments,
       covariates=covariates
   )

Survival Analysis
-----------------

Survival G-Computation
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: causal_inference.estimators.survival_g_computation.SurvivalGComputation
   :members:
   :show-inheritance:

Survival IPW
^^^^^^^^^^^

.. autoclass:: causal_inference.estimators.survival_ipw.SurvivalIPW
   :members:
   :show-inheritance:

Survival AIPW
^^^^^^^^^^^^

.. autoclass:: causal_inference.estimators.survival_aipw.SurvivalAIPW
   :members:
   :show-inheritance:

Example with survival data:

.. code-block:: python

   from causal_inference.estimators import SurvivalGComputation
   from causal_inference.core import SurvivalOutcomeData

   # Survival outcome (time-to-event)
   survival_outcome = SurvivalOutcomeData(
       times=data["time_to_event"],
       events=data["event_occurred"],
       name="customer_lifetime"
   )

   estimator = SurvivalGComputation()
   effect = estimator.estimate_ate(treatment, survival_outcome, covariates)

Estimator Selection Guide
-------------------------

**Choose by Problem Type:**

================================  =============================  ===============================
Problem Type                      Recommended Estimator          Alternative
================================  =============================  ===============================
Simple randomized experiment     GComputation                   AIPW
Observational study              AIPW                           DoublyRobustML
High-dimensional data            DoublyRobustML                 CausalForest
Heterogeneous effects            CausalForest, XLearner         TLearner
Panel data (one treated unit)    SyntheticControl               DiD
Panel data (multiple units)      DifferenceInDifferences        -
Sharp discontinuity             RegressionDiscontinuity        -
Endogeneity with instruments    InstrumentalVariables          -
Time-to-event outcomes          SurvivalAIPW                   SurvivalGComputation
================================  =============================  ===============================

**Choose by Data Characteristics:**

- **Small sample (n < 1000)**: GComputation, IPW, AIPW
- **Large sample (n > 10000)**: DoublyRobustML, CausalForest
- **Many covariates (p > 100)**: DoublyRobustML, CausalForest
- **Binary treatment**: All estimators
- **Continuous treatment**: GComputation, DoublyRobustML, CausalForest
- **Multiple treatments**: GComputation, AIPW

Common Usage Patterns
---------------------

**Basic Pattern:**

.. code-block:: python

   # 1. Choose estimator
   estimator = AIPW()

   # 2. Estimate treatment effect
   effect = estimator.estimate_ate(treatment, outcome, covariates)

   # 3. Check results
   print(f"ATE: {effect.ate:.4f} (p={effect.p_value:.4f})")

   # 4. Run diagnostics
   diagnostics = estimator.diagnose(treatment, outcome, covariates)

**With Custom Models:**

.. code-block:: python

   from lightgbm import LGBMRegressor, LGBMClassifier

   estimator = DoublyRobustML(
       outcome_model=LGBMRegressor(
           n_estimators=100,
           learning_rate=0.1,
           random_state=42
       ),
       propensity_model=LGBMClassifier(
           n_estimators=100,
           learning_rate=0.1,
           random_state=42
       ),
       n_folds=5
   )

   effect = estimator.estimate_ate(treatment, outcome, covariates)

**Heterogeneous Effects:**

.. code-block:: python

   # Estimate individual-level treatment effects
   estimator = CausalForest()
   cate_results = estimator.estimate_cate(treatment, outcome, covariates)

   # Get effects for each person
   individual_effects = cate_results.individual_effects

   # Find most/least responsive individuals
   top_responders = individual_effects.nlargest(10)

   # Subgroup analysis
   estimator = AIPW()
   subgroup_effects = estimator.estimate_ate(
       treatment, outcome, covariates,
       subgroups=data["customer_segment"]
   )

**Sensitivity Analysis:**

.. code-block:: python

   from causal_inference.diagnostics import sensitivity_analysis

   # Estimate effect
   estimator = AIPW()
   effect = estimator.estimate_ate(treatment, outcome, covariates)

   # Test sensitivity to unmeasured confounding
   sensitivity_results = sensitivity_analysis(
       estimator=estimator,
       effect=effect,
       confounding_strength_range=(0, 0.3)
   )
