Troubleshooting Guide
====================

Common issues and their solutions when using causal inference methods.

.. toctree::
   :maxdepth: 2

   common_errors
   debugging_guide

Quick Diagnosis
===============

**Symptoms Checker:**

==================================  ============================  ================
Symptom                             Likely Issue                  Solution
==================================  ============================  ================
``ValidationError`` on data input  Data validation failure       Check data types and formats
Extremely large effect sizes       Model misspecification        Try different estimator
Effect size near zero              Weak/no effect or bias        Check for confounding
Very wide confidence intervals     High variance                 More data or better models
``PositivityError``                No overlap                    Trim data or change design
``ConvergenceWarning``             Model fitting issues          Adjust parameters or data
NaN or infinite values             Numerical instability         Check for extreme values
==================================  ============================  ================

Data Issues
===========

Validation Errors
-----------------

**Problem**: ``pydantic.ValidationError`` when creating data objects

**Common Causes:**

1. **Wrong treatment type**:

.. code-block:: python

   # ❌ This will fail - binary treatment with 3 values
   treatment = TreatmentData(
       values=[0, 1, 2],
       treatment_type="binary"
   )

   # ✅ Correct - specify categorical
   treatment = TreatmentData(
       values=[0, 1, 2],
       treatment_type="categorical",
       categories=[0, 1, 2]
   )

2. **Missing categories for categorical treatment**:

.. code-block:: python

   # ❌ This will fail
   treatment = TreatmentData(
       values=["A", "B", "C"],
       treatment_type="categorical"
   )

   # ✅ Correct - specify categories
   treatment = TreatmentData(
       values=["A", "B", "C"],
       treatment_type="categorical",
       categories=["A", "B", "C"]
   )

3. **Non-numeric data for continuous variables**:

.. code-block:: python

   # ❌ This will fail
   outcome = OutcomeData(
       values=["high", "low", "medium"],
       outcome_type="continuous"
   )

   # ✅ Correct - convert to numeric
   mapping = {"low": 0, "medium": 1, "high": 2}
   outcome = OutcomeData(
       values=[mapping[x] for x in ["high", "low", "medium"]],
       outcome_type="continuous"
   )

Missing Data
------------

**Problem**: NaN values in your data

**Solutions:**

1. **Automatic handling** (recommended):

.. code-block:: python

   import numpy as np

   # Library automatically handles missing values
   treatment = TreatmentData(values=[1, 0, np.nan, 1, 0])

   # Check which observations have missing data
   print(f"Missing indices: {treatment.missing_indices}")

2. **Manual preprocessing**:

.. code-block:: python

   import pandas as pd

   # Remove rows with missing treatment
   data = data.dropna(subset=['treatment_column'])

   # Or impute missing values
   data['covariate'].fillna(data['covariate'].mean(), inplace=True)

Data Type Issues
---------------

**Problem**: Pandas DataFrame columns have wrong data types

**Solution:**

.. code-block:: python

   import pandas as pd

   # Check data types
   print(data.dtypes)

   # Convert treatment to numeric if needed
   data['treatment'] = pd.to_numeric(data['treatment'], errors='coerce')

   # Convert categorical strings to numeric codes
   data['category'] = pd.Categorical(data['category']).codes

Model Issues
============

Extreme Effect Sizes
--------------------

**Problem**: Unrealistically large treatment effects (e.g., ATE > 1 for binary outcome)

**Diagnosis:**

.. code-block:: python

   # Check raw group differences
   treated_mean = data[data['treatment'] == 1]['outcome'].mean()
   control_mean = data[data['treatment'] == 0]['outcome'].mean()
   naive_diff = treated_mean - control_mean

   print(f"Naive difference: {naive_diff}")
   print(f"Causal estimate: {effect.ate}")

   # If very different, likely confounding or model issues

**Solutions:**

1. **Check for confounding**:

.. code-block:: python

   from causal_inference.diagnostics import check_balance

   # Check if treatment groups are balanced on covariates
   balance = check_balance(treatment, covariates)
   print(balance.summary())

2. **Try different estimator**:

.. code-block:: python

   # Compare multiple methods
   from causal_inference.estimators import GComputation, IPW, AIPW

   methods = [GComputation(), IPW(), AIPW()]
   effects = []

   for method in methods:
       effect = method.estimate_ate(treatment, outcome, covariates)
       effects.append(effect.ate)
       print(f"{method.__class__.__name__}: {effect.ate:.4f}")

Near-Zero Effects
----------------

**Problem**: Treatment effect estimates very close to zero

**Possible Explanations:**

1. **True null effect** - treatment genuinely has no impact
2. **Bias toward null** - measurement error or confounding
3. **Insufficient power** - sample too small to detect real effect

**Diagnosis:**

.. code-block:: python

   # Check confidence intervals
   if effect.ci_lower < 0 < effect.ci_upper:
       print("Effect not statistically different from zero")
       print(f"But could be as large as {effect.ci_upper:.4f}")

   # Check power analysis
   from causal_inference.utils import power_analysis
   power = power_analysis(
       effect_size=0.1,  # Minimum effect you care about
       sample_size=len(data),
       alpha=0.05
   )
   print(f"Statistical power: {power:.2f}")

Convergence Issues
------------------

**Problem**: Model fitting fails or gives convergence warnings

**Solutions:**

1. **Scale your features**:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   scaled_covariates = scaler.fit_transform(covariates.values)

   scaled_cov_data = CovariateData(
       values=scaled_covariates,
       names=covariates.names
   )

2. **Reduce model complexity**:

.. code-block:: python

   # Use simpler models if complex ones fail
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier

   # Instead of complex model
   estimator = AIPW(propensity_model=LogisticRegression(max_iter=1000))

3. **Check for perfect separation**:

.. code-block:: python

   # Look for covariates that perfectly predict treatment
   for col in covariates.names:
       crosstab = pd.crosstab(data[col], data['treatment'])
       print(f"{col}:\n{crosstab}\n")

Positivity Issues
=================

No Overlap Problem
-----------------

**Problem**: ``PositivityError`` - no overlap between treatment and control groups

**Diagnosis:**

.. code-block:: python

   from causal_inference.diagnostics import check_overlap

   # Check propensity score distribution
   overlap = check_overlap(treatment, covariates)
   overlap.plot()  # Visualize overlap

**Solutions:**

1. **Trim extreme observations**:

.. code-block:: python

   from causal_inference.estimators import AIPW

   # Trim extreme propensity scores
   estimator = AIPW(trim_weights=True, trim_quantiles=(0.05, 0.95))
   effect = estimator.estimate_ate(treatment, outcome, covariates)

2. **Focus on common support**:

.. code-block:: python

   # Only analyze observations with good overlap
   from causal_inference.utils import common_support_filter

   mask = common_support_filter(treatment, covariates)

   # Subset your data
   filtered_treatment = TreatmentData(values=treatment.values[mask])
   filtered_outcome = OutcomeData(values=outcome.values[mask])
   filtered_covariates = CovariateData(
       values=covariates.values[mask],
       names=covariates.names
   )

3. **Use methods robust to positivity violations**:

.. code-block:: python

   # G-computation is more robust to positivity issues than IPW
   from causal_inference.estimators import GComputation

   estimator = GComputation()
   effect = estimator.estimate_ate(treatment, outcome, covariates)

Performance Issues
==================

Slow Computation
----------------

**Problem**: Analysis takes too long

**Solutions:**

1. **Use faster models**:

.. code-block:: python

   # Fast linear models instead of complex ML
   from sklearn.linear_model import LogisticRegression, LinearRegression

   estimator = AIPW(
       outcome_model=LinearRegression(),
       propensity_model=LogisticRegression()
   )

2. **Reduce cross-validation folds**:

.. code-block:: python

   # Fewer folds for faster computation
   estimator = DoublyRobustML(n_folds=3)  # Instead of 5 or 10

3. **Sample your data**:

.. code-block:: python

   # Work with a subset for initial analysis
   sample_data = data.sample(n=10000, random_state=42)

Memory Issues
------------

**Problem**: Out of memory errors with large datasets

**Solutions:**

1. **Use memory-efficient models**:

.. code-block:: python

   # Use SGD-based models for large data
   from sklearn.linear_model import SGDClassifier, SGDRegressor

   estimator = AIPW(
       outcome_model=SGDRegressor(),
       propensity_model=SGDClassifier()
   )

2. **Process data in chunks**:

.. code-block:: python

   # For very large datasets, process in chunks
   chunk_size = 50000
   effects = []

   for chunk in pd.read_csv("large_file.csv", chunksize=chunk_size):
       # Process each chunk
       effect = estimator.estimate_ate(treatment, outcome, covariates)
       effects.append(effect)

   # Combine results
   overall_effect = combine_effects(effects)

Interpretation Issues
=====================

Confidence Intervals
--------------------

**Problem**: Very wide confidence intervals

**Causes & Solutions:**

1. **Small sample size** → Collect more data
2. **High outcome variance** → Use covariates to reduce noise
3. **Poor model fit** → Try different estimators or better models
4. **Positivity issues** → Address overlap problems

Statistical Significance
------------------------

**Problem**: Interpreting p-values and significance

**Best Practices:**

.. code-block:: python

   # Don't just look at p-values
   print(f"ATE: {effect.ate:.4f}")
   print(f"95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]")
   print(f"P-value: {effect.p_value:.4f}")

   # Consider practical significance too
   if abs(effect.ate) > 0.05:  # Your threshold
       print("Effect is practically significant")

   # Report effect size relative to baseline
   baseline_rate = outcome.values.mean()
   relative_effect = effect.ate / baseline_rate
   print(f"Relative effect: {relative_effect:.1%}")

Causal Interpretation
--------------------

**Problem**: Confusing correlation and causation

**Key Points:**

1. **Causal claims require assumptions** - always state them
2. **Effect estimates are only as good as your identification strategy**
3. **Run diagnostic tests** to validate assumptions
4. **Consider alternative explanations** for your findings

.. code-block:: python

   # Always run diagnostics
   diagnostics = estimator.diagnose(treatment, outcome, covariates)

   if diagnostics.assumptions_satisfied:
       print("✅ Assumptions appear satisfied - causal interpretation likely valid")
   else:
       print("⚠️ Assumption violations detected - interpret causally with caution")
       print("Consider:")
       print("- Alternative identification strategies")
       print("- Sensitivity analysis")
       print("- Additional robustness checks")

Getting Help
============

Diagnostic Checklist
--------------------

Before asking for help, run through this checklist:

.. code-block:: python

   # 1. Data validation
   print("=== Data Validation ===")
   print(f"Treatment type: {treatment.treatment_type}")
   print(f"Outcome type: {outcome.outcome_type}")
   print(f"Missing values: {treatment.missing_indices}")

   # 2. Basic descriptives
   print("\n=== Descriptives ===")
   print(f"Sample size: {len(treatment.values)}")
   print(f"Treatment rate: {treatment.values.mean():.3f}")
   print(f"Outcome mean: {outcome.values.mean():.3f}")

   # 3. Balance check
   print("\n=== Balance ===")
   balance = check_balance(treatment, covariates)
   print(balance.summary())

   # 4. Overlap check
   print("\n=== Overlap ===")
   overlap = check_overlap(treatment, covariates)
   print(f"Overlap score: {overlap.score:.3f}")

   # 5. Effect estimate
   print("\n=== Effect ===")
   effect = estimator.estimate_ate(treatment, outcome, covariates)
   print(f"ATE: {effect.ate:.4f} [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]")

Where to Get Help
----------------

1. **Documentation**: Check the :doc:`../api/index` and :doc:`../tutorials/index`
2. **GitHub Issues**: `Report bugs <https://github.com/datablogin/causal-inference-marketing/issues>`_
3. **GitHub Discussions**: `Ask questions <https://github.com/datablogin/causal-inference-marketing/discussions>`_
4. **Stack Overflow**: Tag questions with ``causal-inference`` and ``python``

When Reporting Issues
--------------------

Please include:

1. **Minimal reproducible example** with synthetic data
2. **Full error traceback**
3. **Library version**: ``print(causal_inference.__version__)``
4. **Environment info**: Python version, OS, key package versions
5. **What you expected vs. what happened**

.. code-block:: python

   # Include this diagnostic info in bug reports
   import sys
   import causal_inference
   import pandas as pd
   import numpy as np

   print(f"causal_inference version: {causal_inference.__version__}")
   print(f"Python version: {sys.version}")
   print(f"pandas version: {pd.__version__}")
   print(f"numpy version: {np.__version__}")
