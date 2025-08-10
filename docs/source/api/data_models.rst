Data Models
===========

Core Pydantic data models for type-safe causal inference.

Overview
--------

All data in the library is represented using Pydantic models that provide:

- **Type validation** - Ensures data types are correct
- **Field validation** - Validates ranges, formats, and constraints
- **Automatic conversion** - Converts compatible types (e.g., lists to arrays)
- **Clear error messages** - Helpful validation error descriptions
- **IDE support** - Full autocompletion and type hints

Treatment Data
--------------

.. autoclass:: causal_inference.core.base.TreatmentData
   :members:
   :show-inheritance:

**Supported Treatment Types:**

- **Binary**: ``treatment_type="binary"`` - Two values (0/1, True/False)
- **Categorical**: ``treatment_type="categorical"`` - Multiple discrete values
- **Continuous**: ``treatment_type="continuous"`` - Numeric continuous values

**Example Usage:**

.. code-block:: python

   import pandas as pd
   from causal_inference.core import TreatmentData

   # Binary treatment (email campaign)
   binary_treatment = TreatmentData(
       values=[0, 1, 1, 0, 1],
       name="email_campaign",
       treatment_type="binary"
   )

   # Categorical treatment (marketing channel)
   categorical_treatment = TreatmentData(
       values=["email", "social", "search", "email", "control"],
       name="channel",
       treatment_type="categorical",
       categories=["control", "email", "social", "search"]
   )

   # Continuous treatment (ad spend)
   continuous_treatment = TreatmentData(
       values=[100.5, 250.0, 0.0, 150.2, 300.1],
       name="ad_spend",
       treatment_type="continuous"
   )

Outcome Data
------------

.. autoclass:: causal_inference.core.base.OutcomeData
   :members:
   :show-inheritance:

**Supported Outcome Types:**

- **Continuous**: ``outcome_type="continuous"`` - Numeric outcomes (revenue, time spent)
- **Binary**: ``outcome_type="binary"`` - Binary outcomes (conversion, churn)
- **Count**: ``outcome_type="count"`` - Count data (page views, purchases)

**Example Usage:**

.. code-block:: python

   from causal_inference.core import OutcomeData

   # Binary outcome (conversion)
   conversion = OutcomeData(
       values=[1, 0, 1, 0, 1],
       name="conversion",
       outcome_type="binary"
   )

   # Continuous outcome (revenue)
   revenue = OutcomeData(
       values=[25.50, 0.0, 47.25, 12.10, 156.75],
       name="revenue",
       outcome_type="continuous"
   )

   # Count outcome (page views)
   page_views = OutcomeData(
       values=[5, 1, 12, 3, 8],
       name="page_views",
       outcome_type="count"
   )

Survival Outcome Data
---------------------

.. autoclass:: causal_inference.core.base.SurvivalOutcomeData
   :members:
   :show-inheritance:

For survival/time-to-event analysis:

.. code-block:: python

   from causal_inference.core import SurvivalOutcomeData

   survival = SurvivalOutcomeData(
       times=[30, 45, 60, 90, 120],  # Time to event or censoring
       events=[1, 0, 1, 1, 0],       # 1=event occurred, 0=censored
       name="customer_lifetime"
   )

Covariate Data
--------------

.. autoclass:: causal_inference.core.base.CovariateData
   :members:
   :show-inheritance:

For confounding variables and other covariates:

.. code-block:: python

   import pandas as pd
   from causal_inference.core import CovariateData

   # Multiple covariates from DataFrame
   df = pd.DataFrame({
       'age': [25, 34, 45, 29, 38],
       'income': [45000, 67000, 89000, 52000, 71000],
       'previous_purchases': [2, 5, 8, 1, 3]
   })

   covariates = CovariateData(
       values=df,
       names=["age", "income", "previous_purchases"]
   )

   # Single covariate
   age_covariate = CovariateData(
       values=df["age"],
       names=["age"]
   )

Instrument Data
---------------

.. autoclass:: causal_inference.core.base.InstrumentData
   :members:
   :show-inheritance:

For instrumental variable analysis:

.. code-block:: python

   from causal_inference.core import InstrumentData

   # Weather as instrument for advertising effectiveness
   instrument = InstrumentData(
       values=[1, 0, 1, 0, 1],  # 1=good weather, 0=bad weather
       name="weather_instrument"
   )

Causal Effect Results
---------------------

.. autoclass:: causal_inference.core.base.CausalEffect
   :members:
   :show-inheritance:

This is the standard result format returned by all estimators:

**Key Attributes:**

- **ate**: Average Treatment Effect
- **ci_lower/ci_upper**: Confidence interval bounds
- **se**: Standard error of the estimate
- **p_value**: Statistical significance test
- **diagnostics**: Model diagnostics and assumptions

**Example Usage:**

.. code-block:: python

   from causal_inference.estimators import GComputation

   estimator = GComputation()
   effect = estimator.estimate_ate(treatment, outcome, covariates)

   print(f"ATE: {effect.ate:.4f}")
   print(f"95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]")
   print(f"P-value: {effect.p_value:.4f}")

   # Check if statistically significant
   if effect.p_value < 0.05:
       print("Effect is statistically significant")

Data Validation
---------------

All models include comprehensive validation:

**Treatment Data Validation:**

.. code-block:: python

   # This will raise a validation error
   try:
       invalid_treatment = TreatmentData(
           values=["A", "B", "C"],
           treatment_type="binary"  # Binary expects 2 unique values
       )
   except ValidationError as e:
       print(f"Validation error: {e}")

**Outcome Data Validation:**

.. code-block:: python

   # Count outcomes must be non-negative integers
   try:
       invalid_outcome = OutcomeData(
           values=[-1, 2.5, 3],  # Negative values and floats not allowed
           outcome_type="count"
       )
   except ValidationError as e:
       print(f"Validation error: {e}")

**Missing Data Handling:**

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Missing values are automatically handled
   data_with_missing = TreatmentData(
       values=[1, 0, np.nan, 1, 0],  # NaN values detected
       treatment_type="binary"
   )

   print(f"Missing indices: {data_with_missing.missing_indices}")

Common Patterns
---------------

**From Pandas DataFrames:**

.. code-block:: python

   import pandas as pd
   from causal_inference.core import TreatmentData, OutcomeData, CovariateData

   df = pd.read_csv("marketing_data.csv")

   treatment = TreatmentData(values=df["treatment_column"])
   outcome = OutcomeData(values=df["outcome_column"])
   covariates = CovariateData(
       values=df[["age", "income", "location"]],
       names=["age", "income", "location"]
   )

**From NumPy Arrays:**

.. code-block:: python

   import numpy as np

   treatment = TreatmentData(values=np.array([0, 1, 1, 0]))
   outcome = OutcomeData(values=np.array([0.1, 0.8, 0.6, 0.2]))

**Type Conversion:**

.. code-block:: python

   # Automatic conversion from lists
   treatment = TreatmentData(values=[True, False, True, False])
   # Converts to: [1, 0, 1, 0]

   # String categorical treatments
   treatment = TreatmentData(
       values=["control", "email", "sms", "email"],
       treatment_type="categorical"
   )

Validation Reference
--------------------

**TreatmentData Validation Rules:**

- Binary: Must have exactly 2 unique non-missing values
- Categorical: Must specify ``categories`` parameter
- Continuous: Must be numeric values

**OutcomeData Validation Rules:**

- Continuous: Must be numeric
- Binary: Must have exactly 2 unique non-missing values
- Count: Must be non-negative integers

**CovariateData Validation Rules:**

- Column count must match length of ``names`` parameter
- All values must be numeric (after encoding)
- Missing values are tracked but allowed

**Common Validation Errors:**

.. code-block:: python

   from pydantic import ValidationError

   # Wrong number of unique values for binary
   try:
       TreatmentData(values=[0, 1, 2], treatment_type="binary")
   except ValidationError:
       pass  # Error: Binary treatment must have exactly 2 unique values

   # Missing categories for categorical
   try:
       TreatmentData(values=["A", "B", "C"], treatment_type="categorical")
   except ValidationError:
       pass  # Error: Must specify categories for categorical treatment

   # Non-numeric continuous treatment
   try:
       TreatmentData(values=["low", "high"], treatment_type="continuous")
   except ValidationError:
       pass  # Error: Continuous treatment must be numeric
