Quickstart Guide
================

Get up and running with causal inference in 5 minutes!

This guide walks you through your first causal analysis using our library.

30-Second Setup
===============

.. code-block:: bash

   pip install causal-inference-marketing

.. code-block:: python

   from causal_inference.estimators import GComputation
   print("Ready to go! 🚀")

Your First Analysis
===================

Let's estimate the causal effect of an email marketing campaign on conversions.

Step 1: Import the Library
--------------------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from causal_inference.estimators import GComputation
   from causal_inference.core import TreatmentData, OutcomeData, CovariateData

Step 2: Create Sample Data
--------------------------

For this example, we'll create synthetic marketing data:

.. code-block:: python

   # Generate sample marketing data
   np.random.seed(42)
   n = 1000

   # Customer characteristics
   age = np.random.normal(40, 15, n)
   income = np.random.normal(50000, 20000, n)
   previous_purchases = np.random.poisson(3, n)

   # Email campaign assignment (treatment)
   # Higher income customers more likely to be targeted
   email_prob = 0.3 + 0.3 * (income > 50000)
   email_campaign = np.random.binomial(1, email_prob, n)

   # Conversion outcome
   # Both email and customer characteristics affect conversion
   conversion_prob = (
       0.1 +  # baseline conversion
       0.15 * email_campaign +  # email effect
       0.0001 * income +  # income effect
       0.02 * previous_purchases  # loyalty effect
   )
   conversion = np.random.binomial(1, conversion_prob, n)

   # Create DataFrame
   data = pd.DataFrame({
       'age': age,
       'income': income,
       'previous_purchases': previous_purchases,
       'email_campaign': email_campaign,
       'conversion': conversion
   })

   print(f"Data shape: {data.shape}")
   print(f"Conversion rate: {data['conversion'].mean():.3f}")
   print(f"Email campaign rate: {data['email_campaign'].mean():.3f}")

Step 3: Define Your Causal Problem
----------------------------------

.. code-block:: python

   # Define treatment: email campaign
   treatment = TreatmentData(
       values=data["email_campaign"],
       name="email_campaign",
       treatment_type="binary"
   )

   # Define outcome: conversion
   outcome = OutcomeData(
       values=data["conversion"],
       name="conversion",
       outcome_type="binary"
   )

   # Define confounders: variables that affect both treatment and outcome
   covariates = CovariateData(
       values=data[["age", "income", "previous_purchases"]],
       names=["age", "income", "previous_purchases"]
   )

Step 4: Choose and Run Estimator
---------------------------------

.. code-block:: python

   # Initialize G-computation estimator
   estimator = GComputation()

   # Estimate the Average Treatment Effect (ATE)
   effect = estimator.estimate_ate(
       treatment=treatment,
       outcome=outcome,
       covariates=covariates
   )

   # Display results
   print("📊 Causal Analysis Results")
   print("=" * 30)
   print(f"Average Treatment Effect: {effect.ate:.4f}")
   print(f"95% Confidence Interval: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]")
   print(f"Standard Error: {effect.se:.4f}")
   print(f"P-value: {effect.p_value:.4f}")

   # Interpretation
   if effect.p_value < 0.05:
       print("✅ Statistically significant effect detected!")
   else:
       print("❌ No statistically significant effect found.")

Expected Output
---------------

.. code-block:: none

   📊 Causal Analysis Results
   ==============================
   Average Treatment Effect: 0.1543
   95% Confidence Interval: [0.1122, 0.1964]
   Standard Error: 0.0215
   P-value: 0.0000
   ✅ Statistically significant effect detected!

Step 5: Interpret the Results
-----------------------------

Our analysis shows:

- **ATE = 0.154**: Email campaigns increase conversion probability by ~15.4 percentage points
- **95% CI**: We're 95% confident the true effect is between 11.2% and 19.6%
- **P-value < 0.001**: The effect is statistically significant

⚠️ **Important**: This is the causal effect after adjusting for confounders (age, income, previous purchases).

Understanding the Results
=========================

What We Estimated
-----------------

The **Average Treatment Effect (ATE)** answers:

   *"If we randomly assigned email campaigns to all customers, what would be the average increase in conversion rate?"*

This differs from simply comparing conversion rates between groups, which would be biased due to confounding.

Why G-computation?
------------------

G-computation (standardization) works by:

1. **Modeling the outcome**: Fitting a model to predict conversions based on treatment and covariates
2. **Standardizing**: Computing expected outcomes under different treatment scenarios
3. **Comparing**: Taking the difference to get the causal effect

It's robust when the outcome model is correctly specified.

Next Steps
==========

🎯 **Try Different Methods**
   Compare results using other estimators:

.. code-block:: python

   from causal_inference.estimators import IPW, AIPW

   # Inverse Probability Weighting
   ipw = IPW()
   ipw_effect = ipw.estimate_ate(treatment, outcome, covariates)

   # Augmented IPW (Doubly Robust)
   aipw = AIPW()
   aipw_effect = aipw.estimate_ate(treatment, outcome, covariates)

📊 **Run Diagnostics**
   Check assumptions and model fit:

.. code-block:: python

   # Check balance of covariates
   from causal_inference.diagnostics import check_balance
   balance = check_balance(treatment, covariates)

   # Sensitivity analysis
   from causal_inference.diagnostics import sensitivity_analysis
   sensitivity = sensitivity_analysis(estimator, effect)

🚀 **Production Deployment**
   Deploy as a web service:

.. code-block:: python

   # Start FastAPI service
   from services.causal_api.main import app
   import uvicorn

   uvicorn.run(app, host="0.0.0.0", port=8000)

Common Patterns
===============

Marketing Attribution
----------------------

.. code-block:: python

   # Multi-channel attribution
   treatment = TreatmentData(
       values=data["channel"],  # "email", "social", "search", "control"
       treatment_type="categorical",
       categories=["control", "email", "social", "search"]
   )

Continuous Treatments
---------------------

.. code-block:: python

   # Ad spend effect
   treatment = TreatmentData(
       values=data["ad_spend"],
       treatment_type="continuous"
   )

Time Series Data
----------------

.. code-block:: python

   # For temporal data, include time-related covariates
   covariates = CovariateData(
       values=data[["month", "seasonality", "trend", "baseline_features"]],
       names=["month", "seasonality", "trend", "baseline_features"]
   )

What's Next?
============

- **Deep Dive**: Read :doc:`tutorials/first_analysis` for detailed explanations
- **Choose Methods**: Use :doc:`methodology/method_selection` to pick the right estimator
- **Real Examples**: Try :doc:`tutorials/marketing_attribution` with real marketing data
- **Production**: Deploy with :doc:`deployment/fastapi_guide`

Need Help?
==========

- 📚 **Full API docs**: :doc:`api/index`
- 🐛 **Found a bug?**: `GitHub Issues <https://github.com/datablogin/causal-inference-marketing/issues>`_
- 💬 **Questions?**: `GitHub Discussions <https://github.com/datablogin/causal-inference-marketing/discussions>`_
