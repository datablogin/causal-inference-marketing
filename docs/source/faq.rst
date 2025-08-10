Frequently Asked Questions
===========================

Common questions about causal inference and this library.

Getting Started
===============

**Q: I'm new to causal inference. Where should I start?**

A: Start with our :doc:`quickstart` guide, which walks through a simple example in 5 minutes.
Then read the :doc:`methodology/index` to understand the key concepts. Finally, try the
:doc:`tutorials/first_analysis` for a more detailed walkthrough.

**Q: What's the difference between causal inference and machine learning?**

A: Machine learning focuses on **prediction** - accurately forecasting future outcomes.
Causal inference focuses on **understanding cause-and-effect** - what would happen if we changed something.

- **ML question**: "Will this customer convert?"
- **Causal question**: "Would this customer convert if we sent them an email?"

**Q: Do I need to know statistics to use this library?**

A: Basic understanding helps, but the library is designed to be accessible. Key concepts to understand:

- Confounding variables (things that affect both treatment and outcome)
- Selection bias (systematic differences between treatment groups)
- The difference between correlation and causation

Method Selection
================

**Q: Which causal inference method should I use?**

A: It depends on your data and problem. Quick guide:

- **Randomized experiment**: Simple difference in means
- **Observational data with good confounders**: AIPW (Augmented IPW)
- **High-dimensional data**: DoublyRobustML
- **Panel/time series data**: Difference-in-Differences
- **Sharp discontinuity**: Regression Discontinuity
- **Individual-level effects**: CausalForest

See our :doc:`methodology/method_selection` for a detailed guide.

**Q: What's the best all-purpose method?**

A: **AIPW (Augmented Inverse Probability Weighting)** is often the best starting point because:

- It's "doubly robust" - works if either the outcome model OR propensity model is correct
- Good performance across various scenarios
- Handles binary, categorical, and continuous treatments
- Available with ``AIPW()`` estimator

**Q: When should I use machine learning vs. simple linear models?**

A: Start simple, then add complexity:

- **Use linear models** for interpretability and when sample size < 10,000
- **Use ML models** when you have complex relationships and large datasets (> 10,000 observations)
- **DoublyRobustML** automatically handles this with cross-fitting

Data Requirements
=================

**Q: How much data do I need?**

A: It depends on the method and effect size:

- **Minimum**: 500-1,000 observations for basic methods
- **Recommended**: 10,000+ observations for stable results
- **For ML methods**: 10,000+ observations minimum
- **For heterogeneous effects**: 50,000+ observations

Smaller effect sizes require larger samples for reliable detection.

**Q: What if I have missing data?**

A: The library handles missing data automatically:

.. code-block:: python

   # Missing values are automatically tracked
   treatment = TreatmentData(values=[1, 0, np.nan, 1])
   print(treatment.missing_indices)  # Shows which rows have missing data

   # Observations with missing treatment/outcome are excluded from analysis

For missing covariates, consider imputation before analysis.

**Q: Can I use this with time series data?**

A: Yes, but you need the right method:

- **Panel data**: Use ``DifferenceInDifferences``
- **Single treated unit**: Use ``SyntheticControl``
- **Time-varying treatments**: Include time-related covariates in observational methods

**Q: What data format should I use?**

A: The library accepts pandas DataFrames, Series, or numpy arrays:

.. code-block:: python

   # From pandas (recommended)
   treatment = TreatmentData(values=df["treatment_column"])

   # From numpy arrays
   treatment = TreatmentData(values=np.array([0, 1, 1, 0]))

   # From lists (automatically converted)
   treatment = TreatmentData(values=[0, 1, 1, 0])

Interpretation
==============

**Q: How do I interpret the Average Treatment Effect (ATE)?**

A: The ATE tells you the average difference in outcomes if everyone received treatment vs. if everyone received control.

**Example**: ATE = 0.15 for a binary outcome means treatment increases the probability by 15 percentage points on average.

**Important**: This is different from simply comparing treated vs. untreated groups, because the ATE adjusts for confounding.

**Q: What's a "statistically significant" result?**

A: Typically p < 0.05, meaning there's less than 5% chance the result is due to random variation alone.

But also consider:
- **Practical significance**: Is the effect large enough to matter?
- **Confidence intervals**: How precise is the estimate?
- **Business context**: What effect size would change your decisions?

**Q: My effect size seems too large/small. Is this normal?**

A: Check these possibilities:

**Large effects** (> 50% change):
- Model misspecification (try different estimator)
- Unmeasured confounding
- Data quality issues

**Small effects** (< 1% change):
- Genuinely small effect (common in marketing)
- Insufficient statistical power
- Bias toward null due to measurement error

Run diagnostic tests to investigate.

**Q: How do I know if my results are reliable?**

A: Run these checks:

.. code-block:: python

   # 1. Balance diagnostics
   from causal_inference.diagnostics import check_balance
   balance = check_balance(treatment, covariates)

   # 2. Overlap diagnostics
   from causal_inference.diagnostics import check_overlap
   overlap = check_overlap(treatment, covariates)

   # 3. Compare multiple methods
   methods = [GComputation(), IPW(), AIPW()]
   effects = [m.estimate_ate(treatment, outcome, covariates) for m in methods]

   # 4. Sensitivity analysis
   from causal_inference.diagnostics import sensitivity_analysis
   sensitivity = sensitivity_analysis(estimator, effect)

If diagnostics look good and multiple methods agree, results are likely reliable.

Technical Issues
================

**Q: I get a `ValidationError` when creating data objects. What's wrong?**

A: Common causes:

1. **Wrong treatment type**:
   - Binary treatments need exactly 2 unique values
   - Categorical treatments need ``categories`` parameter
   - Continuous treatments must be numeric

2. **Data type mismatch**:
   - Convert strings to numbers for continuous variables
   - Use appropriate ``treatment_type`` and ``outcome_type``

See :doc:`troubleshooting/index` for detailed solutions.

**Q: The analysis is very slow. How can I speed it up?**

A: Try these optimizations:

.. code-block:: python

   # 1. Use simpler models
   from sklearn.linear_model import LogisticRegression
   estimator = AIPW(propensity_model=LogisticRegression())

   # 2. Reduce cross-validation folds
   estimator = DoublyRobustML(n_folds=3)  # Instead of 5 or 10

   # 3. Sample your data for initial analysis
   sample = data.sample(n=10000, random_state=42)

**Q: I get convergence warnings. Should I be concerned?**

A: Usually not, but you can:

1. **Scale your features**:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   scaled_covariates = scaler.fit_transform(covariates.values)

2. **Increase max_iter** in sklearn models:

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression(max_iter=1000)  # Default is 100

3. **Try different solvers** or simpler models if problems persist

Business Applications
=====================

**Q: How does this apply to marketing attribution?**

A: Traditional attribution models (last-click, first-click) just describe what happened.
Causal inference estimates what would have happened with different attribution.

.. code-block:: python

   # Multi-channel attribution
   treatment = TreatmentData(
       values=data["channel"],  # "email", "social", "search", "control"
       treatment_type="categorical",
       categories=["control", "email", "social", "search"]
   )

   # Estimates incremental effect of each channel
   effect = estimator.estimate_ate(treatment, outcome, covariates)

**Q: Can I measure the incrementality of my advertising campaigns?**

A: Yes! This is perfect for causal inference:

.. code-block:: python

   # Define treatment as ad exposure
   treatment = TreatmentData(values=data["exposed_to_ad"])
   outcome = OutcomeData(values=data["conversion"])

   # Include important confounders
   covariates = CovariateData(
       values=data[["age", "income", "past_purchases", "website_visits"]],
       names=["age", "income", "past_purchases", "website_visits"]
   )

   effect = estimator.estimate_ate(treatment, outcome, covariates)
   # effect.ate = incremental lift from advertising

**Q: How do I optimize marketing budget allocation?**

A: Estimate effects for different spend levels:

.. code-block:: python

   # Continuous treatment (spend amount)
   treatment = TreatmentData(
       values=data["ad_spend"],
       treatment_type="continuous"
   )

   # Can estimate dose-response curves
   dose_response = estimator.estimate_dose_response(
       treatment, outcome, covariates
   )

**Q: Can this help with pricing experiments?**

A: Absolutely:

.. code-block:: python

   # Price as treatment
   treatment = TreatmentData(
       values=data["price"],
       treatment_type="continuous"
   )

   # Revenue/conversion as outcome
   outcome = OutcomeData(values=data["revenue"])

   # Estimate price elasticity
   elasticity = estimator.estimate_ate(treatment, outcome, covariates)

Advanced Topics
===============

**Q: How do I estimate heterogeneous treatment effects?**

A: Use methods that estimate individual-level effects:

.. code-block:: python

   from causal_inference.estimators import CausalForest, XLearner

   # Causal Forest - tree-based method
   forest = CausalForest()
   cate_results = forest.estimate_cate(treatment, outcome, covariates)
   individual_effects = cate_results.individual_effects

   # X-Learner - meta-learner approach
   xlearner = XLearner()
   cate_results = xlearner.estimate_cate(treatment, outcome, covariates)

**Q: How do I handle multiple treatments?**

A: Use categorical treatments:

.. code-block:: python

   # Multiple marketing channels
   treatment = TreatmentData(
       values=data["channel"],
       treatment_type="categorical",
       categories=["control", "email", "social", "search", "display"]
   )

   # This estimates effect of each channel vs. control
   effects = estimator.estimate_ate(treatment, outcome, covariates)

**Q: What about interference between units?**

A: This is challenging - when one unit's treatment affects another's outcome:

- **Cluster randomization**: Randomize at group level (e.g., by geography)
- **Network analysis**: Use specialized methods for network interference
- **Spillover analysis**: Test for spillover effects in your data

Currently, the library assumes no interference (SUTVA assumption).

**Q: How do I test for unmeasured confounding?**

A: Use sensitivity analysis:

.. code-block:: python

   from causal_inference.diagnostics import sensitivity_analysis

   # Test how sensitive results are to unmeasured confounding
   sensitivity = sensitivity_analysis(
       estimator=estimator,
       effect=effect,
       confounding_strength_range=(0, 0.3)
   )

   sensitivity.plot()  # Shows how effect changes with confounding

Getting Help
============

**Q: Where can I get help if I'm stuck?**

A: Multiple resources available:

1. **Documentation**: :doc:`api/index` and :doc:`tutorials/index`
2. **Troubleshooting**: :doc:`troubleshooting/index`
3. **GitHub Issues**: `Report bugs <https://github.com/datablogin/causal-inference-marketing/issues>`_
4. **GitHub Discussions**: `Ask questions <https://github.com/datablogin/causal-inference-marketing/discussions>`_

**Q: How do I report a bug?**

A: Create a GitHub issue with:
- Minimal reproducible example
- Full error traceback
- System information (Python version, library version, OS)
- What you expected vs. what happened

**Q: Can I contribute to the library?**

A: Yes! See our contributing guide:
- Fork the repository
- Create a feature branch
- Add tests for new functionality
- Submit a pull request

We welcome contributions of all kinds - bug fixes, new features, documentation improvements, and examples.

**Q: Is this library ready for production use?**

A: The library is in active development. Current status:

- **Core estimators**: Production ready (G-computation, IPW, AIPW)
- **ML estimators**: Beta (DoublyRobustML, CausalForest)
- **Quasi-experimental methods**: Alpha (RDD, DiD, Synthetic Control)
- **API service**: Beta

Check the GitHub releases page for the latest stability information.
