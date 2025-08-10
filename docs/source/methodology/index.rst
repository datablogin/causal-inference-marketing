Methodology Guide
=================

Understanding when and how to apply causal inference methods.

.. toctree::
   :maxdepth: 2

   method_selection
   estimator_guide
   assumptions

Overview
--------

Causal inference is fundamentally about answering **counterfactual questions**:
*"What would have happened if we had done something different?"*

This is different from prediction (which asks *"What will happen?"*) or
description (which asks *"What did happen?"*).

The Fundamental Problem
-----------------------

The **fundamental problem of causal inference** is that we can never directly observe counterfactuals.
For any individual, we can observe their outcome under either treatment or control, but never both.

**Example:** A customer either receives an email campaign (treatment) or doesn't (control).
We observe their purchase behavior under one condition, but we can never know what they
would have done under the other condition.

This is where causal inference methods come in - they use assumptions and statistical
techniques to estimate these unobservable counterfactuals.

Key Concepts
============

Treatment Effects
-----------------

**Individual Treatment Effect (ITE)**
   The difference in outcomes for an individual under treatment vs. control:

   .. math:: \tau_i = Y_i(1) - Y_i(0)

   Where :math:`Y_i(1)` is individual i's outcome under treatment and :math:`Y_i(0)` is their outcome under control.

**Average Treatment Effect (ATE)**
   The population average of individual treatment effects:

   .. math:: \tau = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]

**Conditional Average Treatment Effect (CATE)**
   The treatment effect conditional on covariates:

   .. math:: \tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]

Common Causal Questions in Marketing
====================================

Attribution
-----------
- Which marketing channels are driving conversions?
- How much credit should each touchpoint receive?
- What is the incremental impact of each channel?

Incrementality Testing
---------------------
- What is the lift from our advertising campaigns?
- Are we reaching new customers or just existing ones?
- How does performance vary across different audiences?

Media Mix Modeling
------------------
- How should we allocate our marketing budget?
- What are the diminishing returns to each channel?
- How do channels interact and complement each other?

A/B Testing & Experimentation
-----------------------------
- What is the true treatment effect of our experiment?
- How do we handle interference between units?
- What is the optimal experimental design?

Price Testing
-------------
- What is the demand elasticity for our products?
- How do price changes affect customer lifetime value?
- What are optimal pricing strategies?

Identification Strategies
=========================

Selection on Observables
------------------------
**Assumption**: All confounders are observed and controlled for.

**Methods**: G-computation, IPW, AIPW, Doubly Robust ML

**When to use**: Rich observational data with good confounder measurement

**Example**: Using customer demographics, purchase history, and behavior data to estimate email campaign effects

Quasi-Experimental Designs
--------------------------

**Regression Discontinuity**
- Exploits arbitrary cutoffs in treatment assignment
- Example: Loyalty program eligibility based on spending threshold

**Difference-in-Differences**
- Uses time variation and control groups
- Example: Marketing campaign launched in some regions but not others

**Synthetic Control**
- Creates synthetic counterfactuals from control units
- Example: Estimating impact of a new advertising strategy in one market

Instrumental Variables
---------------------
**Assumption**: Have a valid instrument that affects treatment but not outcome directly

**When to use**: When confounding is suspected but unmeasured

**Example**: Using weather as instrument for outdoor advertising effectiveness

Natural Experiments
-------------------
**Idea**: Find naturally occurring randomization

**Examples**:
- Lottery-based targeting
- Geographic or temporal variation in policies
- System outages or technical issues

Method Comparison
=================

Robustness vs. Assumptions
--------------------------

====================  ================  ===================  ==================
Method                Robustness        Key Assumptions      Best Use Case
====================  ================  ===================  ==================
G-computation         Medium            Correct outcome      Good outcome model
                                       model
IPW                   Low               Correct propensity   Good treatment model
                                       model
AIPW                  High              Either model         General purpose
                      (doubly robust)   correct
Doubly Robust ML      High              Either model         High-dimensional
                      (doubly robust)   correct              data
RDD                   Very High         Continuity at        Sharp discontinuity
                                       cutoff
DiD                   Medium            Parallel trends      Panel data
IV                    Medium            Valid instrument     Endogeneity
====================  ================  ===================  ==================

Statistical Power
-----------------

**Sample Size Requirements:**

- **G-computation**: Lower (good outcome model helps)
- **IPW**: Higher (needs strong overlap)
- **AIPW**: Medium (benefits from both models)
- **ML methods**: Higher (more complex models need more data)
- **RDD**: Very high (local estimation near cutoff)

**Precision:**

- Methods that use both outcome and treatment models (AIPW, DML) typically have better precision
- Methods with weaker assumptions (RDD, IV) often have lower precision but higher credibility

Common Pitfalls
================

Model Misspecification
---------------------
- **Outcome model**: G-computation fails if outcome model is wrong
- **Propensity model**: IPW fails if propensity model is wrong
- **Solution**: Use doubly robust methods (AIPW, DML)

Positivity Violations
--------------------
- **Problem**: No overlap between treatment and control groups for some covariate values
- **Symptoms**: Extreme propensity scores (near 0 or 1), unstable results
- **Solutions**: Trim extreme weights, focus on common support region

Unmeasured Confounding
---------------------
- **Problem**: Important confounders not observed in data
- **Symptoms**: Implausible effect sizes, fails falsification tests
- **Solutions**: Use quasi-experimental methods, sensitivity analysis

Multiple Testing
----------------
- **Problem**: Testing many hypotheses increases false positive rate
- **Solutions**: Correct for multiple comparisons, pre-specify primary outcomes

Best Practices
==============

Pre-Analysis Planning
--------------------
1. **Define the research question** clearly
2. **Identify the treatment and outcome** of interest
3. **List potential confounders** based on domain knowledge
4. **Choose identification strategy** before seeing results
5. **Pre-register analysis plan** if possible

Model Selection
--------------
1. **Start simple** with interpretable models
2. **Use domain knowledge** to inform model specification
3. **Compare multiple methods** for robustness
4. **Validate models** on held-out data when possible

Diagnostic Testing
-----------------
1. **Check balance** of covariates across treatment groups
2. **Assess overlap** in propensity scores
3. **Run falsification tests** (placebo outcomes, negative controls)
4. **Perform sensitivity analysis** for unmeasured confounding

Reporting Results
----------------
1. **Report effect sizes** with confidence intervals
2. **Include diagnostic results**
3. **Discuss limitations** and assumptions
4. **Provide practical interpretation** of findings

Decision Framework
==================

Use this decision tree to select the appropriate method:

.. code-block:: none

   Do you have a randomized experiment?
   ├─ Yes → Simple difference in means (t-test)
   └─ No → Do you have a quasi-experimental design?
       ├─ Yes → Use appropriate quasi-experimental method
       │   ├─ Sharp cutoff → Regression Discontinuity
       │   ├─ Panel data → Difference-in-Differences
       │   ├─ One treated unit → Synthetic Control
       │   └─ Valid instrument → Instrumental Variables
       └─ No → Observational study
           ├─ Rich confounders observed → Selection on observables
           │   ├─ Small sample → AIPW
           │   ├─ Large sample → Doubly Robust ML
           │   └─ Want heterogeneous effects → Causal Forest
           └─ Suspected unmeasured confounding → Sensitivity analysis

Next Steps
==========

- :doc:`method_selection` - Detailed method selection guide
- :doc:`estimator_guide` - Statistical details for each estimator
- :doc:`assumptions` - Testing and validating assumptions
- :doc:`../tutorials/index` - Hands-on examples and tutorials
