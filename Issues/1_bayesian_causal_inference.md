# Add Bayesian Causal Inference Module

**Objective:**  
Add Bayesian methods to the library to estimate treatment effects with posterior uncertainty.

**Tasks:**  
- Implement a Bayesian linear model for ATE using PyMC or numpyro.
- Use `qsmk` as treatment, `wt82_71` as outcome, and adjust for age, sex, education, and income.
- Return full posterior, mean estimate, and 95% credible interval.

**Benchmark:**  
- Dataset: `nhefs.csv`
- Compare posterior mean to AIPW/DR estimate from existing library.
- Plot posterior distribution of ATE.

**KPI:**  
- Posterior mean within ±10% of AIPW point estimate.
- Effective sample size > 100 for ATE parameter.

**Labels:** `enhancement`, `causal-estimator`, `bayesian`
