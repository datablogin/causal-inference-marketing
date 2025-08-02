# Add Regression Discontinuity Design (RDD) Support

**Objective:**  
Implement RDD functionality for analyzing causal effects at a cutoff threshold.

**Tasks:**  
- Create generic `estimate_rdd()` function.
- Use income or age as forcing variable with an arbitrary cutoff (e.g. age 50).
- Plot outcome around the cutoff with fitted polynomials.

**Benchmark:**  
- Dataset: `nhefs.csv`
- Simulate a binary treatment assigned by age cutoff (e.g. age ≥ 50).
- Outcome: `wt82_71`

**KPI:**  
- RDD estimate matches simulated treatment effect ±10%.
- Visual shows clear discontinuity if simulated effect exists.

**Labels:** `enhancement`, `causal-estimator`, `RDD`
