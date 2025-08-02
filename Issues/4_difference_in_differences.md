# Support Difference-in-Differences (DID) Estimation

**Objective:**  
Enable two-period and staggered DID for observational campaign data.

**Tasks:**  
- Implement a `difference_in_differences()` estimator.
- Accept treatment indicator, time, and group variables.

**Benchmark:**  
- Dataset: `nhefs.csv`
- Simulate treatment group and time-based exposure.
- Outcome: `wt82_71`

**KPI:**  
- Estimated ATT within ±10% of simulated truth.
- Parallel trends plot included.

**Labels:** `enhancement`, `causal-estimator`, `DID`
