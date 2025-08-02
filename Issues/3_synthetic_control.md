# Implement Synthetic Control Estimator

**Objective:**  
Create synthetic control estimator for single-unit or regional marketing interventions.

**Tasks:**  
- Implement core synthetic control logic using optimization over weights.
- Include visualization of pre/post outcome trajectory vs synthetic.

**Benchmark:**  
- Use `nhefs.csv`, simulate group-level campaign introduced at a time-point.
- Compare post-campaign outcome for treated vs. synthetic control.

**KPI:**  
- RMSPE (pre-intervention) < 0.1
- Post-treatment gap > 2x RMSPE (significant effect)

**Labels:** `enhancement`, `causal-estimator`, `synthetic-control`
