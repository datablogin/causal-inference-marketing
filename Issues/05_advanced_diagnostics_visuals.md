# Advanced Diagnostics & Visualization Tools

**Objective**  
Enhance diagnostics with richer visual outputs and reports.

**Scope / Acceptance**  
- Love plots for covariate balance.  
- Weight distribution plots & trimming suggestions.  
- Propensity score overlap visualization.  
- Residual plots for model-based estimators.

**Tasks**  
1. Implement love plot generator with pre/post adjustment SMD.  
2. Weight distribution diagnostics with tail metrics.  
3. PS overlap hist/density plots.  
4. Residual plots for G-comp, TMLE.

**Test Data**  
- Synthetic with known balance issues.  
- Real marketing data.

**Tests**  
- Post-adjustment SMD ≤ 0.1 for ≥ 90% covariates triggers "good" label.  
- Weight kurtosis warning triggers on heavy-tail synthetic.

**KPIs**  
- Diagnostic report generation < 2s on 50k rows.
