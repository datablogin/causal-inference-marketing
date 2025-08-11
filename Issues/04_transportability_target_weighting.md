# Transportability & Target Population Weighting

**Objective**
Enable generalization of causal estimates to new audiences/markets.

**Scope / Acceptance**
- Implement covariate shift diagnostics and weighting.
- Support Targeted Maximum Transported Likelihood (TMTL).

**Tasks**
1. Diagnostics for covariate distribution shift between source and target populations.
2. Weight estimation for transportability (reweight source sample to match target).
3. Integrate with existing estimators for transported ATE.

**Test Data**
- Synthetic: generate source/target with known shift.
- Two-market marketing dataset.

**Tests**
- Weighting reduces covariate SMD below 0.1 in ≥ 90% of covariates.
- Transported ATE matches target-population ground truth within ±0.05.

**KPIs**
- Covariate shift detection power ≥ 80% for medium shifts.
