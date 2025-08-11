# Unified API & One-Call HTML Report

**Objective**
Provide sklearn-style API and a one-call HTML report summarizing all diagnostics, assumptions, and results.

**Scope / Acceptance**
- Base Estimator class with `fit`, `estimate_effect`, `predict_ite`.
- HTML report with methods used, assumptions, balance, overlap, effects, sensitivity.

**Tasks**
1. Define common estimator interface.
2. Implement adapters for existing methods.
3. HTML template with embedded plots and metrics.

**Test Data**
- Synthetic + real marketing dataset.

**Tests**
- Interface works interchangeably across estimators.
- HTML report opens with embedded plots and all required sections.

**KPIs**
- ≥ 90% test coverage for API methods.
- Report renders < 5s for 10k-row analysis.
