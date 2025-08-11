# Policy Learning & Off-Policy Evaluation

**Objective**  
Optimize targeting policies from CATE/uplift estimates under budget constraints, and evaluate counterfactual policy performance.

**Scope / Acceptance**  
- Policy optimization: treat top-k by uplift or budget constraint.  
- Off-policy evaluation: IPS, DR, DR-SN estimators.

**Tasks**  
1. Implement greedy and ILP-based budgeted policy selection.  
2. Implement IPS/DR/DR-SN off-policy estimators with variance estimates.  
3. Policy simulation API for scenario testing.  
4. Integration with existing CATE output formats.

**Test Data**  
- Synthetic heterogeneity DGP with cost-per-treatment.  
- Historical marketing logs.

**Tests**  
- Top-20% policy yields ≥ 1.3× uplift vs. random targeting in sims.  
- Off-policy estimates within ±10% of ground-truth simulated effect.

**KPIs**  
- Policy regret ≤ 20% vs. oracle under budget=20%.  
- OPE bias ≤ 0.05 in simulations.
