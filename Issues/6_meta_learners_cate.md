# Integrate Meta-Learners (T/X/R Learner) for CATE

**Objective:**  
Add meta-learners (T-learner, X-learner, R-learner) for heterogeneous treatment effect estimation.

**Tasks:**  
- Implement wrappers for T/X/R-learners using scikit-learn or lightgbm.
- Visualize CATE distribution.

**Benchmark:**  
- Dataset: `nhefs.csv`
- Treatment: `qsmk`; Outcome: `wt82_71`
- Covariates: age, sex, education

**KPI:**  
- Model R² > 0.2 on out-of-fold uplift.
- Visual: histogram or KDE of CATE.

**Labels:** `enhancement`, `CATE`, `uplift-modeling`
