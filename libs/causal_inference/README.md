# causal-inference-marketing

Causal inference library for marketing analytics -- AIPW, TMLE, G-computation, IPW, causal discovery, and 30+ estimators.

## Installation

```bash
pip install causal-inference-marketing
```

With optional extras:

```bash
pip install causal-inference-marketing[bayesian]       # PyMC + arviz for Bayesian estimation
pip install causal-inference-marketing[ml]             # LightGBM, SHAP, joblib
pip install causal-inference-marketing[optimization]   # CVXPY for policy optimization
pip install causal-inference-marketing[visualization]  # seaborn for diagnostic plots
pip install causal-inference-marketing[all]            # Everything
```

> **Note:** The Python import name is `causal_inference`, not `causal_inference_marketing`.
> Do not install this package alongside the unrelated `causal-inference` PyPI package --
> the two share the same import namespace and will conflict.

## Quick start

### AIPW (doubly robust) estimator

```python
import numpy as np
import pandas as pd
from causal_inference.core.base import TreatmentData, OutcomeData, CovariateData
from causal_inference.estimators.aipw import AIPWEstimator

# Prepare data
treatment = TreatmentData(values=np.array([0, 1, 0, 1, 1, 0, 1, 0]),
                          treatment_type="binary")
outcome = OutcomeData(values=np.array([2.1, 5.3, 1.9, 4.8, 5.1, 2.3, 4.7, 2.0]),
                      outcome_type="continuous")
covariates = CovariateData(
    values=pd.DataFrame({"age": [25, 30, 28, 35, 40, 22, 33, 27],
                          "income": [50, 80, 55, 90, 85, 45, 75, 60]}),
    names=["age", "income"],
)

estimator = AIPWEstimator(cross_fitting=True, n_folds=2, random_state=42)
estimator.fit(treatment, outcome, covariates)
effect = estimator.estimate_ate()

print(f"ATE: {effect.ate:.3f}")
print(f"95% CI: [{effect.ate_ci_lower:.3f}, {effect.ate_ci_upper:.3f}]")
```

### IPW estimator

```python
import numpy as np
import pandas as pd
from causal_inference.core.base import TreatmentData, OutcomeData, CovariateData
from causal_inference.estimators.ipw import IPWEstimator

treatment = TreatmentData(values=np.array([0, 1, 0, 1, 1, 0, 1, 0]),
                          treatment_type="binary")
outcome = OutcomeData(values=np.array([2.1, 5.3, 1.9, 4.8, 5.1, 2.3, 4.7, 2.0]),
                      outcome_type="continuous")
covariates = CovariateData(
    values=pd.DataFrame({"age": [25, 30, 28, 35, 40, 22, 33, 27],
                          "income": [50, 80, 55, 90, 85, 45, 75, 60]}),
    names=["age", "income"],
)


estimator = IPWEstimator(
    propensity_model_type="logistic",
    stabilized_weights=True,
    bootstrap_samples=500,
    random_state=42,
)
estimator.fit(treatment, outcome, covariates)
effect = estimator.estimate_ate()

print(f"ATE: {effect.ate:.3f}")
```

### Causal discovery with the PC algorithm

```python
import numpy as np
import pandas as pd
from causal_inference.discovery import PCAlgorithm

# Observational data with columns for each variable
data = pd.DataFrame({
    "ad_spend": np.random.normal(100, 20, 500),
    "impressions": np.random.normal(1000, 200, 500),
    "clicks": np.random.normal(50, 10, 500),
    "conversions": np.random.normal(5, 2, 500),
})

pc = PCAlgorithm(independence_test="pearson", alpha=0.05)
result = pc.discover(data)

print("Learned DAG adjacency matrix:")
print(result.dag.adjacency_matrix)
print("Variables:", result.dag.variable_names)
```

### Discovery-to-estimation pipeline

```python
import numpy as np
import pandas as pd
from causal_inference.discovery import PCAlgorithm, DiscoveryEstimatorPipeline
from causal_inference.estimators.aipw import AIPWEstimator

data = pd.DataFrame({
    "ad_spend": np.random.normal(100, 20, 500),
    "impressions": np.random.normal(1000, 200, 500),
    "clicks": np.random.normal(50, 10, 500),
    "conversions": np.random.normal(5, 2, 500),
})

pipeline = DiscoveryEstimatorPipeline(
    discovery_algorithm=PCAlgorithm(alpha=0.05),
    estimator_class=AIPWEstimator,
)
pipeline_result = pipeline.run(
    data=data,
    treatment_col="ad_spend",
    outcome_col="conversions",
)
```

## Features

### Estimators (30+)

| Category | Estimators |
|----------|-----------|
| **Core** | G-computation, IPW, AIPW, TMLE, Doubly Robust ML |
| **Instrumental variables** | IV / 2SLS |
| **Heterogeneous effects** | S-Learner, T-Learner, X-Learner, R-Learner, Causal Forest |
| **Quasi-experimental** | Difference-in-Differences, Regression Discontinuity, Synthetic Control |
| **Time-varying** | Time-varying treatment estimator, G-estimation |
| **Mediation** | Natural direct/indirect effects |
| **Survival** | Survival IPW, Survival G-computation, Survival AIPW |
| **Subgroup discovery** | Virtual Twins, SIDES, Optimal Policy Tree |
| **Bayesian** | Bayesian causal estimation (optional, requires `pymc`) |

### Causal discovery

- **PC algorithm** -- constraint-based, multiple independence tests (Pearson, Spearman, mutual information)
- **FCI** -- handles latent confounders
- **GES** -- Greedy Equivalence Search (score-based)
- **NOTEARS** -- continuous optimization approach to structure learning
- **Benchmarking suite** -- compare algorithms on synthetic data with known ground truth

### Diagnostics and sensitivity

- Propensity score overlap diagnostics
- Covariate balance checking
- Sensitivity analysis (Rosenbaum bounds, Oster's delta, E-values, placebo tests)
- Falsification tests
- Assumption validation suite

### Additional modules

- **Policy learning** -- off-policy evaluation and optimization
- **Interference** -- spillover estimation and network causal inference
- **Transportability** -- generalize effects across populations
- **Target trial emulation** -- emulate randomized trials from observational data
- **Super Learner** -- ensemble ML for nuisance parameter estimation
- **Visualization** -- publication-ready plots for causal analysis

## Python version support

Python 3.11, 3.12, and 3.13.

## License

MIT
