# Causal Inference Tools for Marketing Applications

A comprehensive library for applying causal inference methods to marketing analytics, structured for compatibility with the analytics-backend-monorepo ecosystem.

## Overview

This library provides production-ready implementations of causal inference methods specifically designed for marketing use cases. Built with a **monorepo-compatible architecture**, it can operate as a standalone project or seamlessly integrate into the existing analytics-backend-monorepo infrastructure.

## Key Features

- **Multiple Estimators**: G-computation, IPW, AIPW, TMLE, and Doubly Robust ML
- **Survival Analysis**: Time-to-event analysis with causal methods
- **Time-Varying Treatments**: Handle treatments that change over time
- **Instrumental Variables**: IV estimation for unmeasured confounding
- **G-Estimation**: Structural nested models for complex treatments
- **Propensity Score Methods**: Matching, stratification, and weighting
- **Production-Ready**: FastAPI services, observability, and Docker deployment

## Architecture

### Monorepo-Compatible Structure
```
causal-inference-marketing/
├── libs/
│   ├── causal_inference/         # Core library (ready for monorepo)
│   │   ├── causal_inference/
│   │   │   ├── core/             # Base classes and interfaces
│   │   │   ├── data/             # Data models and validation
│   │   │   ├── estimators/       # All causal estimators
│   │   │   └── utils/            # Helper functions
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── causal_inference_python_code/  # "What If" book examples
├── services/
│   ├── causal_api/               # FastAPI service
│   └── causal_processor/         # Background processing (planned)
├── shared/                       # Extracted monorepo patterns
│   ├── config/                   # Configuration management
│   ├── database/                 # Database abstractions
│   └── observability/            # Metrics and logging
└── docker/                       # Service containerization
```

### Integration Ready
This structure mirrors the analytics-backend-monorepo patterns, enabling seamless integration. See [INTEGRATION.md](INTEGRATION.md) for detailed integration guide.

## Installation

### Requirements

- Python 3.11, 3.12, or 3.13
- uv package manager (optional but recommended)

### Development Installation

```bash
git clone https://github.com/datablogin/causal-inference-marketing.git
cd causal-inference-marketing
make install-dev
```

### Using Make Commands
```bash
make help           # Show available commands
make install-dev    # Install with development dependencies
make ci             # Run full CI pipeline (lint, typecheck, test)
make api            # Start FastAPI development server
```

## Quick Start

### Available Estimators

This library includes comprehensive causal inference estimators across multiple categories:

#### Core Estimators
- **G-Computation**: Outcome regression approach for standardization
- **IPW (Inverse Probability Weighting)**: Propensity score weighting with stabilization
- **AIPW (Augmented IPW)**: Doubly robust estimator combining G-computation and IPW
- **TMLE (Targeted Maximum Likelihood Estimation)**: Efficient doubly robust estimator
- **Doubly Robust ML**: Machine learning-based doubly robust estimation with cross-fitting

#### Advanced Machine Learning Methods
- **Causal Forests**: Honest random forests for heterogeneous treatment effects
- **Meta-Learners**: T-learner, X-learner, R-learner for conditional average treatment effects (CATE)
- **Bayesian Causal Inference**: Posterior inference with uncertainty quantification
- **Subgroup Discovery**: Automated identification of treatment effect heterogeneity

#### Design-Based Methods
- **Instrumental Variables (IV)**: Two-stage least squares with weak instrument diagnostics
- **Regression Discontinuity (RDD)**: Sharp and fuzzy discontinuity designs
- **Difference-in-Differences (DID)**: Panel data estimation with staggered adoption
- **Synthetic Control**: Comparative case study method for policy evaluation

#### Specialized Applications
- **G-Estimation**: Structural nested models for optimal treatment regimes
- **Mediation Analysis**: Direct and indirect effect decomposition
- **Propensity Score Methods**: Matching, stratification, and covariate balance diagnostics
- **Time-Varying Treatment**: Sequential treatment strategies with g-methods
- **Survival Analysis**: Causal survival models (IPW, G-computation, AIPW) with RMST estimation

#### Enhanced Features
- **Parallel Cross-Fitting**: 2x+ speedup with configurable backends
- **Continuous Treatments**: Dose-response estimation and marginal treatment effects
- **Multi-Category Treatments**: K-way treatment comparisons
- **Comprehensive Diagnostics**: Orthogonality checks, residual analysis, model validation
- **Synthetic Data Generation**: Validation pipeline with known ground truth

### Library Usage

#### Basic Example - Average Treatment Effect
```python
from causal_inference.estimators import GComputationEstimator, IPWEstimator, AIPWEstimator
from causal_inference.data import TreatmentData, OutcomeData, CovariateData
import pandas as pd
import numpy as np

# Load your data
df = pd.DataFrame({
    'treatment': np.random.binomial(1, 0.5, 1000),
    'outcome': np.random.normal(0, 1, 1000),
    'age': np.random.normal(50, 10, 1000),
    'income': np.random.normal(60000, 20000, 1000)
})

# Prepare data objects
treatment = TreatmentData(data=df['treatment'])
outcome = OutcomeData(data=df['outcome'])
covariates = CovariateData(data=df[['age', 'income']])

# G-Computation Estimator
g_comp = GComputationEstimator()
g_comp.fit(treatment, outcome, covariates)
g_result = g_comp.estimate_ate()
print(f"G-Computation ATE: {g_result.ate:.3f} (95% CI: {g_result.confidence_interval})")

# IPW Estimator
ipw = IPWEstimator()
ipw.fit(treatment, outcome, covariates)
ipw_result = ipw.estimate_ate()
print(f"IPW ATE: {ipw_result.ate:.3f} (95% CI: {ipw_result.confidence_interval})")

# AIPW (Doubly Robust) Estimator
aipw = AIPWEstimator()
aipw.fit(treatment, outcome, covariates)
aipw_result = aipw.estimate_ate()
print(f"AIPW ATE: {aipw_result.ate:.3f} (95% CI: {aipw_result.confidence_interval})")
```

#### Advanced Example - Propensity Score Analysis
```python
from causal_inference.estimators import PropensityScoreEstimator

# Propensity Score Matching
ps_estimator = PropensityScoreEstimator(
    method='matching',
    n_neighbors=1,
    caliper=0.1
)
ps_estimator.fit(treatment, outcome, covariates)
ps_result = ps_estimator.estimate_ate()

# Get diagnostics
diagnostics = ps_estimator.get_diagnostics()
print(f"Matched ATE: {ps_result.ate:.3f}")
print(f"Balance improvement: {diagnostics['balance_improvement']}")
```

#### Time-Varying Treatment Example
```python
from causal_inference.estimators import TimeVaryingEstimator

# Define treatment strategies to compare
always_treat = lambda history: 1
never_treat = lambda history: 0

# Fit time-varying model
tv_estimator = TimeVaryingEstimator(
    outcome_model_type='linear',
    treatment_model_type='logistic'
)
tv_estimator.fit(treatment_history, outcome_history, time_varying_covariates)

# Compare strategies
comparison = tv_estimator.compare_strategies(always_treat, never_treat)
print(f"Strategy comparison: {comparison.effect:.3f} (95% CI: {comparison.confidence_interval})")
```

#### Instrumental Variables Example
```python
from causal_inference.estimators import IVEstimator
from causal_inference.data import InstrumentData

# Prepare instrument data
instrument = InstrumentData(data=df['instrument'])

# Two-Stage Least Squares
iv_estimator = IVEstimator(method='2sls')
iv_estimator.fit(treatment, outcome, covariates, instrument)
iv_result = iv_estimator.estimate_ate()

# Get first-stage diagnostics
diagnostics = iv_estimator.get_diagnostics()
print(f"IV ATE: {iv_result.ate:.3f}")
print(f"First-stage F-statistic: {diagnostics['first_stage_f']:.2f}")
print(f"Wu-Hausman test p-value: {diagnostics['wu_hausman_p']:.3f}")
```

#### Machine Learning with Doubly Robust Estimation
```python
from causal_inference.estimators import DoublyRobustMLEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Use any sklearn-compatible models
dr_ml = DoublyRobustMLEstimator(
    outcome_model=RandomForestRegressor(n_estimators=100),
    treatment_model=RandomForestClassifier(n_estimators=100),
    n_folds=5
)
dr_ml.fit(treatment, outcome, covariates)
dr_result = dr_ml.estimate_ate()
print(f"DR-ML ATE: {dr_result.ate:.3f} (95% CI: {dr_result.confidence_interval})")
```

#### TMLE (Targeted Maximum Likelihood) Example
```python
from causal_inference.estimators import TMLEEstimator

# TMLE with Super Learner
tmle = TMLEEstimator(
    outcome_model_type='super_learner',
    treatment_model_type='super_learner',
    cv_folds=10
)
tmle.fit(treatment, outcome, covariates)
tmle_result = tmle.estimate_ate()

# Get influence curve for inference
influence_curve = tmle.get_influence_curve()
print(f"TMLE ATE: {tmle_result.ate:.3f} (95% CI: {tmle_result.confidence_interval})")
print(f"Efficient variance: {np.var(influence_curve):.4f}")
```

#### G-Estimation Example
```python
from causal_inference.estimators import GEstimationEstimator

# Structural nested mean model
g_est = GEstimationEstimator(
    model_type='linear',
    link_function='identity'
)
g_est.fit(treatment, outcome, covariates)
g_result = g_est.estimate_ate()

# Get optimal treatment regime
optimal_regime = g_est.get_optimal_regime(covariates)
print(f"G-estimation ATE: {g_result.ate:.3f}")
print(f"Proportion who would benefit from treatment: {optimal_regime.mean():.2%}")
```

#### Survival Analysis Example
```python
from causal_inference.estimators import SurvivalIPWEstimator
from causal_inference.data import SurvivalData

# Prepare survival data
survival_data = SurvivalData(
    time=df['time_to_event'],
    event=df['event_indicator']
)

# Survival IPW
survival_ipw = SurvivalIPWEstimator(
    time_points=[30, 60, 90, 180, 365]  # Days
)
survival_ipw.fit(treatment, survival_data, covariates)

# Get survival curves
survival_curves = survival_ipw.get_survival_curves()
rmst_diff = survival_ipw.estimate_rmst_difference(tau=365)
print(f"RMST difference at 1 year: {rmst_diff.effect:.1f} days (95% CI: {rmst_diff.confidence_interval})")
```

#### Causal Forests for Heterogeneous Treatment Effects
```python
from causal_inference.estimators import CausalForestEstimator

# Honest causal forest for individualized treatment effects
cf_estimator = CausalForestEstimator(
    n_trees=1000,
    min_samples_leaf=10,
    honesty=True,
    honesty_fraction=0.5
)
cf_estimator.fit(treatment, outcome, covariates)

# Get individualized treatment effects
individual_effects = cf_estimator.predict(covariates)
print(f"Mean CATE: {individual_effects.mean():.3f}")
print(f"CATE std: {individual_effects.std():.3f}")

# Variable importance for effect modifiers
importance = cf_estimator.get_feature_importance()
print(f"Top effect modifier: {importance.index[0]} (importance: {importance.iloc[0]:.3f})")

# Confidence intervals for individual effects
ci_lower, ci_upper = cf_estimator.predict_confidence_intervals(covariates, alpha=0.05)
print(f"Individual CIs computed for {len(ci_lower)} observations")
```

#### Meta-Learners for CATE Estimation
```python
from causal_inference.estimators import TLearner, XLearner, RLearner
from sklearn.ensemble import RandomForestRegressor

# T-Learner: Separate models for treated and control
t_learner = TLearner(
    base_learner=RandomForestRegressor(n_estimators=100)
)
t_learner.fit(treatment, outcome, covariates)
t_cate = t_learner.predict_cate(covariates)

# X-Learner: More efficient for imbalanced treatments
x_learner = XLearner(
    outcome_learner=RandomForestRegressor(n_estimators=100),
    propensity_learner=RandomForestRegressor(n_estimators=100)
)
x_learner.fit(treatment, outcome, covariates)
x_cate = x_learner.predict_cate(covariates)

# R-Learner: Residual-based approach
r_learner = RLearner(
    outcome_learner=RandomForestRegressor(n_estimators=100),
    propensity_learner=RandomForestRegressor(n_estimators=100),
    treatment_learner=RandomForestRegressor(n_estimators=100)
)
r_learner.fit(treatment, outcome, covariates)
r_cate = r_learner.predict_cate(covariates)

print(f"T-Learner ATE: {t_cate.mean():.3f}")
print(f"X-Learner ATE: {x_cate.mean():.3f}")
print(f"R-Learner ATE: {r_cate.mean():.3f}")
```

#### Mediation Analysis
```python
from causal_inference.estimators import MediationEstimator

# Prepare mediator data
mediator_data = CovariateData(data=df[['mediator_variable']])

# Mediation analysis with bootstrap CI
mediation = MediationEstimator(
    outcome_model_type='linear',
    mediator_model_type='linear',
    n_bootstrap=1000
)
mediation.fit(treatment, outcome, covariates, mediator_data)

# Decompose total effect
results = mediation.estimate_effects()
print(f"Total Effect: {results.total_effect:.3f} (95% CI: {results.total_effect_ci})")
print(f"Direct Effect: {results.direct_effect:.3f} (95% CI: {results.direct_effect_ci})")
print(f"Indirect Effect: {results.indirect_effect:.3f} (95% CI: {results.indirect_effect_ci})")
print(f"Proportion Mediated: {results.proportion_mediated:.2%}")
```

#### Difference-in-Differences Analysis
```python
from causal_inference.estimators import DifferenceInDifferencesEstimator

# Panel data with pre/post periods and treatment/control groups
did_estimator = DifferenceInDifferencesEstimator(
    time_variable='period',
    unit_variable='unit_id',
    cluster_se=True  # Cluster standard errors by unit
)
did_estimator.fit(treatment, outcome, covariates)
did_result = did_estimator.estimate_ate()

# Parallel trends test
parallel_trends = did_estimator.test_parallel_trends()
print(f"DID ATE: {did_result.ate:.3f} (95% CI: {did_result.confidence_interval})")
print(f"Parallel trends test p-value: {parallel_trends.p_value:.3f}")

# Event study for dynamic effects
if parallel_trends.p_value > 0.05:  # If parallel trends hold
    event_study = did_estimator.event_study(leads=3, lags=5)
    print(f"Pre-treatment effects: {event_study.pre_effects}")
    print(f"Post-treatment effects: {event_study.post_effects}")
```

#### Synthetic Control Method
```python
from causal_inference.estimators import SyntheticControlEstimator

# Comparative case study for policy evaluation
sc_estimator = SyntheticControlEstimator(
    time_variable='year',
    unit_variable='state',
    treatment_period=2010
)
sc_estimator.fit(treatment, outcome, covariates)
sc_result = sc_estimator.estimate_ate()

# Get synthetic control weights
weights = sc_estimator.get_synthetic_weights()
print(f"Synthetic Control ATE: {sc_result.ate:.3f}")
print(f"Top donor states: {weights.head(3)}")

# Placebo tests for validation
placebo_results = sc_estimator.placebo_tests()
print(f"Placebo test p-value: {placebo_results.p_value:.3f}")
```

#### Regression Discontinuity Design
```python
from causal_inference.estimators import RegressionDiscontinuityEstimator

# Sharp RDD with running variable
rdd_estimator = RegressionDiscontinuityEstimator(
    cutoff=0.0,
    bandwidth='optimal',  # Imbens-Kalyanaraman optimal bandwidth
    polynomial_order=1
)
rdd_estimator.fit(treatment, outcome, covariates, running_variable=df['score'])
rdd_result = rdd_estimator.estimate_ate()

# Diagnostics
diagnostics = rdd_estimator.get_diagnostics()
print(f"RDD Local ATE: {rdd_result.ate:.3f} (95% CI: {rdd_result.confidence_interval})")
print(f"Optimal bandwidth: {diagnostics['bandwidth']:.3f}")
print(f"Density test p-value: {diagnostics['density_test_p']:.3f}")

# Robustness checks
robustness = rdd_estimator.robustness_checks(
    bandwidths=[0.5, 1.0, 1.5],
    polynomial_orders=[1, 2, 3]
)
print(f"Robustness range: {robustness.effect_range}")
```

#### Bayesian Causal Inference
```python
from causal_inference.estimators import BayesianCausalEstimator

# Bayesian estimation with uncertainty quantification
bayesian_estimator = BayesianCausalEstimator(
    prior_type='weakly_informative',
    n_samples=2000,
    n_chains=4
)
bayesian_estimator.fit(treatment, outcome, covariates)

# Posterior inference
posterior = bayesian_estimator.get_posterior_samples()
ate_samples = posterior['ate']

print(f"Posterior mean ATE: {ate_samples.mean():.3f}")
print(f"95% Credible Interval: [{np.percentile(ate_samples, 2.5):.3f}, {np.percentile(ate_samples, 97.5):.3f}]")
print(f"P(ATE > 0): {(ate_samples > 0).mean():.3f}")

# Model comparison with Bayes factors
model_comparison = bayesian_estimator.compare_models(['linear', 'nonlinear'])
print(f"Best model: {model_comparison.best_model}")
```

#### Enhanced Doubly Robust ML with Diagnostics
```python
from causal_inference.estimators import DoublyRobustMLEstimator
from causal_inference.testing import generate_synthetic_dml_data

# Generate synthetic data for validation
X_syn, D_syn, Y_syn, true_ate = generate_synthetic_dml_data(
    n=2000,
    n_features=10,
    true_ate=1.5,
    confounding_strength=1.0
)

# Enhanced DML with comprehensive diagnostics
dml_estimator = DoublyRobustMLEstimator(
    outcome_learner=RandomForestRegressor(n_estimators=100),
    propensity_learner=RandomForestClassifier(n_estimators=100),
    n_folds=5,
    moment_function='aipw',  # or 'orthogonal', 'partialling_out'
    performance_config={
        'n_jobs': -1,  # Parallel processing
        'parallel_backend': 'threading',
        'enable_caching': True
    }
)
dml_estimator.fit(TreatmentData(D_syn), OutcomeData(Y_syn), CovariateData(X_syn))
dml_result = dml_estimator.estimate_ate()

# Comprehensive diagnostics
diagnostics = dml_estimator.get_comprehensive_diagnostics()
print(f"DML ATE: {dml_result.ate:.3f} (True: {true_ate:.3f})")
print(f"Orthogonality check: r={diagnostics['orthogonality_correlation']:.4f}")
print(f"Outcome model R²: {diagnostics['outcome_model_r2']:.3f}")
print(f"Propensity model AUC: {diagnostics['propensity_model_auc']:.3f}")

# Performance profiling
profiling = dml_estimator.profile_performance()
print(f"Runtime: {profiling['total_time']:.2f}s")
print(f"Memory peak: {profiling['memory_peak_mb']:.1f}MB")
```

## Method Selection Guide

### Choosing the Right Estimator

#### By Study Design and Data Structure

**Cross-sectional Data (Single Time Point)**
- **G-Computation**: When you have good outcome model specification
- **IPW**: When you have good propensity model specification
- **AIPW/TMLE**: When you want double robustness protection
- **Causal Forests**: For heterogeneous treatment effects with high-dimensional data
- **Meta-Learners**: For CATE estimation with flexible ML models

**Panel Data (Multiple Time Points)**
- **Difference-in-Differences**: For policy evaluation with parallel trends
- **Synthetic Control**: For comparative case studies with few treated units
- **Time-Varying Treatment**: For sequential treatment strategies

**Cross-sectional with Design Elements**
- **Instrumental Variables**: When you have unmeasured confounding but valid instruments
- **Regression Discontinuity**: When treatment assignment follows a cutoff rule
- **Propensity Score Methods**: For covariate balance in observational studies

#### By Treatment Type

**Binary Treatment (0/1)**
- All estimators support binary treatments
- Start with **AIPW** for double robustness
- Use **Causal Forests** for heterogeneity exploration

**Continuous Treatment (doses, amounts)**
- **Enhanced DML**: Supports dose-response estimation
- **Bayesian Methods**: For uncertainty quantification with continuous treatments
- **G-Computation**: With flexible outcome models

**Multi-Category Treatment (A/B/C)**
- **Enhanced DML**: Supports K-way comparisons
- **Meta-Learners**: Can handle multiple treatment arms
- **Bayesian Methods**: For complex treatment structures

#### By Sample Size and Computational Resources

**Small Samples (n < 1,000)**
- **G-Computation**: Simple and interpretable
- **Bayesian Methods**: Incorporates prior information
- **Bootstrap-based methods**: For robust inference

**Medium Samples (1,000 - 10,000)**
- **AIPW/TMLE**: Efficient and robust
- **Propensity Score Methods**: Good balance diagnostics
- **Meta-Learners**: Flexible ML approaches

**Large Samples (n > 10,000)**
- **Doubly Robust ML**: Parallel processing and caching
- **Causal Forests**: Scales well with sample size
- **Enhanced DML**: Memory-efficient processing

#### By Research Question

**Average Treatment Effect (ATE)**
- **AIPW, TMLE**: Most efficient for population-level effects
- **DML**: For high-dimensional confounders

**Heterogeneous Treatment Effects (HTE)**
- **Causal Forests**: Individual-level predictions
- **Meta-Learners**: Different approaches for different scenarios
- **Subgroup Discovery**: Automatic subgroup identification

**Optimal Treatment Assignment**
- **G-Estimation**: Structural nested models for optimal regimes
- **Causal Forests**: Individual treatment recommendations
- **Bayesian Methods**: Incorporating decision costs

**Mechanism Understanding**
- **Mediation Analysis**: Direct vs. indirect effects
- **G-Estimation**: Structural relationships
- **Bayesian Methods**: Full posterior uncertainty

### Performance and Scalability

#### Runtime Expectations (1,000 samples)
- **G-Computation**: < 1s
- **IPW**: < 1s
- **AIPW**: < 2s
- **TMLE**: < 5s
- **DML (parallel)**: < 10s
- **Causal Forests**: < 30s
- **Bayesian Methods**: 1-5 minutes

#### Memory Requirements
- **Basic Estimators**: < 100MB
- **DML with caching**: < 1GB
- **Causal Forests**: < 500MB
- **Bayesian MCMC**: < 2GB

### API Service

#### Development Server
```bash
# Start the causal inference API
make api

# Test the API
curl http://localhost:8000/health/
curl http://localhost:8000/api/v1/attribution/methods

# Interactive API documentation
open http://localhost:8000/docs
```

#### Production Configuration
```bash
# Environment variables for production
export CAUSAL_API_HOST=0.0.0.0
export CAUSAL_API_PORT=8000
export CAUSAL_API_WORKERS=4
export CAUSAL_LOG_LEVEL=INFO
export CAUSAL_CACHE_SIZE=1000

# Start with production settings
uvicorn services.causal_api.main:app \
  --host $CAUSAL_API_HOST \
  --port $CAUSAL_API_PORT \
  --workers $CAUSAL_API_WORKERS
```

#### API Endpoints
```python
# Example API usage
import requests

# Estimate treatment effect via API
response = requests.post("http://localhost:8000/api/v1/estimate", json={
    "method": "aipw",
    "treatment": [1, 0, 1, 0, 1],
    "outcome": [2.1, 1.5, 2.3, 1.8, 2.0],
    "covariates": [[1, 2], [2, 1], [1, 1], [2, 2], [1, 2]],
    "bootstrap_samples": 1000
})
result = response.json()
print(f"ATE: {result['ate']:.3f} (95% CI: {result['confidence_interval']})")
```

### Docker Deployment

#### Development Environment
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up

# API available at http://localhost:8000
# Metrics at http://localhost:9090
# Prometheus at http://localhost:9091
```

#### Production Deployment
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  causal_api:
    image: causal-inference:latest
    environment:
      - CAUSAL_API_WORKERS=8
      - CAUSAL_CACHE_SIZE=5000
      - CAUSAL_LOG_LEVEL=WARNING
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2GB
          cpus: '2'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: causal-api
spec:
  replicas: 5
  selector:
    matchLabels:
      app: causal-api
  template:
    spec:
      containers:
      - name: api
        image: causal-inference:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: CAUSAL_API_WORKERS
          value: "4"
```

#### Performance Monitoring
```python
# Monitoring integration
from causal_inference.monitoring import CausalMetrics

# Track estimation performance
metrics = CausalMetrics()
with metrics.track_estimation("aipw"):
    result = aipw_estimator.estimate_ate()

# Export to Prometheus
metrics.export_prometheus(port=9090)

# Key metrics tracked:
# - estimation_duration_seconds
# - model_fit_duration_seconds
# - memory_usage_bytes
# - estimation_errors_total
# - api_requests_total
```

## Development

### Setup Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/datablogin/causal-inference-marketing.git
   cd causal-inference-marketing
   ```

2. Install development dependencies:
   ```bash
   make install-dev
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific library tests
make ci-causal-inference

# Run specific test modules
pytest libs/causal_inference/tests/
pytest -m integration  # Integration tests only
```

### Code Quality

This project uses several tools to maintain code quality:

- **Ruff**: Fast Python linter and formatter (replaces Black, isort, flake8)
- **MyPy**: Static type checking with strict mode
- **Pytest**: Testing framework with async support

Run all quality checks:
```bash
# Run full CI pipeline
make ci

# Individual commands
make lint       # Run ruff linting
make typecheck  # Run mypy type checking
make test       # Run pytest
make format     # Auto-format code with ruff
```

## Documentation

Full documentation is available at [project documentation](https://github.com/datablogin/causal-inference-marketing#readme).

### Building Documentation Locally

```bash
pip install -e ".[docs]"
cd docs/
make html
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{causal_inference_marketing,
  title = {Causal Inference Tools for Marketing Applications},
  author = {Causal Inference Marketing Team},
  url = {https://github.com/datablogin/causal-inference-marketing},
  version = {0.1.0},
  year = {2024}
}
```

## References

- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC.
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference*. Cambridge University Press.

## Support

- **Issues**: [GitHub Issues](https://github.com/datablogin/causal-inference-marketing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/datablogin/causal-inference-marketing/discussions)

---

**Status**: 🚧 Under active development - Phase 1 (Foundation)# Force CI trigger
