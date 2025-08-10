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

- Python 3.9, 3.10, 3.11, 3.12, or 3.13
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

This library includes the following causal inference estimators:

- **G-Computation**: Outcome regression approach
- **IPW (Inverse Probability Weighting)**: Propensity score weighting
- **AIPW (Augmented IPW)**: Doubly robust estimator combining G-computation and IPW
- **TMLE (Targeted Maximum Likelihood Estimation)**: Efficient doubly robust estimator
- **Doubly Robust ML**: Machine learning-based doubly robust estimation
- **Instrumental Variables (IV)**: Two-stage least squares for unmeasured confounding
- **G-Estimation**: Structural nested models
- **Propensity Score Methods**: Matching, stratification, and diagnostics
- **Time-Varying Treatment**: Sequential treatment strategies
- **Survival Analysis**: Causal survival models (IPW, G-computation, AIPW)

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

### API Service
```bash
# Start the causal inference API
make api

# Test the API
curl http://localhost:8000/health/
curl http://localhost:8000/api/v1/attribution/methods
```

### Docker Deployment
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up

# API available at http://localhost:8000
# Metrics at http://localhost:9090
# Prometheus at http://localhost:9091
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
