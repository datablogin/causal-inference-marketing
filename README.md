# Causal Inference Tools for Marketing Applications

A comprehensive Python library for applying causal inference methods to marketing analytics, including attribution modeling, incrementality testing, and media mix modeling.

## Overview

This library provides production-ready implementations of causal inference methods specifically designed for marketing use cases. Built on the theoretical foundation of "Causal Inference: What If" by Hernán and Robins, it offers both academic rigor and practical applicability for marketing practitioners.

## Key Features

- **Attribution Modeling**: Multi-touch attribution with proper causal identification
- **Incrementality Testing**: Geo-based holdout experiments and difference-in-differences
- **Media Mix Modeling**: Causal decomposition of marketing channel effects
- **Experimental Design**: A/B testing with network effects and interference
- **Customer Analytics**: Causal impact analysis for customer lifetime value
- **Budget Optimization**: Causal-based marketing budget allocation

## Installation

### From PyPI (recommended)

```bash
pip install causal-inference-marketing
```

### Development Installation

```bash
git clone https://github.com/datablogin/causal-inference-marketing.git
cd causal-inference-marketing
pip install -e ".[dev]"
```

## Quick Start

```python
from causal_inference_marketing import Attribution, IncrementalityTest

# Multi-touch attribution analysis
attribution = Attribution(method="doubly_robust")
results = attribution.fit(data, treatment_col="channel", outcome_col="conversion")

# Incrementality testing
increment_test = IncrementalityTest(method="geo_holdout")
lift = increment_test.estimate_lift(data, treatment_periods, control_periods)
```

## Project Structure

```
src/causal_inference_marketing/
├── core/              # Core causal inference estimators
├── data/              # Data preprocessing and simulation utilities
└── utils/             # Common utilities and helper functions
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
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=causal_inference_marketing

# Run specific test modules
pytest tests/test_core/
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Pytest**: Testing framework

Run all quality checks:
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/

# Run tests
pytest
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

**Status**: 🚧 Under active development - Phase 1 (Foundation)