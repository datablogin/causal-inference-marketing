# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Core development workflow:**
- `make install-dev` - Install with development dependencies (dev, test, ml optional dependencies)
- `make ci` - Run complete CI pipeline (lint, typecheck, test) - use this before commits
- `make lint` - Run ruff linting on libs/, services/, shared/
- `make typecheck` - Run mypy type checking on libs/causal_inference/ and shared/
- `make test` - Run pytest on libs/ and shared/ directories
- `make test-cov` - Run tests with coverage reporting
- `make format` - Auto-format code with ruff and apply fixes

**API development:**
- `make api` - Start FastAPI development server at localhost:8000
- Uses uvicorn with auto-reload for development

**Testing specific libraries:**
- `make ci-causal-inference` - Run tests, mypy, and ruff specifically for causal_inference library
- `pytest libs/causal_inference/tests/` - Run tests for specific library
- `pytest -m integration` - Run integration tests only

## Architecture Overview

This is a **monorepo-compatible causal inference library** designed for marketing applications with a multi-layered architecture:

### Core Structure
- **libs/causal_inference/** - Core causal inference library with estimators (G-computation, etc.)
- **services/** - FastAPI microservices (causal_api with health and attribution endpoints)  
- **shared/** - Common infrastructure (config, database, observability)
- **libs/causal_inference_python_code/** - Jupyter notebooks for "Causal Inference: What If" book examples

### Key Design Patterns

**Base Estimator Architecture:**
- All estimators inherit from `BaseEstimator` in `libs/causal_inference/causal_inference/core/base.py`
- Uses Pydantic models for data validation (`TreatmentData`, `OutcomeData`, `CovariateData`)
- Standardized `CausalEffect` result format across all estimators
- Abstract methods: `_fit_implementation()` and `_estimate_ate_implementation()`

**Data Models (Pydantic-based):**
- `TreatmentData` - Supports binary, categorical, continuous treatments with validation
- `OutcomeData` - Handles continuous, binary, count outcomes  
- `CovariateData` - Manages covariate/confounder variables
- `CausalEffect` - Standardized result object with ATE, confidence intervals, diagnostics

**Configuration Management:**
- `shared/config/` uses Pydantic-based configuration with environment-specific validation
- `CausalInferenceConfig` extends `BaseConfiguration` with causal-specific settings
- Configuration validation includes production-specific checks

### Current Estimators
- **G-computation** (`libs/causal_inference/causal_inference/estimators/g_computation.py`) - Standardization method with sklearn model integration, bootstrap confidence intervals
- **IPW** (`libs/causal_inference/causal_inference/estimators/ipw.py`) - Inverse Probability Weighting with propensity score models and weight stabilization
- **AIPW** (`libs/causal_inference/causal_inference/estimators/aipw.py`) - Augmented Inverse Probability Weighting doubly robust estimator combining G-computation and IPW

## Code Quality Standards

**Tooling configuration:**
- **Ruff** (replaces Black + isort + flake8) - line length 88, targets Python 3.11+
- **MyPy** - strict typing on libs/causal_inference/ and shared/, excludes test files
- **Pytest** - with asyncio support, 300s timeout, coverage reporting to XML/HTML
- Supports Python 3.11, 3.12, 3.13

**Code organization:**
- Use absolute imports and explicit `__init__.py` exports
- Pydantic models for all data structures with validation
- Abstract base classes define consistent interfaces
- Type hints required (NDArray[Any] for numpy arrays)

## Development Notes

**Dependencies:**
- Uses `uv` as package manager (UV commands in Makefile)
- Core: numpy, pandas, scipy, scikit-learn, statsmodels
- API: FastAPI, uvicorn, SQLAlchemy, asyncpg
- ML extras: lightgbm, mlflow, shap (optional)
- PyMC temporarily disabled for Python 3.13 compatibility

**Testing approach:**
- Tests in `libs/` and `shared/` directories follow pytest conventions
- Integration tests marked with `@pytest.mark.integration`
- Coverage reports exclude test files and `__init__.py`
- Async test support enabled

**Project status:**
- Currently in "Phase 1 (Foundation)" development
- Recent work: G-computation estimator implementation with ruff formatting fixes
- Branch: `feature/implement-g-computation-estimator`