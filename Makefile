# Makefile for Causal Inference Marketing (compatible with analytics-backend-monorepo)

# Variables
PYTHON := python3
UV := uv
RUFF := ruff
MYPY := mypy
PYTEST := pytest

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  format      - Format code with ruff"
	@echo "  lint        - Lint code with ruff"
	@echo "  typecheck   - Type check with mypy"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  ci          - Run full CI pipeline (lint, typecheck, test)"
	@echo "  clean       - Clean build artifacts"
	@echo "  api         - Start causal inference API server"
	@echo "  review      - Run Claude review of current PR and post as comment"
	@echo "  review-dry  - Preview Claude review without executing"
	@echo "  review-file - Run Claude review and save to file"

# Installation targets
.PHONY: install
install:
	$(UV) pip install -e .

.PHONY: install-dev
install-dev:
	$(UV) pip install -e ".[dev,test,ml]"

# Code quality targets
.PHONY: format
format:
	$(RUFF) format libs/ services/ shared/
	$(RUFF) check --fix libs/ services/ shared/

.PHONY: lint
lint:
	$(RUFF) check libs/ services/ shared/

.PHONY: typecheck
typecheck:
	$(MYPY) libs/causal_inference/ shared/

# Testing targets
.PHONY: test
test:
	$(PYTEST) libs/ shared/

.PHONY: test-cov
test-cov:
	$(PYTEST) --cov=causal_inference --cov=shared --cov-report=html --cov-report=term libs/ shared/

.PHONY: test-integration
test-integration:
	$(PYTEST) -m integration

# CI pipeline
.PHONY: ci
ci: lint typecheck test
	@echo "✅ CI pipeline completed successfully"

# Service targets
.PHONY: api
api:
	cd services/causal_api && $(UV) run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Library-specific targets
.PHONY: ci-causal-inference
ci-causal-inference:
	$(PYTEST) libs/causal_inference/tests/
	$(MYPY) libs/causal_inference/
	$(RUFF) check libs/causal_inference/

# Review targets
.PHONY: review
review:
	@echo "Running Claude review of current PR and posting as comment..."
	./claude-review.sh --focus causal-inference

.PHONY: review-dry
review-dry:
	@echo "Previewing Claude review of current PR..."
	./claude-review.sh --dry-run --focus causal-inference

.PHONY: review-file
review-file:
	@echo "Running Claude review of current PR and saving to file..."
	./claude-review.sh --save-file --focus causal-inference

# Clean targets
.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete