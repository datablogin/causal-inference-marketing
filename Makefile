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
	@echo "🚀 Causal Inference Marketing Tools - Development Commands"
	@echo ""
	@echo "📦 Installation:"
	@echo "  install       - Install core dependencies only"
	@echo "  install-dev   - Install with development dependencies (dev, test, ml)"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  format        - Auto-format code with ruff and apply fixes"
	@echo "  lint          - Run ruff linting on libs/, services/, shared/"
	@echo "  typecheck     - Run mypy type checking on libs/causal_inference/ and shared/"
	@echo "  security      - Run safety check for dependency vulnerabilities"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test          - Run pytest on libs/ and shared/ directories"
	@echo "  test-cov      - Run tests with coverage reporting (HTML/XML)"
	@echo "  test-integration - Run integration tests only"
	@echo ""
	@echo "🔄 CI/CD:"
	@echo "  ci            - Run complete CI pipeline (lint, typecheck, test)"
	@echo "  ci-causal-inference - Run tests, mypy, and ruff for causal_inference library only"
	@echo ""
	@echo "🚀 Services:"
	@echo "  api           - Start FastAPI development server at localhost:8000"
	@echo ""
	@echo "🤖 Code Review:"
	@echo "  review        - Run Claude review of current PR and post as comment"
	@echo "  review-dry    - Preview Claude review without executing"
	@echo "  review-file   - Run Claude review and save to file"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean         - Clean build artifacts, caches, and temporary files"

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
	$(UV) run $(RUFF) format libs/ services/ shared/
	$(UV) run $(RUFF) check --fix libs/ services/ shared/

.PHONY: lint
lint:
	$(UV) run $(RUFF) check libs/ services/ shared/

.PHONY: typecheck
typecheck:
	$(UV) run $(MYPY) libs/causal_inference/ shared/

.PHONY: security
security:
	$(UV) run safety check || echo "⚠️  Security vulnerabilities found (non-blocking for development)"

# Testing targets
.PHONY: test
test:
	$(UV) run $(PYTEST) -n auto libs/ shared/

.PHONY: test-sequential
test-sequential:
	$(UV) run $(PYTEST) libs/ shared/

.PHONY: test-cov
test-cov:
	$(UV) run $(PYTEST) --cov=causal_inference --cov=shared --cov-report=html --cov-report=term libs/ shared/

.PHONY: test-integration
test-integration:
	$(UV) run $(PYTEST) -m integration

# CI pipeline
.PHONY: ci
ci: lint typecheck security test
	@echo "✅ CI pipeline completed successfully"

# Service targets
.PHONY: api
api:
	cd services/causal_api && $(UV) run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Library-specific targets
.PHONY: ci-causal-inference
ci-causal-inference:
	$(UV) run $(PYTEST) libs/causal_inference/tests/
	$(UV) run $(MYPY) libs/causal_inference/
	$(UV) run $(RUFF) check libs/causal_inference/

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
	rm -rf .coverage*
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete