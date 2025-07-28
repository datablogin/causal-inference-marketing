"""Tests for enhanced bootstrap confidence interval methods."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.core.bootstrap import BootstrapConfig
from causal_inference.estimators.g_computation import GComputationEstimator


class TestBootstrapConfig:
    """Test BootstrapConfig validation and functionality."""

    def test_default_config(self):
        """Test default bootstrap configuration."""
        config = BootstrapConfig()
        assert config.n_samples == 1000
        assert config.method == "percentile"
        assert config.confidence_level == 0.95
        assert config.stratified is True
        assert config.parallel is True

    def test_config_validation(self):
        """Test bootstrap configuration validation."""
        # Valid config
        config = BootstrapConfig(
            n_samples=100, method="bca", confidence_level=0.90, random_state=42
        )
        assert config.n_samples == 100
        assert config.method == "bca"
        assert config.confidence_level == 0.90
        assert config.random_state == 42

        # Invalid configurations
        with pytest.raises(ValueError):
            BootstrapConfig(n_samples=-1)  # Negative samples

        with pytest.raises(ValueError):
            BootstrapConfig(confidence_level=1.1)  # Invalid confidence level

        with pytest.raises(ValueError):
            BootstrapConfig(method="invalid")  # Invalid method

        with pytest.raises(ValueError):
            BootstrapConfig(n_jobs=0)  # Invalid n_jobs


class TestBootstrapIntegration:
    """Test bootstrap integration with estimators."""

    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data for testing."""
        np.random.seed(42)
        n = 200

        # Generate covariates
        X = np.random.normal(0, 1, (n, 3))
        covariate_df = pd.DataFrame(X, columns=["X1", "X2", "X3"])

        # Generate treatment (affected by covariates)
        propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1])))
        treatment = np.random.binomial(1, propensity)

        # Generate outcome (affected by treatment and covariates)
        outcome = (
            2 * treatment
            + 0.5 * X[:, 0]
            + 0.3 * X[:, 1]
            + 0.2 * X[:, 2]
            + np.random.normal(0, 1, n)
        )

        treatment_data = TreatmentData(
            values=pd.Series(treatment), treatment_type="binary"
        )
        outcome_data = OutcomeData(values=pd.Series(outcome), outcome_type="continuous")
        covariate_data = CovariateData(
            values=covariate_df, names=list(covariate_df.columns)
        )

        return treatment_data, outcome_data, covariate_data

    def test_basic_bootstrap_functionality(self, simple_data):
        """Test basic bootstrap functionality with G-computation."""
        treatment_data, outcome_data, covariate_data = simple_data

        # Test with basic bootstrap config
        bootstrap_config = BootstrapConfig(
            n_samples=50,  # Small number for faster testing
            method="percentile",
            confidence_level=0.95,
            parallel=False,  # Disable parallel for testing
            random_state=42,
        )

        estimator = GComputationEstimator(
            bootstrap_config=bootstrap_config, verbose=False
        )

        # Fit and estimate
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Check that bootstrap results are available
        assert effect.bootstrap_samples == 50
        assert effect.bootstrap_estimates is not None
        assert len(effect.bootstrap_estimates) <= 50  # May be fewer due to failures
        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None
        assert effect.ate_ci_lower < effect.ate_ci_upper
        assert effect.bootstrap_method == "percentile"

        # Check ATE is reasonable (true effect is ~2)
        assert 1.0 < effect.ate < 3.0

    def test_different_bootstrap_methods(self, simple_data):
        """Test different bootstrap methods."""
        treatment_data, outcome_data, covariate_data = simple_data

        methods = ["percentile", "bias_corrected", "bca"]
        results = {}

        for method in methods:
            bootstrap_config = BootstrapConfig(
                n_samples=50,  # Minimum for bootstrap
                method=method,
                parallel=False,
                random_state=42,
            )

            estimator = GComputationEstimator(
                bootstrap_config=bootstrap_config, verbose=False
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            results[method] = effect

            # Basic checks
            assert effect.bootstrap_method == method
            assert effect.ate_ci_lower is not None
            assert effect.ate_ci_upper is not None

            # Method-specific checks
            if method == "percentile":
                assert effect.ate_ci_lower is not None
                assert effect.ate_ci_upper is not None
            elif method == "bca":
                assert effect.bootstrap_acceleration is not None

        # Results should be similar but not identical
        ates = [results[method].ate for method in methods]
        assert all(1.0 < ate < 3.0 for ate in ates)

    def test_bootstrap_convergence(self, simple_data):
        """Test bootstrap convergence checking."""
        treatment_data, outcome_data, covariate_data = simple_data

        bootstrap_config = BootstrapConfig(
            n_samples=100,
            convergence_check=True,
            min_convergence_samples=50,
            parallel=False,
            random_state=42,
        )

        estimator = GComputationEstimator(
            bootstrap_config=bootstrap_config, verbose=False
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Check convergence information is available
        assert effect.bootstrap_converged is not None

    def test_legacy_compatibility(self, simple_data):
        """Test backward compatibility with legacy parameters."""
        treatment_data, outcome_data, covariate_data = simple_data

        # Test old-style parameters still work
        estimator = GComputationEstimator(
            bootstrap_samples=50, confidence_level=0.90, verbose=False
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Should still work with legacy parameters
        assert effect.bootstrap_samples == 50
        assert abs(effect.confidence_level - 0.90) < 1e-6
        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None

    def test_no_bootstrap(self, simple_data):
        """Test estimator works without bootstrap."""
        treatment_data, outcome_data, covariate_data = simple_data

        bootstrap_config = BootstrapConfig(n_samples=0)

        estimator = GComputationEstimator(
            bootstrap_config=bootstrap_config, verbose=False
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Should work without bootstrap
        assert effect.bootstrap_samples == 0
        assert effect.bootstrap_estimates is None
        assert effect.ate_ci_lower is None
        assert effect.ate_ci_upper is None
        assert effect.bootstrap_method is None

    def test_bootstrap_diagnostics(self, simple_data):
        """Test bootstrap diagnostics functionality."""
        treatment_data, outcome_data, covariate_data = simple_data

        bootstrap_config = BootstrapConfig(
            n_samples=50, method="bca", parallel=False, random_state=42
        )

        estimator = GComputationEstimator(
            bootstrap_config=bootstrap_config, verbose=False
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        estimator.estimate_ate()

        # Get bootstrap diagnostics
        diagnostics = estimator.get_bootstrap_diagnostics()

        assert "method" in diagnostics
        assert "n_samples_requested" in diagnostics
        assert "n_samples_successful" in diagnostics
        assert "success_rate" in diagnostics
        assert "bootstrap_se" in diagnostics

        assert diagnostics["method"] == "bca"
        assert diagnostics["n_samples_requested"] == 50
        assert 0 <= diagnostics["success_rate"] <= 1
