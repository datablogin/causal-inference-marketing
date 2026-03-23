"""Tests for Issue #142: Positivity diagnostics for ensemble predictions."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.g_computation import GComputationEstimator


@pytest.fixture
def normal_continuous_data():
    """Generate well-behaved synthetic data with known treatment effect."""
    np.random.seed(42)
    n = 500

    # Covariates
    X = np.random.randn(n, 3)

    # Treatment with confounding
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = np.random.binomial(1, propensity)

    # Outcome with treatment effect = 2.0 (well-behaved, no extreme predictions)
    outcome = (
        2.0 * treatment
        + X[:, 0]
        + 0.5 * X[:, 1]
        + 0.3 * X[:, 2]
        + np.random.randn(n) * 0.5
    )

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=X, names=["X1", "X2", "X3"]),
    }


@pytest.fixture
def heterogeneous_effect_data():
    """Generate data with highly heterogeneous treatment effects (outlier ITEs)."""
    np.random.seed(123)
    n = 500

    # Covariates with one strong effect modifier
    X = np.random.randn(n, 3)

    # Treatment assignment
    propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0])))
    treatment = np.random.binomial(1, propensity)

    # Outcome with VERY heterogeneous treatment effects:
    # Most units have effect ~1.0, but ~10% have effect ~20.0
    # This creates outlier individual treatment effects
    is_outlier = X[:, 2] > 1.5  # ~7% of observations
    treatment_effect = np.where(is_outlier, 20.0, 1.0)

    outcome = (
        treatment_effect * treatment
        + X[:, 0]
        + 0.5 * X[:, 1]
        + np.random.randn(n) * 0.3
    )

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=X, names=["X1", "X2", "X3"]),
    }


@pytest.fixture
def extreme_binary_data():
    """Generate binary outcome data where predictions are extreme (near 0 or 1)."""
    np.random.seed(456)
    n = 500

    # Covariates
    X = np.random.randn(n, 3)

    # Treatment assignment
    propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0])))
    treatment = np.random.binomial(1, propensity)

    # Binary outcome that is nearly deterministic given covariates
    # This forces model predictions to be very close to 0 or 1
    linear_pred = 5.0 * X[:, 0] + 3.0 * X[:, 1] + 2.0 * treatment
    prob = 1 / (1 + np.exp(-linear_pred))
    outcome = np.random.binomial(1, prob).astype(float)

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="binary"),
        "covariates": CovariateData(values=X, names=["X1", "X2", "X3"]),
    }


class TestPositivityDiagnosticsExist:
    """Test that positivity diagnostics are returned after ensemble estimation."""

    def test_positivity_diagnostics_exist(self, normal_continuous_data):
        """After estimate_ate with ensemble, get_positivity_diagnostics() returns non-None."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            bootstrap_samples=0,
            random_state=42,
        )
        estimator.fit(
            normal_continuous_data["treatment"],
            normal_continuous_data["outcome"],
            normal_continuous_data["covariates"],
        )
        estimator.estimate_ate()

        diagnostics = estimator.get_positivity_diagnostics()
        assert diagnostics is not None


class TestPositivityDiagnosticsNormalData:
    """Test that normal data passes positivity checks."""

    def test_positivity_diagnostics_normal_data(self, normal_continuous_data):
        """Normal, well-behaved data should pass the positivity check."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            bootstrap_samples=0,
            random_state=42,
        )
        estimator.fit(
            normal_continuous_data["treatment"],
            normal_continuous_data["outcome"],
            normal_continuous_data["covariates"],
        )
        estimator.estimate_ate()

        diagnostics = estimator.get_positivity_diagnostics()
        assert diagnostics is not None
        assert diagnostics["positivity_check"] is True
        assert len(diagnostics["warnings"]) == 0


class TestPositivityDiagnosticsKeys:
    """Test that all expected keys are present in diagnostics."""

    def test_positivity_diagnostics_effect_keys(self, normal_continuous_data):
        """Verify all expected keys present in diagnostics dict."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            bootstrap_samples=0,
            random_state=42,
        )
        estimator.fit(
            normal_continuous_data["treatment"],
            normal_continuous_data["outcome"],
            normal_continuous_data["covariates"],
        )
        estimator.estimate_ate()

        diagnostics = estimator.get_positivity_diagnostics()
        assert diagnostics is not None

        # Check required keys
        assert "effect_range" in diagnostics
        assert "effect_std" in diagnostics
        assert "positivity_check" in diagnostics
        assert "warnings" in diagnostics

        # effect_range should be a 2-element list [min, max]
        assert isinstance(diagnostics["effect_range"], list)
        assert len(diagnostics["effect_range"]) == 2
        assert diagnostics["effect_range"][0] <= diagnostics["effect_range"][1]

        # effect_std should be a non-negative float
        assert isinstance(diagnostics["effect_std"], float)
        assert diagnostics["effect_std"] >= 0


class TestPositivityDiagnosticsNotRunWithoutEnsemble:
    """Test that positivity diagnostics are not run for single model."""

    def test_positivity_diagnostics_not_run_without_ensemble(self, normal_continuous_data):
        """Single model mode should return None for get_positivity_diagnostics()."""
        estimator = GComputationEstimator(
            model_type="linear",
            use_ensemble=False,
            bootstrap_samples=0,
            random_state=42,
        )
        estimator.fit(
            normal_continuous_data["treatment"],
            normal_continuous_data["outcome"],
            normal_continuous_data["covariates"],
        )
        estimator.estimate_ate()

        diagnostics = estimator.get_positivity_diagnostics()
        assert diagnostics is None


class TestPositivityDiagnosticsEffectOutliers:
    """Test outlier detection in individual treatment effects."""

    def test_positivity_diagnostics_effect_outliers(self, heterogeneous_effect_data):
        """Data with heterogeneous effects should trigger outlier detection."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            bootstrap_samples=0,
            random_state=42,
        )
        estimator.fit(
            heterogeneous_effect_data["treatment"],
            heterogeneous_effect_data["outcome"],
            heterogeneous_effect_data["covariates"],
        )
        estimator.estimate_ate()

        diagnostics = estimator.get_positivity_diagnostics()
        assert diagnostics is not None

        # Should detect outlier effects
        assert "effect_outlier_fraction" in diagnostics
        assert diagnostics["effect_outlier_fraction"] > 0

        # The effect range should be wide due to heterogeneous effects
        effect_range = diagnostics["effect_range"]
        assert (effect_range[1] - effect_range[0]) > 5.0  # Wide range


class TestPositivityDiagnosticsExtremeBinary:
    """Test extreme prediction detection for binary outcomes."""

    def test_positivity_diagnostics_extreme_binary(self, extreme_binary_data):
        """Binary outcome with extreme predictions should be flagged."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            bootstrap_samples=0,
            random_state=42,
        )
        estimator.fit(
            extreme_binary_data["treatment"],
            extreme_binary_data["outcome"],
            extreme_binary_data["covariates"],
        )
        estimator.estimate_ate()

        diagnostics = estimator.get_positivity_diagnostics()
        assert diagnostics is not None

        # For binary outcomes, should have extreme fraction keys
        assert "treated_extreme_low_frac" in diagnostics
        assert "treated_extreme_high_frac" in diagnostics
        assert "control_extreme_low_frac" in diagnostics
        assert "control_extreme_high_frac" in diagnostics

        # The extreme binary data should trigger some extreme predictions
        total_extreme_treated = (
            diagnostics["treated_extreme_low_frac"]
            + diagnostics["treated_extreme_high_frac"]
        )
        total_extreme_control = (
            diagnostics["control_extreme_low_frac"]
            + diagnostics["control_extreme_high_frac"]
        )
        # At least one group should have some extreme predictions
        assert total_extreme_treated > 0 or total_extreme_control > 0
