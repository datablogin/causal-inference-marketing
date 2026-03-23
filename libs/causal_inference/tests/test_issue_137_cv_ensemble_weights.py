"""Tests for Issue #137: CV-based ensemble weight optimization."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.g_computation import GComputationEstimator


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known treatment effect."""
    np.random.seed(42)
    n = 300

    X = np.random.randn(n, 3)
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = np.random.binomial(1, propensity)
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
        "true_ate": 2.0,
    }


@pytest.fixture
def binary_outcome_data():
    """Generate synthetic data with binary outcome."""
    np.random.seed(42)
    n = 300

    X = np.random.randn(n, 3)
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = np.random.binomial(1, propensity)
    linear_pred = -0.5 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.8 * treatment
    prob = 1 / (1 + np.exp(-linear_pred))
    outcome = np.random.binomial(1, prob)

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="binary"),
        "covariates": CovariateData(values=X, names=["X1", "X2", "X3"]),
    }


@pytest.fixture
def small_data():
    """Generate small dataset for edge case testing."""
    np.random.seed(42)
    n = 30

    X = np.random.randn(n, 2)
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.0 * treatment + X[:, 0] + np.random.randn(n) * 0.5

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=X, names=["X1", "X2"]),
        "true_ate": 2.0,
    }


class TestCVEnsembleWeights:
    """Tests for cross-validated ensemble weight optimization (#137)."""

    def test_cv_weights_basic(self, synthetic_data):
        """CV-optimized weights should be valid (sum to 1, non-negative)."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_use_cv=True,
            ensemble_cv_folds=5,
            ensemble_variance_penalty=0.1,
            random_state=42,
            bootstrap_samples=0,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        assert estimator.ensemble_weights is not None
        assert abs(np.sum(estimator.ensemble_weights) - 1.0) < 1e-6
        assert np.all(estimator.ensemble_weights >= -1e-10)

    def test_cv_backward_compat(self, synthetic_data):
        """ensemble_use_cv=False should preserve old in-sample behavior."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_use_cv=False,
            ensemble_variance_penalty=0.1,
            random_state=42,
            bootstrap_samples=0,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # Should still produce valid weights
        assert estimator.ensemble_weights is not None
        assert abs(np.sum(estimator.ensemble_weights) - 1.0) < 1e-6

        # Should NOT have OOF predictions stored
        assert (
            not hasattr(estimator, "ensemble_oof_predictions_")
            or estimator.ensemble_oof_predictions_ is None
        )

    def test_cv_small_data_fallback(self, small_data):
        """Should auto-reduce folds for small datasets."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge"],
            ensemble_use_cv=True,
            ensemble_cv_folds=5,  # Will be reduced for small data
            ensemble_variance_penalty=0.1,
            random_state=42,
            bootstrap_samples=0,
        )

        # Should not raise, should auto-reduce folds
        estimator.fit(
            small_data["treatment"],
            small_data["outcome"],
            small_data["covariates"],
        )

        assert estimator.ensemble_weights is not None
        assert abs(np.sum(estimator.ensemble_weights) - 1.0) < 1e-6

    def test_cv_ate_recovery(self, synthetic_data):
        """CV ensemble should recover true ATE reasonably."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_use_cv=True,
            ensemble_cv_folds=5,
            ensemble_variance_penalty=0.1,
            random_state=42,
            bootstrap_samples=0,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        effect = estimator.estimate_ate()
        assert abs(effect.ate - synthetic_data["true_ate"]) < 1.0

    def test_cv_oof_predictions_stored(self, synthetic_data):
        """OOF predictions should be accessible after fitting with CV."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_use_cv=True,
            ensemble_cv_folds=5,
            ensemble_variance_penalty=0.1,
            random_state=42,
            bootstrap_samples=0,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # OOF predictions should be stored
        assert hasattr(estimator, "ensemble_oof_predictions_")
        assert estimator.ensemble_oof_predictions_ is not None
        # Shape: (n_samples, n_models)
        n_samples = len(synthetic_data["treatment"].values)
        n_models = len(estimator.ensemble_models_fitted)
        assert estimator.ensemble_oof_predictions_.shape == (n_samples, n_models)

    def test_cv_diagnostics(self, synthetic_data):
        """Diagnostics should include CV information when using CV."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_use_cv=True,
            ensemble_cv_folds=5,
            ensemble_variance_penalty=0.1,
            random_state=42,
            bootstrap_samples=0,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        diag = estimator.get_optimization_diagnostics()
        assert diag is not None
        assert "ensemble_cv_folds" in diag

    def test_cv_binary_outcome(self, binary_outcome_data):
        """CV ensemble should work with binary outcomes."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_use_cv=True,
            ensemble_cv_folds=5,
            ensemble_variance_penalty=0.1,
            random_state=42,
            bootstrap_samples=0,
        )

        estimator.fit(
            binary_outcome_data["treatment"],
            binary_outcome_data["outcome"],
            binary_outcome_data["covariates"],
        )

        assert estimator.ensemble_weights is not None
        assert abs(np.sum(estimator.ensemble_weights) - 1.0) < 1e-6

        effect = estimator.estimate_ate()
        assert effect.ate is not None
        assert np.isfinite(effect.ate)

    def test_cv_reproducibility(self, synthetic_data):
        """CV ensemble should be deterministic with same seed."""
        weights = []
        for _ in range(2):
            estimator = GComputationEstimator(
                use_ensemble=True,
                ensemble_models=["linear", "ridge", "random_forest"],
                ensemble_use_cv=True,
                ensemble_cv_folds=5,
                ensemble_variance_penalty=0.1,
                random_state=42,
                bootstrap_samples=0,
            )

            estimator.fit(
                synthetic_data["treatment"],
                synthetic_data["outcome"],
                synthetic_data["covariates"],
            )

            weights.append(estimator.ensemble_weights.copy())

        np.testing.assert_array_almost_equal(weights[0], weights[1])
