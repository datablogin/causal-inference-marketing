"""Tests for Issue #139 (model-specific params) and Issue #140 (error threshold).

Issue #139: ensemble_model_params allows per-model hyperparameters without
cross-contamination between model types.

Issue #140: ensemble_min_models provides a threshold so that too many ensemble
fit failures raise EstimationError instead of silently falling back.
"""

import numpy as np
import pytest

from causal_inference.core.base import (
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.g_computation import GComputationEstimator


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known treatment effect."""
    np.random.seed(42)
    n = 500

    # Covariates
    X = np.random.randn(n, 3)

    # Treatment with confounding
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = np.random.binomial(1, propensity)

    # Outcome with treatment effect = 2.0
    true_ate = 2.0
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
        "true_ate": true_ate,
    }


# ======================================================================
# Issue #139: Model-Specific Parameter Support
# ======================================================================


class TestModelSpecificParams:
    """Tests for ensemble_model_params (Issue #139)."""

    def test_model_specific_params_rf(self, synthetic_data):
        """RF receives max_depth=3 from ensemble_model_params."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_model_params={"random_forest": {"max_depth": 3}},
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        rf_model = estimator.ensemble_models_fitted["random_forest"]
        assert rf_model.max_depth == 3

    def test_model_specific_params_ridge(self, synthetic_data):
        """Ridge receives custom alpha from ensemble_model_params."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_model_params={"ridge": {"alpha": 5.0}},
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        ridge_model = estimator.ensemble_models_fitted["ridge"]
        assert ridge_model.alpha == 5.0

    def test_model_specific_params_no_cross_contamination(self, synthetic_data):
        """RF params (max_depth) must not leak to linear model."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "random_forest"],
            ensemble_model_params={"random_forest": {"max_depth": 3}},
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # linear model should be fitted without RF-specific params
        linear_model = estimator.ensemble_models_fitted["linear"]
        assert not hasattr(linear_model, "max_depth")

        # RF should have max_depth=3
        rf_model = estimator.ensemble_models_fitted["random_forest"]
        assert rf_model.max_depth == 3

    def test_model_specific_params_backward_compat(self, synthetic_data):
        """No ensemble_model_params -> defaults are used (regression test)."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # All three models should be fitted with defaults
        assert "linear" in estimator.ensemble_models_fitted
        assert "ridge" in estimator.ensemble_models_fitted
        assert "random_forest" in estimator.ensemble_models_fitted

        # Ridge default alpha = 1.0
        ridge_model = estimator.ensemble_models_fitted["ridge"]
        assert ridge_model.alpha == 1.0

        # RF default n_estimators = 100
        rf_model = estimator.ensemble_models_fitted["random_forest"]
        assert rf_model.n_estimators == 100

    def test_model_specific_params_empty_dict(self, synthetic_data):
        """Empty dict for a model -> defaults used for that model."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_model_params={"ridge": {}},
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # Ridge should use default alpha=1.0
        ridge_model = estimator.ensemble_models_fitted["ridge"]
        assert ridge_model.alpha == 1.0

    def test_model_specific_params_propagated_to_bootstrap(self, synthetic_data):
        """Bootstrap sub-estimators receive ensemble_model_params."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_model_params={
                "random_forest": {"max_depth": 3},
                "ridge": {"alpha": 5.0},
            },
            random_state=42,
        )

        # Test _create_bootstrap_estimator propagates params
        bootstrap_est = estimator._create_bootstrap_estimator(random_state=123)
        assert bootstrap_est.ensemble_model_params == {
            "random_forest": {"max_depth": 3},
            "ridge": {"alpha": 5.0},
        }


# ======================================================================
# Issue #140: Error Threshold for Ensemble Fitting Failures
# ======================================================================


class TestErrorThreshold:
    """Tests for ensemble_min_models (Issue #140)."""

    def test_error_threshold_default(self, synthetic_data):
        """Default ensemble_min_models=2; only 1 model fits -> EstimationError."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear"],  # Only 1 model -> can't meet threshold of 2
            ensemble_min_models=2,
            random_state=42,
        )

        with pytest.raises(EstimationError, match="minimum required: 2"):
            estimator.fit(
                synthetic_data["treatment"],
                synthetic_data["outcome"],
                synthetic_data["covariates"],
            )

    def test_error_threshold_custom(self, synthetic_data):
        """ensemble_min_models=1 allows single-model fallback."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear"],  # Only 1 model
            ensemble_min_models=1,
            random_state=42,
        )

        # Should NOT raise; single-model fallback is allowed
        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # The estimator should have fitted model(s)
        assert len(estimator.ensemble_models_fitted) >= 1 or estimator.outcome_model is not None

    def test_error_threshold_all_fail(self, synthetic_data):
        """All models fail -> clear EstimationError."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["nonexistent_model_type"],  # Will be skipped (no match)
            ensemble_min_models=1,
            random_state=42,
        )

        with pytest.raises(EstimationError, match="no models fitted successfully"):
            estimator.fit(
                synthetic_data["treatment"],
                synthetic_data["outcome"],
                synthetic_data["covariates"],
            )

    def test_error_threshold_all_succeed(self, synthetic_data):
        """All models succeed -> no error, ensemble works normally."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_min_models=2,
            random_state=42,
        )

        # Should NOT raise
        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        assert len(estimator.ensemble_models_fitted) == 3
        assert estimator.ensemble_weights is not None

    def test_fit_failure_diagnostics(self, synthetic_data):
        """Failed model names appear in diagnostics."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_min_models=2,
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        diag = estimator.get_optimization_diagnostics()
        assert "ensemble_fit_failures" in diag
        assert "ensemble_fit_success_rate" in diag
        # All succeeded, so success rate should be 1.0
        assert diag["ensemble_fit_success_rate"] == 1.0
        assert diag["ensemble_fit_failures"] == []

    def test_error_threshold_propagated_to_bootstrap(self, synthetic_data):
        """Bootstrap sub-estimators receive ensemble_min_models."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_min_models=3,
            random_state=42,
        )

        bootstrap_est = estimator._create_bootstrap_estimator(random_state=123)
        assert bootstrap_est.ensemble_min_models == 3
