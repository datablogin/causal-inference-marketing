"""Tests for Issue #138: Bootstrap should propagate ensemble settings."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.core.bootstrap import BootstrapConfig
from causal_inference.estimators.g_computation import GComputationEstimator


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known treatment effect."""
    np.random.seed(42)
    n = 200

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


class TestBootstrapEnsemblePropagation:
    """Tests for bootstrap propagation of ensemble settings (#138)."""

    def test_bootstrap_propagates_ensemble_settings(self, synthetic_data):
        """Bootstrap sub-estimator should inherit ensemble config from parent."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge"],
            ensemble_variance_penalty=0.2,
            bootstrap_samples=0,
            random_state=42,
        )

        # Create a bootstrap sub-estimator
        sub_estimator = estimator._create_bootstrap_estimator(random_state=123)

        # Sub-estimator should have ensemble enabled
        assert sub_estimator.use_ensemble is True
        assert sub_estimator.ensemble_models == ["linear", "ridge"]
        assert sub_estimator.ensemble_variance_penalty == 0.2

    def test_bootstrap_ensemble_opt_out(self, synthetic_data):
        """propagate_ensemble=False should disable ensemble in bootstrap."""
        config = BootstrapConfig(
            n_samples=10,
            propagate_ensemble=False,
            random_state=42,
        )
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge"],
            bootstrap_config=config,
            random_state=42,
        )

        sub_estimator = estimator._create_bootstrap_estimator(random_state=123)

        # With opt-out, sub-estimator should NOT have ensemble
        assert sub_estimator.use_ensemble is False

    def test_ensemble_bootstrap_produces_valid_cis(self, synthetic_data):
        """Ensemble + bootstrap should produce valid confidence intervals."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge"],
            ensemble_variance_penalty=0.1,
            bootstrap_samples=30,
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        effect = estimator.estimate_ate()

        # CIs should exist and bracket ATE
        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None
        assert effect.ate_ci_lower < effect.ate < effect.ate_ci_upper

    def test_ensemble_bootstrap_handles_model_failures(self, synthetic_data):
        """Bootstrap should handle model failures gracefully in ensemble mode."""
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge", "random_forest"],
            ensemble_variance_penalty=0.1,
            bootstrap_samples=20,
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # Should produce valid effect even with ensemble in bootstrap
        effect = estimator.estimate_ate()
        assert effect.ate is not None
        assert np.isfinite(effect.ate)
