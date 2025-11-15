"""Tests for G-computation ensemble cross-validation functionality."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
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


def test_ensemble_cv_strategy(synthetic_data):
    """Test that CV strategy runs without errors."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_weight_strategy="cv",
        ensemble_cv_folds=5,
        random_state=42,
        verbose=True,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Check that models were fitted
    assert len(estimator.ensemble_models_fitted) > 0
    assert estimator.ensemble_weights is not None

    # Check diagnostics
    diag = estimator.get_optimization_diagnostics()
    assert diag is not None
    assert diag["ensemble_method"] == "cv"
    assert "ensemble_cv_folds" in diag
    assert "ensemble_cv_objective_mean" in diag

    # Estimate ATE
    effect = estimator.estimate_ate()
    assert effect.ate is not None
    assert abs(effect.ate - synthetic_data["true_ate"]) < 1.0


def test_ensemble_split_strategy(synthetic_data):
    """Test that validation split strategy runs without errors."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_weight_strategy="split",
        ensemble_val_size=0.2,
        random_state=42,
        verbose=True,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Check that models were fitted
    assert len(estimator.ensemble_models_fitted) > 0
    assert estimator.ensemble_weights is not None

    # Check diagnostics
    diag = estimator.get_optimization_diagnostics()
    assert diag is not None
    assert diag["ensemble_method"] == "split"
    assert "ensemble_val_size" in diag
    assert "ensemble_val_objective" in diag

    # Estimate ATE
    effect = estimator.estimate_ate()
    assert effect.ate is not None
    assert abs(effect.ate - synthetic_data["true_ate"]) < 1.0


def test_ensemble_in_sample_strategy(synthetic_data):
    """Test that in-sample strategy (default) still works."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge"],
        ensemble_weight_strategy="in_sample",
        random_state=42,
        verbose=True,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    # Check that models were fitted
    assert len(estimator.ensemble_models_fitted) > 0
    assert estimator.ensemble_weights is not None

    # Check diagnostics - should not have CV-specific fields
    diag = estimator.get_optimization_diagnostics()
    assert diag is not None
    assert "ensemble_success" in diag
    assert "ensemble_cv_folds" not in diag  # Should not have CV fields

    # Estimate ATE
    effect = estimator.estimate_ate()
    assert effect.ate is not None
    assert abs(effect.ate - synthetic_data["true_ate"]) < 1.0


def test_ensemble_cv_vs_in_sample(synthetic_data):
    """Test that CV and in-sample produce similar but not identical results."""
    # In-sample
    estimator_in_sample = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_weight_strategy="in_sample",
        random_state=42,
    )
    estimator_in_sample.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )
    effect_in_sample = estimator_in_sample.estimate_ate()

    # CV
    estimator_cv = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_weight_strategy="cv",
        ensemble_cv_folds=5,
        random_state=42,
    )
    estimator_cv.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )
    effect_cv = estimator_cv.estimate_ate()

    # Both should recover true effect reasonably
    assert abs(effect_in_sample.ate - synthetic_data["true_ate"]) < 1.0
    assert abs(effect_cv.ate - synthetic_data["true_ate"]) < 1.0

    # Weights should be different (but both valid)
    weights_in_sample = estimator_in_sample.ensemble_weights
    weights_cv = estimator_cv.ensemble_weights

    # Check that weights are valid
    assert np.abs(np.sum(weights_in_sample) - 1.0) < 1e-6
    assert np.abs(np.sum(weights_cv) - 1.0) < 1e-6
    assert np.all(weights_in_sample >= 0)
    assert np.all(weights_cv >= 0)

    # Weights should differ (CV should be more conservative)
    assert not np.allclose(weights_in_sample, weights_cv, atol=0.05)


def test_ensemble_cv_small_sample_fallback():
    """Test that CV falls back to in-sample for very small samples."""
    np.random.seed(42)
    n = 80  # Small sample

    X = np.random.randn(n, 2)
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.0 * treatment + X[:, 0] + np.random.randn(n) * 0.5

    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge"],
        ensemble_weight_strategy="cv",
        ensemble_cv_folds=5,
        random_state=42,
        verbose=True,
    )

    estimator.fit(
        TreatmentData(values=treatment, treatment_type="binary"),
        OutcomeData(values=outcome, outcome_type="continuous"),
        CovariateData(values=X, names=["X1", "X2"]),
    )

    # Should still succeed (falls back to in-sample)
    assert estimator.ensemble_weights is not None
    effect = estimator.estimate_ate()
    assert effect.ate is not None


def test_ensemble_split_small_sample_fallback():
    """Test that split falls back to in-sample for very small samples."""
    np.random.seed(42)
    n = 80  # Small sample

    X = np.random.randn(n, 2)
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.0 * treatment + X[:, 0] + np.random.randn(n) * 0.5

    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge"],
        ensemble_weight_strategy="split",
        ensemble_val_size=0.3,
        random_state=42,
        verbose=True,
    )

    estimator.fit(
        TreatmentData(values=treatment, treatment_type="binary"),
        OutcomeData(values=outcome, outcome_type="continuous"),
        CovariateData(values=X, names=["X1", "X2"]),
    )

    # Should still succeed (may fall back to in-sample)
    assert estimator.ensemble_weights is not None
    effect = estimator.estimate_ate()
    assert effect.ate is not None


def test_ensemble_cv_treatment_balance():
    """Test that CV preserves treatment balance across folds."""
    np.random.seed(42)
    n = 600

    X = np.random.randn(n, 3)
    # Create treatment with strong confounding
    propensity = 1 / (1 + np.exp(-(2 * X[:, 0] + X[:, 1])))
    treatment = np.random.binomial(1, propensity)
    outcome = 2.0 * treatment + X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge", "random_forest"],
        ensemble_weight_strategy="cv",
        ensemble_cv_folds=5,
        random_state=42,
        verbose=True,
    )

    estimator.fit(
        TreatmentData(values=treatment, treatment_type="binary"),
        OutcomeData(values=outcome, outcome_type="continuous"),
        CovariateData(values=X, names=["X1", "X2", "X3"]),
    )

    # Should complete successfully
    assert estimator.ensemble_weights is not None

    # Check diagnostics show successful folds
    diag = estimator.get_optimization_diagnostics()
    assert diag["ensemble_cv_folds"] > 0
    assert diag["ensemble_cv_folds"] <= 5


def test_ensemble_cv_with_bootstrap(synthetic_data):
    """Test that CV strategy works with bootstrap confidence intervals."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge"],
        ensemble_weight_strategy="cv",
        ensemble_cv_folds=3,  # Fewer folds for speed with bootstrap
        bootstrap_samples=50,  # Small for speed
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    effect = estimator.estimate_ate()

    # Should have bootstrap CIs
    assert effect.ate_ci_lower is not None
    assert effect.ate_ci_upper is not None
    assert effect.ate_ci_lower < effect.ate < effect.ate_ci_upper


def test_ensemble_split_with_bootstrap(synthetic_data):
    """Test that split strategy works with bootstrap confidence intervals."""
    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge"],
        ensemble_weight_strategy="split",
        ensemble_val_size=0.2,
        bootstrap_samples=50,  # Small for speed
        random_state=42,
    )

    estimator.fit(
        synthetic_data["treatment"],
        synthetic_data["outcome"],
        synthetic_data["covariates"],
    )

    effect = estimator.estimate_ate()

    # Should have bootstrap CIs
    assert effect.ate_ci_lower is not None
    assert effect.ate_ci_upper is not None
    assert effect.ate_ci_lower < effect.ate < effect.ate_ci_upper


def test_ensemble_cv_different_fold_counts(synthetic_data):
    """Test CV with different numbers of folds."""
    for n_folds in [3, 5, 10]:
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge"],
            ensemble_weight_strategy="cv",
            ensemble_cv_folds=n_folds,
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # Should succeed
        assert estimator.ensemble_weights is not None
        effect = estimator.estimate_ate()
        assert effect.ate is not None
        assert abs(effect.ate - synthetic_data["true_ate"]) < 1.5


def test_ensemble_split_different_val_sizes(synthetic_data):
    """Test split strategy with different validation sizes."""
    for val_size in [0.1, 0.2, 0.3]:
        estimator = GComputationEstimator(
            use_ensemble=True,
            ensemble_models=["linear", "ridge"],
            ensemble_weight_strategy="split",
            ensemble_val_size=val_size,
            random_state=42,
        )

        estimator.fit(
            synthetic_data["treatment"],
            synthetic_data["outcome"],
            synthetic_data["covariates"],
        )

        # Should succeed
        assert estimator.ensemble_weights is not None
        effect = estimator.estimate_ate()
        assert effect.ate is not None
        assert abs(effect.ate - synthetic_data["true_ate"]) < 1.5
