"""Test for ensemble CV fallback with small datasets."""

import numpy as np

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.g_computation import GComputationEstimator


def test_ensemble_cv_fallback_small_dataset():
    """Test that very small datasets fall back to in-sample predictions.

    When n is too small for meaningful CV folds, the fit path should skip
    OOF predictions and use in-sample predictions for weight optimization,
    reporting ensemble_cv_folds=None in diagnostics.
    """
    rng = np.random.default_rng(42)
    n = 10  # Small enough to trigger fallback (n < cv_folds * 5)

    X = rng.standard_normal((n, 2))
    treatment = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    outcome = 1.5 * treatment + X[:, 0] + rng.standard_normal(n) * 0.3

    estimator = GComputationEstimator(
        use_ensemble=True,
        ensemble_models=["linear", "ridge"],
        ensemble_variance_penalty=0.1,
        random_state=42,
        verbose=True,
    )

    # Fit triggers the small-dataset fallback path internally
    estimator.fit(
        TreatmentData(values=treatment, treatment_type="binary"),
        OutcomeData(values=outcome, outcome_type="continuous"),
        CovariateData(values=X, names=["X1", "X2"]),
    )

    # Diagnostics should show cv_folds=None (in-sample fallback)
    diag = estimator.get_optimization_diagnostics()
    assert diag is not None
    assert diag["ensemble_cv_folds"] is None, (
        f"Expected ensemble_cv_folds=None (in-sample fallback) but got {diag['ensemble_cv_folds']}"
    )

    # Weights should still be valid
    assert estimator.ensemble_weights is not None
    assert abs(np.sum(estimator.ensemble_weights) - 1.0) < 1e-6
    assert np.all(estimator.ensemble_weights >= 0)
