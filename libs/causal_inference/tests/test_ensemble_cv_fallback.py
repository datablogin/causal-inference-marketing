"""Test for ensemble CV fallback with small datasets."""

import numpy as np

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.g_computation import GComputationEstimator


def test_ensemble_cv_fallback_small_dataset():
    """Test that very small datasets fall back to in-sample predictions.

    When n < cv_folds * 2, cross-validation is infeasible and the ensemble
    weight optimization should fall back to in-sample predictions, reporting
    ensemble_cv_folds=0 in diagnostics.
    """
    rng = np.random.default_rng(42)
    n = 10  # Minimum allowed sample size

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

    # Fit the estimator to populate ensemble_models_fitted
    estimator.fit(
        TreatmentData(values=treatment, treatment_type="binary"),
        OutcomeData(values=outcome, outcome_type="continuous"),
        CovariateData(values=X, names=["X1", "X2"]),
    )

    # Now test the fallback by calling _optimize_ensemble_weights directly
    # with high cv_folds so n < cv_folds * 2 triggers the in-sample path
    import pandas as pd

    features = pd.DataFrame(
        {"treatment": treatment, "X1": X[:, 0], "X2": X[:, 1]}
    )
    weights = estimator._optimize_ensemble_weights(
        models=estimator.ensemble_models_fitted,
        features=features,
        y=outcome,
        cv_folds=6,  # 10 < 6*2 = 12, so in-sample fallback triggers
    )

    # Diagnostics should show cv_folds=0 (in-sample fallback)
    diag = estimator.get_optimization_diagnostics()
    assert diag is not None
    assert diag["ensemble_cv_folds"] == 0, (
        f"Expected ensemble_cv_folds=0 (in-sample fallback) but got {diag['ensemble_cv_folds']}"
    )

    # Weights should still be valid
    assert weights is not None
    assert abs(np.sum(weights) - 1.0) < 1e-6
    assert np.all(weights >= 0)
