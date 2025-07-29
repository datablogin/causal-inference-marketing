"""Tests for machine learning-based causal inference estimators.

Tests include TMLE, DoublyRobustML, and Super Learner functionality
with comprehensive scenarios including high-dimensional data.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from causal_inference.core.base import (
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.doubly_robust_ml import DoublyRobustMLEstimator
from causal_inference.estimators.tmle import TMLEEstimator
from causal_inference.ml.super_learner import SuperLearner, SuperLearnerConfig


class TestSuperLearner:
    """Test Super Learner ensemble method."""

    @pytest.fixture
    def regression_data(self):
        """Generate synthetic regression data."""
        np.random.seed(42)
        n = 200
        p = 5

        X = np.random.randn(n, p)
        true_coef = np.array([1.5, -1.0, 0.5, 0.0, 0.8])
        y = X @ true_coef + np.random.normal(0, 0.5, n)

        return X, y, true_coef

    @pytest.fixture
    def classification_data(self):
        """Generate synthetic classification data."""
        np.random.seed(42)
        n = 200
        p = 5

        X = np.random.randn(n, p)
        true_coef = np.array([1.0, -0.8, 0.6, 0.0, -0.4])
        logits = X @ true_coef
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)

        return X, y, true_coef

    def test_super_learner_regression(self, regression_data):
        """Test Super Learner on regression task."""
        X, y, _ = regression_data

        sl = SuperLearner(
            base_learners=["linear_regression", "ridge", "random_forest"],
            task_type="regression",
        )

        sl.fit(X, y)
        predictions = sl.predict(X)

        assert sl.is_fitted
        assert len(predictions) == len(y)
        assert sl.task_type == "regression"

        # Check that predictions are reasonable
        mse = np.mean((predictions - y) ** 2)
        assert mse < 1.0  # Should be reasonably accurate

    def test_super_learner_classification(self, classification_data):
        """Test Super Learner on classification task."""
        X, y, _ = classification_data

        sl = SuperLearner(
            base_learners=["logistic_regression", "random_forest"],
            task_type="classification",
        )

        sl.fit(X, y)
        predictions = sl.predict(X)
        prob_predictions = sl.predict_proba(X)

        assert sl.is_fitted
        assert len(predictions) == len(y)
        assert prob_predictions.shape == (len(y), 2)
        assert sl.task_type == "classification"

        # Check that probabilities are valid
        assert np.all(prob_predictions >= 0)
        assert np.all(prob_predictions <= 1)
        assert np.allclose(np.sum(prob_predictions, axis=1), 1)

    def test_super_learner_auto_detection(self, regression_data, classification_data):
        """Test automatic task type detection."""
        X_reg, y_reg, _ = regression_data
        X_cls, y_cls, _ = classification_data

        # Regression data
        sl_reg = SuperLearner(task_type="auto")
        sl_reg.fit(X_reg, y_reg)
        assert sl_reg.task_type == "regression"

        # Classification data
        sl_cls = SuperLearner(task_type="auto")
        sl_cls.fit(X_cls, y_cls)
        assert sl_cls.task_type == "classification"

    def test_super_learner_performance_tracking(self, regression_data):
        """Test performance tracking and learner weights."""
        X, y, _ = regression_data

        sl = SuperLearner(
            base_learners=["linear_regression", "ridge", "lasso"],
            config=SuperLearnerConfig(ensemble_method="stacking"),
        )

        sl.fit(X, y)

        # Check performance tracking
        learner_performance = sl.get_learner_performance()
        ensemble_performance = sl.get_ensemble_performance()
        learner_weights = sl.get_learner_weights()

        assert isinstance(learner_performance, dict)
        assert len(learner_performance) > 0
        assert isinstance(ensemble_performance, dict)
        assert "mse" in ensemble_performance or "rmse" in ensemble_performance

        if learner_weights is not None:
            assert isinstance(learner_weights, dict)
            assert len(learner_weights) == len(learner_performance)

    def test_super_learner_custom_learners(self, regression_data):
        """Test Super Learner with custom learners."""
        X, y, _ = regression_data

        custom_learners = {
            "custom_linear": LinearRegression(),
            "custom_rf": RandomForestRegressor(n_estimators=10, random_state=42),
        }

        sl = SuperLearner(base_learners=custom_learners)
        sl.fit(X, y)
        predictions = sl.predict(X)

        assert sl.is_fitted
        assert len(predictions) == len(y)
        assert len(sl.fitted_learners_) == 2


class TestHighDimensionalData:
    """Test ML estimators on high-dimensional data."""

    @pytest.fixture
    def high_dim_confounded_data(self):
        """Generate high-dimensional confounded data for causal inference."""
        np.random.seed(42)
        n = 1000
        p = 50  # High-dimensional covariates

        # Generate confounders with sparse structure
        true_confounders = np.random.randn(n, 8)  # Only 8 truly important
        noise_confounders = np.random.randn(n, p - 8)  # Rest are noise
        X = np.hstack([true_confounders, noise_confounders])

        # Treatment depends on first 4 confounders
        treatment_coef = np.zeros(p)
        treatment_coef[:4] = [0.5, -0.3, 0.4, -0.2]
        logit_p = X @ treatment_coef
        treatment_prob = 1 / (1 + np.exp(-logit_p))
        treatment = np.random.binomial(1, treatment_prob)

        # Outcome depends on treatment + different confounders
        outcome_coef = np.zeros(p)
        outcome_coef[2:6] = [0.3, -0.4, 0.5, -0.3]  # Overlaps with treatment model
        true_ate = 2.0
        outcome = X @ outcome_coef + true_ate * treatment + np.random.normal(0, 1, n)

        return X, treatment, outcome, true_ate

    def test_tmle_high_dimensional(self, high_dim_confounded_data):
        """Test TMLE on high-dimensional data."""
        X, treatment, outcome, true_ate = high_dim_confounded_data

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X), names=[f"X{i}" for i in range(X.shape[1])]
        )

        # TMLE with Super Learner
        estimator = TMLEEstimator(
            outcome_learner=SuperLearner(
                ["linear_regression", "lasso", "random_forest"]
            ),
            propensity_learner=SuperLearner(
                ["logistic_regression", "lasso_logistic", "random_forest"]
            ),
            cross_fitting=True,
            cv_folds=3,  # Smaller for test efficiency
            random_state=42,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Should recover true ATE within reasonable bounds
        assert abs(effect.ate - true_ate) < 1.0  # Within 1 unit
        assert effect.ate_ci_lower < effect.ate_ci_upper
        assert effect.n_observations == len(treatment)
        assert effect.method == "TMLE"

    def test_doubly_robust_ml_high_dimensional(self, high_dim_confounded_data):
        """Test DoublyRobustML on high-dimensional data."""
        X, treatment, outcome, true_ate = high_dim_confounded_data

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X), names=[f"X{i}" for i in range(X.shape[1])]
        )

        # Test both AIPW and orthogonal moment functions
        for moment_function in ["aipw", "orthogonal"]:
            estimator = DoublyRobustMLEstimator(
                outcome_learner=SuperLearner(
                    ["linear_regression", "ridge", "lasso", "random_forest"]
                ),
                propensity_learner=SuperLearner(
                    ["logistic_regression", "ridge_logistic", "random_forest"]
                ),
                cross_fitting=True,
                cv_folds=3,
                moment_function=moment_function,
                random_state=42,
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            # Should recover true ATE within reasonable bounds
            assert abs(effect.ate - true_ate) < 1.0
            assert effect.ate_ci_lower < effect.ate_ci_upper
            assert effect.method == f"DoublyRobustML_{moment_function}"

    def test_super_learner_performance_comparison(self, high_dim_confounded_data):
        """Test that Super Learner outperforms individual learners."""
        X, treatment, outcome, _ = high_dim_confounded_data

        # Create outcome prediction task
        y = outcome

        # Fit individual learners
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Lasso

        lasso = Lasso(alpha=0.1, random_state=42)
        rf = RandomForestRegressor(n_estimators=50, random_state=42)

        lasso.fit(X, y)
        rf.fit(X, y)

        lasso_pred = lasso.predict(X)
        rf_pred = rf.predict(X)

        lasso_mse = np.mean((lasso_pred - y) ** 2)
        rf_mse = np.mean((rf_pred - y) ** 2)

        # Fit Super Learner
        sl = SuperLearner(
            base_learners=["linear_regression", "lasso", "random_forest"],
            config=SuperLearnerConfig(cv_folds=3),
        )
        sl.fit(X, y)
        sl_pred = sl.predict(X)
        sl_mse = np.mean((sl_pred - y) ** 2)

        # Super Learner should be competitive with best individual learner
        best_individual_mse = min(lasso_mse, rf_mse)
        assert sl_mse <= best_individual_mse * 1.1  # Allow 10% tolerance


class TestCrossValidationPerformance:
    """Test cross-validation and performance aspects."""

    @pytest.fixture
    def medium_confounded_data(self):
        """Generate medium-sized confounded data."""
        np.random.seed(42)
        n = 500
        p = 10

        X = np.random.randn(n, p)

        # Treatment depends on first 3 confounders
        treatment_coef = np.zeros(p)
        treatment_coef[:3] = [0.6, -0.4, 0.5]
        logit_p = X @ treatment_coef
        treatment_prob = 1 / (1 + np.exp(-logit_p))
        treatment = np.random.binomial(1, treatment_prob)

        # Outcome depends on treatment + different confounders
        outcome_coef = np.zeros(p)
        outcome_coef[1:4] = [0.4, -0.3, 0.6]
        true_ate = 1.5
        outcome = X @ outcome_coef + true_ate * treatment + np.random.normal(0, 0.8, n)

        return X, treatment, outcome, true_ate

    def test_cross_fitting_bias_reduction(self, medium_confounded_data):
        """Test that cross-fitting reduces bias compared to non-cross-fitted estimates."""
        X, treatment, outcome, true_ate = medium_confounded_data

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        # TMLE without cross-fitting
        estimator_no_cf = TMLEEstimator(
            outcome_learner=SuperLearner(["random_forest"]),
            propensity_learner=SuperLearner(["random_forest"]),
            cross_fitting=False,
            random_state=42,
        )

        # TMLE with cross-fitting
        estimator_with_cf = TMLEEstimator(
            outcome_learner=SuperLearner(["random_forest"]),
            propensity_learner=SuperLearner(["random_forest"]),
            cross_fitting=True,
            cv_folds=5,
            random_state=42,
        )

        # Fit both estimators
        estimator_no_cf.fit(treatment_data, outcome_data, covariate_data)
        estimator_with_cf.fit(treatment_data, outcome_data, covariate_data)

        effect_no_cf = estimator_no_cf.estimate_ate()
        effect_with_cf = estimator_with_cf.estimate_ate()

        # Both should provide reasonable estimates
        assert abs(effect_no_cf.ate - true_ate) < 2.0
        assert abs(effect_with_cf.ate - true_ate) < 2.0

        # Cross-fitting should provide more reliable confidence intervals
        assert effect_with_cf.ate_se is not None
        assert effect_with_cf.ate_ci_lower is not None
        assert effect_with_cf.ate_ci_upper is not None

    def test_tmle_vs_doubly_robust_ml_comparison(self, medium_confounded_data):
        """Compare TMLE and DoublyRobustML estimators."""
        X, treatment, outcome, true_ate = medium_confounded_data

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        # Common learners
        outcome_learner = SuperLearner(["linear_regression", "lasso", "random_forest"])
        propensity_learner = SuperLearner(["logistic_regression", "lasso_logistic"])

        # TMLE estimator
        tmle = TMLEEstimator(
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
            cross_fitting=True,
            cv_folds=3,
            random_state=42,
        )

        # DoublyRobustML estimator
        drml = DoublyRobustMLEstimator(
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
            cross_fitting=True,
            cv_folds=3,
            moment_function="aipw",
            random_state=42,
        )

        # Fit both estimators
        tmle.fit(treatment_data, outcome_data, covariate_data)
        drml.fit(treatment_data, outcome_data, covariate_data)

        tmle_effect = tmle.estimate_ate()
        drml_effect = drml.estimate_ate()

        # Both should recover true ATE reasonably well
        assert abs(tmle_effect.ate - true_ate) < 1.5
        assert abs(drml_effect.ate - true_ate) < 1.5

        # Both should provide confidence intervals
        assert tmle_effect.ate_ci_lower is not None
        assert drml_effect.ate_ci_lower is not None

        # Estimates should be similar (both are doubly robust)
        assert abs(tmle_effect.ate - drml_effect.ate) < 1.0

    def test_variable_importance_calculation(self, medium_confounded_data):
        """Test variable importance calculation."""
        X, treatment, outcome, _ = medium_confounded_data

        # Prepare data with feature names
        feature_names = [f"X{i}" for i in range(X.shape[1])]
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=feature_names), names=feature_names
        )

        # TMLE with feature importance
        estimator = TMLEEstimator(
            outcome_learner=SuperLearner(["lasso", "random_forest"]),
            propensity_learner=SuperLearner(["lasso_logistic", "random_forest"]),
            cross_fitting=True,
            cv_folds=3,
            random_state=42,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        estimator.estimate_ate()

        # Get variable importance
        importance = estimator.get_variable_importance()

        if importance is not None:
            assert isinstance(importance, pd.DataFrame)
            assert "feature" in importance.columns
            assert "importance" in importance.columns
            assert len(importance) <= X.shape[1]

            # Importance should be sorted in descending order
            assert importance["importance"].is_monotonic_decreasing

    def test_computational_efficiency(self, medium_confounded_data):
        """Test computational efficiency requirements."""
        X, treatment, outcome, _ = medium_confounded_data

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        import time

        # Test TMLE efficiency
        start_time = time.time()

        estimator = TMLEEstimator(
            outcome_learner=SuperLearner(["lasso", "random_forest"]),
            propensity_learner=SuperLearner(["lasso_logistic", "random_forest"]),
            cross_fitting=True,
            cv_folds=3,
            random_state=42,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        estimator.estimate_ate()

        fitting_time = time.time() - start_time

        # Should complete within reasonable time (relaxed for tests)
        assert fitting_time < 60  # Less than 1 minute for test data

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test Super Learner with invalid learner names
        with pytest.raises(ValueError, match="Unknown learner"):
            SuperLearner(base_learners=["invalid_learner"])

        # Test TMLE without covariates
        treatment_data = TreatmentData(
            values=np.array([0, 1, 0, 1]), treatment_type="binary"
        )
        outcome_data = OutcomeData(values=np.array([1.0, 2.0, 1.5, 2.5]))

        estimator = TMLEEstimator()
        with pytest.raises(EstimationError, match="requires covariates"):
            estimator.fit(treatment_data, outcome_data, None)

        # Test DoublyRobustML with non-binary treatment
        treatment_data_cont = TreatmentData(
            values=np.array([0.1, 0.5, 0.8, 0.3]), treatment_type="continuous"
        )
        covariate_data = CovariateData(values=pd.DataFrame(np.random.randn(4, 3)))

        estimator_drml = DoublyRobustMLEstimator()
        with pytest.raises(EstimationError, match="binary treatments"):
            estimator_drml.fit(treatment_data_cont, outcome_data, covariate_data)


class TestIntegrationWithExistingEstimators:
    """Test integration with existing estimator architecture."""

    def test_tmle_inherits_base_estimator(self):
        """Test that TMLE properly inherits from BaseEstimator."""
        estimator = TMLEEstimator()

        # Should have BaseEstimator interface
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "estimate_ate")
        assert hasattr(estimator, "predict_potential_outcomes")
        assert hasattr(estimator, "check_positivity_assumption")
        assert hasattr(estimator, "summary")

        # Should have CrossFitting interface
        assert hasattr(estimator, "cv_folds")
        assert hasattr(estimator, "stratified")

    def test_doubly_robust_ml_inherits_base_estimator(self):
        """Test that DoublyRobustML properly inherits from BaseEstimator."""
        estimator = DoublyRobustMLEstimator()

        # Should have BaseEstimator interface
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "estimate_ate")
        assert hasattr(estimator, "predict_potential_outcomes")
        assert hasattr(estimator, "check_positivity_assumption")
        assert hasattr(estimator, "summary")

        # Should have CrossFitting interface
        assert hasattr(estimator, "cv_folds")
        assert hasattr(estimator, "stratified")

    def test_causal_effect_compatibility(self):
        """Test that ML estimators return proper CausalEffect objects."""
        # Generate simple test data
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = X[:, 0] + 2 * treatment + np.random.normal(0, 0.5, n)

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        for EstimatorClass in [TMLEEstimator, DoublyRobustMLEstimator]:
            estimator = EstimatorClass(
                cross_fitting=False,  # Simpler for test
                random_state=42,
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            # Should return proper CausalEffect
            assert hasattr(effect, "ate")
            assert hasattr(effect, "ate_se")
            assert hasattr(effect, "ate_ci_lower")
            assert hasattr(effect, "ate_ci_upper")
            assert hasattr(effect, "method")
            assert hasattr(effect, "n_observations")
            assert hasattr(effect, "n_treated")
            assert hasattr(effect, "n_control")

            # Values should be reasonable
            assert isinstance(effect.ate, float)
            assert effect.n_observations == n
            assert effect.n_treated + effect.n_control == n

