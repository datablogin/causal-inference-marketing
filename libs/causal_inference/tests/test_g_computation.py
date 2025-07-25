"""Tests for G-computation estimator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from causal_inference.core.base import (
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.g_computation import GComputationEstimator


class TestGComputationEstimator:
    """Test cases for G-computation estimator."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)

        # Create synthetic data with known treatment effect
        n = 200

        # Generate covariates
        self.X = np.random.randn(n, 3)

        # Generate treatment (binary)
        # Make treatment slightly correlated with covariates (selection bias)
        treatment_logits = 0.5 * self.X[:, 0] + 0.3 * self.X[:, 1] + np.random.randn(n) * 0.5
        self.treatment_binary = (treatment_logits > 0).astype(int)

        # Generate outcome with known treatment effect of 2.0
        true_ate = 2.0
        noise = np.random.randn(n) * 0.5
        self.outcome_continuous = (
            2 * self.X[:, 0] +         # covariate effect
            1.5 * self.X[:, 1] +       # covariate effect
            0.8 * self.X[:, 2] +       # covariate effect
            true_ate * self.treatment_binary +  # treatment effect
            noise
        )

        # Generate binary outcome
        outcome_logits = (
            1.2 * self.X[:, 0] +
            0.8 * self.X[:, 1] +
            0.5 * self.X[:, 2] +
            1.5 * self.treatment_binary +  # treatment effect on log-odds
            noise * 0.3
        )
        self.outcome_binary = (outcome_logits > 0).astype(int)

        # Create data objects
        self.treatment_data = TreatmentData(
            values=pd.Series(self.treatment_binary),
            name="treatment",
            treatment_type="binary"
        )

        self.outcome_data_continuous = OutcomeData(
            values=pd.Series(self.outcome_continuous),
            name="outcome",
            outcome_type="continuous"
        )

        self.outcome_data_binary = OutcomeData(
            values=pd.Series(self.outcome_binary),
            name="outcome",
            outcome_type="binary"
        )

        self.covariate_data = CovariateData(
            values=pd.DataFrame(self.X, columns=["X1", "X2", "X3"]),
            names=["X1", "X2", "X3"]
        )

        # Store true treatment effect for validation
        self.true_ate = true_ate

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = GComputationEstimator(
            model_type="linear",
            bootstrap_samples=100,
            confidence_level=0.95,
            random_state=42,
            verbose=True
        )

        assert estimator.model_type == "linear"
        assert estimator.bootstrap_samples == 100
        assert estimator.confidence_level == 0.95
        assert estimator.random_state == 42
        assert estimator.verbose is True
        assert estimator.outcome_model is None
        assert not estimator.is_fitted

    def test_model_selection(self):
        """Test automatic model selection."""
        estimator = GComputationEstimator(model_type="auto")

        # Test continuous outcome
        linear_model = estimator._select_model("continuous")
        assert linear_model.__class__.__name__ == "LinearRegression"

        # Test binary outcome
        logistic_model = estimator._select_model("binary")
        assert logistic_model.__class__.__name__ == "LogisticRegression"

        # Test manual selection
        estimator_rf = GComputationEstimator(model_type="random_forest")
        rf_model = estimator_rf._select_model("continuous")
        assert rf_model.__class__.__name__ == "RandomForestRegressor"

    def test_feature_preparation(self):
        """Test feature matrix preparation."""
        estimator = GComputationEstimator()

        # Test with treatment only
        features = estimator._prepare_features(self.treatment_data)
        assert features.shape == (200, 1)
        assert "treatment" in features.columns

        # Test with treatment and covariates
        features = estimator._prepare_features(self.treatment_data, self.covariate_data)
        assert features.shape == (200, 4)
        expected_cols = ["treatment", "X1", "X2", "X3"]
        assert all(col in features.columns for col in expected_cols)

    def test_fitting_continuous_outcome(self):
        """Test fitting with continuous outcome."""
        estimator = GComputationEstimator(
            model_type="linear",
            bootstrap_samples=0,  # Skip bootstrap for speed
            random_state=42
        )

        # Fit the model
        result = estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        # Check fitting results
        assert result is estimator
        assert estimator.is_fitted
        assert estimator.outcome_model is not None
        assert estimator._model_features is not None
        assert len(estimator._model_features) == 4

    def test_fitting_binary_outcome(self):
        """Test fitting with binary outcome."""
        estimator = GComputationEstimator(
            model_type="logistic",
            bootstrap_samples=0,
            random_state=42
        )

        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_binary,
            covariates=self.covariate_data
        )

        assert estimator.is_fitted
        assert estimator.outcome_model.__class__.__name__ == "LogisticRegression"

    def test_ate_estimation_continuous(self):
        """Test ATE estimation with continuous outcome."""
        estimator = GComputationEstimator(
            model_type="linear",
            bootstrap_samples=100,  # Use bootstrap for CI
            confidence_level=0.95,
            random_state=42,
            verbose=True
        )

        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        effect = estimator.estimate_ate()

        # Check effect structure
        assert effect.method == "G-computation"
        assert effect.n_observations == 200
        assert effect.confidence_level == 0.95
        assert effect.bootstrap_samples == 100

        # Check that estimate is reasonably close to true effect (within 20%)
        assert abs(effect.ate - self.true_ate) < 0.4  # Allow some estimation error

        # Check confidence intervals
        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None
        assert effect.ate_ci_lower < effect.ate < effect.ate_ci_upper

        # Check standard error
        assert effect.ate_se is not None
        assert effect.ate_se > 0

    def test_ate_estimation_binary(self):
        """Test ATE estimation with binary outcome."""
        estimator = GComputationEstimator(
            model_type="logistic",
            bootstrap_samples=50,  # Smaller for speed
            random_state=42
        )

        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_binary,
            covariates=self.covariate_data
        )

        effect = estimator.estimate_ate()

        # Check basic structure
        assert effect.method == "G-computation"
        assert isinstance(effect.ate, float)
        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None

    def test_counterfactual_prediction(self):
        """Test counterfactual outcome prediction."""
        estimator = GComputationEstimator(model_type="linear", random_state=42)

        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        # Predict under treatment = 0
        y0_pred = estimator._predict_counterfactuals(0)
        assert len(y0_pred) == 200

        # Predict under treatment = 1
        y1_pred = estimator._predict_counterfactuals(1)
        assert len(y1_pred) == 200

        # Treatment effect should be positive on average
        assert np.mean(y1_pred - y0_pred) > 0

    def test_potential_outcomes_prediction(self):
        """Test potential outcomes prediction interface."""
        estimator = GComputationEstimator(model_type="linear", random_state=42)

        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        # Test prediction on new data
        new_X = np.random.randn(50, 3)
        new_treatment = np.random.binomial(1, 0.5, 50)

        y0_pred, y1_pred = estimator.predict_potential_outcomes(
            treatment_values=new_treatment,
            covariates=new_X
        )

        assert len(y0_pred) == 50
        assert len(y1_pred) == 50
        assert isinstance(y0_pred, np.ndarray)
        assert isinstance(y1_pred, np.ndarray)

    def test_estimation_without_fitting(self):
        """Test that estimation fails without fitting."""
        estimator = GComputationEstimator()

        with pytest.raises(EstimationError, match="must be fitted"):
            estimator.estimate_ate()

    def test_prediction_without_fitting(self):
        """Test that prediction fails without fitting."""
        estimator = GComputationEstimator()

        with pytest.raises(EstimationError, match="must be fitted"):
            estimator.predict_potential_outcomes(
                treatment_values=np.array([0, 1]),
                covariates=np.random.randn(2, 3)
            )

    def test_fitting_without_covariates(self):
        """Test fitting and estimation without covariates."""
        estimator = GComputationEstimator(
            model_type="linear",
            bootstrap_samples=0,
            random_state=42
        )

        # Fit with treatment only
        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_continuous
        )

        assert estimator.is_fitted

        # Estimate effect
        effect = estimator.estimate_ate()
        assert isinstance(effect.ate, float)

        # The estimate might be biased without covariates, but should still work
        assert effect.method == "G-computation"

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        estimator = GComputationEstimator(
            model_type="linear",
            bootstrap_samples=50,
            confidence_level=0.90,
            random_state=42
        )

        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        # Test bootstrap CI method directly
        ci_lower, ci_upper, boot_estimates = estimator._bootstrap_confidence_interval()

        assert ci_lower is not None
        assert ci_upper is not None
        assert boot_estimates is not None
        assert len(boot_estimates) <= 50  # Some bootstrap samples might fail
        assert ci_lower < ci_upper

    def test_random_forest_model(self):
        """Test using random forest models."""
        estimator = GComputationEstimator(
            model_type="random_forest",
            model_params={"n_estimators": 10, "max_depth": 3},  # Small for speed
            bootstrap_samples=0,
            random_state=42
        )

        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        assert estimator.outcome_model.__class__.__name__ == "RandomForestRegressor"

        effect = estimator.estimate_ate()
        assert isinstance(effect.ate, float)

    def test_categorical_treatment(self):
        """Test with categorical treatment."""
        # Create categorical treatment with 3 levels
        n = 200
        treatment_cat = np.random.choice([0, 1, 2], size=n)

        treatment_data_cat = TreatmentData(
            values=pd.Series(treatment_cat),
            name="treatment_cat",
            treatment_type="categorical",
            categories=[0, 1, 2]
        )

        estimator = GComputationEstimator(
            model_type="linear",
            bootstrap_samples=0,
            random_state=42
        )

        estimator.fit(
            treatment=treatment_data_cat,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        effect = estimator.estimate_ate()
        assert isinstance(effect.ate, float)

    def test_continuous_treatment(self):
        """Test with continuous treatment."""
        # Create continuous treatment
        n = 200
        treatment_cont = np.random.uniform(0, 5, size=n)

        treatment_data_cont = TreatmentData(
            values=pd.Series(treatment_cont),
            name="treatment_cont",
            treatment_type="continuous"
        )

        estimator = GComputationEstimator(
            model_type="linear",
            bootstrap_samples=0,
            random_state=42
        )

        estimator.fit(
            treatment=treatment_data_cont,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        effect = estimator.estimate_ate()
        assert isinstance(effect.ate, float)

    def test_model_with_sklearn_datasets(self):
        """Test with sklearn generated datasets."""
        # Generate regression dataset
        X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)

        # Create binary treatment from first feature
        treatment = (X[:, 0] > np.median(X[:, 0])).astype(int)

        treatment_data = TreatmentData(
            values=pd.Series(treatment),
            treatment_type="binary"
        )

        outcome_data = OutcomeData(
            values=pd.Series(y),
            outcome_type="continuous"
        )

        covariate_data = CovariateData(
            values=pd.DataFrame(X[:, 1:], columns=["X1", "X2", "X3"]),
            names=["X1", "X2", "X3"]
        )

        estimator = GComputationEstimator(
            model_type="auto",
            bootstrap_samples=20,
            random_state=42
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        assert isinstance(effect.ate, float)
        assert effect.n_observations == 100

    def test_edge_case_no_treatment_variation(self):
        """Test handling of edge case with no treatment variation."""
        # Create treatment with no variation (all treated)
        treatment_no_var = TreatmentData(
            values=pd.Series([1] * 100),
            treatment_type="binary"
        )

        outcome_no_var = OutcomeData(
            values=pd.Series(np.random.randn(100)),
            outcome_type="continuous"
        )

        estimator = GComputationEstimator(bootstrap_samples=0, random_state=42)

        # This should raise an error during validation in the base class
        with pytest.raises(Exception):  # Could be DataValidationError or EstimationError
            estimator.fit(treatment_no_var, outcome_no_var)

    def test_summary_output(self):
        """Test summary output formatting."""
        estimator = GComputationEstimator(
            model_type="linear",
            bootstrap_samples=10,
            random_state=42,
            verbose=False
        )

        estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data_continuous,
            covariates=self.covariate_data
        )

        # Generate effect for summary
        estimator.estimate_ate()

        summary = estimator.summary()

        assert "GComputationEstimator" in summary
        assert "Fitted: True" in summary
        assert "ATE:" in summary
        assert "95% CI:" in summary


if __name__ == "__main__":
    pytest.main([__file__])
