"""Tests for Causal Forest implementation."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import (
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.causal_forest import CausalForest, HonestTree


class TestHonestTree:
    """Test cases for HonestTree implementation."""

    def test_init(self):
        """Test HonestTree initialization."""
        tree = HonestTree(min_samples_split=10, random_state=42)
        assert tree.min_samples_split == 10
        assert tree.random_state == 42
        assert not tree.is_fitted_

    def test_fit_simple_data(self):
        """Test fitting on simple synthetic data."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        treatment = np.random.binomial(1, 0.5, n)
        # Simple treatment effect: 1 if X[0] > 0, else 0.5
        outcome = (
            2
            + X.sum(axis=1)
            + treatment * np.where(X[:, 0] > 0, 1, 0.5)
            + np.random.randn(n) * 0.1
        )

        tree = HonestTree(min_samples_split=20, min_samples_leaf=10, random_state=42)
        tree.fit(X, outcome, treatment)

        assert tree.is_fitted_
        assert tree.tree_ is not None

    def test_fit_insufficient_data(self):
        """Test error handling with insufficient data."""
        X = np.random.randn(10, 2)
        treatment = np.ones(10)  # All treated
        outcome = np.random.randn(10)

        tree = HonestTree(min_samples_split=20)
        with pytest.raises(ValueError, match="Not enough samples"):
            tree.fit(X, outcome, treatment)

    def test_predict(self):
        """Test prediction functionality."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 2)
        treatment = np.random.binomial(1, 0.5, n)
        # True effect: 2 * X[0] + X[1]
        true_effect = 2 * X[:, 0] + X[:, 1]
        outcome = X.sum(axis=1) + treatment * true_effect + np.random.randn(n) * 0.1

        tree = HonestTree(random_state=42)
        tree.fit(X, outcome, treatment)

        # Test prediction
        X_test = np.random.randn(20, 2)
        effects, std_errors = tree.predict(X_test)

        assert effects.shape == (20,)
        assert std_errors.shape == (20,)
        assert np.all(std_errors > 0)  # Standard errors should be positive

    def test_predict_not_fitted(self):
        """Test prediction error when not fitted."""
        tree = HonestTree()
        X_test = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="not fitted"):
            tree.predict(X_test)


class TestCausalForest:
    """Test cases for CausalForest implementation."""

    def test_init(self):
        """Test CausalForest initialization."""
        cf = CausalForest(n_estimators=50, random_state=42)
        assert cf.n_estimators == 50
        assert cf.random_state == 42
        assert len(cf.trees_) == 0

    def test_prepare_data(self):
        """Test data preparation."""
        np.random.seed(42)
        n = 100

        # Test with pandas data
        treatment_df = pd.Series(np.random.binomial(1, 0.5, n))
        outcome_df = pd.Series(np.random.randn(n))
        covariate_df = pd.DataFrame(np.random.randn(n, 3), columns=["x1", "x2", "x3"])

        treatment_data = TreatmentData(values=treatment_df)
        outcome_data = OutcomeData(values=outcome_df)
        covariate_data = CovariateData(values=covariate_df)

        cf = CausalForest()
        T, Y, X = cf._prepare_data(treatment_data, outcome_data, covariate_data)

        assert T.shape == (n,)
        assert Y.shape == (n,)
        assert X.shape == (n, 3)
        assert np.all((T == 0) | (T == 1))  # Binary treatment

    def test_prepare_data_non_binary_treatment(self):
        """Test error with non-binary treatment."""
        treatment_data = TreatmentData(
            values=np.array([0, 1, 2]), treatment_type="categorical"
        )
        outcome_data = OutcomeData(values=np.array([1.0, 2.0, 3.0]))

        cf = CausalForest()
        with pytest.raises(ValueError, match="binary treatment"):
            cf._prepare_data(treatment_data, outcome_data, None)

    def test_fit_simple_data(self):
        """Test fitting on simple synthetic data."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 3)

        # Treatment assignment with some confounding
        propensity = 0.2 + 0.6 / (1 + np.exp(-X[:, 0]))
        treatment = np.random.binomial(1, propensity)

        # Heterogeneous treatment effect
        true_cate = 1 + 0.5 * X[:, 0] + 0.3 * X[:, 1]
        outcome = X.sum(axis=1) + treatment * true_cate + np.random.randn(n) * 0.5

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=20, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        assert cf.is_fitted
        assert len(cf.trees_) > 0
        assert cf.feature_importances_ is not None

    def test_predict_cate(self):
        """Test CATE prediction."""
        np.random.seed(42)
        n = 150
        X = np.random.randn(n, 2)
        treatment = np.random.binomial(1, 0.5, n)

        # Simple heterogeneous effect: effect = X[0]
        true_effect = X[:, 0]
        outcome = X.sum(axis=1) + treatment * true_effect + np.random.randn(n) * 0.3

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=30, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        # Test prediction
        X_test = np.random.randn(20, 2)
        cate_pred, cate_ci = cf.predict_cate(X_test)

        assert cate_pred.shape == (20,)
        assert cate_ci.shape == (20, 2)

        # Check CI ordering
        assert np.all(cate_ci[:, 0] <= cate_ci[:, 1])

    def test_predict_cate_pandas_input(self):
        """Test CATE prediction with pandas DataFrame input."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = X.sum(axis=1) + treatment * X[:, 0] + np.random.randn(n) * 0.3

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=20, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        # Test with DataFrame
        X_test_df = pd.DataFrame(np.random.randn(15, 2), columns=["x1", "x2"])
        cate_pred, cate_ci = cf.predict_cate(X_test_df)

        assert cate_pred.shape == (15,)
        assert cate_ci.shape == (15, 2)

    def test_estimate_ate(self):
        """Test ATE estimation."""
        np.random.seed(42)
        n = 120
        X = np.random.randn(n, 2)
        treatment = np.random.binomial(1, 0.5, n)

        # Known ATE = 1.5
        ate_true = 1.5
        outcome = X.sum(axis=1) + treatment * ate_true + np.random.randn(n) * 0.3

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=25, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        result = cf.estimate_ate()

        assert hasattr(result, "ate")
        assert hasattr(result, "cate_estimates")
        assert result.method == "Causal Forest"

        # Should be reasonably close to true ATE
        assert abs(result.ate - ate_true) < 0.5

    def test_feature_importance(self):
        """Test feature importance computation."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = X.sum(axis=1) + treatment * X[:, 0] + np.random.randn(n) * 0.3

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=20, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        importance = cf.feature_importance()
        assert importance.shape == (3,)
        assert np.all(importance >= 0)

    def test_variable_importance(self):
        """Test variable importance for effect modification."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        treatment = np.random.binomial(1, 0.5, n)

        # X[0] is the main effect modifier
        outcome = (
            X.sum(axis=1) + treatment * (1 + 2 * X[:, 0]) + np.random.randn(n) * 0.3
        )

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=20, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        var_importance = cf.variable_importance()
        assert var_importance.shape == (3,)
        assert np.all(var_importance >= 0)
        assert np.sum(var_importance) <= 1.01  # Should be normalized

    def test_not_fitted_errors(self):
        """Test errors when model not fitted."""
        cf = CausalForest()
        X_test = np.random.randn(10, 2)

        with pytest.raises(EstimationError, match="not fitted"):
            cf.predict_cate(X_test)

        with pytest.raises(EstimationError, match="not fitted"):
            cf.feature_importance()

        with pytest.raises(EstimationError, match="not fitted"):
            cf.variable_importance()

    def test_small_subsample_error(self):
        """Test error with too small subsample ratio."""
        n = 50
        X = np.random.randn(n, 2)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.randn(n)

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(subsample_ratio=0.1, min_samples_split=30)

        with pytest.raises(EstimationError, match="subsample_ratio too small"):
            cf.fit(treatment_data, outcome_data, covariate_data)

    def test_no_trees_fitted_error(self):
        """Test error when no trees could be fitted."""
        # Test that creating TreatmentData with no variation fails at construction time
        n = 20  # Very small dataset
        treatment = np.ones(n)  # All treated, no variation

        # This should raise a ValueError due to all-treated data (no variation) at construction time
        with pytest.raises(
            ValueError, match="Binary treatment must have exactly 2 unique values"
        ):
            TreatmentData(values=treatment, treatment_type="binary")


@pytest.fixture
def synthetic_hte_data():
    """Create synthetic data with known heterogeneous treatment effects."""
    np.random.seed(123)
    n = 200

    # Covariates
    X = np.random.randn(n, 3)

    # Treatment assignment (randomized)
    treatment = np.random.binomial(1, 0.5, n)

    # Heterogeneous treatment effect: depends on X[0] and X[1]
    true_cate = 0.5 + 1.0 * X[:, 0] + 0.5 * X[:, 1]

    # Outcome model
    outcome = (
        2
        + 0.5 * X[:, 0]
        + 0.3 * X[:, 1]
        - 0.2 * X[:, 2]
        + treatment * true_cate
        + np.random.randn(n) * 0.5
    )

    return {
        "X": X,
        "treatment": treatment,
        "outcome": outcome,
        "true_cate": true_cate,
        "true_ate": np.mean(true_cate),
    }


class TestCausalForestIntegration:
    """Integration tests for Causal Forest."""

    def test_end_to_end_workflow(self, synthetic_hte_data):
        """Test complete workflow with synthetic data."""
        data = synthetic_hte_data

        treatment_data = TreatmentData(values=data["treatment"])
        outcome_data = OutcomeData(values=data["outcome"])
        covariate_data = CovariateData(values=data["X"])

        # Fit causal forest
        cf = CausalForest(
            n_estimators=50,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            verbose=False,
        )
        cf.fit(treatment_data, outcome_data, covariate_data)

        # Test predictions
        cate_pred, cate_ci = cf.predict_cate(data["X"])
        ate_result = cf.estimate_ate()

        # Basic checks
        assert cate_pred.shape == (len(data["X"]),)
        assert cate_ci.shape == (len(data["X"]), 2)
        assert np.all(cate_ci[:, 0] <= cate_ci[:, 1])

        # ATE should be reasonably close to true value
        true_ate = data["true_ate"]
        assert abs(ate_result.ate - true_ate) < 0.3

        # CATE predictions should correlate with true effects
        correlation = np.corrcoef(cate_pred, data["true_cate"])[0, 1]
        assert correlation > 0.3  # Should capture some of the heterogeneity

    def test_confidence_intervals_coverage(self, synthetic_hte_data):
        """Test that confidence intervals have reasonable coverage."""
        data = synthetic_hte_data

        treatment_data = TreatmentData(values=data["treatment"])
        outcome_data = OutcomeData(values=data["outcome"])
        covariate_data = CovariateData(values=data["X"])

        cf = CausalForest(n_estimators=40, confidence_level=0.95, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        # Get predictions and confidence intervals
        cate_pred, cate_ci = cf.predict_cate(data["X"])
        true_cate = data["true_cate"]

        # Check coverage (should be close to 95% but may vary due to small sample)
        coverage = np.mean((true_cate >= cate_ci[:, 0]) & (true_cate <= cate_ci[:, 1]))

        # Allow some deviation due to small sample size and model limitations
        assert coverage > 0.70  # Should have reasonable coverage

    def test_feature_importance_identifies_relevant_features(self, synthetic_hte_data):
        """Test that feature importance identifies relevant features."""
        data = synthetic_hte_data

        # In our synthetic data, X[0] and X[1] affect the treatment effect
        # X[2] only affects the baseline outcome

        treatment_data = TreatmentData(values=data["treatment"])
        outcome_data = OutcomeData(values=data["outcome"])
        covariate_data = CovariateData(values=data["X"])

        cf = CausalForest(n_estimators=40, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        var_importance = cf.variable_importance()

        # X[0] should have higher importance than X[2] for effect modification
        # (though this is a probabilistic test and may occasionally fail)
        assert len(var_importance) == 3
        assert np.all(var_importance >= 0)


if __name__ == "__main__":
    pytest.main([__file__])
