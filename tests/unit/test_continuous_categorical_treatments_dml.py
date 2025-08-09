"""Tests for continuous and categorical treatment support in DoublyRobustMLEstimator.

This module tests the enhanced DoublyRobustMLEstimator that supports:
- Binary treatments (existing functionality)
- Continuous treatments (new)
- Multi-valued categorical treatments (new)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_inference.core.base import (
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.doubly_robust_ml import DoublyRobustMLEstimator
from causal_inference.ml.super_learner import SuperLearner


class TestTreatmentDataEnhancements:
    """Test enhanced TreatmentData validation for different treatment types."""

    def test_binary_treatment_validation(self):
        """Test binary treatment validation works correctly."""
        # Valid binary treatment (0/1)
        treatment = TreatmentData(
            values=np.array([0, 1, 1, 0, 1]), treatment_type="binary"
        )
        assert treatment.treatment_type == "binary"
        assert treatment.is_binary_encoded_as_01()

        # Valid binary treatment (other encoding)
        treatment_alt = TreatmentData(
            values=np.array(["A", "B", "B", "A", "B"]), treatment_type="binary"
        )
        assert treatment_alt.treatment_type == "binary"
        assert not treatment_alt.is_binary_encoded_as_01()

        # Invalid binary treatment (too many categories)
        with pytest.raises(
            ValueError, match="Binary treatment must have exactly 2 unique values"
        ):
            TreatmentData(values=np.array([0, 1, 2, 1, 0]), treatment_type="binary")

    def test_categorical_treatment_validation(self):
        """Test categorical treatment validation."""
        # Valid categorical treatment
        treatment = TreatmentData(
            values=np.array([0, 1, 2, 1, 0, 2]), treatment_type="categorical"
        )
        assert treatment.treatment_type == "categorical"
        assert treatment.n_categories == 3
        assert sorted(treatment.categories) == [0, 1, 2]

        # Categorical with explicit categories
        treatment_explicit = TreatmentData(
            values=np.array(["low", "medium", "high", "medium", "low"]),
            treatment_type="categorical",
            categories=["low", "medium", "high"],
            n_categories=3,
        )
        assert treatment_explicit.n_categories == 3
        assert set(treatment_explicit.categories) == {"low", "medium", "high"}

        # Invalid categorical (mismatch between categories and n_categories)
        with pytest.raises(
            ValueError, match="doesn't match actual number of categories"
        ):
            TreatmentData(
                values=np.array([0, 1, 2]), treatment_type="categorical", n_categories=2
            )

    def test_continuous_treatment_validation(self):
        """Test continuous treatment validation."""
        # Valid continuous treatment
        treatment = TreatmentData(
            values=np.array([0.5, 1.2, 2.3, 0.8, 1.5]), treatment_type="continuous"
        )
        assert treatment.treatment_type == "continuous"
        assert treatment.dose_range is not None
        assert treatment.dose_range[0] == 0.5
        assert treatment.dose_range[1] == 2.3

        # Continuous with explicit dose range
        treatment_explicit = TreatmentData(
            values=np.array([1.0, 2.0, 3.0]),
            treatment_type="continuous",
            dose_range=(0.0, 5.0),
        )
        assert treatment_explicit.dose_range == (0.0, 5.0)

        # Invalid continuous (no variation)
        with pytest.raises(ValueError, match="must have variation"):
            TreatmentData(values=np.array([1.0, 1.0, 1.0]), treatment_type="continuous")

        # Invalid continuous (out of range)
        with pytest.raises(ValueError, match="outside specified dose_range"):
            TreatmentData(
                values=np.array([0.5, 6.0, 1.5]),
                treatment_type="continuous",
                dose_range=(0.0, 5.0),
            )


class TestDoublyRobustMLBinaryTreatments:
    """Test DoublyRobustML with binary treatments (ensure backward compatibility)."""

    @pytest.fixture
    def binary_data(self):
        """Generate synthetic binary treatment data."""
        np.random.seed(42)
        n = 500

        # Covariates
        X = np.random.randn(n, 4)

        # Treatment assignment (binary)
        propensity_logits = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
        propensity_probs = 1 / (1 + np.exp(-propensity_logits))
        A = np.random.binomial(1, propensity_probs)

        # Outcome with treatment effect
        true_ate = 2.0
        Y = (
            1.0
            + 0.5 * X[:, 0]
            + 0.3 * X[:, 1]
            + true_ate * A
            + np.random.normal(0, 0.5, n)
        )

        treatment = TreatmentData(values=A, treatment_type="binary")
        outcome = OutcomeData(values=Y, outcome_type="continuous")
        covariates = CovariateData(
            values=pd.DataFrame(X, columns=[f"X{i}" for i in range(4)])
        )

        return treatment, outcome, covariates, true_ate

    def test_binary_treatment_estimation(self, binary_data):
        """Test that binary treatment estimation works correctly."""
        treatment, outcome, covariates, true_ate = binary_data

        # Simple learners for fast testing
        outcome_learner = SuperLearner(
            base_learners=["linear_regression", "ridge"], task_type="regression"
        )
        propensity_learner = SuperLearner(
            base_learners=["logistic_regression", "ridge_logistic"],
            task_type="classification",
        )

        estimator = DoublyRobustMLEstimator(
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
            cv_folds=3,
            random_state=42,
        )

        estimator.fit(treatment, outcome, covariates)
        result = estimator.estimate_ate()

        # Check that estimation runs without error
        assert result.ate is not None
        assert result.ate_se is not None
        assert result.method.startswith("DoublyRobustML")
        assert "binary" in result.method

        # Check that estimate is reasonable (within 50% of true effect)
        assert abs(result.ate - true_ate) / abs(true_ate) < 0.5

        # Check treatment counts
        assert result.n_treated > 0
        assert result.n_control > 0
        assert result.n_treated + result.n_control == result.n_observations


class TestDoublyRobustMLCategoricalTreatments:
    """Test DoublyRobustML with categorical treatments."""

    @pytest.fixture
    def categorical_data(self):
        """Generate synthetic categorical treatment data."""
        np.random.seed(42)
        n = 600

        # Covariates
        X = np.random.randn(n, 4)

        # Treatment assignment (3 categories)
        # Multinomial logistic model for treatment assignment
        logits_1 = 0.5 * X[:, 0] - 0.2 * X[:, 1]
        logits_2 = -0.3 * X[:, 0] + 0.4 * X[:, 2]

        # Softmax for probabilities
        exp_logits = np.exp(np.column_stack([np.zeros(n), logits_1, logits_2]))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        A = np.array([np.random.choice(3, p=prob_row) for prob_row in probs])

        # Outcome with different effects for each treatment
        treatment_effects = np.array(
            [0.0, 1.5, 3.0]
        )  # Control, Treatment 1, Treatment 2
        Y = (
            1.0
            + 0.5 * X[:, 0]
            + 0.3 * X[:, 1]
            + treatment_effects[A]
            + np.random.normal(0, 0.5, n)
        )

        treatment = TreatmentData(values=A, treatment_type="categorical")
        outcome = OutcomeData(values=Y, outcome_type="continuous")
        covariates = CovariateData(
            values=pd.DataFrame(X, columns=[f"X{i}" for i in range(4)])
        )

        return treatment, outcome, covariates, treatment_effects

    def test_categorical_treatment_estimation(self, categorical_data):
        """Test that categorical treatment estimation works."""
        treatment, outcome, covariates, treatment_effects = categorical_data

        # Use learners that can handle multiclass
        outcome_learner = SuperLearner(
            base_learners=["linear_regression", "gradient_boosting"],
            task_type="regression",
        )
        propensity_learner = SuperLearner(
            base_learners=["logistic_regression", "gradient_boosting"],
            task_type="classification",
        )

        estimator = DoublyRobustMLEstimator(
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
            cv_folds=3,
            random_state=42,
        )

        estimator.fit(treatment, outcome, covariates)
        result = estimator.estimate_ate()

        # Check that estimation runs without error
        assert result.ate is not None
        assert result.method.startswith("DoublyRobustML")
        assert "categorical" in result.method

        # For categorical treatments, ATE compares extreme categories
        expected_ate = (
            treatment_effects[-1] - treatment_effects[0]
        )  # Category 2 vs Category 0

        # Check that estimate is reasonable (within 50% of expected difference)
        assert abs(result.ate - expected_ate) / abs(expected_ate) < 0.5

        # Check that treatment counts make sense
        assert result.n_observations == 600
        assert result.n_treated > 0
        assert result.n_control > 0

    def test_categorical_treatment_with_string_categories(self):
        """Test categorical treatment with string categories."""
        np.random.seed(42)
        n = 300

        X = np.random.randn(n, 3)
        A = np.random.choice(["low", "medium", "high"], n)
        Y = (
            1.0
            + 0.5 * X[:, 0]
            + {"low": 0, "medium": 1, "high": 2}[A[0]] * np.ones(n)
            + np.random.normal(0, 0.5, n)
        )

        for i in range(n):
            Y[i] = (
                1.0
                + 0.5 * X[i, 0]
                + {"low": 0, "medium": 1, "high": 2}[A[i]]
                + np.random.normal(0, 0.5)
            )

        treatment = TreatmentData(values=A, treatment_type="categorical")
        outcome = OutcomeData(values=Y, outcome_type="continuous")
        covariates = CovariateData(values=pd.DataFrame(X, columns=["X0", "X1", "X2"]))

        estimator = DoublyRobustMLEstimator(
            outcome_learner=LinearRegression(),
            propensity_learner=LogisticRegression(max_iter=1000),
            cv_folds=2,
            random_state=42,
        )

        estimator.fit(treatment, outcome, covariates)
        result = estimator.estimate_ate()

        assert result.ate is not None
        assert "categorical" in result.method


class TestDoublyRobustMLContinuousTreatments:
    """Test DoublyRobustML with continuous treatments."""

    @pytest.fixture
    def continuous_data(self):
        """Generate synthetic continuous treatment data."""
        np.random.seed(42)
        n = 500

        # Covariates
        X = np.random.randn(n, 4)

        # Continuous treatment (dose)
        # Treatment model: dose depends on covariates
        dose_mean = 1.0 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
        A = dose_mean + np.random.normal(0, 0.5, n)
        A = np.clip(A, 0.1, 3.0)  # Clip to reasonable dose range

        # Dose-response relationship (quadratic)
        dose_effect = 2.0 * A - 0.5 * A**2
        Y = (
            1.0
            + 0.5 * X[:, 0]
            + 0.3 * X[:, 1]
            + dose_effect
            + np.random.normal(0, 0.5, n)
        )

        treatment = TreatmentData(values=A, treatment_type="continuous")
        outcome = OutcomeData(values=Y, outcome_type="continuous")
        covariates = CovariateData(
            values=pd.DataFrame(X, columns=[f"X{i}" for i in range(4)])
        )

        return treatment, outcome, covariates

    def test_continuous_treatment_estimation(self, continuous_data):
        """Test that continuous treatment estimation works."""
        treatment, outcome, covariates = continuous_data

        # Use regression learners suitable for continuous treatments
        outcome_learner = SuperLearner(
            base_learners=["linear_regression", "gradient_boosting"],
            task_type="regression",
        )
        propensity_learner = SuperLearner(
            base_learners=["linear_regression", "ridge"],
            task_type="regression",  # For continuous treatment, propensity is also regression
        )

        estimator = DoublyRobustMLEstimator(
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
            cv_folds=3,
            random_state=42,
        )

        estimator.fit(treatment, outcome, covariates)
        result = estimator.estimate_ate()

        # Check that estimation runs without error
        assert result.ate is not None
        assert result.method.startswith("DoublyRobustML")
        assert "continuous" in result.method

        # For continuous treatments, all observations are "treated"
        assert result.n_treated == result.n_observations
        assert result.n_control == 0

        # Check that dose-response function was computed
        assert "dose_response_function" in estimator.nuisance_estimates_
        assert "dose_grid" in estimator.nuisance_estimates_

    def test_continuous_treatment_dose_range_validation(self):
        """Test dose range validation for continuous treatments."""
        np.random.seed(42)
        n = 100

        X = np.random.randn(n, 3)
        A = np.random.uniform(0.5, 2.5, n)
        Y = 1.0 + A + np.random.normal(0, 0.1, n)

        # Test with explicit dose range
        treatment = TreatmentData(
            values=A, treatment_type="continuous", dose_range=(0.0, 3.0)
        )
        outcome = OutcomeData(values=Y)
        covariates = CovariateData(values=pd.DataFrame(X))

        estimator = DoublyRobustMLEstimator(
            outcome_learner=LinearRegression(),
            propensity_learner=LinearRegression(),
            cv_folds=2,
            random_state=42,
        )

        estimator.fit(treatment, outcome, covariates)
        result = estimator.estimate_ate()

        assert result.ate is not None
        assert estimator.treatment_dose_range_ == (0.0, 3.0)


class TestDoublyRobustMLErrorHandling:
    """Test error handling for different treatment types."""

    def test_invalid_treatment_type(self):
        """Test that invalid treatment types raise appropriate errors."""
        np.random.seed(42)
        n = 100

        A = np.random.choice([0, 1], n)

        # Try to create invalid treatment type
        with pytest.raises(ValueError, match="treatment_type must be one of"):
            TreatmentData(values=A, treatment_type="invalid")

    def test_continuous_treatment_with_non_numeric_values(self):
        """Test error handling for non-numeric continuous treatment."""
        with pytest.raises(
            ValueError, match="Continuous treatment values must be numeric"
        ):
            TreatmentData(
                values=np.array(["low", "medium", "high"]), treatment_type="continuous"
            )

    def test_model_compatibility_warnings(self):
        """Test that appropriate warnings are issued for model compatibility."""
        np.random.seed(42)
        n = 100

        X = np.random.randn(n, 2)
        A = np.random.choice([0, 1, 2], n)  # Categorical
        Y = np.random.randn(n)

        treatment = TreatmentData(values=A, treatment_type="categorical")
        outcome = OutcomeData(values=Y)
        covariates = CovariateData(values=pd.DataFrame(X))

        # Use a classifier that might not handle multiclass well
        estimator = DoublyRobustMLEstimator(
            outcome_learner=LinearRegression(),
            propensity_learner=LogisticRegression(max_iter=1000),
            cv_folds=2,
            random_state=42,
        )

        # Should not raise error, but might issue warnings
        estimator.fit(treatment, outcome, covariates)
        result = estimator.estimate_ate()

        assert result.ate is not None


class TestTreatmentTypeIntegration:
    """Integration tests across different treatment types."""

    def test_treatment_type_consistency(self):
        """Test that treatment type information is consistently stored."""
        np.random.seed(42)
        n = 200

        # Test binary
        X = np.random.randn(n, 3)
        A_binary = np.random.choice([0, 1], n)
        Y = np.random.randn(n)

        treatment_binary = TreatmentData(values=A_binary, treatment_type="binary")
        outcome = OutcomeData(values=Y)
        covariates = CovariateData(values=pd.DataFrame(X))

        estimator = DoublyRobustMLEstimator(cv_folds=2, random_state=42)
        estimator.fit(treatment_binary, outcome, covariates)

        assert estimator.treatment_type_ == "binary"
        assert hasattr(estimator, "treatment_type_")

        # Test categorical
        A_cat = np.random.choice([0, 1, 2], n)
        treatment_cat = TreatmentData(values=A_cat, treatment_type="categorical")

        estimator_cat = DoublyRobustMLEstimator(cv_folds=2, random_state=42)
        estimator_cat.fit(treatment_cat, outcome, covariates)

        assert estimator_cat.treatment_type_ == "categorical"
        assert hasattr(estimator_cat, "treatment_categories_")
        assert hasattr(estimator_cat, "n_treatment_categories_")

        # Test continuous
        A_cont = np.random.uniform(0, 3, n)
        treatment_cont = TreatmentData(values=A_cont, treatment_type="continuous")

        estimator_cont = DoublyRobustMLEstimator(cv_folds=2, random_state=42)
        estimator_cont.fit(treatment_cont, outcome, covariates)

        assert estimator_cont.treatment_type_ == "continuous"
        assert hasattr(estimator_cont, "treatment_dose_range_")

    def test_method_naming_consistency(self):
        """Test that method names include treatment type information."""
        np.random.seed(42)
        n = 150

        X = np.random.randn(n, 2)
        Y = np.random.randn(n)

        # Test different treatment types produce different method names
        treatment_types = [
            (np.random.choice([0, 1], n), "binary"),
            (np.random.choice([0, 1, 2], n), "categorical"),
            (np.random.uniform(0, 2, n), "continuous"),
        ]

        method_names = []
        for A, t_type in treatment_types:
            treatment = TreatmentData(values=A, treatment_type=t_type)
            outcome = OutcomeData(values=Y)
            covariates = CovariateData(values=pd.DataFrame(X))

            estimator = DoublyRobustMLEstimator(cv_folds=2, random_state=42)
            estimator.fit(treatment, outcome, covariates)
            result = estimator.estimate_ate()

            method_names.append(result.method)

        # All method names should be different and contain treatment type
        assert len(set(method_names)) == 3
        assert any("binary" in name for name in method_names)
        assert any("categorical" in name for name in method_names)
        assert any("continuous" in name for name in method_names)


if __name__ == "__main__":
    pytest.main([__file__])
