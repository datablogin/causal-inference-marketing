"""Tests for base estimator classes and interfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    DataValidationError,
    EstimationError,
    OutcomeData,
    TreatmentData,
)


class MockEstimator(BaseEstimator):
    """Mock estimator for testing the base class."""

    def __init__(self, should_fail: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.should_fail = should_fail
        self._fitted_data = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Mock fitting implementation."""
        if self.should_fail:
            raise ValueError("Mock fitting failure")

        self._fitted_data = {
            "treatment": treatment,
            "outcome": outcome,
            "covariates": covariates,
        }

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Mock ATE estimation implementation."""
        if self.should_fail:
            raise ValueError("Mock estimation failure")

        # Simple mock: treated mean - control mean
        treatment_vals = self._fitted_data["treatment"].values
        outcome_vals = self._fitted_data["outcome"].values

        treated_outcomes = outcome_vals[treatment_vals == 1]
        control_outcomes = outcome_vals[treatment_vals == 0]

        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)

        return CausalEffect(
            ate=ate,
            ate_se=0.1,
            ate_ci_lower=ate - 1.96 * 0.1,
            ate_ci_upper=ate + 1.96 * 0.1,
            method="mock",
            n_observations=len(treatment_vals),
            n_treated=np.sum(treatment_vals == 1),
            n_control=np.sum(treatment_vals == 0),
        )


class TestTreatmentData:
    """Test cases for TreatmentData model."""

    def test_valid_binary_treatment(self):
        """Test creation of valid binary treatment data."""
        values = pd.Series([0, 1, 0, 1, 1])
        treatment = TreatmentData(
            values=values, name="treatment", treatment_type="binary"
        )

        assert treatment.name == "treatment"
        assert treatment.treatment_type == "binary"
        assert len(treatment.values) == 5

    def test_valid_categorical_treatment(self):
        """Test creation of valid categorical treatment data."""
        values = pd.Series(["A", "B", "C", "A", "B"])
        treatment = TreatmentData(
            values=values,
            name="group",
            treatment_type="categorical",
            categories=["A", "B", "C"],
        )

        assert treatment.name == "group"
        assert treatment.treatment_type == "categorical"
        assert treatment.categories == ["A", "B", "C"]

    def test_valid_continuous_treatment(self):
        """Test creation of valid continuous treatment data."""
        values = np.array([0.1, 0.5, 1.2, 2.0, 0.8])
        treatment = TreatmentData(values=values, treatment_type="continuous")

        assert treatment.treatment_type == "continuous"
        assert len(treatment.values) == 5

    def test_invalid_treatment_type(self):
        """Test that invalid treatment type raises error."""
        values = pd.Series([0, 1, 0, 1])

        with pytest.raises(ValueError, match="treatment_type must be one of"):
            TreatmentData(values=values, treatment_type="invalid")

    def test_empty_values(self):
        """Test that empty values raise error."""
        with pytest.raises(ValueError, match="Treatment values cannot be empty"):
            TreatmentData(values=pd.Series([]))


class TestOutcomeData:
    """Test cases for OutcomeData model."""

    def test_valid_continuous_outcome(self):
        """Test creation of valid continuous outcome data."""
        values = pd.Series([1.5, 2.1, 0.8, 3.2, 1.9])
        outcome = OutcomeData(values=values, name="revenue", outcome_type="continuous")

        assert outcome.name == "revenue"
        assert outcome.outcome_type == "continuous"
        assert len(outcome.values) == 5

    def test_valid_binary_outcome(self):
        """Test creation of valid binary outcome data."""
        values = np.array([0, 1, 1, 0, 1])
        outcome = OutcomeData(values=values, outcome_type="binary")

        assert outcome.outcome_type == "binary"
        assert len(outcome.values) == 5

    def test_invalid_outcome_type(self):
        """Test that invalid outcome type raises error."""
        values = pd.Series([1, 2, 3, 4])

        with pytest.raises(ValueError, match="outcome_type must be one of"):
            OutcomeData(values=values, outcome_type="invalid")

    def test_empty_values(self):
        """Test that empty values raise error."""
        with pytest.raises(ValueError, match="Outcome values cannot be empty"):
            OutcomeData(values=np.array([]))


class TestCovariateData:
    """Test cases for CovariateData model."""

    def test_valid_covariate_dataframe(self):
        """Test creation of valid covariate data with DataFrame."""
        df = pd.DataFrame(
            {"age": [25, 30, 35, 40, 45], "income": [50000, 60000, 70000, 80000, 90000]}
        )

        covariates = CovariateData(values=df, names=["age", "income"])

        assert covariates.names == ["age", "income"]
        assert covariates.values.shape == (5, 2)

    def test_valid_covariate_array(self):
        """Test creation of valid covariate data with numpy array."""
        values = np.random.randn(10, 3)
        covariates = CovariateData(values=values, names=["x1", "x2", "x3"])

        assert covariates.names == ["x1", "x2", "x3"]
        assert covariates.values.shape == (10, 3)

    def test_empty_values(self):
        """Test that empty values raise error."""
        with pytest.raises(ValueError, match="Covariate values cannot be empty"):
            CovariateData(values=pd.DataFrame())


class TestCausalEffect:
    """Test cases for CausalEffect data class."""

    def test_valid_causal_effect(self):
        """Test creation of valid causal effect."""
        effect = CausalEffect(
            ate=0.5,
            ate_se=0.1,
            ate_ci_lower=0.3,
            ate_ci_upper=0.7,
            method="test",
            n_observations=100,
        )

        assert effect.ate == 0.5
        assert effect.ate_se == 0.1
        assert effect.method == "test"
        assert effect.n_observations == 100

    def test_invalid_confidence_interval(self):
        """Test that invalid confidence interval raises error."""
        with pytest.raises(
            ValueError, match="Lower confidence bound cannot exceed upper bound"
        ):
            CausalEffect(ate=0.5, ate_ci_lower=0.7, ate_ci_upper=0.3)

    def test_invalid_confidence_level(self):
        """Test that invalid confidence level raises error."""
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            CausalEffect(ate=0.5, confidence_level=1.5)

    def test_is_significant_property(self):
        """Test the is_significant property."""
        # Significant positive effect
        effect1 = CausalEffect(ate=0.5, ate_ci_lower=0.2, ate_ci_upper=0.8)
        assert effect1.is_significant is True

        # Significant negative effect
        effect2 = CausalEffect(ate=-0.5, ate_ci_lower=-0.8, ate_ci_upper=-0.2)
        assert effect2.is_significant is True

        # Non-significant effect (CI contains zero)
        effect3 = CausalEffect(ate=0.1, ate_ci_lower=-0.2, ate_ci_upper=0.4)
        assert effect3.is_significant is False

        # Missing CI
        effect4 = CausalEffect(ate=0.5)
        assert effect4.is_significant is False

    def test_effect_size_interpretation(self):
        """Test the effect size interpretation property."""
        assert CausalEffect(ate=0.05).effect_size_interpretation == "negligible"
        assert CausalEffect(ate=0.2).effect_size_interpretation == "small"
        assert CausalEffect(ate=0.4).effect_size_interpretation == "medium"
        assert CausalEffect(ate=0.8).effect_size_interpretation == "large"
        assert CausalEffect(ate=-0.4).effect_size_interpretation == "medium"


class TestBaseEstimator:
    """Test cases for BaseEstimator abstract class."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)

        # Create simple test data
        n = 100
        self.treatment_values = np.random.binomial(1, 0.5, n)
        self.outcome_values = self.treatment_values * 2 + np.random.normal(0, 1, n)
        self.covariate_values = np.random.randn(n, 2)

        self.treatment_data = TreatmentData(
            values=pd.Series(self.treatment_values), treatment_type="binary"
        )

        self.outcome_data = OutcomeData(
            values=pd.Series(self.outcome_values), outcome_type="continuous"
        )

        self.covariate_data = CovariateData(
            values=pd.DataFrame(self.covariate_values, columns=["x1", "x2"]),
            names=["x1", "x2"],
        )

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = MockEstimator(random_state=42, verbose=True)

        assert estimator.random_state == 42
        assert estimator.verbose is True
        assert estimator.is_fitted is False
        assert estimator.treatment_data is None
        assert estimator.outcome_data is None
        assert estimator.covariate_data is None

    def test_successful_fitting(self):
        """Test successful fitting process."""
        estimator = MockEstimator()

        result = estimator.fit(
            treatment=self.treatment_data,
            outcome=self.outcome_data,
            covariates=self.covariate_data,
        )

        # Should return self
        assert result is estimator
        assert estimator.is_fitted is True
        assert estimator.treatment_data is not None
        assert estimator.outcome_data is not None
        assert estimator.covariate_data is not None

    def test_fitting_failure(self):
        """Test fitting process with failure."""
        estimator = MockEstimator(should_fail=True)

        with pytest.raises(EstimationError, match="Failed to fit estimator"):
            estimator.fit(treatment=self.treatment_data, outcome=self.outcome_data)

        assert estimator.is_fitted is False

    def test_ate_estimation(self):
        """Test ATE estimation."""
        estimator = MockEstimator()
        estimator.fit(treatment=self.treatment_data, outcome=self.outcome_data)

        effect = estimator.estimate_ate()

        assert isinstance(effect, CausalEffect)
        assert effect.method == "mock"
        assert effect.n_observations == 100
        assert effect.ate_se == 0.1

    def test_ate_estimation_without_fitting(self):
        """Test that ATE estimation fails if not fitted."""
        estimator = MockEstimator()

        with pytest.raises(
            EstimationError, match="Estimator must be fitted before estimation"
        ):
            estimator.estimate_ate()

    def test_ate_estimation_caching(self):
        """Test that ATE estimation results are cached."""
        estimator = MockEstimator()
        estimator.fit(treatment=self.treatment_data, outcome=self.outcome_data)

        # First call
        effect1 = estimator.estimate_ate()

        # Second call should return cached result
        effect2 = estimator.estimate_ate(use_cache=True)

        assert effect1 is effect2

        # Force re-computation
        effect3 = estimator.estimate_ate(use_cache=False)
        assert effect3 is not effect1  # New object, but same values

    def test_input_validation_mismatched_lengths(self):
        """Test validation with mismatched data lengths."""
        estimator = MockEstimator()

        short_outcome = OutcomeData(
            values=pd.Series([1, 2, 3]),  # Only 3 observations
            outcome_type="continuous",
        )

        with pytest.raises(
            DataValidationError, match="must have the same number of observations"
        ):
            estimator.fit(
                treatment=self.treatment_data,  # 100 observations
                outcome=short_outcome,  # 3 observations
            )

    def test_input_validation_missing_treatment(self):
        """Test validation with missing treatment values."""
        estimator = MockEstimator()

        treatment_with_na = TreatmentData(
            values=pd.Series([0, 1, np.nan, 1, 0]), treatment_type="binary"
        )

        outcome = OutcomeData(
            values=pd.Series([1, 2, 3, 4, 5]), outcome_type="continuous"
        )

        with pytest.raises(
            DataValidationError, match="Treatment values cannot contain missing data"
        ):
            estimator.fit(treatment=treatment_with_na, outcome=outcome)

    def test_input_validation_minimum_sample_size(self):
        """Test validation with insufficient sample size."""
        estimator = MockEstimator()

        small_treatment = TreatmentData(
            values=pd.Series([0, 1, 0]),  # Only 3 observations
            treatment_type="binary",
        )

        small_outcome = OutcomeData(
            values=pd.Series([1, 2, 3]), outcome_type="continuous"
        )

        with pytest.raises(
            DataValidationError, match="Minimum sample size of 10 observations required"
        ):
            estimator.fit(treatment=small_treatment, outcome=small_outcome)

    def test_input_validation_no_treatment_variation(self):
        """Test validation with no treatment variation."""
        # Test that creating TreatmentData with no variation fails at construction time
        with pytest.raises(
            ValueError,
            match="Binary treatment must have exactly 2 unique values",
        ):
            TreatmentData(
                values=pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),  # All 1s
                treatment_type="binary",
            )

    def test_positivity_assumption_check(self):
        """Test positivity assumption checking."""
        estimator = MockEstimator()
        estimator.fit(treatment=self.treatment_data, outcome=self.outcome_data)

        results = estimator.check_positivity_assumption()

        assert isinstance(results, dict)
        assert "assumption_met" in results
        assert "treatment_probability" in results
        assert "control_probability" in results
        assert results["assumption_met"] is True  # Balanced treatment

    def test_positivity_assumption_violation(self):
        """Test positivity assumption with severe imbalance."""
        # Create severely imbalanced treatment
        imbalanced_treatment = np.concatenate(
            [
                np.ones(95),  # 95 treated
                np.zeros(5),  # 5 control
            ]
        )

        imbalanced_treatment_data = TreatmentData(
            values=pd.Series(imbalanced_treatment), treatment_type="binary"
        )

        imbalanced_outcome_data = OutcomeData(
            values=pd.Series(np.random.randn(100)), outcome_type="continuous"
        )

        estimator = MockEstimator()
        estimator.fit(
            treatment=imbalanced_treatment_data, outcome=imbalanced_outcome_data
        )

        results = estimator.check_positivity_assumption(min_probability=0.1)

        assert results["assumption_met"] is False
        assert len(results["violations"]) > 0

    def test_summary_unfitted(self):
        """Test summary for unfitted estimator."""
        estimator = MockEstimator()
        summary = estimator.summary()

        assert "MockEstimator (not fitted)" in summary

    def test_summary_fitted(self):
        """Test summary for fitted estimator."""
        estimator = MockEstimator()
        estimator.fit(treatment=self.treatment_data, outcome=self.outcome_data)

        # Generate ATE estimate
        estimator.estimate_ate()

        summary = estimator.summary()

        assert "MockEstimator Summary" in summary
        assert "Fitted: True" in summary
        assert "Observations: 100" in summary
        assert "ATE:" in summary
        assert "95% CI:" in summary

    def test_predict_potential_outcomes_not_implemented(self):
        """Test that base class prediction method raises NotImplementedError."""
        estimator = MockEstimator()
        estimator.fit(treatment=self.treatment_data, outcome=self.outcome_data)

        with pytest.raises(NotImplementedError):
            estimator.predict_potential_outcomes(
                treatment_values=np.array([0, 1, 0, 1]), covariates=None
            )


if __name__ == "__main__":
    pytest.main([__file__])
