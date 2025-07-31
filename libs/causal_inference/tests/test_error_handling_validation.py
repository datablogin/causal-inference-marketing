"""Tests for comprehensive error handling and input validation.

This module tests that all estimators properly handle edge cases, invalid inputs,
and error conditions with informative error messages.
"""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import (
    CovariateData,
    DataValidationError,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator


class TestDataValidationErrors:
    """Test proper validation of input data."""

    def test_empty_data_validation(self):
        """Test that empty data raises appropriate errors."""
        # Test empty treatment data
        with pytest.raises(ValueError, match="Treatment values cannot be empty"):
            TreatmentData(values=np.array([]), treatment_type="binary")

        # Test empty outcome data
        with pytest.raises(ValueError, match="Outcome values cannot be empty"):
            OutcomeData(values=np.array([]), outcome_type="continuous")

        # Test empty covariate data
        with pytest.raises(ValueError, match="Covariate values cannot be empty"):
            CovariateData(values=pd.DataFrame())

    def test_invalid_treatment_type(self):
        """Test that invalid treatment types raise errors."""
        with pytest.raises(ValueError, match="treatment_type must be one of"):
            TreatmentData(values=np.array([0, 1, 0, 1]), treatment_type="invalid_type")

    def test_invalid_outcome_type(self):
        """Test that invalid outcome types raise errors."""
        with pytest.raises(ValueError, match="outcome_type must be one of"):
            OutcomeData(values=np.array([1.5, 2.3, 1.8]), outcome_type="invalid_type")

    def test_mismatched_data_lengths(self):
        """Test that mismatched data lengths raise DataValidationError."""
        treatment = TreatmentData(values=np.array([0, 1, 0]), treatment_type="binary")
        outcome = OutcomeData(
            values=np.array([1.5, 2.3]), outcome_type="continuous"
        )  # Different length
        covariates = CovariateData(
            values=pd.DataFrame({"X1": [0.1, 0.2, 0.3], "X2": [1.1, 1.2, 1.3]})
        )

        estimator = GComputationEstimator()

        with pytest.raises(
            DataValidationError,
            match="Treatment .* and outcome .* must have the same number of observations",
        ):
            estimator.fit(treatment, outcome, covariates)

    def test_missing_treatment_values(self):
        """Test that missing treatment values raise DataValidationError."""
        treatment_with_nan = TreatmentData(
            values=pd.Series([0, 1, np.nan, 1]), treatment_type="binary"
        )
        outcome = OutcomeData(
            values=np.array([1.5, 2.3, 1.8, 2.1]), outcome_type="continuous"
        )

        estimator = GComputationEstimator()

        with pytest.raises(
            DataValidationError, match="Treatment values cannot contain missing data"
        ):
            estimator.fit(treatment_with_nan, outcome)

    def test_no_treatment_variation(self):
        """Test that lack of treatment variation raises DataValidationError."""
        # All control units - but enough samples to pass minimum size check
        treatment = TreatmentData(values=np.array([0] * 15), treatment_type="binary")
        outcome = OutcomeData(
            values=np.random.normal(0, 1, 15), outcome_type="continuous"
        )

        estimator = GComputationEstimator()

        with pytest.raises(
            DataValidationError,
            match="Binary treatment must have both treated and control units",
        ):
            estimator.fit(treatment, outcome)

    def test_minimum_sample_size(self):
        """Test that minimum sample size requirement is enforced."""
        # Only 5 observations (below minimum of 10)
        treatment = TreatmentData(
            values=np.array([0, 1, 0, 1, 0]), treatment_type="binary"
        )
        outcome = OutcomeData(
            values=np.array([1.5, 2.3, 1.8, 2.1, 1.7]), outcome_type="continuous"
        )

        estimator = GComputationEstimator()

        with pytest.raises(
            DataValidationError, match="Minimum sample size of 10 observations required"
        ):
            estimator.fit(treatment, outcome)


class TestEstimationErrors:
    """Test proper handling of estimation errors."""

    def test_estimation_before_fitting(self):
        """Test that estimation before fitting raises EstimationError."""
        estimator = GComputationEstimator()

        with pytest.raises(
            EstimationError, match="Estimator must be fitted before estimation"
        ):
            estimator.estimate_ate()

    def test_prediction_before_fitting(self):
        """Test that prediction before fitting raises EstimationError."""
        estimator = GComputationEstimator()

        with pytest.raises(
            EstimationError, match="Estimator must be fitted before prediction"
        ):
            estimator.predict_potential_outcomes(
                treatment_values=np.array([0, 1]), covariates=np.array([[1, 2], [3, 4]])
            )

    def test_ipw_without_covariates(self):
        """Test that IPW without covariates raises EstimationError."""
        treatment = TreatmentData(
            values=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), treatment_type="binary"
        )
        outcome = OutcomeData(
            values=np.array([1.5, 2.3, 1.8, 2.1, 1.7, 2.4, 1.6, 2.2, 1.9, 2.0]),
            outcome_type="continuous",
        )

        estimator = IPWEstimator()

        with pytest.raises(
            EstimationError,
            match="IPW requires covariates for propensity score estimation",
        ):
            estimator.fit(treatment, outcome, covariates=None)

    def test_invalid_model_type(self):
        """Test that invalid model types raise ValueError during fitting."""
        # These errors occur during fitting, not initialization
        treatment = TreatmentData(values=np.array([0, 1] * 8), treatment_type="binary")
        outcome = OutcomeData(
            values=np.random.normal(0, 1, 16), outcome_type="continuous"
        )
        covariates = CovariateData(
            values=pd.DataFrame(np.random.normal(0, 1, (16, 2)), columns=["X1", "X2"])
        )

        g_estimator = GComputationEstimator(model_type="invalid_model")
        with pytest.raises(EstimationError, match="Unknown model type"):
            g_estimator.fit(treatment, outcome, covariates)

        ipw_estimator = IPWEstimator(propensity_model_type="invalid_model")
        with pytest.raises(EstimationError, match="Unknown propensity model type"):
            ipw_estimator.fit(treatment, outcome, covariates)


class TestBusinessFriendlyErrorMessages:
    """Test that error messages are business-friendly and actionable."""

    def test_convergence_error_message(self):
        """Test that convergence issues are handled gracefully."""
        # Create data that might cause convergence issues
        n_samples = 50
        np.random.seed(42)  # For reproducible test
        X = np.random.normal(0, 1, (n_samples, 2))
        # Create some separation but not perfect
        treatment = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        outcome = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n_samples)

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X, columns=["X1", "X2"]))

        estimator = IPWEstimator(
            propensity_model_params={"max_iter": 10, "solver": "liblinear"}
        )

        # The estimator should either succeed or fail with an informative error
        try:
            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()
            assert np.isfinite(effect.ate)  # Should produce valid result
        except EstimationError as e:
            error_message = str(e)
            # Check that error message contains helpful guidance
            assert any(
                keyword in error_message.lower()
                for keyword in [
                    "convergence",
                    "separation",
                    "numerical instability",
                    "multicollinearity",
                    "failed",
                ]
            ), f"Error message should contain helpful guidance: {error_message}"

    def test_sample_size_guidance(self):
        """Test that sample size errors provide clear guidance."""
        treatment = TreatmentData(values=np.array([0, 1]), treatment_type="binary")
        outcome = OutcomeData(values=np.array([1.0, 2.0]), outcome_type="continuous")

        estimator = GComputationEstimator()

        with pytest.raises(DataValidationError) as exc_info:
            estimator.fit(treatment, outcome)

        error_message = str(exc_info.value)
        assert "Minimum sample size of 10 observations required" in error_message

    def test_treatment_variation_guidance(self):
        """Test that treatment variation errors provide clear guidance."""
        treatment = TreatmentData(values=np.array([1] * 15), treatment_type="binary")
        outcome = OutcomeData(
            values=np.random.normal(0, 1, 15), outcome_type="continuous"
        )

        estimator = GComputationEstimator()

        with pytest.raises(DataValidationError) as exc_info:
            estimator.fit(treatment, outcome)

        error_message = str(exc_info.value)
        assert "both treated and control units" in error_message


class TestInputValidationEnhancements:
    """Test enhanced input validation for production use."""

    def test_treatment_data_validation(self):
        """Test comprehensive treatment data validation."""
        # Test categorical treatment without categories
        with pytest.raises(
            DataValidationError, match="Categorical treatment must specify categories"
        ):
            treatment = TreatmentData(
                values=np.array([0, 1, 2, 1, 0]), treatment_type="categorical"
            )
            outcome = OutcomeData(
                values=np.random.normal(0, 1, 5), outcome_type="continuous"
            )
            estimator = GComputationEstimator()
            estimator.fit(treatment, outcome)

    def test_robust_error_recovery(self):
        """Test that estimators can recover from errors gracefully."""
        # Create valid data
        np.random.seed(42)
        treatment = TreatmentData(
            values=np.random.binomial(1, 0.5, 20), treatment_type="binary"
        )
        outcome = OutcomeData(
            values=np.random.normal(0, 1, 20), outcome_type="continuous"
        )
        covariates = CovariateData(
            values=pd.DataFrame(
                np.random.normal(0, 1, (20, 3)), columns=["X1", "X2", "X3"]
            )
        )

        estimator = GComputationEstimator()

        # Should fit successfully
        estimator.fit(treatment, outcome, covariates)
        assert estimator.is_fitted

        # Should estimate successfully
        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)

    def test_assumption_checking(self):
        """Test that assumption checking provides useful diagnostics."""
        np.random.seed(123)
        treatment = TreatmentData(
            values=np.random.binomial(1, 0.5, 30), treatment_type="binary"
        )
        outcome = OutcomeData(
            values=np.random.normal(0, 1, 30), outcome_type="continuous"
        )
        covariates = CovariateData(
            values=pd.DataFrame(np.random.normal(0, 1, (30, 2)), columns=["X1", "X2"])
        )

        estimator = GComputationEstimator()
        estimator.fit(treatment, outcome, covariates)

        # Check positivity assumption
        positivity_result = estimator.check_positivity_assumption()
        assert isinstance(positivity_result, dict)
        assert "assumption_met" in positivity_result
        assert "violations" in positivity_result

        # Check assumptions summary
        assumptions = estimator.check_assumptions(verbose=False)
        assert isinstance(assumptions, dict)


class TestErrorHandlingIntegration:
    """Test error handling in integrated workflows."""

    def test_complete_workflow_error_handling(self):
        """Test that complete workflows handle errors appropriately."""
        np.random.seed(456)

        # Generate data with potential issues
        n_samples = 25
        treatment = TreatmentData(
            values=np.random.binomial(1, 0.3, n_samples), treatment_type="binary"
        )
        outcome = OutcomeData(
            values=np.random.normal(0, 1, n_samples), outcome_type="continuous"
        )
        covariates = CovariateData(
            values=pd.DataFrame(
                np.random.normal(0, 1, (n_samples, 4)), columns=["X1", "X2", "X3", "X4"]
            )
        )

        # Test all three main estimators
        estimators = [
            GComputationEstimator(bootstrap_samples=10),  # Reduced for speed
            IPWEstimator(bootstrap_samples=10),
            AIPWEstimator(bootstrap_samples=10),
        ]

        for estimator in estimators:
            try:
                estimator.fit(treatment, outcome, covariates)
                effect = estimator.estimate_ate()

                # Verify that valid results are returned
                assert np.isfinite(effect.ate)
                assert effect.method is not None
                assert effect.n_observations == n_samples

            except (EstimationError, DataValidationError) as e:
                # If estimation fails, error should be informative
                error_msg = str(e)
                assert len(error_msg) > 20  # Should be descriptive
                assert any(
                    keyword in error_msg.lower()
                    for keyword in ["error", "failed", "cannot", "invalid", "missing"]
                )

    def test_bootstrap_error_handling(self):
        """Test that bootstrap procedures handle errors gracefully."""
        np.random.seed(789)

        # Create small dataset that might cause bootstrap issues
        n_samples = 15
        treatment = TreatmentData(
            values=np.random.binomial(1, 0.5, n_samples), treatment_type="binary"
        )
        outcome = OutcomeData(
            values=np.random.normal(0, 1, n_samples), outcome_type="continuous"
        )
        covariates = CovariateData(
            values=pd.DataFrame(
                np.random.normal(0, 1, (n_samples, 2)), columns=["X1", "X2"]
            )
        )

        estimator = GComputationEstimator(
            bootstrap_samples=5
        )  # Very small for potential issues
        estimator.fit(treatment, outcome, covariates)

        # Should not crash even if bootstrap has issues
        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)
        # Confidence intervals might be None if bootstrap failed, and that's okay
