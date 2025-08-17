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
        """Test that lack of treatment variation fails at TreatmentData construction."""
        # Test that creating TreatmentData with no variation fails at construction time
        with pytest.raises(
            ValueError, match="Binary treatment must have exactly 2 unique values"
        ):
            TreatmentData(values=np.array([0] * 15), treatment_type="binary")

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
        # Test that creating TreatmentData with no variation provides clear error message
        with pytest.raises(
            ValueError, match="Binary treatment must have exactly 2 unique values"
        ) as exc_info:
            TreatmentData(values=np.array([1] * 15), treatment_type="binary")

        error_message = str(exc_info.value)
        assert "both treated and control units" in error_message


class TestInputValidationEnhancements:
    """Test enhanced input validation for production use."""

    def test_treatment_data_validation(self):
        """Test comprehensive treatment data validation."""
        # Test categorical treatment without categories - this should fail at TreatmentData construction
        with pytest.raises(
            ValueError, match="Categorical treatment must specify categories"
        ):
            TreatmentData(
                values=np.array([0, 1, 2, 1, 0]), treatment_type="categorical"
            )

    def test_categorical_treatment_valid(self):
        """Test that categorical treatment with valid categories works correctly."""
        # Test categorical treatment with proper categories specified
        np.random.seed(123)
        treatment = TreatmentData(
            values=np.array([0, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
            treatment_type="categorical",
            categories=[0, 1, 2],  # Valid categories
        )
        outcome = OutcomeData(
            values=np.random.normal(0, 1, 15), outcome_type="continuous"
        )
        covariates = CovariateData(
            values=pd.DataFrame(np.random.normal(0, 1, (15, 2)), columns=["X1", "X2"])
        )

        # Should work without errors
        estimator = GComputationEstimator()
        estimator.fit(treatment, outcome, covariates)
        assert estimator.is_fitted

        # Should be able to estimate effects
        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)

    def test_reproducible_bootstrap(self):
        """Test that same seed produces same bootstrap results."""
        np.random.seed(42)

        # Create consistent test data
        treatment = TreatmentData(values=np.array([0, 1] * 8), treatment_type="binary")
        outcome = OutcomeData(
            values=np.random.normal(0, 1, 16), outcome_type="continuous"
        )
        covariates = CovariateData(
            values=pd.DataFrame(np.random.normal(0, 1, (16, 2)), columns=["X1", "X2"])
        )

        # Test reproducibility across different estimators
        estimators = [
            GComputationEstimator(bootstrap_samples=50, random_state=123),
            IPWEstimator(bootstrap_samples=50, random_state=123),
            AIPWEstimator(bootstrap_samples=50, random_state=123),
        ]

        # Store results for comparison
        results = []

        for estimator in estimators:
            # Run estimation twice with same seed
            estimator.fit(treatment, outcome, covariates)
            effect1 = estimator.estimate_ate()

            # Reset and run again with same seed
            estimator_copy = type(estimator)(bootstrap_samples=50, random_state=123)
            estimator_copy.fit(treatment, outcome, covariates)
            effect2 = estimator_copy.estimate_ate()

            # Results should be identical with same seed
            assert np.isclose(effect1.ate, effect2.ate), (
                f"ATE not reproducible for {type(estimator).__name__}"
            )

            # Bootstrap estimates should also be reproducible (check sorted values to avoid order issues)
            if (
                effect1.bootstrap_estimates is not None
                and effect2.bootstrap_estimates is not None
            ):
                # Sort both arrays to check if they contain the same values (order might vary)
                sorted_estimates1 = np.sort(effect1.bootstrap_estimates)
                sorted_estimates2 = np.sort(effect2.bootstrap_estimates)
                assert np.allclose(sorted_estimates1, sorted_estimates2), (
                    f"Bootstrap estimates not reproducible for {type(estimator).__name__}"
                )

            # Confidence intervals should be reproducible
            if effect1.ate_ci_lower is not None and effect2.ate_ci_lower is not None:
                assert np.isclose(effect1.ate_ci_lower, effect2.ate_ci_lower), (
                    f"CI lower bound not reproducible for {type(estimator).__name__}"
                )
                assert np.isclose(effect1.ate_ci_upper, effect2.ate_ci_upper), (
                    f"CI upper bound not reproducible for {type(estimator).__name__}"
                )

            results.append(effect1)

        # Verify that all estimators produced valid results
        for result in results:
            assert np.isfinite(result.ate)
            assert result.n_observations == 16

    def test_bootstrap_convergence(self):
        """Test bootstrap convergence with challenging cases."""
        np.random.seed(789)

        # Create a challenging case with potential separation issues
        n_samples = 30
        X = np.random.normal(0, 1, (n_samples, 2))

        # Create near-perfect separation case
        # This makes the logistic regression model struggle
        treatment_prob = 1 / (1 + np.exp(-(3 * X[:, 0] + 2 * X[:, 1])))
        treatment = np.random.binomial(1, treatment_prob)

        # Ensure we have some variation
        if np.sum(treatment) == 0:
            treatment[0] = 1
        if np.sum(treatment) == len(treatment):
            treatment[0] = 0

        # Create outcome with strong treatment effect
        outcome = (
            2 * treatment + X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n_samples)
        )

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X, columns=["X1", "X2"]))

        # Test with smaller bootstrap samples for faster execution
        estimators = [
            GComputationEstimator(bootstrap_samples=25, random_state=42),
            # IPW might fail with separation, so we handle that gracefully
            IPWEstimator(
                bootstrap_samples=25,
                random_state=42,
                propensity_model_params={
                    "C": 0.1,
                    "max_iter": 1000,
                },  # Regularized for stability
            ),
            AIPWEstimator(
                bootstrap_samples=25,
                random_state=42,
                propensity_model_params={"C": 0.1, "max_iter": 1000},
            ),
        ]

        for estimator in estimators:
            try:
                estimator.fit(treatment_data, outcome_data, covariate_data)
                effect = estimator.estimate_ate()

                # If bootstrap succeeded, check convergence indicators
                if (
                    hasattr(effect, "bootstrap_converged")
                    and effect.bootstrap_converged is not None
                ):
                    # Bootstrap should either converge or provide meaningful diagnostics
                    if not effect.bootstrap_converged:
                        # If not converged, we should still get reasonable estimates
                        assert np.isfinite(effect.ate), (
                            f"ATE should be finite even with convergence issues for {type(estimator).__name__}"
                        )
                        # Standard error might be None if bootstrap failed completely
                        if effect.ate_se is not None:
                            assert effect.ate_se > 0, (
                                f"Standard error should be positive for {type(estimator).__name__}"
                            )
                    else:
                        # If converged, all bootstrap statistics should be available
                        assert np.isfinite(effect.ate), (
                            f"ATE should be finite when converged for {type(estimator).__name__}"
                        )
                        if effect.ate_se is not None:
                            assert effect.ate_se > 0, (
                                f"Standard error should be positive when converged for {type(estimator).__name__}"
                            )

                        # Check that confidence intervals are reasonable
                        if (
                            effect.ate_ci_lower is not None
                            and effect.ate_ci_upper is not None
                        ):
                            assert effect.ate_ci_lower < effect.ate_ci_upper, (
                                f"CI bounds should be ordered correctly for {type(estimator).__name__}"
                            )

                # Basic sanity checks regardless of convergence
                assert np.isfinite(effect.ate), (
                    f"Point estimate should always be finite for {type(estimator).__name__}"
                )
                assert effect.n_observations == n_samples, (
                    f"Sample size should be correct for {type(estimator).__name__}"
                )

            except EstimationError as e:
                # For challenging cases, some estimators may fail
                # This is acceptable as long as the error is informative
                error_msg = str(e).lower()
                assert any(
                    keyword in error_msg
                    for keyword in [
                        "convergence",
                        "separation",
                        "numerical",
                        "instability",
                        "failed",
                    ]
                ), (
                    f"Error message should be informative for {type(estimator).__name__}: {error_msg}"
                )

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
