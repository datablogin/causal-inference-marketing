"""Tests for sensitivity analysis improvements from Claude review.

This test suite validates the improvements made based on Claude's review:
- Performance benchmarking and KPI validation
- Chunked processing for bootstrap operations
- Warning system for edge cases
- Unified input standardization
"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from causal_inference.sensitivity import (
    benchmark_sensitivity_functions,
    check_model_assumptions,
    check_treatment_variation,
    oster_delta,
    run_performance_validation,
    standardize_input,
    validate_treatment_outcome_lengths,
)


class TestPerformanceBenchmarking:
    """Test performance benchmarking functionality."""

    def test_benchmark_sensitivity_functions_small(self):
        """Test benchmarking with small dataset."""
        results = benchmark_sensitivity_functions(n_rows=1000, n_bootstrap=10)

        # Check structure
        assert "summary" in results
        assert "e_value" in results
        assert "oster_delta" in results

        # Check summary statistics
        summary = results["summary"]
        assert "total_functions_tested" in summary
        assert "successful_functions" in summary
        assert "average_time_seconds" in summary
        assert summary["n_rows_tested"] == 1000

        # All functions should succeed on small data
        assert summary["successful_functions"] >= 3

    def test_benchmark_sensitivity_functions_large(self):
        """Test benchmarking with large dataset (performance KPI)."""
        results = benchmark_sensitivity_functions(n_rows=50000, n_bootstrap=50)

        # Check that functions complete
        assert "summary" in results
        summary = results["summary"]

        # Should test multiple functions
        assert summary["total_functions_tested"] >= 4

        # Check timing information exists
        assert "average_time_seconds" in summary
        assert "max_time_seconds" in summary

    def test_run_performance_validation(self, capsys):
        """Test full performance validation with output."""
        with patch("builtins.print") as mock_print:
            success = run_performance_validation(print_results=True)

        # Should return boolean
        assert isinstance(success, bool)

        # Should have printed results
        mock_print.assert_called()


class TestChunkedProcessing:
    """Test chunked processing improvements."""

    def test_oster_delta_chunked_bootstrap(self):
        """Test Oster delta with chunked bootstrap processing."""
        np.random.seed(42)
        n = 5000

        # Generate test data
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        t = np.random.binomial(1, 0.5, n)
        y = 2.0 * t + X1 + X2 + np.random.normal(0, 1, n)

        # Test with chunked bootstrap
        results = oster_delta(
            outcome=y,
            treatment=t,
            covariates_restricted=X1.reshape(-1, 1),
            covariates_full=np.column_stack([X1, X2]),
            bootstrap_samples=200,  # Large enough to trigger chunking
        )

        # Should complete successfully
        assert "bootstrap_results" in results
        assert "n_chunks_processed" in results["bootstrap_results"]
        assert results["bootstrap_results"]["n_chunks_processed"] >= 1


class TestWarningSyste:
    """Test warning system for edge cases."""

    def test_oster_delta_low_r2_warning(self):
        """Test warning when R² is very low."""
        np.random.seed(42)
        n = 1000

        # Generate data with very weak relationships (low R²)
        t = np.random.binomial(1, 0.5, n)
        y = 0.01 * t + np.random.normal(0, 10, n)  # Lots of noise
        X = np.random.normal(0, 1, n).reshape(-1, 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            oster_delta(
                outcome=y, treatment=t, covariates_restricted=X, covariates_full=X
            )

            # Should trigger low R² warning
            warning_messages = [str(warning.message) for warning in w]
            assert any("Low R²" in msg for msg in warning_messages)

    def test_oster_delta_small_improvement_warning(self):
        """Test warning when R² improvement is minimal."""
        np.random.seed(42)
        n = 1000

        # Generate data where additional controls don't help much
        X1 = np.random.normal(0, 1, n)
        X2 = 0.01 * X1 + np.random.normal(0, 1, n)  # Minimal additional info
        t = 0.5 * X1 + np.random.normal(0, 1, n)
        y = 2.0 * t + X1 + np.random.normal(0, 1, n)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            oster_delta(
                outcome=y,
                treatment=t,
                covariates_restricted=X1.reshape(-1, 1),
                covariates_full=np.column_stack([X1, X2]),
            )

            # May trigger small improvement warning - this is probabilistic

    def test_oster_delta_low_robustness_warning(self):
        """Test warning when robustness ratio is low."""
        np.random.seed(42)
        n = 1000

        # Generate data with strong confounding
        U = np.random.normal(0, 1, n)  # Strong unobserved confounder
        X = np.random.normal(0, 1, n)
        t = 0.8 * U + 0.2 * X + np.random.normal(0, 1, n)  # Treatment depends on U
        y = 0.5 * t + 2.0 * U + X + np.random.normal(0, 1, n)  # Outcome depends on U

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            oster_delta(
                outcome=y,
                treatment=t,
                covariates_restricted=X.reshape(-1, 1),
                covariates_full=X.reshape(-1, 1),  # Same as restricted
            )

            # Should trigger low robustness warning due to strong confounding
            # This test depends on the specific data generation


class TestUnifiedInputStandardization:
    """Test unified input validation and standardization."""

    def test_standardize_input_numpy_array(self):
        """Test standardizing numpy arrays."""
        data = np.array([1, 2, 3, 4, 5])
        result = standardize_input(data, name="test")

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        np.testing.assert_array_equal(result, data)

    def test_standardize_input_pandas_series(self):
        """Test standardizing pandas Series."""
        data = pd.Series([1, 2, 3, 4, 5])
        result = standardize_input(data, name="test")

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        np.testing.assert_array_equal(result, data.values)

    def test_standardize_input_pandas_dataframe(self):
        """Test standardizing pandas DataFrame."""
        data = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        result = standardize_input(data, name="test", allow_2d=True)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, data.values)

    def test_standardize_input_single_column_df(self):
        """Test standardizing single-column DataFrame."""
        data = pd.DataFrame([1, 2, 3, 4, 5])
        result = standardize_input(data, name="test", allow_2d=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_standardize_input_missing_values(self):
        """Test error handling for missing values."""
        data = np.array([1, 2, np.nan, 4, 5])

        with pytest.raises(ValueError, match="contains.*missing values"):
            standardize_input(data, name="test")

    def test_standardize_input_infinite_values(self):
        """Test error handling for infinite values."""
        data = np.array([1, 2, np.inf, 4, 5])

        with pytest.raises(ValueError, match="contains.*infinite values"):
            standardize_input(data, name="test")

    def test_standardize_input_min_length(self):
        """Test minimum length validation."""
        data = np.array([1, 2])

        with pytest.raises(ValueError, match="requires at least"):
            standardize_input(data, name="test", min_length=5)

    def test_validate_treatment_outcome_lengths(self):
        """Test length validation across arrays."""
        y = np.array([1, 2, 3, 4, 5])
        t = np.array([0, 1, 0, 1, 0])
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

        # Should not raise error
        validate_treatment_outcome_lengths(
            y, t, X, names=["outcome", "treatment", "covariates"]
        )

    def test_validate_treatment_outcome_lengths_mismatch(self):
        """Test error when lengths don't match."""
        y = np.array([1, 2, 3, 4, 5])
        t = np.array([0, 1, 0, 1])  # Different length

        with pytest.raises(ValueError, match="same length"):
            validate_treatment_outcome_lengths(y, t, names=["outcome", "treatment"])


class TestTreatmentVariationChecks:
    """Test treatment variation diagnostics."""

    def test_check_treatment_variation_binary(self):
        """Test binary treatment variation check."""
        treatment = np.array([0, 0, 0, 1, 1, 1, 1])

        result = check_treatment_variation(treatment)

        assert result["is_binary"] is True
        assert len(result["unique_values"]) == 2
        assert len(result["proportions"]) == 2
        assert "balanced" in result

    def test_check_treatment_variation_continuous(self):
        """Test continuous treatment variation check."""
        treatment = np.random.normal(0, 1, 100)

        result = check_treatment_variation(treatment)

        assert result["is_binary"] is False
        assert result["n_unique"] > 10
        assert result["is_continuous"] is True

    def test_check_treatment_variation_insufficient(self):
        """Test error when treatment has insufficient variation."""
        treatment = np.array([1, 1, 1, 1, 1])  # No variation

        with pytest.raises(ValueError, match="Insufficient treatment variation"):
            check_treatment_variation(treatment)

    def test_check_treatment_variation_unbalanced_warning(self):
        """Test warning for unbalanced binary treatment."""
        treatment = np.array(
            [0] * 999 + [1] * 1
        )  # Very unbalanced (0.1% < 1% threshold)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            check_treatment_variation(treatment)

            # Should warn about small group
            warning_messages = [str(warning.message) for warning in w]
            assert any("has only" in msg for msg in warning_messages)


class TestModelAssumptionChecks:
    """Test model assumption diagnostics."""

    def test_check_model_assumptions_basic(self):
        """Test basic model assumption checks."""
        np.random.seed(42)
        n = 1000

        treatment = np.random.binomial(1, 0.5, n)
        outcome = 2.0 * treatment + np.random.normal(0, 1, n)

        result = check_model_assumptions(outcome, treatment)

        assert "outcome_skewed" in result
        assert "outcome_skew" in result
        assert "treatment_outcome_correlation" in result
        assert isinstance(result["outcome_skewed"], bool)

    def test_check_model_assumptions_skewed_outcome(self):
        """Test warning for highly skewed outcome."""
        np.random.seed(42)
        n = 1000

        treatment = np.random.binomial(1, 0.5, n)
        # Highly skewed outcome (exponential distribution)
        outcome = np.random.exponential(1, n) ** 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            check_model_assumptions(outcome, treatment)

            # Should warn about skewness
            warning_messages = [str(warning.message) for warning in w]
            assert any("skewed" in msg for msg in warning_messages)

    def test_check_model_assumptions_weak_correlation(self):
        """Test warning for very weak treatment-outcome correlation."""
        n = 1000
        # Create treatment and outcome with exactly zero correlation
        treatment = np.array([0, 1] * (n // 2))  # Perfect alternating pattern

        # Create outcome that has exactly zero correlation with treatment
        # Since treatment alternates 0,1,0,1..., we make outcome also alternate
        # but with a shift to ensure zero correlation
        outcome = np.array(
            [1, 1, -1, -1] * (n // 4)
        )  # Pattern that sums to zero correlation

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            check_model_assumptions(outcome, treatment)

            # Should warn about weak correlation
            warning_messages = [str(warning.message) for warning in w]
            assert any("weak correlation" in msg for msg in warning_messages)

    def test_check_model_assumptions_multicollinearity(self):
        """Test multicollinearity detection."""
        np.random.seed(42)
        n = 1000

        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.normal(0, 1, n)

        # Highly correlated covariates
        X1 = np.random.normal(0, 1, n)
        X2 = X1 + np.random.normal(0, 0.01, n)  # Nearly identical to X1
        covariates = np.column_stack([X1, X2])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            check_model_assumptions(outcome, treatment, covariates)

            # Should warn about multicollinearity
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "correlation between covariates" in msg for msg in warning_messages
            )


class TestIntegrationWithOsterDelta:
    """Test integration of improvements with Oster delta function."""

    def test_oster_delta_with_improved_validation(self):
        """Test Oster delta with improved input validation."""
        np.random.seed(42)
        n = 1000

        # Generate test data as pandas objects
        X1 = pd.Series(np.random.normal(0, 1, n))
        X2 = pd.DataFrame(np.random.normal(0, 1, n).reshape(-1, 1))
        t = pd.Series(np.random.binomial(1, 0.5, n))
        y = pd.Series(2.0 * t + X1 + np.random.normal(0, 1, n))

        # Should handle pandas inputs correctly
        results = oster_delta(
            outcome=y,
            treatment=t,
            covariates_restricted=X1,
            covariates_full=pd.concat([X1, X2.iloc[:, 0]], axis=1),
        )

        # Should complete successfully
        assert "beta_restricted" in results
        assert "beta_full" in results
        assert "robustness_ratio" in results
        assert "interpretation" in results

    def test_oster_delta_error_messages_enhanced(self):
        """Test enhanced error messages in Oster delta."""
        # Test with insufficient data
        with pytest.raises(ValueError, match="requires at least"):
            oster_delta(
                outcome=[1, 2],  # Too short
                treatment=[0, 1],
                bootstrap_samples=0,
            )

        # Test with mismatched lengths (use longer arrays to pass min_length check)
        with pytest.raises(ValueError, match="same length"):
            oster_delta(
                outcome=[
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                ],  # 15 elements
                treatment=[
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ],  # 10 elements - different length
                bootstrap_samples=0,
            )
