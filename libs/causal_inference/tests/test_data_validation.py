"""Tests for data validation utilities."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.data.validation import (
    CausalDataValidator,
    DataValidationError,
    validate_causal_data,
)


class TestCausalDataValidator:
    """Test cases for causal data validator."""

    def setup_method(self):
        """Set up test data."""
        # Good quality data
        self.good_treatment = TreatmentData(
            values=pd.Series([0, 1, 0, 1, 0, 1] * 50),
            name="treatment",
            treatment_type="binary",
        )

        self.good_outcome = OutcomeData(
            values=pd.Series([1.2, 2.3, 1.1, 2.5, 1.0, 2.8] * 50),
            name="outcome",
            outcome_type="continuous",
        )

        self.good_covariates = CovariateData(
            values=pd.DataFrame({
                'age': [25, 30, 35, 40, 45, 50] * 50,
                'income': [30000, 40000, 50000, 60000, 70000, 80000] * 50,
                'education': [1, 2, 3, 1, 2, 3] * 50,
            }),
            names=['age', 'income', 'education'],
        )

        self.validator = CausalDataValidator(verbose=False)

    def test_validate_good_treatment_data(self):
        """Test validation of good treatment data."""
        self.validator.validate_treatment_data(self.good_treatment)
        assert len(self.validator.errors) == 0
        assert len(self.validator.warnings) == 0

    def test_validate_treatment_with_missing_values(self):
        """Test validation of treatment data with missing values."""
        treatment_with_missing = TreatmentData(
            values=pd.Series([0, 1, np.nan, 1, 0, np.nan] * 20),
            name="treatment",
            treatment_type="binary",
        )

        self.validator.warnings.clear()
        self.validator.errors.clear()
        self.validator.validate_treatment_data(treatment_with_missing)

        # Should have warnings about missing values
        assert any("missing values" in warning for warning in self.validator.warnings)

    def test_validate_treatment_wrong_binary_values(self):
        """Test validation of binary treatment with wrong values."""
        wrong_binary_treatment = TreatmentData(
            values=pd.Series([0, 2, 0, 2, 0, 2] * 20),
            name="treatment",
            treatment_type="binary",
        )

        self.validator.warnings.clear()
        self.validator.errors.clear()
        self.validator.validate_treatment_data(wrong_binary_treatment)

        # Should have errors about wrong binary values
        assert any("exactly 2 unique values" in error for error in self.validator.errors)

    def test_validate_treatment_non_standard_binary(self):
        """Test validation of binary treatment with non-0/1 values."""
        non_standard_binary = TreatmentData(
            values=pd.Series(['A', 'B', 'A', 'B'] * 25),
            name="treatment",
            treatment_type="binary",
        )

        self.validator.validate_treatment_data(non_standard_binary)

        # Should have warnings about non-0/1 values
        assert any("not 0/1" in warning for warning in self.validator.warnings)

    def test_validate_treatment_small_sample(self):
        """Test validation of treatment data with small sample size."""
        small_treatment = TreatmentData(
            values=pd.Series([0, 1, 0, 1]),
            name="treatment",
            treatment_type="binary",
        )

        self.validator.validate_treatment_data(small_treatment)

        # Should have warnings about small sample size
        assert any("small sample size" in warning for warning in self.validator.warnings)

    def test_validate_good_outcome_data(self):
        """Test validation of good outcome data."""
        self.validator.validate_outcome_data(self.good_outcome)
        assert len(self.validator.errors) == 0
        assert len(self.validator.warnings) == 0

    def test_validate_outcome_with_missing_values(self):
        """Test validation of outcome data with missing values."""
        outcome_with_missing = OutcomeData(
            values=pd.Series([1.2, np.nan, 1.1, 2.5, np.nan, 2.8] * 20),
            name="outcome",
            outcome_type="continuous",
        )

        self.validator.warnings.clear()
        self.validator.errors.clear()
        self.validator.validate_outcome_data(outcome_with_missing)

        # Should have warnings about missing values
        assert any("missing values" in warning for warning in self.validator.warnings)

    def test_validate_outcome_with_infinite_values(self):
        """Test validation of outcome data with infinite values."""
        outcome_with_inf = OutcomeData(
            values=pd.Series([1.2, np.inf, 1.1, 2.5, -np.inf, 2.8] * 20),
            name="outcome",
            outcome_type="continuous",
        )

        self.validator.validate_outcome_data(outcome_with_inf)

        # Should have errors about infinite values
        assert any("infinite values" in error for error in self.validator.errors)

    def test_validate_outcome_with_outliers(self):
        """Test validation of outcome data with outliers."""
        # Create data with extreme outliers
        normal_values = [1.0, 1.1, 1.2, 0.9, 1.3] * 50
        outlier_values = normal_values + [100.0, -100.0]  # Extreme outliers

        outcome_with_outliers = OutcomeData(
            values=pd.Series(outlier_values),
            name="outcome",
            outcome_type="continuous",
        )

        self.validator.validate_outcome_data(outcome_with_outliers)

        # Should have warnings about outliers
        assert any("outliers" in warning for warning in self.validator.warnings)

    def test_validate_good_covariate_data(self):
        """Test validation of good covariate data."""
        self.validator.validate_covariate_data(self.good_covariates)
        assert len(self.validator.errors) == 0
        # May have warnings about correlations, but that's ok

    def test_validate_covariates_with_missing_data(self):
        """Test validation of covariate data with missing values."""
        covariates_with_missing = CovariateData(
            values=pd.DataFrame({
                'age': [25, np.nan, 35, 40, np.nan, 50] * 20,
                'income': [30000, 40000, np.nan, 60000, 70000, 80000] * 20,
            }),
            names=['age', 'income'],
        )

        self.validator.validate_covariate_data(covariates_with_missing)

        # Should have warnings about missing values
        missing_warnings = [w for w in self.validator.warnings if "missing values" in w]
        assert len(missing_warnings) > 0

    def test_validate_covariates_with_constant_variable(self):
        """Test validation of covariate data with constant variable."""
        covariates_with_constant = CovariateData(
            values=pd.DataFrame({
                'age': [25, 30, 35, 40, 45, 50] * 20,
                'constant': [1, 1, 1, 1, 1, 1] * 20,  # Constant variable
            }),
            names=['age', 'constant'],
        )

        self.validator.validate_covariate_data(covariates_with_constant)

        # Should have warnings about constant variable
        assert any("constant" in warning for warning in self.validator.warnings)

    def test_validate_covariates_with_high_correlation(self):
        """Test validation of covariate data with highly correlated variables."""
        # Create highly correlated variables
        x1 = [1, 2, 3, 4, 5, 6] * 20
        x2 = [1.01, 2.01, 3.01, 4.01, 5.01, 6.01] * 20  # Almost identical

        covariates_correlated = CovariateData(
            values=pd.DataFrame({
                'x1': x1,
                'x2': x2,
            }),
            names=['x1', 'x2'],
        )

        self.validator.validate_covariate_data(covariates_correlated)

        # Should have warnings about high correlation
        assert any("highly correlated" in warning for warning in self.validator.warnings)

    def test_validate_all_good_data(self):
        """Test validation of all good data."""
        self.validator.validate_all(
            self.good_treatment,
            self.good_outcome,
            self.good_covariates
        )

        # Should pass without errors
        assert len(self.validator.errors) == 0

    def test_validate_all_sample_size_mismatch(self):
        """Test validation with sample size mismatch."""
        mismatched_outcome = OutcomeData(
            values=pd.Series([1.2, 2.3, 1.1]),  # Different length
            name="outcome",
            outcome_type="continuous",
        )

        self.validator.validate_all(
            self.good_treatment,
            mismatched_outcome,
            self.good_covariates
        )

        # Should have errors about sample size mismatch
        assert any("Sample size mismatch" in error for error in self.validator.errors)

    def test_validate_all_with_errors_raises_exception(self):
        """Test that validation raises exception when errors are found."""
        # Create data with errors
        bad_treatment = TreatmentData(
            values=pd.Series([0, 1, 2, 3]),  # Wrong values for binary
            name="treatment",
            treatment_type="binary",
        )

        bad_outcome = OutcomeData(
            values=pd.Series([1.2, np.inf, 1.1]),  # Different length + infinite
            name="outcome",
            outcome_type="continuous",
        )

        with pytest.raises(DataValidationError):
            self.validator.validate_all(bad_treatment, bad_outcome)

    def test_validate_overlap_warning(self):
        """Test overlap validation warning."""
        # Create data with poor overlap
        # Treatment strongly depends on first covariate
        X = np.random.randn(100, 3)
        treatment_values = (X[:, 0] > 1).astype(int)  # Only high X1 gets treatment

        poor_overlap_treatment = TreatmentData(
            values=pd.Series(treatment_values),
            name="treatment",
            treatment_type="binary",
        )

        poor_overlap_covariates = CovariateData(
            values=pd.DataFrame(X, columns=['X1', 'X2', 'X3']),
            names=['X1', 'X2', 'X3'],
        )

        self.validator.validate_overlap(poor_overlap_treatment, poor_overlap_covariates)

        # May or may not have overlap warnings depending on random data
        # This test mainly checks that the method runs without error


class TestValidateCausalDataConvenienceFunction:
    """Test cases for the validate_causal_data convenience function."""

    def setup_method(self):
        """Set up test data."""
        self.good_treatment = TreatmentData(
            values=pd.Series([0, 1, 0, 1, 0, 1] * 10),
            name="treatment",
            treatment_type="binary",
        )

        self.good_outcome = OutcomeData(
            values=pd.Series([1.2, 2.3, 1.1, 2.5, 1.0, 2.8] * 10),
            name="outcome",
            outcome_type="continuous",
        )

        self.good_covariates = CovariateData(
            values=pd.DataFrame({
                'age': [25, 30, 35, 40, 45, 50] * 10,
                'income': [30000, 40000, 50000, 60000, 70000, 80000] * 10,
            }),
            names=['age', 'income'],
        )

    def test_validate_causal_data_success(self):
        """Test successful validation."""
        warnings, errors = validate_causal_data(
            self.good_treatment,
            self.good_outcome,
            self.good_covariates,
            verbose=False
        )

        assert len(errors) == 0
        # May have warnings, that's ok

    def test_validate_causal_data_with_errors(self):
        """Test validation with errors."""
        bad_treatment = TreatmentData(
            values=pd.Series([0, 1, 2, 3]),  # Wrong values for binary
            name="treatment",
            treatment_type="binary",
        )

        with pytest.raises(DataValidationError):
            validate_causal_data(
                bad_treatment,
                self.good_outcome,
                verbose=False
            )

    def test_validate_causal_data_without_covariates(self):
        """Test validation without covariates."""
        warnings, errors = validate_causal_data(
            self.good_treatment,
            self.good_outcome,
            covariates=None,
            verbose=False
        )

        assert len(errors) == 0

    def test_validate_causal_data_no_overlap_check(self):
        """Test validation without overlap check."""
        warnings, errors = validate_causal_data(
            self.good_treatment,
            self.good_outcome,
            self.good_covariates,
            check_overlap=False,
            verbose=False
        )

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])
