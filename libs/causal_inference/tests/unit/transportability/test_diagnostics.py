"""Tests for covariate shift diagnostics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from causal_inference.transportability.diagnostics import (
    CovariateShiftDiagnostics,
    DistributionDifference,
    ShiftSeverity,
)


class TestCovariateShiftDiagnostics:
    """Test suite for covariate shift diagnostics."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with known shift."""
        np.random.seed(42)

        # Source population
        n_source = 1000
        X_source, _ = make_classification(
            n_samples=n_source,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42,
        )

        # Target population with shift in first two features
        X_target = X_source.copy()
        X_target[:, 0] += 1.0  # Mean shift
        X_target[:, 1] *= 1.5  # Scale shift

        # Add some noise to target
        X_target += np.random.normal(0, 0.1, X_target.shape)

        return X_source, X_target

    @pytest.fixture
    def identical_data(self):
        """Generate identical source and target data."""
        np.random.seed(42)
        X, _ = make_classification(n_samples=500, n_features=3, random_state=42)
        return X, X.copy()

    def test_initialization(self):
        """Test CovariateShiftDiagnostics initialization."""
        diagnostics = CovariateShiftDiagnostics()

        assert diagnostics.min_samples == 50
        assert diagnostics.alpha == 0.05
        assert diagnostics.smd_threshold_mild == 0.1
        assert diagnostics.smd_threshold_moderate == 0.25
        assert diagnostics.distribution_differences == []

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        diagnostics = CovariateShiftDiagnostics(
            min_samples=100,
            alpha=0.01,
            smd_threshold_mild=0.05,
            smd_threshold_moderate=0.2,
        )

        assert diagnostics.min_samples == 100
        assert diagnostics.alpha == 0.01
        assert diagnostics.smd_threshold_mild == 0.05
        assert diagnostics.smd_threshold_moderate == 0.2

    def test_analyze_covariate_shift_basic(self, sample_data):
        """Test basic covariate shift analysis."""
        X_source, X_target = sample_data

        diagnostics = CovariateShiftDiagnostics()
        results = diagnostics.analyze_covariate_shift(X_source, X_target)

        # Check result structure
        assert "distribution_differences" in results
        assert "overall_shift_score" in results
        assert "discriminative_accuracy" in results
        assert "n_variables" in results
        assert "recommendations" in results

        # Check we have differences for each variable
        assert len(results["distribution_differences"]) == X_source.shape[1]

        # Check overall metrics are reasonable
        assert 0 <= results["overall_shift_score"] <= 5  # Should be reasonable
        assert 0.5 <= results["discriminative_accuracy"] <= 1.0

    def test_analyze_identical_data(self, identical_data):
        """Test analysis on identical source and target data."""
        X_source, X_target = identical_data

        diagnostics = CovariateShiftDiagnostics()
        results = diagnostics.analyze_covariate_shift(X_source, X_target)

        # Should have minimal shift
        assert results["overall_shift_score"] < 0.1
        assert results["n_severe_shifts"] == 0
        assert results["n_moderate_shifts"] == 0

        # All differences should be mild
        for diff in results["distribution_differences"]:
            assert diff.severity == ShiftSeverity.MILD
            assert abs(diff.standardized_mean_diff) < 0.1

    def test_with_dataframes(self, sample_data):
        """Test analysis with pandas DataFrames."""
        X_source, X_target = sample_data

        # Convert to DataFrames
        source_df = pd.DataFrame(
            X_source, columns=[f"var_{i}" for i in range(X_source.shape[1])]
        )
        target_df = pd.DataFrame(
            X_target, columns=[f"var_{i}" for i in range(X_target.shape[1])]
        )

        diagnostics = CovariateShiftDiagnostics()
        results = diagnostics.analyze_covariate_shift(source_df, target_df)

        # Should work the same as with arrays
        assert len(results["distribution_differences"]) == len(source_df.columns)

        # Check variable names are preserved
        var_names = [diff.variable_name for diff in results["distribution_differences"]]
        assert all(name.startswith("var_") for name in var_names)

    def test_custom_variable_names(self, sample_data):
        """Test analysis with custom variable names."""
        X_source, X_target = sample_data

        var_names = ["age", "income", "education", "experience", "location"]

        diagnostics = CovariateShiftDiagnostics()
        results = diagnostics.analyze_covariate_shift(
            X_source, X_target, variable_names=var_names
        )

        # Check variable names are used
        result_names = [
            diff.variable_name for diff in results["distribution_differences"]
        ]
        assert result_names == var_names

    def test_shift_severity_classification(self):
        """Test shift severity classification."""
        diagnostics = CovariateShiftDiagnostics()

        # Test classification thresholds
        assert diagnostics._classify_shift_severity(0.05) == ShiftSeverity.MILD
        assert diagnostics._classify_shift_severity(0.15) == ShiftSeverity.MODERATE
        assert diagnostics._classify_shift_severity(0.35) == ShiftSeverity.SEVERE

        # Test boundary cases
        assert diagnostics._classify_shift_severity(0.1) == ShiftSeverity.MILD
        assert diagnostics._classify_shift_severity(0.25) == ShiftSeverity.MODERATE

    def test_single_variable_analysis(self):
        """Test analysis of a single variable."""
        np.random.seed(42)

        # Create data with known difference
        source_values = np.random.normal(0, 1, 1000)
        target_values = np.random.normal(1, 1, 1000)  # Mean shift of 1

        diagnostics = CovariateShiftDiagnostics()
        diff = diagnostics._analyze_single_variable(
            source_values, target_values, "test_var"
        )

        # Check properties
        assert diff.variable_name == "test_var"
        assert abs(diff.standardized_mean_diff - 1.0) < 0.1  # Should be close to 1
        assert diff.source_mean < diff.target_mean  # Target should be higher
        assert diff.severity == ShiftSeverity.SEVERE  # SMD of 1 is severe
        assert diff.is_significant  # Should be statistically significant

    def test_recommendations_generation(self, sample_data):
        """Test recommendation generation."""
        X_source, X_target = sample_data

        diagnostics = CovariateShiftDiagnostics()
        results = diagnostics.analyze_covariate_shift(X_source, X_target)

        recommendations = results["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should have at least one recommendation
        assert any("shift" in rec.lower() for rec in recommendations)

    def test_summary_table_creation(self, sample_data):
        """Test summary table creation."""
        X_source, X_target = sample_data

        diagnostics = CovariateShiftDiagnostics()
        diagnostics.analyze_covariate_shift(X_source, X_target)

        summary_df = diagnostics.create_shift_summary_table()

        # Check structure
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == X_source.shape[1]

        # Check columns
        expected_cols = [
            "Variable",
            "SMD",
            "Abs_SMD",
            "KS_Statistic",
            "KS_P_Value",
            "Wasserstein",
            "Severity",
            "Effect_Size",
            "Significant",
            "Source_Mean",
            "Target_Mean",
        ]
        for col in expected_cols:
            assert col in summary_df.columns

        # Check sorting (should be by absolute SMD, descending)
        abs_smds = summary_df["Abs_SMD"].values
        assert np.all(abs_smds[:-1] >= abs_smds[1:])  # Should be descending

    def test_error_handling_mismatched_dimensions(self):
        """Test error handling for mismatched dimensions."""
        source_data = np.random.randn(100, 3)
        target_data = np.random.randn(100, 4)  # Different number of features

        diagnostics = CovariateShiftDiagnostics()

        with pytest.raises(ValueError, match="same number of variables"):
            diagnostics.analyze_covariate_shift(source_data, target_data)

    def test_error_handling_mismatched_variable_names(self):
        """Test error handling for mismatched variable names."""
        source_data = np.random.randn(100, 3)
        target_data = np.random.randn(100, 3)
        var_names = ["var1", "var2"]  # Wrong number of names

        diagnostics = CovariateShiftDiagnostics()

        with pytest.raises(ValueError, match="Number of variable names"):
            diagnostics.analyze_covariate_shift(source_data, target_data, var_names)

    def test_small_sample_warning(self):
        """Test warning for small sample sizes."""
        source_data = np.random.randn(10, 2)  # Very small sample
        target_data = np.random.randn(10, 2)

        diagnostics = CovariateShiftDiagnostics(min_samples=50)

        with pytest.warns(UserWarning, match="Sample sizes"):
            diagnostics.analyze_covariate_shift(source_data, target_data)

    def test_missing_values_handling(self):
        """Test handling of missing values."""
        np.random.seed(42)

        # Create data with missing values
        source_data = np.random.randn(200, 3)
        target_data = np.random.randn(200, 3)

        # Add some NaN values
        source_data[0:10, 0] = np.nan
        target_data[0:5, 1] = np.nan

        diagnostics = CovariateShiftDiagnostics()
        results = diagnostics.analyze_covariate_shift(source_data, target_data)

        # Should complete without error
        assert len(results["distribution_differences"]) == 3

        # Check that missing values are handled in single variable analysis
        diff = diagnostics._analyze_single_variable(
            source_data[:, 0], target_data[:, 0], "test_var"
        )
        assert not np.isnan(diff.standardized_mean_diff)


class TestDistributionDifference:
    """Test suite for DistributionDifference dataclass."""

    def test_initialization(self):
        """Test DistributionDifference initialization."""
        diff = DistributionDifference(
            variable_name="test_var",
            standardized_mean_diff=0.5,
            ks_statistic=0.2,
            ks_pvalue=0.01,
            wasserstein_distance=0.3,
            severity=ShiftSeverity.MODERATE,
            source_mean=1.0,
            target_mean=1.5,
            source_std=1.0,
            target_std=1.2,
        )

        assert diff.variable_name == "test_var"
        assert diff.standardized_mean_diff == 0.5
        assert diff.severity == ShiftSeverity.MODERATE

    def test_is_significant_property(self):
        """Test is_significant property."""
        # Significant difference
        diff_sig = DistributionDifference(
            variable_name="test",
            standardized_mean_diff=0.5,
            ks_statistic=0.2,
            ks_pvalue=0.01,  # < 0.05
            wasserstein_distance=0.3,
            severity=ShiftSeverity.MODERATE,
            source_mean=1.0,
            target_mean=1.5,
            source_std=1.0,
            target_std=1.2,
        )
        assert diff_sig.is_significant is True

        # Non-significant difference
        diff_nonsig = DistributionDifference(
            variable_name="test",
            standardized_mean_diff=0.1,
            ks_statistic=0.05,
            ks_pvalue=0.5,  # > 0.05
            wasserstein_distance=0.1,
            severity=ShiftSeverity.MILD,
            source_mean=1.0,
            target_mean=1.1,
            source_std=1.0,
            target_std=1.0,
        )
        assert diff_nonsig.is_significant is False

    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        # Negligible effect
        diff_negligible = DistributionDifference(
            variable_name="test",
            standardized_mean_diff=0.05,
            ks_statistic=0.1,
            ks_pvalue=0.5,
            wasserstein_distance=0.1,
            severity=ShiftSeverity.MILD,
            source_mean=1.0,
            target_mean=1.05,
            source_std=1.0,
            target_std=1.0,
        )
        assert diff_negligible.effect_size_interpretation == "negligible"

        # Small effect
        diff_small = DistributionDifference(
            variable_name="test",
            standardized_mean_diff=0.3,
            ks_statistic=0.2,
            ks_pvalue=0.1,
            wasserstein_distance=0.2,
            severity=ShiftSeverity.MODERATE,
            source_mean=1.0,
            target_mean=1.3,
            source_std=1.0,
            target_std=1.0,
        )
        assert diff_small.effect_size_interpretation == "small"

        # Large effect
        diff_large = DistributionDifference(
            variable_name="test",
            standardized_mean_diff=1.0,
            ks_statistic=0.5,
            ks_pvalue=0.001,
            wasserstein_distance=0.8,
            severity=ShiftSeverity.SEVERE,
            source_mean=1.0,
            target_mean=2.0,
            source_std=1.0,
            target_std=1.0,
        )
        assert diff_large.effect_size_interpretation == "large"


class TestShiftSeverity:
    """Test suite for ShiftSeverity enum."""

    def test_enum_values(self):
        """Test ShiftSeverity enum values."""
        assert ShiftSeverity.MILD.value == "mild"
        assert ShiftSeverity.MODERATE.value == "moderate"
        assert ShiftSeverity.SEVERE.value == "severe"

    def test_enum_comparison(self):
        """Test that enum values can be compared."""
        mild = ShiftSeverity.MILD
        moderate = ShiftSeverity.MODERATE
        severe = ShiftSeverity.SEVERE

        assert mild == ShiftSeverity.MILD
        assert moderate != mild
        assert severe != moderate
