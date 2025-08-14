"""Tests for the unified CausalAnalysis API.

This module tests the sklearn-style unified interface and HTML report generation
functionality for the causal inference library.
"""

import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from causal_inference.api import CausalAnalysis
from causal_inference.core.base import CausalEffect


class TestCausalAnalysisAPI:
    """Test the unified CausalAnalysis API."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 1000

        # Generate synthetic data
        age = np.random.normal(40, 10, n)
        income = np.random.normal(50000, 15000, n)

        # Treatment assignment (with some confounding)
        treatment_prob = 0.3 + 0.1 * (age - 40) / 10 + 0.1 * (income - 50000) / 15000
        treatment = np.random.binomial(1, np.clip(treatment_prob, 0.1, 0.9), n)

        # Outcome (with true treatment effect of 2.0)
        outcome = (
            10
            + 2.0 * treatment
            + 0.1 * age
            + 0.0001 * income
            + np.random.normal(0, 3, n)
        )

        return pd.DataFrame(
            {"treatment": treatment, "outcome": outcome, "age": age, "income": income}
        )

    def test_basic_initialization(self):
        """Test basic initialization of CausalAnalysis."""
        analysis = CausalAnalysis()

        assert analysis.method == "auto"
        assert analysis.is_fitted_ is False
        assert analysis.confidence_level == 0.95
        assert analysis.bootstrap_samples == 1000

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        analysis = CausalAnalysis(
            method="aipw",
            treatment_column="intervention",
            outcome_column="response",
            covariate_columns=["x1", "x2"],
            confidence_level=0.90,
            bootstrap_samples=500,
            random_state=123,
        )

        assert analysis.method == "aipw"
        assert analysis.treatment_column == "intervention"
        assert analysis.outcome_column == "response"
        assert analysis.covariate_columns == ["x1", "x2"]
        assert analysis.confidence_level == 0.90
        assert analysis.bootstrap_samples == 500
        assert analysis.random_state == 123

    def test_fit_with_auto_column_detection(self, sample_data):
        """Test fitting with automatic column detection."""
        analysis = CausalAnalysis(method="g_computation")

        # Should auto-detect columns
        result = analysis.fit(sample_data)

        assert result is analysis  # Method chaining
        assert analysis.is_fitted_ is True
        assert analysis.treatment_column == "treatment"
        assert analysis.outcome_column == "outcome"
        assert set(analysis.covariate_columns) == {"age", "income"}

    def test_fit_with_explicit_columns(self, sample_data):
        """Test fitting with explicitly specified columns."""
        analysis = CausalAnalysis(
            method="ipw",
            treatment_column="treatment",
            outcome_column="outcome",
            covariate_columns=["age"],
        )

        analysis.fit(sample_data)

        assert analysis.is_fitted_ is True
        assert analysis.treatment_column == "treatment"
        assert analysis.outcome_column == "outcome"
        assert analysis.covariate_columns == ["age"]

    def test_method_auto_selection(self, sample_data):
        """Test automatic method selection."""
        analysis = CausalAnalysis(method="auto")
        analysis.fit(sample_data)

        # With 1000 samples and covariates, should select AIPW
        assert analysis.method == "aipw"
        assert analysis.is_fitted_ is True

    def test_method_selection_small_sample(self):
        """Test method selection with small sample."""
        # Create small sample
        np.random.seed(42)
        small_data = pd.DataFrame(
            {
                "treatment": np.random.binomial(1, 0.5, 100),
                "outcome": np.random.normal(0, 1, 100),
                "covar": np.random.normal(0, 1, 100),
            }
        )

        analysis = CausalAnalysis(method="auto")
        analysis.fit(small_data)

        # With small sample, should select G-computation
        assert analysis.method == "g_computation"

    def test_estimate_ate(self, sample_data):
        """Test ATE estimation."""
        analysis = CausalAnalysis(method="g_computation")
        analysis.fit(sample_data)

        effect = analysis.estimate_ate()

        assert isinstance(effect, CausalEffect)
        assert hasattr(effect, "ate")
        assert hasattr(effect, "ate_ci_lower")
        assert hasattr(effect, "ate_ci_upper")
        assert hasattr(effect, "p_value")

        # Should be close to true effect of 2.0
        assert 1.5 < effect.ate < 2.5

    def test_estimate_ate_before_fit(self, sample_data):
        """Test that ATE estimation fails before fitting."""
        analysis = CausalAnalysis()

        with pytest.raises(
            ValueError, match="Must call fit\\(\\) before estimating effects"
        ):
            analysis.estimate_ate()

    def test_report_generation(self, sample_data):
        """Test HTML report generation."""
        analysis = CausalAnalysis(method="g_computation")
        analysis.fit(sample_data)

        report = analysis.report()

        assert isinstance(report, dict)
        assert "html_report" in report
        assert "effect" in report
        assert "method" in report
        assert "sample_size" in report

        # Check HTML content
        html = report["html_report"]
        assert isinstance(html, str)
        assert "<html" in html
        assert "Executive Summary" in html
        assert "Treatment Effect" in html

    def test_report_file_output(self, sample_data):
        """Test saving report to file."""
        analysis = CausalAnalysis(method="g_computation")
        analysis.fit(sample_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"

            report = analysis.report(output_path=str(output_path))

            assert output_path.exists()
            assert report["file_path"] == str(output_path)

            # Check file content
            with open(output_path, encoding="utf-8") as f:
                content = f.read()
                assert "<html" in content
                assert "Causal Analysis Report" in content

    def test_report_different_templates(self, sample_data):
        """Test different report templates."""
        analysis = CausalAnalysis(method="g_computation")
        analysis.fit(sample_data)

        # Test executive template
        exec_report = analysis.report(template="executive")
        assert "Executive Summary" in exec_report["html_report"]

        # Test technical template
        tech_report = analysis.report(template="technical")
        assert "Technical Appendix" in tech_report["html_report"]

        # Test full template
        full_report = analysis.report(template="full")
        assert "Technical Appendix" in full_report["html_report"]
        assert "Executive Summary" in full_report["html_report"]

    def test_sensitivity_analysis(self, sample_data):
        """Test sensitivity analysis integration."""
        analysis = CausalAnalysis(method="g_computation")
        analysis.fit(sample_data)

        # Mock sensitivity analysis to avoid complex dependencies
        with patch(
            "causal_inference.api.unified_estimator.generate_sensitivity_report"
        ) as mock_sensitivity:
            mock_sensitivity.return_value = {
                "overall_assessment": "HIGH ROBUSTNESS",
                "recommendations": ["Test recommendation"],
            }

            sensitivity_results = analysis.sensitivity_analysis()

            assert isinstance(sensitivity_results, dict)
            mock_sensitivity.assert_called_once()

    def test_predict_ite_not_supported(self, sample_data):
        """Test ITE prediction when not supported by estimator."""
        analysis = CausalAnalysis(method="g_computation")
        analysis.fit(sample_data)

        with pytest.warns(UserWarning, match="does not support ITE prediction"):
            ite_predictions = analysis.predict_ite()

        # Should return constant effect for all observations
        assert len(ite_predictions) == len(sample_data)
        assert np.all(ite_predictions == analysis.effect_.ate)

    def test_data_file_loading(self, sample_data):
        """Test loading data from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_data.csv"
            sample_data.to_csv(csv_path, index=False)

            analysis = CausalAnalysis()
            analysis.fit(str(csv_path))

            assert analysis.is_fitted_ is True
            assert len(analysis.data_) == len(sample_data)

    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        analysis = CausalAnalysis()

        with pytest.raises(ValueError, match="Unsupported file format"):
            analysis.fit("test_data.xyz")

    def test_no_binary_treatment_column(self):
        """Test error when no binary treatment column is found."""
        # Data with no binary columns
        data = pd.DataFrame(
            {
                "continuous_var": np.random.normal(0, 1, 100),
                "outcome": np.random.normal(0, 1, 100),
                "multi_category": np.random.choice(["a", "b", "c", "d"], 100),
            }
        )

        analysis = CausalAnalysis()

        with pytest.raises(ValueError, match="No binary treatment column detected"):
            analysis.fit(data)

    def test_multiple_binary_columns_warning(self):
        """Test warning when multiple binary columns exist and no heuristic match."""
        data = pd.DataFrame(
            {
                "binary_a": np.random.binomial(1, 0.5, 100),
                "binary_b": np.random.binomial(1, 0.3, 100),
                "outcome": np.random.normal(0, 1, 100),
            }
        )

        analysis = CausalAnalysis()

        with pytest.warns(UserWarning, match="Multiple binary columns found"):
            analysis.fit(data)

        # Should use first binary column since no heuristic match
        assert analysis.treatment_column == "binary_a"

    def test_multiple_binary_columns_heuristic_no_warning(self):
        """Test no warning when multiple binary columns exist but heuristic matches."""
        data = pd.DataFrame(
            {
                "treatment": np.random.binomial(1, 0.5, 100),
                "another_binary": np.random.binomial(1, 0.3, 100),
                "outcome": np.random.normal(0, 1, 100),
            }
        )

        analysis = CausalAnalysis()

        # Should not emit warning because 'treatment' matches heuristic
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # Filter out the p-value warning which is expected
            warnings.filterwarnings("ignore", message="P-value approximated.*")
            analysis.fit(data)

        # Should use 'treatment' due to name heuristic
        assert analysis.treatment_column == "treatment"

    def test_method_chaining(self, sample_data):
        """Test that method chaining works as expected."""
        # Test the main use case: CausalAnalysis().fit(data).report()
        report = CausalAnalysis(method="g_computation").fit(sample_data).report()

        assert isinstance(report, dict)
        assert "html_report" in report
        assert "effect" in report

    def test_repr_methods(self, sample_data):
        """Test string representations."""
        analysis = CausalAnalysis(method="ipw")

        # Before fitting
        repr_before = repr(analysis)
        assert "fitted=False" in repr_before
        assert "method='ipw'" in repr_before

        # After fitting
        analysis.fit(sample_data)
        repr_after = repr(analysis)
        assert "fitted=True" in repr_after
        assert "ate=" in repr_after


class TestColumnDetectionHeuristics:
    """Test automatic column detection heuristics."""

    def test_treatment_column_detection(self):
        """Test treatment column detection with common names."""
        test_cases = [
            ("treatment", ["treatment", "outcome", "covar"]),
            ("intervention", ["intervention", "response", "age"]),
            ("exposed", ["exposed", "result", "income"]),
            ("campaign", ["campaign", "revenue", "segment"]),
        ]

        for expected_treatment, columns in test_cases:
            data = pd.DataFrame(
                {
                    col: np.random.binomial(1, 0.5, 100)
                    if col == expected_treatment
                    else np.random.normal(0, 1, 100)
                    for col in columns
                }
            )

            analysis = CausalAnalysis()
            analysis.fit(data)

            assert analysis.treatment_column == expected_treatment

    def test_outcome_column_detection(self):
        """Test outcome column detection with common names."""
        test_cases = [
            ("revenue", ["treatment", "revenue", "age"]),
            ("conversion", ["exposed", "conversion", "segment"]),
            ("sales", ["campaign", "sales", "region"]),
            ("outcome", ["intervention", "outcome", "covariate"]),
        ]

        for expected_outcome, columns in test_cases:
            data = pd.DataFrame(
                {
                    col: np.random.binomial(1, 0.5, 100)
                    if col in ["treatment", "exposed", "campaign", "intervention"]
                    else np.random.normal(0, 1, 100)
                    for col in columns
                }
            )

            analysis = CausalAnalysis()
            analysis.fit(data)

            assert analysis.outcome_column == expected_outcome


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_data = pd.DataFrame()
        analysis = CausalAnalysis()

        with pytest.raises(ValueError):
            analysis.fit(empty_data)

    def test_single_column_dataframe(self):
        """Test handling of DataFrame with single column."""
        single_col_data = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        analysis = CausalAnalysis()

        with pytest.raises(ValueError):
            analysis.fit(single_col_data)

    def test_report_before_fit(self):
        """Test that report generation fails before fitting."""
        analysis = CausalAnalysis()

        with pytest.raises(
            ValueError, match="Must call fit\\(\\) before generating report"
        ):
            analysis.report()

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        analysis = CausalAnalysis(method="invalid_method")

        data = pd.DataFrame(
            {
                "treatment": np.random.binomial(1, 0.5, 100),
                "outcome": np.random.normal(0, 1, 100),
            }
        )

        with pytest.raises(ValueError, match="Unknown method"):
            analysis.fit(data)


if __name__ == "__main__":
    pytest.main([__file__])
