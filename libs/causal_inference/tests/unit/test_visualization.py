"""Unit tests for the advanced visualization module.

This module tests all visualization components including Love plots,
weight diagnostics, propensity score analysis, residual analysis,
and automated report generation.
"""

from unittest.mock import patch

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.visualization import (
    DiagnosticReportGenerator,
    LovePlotGenerator,
    PropensityPlotGenerator,
    ResidualAnalyzer,
    WeightDiagnostics,
    create_love_plot,
    create_propensity_plots,
    create_residual_plots,
    create_weight_plots,
    generate_diagnostic_report,
)
from causal_inference.visualization.balance_plots import LovePlotData
from causal_inference.visualization.propensity_plots import PropensityOverlapResult
from causal_inference.visualization.residual_analysis import ResidualAnalysisResult
from causal_inference.visualization.weight_diagnostics import WeightDiagnosticsResult


class TestLovePlotGenerator:
    """Test Love plot generation and covariate balance visualization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200

        # Create treatment data
        treatment = np.random.binomial(1, 0.5, n)

        # Create covariates with some imbalance
        x1 = np.random.normal(0, 1, n) + 0.5 * treatment
        x2 = np.random.normal(2, 1.5, n) + 0.3 * treatment
        x3 = np.random.binomial(1, 0.3 + 0.2 * treatment, n)

        covariates = np.column_stack([x1, x2, x3])

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        covariate_data = CovariateData(
            values=covariates, names=["continuous_1", "continuous_2", "binary_1"]
        )

        return treatment_data, covariate_data

    def test_initialization(self):
        """Test Love plot generator initialization."""
        generator = LovePlotGenerator()

        assert generator.balance_threshold == 0.1
        assert generator.poor_balance_threshold == 0.25
        assert generator.style == "whitegrid"
        assert generator.figsize == (10, 8)

    def test_balance_data_calculation(self, sample_data):
        """Test calculation of balance data for Love plots."""
        treatment_data, covariate_data = sample_data
        generator = LovePlotGenerator()

        love_data = generator.calculate_balance_data(covariate_data, treatment_data)

        assert isinstance(love_data, LovePlotData)
        assert len(love_data.covariate_names) == 3
        assert len(love_data.smd_before) == 3
        assert love_data.smd_after is None
        assert love_data.balance_threshold == 0.1
        assert love_data.poor_balance_threshold == 0.25

        # Check that SMD values are reasonable
        assert all(np.isfinite(love_data.smd_before))
        assert all(np.abs(love_data.smd_before) < 5)  # Should be reasonable values

    def test_balance_data_with_weights(self, sample_data):
        """Test balance calculation with after-adjustment weights."""
        treatment_data, covariate_data = sample_data
        generator = LovePlotGenerator()

        # Create some example weights (uniform for simplicity)
        weights_after = np.ones(len(treatment_data.values))

        love_data = generator.calculate_balance_data(
            covariate_data, treatment_data, weights_after=weights_after
        )

        assert love_data.smd_after is not None
        assert len(love_data.smd_after) == 3
        assert all(np.isfinite(love_data.smd_after))

    @patch("causal_inference.visualization.balance_plots.PLOTTING_AVAILABLE", True)
    @patch("matplotlib.pyplot.show")
    def test_create_love_plot_static(self, mock_show, sample_data):
        """Test static Love plot creation."""
        treatment_data, covariate_data = sample_data
        generator = LovePlotGenerator()

        love_data = generator.calculate_balance_data(covariate_data, treatment_data)
        fig = generator.create_love_plot(love_data, interactive=False)

        # Check that a figure was created
        assert fig is not None
        # The figure should have the expected structure
        assert len(fig.axes) >= 1

    def test_convenience_function(self, sample_data):
        """Test the convenience function for creating Love plots."""
        treatment_data, covariate_data = sample_data

        with patch("matplotlib.pyplot.show"):
            fig = create_love_plot(
                covariate_data,
                treatment_data,
                balance_threshold=0.15,
                interactive=False,
            )

        assert fig is not None


class TestWeightDiagnostics:
    """Test weight distribution diagnostics and visualization."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample weight data for testing."""
        np.random.seed(42)

        # Create weights with some extreme values
        normal_weights = np.random.lognormal(0, 0.5, 180)
        extreme_weights = np.random.lognormal(2, 1, 20)
        weights = np.concatenate([normal_weights, extreme_weights])

        return weights

    def test_initialization(self):
        """Test weight diagnostics initialization."""
        analyzer = WeightDiagnostics()

        assert analyzer.extreme_weight_threshold == 10.0
        assert analyzer.trimming_percentile == 99.0
        assert analyzer.figsize == (12, 8)
        assert analyzer.style == "whitegrid"

    def test_weight_analysis(self, sample_weights):
        """Test comprehensive weight analysis."""
        analyzer = WeightDiagnostics()
        result = analyzer.analyze_weights(sample_weights)

        assert isinstance(result, WeightDiagnosticsResult)
        assert result.n_observations == len(sample_weights)
        assert result.min_weight >= 0
        assert result.max_weight >= result.min_weight
        assert result.mean_weight > 0
        assert result.median_weight > 0
        assert result.std_weight >= 0
        assert np.isfinite(result.skewness)
        assert np.isfinite(result.kurtosis)
        assert result.extreme_weight_count >= 0
        assert 0 <= result.extreme_weight_percentage <= 100
        assert result.effective_sample_size > 0
        assert result.effective_sample_size <= result.n_observations

        # Check weight summary statistics
        assert len(result.weight_summary) == 6
        assert "p50" in result.weight_summary
        assert result.weight_summary["p50"] == result.median_weight

    def test_extreme_weight_detection(self):
        """Test detection of extreme weights."""
        analyzer = WeightDiagnostics(extreme_weight_threshold=5.0)

        # Create weights with known extreme values
        weights = np.array([1, 2, 3, 4, 10, 15, 20])  # 3 extreme weights
        result = analyzer.analyze_weights(weights)

        assert result.extreme_weight_count == 3
        assert result.extreme_weight_percentage == pytest.approx(3 / 7 * 100, rel=1e-2)

    @patch("causal_inference.visualization.weight_diagnostics.PLOTTING_AVAILABLE", True)
    @patch("matplotlib.pyplot.show")
    def test_create_weight_plots(self, mock_show, sample_weights):
        """Test weight distribution plot creation."""
        analyzer = WeightDiagnostics()
        result = analyzer.analyze_weights(sample_weights)

        fig = analyzer.create_weight_plots(result, interactive=False)
        assert fig is not None

    def test_generate_recommendations(self, sample_weights):
        """Test recommendation generation."""
        analyzer = WeightDiagnostics()
        result = analyzer.analyze_weights(sample_weights)

        recommendations = analyzer.generate_recommendations(result)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

        # Check that recommendations contain expected patterns
        rec_text = " ".join(recommendations).lower()
        assert any(word in rec_text for word in ["weight", "effective", "sample"])

    def test_convenience_function(self, sample_weights):
        """Test the convenience function for weight diagnostics."""
        with patch("matplotlib.pyplot.show"):
            fig, result, recommendations = create_weight_plots(
                sample_weights, extreme_weight_threshold=5.0, interactive=False
            )

        assert fig is not None
        assert isinstance(result, WeightDiagnosticsResult)
        assert isinstance(recommendations, list)


class TestPropensityPlotGenerator:
    """Test propensity score visualization and overlap assessment."""

    @pytest.fixture
    def sample_propensity_data(self):
        """Create sample propensity score data."""
        np.random.seed(42)
        n = 200

        # Create treatment assignment
        treatment = np.random.binomial(1, 0.5, n)

        # Create propensity scores with some overlap issues
        # Treated units have higher propensity scores on average
        ps_treated = np.random.beta(6, 3, np.sum(treatment == 1))
        ps_control = np.random.beta(3, 6, np.sum(treatment == 0))

        propensity_scores = np.zeros(n)
        propensity_scores[treatment == 1] = ps_treated
        propensity_scores[treatment == 0] = ps_control

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")

        return propensity_scores, treatment_data

    def test_initialization(self):
        """Test propensity plot generator initialization."""
        generator = PropensityPlotGenerator()

        assert generator.overlap_threshold == 0.1
        assert generator.positivity_threshold == 0.01
        assert generator.figsize == (15, 10)
        assert generator.style == "whitegrid"

    def test_propensity_overlap_analysis(self, sample_propensity_data):
        """Test propensity score overlap analysis."""
        propensity_scores, treatment_data = sample_propensity_data
        generator = PropensityPlotGenerator()

        result = generator.analyze_propensity_overlap(propensity_scores, treatment_data)

        assert isinstance(result, PropensityOverlapResult)
        assert len(result.propensity_scores) == len(treatment_data.values)
        assert 0 <= result.overlap_percentage <= 100
        assert len(result.common_support_range) == 2
        assert result.common_support_range[0] <= result.common_support_range[1]
        assert result.positivity_violations >= 0
        assert 0 <= result.auc_score <= 1
        assert 0 <= result.brier_score <= 1
        assert np.isfinite(result.calibration_slope)
        assert np.isfinite(result.calibration_intercept)

    def test_positivity_violation_detection(self):
        """Test detection of positivity violations."""
        generator = PropensityPlotGenerator(positivity_threshold=0.1)

        # Create propensity scores with extreme values
        treatment = np.array([0, 0, 1, 1])
        propensity_scores = np.array(
            [0.05, 0.95, 0.05, 0.95]
        )  # 4 violations (2 below 0.1, 2 above 0.9)
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")

        result = generator.analyze_propensity_overlap(propensity_scores, treatment_data)

        assert result.positivity_violations == 4

    @patch("causal_inference.visualization.propensity_plots.PLOTTING_AVAILABLE", True)
    @patch("matplotlib.pyplot.show")
    def test_create_propensity_plots(self, mock_show, sample_propensity_data):
        """Test propensity score plot creation."""
        propensity_scores, treatment_data = sample_propensity_data
        generator = PropensityPlotGenerator()

        result = generator.analyze_propensity_overlap(propensity_scores, treatment_data)
        fig = generator.create_propensity_plots(result, interactive=False)

        assert fig is not None

    def test_generate_recommendations(self, sample_propensity_data):
        """Test propensity score recommendation generation."""
        propensity_scores, treatment_data = sample_propensity_data
        generator = PropensityPlotGenerator()

        result = generator.analyze_propensity_overlap(propensity_scores, treatment_data)
        recommendations = generator.generate_recommendations(result)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check for expected content
        rec_text = " ".join(recommendations).lower()
        assert any(
            word in rec_text for word in ["overlap", "propensity", "discrimination"]
        )

    def test_convenience_function(self, sample_propensity_data):
        """Test the convenience function for propensity plots."""
        propensity_scores, treatment_data = sample_propensity_data

        with patch("matplotlib.pyplot.show"):
            fig, result, recommendations = create_propensity_plots(
                propensity_scores, treatment_data, interactive=False
            )

        assert fig is not None
        assert isinstance(result, PropensityOverlapResult)
        assert isinstance(recommendations, list)


class TestResidualAnalyzer:
    """Test residual analysis and model diagnostics."""

    @pytest.fixture
    def sample_residual_data(self):
        """Create sample residual data."""
        np.random.seed(42)
        n = 100

        # Create fitted values
        fitted_values = np.random.uniform(0, 10, n)

        # Create residuals with some patterns
        residuals = np.random.normal(0, 1, n)

        # Add some heteroscedasticity
        residuals = residuals * (1 + 0.3 * fitted_values / 10)

        # Create design matrix
        design_matrix = np.random.normal(0, 1, (n, 3))
        design_matrix = np.column_stack([np.ones(n), design_matrix])  # Add intercept

        return residuals, fitted_values, design_matrix

    def test_initialization(self):
        """Test residual analyzer initialization."""
        analyzer = ResidualAnalyzer()

        assert analyzer.outlier_threshold == 2.5
        assert analyzer.leverage_threshold is None
        assert analyzer.influence_threshold == 1.0
        assert analyzer.figsize == (15, 12)
        assert analyzer.style == "whitegrid"

    def test_residual_analysis(self, sample_residual_data):
        """Test comprehensive residual analysis."""
        residuals, fitted_values, design_matrix = sample_residual_data
        analyzer = ResidualAnalyzer()

        result = analyzer.analyze_residuals(residuals, fitted_values, design_matrix)

        assert isinstance(result, ResidualAnalysisResult)
        assert len(result.residuals) == len(residuals)
        assert len(result.fitted_values) == len(fitted_values)
        assert len(result.standardized_residuals) == len(residuals)
        assert len(result.studentized_residuals) == len(residuals)

        # Check test statistics
        assert np.isfinite(result.shapiro_stat)
        assert 0 <= result.shapiro_pvalue <= 1
        assert np.isfinite(result.jarque_bera_stat)
        assert 0 <= result.jarque_bera_pvalue <= 1
        assert np.isfinite(result.breusch_pagan_stat)
        assert 0 <= result.breusch_pagan_pvalue <= 1

        # Check counts
        assert result.outlier_count >= 0
        assert result.high_leverage_count >= 0
        assert result.influential_count >= 0

        # Check boolean flags
        assert isinstance(result.normality_assumption_met, bool)
        assert isinstance(result.homoscedasticity_assumption_met, bool)
        assert isinstance(result.outliers_detected, bool)

    def test_outlier_detection(self):
        """Test outlier detection in residuals."""
        analyzer = ResidualAnalyzer(outlier_threshold=2.0)

        # Create residuals with known outliers
        residuals = np.array([0, 0, 0, 0, 3, -3, 0])  # 2 outliers
        fitted_values = np.ones(7)

        result = analyzer.analyze_residuals(residuals, fitted_values)

        assert result.outlier_count == 2
        assert result.outliers_detected is True

    @patch("causal_inference.visualization.residual_analysis.PLOTTING_AVAILABLE", True)
    @patch("matplotlib.pyplot.show")
    def test_create_residual_plots(self, mock_show, sample_residual_data):
        """Test residual plot creation."""
        residuals, fitted_values, design_matrix = sample_residual_data
        analyzer = ResidualAnalyzer()

        result = analyzer.analyze_residuals(residuals, fitted_values, design_matrix)
        fig = analyzer.create_residual_plots(result, interactive=False)

        assert fig is not None

    def test_generate_recommendations(self, sample_residual_data):
        """Test residual analysis recommendation generation."""
        residuals, fitted_values, design_matrix = sample_residual_data
        analyzer = ResidualAnalyzer()

        result = analyzer.analyze_residuals(residuals, fitted_values, design_matrix)
        recommendations = analyzer.generate_recommendations(result)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check for expected content
        rec_text = " ".join(recommendations).lower()
        assert any(
            word in rec_text for word in ["residual", "normal", "outlier", "assumption"]
        )

    def test_convenience_function(self, sample_residual_data):
        """Test the convenience function for residual analysis."""
        residuals, fitted_values, design_matrix = sample_residual_data

        with patch("matplotlib.pyplot.show"):
            fig, result, recommendations = create_residual_plots(
                residuals, fitted_values, design_matrix, interactive=False
            )

        assert fig is not None
        assert isinstance(result, ResidualAnalysisResult)
        assert isinstance(recommendations, list)


class TestDiagnosticReportGenerator:
    """Test automated diagnostic report generation."""

    @pytest.fixture
    def comprehensive_data(self):
        """Create comprehensive data for report testing."""
        np.random.seed(42)
        n = 200

        # Treatment data
        treatment = np.random.binomial(1, 0.5, n)
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")

        # Outcome data
        outcome = np.random.normal(5 + 2 * treatment, 2, n)
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")

        # Covariate data
        covariates = np.random.normal(0, 1, (n, 3))
        covariate_data = CovariateData(
            values=covariates, names=["age", "income", "education"]
        )

        # Weights
        weights = np.random.lognormal(0, 0.3, n)

        # Propensity scores
        propensity_scores = np.random.beta(2, 2, n)

        # Residuals and fitted values
        fitted_values = 5 + 2 * treatment + np.sum(covariates * 0.5, axis=1)
        residuals = outcome - fitted_values

        return {
            "treatment_data": treatment_data,
            "outcome_data": outcome_data,
            "covariate_data": covariate_data,
            "weights": weights,
            "propensity_scores": propensity_scores,
            "residuals": residuals,
            "fitted_values": fitted_values,
        }

    def test_initialization(self):
        """Test report generator initialization."""
        generator = DiagnosticReportGenerator()

        assert generator.template_dir is None
        assert generator.max_file_size_mb == 5.0
        assert isinstance(generator.love_plot_generator, LovePlotGenerator)
        assert isinstance(generator.weight_analyzer, WeightDiagnostics)
        assert isinstance(generator.propensity_generator, PropensityPlotGenerator)
        assert isinstance(generator.residual_analyzer, ResidualAnalyzer)

    @patch("matplotlib.pyplot.show")
    def test_comprehensive_report_generation(self, mock_show, comprehensive_data):
        """Test generation of comprehensive diagnostic report."""
        generator = DiagnosticReportGenerator()

        html_content = generator.generate_comprehensive_report(
            treatment_data=comprehensive_data["treatment_data"],
            outcome_data=comprehensive_data["outcome_data"],
            covariates=comprehensive_data["covariate_data"],
            weights=comprehensive_data["weights"],
            propensity_scores=comprehensive_data["propensity_scores"],
            residuals=comprehensive_data["residuals"],
            fitted_values=comprehensive_data["fitted_values"],
            estimator_name="Test Estimator",
            ate_estimate=2.1,
            ate_ci_lower=1.5,
            ate_ci_upper=2.7,
        )

        assert isinstance(html_content, str)
        assert len(html_content) > 1000  # Should be substantial
        assert "<!DOCTYPE html>" in html_content
        assert "Test Estimator" in html_content
        assert "2.1" in html_content  # ATE estimate

        # Check for presence of key sections
        assert "Covariate Balance" in html_content or "Love Plot" in html_content
        assert "Weight Distribution" in html_content or "weight" in html_content.lower()
        assert (
            "Propensity Score" in html_content or "propensity" in html_content.lower()
        )
        assert "Residual Analysis" in html_content or "residual" in html_content.lower()

    @patch("matplotlib.pyplot.show")
    def test_minimal_report_generation(self, mock_show):
        """Test report generation with minimal data."""
        generator = DiagnosticReportGenerator()

        # Create minimal data
        n = 50
        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.normal(0, 1, n)

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")

        html_content = generator.generate_comprehensive_report(
            treatment_data=treatment_data,
            outcome_data=outcome_data,
            estimator_name="Minimal Test",
        )

        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "Minimal Test" in html_content

    @patch("matplotlib.pyplot.show")
    def test_convenience_function(self, mock_show, comprehensive_data):
        """Test the convenience function for report generation."""
        html_content = generate_diagnostic_report(
            treatment_data=comprehensive_data["treatment_data"],
            outcome_data=comprehensive_data["outcome_data"],
            covariates=comprehensive_data["covariate_data"],
            estimator_name="Convenience Test",
        )

        assert isinstance(html_content, str)
        assert "Convenience Test" in html_content


class TestVisualizationIntegration:
    """Test integration between different visualization components."""

    @pytest.fixture
    def integration_data(self):
        """Create data for integration testing."""
        np.random.seed(42)
        n = 150

        # Create realistic causal inference scenario
        age = np.random.normal(45, 15, n)
        income = np.random.lognormal(10, 0.5, n)

        # Treatment assignment with confounding
        propensity = 1 / (1 + np.exp(-(0.1 * age - 0.0001 * income - 2)))
        treatment = np.random.binomial(1, propensity, n)

        # Outcome with treatment effect
        outcome = (
            50 + 0.5 * age + 0.0001 * income + 5 * treatment + np.random.normal(0, 5, n)
        )

        # Create data objects
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=np.column_stack([age, income]), names=["age", "income"]
        )

        return treatment_data, outcome_data, covariate_data, propensity

    @patch("matplotlib.pyplot.show")
    def test_full_workflow_integration(self, mock_show, integration_data):
        """Test the full workflow with all visualization components."""
        treatment_data, outcome_data, covariate_data, true_propensity = integration_data

        # Create IPW weights (simplified)
        weights = 1 / (
            true_propensity * treatment_data.values
            + (1 - true_propensity) * (1 - treatment_data.values)
        )

        # Create simple residuals
        fitted_values = np.mean(outcome_data.values)
        residuals = outcome_data.values - fitted_values

        # Test that all components work together
        # 1. Love plot
        love_fig = create_love_plot(
            covariate_data, treatment_data, weights_after=weights, interactive=False
        )
        assert love_fig is not None

        # 2. Weight diagnostics
        weight_fig, weight_result, weight_recs = create_weight_plots(
            weights, interactive=False
        )
        assert weight_fig is not None
        assert isinstance(weight_result, WeightDiagnosticsResult)
        assert isinstance(weight_recs, list)

        # 3. Propensity score analysis
        prop_fig, prop_result, prop_recs = create_propensity_plots(
            true_propensity, treatment_data, interactive=False
        )
        assert prop_fig is not None
        assert isinstance(prop_result, PropensityOverlapResult)
        assert isinstance(prop_recs, list)

        # 4. Residual analysis
        resid_fig, resid_result, resid_recs = create_residual_plots(
            residuals,
            fitted_values,
            design_matrix=covariate_data.values,
            interactive=False,
        )
        assert resid_fig is not None
        assert isinstance(resid_result, ResidualAnalysisResult)
        assert isinstance(resid_recs, list)

        # 5. Comprehensive report
        html_report = generate_diagnostic_report(
            treatment_data=treatment_data,
            outcome_data=outcome_data,
            covariates=covariate_data,
            weights=weights,
            propensity_scores=true_propensity,
            residuals=residuals,
            fitted_values=fitted_values,
            estimator_name="Integration Test Estimator",
            ate_estimate=5.2,
            ate_ci_lower=3.8,
            ate_ci_upper=6.6,
        )

        assert isinstance(html_report, str)
        assert len(html_report) > 5000  # Should be comprehensive

        # Check that all components are represented
        assert "Integration Test Estimator" in html_report
        assert "5.2" in html_report  # ATE estimate

    def test_error_handling_missing_dependencies(self):
        """Test graceful handling when plotting dependencies are missing."""
        with patch(
            "causal_inference.visualization.balance_plots.PLOTTING_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="Plotting libraries not available"):
                LovePlotGenerator()

        with patch(
            "causal_inference.visualization.weight_diagnostics.PLOTTING_AVAILABLE",
            False,
        ):
            with pytest.raises(ImportError, match="Plotting libraries not available"):
                WeightDiagnostics()

    def test_performance_requirements(self, integration_data):
        """Test that visualization meets performance requirements from issue."""
        treatment_data, outcome_data, covariate_data, _ = integration_data

        # Scale up to test performance (issue requires < 2s for 50k rows)
        # We'll test with smaller data but measure relative performance
        import time

        n_small = 1000
        n_large = 5000

        # Create larger datasets
        def create_large_data(n):
            np.random.seed(42)
            treatment = np.random.binomial(1, 0.5, n)
            outcome = np.random.normal(0, 1, n)
            covariates = np.random.normal(0, 1, (n, 5))

            return (
                TreatmentData(values=treatment, treatment_type="binary"),
                OutcomeData(values=outcome, outcome_type="continuous"),
                CovariateData(values=covariates, names=[f"x{i}" for i in range(5)]),
            )

        # Test with smaller dataset
        small_data = create_large_data(n_small)
        start_time = time.time()

        with patch("matplotlib.pyplot.show"):
            _ = generate_diagnostic_report(
                *small_data, estimator_name="Performance Test"
            )

        small_time = time.time() - start_time

        # Test with larger dataset
        large_data = create_large_data(n_large)
        start_time = time.time()

        with patch("matplotlib.pyplot.show"):
            _ = generate_diagnostic_report(
                *large_data, estimator_name="Performance Test"
            )

        large_time = time.time() - start_time

        # Performance should scale reasonably (not exponentially)
        time_ratio = large_time / small_time
        size_ratio = n_large / n_small

        # Should not be much worse than linear scaling
        assert time_ratio < size_ratio * 2, (
            f"Performance scaling is poor: {time_ratio:.2f}x time for "
            f"{size_ratio:.2f}x data"
        )


@pytest.mark.slow
class TestVisualizationPerformance:
    """Performance and scalability tests for visualization components."""

    def test_large_dataset_memory_usage(self):
        """Test memory efficiency with large datasets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        n = 50000
        np.random.seed(42)

        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.normal(0, 1, n)
        covariates = np.random.normal(0, 1, (n, 10))
        weights = np.random.lognormal(0, 0.3, n)

        TreatmentData(values=treatment, treatment_type="binary")
        OutcomeData(values=outcome, outcome_type="continuous")
        CovariateData(values=covariates, names=[f"covar_{i}" for i in range(10)])

        # Generate diagnostics
        with patch("matplotlib.pyplot.show"):
            analyzer = WeightDiagnostics()
            result = analyzer.analyze_weights(weights)
            _ = analyzer.create_weight_plots(result, interactive=False)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should be reasonable (< 500MB as per issue requirements)
        assert (
            memory_increase < 500
        ), f"Memory usage too high: {memory_increase:.1f}MB increase"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
