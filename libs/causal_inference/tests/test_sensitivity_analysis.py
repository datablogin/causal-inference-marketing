"""Tests for the comprehensive sensitivity analysis suite."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from causal_inference.sensitivity import (
    e_value,
    generate_sensitivity_report,
    negative_control,
    oster_delta,
    placebo_test,
    rosenbaum_bounds,
)


class TestEValue:
    """Tests for E-value calculation."""

    def test_evalue_basic_calculation(self):
        """Test basic E-value calculation for risk ratio."""
        result = e_value(observed_estimate=2.0, ci_lower=1.5, ci_upper=2.5)

        assert isinstance(result, dict)
        assert "evalue_point" in result
        assert "evalue_ci" in result
        assert "interpretation" in result

        # E-value for RR=2.0 should be 2 + sqrt(2*1) = 3.414
        assert_almost_equal(result["evalue_point"], 3.414, decimal=2)

    def test_evalue_protective_effect(self):
        """Test E-value for protective effects (RR < 1)."""
        result = e_value(observed_estimate=0.5, ci_lower=0.3, ci_upper=0.8)

        assert result["evalue_point"] > 1.0
        assert (
            "protective" in result["interpretation"].lower()
            or "below" in result["interpretation"].lower()
        )

    def test_evalue_invalid_inputs(self):
        """Test E-value with invalid inputs."""
        with pytest.raises(ValueError, match="must be positive"):
            e_value(observed_estimate=0.0)

        with pytest.raises(ValueError, match="cannot exceed"):
            e_value(observed_estimate=2.0, ci_lower=3.0, ci_upper=1.0)

    def test_evalue_interpretation_thresholds(self):
        """Test E-value interpretation thresholds."""
        # Low E-value
        result_low = e_value(observed_estimate=1.1)
        assert "low" in result_low["interpretation"].lower()

        # High E-value
        result_high = e_value(observed_estimate=5.0)
        assert (
            "high" in result_high["interpretation"].lower()
            or "strong" in result_high["interpretation"].lower()
        )


class TestRosenbaumBounds:
    """Tests for Rosenbaum bounds analysis."""

    def test_rosenbaum_basic_functionality(self):
        """Test basic Rosenbaum bounds calculation."""
        np.random.seed(42)
        treated = np.random.normal(2, 1, 50)
        control = np.random.normal(0, 1, 50)

        result = rosenbaum_bounds(treated, control)

        assert isinstance(result, dict)
        assert "original_p_value" in result
        assert "critical_gamma" in result
        assert "bounds" in result
        assert "robustness_assessment" in result

        # Should have bounds for multiple gamma values
        assert len(result["bounds"]) > 1

    def test_rosenbaum_insufficient_data(self):
        """Test Rosenbaum bounds with insufficient data."""
        treated = [1, 2]
        control = [0, 1]

        with pytest.raises(ValueError, match="Minimum of 5 pairs"):
            rosenbaum_bounds(treated, control)

    def test_rosenbaum_unequal_lengths(self):
        """Test Rosenbaum bounds with unequal group sizes."""
        treated = np.random.normal(1, 1, 20)
        control = np.random.normal(0, 1, 30)

        with pytest.raises(ValueError, match="same length"):
            rosenbaum_bounds(treated, control)

    def test_rosenbaum_gamma_interpretation(self):
        """Test that higher gamma values produce higher p-values."""
        np.random.seed(42)
        treated = np.random.normal(3, 1, 30)  # Strong effect
        control = np.random.normal(0, 1, 30)

        result = rosenbaum_bounds(
            treated, control, gamma_range=(1.0, 2.0), gamma_steps=5
        )

        # P-values should generally increase with gamma
        p_upper_values = [bound["p_value_upper"] for bound in result["bounds"]]
        assert len(p_upper_values) >= 3

        # At least some trend toward higher p-values with higher gamma
        assert p_upper_values[-1] >= p_upper_values[0]


class TestOsterDelta:
    """Tests for Oster's δ analysis."""

    def test_oster_basic_calculation(self):
        """Test basic Oster delta calculation."""
        np.random.seed(42)
        n = 500

        # Simulate data with omitted variable bias
        X1 = np.random.normal(0, 1, n)  # Observed confounder
        X2 = np.random.normal(0, 1, n)  # Additional control
        U = np.random.normal(0, 1, n)  # Unobserved confounder

        T = 0.5 * X1 + 0.3 * U + np.random.normal(0, 1, n)
        Y = 2.0 * T + X1 + X2 + 0.5 * U + np.random.normal(0, 1, n)

        result = oster_delta(
            outcome=Y,
            treatment=T,
            covariates_restricted=X1.reshape(-1, 1),
            covariates_full=np.column_stack([X1, X2]),
        )

        assert isinstance(result, dict)
        assert "beta_restricted" in result
        assert "beta_full" in result
        assert "beta_star" in result
        assert "robustness_ratio" in result
        assert "interpretation" in result

        # Beta coefficients should be reasonable
        assert abs(result["beta_restricted"]) > 0
        assert abs(result["beta_full"]) > 0

    def test_oster_no_improvement(self):
        """Test Oster analysis when full model doesn't improve fit."""
        np.random.seed(42)
        n = 200

        T = np.random.binomial(1, 0.5, n)
        X = np.random.normal(0, 1, n)
        Y = 2 * T + np.random.normal(0, 1, n)  # X doesn't affect Y

        result = oster_delta(
            outcome=Y,
            treatment=T,
            covariates_restricted=None,
            covariates_full=X.reshape(-1, 1),
        )

        # When there's no improvement, beta_star should equal beta_full
        assert_almost_equal(result["beta_star"], result["beta_full"], decimal=3)

    def test_oster_data_validation(self):
        """Test Oster analysis input validation."""
        T = np.array([1, 0, 1])
        Y = np.array([2, 1])  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            oster_delta(outcome=Y, treatment=T)


class TestNegativeControl:
    """Tests for negative control analysis."""

    def test_negative_control_no_violations(self):
        """Test negative control analysis with no violations."""
        np.random.seed(42)
        n = 500

        # Main variables
        T = np.random.binomial(1, 0.5, n)
        Y = 2 * T + np.random.normal(0, 1, n)

        # Negative controls (should show no effects)
        neg_outcome = np.random.normal(0, 1, n)
        neg_exposure = np.random.binomial(1, 0.3, n)

        result = negative_control(
            treatment=T,
            outcome=Y,
            negative_control_outcome=neg_outcome,
            negative_control_exposure=neg_exposure,
        )

        assert isinstance(result, dict)
        assert "overall_assessment" in result
        assert "n_violations" in result
        assert "interpretation" in result

        # Should detect no violations with random negative controls
        assert result["n_violations"] == 0
        assert "no assumption violations" in result["overall_assessment"].lower()

    def test_negative_control_with_violations(self):
        """Test negative control analysis detecting violations."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n)
        Y = 2 * T + np.random.normal(0, 1, n)

        # Negative controls with spurious associations
        neg_outcome = T + np.random.normal(0, 0.1, n)  # Correlated with treatment
        neg_exposure = np.random.binomial(1, 0.3, n)

        result = negative_control(
            treatment=T,
            outcome=Y,
            negative_control_outcome=neg_outcome,
            negative_control_exposure=neg_exposure,
        )

        # Should detect at least one violation
        assert result["n_violations"] >= 1
        assert "violation" in result["overall_assessment"].lower()

    def test_negative_control_data_validation(self):
        """Test negative control input validation."""
        T = np.array([1, 0, 1])
        Y = np.array([2, 1, 3])
        neg_outcome = np.array([1, 0])  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            negative_control(
                treatment=T, outcome=Y, negative_control_outcome=neg_outcome
            )


class TestPlaceboTest:
    """Tests for placebo tests."""

    def test_placebo_random_treatment(self):
        """Test placebo test with random treatment assignment."""
        np.random.seed(42)
        n = 300

        T = np.random.binomial(1, 0.5, n)
        X = np.random.normal(0, 1, (n, 2))
        Y = 2 * T + X.sum(axis=1) + np.random.normal(0, 1, n)

        result = placebo_test(
            treatment=T,
            outcome=Y,
            covariates=X,
            placebo_type="random_treatment",
            n_placebo_tests=20,
            random_state=42,
        )

        assert isinstance(result, dict)
        assert "false_positive_rate" in result
        assert "mean_placebo_effect" in result
        assert "passes_placebo_test" in result
        assert "individual_results" in result

        # Should have reasonable false positive rate
        assert 0 <= result["false_positive_rate"] <= 1

        # Mean placebo effect should be small
        assert abs(result["mean_placebo_effect"]) < 1.0

    def test_placebo_dummy_outcome(self):
        """Test placebo test with dummy outcome."""
        np.random.seed(42)
        n = 200

        T = np.random.binomial(1, 0.5, n)
        Y = 2 * T + np.random.normal(0, 1, n)

        result = placebo_test(
            treatment=T,
            outcome=Y,
            placebo_type="dummy_outcome",
            n_placebo_tests=15,
            random_state=42,
        )

        # With dummy outcomes, should generally pass placebo test
        assert result["n_tests_run"] > 0
        assert abs(result["mean_placebo_effect"]) < 2.0

    def test_placebo_invalid_type(self):
        """Test placebo test with invalid placebo type."""
        T = np.array([1, 0, 1, 0])
        Y = np.array([2, 1, 3, 1])

        with pytest.raises(ValueError, match="Unknown placebo_type"):
            placebo_test(T, Y, placebo_type="invalid_type")


class TestSensitivityReporting:
    """Tests for unified sensitivity analysis reporting."""

    def test_generate_basic_report(self):
        """Test basic sensitivity report generation."""
        np.random.seed(42)
        n = 200

        T = np.random.binomial(1, 0.5, n)
        Y = 2 * T + np.random.normal(0, 1, n)
        X = np.random.normal(0, 1, (n, 2))

        result = generate_sensitivity_report(
            treatment_data=T,
            outcome_data=Y,
            covariates_data=X,
            observed_effect=2.0,
            ci_lower=1.8,
            ci_upper=2.2,
            include_placebo=True,
        )

        assert isinstance(result, dict)
        assert "report_html" in result
        assert "overall_assessment" in result
        assert "summary_results" in result
        assert "recommendations" in result

        # HTML report should contain key elements
        html = result["report_html"]
        assert "Sensitivity Analysis Report" in html
        assert "Overall Assessment" in html
        assert len(result["recommendations"]) > 0

    def test_generate_report_with_errors(self):
        """Test report generation handling individual analysis failures."""
        # Very small dataset that might cause some analyses to fail
        T = np.array([1, 0])
        Y = np.array([2, 1])

        result = generate_sensitivity_report(
            treatment_data=T,
            outcome_data=Y,
            observed_effect=1.0,
            include_rosenbaum=True,  # Should fail with insufficient data
            include_evalue=True,  # Should work
        )

        # Report should still be generated despite some failures
        assert "report_html" in result
        assert "overall_assessment" in result

    def test_generate_report_file_output(self, tmp_path):
        """Test saving report to file."""
        np.random.seed(42)
        n = 100

        T = np.random.binomial(1, 0.5, n)
        Y = 2 * T + np.random.normal(0, 1, n)

        output_path = tmp_path / "test_report.html"

        result = generate_sensitivity_report(
            treatment_data=T,
            outcome_data=Y,
            observed_effect=2.0,
            output_path=str(output_path),
        )

        # File should be created
        assert output_path.exists()
        assert result["file_path"] == str(output_path)

        # File should contain HTML report
        with open(output_path) as f:
            content = f.read()
            assert "Sensitivity Analysis Report" in content


class TestIntegrationValidation:
    """Integration tests to validate sensitivity analysis results against known examples."""

    def test_known_evalue_results(self):
        """Test E-values against published examples."""
        # Example from VanderWeele & Ding paper
        result = e_value(observed_estimate=3.9, ci_lower=1.8, ci_upper=8.7)

        # E-value for point estimate should be around 7.26
        assert_almost_equal(result["evalue_point"], 7.26, decimal=1)

    def test_comprehensive_analysis_validation(self):
        """Test comprehensive analysis with controlled synthetic data."""
        np.random.seed(42)
        n = 1000

        # Create data with known properties
        U = np.random.normal(0, 1, n)  # Unmeasured confounder
        X1 = U + np.random.normal(0, 1, n)  # Observed confounder
        X2 = np.random.normal(0, 1, n)  # Additional control

        # Treatment affected by confounders
        T = (0.5 * U + 0.3 * X1 + np.random.normal(0, 1, n)) > 0
        T = T.astype(int)

        # Outcome affected by treatment and confounders
        Y = 1.5 * T + U + 0.5 * X1 + np.random.normal(0, 1, n)

        # Negative controls
        neg_outcome = np.random.normal(0, 1, n)  # Should show no effect
        neg_exposure = np.random.binomial(1, 0.3, n)  # Should show no effect

        # Run comprehensive analysis
        report = generate_sensitivity_report(
            treatment_data=T,
            outcome_data=Y,
            covariates_data=np.column_stack([X1, X2]),
            observed_effect=1.2,  # Approximate expected effect
            ci_lower=0.9,
            ci_upper=1.5,
            include_negative_controls=True,
            include_placebo=True,
            negative_control_outcome=neg_outcome,
            negative_control_exposure=neg_exposure,
            covariates_restricted=X1.reshape(-1, 1),
            include_oster=True,
        )

        # Validate overall structure
        assert (
            "HIGH" in report["overall_assessment"]
            or "MODERATE" in report["overall_assessment"]
        )
        assert len(report["recommendations"]) >= 3
        assert len(report["summary_results"]) >= 3

        # Negative controls should generally pass
        neg_results = report["individual_results"].get("negative_control")
        if neg_results and "error" not in neg_results:
            assert neg_results["n_violations"] <= 1  # Allow for some random variation

    def test_performance_requirements(self):
        """Test that analyses meet performance requirements from issue."""
        import time

        np.random.seed(42)
        n = 10000  # Large dataset to test performance

        T = np.random.binomial(1, 0.5, n)
        Y = 2 * T + np.random.normal(0, 1, n)

        # Test E-value performance
        start_time = time.time()
        e_result = e_value(observed_estimate=2.0, ci_lower=1.8, ci_upper=2.2)
        e_time = time.time() - start_time

        assert e_time < 2.0, f"E-value took {e_time:.2f}s, should be < 2s"

        # Test placebo test performance (reduced number of tests for speed)
        start_time = time.time()
        _ = placebo_test(T[:1000], Y[:1000], n_placebo_tests=10)
        placebo_time = time.time() - start_time

        assert placebo_time < 10.0, f"Placebo tests took {placebo_time:.2f}s"

        # Validate that results are interpretable (no complex technical jargon)
        assert "unmeasured confounding" in e_result["interpretation"].lower()
        assert "robustness" in e_result["interpretation"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
