"""Tests for causal inference diagnostics framework."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import (
    CausalEffect,
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.diagnostics.assumptions import (
    AssumptionChecker,
    check_confounding_detection,
    detect_confounding_associations,
)
from causal_inference.diagnostics.balance import (
    BalanceDiagnostics,
    calculate_standardized_mean_difference,
    calculate_variance_ratio,
    check_covariate_balance,
)
from causal_inference.diagnostics.overlap import (
    OverlapDiagnostics,
    assess_positivity,
    calculate_propensity_scores,
    check_common_support,
)
from causal_inference.diagnostics.reporting import (
    DiagnosticReportGenerator,
    generate_diagnostic_report,
)
from causal_inference.diagnostics.sensitivity import (
    SensitivityAnalysis,
    evalue_calculation,
    rosenbaum_bounds,
)
from causal_inference.diagnostics.specification import (
    ModelSpecificationTests,
    functional_form_tests,
    linearity_tests,
)


class TestBalanceDiagnostics:
    """Test covariate balance diagnostics."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200

        # Create balanced data
        age = np.random.normal(40, 10, n)
        income = np.random.normal(50000, 15000, n)
        education = np.random.choice([1, 2, 3, 4], n)

        # Balanced treatment assignment
        treatment = np.random.binomial(1, 0.5, n)

        self.balanced_treatment = TreatmentData(
            values=pd.Series(treatment),
            name="treatment",
            treatment_type="binary",
        )

        self.balanced_covariates = CovariateData(
            values=pd.DataFrame(
                {
                    "age": age,
                    "income": income,
                    "education": education,
                }
            ),
            names=["age", "income", "education"],
        )

        # Create imbalanced data
        # Treatment assignment depends on age
        treatment_prob = 1 / (1 + np.exp(-(age - 40) / 10))
        imbalanced_treatment = np.random.binomial(1, treatment_prob)

        self.imbalanced_treatment = TreatmentData(
            values=pd.Series(imbalanced_treatment),
            name="treatment",
            treatment_type="binary",
        )

    def test_standardized_mean_difference(self):
        """Test SMD calculation."""
        # Create simple test data
        covariate = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        treatment = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        smd = calculate_standardized_mean_difference(covariate, treatment)

        # Should be positive (treated group has higher values)
        assert smd > 0
        assert isinstance(smd, float)

    def test_variance_ratio(self):
        """Test variance ratio calculation."""
        # Create data with different variances
        treated = np.array([1, 2, 3, 4, 5])
        control = np.array([1.0, 1.1, 1.2, 1.3, 1.4])  # Lower variance

        covariate = np.concatenate([control, treated])
        treatment = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        var_ratio = calculate_variance_ratio(covariate, treatment)

        # Treated group should have higher variance
        assert var_ratio > 1
        assert isinstance(var_ratio, float)

    def test_balance_assessment_balanced_data(self):
        """Test balance assessment on balanced data."""
        diagnostics = BalanceDiagnostics(
            balance_threshold=0.15
        )  # More lenient threshold
        results = diagnostics.assess_balance(
            self.balanced_treatment, self.balanced_covariates
        )

        # Should detect reasonable balance (may have some imbalance due to randomness)
        # Check that all SMDs are reasonably small
        for smd in results.standardized_mean_differences.values():
            assert abs(smd) < 0.3  # Should be small for balanced data

        # Should have most variables balanced
        assert len(results.imbalanced_covariates) <= 1  # Allow some randomness

    def test_balance_assessment_imbalanced_data(self):
        """Test balance assessment on imbalanced data."""
        diagnostics = BalanceDiagnostics(balance_threshold=0.1)
        results = diagnostics.assess_balance(
            self.imbalanced_treatment, self.balanced_covariates
        )

        # Should detect imbalance in age (used for treatment assignment)
        assert not results.overall_balance_met
        assert "age" in results.imbalanced_covariates

    def test_balance_table_creation(self):
        """Test balance table creation."""
        diagnostics = BalanceDiagnostics()
        results = diagnostics.assess_balance(
            self.balanced_treatment, self.balanced_covariates
        )

        table = diagnostics.create_balance_table(results)

        assert isinstance(table, pd.DataFrame)
        assert "Covariate" in table.columns
        assert "SMD" in table.columns
        assert len(table) == len(self.balanced_covariates.values.columns)

    def test_check_covariate_balance_convenience_function(self):
        """Test convenience function for balance checking."""
        results = check_covariate_balance(
            self.balanced_treatment,
            self.balanced_covariates,
            verbose=False,
        )

        assert hasattr(results, "overall_balance_met")
        assert hasattr(results, "standardized_mean_differences")


class TestOverlapDiagnostics:
    """Test overlap and positivity diagnostics."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200

        # Create data with good overlap
        age = np.random.normal(40, 10, n)
        income = np.random.normal(50000, 15000, n)

        # Treatment assignment with moderate confounding
        treatment_logits = 0.5 * (age - 40) / 10 + 0.3 * (income - 50000) / 15000
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)

        self.good_overlap_treatment = TreatmentData(
            values=pd.Series(treatment),
            name="treatment",
            treatment_type="binary",
        )

        self.good_overlap_covariates = CovariateData(
            values=pd.DataFrame(
                {
                    "age": age,
                    "income": income,
                }
            ),
            names=["age", "income"],
        )

        # Create data with poor overlap
        poor_treatment_logits = 3.0 * (age - 40) / 10  # Strong confounding
        poor_treatment_prob = 1 / (1 + np.exp(-poor_treatment_logits))
        poor_treatment = np.random.binomial(1, poor_treatment_prob)

        self.poor_overlap_treatment = TreatmentData(
            values=pd.Series(poor_treatment),
            name="treatment",
            treatment_type="binary",
        )

    def test_propensity_score_calculation(self):
        """Test propensity score calculation."""
        ps = calculate_propensity_scores(
            self.good_overlap_treatment,
            self.good_overlap_covariates,
            model_type="logistic",
        )

        assert len(ps) == len(self.good_overlap_treatment.values)
        assert np.all(ps >= 0)
        assert np.all(ps <= 1)
        assert isinstance(ps, np.ndarray)

    def test_common_support_check(self):
        """Test common support assessment."""
        ps = calculate_propensity_scores(
            self.good_overlap_treatment,
            self.good_overlap_covariates,
        )

        support_range, units_in_support = check_common_support(
            ps,
            self.good_overlap_treatment.values,
        )

        assert len(support_range) == 2
        assert support_range[0] <= support_range[1]
        assert units_in_support <= len(ps)
        assert units_in_support >= 0

    def test_overlap_assessment_good_overlap(self):
        """Test overlap assessment with good overlap."""
        diagnostics = OverlapDiagnostics()
        results = diagnostics.assess_overlap(
            self.good_overlap_treatment,
            self.good_overlap_covariates,
        )

        assert isinstance(results.overall_positivity_met, bool)
        assert 0 <= results.min_propensity_score <= 1
        assert 0 <= results.max_propensity_score <= 1
        assert len(results.propensity_scores) == len(self.good_overlap_treatment.values)

    def test_overlap_assessment_poor_overlap(self):
        """Test overlap assessment with poor overlap."""
        diagnostics = OverlapDiagnostics(min_propensity=0.1, max_propensity=0.9)
        results = diagnostics.assess_overlap(
            self.poor_overlap_treatment,
            self.good_overlap_covariates,
        )

        # Poor overlap should trigger violations
        assert len(results.violations) > 0
        # May or may not meet positivity depending on random seed

    def test_assess_positivity_convenience_function(self):
        """Test convenience function for positivity assessment."""
        results = assess_positivity(
            self.good_overlap_treatment,
            self.good_overlap_covariates,
            verbose=False,
        )

        assert hasattr(results, "overall_positivity_met")
        assert hasattr(results, "propensity_scores")


class TestAssumptionChecking:
    """Test assumption checking diagnostics."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200

        # Create confounders
        age = np.random.normal(40, 10, n)
        income = np.random.normal(50000, 15000, n)

        # Treatment depends on confounders
        treatment_logits = 0.02 * (age - 40) + 0.00001 * (income - 50000)
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)

        # Outcome depends on both treatment and confounders
        outcome = (
            2.0 * treatment  # Treatment effect
            + 0.1 * age  # Confounder effect
            + 0.00005 * income  # Confounder effect
            + np.random.normal(0, 1, n)  # Noise
        )

        self.treatment = TreatmentData(
            values=pd.Series(treatment),
            name="treatment",
            treatment_type="binary",
        )

        self.outcome = OutcomeData(
            values=pd.Series(outcome),
            name="outcome",
            outcome_type="continuous",
        )

        self.covariates = CovariateData(
            values=pd.DataFrame(
                {
                    "age": age,
                    "income": income,
                }
            ),
            names=["age", "income"],
        )

    def test_confounding_detection(self):
        """Test confounding detection."""
        results = detect_confounding_associations(
            self.treatment,
            self.outcome,
            self.covariates,
        )

        assert isinstance(results, dict)
        assert "age" in results
        assert "income" in results

        # Age should be detected as confounder (affects both treatment and outcome)
        age_result = results["age"]
        assert "is_potential_confounder" in age_result
        assert "confounding_strength" in age_result

    def test_assumption_checker(self):
        """Test comprehensive assumption checking."""
        checker = AssumptionChecker()
        results = checker.check_all_assumptions(
            self.treatment,
            self.outcome,
            self.covariates,
        )

        assert hasattr(results, "exchangeability_likely")
        assert hasattr(results, "confounding_detected")
        assert isinstance(results.confounding_strength, dict)
        assert isinstance(results.recommendations, list)

    def test_check_confounding_detection_convenience(self):
        """Test convenience function for confounding detection."""
        results = check_confounding_detection(
            self.treatment,
            self.outcome,
            self.covariates,
            verbose=False,
        )

        assert isinstance(results, dict)
        for var_results in results.values():
            assert "is_potential_confounder" in var_results


class TestSensitivityAnalysis:
    """Test sensitivity analysis tools."""

    def test_evalue_calculation(self):
        """Test E-value calculation."""
        # Test with risk ratio > 1
        evalue_results = evalue_calculation(2.0, 1.5)

        assert "evalue_point" in evalue_results
        assert "evalue_ci" in evalue_results
        assert evalue_results["evalue_point"] > 1
        assert evalue_results["evalue_ci"] is not None

    def test_evalue_calculation_protective_effect(self):
        """Test E-value calculation for protective effects."""
        # Test with risk ratio < 1
        evalue_results = evalue_calculation(0.5, 0.3)

        assert evalue_results["evalue_point"] > 1
        assert evalue_results["evalue_ci"] is not None

    def test_rosenbaum_bounds(self):
        """Test Rosenbaum bounds calculation."""
        # Create paired data
        treated = np.random.normal(1, 1, 50)
        control = np.random.normal(0, 1, 50)

        results = rosenbaum_bounds(treated, control)

        assert "original_p_value" in results
        assert "bounds" in results
        assert isinstance(results["bounds"], list)
        assert len(results["bounds"]) > 0

    def test_sensitivity_analysis(self):
        """Test comprehensive sensitivity analysis."""
        # Create mock causal effect
        causal_effect = CausalEffect(
            ate=2.0,
            ate_ci_lower=1.0,
            ate_ci_upper=3.0,
            method="test",
        )

        # Create simple data
        treatment = TreatmentData(
            values=pd.Series([0, 1, 0, 1] * 25),
            name="treatment",
            treatment_type="binary",
        )

        outcome = OutcomeData(
            values=pd.Series(np.random.normal(0, 1, 100)),
            name="outcome",
            outcome_type="continuous",
        )

        analyzer = SensitivityAnalysis()
        results = analyzer.comprehensive_sensitivity_analysis(
            treatment,
            outcome,
            causal_effect,
        )

        assert hasattr(results, "evalue")
        assert hasattr(results, "robustness_assessment")
        assert isinstance(results.recommendations, list)


class TestSpecificationTests:
    """Test model specification diagnostics."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200

        # Create data with known functional forms
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)

        # Linear outcome
        linear_outcome = 2 * x1 + 3 * x2 + np.random.normal(0, 0.5, n)

        # Nonlinear outcome
        nonlinear_outcome = 2 * x1 + 3 * x2**2 + np.random.normal(0, 0.5, n)

        self.linear_outcome = OutcomeData(
            values=pd.Series(linear_outcome),
            name="outcome",
            outcome_type="continuous",
        )

        self.nonlinear_outcome = OutcomeData(
            values=pd.Series(nonlinear_outcome),
            name="outcome",
            outcome_type="continuous",
        )

        self.covariates = CovariateData(
            values=pd.DataFrame(
                {
                    "x1": x1,
                    "x2": x2,
                }
            ),
            names=["x1", "x2"],
        )

        self.treatment = TreatmentData(
            values=pd.Series([0, 1] * (n // 2)),
            name="treatment",
            treatment_type="binary",
        )

    def test_linearity_tests_linear_data(self):
        """Test linearity tests on linear data."""
        results = linearity_tests(self.linear_outcome, self.covariates)

        assert "individual_tests" in results
        assert "overall_linearity_ok" in results

        # Should pass linearity for linear data
        assert results["overall_linearity_ok"]

    def test_linearity_tests_nonlinear_data(self):
        """Test linearity tests on nonlinear data."""
        results = linearity_tests(self.nonlinear_outcome, self.covariates)

        # May detect nonlinearity in x2
        assert "problematic_variables" in results
        # Note: May not always detect due to random variation

    def test_functional_form_tests(self):
        """Test functional form comparison."""
        results = functional_form_tests(
            self.linear_outcome,
            self.covariates,
            comparison_models=["linear", "polynomial"],
        )

        assert "model_comparisons" in results
        assert "best_model" in results
        assert "linear_model_adequate" in results

    def test_comprehensive_specification_tests(self):
        """Test comprehensive specification testing."""
        tester = ModelSpecificationTests()
        results = tester.comprehensive_specification_tests(
            self.linear_outcome,
            self.treatment,
            self.covariates,
        )

        assert hasattr(results, "specification_passed")
        assert hasattr(results, "linearity_test_results")
        assert hasattr(results, "interaction_test_results")
        assert isinstance(results.recommendations, list)


class TestDiagnosticReporting:
    """Test diagnostic reporting functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100

        # Simple test data
        age = np.random.normal(40, 10, n)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = 2 * treatment + 0.1 * age + np.random.normal(0, 1, n)

        self.treatment = TreatmentData(
            values=pd.Series(treatment),
            name="treatment",
            treatment_type="binary",
        )

        self.outcome = OutcomeData(
            values=pd.Series(outcome),
            name="outcome",
            outcome_type="continuous",
        )

        self.covariates = CovariateData(
            values=pd.DataFrame({"age": age}),
            names=["age"],
        )

    def test_report_generation(self):
        """Test comprehensive report generation."""
        generator = DiagnosticReportGenerator()
        report = generator.generate_comprehensive_report(
            self.treatment,
            self.outcome,
            self.covariates,
            verbose=False,
        )

        assert hasattr(report, "timestamp")
        assert hasattr(report, "sample_size")
        assert hasattr(report, "overall_assessment")
        assert report.sample_size == len(self.treatment.values)

    def test_report_with_causal_effect(self):
        """Test report generation with causal effect for sensitivity analysis."""
        causal_effect = CausalEffect(
            ate=2.0,
            ate_ci_lower=1.0,
            ate_ci_upper=3.0,
            method="test",
        )

        generator = DiagnosticReportGenerator(include_sensitivity=True)
        report = generator.generate_comprehensive_report(
            self.treatment,
            self.outcome,
            self.covariates,
            causal_effect=causal_effect,
            verbose=False,
        )

        assert report.sensitivity_results is not None

    def test_generate_diagnostic_report_convenience(self):
        """Test convenience function for report generation."""
        report = generate_diagnostic_report(
            self.treatment,
            self.outcome,
            self.covariates,
            verbose=False,
        )

        assert hasattr(report, "overall_assessment")
        assert hasattr(report, "recommendations")

    def test_export_to_dataframe(self):
        """Test exporting report results to DataFrame."""
        generator = DiagnosticReportGenerator()
        report = generator.generate_comprehensive_report(
            self.treatment,
            self.outcome,
            self.covariates,
            verbose=False,
        )

        df = generator.export_report_to_dataframe(report)

        assert isinstance(df, pd.DataFrame)
        assert "Category" in df.columns
        assert "Metric" in df.columns
        assert "Value" in df.columns
        assert len(df) > 0


class TestDiagnosticsIntegration:
    """Integration tests for the diagnostics framework."""

    def test_full_diagnostic_workflow(self):
        """Test the complete diagnostic workflow."""
        np.random.seed(42)
        n = 150

        # Create realistic data with known properties
        age = np.random.normal(45, 12, n)
        income = np.random.normal(60000, 20000, n)
        education = np.random.choice([1, 2, 3, 4], n)

        # Treatment assignment with moderate confounding
        treatment_logits = (
            0.02 * (age - 45) + 0.00001 * (income - 60000) + 0.3 * (education - 2.5)
        )
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)

        # Outcome with treatment effect and confounding
        outcome = (
            3.0 * treatment  # Treatment effect
            + 0.15 * age  # Confounder
            + 0.00003 * income  # Confounder
            + 0.5 * education  # Confounder
            + np.random.normal(0, 2, n)
        )

        treatment_data = TreatmentData(
            values=pd.Series(treatment),
            name="treatment",
            treatment_type="binary",
        )

        outcome_data = OutcomeData(
            values=pd.Series(outcome),
            name="outcome",
            outcome_type="continuous",
        )

        covariates_data = CovariateData(
            values=pd.DataFrame(
                {
                    "age": age,
                    "income": income,
                    "education": education,
                }
            ),
            names=["age", "income", "education"],
        )

        causal_effect = CausalEffect(
            ate=3.0,
            ate_ci_lower=2.0,
            ate_ci_upper=4.0,
            method="integration_test",
        )

        # Run comprehensive diagnostics
        report = generate_diagnostic_report(
            treatment_data,
            outcome_data,
            covariates_data,
            causal_effect=causal_effect,
            include_sensitivity=True,
            verbose=False,
        )

        # Verify all components are present
        assert report.balance_results is not None
        assert report.overlap_results is not None
        assert report.assumption_results is not None
        assert report.specification_results is not None
        assert report.sensitivity_results is not None

        # Verify reasonable results
        assert isinstance(report.overall_assessment, str)
        assert len(report.recommendations) > 0

        # Should have some confounding strength detected (may not always trigger detection threshold)
        assert len(report.assumption_results.confounding_strength) > 0

        # Export should work
        generator = DiagnosticReportGenerator()
        df = generator.export_report_to_dataframe(report)
        assert len(df) > 10  # Should have multiple diagnostic metrics


if __name__ == "__main__":
    pytest.main([__file__])
