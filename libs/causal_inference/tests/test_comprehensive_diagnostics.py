"""Tests for comprehensive model diagnostics and validation suite."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.diagnostics import (
    calculate_calibration_metrics,
    calculate_distributional_balance,
    calculate_prognostic_score_balance,
    ramsey_reset_test,
    suggest_trimming_thresholds,
    validate_causal_assumptions,
)
from causal_inference.diagnostics.specification import (
    comprehensive_transformation_analysis,
    reset_test,
    suggest_box_cox_transformation,
)
from causal_inference.diagnostics.validation_suite import ComprehensiveValidationSuite


class TestEnhancedBalanceDiagnostics:
    """Test enhanced balance diagnostic features."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 1000

        # Create covariates
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(1, 2, n)
        X3 = np.random.binomial(1, 0.3, n)

        # Create treatment with some imbalance
        treatment_logits = -0.5 + 0.3 * X1 + 0.1 * X2
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)

        # Create outcome
        outcome = (
            2
            + 1.5 * treatment
            + 0.5 * X1
            + 0.3 * X2
            + 0.2 * X3
            + np.random.normal(0, 1, n)
        )

        self.treatment_data = TreatmentData(values=treatment)
        self.outcome_data = OutcomeData(values=outcome)
        self.covariate_data = CovariateData(
            values=pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
        )

    def test_prognostic_score_balance(self):
        """Test prognostic score balance calculation."""
        result = calculate_prognostic_score_balance(
            self.outcome_data, self.treatment_data, self.covariate_data
        )

        assert "prognostic_scores" in result
        assert "smd" in result
        assert "p_value" in result
        assert "balance_met" in result
        assert "model_r2" in result

        # Should have predictions for all observations
        assert len(result["prognostic_scores"]) == len(self.treatment_data.values)

        # SMD should be reasonable
        assert not np.isnan(result["smd"])
        assert result["model_r2"] > 0  # Model should have some predictive power

    def test_distributional_balance(self):
        """Test distributional balance using KS test."""
        # Test with balanced data
        balanced_covariate = np.random.normal(0, 1, 1000)
        balanced_treatment = np.random.binomial(1, 0.5, 1000)

        result = calculate_distributional_balance(
            balanced_covariate, balanced_treatment
        )

        assert "ks_statistic" in result
        assert "p_value" in result
        assert "distributions_differ" in result

        # For balanced data, should not detect differences
        assert result["p_value"] > 0.05  # Should not reject null of same distribution

        # Test with imbalanced data
        imbalanced_covariate = np.concatenate(
            [
                np.random.normal(0, 1, 500),  # Control group
                np.random.normal(1, 1, 500),  # Treated group (different mean)
            ]
        )
        imbalanced_treatment = np.concatenate(
            [
                np.zeros(500),
                np.ones(500),
            ]
        )

        result_imbalanced = calculate_distributional_balance(
            imbalanced_covariate, imbalanced_treatment
        )

        # Should detect significant difference
        assert result_imbalanced["p_value"] < 0.05
        assert result_imbalanced["distributions_differ"]

    def test_enhanced_balance_assessment(self):
        """Test enhanced balance assessment with new features."""
        from causal_inference.diagnostics.balance import BalanceDiagnostics

        diagnostics = BalanceDiagnostics()
        results = diagnostics.assess_balance(
            self.treatment_data, self.covariate_data, outcome=self.outcome_data
        )

        # Check new fields are present
        assert hasattr(results, "ks_test_results")
        assert hasattr(results, "prognostic_score_balance")

        # Check KS test results
        assert len(results.ks_test_results) == len(self.covariate_data.values.columns)
        for var_name, ks_result in results.ks_test_results.items():
            assert "ks_statistic" in ks_result
            assert "p_value" in ks_result
            assert "distributions_differ" in ks_result

        # Check prognostic score balance
        assert results.prognostic_score_balance is not None
        if "error" not in results.prognostic_score_balance:
            assert "smd" in results.prognostic_score_balance
            assert "balance_met" in results.prognostic_score_balance


class TestEnhancedOverlapDiagnostics:
    """Test enhanced overlap diagnostic features."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 1000

        # Create covariates
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)

        # Create treatment with moderate overlap issues
        logits = -2 + 2 * X1 + 1 * X2
        treatment_prob = 1 / (1 + np.exp(-logits))
        treatment = np.random.binomial(1, treatment_prob)

        self.treatment_data = TreatmentData(values=treatment)
        self.covariate_data = CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2}))

        # Create propensity scores for testing
        self.propensity_scores = treatment_prob

    def test_calibration_metrics(self):
        """Test propensity score calibration metrics."""
        result = calculate_calibration_metrics(
            self.treatment_data.values, self.propensity_scores
        )

        assert "fraction_of_positives" in result
        assert "mean_predicted_value" in result
        assert "brier_score" in result
        assert "calibration_slope" in result
        assert "mean_absolute_calibration_error" in result
        assert "well_calibrated" in result

        # Brier score should be reasonable
        assert 0 <= result["brier_score"] <= 1

        # MACE should be reasonable
        assert result["mean_absolute_calibration_error"] >= 0

    def test_trimming_suggestions(self):
        """Test propensity score trimming suggestions."""
        result = suggest_trimming_thresholds(
            self.propensity_scores, self.treatment_data.values
        )

        # Should have suggestions for different percentiles
        expected_keys = ["trim_1.0pct", "trim_2.5pct", "trim_5.0pct", "trim_10.0pct"]
        for key in expected_keys:
            assert key in result

        # Each suggestion should have required fields
        for trim_level, details in result.items():
            assert "lower_bound" in details
            assert "upper_bound" in details
            assert "n_trimmed" in details
            assert "pct_trimmed" in details
            assert "n_treated_remaining" in details
            assert "n_control_remaining" in details

            # Bounds should make sense
            assert details["lower_bound"] <= details["upper_bound"]
            assert 0 <= details["pct_trimmed"] <= 100

    def test_enhanced_overlap_assessment(self):
        """Test enhanced overlap assessment with new features."""
        from causal_inference.diagnostics.overlap import OverlapDiagnostics

        diagnostics = OverlapDiagnostics()
        results = diagnostics.assess_overlap(self.treatment_data, self.covariate_data)

        # Check new fields are present
        assert hasattr(results, "calibration_results")
        assert hasattr(results, "trimming_recommendations")
        assert hasattr(results, "roc_curve_data")

        # Check calibration results
        if results.calibration_results and "error" not in results.calibration_results:
            assert "brier_score" in results.calibration_results
            assert "well_calibrated" in results.calibration_results

        # Check trimming recommendations
        assert results.trimming_recommendations is not None
        assert len(results.trimming_recommendations) > 0

        # Check ROC curve data
        if results.roc_curve_data:
            assert "fpr" in results.roc_curve_data
            assert "tpr" in results.roc_curve_data
            assert "auc" in results.roc_curve_data


class TestEnhancedSpecificationTests:
    """Test enhanced specification test features."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 500

        # Create data with known specification issues
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        treatment = np.random.binomial(1, 0.5, n)

        # Create outcome with non-linear relationship
        outcome = (
            1
            + 2 * treatment
            + 0.5 * X1
            + 0.3 * X1**2
            + 0.2 * X2
            + np.random.normal(0, 1, n)
        )

        self.treatment_data = TreatmentData(values=treatment)
        self.outcome_data = OutcomeData(values=outcome)
        self.covariate_data = CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2}))

        # Create some positive data for Box-Cox testing
        self.positive_data = np.random.lognormal(0, 1, n)

    def test_reset_test(self):
        """Test RESET test for functional form."""
        # Create simple linear model residuals and fitted values
        from sklearn.linear_model import LinearRegression

        X = self.covariate_data.values
        y = self.outcome_data.values

        model = LinearRegression()
        model.fit(X, y)
        fitted = model.predict(X)
        residuals = y - fitted

        result = reset_test(residuals, fitted)

        assert "f_statistic" in result
        assert "p_value" in result
        assert "specification_adequate" in result
        assert "powers_tested" in result
        assert "recommendation" in result

        # Should detect non-linearity in our test data
        assert result["p_value"] < 0.1  # Likely to reject linear form

    def test_box_cox_transformation(self):
        """Test Box-Cox transformation suggestions."""
        result = suggest_box_cox_transformation(self.positive_data, "test_var")

        assert "variable" in result
        assert "lambda_optimal" in result
        assert "transformation_type" in result
        assert "transformation_needed" in result
        assert "recommendation" in result

        # Lambda should be reasonable
        if not np.isnan(result["lambda_optimal"]):
            assert -3 <= result["lambda_optimal"] <= 3

    def test_transformation_analysis(self):
        """Test comprehensive transformation analysis."""
        # Create positive outcome for testing
        positive_outcome = OutcomeData(values=self.positive_data)

        result = comprehensive_transformation_analysis(
            positive_outcome, self.covariate_data
        )

        assert "outcome" in result
        assert isinstance(result, dict)

        # Should have analysis for outcome
        outcome_analysis = result["outcome"]
        assert "recommendation" in outcome_analysis

    def test_ramsey_reset_convenience_function(self):
        """Test RESET test convenience function."""
        result = ramsey_reset_test(
            self.outcome_data, self.covariate_data, verbose=False
        )

        assert "f_statistic" in result
        assert "p_value" in result
        assert "specification_adequate" in result

    def test_enhanced_specification_tests(self):
        """Test enhanced specification tests with new features."""
        from causal_inference.diagnostics.specification import ModelSpecificationTests

        tester = ModelSpecificationTests()
        results = tester.comprehensive_specification_tests(
            self.outcome_data, self.treatment_data, self.covariate_data
        )

        # Check new fields are present
        assert hasattr(results, "reset_test_results")
        assert hasattr(results, "transformation_suggestions")

        # Check RESET test results
        if results.reset_test_results:
            assert "f_statistic" in results.reset_test_results
            assert "specification_adequate" in results.reset_test_results

        # Check transformation suggestions
        if results.transformation_suggestions:
            assert isinstance(results.transformation_suggestions, dict)


class TestComprehensiveValidationSuite:
    """Test the comprehensive validation suite."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 1000

        # Create balanced, well-specified data for positive test
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X3 = np.random.binomial(1, 0.5, n)

        # Balanced treatment assignment
        treatment = np.random.binomial(1, 0.5, n)

        # Well-specified outcome
        outcome = (
            1
            + 1.5 * treatment
            + 0.5 * X1
            + 0.3 * X2
            + 0.2 * X3
            + np.random.normal(0, 1, n)
        )

        self.good_treatment = TreatmentData(values=treatment)
        self.good_outcome = OutcomeData(values=outcome)
        self.good_covariates = CovariateData(
            values=pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
        )

        # Create problematic data
        # Highly imbalanced treatment
        logits = -3 + 4 * X1  # Strong dependence creates overlap issues
        bad_treatment_prob = 1 / (1 + np.exp(-logits))
        bad_treatment = np.random.binomial(1, bad_treatment_prob)

        # Non-linear outcome
        bad_outcome = 1 + 2 * bad_treatment + 0.5 * X1**2 + np.random.normal(0, 1, n)

        self.bad_treatment = TreatmentData(values=bad_treatment)
        self.bad_outcome = OutcomeData(values=bad_outcome)
        self.bad_covariates = CovariateData(
            values=pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
        )

    def test_validation_suite_initialization(self):
        """Test validation suite initialization."""
        suite = ComprehensiveValidationSuite()

        assert hasattr(suite, "balance_diagnostics")
        assert hasattr(suite, "overlap_diagnostics")
        assert hasattr(suite, "specification_tests")

    def test_good_data_validation(self):
        """Test validation on well-behaved data."""
        results = validate_causal_assumptions(
            self.good_treatment,
            self.good_outcome,
            self.good_covariates,
            verbose=False,
        )

        # Should have all components
        assert hasattr(results, "balance_assessment")
        assert hasattr(results, "overlap_assessment")
        assert hasattr(results, "specification_assessment")
        assert hasattr(results, "overall_validation_passed")
        assert hasattr(results, "validation_score")

        # Should generally pass with good data
        assert results.validation_score > 50  # At least moderate score

        # Should have reasonable recommendations
        assert isinstance(results.recommendations, list)
        assert len(results.recommendations) > 0

    def test_problematic_data_validation(self):
        """Test validation on problematic data."""
        results = validate_causal_assumptions(
            self.bad_treatment,
            self.bad_outcome,
            self.bad_covariates,
            verbose=False,
        )

        # Should detect issues
        assert len(results.critical_issues) > 0 or len(results.warnings) > 0

        # Should have lower validation score
        assert results.validation_score < 90  # Should detect some problems

        # Should provide actionable recommendations
        assert len(results.recommendations) > 0

    def test_validation_score_calculation(self):
        """Test validation score calculation."""
        suite = ComprehensiveValidationSuite()

        # Test with good data
        good_results = suite.validate_assumptions(
            self.good_treatment, self.good_outcome, self.good_covariates
        )

        # Test with bad data
        bad_results = suite.validate_assumptions(
            self.bad_treatment, self.bad_outcome, self.bad_covariates
        )

        # Good data should have higher score
        assert good_results.validation_score >= bad_results.validation_score

        # Scores should be in valid range
        assert 0 <= good_results.validation_score <= 100
        assert 0 <= bad_results.validation_score <= 100

    def test_diagnostic_report_generation(self):
        """Test diagnostic report generation."""
        suite = ComprehensiveValidationSuite()
        results = suite.validate_assumptions(
            self.good_treatment, self.good_outcome, self.good_covariates
        )

        report = suite.generate_diagnostic_report(results)

        # Should be a string with expected sections
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial

        # Should contain key sections
        assert "Executive Summary" in report
        assert "Detailed Assessment" in report
        assert "Recommendations" in report
        assert "Covariate Balance" in report
        assert "Overlap and Positivity" in report
        assert "Model Specification" in report

    def test_validation_summary_printing(self):
        """Test validation summary printing (no exceptions)."""
        suite = ComprehensiveValidationSuite()
        results = suite.validate_assumptions(
            self.good_treatment, self.good_outcome, self.good_covariates
        )

        # Should not raise exception
        suite.print_validation_summary(results)

    def test_convenience_function(self):
        """Test convenience function for validation."""
        results = validate_causal_assumptions(
            self.good_treatment,
            self.good_outcome,
            self.good_covariates,
            verbose=False,
            generate_report=False,
        )

        assert hasattr(results, "validation_score")
        assert hasattr(results, "overall_validation_passed")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Very small sample size
        small_n = 50
        small_treatment = TreatmentData(values=np.random.binomial(1, 0.5, small_n))
        small_outcome = OutcomeData(values=np.random.normal(0, 1, small_n))
        small_covariates = CovariateData(
            values=pd.DataFrame({"X1": np.random.normal(0, 1, small_n)})
        )

        # Should handle small sample gracefully
        results = validate_causal_assumptions(
            small_treatment, small_outcome, small_covariates, verbose=False
        )

        assert hasattr(results, "validation_score")

        # All treated or all control
        all_treated = TreatmentData(values=np.ones(100, dtype=int))
        normal_outcome = OutcomeData(values=np.random.normal(0, 1, 100))
        normal_covariates = CovariateData(
            values=pd.DataFrame({"X1": np.random.normal(0, 1, 100)})
        )

        # Should handle gracefully (though will detect issues)
        results = validate_causal_assumptions(
            all_treated, normal_outcome, normal_covariates, verbose=False
        )

        # Should detect critical issues
        assert len(results.critical_issues) > 0


class TestIntegration:
    """Integration tests for the enhanced diagnostics."""

    def test_full_workflow(self):
        """Test complete diagnostic workflow."""
        np.random.seed(42)
        n = 800

        # Create realistic data
        age = np.random.normal(45, 15, n)
        income = np.random.lognormal(10, 0.5, n)
        education = np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.3, 0.3, 0.2])

        # Treatment depends on covariates (selection bias)
        logits = -2 + 0.02 * age + 0.0001 * income + 0.5 * education
        treatment_prob = 1 / (1 + np.exp(-logits))
        treatment = np.random.binomial(1, treatment_prob)

        # Outcome depends on treatment and covariates
        outcome = (
            5000
            + 2000 * treatment
            + 50 * age
            + 0.1 * income
            + 1000 * education
            + np.random.normal(0, 2000, n)
        )

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(
            values=pd.DataFrame({"age": age, "income": income, "education": education})
        )

        # Run comprehensive validation
        results = validate_causal_assumptions(
            treatment_data,
            outcome_data,
            covariate_data,
            verbose=False,
            generate_report=True,
        )

        # Should detect selection bias issues
        assert (
            not results.overlap_assessment.overall_positivity_met
            or results.validation_score < 85
        )

        # Should provide meaningful recommendations
        assert len(results.recommendations) > 2

        # Report should be comprehensive
        suite = ComprehensiveValidationSuite()
        report = suite.generate_diagnostic_report(results)
        assert len(report) > 500  # Substantial report


if __name__ == "__main__":
    pytest.main([__file__])
