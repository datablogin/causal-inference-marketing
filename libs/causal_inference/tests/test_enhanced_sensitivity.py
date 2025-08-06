"""Tests for enhanced sensitivity analysis functionality."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import (
    CausalEffect,
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.diagnostics.sensitivity import (
    LinearSensitivityModel,
    SensitivityAnalysis,
    TippingPointAnalysis,
    evalue_calculation,
    rosenbaum_bounds,
)


class TestEnhancedEvalueCalculation:
    """Test enhanced E-value calculation."""

    def test_evalue_basic_functionality(self):
        """Test basic E-value calculation."""
        result = evalue_calculation(2.0, ci_lower=1.5, ci_upper=3.0)

        assert "evalue_point" in result
        assert "evalue_ci" in result
        assert "evalue_ci_lower" in result
        assert "evalue_ci_upper" in result
        assert result["evalue_point"] > 1
        assert result["risk_ratio_used"] == 2.0

    def test_evalue_protective_effect(self):
        """Test E-value for protective effects."""
        result = evalue_calculation(0.6, ci_lower=0.4, ci_upper=0.8)

        assert result["evalue_point"] > 1
        assert result["risk_ratio_used"] == 0.6

    def test_evalue_different_effect_types(self):
        """Test E-value with different effect types."""
        # Risk ratio
        rr_result = evalue_calculation(2.0, effect_type="risk_ratio")
        assert rr_result["effect_type"] == "risk_ratio"

        # Odds ratio with rare outcome
        or_result = evalue_calculation(2.5, effect_type="odds_ratio", rare_outcome=True)
        assert or_result["effect_type"] == "odds_ratio"

        # Hazard ratio
        hr_result = evalue_calculation(1.8, effect_type="hazard_ratio")
        assert hr_result["effect_type"] == "hazard_ratio"

    def test_evalue_threshold_interpretation(self):
        """Test E-value threshold interpretations."""
        # High E-value
        high_result = evalue_calculation(3.0)
        assert "Above common threshold" in high_result["threshold_interpretation"]

        # Moderate E-value
        mod_result = evalue_calculation(1.8)
        assert "Above common threshold" in mod_result["threshold_interpretation"]

        # Low E-value - use value close to 1.0 to get E-value below 1.25
        low_result = evalue_calculation(1.01)  # This gives E-value around 1.15
        assert "Below minimal threshold" in low_result["threshold_interpretation"]

    def test_evalue_error_handling(self):
        """Test E-value error handling."""
        # Negative effect estimate
        with pytest.raises(ValueError, match="Effect estimate must be positive"):
            evalue_calculation(-0.5)

        # Invalid confidence interval
        with pytest.raises(ValueError, match="Lower CI bound cannot exceed upper"):
            evalue_calculation(2.0, ci_lower=3.0, ci_upper=2.0)


class TestEnhancedRosenbaumBounds:
    """Test enhanced Rosenbaum bounds."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)

        # Create paired data with moderate treatment effect
        n_pairs = 30
        treated = np.random.normal(1.0, 1.0, n_pairs)
        control = np.random.normal(0.0, 1.0, n_pairs)

        self.treated_outcomes = treated
        self.control_outcomes = control

    def test_enhanced_rosenbaum_bounds(self):
        """Test enhanced Rosenbaum bounds calculation."""
        result = rosenbaum_bounds(
            self.treated_outcomes,
            self.control_outcomes,
            gamma_range=(1.0, 3.0),
            gamma_steps=25,
            alpha=0.05,
        )

        assert "original_p_value" in result
        assert "original_statistic" in result
        assert "method_used" in result
        assert "n_pairs" in result
        assert "n_non_zero_differences" in result
        assert "robustness_assessment" in result

        # Check bounds structure
        assert isinstance(result["bounds"], list)
        assert len(result["bounds"]) == 25

        # Check individual bound
        bound = result["bounds"][0]
        assert "gamma" in bound
        assert "p_value_lower" in bound
        assert "p_value_upper" in bound

    def test_rosenbaum_bounds_with_zeros(self):
        """Test Rosenbaum bounds with zero differences."""
        # Create data with some zero differences
        treated = np.array([1, 2, 3, 4, 5, 1, 2, 3])
        control = np.array([1, 1, 2, 3, 4, 1, 2, 3])  # Some zeros

        result = rosenbaum_bounds(treated, control)

        assert result["n_non_zero_differences"] <= result["n_pairs"]
        assert result["n_non_zero_differences"] >= 0

    def test_rosenbaum_bounds_sign_test_fallback(self):
        """Test fallback to sign test when Wilcoxon fails."""
        # Create data that might cause Wilcoxon to fail
        treated = np.array([1, 1, 1, 2, 2])
        control = np.array([0, 0, 0, 1, 1])

        result = rosenbaum_bounds(treated, control, method="sign_test")

        assert result["method_used"] == "sign_test"
        assert "bounds" in result

    def test_rosenbaum_bounds_error_handling(self):
        """Test Rosenbaum bounds error handling."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            rosenbaum_bounds(np.array([1, 2, 3]), np.array([1, 2]))

        # Insufficient data
        with pytest.raises(ValueError, match="Minimum of 5 pairs"):
            rosenbaum_bounds(np.array([1, 2]), np.array([1, 2]))


class TestLinearSensitivityModel:
    """Test linear sensitivity model."""

    def setup_method(self):
        """Set up linear sensitivity model."""
        self.model = LinearSensitivityModel()
        self.observed_ate = 2.0

    def test_linear_sensitivity_analysis(self):
        """Test linear sensitivity analysis."""
        result = self.model.analyze_sensitivity(
            self.observed_ate,
            confounder_treatment_assoc=0.5,
            confounder_outcome_assoc=0.4,
        )

        assert "observed_ate" in result
        assert "confounder_bias" in result
        assert "adjusted_ate" in result
        assert "percent_change" in result
        assert "sign_change" in result
        assert "nullified" in result

        expected_bias = 0.5 * 0.4
        assert abs(result["confounder_bias"] - expected_bias) < 1e-6

    def test_sensitivity_surface(self):
        """Test sensitivity surface generation."""
        result = self.model.sensitivity_surface(
            self.observed_ate,
            confounder_treatment_range=(-0.8, 0.8),
            confounder_outcome_range=(-0.6, 0.6),
            n_points=10,
        )

        assert "treatment_associations" in result
        assert "outcome_associations" in result
        assert "bias_surface" in result
        assert "adjusted_ate_surface" in result
        assert "nullification_boundary" in result

        # Check dimensions
        assert len(result["treatment_associations"]) == 10
        assert len(result["outcome_associations"]) == 10
        assert result["bias_surface"].shape == (10, 10)

    def test_nullification_detection(self):
        """Test nullification detection."""
        # Strong confounding that should nullify effect
        result = self.model.analyze_sensitivity(
            self.observed_ate,
            confounder_treatment_assoc=2.0,
            confounder_outcome_assoc=2.0,
        )

        # Bias should be large
        assert abs(result["confounder_bias"]) > 1.0
        assert result["percent_change"] > 50


class TestTippingPointAnalysis:
    """Test tipping point analysis."""

    def setup_method(self):
        """Set up tipping point analysis."""
        self.analysis = TippingPointAnalysis(significance_level=0.05)

    def test_evalue_tipping_point(self):
        """Test E-value based tipping point analysis."""
        result = self.analysis.find_tipping_point_evalue(
            observed_estimate=2.0,
            ci_lower=1.2,
            ci_upper=3.0,
        )

        assert "tipping_gamma" in result
        assert "tipping_evalue" in result
        assert "interpretation" in result
        assert "robustness_assessment" in result

        # Tipping gamma should be reasonable
        assert 1.0 <= result["tipping_gamma"] <= 10.0

    def test_linear_tipping_point(self):
        """Test linear model tipping point."""
        result = self.analysis.find_tipping_point_linear(
            observed_ate=2.0,
            se_ate=0.5,
        )

        assert "tipping_strength" in result
        assert "critical_bias" in result
        assert "interpretation" in result
        assert "robustness_assessment" in result

    def test_tipping_point_robustness_assessment(self):
        """Test robustness assessment in tipping point."""
        # High robustness scenario
        high_robust = self.analysis.find_tipping_point_linear(
            observed_ate=3.0,
            se_ate=0.3,  # Small SE, strong effect
        )

        # Low robustness scenario
        low_robust = self.analysis.find_tipping_point_linear(
            observed_ate=0.5,
            se_ate=0.4,  # Large SE, weak effect
        )

        # High robust should have higher tipping strength
        assert high_robust["tipping_strength"] > low_robust["tipping_strength"]


class TestComprehensiveSensitivityAnalysis:
    """Test comprehensive sensitivity analysis."""

    def setup_method(self):
        """Set up test data for comprehensive analysis."""
        np.random.seed(42)
        n = 100

        # Create realistic treatment and outcome data
        treatment = np.random.binomial(1, 0.4, n)
        outcome = 2.0 * treatment + np.random.normal(0, 1.5, n)

        self.treatment_data = TreatmentData(
            values=pd.Series(treatment),
            treatment_type="binary",
        )

        self.outcome_data = OutcomeData(
            values=pd.Series(outcome),
            outcome_type="continuous",
        )

        self.causal_effect = CausalEffect(
            ate=2.0,
            ate_se=0.3,
            ate_ci_lower=1.4,
            ate_ci_upper=2.6,
            method="test",
        )

        self.analyzer = SensitivityAnalysis(
            alpha=0.05,
            evalue_threshold=2.0,
            verbose=False,
            random_state=42,
        )

    def test_comprehensive_analysis(self):
        """Test comprehensive sensitivity analysis."""
        result = self.analyzer.comprehensive_sensitivity_analysis(
            self.treatment_data,
            self.outcome_data,
            self.causal_effect,
        )

        # Check all expected attributes
        assert hasattr(result, "evalue")
        assert hasattr(result, "evalue_ci_lower")
        assert hasattr(result, "robustness_assessment")
        assert hasattr(result, "recommendations")

        # Check that E-value is reasonable
        assert result.evalue > 1.0

        # Check that recommendations are generated
        assert len(result.recommendations) > 0

    def test_comprehensive_analysis_with_all_features(self):
        """Test comprehensive analysis with all features enabled."""
        result = self.analyzer.comprehensive_sensitivity_analysis(
            self.treatment_data,
            self.outcome_data,
            self.causal_effect,
            include_tipping_point=True,
            include_linear_sensitivity=True,
        )

        # Check sensitivity plots data contains all components
        plots_data = result.sensitivity_plots_data
        assert "confounding_analysis" in plots_data
        assert "linear_sensitivity" in plots_data
        assert "tipping_point" in plots_data
        assert "evalue_details" in plots_data

    def test_robustness_assessment_levels(self):
        """Test different robustness assessment levels."""
        # High robustness case
        high_robust_effect = CausalEffect(
            ate=3.0,
            ate_se=0.2,
            ate_ci_lower=2.6,
            ate_ci_upper=3.4,
            method="test",
        )

        result_high = self.analyzer.comprehensive_sensitivity_analysis(
            self.treatment_data,
            self.outcome_data,
            high_robust_effect,
        )

        assert "High robustness" in result_high.robustness_assessment

        # Low robustness case
        low_robust_effect = CausalEffect(
            ate=0.2,
            ate_se=0.15,
            ate_ci_lower=0.05,
            ate_ci_upper=0.35,
            method="test",
        )

        result_low = self.analyzer.comprehensive_sensitivity_analysis(
            self.treatment_data,
            self.outcome_data,
            low_robust_effect,
        )

        # The robustness assessment should be lower than high robustness
        # (Could be "Moderate" due to averaging across multiple metrics)
        assert result_low.robustness_assessment != result_high.robustness_assessment
        assert any(
            word in result_low.robustness_assessment.lower()
            for word in ["moderate", "limited", "low"]
        )

    def test_binary_outcome_conversion(self):
        """Test ATE to risk ratio conversion for binary outcomes."""
        # Binary outcome data
        binary_outcome = OutcomeData(
            values=pd.Series(np.random.binomial(1, 0.3, 100)),
            outcome_type="binary",
        )

        result = self.analyzer.comprehensive_sensitivity_analysis(
            self.treatment_data,
            binary_outcome,
            self.causal_effect,
        )

        # Should successfully convert and analyze
        assert result.evalue > 1.0

    def test_recommendation_generation(self):
        """Test recommendation generation quality."""
        result = self.analyzer.comprehensive_sensitivity_analysis(
            self.treatment_data,
            self.outcome_data,
            self.causal_effect,
        )

        recommendations = result.recommendations

        # Should have substantive recommendations
        assert len(recommendations) > 3

        # Should contain technical section
        tech_recs = [r for r in recommendations if "Technical recommendations" in r]
        assert len(tech_recs) > 0

        # Check for appropriate symbols
        symbols_found = any(
            symbol in "".join(recommendations)
            for symbol in ["✅", "⚠️", "❌", "ℹ️", "📊", "🎯", "📋"]
        )
        assert symbols_found

    def test_error_handling_comprehensive(self):
        """Test error handling in comprehensive analysis."""
        # Test with insufficient data for Rosenbaum bounds
        small_treatment = TreatmentData(
            values=pd.Series([0, 1, 0]),
            treatment_type="binary",
        )

        small_outcome = OutcomeData(
            values=pd.Series([1.0, 2.0, 1.5]),
            outcome_type="continuous",
        )

        # Should handle gracefully without crashing
        result = self.analyzer.comprehensive_sensitivity_analysis(
            small_treatment,
            small_outcome,
            self.causal_effect,
        )

        # Should still provide E-value analysis
        assert result.evalue > 0


class TestSensitivityIntegration:
    """Integration tests for sensitivity analysis."""

    def test_sensitivity_with_existing_diagnostics(self):
        """Test integration with existing diagnostic framework."""
        np.random.seed(42)
        n = 80

        treatment = np.random.binomial(1, 0.5, n)
        outcome = 1.5 * treatment + np.random.normal(0, 1, n)
        covariates = np.random.normal(0, 1, (n, 2))

        treatment_data = TreatmentData(
            values=pd.Series(treatment),
            treatment_type="binary",
        )

        outcome_data = OutcomeData(
            values=pd.Series(outcome),
            outcome_type="continuous",
        )

        covariate_data = CovariateData(
            values=pd.DataFrame(covariates, columns=["X1", "X2"]),
            names=["X1", "X2"],
        )

        causal_effect = CausalEffect(
            ate=1.5,
            ate_ci_lower=1.0,
            ate_ci_upper=2.0,
            method="integration_test",
        )

        # Test that sensitivity analysis works with covariates
        analyzer = SensitivityAnalysis(random_state=42)
        result = analyzer.comprehensive_sensitivity_analysis(
            treatment_data,
            outcome_data,
            causal_effect,
            covariates=covariate_data,
        )

        assert result.evalue > 1.0
        assert len(result.recommendations) > 0

    def test_convenience_functions(self):
        """Test convenience functions still work."""
        from causal_inference.diagnostics.sensitivity import (
            assess_sensitivity,
            calculate_evalue,
        )

        # Test calculate_evalue convenience function
        evalue_result = calculate_evalue(2.0, ci_lower=1.5, verbose=False)
        assert "evalue_point" in evalue_result

        # Test assess_sensitivity convenience function
        treatment_data = TreatmentData(
            values=pd.Series([0, 1, 0, 1] * 20),
            treatment_type="binary",
        )

        outcome_data = OutcomeData(
            values=pd.Series(np.random.normal(0, 1, 80)),
            outcome_type="continuous",
        )

        causal_effect = CausalEffect(
            ate=1.0,
            ate_ci_lower=0.5,
            ate_ci_upper=1.5,
            method="convenience_test",
        )

        sensitivity_result = assess_sensitivity(
            treatment_data,
            outcome_data,
            causal_effect,
            verbose=False,
        )

        assert hasattr(sensitivity_result, "evalue")
        assert hasattr(sensitivity_result, "recommendations")


if __name__ == "__main__":
    pytest.main([__file__])
