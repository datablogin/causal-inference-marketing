"""Integration tests for the visualization module with real estimators.

This module tests the visualization components working with actual causal
inference estimators to ensure end-to-end functionality.
"""

from unittest.mock import patch

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator
from causal_inference.visualization import (
    create_love_plot,
    create_propensity_plots,
    create_residual_plots,
    create_weight_plots,
    generate_diagnostic_report,
)


class TestEstimatorIntegration:
    """Test visualization integration with causal inference estimators."""

    @pytest.fixture
    def marketing_data(self):
        """Create realistic marketing campaign data."""
        np.random.seed(42)
        n = 500

        # Customer demographics (confounders)
        age = np.random.normal(35, 12, n)
        income = np.random.lognormal(np.log(50000), 0.4, n)
        previous_purchases = np.random.poisson(3, n)
        segment = np.random.choice(
            ["premium", "regular", "budget"], n, p=[0.2, 0.5, 0.3]
        )

        # Create segment dummy variables
        segment_premium = (segment == "premium").astype(int)
        segment_regular = (segment == "regular").astype(int)

        # Treatment assignment (email campaign) with confounding
        propensity_logit = (
            -2
            + 0.02 * age
            + 0.00001 * income
            + 0.3 * previous_purchases
            + 0.5 * segment_premium
            + 0.2 * segment_regular
        )
        propensity = 1 / (1 + np.exp(-propensity_logit))
        treatment = np.random.binomial(1, propensity, n)

        # Outcome: revenue with treatment effect
        true_ate = 50  # $50 treatment effect
        revenue = (
            100
            + 0.5 * age
            + 0.001 * income
            + 10 * previous_purchases
            + 30 * segment_premium
            + 15 * segment_regular
            + true_ate * treatment
            + np.random.normal(0, 25, n)
        )

        # Package into data objects
        treatment_data = TreatmentData(
            values=treatment, treatment_type="binary", name="email_campaign"
        )

        outcome_data = OutcomeData(
            values=revenue, outcome_type="continuous", name="revenue"
        )

        covariate_data = CovariateData(
            values=np.column_stack(
                [age, income, previous_purchases, segment_premium, segment_regular]
            ),
            names=[
                "age",
                "income",
                "previous_purchases",
                "segment_premium",
                "segment_regular",
            ],
        )

        return {
            "treatment_data": treatment_data,
            "outcome_data": outcome_data,
            "covariate_data": covariate_data,
            "true_ate": true_ate,
            "true_propensity": propensity,
        }

    @patch("matplotlib.pyplot.show")
    def test_g_computation_integration(self, mock_show, marketing_data):
        """Test visualization with G-computation estimator."""
        # Fit G-computation estimator
        estimator = GComputationEstimator(bootstrap_samples=0, verbose=False)
        estimator.fit(
            marketing_data["treatment_data"],
            marketing_data["outcome_data"],
            marketing_data["covariate_data"],
        )

        effect = estimator.estimate_ate()

        # Extract model components for visualization
        fitted_values = estimator.outcome_model.predict(
            np.column_stack(
                [
                    marketing_data["treatment_data"].values,
                    marketing_data["covariate_data"].values,
                ]
            )
        )
        residuals = marketing_data["outcome_data"].values - fitted_values

        # Test Love plot (before adjustment)
        love_fig = create_love_plot(
            marketing_data["covariate_data"],
            marketing_data["treatment_data"],
            title="G-computation Covariate Balance",
            interactive=False,
        )
        assert love_fig is not None

        # Test residual analysis
        design_matrix = np.column_stack(
            [
                np.ones(len(fitted_values)),  # intercept
                marketing_data["treatment_data"].values,
                marketing_data["covariate_data"].values,
            ]
        )

        resid_fig, resid_result, resid_recs = create_residual_plots(
            residuals,
            fitted_values,
            design_matrix,
            title="G-computation Residual Analysis",
            interactive=False,
        )
        assert resid_fig is not None
        assert len(resid_recs) > 0

        # Test comprehensive report
        html_report = generate_diagnostic_report(
            treatment_data=marketing_data["treatment_data"],
            outcome_data=marketing_data["outcome_data"],
            covariates=marketing_data["covariate_data"],
            residuals=residuals,
            fitted_values=fitted_values,
            estimator_name="G-computation",
            ate_estimate=effect.ate,
            ate_ci_lower=getattr(effect, "ate_ci_lower", None),
            ate_ci_upper=getattr(effect, "ate_ci_upper", None),
        )

        assert "G-computation" in html_report
        assert "email_campaign" in html_report
        assert "revenue" in html_report
        assert len(html_report) > 5000

    @patch("matplotlib.pyplot.show")
    def test_ipw_integration(self, mock_show, marketing_data):
        """Test visualization with IPW estimator."""
        # Fit IPW estimator
        estimator = IPWEstimator(bootstrap_samples=0, verbose=False)
        estimator.fit(
            marketing_data["treatment_data"],
            marketing_data["outcome_data"],
            marketing_data["covariate_data"],
        )

        effect = estimator.estimate_ate()

        # Get propensity scores and weights
        propensity_scores = estimator.get_propensity_scores()
        weights = estimator.get_weights()

        # Test Love plot with IPW weights
        love_fig = create_love_plot(
            marketing_data["covariate_data"],
            marketing_data["treatment_data"],
            weights_after=weights,
            title="IPW Weighted Covariate Balance",
            interactive=False,
        )
        assert love_fig is not None

        # Test weight diagnostics
        weight_fig, weight_result, weight_recs = create_weight_plots(
            weights, title="IPW Weight Distribution", interactive=False
        )
        assert weight_fig is not None
        assert weight_result.n_observations == len(weights)
        assert len(weight_recs) > 0

        # Test propensity score analysis
        prop_fig, prop_result, prop_recs = create_propensity_plots(
            propensity_scores,
            marketing_data["treatment_data"],
            title="IPW Propensity Score Analysis",
            interactive=False,
        )
        assert prop_fig is not None
        assert 0 <= prop_result.overlap_percentage <= 100
        assert len(prop_recs) > 0

        # Test comprehensive report
        html_report = generate_diagnostic_report(
            treatment_data=marketing_data["treatment_data"],
            outcome_data=marketing_data["outcome_data"],
            covariates=marketing_data["covariate_data"],
            weights=weights,
            propensity_scores=propensity_scores,
            estimator_name="Inverse Probability Weighting",
            ate_estimate=effect.ate,
            ate_ci_lower=getattr(effect, "ate_ci_lower", None),
            ate_ci_upper=getattr(effect, "ate_ci_upper", None),
        )

        assert "Inverse Probability Weighting" in html_report
        assert "Weight Distribution" in html_report or "weight" in html_report.lower()
        assert "Propensity Score" in html_report or "propensity" in html_report.lower()

    @patch("matplotlib.pyplot.show")
    def test_aipw_integration(self, mock_show, marketing_data):
        """Test visualization with AIPW (doubly robust) estimator."""
        # Fit AIPW estimator
        estimator = AIPWEstimator(
            cross_fitting=False,  # Disable for testing
            bootstrap_samples=0,
            verbose=False,
        )
        estimator.fit(
            marketing_data["treatment_data"],
            marketing_data["outcome_data"],
            marketing_data["covariate_data"],
        )

        effect = estimator.estimate_ate()

        # Get components from AIPW
        propensity_scores = estimator.propensity_estimator.get_propensity_scores()
        weights = estimator.propensity_estimator.get_weights()

        # Get outcome model residuals
        if hasattr(estimator, "outcome_estimator"):
            fitted_values = estimator.outcome_estimator.outcome_model.predict(
                np.column_stack(
                    [
                        marketing_data["treatment_data"].values,
                        marketing_data["covariate_data"].values,
                    ]
                )
            )
            residuals = marketing_data["outcome_data"].values - fitted_values
        else:
            # Fallback if outcome estimator not accessible
            fitted_values = np.mean(marketing_data["outcome_data"].values)
            residuals = marketing_data["outcome_data"].values - fitted_values

        # Test comprehensive AIPW report with all components
        html_report = generate_diagnostic_report(
            treatment_data=marketing_data["treatment_data"],
            outcome_data=marketing_data["outcome_data"],
            covariates=marketing_data["covariate_data"],
            weights=weights,
            propensity_scores=propensity_scores,
            residuals=residuals,
            fitted_values=fitted_values,
            estimator_name="Augmented Inverse Probability Weighting (AIPW)",
            ate_estimate=effect.ate,
            ate_ci_lower=getattr(effect, "ate_ci_lower", None),
            ate_ci_upper=getattr(effect, "ate_ci_upper", None),
        )

        # Check that all diagnostic components are present
        assert "AIPW" in html_report
        assert len(html_report) > 8000  # Should be comprehensive

        # Should contain all major diagnostic sections
        html_lower = html_report.lower()
        assert any(term in html_lower for term in ["balance", "covariate"])
        assert any(term in html_lower for term in ["weight", "distribution"])
        assert any(term in html_lower for term in ["propensity", "overlap"])
        assert any(term in html_lower for term in ["residual", "model"])


class TestReportQuality:
    """Test the quality and completeness of generated reports."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a realistic A/B test scenario."""
        np.random.seed(123)
        n = 1000

        # User characteristics
        user_age = np.random.normal(30, 10, n)
        user_engagement = np.random.exponential(2, n)
        is_premium = np.random.binomial(1, 0.3, n)

        # Treatment assignment (feature rollout)
        treatment_prob = (
            0.3 + 0.1 * is_premium
        )  # Premium users more likely to get feature
        treatment = np.random.binomial(1, treatment_prob, n)

        # Outcome: user retention (with true treatment effect)
        true_effect = 0.15  # 15% increase in retention
        baseline_retention = (
            0.6 + 0.01 * user_age + 0.05 * user_engagement + 0.2 * is_premium
        )
        retention_prob = baseline_retention + true_effect * treatment
        retention_prob = np.clip(retention_prob, 0, 1)
        retention = np.random.binomial(1, retention_prob, n)

        return {
            "treatment": TreatmentData(
                values=treatment, treatment_type="binary", name="feature_rollout"
            ),
            "outcome": OutcomeData(
                values=retention, outcome_type="binary", name="user_retention"
            ),
            "covariates": CovariateData(
                values=np.column_stack([user_age, user_engagement, is_premium]),
                names=["user_age", "user_engagement", "is_premium"],
            ),
            "true_effect": true_effect,
        }

    @patch("matplotlib.pyplot.show")
    def test_report_content_quality(self, mock_show, sample_scenario):
        """Test that generated reports contain high-quality, actionable content."""
        # Generate comprehensive report
        html_report = generate_diagnostic_report(
            treatment_data=sample_scenario["treatment"],
            outcome_data=sample_scenario["outcome"],
            covariates=sample_scenario["covariates"],
            estimator_name="A/B Test Analysis",
            ate_estimate=0.142,
            ate_ci_lower=0.089,
            ate_ci_upper=0.195,
        )

        # Test report structure
        assert "<!DOCTYPE html>" in html_report
        assert "<html" in html_report and "</html>" in html_report
        assert "<head>" in html_report and "</head>" in html_report
        assert "<body>" in html_report and "</body>" in html_report

        # Test content completeness
        assert "A/B Test Analysis" in html_report
        assert "feature_rollout" in html_report
        assert "user_retention" in html_report
        assert "0.142" in html_report  # ATE estimate

        # Test presence of key sections
        html_lower = html_report.lower()
        assert "analysis overview" in html_lower
        assert "overall assessment" in html_lower

        # Test that actionable recommendations are present
        assert any(marker in html_report for marker in ["✅", "⚠️", "❌"])
        assert "recommendation" in html_lower

        # Test file size requirement (< 5MB per issue)
        report_size_mb = len(html_report.encode("utf-8")) / (1024 * 1024)
        assert report_size_mb < 5.0, f"Report too large: {report_size_mb:.2f}MB"

    @patch("matplotlib.pyplot.show")
    def test_report_saves_correctly(self, mock_show, sample_scenario, tmp_path):
        """Test that reports save correctly to files."""
        output_path = tmp_path / "test_report.html"

        # Generate and save report
        html_report = generate_diagnostic_report(
            treatment_data=sample_scenario["treatment"],
            outcome_data=sample_scenario["outcome"],
            covariates=sample_scenario["covariates"],
            estimator_name="File Save Test",
            save_path=str(output_path),
        )

        # Check that file was created
        assert output_path.exists()
        assert output_path.is_file()

        # Check that file content matches returned content
        with open(output_path, encoding="utf-8") as f:
            saved_content = f.read()

        assert saved_content == html_report
        assert "File Save Test" in saved_content

    def test_report_performance_requirements(self, sample_scenario):
        """Test that report generation meets performance requirements."""
        import time

        # Test with requirements from issue: < 2s for 50k rows
        # We'll test proportionally with smaller data
        n_test = 5000  # 1/10th of requirement
        max_time = 0.2  # 1/10th of time requirement

        # Create larger dataset
        np.random.seed(42)
        treatment = np.random.binomial(1, 0.5, n_test)
        outcome = np.random.normal(0, 1, n_test)
        covariates = np.random.normal(0, 1, (n_test, 5))

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=covariates, names=[f"covar_{i}" for i in range(5)]
        )

        # Time the report generation
        start_time = time.time()

        with patch("matplotlib.pyplot.show"):
            html_report = generate_diagnostic_report(
                treatment_data=treatment_data,
                outcome_data=outcome_data,
                covariates=covariate_data,
                estimator_name="Performance Test",
            )

        elapsed_time = time.time() - start_time

        assert elapsed_time < max_time, (
            f"Report generation too slow: {elapsed_time:.3f}s > {max_time:.3f}s "
            f"for {n_test:,} observations"
        )
        assert len(html_report) > 1000  # Should still be substantial


class TestEdgeCases:
    """Test visualization with edge cases and challenging data."""

    def test_perfect_balance_scenario(self):
        """Test visualization when covariates are perfectly balanced."""
        np.random.seed(42)
        n = 200

        # Create perfectly balanced data
        treatment = np.repeat([0, 1], n // 2)

        # Covariates that are identical across groups
        age = np.random.normal(30, 5, n)
        income = np.random.normal(50000, 10000, n)

        # Shuffle to remove any ordering effects
        indices = np.random.permutation(n)
        treatment = treatment[indices]
        age = age[indices]
        income = income[indices]

        outcome = np.random.normal(100, 10, n)

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=np.column_stack([age, income]), names=["age", "income"]
        )

        with patch("matplotlib.pyplot.show"):
            # Should not crash with perfect balance
            love_fig = create_love_plot(
                covariate_data, treatment_data, interactive=False
            )
            assert love_fig is not None

            # Report should handle perfect balance gracefully
            html_report = generate_diagnostic_report(
                treatment_data=treatment_data,
                outcome_data=outcome_data,
                covariates=covariate_data,
                estimator_name="Perfect Balance Test",
            )

            assert "Perfect Balance Test" in html_report
            assert "✅" in html_report  # Should show good balance indicators

    def test_extreme_imbalance_scenario(self):
        """Test visualization with extremely imbalanced treatment groups."""
        np.random.seed(42)
        n = 1000

        # Very small treatment group (1%)
        treatment = np.zeros(n)
        treatment[:10] = 1  # Only 10 treated units

        # Covariates with extreme differences
        outcome = np.random.normal(100 + 50 * treatment, 10, n)
        age = np.random.normal(30 + 20 * treatment, 5, n)
        income = np.random.normal(50000 + 30000 * treatment, 10000, n)

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=np.column_stack([age, income]), names=["age", "income"]
        )

        with patch("matplotlib.pyplot.show"):
            # Should handle extreme imbalance without crashing
            html_report = generate_diagnostic_report(
                treatment_data=treatment_data,
                outcome_data=outcome_data,
                covariates=covariate_data,
                estimator_name="Extreme Imbalance Test",
            )

            assert "Extreme Imbalance Test" in html_report
            assert "10" in html_report  # Should show treated count
            assert "990" in html_report  # Should show control count

    def test_missing_components_gracefully(self):
        """Test that visualization handles missing components gracefully."""
        np.random.seed(42)
        n = 100

        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.normal(0, 1, n)

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")

        with patch("matplotlib.pyplot.show"):
            # Should work with minimal data (no covariates)
            html_report = generate_diagnostic_report(
                treatment_data=treatment_data,
                outcome_data=outcome_data,
                # No covariates, weights, propensity scores, or residuals
                estimator_name="Minimal Data Test",
            )

            assert "Minimal Data Test" in html_report
            assert (
                "None" in html_report or "0" in html_report
            )  # Should indicate no covariates
            assert len(html_report) > 1000  # Should still be substantial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
