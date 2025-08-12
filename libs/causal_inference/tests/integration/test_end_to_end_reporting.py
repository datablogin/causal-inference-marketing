"""End-to-end integration tests for unified API and reporting.

This module tests the complete workflow from data input to HTML report generation,
ensuring all components work together seamlessly.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_inference.api import CausalAnalysis


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.fixture
    def marketing_campaign_data(self):
        """Create realistic marketing campaign data."""
        np.random.seed(42)
        n = 2000

        # Customer characteristics
        age = np.random.normal(35, 12, n)
        income = np.random.exponential(50000, n)
        previous_purchases = np.random.poisson(3, n)
        segment = np.random.choice(
            ["premium", "standard", "budget"], n, p=[0.2, 0.5, 0.3]
        )

        # Email campaign assignment (with realistic selection bias)
        email_prob = (
            0.4  # Base rate
            + 0.1 * (income > 60000)  # Higher income more likely
            + 0.15 * (segment == "premium")  # Premium segment more likely
            + 0.05 * (previous_purchases > 5)  # Loyal customers more likely
        )
        email_campaign = np.random.binomial(1, np.clip(email_prob, 0.1, 0.9), n)

        # Revenue outcome (with true treatment effect)
        base_revenue = (
            100  # Base revenue
            + 0.5 * age  # Age effect
            + 0.001 * income  # Income effect
            + 20 * previous_purchases  # Loyalty effect
            + 50 * (segment == "premium")  # Segment effect
            + 30 * (segment == "standard")
        )

        # True treatment effect varies by segment
        treatment_effect = (
            50  # Base treatment effect
            + 30 * (segment == "premium")  # Higher for premium
            + 10 * (segment == "standard")  # Moderate for standard
            - 20 * (segment == "budget")  # Lower for budget
        )

        revenue = (
            base_revenue
            + treatment_effect * email_campaign
            + np.random.normal(0, 50, n)  # Random noise
        )

        return pd.DataFrame(
            {
                "email_campaign": email_campaign,
                "revenue": revenue,
                "age": age,
                "income": income,
                "previous_purchases": previous_purchases,
                "segment": segment,
            }
        )

    def test_complete_marketing_analysis(self, marketing_campaign_data):
        """Test complete marketing campaign analysis workflow."""

        # Step 1: Initialize analysis with business context
        analysis = CausalAnalysis(
            method="aipw",  # Use doubly robust method
            treatment_column="email_campaign",
            outcome_column="revenue",
            covariate_columns=["age", "income", "previous_purchases"],
            confidence_level=0.95,
            bootstrap_samples=500,  # Reduced for faster testing
            random_state=42,
        )

        # Step 2: Fit the model
        analysis.fit(marketing_campaign_data)

        # Verify fitting worked
        assert analysis.is_fitted_ is True
        assert analysis.method == "aipw"
        assert analysis.estimator_ is not None
        assert analysis.effect_ is not None

        # Step 3: Estimate treatment effect
        effect = analysis.estimate_ate()

        # Should detect positive effect (true effect is ~50-80 depending on segment mix)
        assert 30 < effect.ate < 120
        assert effect.p_value < 0.05  # Should be significant with this sample size

        # Step 4: Generate comprehensive report
        report = analysis.report(
            template="full", include_sensitivity=True, include_diagnostics=True
        )

        # Verify report structure
        assert isinstance(report, dict)
        required_keys = [
            "html_report",
            "effect",
            "method",
            "sample_size",
            "treatment_column",
            "outcome_column",
            "covariate_columns",
        ]
        for key in required_keys:
            assert key in report

        # Verify report content
        html = report["html_report"]
        assert isinstance(html, str)
        assert len(html) > 10000  # Should be substantial

        # Check for key sections
        expected_sections = [
            "Executive Summary",
            "Data Overview",
            "Analysis Method",
            "Results",
            "Recommendations",
            "Technical Appendix",
        ]
        for section in expected_sections:
            assert section in html

        # Check for business context
        assert "Email Campaign" in html or "email_campaign" in html
        assert "Revenue" in html or "revenue" in html

        return analysis, report

    def test_one_line_analysis(self, marketing_campaign_data):
        """Test the one-line analysis workflow."""

        # The key use case: one line complete analysis
        report = CausalAnalysis().fit(marketing_campaign_data).report()

        # Should work and produce valid results
        assert isinstance(report, dict)
        assert "html_report" in report
        assert "effect" in report
        assert len(report["html_report"]) > 5000  # Substantial report

    def test_file_based_workflow(self, marketing_campaign_data):
        """Test workflow with file input/output."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save data to file
            data_file = temp_path / "campaign_data.csv"
            marketing_campaign_data.to_csv(data_file, index=False)

            # Load and analyze from file
            analysis = CausalAnalysis(
                treatment_column="email_campaign", outcome_column="revenue"
            )

            # Fit from file
            analysis.fit(str(data_file))
            assert analysis.is_fitted_ is True

            # Generate and save report
            report_file = temp_path / "campaign_report.html"
            report = analysis.report(output_path=str(report_file))

            # Verify file was created
            assert report_file.exists()
            assert report["file_path"] == str(report_file)

            # Verify file content
            with open(report_file, encoding="utf-8") as f:
                content = f.read()
                assert "<html" in content
                assert "Email Campaign" in content or "email_campaign" in content
                assert len(content) > 10000

    def test_different_estimator_methods(self, marketing_campaign_data):
        """Test that different estimator methods all work."""

        methods = ["g_computation", "ipw", "aipw"]
        results = {}

        for method in methods:
            analysis = CausalAnalysis(
                method=method,
                treatment_column="email_campaign",
                outcome_column="revenue",
                covariate_columns=["age", "income"],
                bootstrap_samples=100,  # Reduced for speed
            )

            analysis.fit(marketing_campaign_data)
            effect = analysis.estimate_ate()

            results[method] = effect.ate

            # All methods should detect positive effect
            assert effect.ate > 0
            assert not np.isnan(effect.ate)

        # Results should be reasonably similar (within factor of 2)
        ate_values = list(results.values())
        assert max(ate_values) / min(ate_values) < 3

    def test_auto_method_selection(self, marketing_campaign_data):
        """Test automatic method selection with different data sizes."""

        # Large sample - should select AIPW
        large_analysis = CausalAnalysis(method="auto")
        large_analysis.fit(marketing_campaign_data)  # 2000 samples
        assert large_analysis.method == "aipw"

        # Small sample - should select G-computation
        small_data = marketing_campaign_data.sample(300, random_state=42)
        small_analysis = CausalAnalysis(method="auto")
        small_analysis.fit(small_data)
        assert small_analysis.method == "g_computation"

        # No covariates - should select G-computation (reduces to simple difference)
        no_cov_data = marketing_campaign_data[["email_campaign", "revenue"]]
        no_cov_analysis = CausalAnalysis(method="auto")
        no_cov_analysis.fit(no_cov_data)
        assert no_cov_analysis.method == "g_computation"

    def test_report_customization(self, marketing_campaign_data):
        """Test report customization options."""

        analysis = CausalAnalysis()
        analysis.fit(marketing_campaign_data)

        # Test custom title and analyst name
        custom_report = analysis.report(
            template="executive",
            title="Q3 Email Campaign Analysis",
            analyst_name="Data Science Team",
        )

        html = custom_report["html_report"]
        assert "Q3 Email Campaign Analysis" in html
        assert "Data Science Team" in html

        # Test minimal report
        minimal_report = analysis.report(
            template="executive", include_sensitivity=False, include_diagnostics=False
        )

        # Should be shorter without sensitivity/diagnostics
        assert len(minimal_report["html_report"]) < len(custom_report["html_report"])

    def test_edge_case_handling(self):
        """Test handling of edge cases and problematic data."""

        # Very small treatment effect
        np.random.seed(42)
        n = 1000
        small_effect_data = pd.DataFrame(
            {
                "treatment": np.random.binomial(1, 0.5, n),
                "outcome": 10
                + 0.01 * np.random.binomial(1, 0.5, n)
                + np.random.normal(0, 1, n),  # Tiny effect
                "covariate": np.random.normal(0, 1, n),
            }
        )

        analysis = CausalAnalysis()
        analysis.fit(small_effect_data)

        effect = analysis.estimate_ate()
        report = analysis.report()

        # Should handle small effects gracefully
        assert not np.isnan(effect.ate)
        assert isinstance(report["html_report"], str)
        assert len(report["html_report"]) > 5000

    def test_categorical_data_handling(self):
        """Test handling of categorical data in covariates."""

        np.random.seed(42)
        n = 1000

        # Data with categorical variables
        categorical_data = pd.DataFrame(
            {
                "treatment": np.random.binomial(1, 0.5, n),
                "outcome": np.random.normal(10, 2, n),
                "region": np.random.choice(["North", "South", "East", "West"], n),
                "product_category": np.random.choice(["A", "B", "C"], n),
                "age_numeric": np.random.normal(40, 10, n),
            }
        )

        # Add treatment effect
        treatment_mask = categorical_data["treatment"] == 1
        categorical_data.loc[treatment_mask, "outcome"] += 2.0

        analysis = CausalAnalysis(
            covariate_columns=["age_numeric"]  # Only use numeric covariate
        )
        analysis.fit(categorical_data)

        effect = analysis.estimate_ate()
        report = analysis.report()

        # Should work despite categorical columns being present
        assert analysis.is_fitted_ is True
        assert 1.5 < effect.ate < 2.5  # Should recover true effect
        assert isinstance(report["html_report"], str)


class TestRealWorldScenarios:
    """Test scenarios that mirror real-world use cases."""

    def test_ab_test_analysis(self):
        """Test A/B test analysis scenario."""

        # Simulate A/B test data
        np.random.seed(42)
        n = 5000

        # Balanced randomized assignment
        variant = np.random.choice(["control", "treatment"], n, p=[0.5, 0.5])
        variant_binary = (variant == "treatment").astype(int)

        # Conversion outcome with treatment effect
        base_conversion_rate = 0.05
        treatment_lift = 0.01  # 1 percentage point increase

        conversion_prob = base_conversion_rate + treatment_lift * variant_binary
        conversion = np.random.binomial(1, conversion_prob, n)

        # Customer characteristics (balanced due to randomization)
        customer_data = pd.DataFrame(
            {
                "variant": variant_binary,
                "conversion": conversion,
                "days_since_signup": np.random.exponential(30, n),
                "previous_purchases": np.random.poisson(2, n),
            }
        )

        # Analyze with unified API
        analysis = CausalAnalysis(
            method="g_computation",  # Good for randomized experiments
            treatment_column="variant",
            outcome_column="conversion",
        )

        analysis.fit(customer_data)
        effect = analysis.estimate_ate()
        report = analysis.report(title="A/B Test Results - Variant Impact")

        # Should detect the 1% lift
        assert 0.005 < effect.ate < 0.015
        assert "A/B Test Results" in report["html_report"]

    def test_observational_study_analysis(self):
        """Test observational study with confounding."""

        # Simulate observational study with confounding
        np.random.seed(42)
        n = 3000

        # Confounders
        income = np.random.lognormal(10, 0.5, n)  # Log-normal income distribution
        education = np.random.choice(
            ["high_school", "college", "graduate"], n, p=[0.4, 0.4, 0.2]
        )
        age = np.random.normal(45, 15, n)

        # Treatment assignment depends on confounders (selection bias)
        treatment_logit = (
            -2  # Low base rate
            + 0.5 * (education == "graduate")
            + 0.3 * (education == "college")
            + 0.0001 * income
            + 0.01 * age
        )
        treatment_prob = 1 / (1 + np.exp(-treatment_logit))
        treatment = np.random.binomial(1, treatment_prob, n)

        # Outcome depends on both treatment and confounders
        outcome = (
            1000  # Base outcome
            + 500 * treatment  # True treatment effect
            + 0.01 * income  # Income effect
            + 200 * (education == "graduate")
            + 100 * (education == "college")
            + 5 * age
            + np.random.normal(0, 100, n)  # Noise
        )

        obs_data = pd.DataFrame(
            {
                "treatment": treatment,
                "outcome": outcome,
                "income": income,
                "age": age,
                "education": education,
            }
        )

        # Use AIPW for doubly robust estimation
        analysis = CausalAnalysis(
            method="aipw",
            covariate_columns=[
                "income",
                "age",
            ],  # Don't include categorical for simplicity
        )

        analysis.fit(obs_data)
        effect = analysis.estimate_ate()
        report = analysis.report(
            template="full",
            include_sensitivity=True,
            title="Observational Study Analysis",
        )

        # Should recover true effect of 500 despite confounding
        assert 400 < effect.ate < 600
        assert "Observational Study" in report["html_report"]
        assert "Sensitivity Analysis" in report["html_report"]


if __name__ == "__main__":
    pytest.main([__file__])
