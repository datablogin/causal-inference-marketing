"""Comprehensive tests that validate the entire tutorial workflow.

This module tests complete end-to-end workflows demonstrated in the tutorial
notebooks to ensure they work correctly and produce expected results.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.data.synthetic import SyntheticDataGenerator
from causal_inference.diagnostics.balance import check_covariate_balance
from causal_inference.diagnostics.overlap import assess_positivity
from causal_inference.diagnostics.reporting import generate_diagnostic_report
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator


class TestBasicUsageTutorial:
    """Test the complete basic usage tutorial workflow."""

    def test_basic_tutorial_end_to_end(self):
        """Test complete basic usage tutorial workflow."""
        # Step 1: Data generation (as in tutorial)
        np.random.seed(42)
        generator = SyntheticDataGenerator(random_state=42)

        treatment, outcome, covariates = generator.generate_linear_binary_treatment(
            n_samples=500,  # Smaller for test performance
            n_confounders=4,
            treatment_effect=2.5,
            confounding_strength=0.5,
        )

        # Verify data structure
        assert len(treatment.values) == 500
        assert treatment.treatment_type == "binary"
        assert outcome.outcome_type == "continuous"
        assert covariates.values.shape == (500, 4)

        # Step 2: Data validation (as in tutorial)
        from causal_inference.data.validation import validate_causal_data

        # Should not raise errors
        validate_causal_data(treatment, outcome, covariates, check_overlap=True)

        # Step 3: Estimate effects with all three methods
        estimators = {
            "g_computation": GComputationEstimator(bootstrap_samples=50),
            "ipw": IPWEstimator(bootstrap_samples=50),
            "aipw": AIPWEstimator(bootstrap_samples=50),
        }

        results = {}
        for name, estimator in estimators.items():
            estimator.fit(treatment, outcome, covariates)
            effect = estimator.estimate_ate()
            results[name] = effect.ate

            # Check that estimates are reasonable
            assert np.isfinite(effect.ate)
            assert effect.confidence_interval is not None
            assert len(effect.confidence_interval) == 2

            # Should be reasonably close to true effect (2.5)
            assert abs(effect.ate - 2.5) < 1.5, (
                f"{name} estimate {effect.ate:.2f} too far from true effect 2.5"
            )

        # Step 4: Run diagnostics (as in tutorial)
        balance_result = check_covariate_balance(treatment, covariates)
        overlap_result = assess_positivity(treatment, covariates)

        assert balance_result is not None
        assert hasattr(balance_result, "overall_balance_met")
        assert overlap_result is not None
        assert hasattr(overlap_result, "overall_positivity_met")

        # Step 5: Generate diagnostic report
        diagnostic_report = generate_diagnostic_report(
            treatment=treatment, outcome=outcome, covariates=covariates
        )

        assert diagnostic_report is not None
        assert hasattr(diagnostic_report, "overall_assessment")


class TestMarketingUseCaseTutorial:
    """Test the marketing use case tutorial workflows."""

    def test_email_marketing_workflow(self):
        """Test email marketing campaign analysis workflow."""
        np.random.seed(123)
        n_customers = 1000  # Reduced for test performance

        # Generate realistic email marketing data (as in tutorial)
        age = np.random.uniform(18, 70, n_customers)
        income = np.random.lognormal(10.5, 0.5, n_customers)
        past_purchases = np.random.poisson(3, n_customers)
        email_engagement = np.random.beta(2, 5, n_customers)
        days_since_last_purchase = np.random.exponential(30, n_customers)

        # Email targeting (confounded)
        email_propensity = (
            -2.5
            + 0.02 * age
            + 0.00001 * income
            + 0.3 * past_purchases
            + 2.0 * email_engagement
            - 0.01 * days_since_last_purchase
        )

        email_prob = 1 / (1 + np.exp(-email_propensity))
        received_email = np.random.binomial(1, email_prob, n_customers)

        # Purchase outcome with true effect of 25
        purchase_amount = (
            50
            + 1.5 * age
            + 0.002 * income
            + 15 * past_purchases
            + 100 * email_engagement
            - 0.5 * days_since_last_purchase
            + 25 * received_email
            + np.random.normal(0, 20, n_customers)
        )
        purchase_amount = np.maximum(purchase_amount, 0)

        # Create data objects
        email_data = {
            "treatment": TreatmentData(values=received_email, treatment_type="binary"),
            "outcome": OutcomeData(values=purchase_amount, outcome_type="continuous"),
            "covariates": CovariateData(
                values=pd.DataFrame(
                    {
                        "age": age,
                        "income": income,
                        "past_purchases": past_purchases,
                        "email_engagement": email_engagement,
                        "days_since_last_purchase": days_since_last_purchase,
                    }
                )
            ),
        }

        # Analyze with AIPW using flexible models
        aipw_email = AIPWEstimator(
            outcome_model=RandomForestRegressor(n_estimators=50, random_state=42),
            propensity_model=RandomForestClassifier(n_estimators=50, random_state=42),
            bootstrap_samples=50,
        )

        aipw_email.fit(
            email_data["treatment"], email_data["outcome"], email_data["covariates"]
        )
        email_effect = aipw_email.estimate_ate()

        # Verify results
        assert np.isfinite(email_effect.ate)
        assert email_effect.confidence_interval is not None

        # Should recover something close to true effect (25)
        assert abs(email_effect.ate - 25) < 15, (
            f"Email effect {email_effect.ate:.2f} too far from true effect 25"
        )

        # Check that naive analysis is different (biased)
        naive_diff = np.mean(purchase_amount[received_email == 1]) - np.mean(
            purchase_amount[received_email == 0]
        )
        assert abs(naive_diff - email_effect.ate) > 5, (
            "Causal and naive estimates should differ due to confounding"
        )

    def test_price_promotion_workflow(self):
        """Test price promotion impact analysis workflow."""
        np.random.seed(456)
        n_products = 800  # Reduced for test performance

        # Product characteristics (as in tutorial)
        base_price = np.random.uniform(10, 100, n_products)
        product_age = np.random.uniform(0, 365, n_products)
        category_popularity = np.random.normal(0, 1, n_products)
        inventory_level = np.random.exponential(50, n_products)
        seasonality = np.sin(2 * np.pi * np.arange(n_products) / 365)

        # Promotions target problem products (confounded)
        promotion_propensity = (
            -1.0
            - 0.02 * base_price
            + 0.005 * product_age
            - 0.5 * category_popularity
            + 0.01 * inventory_level
            - 0.5 * seasonality
        )

        promotion_prob = 1 / (1 + np.exp(-promotion_propensity))
        has_promotion = np.random.binomial(1, promotion_prob, n_products)

        # Sales with true 40% promotion effect (log scale)
        log_sales = (
            3.0
            - 0.02 * base_price
            - 0.001 * product_age
            + 0.8 * category_popularity
            - 0.005 * inventory_level
            + 0.3 * seasonality
            + 0.4 * has_promotion
            + np.random.normal(0, 0.3, n_products)
        )

        sales_units = np.exp(log_sales)

        # Create promotion data
        promotion_data = {
            "treatment": TreatmentData(values=has_promotion, treatment_type="binary"),
            "outcome": OutcomeData(values=sales_units, outcome_type="continuous"),
            "covariates": CovariateData(
                values=pd.DataFrame(
                    {
                        "base_price": base_price,
                        "product_age": product_age,
                        "category_popularity": category_popularity,
                        "inventory_level": inventory_level,
                        "seasonality": seasonality,
                    }
                )
            ),
        }

        # Analyze with AIPW
        aipw_promotion = AIPWEstimator(
            outcome_model=RandomForestRegressor(n_estimators=50, random_state=42),
            propensity_model=RandomForestClassifier(n_estimators=50, random_state=42),
            bootstrap_samples=50,
        )

        aipw_promotion.fit(
            promotion_data["treatment"],
            promotion_data["outcome"],
            promotion_data["covariates"],
        )
        promotion_effect = aipw_promotion.estimate_ate()

        # Verify results
        assert np.isfinite(promotion_effect.ate)
        assert promotion_effect.confidence_interval is not None

        # Convert to percentage effect
        baseline_sales = np.mean(sales_units[has_promotion == 0])
        percentage_lift = (promotion_effect.ate / baseline_sales) * 100

        # Should be reasonably close to true 40% effect
        assert abs(percentage_lift - 40) < 25, (
            f"Promotion effect {percentage_lift:.1f}% too far from true 40%"
        )

    def test_loyalty_program_workflow(self):
        """Test loyalty program evaluation workflow."""
        np.random.seed(789)
        n_customers = 1500  # Reduced for test performance

        # Customer characteristics (as in tutorial)
        customer_age = np.random.uniform(25, 65, n_customers)
        income_segment = np.random.choice([1, 2, 3], n_customers, p=[0.3, 0.5, 0.2])
        frequency_segment = np.random.choice([1, 2, 3], n_customers, p=[0.4, 0.4, 0.2])
        years_as_customer = np.random.exponential(2, n_customers)
        channel_preference = np.random.choice([0, 1], n_customers, p=[0.6, 0.4])

        # Loyalty program enrollment (selection bias)
        loyalty_propensity = (
            -3.0
            + 0.03 * customer_age
            + 0.8 * income_segment
            + 1.2 * frequency_segment
            + 0.3 * years_as_customer
            + 0.5 * channel_preference
        )

        loyalty_prob = 1 / (1 + np.exp(-loyalty_propensity))
        in_loyalty_program = np.random.binomial(1, loyalty_prob, n_customers)

        # Annual spend with true $300 loyalty effect
        annual_spend = (
            1000
            + 20 * customer_age
            + 800 * income_segment
            + 600 * frequency_segment
            + 100 * years_as_customer
            + 200 * channel_preference
            + 300 * in_loyalty_program
            + np.random.normal(0, 400, n_customers)
        )
        annual_spend = np.maximum(annual_spend, 100)

        # Create loyalty data
        loyalty_data = {
            "treatment": TreatmentData(
                values=in_loyalty_program, treatment_type="binary"
            ),
            "outcome": OutcomeData(values=annual_spend, outcome_type="continuous"),
            "covariates": CovariateData(
                values=pd.DataFrame(
                    {
                        "customer_age": customer_age,
                        "income_segment": income_segment,
                        "frequency_segment": frequency_segment,
                        "years_as_customer": years_as_customer,
                        "channel_preference": channel_preference,
                    }
                )
            ),
        }

        # Analyze with AIPW
        aipw_loyalty = AIPWEstimator(
            outcome_model=RandomForestRegressor(n_estimators=50, random_state=42),
            propensity_model=RandomForestClassifier(n_estimators=50, random_state=42),
            bootstrap_samples=50,
        )

        aipw_loyalty.fit(
            loyalty_data["treatment"],
            loyalty_data["outcome"],
            loyalty_data["covariates"],
        )
        loyalty_effect = aipw_loyalty.estimate_ate()

        # Verify results
        assert np.isfinite(loyalty_effect.ate)
        assert loyalty_effect.confidence_interval is not None

        # Should be reasonably close to true effect ($300)
        assert abs(loyalty_effect.ate - 300) < 150, (
            f"Loyalty effect ${loyalty_effect.ate:.0f} too far from true $300"
        )

        # Check that selection bias exists
        naive_diff = np.mean(annual_spend[in_loyalty_program == 1]) - np.mean(
            annual_spend[in_loyalty_program == 0]
        )
        assert naive_diff > loyalty_effect.ate, (
            "Naive analysis should overestimate due to selection bias"
        )


class TestAdvancedTutorialConcepts:
    """Test advanced tutorial concepts and edge cases."""

    def test_multi_estimator_consistency(self):
        """Test that multiple estimators give consistent results on clean data."""
        # Generate clean, simple data with known effect
        np.random.seed(999)
        n_samples = 500

        # Simple confounders
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)

        # Random treatment (minimal confounding)
        treatment = np.random.binomial(1, 0.5, n_samples)

        # Outcome with clear treatment effect
        outcome = (
            10 + 2 * X1 + 3 * X2 + 5 * treatment + np.random.normal(0, 1, n_samples)
        )

        # Create data objects
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2}))

        # Estimate with all methods
        estimators = [
            GComputationEstimator(bootstrap_samples=50),
            IPWEstimator(bootstrap_samples=50),
            AIPWEstimator(bootstrap_samples=50),
        ]

        estimates = []
        for estimator in estimators:
            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()
            estimates.append(effect.ate)

        # All estimates should be close to true effect (5) and each other
        for estimate in estimates:
            assert abs(estimate - 5) < 1.0, (
                f"Estimate {estimate:.2f} too far from true effect 5"
            )

        # Estimates should be reasonably consistent
        estimate_range = max(estimates) - min(estimates)
        assert estimate_range < 1.5, (
            f"Estimates too variable: range = {estimate_range:.2f}"
        )

    def test_tutorial_robustness_to_model_choice(self):
        """Test that tutorial results are robust to different model choices."""
        # Generate synthetic marketing data
        np.random.seed(555)
        generator = SyntheticDataGenerator(random_state=555)

        treatment, outcome, covariates = generator.generate_linear_binary_treatment(
            n_samples=400, n_confounders=3, treatment_effect=3.0
        )

        # Test different model combinations
        model_combinations = [
            {"outcome": "linear", "propensity": "logistic"},
            {
                "outcome": RandomForestRegressor(n_estimators=50, random_state=42),
                "propensity": RandomForestClassifier(n_estimators=50, random_state=42),
            },
        ]

        estimates = []
        for models in model_combinations:
            aipw = AIPWEstimator(
                outcome_model=models["outcome"],
                propensity_model=models["propensity"],
                bootstrap_samples=50,
            )

            aipw.fit(treatment, outcome, covariates)
            effect = aipw.estimate_ate()
            estimates.append(effect.ate)

        # Results should be reasonably similar across model choices
        estimate_range = max(estimates) - min(estimates)
        assert estimate_range < 1.0, (
            f"Results too sensitive to model choice: range = {estimate_range:.2f}"
        )

        # All should be reasonably close to true effect
        for estimate in estimates:
            assert abs(estimate - 3.0) < 1.0, (
                f"Estimate {estimate:.2f} too far from true effect 3.0"
            )

    def test_tutorial_diagnostics_workflow(self):
        """Test complete diagnostics workflow from tutorials."""
        # Generate data with known confounding structure
        np.random.seed(777)
        n_samples = 600

        # Strong confounders
        age = np.random.uniform(18, 80, n_samples)
        income = np.random.exponential(50000, n_samples)

        # Treatment depends on confounders
        propensity = 1 / (
            1 + np.exp(-(-2 + 0.02 * (age - 40) + 0.00002 * (income - 50000)))
        )
        treatment = np.random.binomial(1, propensity, n_samples)

        # Outcome depends on confounders and treatment
        outcome = (
            1000
            + 20 * age
            + 0.1 * income
            + 5000 * treatment
            + np.random.normal(0, 1000, n_samples)
        )

        # Create data objects
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame({"age": age, "income": income})
        )

        # Run all diagnostic checks
        balance_result = check_covariate_balance(treatment_data, covariate_data)
        overlap_result = assess_positivity(treatment_data, covariate_data)
        diagnostic_report = generate_diagnostic_report(
            treatment_data, outcome_data, covariate_data
        )

        # Verify diagnostics structure
        assert hasattr(balance_result, "standardized_mean_differences")
        assert hasattr(balance_result, "overall_balance_met")
        assert hasattr(overlap_result, "overall_positivity_met")
        assert hasattr(diagnostic_report, "overall_assessment")

        # Should detect confounding (poor balance expected)
        assert not balance_result.overall_balance_met, (
            "Should detect confounding in this data"
        )

        # Should have reasonable overlap
        assert overlap_result.overall_positivity_met, (
            "Should have common support with this data"
        )


@pytest.mark.slow  # Changed from integration to avoid warning
class TestTutorialIntegration:
    """Integration tests that validate complete tutorial workflows."""

    def test_complete_marketing_analysis_pipeline(self):
        """Test complete marketing analysis pipeline from data to business insights."""
        # This test validates the entire workflow that would be used in practice

        # 1. Data generation (simulating real marketing scenario)
        np.random.seed(12345)
        generator = SyntheticDataGenerator(random_state=12345)

        # Generate email marketing data
        treatment, outcome, covariates = generator.generate_linear_binary_treatment(
            n_samples=1000,
            n_confounders=5,
            treatment_effect=15.0,  # $15 lift from email
            confounding_strength=0.7,  # Strong confounding
        )

        # 2. Data validation
        from causal_inference.data.validation import validate_causal_data

        validate_causal_data(treatment, outcome, covariates, check_overlap=True)

        # 3. Diagnostic analysis
        balance_result = check_covariate_balance(treatment, covariates)
        overlap_result = assess_positivity(treatment, covariates)

        # 4. Causal effect estimation
        aipw = AIPWEstimator(bootstrap_samples=100)
        aipw.fit(treatment, outcome, covariates)
        effect = aipw.estimate_ate()

        # 5. Business impact calculation
        customers_per_month = 10000
        treatment_rate = np.mean(treatment.values)
        monthly_treated = customers_per_month * treatment_rate
        monthly_lift = monthly_treated * effect.ate
        annual_lift = monthly_lift * 12

        # 6. Verify complete pipeline
        assert np.isfinite(effect.ate)
        assert effect.confidence_interval is not None
        assert np.isfinite(monthly_lift)
        assert np.isfinite(annual_lift)
        assert abs(effect.ate - 15.0) < 5.0  # Should recover true effect

        # 7. Generate business summary
        summary = {
            "causal_effect": effect.ate,
            "confidence_interval": effect.confidence_interval,
            "monthly_lift": monthly_lift,
            "annual_lift": annual_lift,
            "is_significant": effect.is_significant,
            "balance_assessment": balance_result.overall_balance_met,
            "overlap_satisfied": overlap_result.overall_positivity_met,
        }

        # Verify business summary is complete and reasonable
        assert all(
            key in summary
            for key in [
                "causal_effect",
                "confidence_interval",
                "monthly_lift",
                "annual_lift",
                "is_significant",
                "balance_assessment",
                "overlap_satisfied",
            ]
        )

        assert summary["is_significant"], (
            "Effect should be significant with this effect size"
        )
        assert summary["monthly_lift"] > 0, "Should show positive business impact"

    def test_multi_use_case_comparison(self):
        """Test comparison across multiple marketing use cases."""
        # Generate different marketing scenarios and compare results

        scenarios = {}
        np.random.seed(54321)

        # Email marketing
        generator_email = SyntheticDataGenerator(random_state=100)
        treatment_email, outcome_email, covariates_email = (
            generator_email.generate_linear_binary_treatment(
                n_samples=500, n_confounders=3, treatment_effect=20.0
            )
        )

        # Social media advertising
        generator_social = SyntheticDataGenerator(random_state=200)
        treatment_social, outcome_social, covariates_social = (
            generator_social.generate_linear_binary_treatment(
                n_samples=500, n_confounders=3, treatment_effect=35.0
            )
        )

        # Loyalty program
        generator_loyalty = SyntheticDataGenerator(random_state=300)
        treatment_loyalty, outcome_loyalty, covariates_loyalty = (
            generator_loyalty.generate_linear_binary_treatment(
                n_samples=500, n_confounders=3, treatment_effect=50.0
            )
        )

        scenarios = {
            "Email Marketing": (treatment_email, outcome_email, covariates_email, 20.0),
            "Social Media": (treatment_social, outcome_social, covariates_social, 35.0),
            "Loyalty Program": (
                treatment_loyalty,
                outcome_loyalty,
                covariates_loyalty,
                50.0,
            ),
        }

        results = {}
        for name, (treatment, outcome, covariates, true_effect) in scenarios.items():
            aipw = AIPWEstimator(bootstrap_samples=50)
            aipw.fit(treatment, outcome, covariates)
            effect = aipw.estimate_ate()

            results[name] = {
                "estimated_effect": effect.ate,
                "true_effect": true_effect,
                "error": abs(effect.ate - true_effect),
                "ci_width": effect.confidence_interval[1]
                - effect.confidence_interval[0],
            }

        # Verify all scenarios work
        for name, result in results.items():
            assert np.isfinite(result["estimated_effect"])
            assert (
                result["error"] < result["true_effect"] * 0.5
            )  # Within 50% of true effect
            assert result["ci_width"] > 0  # Valid confidence interval

        # Verify ranking matches true effects
        estimated_ranking = sorted(
            results.keys(), key=lambda x: results[x]["estimated_effect"]
        )

        # Should get the correct ordering (may allow some flexibility)
        assert estimated_ranking.index("Email Marketing") < estimated_ranking.index(
            "Loyalty Program"
        )
        assert estimated_ranking.index("Social Media") < estimated_ranking.index(
            "Loyalty Program"
        )
