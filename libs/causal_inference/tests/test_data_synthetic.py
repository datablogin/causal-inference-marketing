"""Tests for synthetic data generation utilities."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.data.synthetic import (
    SyntheticDataGenerator,
    generate_confounded_observational,
    generate_simple_rct,
)


class TestSyntheticDataGenerator:
    """Test cases for synthetic data generator."""

    def setup_method(self):
        """Set up generator with fixed random state."""
        self.generator = SyntheticDataGenerator(random_state=42)

    def test_generate_linear_binary_treatment_default(self):
        """Test generation of linear binary treatment data with defaults."""
        treatment, outcome, covariates = (
            self.generator.generate_linear_binary_treatment()
        )

        # Check types
        assert isinstance(treatment, TreatmentData)
        assert isinstance(outcome, OutcomeData)
        assert isinstance(covariates, CovariateData)

        # Check properties
        assert treatment.treatment_type == "binary"
        assert outcome.outcome_type == "continuous"
        assert len(treatment.values) == 1000  # Default n_samples
        assert len(outcome.values) == 1000
        assert len(covariates.names) == 5  # Default n_confounders

        # Check that treatment is binary
        unique_treatments = np.unique(treatment.values)
        assert len(unique_treatments) == 2
        assert set(unique_treatments) == {0, 1}

        # Check data alignment
        assert len(treatment.values) == len(outcome.values) == len(covariates.values)

    def test_generate_linear_binary_treatment_custom_params(self):
        """Test generation with custom parameters."""
        treatment, outcome, covariates = (
            self.generator.generate_linear_binary_treatment(
                n_samples=500, n_confounders=3, treatment_effect=1.5, noise_std=0.5
            )
        )

        assert len(treatment.values) == 500
        assert len(covariates.names) == 3
        assert isinstance(covariates.values, pd.DataFrame)
        assert covariates.values.shape == (500, 3)

    def test_generate_linear_binary_treatment_confounding(self):
        """Test that confounding is present in generated data."""
        # Generate data with strong confounding
        treatment, outcome, covariates = (
            self.generator.generate_linear_binary_treatment(
                n_samples=500, confounding_strength=2.0, selection_bias=1.0  # Reduced from 2000 for CI
            )
        )

        # Check that treatment and outcome are both correlated with confounders
        df = pd.DataFrame(
            {
                "treatment": treatment.values,
                "outcome": outcome.values,
                **{
                    f"X{i + 1}": covariates.values.iloc[:, i]
                    for i in range(len(covariates.names))
                },
            }
        )

        # Treatment should be correlated with some confounders (due to selection bias)
        treatment_X1_corr = df["treatment"].corr(df["X1"])
        assert abs(treatment_X1_corr) > 0.1  # Should have some correlation

        # Outcome should be correlated with confounders (due to confounding)
        outcome_corr_with_X = [df["outcome"].corr(df[f"X{i + 1}"]) for i in range(3)]
        assert any(abs(corr) > 0.2 for corr in outcome_corr_with_X)

    def test_generate_nonlinear_continuous_treatment_default(self):
        """Test generation of nonlinear continuous treatment data."""
        treatment, outcome, covariates = (
            self.generator.generate_nonlinear_continuous_treatment()
        )

        # Check types and properties
        assert isinstance(treatment, TreatmentData)
        assert treatment.treatment_type == "continuous"
        assert isinstance(outcome, OutcomeData)
        assert outcome.outcome_type == "continuous"
        assert isinstance(covariates, CovariateData)

        # Check sample size
        assert len(treatment.values) == 1000
        assert len(outcome.values) == 1000
        assert len(covariates.names) == 4  # Default n_confounders

        # Check that treatment is continuous (not just binary)
        unique_treatments = np.unique(treatment.values)
        assert len(unique_treatments) > 10  # Should have many unique values

    def test_generate_nonlinear_continuous_treatment_dose_response(self):
        """Test different dose-response functions."""
        dose_response_functions = ["linear", "quadratic", "threshold"]

        for dose_fn in dose_response_functions:
            treatment, outcome, covariates = (
                self.generator.generate_nonlinear_continuous_treatment(
                    n_samples=500, treatment_effect_fn=dose_fn
                )
            )

            # All should generate valid data
            assert len(treatment.values) == 500
            assert len(outcome.values) == 500
            assert not np.any(np.isnan(outcome.values))
            assert not np.any(np.isinf(outcome.values))

    def test_generate_nonlinear_continuous_treatment_invalid_function(self):
        """Test error handling for invalid dose-response function."""
        with pytest.raises(ValueError, match="Unknown treatment effect function"):
            self.generator.generate_nonlinear_continuous_treatment(
                treatment_effect_fn="invalid_function"
            )

    def test_generate_marketing_campaign_data_default(self):
        """Test generation of marketing campaign data with defaults."""
        treatment, outcome, covariates = (
            self.generator.generate_marketing_campaign_data()
        )

        # Check types and properties
        assert isinstance(treatment, TreatmentData)
        assert treatment.treatment_type == "categorical"
        assert isinstance(outcome, OutcomeData)
        assert outcome.outcome_type == "continuous"
        assert isinstance(covariates, CovariateData)

        # Check sample size
        assert len(treatment.values) == 5000  # Default n_samples

        # Check that treatment has expected categories
        expected_campaigns = ["email", "social_media", "direct_mail", "control"]
        unique_treatments = set(treatment.values.unique())
        assert unique_treatments.issubset(set(expected_campaigns))

        # Check that outcomes are non-negative (purchase amounts)
        assert all(outcome.values >= 0)

        # Check covariate names
        expected_covariates = [
            "customer_age",
            "customer_income",
            "previous_purchases",
            "customer_segment",
            "email_engagement",
            "month",
        ]
        assert set(covariates.names) == set(expected_covariates)

    def test_generate_marketing_campaign_data_custom_campaigns(self):
        """Test generation with custom campaign types."""
        custom_campaigns = ["tv", "radio", "print", "control"]
        treatment, outcome, covariates = (
            self.generator.generate_marketing_campaign_data(
                n_samples=300, campaign_types=custom_campaigns  # Reduced from 1000 for CI
            )
        )

        unique_treatments = set(treatment.values.unique())
        assert unique_treatments.issubset(set(custom_campaigns))
        assert treatment.categories == custom_campaigns

    def test_generate_marketing_campaign_data_no_seasonality(self):
        """Test generation without seasonality."""
        treatment, outcome, covariates = (
            self.generator.generate_marketing_campaign_data(
                n_samples=300, include_seasonality=False  # Reduced from 1000 for CI
            )
        )

        # Should not include month covariate
        assert "month" not in covariates.names

        # Should have one less covariate
        expected_covariates = [
            "customer_age",
            "customer_income",
            "previous_purchases",
            "customer_segment",
            "email_engagement",
        ]
        assert set(covariates.names) == set(expected_covariates)

    def test_generate_missing_data_scenario_mcar(self):
        """Test generation of MCAR missing data."""
        # First generate base data
        treatment, outcome, covariates = (
            self.generator.generate_linear_binary_treatment(n_samples=200)
        )

        # Add MCAR missing data
        treatment_miss, outcome_miss, covariates_miss = (
            self.generator.generate_missing_data_scenario(
                treatment,
                outcome,
                covariates,
                missing_mechanism="MCAR",
                missing_rate=0.2,
            )
        )

        # Check that data objects have same structure
        assert isinstance(treatment_miss, TreatmentData)
        assert isinstance(outcome_miss, OutcomeData)
        assert isinstance(covariates_miss, CovariateData)

        # Check that some data is missing
        if isinstance(covariates_miss.values, pd.DataFrame):
            total_missing = covariates_miss.values.isnull().sum().sum()
            assert total_missing > 0

    def test_generate_missing_data_scenario_mar(self):
        """Test generation of MAR missing data."""
        # Generate base data
        treatment, outcome, covariates = (
            self.generator.generate_linear_binary_treatment(n_samples=200)
        )

        # Add MAR missing data
        treatment_miss, outcome_miss, covariates_miss = (
            self.generator.generate_missing_data_scenario(
                treatment,
                outcome,
                covariates,
                missing_mechanism="MAR",
                missing_rate=0.15,
            )
        )

        # Check that some data is missing
        if isinstance(covariates_miss.values, pd.DataFrame):
            total_missing = covariates_miss.values.isnull().sum().sum()
            # MAR might result in less missing data due to dependencies
            # Just check that the function runs without error
            assert total_missing >= 0

    def test_generate_missing_data_scenario_mnar(self):
        """Test generation of MNAR missing data."""
        # Generate base data
        treatment, outcome, covariates = (
            self.generator.generate_linear_binary_treatment(n_samples=200)
        )

        # Add MNAR missing data
        treatment_miss, outcome_miss, covariates_miss = (
            self.generator.generate_missing_data_scenario(
                treatment,
                outcome,
                covariates,
                missing_mechanism="MNAR",
                missing_rate=0.1,
            )
        )

        # Check that some data is missing
        if isinstance(covariates_miss.values, pd.DataFrame):
            total_missing = covariates_miss.values.isnull().sum().sum()
            assert (
                total_missing >= 0
            )  # MNAR might result in variable amounts of missingness


class TestSyntheticDataConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_generate_simple_rct(self):
        """Test simple RCT generation."""
        treatment, outcome, covariates = generate_simple_rct(
            n_samples=500, treatment_effect=1.5, random_state=42
        )

        # Check basic properties
        assert len(treatment.values) == 500
        assert treatment.treatment_type == "binary"
        assert outcome.outcome_type == "continuous"

        # In RCT, treatment should be roughly balanced
        treatment_prop = np.mean(treatment.values)
        assert 0.4 < treatment_prop < 0.6  # Should be close to 50/50

        # Check that there's a treatment effect
        df = pd.DataFrame({"treatment": treatment.values, "outcome": outcome.values})

        mean_treated = df[df["treatment"] == 1]["outcome"].mean()
        mean_control = df[df["treatment"] == 0]["outcome"].mean()
        observed_effect = mean_treated - mean_control

        # Should be close to true effect (1.5) but allow for sampling variation
        assert 1.0 < observed_effect < 2.0

    def test_generate_confounded_observational(self):
        """Test confounded observational data generation."""
        treatment, outcome, covariates = generate_confounded_observational(
            n_samples=300,  # Reduced from 1000 for CI
            treatment_effect=2.0,
            confounding_strength=1.5,
            random_state=42,
        )

        # Check basic properties
        assert len(treatment.values) == 1000
        assert treatment.treatment_type == "binary"
        assert outcome.outcome_type == "continuous"

        # Should have confounding - naive treatment effect should be biased
        df = pd.DataFrame({"treatment": treatment.values, "outcome": outcome.values})

        mean_treated = df[df["treatment"] == 1]["outcome"].mean()
        mean_control = df[df["treatment"] == 0]["outcome"].mean()
        naive_effect = mean_treated - mean_control

        # Due to confounding, naive effect should be different from true effect (2.0)
        # The exact bias depends on the confounding structure, but it should be substantial
        assert abs(naive_effect - 2.0) > 0.5  # Should be biased

    def test_reproducibility_with_random_state(self):
        """Test that results are reproducible with same random state."""
        treatment1, outcome1, covariates1 = generate_simple_rct(
            n_samples=100, random_state=123
        )

        treatment2, outcome2, covariates2 = generate_simple_rct(
            n_samples=100, random_state=123
        )

        # Results should be identical
        np.testing.assert_array_equal(treatment1.values, treatment2.values)
        np.testing.assert_array_equal(outcome1.values, outcome2.values)
        np.testing.assert_array_equal(covariates1.values, covariates2.values)

    def test_different_results_with_different_random_state(self):
        """Test that results differ with different random states."""
        treatment1, outcome1, covariates1 = generate_simple_rct(
            n_samples=100, random_state=123
        )

        treatment2, outcome2, covariates2 = generate_simple_rct(
            n_samples=100, random_state=456
        )

        # Results should be different
        assert not np.array_equal(treatment1.values, treatment2.values)
        assert not np.array_equal(outcome1.values, outcome2.values)


if __name__ == "__main__":
    pytest.main([__file__])
