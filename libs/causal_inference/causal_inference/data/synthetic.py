"""Synthetic data generation for causal inference testing and examples.

This module provides utilities to generate realistic synthetic datasets
for testing causal inference methods and creating educational examples.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from ..core.base import CovariateData, OutcomeData, TreatmentData


class SyntheticDataGenerator:
    """Generator for synthetic causal inference datasets."""

    def __init__(self, random_state: Optional[int] = None):
        """Initialize the synthetic data generator.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def generate_linear_binary_treatment(
        self,
        n_samples: int = 1000,
        n_confounders: int = 5,
        treatment_effect: float = 2.0,
        confounding_strength: float = 1.0,
        noise_std: float = 1.0,
        selection_bias: float = 0.5,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData]:
        """Generate synthetic data with linear relationships and binary treatment.

        Args:
            n_samples: Number of observations to generate
            n_confounders: Number of confounding variables
            treatment_effect: True average treatment effect
            confounding_strength: Strength of confounding relationships
            noise_std: Standard deviation of noise terms
            selection_bias: Strength of treatment selection bias

        Returns:
            Tuple of (treatment, outcome, covariates) data objects
        """
        # Generate confounding variables
        X = np.random.multivariate_normal(
            mean=np.zeros(n_confounders), cov=np.eye(n_confounders), size=n_samples
        )

        # Generate treatment with confounding
        # Treatment probability depends on confounders
        treatment_logits = selection_bias * np.sum(
            X[:, :3], axis=1
        )  # Use first 3 confounders
        treatment_probs = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_probs)

        # Generate outcome with confounding and treatment effect
        confounder_effects = confounding_strength * np.sum(X, axis=1)
        treatment_effects = treatment_effect * treatment
        noise = np.random.normal(0, noise_std, n_samples)

        outcome = confounder_effects + treatment_effects + noise

        # Create data objects
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

        covariate_names = [f"X{i + 1}" for i in range(n_confounders)]
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=covariate_names),
            names=covariate_names,
        )

        return treatment_data, outcome_data, covariate_data

    def generate_nonlinear_continuous_treatment(
        self,
        n_samples: int = 1000,
        n_confounders: int = 4,
        treatment_effect_fn: str = "quadratic",
        noise_std: float = 0.5,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData]:
        """Generate synthetic data with nonlinear relationships and continuous treatment.

        Args:
            n_samples: Number of observations to generate
            n_confounders: Number of confounding variables
            treatment_effect_fn: Type of dose-response function ('linear', 'quadratic', 'threshold')
            noise_std: Standard deviation of noise terms

        Returns:
            Tuple of (treatment, outcome, covariates) data objects
        """
        # Generate confounding variables
        X = np.random.randn(n_samples, n_confounders)

        # Generate continuous treatment with confounding
        treatment_mean = 2.0 + 0.5 * X[:, 0]
        if n_confounders >= 2:
            treatment_mean -= 0.3 * X[:, 1]
        treatment = np.random.normal(treatment_mean, 1.0)

        # Generate outcome with nonlinear treatment effect
        confounder_effects = 1.5 * X[:, 0] + 0.3 * np.sin(X[:, 0])  # Basic effects

        if n_confounders >= 2:
            confounder_effects += 0.8 * X[:, 1]

        if n_confounders >= 4:
            confounder_effects += 0.5 * X[:, 2] * X[:, 3]  # Interaction effect
        elif n_confounders >= 3:
            confounder_effects += 0.5 * X[:, 2]

        # Apply different dose-response functions
        if treatment_effect_fn == "linear":
            treatment_effects = 1.0 * treatment
        elif treatment_effect_fn == "quadratic":
            treatment_effects = 0.5 * treatment + 0.2 * treatment**2
        elif treatment_effect_fn == "threshold":
            treatment_effects = 2.0 * (treatment > 2.0).astype(float)
        else:
            raise ValueError(
                f"Unknown treatment effect function: {treatment_effect_fn}"
            )

        noise = np.random.normal(0, noise_std, n_samples)
        outcome = confounder_effects + treatment_effects + noise

        # Create data objects
        treatment_data = TreatmentData(
            values=pd.Series(treatment),
            name="treatment",
            treatment_type="continuous",
        )

        outcome_data = OutcomeData(
            values=pd.Series(outcome),
            name="outcome",
            outcome_type="continuous",
        )

        covariate_names = [f"X{i + 1}" for i in range(n_confounders)]
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=covariate_names),
            names=covariate_names,
        )

        return treatment_data, outcome_data, covariate_data

    def generate_marketing_campaign_data(
        self,
        n_samples: int = 5000,
        campaign_types: Optional[list[str]] = None,
        include_seasonality: bool = True,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData]:
        """Generate synthetic marketing campaign dataset.

        Args:
            n_samples: Number of observations (customers)
            campaign_types: List of campaign types. If None, uses default marketing campaigns.
            include_seasonality: Whether to include seasonal effects

        Returns:
            Tuple of (treatment, outcome, covariates) data objects representing
            marketing campaign assignment, customer response, and customer attributes
        """
        if campaign_types is None:
            campaign_types = ["email", "social_media", "direct_mail", "control"]

        n_campaigns = len(campaign_types)

        # Generate customer attributes (confounders)
        customer_age = np.random.normal(45, 15, n_samples)
        customer_age = np.clip(customer_age, 18, 80)  # Realistic age range

        customer_income = np.random.lognormal(10.5, 0.5, n_samples)  # Log-normal income
        customer_income = np.clip(customer_income, 20000, 200000)

        previous_purchases = np.random.poisson(
            3, n_samples
        )  # Historical purchase count

        # Customer segment (affects both treatment assignment and outcome)
        segment_probs = np.array([0.3, 0.4, 0.2, 0.1])  # Premium, Standard, Budget, New
        customer_segment = np.random.choice(4, n_samples, p=segment_probs)

        # Email engagement score (affects treatment assignment)
        email_engagement = np.random.beta(
            2, 5, n_samples
        )  # Skewed toward lower engagement

        # Seasonality (if enabled)
        if include_seasonality:
            # Simulate monthly data
            month = np.random.randint(1, 13, n_samples)
            seasonal_effect = np.sin(2 * np.pi * month / 12)  # Seasonal pattern
        else:
            seasonal_effect = np.zeros(n_samples)

        # Treatment assignment (marketing campaign) - not purely random
        # Premium customers more likely to get direct mail, high engagement gets email
        treatment_logits = np.zeros((n_samples, n_campaigns))

        # Email campaign: higher for high engagement customers
        treatment_logits[:, 0] = (
            2.0 * email_engagement
            + 0.3 * (customer_segment == 0)  # Premium customers
            + 0.1 * (customer_age < 40)  # Younger customers
        )

        # Social media: higher for younger customers
        treatment_logits[:, 1] = (
            -0.05 * customer_age
            + 2.0
            + 0.2 * (customer_segment <= 1)  # Premium/Standard
        )

        # Direct mail: higher for older, premium customers
        treatment_logits[:, 2] = (
            0.03 * customer_age
            + 0.5 * (customer_segment == 0)  # Premium
            + 0.01 * customer_income / 1000
        )

        # Control group: baseline probability
        treatment_logits[:, 3] = np.ones(n_samples)

        # Convert to probabilities and sample treatment
        treatment_probs = np.exp(treatment_logits) / np.sum(
            np.exp(treatment_logits), axis=1, keepdims=True
        )
        treatment_idx = np.array(
            [np.random.choice(n_campaigns, p=probs) for probs in treatment_probs]
        )
        treatment = np.array([campaign_types[idx] for idx in treatment_idx])

        # Generate outcome (customer response/purchase amount)
        # Base response depends on customer characteristics
        base_response = (
            0.02 * customer_age
            + 0.00005 * customer_income
            + 2.0 * previous_purchases
            + 5.0 * (customer_segment == 0)  # Premium customers spend more
            + 2.0 * (customer_segment == 1)  # Standard customers
            + 10.0 * seasonal_effect  # Seasonal effects
        )

        # Treatment effects (different for each campaign type)
        treatment_effects = np.zeros(n_samples)

        for i, campaign in enumerate(campaign_types):
            mask = treatment == campaign
            if campaign == "email":
                # Email effect depends on engagement
                treatment_effects[mask] = 15.0 * email_engagement[mask] + 5.0
            elif campaign == "social_media":
                # Social media works better for younger customers
                treatment_effects[mask] = 12.0 - 0.1 * customer_age[mask]
            elif campaign == "direct_mail":
                # Direct mail works better for high-income customers
                treatment_effects[mask] = 8.0 + 0.00003 * customer_income[mask]
            else:  # control
                treatment_effects[mask] = 0.0

        # Add noise and ensure non-negative outcomes
        noise = np.random.normal(0, 5, n_samples)
        outcome = base_response + treatment_effects + noise
        outcome = np.maximum(outcome, 0)  # Purchase amounts can't be negative

        # Create data objects
        treatment_data = TreatmentData(
            values=pd.Series(treatment),
            name="campaign_type",
            treatment_type="categorical",
            categories=list(campaign_types),
        )

        outcome_data = OutcomeData(
            values=pd.Series(outcome),
            name="purchase_amount",
            outcome_type="continuous",
        )

        # Create covariate dataframe
        covariates_df = pd.DataFrame(
            {
                "customer_age": customer_age,
                "customer_income": customer_income,
                "previous_purchases": previous_purchases,
                "customer_segment": customer_segment,
                "email_engagement": email_engagement,
            }
        )

        if include_seasonality:
            covariates_df["month"] = month

        covariate_data = CovariateData(
            values=covariates_df,
            names=list(covariates_df.columns),
        )

        return treatment_data, outcome_data, covariate_data

    def generate_missing_data_scenario(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData,
        missing_mechanism: Literal["MCAR", "MAR", "MNAR"] = "MAR",
        missing_rate: float = 0.1,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData]:
        """Add missing data to existing dataset according to specified mechanism.

        Args:
            treatment: Original treatment data
            outcome: Original outcome data
            covariates: Original covariate data
            missing_mechanism: Type of missingness ('MCAR', 'MAR', 'MNAR')
            missing_rate: Overall proportion of data to make missing

        Returns:
            Tuple of data objects with missing values introduced
        """
        n_samples = len(treatment.values)

        # Create copies of the data
        new_treatment = TreatmentData(
            values=treatment.values.copy(),
            name=treatment.name,
            treatment_type=treatment.treatment_type,
            categories=treatment.categories,
        )

        new_outcome = OutcomeData(
            values=outcome.values.copy(),
            name=outcome.name,
            outcome_type=outcome.outcome_type,
        )

        if isinstance(covariates.values, pd.DataFrame):
            new_cov_values = covariates.values.copy()
        else:
            new_cov_values = pd.DataFrame(
                covariates.values.copy(),
                columns=covariates.names
                or [f"X{i}" for i in range(covariates.values.shape[1])],
            )

        new_covariates = CovariateData(
            values=new_cov_values,
            names=covariates.names,
        )

        if missing_mechanism == "MCAR":
            # Missing Completely at Random - uniform probability
            for col in new_cov_values.columns:
                missing_mask = np.random.random(n_samples) < missing_rate
                new_cov_values.loc[missing_mask, col] = np.nan

        elif missing_mechanism == "MAR":
            # Missing at Random - depends on observed variables
            # Make missingness depend on treatment and other covariates
            if len(new_cov_values.columns) > 1:
                first_covariate = new_cov_values.columns[0]
                # Higher missingness for treated units and high values of first covariate
                if new_treatment.treatment_type == "binary":
                    treatment_numeric = (new_treatment.values == 1).astype(int)
                elif new_treatment.treatment_type == "categorical":
                    # For categorical treatments, encode as 1 for non-control, 0 for control
                    treatment_numeric = (new_treatment.values != "control").astype(int)
                else:  # continuous
                    if isinstance(new_treatment.values, pd.Series):
                        treatment_numeric = pd.to_numeric(
                            new_treatment.values, errors="coerce"
                        ).fillna(0)
                    else:
                        treatment_numeric = pd.to_numeric(
                            pd.Series(new_treatment.values), errors="coerce"
                        ).fillna(0)

                for i, col in enumerate(
                    new_cov_values.columns[1:], 1
                ):  # Skip first covariate
                    missing_logits = (
                        -2.0  # Base probability
                        + 1.0 * treatment_numeric  # Treatment increases missingness
                        + 0.5
                        * np.array(
                            new_cov_values[first_covariate]
                        )  # First covariate affects missingness
                    )
                    missing_probs = 1 / (1 + np.exp(-missing_logits))
                    missing_mask = (
                        np.random.random(n_samples) < missing_probs * missing_rate * 3
                    )  # Scale to get desired rate
                    new_cov_values.loc[missing_mask, col] = np.nan

        elif missing_mechanism == "MNAR":
            # Missing Not at Random - depends on unobserved values
            # Make high values more likely to be missing
            for col in new_cov_values.columns:
                col_values = np.array(new_cov_values[col])
                # Standardize values
                standardized = (col_values - np.mean(col_values)) / np.std(col_values)
                # Higher values more likely to be missing
                missing_logits = -2.0 + 1.0 * standardized
                missing_probs = 1 / (1 + np.exp(-missing_logits))
                missing_mask = (
                    np.random.random(n_samples) < missing_probs * missing_rate * 2
                )
                new_cov_values.loc[missing_mask, col] = np.nan

        new_covariates.values = new_cov_values
        return new_treatment, new_outcome, new_covariates


def generate_simple_rct(
    n_samples: int = 1000,
    treatment_effect: float = 2.0,
    noise_std: float = 1.0,
    random_state: Optional[int] = None,
) -> tuple[TreatmentData, OutcomeData, CovariateData]:
    """Generate simple randomized controlled trial data.

    Args:
        n_samples: Number of observations
        treatment_effect: True average treatment effect
        noise_std: Standard deviation of noise
        random_state: Random seed

    Returns:
        Tuple of (treatment, outcome, covariates) data objects
    """
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Purely random treatment assignment
    treatment = np.random.binomial(1, 0.5, n_samples)

    # Generate some baseline covariates (not confounders in RCT)
    X = np.random.randn(n_samples, 3)

    # Outcome only depends on treatment and noise (no confounding)
    outcome = treatment_effect * treatment + np.random.normal(0, noise_std, n_samples)

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

    covariate_data = CovariateData(
        values=pd.DataFrame(X, columns=["X1", "X2", "X3"]),
        names=["X1", "X2", "X3"],
    )

    return treatment_data, outcome_data, covariate_data


def generate_confounded_observational(
    n_samples: int = 1000,
    treatment_effect: float = 2.0,
    confounding_strength: float = 1.0,
    selection_bias: float = 0.5,
    random_state: Optional[int] = None,
) -> tuple[TreatmentData, OutcomeData, CovariateData]:
    """Generate confounded observational data.

    Args:
        n_samples: Number of observations
        treatment_effect: True average treatment effect
        confounding_strength: Strength of confounding
        selection_bias: Strength of treatment selection bias
        random_state: Random seed

    Returns:
        Tuple of (treatment, outcome, covariates) data objects
    """
    generator = SyntheticDataGenerator(random_state=random_state)
    return generator.generate_linear_binary_treatment(
        n_samples=n_samples,
        treatment_effect=treatment_effect,
        confounding_strength=confounding_strength,
        selection_bias=selection_bias,
    )
