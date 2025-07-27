"""Shared test fixtures for the causal inference library.

This module provides reusable fixtures for testing estimators, data models,
and diagnostic functions across the causal inference library.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.data.synthetic import SyntheticDataGenerator


@pytest.fixture
def random_state():
    """Provide a consistent random state for reproducible tests."""
    return 42


@pytest.fixture
def small_sample_size():
    """Small sample size for quick tests."""
    return 100


@pytest.fixture
def medium_sample_size():
    """Medium sample size for more realistic tests."""
    return 500


@pytest.fixture
def large_sample_size():
    """Large sample size for performance and robustness tests."""
    return 1000


@pytest.fixture
def synthetic_data_generator(random_state):
    """Provide a configured synthetic data generator."""
    return SyntheticDataGenerator(random_state=random_state)


@pytest.fixture
def simple_binary_data(small_sample_size, random_state):
    """Generate simple binary treatment and continuous outcome data."""
    np.random.seed(random_state)

    # Generate confounders
    X1 = np.random.normal(0, 1, small_sample_size)
    X2 = np.random.normal(0, 1, small_sample_size)

    # Generate treatment (binary)
    propensity = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2)))
    treatment = np.random.binomial(1, propensity, small_sample_size)

    # Generate outcome (continuous with treatment effect = 2)
    outcome = (
        1
        + 0.5 * X1
        + 0.3 * X2
        + 2 * treatment
        + np.random.normal(0, 0.5, small_sample_size)
    )

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2})),
        "true_ate": 2.0,
    }


@pytest.fixture
def simple_categorical_data(small_sample_size, random_state):
    """Generate categorical treatment data."""
    np.random.seed(random_state)

    # Generate confounders
    X1 = np.random.normal(0, 1, small_sample_size)
    X2 = np.random.normal(0, 1, small_sample_size)

    # Generate treatment (categorical: 0, 1, 2)
    linear_pred = 0.5 * X1 + 0.3 * X2
    probs = np.column_stack(
        [
            1 / (1 + np.exp(linear_pred) + np.exp(2 * linear_pred)),
            np.exp(linear_pred) / (1 + np.exp(linear_pred) + np.exp(2 * linear_pred)),
            np.exp(2 * linear_pred)
            / (1 + np.exp(linear_pred) + np.exp(2 * linear_pred)),
        ]
    )
    treatment = np.array([np.random.choice(3, p=p) for p in probs])

    # Generate outcome (treatment effects: 0, 1.5, 3.0)
    treatment_effects = np.array([0, 1.5, 3.0])
    outcome = (
        1
        + 0.5 * X1
        + 0.3 * X2
        + treatment_effects[treatment]
        + np.random.normal(0, 0.5, small_sample_size)
    )

    return {
        "treatment": TreatmentData(
            values=treatment, treatment_type="categorical", categories=[0, 1, 2]
        ),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2})),
        "true_ate_1_vs_0": 1.5,
        "true_ate_2_vs_0": 3.0,
    }


@pytest.fixture
def simple_continuous_treatment_data(small_sample_size, random_state):
    """Generate continuous treatment data."""
    np.random.seed(random_state)

    # Generate confounders
    X1 = np.random.normal(0, 1, small_sample_size)
    X2 = np.random.normal(0, 1, small_sample_size)

    # Generate continuous treatment
    treatment = 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, small_sample_size)

    # Generate outcome (linear dose-response with coefficient 1.2)
    outcome = (
        1
        + 0.5 * X1
        + 0.3 * X2
        + 1.2 * treatment
        + np.random.normal(0, 0.5, small_sample_size)
    )

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="continuous"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2})),
        "true_dose_response_coef": 1.2,
    }


@pytest.fixture
def binary_outcome_data(small_sample_size, random_state):
    """Generate binary outcome data."""
    np.random.seed(random_state)

    # Generate confounders
    X1 = np.random.normal(0, 1, small_sample_size)
    X2 = np.random.normal(0, 1, small_sample_size)

    # Generate binary treatment
    propensity = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2)))
    treatment = np.random.binomial(1, propensity, small_sample_size)

    # Generate binary outcome (logistic model with treatment effect = 0.8)
    linear_pred = -0.5 + 0.5 * X1 + 0.3 * X2 + 0.8 * treatment
    prob_outcome = 1 / (1 + np.exp(-linear_pred))
    outcome = np.random.binomial(1, prob_outcome, small_sample_size)

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="binary"),
        "covariates": CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2})),
        "true_log_odds_ratio": 0.8,
    }


@pytest.fixture
def sklearn_classification_data(medium_sample_size, random_state):
    """Generate sklearn-compatible classification data."""
    X, y = make_classification(
        n_samples=medium_sample_size,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    # Use first feature as treatment
    treatment = (X[:, 0] > np.median(X[:, 0])).astype(int)

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=y, outcome_type="binary"),
        "covariates": CovariateData(
            values=pd.DataFrame(X[:, 1:], columns=[f"X{i}" for i in range(1, 5)])
        ),
    }


@pytest.fixture
def sklearn_regression_data(medium_sample_size, random_state):
    """Generate sklearn-compatible regression data."""
    X, y = make_regression(
        n_samples=medium_sample_size,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=random_state,
    )

    # Use first feature as continuous treatment
    treatment = X[:, 0]

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="continuous"),
        "outcome": OutcomeData(values=y, outcome_type="continuous"),
        "covariates": CovariateData(
            values=pd.DataFrame(X[:, 1:], columns=[f"X{i}" for i in range(1, 5)])
        ),
    }


@pytest.fixture
def confounded_data(medium_sample_size, random_state):
    """Generate data with strong confounding."""
    np.random.seed(random_state)

    # Strong confounders
    age = np.random.uniform(18, 80, medium_sample_size)
    income = np.random.exponential(50000, medium_sample_size)

    # Treatment strongly depends on confounders
    propensity = 1 / (1 + np.exp(-(0.02 * (age - 40) + 0.00002 * (income - 50000))))
    treatment = np.random.binomial(1, propensity, medium_sample_size)

    # Outcome strongly depends on confounders and treatment
    outcome = (
        1000
        + 20 * age
        + 0.1 * income
        + 5000 * treatment
        + np.random.normal(0, 1000, medium_sample_size)
    )

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(
            values=pd.DataFrame({"age": age, "income": income})
        ),
        "true_ate": 5000.0,
    }


@pytest.fixture
def missing_data_scenario(simple_binary_data, random_state):
    """Create data with missing values for testing."""
    np.random.seed(random_state)
    data = simple_binary_data.copy()

    # Introduce 20% missing values in covariates
    n_samples = len(data["treatment"].values)
    missing_mask = np.random.random(n_samples) < 0.2

    covariates_df = data["covariates"].values.copy()
    covariates_df.loc[missing_mask, "X1"] = np.nan

    data["covariates"] = CovariateData(values=covariates_df)
    return data


@pytest.fixture
def edge_case_data(random_state):
    """Generate edge case data for robustness testing."""
    np.random.seed(random_state)

    scenarios = {
        "no_treatment_variation": {
            "treatment": TreatmentData(values=np.ones(100), treatment_type="binary"),
            "outcome": OutcomeData(
                values=np.random.normal(0, 1, 100), outcome_type="continuous"
            ),
            "covariates": CovariateData(
                values=pd.DataFrame({"X1": np.random.normal(0, 1, 100)})
            ),
        },
        "extreme_propensity": {
            "treatment": TreatmentData(
                values=np.concatenate([np.zeros(95), np.ones(5)]),
                treatment_type="binary",
            ),
            "outcome": OutcomeData(
                values=np.random.normal(0, 1, 100), outcome_type="continuous"
            ),
            "covariates": CovariateData(
                values=pd.DataFrame({"X1": np.random.normal(0, 1, 100)})
            ),
        },
        "small_sample": {
            # Use minimum sample size of 12 (just above the 10 minimum requirement)
            "treatment": TreatmentData(
                values=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                treatment_type="binary",
            ),
            "outcome": OutcomeData(
                values=np.array([1, 3, 2, 4, 1, 5, 2, 6, 3, 7, 4, 8]),
                outcome_type="continuous",
            ),
            "covariates": CovariateData(
                values=pd.DataFrame({"X1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
            ),
        },
        "perfect_separation": {
            # Perfect separation: treatment perfectly predicted by covariates
            "treatment": TreatmentData(
                values=np.concatenate([np.zeros(50), np.ones(50)]),
                treatment_type="binary",
            ),
            "outcome": OutcomeData(
                values=np.random.normal(0, 1, 100), outcome_type="continuous"
            ),
            "covariates": CovariateData(
                values=pd.DataFrame(
                    {
                        "X1": np.concatenate(
                            [
                                np.random.normal(-3, 0.1, 50),
                                np.random.normal(3, 0.1, 50),
                            ]
                        ),  # Perfect separation
                        "X2": np.random.normal(0, 1, 100),  # Normal covariate
                    }
                )
            ),
        },
        "multicollinearity": {
            # Highly correlated covariates creating multicollinearity
            "treatment": TreatmentData(
                values=np.random.binomial(1, 0.5, 100), treatment_type="binary"
            ),
            "outcome": OutcomeData(
                values=np.random.normal(0, 1, 100), outcome_type="continuous"
            ),
            "covariates": CovariateData(
                values=pd.DataFrame(
                    {
                        "X1": np.random.normal(0, 1, 100),
                        "X2": lambda df: df["X1"]
                        + np.random.normal(0, 0.01, 100),  # Almost perfectly correlated
                        "X3": lambda df: 2 * df["X1"]
                        + np.random.normal(0, 0.01, 100),  # Linear combination
                        "X4": np.random.normal(0, 1, 100),  # Independent
                    }
                ).assign(
                    X2=lambda df: df["X1"] + np.random.normal(0, 0.01, 100),
                    X3=lambda df: 2 * df["X1"] + np.random.normal(0, 0.01, 100),
                )
            ),
        },
        "rank_deficient": {
            # Rank deficient covariate matrix
            "treatment": TreatmentData(
                values=np.random.binomial(1, 0.5, 50), treatment_type="binary"
            ),
            "outcome": OutcomeData(
                values=np.random.normal(0, 1, 50), outcome_type="continuous"
            ),
            "covariates": CovariateData(
                values=pd.DataFrame(
                    {
                        "X1": np.ones(50),  # Constant column
                        "X2": np.random.normal(0, 1, 50),
                        "X3": np.random.normal(0, 1, 50),
                        "X4": lambda df: df["X2"] + df["X3"],  # Linear combination
                    }
                ).assign(X4=lambda df: df["X2"] + df["X3"])
            ),
        },
    }

    return scenarios


@pytest.fixture
def model_combinations():
    """Provide various sklearn model combinations for testing."""
    return {
        "linear": {
            "outcome_model": LinearRegression(),
            "propensity_model": LogisticRegression(random_state=42),
        },
        "random_forest": {
            "outcome_model": RandomForestRegressor(n_estimators=10, random_state=42),
            "propensity_model": RandomForestClassifier(
                n_estimators=10, random_state=42
            ),
        },
        "mixed": {
            "outcome_model": RandomForestRegressor(n_estimators=10, random_state=42),
            "propensity_model": LogisticRegression(random_state=42),
        },
    }


@pytest.fixture
def marketing_campaign_data(medium_sample_size, random_state):
    """Generate realistic marketing campaign data."""
    np.random.seed(random_state)

    # Customer demographics
    age = np.random.uniform(18, 70, medium_sample_size)
    income = np.random.lognormal(10, 0.5, medium_sample_size)
    previous_purchases = np.random.poisson(2, medium_sample_size)

    # Treatment assignment (email campaign)
    propensity = 1 / (
        1 + np.exp(-(-2 + 0.02 * age + 0.00001 * income + 0.1 * previous_purchases))
    )
    treatment = np.random.binomial(1, propensity, medium_sample_size)

    # Outcome (purchase amount)
    outcome = (
        100
        + 2 * age
        + 0.01 * income
        + 20 * previous_purchases
        + 50 * treatment
        + np.random.normal(0, 30, medium_sample_size)
    )
    outcome = np.maximum(outcome, 0)  # Non-negative purchases

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(
            values=pd.DataFrame(
                {"age": age, "income": income, "previous_purchases": previous_purchases}
            )
        ),
        "true_ate": 50.0,
        "context": "marketing_campaign",
    }


@pytest.fixture
def performance_benchmark_data(large_sample_size, random_state):
    """Generate large dataset for performance benchmarking."""
    np.random.seed(random_state)

    # Optimized: Reduce covariates for memory efficiency while maintaining complexity
    n_covariates = min(15, large_sample_size // 50)  # Scale with sample size, max 15
    n_samples = min(large_sample_size, 800)  # Cap at 800 for memory efficiency
    X = np.random.normal(0, 1, (n_samples, n_covariates))

    # Treatment depends on subset of covariates
    propensity_weights = np.random.normal(0, 0.1, n_covariates)
    propensity_weights[: min(5, n_covariates)] = np.random.normal(
        0, 0.5, min(5, n_covariates)
    )  # First few are important

    linear_pred = X @ propensity_weights
    propensity = 1 / (1 + np.exp(-linear_pred))
    treatment = np.random.binomial(1, propensity, n_samples)

    # Outcome depends on different subset of covariates
    outcome_weights = np.random.normal(0, 0.1, n_covariates)
    outcome_weights[5 : min(10, n_covariates)] = np.random.normal(
        0, 0.5, min(5, n_covariates - 5)
    )  # Adapt to n_covariates

    outcome = X @ outcome_weights + 2 * treatment + np.random.normal(0, 1, n_samples)

    covariate_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(n_covariates)])

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=covariate_df),
        "true_ate": 2.0,
        "context": "performance_benchmark",
    }
