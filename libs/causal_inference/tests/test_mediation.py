"""Tests for mediation analysis estimator."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.core.bootstrap import BootstrapConfig
from causal_inference.estimators.mediation import (
    MediationEffect,
    MediationEstimator,
    MediatorData,
)


class TestMediatorData:
    """Test the MediatorData class."""

    def test_mediator_data_creation(self):
        """Test basic MediatorData creation."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mediator = MediatorData(values=values, name="test_mediator")

        assert mediator.name == "test_mediator"
        assert mediator.mediator_type == "continuous"
        assert len(mediator.values) == 5

    def test_mediator_data_validation(self):
        """Test MediatorData validation."""
        # Empty values should raise error
        with pytest.raises(ValueError, match="Mediator values cannot be empty"):
            MediatorData(values=pd.Series([]))


class TestMediationEffect:
    """Test the MediationEffect class."""

    def test_mediation_effect_creation(self):
        """Test basic MediationEffect creation."""
        effect = MediationEffect(
            ate=0.5,
            nde=0.3,
            nie=0.2,
            mediated_proportion=0.4,
            method="mediation_analysis",
        )

        assert effect.ate == 0.5
        assert effect.nde == 0.3
        assert effect.nie == 0.2
        assert effect.mediated_proportion == 0.4
        assert effect.method == "mediation_analysis"

    def test_is_mediated_property(self):
        """Test the is_mediated property."""
        # Case with significant mediation
        effect = MediationEffect(
            ate=0.5, nde=0.3, nie=0.2, nie_ci_lower=0.1, nie_ci_upper=0.3
        )
        assert effect.is_mediated is True

        # Case without significant mediation (CI contains zero)
        effect = MediationEffect(
            ate=0.5, nde=0.3, nie=0.2, nie_ci_lower=-0.1, nie_ci_upper=0.3
        )
        assert effect.is_mediated is False

    def test_mediation_effect_validation(self):
        """Test MediationEffect validation warns for inconsistent values."""
        # This should print a warning but not raise an error
        effect = MediationEffect(
            ate=0.5,
            nde=0.2,
            nie=0.2,  # NDE + NIE = 0.4, but ATE = 0.5
            method="mediation_analysis",
        )
        assert effect.ate == 0.5


class TestMediationEstimator:
    """Test the MediationEstimator class."""

    @pytest.fixture
    def synthetic_mediation_data(self):
        """Create synthetic data for mediation analysis testing."""
        np.random.seed(42)
        n_samples = 500

        # Generate confounders
        X = np.random.normal(0, 1, (n_samples, 3))

        # Generate treatment (binary)
        treatment_logits = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
        treatment_probs = 1 / (1 + np.exp(-treatment_logits))
        T = np.random.binomial(1, treatment_probs)

        # Generate mediator (affected by treatment and confounders)
        mediator_noise = np.random.normal(0, 0.5, n_samples)
        M = 0.8 * T + 0.4 * X[:, 0] + 0.2 * X[:, 1] + mediator_noise

        # Generate outcome (affected by treatment, mediator, and confounders)
        outcome_noise = np.random.normal(0, 0.5, n_samples)
        Y = (
            0.3 * T  # Direct effect
            + 0.5 * M  # Mediated effect
            + 0.2 * X[:, 0]
            + 0.1 * X[:, 1]
            - 0.15 * X[:, 2]
            + outcome_noise
        )

        # Convert to data objects
        treatment_data = TreatmentData(values=pd.Series(T), treatment_type="binary")
        outcome_data = OutcomeData(values=pd.Series(Y), outcome_type="continuous")
        mediator_data = MediatorData(values=pd.Series(M), mediator_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=["X1", "X2", "X3"]), names=["X1", "X2", "X3"]
        )

        # True causal effects (approximate)
        # Direct effect: 0.3
        # Indirect effect: 0.8 * 0.5 = 0.4
        # Total effect: 0.3 + 0.4 = 0.7
        # Mediated proportion: 0.4 / 0.7 ≈ 0.57

        return {
            "treatment": treatment_data,
            "outcome": outcome_data,
            "mediator": mediator_data,
            "covariates": covariate_data,
            "true_nde": 0.3,
            "true_nie": 0.4,
            "true_ate": 0.7,
            "true_mediated_prop": 0.57,
        }

    def test_estimator_initialization(self):
        """Test MediationEstimator initialization."""
        estimator = MediationEstimator(
            mediator_model_type="linear",
            outcome_model_type="linear",
            bootstrap_samples=100,
            random_state=42,
        )

        assert estimator.mediator_model_type == "linear"
        assert estimator.outcome_model_type == "linear"
        assert estimator.bootstrap_config.n_samples == 100
        assert estimator.random_state == 42

    def test_model_creation(self, synthetic_mediation_data):
        """Test internal model creation."""
        estimator = MediationEstimator(random_state=42)

        # Test linear model creation
        model = estimator._create_model("linear", "continuous", {})
        assert model.__class__.__name__ == "LinearRegression"

        # Test logistic model creation
        model = estimator._create_model("logistic", "binary", {})
        assert model.__class__.__name__ == "LogisticRegression"

        # Test auto model selection
        model = estimator._create_model("auto", "continuous", {})
        assert model.__class__.__name__ == "LinearRegression"

        model = estimator._create_model("auto", "binary", {})
        assert model.__class__.__name__ == "LogisticRegression"

    def test_fitting_without_covariates(self, synthetic_mediation_data):
        """Test fitting the estimator without covariates."""
        data = synthetic_mediation_data

        estimator = MediationEstimator(
            bootstrap_samples=0,  # Skip bootstrap for speed
            random_state=42,
        )

        fitted_estimator = estimator.fit(
            treatment=data["treatment"],
            outcome=data["outcome"],
            mediator=data["mediator"],
        )

        assert fitted_estimator.is_fitted
        assert fitted_estimator.mediator_model is not None
        assert fitted_estimator.outcome_model is not None

    def test_fitting_with_covariates(self, synthetic_mediation_data):
        """Test fitting the estimator with covariates."""
        data = synthetic_mediation_data

        estimator = MediationEstimator(
            bootstrap_samples=0,  # Skip bootstrap for speed
            random_state=42,
        )

        fitted_estimator = estimator.fit(
            treatment=data["treatment"],
            outcome=data["outcome"],
            mediator=data["mediator"],
            covariates=data["covariates"],
        )

        assert fitted_estimator.is_fitted
        assert fitted_estimator.mediator_model is not None
        assert fitted_estimator.outcome_model is not None

    def test_mediation_effect_estimation(self, synthetic_mediation_data):
        """Test mediation effect estimation."""
        data = synthetic_mediation_data

        estimator = MediationEstimator(
            bootstrap_samples=0,  # Skip bootstrap for speed
            random_state=42,
        )

        estimator.fit(
            treatment=data["treatment"],
            outcome=data["outcome"],
            mediator=data["mediator"],
            covariates=data["covariates"],
        )

        effect = estimator.estimate_ate()

        # Check that effect object is created
        assert isinstance(effect, MediationEffect)
        assert effect.ate is not None
        assert effect.nde is not None
        assert effect.nie is not None
        assert effect.mediated_proportion is not None

        # Check that effects are reasonable (within broad bounds)
        assert abs(effect.nde - data["true_nde"]) < 0.5  # Allow some estimation error
        assert abs(effect.nie - data["true_nie"]) < 0.5
        assert abs(effect.ate - data["true_ate"]) < 0.5

        # Check that NDE + NIE approximately equals ATE
        assert abs((effect.nde + effect.nie) - effect.ate) < 1e-10

    def test_bootstrap_confidence_intervals(self, synthetic_mediation_data):
        """Test bootstrap confidence interval estimation."""
        data = synthetic_mediation_data

        bootstrap_config = BootstrapConfig(
            n_samples=50,  # Small number for speed
            confidence_level=0.95,
            random_state=42,
        )

        estimator = MediationEstimator(
            bootstrap_config=bootstrap_config, random_state=42
        )

        estimator.fit(
            treatment=data["treatment"],
            outcome=data["outcome"],
            mediator=data["mediator"],
            covariates=data["covariates"],
        )

        effect = estimator.estimate_ate()

        # Check that confidence intervals are present
        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None
        assert effect.nde_ci_lower is not None
        assert effect.nde_ci_upper is not None
        assert effect.nie_ci_lower is not None
        assert effect.nie_ci_upper is not None
        assert effect.mediated_prop_ci_lower is not None
        assert effect.mediated_prop_ci_upper is not None

        # Check that confidence intervals make sense
        assert effect.ate_ci_lower < effect.ate_ci_upper
        assert effect.nde_ci_lower < effect.nde_ci_upper
        assert effect.nie_ci_lower < effect.nie_ci_upper

        # Check that standard errors are positive
        assert effect.ate_se > 0
        assert effect.nde_se > 0
        assert effect.nie_se > 0

    def test_different_model_types(self, synthetic_mediation_data):
        """Test different model types for mediator and outcome."""
        data = synthetic_mediation_data

        # Test random forest models
        estimator = MediationEstimator(
            mediator_model_type="random_forest",
            outcome_model_type="random_forest",
            mediator_model_params={"n_estimators": 10, "max_depth": 3},
            outcome_model_params={"n_estimators": 10, "max_depth": 3},
            bootstrap_samples=0,
            random_state=42,
        )

        estimator.fit(
            treatment=data["treatment"],
            outcome=data["outcome"],
            mediator=data["mediator"],
            covariates=data["covariates"],
        )

        effect = estimator.estimate_ate()

        assert isinstance(effect, MediationEffect)
        assert effect.ate is not None
        assert effect.nde is not None
        assert effect.nie is not None

    def test_error_handling(self, synthetic_mediation_data):
        """Test error handling for common mistakes."""
        data = synthetic_mediation_data

        estimator = MediationEstimator(random_state=42)

        # Test estimation before fitting
        with pytest.raises(Exception):
            estimator.estimate_ate()

        # Test fitting without mediator
        with pytest.raises(Exception):
            estimator.fit(treatment=data["treatment"], outcome=data["outcome"])

    def test_nhefs_integration(self):
        """Test integration with NHEFS data loader."""
        from causal_inference.data.nhefs import load_nhefs

        try:
            # Try to load NHEFS data
            treatment_data, outcome_data, covariate_data = load_nhefs(
                treatment="qsmk",
                outcome="wt82_71",
                confounders=[
                    "sex",
                    "age",
                    "race",
                    "education",
                    "smokeintensity",
                    "smokeyrs",
                    "exercise",
                    "active",
                    "wt71",
                ],
            )

            # Extract mediator from covariates (smokeintensity)
            mediator_values = None
            if "smokeintensity" in covariate_data.names:
                mediator_idx = covariate_data.names.index("smokeintensity")
                if isinstance(covariate_data.values, pd.DataFrame):
                    mediator_values = covariate_data.values.iloc[:, mediator_idx]
                else:
                    mediator_values = covariate_data.values[:, mediator_idx]

                # Remove mediator from covariates to avoid confounding
                remaining_covariates = [
                    name for name in covariate_data.names if name != "smokeintensity"
                ]
                if remaining_covariates:
                    remaining_indices = [
                        i
                        for i, name in enumerate(covariate_data.names)
                        if name != "smokeintensity"
                    ]
                    if isinstance(covariate_data.values, pd.DataFrame):
                        covariate_values = covariate_data.values.iloc[
                            :, remaining_indices
                        ]
                    else:
                        covariate_values = covariate_data.values[:, remaining_indices]

                    covariate_data = CovariateData(
                        values=covariate_values, names=remaining_covariates
                    )
                else:
                    covariate_data = None

            if mediator_values is not None:
                mediator_data = MediatorData(
                    values=mediator_values,
                    name="smokeintensity",
                    mediator_type="continuous",
                )

                # Test with a small sample for speed
                n_sample = min(200, len(treatment_data.values))
                sample_indices = np.random.choice(
                    len(treatment_data.values), n_sample, replace=False
                )

                # Create sample data
                sample_treatment = TreatmentData(
                    values=treatment_data.values.iloc[sample_indices],
                    name=treatment_data.name,
                    treatment_type=treatment_data.treatment_type,
                )
                sample_outcome = OutcomeData(
                    values=outcome_data.values.iloc[sample_indices],
                    name=outcome_data.name,
                    outcome_type=outcome_data.outcome_type,
                )
                sample_mediator = MediatorData(
                    values=mediator_data.values.iloc[sample_indices],
                    name=mediator_data.name,
                    mediator_type=mediator_data.mediator_type,
                )
                sample_covariates = None
                if covariate_data is not None:
                    sample_covariates = CovariateData(
                        values=covariate_data.values.iloc[sample_indices]
                        if isinstance(covariate_data.values, pd.DataFrame)
                        else covariate_data.values[sample_indices],
                        names=covariate_data.names,
                    )

                # Fit mediation estimator
                estimator = MediationEstimator(
                    bootstrap_samples=0,  # Skip bootstrap for speed
                    random_state=42,
                )

                estimator.fit(
                    treatment=sample_treatment,
                    outcome=sample_outcome,
                    mediator=sample_mediator,
                    covariates=sample_covariates,
                )

                effect = estimator.estimate_ate()

                # Check that we get reasonable results
                assert isinstance(effect, MediationEffect)
                assert effect.ate is not None
                assert effect.nde is not None
                assert effect.nie is not None
                assert effect.mediated_proportion is not None

                # Results should be finite
                assert np.isfinite(effect.ate)
                assert np.isfinite(effect.nde)
                assert np.isfinite(effect.nie)
                assert np.isfinite(effect.mediated_proportion)

        except FileNotFoundError:
            # NHEFS data not available in test environment
            pytest.skip("NHEFS data not available for integration test")

    def test_feature_matrix_preparation(self, synthetic_mediation_data):
        """Test internal feature matrix preparation."""
        data = synthetic_mediation_data

        estimator = MediationEstimator(random_state=42)
        estimator.mediator_data = data["mediator"]

        # Test feature matrix preparation with covariates
        X_mediator, X_outcome = estimator._prepare_feature_matrices(
            data["treatment"], data["covariates"]
        )

        # Check dimensions
        n_obs = len(data["treatment"].values)
        n_covariates = len(data["covariates"].names)

        assert X_mediator.shape == (n_obs, 1 + n_covariates)  # [T, X]
        assert X_outcome.shape == (n_obs, 1 + 1 + n_covariates)  # [T, M, X]

        # Test feature matrix preparation without covariates
        X_mediator_no_cov, X_outcome_no_cov = estimator._prepare_feature_matrices(
            data["treatment"], None
        )

        assert X_mediator_no_cov.shape == (n_obs, 1)  # [T]
        assert X_outcome_no_cov.shape == (n_obs, 2)  # [T, M]
