"""Tests for Targeted Maximum Transported Likelihood (TMTL) estimator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causal_inference.core.base import (
    CausalEffect,
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.transportability.tmtl import TargetedMaximumTransportedLikelihood


class TestTargetedMaximumTransportedLikelihood:
    """Test suite for TMTL estimator."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample causal inference data."""
        np.random.seed(42)
        n = 1000

        # Generate covariates
        X = np.random.randn(n, 4)

        # Generate treatment with some dependence on covariates
        treatment_probs = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1])))
        T = np.random.binomial(1, treatment_probs)

        # Generate outcome with treatment effect
        true_ate = 2.0
        noise = np.random.randn(n) * 0.5
        Y = 1.0 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + true_ate * T + noise

        return X, T, Y, true_ate

    @pytest.fixture
    def target_data(self):
        """Generate target population data with covariate shift."""
        np.random.seed(123)
        n_target = 500

        # Target population with shifted covariates
        X_target = np.random.randn(n_target, 4)
        X_target[:, 0] += 0.5  # Mean shift in first covariate
        X_target[:, 1] *= 1.2  # Scale shift in second covariate

        return X_target

    def test_initialization_default(self):
        """Test TMTL initialization with default parameters."""
        tmtl = TargetedMaximumTransportedLikelihood()

        assert tmtl.transport_weighting_method == "classification"
        assert tmtl.max_transport_iterations == 100
        assert tmtl.transport_tolerance == 1e-6
        assert tmtl.trim_weights is True
        assert tmtl.max_weight == 10.0
        assert tmtl.cross_fit is True
        assert tmtl.n_folds == 5
        assert tmtl.is_fitted is False

    def test_initialization_custom(self):
        """Test TMTL initialization with custom parameters."""
        outcome_model = RandomForestRegressor(n_estimators=50, random_state=42)
        treatment_model = LogisticRegression(random_state=42)

        tmtl = TargetedMaximumTransportedLikelihood(
            outcome_model=outcome_model,
            treatment_model=treatment_model,
            transport_weighting_method="optimal_transport",
            max_transport_iterations=50,
            transport_tolerance=1e-4,
            trim_weights=False,
            max_weight=5.0,
            cross_fit=False,
            n_folds=3,
            random_state=123,
            verbose=True,
        )

        assert tmtl.outcome_model == outcome_model
        assert tmtl.treatment_model == treatment_model
        assert tmtl.transport_weighting_method == "optimal_transport"
        assert tmtl.max_transport_iterations == 50
        assert tmtl.transport_tolerance == 1e-4
        assert tmtl.trim_weights is False
        assert tmtl.max_weight == 5.0
        assert tmtl.cross_fit is False
        assert tmtl.n_folds == 3
        assert tmtl.random_state == 123
        assert tmtl.verbose is True

    def test_fit_basic(self, sample_data):
        """Test basic fitting functionality."""
        X, T, Y, _ = sample_data

        # Create data objects
        treatment_data = TreatmentData(values=T, name="treatment")
        outcome_data = OutcomeData(values=Y, name="outcome")
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)

        # Fit the estimator
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        assert tmtl.is_fitted is True
        assert tmtl.source_treatment == treatment_data
        assert tmtl.source_outcome == outcome_data
        assert tmtl.source_covariates == covariate_data
        assert tmtl.initial_estimates is not None

    def test_fit_without_covariates_error(self, sample_data):
        """Test that fitting without covariates raises error."""
        X, T, Y, _ = sample_data

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)

        tmtl = TargetedMaximumTransportedLikelihood()

        with pytest.raises(ValueError, match="TMTL requires covariates"):
            tmtl.fit(treatment_data, outcome_data, covariates=None)

    def test_estimate_transported_ate_basic(self, sample_data, target_data):
        """Test basic transported ATE estimation."""
        X, T, Y, true_ate = sample_data
        X_target = target_data

        # Create data objects
        treatment_data = TreatmentData(values=T, name="treatment")
        outcome_data = OutcomeData(values=Y, name="outcome")
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        # Estimate transported ATE
        transported_effect = tmtl.estimate_transported_ate(X_target)

        assert isinstance(transported_effect, CausalEffect)
        assert transported_effect.method == "TMTL"
        assert transported_effect.n_observations == len(T)
        assert transported_effect.ate is not None
        assert transported_effect.diagnostics is not None

        # ATE should be reasonably close to true value (within broad range due to transport)
        assert (
            abs(transported_effect.ate - true_ate) < 5.0
        )  # Broad tolerance for transported estimate

    def test_estimate_transported_ate_not_fitted_error(self, target_data):
        """Test error when estimating transported ATE before fitting."""
        X_target = target_data

        tmtl = TargetedMaximumTransportedLikelihood()

        with pytest.raises(
            ValueError, match="must be fitted before transported estimation"
        ):
            tmtl.estimate_transported_ate(X_target)

    def test_cross_fitting_vs_no_cross_fitting(self, sample_data, target_data):
        """Test cross-fitting vs no cross-fitting."""
        X, T, Y, _ = sample_data
        X_target = target_data

        # Create data objects
        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        # With cross-fitting
        tmtl_cv = TargetedMaximumTransportedLikelihood(
            cross_fit=True, n_folds=3, random_state=42, verbose=False
        )
        tmtl_cv.fit(treatment_data, outcome_data, covariate_data)
        effect_cv = tmtl_cv.estimate_transported_ate(X_target)

        # Without cross-fitting
        tmtl_no_cv = TargetedMaximumTransportedLikelihood(
            cross_fit=False, random_state=42, verbose=False
        )
        tmtl_no_cv.fit(treatment_data, outcome_data, covariate_data)
        effect_no_cv = tmtl_no_cv.estimate_transported_ate(X_target)

        # Both should produce valid results
        assert isinstance(effect_cv, CausalEffect)
        assert isinstance(effect_no_cv, CausalEffect)

        # Results may differ but should be in similar range
        assert abs(effect_cv.ate - effect_no_cv.ate) < 3.0  # Reasonable difference

    def test_transport_weight_computation(self, sample_data, target_data):
        """Test transport weight computation."""
        X, T, Y, _ = sample_data
        X_target = target_data

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        # Estimate transported effect
        tmtl.estimate_transported_ate(X_target)

        # Check that transport weights were computed
        assert tmtl.transport_weights is not None
        assert tmtl.transport_weighting_result is not None

        # Check transport weight properties
        assert len(tmtl.transport_weights) == len(X)
        assert np.all(tmtl.transport_weights > 0)

        # Check weighting result
        assert tmtl.transport_weighting_result.effective_sample_size > 0
        assert tmtl.transport_weighting_result.max_weight > 0

    def test_targeting_convergence(self, sample_data, target_data):
        """Test targeting step convergence."""
        X, T, Y, _ = sample_data
        X_target = target_data

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(
            max_transport_iterations=20,
            transport_tolerance=1e-4,
            random_state=42,
            verbose=False,
        )
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        # Estimate transported effect
        tmtl.estimate_transported_ate(X_target)

        # Check targeting estimates
        assert tmtl.targeted_estimates is not None
        assert "convergence_achieved" in tmtl.targeted_estimates
        assert "n_iterations" in tmtl.targeted_estimates

        # Should converge within reasonable iterations
        assert tmtl.targeted_estimates["n_iterations"] <= 20

    def test_diagnostics_information(self, sample_data, target_data):
        """Test diagnostic information in results."""
        X, T, Y, _ = sample_data
        X_target = target_data

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        effect = tmtl.estimate_transported_ate(X_target)

        # Check diagnostic information
        assert effect.diagnostics is not None

        expected_keys = [
            "transport_effective_sample_size",
            "transport_max_weight",
            "transport_weight_stability",
            "transport_convergence",
            "targeting_convergence",
            "targeting_iterations",
        ]

        for key in expected_keys:
            assert key in effect.diagnostics

    def test_standard_ate_estimation_warning(self, sample_data):
        """Test warning for standard ATE estimation."""
        X, T, Y, _ = sample_data

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        # Test standard ATE estimation (should warn)
        with pytest.warns(UserWarning, match="without transport"):
            effect = tmtl.estimate_ate()

        assert isinstance(effect, CausalEffect)
        assert effect.method == "TMTL_no_transport"

    def test_weight_trimming_effect(self, sample_data, target_data):
        """Test effect of weight trimming."""
        X, T, Y, _ = sample_data
        X_target = target_data

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        # With weight trimming
        tmtl_trim = TargetedMaximumTransportedLikelihood(
            trim_weights=True, max_weight=5.0, random_state=42, verbose=False
        )
        tmtl_trim.fit(treatment_data, outcome_data, covariate_data)
        effect_trim = tmtl_trim.estimate_transported_ate(X_target)

        # Without weight trimming
        tmtl_no_trim = TargetedMaximumTransportedLikelihood(
            trim_weights=False, random_state=42, verbose=False
        )
        tmtl_no_trim.fit(treatment_data, outcome_data, covariate_data)
        effect_no_trim = tmtl_no_trim.estimate_transported_ate(X_target)

        # With trimming should have more stable weights
        max_weight_trim = tmtl_trim.transport_weighting_result.max_weight
        max_weight_no_trim = tmtl_no_trim.transport_weighting_result.max_weight

        assert max_weight_trim <= max_weight_no_trim

        # Both should produce reasonable estimates
        assert isinstance(effect_trim, CausalEffect)
        assert isinstance(effect_no_trim, CausalEffect)

    def test_identical_populations(self, sample_data):
        """Test behavior when source and target populations are identical."""
        X, T, Y, true_ate = sample_data

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        # Use same population as target (no shift)
        effect = tmtl.estimate_transported_ate(X)

        # Should get similar result to standard ATE
        standard_effect = tmtl.estimate_ate()

        # Results should be close (transport weights should be nearly uniform)
        assert abs(effect.ate - standard_effect.ate) < 0.5

        # Transport weights should be close to uniform
        weight_cv = np.std(tmtl.transport_weights) / np.mean(tmtl.transport_weights)
        assert weight_cv < 0.2  # Low coefficient of variation

    def test_dataframe_target_input(self, sample_data):
        """Test with DataFrame target input."""
        X, T, Y, _ = sample_data
        X_target = np.random.randn(300, 4)

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        # Convert target to DataFrame
        target_df = pd.DataFrame(
            X_target, columns=[f"X{i}" for i in range(X_target.shape[1])]
        )

        effect = tmtl.estimate_transported_ate(target_df)

        assert isinstance(effect, CausalEffect)
        assert effect.ate is not None

    def test_small_target_population(self, sample_data):
        """Test with very small target population."""
        X, T, Y, _ = sample_data
        X_target = np.random.randn(50, 4)  # Small target population

        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X, names=[f"X{i}" for i in range(X.shape[1])]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        # Should still work with small target population
        effect = tmtl.estimate_transported_ate(X_target)

        assert isinstance(effect, CausalEffect)
        assert effect.ate is not None

        # Effective sample size should be reasonable
        eff_n = tmtl.transport_weighting_result.effective_sample_size
        assert eff_n > 10  # Should have reasonable effective sample size
