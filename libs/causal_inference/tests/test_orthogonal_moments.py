"""Tests for orthogonal moment functions in Double Machine Learning.

This module tests the OrthogonalMoments class and its various
moment function implementations for correctness, orthogonality,
and statistical properties.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from causal_inference.core.base import (
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.doubly_robust_ml import DoublyRobustMLEstimator
from causal_inference.estimators.orthogonal_moments import OrthogonalMoments


class TestOrthogonalMomentsBasic:
    """Test basic functionality of OrthogonalMoments class."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        np.random.seed(42)
        n = 200
        p = 5

        # Generate confounders
        X = np.random.randn(n, p)

        # Generate treatment with confounding
        treatment_coef = np.array([0.5, -0.3, 0.2, 0.0, -0.1])
        logit_p = X @ treatment_coef
        treatment_prob = 1 / (1 + np.exp(-logit_p))
        treatment = np.random.binomial(1, treatment_prob)

        # Generate outcome with treatment effect and confounding
        outcome_coef = np.array([0.3, -0.2, 0.4, -0.1, 0.2])
        true_ate = 2.0
        outcome = X @ outcome_coef + true_ate * treatment + np.random.normal(0, 0.5, n)

        return X, treatment, outcome, true_ate

    @pytest.fixture
    def nuisance_estimates(self, synthetic_data):
        """Generate mock nuisance parameter estimates."""
        X, treatment, outcome, _ = synthetic_data
        n = len(treatment)

        # Mock propensity scores (slightly different from true to simulate estimation error)
        g_true = 1 / (1 + np.exp(-(X @ np.array([0.4, -0.25, 0.15, 0.05, -0.05]))))
        g = np.clip(g_true + np.random.normal(0, 0.05, n), 0.01, 0.99)

        # Mock outcome models (with some estimation error)
        treated_mask = treatment == 1
        control_mask = treatment == 0

        # Separate outcome models
        mu1 = np.mean(outcome[treated_mask]) + 0.1 * np.random.randn(n)
        mu0 = np.mean(outcome[control_mask]) + 0.1 * np.random.randn(n)

        # Combined outcome model predictions
        mu1_combined = mu1 + np.random.normal(0, 0.1, n)
        mu0_combined = mu0 + np.random.normal(0, 0.1, n)
        mu_observed = treatment * mu1_combined + (1 - treatment) * mu0_combined

        return {
            "propensity_scores": g,
            "mu1": mu1,
            "mu0": mu0,
            "mu1_combined": mu1_combined,
            "mu0_combined": mu0_combined,
            "mu_observed": mu_observed,
        }

    def test_available_methods(self):
        """Test that all expected methods are available."""
        methods = OrthogonalMoments.get_available_methods()
        expected_methods = [
            "aipw",
            "orthogonal",
            "partialling_out",
            "interactive_iv",
            "plr",
            "pliv",
        ]

        assert isinstance(methods, list)
        for method in expected_methods:
            assert method in methods

    def test_compute_scores_dispatch(self, synthetic_data, nuisance_estimates):
        """Test that compute_scores correctly dispatches to appropriate methods."""
        _, treatment, outcome, _ = synthetic_data

        for method in OrthogonalMoments.get_available_methods():
            scores = OrthogonalMoments.compute_scores(
                method, nuisance_estimates, treatment, outcome
            )

            assert isinstance(scores, np.ndarray)
            assert len(scores) == len(treatment)
            assert np.all(np.isfinite(scores))

    def test_invalid_method_raises_error(self, synthetic_data, nuisance_estimates):
        """Test that invalid method name raises ValueError."""
        _, treatment, outcome, _ = synthetic_data

        with pytest.raises(ValueError, match="Unknown method 'invalid_method'"):
            OrthogonalMoments.compute_scores(
                "invalid_method", nuisance_estimates, treatment, outcome
            )


class TestAIPWMoments:
    """Test AIPW moment function implementation."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for AIPW testing."""
        np.random.seed(123)
        n = 150

        X = np.random.randn(n, 3)
        treatment_prob = 0.3 + 0.4 * (1 / (1 + np.exp(-X[:, 0])))
        treatment = np.random.binomial(1, treatment_prob)

        # True potential outcomes
        mu1_true = 1 + X[:, 0] + X[:, 1]
        mu0_true = X[:, 1] - 0.5 * X[:, 2]

        outcome = (
            treatment * mu1_true
            + (1 - treatment) * mu0_true
            + np.random.normal(0, 0.3, n)
        )

        return X, treatment, outcome, mu1_true, mu0_true, treatment_prob

    def test_aipw_with_perfect_nuisance(self, synthetic_data):
        """Test AIPW with perfect nuisance parameter estimates."""
        X, treatment, outcome, mu1_true, mu0_true, g_true = synthetic_data

        # Perfect nuisance estimates
        nuisance_estimates = {
            "mu1": mu1_true,
            "mu0": mu0_true,
            "propensity_scores": g_true,
        }

        scores = OrthogonalMoments.aipw(nuisance_estimates, treatment, outcome)
        ate_estimate = np.mean(scores)

        # Should recover true ATE closely with perfect nuisance functions
        true_ate = np.mean(mu1_true - mu0_true)
        assert abs(ate_estimate - true_ate) < 0.1

    def test_aipw_doubly_robust_property(self, synthetic_data):
        """Test doubly robust property: consistency when one model is correct."""
        X, treatment, outcome, mu1_true, mu0_true, g_true = synthetic_data
        true_ate = np.mean(mu1_true - mu0_true)

        # Case 1: Correct outcome model, wrong propensity model
        nuisance_estimates_1 = {
            "mu1": mu1_true,
            "mu0": mu0_true,
            "propensity_scores": np.full(len(treatment), 0.5),  # Wrong but overlap
        }

        scores_1 = OrthogonalMoments.aipw(nuisance_estimates_1, treatment, outcome)
        ate_1 = np.mean(scores_1)

        # Case 2: Wrong outcome model, correct propensity model
        nuisance_estimates_2 = {
            "mu1": np.full(len(treatment), np.mean(outcome)),  # Wrong
            "mu0": np.full(len(treatment), np.mean(outcome)),  # Wrong
            "propensity_scores": g_true,
        }

        scores_2 = OrthogonalMoments.aipw(nuisance_estimates_2, treatment, outcome)
        ate_2 = np.mean(scores_2)

        # Both should be reasonably close to true ATE (doubly robust property)
        assert abs(ate_1 - true_ate) < 0.5
        assert abs(ate_2 - true_ate) < 0.5

    def test_aipw_extreme_propensity_handling(self, synthetic_data):
        """Test AIPW behavior with extreme propensity scores."""
        X, treatment, outcome, mu1_true, mu0_true, _ = synthetic_data

        # Extreme propensity scores (but still within bounds)
        g_extreme = np.where(treatment == 1, 0.99, 0.01)

        nuisance_estimates = {
            "mu1": mu1_true,
            "mu0": mu0_true,
            "propensity_scores": g_extreme,
        }

        scores = OrthogonalMoments.aipw(nuisance_estimates, treatment, outcome)

        # Should not produce infinite or extremely large values
        assert np.all(np.isfinite(scores))
        assert np.std(scores) < 100  # Reasonable variance


class TestOrthogonalMoments:
    """Test orthogonal moment function implementation."""

    @pytest.fixture
    def synthetic_data_orthogonal(self):
        """Generate synthetic data suitable for orthogonal moments testing."""
        np.random.seed(456)
        n = 200

        X = np.random.randn(n, 4)

        # Treatment model
        g_true = 1 / (1 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])))
        treatment = np.random.binomial(1, g_true)

        # Outcome model (depends on treatment and covariates)
        outcome = (
            2.0 * treatment
            + X[:, 0]
            + 0.5 * X[:, 1]
            - 0.3 * X[:, 2]
            + np.random.normal(0, 0.4, n)
        )

        return X, treatment, outcome, g_true

    def test_orthogonal_moment_computation(self, synthetic_data_orthogonal):
        """Test basic orthogonal moment computation."""
        X, treatment, outcome, g_true = synthetic_data_orthogonal

        # Mock combined outcome model predictions
        mu_observed = outcome + np.random.normal(0, 0.1, len(outcome))
        mu1_combined = outcome + 2.0 + np.random.normal(0, 0.1, len(outcome))
        mu0_combined = outcome - 2.0 + np.random.normal(0, 0.1, len(outcome))

        nuisance_estimates = {
            "mu1_combined": mu1_combined,
            "mu0_combined": mu0_combined,
            "mu_observed": mu_observed,
            "propensity_scores": g_true,
        }

        scores = OrthogonalMoments.orthogonal(nuisance_estimates, treatment, outcome)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(treatment)
        assert np.all(np.isfinite(scores))

    def test_orthogonal_score_structure(self, synthetic_data_orthogonal):
        """Test that orthogonal scores have the expected mathematical structure."""
        X, treatment, outcome, g = synthetic_data_orthogonal

        # Create nuisance estimates
        mu_observed = np.mean(outcome) * np.ones(len(outcome))
        mu1 = np.mean(outcome) + 1.0
        mu0 = np.mean(outcome) - 1.0

        nuisance_estimates = {
            "mu1_combined": np.full(len(outcome), mu1),
            "mu0_combined": np.full(len(outcome), mu0),
            "mu_observed": mu_observed,
            "propensity_scores": g,
        }

        scores = OrthogonalMoments.orthogonal(nuisance_estimates, treatment, outcome)

        # Manual computation for verification
        expected_scores = (treatment - g) * (outcome - mu_observed) + mu1 - mu0

        assert_allclose(scores, expected_scores, rtol=1e-10)


class TestPartiallingOut:
    """Test partialling out moment function."""

    @pytest.fixture
    def partialling_out_data(self):
        """Generate data suitable for partialling out method."""
        np.random.seed(789)
        n = 250

        X = np.random.randn(n, 3)

        # Treatment depends on X
        g_true = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
        treatment = np.random.binomial(1, g_true)

        # Outcome depends on X and treatment
        outcome = (
            1.5 * treatment  # True ATE = 1.5
            + 0.8 * X[:, 0]
            + 0.4 * X[:, 1]
            - 0.3 * X[:, 2]
            + np.random.normal(0, 0.5, n)
        )

        return X, treatment, outcome, g_true

    def test_partialling_out_basic(self, partialling_out_data):
        """Test basic partialling out computation."""
        X, treatment, outcome, g_true = partialling_out_data

        # Outcome model E[Y|X] (marginal, not conditional on treatment)
        mu_observed = (
            0.8 * X[:, 0] + 0.4 * X[:, 1] - 0.3 * X[:, 2] + 1.5 * g_true
        )  # Include expected treatment effect

        nuisance_estimates = {
            "propensity_scores": g_true,
            "mu_observed": mu_observed,
        }

        scores = OrthogonalMoments.partialling_out(
            nuisance_estimates, treatment, outcome
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(treatment)
        assert np.all(np.isfinite(scores))

    def test_partialling_out_weak_treatment_warning(self):
        """Test warning when treatment variation is weak after partialling out."""
        n = 100
        treatment = np.ones(n)  # No variation
        outcome = np.random.randn(n)
        g = np.ones(n)  # Perfect prediction -> no residual variation

        nuisance_estimates = {
            "propensity_scores": g,
            "mu_observed": np.zeros(n),
        }

        with pytest.warns(UserWarning, match="Weak treatment variation"):
            scores = OrthogonalMoments.partialling_out(
                nuisance_estimates, treatment, outcome
            )

        # Should still return valid scores
        assert len(scores) == n
        assert np.all(np.isfinite(scores))


class TestPLRMoments:
    """Test Partially Linear Regression moment function."""

    @pytest.fixture
    def plr_data(self):
        """Generate data suitable for PLR testing."""
        np.random.seed(101)
        n = 180

        X = np.random.randn(n, 4)

        # Treatment model
        g_true = 1 / (1 + np.exp(-(0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2])))
        treatment = np.random.binomial(1, g_true)

        # PLR outcome model: Y = θD + g(X) + ε
        g_x = 0.6 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * X[:, 3]
        true_ate = 2.5
        outcome = true_ate * treatment + g_x + np.random.normal(0, 0.4, n)

        # E[Y|D=0,X] = g(X)
        mu0_true = g_x

        return X, treatment, outcome, g_true, mu0_true, true_ate

    def test_plr_basic_computation(self, plr_data):
        """Test basic PLR computation."""
        X, treatment, outcome, g_true, mu0_true, true_ate = plr_data

        nuisance_estimates = {
            "propensity_scores": g_true,
            "mu0": mu0_true,
        }

        scores = OrthogonalMoments.plr(nuisance_estimates, treatment, outcome)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(treatment)
        assert np.all(np.isfinite(scores))

        # The score should have mean approximately zero under correct specification
        assert abs(np.mean(scores)) < 0.5

    def test_plr_with_estimation_error(self, plr_data):
        """Test PLR robustness to estimation error in nuisance functions."""
        X, treatment, outcome, g_true, mu0_true, true_ate = plr_data

        # Add estimation error to nuisance functions
        g_estimated = np.clip(
            g_true + np.random.normal(0, 0.05, len(treatment)), 0.01, 0.99
        )
        mu0_estimated = mu0_true + np.random.normal(0, 0.2, len(treatment))

        nuisance_estimates = {
            "propensity_scores": g_estimated,
            "mu0": mu0_estimated,
        }

        scores = OrthogonalMoments.plr(nuisance_estimates, treatment, outcome)

        # Should still produce reasonable estimates
        assert np.all(np.isfinite(scores))
        assert np.std(scores) < 10  # Reasonable variance


class TestInstrumentalVariableMethods:
    """Test IV-based moment functions (Interactive IV and PLIV)."""

    @pytest.fixture
    def iv_data(self):
        """Generate synthetic data with instrumental variable."""
        np.random.seed(202)
        n = 200

        X = np.random.randn(n, 3)

        # Instrument Z (correlated with treatment but not directly with outcome)
        Z = np.random.binomial(1, 0.4, n) + 0.3 * X[:, 0] + np.random.normal(0, 0.2, n)

        # Treatment (endogenous, correlated with unobserved confounder)
        U = np.random.randn(n)  # Unobserved confounder
        treatment_prob = 1 / (1 + np.exp(-(0.5 * Z + 0.3 * X[:, 0] + 0.2 * U)))
        treatment = np.random.binomial(1, treatment_prob)

        # Outcome (affected by treatment and unobserved confounder)
        true_ate = 1.8
        outcome = (
            true_ate * treatment
            + X[:, 0]
            + 0.5 * X[:, 1]
            - 0.3 * X[:, 2]
            + 0.4 * U  # This creates endogeneity
            + np.random.normal(0, 0.5, n)
        )

        # Propensity score (ignores unobserved confounder - misspecified)
        g_misspec = 1 / (1 + np.exp(-(0.4 * Z + 0.25 * X[:, 0])))

        return X, treatment, outcome, Z, g_misspec, true_ate

    def test_interactive_iv_with_instrument(self, iv_data):
        """Test interactive IV with valid instrument."""
        X, treatment, outcome, Z, g, true_ate = iv_data

        # Mock outcome model
        mu_observed = np.mean(outcome) * np.ones(len(outcome))

        nuisance_estimates = {
            "propensity_scores": g,
            "mu_observed": mu_observed,
        }

        scores = OrthogonalMoments.interactive_iv(
            nuisance_estimates, treatment, outcome, instrument=Z
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(treatment)
        assert np.all(np.isfinite(scores))

    def test_interactive_iv_without_instrument(self, iv_data):
        """Test interactive IV fallback when no instrument provided."""
        X, treatment, outcome, Z, g, true_ate = iv_data

        mu1_combined = np.full(len(treatment), np.mean(outcome) + 1)
        mu0_combined = np.full(len(treatment), np.mean(outcome) - 1)
        mu_observed = np.mean(outcome) * np.ones(len(treatment))

        nuisance_estimates = {
            "propensity_scores": g,
            "mu1_combined": mu1_combined,
            "mu0_combined": mu0_combined,
            "mu_observed": mu_observed,
        }

        with pytest.warns(UserWarning, match="No instrument provided"):
            scores = OrthogonalMoments.interactive_iv(
                nuisance_estimates, treatment, outcome
            )

        # Should fall back to orthogonal method
        expected_scores = OrthogonalMoments.orthogonal(
            nuisance_estimates, treatment, outcome
        )
        assert_allclose(scores, expected_scores)

    def test_pliv_basic(self, iv_data):
        """Test PLIV basic computation."""
        X, treatment, outcome, Z, g, true_ate = iv_data

        # E[Y|D=0,X] for PLIV
        mu0 = np.mean(outcome) * np.ones(len(treatment))

        nuisance_estimates = {
            "propensity_scores": g,
            "mu0": mu0,
        }

        scores = OrthogonalMoments.pliv(
            nuisance_estimates, treatment, outcome, instrument=Z
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(treatment)
        assert np.all(np.isfinite(scores))

    def test_pliv_without_instrument(self, iv_data):
        """Test PLIV fallback to PLR when no instrument provided."""
        X, treatment, outcome, Z, g, true_ate = iv_data

        mu0 = np.mean(outcome) * np.ones(len(treatment))

        nuisance_estimates = {
            "propensity_scores": g,
            "mu0": mu0,
        }

        with pytest.warns(UserWarning, match="No instrument provided"):
            scores = OrthogonalMoments.pliv(nuisance_estimates, treatment, outcome)

        # Should fall back to PLR method
        expected_scores = OrthogonalMoments.plr(nuisance_estimates, treatment, outcome)
        assert_allclose(scores, expected_scores)


class TestOrthogonalityValidation:
    """Test orthogonality validation functionality."""

    @pytest.fixture
    def validation_data(self):
        """Generate data for orthogonality validation testing."""
        np.random.seed(303)
        n = 150

        X = np.random.randn(n, 3)
        g_true = 1 / (1 + np.exp(-X[:, 0]))
        treatment = np.random.binomial(1, g_true)
        outcome = X[:, 0] + 2 * treatment + np.random.randn(n)

        nuisance_estimates = {
            "propensity_scores": g_true + np.random.normal(0, 0.02, n),
        }

        return treatment, outcome, nuisance_estimates

    def test_orthogonality_validation_structure(self, validation_data):
        """Test structure of orthogonality validation results."""
        treatment, outcome, nuisance_estimates = validation_data

        # Generate some scores
        scores = np.random.randn(len(treatment))

        results = OrthogonalMoments.validate_orthogonality(
            scores, nuisance_estimates, treatment, threshold=0.05
        )

        # Check required fields
        assert "is_orthogonal" in results
        assert "correlations" in results
        assert "threshold" in results
        assert "failed_checks" in results
        assert "interpretation" in results

        assert isinstance(results["is_orthogonal"], bool)
        assert isinstance(results["correlations"], dict)
        assert results["threshold"] == 0.05

    def test_orthogonality_violation_detection(self, validation_data):
        """Test detection of orthogonality violations."""
        treatment, outcome, nuisance_estimates = validation_data

        # Create scores that are deliberately correlated with treatment residuals
        g = nuisance_estimates["propensity_scores"]
        treatment_residual = treatment - g
        correlated_scores = 0.3 * treatment_residual + np.random.randn(len(treatment))

        results = OrthogonalMoments.validate_orthogonality(
            correlated_scores, nuisance_estimates, treatment, threshold=0.05
        )

        # Should detect violation
        assert not results["is_orthogonal"]
        assert len(results["failed_checks"]) > 0
        assert "treatment_residual" in results["correlations"]


class TestAutomaticMethodSelection:
    """Test automatic moment function selection."""

    @pytest.fixture
    def selection_test_data(self):
        """Generate various data scenarios for method selection testing."""
        np.random.seed(404)

        # Scenario 1: Balanced, good overlap, moderate dimensionality
        n1 = 200
        X1 = np.random.randn(n1, 5)
        g1 = 0.3 + 0.4 * (1 / (1 + np.exp(-X1[:, 0])))  # Good overlap
        treatment1 = np.random.binomial(1, g1)
        outcome1 = X1.sum(axis=1) + 2 * treatment1 + np.random.randn(n1)

        # Scenario 2: Small sample, high dimensional
        n2 = 50
        X2 = np.random.randn(n2, 25)  # High dimensional
        g2 = 0.5 * np.ones(n2)  # Balanced
        treatment2 = np.random.binomial(1, g2)
        outcome2 = X2[:, :3].sum(axis=1) + 1.5 * treatment2 + np.random.randn(n2)

        # Scenario 3: Poor overlap
        n3 = 150
        X3 = np.random.randn(n3, 4)
        g3 = 1 / (1 + np.exp(-3 * X3[:, 0]))  # Poor overlap
        treatment3 = np.random.binomial(1, g3)
        outcome3 = X3.sum(axis=1) + 2 * treatment3 + np.random.randn(n3)

        return {
            "balanced_good_overlap": (X1, treatment1, outcome1, g1),
            "small_high_dim": (X2, treatment2, outcome2, g2),
            "poor_overlap": (X3, treatment3, outcome3, g3),
        }

    def create_nuisance_estimates(self, x, treatment, outcome, g):
        """Helper to create nuisance estimates."""
        n = len(treatment)
        mu1 = np.mean(outcome[treatment == 1]) * np.ones(n)
        mu0 = np.mean(outcome[treatment == 0]) * np.ones(n)
        mu_observed = treatment * mu1 + (1 - treatment) * mu0

        return {
            "propensity_scores": g,
            "mu1": mu1,
            "mu0": mu0,
            "mu1_combined": mu1,
            "mu0_combined": mu0,
            "mu_observed": mu_observed,
        }

    def test_method_selection_balanced_case(self, selection_test_data):
        """Test method selection for balanced, good overlap case."""
        X, treatment, outcome, g = selection_test_data["balanced_good_overlap"]
        nuisance_estimates = self.create_nuisance_estimates(X, treatment, outcome, g)

        selected_method, rationale = OrthogonalMoments.select_optimal_method(
            nuisance_estimates, treatment, outcome, X
        )

        # Should select AIPW for well-behaved case
        assert selected_method == "aipw"
        assert "AIPW selected" in " ".join(rationale["decision_factors"])

    def test_method_selection_high_dimensional_case(self, selection_test_data):
        """Test method selection for high-dimensional case."""
        X, treatment, outcome, g = selection_test_data["small_high_dim"]
        nuisance_estimates = self.create_nuisance_estimates(X, treatment, outcome, g)

        selected_method, rationale = OrthogonalMoments.select_optimal_method(
            nuisance_estimates, treatment, outcome, X
        )

        # Should select partialling out for high-dimensional case
        assert selected_method == "partialling_out"
        assert "high-dimensional" in " ".join(rationale["decision_factors"])

    def test_method_selection_poor_overlap_case(self, selection_test_data):
        """Test method selection for poor overlap case."""
        X, treatment, outcome, g = selection_test_data["poor_overlap"]
        nuisance_estimates = self.create_nuisance_estimates(X, treatment, outcome, g)

        selected_method, rationale = OrthogonalMoments.select_optimal_method(
            nuisance_estimates, treatment, outcome, X
        )

        # Should select PLR or partialling_out for poor overlap case (no instrument available)
        assert selected_method in ["plr", "partialling_out"]
        # Should mention either poor overlap or small sample as decision factor
        factors = " ".join(rationale["decision_factors"])
        assert "poor overlap" in factors or "small sample" in factors

    def test_method_selection_with_instrument(self, selection_test_data):
        """Test method selection when instrument is available."""
        X, treatment, outcome, g = selection_test_data["poor_overlap"]
        nuisance_estimates = self.create_nuisance_estimates(X, treatment, outcome, g)

        # Provide an instrument
        instrument = np.random.randn(len(treatment))

        selected_method, rationale = OrthogonalMoments.select_optimal_method(
            nuisance_estimates, treatment, outcome, X, instrument
        )

        # Should prefer IV method when instrument available and overlap is poor
        assert selected_method in ["interactive_iv", "pliv"]
        assert "instrument available" in " ".join(rationale["decision_factors"])


class TestCrossValidationComparison:
    """Test cross-validation method comparison functionality."""

    @pytest.fixture
    def cv_test_data(self):
        """Generate data for cross-validation testing."""
        np.random.seed(505)
        n = 100  # Smaller for faster CV

        X = np.random.randn(n, 3)
        g = 1 / (1 + np.exp(-X[:, 0]))
        treatment = np.random.binomial(1, g)
        outcome = X.sum(axis=1) + 2 * treatment + np.random.randn(n)

        # Create nuisance estimates
        mu1 = np.mean(outcome[treatment == 1]) * np.ones(n)
        mu0 = np.mean(outcome[treatment == 0]) * np.ones(n)
        mu_observed = treatment * mu1 + (1 - treatment) * mu0

        nuisance_estimates = {
            "propensity_scores": g,
            "mu1": mu1,
            "mu0": mu0,
            "mu1_combined": mu1,
            "mu0_combined": mu0,
            "mu_observed": mu_observed,
        }

        return treatment, outcome, nuisance_estimates

    def test_cv_comparison_basic_functionality(self, cv_test_data):
        """Test basic cross-validation comparison functionality."""
        treatment, outcome, nuisance_estimates = cv_test_data

        candidate_methods = ["aipw", "orthogonal", "partialling_out"]

        results = OrthogonalMoments.cross_validate_methods(
            candidate_methods, nuisance_estimates, treatment, outcome, cv_folds=2
        )

        # Check structure
        assert "method_performance" in results
        assert "rankings" in results
        assert "recommended_method" in results

        # Check that all methods were evaluated
        for method in candidate_methods:
            assert method in results["method_performance"]

        # Check rankings structure
        assert "by_combined_score" in results["rankings"]
        rankings = results["rankings"]["by_combined_score"]

        assert len(rankings) == len(candidate_methods)
        assert all("method" in rank for rank in rankings)
        assert all("combined_score" in rank for rank in rankings)

    def test_cv_comparison_ranking_order(self, cv_test_data):
        """Test that CV comparison produces sensible ranking order."""
        treatment, outcome, nuisance_estimates = cv_test_data

        # Compare just AIPW vs a method that should perform worse
        candidate_methods = ["aipw", "partialling_out"]

        results = OrthogonalMoments.cross_validate_methods(
            candidate_methods, nuisance_estimates, treatment, outcome, cv_folds=2
        )

        rankings = results["rankings"]["by_combined_score"]

        # Rankings should be sorted by combined_score in descending order
        scores = [rank["combined_score"] for rank in rankings]
        assert scores == sorted(scores, reverse=True)

        # Recommended method should be the top-ranked one
        assert results["recommended_method"] == rankings[0]["method"]


class TestIntegrationWithDoublyRobustML:
    """Test integration of OrthogonalMoments with DoublyRobustMLEstimator."""

    @pytest.fixture
    def integration_data(self):
        """Generate data for integration testing."""
        np.random.seed(606)
        n = 100
        p = 4

        X = np.random.randn(n, p)
        treatment_coef = np.array([0.5, -0.3, 0.2, 0.1])
        logit_p = X @ treatment_coef
        treatment_prob = 1 / (1 + np.exp(-logit_p))
        treatment = np.random.binomial(1, treatment_prob)

        outcome_coef = np.array([0.3, -0.2, 0.4, -0.1])
        true_ate = 1.8
        outcome = X @ outcome_coef + true_ate * treatment + np.random.normal(0, 0.5, n)

        return X, treatment, outcome, true_ate

    def test_estimator_with_new_moment_functions(self, integration_data):
        """Test DoublyRobustMLEstimator with new moment functions."""
        X, treatment, outcome, true_ate = integration_data

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X), names=[f"X{i}" for i in range(X.shape[1])]
        )

        # Test different moment functions
        moment_functions = ["aipw", "orthogonal", "partialling_out", "plr"]

        for moment_function in moment_functions:
            estimator = DoublyRobustMLEstimator(
                cross_fitting=False,  # Simpler for test
                moment_function=moment_function,
                random_state=42,
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            # Should produce reasonable estimates
            assert isinstance(effect.ate, float)
            assert (
                abs(effect.ate - true_ate) < 3.0
            )  # Relaxed tolerance for small sample
            assert effect.method.startswith("DoublyRobustML_")
            assert moment_function in effect.method

    def test_estimator_auto_selection(self, integration_data):
        """Test DoublyRobustMLEstimator with automatic method selection."""
        X, treatment, outcome, true_ate = integration_data

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        estimator = DoublyRobustMLEstimator(
            cross_fitting=False,
            moment_function="auto",  # Auto selection
            verbose=True,
            random_state=42,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Should have selected a method and stored results
        selection_results = estimator.get_moment_selection_results()

        assert isinstance(selection_results, dict)
        assert "selected_method" in selection_results
        assert "decision_factors" in selection_results
        assert (
            selection_results["selected_method"]
            in OrthogonalMoments.get_available_methods()
        )

        # Effect method should reflect selected method
        assert selection_results["selected_method"] in effect.method

    def test_estimator_validation_methods(self, integration_data):
        """Test new validation methods in DoublyRobustMLEstimator."""
        X, treatment, outcome, true_ate = integration_data

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        estimator = DoublyRobustMLEstimator(
            cross_fitting=False,
            moment_function="aipw",
            random_state=42,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        estimator.estimate_ate()

        # Test moment function validation
        validation_results = estimator.validate_moment_function_choice()

        assert isinstance(validation_results, dict)
        assert "moment_method" in validation_results
        assert "validation_passed" in validation_results
        assert validation_results["moment_method"] == "aipw"

        # Test method comparison
        comparison_results = estimator.compare_moment_functions(
            candidate_methods=["aipw", "orthogonal"], cv_folds=2
        )

        assert isinstance(comparison_results, dict)
        assert "method_performance" in comparison_results
        assert "recommended_method" in comparison_results
        assert "current_method" in comparison_results
        assert comparison_results["current_method"] == "aipw"

    def test_error_handling_before_fitting(self):
        """Test error handling when methods are called before fitting."""
        estimator = DoublyRobustMLEstimator()

        # Should raise errors before fitting
        with pytest.raises(Exception):  # EstimationError
            estimator.get_moment_selection_results()

        with pytest.raises(Exception):  # EstimationError
            estimator.validate_moment_function_choice()

        with pytest.raises(Exception):  # EstimationError
            estimator.compare_moment_functions()
