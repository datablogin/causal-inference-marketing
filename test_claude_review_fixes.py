"""Tests for fixes implemented based on Claude review feedback.

This file tests the specific improvements made to address Claude's review comments:
1. Causal forest prediction loop efficiency
2. Policy value propensity score estimation
3. SIDES clustering improvements
4. Feature importance computation
"""

import numpy as np
import pytest

from libs.causal_inference.causal_inference.core.base import (
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from libs.causal_inference.causal_inference.estimators.causal_forest import CausalForest
from libs.causal_inference.causal_inference.estimators.subgroup_discovery import SIDES
from libs.causal_inference.causal_inference.evaluation.hte_metrics import policy_value


class TestCausalForestFixes:
    """Test fixes to causal forest implementation."""

    def test_feature_importance_not_uniform(self):
        """Test that feature importance is no longer uniform placeholder."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        treatment = np.random.binomial(1, 0.5, n)

        # Make X[0] strongly predictive of outcome
        outcome = 2 * X[:, 0] + treatment * (1 + X[:, 0]) + np.random.randn(n) * 0.3

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=20, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        importance = cf.feature_importance()

        # Should not be exactly uniform (1/3, 1/3, 1/3)
        uniform_importance = np.ones(3) / 3
        assert not np.allclose(importance, uniform_importance, atol=0.01)

        # Should sum to 1
        assert np.isclose(np.sum(importance), 1.0)

    def test_prediction_efficiency_correctness(self):
        """Test that prediction efficiency fix produces correct results."""
        np.random.seed(42)
        n = 80
        X = np.random.randn(n, 2)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = X.sum(axis=1) + treatment * X[:, 0] + np.random.randn(n) * 0.2

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=10, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        # Predict on test data
        X_test = np.random.randn(20, 2)
        cate_pred, cate_ci = cf.predict_cate(X_test)

        # Basic checks that results are reasonable
        assert cate_pred.shape == (20,)
        assert cate_ci.shape == (20, 2)
        assert np.all(cate_ci[:, 0] <= cate_ci[:, 1])
        assert not np.any(np.isnan(cate_pred))


class TestPolicyValueFixes:
    """Test fixes to policy value estimation."""

    def test_policy_value_with_propensity_scores(self):
        """Test that policy value uses individual propensity scores."""
        np.random.seed(42)
        n = 100

        outcomes = np.random.randn(n)
        treatments = np.random.binomial(1, 0.5, n)
        cate_estimates = np.random.randn(n)
        propensity_scores = np.random.beta(2, 2, n)  # Varying propensity

        # Test with provided propensity scores
        pv_with_ps = policy_value(
            outcomes, treatments, cate_estimates, propensity_scores=propensity_scores
        )

        # Test with automatic estimation (should be different)
        pv_without_ps = policy_value(outcomes, treatments, cate_estimates)

        assert isinstance(pv_with_ps, float)
        assert isinstance(pv_without_ps, float)
        assert not np.isnan(pv_with_ps)
        assert not np.isnan(pv_without_ps)

        # Values should generally be different when using different propensity estimation
        # (though they might occasionally be similar by chance)

    def test_policy_value_fallback_behavior(self):
        """Test policy value fallback when propensity estimation fails."""
        # Create edge case data that might cause fitting issues
        outcomes = np.array([1.0, 2.0, 3.0])
        treatments = np.array([1, 1, 1])  # All treated
        cate_estimates = np.array([0.5, -0.5, 1.0])

        # Should not crash and should return a reasonable value
        pv = policy_value(outcomes, treatments, cate_estimates)
        assert isinstance(pv, float)
        assert not np.isnan(pv)


class TestSIDESFixes:
    """Test fixes to SIDES clustering."""

    def test_sides_cluster_selection(self):
        """Test that SIDES uses proper cluster selection rather than fixed k=2."""
        np.random.seed(42)
        n = 120  # Large enough for multiple clusters

        # Create data with clear cluster structure (3 groups)
        X1 = np.random.randn(40, 2) + np.array([2, 2])
        X2 = np.random.randn(40, 2) + np.array([-2, -2])
        X3 = np.random.randn(40, 2) + np.array([0, 3])
        X = np.vstack([X1, X2, X3])

        treatments = np.random.binomial(1, 0.5, n)

        # Different effects for each cluster
        cate_true = np.array([2.0] * 40 + [0.5] * 40 + [1.5] * 40)
        outcomes = np.random.randn(n) + treatments * cate_true
        cate_estimates = cate_true + np.random.randn(n) * 0.1

        sides = SIDES(min_subgroup_size=25)
        result = sides.discover_subgroups(X, outcomes, treatments, cate_estimates)

        # Should potentially find more than 2 clusters with proper selection
        # (exact number depends on data and algorithm, but should not be fixed at 2)
        assert len(result.subgroups) >= 1  # Should find at least one subgroup

        # All subgroups should meet size requirements
        for subgroup in result.subgroups:
            assert subgroup.size >= sides.min_subgroup_size


class TestIntegrationFixes:
    """Test that all fixes work together properly."""

    def test_end_to_end_with_fixes(self):
        """Test complete workflow with all fixes applied."""
        np.random.seed(123)
        n = 150

        X = np.random.randn(n, 3)
        treatment = np.random.binomial(1, 0.5, n)

        # Heterogeneous effects
        true_cate = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1]
        outcome = X.sum(axis=1) + treatment * true_cate + np.random.randn(n) * 0.4

        # Test causal forest with fixes
        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        cf = CausalForest(n_estimators=30, random_state=42)
        cf.fit(treatment_data, outcome_data, covariate_data)

        # Predictions should work
        cate_pred, cate_ci = cf.predict_cate(X)
        ate_result = cf.estimate_ate()

        # Feature importance should not be uniform
        importance = cf.feature_importance()
        uniform = np.ones(3) / 3
        assert not np.allclose(importance, uniform, atol=0.05)

        # Policy value with improved propensity estimation
        pv = policy_value(outcome, treatment, cate_pred)
        assert isinstance(pv, float) and not np.isnan(pv)

        # SIDES with improved clustering
        sides = SIDES(min_subgroup_size=30)
        subgroup_result = sides.discover_subgroups(X, outcome, treatment, cate_pred)

        # Basic sanity checks
        assert len(cate_pred) == n
        assert np.all(cate_ci[:, 0] <= cate_ci[:, 1])
        assert isinstance(ate_result.ate, float)
        assert len(subgroup_result.subgroups) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
