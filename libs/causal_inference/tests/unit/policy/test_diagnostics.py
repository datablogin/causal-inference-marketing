"""Tests for diagnostic methods in off-policy evaluation."""

import numpy as np
import pytest

from causal_inference.policy.off_policy_evaluation import OffPolicyEvaluator


class TestDiagnosticMethods:
    """Test diagnostic methods for assumption checking."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_individuals = 200
        self.features = np.random.randn(self.n_individuals, 4)
        self.treatments = np.random.choice([True, False], self.n_individuals)
        self.evaluator = OffPolicyEvaluator(method="dr", random_state=42)

    def test_check_positivity_good_case(self):
        """Test positivity check with good propensity scores."""
        # Create reasonable propensity scores
        propensity_scores = np.random.uniform(0.1, 0.9, self.n_individuals)

        result = self.evaluator.check_positivity(
            self.treatments, self.features, propensity_scores
        )

        assert result["positivity_ok"] is True
        assert result["n_violations_low"] == 0
        assert result["n_violations_high"] == 0
        assert result["violation_rate"] == 0.0
        assert 0.1 <= result["min_propensity"] <= 0.9
        assert 0.1 <= result["max_propensity"] <= 0.9

    def test_check_positivity_violations(self):
        """Test positivity check with violations."""
        # Create problematic propensity scores
        propensity_scores = np.concatenate(
            [
                [0.005, 0.995],  # Extreme values
                np.random.uniform(0.1, 0.9, self.n_individuals - 2),
            ]
        )

        result = self.evaluator.check_positivity(
            self.treatments[: len(propensity_scores)],
            self.features[: len(propensity_scores)],
            propensity_scores,
        )

        assert result["positivity_ok"] is False
        assert result["n_violations_low"] == 1
        assert result["n_violations_high"] == 1
        assert result["violation_rate"] == 2.0 / len(propensity_scores)
        assert len(result["extreme_indices"]) == 2

    def test_check_positivity_without_propensity(self):
        """Test positivity check when propensity scores need to be estimated."""
        result = self.evaluator.check_positivity(self.treatments, self.features)

        # Should work with estimated propensities
        assert isinstance(result["positivity_ok"], bool)
        assert isinstance(result["violation_rate"], float)
        assert result["violation_rate"] >= 0.0

    def test_check_overlap_good_case(self):
        """Test overlap check with good covariate balance."""
        # Create balanced features
        n_half = self.n_individuals // 2
        features_balanced = np.random.randn(self.n_individuals, 3)
        treatments_balanced = np.array(
            [True] * n_half + [False] * (self.n_individuals - n_half)
        )

        result = self.evaluator.check_overlap(treatments_balanced, features_balanced)

        assert result["overlap_ok"] is True
        assert result["n_treated"] == n_half
        assert result["n_control"] == self.n_individuals - n_half
        assert result["n_problematic"] == 0
        assert result["max_std_diff"] < 0.5

    def test_check_overlap_poor_case(self):
        """Test overlap check with poor covariate balance."""
        # Create unbalanced features
        n_treated = 50
        n_control = 150

        # Treated group has much higher feature values
        treated_features = np.random.normal(2.0, 1.0, (n_treated, 3))
        control_features = np.random.normal(-1.0, 1.0, (n_control, 3))

        features_unbalanced = np.vstack([treated_features, control_features])
        treatments_unbalanced = np.array([True] * n_treated + [False] * n_control)

        result = self.evaluator.check_overlap(
            treatments_unbalanced, features_unbalanced
        )

        assert result["overlap_ok"] is False
        assert result["n_treated"] == n_treated
        assert result["n_control"] == n_control
        assert result["n_problematic"] > 0
        assert result["max_std_diff"] > 0.5

    def test_check_overlap_missing_group(self):
        """Test overlap check when one group is missing."""
        # All treated
        treatments_all_treated = np.array([True] * self.n_individuals)

        result = self.evaluator.check_overlap(treatments_all_treated, self.features)

        assert result["overlap_ok"] is False
        assert "missing" in result["reason"]
        assert result["n_treated"] == self.n_individuals
        assert result["n_control"] == 0

    def test_diagnose_assumptions_comprehensive(self):
        """Test comprehensive assumption diagnostics."""
        result = self.evaluator.diagnose_assumptions(self.treatments, self.features)

        assert "assumptions_ok" in result
        assert "warnings" in result
        assert "positivity" in result
        assert "overlap" in result
        assert "recommendation" in result
        assert isinstance(result["assumptions_ok"], bool)
        assert isinstance(result["warnings"], list)

    def test_diagnose_assumptions_with_problems(self):
        """Test diagnostics when assumptions are violated."""
        # Create problematic scenario
        n_treated = 20
        n_control = 180

        # Poor overlap
        treated_features = np.random.normal(3.0, 0.5, (n_treated, 2))
        control_features = np.random.normal(-2.0, 0.5, (n_control, 2))

        features_bad = np.vstack([treated_features, control_features])
        treatments_bad = np.array([True] * n_treated + [False] * n_control)

        # Extreme propensities
        propensity_bad = np.concatenate(
            [
                [0.001] * 5,  # Very low
                [0.5] * (len(treatments_bad) - 10),
                [0.999] * 5,  # Very high
            ]
        )

        result = self.evaluator.diagnose_assumptions(
            treatments_bad, features_bad, propensity_bad
        )

        assert result["assumptions_ok"] is False
        assert len(result["warnings"]) > 0
        assert "sensitivity analysis" in result["recommendation"]

    def test_diagnostic_edge_cases(self):
        """Test diagnostic methods with edge cases."""
        # Empty groups
        empty_treatments = np.array([])
        empty_features = np.empty((0, 3))

        with pytest.raises(ValueError, match="Treatments and features cannot be empty"):
            self.evaluator.check_overlap(empty_treatments, empty_features)

        # Single observation
        single_treatment = np.array([True])
        single_features = np.array([[1.0, 2.0]])

        result = self.evaluator.check_overlap(single_treatment, single_features)
        assert result["overlap_ok"] is False

    def test_diagnostic_numerical_stability(self):
        """Test diagnostic methods handle numerical edge cases."""
        # Features with zero variance
        constant_features = np.ones((100, 3))
        treatments_balanced = np.array([True] * 50 + [False] * 50)

        result = self.evaluator.check_overlap(treatments_balanced, constant_features)

        # Should handle zero variance gracefully
        assert isinstance(result["overlap_ok"], bool)
        assert not np.any(np.isnan(result["standardized_diffs"]))
        assert not np.any(np.isinf(result["standardized_diffs"]))
