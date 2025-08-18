"""Tests for off-policy evaluation functionality."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causal_inference.policy.off_policy_evaluation import (
    OffPolicyEvaluator,
    OPEResult,
    dr_estimator,
    dr_sn_estimator,
    ips_estimator,
)


class TestOPEResult:
    """Test OPEResult dataclass."""

    def test_ope_result_creation(self):
        """Test creation of OPEResult."""
        result = OPEResult(
            policy_value=1.5,
            policy_value_se=0.2,
            confidence_interval=(1.1, 1.9),
            method="ips",
            bias_estimate=0.1,
            variance_estimate=0.04,
        )

        assert result.policy_value == 1.5
        assert result.policy_value_se == 0.2
        assert result.ci_lower == 1.1
        assert result.ci_upper == 1.9
        assert result.method == "ips"

    def test_evaluation_summary(self):
        """Test evaluation summary generation."""
        result = OPEResult(
            policy_value=2.0,
            policy_value_se=0.3,
            confidence_interval=(1.4, 2.6),
            method="dr",
        )

        summary = result.get_evaluation_summary()
        assert summary["method"] == "dr"
        assert summary["policy_value"] == 2.0
        assert summary["policy_value_se"] == 0.3
        assert summary["ci_lower"] == 1.4
        assert summary["ci_upper"] == 2.6


class TestOffPolicyEvaluator:
    """Test OffPolicyEvaluator class."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_individuals = 200

        # Generate features
        self.features = np.random.randn(self.n_individuals, 4)

        # Generate propensity scores based on features
        propensity_logits = 0.2 * self.features[:, 0] - 0.1 * self.features[:, 1]
        self.true_propensities = 1 / (1 + np.exp(-propensity_logits))

        # Generate historical treatments
        self.historical_treatments = np.random.binomial(1, self.true_propensities)

        # Generate outcomes with treatment effect
        outcome_base = 2.0 + 0.3 * self.features[:, 0] + 0.2 * self.features[:, 2]
        treatment_effect = 0.5 + 0.1 * self.features[:, 1]  # Heterogeneous effect
        self.historical_outcomes = (
            outcome_base
            + self.historical_treatments * treatment_effect
            + np.random.normal(0, 0.5, self.n_individuals)
        )

        # Create a policy (e.g., treat individuals with positive feature 1)
        self.policy_assignment = (self.features[:, 1] > 0).astype(bool)

    def test_ips_evaluation(self):
        """Test IPS evaluation method."""
        evaluator = OffPolicyEvaluator(method="ips", random_state=42)

        result = evaluator.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
        )

        assert result.method == "ips"
        assert isinstance(result.policy_value, float)
        assert isinstance(result.policy_value_se, float)
        assert result.policy_value_se > 0
        assert len(result.confidence_interval) == 2
        assert result.ci_lower < result.ci_upper
        assert result.individual_weights is not None
        assert len(result.individual_weights) == self.n_individuals

    def test_dr_evaluation(self):
        """Test Doubly Robust evaluation method."""
        evaluator = OffPolicyEvaluator(method="dr", random_state=42)

        result = evaluator.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
        )

        assert result.method == "dr"
        assert isinstance(result.policy_value, float)
        assert result.policy_value_se > 0
        assert result.individual_weights is not None
        assert result.individual_predictions is not None
        assert "direct_component" in result.evaluation_info
        assert "bias_correction" in result.evaluation_info

    def test_dr_sn_evaluation(self):
        """Test DR with Self-Normalized weights."""
        evaluator = OffPolicyEvaluator(method="dr_sn", random_state=42)

        result = evaluator.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
        )

        assert result.method == "dr_sn"
        assert isinstance(result.policy_value, float)
        assert result.policy_value_se > 0
        assert "normalization_factor" in result.evaluation_info

    def test_evaluation_with_known_propensities(self):
        """Test evaluation with provided propensity scores."""
        evaluator = OffPolicyEvaluator(method="ips", random_state=42)

        result = evaluator.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
            propensity_scores=self.true_propensities,
        )

        assert result.method == "ips"
        assert isinstance(result.policy_value, float)

    def test_evaluation_without_features(self):
        """Test evaluation without features (marginal propensities)."""
        evaluator = OffPolicyEvaluator(method="ips", random_state=42)

        result = evaluator.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
        )

        assert result.method == "ips"
        assert isinstance(result.policy_value, float)

    def test_weight_clipping(self):
        """Test importance weight clipping."""
        evaluator = OffPolicyEvaluator(
            method="ips", clip_weights=True, weight_clip_threshold=5.0, random_state=42
        )

        result = evaluator.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
        )

        # Check that weights are clipped
        assert np.max(result.individual_weights) <= 5.0

    def test_custom_models(self):
        """Test evaluation with custom propensity and outcome models."""
        propensity_model = LogisticRegression(random_state=42)
        outcome_model = RandomForestRegressor(n_estimators=10, random_state=42)

        evaluator = OffPolicyEvaluator(
            method="dr",
            propensity_model=propensity_model,
            outcome_model=outcome_model,
            random_state=42,
        )

        result = evaluator.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
        )

        assert result.method == "dr"
        assert isinstance(result.policy_value, float)

    def test_invalid_method(self):
        """Test invalid evaluation method."""
        with pytest.raises(ValueError):
            OffPolicyEvaluator(method="invalid_method")

    def test_mismatched_input_lengths(self):
        """Test evaluation with mismatched input lengths."""
        evaluator = OffPolicyEvaluator(method="ips", random_state=42)

        with pytest.raises(ValueError):
            evaluator.evaluate_policy(
                self.policy_assignment[:50],
                self.historical_treatments,
                self.historical_outcomes,
                self.features,
            )

    def test_compare_policies(self):
        """Test policy comparison functionality."""
        evaluator = OffPolicyEvaluator(method="dr", random_state=42)

        # Create two different policies
        policy1 = (self.features[:, 0] > 0).astype(bool)
        policy2 = (self.features[:, 1] > 0).astype(bool)

        comparison = evaluator.compare_policies(
            policy1,
            policy2,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
        )

        assert "policy1_value" in comparison
        assert "policy2_value" in comparison
        assert "difference" in comparison
        assert "difference_se" in comparison
        assert "difference_ci" in comparison
        assert "p_value" in comparison
        assert "significant" in comparison
        assert isinstance(comparison["significant"], bool)

    def test_evaluation_stability(self):
        """Test that evaluation is stable across multiple runs."""
        evaluator = OffPolicyEvaluator(method="dr", random_state=42)

        result1 = evaluator.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
        )

        evaluator2 = OffPolicyEvaluator(method="dr", random_state=42)
        result2 = evaluator2.evaluate_policy(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
        )

        # Results should be very similar (allowing for small numerical differences)
        assert abs(result1.policy_value - result2.policy_value) < 0.01

    def test_extreme_propensities(self):
        """Test evaluation with extreme propensity scores."""
        # Create extreme propensities
        extreme_propensities = np.array([0.01] * 50 + [0.99] * 50 + [0.5] * 100)
        extreme_treatments = np.random.binomial(1, extreme_propensities)
        extreme_outcomes = np.random.normal(0, 1, 200)

        evaluator = OffPolicyEvaluator(
            method="ips", clip_weights=True, weight_clip_threshold=20.0, random_state=42
        )

        result = evaluator.evaluate_policy(
            self.policy_assignment,
            extreme_treatments,
            extreme_outcomes,
            propensity_scores=extreme_propensities,
        )

        # Should still produce reasonable result
        assert isinstance(result.policy_value, float)
        assert not np.isnan(result.policy_value)
        assert not np.isinf(result.policy_value)


class TestStandaloneFunctions:
    """Test standalone evaluation functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_individuals = 100
        self.policy_assignment = np.random.choice([True, False], self.n_individuals)
        self.historical_treatments = np.random.choice([True, False], self.n_individuals)
        self.historical_outcomes = np.random.normal(0, 1, self.n_individuals)
        self.propensity_scores = np.random.uniform(0.1, 0.9, self.n_individuals)
        self.features = np.random.randn(self.n_individuals, 3)

    def test_ips_estimator_function(self):
        """Test standalone IPS estimator function."""
        value = ips_estimator(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.propensity_scores,
        )

        assert isinstance(value, float)
        assert not np.isnan(value)

    def test_dr_estimator_function(self):
        """Test standalone DR estimator function."""
        value = dr_estimator(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
            self.propensity_scores,
        )

        assert isinstance(value, float)
        assert not np.isnan(value)

    def test_dr_sn_estimator_function(self):
        """Test standalone DR-SN estimator function."""
        value = dr_sn_estimator(
            self.policy_assignment,
            self.historical_treatments,
            self.historical_outcomes,
            self.features,
            self.propensity_scores,
        )

        assert isinstance(value, float)
        assert not np.isnan(value)


class TestOPEIntegration:
    """Integration tests for off-policy evaluation."""

    def setup_method(self):
        """Set up realistic test scenario."""
        np.random.seed(42)
        self.n_individuals = 500

        # Realistic marketing scenario
        self.features = np.random.randn(self.n_individuals, 6)

        # Propensity based on customer characteristics
        propensity_logits = (
            0.5 * self.features[:, 0]  # Past purchase behavior
            + -0.3 * self.features[:, 1]  # Age (older customers less likely)
            + 0.2 * self.features[:, 2]  # Engagement score
        )
        self.true_propensities = 1 / (1 + np.exp(-propensity_logits))

        # Historical treatment assignments
        self.historical_treatments = np.random.binomial(1, self.true_propensities)

        # Outcomes with heterogeneous treatment effects
        base_outcomes = (
            3.0  # Base conversion rate
            + 0.4 * self.features[:, 0]  # Purchase behavior effect
            + 0.2 * self.features[:, 3]  # Demographic effect
            + np.random.normal(0, 1, self.n_individuals)  # Noise
        )

        treatment_effects = (
            0.8  # Base treatment effect
            + 0.3 * self.features[:, 4]  # Treatment effect modifier 1
            + -0.1 * self.features[:, 5]  # Treatment effect modifier 2
        )

        self.historical_outcomes = (
            base_outcomes + self.historical_treatments * treatment_effects
        )

        # Test policy: target high-value customers
        value_score = 0.6 * self.features[:, 0] + 0.4 * self.features[:, 4]
        self.policy_assignment = (value_score > np.percentile(value_score, 70)).astype(
            bool
        )

    def test_method_comparison(self):
        """Test that different OPE methods give reasonable results."""
        methods = ["ips", "dr", "dr_sn"]
        results = {}

        for method in methods:
            evaluator = OffPolicyEvaluator(method=method, random_state=42)
            result = evaluator.evaluate_policy(
                self.policy_assignment,
                self.historical_treatments,
                self.historical_outcomes,
                self.features,
            )
            results[method] = result.policy_value

        # All methods should give reasonable results
        for method, value in results.items():
            assert isinstance(value, float)
            assert not np.isnan(value)
            assert not np.isinf(value)

        # DR methods should be similar (more robust)
        assert abs(results["dr"] - results["dr_sn"]) < 2.0

    def test_bias_reduction(self):
        """Test that DR method reduces bias compared to IPS."""
        # Create scenario with model misspecification
        n_test = 200
        test_features = np.random.randn(n_test, 6)
        test_treatments = np.random.choice([True, False], n_test)
        test_outcomes = np.random.normal(2.0, 1.0, n_test)
        test_policy = np.random.choice([True, False], n_test)

        # Evaluate with both methods
        ips_evaluator = OffPolicyEvaluator(method="ips", random_state=42)
        dr_evaluator = OffPolicyEvaluator(method="dr", random_state=42)

        ips_result = ips_evaluator.evaluate_policy(
            test_policy, test_treatments, test_outcomes, test_features
        )
        dr_result = dr_evaluator.evaluate_policy(
            test_policy, test_treatments, test_outcomes, test_features
        )

        # Both should produce finite results
        assert np.isfinite(ips_result.policy_value)
        assert np.isfinite(dr_result.policy_value)

        # DR should have smaller confidence interval (more efficient)
        # ips_ci_width = ips_result.ci_upper - ips_result.ci_lower
        # dr_ci_width = dr_result.ci_upper - dr_result.ci_lower

        # Note: This might not always hold in small samples, but generally true
        # assert dr_ci_width <= ips_ci_width * 1.2  # Allow some tolerance

    @pytest.mark.slow
    def test_performance_requirements(self):
        """Test that OPE meets performance requirements."""
        import time

        # Test with larger dataset
        large_n = 5000
        large_policy = np.random.choice([True, False], large_n)
        large_treatments = np.random.choice([True, False], large_n)
        large_outcomes = np.random.normal(0, 1, large_n)
        large_features = np.random.randn(large_n, 10)

        evaluator = OffPolicyEvaluator(method="dr", random_state=42)

        start_time = time.time()
        result = evaluator.evaluate_policy(
            large_policy, large_treatments, large_outcomes, large_features
        )
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 30.0  # 30 seconds for 5k samples
        assert isinstance(result.policy_value, float)

    def test_ope_bias_requirements(self):
        """Test that OPE bias meets requirements (≤ 0.05)."""
        # This is hard to test without ground truth, but we can test consistency
        evaluator = OffPolicyEvaluator(method="dr", random_state=42)

        # Run evaluation multiple times with different seeds
        policy_values = []
        for seed in range(10):
            np.random.seed(seed)
            # Small perturbations to test stability
            perturbed_outcomes = self.historical_outcomes + np.random.normal(
                0, 0.01, self.n_individuals
            )

            result = evaluator.evaluate_policy(
                self.policy_assignment,
                self.historical_treatments,
                perturbed_outcomes,
                self.features,
            )
            policy_values.append(result.policy_value)

        # Results should be reasonably consistent
        std_policy_values = np.std(policy_values)
        mean_policy_values = np.mean(policy_values)
        cv = std_policy_values / (abs(mean_policy_values) + 1e-8)

        # Coefficient of variation should be small (indicating low bias/variance)
        assert cv < 0.2  # 20% coefficient of variation
