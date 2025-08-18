"""Tests for policy integration functionality."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from causal_inference.estimators.meta_learners import TLearner
from causal_inference.policy.integration import (
    CATEPolicyResult,
    PolicyIntegrator,
    extract_treatment_effects,
    integrate_with_cate,
)


class MockCATEEstimator:
    """Mock CATE estimator for testing."""

    def __init__(self, effects=None):
        self.effects = effects
        self.last_result_ = None

    def predict(self, x):
        if self.effects is not None:
            return self.effects
        return np.random.normal(0.5, 0.2, len(x))

    def predict_cate(self, x):
        return self.predict(x)


class TestCATEPolicyResult:
    """Test CATEPolicyResult dataclass."""

    def test_cate_policy_result_creation(self):
        """Test creation of CATEPolicyResult."""
        from causal_inference.estimators.meta_learners import CATEResult
        from causal_inference.policy.optimization import PolicyResult

        # Create mock results
        cate_result = CATEResult(
            ate=0.5,
            confidence_interval=(0.4, 0.6),
            cate_estimates=np.array([0.4, 0.6, 0.3, 0.7]),
            method="T-learner",
        )

        policy_result = PolicyResult(
            treatment_assignment=np.array([True, True, False, True]),
            expected_value=1.7,
            total_cost=3.0,
            policy_type="greedy",
            individual_uplifts=np.array([0.4, 0.6, 0.3, 0.7]),
        )

        integrated_result = CATEPolicyResult(
            cate_result=cate_result,
            policy_result=policy_result,
        )

        assert np.array_equal(
            integrated_result.treatment_assignment, np.array([True, True, False, True])
        )
        assert np.array_equal(
            integrated_result.individual_treatment_effects,
            np.array([0.4, 0.6, 0.3, 0.7]),
        )
        assert integrated_result.policy_value_estimate == 1.7

    def test_policy_summary(self):
        """Test policy summary generation."""
        from causal_inference.estimators.meta_learners import CATEResult
        from causal_inference.policy.optimization import PolicyResult

        cate_result = CATEResult(
            ate=0.5,
            confidence_interval=(0.4, 0.6),
            cate_estimates=np.array([0.4, 0.6]),
            method="S-learner",
        )

        policy_result = PolicyResult(
            treatment_assignment=np.array([True, False]),
            expected_value=0.4,
            total_cost=1.0,
            policy_type="ilp",
        )

        integrated_result = CATEPolicyResult(
            cate_result=cate_result,
            policy_result=policy_result,
        )

        summary = integrated_result.get_policy_summary()
        assert summary["cate_method"] == "S-learner"
        assert summary["policy_type"] == "ilp"
        assert summary["n_treated"] == 1
        assert summary["treatment_rate"] == 0.5


class TestPolicyIntegrator:
    """Test PolicyIntegrator class."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_individuals = 100
        self.features = np.random.randn(self.n_individuals, 4)
        self.historical_treatments = np.random.choice([True, False], self.n_individuals)
        self.historical_outcomes = np.random.normal(0, 1, self.n_individuals)

        # Create mock CATE estimator
        self.effects = np.random.normal(0.5, 0.3, self.n_individuals)
        self.cate_estimator = MockCATEEstimator(effects=self.effects)

    @pytest.mark.fast
    def test_integration_basic(self):
        """Test basic CATE integration."""
        integrator = PolicyIntegrator(verbose=False)

        result = integrator.integrate_cate_with_policy(
            self.cate_estimator,
            self.features,
            evaluate_policy=False,
        )

        assert isinstance(result, CATEPolicyResult)
        assert len(result.treatment_assignment) == self.n_individuals
        assert result.policy_result.individual_uplifts is not None
        assert len(result.policy_result.individual_uplifts) == self.n_individuals

    @pytest.mark.fast
    def test_integration_with_ope(self):
        """Test integration with off-policy evaluation."""
        integrator = PolicyIntegrator(verbose=False)

        result = integrator.integrate_cate_with_policy(
            self.cate_estimator,
            self.features,
            self.historical_treatments,
            self.historical_outcomes,
            evaluate_policy=True,
        )

        assert isinstance(result, CATEPolicyResult)
        assert result.ope_result is not None
        assert isinstance(result.ope_result.policy_value, float)

    @pytest.mark.fast
    def test_integration_with_constraints(self):
        """Test integration with budget and rate constraints."""
        integrator = PolicyIntegrator(verbose=False)
        costs = np.random.uniform(0.5, 2.0, self.n_individuals)

        result = integrator.integrate_cate_with_policy(
            self.cate_estimator,
            self.features,
            costs=costs,
            budget=50.0,
            max_treatment_rate=0.3,
            evaluate_policy=False,
        )

        assert result.policy_result.total_cost <= 50.0
        assert result.policy_result.treatment_rate <= 0.3
        assert len(result.policy_result.individual_costs) == self.n_individuals

    @pytest.mark.slow
    def test_integration_with_real_estimator(self):
        """Test integration with real CATE estimator."""
        # Create synthetic data for T-learner
        np.random.seed(42)
        n_train = 200
        X_train = np.random.randn(n_train, 3)
        T_train = np.random.choice([0, 1], n_train)

        # Generate outcomes with treatment effect
        base_outcome = 1.0 + 0.5 * X_train[:, 0] + 0.3 * X_train[:, 1]
        treatment_effect = 0.4 + 0.2 * X_train[:, 2]
        Y_train = (
            base_outcome
            + T_train * treatment_effect
            + np.random.normal(0, 0.5, n_train)
        )

        # Train T-learner
        from causal_inference.core import CovariateData, OutcomeData, TreatmentData

        treatment_data = TreatmentData(values=T_train, treatment_type="binary")
        outcome_data = OutcomeData(values=Y_train, outcome_type="continuous")
        covariate_data = CovariateData(values=X_train)

        t_learner = TLearner(
            control_model=RandomForestRegressor(n_estimators=10, random_state=42),
            treatment_model=RandomForestRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        t_learner.fit(treatment_data, outcome_data, covariate_data)

        # Test integration
        integrator = PolicyIntegrator(verbose=False)
        test_features = np.random.randn(50, 3)

        result = integrator.integrate_cate_with_policy(
            t_learner,
            test_features,
            evaluate_policy=False,
        )

        assert isinstance(result, CATEPolicyResult)
        assert len(result.treatment_assignment) == 50
        assert result.integration_info["cate_estimator_type"] == "TLearner"

    def test_compare_cate_policies(self):
        """Test comparison of multiple CATE estimators."""
        # Create two different mock estimators
        estimator1 = MockCATEEstimator(
            effects=np.random.normal(0.6, 0.2, self.n_individuals)
        )
        estimator2 = MockCATEEstimator(
            effects=np.random.normal(0.4, 0.3, self.n_individuals)
        )

        integrator = PolicyIntegrator(verbose=False)

        comparison = integrator.compare_cate_policies(
            [estimator1, estimator2],
            ["Estimator1", "Estimator2"],
            self.features,
            self.historical_treatments,
            self.historical_outcomes,
        )

        assert "integrated_results" in comparison
        assert "policy_comparisons" in comparison
        assert "policy_values" in comparison
        assert "ranked_estimators" in comparison
        assert "best_estimator" in comparison

        assert len(comparison["integrated_results"]) == 2
        assert "Estimator1" in comparison["integrated_results"]
        assert "Estimator2" in comparison["integrated_results"]

    @pytest.mark.slow
    def test_simulate_cate_policy(self):
        """Test CATE policy simulation."""

        def data_generator(**kwargs):
            n_samples = kwargs.get("n_samples", 100)
            return {
                "features": np.random.randn(n_samples, 4),
                "historical_treatments": np.random.choice([True, False], n_samples),
                "historical_outcomes": np.random.normal(0, 1, n_samples),
                "costs": np.ones(n_samples),
            }

        integrator = PolicyIntegrator(verbose=False)

        result = integrator.simulate_cate_policy(
            self.cate_estimator,
            data_generator,
            n_simulations=10,  # Small number for testing
        )

        assert result.simulation_type == "monte_carlo"
        assert len(result.policy_values) <= 10
        assert isinstance(result.mean_policy_value, float)

    def test_extract_treatment_effects(self):
        """Test treatment effect extraction."""
        integrator = PolicyIntegrator()

        # Test with predict method
        effects = integrator._extract_treatment_effects(
            self.cate_estimator, self.features
        )
        assert len(effects) == self.n_individuals
        assert isinstance(effects, np.ndarray)

        # Test with predict_cate method
        class MockCATEWithPredictCATE:
            def predict_cate(self, x):
                return np.random.normal(0.3, 0.1, len(x))

        estimator_cate = MockCATEWithPredictCATE()
        effects_cate = integrator._extract_treatment_effects(
            estimator_cate, self.features
        )
        assert len(effects_cate) == self.n_individuals

    def test_policy_recommendations(self):
        """Test policy recommendation generation."""
        integrator = PolicyIntegrator(verbose=False)

        result = integrator.integrate_cate_with_policy(
            self.cate_estimator,
            self.features,
            evaluate_policy=False,
        )

        assert result.policy_recommendations is not None
        recommendations = result.policy_recommendations

        assert "treatment_assignment" in recommendations
        assert "targeting_insights" in recommendations
        assert "implementation" in recommendations

        # Check treatment assignment details
        treatment_info = recommendations["treatment_assignment"]
        assert "n_treated" in treatment_info
        assert "treatment_rate" in treatment_info
        assert "total_cost" in treatment_info

        # Check targeting insights
        insights = recommendations["targeting_insights"]
        assert "high_value_segments" in insights
        assert "cost_efficiency" in insights
        assert "uplift_distribution" in insights


class TestStandaloneFunctions:
    """Test standalone integration functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.features = np.random.randn(50, 3)
        self.effects = np.random.normal(0.4, 0.2, 50)
        self.estimator = MockCATEEstimator(effects=self.effects)

    def test_integrate_with_cate_function(self):
        """Test standalone integrate_with_cate function."""
        result = integrate_with_cate(
            self.estimator,
            self.features,
            optimization_method="greedy",
            budget=25.0,
        )

        assert result.policy_type == "greedy"
        assert result.total_cost <= 25.0
        assert len(result.treatment_assignment) == 50

    def test_extract_treatment_effects_function(self):
        """Test standalone extract_treatment_effects function."""
        effects = extract_treatment_effects(self.estimator, self.features)

        assert len(effects) == 50
        assert isinstance(effects, np.ndarray)
        np.testing.assert_array_equal(effects, self.effects)


class TestPolicyIntegratorEdgeCases:
    """Test edge cases for PolicyIntegrator."""

    def test_estimator_without_predict_method(self):
        """Test with estimator that has no standard predict method."""

        class BadEstimator:
            def unknown_method(self, x):
                return np.random.normal(0, 1, len(x))

        bad_estimator = BadEstimator()
        integrator = PolicyIntegrator(verbose=False)
        features = np.random.randn(10, 2)

        # Should fallback to random effects
        result = integrator.integrate_cate_with_policy(
            bad_estimator,
            features,
            evaluate_policy=False,
        )

        assert isinstance(result, CATEPolicyResult)
        assert len(result.treatment_assignment) == 10

    def test_ope_failure_handling(self):
        """Test handling of OPE failures."""
        # Create data that might cause OPE to fail
        features = np.random.randn(10, 2)
        treatments = np.array([True] * 10)  # All treated
        outcomes = np.random.normal(0, 1, 10)

        integrator = PolicyIntegrator(verbose=False)
        estimator = MockCATEEstimator(effects=np.random.normal(0.5, 0.1, 10))

        result = integrator.integrate_cate_with_policy(
            estimator,
            features,
            treatments,
            outcomes,
            evaluate_policy=True,
        )

        # Should handle gracefully even if OPE fails
        assert isinstance(result, CATEPolicyResult)
        # OPE might fail with all-treated data, but that's okay

    def test_zero_budget_constraint(self):
        """Test with zero budget constraint."""
        integrator = PolicyIntegrator(verbose=False)
        features = np.random.randn(20, 3)
        estimator = MockCATEEstimator(effects=np.random.normal(1.0, 0.2, 20))
        costs = np.ones(20)

        result = integrator.integrate_cate_with_policy(
            estimator,
            features,
            costs=costs,
            budget=0.0,
            evaluate_policy=False,
        )

        # Should treat nobody with zero budget
        assert result.policy_result.n_treated == 0
        assert result.policy_result.total_cost == 0.0

    def test_high_treatment_rate_constraint(self):
        """Test with very high treatment rate constraint."""
        integrator = PolicyIntegrator(verbose=False)
        features = np.random.randn(30, 3)
        estimator = MockCATEEstimator(effects=np.random.normal(0.8, 0.1, 30))

        result = integrator.integrate_cate_with_policy(
            estimator,
            features,
            max_treatment_rate=0.9,
            evaluate_policy=False,
        )

        # Should treat most people with high rate constraint and positive effects
        assert result.policy_result.treatment_rate <= 0.9
        assert result.policy_result.n_treated >= 20  # Most should be treated


class TestPolicyIntegratorPerformance:
    """Test performance aspects of PolicyIntegrator."""

    @pytest.mark.slow
    def test_large_scale_integration(self):
        """Test integration with larger datasets."""
        import time

        np.random.seed(42)
        n_large = 1000
        features = np.random.randn(n_large, 5)
        effects = np.random.normal(0.5, 0.3, n_large)
        estimator = MockCATEEstimator(effects=effects)

        integrator = PolicyIntegrator(verbose=False)

        start_time = time.time()
        result = integrator.integrate_cate_with_policy(
            estimator,
            features,
            evaluate_policy=False,
        )
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 5.0  # 5 seconds for 1k samples
        assert len(result.treatment_assignment) == n_large

    def test_integration_memory_efficiency(self):
        """Test memory efficiency of integration."""
        # This is more of a smoke test to ensure no obvious memory leaks
        integrator = PolicyIntegrator(verbose=False)

        for i in range(10):
            features = np.random.randn(100, 3)
            estimator = MockCATEEstimator(effects=np.random.normal(0.5, 0.2, 100))

            result = integrator.integrate_cate_with_policy(
                estimator,
                features,
                evaluate_policy=False,
            )

            # Just ensure it completes without issues
            assert isinstance(result, CATEPolicyResult)

            # Clean up
            del result, features, estimator
