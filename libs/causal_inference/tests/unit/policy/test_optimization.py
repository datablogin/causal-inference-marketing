"""Tests for policy optimization functionality."""

import numpy as np
import pytest

from causal_inference.policy.optimization import (
    PolicyOptimizer,
    PolicyResult,
    budget_constrained_policy,
    greedy_policy,
)


class TestPolicyResult:
    """Test PolicyResult dataclass."""

    def test_policy_result_creation(self):
        """Test creation of PolicyResult."""
        treatment = np.array([True, False, True, False, True])
        uplifts = np.array([1.0, 0.5, 2.0, -0.5, 1.5])
        costs = np.array([1.0, 1.0, 2.0, 1.0, 1.5])

        result = PolicyResult(
            treatment_assignment=treatment,
            expected_value=4.5,
            total_cost=5.5,
            policy_type="greedy",
            individual_uplifts=uplifts,
            individual_costs=costs,
        )

        assert result.treatment_rate == 0.6
        assert result.n_treated == 3
        assert result.expected_value == 4.5
        assert result.total_cost == 5.5
        assert result.constraints_satisfied is True

    def test_policy_summary(self):
        """Test policy summary generation."""
        treatment = np.array([True, False, True])
        result = PolicyResult(
            treatment_assignment=treatment,
            expected_value=2.5,
            total_cost=3.0,
            policy_type="ilp",
        )

        summary = result.get_policy_summary()
        assert summary["policy_type"] == "ilp"
        assert summary["n_treated"] == 2
        assert summary["treatment_rate"] == pytest.approx(0.667, abs=0.01)
        assert summary["expected_value"] == 2.5
        assert summary["value_per_cost"] == pytest.approx(0.833, abs=0.01)


class TestPolicyOptimizer:
    """Test PolicyOptimizer class."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.uplifts = np.array([2.0, 1.5, 3.0, 0.5, -0.5, 2.5, 1.0, 0.0])
        self.costs = np.array([1.0, 1.5, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0])

    def test_greedy_optimization(self):
        """Test greedy policy optimization."""
        optimizer = PolicyOptimizer(optimization_method="greedy", random_state=42)

        result = optimizer.optimize_policy(self.uplifts, self.costs, budget=5.0)

        assert result.policy_type == "greedy"
        assert result.total_cost <= 5.0
        assert result.expected_value > 0
        assert np.sum(result.treatment_assignment) <= len(self.uplifts)

        # Check that treated individuals have positive uplift-to-cost ratios
        treated_mask = result.treatment_assignment
        if np.any(treated_mask):
            treated_ratios = self.uplifts[treated_mask] / self.costs[treated_mask]
            assert np.all(treated_ratios > 0)

    def test_greedy_with_treatment_rate_constraint(self):
        """Test greedy optimization with treatment rate constraint."""
        optimizer = PolicyOptimizer(optimization_method="greedy", random_state=42)

        result = optimizer.optimize_policy(
            self.uplifts, self.costs, max_treatment_rate=0.5
        )

        assert result.treatment_rate <= 0.5
        assert result.n_treated <= len(self.uplifts) * 0.5

    @pytest.mark.slow
    def test_ilp_optimization_without_cvxpy(self):
        """Test ILP optimization fallback when cvxpy not available."""
        # Skip this test if cvxpy is available (most environments have it)
        try:
            import cvxpy  # noqa: F401

            pytest.skip("cvxpy is available, cannot test fallback behavior")
        except ImportError:
            pass

        optimizer = PolicyOptimizer(optimization_method="ilp", random_state=42)

        # This should fall back to greedy if cvxpy import fails
        result = optimizer.optimize_policy(self.uplifts, self.costs, budget=5.0)

        assert result.policy_type in ["ilp", "greedy"]  # Could be either
        assert result.total_cost <= 5.0

    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        optimizer = PolicyOptimizer(
            optimization_method="multi_objective", random_state=42
        )

        result = optimizer.optimize_policy(self.uplifts, self.costs, budget=6.0)

        assert result.policy_type == "multi_objective"
        assert result.total_cost <= 6.0
        assert result.expected_value >= 0

    def test_dynamic_optimization(self):
        """Test dynamic policy optimization."""
        optimizer = PolicyOptimizer(optimization_method="dynamic", random_state=42)
        features = np.random.rand(len(self.uplifts), 3)

        result = optimizer.optimize_policy(
            self.uplifts, self.costs, features=features, budget=5.0
        )

        assert result.policy_type == "dynamic"
        assert result.total_cost <= 5.0
        assert "feature_adjustment" in result.optimization_info

    def test_optimization_with_fairness_constraints(self):
        """Test optimization with fairness constraints."""
        optimizer = PolicyOptimizer(
            optimization_method="multi_objective", random_state=42
        )

        # Create group indicators
        group1 = np.array([True, False, True, False, True, False, True, False])
        group2 = ~group1
        fairness_constraints = {
            "group_indicators": [group1, group2],
            "max_difference": 0.1,
            "fairness_weight": 0.2,
        }

        result = optimizer.optimize_policy(
            self.uplifts,
            self.costs,
            budget=5.0,
            fairness_constraints=fairness_constraints,
        )

        assert result.policy_type == "multi_objective"
        assert result.total_cost <= 5.0

    def test_empty_uplifts(self):
        """Test optimization with empty inputs."""
        optimizer = PolicyOptimizer(random_state=42)

        with pytest.raises(ValueError):
            optimizer.optimize_policy(np.array([]), np.array([]))

    def test_mismatched_input_lengths(self):
        """Test optimization with mismatched input lengths."""
        optimizer = PolicyOptimizer(random_state=42)

        with pytest.raises(ValueError):
            optimizer.optimize_policy(self.uplifts, self.costs[:3])

    def test_invalid_optimization_method(self):
        """Test invalid optimization method."""
        with pytest.raises(ValueError):
            PolicyOptimizer(optimization_method="invalid_method")

    def test_dynamic_without_features(self):
        """Test dynamic optimization without features."""
        optimizer = PolicyOptimizer(optimization_method="dynamic", random_state=42)

        with pytest.raises(ValueError):
            optimizer.optimize_policy(self.uplifts, self.costs)


class TestGreedyPolicy:
    """Test greedy policy function."""

    def setup_method(self):
        """Set up test data."""
        self.uplifts = np.array([2.0, 1.0, 3.0, 0.5, 2.5])

    def test_greedy_policy_with_k(self):
        """Test greedy policy with k parameter."""
        treatment = greedy_policy(self.uplifts, k=3)

        assert len(treatment) == len(self.uplifts)
        assert np.sum(treatment) == 3
        assert treatment.dtype == bool

        # Check that top-k individuals are selected
        top_k_indices = np.argsort(-self.uplifts)[:3]
        assert np.all(treatment[top_k_indices])

    def test_greedy_policy_with_treatment_rate(self):
        """Test greedy policy with treatment rate."""
        treatment = greedy_policy(self.uplifts, treatment_rate=0.4)

        assert len(treatment) == len(self.uplifts)
        assert np.sum(treatment) == 2  # 40% of 5
        assert treatment.dtype == bool

    def test_greedy_policy_without_parameters(self):
        """Test greedy policy without k or treatment_rate."""
        with pytest.raises(ValueError):
            greedy_policy(self.uplifts)

    def test_greedy_policy_k_larger_than_population(self):
        """Test greedy policy with k larger than population."""
        treatment = greedy_policy(self.uplifts, k=10)

        assert np.sum(treatment) == len(self.uplifts)  # Treat everyone


class TestBudgetConstrainedPolicy:
    """Test budget-constrained policy function."""

    def setup_method(self):
        """Set up test data."""
        self.uplifts = np.array([2.0, 1.5, 3.0, 0.5, 2.5])
        self.costs = np.array([1.0, 1.5, 2.0, 1.0, 2.0])

    def test_budget_constrained_policy(self):
        """Test budget-constrained policy."""
        treatment = budget_constrained_policy(self.uplifts, self.costs, budget=4.0)

        assert len(treatment) == len(self.uplifts)
        assert treatment.dtype == bool

        # Check that budget constraint is satisfied
        total_cost = np.sum(self.costs[treatment])
        assert total_cost <= 4.0

    def test_budget_constrained_with_zero_budget(self):
        """Test budget-constrained policy with zero budget."""
        treatment = budget_constrained_policy(self.uplifts, self.costs, budget=0.0)

        assert np.sum(treatment) == 0

    def test_budget_constrained_with_large_budget(self):
        """Test budget-constrained policy with large budget."""
        treatment = budget_constrained_policy(self.uplifts, self.costs, budget=1000.0)

        # Should treat everyone with positive uplift
        positive_uplift_mask = self.uplifts > 0
        treated_positive = treatment[positive_uplift_mask]
        assert np.all(treated_positive)


class TestPolicyOptimizerIntegration:
    """Integration tests for PolicyOptimizer."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_individuals = 100
        self.uplifts = np.random.normal(1.0, 2.0, self.n_individuals)
        self.costs = np.random.uniform(0.5, 2.0, self.n_individuals)
        self.features = np.random.rand(self.n_individuals, 5)

    @pytest.mark.slow
    def test_large_scale_optimization(self):
        """Test optimization with larger dataset."""
        optimizer = PolicyOptimizer(optimization_method="greedy", random_state=42)

        result = optimizer.optimize_policy(
            self.uplifts, self.costs, budget=50.0, max_treatment_rate=0.3
        )

        assert result.treatment_rate <= 0.3
        assert result.total_cost <= 50.0
        assert result.n_treated <= self.n_individuals * 0.3

    def test_optimization_consistency(self):
        """Test that optimization is consistent across runs."""
        optimizer = PolicyOptimizer(optimization_method="greedy", random_state=42)

        result1 = optimizer.optimize_policy(self.uplifts, self.costs, budget=30.0)
        result2 = optimizer.optimize_policy(self.uplifts, self.costs, budget=30.0)

        assert np.array_equal(
            result1.treatment_assignment, result2.treatment_assignment
        )
        assert result1.expected_value == result2.expected_value

    def test_budget_vs_treatment_rate_constraints(self):
        """Test interaction between budget and treatment rate constraints."""
        optimizer = PolicyOptimizer(optimization_method="greedy", random_state=42)

        # Very tight budget constraint (should select fewer, higher-value individuals)
        result_budget = optimizer.optimize_policy(self.uplifts, self.costs, budget=2.0)

        # Very relaxed treatment rate constraint (should select more individuals)
        result_rate = optimizer.optimize_policy(
            self.uplifts, self.costs, max_treatment_rate=0.9
        )

        # Results should be different (unless all uplifts are negative)
        budget_treated = np.sum(result_budget.treatment_assignment)
        rate_treated = np.sum(result_rate.treatment_assignment)

        # At minimum, the counts should be different due to different constraints
        assert budget_treated != rate_treated or np.sum(self.uplifts > 0) == 0

    @pytest.mark.slow
    def test_performance_requirements(self):
        """Test that optimization meets performance requirements."""
        import time

        optimizer = PolicyOptimizer(optimization_method="greedy", random_state=42)

        # Test with 10k individuals (should be < 60s as per requirements)
        large_uplifts = np.random.normal(1.0, 2.0, 10000)
        large_costs = np.random.uniform(0.5, 2.0, 10000)

        start_time = time.time()
        result = optimizer.optimize_policy(
            large_uplifts, large_costs, budget=5000.0, max_treatment_rate=0.2
        )
        end_time = time.time()

        assert end_time - start_time < 60.0  # Performance requirement
        assert result.n_treated <= 2000  # 20% of 10k
        assert result.total_cost <= 5000.0

    def test_value_optimization_quality(self):
        """Test that optimization finds good solutions."""
        optimizer = PolicyOptimizer(optimization_method="greedy", random_state=42)

        # Create scenario with clear best choices
        clear_uplifts = np.array([10.0, 8.0, 1.0, -2.0, 9.0, 0.5, -1.0, 7.0])
        clear_costs = np.ones(8)

        result = optimizer.optimize_policy(clear_uplifts, clear_costs, budget=4.0)

        # Should select the 4 highest uplifts
        expected_selection = clear_uplifts >= 7.0
        assert np.array_equal(result.treatment_assignment, expected_selection)
        assert result.expected_value == 10.0 + 8.0 + 9.0 + 7.0
