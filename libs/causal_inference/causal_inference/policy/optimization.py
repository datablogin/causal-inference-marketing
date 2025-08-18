"""Policy optimization for treatment assignment.

This module provides methods to optimize treatment assignment policies based on
predicted heterogeneous treatment effects (CATE/uplift), including:
- Greedy top-k selection
- Budget-constrained optimization using Integer Linear Programming
- Multi-objective optimization with fairness constraints
- Dynamic policy learning with contextual information

Classes:
    PolicyOptimizer: Main policy optimization engine
    PolicyResult: Results container for policy optimization

Functions:
    greedy_policy: Simple greedy selection by uplift ranking
    budget_constrained_policy: ILP-based optimization under budget constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


__all__ = [
    "PolicyOptimizer",
    "PolicyResult",
    "greedy_policy",
    "budget_constrained_policy",
]


@dataclass
class PolicyResult:
    """Results container for policy optimization.

    Attributes:
        treatment_assignment: Binary array indicating treatment assignment
        expected_value: Expected value of the policy
        total_cost: Total cost of implementing the policy
        policy_type: Type of optimization used
        optimization_info: Additional optimization details
        individual_uplifts: Predicted uplift for each individual
        individual_costs: Cost for treating each individual
        constraints_satisfied: Whether all constraints were satisfied
    """

    treatment_assignment: NDArray[np.bool_]
    expected_value: float
    total_cost: float
    policy_type: str
    optimization_info: dict[str, Any] = field(default_factory=dict)
    individual_uplifts: NDArray[np.floating] | None = None
    individual_costs: NDArray[np.floating] | None = None
    constraints_satisfied: bool = True

    @property
    def treatment_rate(self) -> float:
        """Calculate treatment rate."""
        return np.mean(self.treatment_assignment)

    @property
    def n_treated(self) -> int:
        """Number of individuals treated."""
        return int(np.sum(self.treatment_assignment))

    def get_policy_summary(self) -> dict[str, Any]:
        """Get summary statistics for the policy."""
        return {
            "policy_type": self.policy_type,
            "n_treated": self.n_treated,
            "treatment_rate": self.treatment_rate,
            "expected_value": self.expected_value,
            "total_cost": self.total_cost,
            "value_per_cost": self.expected_value / max(self.total_cost, 1e-8),
            "constraints_satisfied": self.constraints_satisfied,
        }


class PolicyOptimizer:
    """Policy optimization engine for treatment assignment.

    This class provides methods to optimize treatment assignment policies
    based on predicted treatment effects and various constraints.

    Args:
        optimization_method: Method to use ('greedy', 'ilp', 'multi_objective')
        solver: Solver to use for ILP (when available)
        verbose: Whether to print optimization progress
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        optimization_method: str = "greedy",
        solver: str = "ECOS",
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.optimization_method = optimization_method
        self.solver = solver
        self.verbose = verbose
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Validate method
        valid_methods = ["greedy", "ilp", "multi_objective", "dynamic"]
        if optimization_method not in valid_methods:
            raise ValueError(f"optimization_method must be one of {valid_methods}")

    def optimize_policy(
        self,
        uplifts: NDArray[np.floating],
        costs: NDArray[np.floating] | None = None,
        budget: float | None = None,
        max_treatment_rate: float | None = None,
        fairness_constraints: dict[str, Any] | None = None,
        features: NDArray[np.floating] | None = None,
    ) -> PolicyResult:
        """Optimize treatment assignment policy.

        Args:
            uplifts: Predicted treatment effects for each individual
            costs: Cost of treating each individual (default: all 1.0)
            budget: Maximum budget constraint
            max_treatment_rate: Maximum fraction of individuals to treat
            fairness_constraints: Fairness constraints (group-based)
            features: Individual features for dynamic policies

        Returns:
            PolicyResult: Optimized policy and metadata
        """
        n_individuals = len(uplifts)

        # Validate inputs
        if n_individuals == 0:
            raise ValueError("uplifts cannot be empty")

        # Set default costs
        if costs is None:
            costs = np.ones(n_individuals)

        if len(costs) != n_individuals:
            raise ValueError("uplifts and costs must have same length")

        if self.optimization_method == "greedy":
            return self._optimize_greedy(uplifts, costs, budget, max_treatment_rate)
        elif self.optimization_method == "ilp":
            return self._optimize_ilp(
                uplifts, costs, budget, max_treatment_rate, fairness_constraints
            )
        elif self.optimization_method == "multi_objective":
            return self._optimize_multi_objective(
                uplifts, costs, budget, max_treatment_rate, fairness_constraints
            )
        elif self.optimization_method == "dynamic":
            if features is None:
                raise ValueError("features required for dynamic policy optimization")
            return self._optimize_dynamic(
                uplifts, costs, features, budget, max_treatment_rate
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

    def _optimize_greedy(
        self,
        uplifts: NDArray[np.floating],
        costs: NDArray[np.floating],
        budget: float | None = None,
        max_treatment_rate: float | None = None,
    ) -> PolicyResult:
        """Greedy optimization by uplift-to-cost ratio."""
        n_individuals = len(uplifts)

        # Calculate uplift-to-cost ratios with numerical stability
        ratios = uplifts / (costs + 1e-8)

        # Sort by ratio (descending)
        sorted_indices = np.argsort(-ratios)

        # Initialize treatment assignment
        treatment = np.zeros(n_individuals, dtype=bool)
        total_cost = 0.0
        total_value = 0.0

        # Greedy selection
        for idx in sorted_indices:
            candidate_cost = total_cost + costs[idx]
            candidate_rate = (np.sum(treatment) + 1) / n_individuals

            # Check constraints
            budget_ok = budget is None or candidate_cost <= budget
            rate_ok = max_treatment_rate is None or candidate_rate <= max_treatment_rate

            if budget_ok and rate_ok and ratios[idx] > 0:
                treatment[idx] = True
                total_cost = candidate_cost
                total_value += uplifts[idx]
            else:
                break

        return PolicyResult(
            treatment_assignment=treatment,
            expected_value=total_value,
            total_cost=total_cost,
            policy_type="greedy",
            optimization_info={
                "sorted_ratios": ratios[sorted_indices][:10],  # Top 10 ratios
                "n_considered": len(sorted_indices),
            },
            individual_uplifts=uplifts,
            individual_costs=costs,
        )

    def _optimize_ilp(
        self,
        uplifts: NDArray[np.floating],
        costs: NDArray[np.floating],
        budget: float | None = None,
        max_treatment_rate: float | None = None,
        fairness_constraints: dict[str, Any] | None = None,
    ) -> PolicyResult:
        """Integer Linear Programming optimization."""
        if not HAS_CVXPY:
            raise ImportError(
                "cvxpy is required for ILP optimization. "
                "Install with: pip install cvxpy"
            )

        n_individuals = len(uplifts)

        # Decision variables (binary)
        x = cp.Variable(n_individuals, boolean=True)

        # Objective: maximize total uplift
        objective = cp.Maximize(uplifts @ x)

        # Constraints
        constraints = []

        # Budget constraint
        if budget is not None:
            constraints.append(costs @ x <= budget)

        # Treatment rate constraint
        if max_treatment_rate is not None:
            constraints.append(cp.sum(x) <= max_treatment_rate * n_individuals)

        # Fairness constraints
        if fairness_constraints is not None:
            # Example: demographic parity
            if "group_indicators" in fairness_constraints:
                groups = fairness_constraints["group_indicators"]
                max_diff = fairness_constraints.get("max_difference", 0.1)

                for i, group1 in enumerate(groups):
                    for j, group2 in enumerate(groups[i + 1 :], i + 1):
                        # Treatment rate difference constraint
                        rate1 = cp.sum(cp.multiply(group1, x)) / cp.sum(group1)
                        rate2 = cp.sum(cp.multiply(group2, x)) / cp.sum(group2)
                        constraints.append(cp.abs(rate1 - rate2) <= max_diff)

        # Solve the problem
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.solver, verbose=self.verbose)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                # Fallback to greedy
                if self.verbose:
                    print(
                        f"ILP solver status: {problem.status}, falling back to greedy"
                    )
                return self._optimize_greedy(uplifts, costs, budget, max_treatment_rate)

            # Extract solution
            treatment = np.array(x.value, dtype=bool)
            total_value = float(problem.value or 0.0)
            total_cost = float(np.sum(costs * treatment))

            return PolicyResult(
                treatment_assignment=treatment,
                expected_value=total_value,
                total_cost=total_cost,
                policy_type="ilp",
                optimization_info={
                    "solver_status": problem.status,
                    "solve_time": getattr(problem, "solver_stats", {}).get(
                        "solve_time", 0
                    ),
                    "n_constraints": len(constraints),
                },
                individual_uplifts=uplifts,
                individual_costs=costs,
                constraints_satisfied=problem.status == "optimal",
            )

        except Exception as e:
            if self.verbose:
                print(f"ILP optimization failed: {e}, falling back to greedy")
            return self._optimize_greedy(uplifts, costs, budget, max_treatment_rate)

    def _optimize_multi_objective(
        self,
        uplifts: NDArray[np.floating],
        costs: NDArray[np.floating],
        budget: float | None = None,
        max_treatment_rate: float | None = None,
        fairness_constraints: dict[str, Any] | None = None,
    ) -> PolicyResult:
        """Multi-objective optimization balancing value, cost, and fairness."""
        # For now, use weighted combination approach
        # In future, could use Pareto optimization

        fairness_weight = 0.1
        if fairness_constraints is not None:
            fairness_weight = fairness_constraints.get("fairness_weight", 0.1)

        # Normalize uplifts and costs for fair weighting
        norm_uplifts = uplifts / (np.std(uplifts) + 1e-8)
        norm_costs = costs / (np.std(costs) + 1e-8)

        # Multi-objective score (higher is better)
        scores = norm_uplifts - 0.5 * norm_costs

        # Add fairness penalty if needed
        if (
            fairness_constraints is not None
            and "group_indicators" in fairness_constraints
        ):
            groups = fairness_constraints["group_indicators"]
            # Penalize individuals in over-represented groups
            for group in groups:
                group_mean_uplift = np.mean(uplifts[group])
                overall_mean_uplift = np.mean(uplifts)
                if group_mean_uplift > overall_mean_uplift:
                    scores[group] -= fairness_weight * (
                        group_mean_uplift - overall_mean_uplift
                    )

        # Use greedy selection on multi-objective scores
        return self._optimize_greedy_by_scores(
            scores, uplifts, costs, budget, max_treatment_rate
        )

    def _optimize_dynamic(
        self,
        uplifts: NDArray[np.floating],
        costs: NDArray[np.floating],
        features: NDArray[np.floating],
        budget: float | None = None,
        max_treatment_rate: float | None = None,
    ) -> PolicyResult:
        """Dynamic policy optimization using contextual information."""
        # Simplified dynamic policy: adjust uplifts based on features
        # In practice, would use more sophisticated contextual bandits

        # Use feature-weighted uplifts
        feature_importance = np.random.uniform(0.5, 1.5, features.shape[1])
        feature_scores = np.dot(features, feature_importance)

        # Combine with original uplifts
        adjusted_uplifts = uplifts * (1 + 0.2 * feature_scores)

        # Use greedy optimization with adjusted uplifts
        result = self._optimize_greedy(
            adjusted_uplifts, costs, budget, max_treatment_rate
        )
        result.policy_type = "dynamic"
        result.optimization_info["feature_adjustment"] = True
        result.optimization_info["feature_importance"] = feature_importance

        return result

    def _optimize_greedy_by_scores(
        self,
        scores: NDArray[np.floating],
        uplifts: NDArray[np.floating],
        costs: NDArray[np.floating],
        budget: float | None = None,
        max_treatment_rate: float | None = None,
    ) -> PolicyResult:
        """Helper method for greedy selection by arbitrary scores."""
        n_individuals = len(scores)

        # Sort by scores (descending)
        sorted_indices = np.argsort(-scores)

        # Initialize treatment assignment
        treatment = np.zeros(n_individuals, dtype=bool)
        total_cost = 0.0
        total_value = 0.0

        # Greedy selection
        for idx in sorted_indices:
            candidate_cost = total_cost + costs[idx]
            candidate_rate = (np.sum(treatment) + 1) / n_individuals

            # Check constraints
            budget_ok = budget is None or candidate_cost <= budget
            rate_ok = max_treatment_rate is None or candidate_rate <= max_treatment_rate

            if budget_ok and rate_ok and scores[idx] > 0:
                treatment[idx] = True
                total_cost = candidate_cost
                total_value += uplifts[idx]
            else:
                break

        return PolicyResult(
            treatment_assignment=treatment,
            expected_value=total_value,
            total_cost=total_cost,
            policy_type="multi_objective",
            optimization_info={
                "sorted_scores": scores[sorted_indices][:10],
                "n_considered": len(sorted_indices),
            },
            individual_uplifts=uplifts,
            individual_costs=costs,
        )


def greedy_policy(
    uplifts: NDArray[np.floating],
    k: int | None = None,
    treatment_rate: float | None = None,
) -> NDArray[np.bool_]:
    """Simple greedy policy: treat top-k individuals by uplift.

    Args:
        uplifts: Predicted treatment effects
        k: Number of individuals to treat
        treatment_rate: Fraction of individuals to treat (alternative to k)

    Returns:
        Binary treatment assignment array
    """
    n_individuals = len(uplifts)

    if k is None and treatment_rate is None:
        raise ValueError("Must specify either k or treatment_rate")

    if k is None:
        k = int(treatment_rate * n_individuals)

    k = min(k, n_individuals)

    # Get top-k indices
    top_k_indices = np.argsort(-uplifts)[:k]

    # Create treatment assignment
    treatment = np.zeros(n_individuals, dtype=bool)
    treatment[top_k_indices] = True

    return treatment


def budget_constrained_policy(
    uplifts: NDArray[np.floating],
    costs: NDArray[np.floating],
    budget: float,
) -> NDArray[np.bool_]:
    """Budget-constrained policy using greedy uplift-to-cost ratio.

    Args:
        uplifts: Predicted treatment effects
        costs: Cost of treating each individual
        budget: Maximum budget

    Returns:
        Binary treatment assignment array
    """
    optimizer = PolicyOptimizer(optimization_method="greedy")
    result = optimizer.optimize_policy(uplifts, costs, budget=budget)
    return result.treatment_assignment
