"""Policy Learning & Off-Policy Evaluation Framework.

This module provides tools for optimizing targeting policies from CATE estimates
and evaluating counterfactual policy performance using off-policy evaluation methods.

Key Components:
- Policy optimization (greedy, budget-constrained, multi-objective)
- Off-policy evaluation (IPS, DR, DR-SN)
- Policy simulation and testing framework
- Integration with CATE estimators

Classes:
    PolicyOptimizer: Optimize treatment assignment policies
    OffPolicyEvaluator: Evaluate policy performance using historical data
    PolicySimulator: Simulate policy performance under different conditions
    PolicyIntegrator: Integration layer with CATE estimators

Functions:
    greedy_policy: Select top-k individuals by uplift
    budget_constrained_policy: ILP optimization under cost constraints
    evaluate_policy: Compute policy value using off-policy methods
"""

from .integration import PolicyIntegrator
from .off_policy_evaluation import OffPolicyEvaluator, OPEResult
from .optimization import PolicyOptimizer, PolicyResult
from .simulation import PolicySimulator, SimulationResult

__all__ = [
    "PolicyOptimizer",
    "PolicyResult",
    "OffPolicyEvaluator",
    "OPEResult",
    "PolicySimulator",
    "SimulationResult",
    "PolicyIntegrator",
]
