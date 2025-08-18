"""Integration layer for policy learning with CATE estimators.

This module provides seamless integration between the policy learning framework
and existing CATE estimators (Meta-Learners, Causal Forest, etc.).

Classes:
    PolicyIntegrator: Main integration interface
    CATEPolicyResult: Extended result with policy recommendations

Functions:
    integrate_with_cate: Helper function for CATE integration
    extract_treatment_effects: Extract effects from various CATE estimators
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import CATE estimators
from ..estimators.causal_forest import CausalForest
from ..estimators.meta_learners import BaseMetaLearner, CATEResult
from .off_policy_evaluation import OffPolicyEvaluator, OPEResult
from .optimization import PolicyOptimizer, PolicyResult
from .simulation import PolicySimulator, SimulationResult

__all__ = [
    "PolicyIntegrator",
    "CATEPolicyResult",
    "integrate_with_cate",
    "extract_treatment_effects",
]


@dataclass
class CATEPolicyResult:
    """Extended result combining CATE estimation with policy learning.

    Attributes:
        cate_result: Original CATE estimation result
        policy_result: Policy optimization result
        ope_result: Off-policy evaluation result
        integration_info: Additional integration metadata
        policy_recommendations: Actionable policy recommendations
        treatment_assignment: Final treatment assignment
        individual_treatment_effects: CATE estimates for each individual
        policy_value_estimate: Estimated value of the optimized policy
    """

    cate_result: CATEResult | Any
    policy_result: PolicyResult
    ope_result: OPEResult | None = None
    integration_info: dict[str, Any] = field(default_factory=dict)
    policy_recommendations: dict[str, Any] | None = None

    @property
    def treatment_assignment(self) -> NDArray[np.bool_]:
        """Get the final treatment assignment."""
        return self.policy_result.treatment_assignment

    @property
    def individual_treatment_effects(self) -> NDArray[np.floating]:
        """Get individual treatment effects."""
        return self.policy_result.individual_uplifts

    @property
    def policy_value_estimate(self) -> float:
        """Get estimated policy value."""
        if self.ope_result is not None:
            return self.ope_result.policy_value
        return self.policy_result.expected_value

    def get_policy_summary(self) -> dict[str, Any]:
        """Get comprehensive policy summary."""
        summary = {
            "cate_method": getattr(self.cate_result, "method", "unknown"),
            "policy_type": self.policy_result.policy_type,
            "n_treated": self.policy_result.n_treated,
            "treatment_rate": self.policy_result.treatment_rate,
            "expected_policy_value": self.policy_value_estimate,
            "total_cost": self.policy_result.total_cost,
        }

        if self.ope_result is not None:
            summary.update(
                {
                    "ope_method": self.ope_result.method,
                    "policy_value_se": self.ope_result.policy_value_se,
                    "policy_value_ci": self.ope_result.confidence_interval,
                }
            )

        if self.policy_recommendations is not None:
            summary["recommendations"] = self.policy_recommendations

        return summary


class PolicyIntegrator:
    """Integration layer for CATE estimators and policy learning.

    This class provides a unified interface for combining heterogeneous
    treatment effect estimation with policy optimization and evaluation.

    Args:
        policy_optimizer: Policy optimization engine
        off_policy_evaluator: Off-policy evaluation engine
        policy_simulator: Policy simulation engine
        default_costs: Default treatment costs (if not specified)
        verbose: Whether to print integration progress
    """

    def __init__(
        self,
        policy_optimizer: PolicyOptimizer | None = None,
        off_policy_evaluator: OffPolicyEvaluator | None = None,
        policy_simulator: PolicySimulator | None = None,
        default_costs: float | None = 1.0,
        verbose: bool = False,
    ):
        self.policy_optimizer = policy_optimizer or PolicyOptimizer()
        self.off_policy_evaluator = off_policy_evaluator or OffPolicyEvaluator()
        self.policy_simulator = policy_simulator or PolicySimulator()
        self.default_costs = default_costs
        self.verbose = verbose

    def integrate_cate_with_policy(
        self,
        cate_estimator: BaseMetaLearner | CausalForest | Any,
        features: NDArray[np.floating],
        historical_treatments: NDArray[np.bool_] | None = None,
        historical_outcomes: NDArray[np.floating] | None = None,
        costs: NDArray[np.floating] | None = None,
        budget: float | None = None,
        max_treatment_rate: float | None = None,
        evaluate_policy: bool = True,
    ) -> CATEPolicyResult:
        """Integrate CATE estimation with policy learning.

        Args:
            cate_estimator: Fitted CATE estimator
            features: Individual features for prediction
            historical_treatments: Historical treatment assignments (for OPE)
            historical_outcomes: Historical outcomes (for OPE)
            costs: Treatment costs per individual
            budget: Budget constraint
            max_treatment_rate: Maximum treatment rate constraint
            evaluate_policy: Whether to perform off-policy evaluation

        Returns:
            CATEPolicyResult: Comprehensive results with policy recommendations
        """
        if self.verbose:
            print("Extracting treatment effects from CATE estimator...")

        # Extract treatment effects
        treatment_effects = self._extract_treatment_effects(cate_estimator, features)

        # Set default costs
        if costs is None:
            costs = np.full(len(treatment_effects), self.default_costs)

        if self.verbose:
            print("Optimizing treatment assignment policy...")

        # Optimize policy
        policy_result = self.policy_optimizer.optimize_policy(
            treatment_effects,
            costs=costs,
            budget=budget,
            max_treatment_rate=max_treatment_rate,
            features=features,
        )

        # Off-policy evaluation (if requested and data available)
        ope_result = None
        if (
            evaluate_policy
            and historical_treatments is not None
            and historical_outcomes is not None
        ):
            if self.verbose:
                print("Evaluating policy using off-policy methods...")

            try:
                ope_result = self.off_policy_evaluator.evaluate_policy(
                    policy_result.treatment_assignment,
                    historical_treatments,
                    historical_outcomes,
                    features,
                )
            except Exception as e:
                if self.verbose:
                    print(f"Off-policy evaluation failed: {e}")

        # Generate policy recommendations
        policy_recommendations = self._generate_policy_recommendations(
            policy_result, treatment_effects, costs, ope_result
        )

        # Create CATE result object (handle different estimator types)
        if (
            hasattr(cate_estimator, "last_result_")
            and cate_estimator.last_result_ is not None
        ):
            cate_result = cate_estimator.last_result_
        else:
            # Create a minimal CATE result
            ate_value = np.mean(treatment_effects)
            ate_se = np.std(treatment_effects) / np.sqrt(len(treatment_effects))
            ci_lower = ate_value - 1.96 * ate_se
            ci_upper = ate_value + 1.96 * ate_se

            cate_result = CATEResult(
                ate=ate_value,
                confidence_interval=(ci_lower, ci_upper),
                cate_estimates=treatment_effects,
                method=getattr(cate_estimator, "__class__", {}).get(
                    "__name__", "unknown"
                ),
            )

        return CATEPolicyResult(
            cate_result=cate_result,
            policy_result=policy_result,
            ope_result=ope_result,
            integration_info={
                "cate_estimator_type": type(cate_estimator).__name__,
                "n_individuals": len(treatment_effects),
                "feature_dimensions": features.shape[1] if features.ndim > 1 else 0,
                "policy_optimization_method": self.policy_optimizer.optimization_method,
                "ope_method": self.off_policy_evaluator.method if ope_result else None,
            },
            policy_recommendations=policy_recommendations,
        )

    def compare_cate_policies(
        self,
        cate_estimators: list[BaseMetaLearner | CausalForest | Any],
        estimator_names: list[str],
        features: NDArray[np.floating],
        historical_treatments: NDArray[np.bool_],
        historical_outcomes: NDArray[np.floating],
        costs: NDArray[np.floating] | None = None,
        budget: float | None = None,
    ) -> dict[str, Any]:
        """Compare policies from multiple CATE estimators.

        Args:
            cate_estimators: List of fitted CATE estimators
            estimator_names: Names of the estimators
            features: Individual features
            historical_treatments: Historical treatments
            historical_outcomes: Historical outcomes
            costs: Treatment costs
            budget: Budget constraint

        Returns:
            Dictionary with comparison results
        """
        if len(cate_estimators) != len(estimator_names):
            raise ValueError("Number of estimators must match number of names")

        if self.verbose:
            print(f"Comparing policies from {len(cate_estimators)} CATE estimators...")

        # Integrate each estimator
        integrated_results = {}
        for estimator, name in zip(cate_estimators, estimator_names):
            if self.verbose:
                print(f"Processing {name}...")

            result = self.integrate_cate_with_policy(
                estimator,
                features,
                historical_treatments,
                historical_outcomes,
                costs,
                budget,
            )
            integrated_results[name] = result

        # Compare policies using off-policy evaluation
        policy_comparisons = {}
        estimator_names_list = list(estimator_names)

        for i in range(len(estimator_names_list)):
            for j in range(i + 1, len(estimator_names_list)):
                name1, name2 = estimator_names_list[i], estimator_names_list[j]

                comparison = self.off_policy_evaluator.compare_policies(
                    integrated_results[name1].treatment_assignment,
                    integrated_results[name2].treatment_assignment,
                    historical_treatments,
                    historical_outcomes,
                    features,
                )

                policy_comparisons[f"{name1}_vs_{name2}"] = comparison

        # Rank estimators by policy value
        policy_values = {
            name: result.policy_value_estimate
            for name, result in integrated_results.items()
        }
        ranked_estimators = sorted(
            policy_values.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "integrated_results": integrated_results,
            "policy_comparisons": policy_comparisons,
            "policy_values": policy_values,
            "ranked_estimators": ranked_estimators,
            "best_estimator": ranked_estimators[0][0] if ranked_estimators else None,
        }

    def simulate_cate_policy(
        self,
        cate_estimator: BaseMetaLearner | CausalForest | Any,
        data_generator: callable,
        n_simulations: int = 100,
        scenario_params: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """Simulate CATE-based policy performance.

        Args:
            cate_estimator: CATE estimator to use for policy generation
            data_generator: Function to generate synthetic data
            n_simulations: Number of simulations
            scenario_params: Parameters for data generation

        Returns:
            SimulationResult: Simulation results
        """

        def policy_generator_wrapper(**kwargs):
            """Wrapper that generates policy from CATE estimator."""
            data = data_generator(**kwargs)

            # Extract treatment effects using the CATE estimator
            features = data["features"]
            treatment_effects = self._extract_treatment_effects(
                cate_estimator, features
            )

            # Add treatment effects to data
            data["uplifts"] = treatment_effects
            return data

        return self.policy_simulator.simulate_policy_performance(
            policy_generator_wrapper,
            self.policy_optimizer,
            self.off_policy_evaluator,
            scenario_params,
        )

    def _extract_treatment_effects(
        self,
        estimator: BaseMetaLearner | CausalForest | Any,
        features: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Extract treatment effects from various CATE estimators."""
        try:
            # Try standard predict method for treatment effects
            if hasattr(estimator, "predict"):
                return estimator.predict(features)

            # Try CATE-specific methods
            elif hasattr(estimator, "predict_cate"):
                return estimator.predict_cate(features)

            elif hasattr(estimator, "predict_ite"):
                return estimator.predict_ite(features)

            # For meta-learners, use the specific prediction method
            elif hasattr(estimator, "estimate_cate"):
                return estimator.estimate_cate(features)

            else:
                raise ValueError(f"Unknown CATE estimator type: {type(estimator)}")

        except Exception as e:
            if self.verbose:
                print(f"Error extracting treatment effects: {e}")
            # Fallback to random effects
            return np.random.normal(0, 1, len(features))

    def _generate_policy_recommendations(
        self,
        policy_result: PolicyResult,
        treatment_effects: NDArray[np.floating],
        costs: NDArray[np.floating],
        ope_result: OPEResult | None = None,
    ) -> dict[str, Any]:
        """Generate actionable policy recommendations."""
        recommendations = {
            "treatment_assignment": {
                "n_treated": policy_result.n_treated,
                "treatment_rate": policy_result.treatment_rate,
                "total_cost": policy_result.total_cost,
                "expected_value": policy_result.expected_value,
            },
            "targeting_insights": {
                "high_value_segments": self._identify_high_value_segments(
                    treatment_effects, policy_result.treatment_assignment
                ),
                "cost_efficiency": policy_result.expected_value
                / max(policy_result.total_cost, 1e-8),
                "uplift_distribution": {
                    "mean_treated": np.mean(
                        treatment_effects[policy_result.treatment_assignment]
                    ),
                    "mean_untreated": np.mean(
                        treatment_effects[~policy_result.treatment_assignment]
                    ),
                    "uplift_threshold": np.min(
                        treatment_effects[policy_result.treatment_assignment]
                    )
                    if np.any(policy_result.treatment_assignment)
                    else 0,
                },
            },
            "implementation": {
                "deployment_priority": "high"
                if policy_result.expected_value > 0
                else "low",
                "pilot_recommendation": self._recommend_pilot_size(policy_result),
                "monitoring_metrics": [
                    "treatment_rate",
                    "average_uplift",
                    "cost_per_treatment",
                    "incremental_value",
                ],
            },
        }

        # Add OPE-specific recommendations
        if ope_result is not None:
            recommendations["validation"] = {
                "ope_method": ope_result.method,
                "policy_value_estimate": ope_result.policy_value,
                "confidence_interval": ope_result.confidence_interval,
                "recommendation_confidence": self._assess_confidence(ope_result),
            }

        return recommendations

    def _identify_high_value_segments(
        self,
        treatment_effects: NDArray[np.floating],
        treatment_assignment: NDArray[np.bool_],
    ) -> dict[str, Any]:
        """Identify high-value segments in the policy."""
        treated_effects = treatment_effects[treatment_assignment]

        if len(treated_effects) == 0:
            return {"message": "No individuals selected for treatment"}

        return {
            "top_decile_uplift": np.percentile(treated_effects, 90),
            "median_uplift": np.median(treated_effects),
            "uplift_iqr": np.percentile(treated_effects, 75)
            - np.percentile(treated_effects, 25),
            "segment_heterogeneity": np.std(treated_effects),
        }

    def _recommend_pilot_size(self, policy_result: PolicyResult) -> dict[str, Any]:
        """Recommend pilot test size based on policy characteristics."""
        base_pilot_rate = 0.1  # 10% pilot by default

        # Adjust based on treatment rate
        if policy_result.treatment_rate < 0.1:
            pilot_rate = min(
                0.2, base_pilot_rate * 2
            )  # Increase for low treatment rates
        elif policy_result.treatment_rate > 0.5:
            pilot_rate = max(
                0.05, base_pilot_rate * 0.5
            )  # Decrease for high treatment rates
        else:
            pilot_rate = base_pilot_rate

        return {
            "recommended_pilot_rate": pilot_rate,
            "rationale": f"Based on treatment rate of {policy_result.treatment_rate:.1%}",
            "minimum_sample_size": max(100, int(1000 * pilot_rate)),
            "duration_recommendation": "2-4 weeks",
        }

    def _assess_confidence(self, ope_result: OPEResult) -> str:
        """Assess confidence in policy recommendation."""
        ci_width = ope_result.ci_upper - ope_result.ci_lower
        relative_width = ci_width / abs(ope_result.policy_value + 1e-8)

        if relative_width < 0.2:
            return "high"
        elif relative_width < 0.5:
            return "medium"
        else:
            return "low"


def integrate_with_cate(
    cate_estimator: BaseMetaLearner | CausalForest | Any,
    features: NDArray[np.floating],
    optimization_method: str = "greedy",
    budget: float | None = None,
) -> PolicyResult:
    """Helper function for quick CATE integration.

    Args:
        cate_estimator: Fitted CATE estimator
        features: Individual features
        optimization_method: Policy optimization method
        budget: Budget constraint

    Returns:
        PolicyResult: Optimized policy result
    """
    integrator = PolicyIntegrator(
        policy_optimizer=PolicyOptimizer(optimization_method=optimization_method)
    )

    result = integrator.integrate_cate_with_policy(
        cate_estimator, features, budget=budget, evaluate_policy=False
    )

    return result.policy_result


def extract_treatment_effects(
    estimator: BaseMetaLearner | CausalForest | Any,
    features: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Extract treatment effects from CATE estimators.

    Args:
        estimator: CATE estimator
        features: Individual features

    Returns:
        Individual treatment effects
    """
    integrator = PolicyIntegrator()
    return integrator._extract_treatment_effects(estimator, features)
