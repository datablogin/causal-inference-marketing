"""Policy simulation and testing framework.

This module provides tools for simulating policy performance under different
conditions, conducting scenario analysis, and performing sensitivity tests.

Classes:
    PolicySimulator: Main policy simulation engine
    SimulationResult: Results container for policy simulations

Functions:
    monte_carlo_policy_evaluation: MC evaluation of policy performance
    sensitivity_analysis: Test policy robustness to model misspecification
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .off_policy_evaluation import OffPolicyEvaluator
from .optimization import PolicyOptimizer

__all__ = [
    "PolicySimulator",
    "SimulationResult",
    "monte_carlo_policy_evaluation",
    "sensitivity_analysis",
]


@dataclass
class SimulationResult:
    """Results container for policy simulation.

    Attributes:
        simulation_type: Type of simulation performed
        policy_values: Distribution of policy values across simulations
        mean_policy_value: Mean policy value
        std_policy_value: Standard deviation of policy values
        confidence_interval: Confidence interval for policy value
        regret_vs_oracle: Regret compared to oracle policy
        simulation_info: Additional simulation details
        scenario_results: Results for different scenarios (if applicable)
    """

    simulation_type: str
    policy_values: NDArray[np.floating[Any]]
    mean_policy_value: float
    std_policy_value: float
    confidence_interval: tuple[float, float]
    regret_vs_oracle: Optional[float] = None
    simulation_info: dict[str, Any] = field(default_factory=dict)
    scenario_results: Optional[dict[str, Any]] = None

    @property
    def ci_lower(self) -> float:
        """Lower bound of confidence interval."""
        return self.confidence_interval[0]

    @property
    def ci_upper(self) -> float:
        """Upper bound of confidence interval."""
        return self.confidence_interval[1]

    def get_simulation_summary(self) -> dict[str, Any]:
        """Get summary statistics for the simulation."""
        return {
            "simulation_type": self.simulation_type,
            "n_simulations": len(self.policy_values),
            "mean_policy_value": self.mean_policy_value,
            "std_policy_value": self.std_policy_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "regret_vs_oracle": self.regret_vs_oracle,
            "min_value": np.min(self.policy_values),
            "max_value": np.max(self.policy_values),
            "median_value": np.median(self.policy_values),
        }


class PolicySimulator:
    """Policy simulation and testing engine.

    This class provides comprehensive simulation capabilities for testing
    policy performance under various conditions and assumptions.

    Args:
        n_simulations: Number of Monte Carlo simulations
        confidence_level: Confidence level for intervals (default: 0.95)
        random_state: Random seed for reproducibility
        verbose: Whether to print simulation progress
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

    def simulate_policy_performance(
        self,
        data_generator: Callable[..., Any],
        policy_optimizer: PolicyOptimizer,
        evaluator: OffPolicyEvaluator,
        scenario_params: Optional[dict[str, Any]] = None,
    ) -> SimulationResult:
        """Simulate policy performance using Monte Carlo.

        Args:
            data_generator: Function to generate synthetic data
            policy_optimizer: Policy optimization engine
            evaluator: Off-policy evaluation engine
            scenario_params: Parameters for data generation scenarios

        Returns:
            SimulationResult: Comprehensive simulation results
        """
        policy_values = []
        oracle_values = []
        simulation_details = []

        if self.verbose:
            print(f"Running {self.n_simulations} policy simulations...")

        for sim_idx in range(self.n_simulations):
            if self.verbose and sim_idx % 100 == 0:
                print(f"Simulation {sim_idx + 1}/{self.n_simulations}")

            try:
                # Generate synthetic data
                data = data_generator(**(scenario_params or {}))

                # Extract components
                uplifts = data["uplifts"]
                costs = data.get("costs", np.ones(len(uplifts)))
                historical_treatments = data["historical_treatments"]
                historical_outcomes = data["historical_outcomes"]
                features = data.get("features")
                budget = data.get("budget")

                # Optimize policy
                policy_result = policy_optimizer.optimize_policy(
                    uplifts, costs, budget=budget
                )

                # Evaluate policy
                ope_result = evaluator.evaluate_policy(
                    policy_result.treatment_assignment,
                    historical_treatments,
                    historical_outcomes,
                    features,
                )

                policy_values.append(ope_result.policy_value)

                # Calculate oracle value (perfect policy with true uplifts)
                if "true_uplifts" in data:
                    oracle_policy = policy_optimizer.optimize_policy(
                        data["true_uplifts"], costs, budget=budget
                    )
                    oracle_ope = evaluator.evaluate_policy(
                        oracle_policy.treatment_assignment,
                        historical_treatments,
                        historical_outcomes,
                        features,
                    )
                    oracle_values.append(oracle_ope.policy_value)

                # Store simulation details
                simulation_details.append(
                    {
                        "simulation_id": sim_idx,
                        "policy_value": ope_result.policy_value,
                        "policy_value_se": ope_result.policy_value_se,
                        "n_treated": policy_result.n_treated,
                        "treatment_rate": policy_result.treatment_rate,
                        "total_cost": policy_result.total_cost,
                    }
                )

            except Exception as e:
                if self.verbose:
                    print(f"Simulation {sim_idx} failed: {e}")
                continue

        # Convert to numpy arrays
        policy_values_arr = np.array(policy_values)
        oracle_values_arr = np.array(oracle_values) if oracle_values else None

        # Calculate summary statistics
        mean_value = float(np.mean(policy_values_arr))
        std_value = float(np.std(policy_values_arr))

        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = float(np.percentile(policy_values_arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(policy_values_arr, 100 * (1 - alpha / 2)))

        # Calculate regret vs oracle
        regret_vs_oracle: Optional[float] = None
        if oracle_values_arr is not None:
            regret_vs_oracle = float(np.mean(oracle_values_arr - policy_values_arr))

        return SimulationResult(
            simulation_type="monte_carlo",
            policy_values=policy_values_arr,
            mean_policy_value=mean_value,
            std_policy_value=std_value,
            confidence_interval=(ci_lower, ci_upper),
            regret_vs_oracle=regret_vs_oracle,
            simulation_info={
                "n_successful_simulations": len(policy_values_arr),
                "n_failed_simulations": self.n_simulations - len(policy_values_arr),
                "simulation_details": simulation_details,
                "oracle_available": oracle_values_arr is not None,
            },
        )

    def scenario_analysis(
        self,
        data_generator: Callable[..., Any],
        policy_optimizer: PolicyOptimizer,
        evaluator: OffPolicyEvaluator,
        scenarios: dict[str, dict[str, Any]],
    ) -> dict[str, SimulationResult]:
        """Conduct scenario analysis across different conditions.

        Args:
            data_generator: Function to generate synthetic data
            policy_optimizer: Policy optimization engine
            evaluator: Off-policy evaluation engine
            scenarios: Dictionary of scenario name -> parameters

        Returns:
            Dictionary of scenario name -> SimulationResult
        """
        scenario_results = {}

        if self.verbose:
            print(f"Running scenario analysis across {len(scenarios)} scenarios...")

        for scenario_name, scenario_params in scenarios.items():
            if self.verbose:
                print(f"Running scenario: {scenario_name}")

            result = self.simulate_policy_performance(
                data_generator, policy_optimizer, evaluator, scenario_params
            )
            result.simulation_info["scenario_name"] = scenario_name
            result.simulation_info["scenario_params"] = scenario_params
            scenario_results[scenario_name] = result

        return scenario_results

    def sensitivity_analysis(
        self,
        base_data: dict[str, Any],
        policy_optimizer: PolicyOptimizer,
        evaluator: OffPolicyEvaluator,
        perturbation_params: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Conduct sensitivity analysis by perturbing model inputs.

        Args:
            base_data: Base data dictionary
            policy_optimizer: Policy optimization engine
            evaluator: Off-policy evaluation engine
            perturbation_params: Parameters for different perturbations

        Returns:
            Dictionary with sensitivity analysis results
        """
        base_uplifts = base_data["uplifts"]
        base_costs = base_data.get("costs", np.ones(len(base_uplifts)))
        historical_treatments = base_data["historical_treatments"]
        historical_outcomes = base_data["historical_outcomes"]
        features = base_data.get("features")
        budget = base_data.get("budget")

        # Evaluate base policy
        base_policy = policy_optimizer.optimize_policy(
            base_uplifts, base_costs, budget=budget
        )
        base_result = evaluator.evaluate_policy(
            base_policy.treatment_assignment,
            historical_treatments,
            historical_outcomes,
            features,
        )
        base_value = base_result.policy_value

        sensitivity_results = {}

        for perturbation_name, perturbation_config in perturbation_params.items():
            perturbation_type = perturbation_config["type"]
            perturbation_levels = perturbation_config["levels"]

            sensitivity_values = []

            for level in perturbation_levels:
                try:
                    # Apply perturbation
                    if perturbation_type == "uplift_noise":
                        # Add noise to uplifts
                        noise = np.random.normal(0, level, len(base_uplifts))
                        perturbed_uplifts = base_uplifts + noise
                    elif perturbation_type == "cost_multiplier":
                        # Multiply costs
                        perturbed_costs = base_costs * level
                        perturbed_uplifts = base_uplifts
                    elif perturbation_type == "budget_change":
                        # Change budget
                        perturbed_budget = budget * level if budget else None
                        perturbed_uplifts = base_uplifts
                        perturbed_costs = base_costs
                    else:
                        continue

                    # Optimize policy with perturbation
                    if perturbation_type == "budget_change":
                        perturbed_policy = policy_optimizer.optimize_policy(
                            perturbed_uplifts, base_costs, budget=perturbed_budget
                        )
                    elif perturbation_type == "cost_multiplier":
                        perturbed_policy = policy_optimizer.optimize_policy(
                            perturbed_uplifts, perturbed_costs, budget=budget
                        )
                    else:
                        perturbed_policy = policy_optimizer.optimize_policy(
                            perturbed_uplifts, base_costs, budget=budget
                        )

                    # Evaluate perturbed policy
                    perturbed_result = evaluator.evaluate_policy(
                        perturbed_policy.treatment_assignment,
                        historical_treatments,
                        historical_outcomes,
                        features,
                    )

                    sensitivity_values.append(
                        {
                            "level": level,
                            "policy_value": perturbed_result.policy_value,
                            "value_change": perturbed_result.policy_value - base_value,
                            "relative_change": (
                                perturbed_result.policy_value - base_value
                            )
                            / abs(base_value + 1e-8),
                            "n_treated": perturbed_policy.n_treated,
                            "treatment_rate": perturbed_policy.treatment_rate,
                        }
                    )

                except Exception as e:
                    if self.verbose:
                        print(
                            f"Sensitivity analysis failed for {perturbation_name}, level {level}: {e}"
                        )
                    continue

            sensitivity_results[perturbation_name] = {
                "perturbation_type": perturbation_type,
                "results": sensitivity_values,
                "max_absolute_change": max(
                    [abs(r["value_change"]) for r in sensitivity_values], default=0
                ),
                "max_relative_change": max(
                    [abs(r["relative_change"]) for r in sensitivity_values], default=0
                ),
            }

        return {
            "base_policy_value": base_value,
            "perturbations": sensitivity_results,
            "summary": {
                "most_sensitive_to": max(
                    sensitivity_results.keys(),
                    key=lambda k: sensitivity_results[k]["max_relative_change"],
                    default=None,
                ),
                "max_sensitivity": max(
                    [r["max_relative_change"] for r in sensitivity_results.values()],
                    default=0,
                ),
            },
        }

    def ab_test_simulation(
        self,
        policy_assignments: list[NDArray[np.bool_]],
        policy_names: list[str],
        data_generator: Callable[..., Any],
        test_duration: int = 30,  # days
        daily_samples: int = 1000,
    ) -> dict[str, Any]:
        """Simulate A/B testing of multiple policies.

        Args:
            policy_assignments: List of treatment assignments for each policy
            policy_names: Names of the policies
            data_generator: Function to generate daily data
            test_duration: Duration of test in days
            daily_samples: Number of samples per day

        Returns:
            A/B test simulation results
        """
        n_policies = len(policy_assignments)

        # Track cumulative results
        cumulative_values: dict[str, list[float]] = {name: [] for name in policy_names}
        daily_results = []

        for day in range(test_duration):
            if self.verbose and day % 7 == 0:
                print(f"Day {day + 1}/{test_duration}")

            # Generate daily data
            daily_data = data_generator(n_samples=daily_samples)

            day_results: dict[str, Any] = {"day": day + 1}

            for i, (policy, name) in enumerate(zip(policy_assignments, policy_names)):
                # Subset policy to daily samples
                daily_policy = (
                    policy[:daily_samples] if len(policy) >= daily_samples else policy
                )

                # Evaluate policy on daily data
                evaluator = OffPolicyEvaluator(
                    method="dr", random_state=self.random_state
                )
                result = evaluator.evaluate_policy(
                    daily_policy,
                    daily_data["historical_treatments"][: len(daily_policy)],
                    daily_data["historical_outcomes"][: len(daily_policy)],
                    daily_data.get("features", {}).get(len(daily_policy))
                    if "features" in daily_data
                    else None,
                )

                daily_value = result.policy_value
                cumulative_values[name].append(daily_value)

                day_results[f"{name}_value"] = daily_value
                day_results[f"{name}_cumulative_mean"] = np.mean(
                    cumulative_values[name]
                )
                day_results[f"{name}_cumulative_std"] = np.std(cumulative_values[name])

            daily_results.append(day_results)

        # Statistical analysis
        final_means = {
            name: np.mean(values) for name, values in cumulative_values.items()
        }
        final_stds = {
            name: np.std(values) for name, values in cumulative_values.items()
        }

        # Pairwise comparisons
        comparisons = {}
        for i in range(n_policies):
            for j in range(i + 1, n_policies):
                name1, name2 = policy_names[i], policy_names[j]
                values1, values2 = cumulative_values[name1], cumulative_values[name2]

                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(values1, values2)

                comparisons[f"{name1}_vs_{name2}"] = {
                    "mean_difference": final_means[name1] - final_means[name2],
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "winner": name1
                    if final_means[name1] > final_means[name2]
                    else name2,
                }

        return {
            "test_duration": test_duration,
            "daily_samples": daily_samples,
            "policy_names": policy_names,
            "final_means": final_means,
            "final_stds": final_stds,
            "daily_results": daily_results,
            "comparisons": comparisons,
            "best_policy": max(policy_names, key=lambda name: final_means[name]),
            "cumulative_values": cumulative_values,
        }


def monte_carlo_policy_evaluation(
    policy_assignment: NDArray[np.bool_],
    data_generator: Callable[..., Any],
    n_simulations: int = 1000,
    evaluator: Optional[OffPolicyEvaluator] = None,
) -> tuple[float, float]:
    """Monte Carlo evaluation of policy performance.

    Args:
        policy_assignment: Treatment assignment policy
        data_generator: Function to generate synthetic data
        n_simulations: Number of Monte Carlo simulations
        evaluator: Off-policy evaluator (uses DR by default)

    Returns:
        Tuple of (mean_policy_value, std_policy_value)
    """
    if evaluator is None:
        evaluator = OffPolicyEvaluator(method="dr")

    policy_values = []

    for _ in range(n_simulations):
        try:
            # Generate data
            data = data_generator()

            # Evaluate policy
            result = evaluator.evaluate_policy(
                policy_assignment,
                data["historical_treatments"],
                data["historical_outcomes"],
                data.get("features"),
            )

            policy_values.append(result.policy_value)

        except Exception:
            continue

    policy_values_arr = np.array(policy_values)
    return float(np.mean(policy_values_arr)), float(np.std(policy_values_arr))


def sensitivity_analysis(
    base_uplifts: NDArray[np.floating[Any]],
    policy_optimizer: PolicyOptimizer,
    perturbation_levels: list[float],
    perturbation_type: str = "additive_noise",
) -> dict[str, list[float]]:
    """Simple sensitivity analysis for policy optimization.

    Args:
        base_uplifts: Base uplift predictions
        policy_optimizer: Policy optimization engine
        perturbation_levels: Levels of perturbation to test
        perturbation_type: Type of perturbation ('additive_noise', 'multiplicative')

    Returns:
        Dictionary with sensitivity analysis results
    """
    base_policy = policy_optimizer.optimize_policy(base_uplifts)
    base_value = base_policy.expected_value

    sensitivity_results = {
        "perturbation_levels": perturbation_levels,
        "policy_values": [],
        "value_changes": [],
        "relative_changes": [],
    }

    for level in perturbation_levels:
        if perturbation_type == "additive_noise":
            noise = np.random.normal(0, level, len(base_uplifts))
            perturbed_uplifts = base_uplifts + noise
        elif perturbation_type == "multiplicative":
            perturbed_uplifts = base_uplifts * level
        else:
            continue

        # Optimize policy with perturbation
        perturbed_policy = policy_optimizer.optimize_policy(perturbed_uplifts)

        value_change = perturbed_policy.expected_value - base_value
        relative_change = value_change / (abs(base_value) + 1e-8)

        sensitivity_results["policy_values"].append(perturbed_policy.expected_value)
        sensitivity_results["value_changes"].append(value_change)
        sensitivity_results["relative_changes"].append(relative_change)

    return sensitivity_results
