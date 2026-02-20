#!/usr/bin/env python3
"""
Marketing Policy Optimization Example

This example demonstrates how to use the policy learning framework to optimize
marketing campaigns based on heterogeneous treatment effects (CATE) with
budget constraints and off-policy evaluation.

Scenario:
- Email marketing campaign for an e-commerce company
- Goal: Maximize incremental revenue while staying within budget
- Constraints: Limited marketing budget, customer fairness considerations
- Evaluation: Off-policy evaluation using historical campaign data

Key Steps:
1. Load and prepare historical campaign data
2. Estimate heterogeneous treatment effects using Meta-Learners
3. Optimize targeting policy under budget constraints
4. Evaluate policy performance using off-policy methods
5. Simulate policy performance under different scenarios
6. Generate actionable business recommendations
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Causal inference imports
from causal_inference.core import (  # type: ignore[import-not-found]
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.meta_learners import (  # type: ignore[import-not-found]
    SLearner,
    TLearner,
)
from causal_inference.policy import (  # type: ignore[import-not-found]
    OffPolicyEvaluator,
    PolicyIntegrator,
    PolicyOptimizer,
    PolicySimulator,
)


def generate_marketing_data(n_customers: int = 5000, seed: Optional[int] = 42) -> dict[str, Any]:
    """Generate realistic marketing campaign data.

    This simulates a historical email marketing campaign for an e-commerce company.

    Returns:
        dict: Dictionary containing customer features, historical treatments, outcomes, and metadata
    """
    np.random.seed(seed)

    # Customer demographics and behavior
    customer_data = {
        "customer_id": np.arange(n_customers),
        "age": np.random.normal(40, 12, n_customers),
        "income": np.random.lognormal(
            10.5, 0.8, n_customers
        ),  # Log-normal income distribution
        "past_purchases": np.random.poisson(3, n_customers),
        "days_since_last_purchase": np.random.exponential(30, n_customers),
        "email_engagement_score": np.random.beta(
            2, 3, n_customers
        ),  # Typically skewed low
        "geographic_segment": np.random.choice(
            ["urban", "suburban", "rural"], n_customers, p=[0.4, 0.4, 0.2]
        ),
    }

    df = pd.DataFrame(customer_data)

    # Engineer features
    df["log_income"] = np.log(df["income"])
    df["recency_score"] = 1 / (
        1 + df["days_since_last_purchase"] / 30
    )  # Higher score = more recent
    df["is_urban"] = (df["geographic_segment"] == "urban").astype(int)
    df["is_high_value"] = (
        (df["income"] > np.percentile(df["income"], 75)) & (df["past_purchases"] > 2)
    ).astype(int)

    # Historical treatment assignment (not optimal - based on simple rules)
    treatment_propensity = 1 / (
        1
        + np.exp(
            -(
                -1.5  # Base propensity
                + 0.3
                * df[
                    "email_engagement_score"
                ]  # More engaged customers more likely to be treated
                + 0.2 * df["is_high_value"]  # High-value customers more likely
                + 0.1 * (df["age"] - 40) / 12  # Slight age bias
                + np.random.normal(0, 0.2, n_customers)  # Random variation
            )
        )
    )

    historical_treatments = np.random.binomial(1, treatment_propensity)

    # True heterogeneous treatment effects (unknown in practice)
    # Email campaigns work better for:
    # - Engaged customers (higher email engagement score)
    # - Customers who haven't purchased recently (recency effect)
    # - Mid-income customers (not too low, not too high)
    true_treatment_effects = (
        0.15  # Base email effect (15% conversion lift)
        + 0.25 * df["email_engagement_score"]  # Engagement boosts effectiveness
        + 0.10 * df["recency_score"]  # Recent customers less responsive
        + 0.05
        * np.maximum(
            0, 1 - df["log_income"] / 12
        )  # Diminishing returns for high income
        + -0.02 * (df["age"] - 40) / 12  # Slightly less effective for older customers
        + np.random.normal(0, 0.05, n_customers)  # Individual variation
    )

    # Historical outcomes (conversion rates)
    base_conversion = (
        0.08  # Base conversion rate (8%)
        + 0.02 * df["email_engagement_score"]  # Engaged customers convert more
        + 0.01 * np.log(df["past_purchases"] + 1)  # Purchase history effect
        + 0.005 * df["is_high_value"]  # High-value customers convert more
        + np.random.normal(0, 0.02, n_customers)  # Noise
    )

    historical_outcomes = np.clip(
        base_conversion + historical_treatments * true_treatment_effects,
        0,
        1,  # Conversion rates between 0 and 1
    )

    # Campaign costs (varies by channel and targeting complexity)
    campaign_costs = np.random.gamma(2, 0.5)  # Base cost ~$1 per customer
    # Higher costs for complex targeting
    campaign_costs *= 1 + 0.3 * df["is_high_value"] + 0.2 * df["email_engagement_score"]

    # Prepare feature matrix for ML models
    feature_columns = [
        "age",
        "log_income",
        "past_purchases",
        "recency_score",
        "email_engagement_score",
        "is_urban",
        "is_high_value",
    ]
    X = df[feature_columns].values

    return {
        "customer_data": df,
        "features": X,
        "feature_names": feature_columns,
        "historical_treatments": historical_treatments,
        "historical_outcomes": historical_outcomes,
        "true_treatment_effects": true_treatment_effects,
        "campaign_costs": campaign_costs,
        "treatment_propensity": treatment_propensity,
    }


def train_cate_models(
    data: dict[str, Any], test_size: float = 0.3, random_state: int = 42
) -> dict[str, Any]:
    """Train multiple CATE estimators and compare performance.

    Args:
        data: Marketing data dictionary from generate_marketing_data()
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        dict: Trained CATE estimators and evaluation data
    """
    print("Training CATE estimators...")

    # Split data
    indices = np.arange(len(data["features"]))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    X_train, X_test = data["features"][train_idx], data["features"][test_idx]
    T_train, T_test = (
        data["historical_treatments"][train_idx],
        data["historical_treatments"][test_idx],
    )
    Y_train, Y_test = (
        data["historical_outcomes"][train_idx],
        data["historical_outcomes"][test_idx],
    )

    # Prepare data objects
    treatment_data = TreatmentData(values=T_train, treatment_type="binary")
    outcome_data = OutcomeData(values=Y_train, outcome_type="continuous")
    covariate_data = CovariateData(values=X_train)

    # Train T-learner (separate models for treated and control)
    t_learner = TLearner(
        control_model=RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=random_state
        ),
        treatment_model=RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=random_state
        ),
        random_state=random_state,
    )
    t_learner.fit(treatment_data, outcome_data, covariate_data)

    # Train S-learner (single model with treatment indicator)
    s_learner = SLearner(
        base_model=RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=random_state
        ),
        random_state=random_state,
    )
    s_learner.fit(treatment_data, outcome_data, covariate_data)

    # Predict treatment effects on test set
    t_learner_effects = t_learner.predict(X_test)
    s_learner_effects = s_learner.predict(X_test)
    true_effects_test = data["true_treatment_effects"][test_idx]

    # Evaluate CATE estimation quality
    from sklearn.metrics import mean_squared_error, r2_score

    t_learner_mse = mean_squared_error(true_effects_test, t_learner_effects)
    s_learner_mse = mean_squared_error(true_effects_test, s_learner_effects)

    t_learner_r2 = r2_score(true_effects_test, t_learner_effects)
    s_learner_r2 = r2_score(true_effects_test, s_learner_effects)

    print(
        f"T-learner CATE estimation - MSE: {t_learner_mse:.4f}, R²: {t_learner_r2:.4f}"
    )
    print(
        f"S-learner CATE estimation - MSE: {s_learner_mse:.4f}, R²: {s_learner_r2:.4f}"
    )

    # Select best model
    best_estimator = t_learner if t_learner_r2 > s_learner_r2 else s_learner
    best_name = "T-learner" if t_learner_r2 > s_learner_r2 else "S-learner"

    print(f"Selected best CATE estimator: {best_name}")

    return {
        "estimators": {"t_learner": t_learner, "s_learner": s_learner},
        "best_estimator": best_estimator,
        "best_name": best_name,
        "test_data": {
            "features": X_test,
            "treatments": T_test,
            "outcomes": Y_test,
            "true_effects": true_effects_test,
            "costs": data["campaign_costs"][test_idx],
        },
        "evaluation": {
            "t_learner_mse": t_learner_mse,
            "s_learner_mse": s_learner_mse,
            "t_learner_r2": t_learner_r2,
            "s_learner_r2": s_learner_r2,
        },
    }


def optimize_marketing_policy(
    cate_results: dict[str, Any],
    data: dict[str, Any],
    budget_scenarios: list[float] | None = None,
) -> dict[str, Any]:
    """Optimize marketing policy under different budget constraints.

    Args:
        cate_results: Results from train_cate_models()
        data: Original marketing data
        budget_scenarios: List of budget fractions to test (default: [0.1, 0.2, 0.3, 0.5])

    Returns:
        dict: Policy optimization results for different scenarios
    """
    print("\\nOptimizing marketing policies...")

    if budget_scenarios is None:
        budget_scenarios = [0.1, 0.2, 0.3, 0.5]  # 10%, 20%, 30%, 50% of total budget

    # Use test data for policy optimization
    test_data = cate_results["test_data"]
    total_possible_cost = np.sum(test_data["costs"])

    # Set up policy integrator
    integrator = PolicyIntegrator(
        policy_optimizer=PolicyOptimizer(optimization_method="greedy", random_state=42),
        off_policy_evaluator=OffPolicyEvaluator(method="dr", random_state=42),
        verbose=True,
    )

    policy_results = {}

    for budget_fraction in budget_scenarios:
        budget = total_possible_cost * budget_fraction
        print(
            f"\\nOptimizing policy with {budget_fraction:.0%} budget (${budget:.0f})..."
        )

        # Integrate CATE estimation with policy optimization
        result = integrator.integrate_cate_with_policy(
            cate_results["best_estimator"],
            test_data["features"],
            test_data["treatments"],
            test_data["outcomes"],
            costs=test_data["costs"],
            budget=budget,
            evaluate_policy=True,
        )

        policy_results[f"budget_{budget_fraction:.0%}"] = {
            "budget_fraction": budget_fraction,
            "budget_amount": budget,
            "result": result,
            "summary": result.get_policy_summary(),
        }

        # Print key metrics
        summary = result.get_policy_summary()
        print(f"  Policy type: {summary['policy_type']}")
        print(
            f"  Customers treated: {summary['n_treated']:,} ({summary['treatment_rate']:.1%})"
        )
        print(f"  Expected policy value: {summary['expected_policy_value']:.4f}")
        print(f"  Total cost: ${result.policy_result.total_cost:.0f}")

        if result.ope_result:
            ci = result.ope_result.confidence_interval
            print(f"  Policy value CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    return policy_results


def evaluate_policy_performance(
    policy_results: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate and compare policy performance across scenarios.

    Args:
        policy_results: Results from optimize_marketing_policy()
        data: Original marketing data

    Returns:
        dict: Performance evaluation results
    """
    print("\\n" + "=" * 60)
    print("POLICY PERFORMANCE EVALUATION")
    print("=" * 60)

    # Create performance comparison
    performance_data = []

    for scenario_name, scenario_result in policy_results.items():
        result = scenario_result["result"]
        summary = scenario_result["summary"]

        performance_data.append(
            {
                "scenario": scenario_name,
                "budget_fraction": scenario_result["budget_fraction"],
                "n_treated": summary["n_treated"],
                "treatment_rate": summary["treatment_rate"],
                "expected_value": summary["expected_policy_value"],
                "cost": result.policy_result.total_cost,
                "value_per_cost": summary["expected_policy_value"]
                / max(result.policy_result.total_cost, 1),
                "ope_method": result.ope_result.method if result.ope_result else None,
                "ci_width": (result.ope_result.ci_upper - result.ope_result.ci_lower)
                if result.ope_result
                else None,
            }
        )

    performance_df = pd.DataFrame(performance_data)

    # Find best policy by value per cost
    best_scenario_idx = performance_df["value_per_cost"].idxmax()
    best_scenario = performance_df.loc[best_scenario_idx]

    print("\\nPolicy Performance Comparison:")
    print(performance_df.round(4))

    print(f"\\nBest Policy (by value per cost): {best_scenario['scenario']}")
    print(f"  Budget: {best_scenario['budget_fraction']:.0%}")
    print(f"  Treatment Rate: {best_scenario['treatment_rate']:.1%}")
    print(f"  Expected Value: {best_scenario['expected_value']:.4f}")
    print(f"  Value per Dollar: {best_scenario['value_per_cost']:.4f}")

    return {
        "performance_comparison": performance_df,
        "best_scenario": best_scenario,
        "best_policy_result": policy_results[best_scenario["scenario"]]["result"],
    }


def simulate_policy_scenarios(
    cate_results: dict[str, Any], n_simulations: int = 50
) -> dict[str, Any]:
    """Simulate policy performance under different market conditions.

    Args:
        cate_results: Results from train_cate_models()
        n_simulations: Number of Monte Carlo simulations per scenario

    Returns:
        dict: Simulation results for different scenarios
    """
    print("\\n" + "=" * 60)
    print("POLICY SCENARIO SIMULATION")
    print("=" * 60)

    def marketing_data_simulator(market_condition: str = "normal", n_customers: int = 1000) -> dict[str, Any]:
        """Simulate marketing data under different conditions."""
        # Use the original data generation function with modifications
        base_data = generate_marketing_data(
            n_customers=n_customers
        )  # Different seed each time

        # Modify based on market condition
        if market_condition == "recession":
            # Lower treatment effects, higher price sensitivity
            base_data["true_treatment_effects"] *= 0.7  # 30% reduction in effectiveness
            base_data["campaign_costs"] *= 1.2  # 20% higher costs
        elif market_condition == "expansion":
            # Higher treatment effects, customers more responsive
            base_data["true_treatment_effects"] *= 1.3  # 30% increase in effectiveness
            base_data["campaign_costs"] *= 0.9  # 10% lower costs
        # "normal" condition uses base data as-is

        return {
            "uplifts": base_data["true_treatment_effects"],
            "true_uplifts": base_data[
                "true_treatment_effects"
            ],  # For oracle comparison
            "historical_treatments": base_data["historical_treatments"].astype(bool),
            "historical_outcomes": base_data["historical_outcomes"],
            "features": base_data["features"],
            "costs": base_data["campaign_costs"],
            "budget": np.sum(base_data["campaign_costs"]) * 0.25,  # 25% budget
        }

    # Define scenarios
    scenarios = {
        "normal_market": {"market_condition": "normal"},
        "recession": {"market_condition": "recession"},
        "expansion": {"market_condition": "expansion"},
    }

    # Set up simulation
    policy_optimizer = PolicyOptimizer(optimization_method="greedy", random_state=42)
    evaluator = OffPolicyEvaluator(method="dr", random_state=42)
    simulator = PolicySimulator(
        n_simulations=n_simulations, random_state=42, verbose=True
    )

    print(f"Running {n_simulations} simulations per scenario...")
    scenario_results = simulator.scenario_analysis(
        marketing_data_simulator,
        policy_optimizer,
        evaluator,
        scenarios,
    )

    # Analyze results
    print("\\nScenario Analysis Results:")
    for scenario_name, result in scenario_results.items():
        print(f"\\n{scenario_name.upper()}:")
        print(
            f"  Mean policy value: {result.mean_policy_value:.4f} ± {result.std_policy_value:.4f}"
        )
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        if result.regret_vs_oracle is not None:
            print(f"  Average regret vs oracle: {result.regret_vs_oracle:.4f}")
        print(
            f"  Successful simulations: {result.simulation_info['n_successful_simulations']}"
        )

    return dict(scenario_results)


def generate_business_recommendations(
    evaluation_results: dict[str, Any],
    scenario_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate actionable business recommendations based on analysis.

    Args:
        evaluation_results: Results from evaluate_policy_performance()
        scenario_results: Results from simulate_policy_scenarios() (optional)

    Returns:
        dict: Structured business recommendations
    """
    print("\\n" + "=" * 60)
    print("BUSINESS RECOMMENDATIONS")
    print("=" * 60)

    best_policy = evaluation_results["best_policy_result"]

    recommendations: dict[str, Any] = {
        "executive_summary": {
            "recommended_budget": f"{evaluation_results['best_scenario']['budget_fraction']:.0%} of total marketing budget",
            "expected_customers_treated": f"{evaluation_results['best_scenario']['n_treated']:,}",
            "treatment_rate": f"{evaluation_results['best_scenario']['treatment_rate']:.1%}",
            "expected_roi": f"${evaluation_results['best_scenario']['value_per_cost']:.2f} per dollar spent",
        },
        "implementation": best_policy.policy_recommendations["implementation"]
        if best_policy.policy_recommendations
        else {},
        "targeting_insights": best_policy.policy_recommendations["targeting_insights"]
        if best_policy.policy_recommendations
        else {},
        "risk_assessment": {},
        "next_steps": [],
    }

    # Risk assessment
    if scenario_results:
        normal_value = scenario_results["normal_market"].mean_policy_value
        recession_value = scenario_results["recession"].mean_policy_value
        expansion_value = scenario_results["expansion"].mean_policy_value

        downside_risk = (normal_value - recession_value) / normal_value
        upside_potential = (expansion_value - normal_value) / normal_value

        recommendations["risk_assessment"] = {
            "downside_risk": f"{downside_risk:.1%} value reduction in recession",
            "upside_potential": f"{upside_potential:.1%} value increase in expansion",
            "policy_robustness": "high"
            if downside_risk < 0.2
            else "medium"
            if downside_risk < 0.4
            else "low",
        }

    # Next steps
    recommendations["next_steps"] = [
        "Implement A/B test with recommended targeting policy vs. current strategy",
        "Set up monitoring dashboard for key policy metrics",
        "Plan for policy recalibration based on A/B test results",
        "Consider expanding budget if pilot results exceed expectations",
    ]

    # Add scenario-specific recommendations
    if scenario_results and downside_risk > 0.3:
        recommendations["next_steps"].append(
            "Develop contingency policy for economic downturn scenarios"
        )

    # Print recommendations
    print("\\nEXECUTIVE SUMMARY:")
    for key, value in recommendations["executive_summary"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    if recommendations["risk_assessment"]:
        print("\\nRISK ASSESSMENT:")
        for key, value in recommendations["risk_assessment"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\\nNEXT STEPS:")
    for i, step in enumerate(recommendations["next_steps"], 1):
        print(f"  {i}. {step}")

    return recommendations


def main() -> None:
    """Run the complete marketing policy optimization workflow."""
    print("=" * 60)
    print("MARKETING POLICY OPTIMIZATION EXAMPLE")
    print("=" * 60)
    print("\\nGenerating synthetic marketing campaign data...")

    # Step 1: Generate marketing data
    data = generate_marketing_data(n_customers=3000, seed=42)
    print(f"Generated data for {len(data['features'])} customers")
    print(f"Historical treatment rate: {np.mean(data['historical_treatments']):.1%}")
    print(f"Historical conversion rate: {np.mean(data['historical_outcomes']):.1%}")

    # Step 2: Train CATE models
    cate_results = train_cate_models(data)

    # Step 3: Optimize policies under different budget scenarios
    policy_results = optimize_marketing_policy(cate_results, data)

    # Step 4: Evaluate policy performance
    evaluation_results = evaluate_policy_performance(policy_results, data)

    # Step 5: Simulate scenarios (optional - can be time-consuming)
    print("\\nRunning scenario simulations (this may take a moment)...")
    scenario_results = simulate_policy_scenarios(cate_results, n_simulations=20)

    # Step 6: Generate business recommendations
    generate_business_recommendations(evaluation_results, scenario_results)

    print("\\n" + "=" * 60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\\nKey Insights:")
    print("- Learned optimal targeting policy from heterogeneous treatment effects")
    print("- Evaluated policy performance using robust off-policy evaluation")
    print("- Simulated policy performance under different market conditions")
    print("- Generated actionable business recommendations")
    print(
        "\\nThis framework can be applied to real campaign data for optimal targeting!"
    )


if __name__ == "__main__":
    main()
