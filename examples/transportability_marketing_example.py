"""
Transportability & Target Population Weighting Example: Marketing Campaign Analysis

This example demonstrates how to use the transportability module to generalize
causal effects from one population to another, addressing external validity concerns
in marketing applications.

Scenario:
- Source Population: US customers (pilot campaign)
- Target Population: European customers (expansion target)
- Goal: Estimate treatment effect of email marketing campaign for EU market
"""

import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.aipw import AIPW
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.transportability import (
    CovariateShiftDiagnostics,
    DensityRatioEstimator,
    TargetedMaximumTransportedLikelihood,
    TransportabilityEstimator,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def generate_marketing_data() -> Dict[str, Any]:
    """Generate realistic marketing campaign data with known population differences."""
    print("📊 Generating Marketing Campaign Data...")

    np.random.seed(42)

    # Source Population: US Customers
    n_us = 3000
    print(f"   Source (US): {n_us:,} customers")

    # Customer characteristics (US)
    age_us = np.random.normal(42, 15, n_us)
    age_us = np.clip(age_us, 18, 80)  # Realistic age range

    income_us = np.random.lognormal(np.log(55000), 0.6, n_us)
    income_us = np.clip(income_us, 20000, 200000)  # Realistic income range

    education_us = np.random.choice([0, 1, 2, 3], n_us, p=[0.25, 0.35, 0.25, 0.15])
    # 0: High school, 1: Some college, 2: Bachelor's, 3: Graduate

    engagement_score_us = (
        0.01 * age_us
        + 0.00001 * income_us
        + 0.5 * education_us
        + np.random.normal(0, 0.3, n_us)
    )
    engagement_score_us = np.clip(engagement_score_us, 0, 5)

    X_us = np.column_stack([age_us, income_us, education_us, engagement_score_us])

    # Treatment Assignment (Email Campaign)
    # Treatment probability depends on engagement and demographics
    treatment_logits = (
        -1.5
        + 0.02 * age_us
        + 0.00002 * income_us
        + 0.3 * education_us
        + 0.5 * engagement_score_us
        + np.random.normal(0, 0.2, n_us)
    )
    treatment_probs = 1 / (1 + np.exp(-treatment_logits))
    T_us = np.random.binomial(1, treatment_probs)

    # Outcome: Purchase Amount (in USD)
    # Base spending propensity
    base_spending = (
        5 * age_us
        + 0.05 * income_us
        + 15 * education_us
        + 10 * engagement_score_us
        + np.random.normal(0, 20, n_us)
    )

    # Treatment effect (heterogeneous)
    treatment_effect = (
        25
        + 0.1 * age_us
        + 0.0002 * income_us
        + 5 * education_us
        + 8 * engagement_score_us
        + np.random.normal(0, 5, n_us)
    )

    Y_us = base_spending + T_us * treatment_effect
    Y_us = np.maximum(Y_us, 0)  # No negative purchases

    # Target Population: European Customers
    n_eu = 2000
    print(f"   Target (EU): {n_eu:,} customers")

    # European customers have different characteristics
    age_eu = np.random.normal(47, 12, n_eu)  # Slightly older
    age_eu = np.clip(age_eu, 18, 80)

    income_eu = np.random.lognormal(np.log(48000), 0.5, n_eu)  # Lower income (EUR)
    income_eu = np.clip(income_eu, 18000, 150000)

    education_eu = np.random.choice([0, 1, 2, 3], n_eu, p=[0.15, 0.30, 0.35, 0.20])
    # More educated population

    engagement_score_eu = (
        0.01 * age_eu
        + 0.00001 * income_eu
        + 0.6 * education_eu
        + np.random.normal(0, 0.25, n_eu)
    )
    engagement_score_eu = np.clip(engagement_score_eu, 0, 5)

    X_eu = np.column_stack([age_eu, income_eu, education_eu, engagement_score_eu])

    # Feature names
    feature_names = ["age", "income", "education", "engagement_score"]

    # True ATE in source population
    true_ate_us = np.mean(treatment_effect)

    return {
        "source": {
            "X": X_us,
            "T": T_us,
            "Y": Y_us,
            "n": n_us,
            "population": "US Customers",
        },
        "target": {"X": X_eu, "n": n_eu, "population": "European Customers"},
        "feature_names": feature_names,
        "true_ate_source": true_ate_us,
        "metadata": {
            "treatment": "Email Marketing Campaign",
            "outcome": "Purchase Amount (USD)",
            "features": {
                "age": "Customer age (years)",
                "income": "Annual income (USD/EUR)",
                "education": "Education level (0-3)",
                "engagement_score": "Platform engagement score (0-5)",
            },
        },
    }


def analyze_covariate_shift(data: Dict[str, Any]) -> None:
    """Analyze and visualize covariate shift between populations."""
    print("\n🔍 Step 1: Covariate Shift Analysis")
    print("=" * 50)

    source_X = data["source"]["X"]
    target_X = data["target"]["X"]
    feature_names = data["feature_names"]

    # Initialize diagnostics
    diagnostics = CovariateShiftDiagnostics(
        smd_threshold_mild=0.1, smd_threshold_moderate=0.25
    )

    # Run comprehensive analysis
    results = diagnostics.analyze_covariate_shift(
        source_data=source_X, target_data=target_X, variable_names=feature_names
    )

    # Print summary
    print(f"Overall Shift Score: {results['overall_shift_score']:.3f}")
    print(f"Discriminative Accuracy: {results['discriminative_accuracy']:.3f}")
    print(f"Variables analyzed: {results['n_variables']}")
    print(f"Severe shifts: {results['n_severe_shifts']}")
    print(f"Moderate shifts: {results['n_moderate_shifts']}")
    print(f"Mild shifts: {results['n_mild_shifts']}")

    # Create summary table
    summary_df = diagnostics.create_shift_summary_table()
    print("\nCovariate Shift Summary:")
    print(summary_df.round(3))

    # Show recommendations
    print("\n📋 Recommendations:")
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"  {i}. {rec}")

    # Visualize distributions
    _plot_covariate_distributions(data)


def _plot_covariate_distributions(data: Dict[str, Any]) -> None:
    """Plot covariate distributions for source vs target populations."""
    source_X = data["source"]["X"]
    target_X = data["target"]["X"]
    feature_names = data["feature_names"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, feature in enumerate(feature_names):
        ax = axes[i]

        # Plot distributions
        ax.hist(
            source_X[:, i],
            bins=30,
            alpha=0.6,
            label=f'Source ({data["source"]["population"]})',
            density=True,
        )
        ax.hist(
            target_X[:, i],
            bins=30,
            alpha=0.6,
            label=f'Target ({data["target"]["population"]})',
            density=True,
        )

        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("covariate_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("💾 Saved: covariate_distributions.png")


def estimate_transport_weights(data: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate transport weights using different methods."""
    print("\n⚖️  Step 2: Transport Weight Estimation")
    print("=" * 50)

    source_X = data["source"]["X"]
    target_X = data["target"]["X"]

    # Method 1: Density Ratio Estimation (Classification)
    print("🔬 Method 1: Density Ratio Estimation (Classification)")
    density_estimator = DensityRatioEstimator(
        trim_weights=True,
        max_weight=10.0,
        cross_validate=True,
        cv_folds=5,
        random_state=42,
    )

    weight_result = density_estimator.fit_weights(source_X, target_X)

    print(f"   Effective Sample Size: {weight_result.effective_sample_size:.1f}")
    print(f"   Maximum Weight: {weight_result.max_weight:.2f}")
    print(f"   Stability Ratio: {weight_result.weight_stability_ratio:.2f}")
    print(f"   Relative Efficiency: {weight_result.relative_efficiency:.3f}")
    print(f"   Stable Weights: {'✅' if weight_result.is_stable else '❌'}")

    # Validate weight quality
    print("\n🔍 Weight Quality Validation:")
    validation = density_estimator.validate_weights(
        weight_result.weights, source_X, target_X
    )

    print(
        f"   Good Balance Achieved: {'✅' if validation['good_balance_achieved'] else '❌'}"
    )
    print(f"   Mean Absolute SMD: {validation['mean_absolute_smd']:.3f}")
    print(
        f"   Variables Balanced: {validation['n_variables_balanced']}/{validation['total_variables']}"
    )

    # Visualize weights
    _plot_transport_weights(weight_result.weights, data)

    return {
        "weights": weight_result.weights,
        "result": weight_result,
        "validation": validation,
    }


def _plot_transport_weights(weights: np.ndarray, data: Dict[str, Any]) -> None:
    """Plot transport weight distribution and diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Weight distribution
    axes[0].hist(weights, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(
        np.mean(weights),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(weights):.2f}",
    )
    axes[0].axvline(
        np.median(weights),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(weights):.2f}",
    )
    axes[0].set_xlabel("Transport Weight")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Transport Weights")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Weight vs sample index
    axes[1].scatter(range(len(weights)), weights, alpha=0.6, s=10)
    axes[1].axhline(np.mean(weights), color="red", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Transport Weight")
    axes[1].set_title("Transport Weights by Sample")
    axes[1].grid(True, alpha=0.3)

    # Effective sample size illustration
    eff_n = (np.sum(weights) ** 2) / np.sum(weights**2)
    uniform_eff_n = len(weights)

    categories = ["Actual\n(Weighted)", "Ideal\n(Uniform)"]
    eff_sizes = [eff_n, uniform_eff_n]
    colors = ["lightcoral", "lightblue"]

    bars = axes[2].bar(categories, eff_sizes, color=colors, edgecolor="black")
    axes[2].set_ylabel("Effective Sample Size")
    axes[2].set_title("Effective Sample Size")
    axes[2].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, value in zip(bars, eff_sizes):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("transport_weights.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("💾 Saved: transport_weights.png")


def compare_estimation_methods(
    data: Dict[str, Any], transport_weights: Dict[str, Any]
) -> Dict[str, float]:
    """Compare different causal inference methods with and without transportability."""
    print("\n🎯 Step 3: Causal Effect Estimation & Comparison")
    print("=" * 50)

    source = data["source"]
    target = data["target"]
    feature_names = data["feature_names"]

    # Prepare data objects
    treatment_data = TreatmentData(values=source["T"], name="email_campaign")
    outcome_data = OutcomeData(values=source["Y"], name="purchase_amount")
    covariate_data = CovariateData(values=source["X"], names=feature_names)

    results = {}

    # Method 1: AIPW (standard and transported)
    print("🔬 Method 1: Augmented Inverse Probability Weighting (AIPW)")

    aipw = AIPW(
        outcome_model=RandomForestRegressor(n_estimators=100, random_state=42),
        treatment_model=LogisticRegression(random_state=42, max_iter=1000),
        random_state=42,
        verbose=False,
    )
    aipw.fit(treatment_data, outcome_data, covariate_data)

    # Standard estimate
    aipw_effect = aipw.estimate_ate()
    results["AIPW (Source)"] = aipw_effect.ate
    print(f"   Source Population ATE: ${aipw_effect.ate:.2f}")

    # Transported estimate
    transport_aipw = TransportabilityEstimator(
        base_estimator=aipw, weighting_method="classification", auto_diagnostics=False
    )
    transported_aipw = transport_aipw.estimate_transported_effect(target["X"])
    results["AIPW (Transported)"] = transported_aipw.ate
    print(f"   Transported to Target: ${transported_aipw.ate:.2f}")

    # Method 2: G-Computation
    print("\n🔬 Method 2: G-Computation")

    gcomp = GComputationEstimator(
        outcome_model=RandomForestRegressor(n_estimators=100, random_state=42),
        random_state=42,
        verbose=False,
    )
    gcomp.fit(treatment_data, outcome_data, covariate_data)

    gcomp_effect = gcomp.estimate_ate()
    results["G-Computation (Source)"] = gcomp_effect.ate
    print(f"   Source Population ATE: ${gcomp_effect.ate:.2f}")

    transport_gcomp = TransportabilityEstimator(
        base_estimator=gcomp, auto_diagnostics=False
    )
    transported_gcomp = transport_gcomp.estimate_transported_effect(target["X"])
    results["G-Computation (Transported)"] = transported_gcomp.ate
    print(f"   Transported to Target: ${transported_gcomp.ate:.2f}")

    # Method 3: TMTL (Targeted Maximum Transported Likelihood)
    print("\n🔬 Method 3: Targeted Maximum Transported Likelihood (TMTL)")

    tmtl = TargetedMaximumTransportedLikelihood(
        outcome_model=RandomForestRegressor(n_estimators=100, random_state=42),
        treatment_model=LogisticRegression(random_state=42, max_iter=1000),
        cross_fit=True,
        n_folds=5,
        trim_weights=True,
        max_weight=10.0,
        random_state=42,
        verbose=False,
    )
    tmtl.fit(treatment_data, outcome_data, covariate_data)

    tmtl_effect = tmtl.estimate_transported_ate(target["X"])
    results["TMTL (Transported)"] = tmtl_effect.ate
    print(f"   Transported to Target: ${tmtl_effect.ate:.2f}")

    # Print diagnostics for TMTL
    if tmtl_effect.diagnostics:
        print(
            f"   Transport ESS: {tmtl_effect.diagnostics.get('transport_effective_sample_size', 'N/A'):.1f}"
        )
        print(
            f"   Targeting Converged: {'✅' if tmtl_effect.diagnostics.get('targeting_convergence', False) else '❌'}"
        )

    # True ATE in source
    results["True ATE (Source)"] = data["true_ate_source"]

    return results


def visualize_results(results: Dict[str, float], data: Dict[str, Any]) -> None:
    """Visualize comparison of different methods."""
    print("\n📊 Step 4: Results Visualization")
    print("=" * 50)

    # Separate source and transported estimates
    source_methods = {k: v for k, v in results.items() if "Source" in k or "True" in k}
    transported_methods = {k: v for k, v in results.items() if "Transported" in k}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Source population estimates
    methods = list(source_methods.keys())
    estimates = list(source_methods.values())
    colors = ["lightblue", "lightgreen", "gold"]

    bars1 = axes[0].bar(methods, estimates, color=colors, edgecolor="black", alpha=0.8)
    axes[0].set_ylabel("Average Treatment Effect ($)")
    axes[0].set_title(f'ATE Estimates: {data["source"]["population"]}')
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, value in zip(bars1, estimates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"${value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Transported estimates
    t_methods = [k.replace(" (Transported)", "") for k in transported_methods.keys()]
    t_estimates = list(transported_methods.values())
    t_colors = ["lightcoral", "lightpink", "lightsalmon"]

    bars2 = axes[1].bar(
        t_methods, t_estimates, color=t_colors, edgecolor="black", alpha=0.8
    )
    axes[1].set_ylabel("Average Treatment Effect ($)")
    axes[1].set_title(f'Transported ATE Estimates: {data["target"]["population"]}')
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, value in zip(bars2, t_estimates):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"${value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("ate_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("💾 Saved: ate_comparison.png")

    # Summary table
    print("\n📋 Summary of Results:")
    print("-" * 60)
    print(f"{'Method':<35} {'ATE Estimate':<15}")
    print("-" * 60)
    for method, ate in results.items():
        print(f"{method:<35} ${ate:>10.2f}")
    print("-" * 60)


def generate_business_insights(results: Dict[str, float], data: Dict[str, Any]) -> None:
    """Generate business insights from transportability analysis."""
    print("\n💼 Step 5: Business Insights & Recommendations")
    print("=" * 50)

    # Calculate differences
    source_ate = results.get("AIPW (Source)", 0)
    transported_ate = results.get("AIPW (Transported)", 0)
    tmtl_ate = results.get("TMTL (Transported)", 0)

    difference = transported_ate - source_ate
    percent_change = (difference / source_ate * 100) if source_ate != 0 else 0

    print("🎯 Campaign Performance Analysis:")
    print(f"   Source Market ({data['source']['population']}):")
    print(f"      Expected ATE: ${source_ate:.2f} per customer")
    print(f"      Total customers: {data['source']['n']:,}")
    print(f"      Estimated total impact: ${source_ate * data['source']['n']:,.0f}")

    print(f"\n   Target Market ({data['target']['population']}):")
    print(f"      Transported ATE: ${transported_ate:.2f} per customer")
    print(f"      TMTL ATE: ${tmtl_ate:.2f} per customer")
    print(f"      Total customers: {data['target']['n']:,}")
    print(
        f"      Estimated total impact: ${transported_ate * data['target']['n']:,.0f}"
    )

    print("\n📊 Cross-Population Comparison:")
    print(f"   Difference: ${difference:.2f} per customer ({percent_change:+.1f}%)")

    if abs(percent_change) < 10:
        recommendation = "✅ Similar effectiveness expected. Proceed with confidence."
    elif percent_change > 10:
        recommendation = (
            "📈 Higher effectiveness expected in target market. Strong expansion case."
        )
    else:
        recommendation = (
            "📉 Lower effectiveness expected. Consider market adaptation strategies."
        )

    print(f"   Recommendation: {recommendation}")

    print("\n🔍 Key Insights:")
    print("   • Population differences detected and adjusted for")
    print("   • Transport weighting maintains validity of causal estimates")
    print(
        "   • TMTL provides robust transported estimates with uncertainty quantification"
    )
    print("   • Consider local market factors not captured in covariates")

    print("\n🚀 Next Steps:")
    print("   1. Validate transported estimates with small-scale pilot")
    print("   2. Monitor performance and update transport weights with new data")
    print("   3. Consider market-specific adaptations based on cultural differences")
    print("   4. Set up continuous monitoring for population drift")


def main():
    """Run the complete transportability analysis example."""
    print("🌍 Transportability & Target Population Weighting")
    print("📧 Marketing Campaign Analysis Example")
    print("=" * 60)

    # Generate data
    data = generate_marketing_data()

    # Step 1: Analyze covariate shift
    analyze_covariate_shift(data)

    # Step 2: Estimate transport weights
    transport_weights = estimate_transport_weights(data)

    # Step 3: Compare estimation methods
    results = compare_estimation_methods(data, transport_weights)

    # Step 4: Visualize results
    visualize_results(results, data)

    # Step 5: Generate business insights
    generate_business_insights(results, data)

    print("\n✅ Analysis Complete!")
    print(
        "📁 Generated files: covariate_distributions.png, transport_weights.png, ate_comparison.png"
    )
    print("🎉 Your marketing campaign is ready for international expansion!")


if __name__ == "__main__":
    main()
