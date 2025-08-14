#!/usr/bin/env python3
"""
Advanced Diagnostic Visualization Example

This script demonstrates the comprehensive diagnostic visualization capabilities
for causal inference analysis, including Love plots, weight diagnostics,
propensity score analysis, residual diagnostics, and automated HTML reports.

The example uses a realistic marketing scenario to show how these tools can
help assess analysis quality and communicate results to stakeholders.
"""

from pathlib import Path

import numpy as np

# Import causal inference components
from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator

# Import visualization components
from causal_inference.visualization import (
    create_love_plot,
    create_propensity_plots,
    create_residual_plots,
    create_weight_plots,
    generate_diagnostic_report,
)


def create_marketing_campaign_data(n=1000, seed=42):
    """
    Create realistic marketing campaign data with confounding.

    Scenario: Email marketing campaign for an e-commerce company
    - Treatment: Receiving personalized email campaign
    - Outcome: Purchase amount in next 30 days
    - Confounders: Customer demographics and behavior
    """
    np.random.seed(seed)

    print(f"📊 Generating realistic marketing data with {n:,} customers...")

    # Customer demographics and behavior (confounders)
    customer_age = np.random.normal(35, 12, n)
    customer_age = np.clip(customer_age, 18, 80)  # Realistic age range

    annual_income = np.random.lognormal(np.log(50000), 0.4, n)
    annual_income = np.clip(annual_income, 20000, 200000)  # Realistic income range

    past_purchases = np.random.poisson(2.5, n)
    engagement_score = np.random.beta(2, 5, n)  # Skewed toward low engagement

    # Customer segment (premium/regular/budget)
    segment_prob = np.array([0.15, 0.6, 0.25])  # 15% premium, 60% regular, 25% budget
    segment = np.random.choice(["premium", "regular", "budget"], n, p=segment_prob)

    # Create dummy variables for segment
    is_premium = (segment == "premium").astype(int)
    is_regular = (segment == "regular").astype(int)

    # Treatment assignment with realistic confounding
    # Marketing teams often target engaged, high-value customers
    propensity_logit = (
        -1.5  # Base propensity
        + 0.01 * customer_age  # Slightly prefer older customers
        + 0.00002 * annual_income  # Prefer higher income
        + 0.2 * past_purchases  # Target frequent buyers
        + 2.0 * engagement_score  # Strongly target engaged users
        + 0.8 * is_premium  # Prefer premium customers
        + 0.3 * is_regular  # Moderate preference for regular customers
    )

    propensity_scores = 1 / (1 + np.exp(-propensity_logit))
    treatment_assignment = np.random.binomial(1, propensity_scores, n)

    # Outcome: Purchase amount with realistic treatment effect
    # Base purchase amount depends on customer characteristics
    base_purchase = (
        20  # Base amount
        + 0.8 * customer_age  # Age effect
        + 0.0008 * annual_income  # Income effect
        + 15 * past_purchases  # Past behavior strong predictor
        + 50 * engagement_score  # Engagement drives purchases
        + 80 * is_premium  # Premium customers spend more
        + 30 * is_regular  # Regular vs budget difference
    )

    # Treatment effect: Personalized emails increase purchase amount
    true_treatment_effect = 25  # $25 average increase

    # Add treatment effect and noise
    purchase_amount = (
        base_purchase
        + true_treatment_effect * treatment_assignment
        + np.random.normal(0, 20, n)  # Random noise
    )

    # Ensure non-negative purchases
    purchase_amount = np.maximum(purchase_amount, 0)

    # Create data objects
    treatment_data = TreatmentData(
        values=treatment_assignment, treatment_type="binary", name="email_campaign"
    )

    outcome_data = OutcomeData(
        values=purchase_amount, outcome_type="continuous", name="purchase_amount_30d"
    )

    covariate_data = CovariateData(
        values=np.column_stack(
            [
                customer_age,
                annual_income,
                past_purchases,
                engagement_score,
                is_premium,
                is_regular,
            ]
        ),
        names=[
            "customer_age",
            "annual_income",
            "past_purchases",
            "engagement_score",
            "is_premium",
            "is_regular",
        ],
    )

    # Print data summary
    print(f"   Treatment rate: {np.mean(treatment_assignment):.1%}")
    print(f"   True treatment effect: ${true_treatment_effect}")
    print(
        f"   Average purchase (control): ${np.mean(purchase_amount[treatment_assignment == 0]):.2f}"
    )
    print(
        f"   Average purchase (treated): ${np.mean(purchase_amount[treatment_assignment == 1]):.2f}"
    )
    print()

    return {
        "treatment_data": treatment_data,
        "outcome_data": outcome_data,
        "covariate_data": covariate_data,
        "true_effect": true_treatment_effect,
        "true_propensity": propensity_scores,
    }


def demonstrate_love_plots(data):
    """Demonstrate Love plot creation and covariate balance assessment."""
    print("📈 Creating Love Plots for Covariate Balance Assessment...")

    # Create Love plot before adjustment
    print("   Generating static Love plot...")
    _ = create_love_plot(
        covariates=data["covariate_data"],
        treatment=data["treatment_data"],
        title="Covariate Balance Before Adjustment",
        save_path="love_plot_before.png",
        interactive=False,
    )

    # Create interactive Love plot (if Plotly available)
    try:
        print("   Generating interactive Love plot...")
        _ = create_love_plot(
            covariates=data["covariate_data"],
            treatment=data["treatment_data"],
            title="Interactive Covariate Balance Assessment",
            save_path="love_plot_interactive.html",
            interactive=True,
        )
        print("   ✅ Interactive Love plot saved to love_plot_interactive.html")
    except ImportError:
        print("   ⚠️ Plotly not available, skipping interactive plot")

    print("   ✅ Love plot saved to love_plot_before.png")
    print()


def demonstrate_weight_diagnostics(data):
    """Demonstrate weight distribution analysis."""
    print("⚖️ Analyzing Weight Distributions...")

    # Fit IPW estimator to get weights
    print("   Fitting IPW estimator to obtain weights...")
    ipw_estimator = IPWEstimator(bootstrap_samples=0, verbose=False)
    ipw_estimator.fit(
        data["treatment_data"], data["outcome_data"], data["covariate_data"]
    )

    weights = ipw_estimator.get_weights()

    # Create weight diagnostic plots
    print("   Generating weight distribution diagnostics...")
    weight_fig, weight_result, weight_recommendations = create_weight_plots(
        weights=weights,
        title="IPW Weight Distribution Analysis",
        save_path="weight_diagnostics.png",
        interactive=False,
    )

    print("   📊 Weight Statistics:")
    print(f"      Mean weight: {weight_result.mean_weight:.3f}")
    print(f"      Max weight: {weight_result.max_weight:.3f}")
    print(
        f"      Extreme weights: {weight_result.extreme_weight_count} ({weight_result.extreme_weight_percentage:.1f}%)"
    )
    print(f"      Effective sample size: {weight_result.effective_sample_size:.0f}")

    print("   💡 Weight Recommendations:")
    for rec in weight_recommendations[:3]:  # Show first 3 recommendations
        print(f"      {rec}")

    print("   ✅ Weight diagnostics saved to weight_diagnostics.png")
    print()


def demonstrate_propensity_analysis(data):
    """Demonstrate propensity score overlap analysis."""
    print("🎲 Analyzing Propensity Score Overlap...")

    # Use true propensity scores for demonstration
    propensity_scores = data["true_propensity"]

    # Create propensity score diagnostic plots
    print("   Generating propensity score diagnostics...")
    prop_fig, prop_result, prop_recommendations = create_propensity_plots(
        propensity_scores=propensity_scores,
        treatment=data["treatment_data"],
        title="Propensity Score Overlap Analysis",
        save_path="propensity_diagnostics.png",
        interactive=False,
    )

    print("   📊 Propensity Score Statistics:")
    print(f"      Overlap percentage: {prop_result.overlap_percentage:.1f}%")
    print(f"      AUC score: {prop_result.auc_score:.3f}")
    print(f"      Positivity violations: {prop_result.positivity_violations}")
    print(
        f"      Common support range: [{prop_result.common_support_range[0]:.3f}, {prop_result.common_support_range[1]:.3f}]"
    )

    print("   💡 Propensity Score Recommendations:")
    for rec in prop_recommendations[:3]:
        print(f"      {rec}")

    print("   ✅ Propensity diagnostics saved to propensity_diagnostics.png")
    print()


def demonstrate_residual_analysis(data):
    """Demonstrate residual analysis for model diagnostics."""
    print("📈 Performing Residual Analysis...")

    # Fit G-computation estimator to get residuals
    print("   Fitting G-computation estimator...")
    gcomp_estimator = GComputationEstimator(bootstrap_samples=0, verbose=False)
    gcomp_estimator.fit(
        data["treatment_data"], data["outcome_data"], data["covariate_data"]
    )

    # Get fitted values and residuals
    X_design = np.column_stack(
        [data["treatment_data"].values, data["covariate_data"].values]
    )
    fitted_values = gcomp_estimator.outcome_model.predict(X_design)
    residuals = data["outcome_data"].values - fitted_values

    # Create residual diagnostic plots
    print("   Generating residual diagnostics...")
    design_matrix = np.column_stack(
        [
            np.ones(len(fitted_values)),  # intercept
            X_design,
        ]
    )

    resid_fig, resid_result, resid_recommendations = create_residual_plots(
        residuals=residuals,
        fitted_values=fitted_values,
        design_matrix=design_matrix,
        title="G-computation Residual Analysis",
        save_path="residual_diagnostics.png",
        interactive=False,
    )

    print("   📊 Residual Analysis Results:")
    print(f"      Normality test p-value: {resid_result.shapiro_pvalue:.4f}")
    print(
        f"      Homoscedasticity test p-value: {resid_result.breusch_pagan_pvalue:.4f}"
    )
    print(f"      Outliers detected: {resid_result.outlier_count}")
    print(f"      Normality assumption met: {resid_result.normality_assumption_met}")
    print(
        f"      Homoscedasticity assumption met: {resid_result.homoscedasticity_assumption_met}"
    )

    print("   💡 Residual Analysis Recommendations:")
    for rec in resid_recommendations[:3]:
        print(f"      {rec}")

    print("   ✅ Residual diagnostics saved to residual_diagnostics.png")
    print()


def demonstrate_comprehensive_report(data):
    """Demonstrate comprehensive HTML report generation."""
    print("📋 Generating Comprehensive Diagnostic Report...")

    # Fit AIPW estimator for comprehensive analysis
    print("   Fitting AIPW (doubly robust) estimator...")
    aipw_estimator = AIPWEstimator(
        cross_fitting=False,  # Disable for faster demo
        bootstrap_samples=50,  # Small number for demo
        verbose=False,
    )
    aipw_estimator.fit(
        data["treatment_data"], data["outcome_data"], data["covariate_data"]
    )

    effect_result = aipw_estimator.estimate_ate()

    # Get all diagnostic components
    propensity_scores = aipw_estimator.propensity_estimator.get_propensity_scores()
    weights = aipw_estimator.propensity_estimator.get_weights()

    # Get residuals from outcome model
    if hasattr(aipw_estimator, "outcome_estimator"):
        X_design = np.column_stack(
            [data["treatment_data"].values, data["covariate_data"].values]
        )
        fitted_values = aipw_estimator.outcome_estimator.outcome_model.predict(X_design)
        residuals = data["outcome_data"].values - fitted_values
    else:
        fitted_values = np.mean(data["outcome_data"].values)
        residuals = data["outcome_data"].values - fitted_values

    # Generate comprehensive report
    print("   Generating comprehensive HTML report...")
    html_report = generate_diagnostic_report(
        treatment_data=data["treatment_data"],
        outcome_data=data["outcome_data"],
        covariates=data["covariate_data"],
        weights=weights,
        propensity_scores=propensity_scores,
        residuals=residuals,
        fitted_values=fitted_values,
        estimator_name="Augmented Inverse Probability Weighting (AIPW)",
        ate_estimate=effect_result.ate,
        ate_ci_lower=getattr(effect_result, "ate_ci_lower", None),
        ate_ci_upper=getattr(effect_result, "ate_ci_upper", None),
        save_path="comprehensive_diagnostic_report.html",
        template_type="comprehensive",
    )

    print("   📊 Analysis Results:")
    print(f"      Estimated ATE: ${effect_result.ate:.2f}")
    print(f"      True ATE: ${data['true_effect']:.2f}")
    print(
        f"      Estimation error: ${abs(effect_result.ate - data['true_effect']):.2f}"
    )

    if (
        hasattr(effect_result, "ate_ci_lower")
        and effect_result.ate_ci_lower is not None
    ):
        print(
            f"      95% CI: [${effect_result.ate_ci_lower:.2f}, ${effect_result.ate_ci_upper:.2f}]"
        )
        ci_contains_truth = (
            effect_result.ate_ci_lower
            <= data["true_effect"]
            <= effect_result.ate_ci_upper
        )
        print(f"      CI contains true effect: {ci_contains_truth}")

    print(f"   📄 Report size: {len(html_report):,} characters")
    print("   ✅ Comprehensive report saved to comprehensive_diagnostic_report.html")
    print()


def main():
    """Main demonstration function."""
    print("🔬 Advanced Diagnostic Visualization Demonstration")
    print("=" * 60)
    print()
    print("This example demonstrates the comprehensive diagnostic visualization")
    print("capabilities for causal inference analysis, using a realistic")
    print("marketing campaign scenario.")
    print()

    # Create output directory
    output_dir = Path("diagnostic_outputs")
    output_dir.mkdir(exist_ok=True)

    # Change to output directory
    import os

    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        # Generate realistic data
        data = create_marketing_campaign_data(n=2000, seed=42)

        # Demonstrate each visualization component
        demonstrate_love_plots(data)
        demonstrate_weight_diagnostics(data)
        demonstrate_propensity_analysis(data)
        demonstrate_residual_analysis(data)
        demonstrate_comprehensive_report(data)

        print("🎉 Demonstration Complete!")
        print("=" * 60)
        print()
        print("Generated files in ./diagnostic_outputs/:")
        print("   📊 love_plot_before.png - Covariate balance visualization")
        print(
            "   📊 love_plot_interactive.html - Interactive balance plot (if Plotly available)"
        )
        print("   ⚖️ weight_diagnostics.png - Weight distribution analysis")
        print("   🎲 propensity_diagnostics.png - Propensity score overlap")
        print("   📈 residual_diagnostics.png - Model residual analysis")
        print("   📋 comprehensive_diagnostic_report.html - Complete diagnostic report")
        print()
        print("💡 Open the HTML files in a web browser to view interactive content!")
        print()
        print("Key Benefits of This Diagnostic Framework:")
        print("   ✅ Automated quality assessment with actionable recommendations")
        print(
            "   ✅ Professional visualizations suitable for stakeholder communication"
        )
        print("   ✅ Comprehensive coverage of causal inference assumptions")
        print("   ✅ Integration with all major causal inference estimators")
        print("   ✅ Scalable to large datasets with fast generation times")

    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
