"""Example: Bayesian Causal Inference with NHEFS Dataset.

This example demonstrates the Bayesian causal inference estimator
using the NHEFS dataset to estimate the effect of smoking cessation
on weight change, as specified in Issue #60.

The analysis compares the Bayesian posterior mean to the AIPW estimate
and provides full posterior uncertainty quantification.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.data.nhefs import load_nhefs_data
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.bayesian import BayesianEstimator


def main():
    """Run Bayesian causal inference example with NHEFS data."""
    print("=== Bayesian Causal Inference Example with NHEFS Dataset ===")
    print()

    # Load NHEFS data
    print("Loading NHEFS dataset...")
    nhefs_data = load_nhefs_data()

    # Select variables as specified in Issue #60
    # Treatment: qsmk (quit smoking)
    # Outcome: wt82_71 (weight change from 1971 to 1982)
    # Covariates: age, sex, education, income

    # Filter for complete cases
    required_vars = ["qsmk", "wt82_71", "age", "sex", "education", "income"]
    complete_data = nhefs_data.dropna(subset=required_vars)

    print(f"Complete cases: {len(complete_data)}")
    print(f"Treatment distribution: {complete_data['qsmk'].value_counts().to_dict()}")
    print(f"Outcome mean: {complete_data['wt82_71'].mean():.2f}")
    print()

    # Prepare data objects
    treatment_data = TreatmentData(
        values=complete_data["qsmk"], treatment_type="binary", name="qsmk"
    )

    outcome_data = OutcomeData(
        values=complete_data["wt82_71"], outcome_type="continuous", name="wt82_71"
    )

    covariates_df = complete_data[["age", "sex", "education", "income"]]
    covariate_data = CovariateData(
        values=covariates_df, names=["age", "sex", "education", "income"]
    )

    print("Data objects created successfully.")
    print()

    # Benchmark with AIPW estimator
    print("=== AIPW Benchmark ===")
    aipw_estimator = AIPWEstimator(
        outcome_model_type="linear",
        propensity_model_type="logistic",
        random_state=42,
        verbose=True,
    )

    print("Fitting AIPW estimator...")
    aipw_estimator.fit(treatment_data, outcome_data, covariate_data)
    aipw_effect = aipw_estimator.estimate_ate()

    print(f"AIPW ATE: {aipw_effect.ate:.3f}")
    if aipw_effect.ate_ci_lower and aipw_effect.ate_ci_upper:
        print(
            f"AIPW 95% CI: [{aipw_effect.ate_ci_lower:.3f}, {aipw_effect.ate_ci_upper:.3f}]"
        )
    print()

    # Bayesian estimation
    print("=== Bayesian Estimation ===")
    bayesian_estimator = BayesianEstimator(
        prior_treatment_scale=2.5,  # Weakly informative prior
        mcmc_draws=2000,
        mcmc_tune=1000,
        mcmc_chains=4,
        credible_level=0.95,
        random_state=42,
        verbose=True,
    )

    print("Fitting Bayesian estimator...")
    bayesian_estimator.fit(treatment_data, outcome_data, covariate_data)
    bayesian_effect = bayesian_estimator.estimate_ate()

    print(f"Bayesian ATE (posterior mean): {bayesian_effect.ate:.3f}")
    print(f"Bayesian ATE (posterior std): {bayesian_effect.ate_se:.3f}")
    print(
        f"Bayesian 95% credible interval: [{bayesian_effect.ate_credible_lower:.3f}, {bayesian_effect.ate_credible_upper:.3f}]"
    )
    print()

    # MCMC diagnostics
    print("=== MCMC Diagnostics ===")
    print(f"Effective sample size: {bayesian_effect.effective_sample_size:.0f}")
    print(f"R-hat: {bayesian_effect.r_hat:.4f}")
    print()

    # Check convergence
    if bayesian_effect.r_hat > 1.1:
        print("⚠️  Warning: R-hat > 1.1 indicates potential convergence issues")
    else:
        print("✅ R-hat < 1.1 indicates good convergence")

    if bayesian_effect.effective_sample_size < 100:
        print("⚠️  Warning: Effective sample size < 100, consider more MCMC draws")
    else:
        print("✅ Effective sample size > 100")
    print()

    # Compare estimates
    print("=== Comparison of Methods ===")
    print(f"AIPW ATE:     {aipw_effect.ate:.3f}")
    print(f"Bayesian ATE: {bayesian_effect.ate:.3f}")

    difference = abs(bayesian_effect.ate - aipw_effect.ate)
    relative_difference = difference / abs(aipw_effect.ate) * 100

    print(f"Absolute difference: {difference:.3f}")
    print(f"Relative difference: {relative_difference:.1f}%")

    # Check KPI from Issue #60: within ±10%
    if relative_difference <= 10:
        print("✅ KPI met: Posterior mean within ±10% of AIPW estimate")
    else:
        print("❌ KPI not met: Posterior mean > ±10% from AIPW estimate")
    print()

    # Print summary table
    print("=== Summary Table ===")
    summary_df = pd.DataFrame(
        {
            "Method": ["AIPW", "Bayesian"],
            "ATE": [aipw_effect.ate, bayesian_effect.ate],
            "CI_Lower": [
                aipw_effect.ate_ci_lower or np.nan,
                bayesian_effect.ate_credible_lower,
            ],
            "CI_Upper": [
                aipw_effect.ate_ci_upper or np.nan,
                bayesian_effect.ate_credible_upper,
            ],
            "CI_Width": [
                (aipw_effect.ate_ci_upper - aipw_effect.ate_ci_lower)
                if aipw_effect.ate_ci_upper
                else np.nan,
                bayesian_effect.ate_credible_upper - bayesian_effect.ate_credible_lower,
            ],
        }
    )
    print(summary_df.round(3).to_string(index=False))
    print()

    # Visualizations
    print("=== Creating Visualizations ===")

    # Plot posterior distribution
    plt.figure(figsize=(12, 8))

    # Subplot 1: Posterior distribution
    plt.subplot(2, 2, 1)
    bayesian_estimator.plot_posterior(var_names=["treatment_effect"])
    plt.title("Posterior Distribution of Treatment Effect")

    # Subplot 2: Trace plot
    plt.subplot(2, 2, 2)
    bayesian_estimator.plot_trace(var_names=["treatment_effect"])

    # Subplot 3: Comparison of estimates
    plt.subplot(2, 2, 3)
    methods = ["AIPW", "Bayesian"]
    estimates = [aipw_effect.ate, bayesian_effect.ate]
    ci_lower = [
        aipw_effect.ate_ci_lower or bayesian_effect.ate - 1,
        bayesian_effect.ate_credible_lower,
    ]
    ci_upper = [
        aipw_effect.ate_ci_upper or bayesian_effect.ate + 1,
        bayesian_effect.ate_credible_upper,
    ]

    plt.errorbar(
        methods,
        estimates,
        yerr=[
            np.array(estimates) - np.array(ci_lower),
            np.array(ci_upper) - np.array(estimates),
        ],
        fmt="o",
        capsize=5,
        capthick=2,
        markersize=8,
    )
    plt.ylabel("Average Treatment Effect")
    plt.title("Comparison of ATE Estimates")
    plt.grid(True, alpha=0.3)

    # Subplot 4: Posterior histogram
    plt.subplot(2, 2, 4)
    plt.hist(
        bayesian_effect.posterior_samples,
        bins=50,
        alpha=0.7,
        density=True,
        color="skyblue",
        edgecolor="black",
    )
    plt.axvline(
        bayesian_effect.ate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Posterior Mean: {bayesian_effect.ate:.3f}",
    )
    plt.axvline(
        aipw_effect.ate,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"AIPW Estimate: {aipw_effect.ate:.3f}",
    )
    plt.xlabel("Treatment Effect")
    plt.ylabel("Density")
    plt.title("Posterior Samples Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bayesian_nhefs_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Analysis complete! Visualization saved as 'bayesian_nhefs_analysis.png'")

    # Additional diagnostics
    print("\n=== Full Model Summary ===")
    print(bayesian_estimator.parameter_summary())

    # Posterior predictive check
    print("\n=== Posterior Predictive Check ===")
    try:
        bayesian_estimator.posterior_predictive_check(n_samples=100)
        plt.title("Posterior Predictive Check")
        plt.savefig("bayesian_ppc.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("Posterior predictive check completed. Saved as 'bayesian_ppc.png'")
    except Exception as e:
        print(f"Posterior predictive check failed: {e}")


if __name__ == "__main__":
    main()
