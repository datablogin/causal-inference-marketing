"""
Example: Causal Mediation Analysis with NHEFS Dataset

This example demonstrates how to use the MediationEstimator to analyze
the causal pathway: qsmk → smokeintensity (mediator) → wt82_71

The analysis estimates:
- Natural Direct Effect (NDE): Direct effect of smoking cessation on weight change
- Natural Indirect Effect (NIE): Indirect effect through smoking intensity
- Mediated Proportion: Fraction of total effect mediated by smoking intensity
"""

import numpy as np
import pandas as pd

from causal_inference.core.base import CovariateData
from causal_inference.core.bootstrap import BootstrapConfig
from causal_inference.data.nhefs import load_nhefs
from causal_inference.estimators.mediation import MediationEstimator, MediatorData


def main():
    """Run mediation analysis example with NHEFS data."""

    print("=== Causal Mediation Analysis: NHEFS Dataset ===")
    print("Analyzing: qsmk → smokeintensity → wt82_71")
    print()

    # Load NHEFS data
    print("Loading NHEFS dataset...")
    try:
        treatment_data, outcome_data, covariate_data = load_nhefs(
            treatment="qsmk",
            outcome="wt82_71",
            confounders=[
                "sex",
                "age",
                "race",
                "education",
                "smokeintensity",  # This will be our mediator
                "smokeyrs",
                "exercise",
                "active",
                "wt71",
                "asthma",
                "bronch",
            ],
        )

        print(f"Loaded {len(treatment_data.values)} observations")
        print(
            f"Treatment distribution: {pd.Series(treatment_data.values).value_counts().to_dict()}"
        )
        print()

    except FileNotFoundError as e:
        print(f"Error loading NHEFS data: {e}")
        print("Please ensure nhefs.csv is in the project root directory")
        return

    # Extract mediator from covariates
    print("Extracting mediator variable (smokeintensity)...")

    if "smokeintensity" not in covariate_data.names:
        print("Error: smokeintensity not found in covariates")
        return

    # Get mediator values
    mediator_idx = covariate_data.names.index("smokeintensity")
    if isinstance(covariate_data.values, pd.DataFrame):
        mediator_values = covariate_data.values.iloc[:, mediator_idx]
    else:
        mediator_values = covariate_data.values[:, mediator_idx]

    mediator_data = MediatorData(
        values=mediator_values, name="smokeintensity", mediator_type="continuous"
    )

    # Remove mediator from covariates to avoid confounding
    remaining_covariates = [
        name for name in covariate_data.names if name != "smokeintensity"
    ]
    remaining_indices = [
        i for i, name in enumerate(covariate_data.names) if name != "smokeintensity"
    ]

    if isinstance(covariate_data.values, pd.DataFrame):
        covariate_values = covariate_data.values.iloc[:, remaining_indices]
    else:
        covariate_values = covariate_data.values[:, remaining_indices]

    adjusted_covariate_data = CovariateData(
        values=covariate_values, names=remaining_covariates
    )

    print("Mediator statistics:")
    print(f"  Mean: {np.mean(mediator_values):.2f}")
    print(f"  Std:  {np.std(mediator_values):.2f}")
    print(f"  Range: [{np.min(mediator_values):.2f}, {np.max(mediator_values):.2f}]")
    print(f"Remaining covariates: {len(remaining_covariates)}")
    print()

    # Set up bootstrap configuration for confidence intervals
    bootstrap_config = BootstrapConfig(
        n_samples=1000, confidence_level=0.95, random_state=42
    )

    # Initialize mediation estimator
    print("Initializing mediation estimator...")
    estimator = MediationEstimator(
        mediator_model_type="auto",  # Will select appropriate model automatically
        outcome_model_type="auto",
        bootstrap_config=bootstrap_config,
        random_state=42,
        verbose=True,
    )

    # Fit the estimator
    print("Fitting mediation models...")
    print("  1. Mediator model: smokeintensity ~ qsmk + covariates")
    print("  2. Outcome model: wt82_71 ~ qsmk + smokeintensity + covariates")

    estimator.fit(
        treatment=treatment_data,
        outcome=outcome_data,
        mediator=mediator_data,
        covariates=adjusted_covariate_data,
    )

    print("Models fitted successfully!")
    print()

    # Estimate mediation effects
    print("Estimating mediation effects...")
    print("(This may take a few minutes due to bootstrap resampling)")

    effect = estimator.estimate_ate()

    # Display results
    print("=== MEDIATION ANALYSIS RESULTS ===")
    print()

    print("Total Effect (ATE):")
    print(f"  Estimate: {effect.ate:.4f}")
    if effect.ate_ci_lower is not None:
        print(f"  95% CI: [{effect.ate_ci_lower:.4f}, {effect.ate_ci_upper:.4f}]")
        print(f"  Significant: {'Yes' if effect.is_significant else 'No'}")
    print()

    print("Natural Direct Effect (NDE):")
    print(f"  Estimate: {effect.nde:.4f}")
    if effect.nde_ci_lower is not None:
        print(f"  95% CI: [{effect.nde_ci_lower:.4f}, {effect.nde_ci_upper:.4f}]")
    print()

    print("Natural Indirect Effect (NIE):")
    print(f"  Estimate: {effect.nie:.4f}")
    if effect.nie_ci_lower is not None:
        print(f"  95% CI: [{effect.nie_ci_lower:.4f}, {effect.nie_ci_upper:.4f}]")
        print(f"  Evidence of mediation: {'Yes' if effect.is_mediated else 'No'}")
    print()

    print("Mediated Proportion:")
    print(
        f"  Estimate: {effect.mediated_proportion:.4f} ({effect.mediated_proportion * 100:.1f}%)"
    )
    if effect.mediated_prop_ci_lower is not None:
        print(
            f"  95% CI: [{effect.mediated_prop_ci_lower:.4f}, {effect.mediated_prop_ci_upper:.4f}]"
        )
        print(
            f"           ({effect.mediated_prop_ci_lower * 100:.1f}%, {effect.mediated_prop_ci_upper * 100:.1f}%)"
        )
    print()

    # Interpretation
    print("=== INTERPRETATION ===")
    print()

    if effect.mediated_proportion > 0:
        mediation_strength = (
            "strong"
            if effect.mediated_proportion > 0.5
            else "moderate"
            if effect.mediated_proportion > 0.2
            else "weak"
        )
        print(
            f"The analysis suggests {mediation_strength} mediation by smoking intensity."
        )
        print(
            f"Approximately {effect.mediated_proportion * 100:.1f}% of the total effect of"
        )
        print("smoking cessation on weight change operates through changes in")
        print("smoking intensity.")
    else:
        print("The analysis suggests little to no mediation by smoking intensity.")
    print()

    if effect.is_mediated:
        print("The confidence interval for the indirect effect excludes zero,")
        print("providing evidence for a mediation mechanism.")
    else:
        print("The confidence interval for the indirect effect includes zero,")
        print("suggesting limited evidence for mediation.")
    print()

    # Model diagnostics
    print("=== MODEL DIAGNOSTICS ===")
    print(f"Sample size: {effect.n_observations}")
    print(f"Bootstrap samples: {effect.bootstrap_samples}")
    print(f"Estimation method: {effect.method}")

    # Check effect decomposition
    decomposition_error = abs((effect.nde + effect.nie) - effect.ate)
    print(f"Effect decomposition error: {decomposition_error:.6f}")
    if decomposition_error < 1e-10:
        print("✓ Effect decomposition is numerically consistent")
    else:
        print("⚠ Effect decomposition has numerical inconsistencies")

    print()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
