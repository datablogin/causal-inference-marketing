"""Example of Regression Discontinuity Design (RDD) with NHEFS-like data.

This example demonstrates the RDD functionality as specified in GitHub issue #61:
- Uses age as forcing variable with cutoff at age 50
- Simulates binary treatment assignment based on age cutoff
- Uses weight change (wt82_71) as outcome
- Tests the KPI: RDD estimate should match simulated effect ±10%
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from causal_inference.estimators.regression_discontinuity import (  # type: ignore[import-not-found]
    RDDEstimator,
)


def simulate_nhefs_rdd_data(
    n: int = 1000, random_seed: int = 42
) -> tuple[NDArray[Any], NDArray[Any], float]:
    """Simulate NHEFS-like data for RDD analysis.

    Args:
        n: Number of observations
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (age, weight_change, true_treatment_effect)
    """
    np.random.seed(random_seed)

    # Generate age as forcing variable (similar to NHEFS distribution)
    age = np.random.normal(45, 12, n)
    age = np.clip(age, 25, 70)  # Reasonable age range

    # Treatment assignment rule: age >= 50
    cutoff = 50.0
    treatment = (age >= cutoff).astype(int)

    # Simulate weight change with realistic effects
    true_treatment_effect = 2.0  # kg - meaningful weight change

    # Age has a natural effect on weight change (older people tend to gain less)
    age_effect = -0.05 * (age - 45)

    # Add some polynomial age effect for realism
    age_squared_effect = 0.001 * ((age - 45) ** 2)

    # Random noise
    noise = np.random.normal(0, 3, n)

    # Generate outcome: weight change from baseline to follow-up
    wt82_71 = (
        age_effect - age_squared_effect + true_treatment_effect * treatment + noise
    )

    return age, wt82_71, true_treatment_effect


def run_rdd_analysis(
    age: NDArray[Any], weight_change: NDArray[Any], cutoff: float = 50.0
) -> Any:
    """Run comprehensive RDD analysis.

    Args:
        age: Age values (forcing variable)
        weight_change: Weight change outcome
        cutoff: Treatment assignment cutoff
    """
    print("=" * 60)
    print("REGRESSION DISCONTINUITY DESIGN (RDD) ANALYSIS")
    print("=" * 60)

    # Initialize RDD estimator
    estimator = RDDEstimator(
        cutoff=cutoff,
        bandwidth=None,  # Auto-select optimal bandwidth
        polynomial_order=1,  # Start with linear
        kernel="triangular",
        verbose=True,
    )

    print("\n1. DATA SUMMARY")
    print(f"   Sample size: {len(age)}")
    print(f"   Age range: {age.min():.1f} - {age.max():.1f}")
    print(f"   Cutoff: {cutoff}")
    print(f"   Observations below cutoff: {np.sum(age < cutoff)}")
    print(f"   Observations above cutoff: {np.sum(age >= cutoff)}")

    # Estimate RDD effect using the generic estimate_rdd function
    print("\n2. RDD ESTIMATION")
    result = estimator.estimate_rdd(
        forcing_variable=age, outcome=weight_change, cutoff=cutoff
    )

    print(f"   RDD Treatment Effect: {result.ate:.3f} kg")
    if result.ate_se is not None:
        print(f"   Standard Error: {result.ate_se:.3f}")
    if result.ate_ci_lower is not None and result.ate_ci_upper is not None:
        print(
            f"   95% Confidence Interval: [{result.ate_ci_lower:.3f}, {result.ate_ci_upper:.3f}]"
        )

    print(f"   Bandwidth used: {result.bandwidth:.2f} years")
    print(f"   Left sample size: {result.n_left}")
    print(f"   Right sample size: {result.n_right}")

    if result.left_model_r2 is not None and result.right_model_r2 is not None:
        print(f"   Left model R²: {result.left_model_r2:.3f}")
        print(f"   Right model R²: {result.right_model_r2:.3f}")

    # Test different polynomial orders
    print("\n3. ROBUSTNESS CHECKS")
    print("   Testing different polynomial orders:")

    for poly_order in [1, 2, 3]:
        estimator_poly = RDDEstimator(
            cutoff=cutoff,
            bandwidth=result.bandwidth,  # Use same bandwidth
            polynomial_order=poly_order,
            kernel="triangular",
            verbose=False,
        )

        result_poly = estimator_poly.estimate_rdd(
            forcing_variable=age, outcome=weight_change, cutoff=cutoff
        )

        print(f"     Polynomial order {poly_order}: {result_poly.ate:.3f} kg")

    # Placebo test
    placebo_cutoff = cutoff - 5  # Test at age 45
    try:
        p_value = estimator.run_placebo_test(placebo_cutoff)
        print(f"   Placebo test (cutoff at {placebo_cutoff}): p-value = {p_value:.3f}")
        if p_value > 0.05:
            print("     ✓ Placebo test passed (non-significant)")
        else:
            print("     ⚠ Placebo test suggests possible confounding")
    except Exception as e:
        print(f"     Placebo test failed: {str(e)}")

    return result


def create_rdd_visualization(
    age: NDArray[Any], weight_change: NDArray[Any], cutoff: float = 50.0
) -> Any:
    """Create comprehensive RDD visualization.

    Args:
        age: Age values (forcing variable)
        weight_change: Weight change outcome
        cutoff: Treatment assignment cutoff
    """
    estimator = RDDEstimator(
        cutoff=cutoff, bandwidth=None, polynomial_order=1, kernel="triangular"
    )

    # Fit the model
    estimator.estimate_rdd(forcing_variable=age, outcome=weight_change, cutoff=cutoff)

    # Create the RDD plot
    fig = estimator.plot_rdd(figsize=(12, 8))

    # Enhance the plot
    ax = fig.gca()
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Weight Change (kg)", fontsize=12)
    ax.set_title(
        "Regression Discontinuity Design: Effect of Treatment on Weight Change",
        fontsize=14,
    )

    # Add some statistics to the plot
    result = estimator._rdd_result
    if result:
        textstr = f"RDD Effect: {result.ate:.2f} kg\nBandwidth: {result.bandwidth:.1f} years\nN: {result.n_left + result.n_right}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

    plt.tight_layout()
    plt.show()

    return fig


def main() -> None:
    """Run the complete RDD example."""
    print("RDD Analysis with NHEFS-like Data")
    print("GitHub Issue #61 Implementation")

    # Simulate data
    age, weight_change, true_effect = simulate_nhefs_rdd_data(n=800, random_seed=42)

    # Run analysis
    result = run_rdd_analysis(age, weight_change, cutoff=50.0)

    # Check KPI from GitHub issue
    print("\n4. KPI VALIDATION (Issue #61)")
    print(f"   True simulated effect: {true_effect:.3f} kg")
    print(f"   RDD estimated effect: {result.ate:.3f} kg")

    error_pct = abs(result.ate - true_effect) / abs(true_effect) * 100
    print(f"   Estimation error: {error_pct:.1f}%")

    if error_pct <= 10:  # Within 10% as specified in issue
        print("   ✅ KPI MET: RDD estimate within ±10% of true effect")
    else:
        print("   ❌ KPI NOT MET: RDD estimate exceeds ±10% tolerance")

    # Visual check
    print("\n5. VISUALIZATION")
    if abs(result.ate) > 2 * (result.ate_se or 1):  # Rough significance check
        print("   ✅ Effect should be visually apparent in discontinuity plot")
    else:
        print("   ⚠ Effect may be difficult to see visually")

    # Create visualization
    try:
        _ = create_rdd_visualization(age, weight_change, cutoff=50.0)
        print("   📊 RDD plot created successfully")
    except Exception as e:
        print(f"   ❌ Visualization failed: {str(e)}")

    print(f"\n{'=' * 60}")
    print("RDD Analysis Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
