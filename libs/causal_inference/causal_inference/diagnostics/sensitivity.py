"""Sensitivity analysis tools for causal inference.

This module implements methods to assess the robustness of causal effect
estimates to violations of the unconfoundedness assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ..core.base import CausalEffect, CovariateData, OutcomeData, TreatmentData


@dataclass
class SensitivityResults:
    """Results from sensitivity analysis."""

    evalue: float
    evalue_ci_lower: float | None
    rosenbaum_bounds: dict[str, Any] | None
    confounding_strength_needed: float
    robustness_assessment: str
    critical_gamma: float | None
    sensitivity_plots_data: dict[str, Any] | None
    recommendations: list[str]


def evalue_calculation(
    observed_estimate: float,
    ci_lower: float | None = None,
    rare_outcome: bool = False,
) -> dict[str, float]:
    """Calculate E-value for sensitivity analysis.

    The E-value quantifies the minimum strength of association that an unmeasured
    confounder would need to have with both treatment and outcome to explain away
    the observed association.

    Args:
        observed_estimate: Observed treatment effect (risk ratio, odds ratio, etc.)
        ci_lower: Lower confidence interval bound
        rare_outcome: Whether outcome is rare (affects approximation)

    Returns:
        Dictionary with E-value results
    """
    # Convert to risk ratio if needed
    if observed_estimate <= 0:
        raise ValueError("Effect estimate must be positive for E-value calculation")

    # Calculate E-value for point estimate
    if observed_estimate >= 1:
        evalue_point = observed_estimate + np.sqrt(
            observed_estimate * (observed_estimate - 1)
        )
    else:
        # For protective effects
        rr_inv = 1 / observed_estimate
        evalue_point = rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

    # Calculate E-value for confidence interval
    evalue_ci = None
    if ci_lower is not None:
        if ci_lower > 1:
            evalue_ci = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
        elif ci_lower < 1:
            # Conservative approach: use the bound closest to null
            if abs(ci_lower - 1) < abs(observed_estimate - 1):
                evalue_ci = 1.0  # No unmeasured confounding needed
            else:
                ci_inv = 1 / ci_lower
                evalue_ci = ci_inv + np.sqrt(ci_inv * (ci_inv - 1))
        else:
            evalue_ci = 1.0

    return {
        "evalue_point": float(evalue_point),
        "evalue_ci": float(evalue_ci) if evalue_ci is not None else None,
        "interpretation": _interpret_evalue(evalue_point),
    }


def _interpret_evalue(evalue: float) -> str:
    """Provide interpretation for E-value."""
    if evalue < 1.5:
        return "Low robustness - weak confounding could explain association"
    elif evalue < 2.0:
        return "Moderate robustness - moderate confounding needed"
    elif evalue < 3.0:
        return "Good robustness - strong confounding needed"
    else:
        return "High robustness - very strong confounding needed"


def rosenbaum_bounds(
    treated_outcomes: NDArray[Any],
    control_outcomes: NDArray[Any],
    gamma_range: tuple[float, float] = (1.0, 3.0),
    gamma_steps: int = 20,
) -> dict[str, Any]:
    """Calculate Rosenbaum bounds for sensitivity analysis.

    Rosenbaum bounds assess how sensitive results are to hidden bias
    in observational studies, particularly for matched data.

    Args:
        treated_outcomes: Outcomes for treated units
        control_outcomes: Outcomes for control units
        gamma_range: Range of sensitivity parameters to test
        gamma_steps: Number of gamma values to test

    Returns:
        Dictionary with Rosenbaum bounds results
    """
    treated_outcomes = np.asarray(treated_outcomes)
    control_outcomes = np.asarray(control_outcomes)

    if len(treated_outcomes) != len(control_outcomes):
        raise ValueError(
            "Treated and control outcomes must have same length for paired test"
        )

    # Calculate paired differences
    differences = treated_outcomes - control_outcomes

    # Original Wilcoxon signed rank test
    original_stat, original_p = stats.wilcoxon(differences, alternative="two-sided")

    # Calculate bounds for different gamma values
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], gamma_steps)
    bounds_results = []

    for gamma in gamma_values:
        # Calculate bounds (simplified approximation)
        # In practice, this requires more complex calculations
        n = len(differences)

        # Upper and lower bounds on p-values
        # This is a simplified approximation - full implementation requires
        # complex combinatorial calculations

        if gamma == 1.0:
            p_upper = p_lower = original_p
        else:
            # Approximation based on normal distribution
            # Real implementation would use exact permutation distribution
            z_stat = original_stat / np.sqrt(n * (n + 1) * (2 * n + 1) / 6)

            # Adjust for bias
            bias_factor = np.log(gamma)
            z_upper = z_stat - bias_factor
            z_lower = z_stat + bias_factor

            p_upper = 2 * (1 - stats.norm.cdf(abs(z_upper)))
            p_lower = 2 * (1 - stats.norm.cdf(abs(z_lower)))

        bounds_results.append(
            {
                "gamma": float(gamma),
                "p_value_lower": float(max(0, p_lower)),
                "p_value_upper": float(min(1, p_upper)),
            }
        )

    # Find critical gamma (where upper bound crosses significance threshold)
    critical_gamma = None
    alpha = 0.05

    for result in bounds_results:
        if result["p_value_upper"] > alpha:
            critical_gamma = result["gamma"]
            break

    return {
        "original_p_value": float(original_p),
        "bounds": bounds_results,
        "critical_gamma": critical_gamma,
        "interpretation": f"Results robust up to Γ = {critical_gamma:.2f}"
        if critical_gamma
        else "Results robust across tested range",
    }


def unmeasured_confounding_analysis(
    treatment: TreatmentData,
    outcome: OutcomeData,
    observed_effect: CausalEffect,
    confounder_strength_range: tuple[float, float] = (0.1, 0.9),
) -> dict[str, Any]:
    """Analyze impact of unmeasured confounding on causal estimates.

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        observed_effect: Observed causal effect estimate
        confounder_strength_range: Range of confounder strengths to test

    Returns:
        Dictionary with unmeasured confounding analysis
    """
    # Simulate unmeasured confounder with different strengths
    results = []
    strengths = np.linspace(
        confounder_strength_range[0], confounder_strength_range[1], 10
    )

    for strength in strengths:
        # Calculate how much the effect would change
        # This is a simplified simulation - real analysis would be more complex

        # Effect of confounder on outcome
        outcome_effect = strength

        # Bias introduced by unmeasured confounding
        # Simplified formula: bias ≈ β_UY * (mean(U|T=1) - mean(U|T=0))

        # Assume confounder is higher in treated group
        confounder_diff = strength  # Difference in confounder means between groups
        bias = outcome_effect * confounder_diff

        # Adjusted effect estimate
        adjusted_ate = observed_effect.ate - bias

        results.append(
            {
                "confounder_strength": float(strength),
                "bias_introduced": float(bias),
                "adjusted_ate": float(adjusted_ate),
                "percent_change": float(abs(bias / observed_effect.ate) * 100)
                if observed_effect.ate != 0
                else np.inf,
            }
        )

    # Find strength needed to nullify effect
    nullifying_strength = None
    for result in results:
        if abs(result["adjusted_ate"]) < 0.1 * abs(
            observed_effect.ate
        ):  # 90% reduction
            nullifying_strength = result["confounder_strength"]
            break

    return {
        "sensitivity_analysis": results,
        "nullifying_strength": nullifying_strength,
        "robustness_assessment": _assess_robustness(nullifying_strength),
        "recommendation": _recommend_based_on_sensitivity(nullifying_strength),
    }


def _assess_robustness(nullifying_strength: float | None) -> str:
    """Assess robustness based on confounding strength needed to nullify effect."""
    if nullifying_strength is None:
        return "Highly robust - effect remains significant across tested range"
    elif nullifying_strength > 0.7:
        return "Robust - strong confounding needed to nullify effect"
    elif nullifying_strength > 0.4:
        return "Moderately robust - moderate confounding could nullify effect"
    else:
        return "Not robust - weak confounding could nullify effect"


def _recommend_based_on_sensitivity(nullifying_strength: float | None) -> str:
    """Generate recommendation based on sensitivity analysis."""
    if nullifying_strength is None or nullifying_strength > 0.6:
        return "Results appear robust to unmeasured confounding"
    elif nullifying_strength > 0.3:
        return (
            "Consider collecting additional covariates or using instrumental variables"
        )
    else:
        return "Results highly sensitive - strong additional validation needed"


class SensitivityAnalysis:
    """Comprehensive sensitivity analysis for causal inference."""

    def __init__(
        self,
        alpha: float = 0.05,
        evalue_threshold: float = 2.0,
        robustness_threshold: float = 0.5,
    ):
        """Initialize sensitivity analysis.

        Args:
            alpha: Significance level
            evalue_threshold: Threshold for considering E-value adequate
            robustness_threshold: Threshold for robustness assessment
        """
        self.alpha = alpha
        self.evalue_threshold = evalue_threshold
        self.robustness_threshold = robustness_threshold

    def comprehensive_sensitivity_analysis(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        causal_effect: CausalEffect,
        covariates: CovariateData | None = None,
    ) -> SensitivityResults:
        """Perform comprehensive sensitivity analysis.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            causal_effect: Estimated causal effect
            covariates: Optional covariate data

        Returns:
            SensitivityResults with comprehensive assessment
        """
        # Convert ATE to risk ratio for E-value calculation
        # This is simplified - proper conversion depends on outcome type and baseline risk
        if causal_effect.ate > 0:
            risk_ratio = 1 + abs(causal_effect.ate)  # Approximation
        else:
            risk_ratio = 1 / (1 + abs(causal_effect.ate))

        # Calculate E-value
        evalue_results = evalue_calculation(
            risk_ratio,
            1 + abs(causal_effect.ate_ci_lower) if causal_effect.ate_ci_lower else None,
        )

        # Rosenbaum bounds (if applicable)
        rosenbaum_results = None
        if treatment.treatment_type == "binary":
            try:
                treated_outcomes = np.asarray(outcome.values)[
                    np.asarray(treatment.values) == 1
                ]
                control_outcomes = np.asarray(outcome.values)[
                    np.asarray(treatment.values) == 0
                ]

                # For Rosenbaum bounds, we need paired data
                # This is a simplified version - proper implementation requires matching
                min_len = min(len(treated_outcomes), len(control_outcomes))
                if min_len > 10:
                    rosenbaum_results = rosenbaum_bounds(
                        treated_outcomes[:min_len], control_outcomes[:min_len]
                    )
            except Exception:
                pass  # Skip if not applicable

        # Unmeasured confounding analysis
        confounding_analysis = unmeasured_confounding_analysis(
            treatment, outcome, causal_effect
        )

        # Generate recommendations
        recommendations = self._generate_sensitivity_recommendations(
            evalue_results, rosenbaum_results, confounding_analysis
        )

        return SensitivityResults(
            evalue=evalue_results["evalue_point"],
            evalue_ci_lower=evalue_results["evalue_ci"],
            rosenbaum_bounds=rosenbaum_results,
            confounding_strength_needed=confounding_analysis.get(
                "nullifying_strength", 1.0
            ),
            robustness_assessment=confounding_analysis["robustness_assessment"],
            critical_gamma=rosenbaum_results.get("critical_gamma")
            if rosenbaum_results
            else None,
            sensitivity_plots_data=confounding_analysis,
            recommendations=recommendations,
        )

    def _generate_sensitivity_recommendations(
        self,
        evalue_results: dict[str, Any],
        rosenbaum_results: dict[str, Any] | None,
        confounding_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on sensitivity analysis."""
        recommendations = []

        # E-value recommendations
        if evalue_results["evalue_point"] >= self.evalue_threshold:
            recommendations.append(
                f"✅ E-value ({evalue_results['evalue_point']:.2f}) suggests robust results"
            )
        else:
            recommendations.append(
                f"⚠️ Low E-value ({evalue_results['evalue_point']:.2f}) - results may be sensitive"
            )

        # Rosenbaum bounds recommendations
        if rosenbaum_results and rosenbaum_results.get("critical_gamma"):
            gamma = rosenbaum_results["critical_gamma"]
            if gamma >= 2.0:
                recommendations.append(
                    f"✅ Rosenbaum bounds suggest robustness (Γ_crit = {gamma:.2f})"
                )
            else:
                recommendations.append(
                    f"⚠️ Limited robustness in Rosenbaum bounds (Γ_crit = {gamma:.2f})"
                )

        # Unmeasured confounding recommendations
        recommendations.append(confounding_analysis["recommendation"])

        # General recommendations
        if any("⚠️" in rec for rec in recommendations):
            recommendations.extend(
                [
                    "Consider instrumental variable analysis if available",
                    "Collect additional covariates to reduce confounding",
                    "Use multiple identification strategies for robustness",
                ]
            )

        return recommendations

    def print_sensitivity_summary(self, results: SensitivityResults) -> None:
        """Print summary of sensitivity analysis results."""
        print("=== Sensitivity Analysis Summary ===")
        print()

        print(f"E-value: {results.evalue:.2f}")
        if results.evalue_ci_lower:
            print(f"E-value (CI lower bound): {results.evalue_ci_lower:.2f}")

        if results.rosenbaum_bounds:
            print(
                f"Rosenbaum critical Γ: {results.critical_gamma:.2f}"
                if results.critical_gamma
                else "Robust across tested range"
            )

        print(
            f"Confounding strength needed to nullify: {results.confounding_strength_needed:.2f}"
        )
        print(f"Robustness assessment: {results.robustness_assessment}")
        print()

        print("Recommendations:")
        for i, rec in enumerate(results.recommendations, 1):
            print(f"  {i}. {rec}")


# Convenience functions
def calculate_evalue(
    observed_estimate: float,
    ci_lower: float | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Convenience function to calculate E-value."""
    results = evalue_calculation(observed_estimate, ci_lower)

    if verbose:
        print(f"E-value: {results['evalue_point']:.2f}")
        if results["evalue_ci"]:
            print(f"E-value (CI): {results['evalue_ci']:.2f}")
        print(f"Interpretation: {results['interpretation']}")

    return results


def assess_sensitivity(
    treatment: TreatmentData,
    outcome: OutcomeData,
    causal_effect: CausalEffect,
    verbose: bool = True,
) -> SensitivityResults:
    """Convenience function for comprehensive sensitivity analysis."""
    analyzer = SensitivityAnalysis()
    results = analyzer.comprehensive_sensitivity_analysis(
        treatment, outcome, causal_effect
    )

    if verbose:
        analyzer.print_sensitivity_summary(results)

    return results
