"""Sensitivity analysis tools for causal inference.

This module implements methods to assess the robustness of causal effect
estimates to violations of the unconfoundedness assumption.
"""

from __future__ import annotations

import warnings
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
    ci_upper: float | None = None,
    rare_outcome: bool = False,
    effect_type: str = "risk_ratio",
) -> dict[str, Any]:
    """Calculate E-value for sensitivity analysis following VanderWeele & Ding methods.

    The E-value quantifies the minimum strength of association that an unmeasured
    confounder would need to have with both treatment and outcome to explain away
    the observed association.

    Args:
        observed_estimate: Observed treatment effect (risk ratio, odds ratio, etc.)
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        rare_outcome: Whether outcome is rare (affects approximation)
        effect_type: Type of effect measure ('risk_ratio', 'odds_ratio', 'hazard_ratio')

    Returns:
        Dictionary with E-value results

    Raises:
        ValueError: If effect estimate is non-positive or invalid
    """
    # Validate inputs
    if observed_estimate <= 0:
        raise ValueError("Effect estimate must be positive for E-value calculation")

    if ci_lower is not None and ci_upper is not None:
        if ci_lower > ci_upper:
            raise ValueError("Lower CI bound cannot exceed upper CI bound")

    # Convert odds ratio to risk ratio approximation if needed
    if effect_type == "odds_ratio" and rare_outcome:
        # For rare outcomes, OR ≈ RR
        risk_ratio = observed_estimate
    elif effect_type == "odds_ratio" and not rare_outcome:
        # Convert OR to RR using approximation
        # This is simplified - in practice would need baseline risk
        risk_ratio = observed_estimate  # Conservative approach
    else:
        risk_ratio = observed_estimate

    # Calculate E-value for point estimate
    if risk_ratio >= 1:
        evalue_point = risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1))
    else:
        # For protective effects (RR < 1)
        rr_inv = 1 / risk_ratio
        evalue_point = rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

    # Calculate E-value for confidence interval bounds
    evalue_ci_lower = None
    evalue_ci_upper = None

    if ci_lower is not None:
        if ci_lower > 0:  # Valid lower bound
            if ci_lower >= 1:
                evalue_ci_lower = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
            elif ci_lower < 1:
                # For protective CI bound
                if abs(ci_lower - 1) < abs(risk_ratio - 1):
                    evalue_ci_lower = 1.0  # No unmeasured confounding needed
                else:
                    ci_inv = 1 / ci_lower
                    evalue_ci_lower = ci_inv + np.sqrt(ci_inv * (ci_inv - 1))
            else:
                evalue_ci_lower = 1.0

    if ci_upper is not None and ci_upper > 0:
        if ci_upper >= 1:
            evalue_ci_upper = ci_upper + np.sqrt(ci_upper * (ci_upper - 1))
        else:
            # Upper bound is protective
            ci_inv = 1 / ci_upper
            evalue_ci_upper = ci_inv + np.sqrt(ci_inv * (ci_inv - 1))

    # For practical interpretation, use the more conservative bound
    evalue_ci = None
    if evalue_ci_lower is not None and evalue_ci_upper is not None:
        evalue_ci = min(evalue_ci_lower, evalue_ci_upper)
    elif evalue_ci_lower is not None:
        evalue_ci = evalue_ci_lower
    elif evalue_ci_upper is not None:
        evalue_ci = evalue_ci_upper

    # Add protective effect information to interpretation
    base_interpretation = _interpret_evalue(evalue_point)
    if risk_ratio < 1.0:
        interpretation = f"{base_interpretation} (protective effect below null)"
    else:
        interpretation = base_interpretation

    return {
        "evalue_point": float(evalue_point),
        "evalue_ci": float(evalue_ci) if evalue_ci is not None else None,
        "evalue_ci_lower": float(evalue_ci_lower)
        if evalue_ci_lower is not None
        else None,
        "evalue_ci_upper": float(evalue_ci_upper)
        if evalue_ci_upper is not None
        else None,
        "risk_ratio_used": float(risk_ratio),
        "interpretation": interpretation,
        "effect_type": effect_type,
        "threshold_interpretation": _get_evalue_threshold_interpretation(evalue_point),
    }


def _interpret_evalue(evalue: float) -> str:
    """Provide interpretation for E-value following VanderWeele & Ding guidelines."""
    if evalue < 1.5:
        return "Low robustness - weak unmeasured confounding could explain association"
    elif evalue < 2.0:
        return "Moderate robustness - moderate unmeasured confounding needed"
    elif evalue < 3.0:
        return "Good robustness - strong unmeasured confounding needed"
    elif evalue < 5.0:
        return "High robustness - very strong unmeasured confounding needed"
    else:
        return "Very high robustness - extreme unmeasured confounding needed"


def _get_evalue_threshold_interpretation(evalue: float) -> str:
    """Get threshold-based interpretation for E-value."""
    if evalue >= 2.0:
        return "Above common threshold (E-value ≥ 2.0) - results considered robust"
    elif evalue >= 1.25:
        return "Above minimal threshold (E-value ≥ 1.25) - moderate robustness"
    else:
        return (
            "Below minimal threshold - results may be fragile to unmeasured confounding"
        )


def rosenbaum_bounds(
    treated_outcomes: NDArray[Any],
    control_outcomes: NDArray[Any],
    gamma_range: tuple[float, float] = (1.0, 3.0),
    gamma_steps: int = 20,
    alpha: float = 0.05,
    method: str = "wilcoxon",
) -> dict[str, Any]:
    """Calculate Rosenbaum bounds for sensitivity analysis.

    Rosenbaum bounds assess how sensitive results are to hidden bias
    in observational studies, particularly for matched data. This implementation
    follows Rosenbaum (2002) methodology.

    Note on approximations:
        This implementation uses normal approximations for the Wilcoxon signed-rank
        test bounds rather than exact permutation distributions. For samples with
        n < 50 non-zero differences, exact methods would provide better precision,
        but the normal approximation is adequate for larger samples and provides
        computational efficiency.

    Args:
        treated_outcomes: Outcomes for treated units
        control_outcomes: Outcomes for control units
        gamma_range: Range of sensitivity parameters (Γ) to test
        gamma_steps: Number of gamma values to test
        alpha: Significance level for critical gamma calculation
        method: Statistical test method ('wilcoxon', 'sign_test')

    Returns:
        Dictionary with Rosenbaum bounds results

    Raises:
        ValueError: If input data is invalid
    """
    treated_outcomes = np.asarray(treated_outcomes)
    control_outcomes = np.asarray(control_outcomes)

    if len(treated_outcomes) != len(control_outcomes):
        raise ValueError(
            "Treated and control outcomes must have same length for paired test"
        )

    if len(treated_outcomes) < 5:
        raise ValueError("Minimum of 5 pairs required for Rosenbaum bounds")

    # Calculate paired differences
    differences = treated_outcomes - control_outcomes

    # Remove zero differences for signed rank test
    non_zero_diffs = differences[differences != 0]
    n_pairs = len(differences)
    n_non_zero = len(non_zero_diffs)

    if n_non_zero < 3:
        raise ValueError("Insufficient non-zero differences for analysis")

    # Original test statistic
    if method == "wilcoxon":
        try:
            original_stat, original_p = stats.wilcoxon(
                differences, alternative="two-sided", zero_method="wilcox"
            )
        except ValueError:
            # Fallback to sign test if Wilcoxon fails
            method = "sign_test"

    if method == "sign_test":
        # Simple sign test
        n_positive = np.sum(differences > 0)
        original_p = 2 * stats.binom.sf(
            max(n_positive, n_pairs - n_positive) - 1, n_pairs, 0.5
        )
        original_stat = n_positive

    # Calculate bounds for different gamma values
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], gamma_steps)
    bounds_results = []

    for gamma in gamma_values:
        if gamma == 1.0:
            p_upper = p_lower = original_p
        else:
            # Calculate bounds using Rosenbaum methodology
            if method == "wilcoxon":
                p_upper, p_lower = _calculate_wilcoxon_bounds(
                    differences, gamma, original_stat
                )
            else:  # sign_test
                p_upper, p_lower = _calculate_sign_test_bounds(
                    n_pairs, n_positive, gamma
                )

        bounds_results.append(
            {
                "gamma": float(gamma),
                "p_value_lower": float(max(0, min(1, p_lower))),
                "p_value_upper": float(max(0, min(1, p_upper))),
            }
        )

    # Find critical gamma (where upper bound crosses significance threshold)
    critical_gamma = None

    for result in bounds_results:
        if result["p_value_upper"] > alpha:
            critical_gamma = result["gamma"]
            break

    # Additional diagnostics
    robustness_assessment = _assess_rosenbaum_robustness(critical_gamma, gamma_range[1])

    return {
        "original_p_value": float(original_p),
        "original_statistic": float(original_stat),
        "method_used": method,
        "n_pairs": n_pairs,
        "n_non_zero_differences": n_non_zero,
        "bounds": bounds_results,
        "critical_gamma": critical_gamma,
        "alpha_level": alpha,
        "robustness_assessment": robustness_assessment,
        "interpretation": f"Results robust up to Γ = {critical_gamma:.2f}"
        if critical_gamma
        else f"Results robust across tested range (up to Γ = {gamma_range[1]:.2f})",
    }


def _calculate_wilcoxon_bounds(
    differences: NDArray[Any], gamma: float, original_stat: float
) -> tuple[float, float]:
    """Calculate Wilcoxon signed rank test bounds for given gamma."""
    n = len(differences[differences != 0])

    if n < 3:
        return 1.0, 0.0

    # Simplified approximation using normal distribution
    # Full implementation would use exact permutation distribution
    mean_null = n * (n + 1) / 4
    var_null = n * (n + 1) * (2 * n + 1) / 24

    # Bias adjustment for gamma
    bias_factor = np.log(gamma) * np.sqrt(var_null)

    # Upper and lower bounds
    z_upper = (original_stat - mean_null + bias_factor) / np.sqrt(var_null)
    z_lower = (original_stat - mean_null - bias_factor) / np.sqrt(var_null)

    p_upper = 2 * (1 - stats.norm.cdf(abs(z_upper)))
    p_lower = 2 * (1 - stats.norm.cdf(abs(z_lower)))

    return p_upper, p_lower


def _calculate_sign_test_bounds(
    n_pairs: int, n_positive: int, gamma: float
) -> tuple[float, float]:
    """Calculate sign test bounds for given gamma."""
    # Under bias, probability of positive difference changes
    p_biased_upper = gamma / (1 + gamma)
    p_biased_lower = 1 / (1 + gamma)

    # Calculate p-values under biased scenarios
    p_upper = 2 * stats.binom.sf(
        max(n_positive, n_pairs - n_positive) - 1, n_pairs, p_biased_lower
    )
    p_lower = 2 * stats.binom.sf(
        max(n_positive, n_pairs - n_positive) - 1, n_pairs, p_biased_upper
    )

    return p_upper, p_lower


def _assess_rosenbaum_robustness(critical_gamma: float | None, max_gamma: float) -> str:
    """Assess robustness based on critical gamma value."""
    if critical_gamma is None:
        return f"Highly robust - significant across all tested values (up to Γ = {max_gamma:.2f})"
    elif critical_gamma >= 2.0:
        return f"Robust - significant up to moderate hidden bias (Γ = {critical_gamma:.2f})"
    elif critical_gamma >= 1.5:
        return f"Moderately robust - significant up to small hidden bias (Γ = {critical_gamma:.2f})"
    else:
        return f"Low robustness - sensitive to small hidden bias (Γ = {critical_gamma:.2f})"


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


class LinearSensitivityModel:
    """Linear sensitivity model for continuous outcomes.

    This implements the selection bias function approach for assessing
    sensitivity to unmeasured confounding in linear models.
    """

    def __init__(self, treatment_effect_range: tuple[float, float] = (-2.0, 2.0)):
        """Initialize linear sensitivity model.

        Args:
            treatment_effect_range: Range of treatment effects to consider
        """
        self.treatment_effect_range = treatment_effect_range

    def analyze_sensitivity(
        self,
        observed_ate: float,
        confounder_treatment_assoc: float,
        confounder_outcome_assoc: float,
    ) -> dict[str, Any]:
        """Analyze sensitivity using linear model.

        Args:
            observed_ate: Observed average treatment effect
            confounder_treatment_assoc: Association between confounder and treatment
            confounder_outcome_assoc: Association between confounder and outcome

        Returns:
            Dictionary with sensitivity analysis results
        """
        # Calculate bias due to unmeasured confounding
        bias = confounder_treatment_assoc * confounder_outcome_assoc

        # Adjusted treatment effect
        adjusted_ate = observed_ate - bias

        # Calculate percent change
        percent_change = abs(bias / observed_ate) * 100 if observed_ate != 0 else np.inf

        # Determine if effect sign changes
        sign_change = (observed_ate > 0) != (adjusted_ate > 0)

        return {
            "observed_ate": observed_ate,
            "confounder_bias": bias,
            "adjusted_ate": adjusted_ate,
            "percent_change": percent_change,
            "sign_change": sign_change,
            "nullified": abs(adjusted_ate) < 0.1 * abs(observed_ate),
        }

    def sensitivity_surface(
        self,
        observed_ate: float,
        confounder_treatment_range: tuple[float, float] = (-1.0, 1.0),
        confounder_outcome_range: tuple[float, float] = (-1.0, 1.0),
        n_points: int = 20,
    ) -> dict[str, Any]:
        """Generate sensitivity analysis surface.

        Args:
            observed_ate: Observed average treatment effect
            confounder_treatment_range: Range for treatment-confounder association
            confounder_outcome_range: Range for outcome-confounder association
            n_points: Number of points along each dimension

        Returns:
            Dictionary with surface analysis results
        """
        treatment_assocs = np.linspace(
            confounder_treatment_range[0], confounder_treatment_range[1], n_points
        )
        outcome_assocs = np.linspace(
            confounder_outcome_range[0], confounder_outcome_range[1], n_points
        )

        # Create meshgrid
        treatment_mesh, outcome_mesh = np.meshgrid(treatment_assocs, outcome_assocs)

        # Calculate adjusted effects across the surface
        bias_surface = treatment_mesh * outcome_mesh
        adjusted_ate_surface = observed_ate - bias_surface

        # Find nullification boundary
        nullification_boundary = np.abs(adjusted_ate_surface) < 0.1 * abs(observed_ate)

        return {
            "treatment_associations": treatment_assocs,
            "outcome_associations": outcome_assocs,
            "bias_surface": bias_surface,
            "adjusted_ate_surface": adjusted_ate_surface,
            "nullification_boundary": nullification_boundary,
            "observed_ate": observed_ate,
        }


class TippingPointAnalysis:
    """Tipping point analysis to find threshold values where conclusions change."""

    def __init__(self, significance_level: float = 0.05):
        """Initialize tipping point analysis.

        Args:
            significance_level: Statistical significance threshold
        """
        self.alpha = significance_level

    def find_tipping_point_evalue(
        self,
        observed_estimate: float,
        ci_lower: float,
        ci_upper: float,
        search_precision: float = 0.01,
    ) -> dict[str, Any]:
        """Find E-value tipping point where conclusions change.

        Args:
            observed_estimate: Observed effect estimate
            ci_lower: Lower confidence interval
            ci_upper: Upper confidence interval
            search_precision: Precision for binary search

        Returns:
            Dictionary with tipping point results
        """
        # Binary search for tipping point
        low_gamma = 1.0
        high_gamma = 10.0

        # Find point where CI includes null
        while high_gamma - low_gamma > search_precision:
            mid_gamma = (low_gamma + high_gamma) / 2

            # Simulate effect of confounding strength mid_gamma
            # This is simplified - full implementation would depend on specific model
            bias_factor = np.log(mid_gamma)
            adjusted_lower = ci_lower - bias_factor * (observed_estimate - ci_lower)
            adjusted_upper = ci_upper - bias_factor * (ci_upper - observed_estimate)

            # Check if CI includes null
            if adjusted_lower <= 1.0 <= adjusted_upper:  # Assumes risk ratio scale
                high_gamma = mid_gamma
            else:
                low_gamma = mid_gamma

        tipping_gamma = high_gamma

        # Convert to E-value
        tipping_evalue = tipping_gamma + np.sqrt(tipping_gamma * (tipping_gamma - 1))

        return {
            "tipping_gamma": tipping_gamma,
            "tipping_evalue": tipping_evalue,
            "interpretation": f"Conclusions change with confounding strength Γ = {tipping_gamma:.3f}",
            "robustness_assessment": "High"
            if tipping_gamma > 2.0
            else "Moderate"
            if tipping_gamma > 1.5
            else "Low",
        }

    def find_tipping_point_linear(
        self,
        observed_ate: float,
        se_ate: float,
        confounder_strength_range: tuple[float, float] = (0.1, 3.0),
        search_precision: float = 0.01,
    ) -> dict[str, Any]:
        """Find tipping point for linear model where effect becomes non-significant.

        Args:
            observed_ate: Observed average treatment effect
            se_ate: Standard error of ATE
            confounder_strength_range: Range to search for tipping point
            search_precision: Precision for search

        Returns:
            Dictionary with tipping point results
        """
        # Critical value for significance
        critical_value = stats.norm.ppf(1 - self.alpha / 2) * se_ate

        # Binary search for strength that makes effect non-significant
        low_strength = confounder_strength_range[0]
        high_strength = confounder_strength_range[1]

        while high_strength - low_strength > search_precision:
            mid_strength = (low_strength + high_strength) / 2

            # Assume bias proportional to confounding strength
            bias = mid_strength * observed_ate * 0.5  # Simplified relationship
            adjusted_ate = observed_ate - bias

            # Check if still significant
            if abs(adjusted_ate) > critical_value:
                low_strength = mid_strength
            else:
                high_strength = mid_strength

        tipping_strength = high_strength

        return {
            "tipping_strength": tipping_strength,
            "critical_bias": tipping_strength * observed_ate * 0.5,
            "interpretation": f"Effect becomes non-significant with confounding strength {tipping_strength:.3f}",
            "robustness_assessment": "High"
            if tipping_strength > 0.8
            else "Moderate"
            if tipping_strength > 0.5
            else "Low",
        }


class SensitivityAnalysis:
    """Comprehensive sensitivity analysis for causal inference."""

    def __init__(
        self,
        alpha: float = 0.05,
        evalue_threshold: float = 2.0,
        robustness_threshold: float = 0.5,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        """Initialize sensitivity analysis.

        Args:
            alpha: Significance level
            evalue_threshold: Threshold for considering E-value adequate
            robustness_threshold: Threshold for robustness assessment
            verbose: Whether to print verbose output
            random_state: Random seed for reproducible results
        """
        self.alpha = alpha
        self.evalue_threshold = evalue_threshold
        self.robustness_threshold = robustness_threshold
        self.verbose = verbose
        self.random_state = random_state
        self.linear_model = LinearSensitivityModel()
        self.tipping_analysis = TippingPointAnalysis(alpha)

    def comprehensive_sensitivity_analysis(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        causal_effect: CausalEffect,
        covariates: CovariateData | None = None,
        include_tipping_point: bool = True,
        include_linear_sensitivity: bool = True,
    ) -> SensitivityResults:
        """Perform comprehensive sensitivity analysis.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            causal_effect: Estimated causal effect
            covariates: Optional covariate data
            include_tipping_point: Whether to include tipping point analysis
            include_linear_sensitivity: Whether to include linear sensitivity model

        Returns:
            SensitivityResults with comprehensive assessment
        """
        # Convert ATE to risk ratio for E-value calculation with improved method
        risk_ratio = self._convert_ate_to_risk_ratio(causal_effect, outcome)

        # Calculate E-value with enhanced method
        evalue_results = evalue_calculation(
            risk_ratio,
            ci_lower=self._convert_ci_to_risk_ratio(
                causal_effect.ate_ci_lower, causal_effect, outcome
            )
            if causal_effect.ate_ci_lower
            else None,
            ci_upper=self._convert_ci_to_risk_ratio(
                causal_effect.ate_ci_upper, causal_effect, outcome
            )
            if causal_effect.ate_ci_upper
            else None,
            rare_outcome=self._is_rare_outcome(outcome),
            effect_type="risk_ratio",
        )

        # Rosenbaum bounds (enhanced implementation)
        rosenbaum_results = None
        if (
            treatment.treatment_type == "binary"
            and len(np.unique(treatment.values)) == 2
        ):
            try:
                rosenbaum_results = self._calculate_enhanced_rosenbaum_bounds(
                    treatment, outcome
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not calculate Rosenbaum bounds: {e}")

        # Linear sensitivity model analysis
        linear_sensitivity_results = None
        if include_linear_sensitivity and outcome.outcome_type == "continuous":
            linear_sensitivity_results = self._perform_linear_sensitivity_analysis(
                causal_effect
            )

        # Tipping point analysis
        tipping_point_results = None
        if include_tipping_point:
            tipping_point_results = self._perform_tipping_point_analysis(
                causal_effect, risk_ratio
            )

        # Enhanced unmeasured confounding analysis
        confounding_analysis = self._enhanced_unmeasured_confounding_analysis(
            treatment, outcome, causal_effect
        )

        # Generate comprehensive recommendations
        recommendations = self._generate_comprehensive_recommendations(
            evalue_results,
            rosenbaum_results,
            confounding_analysis,
            linear_sensitivity_results,
            tipping_point_results,
        )

        return SensitivityResults(
            evalue=evalue_results["evalue_point"],
            evalue_ci_lower=evalue_results["evalue_ci"],
            rosenbaum_bounds=rosenbaum_results,
            confounding_strength_needed=confounding_analysis.get(
                "nullifying_strength", 1.0
            ),
            robustness_assessment=self._overall_robustness_assessment(
                evalue_results, rosenbaum_results, confounding_analysis
            ),
            critical_gamma=rosenbaum_results.get("critical_gamma")
            if rosenbaum_results
            else None,
            sensitivity_plots_data={
                "confounding_analysis": confounding_analysis,
                "linear_sensitivity": linear_sensitivity_results,
                "tipping_point": tipping_point_results,
                "evalue_details": evalue_results,
            },
            recommendations=recommendations,
        )

    def _convert_ate_to_risk_ratio(
        self, causal_effect: CausalEffect, outcome: OutcomeData
    ) -> float:
        """Convert ATE to risk ratio with improved methodology."""
        if outcome.outcome_type == "binary":
            # For binary outcomes, estimate baseline risk
            original_baseline_risk = float(np.mean(outcome.values))
            baseline_risk = float(
                np.clip(original_baseline_risk, 0.01, 0.99)
            )  # Avoid extremes

            # Warn if extreme baseline risk was clipped
            if abs(original_baseline_risk - baseline_risk) > 1e-6:
                warnings.warn(
                    f"Extreme baseline risk ({original_baseline_risk:.4f}) clipped to "
                    f"{baseline_risk:.4f} for E-value calculation. Results may be less "
                    f"reliable for very rare or very common outcomes.",
                    UserWarning,
                    stacklevel=3,
                )

            # Convert ATE to risk ratio
            original_treated_risk = baseline_risk + causal_effect.ate
            treated_risk = float(np.clip(original_treated_risk, 0.01, 0.99))

            # Warn if treated risk was clipped
            if abs(original_treated_risk - treated_risk) > 1e-6:
                warnings.warn(
                    f"Extreme treated risk ({original_treated_risk:.4f}) clipped to "
                    f"{treated_risk:.4f} for E-value calculation. Results may be less "
                    f"reliable.",
                    UserWarning,
                    stacklevel=3,
                )

            return treated_risk / baseline_risk
        else:
            # For continuous outcomes, use standardized effect size conversion
            outcome_std = (
                float(np.std(outcome.values))
                if hasattr(outcome.values, "__iter__")
                else 1.0
            )
            standardized_effect = abs(causal_effect.ate) / outcome_std

            # Convert to approximate risk ratio (Cohen's conventions)
            return 1 + standardized_effect

    def _convert_ci_to_risk_ratio(
        self, ci_bound: float, causal_effect: CausalEffect, outcome: OutcomeData
    ) -> float:
        """Convert confidence interval bound to risk ratio scale."""
        if outcome.outcome_type == "binary":
            original_baseline_risk = float(np.mean(outcome.values))
            baseline_risk = float(np.clip(original_baseline_risk, 0.01, 0.99))

            # Warn if extreme baseline risk was clipped
            if abs(original_baseline_risk - baseline_risk) > 1e-6:
                warnings.warn(
                    f"Extreme baseline risk ({original_baseline_risk:.4f}) clipped to "
                    f"{baseline_risk:.4f} for CI E-value calculation. Results may be less "
                    f"reliable for very rare or very common outcomes.",
                    UserWarning,
                    stacklevel=3,
                )

            original_treated_risk = baseline_risk + ci_bound
            treated_risk = float(np.clip(original_treated_risk, 0.01, 0.99))

            # Warn if treated risk was clipped
            if abs(original_treated_risk - treated_risk) > 1e-6:
                warnings.warn(
                    f"Extreme CI bound risk ({original_treated_risk:.4f}) clipped to "
                    f"{treated_risk:.4f} for E-value calculation. Results may be less "
                    f"reliable.",
                    UserWarning,
                    stacklevel=3,
                )

            return treated_risk / baseline_risk
        else:
            outcome_std = (
                float(np.std(outcome.values))
                if hasattr(outcome.values, "__iter__")
                else 1.0
            )
            standardized_effect = abs(ci_bound) / outcome_std
            return 1 + standardized_effect

    def _is_rare_outcome(self, outcome: OutcomeData) -> bool:
        """Determine if outcome is rare (< 10% prevalence)."""
        if outcome.outcome_type == "binary":
            prevalence = float(np.mean(outcome.values))
            return prevalence < 0.1 or prevalence > 0.9
        return False

    def _calculate_enhanced_rosenbaum_bounds(
        self, treatment: TreatmentData, outcome: OutcomeData
    ) -> dict[str, Any]:
        """Calculate Rosenbaum bounds with enhanced methodology."""
        treated_outcomes = np.asarray(outcome.values)[np.asarray(treatment.values) == 1]
        control_outcomes = np.asarray(outcome.values)[np.asarray(treatment.values) == 0]

        # Use random pairing if no natural pairing exists
        min_len = min(len(treated_outcomes), len(control_outcomes))

        if min_len < 5:
            raise ValueError("Insufficient data for Rosenbaum bounds analysis")

        # Random matching for unpaired data
        np.random.seed(self.random_state if hasattr(self, "random_state") else 42)
        treated_sample = np.random.choice(treated_outcomes, min_len, replace=False)
        control_sample = np.random.choice(control_outcomes, min_len, replace=False)

        return rosenbaum_bounds(
            treated_sample,
            control_sample,
            gamma_range=(1.0, 4.0),
            gamma_steps=30,
            alpha=self.alpha,
        )

    def _perform_linear_sensitivity_analysis(
        self, causal_effect: CausalEffect
    ) -> dict[str, Any]:
        """Perform linear sensitivity model analysis."""
        # Create sensitivity surface
        surface_results = self.linear_model.sensitivity_surface(
            causal_effect.ate,
            confounder_treatment_range=(-1.0, 1.0),
            confounder_outcome_range=(-1.0, 1.0),
            n_points=25,
        )

        # Analyze specific scenarios
        scenarios: list[dict[str, Any]] = [
            {"name": "Weak confounding", "treat_assoc": 0.2, "outcome_assoc": 0.2},
            {"name": "Moderate confounding", "treat_assoc": 0.5, "outcome_assoc": 0.5},
            {"name": "Strong confounding", "treat_assoc": 0.8, "outcome_assoc": 0.8},
        ]

        scenario_results = []
        for scenario in scenarios:
            treat_assoc = scenario["treat_assoc"]
            outcome_assoc = scenario["outcome_assoc"]
            assert isinstance(treat_assoc, (int, float))  # noqa: UP038
            assert isinstance(outcome_assoc, (int, float))  # noqa: UP038

            result = self.linear_model.analyze_sensitivity(
                causal_effect.ate,
                float(treat_assoc),
                float(outcome_assoc),
            )
            result["scenario_name"] = scenario["name"]
            scenario_results.append(result)

        return {
            "surface_analysis": surface_results,
            "scenario_analysis": scenario_results,
        }

    def _perform_tipping_point_analysis(
        self, causal_effect: CausalEffect, risk_ratio: float
    ) -> dict[str, Any]:
        """Perform tipping point analysis."""
        results = {}

        # E-value based tipping point
        if (
            causal_effect.ate_ci_lower is not None
            and causal_effect.ate_ci_upper is not None
        ):
            ci_lower_rr = self._convert_ci_to_risk_ratio(
                causal_effect.ate_ci_lower,
                causal_effect,
                OutcomeData(values=np.array([0, 1])),
            )
            ci_upper_rr = self._convert_ci_to_risk_ratio(
                causal_effect.ate_ci_upper,
                causal_effect,
                OutcomeData(values=np.array([0, 1])),
            )

            evalue_tipping = self.tipping_analysis.find_tipping_point_evalue(
                risk_ratio, ci_lower_rr, ci_upper_rr
            )
            results["evalue_tipping"] = evalue_tipping

        # Linear model tipping point
        if causal_effect.ate_se is not None:
            linear_tipping = self.tipping_analysis.find_tipping_point_linear(
                causal_effect.ate, causal_effect.ate_se
            )
            results["linear_tipping"] = linear_tipping

        return results

    def _enhanced_unmeasured_confounding_analysis(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        causal_effect: CausalEffect,
    ) -> dict[str, Any]:
        """Enhanced unmeasured confounding analysis."""
        # Use existing function but enhance results
        base_results = unmeasured_confounding_analysis(
            treatment,
            outcome,
            causal_effect,
            confounder_strength_range=(0.05, 1.0),  # More granular range
        )

        # Add enhanced interpretations
        base_results["detailed_interpretation"] = self._interpret_confounding_strength(
            base_results.get("nullifying_strength")
        )

        return base_results

    def _interpret_confounding_strength(self, nullifying_strength: float | None) -> str:
        """Provide detailed interpretation of confounding strength."""
        if nullifying_strength is None:
            return "Effect is highly robust - remains significant even with very strong unmeasured confounding"
        elif nullifying_strength > 0.8:
            return f"Effect is robust - would require very strong unmeasured confounding (strength > {nullifying_strength:.2f}) to nullify"
        elif nullifying_strength > 0.5:
            return f"Effect has moderate robustness - strong confounding (strength > {nullifying_strength:.2f}) could nullify the effect"
        elif nullifying_strength > 0.2:
            return f"Effect has limited robustness - moderate confounding (strength > {nullifying_strength:.2f}) could nullify the effect"
        else:
            return f"Effect is fragile - weak confounding (strength > {nullifying_strength:.2f}) could nullify the effect"

    def _overall_robustness_assessment(
        self,
        evalue_results: dict[str, Any],
        rosenbaum_results: dict[str, Any] | None,
        confounding_analysis: dict[str, Any],
    ) -> str:
        """Provide overall robustness assessment."""
        robustness_scores = []

        # E-value score
        evalue = evalue_results["evalue_point"]
        if evalue >= 3.0:
            robustness_scores.append(4)
        elif evalue >= 2.0:
            robustness_scores.append(3)
        elif evalue >= 1.5:
            robustness_scores.append(2)
        else:
            robustness_scores.append(1)

        # Rosenbaum bounds score
        if rosenbaum_results:
            critical_gamma = rosenbaum_results.get("critical_gamma")
            if critical_gamma is None or critical_gamma >= 3.0:
                robustness_scores.append(4)
            elif critical_gamma >= 2.0:
                robustness_scores.append(3)
            elif critical_gamma >= 1.5:
                robustness_scores.append(2)
            else:
                robustness_scores.append(1)

        # Confounding strength score
        nullifying_strength = confounding_analysis.get("nullifying_strength")
        if nullifying_strength is None or nullifying_strength > 0.8:
            robustness_scores.append(4)
        elif nullifying_strength > 0.5:
            robustness_scores.append(3)
        elif nullifying_strength > 0.2:
            robustness_scores.append(2)
        else:
            robustness_scores.append(1)

        # Overall assessment
        avg_score = np.mean(robustness_scores)

        if avg_score >= 3.5:
            return "High robustness - results appear robust across multiple sensitivity analyses"
        elif avg_score >= 2.5:
            return "Moderate robustness - results show reasonable robustness but warrant caution"
        elif avg_score >= 1.5:
            return (
                "Limited robustness - results are sensitive to unmeasured confounding"
            )
        else:
            return "Low robustness - results are highly sensitive to unmeasured confounding"

    def _generate_comprehensive_recommendations(
        self,
        evalue_results: dict[str, Any],
        rosenbaum_results: dict[str, Any] | None,
        confounding_analysis: dict[str, Any],
        linear_sensitivity_results: dict[str, Any] | None,
        tipping_point_results: dict[str, Any] | None,
    ) -> list[str]:
        """Generate comprehensive recommendations based on all sensitivity analyses."""
        recommendations = []
        warning_count = 0

        # E-value recommendations with enhanced thresholds
        evalue = evalue_results["evalue_point"]
        if evalue >= 3.0:
            recommendations.append(
                f"✅ Strong E-value ({evalue:.2f}) - results highly robust to unmeasured confounding"
            )
        elif evalue >= self.evalue_threshold:
            recommendations.append(
                f"✅ Adequate E-value ({evalue:.2f}) - results reasonably robust"
            )
        elif evalue >= 1.25:
            recommendations.append(
                f"⚠️ Moderate E-value ({evalue:.2f}) - some sensitivity to unmeasured confounding"
            )
            warning_count += 1
        else:
            recommendations.append(
                f"❌ Low E-value ({evalue:.2f}) - high sensitivity to unmeasured confounding"
            )
            warning_count += 1

        # Rosenbaum bounds recommendations with enhanced interpretation
        if rosenbaum_results:
            critical_gamma = rosenbaum_results.get("critical_gamma")
            if critical_gamma is None:
                recommendations.append(
                    "✅ Rosenbaum analysis: Robust across all tested bias levels"
                )
            elif critical_gamma >= 2.5:
                recommendations.append(
                    f"✅ Strong Rosenbaum robustness (Γ_crit = {critical_gamma:.2f})"
                )
            elif critical_gamma >= 2.0:
                recommendations.append(
                    f"✅ Good Rosenbaum robustness (Γ_crit = {critical_gamma:.2f})"
                )
            elif critical_gamma >= 1.5:
                recommendations.append(
                    f"⚠️ Moderate Rosenbaum robustness (Γ_crit = {critical_gamma:.2f})"
                )
                warning_count += 1
            else:
                recommendations.append(
                    f"❌ Limited Rosenbaum robustness (Γ_crit = {critical_gamma:.2f})"
                )
                warning_count += 1
        else:
            recommendations.append(
                "ℹ️ Rosenbaum bounds not applicable (requires paired/matched data)"
            )

        # Linear sensitivity model recommendations
        if linear_sensitivity_results:
            scenario_analysis = linear_sensitivity_results.get("scenario_analysis", [])
            moderate_scenario = next(
                (
                    s
                    for s in scenario_analysis
                    if s["scenario_name"] == "Moderate confounding"
                ),
                None,
            )
            if moderate_scenario:
                if not moderate_scenario["nullified"]:
                    recommendations.append(
                        "✅ Linear sensitivity: Effect robust to moderate unmeasured confounding"
                    )
                else:
                    recommendations.append(
                        "⚠️ Linear sensitivity: Effect vulnerable to moderate unmeasured confounding"
                    )
                    warning_count += 1

        # Tipping point recommendations
        if tipping_point_results:
            if "evalue_tipping" in tipping_point_results:
                tipping = tipping_point_results["evalue_tipping"]
                assessment = tipping["robustness_assessment"]
                if assessment == "High":
                    recommendations.append(
                        f"✅ Tipping point analysis: High robustness (Γ_tip = {tipping['tipping_gamma']:.3f})"
                    )
                elif assessment == "Moderate":
                    recommendations.append(
                        f"⚠️ Tipping point analysis: Moderate robustness (Γ_tip = {tipping['tipping_gamma']:.3f})"
                    )
                    warning_count += 1
                else:
                    recommendations.append(
                        f"❌ Tipping point analysis: Low robustness (Γ_tip = {tipping['tipping_gamma']:.3f})"
                    )
                    warning_count += 1

        # Enhanced unmeasured confounding recommendations
        nullifying_strength = confounding_analysis.get("nullifying_strength")
        if nullifying_strength:
            detailed_interp = confounding_analysis.get("detailed_interpretation", "")
            if "robust" in detailed_interp.lower():
                recommendations.append(f"✅ {detailed_interp}")
            elif "moderate" in detailed_interp.lower():
                recommendations.append(f"⚠️ {detailed_interp}")
                warning_count += 1
            else:
                recommendations.append(f"❌ {detailed_interp}")
                warning_count += 1

        # Method-specific recommendations based on warning count
        if warning_count == 0:
            recommendations.extend(
                [
                    "🎯 Results appear robust across all sensitivity analyses",
                    "Consider presenting sensitivity analysis as evidence of robustness",
                    "Standard causal inference assumptions likely adequate",
                ]
            )
        elif warning_count <= 2:
            recommendations.extend(
                [
                    "📊 Results show mixed robustness - interpret with appropriate caution",
                    "Consider collecting additional covariates if possible",
                    "Triangulation with other identification strategies recommended",
                    "Sensitivity analysis should be reported alongside main results",
                ]
            )
        else:
            recommendations.extend(
                [
                    "⚠️ Results show limited robustness across multiple analyses",
                    "Strong caution warranted in causal interpretation",
                    "Priority recommendations:",
                    "  • Instrumental variable analysis if instruments available",
                    "  • Negative control outcomes to test assumptions",
                    "  • Additional covariate collection and adjustment",
                    "  • Alternative identification strategies (RDD, diff-in-diff, etc.)",
                    "  • Qualitative assessment of potential unmeasured confounders",
                    "Consider whether observational approach is appropriate for this question",
                ]
            )

        # Technical recommendations
        recommendations.append("")
        recommendations.append("📋 Technical recommendations:")

        if evalue < 1.5:
            recommendations.append("  • Report E-value limitations in manuscript")

        if (
            rosenbaum_results
            and rosenbaum_results.get("critical_gamma") is not None
            and rosenbaum_results.get("critical_gamma", 0) < 1.5
        ):
            recommendations.append("  • Consider propensity score matching refinements")

        recommendations.append(
            "  • Document all sensitivity analyses in supplementary materials"
        )
        recommendations.append("  • Consider domain expert input on likely confounders")

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
