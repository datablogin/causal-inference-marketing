"""E-value calculation for sensitivity analysis following VanderWeele & Ding.

This module provides the e_value function that quantifies the minimum strength
of association an unmeasured confounder would need to explain away observed
causal effects.
"""

from __future__ import annotations

from typing import Any, Optional

from ..diagnostics.sensitivity import evalue_calculation


def e_value(
    observed_estimate: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    rare_outcome: bool = False,
    effect_type: str = "risk_ratio",
) -> dict[str, Any]:
    """Calculate E-value for sensitivity analysis following VanderWeele & Ding.

    The E-value quantifies the minimum strength of association that an unmeasured
    confounder would need to have with both treatment and outcome to explain away
    the observed association. This provides a straightforward way to assess
    robustness to unmeasured confounding.

    Args:
        observed_estimate: Observed treatment effect (risk ratio, odds ratio, etc.)
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        rare_outcome: Whether outcome is rare (<10% prevalence)
        effect_type: Type of effect measure ('risk_ratio', 'odds_ratio', 'hazard_ratio')

    Returns:
        Dictionary containing:
            - evalue_point: E-value for point estimate
            - evalue_ci: E-value for confidence interval (more conservative)
            - risk_ratio_used: Risk ratio used in calculation
            - interpretation: Qualitative interpretation
            - threshold_interpretation: Assessment against common thresholds

    Raises:
        ValueError: If effect estimate is non-positive or CI bounds invalid

    Example:
        >>> from causal_inference.sensitivity import e_value
        >>>
        >>> # Risk ratio of 2.5 with 95% CI [1.8, 3.5]
        >>> results = e_value(2.5, ci_lower=1.8, ci_upper=3.5)
        >>> print(f"E-value: {results['evalue_point']:.2f}")
        >>> print(f"Interpretation: {results['interpretation']}")
        >>>
        >>> # For protective effects (RR < 1)
        >>> results = e_value(0.6, ci_lower=0.4, ci_upper=0.9)
        >>> print(f"E-value: {results['evalue_point']:.2f}")

    Notes:
        E-values above 2.0 are generally considered evidence of robustness,
        while values above 1.25 suggest moderate robustness. The E-value for
        the confidence interval bound is often more informative as it
        represents the threshold where the effect could become non-significant.
    """
    return evalue_calculation(
        observed_estimate=observed_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        rare_outcome=rare_outcome,
        effect_type=effect_type,
    )
