"""Rosenbaum bounds for sensitivity analysis in matched/weighted designs.

This module provides the rosenbaum_bounds function that implements sensitivity
analysis for matched data following Rosenbaum (2002) methodology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..diagnostics.sensitivity import rosenbaum_bounds as _rosenbaum_bounds_impl


def rosenbaum_bounds(
    treated_outcomes: NDArray[Any] | list[float],
    control_outcomes: NDArray[Any] | list[float],
    gamma_range: tuple[float, float] = (1.0, 3.0),
    gamma_steps: int = 20,
    alpha: float = 0.05,
    method: str = "wilcoxon",
) -> dict[str, Any]:
    """Calculate Rosenbaum bounds for sensitivity analysis in matched designs.

    Rosenbaum bounds assess how sensitive results are to hidden bias in
    observational studies, particularly for matched or weighted data. This
    implementation follows Rosenbaum (2002) methodology.

    Args:
        treated_outcomes: Outcomes for treated units in matched pairs
        control_outcomes: Outcomes for control units in matched pairs
        gamma_range: Range of sensitivity parameters (Γ) to test
        gamma_steps: Number of gamma values to test across the range
        alpha: Significance level for critical gamma calculation
        method: Statistical test method ('wilcoxon' or 'sign_test')

    Returns:
        Dictionary containing:
            - original_p_value: P-value without hidden bias
            - original_statistic: Test statistic value
            - method_used: Statistical test method used
            - bounds: List of bounds results for each gamma value
            - critical_gamma: Gamma value where significance is lost
            - robustness_assessment: Qualitative assessment
            - interpretation: Human-readable interpretation

    Raises:
        ValueError: If input data is invalid or insufficient

    Example:
        >>> import numpy as np
        >>> from causal_inference.sensitivity import rosenbaum_bounds
        >>>
        >>> # Simulate matched pair data
        >>> treated = np.random.normal(2, 1, 50)  # Treatment group
        >>> control = np.random.normal(0, 1, 50)  # Control group
        >>>
        >>> # Calculate Rosenbaum bounds
        >>> results = rosenbaum_bounds(treated, control)
        >>> print(f"Critical Gamma: {results['critical_gamma']:.2f}")
        >>> print(f"Robustness: {results['robustness_assessment']}")
    """
    # Convert to numpy arrays
    treated_array = np.asarray(treated_outcomes, dtype=float)
    control_array = np.asarray(control_outcomes, dtype=float)

    # Call the existing implementation
    return _rosenbaum_bounds_impl(
        treated_array,
        control_array,
        gamma_range=gamma_range,
        gamma_steps=gamma_steps,
        alpha=alpha,
        method=method,
    )
