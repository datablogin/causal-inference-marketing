"""Comprehensive sensitivity analysis suite for causal inference.

This module provides industry-standard sensitivity analysis methods to assess
the robustness of causal effect estimates to unmeasured confounding and
modeling assumptions.

Main Functions:
    rosenbaum_bounds: Rosenbaum bounds for matched/weighted designs
    e_value: E-value calculation following VanderWeele & Ding
    oster_delta: Oster's δ and partial R² for omitted variable bias
    negative_control: Negative control outcome & exposure analysis
    placebo_test: Placebo treatment & dummy outcome refutation tests
"""

from .controls import negative_control
from .e_values import e_value
from .oster import oster_delta
from .placebo import placebo_test
from .reporting import generate_sensitivity_report
from .rosenbaum import rosenbaum_bounds

__all__ = [
    "rosenbaum_bounds",
    "e_value",
    "oster_delta",
    "negative_control",
    "placebo_test",
    "generate_sensitivity_report",
]
