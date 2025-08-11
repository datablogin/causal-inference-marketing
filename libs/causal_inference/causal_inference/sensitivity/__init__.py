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

Utilities:
    benchmark_sensitivity_functions: Performance benchmarking
    run_performance_validation: KPI validation for 100k rows < 2s
    standardize_input: Unified input validation and standardization
"""

from .benchmarks import benchmark_sensitivity_functions, run_performance_validation
from .controls import negative_control
from .e_values import e_value
from .oster import oster_delta
from .placebo import placebo_test
from .reporting import generate_sensitivity_report
from .rosenbaum import rosenbaum_bounds
from .utils import (
    check_model_assumptions,
    check_treatment_variation,
    format_sensitivity_warnings,
    standardize_input,
    validate_treatment_outcome_lengths,
)

__all__ = [
    "rosenbaum_bounds",
    "e_value",
    "oster_delta",
    "negative_control",
    "placebo_test",
    "generate_sensitivity_report",
    "benchmark_sensitivity_functions",
    "run_performance_validation",
    "standardize_input",
    "validate_treatment_outcome_lengths",
    "check_treatment_variation",
    "check_model_assumptions",
    "format_sensitivity_warnings",
]
