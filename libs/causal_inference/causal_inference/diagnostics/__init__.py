"""Diagnostic tools for causal inference analysis.

This module provides comprehensive diagnostic utilities to validate causal inference
assumptions and detect common issues in causal analysis.
"""

from .assumptions import (
    AssumptionChecker,
    check_confounding_detection,
    check_exchangeability,
    detect_confounding_associations,
    run_all_assumption_checks,
)
from .balance import (
    BalanceDiagnostics,
    calculate_standardized_mean_difference,
    calculate_variance_ratio,
    check_covariate_balance,
)
from .falsification import (
    FalsificationTester,
    run_all_falsification_tests,
    run_placebo_outcome_test,
    run_placebo_treatment_test,
)
from .overlap import (
    OverlapDiagnostics,
    assess_positivity,
    calculate_propensity_overlap,
    calculate_propensity_scores,
    check_common_support,
)
from .reporting import (
    DiagnosticReport,
    DiagnosticReportGenerator,
    create_assumption_summary,
    generate_diagnostic_report,
)
from .sensitivity import (
    SensitivityAnalysis,
    assess_sensitivity,
    evalue_calculation,
    rosenbaum_bounds,
    unmeasured_confounding_analysis,
)
from .specification import (
    ModelSpecificationTests,
    functional_form_tests,
    interaction_tests,
    linearity_tests,
    test_model_specification,
)

# Import visualization with optional dependency handling
try:
    from .visualization import (
        DiagnosticVisualizer,
        plot_balance_diagnostics,
        plot_overlap_diagnostics,
        plot_sensitivity_analysis,
        plot_specification_tests,
    )

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

__all__ = [
    # Balance diagnostics
    "BalanceDiagnostics",
    "calculate_standardized_mean_difference",
    "calculate_variance_ratio",
    "check_covariate_balance",
    # Overlap diagnostics
    "OverlapDiagnostics",
    "assess_positivity",
    "calculate_propensity_overlap",
    "calculate_propensity_scores",
    "check_common_support",
    # Assumption checking
    "AssumptionChecker",
    "check_confounding_detection",
    "check_exchangeability",
    "run_all_assumption_checks",
    "detect_confounding_associations",
    # Sensitivity analysis
    "SensitivityAnalysis",
    "rosenbaum_bounds",
    "evalue_calculation",
    "unmeasured_confounding_analysis",
    "assess_sensitivity",
    # Model specification
    "ModelSpecificationTests",
    "linearity_tests",
    "interaction_tests",
    "functional_form_tests",
    "test_model_specification",
    # Falsification tests
    "FalsificationTester",
    "run_all_falsification_tests",
    "run_placebo_outcome_test",
    "run_placebo_treatment_test",
    # Reporting
    "DiagnosticReport",
    "DiagnosticReportGenerator",
    "generate_diagnostic_report",
    "create_assumption_summary",
]

# Add visualization exports if available
if _VISUALIZATION_AVAILABLE:
    __all__.extend(
        [
            "DiagnosticVisualizer",
            "plot_balance_diagnostics",
            "plot_overlap_diagnostics",
            "plot_sensitivity_analysis",
            "plot_specification_tests",
        ]
    )
