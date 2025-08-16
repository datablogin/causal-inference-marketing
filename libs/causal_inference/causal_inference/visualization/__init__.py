"""Advanced visualization tools for causal inference diagnostics.

This module provides comprehensive visualization capabilities including Love plots,
weight diagnostics, propensity score analysis, residual plots, and automated
HTML report generation for causal inference analyses.
"""

from .balance_plots import LovePlotGenerator, create_love_plot
from .propensity_plots import PropensityPlotGenerator, create_propensity_plots
from .report_generator import DiagnosticReportGenerator, generate_diagnostic_report
from .residual_analysis import ResidualAnalyzer, create_residual_plots
from .weight_diagnostics import WeightDiagnostics, create_weight_plots

__all__ = [
    # Love plots and balance visualization
    "LovePlotGenerator",
    "create_love_plot",
    # Propensity score visualization
    "PropensityPlotGenerator",
    "create_propensity_plots",
    # Weight diagnostics
    "WeightDiagnostics",
    "create_weight_plots",
    # Residual analysis
    "ResidualAnalyzer",
    "create_residual_plots",
    # Report generation
    "DiagnosticReportGenerator",
    "generate_diagnostic_report",
]
