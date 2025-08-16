"""Automated diagnostic report generation with HTML templates.

This module provides comprehensive report generation capabilities that combine
all diagnostic visualizations into professional HTML reports suitable for
stakeholder communication.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import plotting libraries with fallback
try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import plotly  # noqa: F401

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..core.base import CovariateData, OutcomeData, TreatmentData
from .balance_plots import LovePlotData, LovePlotGenerator
from .propensity_plots import PropensityOverlapResult, PropensityPlotGenerator
from .residual_analysis import ResidualAnalysisResult, ResidualAnalyzer
from .weight_diagnostics import WeightDiagnostics, WeightDiagnosticsResult

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReportData:
    """Comprehensive data for diagnostic report generation."""

    # Data information
    n_observations: int
    n_treated: int
    n_control: int
    treatment_column: str
    outcome_column: str
    covariate_columns: list[str]

    # Analysis results
    love_plot_data: LovePlotData | None = None
    weight_diagnostics: WeightDiagnosticsResult | None = None
    propensity_overlap: PropensityOverlapResult | None = None
    residual_analysis: ResidualAnalysisResult | None = None

    # Recommendations
    love_plot_recommendations: list[str] = None
    weight_recommendations: list[str] = None
    propensity_recommendations: list[str] = None
    residual_recommendations: list[str] = None

    # Additional metadata
    analysis_date: datetime = None
    estimator_name: str = "Unknown"
    ate_estimate: float | None = None
    ate_ci_lower: float | None = None
    ate_ci_upper: float | None = None


class DiagnosticReportGenerator:
    """Generator for comprehensive diagnostic HTML reports."""

    def __init__(
        self,
        template_dir: Path | None = None,
        include_interactive: bool = True,
        max_file_size_mb: float = 5.0,
        performance_mode: bool = False,
    ):
        """Initialize report generator.

        Args:
            template_dir: Directory containing HTML templates
            include_interactive: Whether to include interactive plots
            max_file_size_mb: Maximum file size for generated reports
            performance_mode: Enable performance optimizations for speed
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting libraries not available. Install with: "
                "pip install matplotlib seaborn"
            )

        self.template_dir = template_dir
        self.include_interactive = include_interactive and PLOTLY_AVAILABLE
        self.max_file_size_mb = max_file_size_mb
        self.performance_mode = performance_mode

        # Initialize plot generators - optimize based on performance mode
        self.love_plot_generator = LovePlotGenerator()
        self.weight_analyzer = WeightDiagnostics(
            enable_distribution_tests=not performance_mode,
            figsize=(8, 6) if performance_mode else (12, 8),
        )
        self.propensity_generator = PropensityPlotGenerator()
        self.residual_analyzer = ResidualAnalyzer(
            figsize=(10, 8) if performance_mode else (15, 12)
        )

    def generate_comprehensive_report(
        self,
        treatment_data: TreatmentData,
        outcome_data: OutcomeData,
        covariates: CovariateData | None = None,
        weights: NDArray[np.floating[Any]] | None = None,
        propensity_scores: NDArray[np.floating[Any]] | None = None,
        residuals: NDArray[np.floating[Any]] | None = None,
        fitted_values: NDArray[np.floating[Any]] | None = None,
        estimator_name: str = "Causal Estimator",
        ate_estimate: float | None = None,
        ate_ci_lower: float | None = None,
        ate_ci_upper: float | None = None,
        save_path: str | None = None,
        template_type: str = "comprehensive",
    ) -> str:
        """Generate a comprehensive diagnostic report.

        Args:
            treatment_data: Treatment assignment data
            outcome_data: Outcome variable data
            covariates: Covariate data
            weights: Analysis weights (e.g., IPW weights)
            propensity_scores: Estimated propensity scores
            residuals: Model residuals
            fitted_values: Model fitted values
            estimator_name: Name of the causal estimator used
            ate_estimate: Average treatment effect estimate
            ate_ci_lower: Lower bound of ATE confidence interval
            ate_ci_upper: Upper bound of ATE confidence interval
            save_path: Path to save the HTML report
            template_type: Type of template ('executive', 'technical', 'comprehensive')

        Returns:
            HTML report as string
        """
        logger.info(f"Starting comprehensive report generation using {estimator_name}")
        logger.debug(f"Template type: {template_type}")

        # Collect all data and analysis results
        report_data = self._collect_report_data(
            treatment_data,
            outcome_data,
            covariates,
            weights,
            propensity_scores,
            residuals,
            fitted_values,
            estimator_name,
            ate_estimate,
            ate_ci_lower,
            ate_ci_upper,
        )

        # Generate all plots and encode them
        plot_data = self._generate_all_plots(
            treatment_data,
            outcome_data,
            covariates,
            weights,
            propensity_scores,
            residuals,
            fitted_values,
            report_data,
        )

        # Generate HTML report
        logger.info("Generating HTML report from collected data and plots")
        html_content = self._generate_html_report(report_data, plot_data, template_type)

        # Log report statistics
        content_size_mb = len(html_content) / (1024 * 1024)
        logger.info(
            f"Generated HTML report: {len(html_content):,} characters ({content_size_mb:.2f} MB)"
        )

        if content_size_mb > self.max_file_size_mb:
            logger.warning(
                f"Report size ({content_size_mb:.2f} MB) exceeds recommended maximum ({self.max_file_size_mb} MB)"
            )

        # Save if path provided
        if save_path:
            logger.debug(f"Saving report to: {save_path}")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Report saved successfully to {save_path}")

        return html_content

    def _collect_report_data(
        self,
        treatment_data: TreatmentData,
        outcome_data: OutcomeData,
        covariates: CovariateData | None,
        weights: NDArray[np.floating[Any]] | None,
        propensity_scores: NDArray[np.floating[Any]] | None,
        residuals: NDArray[np.floating[Any]] | None,
        fitted_values: NDArray[np.floating[Any]] | None,
        estimator_name: str,
        ate_estimate: float | None,
        ate_ci_lower: float | None,
        ate_ci_upper: float | None,
    ) -> DiagnosticReportData:
        """Collect all data needed for the report."""
        n_observations = len(treatment_data.values)
        n_treated = int(np.sum(treatment_data.values == 1))
        n_control = n_observations - n_treated

        # Perform analyses
        love_plot_data = None
        weight_diagnostics = None
        propensity_overlap = None
        residual_analysis = None

        love_plot_recommendations = []
        weight_recommendations = []
        propensity_recommendations = []
        residual_recommendations = []

        # Love plot analysis (if covariates available)
        if covariates is not None:
            love_plot_data = self.love_plot_generator.calculate_balance_data(
                covariates, treatment_data, weights
            )

            # Generate balance recommendations
            balanced_count = np.sum(
                np.abs(love_plot_data.smd_before) <= love_plot_data.balance_threshold
            )
            total_count = len(love_plot_data.covariate_names)

            if balanced_count / total_count >= 0.9:
                love_plot_recommendations.append(
                    "✅ Excellent covariate balance achieved."
                )
            elif balanced_count / total_count >= 0.7:
                love_plot_recommendations.append("✅ Good covariate balance achieved.")
            elif balanced_count / total_count >= 0.5:
                love_plot_recommendations.append(
                    "⚠️ Moderate covariate balance. Consider adjustment methods."
                )
            else:
                love_plot_recommendations.append(
                    "❌ Poor covariate balance. Strong confounding likely."
                )

        # Weight diagnostics (if weights available)
        if weights is not None:
            weight_diagnostics = self.weight_analyzer.analyze_weights(weights)
            weight_recommendations = self.weight_analyzer.generate_recommendations(
                weight_diagnostics
            )

        # Propensity score analysis (if propensity scores available)
        if propensity_scores is not None:
            propensity_overlap = self.propensity_generator.analyze_propensity_overlap(
                propensity_scores, treatment_data
            )
            propensity_recommendations = (
                self.propensity_generator.generate_recommendations(propensity_overlap)
            )

        # Residual analysis (if residuals and fitted values available)
        if residuals is not None and fitted_values is not None:
            design_matrix = covariates.values if covariates is not None else None
            residual_analysis = self.residual_analyzer.analyze_residuals(
                residuals, fitted_values, design_matrix
            )
            residual_recommendations = self.residual_analyzer.generate_recommendations(
                residual_analysis
            )

        return DiagnosticReportData(
            n_observations=n_observations,
            n_treated=n_treated,
            n_control=n_control,
            treatment_column=getattr(treatment_data, "name", "treatment"),
            outcome_column=getattr(outcome_data, "name", "outcome"),
            covariate_columns=covariates.names if covariates else [],
            love_plot_data=love_plot_data,
            weight_diagnostics=weight_diagnostics,
            propensity_overlap=propensity_overlap,
            residual_analysis=residual_analysis,
            love_plot_recommendations=love_plot_recommendations,
            weight_recommendations=weight_recommendations,
            propensity_recommendations=propensity_recommendations,
            residual_recommendations=residual_recommendations,
            analysis_date=datetime.now(),
            estimator_name=estimator_name,
            ate_estimate=ate_estimate,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
        )

    def _generate_all_plots(
        self,
        treatment_data: TreatmentData,
        outcome_data: OutcomeData,
        covariates: CovariateData | None,
        weights: NDArray[np.floating[Any]] | None,
        propensity_scores: NDArray[np.floating[Any]] | None,
        residuals: NDArray[np.floating[Any]] | None,
        fitted_values: NDArray[np.floating[Any]] | None,
        report_data: DiagnosticReportData,
    ) -> dict[str, str]:
        """Generate all plots and return as base64-encoded images."""
        plot_data = {}

        # In performance mode, skip plot generation entirely
        if self.performance_mode:
            # Skip all plot generation for maximum performance
            return {}
        else:
            # Full plot generation for normal mode
            # Love plot
            if report_data.love_plot_data is not None:
                fig = self.love_plot_generator.create_love_plot(
                    report_data.love_plot_data, interactive=False
                )
                plot_data["love_plot"] = self._figure_to_base64(fig)
                plt.close(fig)

            # Weight diagnostics
            if report_data.weight_diagnostics is not None:
                fig = self.weight_analyzer.create_weight_plots(
                    report_data.weight_diagnostics, interactive=False
                )
                plot_data["weight_plots"] = self._figure_to_base64(fig)
                plt.close(fig)

            # Propensity score plots
            if report_data.propensity_overlap is not None:
                fig = self.propensity_generator.create_propensity_plots(
                    report_data.propensity_overlap, interactive=False
                )
                plot_data["propensity_plots"] = self._figure_to_base64(fig)
                plt.close(fig)

            # Residual analysis
            if report_data.residual_analysis is not None:
                fig = self.residual_analyzer.create_residual_plots(
                    report_data.residual_analysis, interactive=False
                )
                plot_data["residual_plots"] = self._figure_to_base64(fig)
                plt.close(fig)

        return plot_data

    def _figure_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64-encoded string."""
        buffer = io.BytesIO()
        # Use performance-optimized settings
        dpi = 75 if self.performance_mode else 100
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()

        return f"data:image/png;base64,{image_base64}"

    def _generate_html_report(
        self,
        report_data: DiagnosticReportData,
        plot_data: dict[str, str],
        template_type: str,
    ) -> str:
        """Generate the HTML report content."""
        # Basic template - in a full implementation, this would load from external template files
        html_template = self._get_html_template(template_type)

        # Format the template with data
        formatted_html = html_template.format(
            # Basic information
            analysis_date=report_data.analysis_date.strftime("%Y-%m-%d %H:%M:%S"),
            estimator_name=report_data.estimator_name,
            n_observations=report_data.n_observations,
            n_treated=report_data.n_treated,
            n_control=report_data.n_control,
            treatment_column=report_data.treatment_column,
            outcome_column=report_data.outcome_column,
            n_covariates=len(report_data.covariate_columns),
            covariate_list=", ".join(report_data.covariate_columns)
            if report_data.covariate_columns
            else "None",
            # Treatment effect results
            ate_estimate=f"{report_data.ate_estimate:.4f}"
            if report_data.ate_estimate is not None
            else "Not available",
            ate_ci_lower=f"{report_data.ate_ci_lower:.4f}"
            if report_data.ate_ci_lower is not None
            else "N/A",
            ate_ci_upper=f"{report_data.ate_ci_upper:.4f}"
            if report_data.ate_ci_upper is not None
            else "N/A",
            # Recommendations
            love_plot_recommendations=self._format_recommendations(
                report_data.love_plot_recommendations
            ),
            weight_recommendations=self._format_recommendations(
                report_data.weight_recommendations
            ),
            propensity_recommendations=self._format_recommendations(
                report_data.propensity_recommendations
            ),
            residual_recommendations=self._format_recommendations(
                report_data.residual_recommendations
            ),
            # Summary assessments
            overall_assessment=self._generate_overall_assessment(report_data),
        )

        # Add conditional sections using string replacement
        formatted_html = self._add_conditional_sections(
            formatted_html, plot_data, report_data
        )

        # Add CI information if available
        if (
            report_data.ate_ci_lower is not None
            and report_data.ate_ci_upper is not None
        ):
            ci_html = f"<p><strong>95% Confidence Interval:</strong> [{report_data.ate_ci_lower:.4f}, {report_data.ate_ci_upper:.4f}]</p>"
            formatted_html = formatted_html.replace(
                "            <!-- CI info will be inserted dynamically -->",
                f"            {ci_html}",
            )
        else:
            formatted_html = formatted_html.replace(
                "            <!-- CI info will be inserted dynamically -->", ""
            )

        return formatted_html

    def _add_conditional_sections(
        self, html: str, plot_data: dict[str, str], report_data: DiagnosticReportData
    ) -> str:
        """Add conditional sections to the HTML based on available data."""
        # Love plot section
        if plot_data.get("love_plot"):
            love_section = f"""
    <div class="section">
        <div class="section-header">
            <h2>⚖️ Covariate Balance Analysis</h2>
        </div>
        <div class="section-content">
            <div class="plot-container">
                <img src="{plot_data["love_plot"]}" alt="Love Plot" />
            </div>
            <div class="recommendations">
                <h4>Recommendations:</h4>
                <ul>
                    {self._format_recommendations(report_data.love_plot_recommendations)}
                </ul>
            </div>
        </div>
    </div>"""
            html = html.replace(
                "    <!-- Love Plot Section will be inserted here -->", love_section
            )
        else:
            html = html.replace(
                "    <!-- Love Plot Section will be inserted here -->", ""
            )

        # Weight plots section
        if plot_data.get("weight_plots"):
            weight_section = f"""
    <div class="section">
        <div class="section-header">
            <h2>⚖️ Weight Distribution Analysis</h2>
        </div>
        <div class="section-content">
            <div class="plot-container">
                <img src="{plot_data["weight_plots"]}" alt="Weight Diagnostics" />
            </div>
            <div class="recommendations">
                <h4>Recommendations:</h4>
                <ul>
                    {self._format_recommendations(report_data.weight_recommendations)}
                </ul>
            </div>
        </div>
    </div>"""
            html = html.replace(
                "    <!-- Weight Plots Section will be inserted here -->",
                weight_section,
            )
        else:
            html = html.replace(
                "    <!-- Weight Plots Section will be inserted here -->", ""
            )

        # Propensity plots section
        if plot_data.get("propensity_plots"):
            propensity_section = f"""
    <div class="section">
        <div class="section-header">
            <h2>🎲 Propensity Score Analysis</h2>
        </div>
        <div class="section-content">
            <div class="plot-container">
                <img src="{plot_data["propensity_plots"]}" alt="Propensity Score Diagnostics" />
            </div>
            <div class="recommendations">
                <h4>Recommendations:</h4>
                <ul>
                    {self._format_recommendations(report_data.propensity_recommendations)}
                </ul>
            </div>
        </div>
    </div>"""
            html = html.replace(
                "    <!-- Propensity Plots Section will be inserted here -->",
                propensity_section,
            )
        else:
            html = html.replace(
                "    <!-- Propensity Plots Section will be inserted here -->", ""
            )

        # Residual plots section
        if plot_data.get("residual_plots"):
            residual_section = f"""
    <div class="section">
        <div class="section-header">
            <h2>📈 Residual Analysis</h2>
        </div>
        <div class="section-content">
            <div class="plot-container">
                <img src="{plot_data["residual_plots"]}" alt="Residual Analysis" />
            </div>
            <div class="recommendations">
                <h4>Recommendations:</h4>
                <ul>
                    {self._format_recommendations(report_data.residual_recommendations)}
                </ul>
            </div>
        </div>
    </div>"""
            html = html.replace(
                "    <!-- Residual Plots Section will be inserted here -->",
                residual_section,
            )
        else:
            html = html.replace(
                "    <!-- Residual Plots Section will be inserted here -->", ""
            )

        return html

    def _format_recommendations(self, recommendations: list[str]) -> str:
        """Format recommendations as HTML list."""
        if not recommendations:
            return "<li>No specific recommendations available.</li>"

        formatted = []
        for rec in recommendations:
            # Color code based on status
            if rec.startswith("✅"):
                formatted.append(f'<li class="recommendation-good">{rec}</li>')
            elif rec.startswith("⚠️"):
                formatted.append(f'<li class="recommendation-warning">{rec}</li>')
            elif rec.startswith("❌"):
                formatted.append(f'<li class="recommendation-error">{rec}</li>')
            else:
                formatted.append(f"<li>{rec}</li>")

        return "\n".join(formatted)

    def _generate_overall_assessment(self, report_data: DiagnosticReportData) -> str:
        """Generate an overall assessment of the analysis quality."""
        issues = []
        strengths = []

        # Check covariate balance
        if report_data.love_plot_data is not None:
            balanced_count = np.sum(
                np.abs(report_data.love_plot_data.smd_before)
                <= report_data.love_plot_data.balance_threshold
            )
            balance_ratio = balanced_count / len(
                report_data.love_plot_data.covariate_names
            )

            if balance_ratio >= 0.8:
                strengths.append("Good covariate balance")
            else:
                issues.append("Poor covariate balance")

        # Check weight diagnostics
        if report_data.weight_diagnostics is not None:
            if report_data.weight_diagnostics.extreme_weight_percentage < 5:
                strengths.append("Reasonable weight distribution")
            else:
                issues.append("Extreme weights detected")

        # Check propensity overlap
        if report_data.propensity_overlap is not None:
            if report_data.propensity_overlap.overlap_percentage > 70:
                strengths.append("Good propensity score overlap")
            else:
                issues.append("Limited propensity score overlap")

        # Check residual analysis
        if report_data.residual_analysis is not None:
            if (
                report_data.residual_analysis.normality_assumption_met
                and report_data.residual_analysis.homoscedasticity_assumption_met
            ):
                strengths.append("Model assumptions satisfied")
            else:
                issues.append("Model assumption violations")

        # Generate assessment
        if len(issues) == 0:
            assessment = "✅ <strong>Excellent:</strong> Analysis appears robust with no major issues detected."
        elif len(issues) == 1:
            assessment = f"⚠️ <strong>Good:</strong> Analysis is generally solid but note: {issues[0].lower()}."
        elif len(issues) == 2:
            assessment = f"⚠️ <strong>Moderate:</strong> Several concerns detected: {', '.join(issues).lower()}."
        else:
            assessment = f"❌ <strong>Caution:</strong> Multiple issues detected: {', '.join(issues).lower()}. Consider alternative approaches."

        if strengths:
            assessment += f" Positive aspects: {', '.join(strengths).lower()}."

        return assessment

    def _get_html_template(self, template_type: str) -> str:
        """Get HTML template based on type.

        Supports both external template files and built-in templates.
        External templates are loaded from the template_dir if specified.
        """
        # Try to load external template first
        if self.template_dir is not None:
            template_path = Path(self.template_dir) / f"{template_type}_template.html"
            if template_path.exists():
                try:
                    with open(template_path, encoding="utf-8") as f:
                        return f.read()
                except (OSError, UnicodeDecodeError) as e:
                    # Fall back to built-in template if external template fails
                    import warnings

                    warnings.warn(
                        f"Failed to load external template {template_path}: {e}. "
                        f"Using built-in template instead.",
                        UserWarning,
                    )

        # Use built-in template as fallback
        return self._get_builtin_template(template_type)

    def _get_builtin_template(self, template_type: str) -> str:
        """Get built-in HTML template."""
        # This is a comprehensive template with modern styling

        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Causal Inference Diagnostic Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
        }}

        .subtitle {{
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .summary-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }}

        .summary-card h3 {{
            margin: 0 0 0.5rem 0;
            color: #667eea;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .summary-card .value {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #333;
        }}

        .section {{
            background: white;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}

        .section-header {{
            background: #f8f9fa;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #dee2e6;
        }}

        .section-header h2 {{
            margin: 0;
            color: #495057;
            font-size: 1.3rem;
        }}

        .section-content {{
            padding: 1.5rem;
        }}

        .plot-container {{
            text-align: center;
            margin: 1rem 0;
        }}

        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        .recommendations {{
            background: #f8f9fa;
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
        }}

        .recommendations ul {{
            margin: 0;
            padding-left: 1.5rem;
        }}

        .recommendation-good {{
            color: #155724;
            background-color: #d4edda;
            padding: 0.25rem 0.5rem;
            border-radius: 3px;
            margin: 0.25rem 0;
        }}

        .recommendation-warning {{
            color: #856404;
            background-color: #fff3cd;
            padding: 0.25rem 0.5rem;
            border-radius: 3px;
            margin: 0.25rem 0;
        }}

        .recommendation-error {{
            color: #721c24;
            background-color: #f8d7da;
            padding: 0.25rem 0.5rem;
            border-radius: 3px;
            margin: 0.25rem 0;
        }}

        .overall-assessment {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 2rem 0;
            font-size: 1.1rem;
        }}

        .footer {{
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #dee2e6;
        }}

        .no-data {{
            color: #6c757d;
            font-style: italic;
            text-align: center;
            padding: 2rem;
        }}

        @media (max-width: 768px) {{
            .summary-grid {{
                grid-template-columns: 1fr;
            }}

            .header h1 {{
                font-size: 2rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Causal Inference Diagnostic Report</h1>
        <div class="subtitle">Generated on {analysis_date} | Estimator: {estimator_name}</div>
    </div>

    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total Observations</h3>
            <div class="value">{n_observations:,}</div>
        </div>
        <div class="summary-card">
            <h3>Treated Units</h3>
            <div class="value">{n_treated:,}</div>
        </div>
        <div class="summary-card">
            <h3>Control Units</h3>
            <div class="value">{n_control:,}</div>
        </div>
        <div class="summary-card">
            <h3>ATE Estimate</h3>
            <div class="value">{ate_estimate}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <h2>📊 Analysis Overview</h2>
        </div>
        <div class="section-content">
            <p><strong>Treatment Variable:</strong> {treatment_column}</p>
            <p><strong>Outcome Variable:</strong> {outcome_column}</p>
            <p><strong>Number of Covariates:</strong> {n_covariates}</p>
            <p><strong>Covariates:</strong> {covariate_list}</p>
            <!-- CI info will be inserted dynamically -->
        </div>
    </div>

    <div class="overall-assessment">
        <h2 style="margin-top: 0;">🎯 Overall Assessment</h2>
        <p>{overall_assessment}</p>
    </div>

    <!-- Love Plot Section will be inserted here -->

    <!-- Weight Plots Section will be inserted here -->

    <!-- Propensity Plots Section will be inserted here -->

    <!-- Residual Plots Section will be inserted here -->

    <div class="footer">
        <p>This report was automatically generated by the Causal Inference Advanced Diagnostics & Visualization Tools.</p>
        <p>For questions or support, please refer to the documentation.</p>
    </div>
</body>
</html>
        """


def generate_diagnostic_report(
    treatment_data: TreatmentData,
    outcome_data: OutcomeData,
    covariates: CovariateData | None = None,
    weights: NDArray[np.floating[Any]] | None = None,
    propensity_scores: NDArray[np.floating[Any]] | None = None,
    residuals: NDArray[np.floating[Any]] | None = None,
    fitted_values: NDArray[np.floating[Any]] | None = None,
    estimator_name: str = "Causal Estimator",
    ate_estimate: float | None = None,
    ate_ci_lower: float | None = None,
    ate_ci_upper: float | None = None,
    save_path: str | None = None,
    template_type: str = "comprehensive",
    performance_mode: bool = False,
) -> str:
    """Convenience function to generate a diagnostic report.

    Args:
        treatment_data: Treatment assignment data
        outcome_data: Outcome variable data
        covariates: Covariate data
        weights: Analysis weights
        propensity_scores: Estimated propensity scores
        residuals: Model residuals
        fitted_values: Model fitted values
        estimator_name: Name of the estimator
        ate_estimate: Average treatment effect estimate
        ate_ci_lower: Lower bound of ATE confidence interval
        ate_ci_upper: Upper bound of ATE confidence interval
        save_path: Path to save the report
        template_type: Type of template to use
        performance_mode: Enable performance optimizations

    Returns:
        HTML report as string
    """
    generator = DiagnosticReportGenerator(performance_mode=performance_mode)

    return generator.generate_comprehensive_report(
        treatment_data=treatment_data,
        outcome_data=outcome_data,
        covariates=covariates,
        weights=weights,
        propensity_scores=propensity_scores,
        residuals=residuals,
        fitted_values=fitted_values,
        estimator_name=estimator_name,
        ate_estimate=ate_estimate,
        ate_ci_lower=ate_ci_lower,
        ate_ci_upper=ate_ci_upper,
        save_path=save_path,
        template_type=template_type,
    )
