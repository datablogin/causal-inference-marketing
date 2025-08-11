"""HTML report generator for comprehensive causal inference analysis.

This module creates professional HTML reports that combine executive summaries,
technical details, diagnostics, and sensitivity analysis in a business-friendly format.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.base import BaseEstimator, CausalEffect
from .nlg import BusinessInsightsGenerator


class HTMLReportGenerator:
    """Generate comprehensive HTML reports for causal inference analysis.

    This class creates professional, business-ready reports that combine:
    - Executive summary with key findings
    - Technical details and diagnostics
    - Interactive visualizations
    - Sensitivity analysis results
    - Actionable business recommendations
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        effect: CausalEffect,
        data: pd.DataFrame,
        treatment_column: str,
        outcome_column: str,
        covariate_columns: list[str] | None = None,
        method_name: str = "causal_analysis",
        confidence_level: float = 0.95,
    ):
        """Initialize the HTML report generator.

        Args:
            estimator: Fitted causal inference estimator
            effect: Estimated causal effect
            data: Original analysis data
            treatment_column: Name of treatment column
            outcome_column: Name of outcome column
            covariate_columns: List of covariate column names
            method_name: Name of estimation method used
            confidence_level: Confidence level used in analysis
        """
        self.estimator = estimator
        self.effect = effect
        self.data = data
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.covariate_columns = covariate_columns or []
        self.method_name = method_name
        self.confidence_level = confidence_level

        # Initialize business insights generator
        self.insights_generator = BusinessInsightsGenerator(
            effect=effect,
            data=data,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            method_name=method_name,
        )

    def generate_report(
        self,
        template: str = "executive",
        include_sensitivity: bool = True,
        include_diagnostics: bool = True,
        title: str | None = None,
        analyst_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate complete HTML report.

        Args:
            template: Report template ('executive', 'technical', 'full')
            include_sensitivity: Whether to include sensitivity analysis
            include_diagnostics: Whether to include diagnostic plots
            title: Custom report title
            analyst_name: Name of analyst conducting analysis
            **kwargs: Additional template-specific arguments

        Returns:
            Complete HTML report as string
        """
        # Set default title
        if title is None:
            title = f"Causal Analysis Report - {self.treatment_column.title()} Impact"

        # Generate report sections
        sections = {}

        # Executive summary (always included)
        sections["executive_summary"] = self._generate_executive_summary()

        # Data overview
        sections["data_overview"] = self._generate_data_overview()

        # Method explanation
        sections["method_explanation"] = self._generate_method_explanation()

        # Results section
        sections["results"] = self._generate_results_section()

        # Diagnostics (if requested)
        if include_diagnostics:
            sections["diagnostics"] = self._generate_diagnostics_section()

        # Sensitivity analysis (if requested)
        if include_sensitivity:
            sections["sensitivity"] = self._generate_sensitivity_section()

        # Business recommendations
        sections["recommendations"] = self._generate_recommendations_section()

        # Technical appendix (for technical/full templates)
        if template in ["technical", "full"]:
            sections["technical_appendix"] = self._generate_technical_appendix()

        # Generate final HTML
        return self._compile_html_report(
            title=title,
            sections=sections,
            template=template,
            analyst_name=analyst_name,
            **kwargs,
        )

    def _generate_executive_summary(self) -> dict[str, Any]:
        """Generate executive summary section."""
        return {
            "html": self.insights_generator.generate_executive_summary(),
            "key_findings": [
                f"Treatment effect: {self.effect.ate:.3f}",
                f"Statistical significance: {'Yes' if self.effect.p_value < 0.05 else 'No'}",
                f"Confidence interval: [{self.effect.ate_ci_lower:.3f}, {self.effect.ate_ci_upper:.3f}]",
                f"Sample size: {len(self.data):,} observations",
            ],
        }

    def _generate_data_overview(self) -> dict[str, Any]:
        """Generate data overview section."""
        # Basic statistics
        treatment_stats = self.data[self.treatment_column].value_counts()
        outcome_stats = self.data[self.outcome_column].describe()

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Treatment distribution
        treatment_stats.plot(kind="bar", ax=axes[0], color=["#3498db", "#e74c3c"])
        axes[0].set_title("Treatment Distribution")
        axes[0].set_xlabel("Treatment Group")
        axes[0].set_ylabel("Count")

        # Outcome by treatment
        self.data.boxplot(
            column=self.outcome_column, by=self.treatment_column, ax=axes[1]
        )
        axes[1].set_title(f"{self.outcome_column.title()} by Treatment")

        plt.tight_layout()
        plot_html = self._fig_to_html(fig)
        plt.close()

        return {
            "treatment_stats": treatment_stats.to_dict(),
            "outcome_stats": outcome_stats.to_dict(),
            "n_covariates": len(self.covariate_columns),
            "sample_size": len(self.data),
            "plot_html": plot_html,
        }

    def _generate_method_explanation(self) -> dict[str, Any]:
        """Generate method explanation section."""
        method_explanations = {
            "g_computation": {
                "name": "G-Computation (Standardization)",
                "description": "Estimates causal effects by modeling the outcome and predicting counterfactual scenarios.",
                "assumptions": [
                    "No unmeasured confounders",
                    "Correct outcome model specification",
                ],
                "strengths": [
                    "Direct modeling of outcome",
                    "Flexible model specification",
                ],
                "limitations": ["Sensitive to outcome model misspecification"],
            },
            "ipw": {
                "name": "Inverse Probability Weighting (IPW)",
                "description": "Estimates causal effects by weighting observations by inverse treatment probability.",
                "assumptions": [
                    "No unmeasured confounders",
                    "Correct propensity score model",
                ],
                "strengths": [
                    "Does not model outcome directly",
                    "Well-established method",
                ],
                "limitations": [
                    "Sensitive to extreme propensity scores",
                    "Can be unstable",
                ],
            },
            "aipw": {
                "name": "Augmented Inverse Probability Weighting (AIPW)",
                "description": "Doubly robust method combining G-computation and IPW.",
                "assumptions": ["No unmeasured confounders"],
                "strengths": ["Doubly robust", "Consistent if either model is correct"],
                "limitations": ["More complex", "Requires fitting multiple models"],
            },
        }

        method_info = method_explanations.get(
            self.method_name,
            {
                "name": self.method_name.title(),
                "description": f"Causal inference using {self.method_name} method.",
                "assumptions": ["Standard causal inference assumptions"],
                "strengths": ["Estimates causal effects"],
                "limitations": ["Method-specific limitations apply"],
            },
        )

        return method_info

    def _generate_results_section(self) -> dict[str, Any]:
        """Generate results section with effect estimates."""
        # Create results visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot treatment effect with confidence interval
        ax.errorbar(
            [0],
            [self.effect.ate],
            yerr=[
                [self.effect.ate - self.effect.ate_ci_lower],
                [self.effect.ate_ci_upper - self.effect.ate],
            ],
            fmt="o",
            markersize=10,
            capsize=10,
            capthick=2,
            color="#2E86C1",
            ecolor="#2E86C1",
        )

        # Add reference line at zero
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel("Treatment Effect")
        ax.set_title(
            f"Estimated Treatment Effect\n{self.confidence_level * 100:.0f}% Confidence Interval"
        )
        ax.set_xticks([])

        # Add effect size annotation
        ax.annotate(
            f"ATE = {self.effect.ate:.3f}\\np = {self.effect.p_value:.3f}",
            xy=(0, self.effect.ate),
            xytext=(0.2, self.effect.ate),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=12,
            ha="left",
        )

        plt.tight_layout()
        plot_html = self._fig_to_html(fig)
        plt.close()

        return {
            "ate": self.effect.ate,
            "ci_lower": self.effect.ate_ci_lower,
            "ci_upper": self.effect.ate_ci_upper,
            "p_value": getattr(self.effect, "p_value", None),
            "std_error": getattr(self.effect, "std_error", None),
            "confidence_level": self.confidence_level,
            "plot_html": plot_html,
            "interpretation": self.insights_generator.interpret_effect_size(),
        }

    def _generate_diagnostics_section(self) -> dict[str, Any]:
        """Generate diagnostics section."""
        diagnostics = {}

        # Basic balance check
        if self.covariate_columns:
            balance_stats = self._compute_balance_statistics()
            diagnostics["balance"] = balance_stats
            diagnostics["balance_plot"] = self._create_balance_plot()

        # Residual analysis (if available)
        if hasattr(self.estimator, "outcome_model") and hasattr(
            self.estimator.outcome_model, "predict"
        ):
            diagnostics["residuals"] = self._create_residual_plots()

        return diagnostics

    def _generate_sensitivity_section(self) -> dict[str, Any]:
        """Generate sensitivity analysis section."""
        # Import here to avoid circular imports
        from ..sensitivity import generate_sensitivity_report

        try:
            sensitivity_results = generate_sensitivity_report(
                treatment_data=self.data[self.treatment_column].values,
                outcome_data=self.data[self.outcome_column].values,
                covariates_data=self.data[self.covariate_columns].values
                if self.covariate_columns
                else None,
                observed_effect=self.effect.ate,
                ci_lower=self.effect.ate_ci_lower,
                ci_upper=self.effect.ate_ci_upper,
                output_format="dict",
            )

            return {
                "results": sensitivity_results,
                "summary": sensitivity_results.get(
                    "overall_assessment", "Analysis completed"
                ),
                "recommendations": sensitivity_results.get("recommendations", []),
            }
        except Exception as e:
            return {"error": f"Sensitivity analysis failed: {str(e)}", "results": None}

    def _generate_recommendations_section(self) -> dict[str, Any]:
        """Generate business recommendations section."""
        return {
            "recommendations": self.insights_generator.generate_recommendations(),
            "next_steps": self.insights_generator.suggest_next_steps(),
            "caveats": self.insights_generator.list_caveats(),
        }

    def _generate_technical_appendix(self) -> dict[str, Any]:
        """Generate technical appendix with detailed statistics."""
        appendix = {
            "estimator_details": {
                "class": self.estimator.__class__.__name__,
                "parameters": getattr(self.estimator, "get_params", lambda: {})(),
            },
            "sample_characteristics": self.data.describe().to_dict(),
            "effect_details": {
                "ate": self.effect.ate,
                "ate_ci_lower": self.effect.ate_ci_lower,
                "ate_ci_upper": self.effect.ate_ci_upper,
                "p_value": getattr(self.effect, "p_value", None),
                "interpretation": self.effect.interpretation,
            },
        }

        # Add diagnostics if available
        if hasattr(self.estimator, "diagnostics_"):
            appendix["diagnostics"] = self.estimator.diagnostics_

        return appendix

    def _compute_balance_statistics(self) -> dict[str, Any]:
        """Compute covariate balance statistics."""
        balance_stats = {}

        for col in self.covariate_columns:
            if col in self.data.columns:
                treated = self.data[self.data[self.treatment_column] == 1][col]
                control = self.data[self.data[self.treatment_column] == 0][col]

                # Standardized mean difference
                pooled_std = np.sqrt((treated.var() + control.var()) / 2)
                smd = (
                    (treated.mean() - control.mean()) / pooled_std
                    if pooled_std > 0
                    else 0
                )

                balance_stats[col] = {
                    "treated_mean": treated.mean(),
                    "control_mean": control.mean(),
                    "smd": smd,
                    "smd_abs": abs(smd),
                }

        return balance_stats

    def _create_balance_plot(self) -> str:
        """Create covariate balance plot."""
        balance_stats = self._compute_balance_statistics()

        if not balance_stats:
            return ""

        # Create balance plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(balance_stats) * 0.5)))

        variables = list(balance_stats.keys())
        smds = [balance_stats[var]["smd"] for var in variables]

        colors = ["red" if abs(smd) > 0.1 else "green" for smd in smds]

        ax.barh(variables, smds, color=colors, alpha=0.7)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax.axvline(
            x=0.1, color="red", linestyle="--", alpha=0.5, label="SMD = 0.1 threshold"
        )
        ax.axvline(x=-0.1, color="red", linestyle="--", alpha=0.5)

        ax.set_xlabel("Standardized Mean Difference")
        ax.set_title("Covariate Balance Assessment")
        ax.legend()

        plt.tight_layout()
        plot_html = self._fig_to_html(fig)
        plt.close()

        return plot_html

    def _create_residual_plots(self) -> str:
        """Create residual analysis plots."""
        # This is a simplified version - would need access to fitted models
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Residual Analysis\\n(Feature under development)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            style="italic",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        plot_html = self._fig_to_html(fig)
        plt.close()

        return plot_html

    def _fig_to_html(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to HTML img tag."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()

        return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'

    def _compile_html_report(
        self,
        title: str,
        sections: dict[str, Any],
        template: str,
        analyst_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Compile all sections into final HTML report."""

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # CSS styles
        css_styles = """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .section {
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .executive-summary {
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .key-finding {
            background: #e8f5e8;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #27ae60;
            border-radius: 5px;
        }
        .recommendation {
            background: #fff3cd;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
            border-radius: 5px;
        }
        .warning {
            background: #f8d7da;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #dc3545;
            border-radius: 5px;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .stats-table th, .stats-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .stats-table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .highlight {
            background-color: #fff3cd;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .footer {
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            padding-top: 20px;
            margin-top: 40px;
            font-size: 14px;
        }
        </style>
        """

        # Start building HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            {css_styles}
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Comprehensive Causal Inference Analysis Report</p>
                <div style="font-size: 14px; margin-top: 15px;">
                    Generated on {timestamp}
                    {f" • Analyst: {analyst_name}" if analyst_name else ""}
                </div>
            </div>
        """

        # Executive Summary
        exec_summary = sections.get("executive_summary", {})
        html += f"""
            <div class="executive-summary">
                <h2>📊 Executive Summary</h2>
                {exec_summary.get("html", "Executive summary not available")}
                <div style="margin-top: 20px;">
                    <h3>Key Findings:</h3>
                    <ul>
        """
        for finding in exec_summary.get("key_findings", []):
            html += f"<li>{finding}</li>"
        html += "</ul></div></div>"

        # Data Overview
        if "data_overview" in sections:
            data_section = sections["data_overview"]
            html += f"""
                <div class="section">
                    <h2>📈 Data Overview</h2>
                    <p><strong>Sample Size:</strong> {data_section.get("sample_size", "N/A"):,} observations</p>
                    <p><strong>Treatment Groups:</strong> {len(data_section.get("treatment_stats", {}))}</p>
                    <p><strong>Covariates:</strong> {data_section.get("n_covariates", 0)} variables</p>
                    {data_section.get("plot_html", "")}
                </div>
            """

        # Method Explanation
        if "method_explanation" in sections:
            method = sections["method_explanation"]
            html += f"""
                <div class="section">
                    <h2>🔬 Analysis Method</h2>
                    <h3>{method.get("name", "Analysis Method")}</h3>
                    <p>{method.get("description", "No description available")}</p>

                    <h4>Key Assumptions:</h4>
                    <ul>
            """
            for assumption in method.get("assumptions", []):
                html += f"<li>{assumption}</li>"
            html += """
                    </ul>

                    <h4>Strengths:</h4>
                    <ul>
            """
            for strength in method.get("strengths", []):
                html += f"<li>{strength}</li>"
            html += "</ul></div>"

        # Results
        if "results" in sections:
            results = sections["results"]
            p_value = results.get("p_value")
            significance = (
                "Statistically Significant"
                if p_value is not None and p_value < 0.05
                else "Not Statistically Significant"
            )
            html += f"""
                <div class="section">
                    <h2>📋 Results</h2>
                    <div class="key-finding">
                        <h3>Treatment Effect: <span class="highlight">{results.get("ate", 0):.3f}</span></h3>
                        <p><strong>95% Confidence Interval:</strong> [{results.get("ci_lower", 0):.3f}, {results.get("ci_upper", 0):.3f}]</p>
                        <p><strong>Statistical Significance:</strong> {significance}{f" (p = {p_value:.3f})" if p_value is not None else ""}</p>
                    </div>
                    {results.get("plot_html", "")}
                    <p><strong>Interpretation:</strong> {results.get("interpretation", "Effect interpretation not available")}</p>
                </div>
            """

        # Diagnostics
        if "diagnostics" in sections and sections["diagnostics"]:
            diagnostics = sections["diagnostics"]
            html += f"""
                <div class="section">
                    <h2>🔍 Diagnostics</h2>
                    {diagnostics.get("balance_plot", "")}
                    {diagnostics.get("residuals", "")}
                </div>
            """

        # Sensitivity Analysis
        if "sensitivity" in sections:
            sensitivity = sections["sensitivity"]
            if sensitivity.get("results"):
                html += f"""
                    <div class="section">
                        <h2>⚖️ Sensitivity Analysis</h2>
                        <p><strong>Overall Assessment:</strong> {sensitivity.get("summary", "Analysis completed")}</p>

                        <h3>Robustness Checks:</h3>
                        <ul>
                """
                for rec in sensitivity.get("recommendations", [])[:5]:  # Limit to top 5
                    html += f"<li>{rec}</li>"
                html += """
                        </ul>
                    </div>
                """
            elif sensitivity.get("error"):
                html += f"""
                    <div class="section">
                        <h2>⚖️ Sensitivity Analysis</h2>
                        <div class="warning">
                            <p><strong>Note:</strong> {sensitivity.get("error")}</p>
                        </div>
                    </div>
                """

        # Recommendations
        if "recommendations" in sections:
            rec_section = sections["recommendations"]
            html += """
                <div class="section">
                    <h2>💡 Recommendations</h2>
                    <h3>Business Recommendations:</h3>
            """
            for rec in rec_section.get("recommendations", []):
                html += f'<div class="recommendation">{rec}</div>'

            html += "<h3>Next Steps:</h3>"
            for step in rec_section.get("next_steps", []):
                html += f'<div class="recommendation">{step}</div>'

            if rec_section.get("caveats"):
                html += "<h3>Important Caveats:</h3>"
                for caveat in rec_section.get("caveats", []):
                    html += f'<div class="warning">{caveat}</div>'

            html += "</div>"

        # Technical Appendix (for technical/full templates)
        if template in ["technical", "full"] and "technical_appendix" in sections:
            appendix = sections["technical_appendix"]
            html += f"""
                <div class="section">
                    <h2>📚 Technical Appendix</h2>
                    <h3>Estimator Details:</h3>
                    <p><strong>Method:</strong> {appendix["estimator_details"].get("class", "N/A")}</p>

                    <h3>Effect Details:</h3>
                    <table class="stats-table">
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Average Treatment Effect (ATE)</td><td>{appendix["effect_details"].get("ate", 0):.6f}</td></tr>
                        <tr><td>95% CI Lower</td><td>{appendix["effect_details"].get("ate_ci_lower", 0):.6f}</td></tr>
                        <tr><td>95% CI Upper</td><td>{appendix["effect_details"].get("ate_ci_upper", 0):.6f}</td></tr>
                        <tr><td>P-value</td><td>{appendix["effect_details"].get("p_value") if appendix["effect_details"].get("p_value") is not None else "N/A"}</td></tr>
                    </table>
                </div>
            """

        # Footer
        html += f"""
            <div class="footer">
                <p>This report was generated using the Causal Inference Tools library.</p>
                <p>For questions about interpretation or methodology, consult with a causal inference expert.</p>
                <p><em>Report generated on {timestamp}</em></p>
            </div>
        </body>
        </html>
        """

        return html
