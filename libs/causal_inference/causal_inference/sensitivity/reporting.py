"""Unified sensitivity analysis reporting with HTML/PDF output.

This module provides comprehensive reporting capabilities for sensitivity
analysis results, generating analyst-friendly reports with interpretations
and recommendations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .controls import negative_control

# Import functions directly to avoid circular imports
from .e_values import e_value
from .oster import oster_delta
from .placebo import placebo_test
from .rosenbaum import rosenbaum_bounds


def generate_sensitivity_report(
    treatment_data: Any,
    outcome_data: Any,
    covariates_data: Any | None = None,
    observed_effect: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    include_rosenbaum: bool = True,
    include_evalue: bool = True,
    include_oster: bool = False,
    include_negative_controls: bool = False,
    include_placebo: bool = False,
    negative_control_outcome: Any | None = None,
    negative_control_exposure: Any | None = None,
    covariates_restricted: Any | None = None,
    output_format: str = "html",
    output_path: str | None = None,
    report_title: str = "Sensitivity Analysis Report",
    analyst_name: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate comprehensive sensitivity analysis report.

    This function runs multiple sensitivity analyses and generates a unified
    report with interpretations suitable for analysts without deep causal
    inference background.

    Args:
        treatment_data: Treatment variable data
        outcome_data: Outcome variable data
        covariates_data: Covariate data for adjustment
        observed_effect: Observed treatment effect estimate
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        include_rosenbaum: Whether to include Rosenbaum bounds analysis
        include_evalue: Whether to include E-value analysis
        include_oster: Whether to include Oster delta analysis
        include_negative_controls: Whether to include negative control tests
        include_placebo: Whether to include placebo tests
        negative_control_outcome: Negative control outcome for testing
        negative_control_exposure: Negative control exposure for testing
        covariates_restricted: Restricted covariates for Oster analysis
        output_format: Output format ('html', 'pdf', 'dict')
        output_path: Path to save report file
        report_title: Title for the report
        analyst_name: Name of analyst conducting analysis
        **kwargs: Additional arguments for individual sensitivity methods

    Returns:
        Dictionary containing:
            - report_html: HTML report string
            - summary_results: Summary of all analyses
            - individual_results: Detailed results from each method
            - recommendations: List of recommendations
            - overall_assessment: Overall robustness assessment
            - file_path: Path to saved report file (if output_path provided)

    Example:
        >>> import numpy as np
        >>> from causal_inference.sensitivity import generate_sensitivity_report
        >>>
        >>> # Simulate data
        >>> n = 1000
        >>> T = np.random.binomial(1, 0.5, n)
        >>> Y = 2 * T + np.random.normal(0, 1, n)
        >>> X = np.random.normal(0, 1, (n, 2))
        >>>
        >>> # Generate comprehensive report
        >>> report = generate_sensitivity_report(
        >>>     treatment_data=T,
        >>>     outcome_data=Y,
        >>>     covariates_data=X,
        >>>     observed_effect=2.0,
        >>>     ci_lower=1.8,
        >>>     ci_upper=2.2,
        >>>     output_path="sensitivity_report.html"
        >>> )
        >>> print(f"Overall assessment: {report['overall_assessment']}")
    """
    # Initialize results storage
    results = {}
    all_recommendations = []
    analysis_summary = []

    # Run E-value analysis
    if include_evalue and observed_effect is not None:
        try:
            evalue_results = e_value(
                observed_estimate=observed_effect,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                **kwargs.get("evalue_kwargs", {}),
            )
            results["evalue"] = evalue_results
            analysis_summary.append(
                {
                    "method": "E-value",
                    "result": f"{evalue_results['evalue_point']:.2f}",
                    "interpretation": evalue_results["interpretation"],
                }
            )

            if evalue_results["evalue_point"] >= 2.0:
                all_recommendations.append(
                    "✅ E-value analysis suggests good robustness to unmeasured confounding"
                )
            else:
                all_recommendations.append(
                    "⚠️ E-value analysis suggests limited robustness - consider additional controls"
                )

        except Exception as e:
            results["evalue"] = {"error": str(e)}
            all_recommendations.append(
                "❌ E-value analysis failed - check input parameters"
            )

    # Run Rosenbaum bounds analysis
    if include_rosenbaum:
        try:
            # Extract treated and control outcomes
            treatment_array = np.asarray(treatment_data).flatten()
            outcome_array = np.asarray(outcome_data).flatten()

            treated_outcomes = outcome_array[treatment_array == 1]
            control_outcomes = outcome_array[treatment_array == 0]

            if len(treated_outcomes) >= 5 and len(control_outcomes) >= 5:
                # Use random matching for unmatched data
                min_n = min(len(treated_outcomes), len(control_outcomes))
                np.random.seed(42)
                treated_sample = np.random.choice(
                    treated_outcomes, min_n, replace=False
                )
                control_sample = np.random.choice(
                    control_outcomes, min_n, replace=False
                )

                rosenbaum_results = rosenbaum_bounds(
                    treated_sample, control_sample, **kwargs.get("rosenbaum_kwargs", {})
                )
                results["rosenbaum"] = rosenbaum_results

                critical_gamma = rosenbaum_results.get("critical_gamma")
                analysis_summary.append(
                    {
                        "method": "Rosenbaum Bounds",
                        "result": f"Γ_critical = {critical_gamma:.2f}"
                        if critical_gamma
                        else "Robust",
                        "interpretation": rosenbaum_results["robustness_assessment"],
                    }
                )

                if critical_gamma is None or critical_gamma >= 2.0:
                    all_recommendations.append(
                        "✅ Rosenbaum bounds suggest strong robustness to hidden bias"
                    )
                elif critical_gamma >= 1.5:
                    all_recommendations.append(
                        "⚠️ Rosenbaum bounds suggest moderate robustness"
                    )
                else:
                    all_recommendations.append(
                        "❌ Rosenbaum bounds suggest low robustness to hidden bias"
                    )
            else:
                results["rosenbaum"] = {
                    "error": "Insufficient data for Rosenbaum bounds"
                }
                all_recommendations.append(
                    "❌ Rosenbaum bounds analysis requires more data"
                )

        except Exception as e:
            results["rosenbaum"] = {"error": str(e)}
            all_recommendations.append("❌ Rosenbaum bounds analysis failed")

    # Run Oster delta analysis
    if include_oster and covariates_restricted is not None:
        try:
            oster_results = oster_delta(
                outcome=outcome_data,
                treatment=treatment_data,
                covariates_restricted=covariates_restricted,
                covariates_full=covariates_data,
                **kwargs.get("oster_kwargs", {}),
            )
            results["oster"] = oster_results

            analysis_summary.append(
                {
                    "method": "Oster δ Analysis",
                    "result": f"β* = {oster_results['beta_star']:.3f}",
                    "interpretation": oster_results["interpretation"],
                }
            )

            if oster_results["passes_robustness_test"]:
                all_recommendations.append(
                    "✅ Oster analysis suggests results are robust to omitted variable bias"
                )
            else:
                all_recommendations.append(
                    "⚠️ Oster analysis suggests sensitivity to omitted variables"
                )

        except Exception as e:
            results["oster"] = {"error": str(e)}
            all_recommendations.append("❌ Oster delta analysis failed")

    # Run negative control analysis
    if include_negative_controls:
        try:
            negative_results = negative_control(
                treatment=treatment_data,
                outcome=outcome_data,
                negative_control_outcome=negative_control_outcome,
                negative_control_exposure=negative_control_exposure,
                covariates=covariates_data,
                **kwargs.get("negative_control_kwargs", {}),
            )
            results["negative_control"] = negative_results

            analysis_summary.append(
                {
                    "method": "Negative Controls",
                    "result": f"{negative_results['n_violations']} violations",
                    "interpretation": negative_results["overall_assessment"],
                }
            )

            if negative_results["n_violations"] == 0:
                all_recommendations.append(
                    "✅ Negative control analysis supports assumption validity"
                )
            else:
                all_recommendations.append(
                    "❌ Negative control analysis suggests assumption violations"
                )

        except Exception as e:
            results["negative_control"] = {"error": str(e)}
            all_recommendations.append("❌ Negative control analysis failed")

    # Run placebo tests
    if include_placebo:
        try:
            placebo_results = placebo_test(
                treatment=treatment_data,
                outcome=outcome_data,
                covariates=covariates_data,
                **kwargs.get("placebo_kwargs", {}),
            )
            results["placebo"] = placebo_results

            analysis_summary.append(
                {
                    "method": "Placebo Tests",
                    "result": f"FPR = {placebo_results['false_positive_rate']:.1%}",
                    "interpretation": "Passes"
                    if placebo_results["passes_placebo_test"]
                    else "Fails",
                }
            )

            if placebo_results["passes_placebo_test"]:
                all_recommendations.append(
                    "✅ Placebo tests support identification assumptions"
                )
            else:
                all_recommendations.append(
                    "❌ Placebo tests suggest specification issues"
                )

        except Exception as e:
            results["placebo"] = {"error": str(e)}
            all_recommendations.append("❌ Placebo test analysis failed")

    # Generate overall assessment
    n_methods = len(
        [
            x
            for x in [
                include_evalue,
                include_rosenbaum,
                include_oster,
                include_negative_controls,
                include_placebo,
            ]
            if x
        ]
    )
    positive_assessments = len([r for r in all_recommendations if r.startswith("✅")])
    len([r for r in all_recommendations if r.startswith("⚠️")])

    if positive_assessments >= 0.8 * n_methods:
        overall_assessment = "HIGH ROBUSTNESS - Results appear robust across multiple sensitivity analyses"
    elif positive_assessments >= 0.5 * n_methods:
        overall_assessment = (
            "MODERATE ROBUSTNESS - Mixed results suggest cautious interpretation"
        )
    else:
        overall_assessment = (
            "LOW ROBUSTNESS - Multiple analyses suggest sensitivity to assumptions"
        )

    # Generate HTML report
    html_report = _generate_html_report(
        results=results,
        analysis_summary=analysis_summary,
        recommendations=all_recommendations,
        overall_assessment=overall_assessment,
        report_title=report_title,
        analyst_name=analyst_name,
        treatment_data=treatment_data,
        outcome_data=outcome_data,
    )

    # Save report if path provided
    file_path = None
    if output_path:
        file_path = Path(output_path)

        if output_format.lower() == "html":
            file_path = file_path.with_suffix(".html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_report)
        elif output_format.lower() == "pdf":
            # Note: PDF generation would require additional dependencies like weasyprint
            raise NotImplementedError("PDF output requires additional dependencies")

    return {
        "report_html": html_report,
        "summary_results": analysis_summary,
        "individual_results": results,
        "recommendations": all_recommendations,
        "overall_assessment": overall_assessment,
        "file_path": str(file_path) if file_path else None,
    }


def _generate_html_report(
    results: dict[str, Any],
    analysis_summary: list[dict[str, Any]],
    recommendations: list[str],
    overall_assessment: str,
    report_title: str,
    analyst_name: str | None,
    treatment_data: Any,
    outcome_data: Any,
) -> str:
    """Generate HTML sensitivity analysis report."""

    # Generate summary visualizations
    viz_html = _generate_visualizations(results, treatment_data, outcome_data)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{report_title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .assessment {{
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                font-weight: bold;
                font-size: 18px;
            }}
            .high-robustness {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
            .moderate-robustness {{ background-color: #fff3cd; color: #856404; border: 1px solid #faeeba; }}
            .low-robustness {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .summary-table th {{
                background-color: #f8f9fa;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}
            .summary-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #e9ecef;
            }}
            .summary-table tr:hover {{ background-color: #f5f5f5; }}
            .recommendations {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 5px solid #007bff;
            }}
            .recommendations ul {{ list-style-type: none; padding: 0; }}
            .recommendations li {{
                margin: 8px 0;
                padding: 8px 12px;
                border-radius: 5px;
                background: white;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .section h2 {{
                color: #495057;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 10px;
            }}
            .error {{ color: #dc3545; font-style: italic; }}
            .visualization {{
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .footer {{
                text-align: center;
                color: #6c757d;
                border-top: 1px solid #e9ecef;
                padding-top: 20px;
                margin-top: 40px;
            }}
            .metadata {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                font-size: 14px;
                color: #6c757d;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{report_title}</h1>
            <p>Comprehensive sensitivity analysis for causal inference robustness assessment</p>
            <div class="metadata">
                {"<strong>Analyst:</strong> " + analyst_name + "<br>" if analyst_name else ""}
                <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
                <strong>Sample Size:</strong> {len(treatment_data)} observations
            </div>
        </div>

        <div class="assessment {"high-robustness" if "HIGH" in overall_assessment else "moderate-robustness" if "MODERATE" in overall_assessment else "low-robustness"}">
            🎯 Overall Assessment: {overall_assessment}
        </div>

        <div class="section">
            <h2>📊 Analysis Summary</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>Result</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([f"<tr><td><strong>{item['method']}</strong></td><td>{item['result']}</td><td>{item['interpretation']}</td></tr>" for item in analysis_summary])}
                </tbody>
            </table>
        </div>

        {viz_html}

        <div class="recommendations">
            <h2>💡 Recommendations</h2>
            <ul>
                {"".join([f"<li>{rec}</li>" for rec in recommendations])}
            </ul>
        </div>

        <div class="section">
            <h2>📋 Detailed Results</h2>
            {_generate_detailed_results_html(results)}
        </div>

        <div class="section">
            <h2>ℹ️ Interpretation Guide</h2>
            <p><strong>E-values:</strong> Higher values indicate greater robustness. E-values ≥ 2.0 suggest good robustness to unmeasured confounding.</p>
            <p><strong>Rosenbaum Bounds:</strong> Critical Γ values ≥ 1.5 suggest reasonable robustness to hidden bias in matched studies.</p>
            <p><strong>Oster δ:</strong> Robustness ratios ≥ 0.5 suggest results are not overly sensitive to omitted variables.</p>
            <p><strong>Negative Controls:</strong> Should show no significant effects if assumptions hold.</p>
            <p><strong>Placebo Tests:</strong> False positive rates should be ≤ 10% for valid identification.</p>
        </div>

        <div class="footer">
            <p>This report was generated using the Causal Inference Tools sensitivity analysis suite.<br>
            For questions about interpretation, consult a causal inference expert.</p>
        </div>
    </body>
    </html>
    """

    return html_template


def _generate_detailed_results_html(results: dict[str, Any]) -> str:
    """Generate detailed results section of HTML report."""
    detailed_html = ""

    for method, result in results.items():
        if isinstance(result, dict) and "error" in result:
            detailed_html += f"""
            <div class="error">
                <h3>{method.replace("_", " ").title()}</h3>
                <p>Analysis failed: {result["error"]}</p>
            </div>
            """
        else:
            method_name = method.replace("_", " ").title()
            detailed_html += f"<h3>{method_name}</h3>"

            if method == "evalue" and isinstance(result, dict):
                detailed_html += f"""
                <ul>
                    <li><strong>Point E-value:</strong> {result.get("evalue_point", "N/A"):.2f}</li>
                    <li><strong>CI E-value:</strong> {result.get("evalue_ci", "N/A")}</li>
                    <li><strong>Interpretation:</strong> {result.get("interpretation", "N/A")}</li>
                </ul>
                """
            elif method == "rosenbaum" and isinstance(result, dict):
                detailed_html += f"""
                <ul>
                    <li><strong>Critical Gamma:</strong> {result.get("critical_gamma", "Robust across range")}</li>
                    <li><strong>Original p-value:</strong> {result.get("original_p_value", "N/A"):.4f}</li>
                    <li><strong>Assessment:</strong> {result.get("robustness_assessment", "N/A")}</li>
                </ul>
                """
            elif method == "oster" and isinstance(result, dict):
                detailed_html += f"""
                <ul>
                    <li><strong>Bias-adjusted coefficient (β*):</strong> {result.get("beta_star", "N/A"):.3f}</li>
                    <li><strong>Robustness ratio:</strong> {result.get("robustness_ratio", "N/A"):.3f}</li>
                    <li><strong>Passes robustness test:</strong> {result.get("passes_robustness_test", "N/A")}</li>
                </ul>
                """
            elif method == "negative_control" and isinstance(result, dict):
                detailed_html += f"""
                <ul>
                    <li><strong>Assumption violations:</strong> {result.get("n_violations", "N/A")}</li>
                    <li><strong>Assessment:</strong> {result.get("overall_assessment", "N/A")}</li>
                    <li><strong>Bias indicators:</strong> {", ".join(result.get("bias_indicators", []))}</li>
                </ul>
                """
            elif method == "placebo" and isinstance(result, dict):
                detailed_html += f"""
                <ul>
                    <li><strong>False positive rate:</strong> {result.get("false_positive_rate", "N/A"):.1%}</li>
                    <li><strong>Mean placebo effect:</strong> {result.get("mean_placebo_effect", "N/A"):.3f}</li>
                    <li><strong>Passes test:</strong> {result.get("passes_placebo_test", "N/A")}</li>
                </ul>
                """

    return detailed_html


def _generate_visualizations(
    results: dict[str, Any],
    treatment_data: Any,
    outcome_data: Any,
) -> str:
    """Generate visualization section for HTML report."""
    # This is a simplified version - full implementation would create actual plots
    viz_html = """
    <div class="section">
        <h2>📈 Visualizations</h2>
        <div class="visualization">
            <p><em>Visualization plots would appear here in a full implementation.</em></p>
            <p>Potential plots include:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>E-value sensitivity curve</li>
                <li>Rosenbaum bounds plot</li>
                <li>Oster delta surface</li>
                <li>Placebo effect distribution</li>
                <li>Treatment/outcome distributions</li>
            </ul>
        </div>
    </div>
    """

    return viz_html
