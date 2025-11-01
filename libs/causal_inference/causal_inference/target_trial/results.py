"""Target trial emulation results and reporting.

This module implements classes for storing and reporting results from target trial emulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ..core.base import CausalEffect


@dataclass
class EmulationDiagnostics:
    """Diagnostic information from target trial emulation."""

    # Sample sizes and eligibility
    total_sample_size: int
    eligible_sample_size: int
    final_analysis_sample_size: int
    eligibility_rate: float

    # Treatment group information
    treatment_group_sizes: dict[str, int]
    treatment_group_rates: dict[str, float]

    # Adherence and censoring
    adherence_rates: dict[str, float]
    censoring_rates: dict[str, float]
    lost_to_followup_rate: float

    # Grace period analysis (if applicable)
    grace_period_compliance_rate: Optional[float] = None
    subjects_treated_in_grace: Optional[int] = None

    # Balance and overlap diagnostics
    covariate_balance: Optional[dict[str, float]] = None
    propensity_score_overlap: Optional[dict[str, Any]] = None

    # Emulation quality metrics
    emulation_quality_score: Optional[float] = None
    protocol_adherence_score: Optional[float] = None


@dataclass
class TargetTrialResults:
    """Results from target trial emulation analysis."""

    # Core causal effects
    intention_to_treat_effect: CausalEffect
    per_protocol_effect: Optional[CausalEffect] = None

    # Protocol and emulation details
    protocol_summary: str = ""
    emulation_method: str = ""
    estimation_method: str = ""

    # Diagnostic information
    diagnostics: Optional[EmulationDiagnostics] = None

    # Grace period analysis
    grace_period_analysis: Optional[dict[str, Any]] = None

    # Sensitivity analyses
    sensitivity_analyses: Optional[dict[str, CausalEffect]] = None

    # Raw emulation data (for further analysis)
    emulated_data: Optional[pd.DataFrame] = None
    cloned_data: Optional[pd.DataFrame] = None

    def generate_report(
        self,
        include_protocol: bool = True,
        include_feasibility: bool = True,
        include_balance_diagnostics: bool = True,
        include_sensitivity_analysis: bool = False,
    ) -> EmulationReport:
        """Generate comprehensive emulation report.

        Args:
            include_protocol: Include protocol specification
            include_feasibility: Include feasibility assessment
            include_balance_diagnostics: Include covariate balance diagnostics
            include_sensitivity_analysis: Include sensitivity analysis results

        Returns:
            EmulationReport with formatted results
        """
        sections = {}

        # Protocol specification
        if include_protocol and self.protocol_summary:
            sections["protocol_specification"] = self.protocol_summary

        # Feasibility assessment
        if include_feasibility and self.diagnostics:
            sections["feasibility_assessment"] = self._format_feasibility_section()

        # Main results
        sections["emulation_results"] = self._format_results_section()

        # Balance diagnostics
        if (
            include_balance_diagnostics
            and self.diagnostics
            and self.diagnostics.covariate_balance
        ):
            sections["balance_diagnostics"] = self._format_balance_section()

        # Sensitivity analysis
        if include_sensitivity_analysis and self.sensitivity_analyses:
            sections["sensitivity_analysis"] = self._format_sensitivity_section()

        # Interpretation guidelines
        sections["interpretation_guidelines"] = self._format_interpretation_section()

        return EmulationReport(
            sections=sections,
            intention_to_treat_effect=self.intention_to_treat_effect,
            per_protocol_effect=self.per_protocol_effect,
            diagnostics=self.diagnostics,
        )

    def _format_feasibility_section(self) -> str:
        """Format feasibility assessment section."""
        if not self.diagnostics:
            return "No diagnostic information available."

        lines = [
            "Feasibility Assessment",
            "-" * 20,
            f"Total sample size: {self.diagnostics.total_sample_size:,}",
            f"Eligible participants: {self.diagnostics.eligible_sample_size:,} ({self.diagnostics.eligibility_rate:.1%})",
            f"Final analysis sample: {self.diagnostics.final_analysis_sample_size:,}",
            "",
            "Treatment Group Sizes:",
        ]

        for group, size in self.diagnostics.treatment_group_sizes.items():
            rate = self.diagnostics.treatment_group_rates.get(group, 0)
            lines.append(f"  {group}: {size:,} ({rate:.1%})")

        return "\n".join(lines)

    def _format_results_section(self) -> str:
        """Format main results section."""
        lines = [
            "Emulation Results",
            "-" * 16,
            f"Estimation method: {self.estimation_method}",
            "",
            "Intention-to-Treat Analysis:",
            f"  ATE: {self.intention_to_treat_effect.ate:.3f}",
        ]

        if self.intention_to_treat_effect.ate_ci_lower is not None:
            lines.append(
                f"  95% CI: [{self.intention_to_treat_effect.ate_ci_lower:.3f}, "
                f"{self.intention_to_treat_effect.ate_ci_upper:.3f}]"
            )
            lines.append(
                f"  Significant: {self.intention_to_treat_effect.is_significant}"
            )

        if self.per_protocol_effect:
            lines.extend(
                [
                    "",
                    "Per-Protocol Analysis:",
                    f"  ATE: {self.per_protocol_effect.ate:.3f}",
                ]
            )

            if self.per_protocol_effect.ate_ci_lower is not None:
                lines.append(
                    f"  95% CI: [{self.per_protocol_effect.ate_ci_lower:.3f}, "
                    f"{self.per_protocol_effect.ate_ci_upper:.3f}]"
                )
                lines.append(
                    f"  Significant: {self.per_protocol_effect.is_significant}"
                )

        return "\n".join(lines)

    def _format_balance_section(self) -> str:
        """Format covariate balance section."""
        if not self.diagnostics or not self.diagnostics.covariate_balance:
            return "No balance diagnostics available."

        lines = ["Covariate Balance", "-" * 17, "Standardized Mean Differences:"]

        for covariate, smd in self.diagnostics.covariate_balance.items():
            balance_status = "✓" if abs(smd) < 0.1 else "⚠" if abs(smd) < 0.25 else "✗"
            lines.append(f"  {covariate}: {smd:.3f} {balance_status}")

        return "\n".join(lines)

    def _format_sensitivity_section(self) -> str:
        """Format sensitivity analysis section."""
        if not self.sensitivity_analyses:
            return "No sensitivity analyses performed."

        lines = ["Sensitivity Analysis", "-" * 19]

        for analysis_name, effect in self.sensitivity_analyses.items():
            lines.extend(
                [
                    f"{analysis_name}:",
                    f"  ATE: {effect.ate:.3f}",
                ]
            )

            if effect.ate_ci_lower is not None:
                lines.append(
                    f"  95% CI: [{effect.ate_ci_lower:.3f}, {effect.ate_ci_upper:.3f}]"
                )

        return "\n".join(lines)

    def _format_interpretation_section(self) -> str:
        """Format interpretation guidelines section."""
        lines = [
            "Interpretation Guidelines",
            "-" * 25,
            "",
            "Key Considerations:",
            "• ITT effect represents the policy-relevant impact of implementing the intervention",
            "• Per-protocol effect represents the biological efficacy among those who adhere",
            "• Confidence intervals account for sampling uncertainty but not unmeasured confounding",
            "• Results assume all measured confounders were adequately controlled",
            "",
            "Limitations:",
            "• Target trial emulation cannot fully eliminate selection bias",
            "• Grace period choices may affect results",
            "• Adherence definitions impact per-protocol estimates",
        ]

        # Add method-specific considerations
        if "g_computation" in self.estimation_method.lower():
            lines.extend(
                [
                    "",
                    "G-computation specific considerations:",
                    "• Results rely on correct specification of outcome models",
                    "• May be sensitive to model misspecification",
                    "• Extrapolation beyond observed covariate patterns",
                ]
            )
        elif "ipw" in self.estimation_method.lower():
            lines.extend(
                [
                    "",
                    "IPW specific considerations:",
                    "• Results rely on correct specification of treatment models",
                    "• Sensitive to extreme propensity score weights",
                    "• May be unstable with strong confounding",
                ]
            )

        return "\n".join(lines)

    def compare_itt_vs_pp(self) -> dict[str, Any]:
        """Compare intention-to-treat vs per-protocol effects.

        Returns:
            Dictionary with comparison results
        """
        if not self.per_protocol_effect:
            return {"error": "Per-protocol effect not available"}

        comparison: dict[str, Any] = {
            "itt_ate": self.intention_to_treat_effect.ate,
            "pp_ate": self.per_protocol_effect.ate,
            "difference": self.per_protocol_effect.ate
            - self.intention_to_treat_effect.ate,
            "ratio": self.per_protocol_effect.ate / self.intention_to_treat_effect.ate
            if self.intention_to_treat_effect.ate != 0
            else None,
        }

        # Add confidence interval comparison if available
        if (
            self.intention_to_treat_effect.ate_ci_lower is not None
            and self.intention_to_treat_effect.ate_ci_upper is not None
            and self.per_protocol_effect.ate_ci_lower is not None
            and self.per_protocol_effect.ate_ci_upper is not None
        ):
            itt_width = (
                self.intention_to_treat_effect.ate_ci_upper
                - self.intention_to_treat_effect.ate_ci_lower
            )
            pp_width = (
                self.per_protocol_effect.ate_ci_upper
                - self.per_protocol_effect.ate_ci_lower
            )

            comparison.update(
                {
                    "itt_ci_width": itt_width,
                    "pp_ci_width": pp_width,
                    "ci_width_ratio": pp_width / itt_width if itt_width != 0 else None,
                }
            )

        # Interpretation
        difference = comparison["difference"]
        if difference is not None:
            if abs(difference) < 0.1:
                comparison["interpretation"] = (
                    "ITT and per-protocol effects are similar"
                )
            elif difference > 0:
                comparison["interpretation"] = (
                    "Per-protocol effect is larger than ITT effect"
                )
            else:
                comparison["interpretation"] = (
                    "Per-protocol effect is smaller than ITT effect"
                )
        else:
            comparison["interpretation"] = "Cannot compare effects due to missing data"

        return comparison


class EmulationReport:
    """Formatted report from target trial emulation."""

    def __init__(
        self,
        sections: dict[str, str],
        intention_to_treat_effect: CausalEffect,
        per_protocol_effect: Optional[CausalEffect] = None,
        diagnostics: Optional[EmulationDiagnostics] = None,
    ):
        """Initialize emulation report.

        Args:
            sections: Dictionary of report sections
            intention_to_treat_effect: ITT causal effect
            per_protocol_effect: Optional per-protocol effect
            diagnostics: Optional diagnostic information
        """
        self.sections = sections
        self.intention_to_treat_effect = intention_to_treat_effect
        self.per_protocol_effect = per_protocol_effect
        self.diagnostics = diagnostics

    def to_string(self) -> str:
        """Convert report to formatted string.

        Returns:
            Formatted report string
        """
        lines = ["TARGET TRIAL EMULATION REPORT", "=" * 40, ""]

        for section_name, content in self.sections.items():
            lines.extend([content, ""])

        # Add timestamp
        from datetime import datetime

        lines.extend(
            ["-" * 40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        )

        return "\n".join(lines)

    def to_html(self) -> str:
        """Convert report to HTML format.

        Returns:
            HTML formatted report
        """
        html_lines = [
            "<html><head><title>Target Trial Emulation Report</title></head><body>",
            "<h1>Target Trial Emulation Report</h1>",
        ]

        for section_name, content in self.sections.items():
            section_title = section_name.replace("_", " ").title()
            html_lines.extend([f"<h2>{section_title}</h2>", f"<pre>{content}</pre>"])

        html_lines.append("</body></html>")
        return "\n".join(html_lines)

    def save(self, filename: str, format: str = "txt") -> None:
        """Save report to file.

        Args:
            filename: Output filename
            format: Output format ("txt" or "html")
        """
        if format == "html":
            content = self.to_html()
        else:
            content = self.to_string()

        with open(filename, "w") as f:
            f.write(content)
