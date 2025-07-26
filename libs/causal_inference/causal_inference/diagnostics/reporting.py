"""Automated diagnostic reporting for causal inference.

This module provides tools to generate comprehensive diagnostic reports
that summarize all assumption checks and provide actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..core.base import CausalEffect, CovariateData, OutcomeData, TreatmentData
from .assumptions import AssumptionChecker, AssumptionResults
from .balance import BalanceDiagnostics, BalanceResults
from .overlap import OverlapDiagnostics, OverlapResults
from .sensitivity import SensitivityAnalysis, SensitivityResults
from .specification import ModelSpecificationTests, SpecificationResults


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report for causal inference analysis."""

    # Metadata
    timestamp: str
    sample_size: int
    treatment_type: str
    outcome_type: str

    # Diagnostic results
    balance_results: BalanceResults | None = None
    overlap_results: OverlapResults | None = None
    assumption_results: AssumptionResults | None = None
    specification_results: SpecificationResults | None = None
    sensitivity_results: SensitivityResults | None = None

    # Summary assessments
    overall_assessment: str = ""
    critical_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class DiagnosticReportGenerator:
    """Generator for comprehensive diagnostic reports."""

    def __init__(
        self,
        include_balance: bool = True,
        include_overlap: bool = True,
        include_assumptions: bool = True,
        include_specification: bool = True,
        include_sensitivity: bool = False,  # Requires causal effect estimate
    ):
        """Initialize diagnostic report generator.

        Args:
            include_balance: Whether to include balance diagnostics
            include_overlap: Whether to include overlap diagnostics
            include_assumptions: Whether to include assumption checking
            include_specification: Whether to include specification tests
            include_sensitivity: Whether to include sensitivity analysis
        """
        self.include_balance = include_balance
        self.include_overlap = include_overlap
        self.include_assumptions = include_assumptions
        self.include_specification = include_specification
        self.include_sensitivity = include_sensitivity

    def generate_comprehensive_report(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData,
        causal_effect: CausalEffect | None = None,
        verbose: bool = True,
    ) -> DiagnosticReport:
        """Generate comprehensive diagnostic report.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            covariates: Covariate data
            causal_effect: Optional causal effect estimate for sensitivity analysis
            verbose: Whether to print progress

        Returns:
            DiagnosticReport with comprehensive assessment
        """
        if verbose:
            print("Generating comprehensive diagnostic report...")
            print("=" * 50)

        # Initialize report
        report = DiagnosticReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sample_size=len(treatment.values),
            treatment_type=treatment.treatment_type,
            outcome_type=outcome.outcome_type,
        )

        # Balance diagnostics
        if self.include_balance:
            if verbose:
                print("1. Running covariate balance diagnostics...")
            balance_checker = BalanceDiagnostics()
            report.balance_results = balance_checker.assess_balance(
                treatment, covariates
            )

            if not report.balance_results.overall_balance_met:
                report.critical_issues.append("Covariate imbalance detected")

        # Overlap diagnostics
        if self.include_overlap:
            if verbose:
                print("2. Running overlap and positivity diagnostics...")
            overlap_checker = OverlapDiagnostics()
            report.overlap_results = overlap_checker.assess_overlap(
                treatment, covariates
            )

            if not report.overlap_results.overall_positivity_met:
                report.critical_issues.append("Positivity assumption violated")

        # Assumption checking
        if self.include_assumptions:
            if verbose:
                print("3. Running assumption checks...")
            assumption_checker = AssumptionChecker()
            report.assumption_results = assumption_checker.check_all_assumptions(
                treatment, outcome, covariates
            )

            if not report.assumption_results.exchangeability_likely:
                report.critical_issues.append("Exchangeability assumption questionable")

        # Specification tests
        if self.include_specification:
            if verbose:
                print("4. Running model specification tests...")
            spec_tester = ModelSpecificationTests()
            report.specification_results = (
                spec_tester.comprehensive_specification_tests(
                    outcome, treatment, covariates
                )
            )

            if not report.specification_results.specification_passed:
                report.critical_issues.append("Model specification issues detected")

        # Sensitivity analysis
        if self.include_sensitivity and causal_effect is not None:
            if verbose:
                print("5. Running sensitivity analysis...")
            sensitivity_analyzer = SensitivityAnalysis()
            report.sensitivity_results = (
                sensitivity_analyzer.comprehensive_sensitivity_analysis(
                    treatment, outcome, causal_effect, covariates
                )
            )

        # Generate overall assessment and recommendations
        report.overall_assessment = self._generate_overall_assessment(report)
        report.recommendations = self._generate_comprehensive_recommendations(report)

        if verbose:
            print("✅ Diagnostic report generation complete!")
            print()

        return report

    def _generate_overall_assessment(self, report: DiagnosticReport) -> str:
        """Generate overall assessment based on all diagnostic results."""
        if len(report.critical_issues) == 0:
            return "✅ EXCELLENT - All diagnostic checks passed. Causal analysis appears robust."
        elif len(report.critical_issues) == 1:
            return "⚠️ GOOD - Minor issues detected. Causal analysis likely robust with caveats."
        elif len(report.critical_issues) <= 2:
            return "⚠️ FAIR - Multiple issues detected. Proceed with caution and address concerns."
        else:
            return "❌ POOR - Serious issues detected. Causal analysis may not be reliable."

    def _generate_comprehensive_recommendations(
        self, report: DiagnosticReport
    ) -> list[str]:
        """Generate comprehensive recommendations based on all results."""
        recommendations = []

        # Balance recommendations
        if report.balance_results and not report.balance_results.overall_balance_met:
            recommendations.append(
                "Address covariate imbalance through matching, weighting, or stratification"
            )
            if len(report.balance_results.imbalanced_covariates) > 3:
                recommendations.append(
                    "Consider propensity score methods due to multiple imbalanced covariates"
                )

        # Overlap recommendations
        if report.overlap_results and not report.overlap_results.overall_positivity_met:
            recommendations.extend(report.overlap_results.recommendation.split(". "))

        # Assumption recommendations
        if report.assumption_results:
            recommendations.extend(
                report.assumption_results.recommendations[:3]
            )  # Top 3

        # Specification recommendations
        if (
            report.specification_results
            and not report.specification_results.specification_passed
        ):
            recommendations.extend(
                report.specification_results.recommendations[:2]
            )  # Top 2

        # Sensitivity recommendations
        if report.sensitivity_results:
            recommendations.extend(
                report.sensitivity_results.recommendations[:2]
            )  # Top 2

        # General recommendations based on issues
        if len(report.critical_issues) > 2:
            recommendations.append(
                "Consider alternative identification strategies (IV, RDD, etc.)"
            )
            recommendations.append("Conduct robustness checks with multiple estimators")

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen and rec.strip():
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:8]  # Limit to top 8 recommendations

    def print_diagnostic_report(self, report: DiagnosticReport) -> None:
        """Print comprehensive diagnostic report."""
        print("=" * 60)
        print("           CAUSAL INFERENCE DIAGNOSTIC REPORT")
        print("=" * 60)
        print(f"Generated: {report.timestamp}")
        print(f"Sample size: {report.sample_size:,}")
        print(f"Treatment type: {report.treatment_type}")
        print(f"Outcome type: {report.outcome_type}")
        print()

        # Overall assessment
        print("OVERALL ASSESSMENT")
        print("-" * 20)
        print(report.overall_assessment)
        print()

        # Critical issues
        if report.critical_issues:
            print("CRITICAL ISSUES")
            print("-" * 15)
            for i, issue in enumerate(report.critical_issues, 1):
                print(f"  {i}. ❌ {issue}")
            print()

        # Detailed results
        print("DETAILED DIAGNOSTIC RESULTS")
        print("-" * 30)

        # Balance
        if report.balance_results:
            status = (
                "✅ PASSED"
                if report.balance_results.overall_balance_met
                else "❌ FAILED"
            )
            print(f"Covariate Balance: {status}")
            if not report.balance_results.overall_balance_met:
                print(
                    f"  Imbalanced variables: {len(report.balance_results.imbalanced_covariates)}"
                )

        # Overlap
        if report.overlap_results:
            status = (
                "✅ PASSED"
                if report.overlap_results.overall_positivity_met
                else "❌ FAILED"
            )
            print(f"Positivity/Overlap: {status}")
            print(
                f"  Propensity range: [{report.overlap_results.min_propensity_score:.3f}, {report.overlap_results.max_propensity_score:.3f}]"
            )

        # Assumptions
        if report.assumption_results:
            status = (
                "✅ LIKELY"
                if report.assumption_results.exchangeability_likely
                else "⚠️ QUESTIONABLE"
            )
            print(f"Exchangeability: {status}")
            if report.assumption_results.confounding_detected:
                print(
                    f"  Confounders detected: {len(report.assumption_results.confounding_strength)}"
                )

        # Specification
        if report.specification_results:
            status = (
                "✅ PASSED"
                if report.specification_results.specification_passed
                else "⚠️ ISSUES"
            )
            print(f"Model Specification: {status}")
            if report.specification_results.problematic_variables:
                print(
                    f"  Problematic variables: {len(report.specification_results.problematic_variables)}"
                )

        # Sensitivity
        if report.sensitivity_results:
            print("Sensitivity Analysis:")
            print(f"  E-value: {report.sensitivity_results.evalue:.2f}")
            print(f"  Robustness: {report.sensitivity_results.robustness_assessment}")

        print()

        # Recommendations
        print("RECOMMENDATIONS")
        print("-" * 15)
        if report.recommendations:
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  No specific recommendations - analysis appears robust.")

        print()
        print("=" * 60)

    def export_report_to_dataframe(self, report: DiagnosticReport) -> pd.DataFrame:
        """Export diagnostic results to DataFrame for further analysis."""
        data = []

        # Basic information
        data.append(
            {
                "Category": "Metadata",
                "Metric": "Sample Size",
                "Value": report.sample_size,
                "Status": "Info",
            }
        )
        data.append(
            {
                "Category": "Metadata",
                "Metric": "Treatment Type",
                "Value": report.treatment_type,
                "Status": "Info",
            }
        )
        data.append(
            {
                "Category": "Metadata",
                "Metric": "Outcome Type",
                "Value": report.outcome_type,
                "Status": "Info",
            }
        )

        # Balance results
        if report.balance_results:
            data.append(
                {
                    "Category": "Balance",
                    "Metric": "Overall Balance",
                    "Value": "Met"
                    if report.balance_results.overall_balance_met
                    else "Not Met",
                    "Status": "Pass"
                    if report.balance_results.overall_balance_met
                    else "Fail",
                }
            )
            data.append(
                {
                    "Category": "Balance",
                    "Metric": "Imbalanced Variables",
                    "Value": len(report.balance_results.imbalanced_covariates),
                    "Status": "Pass"
                    if len(report.balance_results.imbalanced_covariates) == 0
                    else "Fail",
                }
            )

        # Overlap results
        if report.overlap_results:
            data.append(
                {
                    "Category": "Overlap",
                    "Metric": "Positivity",
                    "Value": "Met"
                    if report.overlap_results.overall_positivity_met
                    else "Violated",
                    "Status": "Pass"
                    if report.overlap_results.overall_positivity_met
                    else "Fail",
                }
            )
            data.append(
                {
                    "Category": "Overlap",
                    "Metric": "Min Propensity",
                    "Value": f"{report.overlap_results.min_propensity_score:.4f}",
                    "Status": "Info",
                }
            )
            data.append(
                {
                    "Category": "Overlap",
                    "Metric": "Max Propensity",
                    "Value": f"{report.overlap_results.max_propensity_score:.4f}",
                    "Status": "Info",
                }
            )

        # Assumption results
        if report.assumption_results:
            data.append(
                {
                    "Category": "Assumptions",
                    "Metric": "Exchangeability",
                    "Value": "Likely"
                    if report.assumption_results.exchangeability_likely
                    else "Questionable",
                    "Status": "Pass"
                    if report.assumption_results.exchangeability_likely
                    else "Warn",
                }
            )
            data.append(
                {
                    "Category": "Assumptions",
                    "Metric": "Confounding Detected",
                    "Value": "Yes"
                    if report.assumption_results.confounding_detected
                    else "No",
                    "Status": "Warn"
                    if report.assumption_results.confounding_detected
                    else "Pass",
                }
            )

        # Specification results
        if report.specification_results:
            data.append(
                {
                    "Category": "Specification",
                    "Metric": "Overall Specification",
                    "Value": "Passed"
                    if report.specification_results.specification_passed
                    else "Issues",
                    "Status": "Pass"
                    if report.specification_results.specification_passed
                    else "Warn",
                }
            )

        # Sensitivity results
        if report.sensitivity_results:
            data.append(
                {
                    "Category": "Sensitivity",
                    "Metric": "E-value",
                    "Value": f"{report.sensitivity_results.evalue:.2f}",
                    "Status": "Info",
                }
            )

        return pd.DataFrame(data)


def generate_diagnostic_report(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    causal_effect: CausalEffect | None = None,
    include_sensitivity: bool = False,
    verbose: bool = True,
) -> DiagnosticReport:
    """Convenience function to generate comprehensive diagnostic report.

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        covariates: Covariate data
        causal_effect: Optional causal effect for sensitivity analysis
        include_sensitivity: Whether to include sensitivity analysis
        verbose: Whether to print detailed output

    Returns:
        DiagnosticReport with comprehensive assessment
    """
    generator = DiagnosticReportGenerator(include_sensitivity=include_sensitivity)
    report = generator.generate_comprehensive_report(
        treatment, outcome, covariates, causal_effect, verbose=False
    )

    if verbose:
        generator.print_diagnostic_report(report)

    return report


def create_assumption_summary(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
) -> dict[str, bool]:
    """Create a simple summary of key assumption checks.

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        covariates: Covariate data

    Returns:
        Dictionary with assumption check results
    """
    # Quick checks
    balance_checker = BalanceDiagnostics()
    balance_results = balance_checker.assess_balance(treatment, covariates)

    overlap_checker = OverlapDiagnostics()
    overlap_results = overlap_checker.assess_overlap(treatment, covariates)

    assumption_checker = AssumptionChecker()
    assumption_results = assumption_checker.check_all_assumptions(
        treatment, outcome, covariates
    )

    return {
        "covariate_balance": balance_results.overall_balance_met,
        "positivity": overlap_results.overall_positivity_met,
        "exchangeability": assumption_results.exchangeability_likely,
        "no_measured_confounding": not assumption_results.confounding_detected,
        "overall_assessment": (
            balance_results.overall_balance_met
            and overlap_results.overall_positivity_met
            and assumption_results.exchangeability_likely
        ),
    }
