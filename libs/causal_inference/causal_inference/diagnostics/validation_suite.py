"""Comprehensive Model Diagnostics and Validation Suite.

This module implements a comprehensive diagnostic and validation framework
to assess the quality and assumptions of causal inference models, as specified
in GitHub issue #74.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.base import CovariateData, OutcomeData, TreatmentData
from .balance import BalanceDiagnostics, BalanceResults
from .overlap import OverlapDiagnostics, OverlapResults
from .specification import ModelSpecificationTests, SpecificationResults


@dataclass
class ComprehensiveValidationResults:
    """Results from comprehensive model validation."""

    balance_assessment: BalanceResults
    overlap_assessment: OverlapResults
    specification_assessment: SpecificationResults
    overall_validation_passed: bool
    critical_issues: list[str]
    warnings: list[str]
    recommendations: list[str]
    validation_score: float  # 0-100 scale


class ComprehensiveValidationSuite:
    """Comprehensive diagnostic and validation framework for causal inference models."""

    def __init__(
        self,
        balance_threshold: float = 0.1,
        min_propensity: float = 0.01,
        max_propensity: float = 0.99,
        alpha: float = 0.05,
    ):
        """Initialize the validation suite.

        Args:
            balance_threshold: SMD threshold for declaring imbalance
            min_propensity: Minimum acceptable propensity score
            max_propensity: Maximum acceptable propensity score
            alpha: Significance level for statistical tests
        """
        self.balance_diagnostics = BalanceDiagnostics(
            balance_threshold=balance_threshold
        )
        self.overlap_diagnostics = OverlapDiagnostics(
            min_propensity=min_propensity, max_propensity=max_propensity
        )
        self.specification_tests = ModelSpecificationTests(alpha=alpha)

    def validate_assumptions(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData,
        generate_report: bool = False,
    ) -> ComprehensiveValidationResults:
        """Run comprehensive validation of causal inference assumptions.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            covariates: Covariate data
            generate_report: Whether to generate a detailed report

        Returns:
            ComprehensiveValidationResults with full assessment
        """
        # 1. Covariate Balance Assessment
        balance_results = self.balance_diagnostics.assess_balance(
            treatment, covariates, outcome=outcome
        )

        # 2. Positivity/Overlap Diagnostics
        overlap_results = self.overlap_diagnostics.assess_overlap(treatment, covariates)

        # 3. Model Specification Checks
        specification_results = (
            self.specification_tests.comprehensive_specification_tests(
                outcome, treatment, covariates
            )
        )

        # Compile overall assessment
        critical_issues = []
        warnings = []
        recommendations = []

        # Analyze balance issues
        if not balance_results.overall_balance_met:
            critical_issues.append("Covariate balance not achieved")
            recommendations.append(
                f"Address imbalanced covariates: {', '.join(balance_results.imbalanced_covariates)}"
            )

        # Check prognostic score balance
        if (
            balance_results.prognostic_score_balance
            and not balance_results.prognostic_score_balance.get("balance_met", True)
        ):
            warnings.append("Prognostic score imbalance detected")
            recommendations.append(
                "Consider additional covariates or different matching strategy"
            )

        # Analyze overlap issues
        if not overlap_results.overall_positivity_met:
            critical_issues.append("Positivity assumption violated")

        for violation in overlap_results.violations:
            if violation["severity"] == "high":
                critical_issues.append(violation["description"])
            else:
                warnings.append(violation["description"])

        # Check model calibration
        if (
            overlap_results.calibration_results
            and not overlap_results.calibration_results.get("well_calibrated", True)
        ):
            warnings.append("Propensity score model poorly calibrated")
            recommendations.append(
                "Consider model recalibration or alternative propensity score methods"
            )

        # Analyze specification issues
        if not specification_results.specification_passed:
            critical_issues.append("Model specification issues detected")
            recommendations.extend(specification_results.recommendations)

        # Check for high propensity model AUC (suggests strong confounding)
        if (
            overlap_results.propensity_model_auc
            and overlap_results.propensity_model_auc > 0.8
        ):
            warnings.append(
                f"High propensity model AUC ({overlap_results.propensity_model_auc:.3f}) suggests strong confounding"
            )
            recommendations.append(
                "Consider additional covariates or sensitivity analysis"
            )

        # Calculate validation score (0-100)
        validation_score = self._calculate_validation_score(
            balance_results, overlap_results, specification_results
        )

        # Overall validation status
        overall_validation_passed = len(critical_issues) == 0 and validation_score >= 70

        # Add general recommendations
        if not overall_validation_passed:
            if validation_score < 50:
                recommendations.append(
                    "⚠️ Caution: Consider alternative causal identification strategies"
                )
            elif validation_score < 70:
                recommendations.append(
                    "⚠️ Proceed with caution and conduct sensitivity analyses"
                )
        else:
            recommendations.append(
                "✅ Assumptions appear adequately satisfied for causal analysis"
            )

        return ComprehensiveValidationResults(
            balance_assessment=balance_results,
            overlap_assessment=overlap_results,
            specification_assessment=specification_results,
            overall_validation_passed=overall_validation_passed,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            validation_score=validation_score,
        )

    def _calculate_validation_score(
        self,
        balance_results: BalanceResults,
        overlap_results: OverlapResults,
        specification_results: SpecificationResults,
    ) -> float:
        """Calculate overall validation score (0-100)."""

        # Balance component (30 points max)
        if balance_results.overall_balance_met:
            balance_score = 30
        else:
            # Penalize based on number and severity of imbalanced covariates
            n_imbalanced = len(balance_results.imbalanced_covariates)
            balance_score = max(0, 30 - (n_imbalanced * 5))

        # Prognostic score penalty
        if (
            balance_results.prognostic_score_balance
            and not balance_results.prognostic_score_balance.get("balance_met", True)
        ):
            balance_score -= 5

        # Overlap component (30 points max)
        if overlap_results.overall_positivity_met:
            overlap_score = 30
        else:
            overlap_score = 15  # Partial credit for some overlap

        # Penalty for extreme weights
        extreme_pct = (
            overlap_results.extreme_weights_count / overlap_results.total_units
        )
        overlap_score -= min(10.0, extreme_pct * 100)  # Up to 10 point penalty

        # Bonus for good calibration
        if (
            overlap_results.calibration_results
            and overlap_results.calibration_results.get("well_calibrated", False)
        ):
            overlap_score += 5

        # Specification component (30 points max)
        if specification_results.specification_passed:
            spec_score = 30
        else:
            spec_score = max(
                0, 30 - len(specification_results.problematic_variables) * 3
            )

        # Additional penalties (up to 10 points)
        penalties = 0

        # High AUC penalty
        if (
            overlap_results.propensity_model_auc
            and overlap_results.propensity_model_auc > 0.8
        ):
            penalties += 5

        # Poor common support penalty
        support_coverage = (
            overlap_results.units_in_common_support / overlap_results.total_units
        )
        if support_coverage < 0.8:
            penalties += 5

        final_score = max(0, balance_score + overlap_score + spec_score - penalties)
        return min(100, final_score)

    def generate_diagnostic_report(
        self,
        results: ComprehensiveValidationResults,
        title: str = "Causal Inference Validation Report",
    ) -> str:
        """Generate a comprehensive diagnostic report.

        Args:
            results: Validation results
            title: Report title

        Returns:
            Formatted report string
        """
        report = []
        report.append(f"{'=' * len(title)}")
        report.append(title)
        report.append(f"{'=' * len(title)}")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        report.append("")

        overall_status = "PASS" if results.overall_validation_passed else "FAIL"
        status_icon = "✅" if results.overall_validation_passed else "❌"

        report.append(f"**Overall Validation Status:** {status_icon} {overall_status}")
        report.append(f"**Validation Score:** {results.validation_score:.1f}/100")
        report.append("")

        if results.critical_issues:
            report.append("**Critical Issues:**")
            for issue in results.critical_issues:
                report.append(f"- 🔴 {issue}")
            report.append("")

        if results.warnings:
            report.append("**Warnings:**")
            for warning in results.warnings:
                report.append(f"- ⚠️ {warning}")
            report.append("")

        # Detailed Assessment
        report.append("## Detailed Assessment")
        report.append("")

        # Balance Assessment
        report.append("### 1. Covariate Balance")
        balance_status = (
            "✅ ACHIEVED"
            if results.balance_assessment.overall_balance_met
            else "❌ NOT ACHIEVED"
        )
        report.append(f"**Status:** {balance_status}")

        if results.balance_assessment.imbalanced_covariates:
            report.append(
                f"**Imbalanced covariates:** {', '.join(results.balance_assessment.imbalanced_covariates)}"
            )

        # Sample sizes
        report.append(
            f"**Sample sizes:** Treated = {results.balance_assessment.sample_sizes['treated']}, Control = {results.balance_assessment.sample_sizes['control']}"
        )

        # Prognostic score
        if results.balance_assessment.prognostic_score_balance:
            ps_balance = results.balance_assessment.prognostic_score_balance
            if "error" not in ps_balance:
                ps_status = "✅" if ps_balance["balance_met"] else "❌"
                report.append(
                    f"**Prognostic score balance:** {ps_status} (SMD = {ps_balance['smd']:.3f})"
                )

        report.append("")

        # Overlap Assessment
        report.append("### 2. Overlap and Positivity")
        overlap_status = (
            "✅ SATISFIED"
            if results.overlap_assessment.overall_positivity_met
            else "❌ VIOLATED"
        )
        report.append(f"**Status:** {overlap_status}")

        report.append(
            f"**Propensity score range:** [{results.overlap_assessment.min_propensity_score:.4f}, {results.overlap_assessment.max_propensity_score:.4f}]"
        )

        support_pct = (
            results.overlap_assessment.units_in_common_support
            / results.overlap_assessment.total_units
            * 100
        )
        report.append(f"**Common support coverage:** {support_pct:.1f}%")

        if results.overlap_assessment.propensity_model_auc:
            report.append(
                f"**Model AUC:** {results.overlap_assessment.propensity_model_auc:.3f}"
            )

        if (
            results.overlap_assessment.calibration_results
            and "error" not in results.overlap_assessment.calibration_results
        ):
            cal = results.overlap_assessment.calibration_results
            cal_status = "Good" if cal["well_calibrated"] else "Poor"
            report.append(
                f"**Model calibration:** {cal_status} (Brier score = {cal['brier_score']:.4f})"
            )

        report.append("")

        # Specification Assessment
        report.append("### 3. Model Specification")
        spec_status = (
            "✅ ADEQUATE"
            if results.specification_assessment.specification_passed
            else "❌ ISSUES DETECTED"
        )
        report.append(f"**Status:** {spec_status}")

        if results.specification_assessment.problematic_variables:
            report.append(
                f"**Problematic variables:** {', '.join(results.specification_assessment.problematic_variables)}"
            )

        # Individual tests
        tests = [
            (
                "Linearity",
                results.specification_assessment.linearity_test_results[
                    "overall_linearity_ok"
                ],
            ),
            (
                "Functional form",
                results.specification_assessment.functional_form_results[
                    "linear_model_adequate"
                ],
            ),
            (
                "Heteroskedasticity",
                not results.specification_assessment.heteroskedasticity_test[
                    "heteroskedasticity_detected"
                ],
            ),
        ]

        if results.specification_assessment.reset_test_results:
            tests.append(
                (
                    "RESET test",
                    results.specification_assessment.reset_test_results[
                        "specification_adequate"
                    ],
                )
            )

        for test_name, passed in tests:
            status = "✅" if passed else "❌"
            report.append(f"**{test_name}:** {status}")

        report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        for i, rec in enumerate(results.recommendations, 1):
            report.append(f"{i}. {rec}")

        report.append("")

        # Trimming suggestions (if available)
        if results.overlap_assessment.trimming_recommendations:
            report.append("## Trimming Suggestions")
            report.append("")

            for (
                trim_level,
                details,
            ) in results.overlap_assessment.trimming_recommendations.items():
                if trim_level in [
                    "trim_1.0pct",
                    "trim_2.5pct",
                    "trim_5.0pct",
                ]:  # Show only common levels
                    report.append(
                        f"**{trim_level}:** Remove {details['pct_trimmed']:.1f}% of sample"
                    )
                    report.append(
                        f"  - Bounds: [{details['lower_bound']:.4f}, {details['upper_bound']:.4f}]"
                    )
                    report.append(
                        f"  - Remaining: {details['n_treated_remaining']} treated, {details['n_control_remaining']} control"
                    )
                    report.append("")

        return "\n".join(report)

    def print_validation_summary(self, results: ComprehensiveValidationResults) -> None:
        """Print a concise validation summary."""
        print("=" * 60)
        print("COMPREHENSIVE CAUSAL INFERENCE VALIDATION SUMMARY")
        print("=" * 60)
        print()

        # Overall status
        status_icon = "✅" if results.overall_validation_passed else "❌"
        status_text = "PASS" if results.overall_validation_passed else "FAIL"
        print(f"Overall Status: {status_icon} {status_text}")
        print(f"Validation Score: {results.validation_score:.1f}/100")
        print()

        # Quick status of each component
        balance_icon = "✅" if results.balance_assessment.overall_balance_met else "❌"
        overlap_icon = (
            "✅" if results.overlap_assessment.overall_positivity_met else "❌"
        )
        spec_icon = (
            "✅" if results.specification_assessment.specification_passed else "❌"
        )

        print("Component Status:")
        print(f"  {balance_icon} Covariate Balance")
        print(f"  {overlap_icon} Positivity/Overlap")
        print(f"  {spec_icon} Model Specification")
        print()

        # Critical issues
        if results.critical_issues:
            print("🔴 Critical Issues:")
            for issue in results.critical_issues:
                print(f"   • {issue}")
            print()

        # Warnings
        if results.warnings:
            print("⚠️  Warnings:")
            for warning in results.warnings:
                print(f"   • {warning}")
            print()

        # Key recommendations
        print("📋 Key Recommendations:")
        for i, rec in enumerate(results.recommendations[:3], 1):  # Show top 3
            print(f"   {i}. {rec}")

        if len(results.recommendations) > 3:
            print(f"   ... and {len(results.recommendations) - 3} more")

        print()


def validate_causal_assumptions(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    balance_threshold: float = 0.1,
    min_propensity: float = 0.01,
    max_propensity: float = 0.99,
    verbose: bool = True,
    generate_report: bool = False,
) -> ComprehensiveValidationResults:
    """Convenience function for comprehensive causal inference validation.

    Args:
        treatment: Treatment assignment data
        outcome: Outcome data
        covariates: Covariate data
        balance_threshold: SMD threshold for declaring imbalance
        min_propensity: Minimum acceptable propensity score
        max_propensity: Maximum acceptable propensity score
        verbose: Whether to print summary
        generate_report: Whether to generate detailed report

    Returns:
        ComprehensiveValidationResults
    """
    suite = ComprehensiveValidationSuite(
        balance_threshold=balance_threshold,
        min_propensity=min_propensity,
        max_propensity=max_propensity,
    )

    results = suite.validate_assumptions(
        treatment, outcome, covariates, generate_report=generate_report
    )

    if verbose:
        suite.print_validation_summary(results)

        if generate_report:
            print("\n" + "=" * 60)
            print("DETAILED DIAGNOSTIC REPORT")
            print("=" * 60)
            report = suite.generate_diagnostic_report(results)
            print(report)

    return results
