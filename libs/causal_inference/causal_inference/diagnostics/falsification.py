"""Falsification tests and placebo checks for causal inference.

This module provides various falsification tests and placebo checks to validate
causal inference assumptions and detect potential violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.base import CovariateData, EstimationError, OutcomeData, TreatmentData


@dataclass
class FalsificationResults:
    """Results from falsification tests."""

    placebo_outcome_test: dict[str, Any]
    placebo_treatment_test: dict[str, Any]
    future_outcome_test: dict[str, Any] | None
    negative_control_test: dict[str, Any] | None
    pre_treatment_balance: dict[str, Any] | None
    dose_response_test: dict[str, Any] | None
    overall_assessment: str
    recommendations: list[str]


class FalsificationTester:
    """Class for conducting falsification tests and placebo checks."""

    def __init__(self, random_state: int = 42):
        """Initialize the falsification tester.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def placebo_outcome_test(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData,
        estimator: Any,
        n_placebo_outcomes: int = 10,
    ) -> dict[str, Any]:
        """Test using placebo (fake) outcomes that should not be affected by treatment.

        Args:
            treatment: Treatment data
            outcome: Real outcome data (used for distribution reference)
            covariates: Covariate data
            estimator: Fitted causal estimator
            n_placebo_outcomes: Number of placebo outcomes to test

        Returns:
            Dictionary with placebo outcome test results
        """
        placebo_effects = []
        significant_count = 0

        # Get real outcome statistics for generating realistic placebo outcomes
        if isinstance(outcome.values, pd.Series):
            outcome_values = outcome.values.values
        else:
            outcome_values = np.asarray(outcome.values)

        outcome_mean = np.mean(outcome_values)
        outcome_std = np.std(outcome_values)

        for i in range(n_placebo_outcomes):
            # Generate placebo outcome with similar distribution but no causal effect
            placebo_values = np.random.normal(
                outcome_mean, outcome_std, size=len(outcome_values)
            )

            # Add some correlation with covariates but not treatment
            if isinstance(covariates.values, pd.DataFrame):
                cov_matrix = covariates.values.select_dtypes(include=[np.number])
                if not cov_matrix.empty:
                    # Add small correlation with first covariate
                    first_cov = cov_matrix.iloc[:, 0].values
                    placebo_values += 0.1 * (first_cov - np.mean(first_cov))

            # Create placebo outcome data
            placebo_outcome = OutcomeData(
                values=placebo_values,
                name=f"placebo_outcome_{i}",
                outcome_type=outcome.outcome_type,
            )

            # Fit estimator with placebo outcome
            try:
                placebo_estimator = type(estimator)()
                placebo_estimator.fit(treatment, placebo_outcome, covariates)
                placebo_effect = placebo_estimator.estimate_ate()

                placebo_effects.append(placebo_effect.ate)

                # Check if effect is "significant" (CI doesn't include 0)
                if placebo_effect.confidence_interval:
                    ci_lower, ci_upper = placebo_effect.confidence_interval
                    if ci_lower > 0 or ci_upper < 0:
                        significant_count += 1

            except Exception:
                # If fitting fails, record as zero effect
                placebo_effects.append(0.0)

        # Calculate test statistics
        mean_placebo_effect = np.mean(placebo_effects)
        std_placebo_effect = np.std(placebo_effects)
        false_positive_rate = significant_count / n_placebo_outcomes

        # Assessment
        if false_positive_rate <= 0.05:
            assessment = "PASS - Low false positive rate"
        elif false_positive_rate <= 0.10:
            assessment = "CAUTION - Moderate false positive rate"
        else:
            assessment = "FAIL - High false positive rate suggests bias"

        return {
            "placebo_effects": placebo_effects,
            "mean_effect": mean_placebo_effect,
            "std_effect": std_placebo_effect,
            "false_positive_rate": false_positive_rate,
            "significant_count": significant_count,
            "total_tests": n_placebo_outcomes,
            "assessment": assessment,
        }

    def placebo_treatment_test(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData,
        estimator: Any,
        n_placebo_treatments: int = 10,
    ) -> dict[str, Any]:
        """Test using placebo (fake) treatments that should have no effect.

        Args:
            treatment: Real treatment data (used for distribution reference)
            outcome: Outcome data
            covariates: Covariate data
            estimator: Causal estimator class
            n_placebo_treatments: Number of placebo treatments to test

        Returns:
            Dictionary with placebo treatment test results
        """
        placebo_effects = []
        significant_count = 0

        for i in range(n_placebo_treatments):
            # Generate random placebo treatment
            n_obs = len(treatment.values)
            if treatment.treatment_type == "binary":
                # Random binary assignment with similar proportion
                real_prop = np.mean(np.asarray(treatment.values))
                placebo_values = np.random.binomial(1, real_prop, n_obs)
            else:
                # Random assignment for continuous treatment
                treatment_vals = np.asarray(treatment.values)
                placebo_values = np.random.permutation(treatment_vals)

            # Create placebo treatment data
            placebo_treatment = TreatmentData(
                values=placebo_values,
                name=f"placebo_treatment_{i}",
                treatment_type=treatment.treatment_type,
                categories=treatment.categories,
            )

            # Fit estimator with placebo treatment
            try:
                placebo_estimator = type(estimator)()
                placebo_estimator.fit(placebo_treatment, outcome, covariates)
                placebo_effect = placebo_estimator.estimate_ate()

                placebo_effects.append(placebo_effect.ate)

                # Check if effect is "significant"
                if placebo_effect.confidence_interval:
                    ci_lower, ci_upper = placebo_effect.confidence_interval
                    if ci_lower > 0 or ci_upper < 0:
                        significant_count += 1

            except Exception:
                placebo_effects.append(0.0)

        # Calculate test statistics
        mean_placebo_effect = np.mean(placebo_effects)
        std_placebo_effect = np.std(placebo_effects)
        false_positive_rate = significant_count / n_placebo_treatments

        # Assessment
        if false_positive_rate <= 0.05:
            assessment = "PASS - Low false positive rate"
        elif false_positive_rate <= 0.10:
            assessment = "CAUTION - Moderate false positive rate"
        else:
            assessment = "FAIL - High false positive rate suggests confounding"

        return {
            "placebo_effects": placebo_effects,
            "mean_effect": mean_placebo_effect,
            "std_effect": std_placebo_effect,
            "false_positive_rate": false_positive_rate,
            "significant_count": significant_count,
            "total_tests": n_placebo_treatments,
            "assessment": assessment,
        }

    def future_outcome_test(
        self,
        treatment: TreatmentData,
        pre_treatment_outcome: OutcomeData,
        covariates: CovariateData,
        estimator: Any,
    ) -> dict[str, Any]:
        """Test treatment effect on pre-treatment outcome (should be zero).

        Args:
            treatment: Treatment data
            pre_treatment_outcome: Outcome measured before treatment
            covariates: Covariate data
            estimator: Causal estimator class

        Returns:
            Dictionary with future outcome test results
        """
        try:
            # Fit estimator with pre-treatment outcome
            future_estimator = type(estimator)()
            future_estimator.fit(treatment, pre_treatment_outcome, covariates)
            future_effect = future_estimator.estimate_ate()

            # Check if effect is significant
            is_significant = False
            if future_effect.confidence_interval:
                ci_lower, ci_upper = future_effect.confidence_interval
                is_significant = ci_lower > 0 or ci_upper < 0

            # Assessment
            if not is_significant and abs(future_effect.ate) < 0.1:
                assessment = "PASS - No significant effect on pre-treatment outcome"
            elif abs(future_effect.ate) < 0.2:
                assessment = "CAUTION - Small effect on pre-treatment outcome"
            else:
                assessment = "FAIL - Significant effect on pre-treatment outcome suggests confounding"

            return {
                "estimated_effect": future_effect.ate,
                "confidence_interval": future_effect.confidence_interval,
                "is_significant": is_significant,
                "assessment": assessment,
            }

        except (ValueError, RuntimeError, EstimationError) as e:
            return {
                "estimated_effect": None,
                "confidence_interval": None,
                "is_significant": None,
                "assessment": f"ERROR - Could not perform test: {str(e)}",
            }
        except Exception as e:
            # Re-raise unexpected errors
            raise RuntimeError(f"Unexpected error in future outcome test: {e}") from e

    def negative_control_test(
        self,
        treatment: TreatmentData,
        negative_control_outcome: OutcomeData,
        covariates: CovariateData,
        estimator: Any,
    ) -> dict[str, Any]:
        """Test treatment effect on negative control outcome (should be zero).

        Args:
            treatment: Treatment data
            negative_control_outcome: Outcome that should not be affected by treatment
            covariates: Covariate data
            estimator: Causal estimator class

        Returns:
            Dictionary with negative control test results
        """
        try:
            # Fit estimator with negative control outcome
            control_estimator = type(estimator)()
            control_estimator.fit(treatment, negative_control_outcome, covariates)
            control_effect = control_estimator.estimate_ate()

            # Check if effect is significant
            is_significant = False
            if control_effect.confidence_interval:
                ci_lower, ci_upper = control_effect.confidence_interval
                is_significant = ci_lower > 0 or ci_upper < 0

            # Assessment
            if not is_significant and abs(control_effect.ate) < 0.1:
                assessment = "PASS - No significant effect on negative control"
            elif abs(control_effect.ate) < 0.2:
                assessment = "CAUTION - Small effect on negative control"
            else:
                assessment = "FAIL - Significant effect on negative control suggests bias"

            return {
                "estimated_effect": control_effect.ate,
                "confidence_interval": control_effect.confidence_interval,
                "is_significant": is_significant,
                "assessment": assessment,
            }

        except (ValueError, RuntimeError, EstimationError) as e:
            return {
                "estimated_effect": None,
                "confidence_interval": None,
                "is_significant": None,
                "assessment": f"ERROR - Could not perform test: {str(e)}",
            }
        except Exception as e:
            # Re-raise unexpected errors
            raise RuntimeError(f"Unexpected error in negative control test: {e}") from e

    def pre_treatment_balance_test(
        self,
        treatment: TreatmentData,
        pre_treatment_covariates: CovariateData,
    ) -> dict[str, Any]:
        """Test balance on pre-treatment covariates.

        Args:
            treatment: Treatment data
            pre_treatment_covariates: Covariates measured before treatment

        Returns:
            Dictionary with pre-treatment balance test results
        """
        from .balance import BalanceDiagnostics

        try:
            # Run balance diagnostics on pre-treatment covariates
            balance_checker = BalanceDiagnostics()
            balance_results = balance_checker.assess_balance(treatment, pre_treatment_covariates)

            # Calculate summary statistics
            imbalanced_vars = [
                var for var, balanced in balance_results.is_balanced.items()
                if not balanced
            ]

            imbalance_rate = len(imbalanced_vars) / len(balance_results.is_balanced)

            # Assessment
            if imbalance_rate <= 0.1:
                assessment = "PASS - Good balance on pre-treatment covariates"
            elif imbalance_rate <= 0.3:
                assessment = "CAUTION - Moderate imbalance on pre-treatment covariates"
            else:
                assessment = "FAIL - Poor balance suggests systematic differences"

            return {
                "balance_results": balance_results,
                "imbalanced_variables": imbalanced_vars,
                "imbalance_rate": imbalance_rate,
                "assessment": assessment,
            }

        except (ValueError, RuntimeError, EstimationError) as e:
            return {
                "balance_results": None,
                "imbalanced_variables": [],
                "imbalance_rate": None,
                "assessment": f"ERROR - Could not perform test: {str(e)}",
            }
        except Exception as e:
            # Re-raise unexpected errors
            raise RuntimeError(f"Unexpected error in pre-treatment balance test: {e}") from e

    def dose_response_test(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData,
        n_dose_levels: int = 5,
    ) -> dict[str, Any]:
        """Test for monotonic dose-response relationship.

        Args:
            treatment: Treatment data (should be continuous or ordinal)
            outcome: Outcome data
            covariates: Covariate data
            n_dose_levels: Number of dose levels to create

        Returns:
            Dictionary with dose-response test results
        """
        try:
            # Convert treatment to dose levels
            treatment_vals = np.asarray(treatment.values)
            if treatment.treatment_type == "binary":
                return {
                    "dose_levels": None,
                    "effects": None,
                    "is_monotonic": None,
                    "assessment": "SKIP - Binary treatment not suitable for dose-response test",
                }

            # Create dose level bins
            dose_bins = np.quantile(treatment_vals, np.linspace(0, 1, n_dose_levels + 1))
            dose_labels = np.digitize(treatment_vals, dose_bins) - 1
            dose_labels = np.clip(dose_labels, 0, n_dose_levels - 1)

            # Calculate mean outcome for each dose level
            outcome_vals = np.asarray(outcome.values)
            dose_effects = []

            for dose in range(n_dose_levels):
                dose_mask = dose_labels == dose
                if np.sum(dose_mask) > 0:
                    dose_mean = np.mean(outcome_vals[dose_mask])
                    dose_effects.append(dose_mean)
                else:
                    dose_effects.append(np.nan)

            # Test for monotonicity
            valid_effects = [e for e in dose_effects if not np.isnan(e)]
            if len(valid_effects) >= 3:
                # Check if effects are roughly monotonic
                differences = np.diff(valid_effects)
                positive_diffs = np.sum(differences > 0)
                negative_diffs = np.sum(differences < 0)

                monotonic_ratio = max(positive_diffs, negative_diffs) / len(differences)
                is_monotonic = monotonic_ratio >= 0.7

                if is_monotonic:
                    assessment = "PASS - Monotonic dose-response relationship observed"
                else:
                    assessment = "CAUTION - Non-monotonic dose-response relationship"
            else:
                is_monotonic = None
                assessment = "INCONCLUSIVE - Insufficient dose levels for assessment"

            return {
                "dose_levels": list(range(n_dose_levels)),
                "effects": dose_effects,
                "is_monotonic": is_monotonic,
                "assessment": assessment,
            }

        except (ValueError, RuntimeError, EstimationError) as e:
            return {
                "dose_levels": None,
                "effects": None,
                "is_monotonic": None,
                "assessment": f"ERROR - Could not perform test: {str(e)}",
            }
        except Exception as e:
            # Re-raise unexpected errors
            raise RuntimeError(f"Unexpected error in dose response test: {e}") from e

    def run_all_falsification_tests(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData,
        estimator: Any,
        pre_treatment_outcome: OutcomeData | None = None,
        negative_control_outcome: OutcomeData | None = None,
        pre_treatment_covariates: CovariateData | None = None,
    ) -> FalsificationResults:
        """Run comprehensive falsification tests.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Covariate data
            estimator: Fitted causal estimator
            pre_treatment_outcome: Optional pre-treatment outcome for future outcome test
            negative_control_outcome: Optional negative control outcome
            pre_treatment_covariates: Optional pre-treatment covariates

        Returns:
            FalsificationResults with comprehensive assessment
        """
        # Run placebo outcome test
        placebo_outcome = self.placebo_outcome_test(
            treatment, outcome, covariates, estimator
        )

        # Run placebo treatment test
        placebo_treatment = self.placebo_treatment_test(
            treatment, outcome, covariates, estimator
        )

        # Run future outcome test if data available
        future_outcome = None
        if pre_treatment_outcome:
            future_outcome = self.future_outcome_test(
                treatment, pre_treatment_outcome, covariates, estimator
            )

        # Run negative control test if data available
        negative_control = None
        if negative_control_outcome:
            negative_control = self.negative_control_test(
                treatment, negative_control_outcome, covariates, estimator
            )

        # Run pre-treatment balance test if data available
        pre_balance = None
        if pre_treatment_covariates:
            pre_balance = self.pre_treatment_balance_test(
                treatment, pre_treatment_covariates
            )

        # Run dose-response test
        dose_response = self.dose_response_test(treatment, outcome, covariates)

        # Overall assessment
        assessments = [
            placebo_outcome["assessment"],
            placebo_treatment["assessment"],
        ]

        if future_outcome:
            assessments.append(future_outcome["assessment"])
        if negative_control:
            assessments.append(negative_control["assessment"])
        if pre_balance:
            assessments.append(pre_balance["assessment"])
        if dose_response["assessment"] != "SKIP":
            assessments.append(dose_response["assessment"])

        # Count passes, cautions, and failures
        passes = sum(1 for a in assessments if a.startswith("PASS"))
        cautions = sum(1 for a in assessments if a.startswith("CAUTION"))
        failures = sum(1 for a in assessments if a.startswith("FAIL"))

        if failures > 0:
            overall = f"CONCERNS - {failures} test(s) failed, suggesting potential bias"
        elif cautions > passes:
            overall = f"MIXED - {cautions} test(s) show caution, {passes} passed"
        else:
            overall = f"ROBUST - {passes} test(s) passed, suggesting valid causal inference"

        # Generate recommendations
        recommendations = []
        if placebo_outcome["false_positive_rate"] > 0.1:
            recommendations.append("High false positive rate in placebo outcomes suggests model mis-specification")
        if placebo_treatment["false_positive_rate"] > 0.1:
            recommendations.append("High false positive rate in placebo treatments suggests unmeasured confounding")
        if future_outcome and future_outcome["is_significant"]:
            recommendations.append("Significant effect on pre-treatment outcome indicates confounding")
        if negative_control and negative_control["is_significant"]:
            recommendations.append("Significant effect on negative control suggests bias or model issues")
        if pre_balance and pre_balance["imbalance_rate"] and pre_balance["imbalance_rate"] > 0.3:
            recommendations.append("Poor balance on pre-treatment covariates suggests systematic differences")
        if dose_response["is_monotonic"] is False:
            recommendations.append("Non-monotonic dose-response relationship may indicate confounding or effect modification")

        if not recommendations:
            recommendations.append("Falsification tests support causal interpretation")
            recommendations.append("Consider additional robustness checks for stronger evidence")

        return FalsificationResults(
            placebo_outcome_test=placebo_outcome,
            placebo_treatment_test=placebo_treatment,
            future_outcome_test=future_outcome,
            negative_control_test=negative_control,
            pre_treatment_balance=pre_balance,
            dose_response_test=dose_response,
            overall_assessment=overall,
            recommendations=recommendations,
        )

    def print_falsification_summary(self, results: FalsificationResults) -> None:
        """Print a summary of falsification test results.

        Args:
            results: FalsificationResults object
        """
        print("=== Falsification Test Results ===")
        print()

        print("Placebo Outcome Test:")
        print(f"  Assessment: {results.placebo_outcome_test['assessment']}")
        print(f"  False Positive Rate: {results.placebo_outcome_test['false_positive_rate']:.3f}")
        print()

        print("Placebo Treatment Test:")
        print(f"  Assessment: {results.placebo_treatment_test['assessment']}")
        print(f"  False Positive Rate: {results.placebo_treatment_test['false_positive_rate']:.3f}")
        print()

        if results.future_outcome_test:
            print("Future Outcome Test:")
            print(f"  Assessment: {results.future_outcome_test['assessment']}")
            if results.future_outcome_test['estimated_effect'] is not None:
                print(f"  Effect Estimate: {results.future_outcome_test['estimated_effect']:.3f}")
            print()

        if results.negative_control_test:
            print("Negative Control Test:")
            print(f"  Assessment: {results.negative_control_test['assessment']}")
            if results.negative_control_test['estimated_effect'] is not None:
                print(f"  Effect Estimate: {results.negative_control_test['estimated_effect']:.3f}")
            print()

        if results.pre_treatment_balance:
            print("Pre-treatment Balance Test:")
            print(f"  Assessment: {results.pre_treatment_balance['assessment']}")
            if results.pre_treatment_balance['imbalance_rate'] is not None:
                print(f"  Imbalance Rate: {results.pre_treatment_balance['imbalance_rate']:.3f}")
            print()

        if results.dose_response_test and results.dose_response_test['assessment'] != "SKIP":
            print("Dose-Response Test:")
            print(f"  Assessment: {results.dose_response_test['assessment']}")
            if results.dose_response_test['is_monotonic'] is not None:
                print(f"  Is Monotonic: {results.dose_response_test['is_monotonic']}")
            print()

        print("Overall Assessment:")
        print(f"  {results.overall_assessment}")
        print()

        print("Recommendations:")
        for rec in results.recommendations:
            print(f"  • {rec}")


# Convenience functions
def run_placebo_outcome_test(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    estimator: Any,
    n_placebo_outcomes: int = 10,
) -> dict[str, Any]:
    """Convenience function for placebo outcome test."""
    tester = FalsificationTester()
    return tester.placebo_outcome_test(
        treatment, outcome, covariates, estimator, n_placebo_outcomes
    )


def run_placebo_treatment_test(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    estimator: Any,
    n_placebo_treatments: int = 10,
) -> dict[str, Any]:
    """Convenience function for placebo treatment test."""
    tester = FalsificationTester()
    return tester.placebo_treatment_test(
        treatment, outcome, covariates, estimator, n_placebo_treatments
    )


def run_all_falsification_tests(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData,
    estimator: Any,
    pre_treatment_outcome: OutcomeData | None = None,
    negative_control_outcome: OutcomeData | None = None,
    pre_treatment_covariates: CovariateData | None = None,
) -> FalsificationResults:
    """Convenience function for running all falsification tests."""
    tester = FalsificationTester()
    return tester.run_all_falsification_tests(
        treatment,
        outcome,
        covariates,
        estimator,
        pre_treatment_outcome,
        negative_control_outcome,
        pre_treatment_covariates,
    )
