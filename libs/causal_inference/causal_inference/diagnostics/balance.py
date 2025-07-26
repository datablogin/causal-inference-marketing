"""Covariate balance diagnostics for causal inference.

This module implements tools to assess whether covariates are balanced between
treatment groups, which is crucial for valid causal inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from ..core.base import CovariateData, TreatmentData


@dataclass
class BalanceResults:
    """Results from covariate balance assessment."""

    standardized_mean_differences: dict[str, float]
    variance_ratios: dict[str, float]
    p_values: dict[str, float]
    balance_threshold: float
    imbalanced_covariates: list[str]
    overall_balance_met: bool
    sample_sizes: dict[str, int]


def calculate_standardized_mean_difference(
    covariate_values: NDArray[Any] | pd.Series,
    treatment_values: NDArray[Any] | pd.Series,
    treatment_level_1: Any = 1,
    treatment_level_0: Any = 0,
) -> float:
    """Calculate standardized mean difference (SMD) for a covariate.

    SMD is defined as the difference in means divided by the pooled standard deviation:
    SMD = (mean_treated - mean_control) / pooled_std

    Args:
        covariate_values: Values of the covariate
        treatment_values: Treatment assignment values
        treatment_level_1: Value representing treatment group
        treatment_level_0: Value representing control group

    Returns:
        Standardized mean difference
    """
    # Convert to numpy arrays
    covariate_values = np.asarray(covariate_values)
    treatment_values = np.asarray(treatment_values)

    # Remove missing values
    mask = ~(np.isnan(covariate_values) | np.isnan(treatment_values))
    covariate_clean = covariate_values[mask]
    treatment_clean = treatment_values[mask]

    # Split by treatment
    treated_mask = treatment_clean == treatment_level_1
    control_mask = treatment_clean == treatment_level_0

    treated_values = covariate_clean[treated_mask]
    control_values = covariate_clean[control_mask]

    if len(treated_values) == 0 or len(control_values) == 0:
        return np.nan

    # Calculate means
    mean_treated = np.mean(treated_values)
    mean_control = np.mean(control_values)

    # Calculate pooled standard deviation
    var_treated = np.var(treated_values, ddof=1) if len(treated_values) > 1 else 0
    var_control = np.var(control_values, ddof=1) if len(control_values) > 1 else 0

    pooled_std = np.sqrt((var_treated + var_control) / 2)

    if pooled_std == 0:
        return 0.0

    smd = (mean_treated - mean_control) / pooled_std
    return float(smd)


def calculate_variance_ratio(
    covariate_values: NDArray[Any] | pd.Series,
    treatment_values: NDArray[Any] | pd.Series,
    treatment_level_1: Any = 1,
    treatment_level_0: Any = 0,
) -> float:
    """Calculate variance ratio between treatment groups.

    Args:
        covariate_values: Values of the covariate
        treatment_values: Treatment assignment values
        treatment_level_1: Value representing treatment group
        treatment_level_0: Value representing control group

    Returns:
        Ratio of treated group variance to control group variance
    """
    # Convert to numpy arrays
    covariate_values = np.asarray(covariate_values)
    treatment_values = np.asarray(treatment_values)

    # Remove missing values
    mask = ~(np.isnan(covariate_values) | np.isnan(treatment_values))
    covariate_clean = covariate_values[mask]
    treatment_clean = treatment_values[mask]

    # Split by treatment
    treated_mask = treatment_clean == treatment_level_1
    control_mask = treatment_clean == treatment_level_0

    treated_values = covariate_clean[treated_mask]
    control_values = covariate_clean[control_mask]

    if len(treated_values) <= 1 or len(control_values) <= 1:
        return np.nan

    var_treated = np.var(treated_values, ddof=1)
    var_control = np.var(control_values, ddof=1)

    if var_control == 0:
        return np.inf if var_treated > 0 else 1.0

    return float(var_treated / var_control)


class BalanceDiagnostics:
    """Comprehensive covariate balance assessment tools."""

    def __init__(
        self,
        balance_threshold: float = 0.1,
        variance_ratio_bounds: tuple[float, float] = (0.5, 2.0),
        alpha: float = 0.05,
    ):
        """Initialize balance diagnostics.

        Args:
            balance_threshold: SMD threshold for declaring imbalance (default 0.1)
            variance_ratio_bounds: Acceptable range for variance ratios
            alpha: Significance level for statistical tests
        """
        self.balance_threshold = balance_threshold
        self.variance_ratio_bounds = variance_ratio_bounds
        self.alpha = alpha

    def assess_balance(
        self,
        treatment: TreatmentData,
        covariates: CovariateData,
        treatment_level_1: Any = 1,
        treatment_level_0: Any = 0,
    ) -> BalanceResults:
        """Assess covariate balance between treatment groups.

        Args:
            treatment: Treatment assignment data
            covariates: Covariate data
            treatment_level_1: Value representing treatment group
            treatment_level_0: Value representing control group

        Returns:
            BalanceResults object with comprehensive balance assessment
        """
        if not isinstance(covariates.values, pd.DataFrame):
            raise ValueError("Covariates must be a DataFrame for balance assessment")

        # Initialize results
        smds = {}
        variance_ratios = {}
        p_values = {}
        imbalanced_covariates = []

        # Calculate sample sizes
        treatment_vals = np.asarray(treatment.values)
        n_treated = np.sum(treatment_vals == treatment_level_1)
        n_control = np.sum(treatment_vals == treatment_level_0)
        sample_sizes = {
            "treated": int(n_treated),
            "control": int(n_control),
            "total": len(treatment_vals),
        }

        # Assess each covariate
        for covariate_name in covariates.values.columns:
            covariate_vals = covariates.values[covariate_name]

            # Calculate SMD
            smd = calculate_standardized_mean_difference(
                covariate_vals, treatment.values, treatment_level_1, treatment_level_0
            )
            smds[covariate_name] = smd

            # Calculate variance ratio
            var_ratio = calculate_variance_ratio(
                covariate_vals, treatment.values, treatment_level_1, treatment_level_0
            )
            variance_ratios[covariate_name] = var_ratio

            # Statistical test (t-test for continuous, chi-square for categorical)
            p_val = self._statistical_test(
                covariate_vals, treatment.values, treatment_level_1, treatment_level_0
            )
            p_values[covariate_name] = p_val

            # Check if imbalanced
            if abs(smd) > self.balance_threshold:
                imbalanced_covariates.append(covariate_name)

        # Overall balance assessment
        overall_balance_met = len(imbalanced_covariates) == 0

        return BalanceResults(
            standardized_mean_differences=smds,
            variance_ratios=variance_ratios,
            p_values=p_values,
            balance_threshold=self.balance_threshold,
            imbalanced_covariates=imbalanced_covariates,
            overall_balance_met=overall_balance_met,
            sample_sizes=sample_sizes,
        )

    def _statistical_test(
        self,
        covariate_values: pd.Series,
        treatment_values: NDArray[Any] | pd.Series,
        treatment_level_1: Any,
        treatment_level_0: Any,
    ) -> float:
        """Perform appropriate statistical test for covariate balance."""
        # Convert to numpy arrays
        covariate_vals = np.asarray(covariate_values)
        treatment_vals = np.asarray(treatment_values)

        # Remove missing values
        mask = ~(np.isnan(covariate_vals) | np.isnan(treatment_vals))
        covariate_clean = covariate_vals[mask]
        treatment_clean = treatment_vals[mask]

        # Split by treatment
        treated_mask = treatment_clean == treatment_level_1
        control_mask = treatment_clean == treatment_level_0

        treated_values = covariate_clean[treated_mask]
        control_values = covariate_clean[control_mask]

        if len(treated_values) == 0 or len(control_values) == 0:
            return np.nan

        # Check if covariate is binary/categorical (limited unique values)
        unique_values = len(np.unique(covariate_clean))

        if unique_values <= 10:
            # Use chi-square test for categorical variables
            try:
                # Create contingency table
                treated_counts = np.bincount(
                    treated_values.astype(int), minlength=unique_values
                )
                control_counts = np.bincount(
                    control_values.astype(int), minlength=unique_values
                )

                contingency_table = np.array([treated_counts, control_counts])
                chi2, p_val = stats.chi2_contingency(contingency_table)[:2]
                return float(p_val)
            except (ValueError, TypeError):
                # Fall back to t-test
                pass

        # Use t-test for continuous variables
        try:
            _, p_val = stats.ttest_ind(treated_values, control_values, equal_var=False)
            return float(p_val)
        except (ValueError, ZeroDivisionError):
            return np.nan

    def create_balance_table(self, balance_results: BalanceResults) -> pd.DataFrame:
        """Create a formatted balance table for reporting.

        Args:
            balance_results: Results from balance assessment

        Returns:
            DataFrame with balance statistics
        """
        data = []

        for covariate in balance_results.standardized_mean_differences.keys():
            smd = balance_results.standardized_mean_differences[covariate]
            var_ratio = balance_results.variance_ratios[covariate]
            p_val = balance_results.p_values[covariate]

            # Determine balance status
            if abs(smd) > self.balance_threshold:
                balance_status = "Imbalanced"
            else:
                balance_status = "Balanced"

            data.append(
                {
                    "Covariate": covariate,
                    "SMD": f"{smd:.3f}",
                    "Variance_Ratio": f"{var_ratio:.3f}"
                    if not np.isnan(var_ratio)
                    else "N/A",
                    "P_Value": f"{p_val:.3f}" if not np.isnan(p_val) else "N/A",
                    "Balance_Status": balance_status,
                }
            )

        df = pd.DataFrame(data)
        return df

    def print_balance_summary(self, balance_results: BalanceResults) -> None:
        """Print a summary of balance assessment results."""
        print("=== Covariate Balance Assessment ===")
        print(f"Balance threshold (SMD): {balance_results.balance_threshold}")
        print(
            f"Sample sizes - Treated: {balance_results.sample_sizes['treated']}, "
            f"Control: {balance_results.sample_sizes['control']}"
        )
        print()

        if balance_results.overall_balance_met:
            print("✅ Overall balance: ACHIEVED")
            print("All covariates meet the balance threshold.")
        else:
            print("❌ Overall balance: NOT ACHIEVED")
            print(
                f"Imbalanced covariates ({len(balance_results.imbalanced_covariates)}):"
            )
            for covariate in balance_results.imbalanced_covariates:
                smd = balance_results.standardized_mean_differences[covariate]
                print(f"  - {covariate}: SMD = {smd:.3f}")

        print()
        print("Balance Table:")
        balance_table = self.create_balance_table(balance_results)
        print(balance_table.to_string(index=False))


def check_covariate_balance(
    treatment: TreatmentData,
    covariates: CovariateData,
    balance_threshold: float = 0.1,
    verbose: bool = True,
) -> BalanceResults:
    """Convenience function to check covariate balance.

    Args:
        treatment: Treatment assignment data
        covariates: Covariate data
        balance_threshold: SMD threshold for declaring imbalance
        verbose: Whether to print results

    Returns:
        BalanceResults object
    """
    diagnostics = BalanceDiagnostics(balance_threshold=balance_threshold)
    results = diagnostics.assess_balance(treatment, covariates)

    if verbose:
        diagnostics.print_balance_summary(results)

    return results
