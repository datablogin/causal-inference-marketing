"""Data validation utilities for causal inference datasets.

This module provides comprehensive validation tools to catch common issues
in causal inference data before analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..core.base import CovariateData, DataValidationError, OutcomeData, TreatmentData


class CausalDataValidator:
    """Validator for causal inference datasets."""

    def __init__(
        self,
        verbose: bool = True,
        outlier_threshold: float = 5.0,
        sample_size_for_checks: Optional[int] = None,
    ) -> None:
        """Initialize the validator.

        Args:
            verbose: Whether to print detailed validation messages
            outlier_threshold: Number of standard deviations for outlier detection
            sample_size_for_checks: If provided, sample this many observations for expensive checks (correlation, overlap) to improve performance
        """
        self.verbose: bool = verbose
        self.outlier_threshold: float = outlier_threshold
        self.sample_size_for_checks: Optional[int] = sample_size_for_checks
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def validate_treatment_data(self, treatment: TreatmentData) -> None:
        """Validate treatment data.

        Args:
            treatment: Treatment data to validate

        Raises:
            DataValidationError: If validation fails
        """
        if self.verbose:
            print("Validating treatment data...")

        # Check for missing values
        if isinstance(treatment.values, pd.Series):
            missing_count = treatment.values.isnull().sum()
            total_count = len(treatment.values)
        else:
            missing_count = np.isnan(treatment.values).sum()
            total_count = len(treatment.values)

        if missing_count > 0:
            pct_missing = 100 * missing_count / total_count
            msg = f"Treatment has {missing_count} missing values ({pct_missing:.1f}%)"
            if pct_missing > 5.0:
                self.errors.append(msg)
            else:
                self.warnings.append(msg)

        # Check treatment type consistency
        if treatment.treatment_type == "binary":
            unique_vals = np.unique(np.asarray(treatment.values))
            unique_vals = unique_vals[~pd.isnull(unique_vals)]

            if len(unique_vals) != 2:
                self.errors.append(
                    f"Binary treatment should have exactly 2 unique values, found {len(unique_vals)}: {unique_vals}"
                )
            elif not all(val in [0, 1] for val in unique_vals):
                self.warnings.append(
                    f"Binary treatment values are not 0/1: {unique_vals}"
                )

        elif treatment.treatment_type == "categorical":
            unique_vals = np.unique(np.asarray(treatment.values))
            unique_vals = unique_vals[~pd.isnull(unique_vals)]

            if len(unique_vals) < 2:
                self.errors.append(
                    f"Categorical treatment should have at least 2 categories, found {len(unique_vals)}"
                )

        # Check sample size
        if total_count < 30:
            self.warnings.append(
                f"Very small sample size for treatment: {total_count} observations"
            )

        if self.verbose:
            print(f"  - {total_count:,} observations")
            print(f"  - {missing_count} missing values")
            if treatment.treatment_type == "binary":
                treated = np.sum(treatment.values == 1)
                control = np.sum(treatment.values == 0)
                print(f"  - Treated: {treated}, Control: {control}")

    def validate_outcome_data(self, outcome: OutcomeData) -> None:
        """Validate outcome data.

        Args:
            outcome: Outcome data to validate

        Raises:
            DataValidationError: If validation fails
        """
        if self.verbose:
            print("Validating outcome data...")

        # Check for missing values
        if isinstance(outcome.values, pd.Series):
            missing_count = outcome.values.isnull().sum()
            total_count = len(outcome.values)
            values = np.asarray(outcome.values.dropna().values)
        else:
            missing_count = np.isnan(outcome.values).sum()
            total_count = len(outcome.values)
            values = np.asarray(outcome.values[~np.isnan(outcome.values)])

        if missing_count > 0:
            pct_missing = 100 * missing_count / total_count
            msg = f"Outcome has {missing_count} missing values ({pct_missing:.1f}%)"
            if pct_missing > 10.0:
                self.errors.append(msg)
            else:
                self.warnings.append(msg)

        # Check for infinite values
        if len(values) > 0:
            inf_count = np.isinf(np.asarray(values)).sum()
            if inf_count > 0:
                self.errors.append(f"Outcome has {inf_count} infinite values")

        # Check outcome type consistency
        if outcome.outcome_type == "binary":
            unique_vals = np.unique(np.asarray(values))
            if len(unique_vals) != 2:
                self.errors.append(
                    f"Binary outcome should have exactly 2 unique values, found {len(unique_vals)}"
                )
            elif not all(val in [0, 1] for val in unique_vals):
                self.warnings.append(
                    f"Binary outcome values are not 0/1: {unique_vals}"
                )

        elif outcome.outcome_type == "continuous":
            if len(values) > 0:
                # Check for outliers (values > outlier_threshold standard deviations from mean)
                values_array = np.asarray(values)
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                if std_val > 0:
                    outliers = (
                        np.abs(values_array - mean_val)
                        > self.outlier_threshold * std_val
                    )
                    outlier_count = outliers.sum()
                    if outlier_count > 0:
                        pct_outliers = 100 * outlier_count / len(values_array)
                        self.warnings.append(
                            f"Outcome has {outlier_count} potential outliers ({pct_outliers:.1f}%)"
                        )

        if self.verbose:
            print(f"  - {total_count:,} observations")
            print(f"  - {missing_count} missing values")
            if len(values) > 0:
                values_array = np.asarray(values)
                print(
                    f"  - Range: [{np.min(values_array):.3f}, {np.max(values_array):.3f}]"
                )
                print(
                    f"  - Mean: {np.mean(values_array):.3f}, Std: {np.std(values_array):.3f}"
                )

    def validate_covariate_data(self, covariates: CovariateData) -> None:
        """Validate covariate data.

        Args:
            covariates: Covariate data to validate
        """
        if self.verbose:
            print("Validating covariate data...")

        if isinstance(covariates.values, pd.DataFrame):
            df = covariates.values
            total_count = len(df)

            # Check missing data per variable
            missing_by_var = df.isnull().sum()
            vars_with_missing = missing_by_var[missing_by_var > 0]

            if len(vars_with_missing) > 0:
                for var, missing_count in vars_with_missing.items():
                    pct_missing = 100 * missing_count / total_count
                    msg = f"Covariate '{var}' has {missing_count} missing values ({pct_missing:.1f}%)"
                    if pct_missing > 25.0:
                        self.errors.append(msg)
                    elif pct_missing > 10.0:
                        self.warnings.append(msg)

            # Check for constant variables
            for col in df.columns:
                if df[col].dtype.name in ["int64", "float64"]:
                    if df[col].nunique() == 1:
                        self.warnings.append(f"Covariate '{col}' is constant")

            # Check for highly correlated variables
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # For large datasets, sample data for correlation check to improve performance
                if (
                    self.sample_size_for_checks
                    and len(df) > self.sample_size_for_checks
                ):
                    sample_df = df.sample(
                        n=self.sample_size_for_checks, random_state=42
                    )
                    corr_matrix = sample_df[numeric_cols].corr().abs()
                    if self.verbose:
                        print(
                            f"  - Correlation check performed on sample of {self.sample_size_for_checks:,} observations"
                        )
                else:
                    corr_matrix = df[numeric_cols].corr().abs()
                # Find pairs with correlation > 0.95
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if (
                            pd.notna(corr_val)
                            and isinstance(corr_val, (int, float))  # noqa: UP038
                            and corr_val > 0.95
                        ):
                            pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                            high_corr_pairs.append((pair, corr_val))

                for (var1, var2), corr_val in high_corr_pairs:
                    self.warnings.append(
                        f"Covariates '{var1}' and '{var2}' are highly correlated (r={corr_val:.3f})"
                    )

        else:
            # Handle numpy array case
            if len(covariates.values.shape) == 2:
                n_obs, n_vars = covariates.values.shape
                missing_count = np.isnan(covariates.values).sum()

                if missing_count > 0:
                    pct_missing = 100 * missing_count / (n_obs * n_vars)
                    self.warnings.append(
                        f"Covariates have {missing_count} missing values ({pct_missing:.1f}%)"
                    )

        if self.verbose:
            if isinstance(covariates.values, pd.DataFrame):
                print(f"  - {len(covariates.values):,} observations")
                print(f"  - {len(covariates.values.columns)} variables")
                print(f"  - Variables: {list(covariates.values.columns)}")
            else:
                print(f"  - Shape: {covariates.values.shape}")

    def validate_overlap(
        self,
        treatment: TreatmentData,
        covariates: CovariateData,
        min_overlap: float = 0.1,
    ) -> None:
        """Check for sufficient overlap in covariate distributions between treatment groups.

        Args:
            treatment: Treatment data
            covariates: Covariate data
            min_overlap: Minimum required overlap proportion
        """
        if self.verbose:
            print("Checking overlap assumption...")

        if treatment.treatment_type != "binary":
            self.warnings.append("Overlap check only implemented for binary treatments")
            return

        # Simple overlap check: ensure both treatment groups are present in most covariate strata
        if isinstance(covariates.values, pd.DataFrame):
            df = covariates.values.copy()
            df["treatment"] = treatment.values

            # For large datasets, sample data for overlap check to improve performance
            if self.sample_size_for_checks and len(df) > self.sample_size_for_checks:
                df = df.sample(n=self.sample_size_for_checks, random_state=42)
                if self.verbose:
                    print(
                        f"  - Overlap check performed on sample of {self.sample_size_for_checks:,} observations"
                    )

            # For continuous variables, create quintiles
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != "treatment" and df[col].nunique() > 10:
                    try:
                        df[f"{col}_quintile"] = pd.qcut(
                            df[col], q=5, labels=False, duplicates="drop"
                        )
                    except (ValueError, IndexError) as e:
                        # Handle case where qcut fails (insufficient unique values, etc.)
                        if self.verbose:
                            print(
                                f"Warning: Could not create quintiles for '{col}': {e}"
                            )
                        continue

            # Check categorical variables for overlap
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            for col in categorical_cols:
                if col != "treatment":
                    overlap_check = df.groupby(col)["treatment"].apply(
                        lambda x: (x == 0).any() and (x == 1).any()
                    )
                    overlap_rate = overlap_check.mean()

                    if overlap_rate < min_overlap:
                        self.warnings.append(
                            f"Poor overlap in '{col}': only {overlap_rate:.1%} of strata have both treatment groups"
                        )

        if self.verbose:
            print("  - Overlap check completed")

    def validate_all(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Optional[CovariateData] = None,
        check_overlap: bool = True,
    ) -> None:
        """Run all validation checks.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Optional covariate data
            check_overlap: Whether to check overlap assumption

        Raises:
            DataValidationError: If critical validation errors are found
        """
        self.warnings.clear()
        self.errors.clear()

        if self.verbose:
            print("=== Causal Data Validation ===")

        # Validate individual data components
        self.validate_treatment_data(treatment)
        self.validate_outcome_data(outcome)

        if covariates is not None:
            self.validate_covariate_data(covariates)

            if check_overlap:
                self.validate_overlap(treatment, covariates)

        # Check sample size alignment
        treatment_n = len(treatment.values)
        outcome_n = len(outcome.values)

        if treatment_n != outcome_n:
            self.errors.append(
                f"Sample size mismatch: treatment ({treatment_n}) vs outcome ({outcome_n})"
            )

        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                covariate_n = len(covariates.values)
            else:
                covariate_n = covariates.values.shape[0]

            if covariate_n != treatment_n:
                self.errors.append(
                    f"Sample size mismatch: treatment ({treatment_n}) vs covariates ({covariate_n})"
                )

        # Print summary
        if self.verbose:
            print("\n=== Validation Summary ===")
            if len(self.errors) == 0 and len(self.warnings) == 0:
                print("✅ All validation checks passed!")
            else:
                if len(self.errors) > 0:
                    print(f"❌ {len(self.errors)} ERROR(S):")
                    for error in self.errors:
                        print(f"   - {error}")

                if len(self.warnings) > 0:
                    print(f"⚠️  {len(self.warnings)} WARNING(S):")
                    for warning in self.warnings:
                        print(f"   - {warning}")

        # Raise exception if there are errors
        if len(self.errors) > 0:
            error_msg = f"Data validation failed with {len(self.errors)} error(s):\n"
            error_msg += "\n".join(f"- {error}" for error in self.errors)
            raise DataValidationError(error_msg)


def validate_causal_data(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: Optional[CovariateData] = None,
    check_overlap: bool = True,
    verbose: bool = True,
    outlier_threshold: float = 5.0,
    sample_size_for_checks: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """Convenience function to validate causal inference data.

    Args:
        treatment: Treatment data
        outcome: Outcome data
        covariates: Optional covariate data
        check_overlap: Whether to check overlap assumption
        verbose: Whether to print validation details
        outlier_threshold: Number of standard deviations for outlier detection
        sample_size_for_checks: If provided, sample this many observations for expensive checks

    Returns:
        Tuple of (warnings, errors) lists

    Raises:
        DataValidationError: If critical validation errors are found
    """
    validator = CausalDataValidator(
        verbose=verbose,
        outlier_threshold=outlier_threshold,
        sample_size_for_checks=sample_size_for_checks,
    )
    validator.validate_all(treatment, outcome, covariates, check_overlap)
    return validator.warnings, validator.errors
