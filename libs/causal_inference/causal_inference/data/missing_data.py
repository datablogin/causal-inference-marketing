"""Missing data handling strategies for causal inference.

This module provides various approaches to handle missing data in causal inference
datasets, including listwise deletion, imputation methods, and missing data diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from ..core.base import CovariateData, OutcomeData, TreatmentData


def _create_combined_dataframe(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData | None = None,
) -> pd.DataFrame:
    """Create a combined DataFrame from treatment, outcome, and covariate data.

    Args:
        treatment: Treatment data
        outcome: Outcome data
        covariates: Optional covariate data

    Returns:
        Combined DataFrame with all data
    """
    # Create base DataFrame with treatment and outcome
    if isinstance(treatment.values, pd.Series):
        df = pd.DataFrame({"treatment": treatment.values})
    else:
        df = pd.DataFrame({"treatment": treatment.values})

    if isinstance(outcome.values, pd.Series):
        df["outcome"] = outcome.values
    else:
        df["outcome"] = outcome.values

    # Add covariates if provided
    if covariates is not None:
        if isinstance(covariates.values, pd.DataFrame):
            for col in covariates.values.columns:
                df[col] = covariates.values[col]
        else:
            # Handle numpy array
            cov_names = covariates.names or [
                f"X{i}" for i in range(covariates.values.shape[1])
            ]
            for i, name in enumerate(cov_names):
                df[name] = covariates.values[:, i]

    return df


class MissingDataHandler:
    """Handler for missing data in causal inference datasets."""

    def __init__(self, strategy: str = "listwise", verbose: bool = True):
        """Initialize the missing data handler.

        Args:
            strategy: Missing data strategy ('listwise', 'mean', 'median', 'mode',
                     'knn', 'iterative')
            verbose: Whether to print processing information
        """
        self.strategy = strategy
        self.verbose = verbose
        self.imputer: SimpleImputer | KNNImputer | IterativeImputer | None = None
        self._fitted: bool = False

        # Validate strategy
        valid_strategies = ["listwise", "mean", "median", "mode", "knn", "iterative"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Strategy must be one of {valid_strategies}, got '{strategy}'"
            )

    def _create_imputer(
        self, data: pd.DataFrame
    ) -> SimpleImputer | KNNImputer | IterativeImputer:
        """Create appropriate imputer based on strategy.

        Args:
            data: Data to fit imputer on

        Returns:
            Fitted imputer object
        """
        if self.strategy == "mean":
            return SimpleImputer(strategy="mean")
        elif self.strategy == "median":
            return SimpleImputer(strategy="median")
        elif self.strategy == "mode":
            return SimpleImputer(strategy="most_frequent")
        elif self.strategy == "knn":
            return KNNImputer(n_neighbors=5)
        elif self.strategy == "iterative":
            return IterativeImputer(random_state=42, max_iter=10)
        else:
            raise ValueError(f"No imputer available for strategy '{self.strategy}'")

    def fit(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> MissingDataHandler:
        """Fit the missing data handler on the provided data.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Optional covariate data

        Returns:
            Self for method chaining
        """
        if self.strategy == "listwise":
            # No fitting needed for listwise deletion
            self._fitted = True
            return self

        # Combine all data for imputation fitting
        df = _create_combined_dataframe(treatment, outcome, covariates)

        # Only fit imputer on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.imputer = self._create_imputer(df[numeric_cols])
            if self.imputer is not None:
                self.imputer.fit(df[numeric_cols])

        self._fitted = True
        return self

    def transform(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData | None]:
        """Apply missing data handling to the provided data.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Optional covariate data

        Returns:
            Tuple of processed (treatment, outcome, covariates) data
        """
        if not self._fitted:
            raise ValueError("Handler must be fitted before transform")

        if self.strategy == "listwise":
            return self._listwise_deletion(treatment, outcome, covariates)
        else:
            return self._impute_missing(treatment, outcome, covariates)

    def fit_transform(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData | None]:
        """Fit and transform data in one step.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Optional covariate data

        Returns:
            Tuple of processed (treatment, outcome, covariates) data
        """
        return self.fit(treatment, outcome, covariates).transform(
            treatment, outcome, covariates
        )

    def _listwise_deletion(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData | None]:
        """Apply listwise deletion to remove any observation with missing data.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Optional covariate data

        Returns:
            Tuple of data with complete cases only
        """
        # Create combined DataFrame to identify complete cases
        df = _create_combined_dataframe(treatment, outcome, covariates)

        # Identify complete cases
        initial_n = len(df)
        complete_cases = df.dropna()
        final_n = len(complete_cases)

        if self.verbose:
            dropped = initial_n - final_n
            print(
                f"Listwise deletion: removed {dropped} observations with missing data ({dropped / initial_n:.1%})"
            )
            print(f"Final sample size: {final_n:,}")

        # Extract processed data
        processed_treatment = TreatmentData(
            values=complete_cases["treatment"],
            name=treatment.name,
            treatment_type=treatment.treatment_type,
            categories=treatment.categories,
        )

        processed_outcome = OutcomeData(
            values=complete_cases["outcome"],
            name=outcome.name,
            outcome_type=outcome.outcome_type,
        )

        processed_covariates = None
        if covariates is not None:
            covariate_cols = [
                col
                for col in complete_cases.columns
                if col not in ["treatment", "outcome"]
            ]
            if covariate_cols:
                processed_covariates = CovariateData(
                    values=complete_cases[covariate_cols],
                    names=covariate_cols,
                )

        return processed_treatment, processed_outcome, processed_covariates

    def _impute_missing(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData | None]:
        """Apply imputation to handle missing data.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Optional covariate data

        Returns:
            Tuple of data with missing values imputed
        """
        # Create combined DataFrame
        df = _create_combined_dataframe(treatment, outcome, covariates)

        initial_missing = df.isnull().sum().sum()

        # Apply imputation to numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and self.imputer is not None:
            df[numeric_cols] = self.imputer.transform(df[numeric_cols])

        # Handle categorical columns with mode imputation
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = (
                    df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown"
                )
                df[col] = df[col].fillna(mode_value)

        final_missing = df.isnull().sum().sum()

        if self.verbose:
            print(
                f"Imputation ({self.strategy}): handled {initial_missing} missing values"
            )
            if final_missing > 0:
                print(
                    f"Warning: {final_missing} missing values remain after imputation"
                )

        # Extract processed data
        processed_treatment = TreatmentData(
            values=df["treatment"],
            name=treatment.name,
            treatment_type=treatment.treatment_type,
            categories=treatment.categories,
        )

        processed_outcome = OutcomeData(
            values=df["outcome"],
            name=outcome.name,
            outcome_type=outcome.outcome_type,
        )

        processed_covariates = None
        if covariates is not None:
            covariate_cols = [
                col for col in df.columns if col not in ["treatment", "outcome"]
            ]
            if covariate_cols:
                processed_covariates = CovariateData(
                    values=df[covariate_cols],
                    names=covariate_cols,
                )

        return processed_treatment, processed_outcome, processed_covariates


def diagnose_missing_data(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData | None = None,
) -> dict[str, object]:
    """Diagnose missing data patterns in causal inference dataset.

    Args:
        treatment: Treatment data
        outcome: Outcome data
        covariates: Optional covariate data

    Returns:
        Dictionary with missing data diagnostics
    """
    # Create combined DataFrame
    df = _create_combined_dataframe(treatment, outcome, covariates)

    # Calculate missing data statistics
    total_obs = len(df)
    missing_by_var = df.isnull().sum()
    complete_cases = total_obs - df.isnull().any(axis=1).sum()

    # Missing data patterns
    missing_patterns = df.isnull().value_counts()

    diagnostics = {
        "total_observations": total_obs,
        "complete_cases": complete_cases,
        "incomplete_cases": total_obs - complete_cases,
        "missing_by_variable": missing_by_var.to_dict(),
        "missing_patterns": missing_patterns.head(10).to_dict(),  # Top 10 patterns
        "any_missing": df.isnull().any().sum(),
        "all_missing": df.isnull().all().sum(),
    }

    # Calculate percentages
    diagnostics["complete_case_rate"] = complete_cases / total_obs
    missing_by_var = diagnostics["missing_by_variable"]  # type: ignore
    diagnostics["missing_rate_by_variable"] = {
        var: count / total_obs
        for var, count in missing_by_var.items()  # type: ignore
    }

    return diagnostics


def print_missing_data_report(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData | None = None,
) -> None:
    """Print a comprehensive missing data report.

    Args:
        treatment: Treatment data
        outcome: Outcome data
        covariates: Optional covariate data
    """
    diagnostics = diagnose_missing_data(treatment, outcome, covariates)

    print("=== Missing Data Report ===")
    print(f"Total observations: {diagnostics['total_observations']:,}")
    print(
        f"Complete cases: {diagnostics['complete_cases']:,} ({diagnostics['complete_case_rate']:.1%})"
    )
    print(f"Incomplete cases: {diagnostics['incomplete_cases']:,}")
    print()

    # Variables with missing data
    missing_by_var = diagnostics["missing_by_variable"]
    assert isinstance(missing_by_var, dict)
    vars_with_missing = {k: v for k, v in missing_by_var.items() if v > 0}
    if vars_with_missing:
        print("Variables with missing data:")
        for var, count in sorted(
            vars_with_missing.items(), key=lambda x: x[1], reverse=True
        ):
            missing_rate_by_var = diagnostics["missing_rate_by_variable"]
            assert isinstance(missing_rate_by_var, dict)
            pct = missing_rate_by_var[var]
            print(f"  {var}: {count:,} ({pct:.1%})")
    else:
        print("No missing data found.")
    print()

    # Missing data patterns
    missing_patterns = diagnostics["missing_patterns"]
    assert isinstance(missing_patterns, dict)
    if len(missing_patterns) > 1:
        print("Top missing data patterns:")
        for i, (pattern, count) in enumerate(missing_patterns.items()):
            total_obs = diagnostics["total_observations"]
            assert isinstance(total_obs, int)
            pct = count / total_obs
            pattern_str = ", ".join(
                [
                    f"{col}={val}"
                    for col, val in zip(
                        ["treatment", "outcome"]
                        + (covariates.names if covariates else []),
                        pattern,
                    )
                ]
            )
            print(f"  {i + 1}. {pattern_str}: {count:,} ({pct:.1%})")

    print()

    # Recommendations
    complete_rate = diagnostics["complete_case_rate"]
    assert isinstance(complete_rate, float)
    if complete_rate > 0.95:
        print(
            "✅ Recommendation: Listwise deletion is appropriate (>95% complete cases)"
        )
    elif complete_rate > 0.80:
        print("⚠️  Recommendation: Consider listwise deletion or simple imputation")
    else:
        print(
            "❌ Recommendation: Substantial missing data - consider advanced imputation methods"
        )


def handle_missing_data(
    treatment: TreatmentData,
    outcome: OutcomeData,
    covariates: CovariateData | None = None,
    strategy: str = "listwise",
    verbose: bool = True,
) -> tuple[TreatmentData, OutcomeData, CovariateData | None]:
    """Convenience function to handle missing data.

    Args:
        treatment: Treatment data
        outcome: Outcome data
        covariates: Optional covariate data
        strategy: Missing data strategy ('listwise', 'mean', 'median', 'mode', 'knn', 'iterative')
        verbose: Whether to print processing information

    Returns:
        Tuple of processed (treatment, outcome, covariates) data
    """
    handler = MissingDataHandler(strategy=strategy, verbose=verbose)
    return handler.fit_transform(treatment, outcome, covariates)
