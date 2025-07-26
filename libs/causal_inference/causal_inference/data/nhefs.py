"""NHEFS (National Health and Nutrition Examination Survey Epidemiologic Follow-up Study) dataset utilities.

The NHEFS dataset is a longitudinal study that is commonly used for causal inference examples,
particularly in the "Causal Inference: What If" book by Hernán and Robins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..core.base import CovariateData, OutcomeData, TreatmentData


class NHEFSDataLoader:
    """Loader for the NHEFS dataset with proper preprocessing and validation.

    The NHEFS dataset contains information about smoking cessation and weight change
    among participants in the National Health and Nutrition Examination Survey.

    Key variables:
    - qsmk: Treatment (1 = quit smoking, 0 = continued smoking)
    - wt82_71: Outcome (weight change from 1971 to 1982)
    - Various covariates for confounding adjustment
    """

    def __init__(self, data_path: str | None = None):
        """Initialize the NHEFS data loader.

        Args:
            data_path: Path to NHEFS CSV file. If None, will search for 'nhefs.csv'
                      in the current directory and parent directories.
        """
        self.data_path = self._find_nhefs_file(data_path)
        self._raw_data: pd.DataFrame | None = None
        self._processed_data: pd.DataFrame | None = None

    def _find_nhefs_file(self, data_path: str | None) -> Path:
        """Find the NHEFS dataset file.

        Args:
            data_path: Explicitly provided path, or None to search

        Returns:
            Path to the NHEFS CSV file

        Raises:
            FileNotFoundError: If NHEFS file cannot be found
        """
        if data_path:
            path = Path(data_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"NHEFS data file not found at: {data_path}")

        # Search in current directory and parent directories
        search_paths = [
            Path.cwd() / "nhefs.csv",
            Path.cwd().parent / "nhefs.csv",
            Path.cwd().parent.parent / "nhefs.csv",
            Path.cwd().parent.parent.parent / "nhefs.csv",
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            "NHEFS data file 'nhefs.csv' not found. Please provide explicit path "
            "or ensure the file is in the current directory or parent directories."
        )

    def load_raw_data(self) -> pd.DataFrame:
        """Load the raw NHEFS dataset.

        Returns:
            Raw NHEFS dataframe with all original columns
        """
        if self._raw_data is None:
            self._raw_data = pd.read_csv(self.data_path)

        return self._raw_data.copy()

    def load_processed_data(
        self,
        outcome: str = "wt82_71",
        treatment: str = "qsmk",
        confounders: list[str] | None = None,
        exclude_missing_outcome: bool = True,
        exclude_missing_treatment: bool = True,
    ) -> pd.DataFrame:
        """Load and preprocess the NHEFS dataset for causal inference.

        Args:
            outcome: Name of outcome variable (default: weight change)
            treatment: Name of treatment variable (default: smoking cessation)
            confounders: List of confounder variables. If None, uses standard set.
            exclude_missing_outcome: Whether to exclude observations with missing outcomes
            exclude_missing_treatment: Whether to exclude observations with missing treatment

        Returns:
            Processed NHEFS dataframe ready for causal inference
        """
        if confounders is None:
            # Standard confounders from Hernán & Robins book
            confounders = [
                "sex",
                "age",
                "race",
                "education",
                "smokeintensity",
                "smokeyrs",
                "exercise",
                "active",
                "wt71",
                "asthma",
                "bronch",
            ]

        # Load raw data
        df = self.load_raw_data()

        # Select relevant columns
        columns = [outcome, treatment] + confounders
        available_columns = [col for col in columns if col in df.columns]
        missing_columns = [col for col in columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns in dataset: {missing_columns}")

        df_subset = df[available_columns].copy()

        # Handle missing data
        if exclude_missing_outcome and outcome in df_subset.columns:
            initial_n = len(df_subset)
            df_subset = df_subset.dropna(subset=[outcome])
            dropped_outcome = initial_n - len(df_subset)
            if dropped_outcome > 0:
                print(
                    f"Excluded {dropped_outcome} observations with missing outcome '{outcome}'"
                )

        if exclude_missing_treatment and treatment in df_subset.columns:
            initial_n = len(df_subset)
            df_subset = df_subset.dropna(subset=[treatment])
            dropped_treatment = initial_n - len(df_subset)
            if dropped_treatment > 0:
                print(
                    f"Excluded {dropped_treatment} observations with missing treatment '{treatment}'"
                )

        # Convert treatment to binary if needed
        if treatment in df_subset.columns:
            unique_vals = df_subset[treatment].dropna().unique()
            if len(unique_vals) == 2 and not all(val in [0, 1] for val in unique_vals):
                print(f"Converting treatment '{treatment}' to binary (0/1)")
                df_subset[treatment] = (df_subset[treatment] == unique_vals[1]).astype(
                    int
                )

        self._processed_data = df_subset
        return df_subset.copy()

    def get_causal_data_objects(
        self,
        outcome: str = "wt82_71",
        treatment: str = "qsmk",
        confounders: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[TreatmentData, OutcomeData, CovariateData]:
        """Get the NHEFS data as causal inference data objects.

        Args:
            outcome: Name of outcome variable
            treatment: Name of treatment variable
            confounders: List of confounder variables
            **kwargs: Additional arguments passed to load_processed_data

        Returns:
            Tuple of (TreatmentData, OutcomeData, CovariateData) objects
        """
        df = self.load_processed_data(
            outcome=outcome, treatment=treatment, confounders=confounders, **kwargs
        )

        # Create treatment data
        treatment_data = TreatmentData(
            values=df[treatment],
            name=treatment,
            treatment_type="binary",
        )

        # Create outcome data
        outcome_data = OutcomeData(
            values=df[outcome],
            name=outcome,
            outcome_type="continuous",
        )

        # Create covariate data
        if confounders is None:
            confounders = [col for col in df.columns if col not in [outcome, treatment]]

        available_confounders = [col for col in confounders if col in df.columns]

        if available_confounders:
            covariate_data = CovariateData(
                values=df[available_confounders],
                names=available_confounders,
            )
        else:
            # Create empty covariate data if no confounders available
            covariate_data = CovariateData(
                values=pd.DataFrame(index=df.index),
                names=[],
            )

        return treatment_data, outcome_data, covariate_data

    def get_dataset_info(self) -> dict[str, Any]:
        """Get information about the NHEFS dataset.

        Returns:
            Dictionary with dataset statistics and information
        """
        df = self.load_raw_data()

        info = {
            "n_observations": len(df),
            "n_variables": len(df.columns),
            "variables": list(df.columns),
            "missing_data": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
        }

        # Add specific information about key variables
        if "qsmk" in df.columns:
            info["treatment_distribution"] = df["qsmk"].value_counts().to_dict()

        if "wt82_71" in df.columns:
            outcome_stats = df["wt82_71"].describe()
            info["outcome_statistics"] = outcome_stats.to_dict()

        return info

    def print_dataset_summary(self) -> None:
        """Print a comprehensive summary of the NHEFS dataset."""
        info = self.get_dataset_info()

        print("=== NHEFS Dataset Summary ===")
        print(f"Observations: {info['n_observations']:,}")
        print(f"Variables: {info['n_variables']}")
        print()

        if "treatment_distribution" in info:
            print("Treatment Distribution (qsmk):")
            for value, count in info["treatment_distribution"].items():
                pct = 100 * count / info["n_observations"]
                print(f"  {value}: {count:,} ({pct:.1f}%)")
            print()

        if "outcome_statistics" in info:
            print("Outcome Statistics (wt82_71):")
            stats = info["outcome_statistics"]
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std:  {stats['std']:.2f}")
            print(f"  Min:  {stats['min']:.2f}")
            print(f"  Max:  {stats['max']:.2f}")
            print()

        # Show variables with substantial missing data
        missing = {k: v for k, v in info["missing_data"].items() if v > 0}
        if missing:
            print("Variables with Missing Data:")
            for var, count in sorted(missing.items(), key=lambda x: x[1], reverse=True):
                pct = 100 * count / info["n_observations"]
                if pct >= 1.0:  # Only show if >= 1% missing
                    print(f"  {var}: {count:,} ({pct:.1f}%)")
        else:
            print("No missing data found.")


def load_nhefs(
    data_path: str | None = None,
    outcome: str = "wt82_71",
    treatment: str = "qsmk",
    confounders: list[str] | None = None,
    return_objects: bool = True,
    **kwargs: Any,
) -> tuple[TreatmentData, OutcomeData, CovariateData] | pd.DataFrame:
    """Convenience function to load NHEFS data.

    Args:
        data_path: Path to NHEFS CSV file
        outcome: Name of outcome variable
        treatment: Name of treatment variable
        confounders: List of confounder variables
        return_objects: If True, return causal data objects; if False, return DataFrame
        **kwargs: Additional arguments passed to load_processed_data

    Returns:
        Either tuple of causal data objects or processed DataFrame
    """
    loader = NHEFSDataLoader(data_path=data_path)

    if return_objects:
        return loader.get_causal_data_objects(
            outcome=outcome, treatment=treatment, confounders=confounders, **kwargs
        )
    else:
        return loader.load_processed_data(
            outcome=outcome, treatment=treatment, confounders=confounders, **kwargs
        )
