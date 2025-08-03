"""Longitudinal data structures for time-varying causal inference.

This module provides data models and utilities for handling longitudinal
(panel) data with time-varying treatments, outcomes, and confounders.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "LongitudinalData",
    "TreatmentStrategy",
    "TimeVaryingTreatmentData",
    "TimeVaryingOutcomeData",
    "TimeVaryingCovariateData",
]


class TimeVaryingTreatmentData(BaseModel):
    """Data model for time-varying treatment assignments.

    Represents treatment assignments that can change over time in
    longitudinal causal inference analyses.
    """

    values: pd.DataFrame = Field(
        ..., description="Treatment values with columns for each time period"
    )
    treatment_names: list[str] = Field(
        default_factory=list, description="Names of treatment variables"
    )
    time_periods: list[int | str] = Field(
        default_factory=list, description="Time period labels"
    )
    treatment_type: str = Field(
        default="binary",
        description="Type of treatment: 'binary', 'categorical', or 'continuous'",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("treatment_type")
    @classmethod
    def validate_treatment_type(cls, v: str) -> str:
        """Validate treatment type is one of allowed values."""
        allowed_types = {"binary", "categorical", "continuous"}
        if v not in allowed_types:
            raise ValueError(f"treatment_type must be one of {allowed_types}")
        return v


class TimeVaryingOutcomeData(BaseModel):
    """Data model for time-varying outcome variables.

    Represents outcomes that can be measured at multiple time points
    in longitudinal causal inference analyses.
    """

    values: pd.DataFrame = Field(
        ..., description="Outcome values with columns for each time period"
    )
    outcome_names: list[str] = Field(
        default_factory=list, description="Names of outcome variables"
    )
    time_periods: list[int | str] = Field(
        default_factory=list, description="Time period labels"
    )
    outcome_type: str = Field(
        default="continuous",
        description="Type of outcome: 'continuous', 'binary', 'count'",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("outcome_type")
    @classmethod
    def validate_outcome_type(cls, v: str) -> str:
        """Validate outcome type is one of allowed values."""
        allowed_types = {"continuous", "binary", "count"}
        if v not in allowed_types:
            raise ValueError(f"outcome_type must be one of {allowed_types}")
        return v


class TimeVaryingCovariateData(BaseModel):
    """Data model for time-varying covariate variables.

    Represents covariates/confounders that can change over time
    in longitudinal causal inference analyses.
    """

    values: pd.DataFrame = Field(
        ..., description="Covariate values with columns for each time period"
    )
    covariate_names: list[str] = Field(
        default_factory=list, description="Names of covariate variables"
    )
    time_periods: list[int | str] = Field(
        default_factory=list, description="Time period labels"
    )

    model_config = {"arbitrary_types_allowed": True}


# Type alias for treatment strategy functions
TreatmentStrategy = Callable[[pd.DataFrame, int | str], NDArray[Any]]


class LongitudinalData(BaseModel):
    """Data structure for longitudinal (panel) causal inference.

    This class manages longitudinal data with time-varying treatments,
    outcomes, and covariates. It provides methods for data validation,
    transformation, and extraction of subsets for analysis.

    The data is expected to be in long format with one row per individual
    per time period, containing columns for individual ID, time period,
    treatment(s), outcome(s), and covariate(s).
    """

    data: pd.DataFrame = Field(..., description="Longitudinal data in long format")
    id_col: str = Field(..., description="Column name for individual identifiers")
    time_col: str = Field(..., description="Column name for time periods")
    treatment_cols: list[str] = Field(
        ..., description="Column names for treatment variables"
    )
    outcome_cols: list[str] = Field(
        ..., description="Column names for outcome variables"
    )
    confounder_cols: list[str] = Field(
        default_factory=list,
        description="Column names for time-varying confounder variables",
    )
    baseline_cols: list[str] = Field(
        default_factory=list,
        description="Column names for baseline (time-invariant) variables",
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        """Initialize longitudinal data with validation."""
        super().__init__(**data)
        self._validate_data_structure()
        self._sort_data()

    def _validate_data_structure(self) -> None:
        """Validate the longitudinal data structure."""
        # Check required columns exist
        required_cols = (
            [self.id_col, self.time_col] + self.treatment_cols + self.outcome_cols
        )
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for missing individual IDs
        if self.data[self.id_col].isna().any():
            raise ValueError("Individual IDs cannot be missing")

        # Check for missing time periods
        if self.data[self.time_col].isna().any():
            raise ValueError("Time periods cannot be missing")

        # Validate panel structure (balanced panel preferred but not required)
        id_time_counts = self.data.groupby([self.id_col, self.time_col]).size()
        if (id_time_counts > 1).any():
            raise ValueError("Duplicate observations for individual-time combinations")

    def _sort_data(self) -> None:
        """Sort data by individual ID and time period."""
        self.data = self.data.sort_values([self.id_col, self.time_col]).reset_index(
            drop=True
        )

    @property
    def n_individuals(self) -> int:
        """Number of unique individuals in the data."""
        return int(self.data[self.id_col].nunique())

    @property
    def n_time_periods(self) -> int:
        """Number of unique time periods in the data."""
        return int(self.data[self.time_col].nunique())

    @property
    def time_periods(self) -> list[int | str]:
        """List of unique time periods."""
        return sorted(self.data[self.time_col].unique())

    @property
    def individuals(self) -> list[int | str]:
        """List of unique individual IDs."""
        return sorted(self.data[self.id_col].unique())

    @property
    def is_balanced_panel(self) -> bool:
        """Check if this is a balanced panel (all individuals observed at all times)."""
        expected_obs = self.n_individuals * self.n_time_periods
        actual_obs = len(self.data)
        return expected_obs == actual_obs

    def get_treatment_data_at_time(
        self, time_period: int | str, treatment_col: str | None = None
    ) -> pd.Series:
        """Get treatment data for a specific time period.

        Args:
            time_period: Time period to extract
            treatment_col: Specific treatment column (uses first if None)

        Returns:
            Series with treatment values for the specified time period
        """
        if treatment_col is None:
            treatment_col = self.treatment_cols[0]

        subset = self.data[self.data[self.time_col] == time_period]
        return subset[treatment_col]

    def get_outcome_data_at_time(
        self, time_period: int | str, outcome_col: str | None = None
    ) -> pd.Series:
        """Get outcome data for a specific time period.

        Args:
            time_period: Time period to extract
            outcome_col: Specific outcome column (uses first if None)

        Returns:
            Series with outcome values for the specified time period
        """
        if outcome_col is None:
            outcome_col = self.outcome_cols[0]

        subset = self.data[self.data[self.time_col] == time_period]
        return subset[outcome_col]

    def get_confounder_data_at_time(self, time_period: int | str) -> pd.DataFrame:
        """Get confounder data for a specific time period.

        Args:
            time_period: Time period to extract

        Returns:
            DataFrame with confounder values for the specified time period
        """
        subset = self.data[self.data[self.time_col] == time_period]
        all_confounders = self.confounder_cols + self.baseline_cols
        return subset[all_confounders]

    def get_individual_trajectory(self, individual_id: int | str) -> pd.DataFrame:
        """Get complete trajectory for a specific individual.

        Args:
            individual_id: Individual ID to extract

        Returns:
            DataFrame with all observations for the specified individual
        """
        return self.data[self.data[self.id_col] == individual_id].copy()

    def apply_treatment_strategy(
        self, strategy: TreatmentStrategy, strategy_name: str = "strategy"
    ) -> pd.DataFrame:
        """Apply a treatment strategy to generate counterfactual treatments.

        Args:
            strategy: Function that takes (data, time) and returns treatment assignments
            strategy_name: Name for the strategy column

        Returns:
            DataFrame with original data plus strategy treatment column
        """
        result_data = self.data.copy()
        strategy_treatments: list[Any] = []

        for time_period in self.time_periods:
            time_data = self.data[self.data[self.time_col] == time_period]
            strategy_treatment = strategy(time_data, time_period)
            strategy_treatments.extend(strategy_treatment)

        result_data[f"{strategy_name}_treatment"] = strategy_treatments
        return result_data

    def check_sequential_exchangeability(
        self, treatment_col: str | None = None
    ) -> dict[str, Any]:
        """Check indicators of sequential exchangeability assumption.

        This performs basic checks for the sequential exchangeability assumption,
        though full verification requires domain knowledge and design considerations.

        Args:
            treatment_col: Treatment column to analyze (uses first if None)

        Returns:
            Dictionary with exchangeability indicators
        """
        if treatment_col is None:
            treatment_col = self.treatment_cols[0]

        results: dict[str, Any] = {
            "baseline_balance": {},
            "time_varying_balance": {},
            "treatment_confounder_feedback": {},
            "positivity_by_period": {},
        }

        # Check baseline balance (exchangeability at t=0)
        if self.baseline_cols:
            baseline_time = self.time_periods[0]
            baseline_data = self.data[self.data[self.time_col] == baseline_time]

            for baseline_var in self.baseline_cols:
                treated = baseline_data[baseline_data[treatment_col] == 1][baseline_var]
                control = baseline_data[baseline_data[treatment_col] == 0][baseline_var]

                if len(treated) > 0 and len(control) > 0:
                    # Simple mean difference test
                    mean_diff = treated.mean() - control.mean()
                    results["baseline_balance"][baseline_var] = {
                        "mean_difference": mean_diff,
                        "treated_mean": treated.mean(),
                        "control_mean": control.mean(),
                    }

        # Check positivity by time period
        for time_period in self.time_periods:
            time_data = self.data[self.data[self.time_col] == time_period]
            treatment_data = time_data[treatment_col]

            n_treated = (treatment_data == 1).sum()
            n_control = (treatment_data == 0).sum()
            total = len(treatment_data)

            results["positivity_by_period"][time_period] = {
                "n_treated": n_treated,
                "n_control": n_control,
                "total": total,
                "prop_treated": n_treated / total if total > 0 else 0,
                "prop_control": n_control / total if total > 0 else 0,
                "positivity_violation": (n_treated < 5) or (n_control < 5),
            }

        return results

    def test_treatment_confounder_feedback(
        self, treatment_col: str | None = None
    ) -> dict[str, Any]:
        """Test for treatment-confounder feedback.

        This tests whether past treatments predict future confounders,
        which would indicate treatment-confounder feedback.

        Args:
            treatment_col: Treatment column to analyze (uses first if None)

        Returns:
            Dictionary with feedback test results
        """
        if treatment_col is None:
            treatment_col = self.treatment_cols[0]

        results: dict[str, Any] = {
            "feedback_detected": False,
            "feedback_strength": {},
            "significant_associations": [],
        }

        # For each time period > 0, test if previous treatments predict current confounders
        time_periods = self.time_periods
        if len(time_periods) < 2:
            return results

        for i in range(1, len(time_periods)):
            current_time = time_periods[i]
            previous_time = time_periods[i - 1]

            # Get data for both time periods
            current_data = self.data[self.data[self.time_col] == current_time]
            previous_data = self.data[self.data[self.time_col] == previous_time]

            # Merge on individual ID
            merged = pd.merge(
                current_data[[self.id_col] + self.confounder_cols],
                previous_data[[self.id_col, treatment_col]],
                on=self.id_col,
                suffixes=("_current", "_previous"),
            )

            # Test association between previous treatment and current confounders
            previous_treatment_col = f"{treatment_col}_previous"
            for confounder in self.confounder_cols:
                if (
                    confounder in merged.columns
                    and previous_treatment_col in merged.columns
                ):
                    # Simple correlation test
                    corr = merged[previous_treatment_col].corr(merged[confounder])

                    if abs(corr) > 0.1:  # Threshold for meaningful association
                        results["feedback_detected"] = True
                        if isinstance(results["significant_associations"], list):
                            results["significant_associations"].append(
                                {
                                    "previous_treatment": treatment_col,
                                    "current_confounder": confounder,
                                    "time_lag": f"{previous_time} -> {current_time}",
                                    "correlation": corr,
                                }
                            )

                    results["feedback_strength"][f"{confounder}_{current_time}"] = corr

        return results

    def to_wide_format(self) -> pd.DataFrame:
        """Convert longitudinal data to wide format.

        Returns:
            DataFrame in wide format with one row per individual
        """
        # Identify columns to pivot
        value_cols = self.treatment_cols + self.outcome_cols + self.confounder_cols

        # Create wide format
        wide_data = self.data.pivot(
            index=self.id_col, columns=self.time_col, values=value_cols
        )

        # Flatten column names
        if isinstance(wide_data.columns, pd.MultiIndex):
            wide_data.columns = [f"{col[0]}_t{col[1]}" for col in wide_data.columns]

        # Add baseline variables (they should be the same across time periods)
        if self.baseline_cols:
            baseline_data = self.data.groupby(self.id_col)[self.baseline_cols].first()
            wide_data = pd.concat([wide_data, baseline_data], axis=1)

        return wide_data.reset_index()

    def summary(self) -> str:
        """Provide a summary of the longitudinal data structure.

        Returns:
            String summary of the data
        """
        summary_lines = [
            "Longitudinal Data Summary",
            "=" * 30,
            f"Individuals: {self.n_individuals}",
            f"Time periods: {self.n_time_periods}",
            f"Total observations: {len(self.data)}",
            f"Balanced panel: {self.is_balanced_panel}",
            f"Treatment variables: {', '.join(self.treatment_cols)}",
            f"Outcome variables: {', '.join(self.outcome_cols)}",
            f"Time-varying confounders: {', '.join(self.confounder_cols)}",
            f"Baseline variables: {', '.join(self.baseline_cols)}",
        ]

        return "\n".join(summary_lines)
