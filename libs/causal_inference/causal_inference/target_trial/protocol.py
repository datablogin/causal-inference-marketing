"""Target trial protocol specification and validation.

This module implements the TargetTrialProtocol class for specifying the hypothetical
randomized trial that would answer the causal question of interest.
"""

from __future__ import annotations

from typing import Any, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class EligibilityCriteria(BaseModel):
    """Eligibility criteria for target trial participants."""

    age_min: Union[int, None] = Field(None, description="Minimum age for eligibility")
    age_max: Union[int, None] = Field(None, description="Maximum age for eligibility")
    baseline_smoker: Union[bool, None] = Field(
        None, description="Smoking status requirement"
    )
    no_missing_weight: Union[bool, None] = Field(
        None, description="Require complete weight data"
    )
    custom_criteria: dict[str, Any] = Field(
        default_factory=dict, description="Custom eligibility criteria"
    )

    @field_validator("age_min", "age_max")
    @classmethod
    def validate_age(cls, v: Union[int, None]) -> Union[int, None]:
        """Validate age values are reasonable."""
        if v is not None and (v < 0 or v > 120):
            raise ValueError("Age must be between 0 and 120")
        return v

    def check_eligibility(self, data: pd.DataFrame) -> pd.Series:
        """Check which participants meet eligibility criteria.

        Args:
            data: DataFrame with participant data

        Returns:
            Boolean Series indicating eligibility
        """
        eligible = pd.Series(True, index=data.index)

        # Age criteria
        if self.age_min is not None and "age" in data.columns:
            eligible &= data["age"] >= self.age_min
        if self.age_max is not None and "age" in data.columns:
            eligible &= data["age"] <= self.age_max

        # Smoking status
        if self.baseline_smoker is not None and "smoker" in data.columns:
            eligible &= data["smoker"] == self.baseline_smoker

        # Missing weight data
        if self.no_missing_weight and "wt82_71" in data.columns:
            eligible &= data["wt82_71"].notna()

        # Custom criteria
        for column, criteria in self.custom_criteria.items():
            if column in data.columns:
                if isinstance(criteria, (list, tuple)):
                    # Range criteria
                    if len(criteria) == 2:
                        eligible &= (data[column] >= criteria[0]) & (
                            data[column] <= criteria[1]
                        )
                elif isinstance(criteria, dict):
                    # Dictionary criteria (e.g., {'in': [1, 2, 3]})
                    if "in" in criteria:
                        eligible &= data[column].isin(criteria["in"])
                    elif "not_in" in criteria:
                        eligible &= ~data[column].isin(criteria["not_in"])
                else:
                    # Exact match
                    eligible &= data[column] == criteria

        return eligible


class TreatmentStrategy(BaseModel):
    """Treatment strategy specification."""

    treatment_assignment: dict[str, Any] = Field(
        ..., description="Treatment variable assignments"
    )
    sustained: bool = Field(
        default=False, description="Whether treatment must be sustained"
    )
    grace_period_days: int = Field(
        default=0, description="Grace period for treatment initiation"
    )

    def apply_strategy(self, data: pd.DataFrame, treatment_col: str) -> pd.Series:
        """Apply treatment strategy to determine treatment assignment.

        Args:
            data: DataFrame with participant data
            treatment_col: Name of treatment column

        Returns:
            Boolean Series indicating treatment assignment
        """
        if treatment_col in self.treatment_assignment:
            target_value = self.treatment_assignment[treatment_col]
            comparison_result = data[treatment_col] == target_value
            # Ensure we return a proper pandas Series
            if isinstance(comparison_result, pd.Series):
                return comparison_result
            else:
                return pd.Series(comparison_result, dtype=bool)
        return pd.Series(False, index=data.index)

    def get_assigned_treatment_value(self, treatment_col: str) -> Any:
        """Get the treatment value assigned by this strategy.

        Args:
            treatment_col: Name of treatment column

        Returns:
            The treatment value this strategy assigns
        """
        return self.treatment_assignment.get(treatment_col, None)


class FollowUpPeriod(BaseModel):
    """Follow-up period specification."""

    duration: int = Field(..., description="Duration of follow-up")
    unit: str = Field("years", description="Time unit (years, months, days)")

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        """Validate time unit."""
        allowed_units = {"years", "months", "days"}
        if v not in allowed_units:
            raise ValueError(f"unit must be one of {allowed_units}")
        return v

    def to_days(self) -> int:
        """Convert follow-up period to days."""
        multipliers = {"years": 365, "months": 30, "days": 1}
        return self.duration * multipliers[self.unit]


class GracePeriod(BaseModel):
    """Grace period specification for treatment initiation."""

    duration: int = Field(..., description="Duration of grace period")
    unit: str = Field("months", description="Time unit (years, months, days)")

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        """Validate time unit."""
        allowed_units = {"years", "months", "days"}
        if v not in allowed_units:
            raise ValueError(f"unit must be one of {allowed_units}")
        return v

    def to_days(self) -> int:
        """Convert grace period to days."""
        multipliers = {"years": 365, "months": 30, "days": 1}
        return self.duration * multipliers[self.unit]


class TargetTrialProtocol(BaseModel):
    """Specification of a target trial protocol.

    This class defines all components of the hypothetical randomized trial
    that would ideally answer the causal question of interest.
    """

    # Core protocol components
    eligibility_criteria: EligibilityCriteria = Field(
        default_factory=lambda: EligibilityCriteria(),  # type: ignore[call-arg]
        description="Eligibility criteria for trial participants",
    )
    treatment_strategies: dict[str, TreatmentStrategy] = Field(
        ..., description="Treatment strategies to compare"
    )
    assignment_procedure: str = Field(
        "randomized", description="How treatments would be assigned"
    )
    follow_up_period: FollowUpPeriod = Field(
        ..., description="Duration and timing of follow-up"
    )
    primary_outcome: str = Field(..., description="Primary outcome variable name")
    time_zero_definition: str = Field(
        "baseline", description="Definition of time zero for the trial"
    )
    grace_period: Union[GracePeriod, None] = Field(
        None, description="Grace period for treatment initiation"
    )

    # Secondary outcomes and analysis details
    secondary_outcomes: list[str] = Field(
        default_factory=list, description="Secondary outcome variables"
    )
    adherence_definition: dict[str, str] = Field(
        default_factory=dict, description="Definition of adherence for each strategy"
    )
    censoring_events: list[str] = Field(
        default_factory=list, description="Events that lead to censoring"
    )

    @field_validator("assignment_procedure")
    @classmethod
    def validate_assignment_procedure(cls, v: str) -> str:
        """Validate assignment procedure."""
        allowed_procedures = {"randomized", "stratified_randomized", "sequential"}
        if v not in allowed_procedures:
            raise ValueError(
                f"assignment_procedure must be one of {allowed_procedures}"
            )
        return v

    @field_validator("treatment_strategies")
    @classmethod
    def validate_treatment_strategies(
        cls, v: dict[str, TreatmentStrategy]
    ) -> dict[str, TreatmentStrategy]:
        """Validate treatment strategies."""
        if len(v) < 2:
            raise ValueError(
                "Must specify at least 2 treatment strategies for comparison"
            )
        return v

    def check_feasibility(self, data: pd.DataFrame) -> dict[str, Any]:
        """Check feasibility of the protocol with given data.

        Args:
            data: Observational data to check against protocol

        Returns:
            Dictionary with feasibility assessment
        """
        results: dict[str, Any] = {
            "is_feasible": True,
            "warnings": [],
            "infeasibility_reasons": [],
            "eligible_sample_size": 0,
            "treatment_group_sizes": {},
            "outcome_availability": {},
        }

        # Check eligibility criteria
        eligible = self.eligibility_criteria.check_eligibility(data)
        eligible_data = data[eligible]
        results["eligible_sample_size"] = len(eligible_data)

        if len(eligible_data) < 100:
            results["is_feasible"] = False
            results["infeasibility_reasons"].append("insufficient_sample_size")
        elif len(eligible_data) < 500:
            results["warnings"].append("small_sample_size")

        # Check treatment group sizes
        for strategy_name, strategy in self.treatment_strategies.items():
            if (
                "qsmk" in strategy.treatment_assignment
            ):  # Assuming smoking cessation example
                treatment_col = "qsmk"
                if treatment_col in eligible_data.columns:
                    assigned = strategy.apply_strategy(eligible_data, treatment_col)
                    group_size = assigned.sum()
                    results["treatment_group_sizes"][strategy_name] = int(group_size)

                    if group_size < 50:
                        results["is_feasible"] = False
                        results["infeasibility_reasons"].append(
                            f"insufficient_{strategy_name}_group_size"
                        )

        # Check outcome availability
        if self.primary_outcome in data.columns:
            outcome_available = data[self.primary_outcome].notna()
            eligible_with_outcome = eligible & outcome_available
            results["outcome_availability"]["primary"] = int(
                eligible_with_outcome.sum()
            )

            if eligible_with_outcome.sum() < len(eligible_data) * 0.8:
                results["warnings"].append("high_missing_outcome_data")
        else:
            results["is_feasible"] = False
            results["infeasibility_reasons"].append("primary_outcome_not_available")

        # Check for temporal ordering (if time variables available)
        if "baseline_date" in data.columns and "outcome_date" in data.columns:
            temporal_order = (data["outcome_date"] > data["baseline_date"]).all()
            if not temporal_order:
                results["warnings"].append("temporal_ordering_issues")

        return results

    def get_protocol_summary(self) -> str:
        """Generate a human-readable summary of the protocol.

        Returns:
            String summary of the trial protocol
        """
        lines = [
            "Target Trial Protocol Summary",
            "=" * 40,
            "",
            f"Primary Outcome: {self.primary_outcome}",
            f"Follow-up Period: {self.follow_up_period.duration} {self.follow_up_period.unit}",
            f"Assignment: {self.assignment_procedure}",
            "",
            "Treatment Strategies:",
        ]

        for name, strategy in self.treatment_strategies.items():
            lines.append(f"  - {name}: {strategy.treatment_assignment}")

        lines.extend(
            [
                "",
                "Eligibility Criteria:",
            ]
        )

        criteria = self.eligibility_criteria
        if criteria.age_min is not None or criteria.age_max is not None:
            age_range = f"{criteria.age_min or 'any'}-{criteria.age_max or 'any'} years"
            lines.append(f"  - Age: {age_range}")

        if criteria.baseline_smoker is not None:
            smoker_status = (
                "smokers only" if criteria.baseline_smoker else "non-smokers only"
            )
            lines.append(f"  - Baseline smoking: {smoker_status}")

        if criteria.no_missing_weight:
            lines.append("  - Complete weight data required")

        if self.grace_period:
            lines.extend(
                [
                    "",
                    f"Grace Period: {self.grace_period.duration} {self.grace_period.unit}",
                ]
            )

        return "\n".join(lines)

    def validate_against_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Validate protocol specification against available data.

        Args:
            data: Available observational data

        Returns:
            Dictionary with validation results
        """
        validation: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_variables": [],
        }

        # Check for required variables
        required_vars = [self.primary_outcome]

        # Extract treatment variables from strategies
        for strategy in self.treatment_strategies.values():
            required_vars.extend(strategy.treatment_assignment.keys())

        # Check if variables exist in data
        missing_vars = [var for var in required_vars if var not in data.columns]
        if missing_vars:
            validation["valid"] = False
            validation["missing_variables"] = missing_vars
            validation["errors"].extend(
                [f"Missing variable: {var}" for var in missing_vars]
            )

        # Check eligibility criteria variables
        if (
            self.eligibility_criteria.age_min is not None
            or self.eligibility_criteria.age_max is not None
        ):
            if "age" not in data.columns:
                validation["warnings"].append(
                    "Age criteria specified but 'age' column not found"
                )

        if self.eligibility_criteria.baseline_smoker is not None:
            if "smoker" not in data.columns:
                validation["warnings"].append(
                    "Smoking criteria specified but 'smoker' column not found"
                )

        return validation
