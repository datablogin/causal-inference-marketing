"""Base classes and interfaces for causal inference estimators.

This module provides the foundational abstract classes and data models
that establish consistent patterns across all causal inference estimators.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..diagnostics.reporting import DiagnosticReport

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator


class TreatmentData(BaseModel):
    """Data model for treatment assignments.

    Represents the treatment variable(s) in a causal inference analysis.
    Supports binary, categorical, and continuous treatments.
    """

    values: pd.Series | NDArray[Any] = Field(
        ..., description="Treatment assignment values"
    )
    name: str = Field(default="treatment", description="Name of the treatment variable")
    treatment_type: str = Field(
        default="binary",
        description="Type of treatment: 'binary', 'categorical', or 'continuous'",
    )
    categories: list[str | int | float] | None = Field(
        default=None, description="For categorical treatments, the possible categories"
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

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: pd.Series | NDArray[Any]) -> pd.Series | NDArray[Any]:
        """Validate treatment values are not empty."""
        if len(v) == 0:
            raise ValueError("Treatment values cannot be empty")
        return v


class OutcomeData(BaseModel):
    """Data model for outcome variables.

    Represents the outcome variable(s) in a causal inference analysis.
    """

    values: pd.Series | NDArray[Any] = Field(..., description="Outcome values")
    name: str = Field(default="outcome", description="Name of the outcome variable")
    outcome_type: str = Field(
        default="continuous",
        description="Type of outcome: 'continuous', 'binary', or 'count'",
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

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: pd.Series | NDArray[Any]) -> pd.Series | NDArray[Any]:
        """Validate outcome values are not empty."""
        if len(v) == 0:
            raise ValueError("Outcome values cannot be empty")
        return v


class CovariateData(BaseModel):
    """Data model for covariate/confounder variables.

    Represents the covariates used for adjustment in causal inference.
    """

    values: pd.DataFrame | NDArray[Any] = Field(..., description="Covariate values")
    names: list[str] = Field(
        default_factory=list, description="Names of the covariate variables"
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("values")
    @classmethod
    def validate_values(
        cls, v: pd.DataFrame | NDArray[Any]
    ) -> pd.DataFrame | NDArray[Any]:
        """Validate covariate values are not empty."""
        if len(v) == 0:
            raise ValueError("Covariate values cannot be empty")
        return v


@dataclass
class CausalEffect:
    """Data class representing the result of a causal inference analysis.

    This standardized result format is returned by all estimators and provides
    consistent access to causal effect estimates, confidence intervals, and
    diagnostic information.
    """

    # Core estimates
    ate: float  # Average Treatment Effect
    ate_se: float | None = None  # Standard error of ATE
    ate_ci_lower: float | None = None  # Lower confidence interval
    ate_ci_upper: float | None = None  # Upper confidence interval
    confidence_level: float = 0.95  # Confidence level for intervals

    # Additional estimates for specific contexts
    att: float | None = None  # Average Treatment Effect on the Treated
    atc: float | None = None  # Average Treatment Effect on the Controls

    # Potential outcomes means
    potential_outcome_treated: float | None = None  # E[Y(1)]
    potential_outcome_control: float | None = None  # E[Y(0)]

    # Method-specific information
    method: str = "unknown"  # Name of the estimation method used
    n_observations: int | None = None  # Number of observations used
    n_treated: int | None = None  # Number of treated units
    n_control: int | None = None  # Number of control units

    # Model diagnostics and assumptions
    diagnostics: dict[str, Any] | None = None  # Method-specific diagnostics
    assumptions_checked: dict[str, bool] | None = None  # Assumption violations

    # Bootstrap/simulation details
    bootstrap_samples: int | None = None  # Number of bootstrap samples
    bootstrap_estimates: NDArray[Any] | None = None  # Bootstrap ATE estimates

    def __post_init__(self) -> None:
        """Validate the causal effect estimates after initialization."""
        if self.ate_ci_lower is not None and self.ate_ci_upper is not None:
            if self.ate_ci_lower > self.ate_ci_upper:
                raise ValueError("Lower confidence bound cannot exceed upper bound")

        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")

    @property
    def is_significant(self) -> bool:
        """Check if the causal effect is statistically significant.

        Returns True if the confidence interval does not contain zero,
        False otherwise. Returns None if confidence interval is not available.
        """
        if self.ate_ci_lower is None or self.ate_ci_upper is None:
            return False
        return self.ate_ci_lower > 0 or self.ate_ci_upper < 0

    @property
    def confidence_interval(self) -> tuple[float, float] | None:
        """Get confidence interval as a tuple.

        Returns:
            Tuple of (lower_bound, upper_bound) or None if not available
        """
        if self.ate_ci_lower is not None and self.ate_ci_upper is not None:
            return (self.ate_ci_lower, self.ate_ci_upper)
        return None

    @property
    def effect_size_interpretation(self) -> str:
        """Provide a qualitative interpretation of effect size."""
        abs_ate = abs(self.ate)

        if abs_ate < 0.1:
            return "negligible"
        elif abs_ate < 0.3:
            return "small"
        elif abs_ate < 0.5:
            return "medium"
        else:
            return "large"


class CausalInferenceError(Exception):
    """Base exception class for causal inference specific errors."""

    pass


class AssumptionViolationError(CausalInferenceError):
    """Raised when causal inference assumptions are violated."""

    pass


class DataValidationError(CausalInferenceError):
    """Raised when input data fails validation."""

    pass


class EstimationError(CausalInferenceError):
    """Raised when estimation process fails."""

    pass


class EstimatorProtocol(Protocol):
    """Protocol defining the interface for causal inference estimators."""

    def fit(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> BaseEstimator:
        """Fit the causal inference estimator to data."""
        ...

    def estimate_ate(self) -> CausalEffect:
        """Estimate the Average Treatment Effect."""
        ...

    def predict_potential_outcomes(
        self,
        treatment_values: pd.Series | NDArray[Any],
        covariates: pd.DataFrame | NDArray[Any] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Predict potential outcomes Y(0) and Y(1)."""
        ...


class BaseEstimator(abc.ABC):
    """Abstract base class for all causal inference estimators.

    This class establishes the common interface and shared functionality
    for all causal inference methods in the library. All specific estimators
    (G-computation, IPW, AIPW, etc.) should inherit from this class.

    Attributes:
        is_fitted: Whether the estimator has been fitted to data
        treatment_data: The treatment assignment data
        outcome_data: The outcome variable data
        covariate_data: The covariate/confounder data
        _causal_effect: Cached causal effect estimate
    """

    def __init__(
        self,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the base estimator.

        Args:
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output during estimation
        """
        self.random_state = random_state
        self.verbose = verbose
        self.is_fitted = False

        # Data containers
        self.treatment_data: TreatmentData | None = None
        self.outcome_data: OutcomeData | None = None
        self.covariate_data: CovariateData | None = None

        # Results cache
        self._causal_effect: CausalEffect | None = None

        # Set random state
        if random_state is not None:
            np.random.seed(random_state)

    @abc.abstractmethod
    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Implement the specific fitting logic for this estimator.

        This method contains the core estimation logic and must be
        implemented by each concrete estimator class.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data for adjustment
        """
        pass

    @abc.abstractmethod
    def _estimate_ate_implementation(self) -> CausalEffect:
        """Implement the specific ATE estimation logic for this estimator.

        Returns:
            CausalEffect object with the estimated average treatment effect
        """
        pass

    def fit(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> BaseEstimator:
        """Fit the causal inference estimator to data.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data for adjustment

        Returns:
            self: The fitted estimator instance

        Raises:
            DataValidationError: If input data fails validation
            EstimationError: If fitting process fails
        """
        # Validate inputs
        self._validate_inputs(treatment, outcome, covariates)

        # Store data
        self.treatment_data = treatment
        self.outcome_data = outcome
        self.covariate_data = covariates

        # Clear cached results
        self._causal_effect = None

        try:
            # Call the implementation-specific fitting logic
            self._fit_implementation(treatment, outcome, covariates)
            self.is_fitted = True

            if self.verbose:
                print(f"Successfully fitted {self.__class__.__name__}")

        except Exception as e:
            raise EstimationError(f"Failed to fit estimator: {str(e)}") from e

        return self

    def estimate_ate(self, use_cache: bool = True) -> CausalEffect:
        """Estimate the Average Treatment Effect.

        Args:
            use_cache: Whether to use cached results if available

        Returns:
            CausalEffect object with the estimated average treatment effect

        Raises:
            EstimationError: If estimator is not fitted or estimation fails
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        # Return cached result if available and requested
        if use_cache and self._causal_effect is not None:
            return self._causal_effect

        try:
            # Call the implementation-specific estimation logic
            causal_effect = self._estimate_ate_implementation()

            # Cache the result
            self._causal_effect = causal_effect

            if self.verbose:
                print(f"Estimated ATE: {causal_effect.ate:.4f}")
                if causal_effect.ate_ci_lower is not None:
                    print(
                        f"95% CI: [{causal_effect.ate_ci_lower:.4f}, {causal_effect.ate_ci_upper:.4f}]"
                    )

            return causal_effect

        except Exception as e:
            raise EstimationError(f"Failed to estimate ATE: {str(e)}") from e

    def predict_potential_outcomes(
        self,
        treatment_values: pd.Series | NDArray[Any],
        covariates: pd.DataFrame | NDArray[Any] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Predict potential outcomes Y(0) and Y(1) for given inputs.

        This is a generic implementation that may be overridden by specific
        estimators that have more efficient prediction methods.

        Args:
            treatment_values: Treatment assignment values to predict for
            covariates: Optional covariate values for prediction

        Returns:
            Tuple of (Y0_predictions, Y1_predictions)

        Raises:
            EstimationError: If estimator is not fitted
            NotImplementedError: If estimator doesn't support prediction
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before prediction")

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement potential outcome prediction"
        )

    def _validate_inputs(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Validate input data for causal inference.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data

        Raises:
            DataValidationError: If any validation checks fail
        """
        # Check that treatment and outcome have same length
        if len(treatment.values) != len(outcome.values):
            raise DataValidationError(
                f"Treatment ({len(treatment.values)}) and outcome ({len(outcome.values)}) "
                "must have the same number of observations"
            )

        # Check covariate dimensions if provided
        if covariates is not None:
            if len(covariates.values) != len(treatment.values):
                raise DataValidationError(
                    f"Covariates ({len(covariates.values)}) must have the same number "
                    f"of observations as treatment ({len(treatment.values)})"
                )

        # Check for missing values in treatment (not allowed)
        if isinstance(treatment.values, pd.Series):
            if treatment.values.isna().any():
                raise DataValidationError(
                    "Treatment values cannot contain missing data"
                )
        elif isinstance(treatment.values, np.ndarray):
            if np.isnan(treatment.values).any():
                raise DataValidationError(
                    "Treatment values cannot contain missing data"
                )

        # Validate treatment type specific constraints
        self._validate_treatment_constraints(treatment)

        # Check minimum sample size
        if len(treatment.values) < 10:
            raise DataValidationError("Minimum sample size of 10 observations required")

        # Check treatment variation
        if isinstance(treatment.values, (pd.Series, np.ndarray)):  # noqa: UP038
            if treatment.treatment_type == "binary":
                if len(np.unique(treatment.values)) < 2:
                    raise DataValidationError(
                        "Binary treatment must have both treated and control units"
                    )

    def _validate_treatment_constraints(self, treatment: TreatmentData) -> None:
        """Validate treatment-specific constraints.

        Args:
            treatment: Treatment data to validate

        Raises:
            DataValidationError: If treatment constraints are violated
        """
        if treatment.treatment_type == "binary":
            unique_values = np.unique(treatment.values)
            if (
                not np.array_equal(unique_values, [0, 1])
                and not np.array_equal(unique_values, [0])
                and not np.array_equal(unique_values, [1])
            ):
                if len(unique_values) != 2:
                    raise DataValidationError(
                        "Binary treatment must have exactly 2 unique values"
                    )

        elif treatment.treatment_type == "categorical":
            if treatment.categories is None:
                raise DataValidationError(
                    "Categorical treatment must specify categories"
                )

            unique_values = set(np.unique(treatment.values))
            expected_categories = set(treatment.categories)
            if not unique_values.issubset(expected_categories):
                raise DataValidationError(
                    f"Treatment values {unique_values - expected_categories} "
                    f"not in specified categories {treatment.categories}"
                )

    def check_positivity_assumption(
        self, min_probability: float = 0.01
    ) -> dict[str, Any]:
        """Check the positivity assumption for causal inference.

        The positivity assumption requires that all units have a non-zero
        probability of receiving each treatment level.

        Args:
            min_probability: Minimum probability threshold for positivity

        Returns:
            Dictionary with positivity check results

        Raises:
            EstimationError: If estimator is not fitted
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before assumption checking")

        if self.treatment_data is None:
            raise EstimationError("No treatment data available for positivity check")

        results: dict[str, Any] = {
            "assumption_met": True,
            "min_probability": min_probability,
            "violations": [],
            "warnings": [],
        }

        # For binary treatment
        if self.treatment_data.treatment_type == "binary":
            n_treated = np.sum(self.treatment_data.values == 1)
            n_control = np.sum(self.treatment_data.values == 0)
            total_n = len(self.treatment_data.values)

            prob_treated = n_treated / total_n
            prob_control = n_control / total_n

            if prob_treated < min_probability:
                results["assumption_met"] = False
                results["violations"].append(
                    f"Probability of treatment ({prob_treated:.4f}) below threshold"
                )

            if prob_control < min_probability:
                results["assumption_met"] = False
                results["violations"].append(
                    f"Probability of control ({prob_control:.4f}) below threshold"
                )

            results["treatment_probability"] = prob_treated
            results["control_probability"] = prob_control

        return results

    def summary(self) -> str:
        """Provide a summary of the fitted estimator and results.

        Returns:
            String summary of estimator status and results
        """
        if not self.is_fitted:
            return f"{self.__class__.__name__} (not fitted)"

        summary_lines = [
            f"{self.__class__.__name__} Summary",
            "=" * 40,
            f"Fitted: {self.is_fitted}",
            f"Observations: {len(self.treatment_data.values) if self.treatment_data else 'N/A'}",
        ]

        if self.treatment_data:
            if self.treatment_data.treatment_type == "binary":
                n_treated = np.sum(self.treatment_data.values == 1)
                n_control = np.sum(self.treatment_data.values == 0)
                summary_lines.extend(
                    [
                        f"Treated units: {n_treated}",
                        f"Control units: {n_control}",
                    ]
                )

        if self._causal_effect:
            summary_lines.extend(
                [
                    "",
                    "Causal Effect Estimate:",
                    f"ATE: {self._causal_effect.ate:.4f}",
                ]
            )

            if self._causal_effect.ate_ci_lower is not None:
                summary_lines.append(
                    f"95% CI: [{self._causal_effect.ate_ci_lower:.4f}, "
                    f"{self._causal_effect.ate_ci_upper:.4f}]"
                )
                summary_lines.append(
                    f"Significant: {self._causal_effect.is_significant}"
                )

        return "\n".join(summary_lines)

    def run_diagnostics(
        self,
        include_balance: bool = True,
        include_overlap: bool = True,
        include_assumptions: bool = True,
        include_specification: bool = True,
        include_sensitivity: bool = False,
        verbose: bool = True,
    ) -> DiagnosticReport:
        """Run comprehensive diagnostics for the fitted estimator.

        Args:
            include_balance: Whether to include balance diagnostics
            include_overlap: Whether to include overlap diagnostics
            include_assumptions: Whether to include assumption checking
            include_specification: Whether to include specification tests
            include_sensitivity: Whether to include sensitivity analysis
            verbose: Whether to print detailed output

        Returns:
            DiagnosticReport with comprehensive assessment

        Raises:
            EstimationError: If estimator is not fitted
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before running diagnostics")

        if self.treatment_data is None or self.outcome_data is None or self.covariate_data is None:
            raise EstimationError("No data available for diagnostics")

        # Import here to avoid circular imports
        from ..diagnostics.reporting import DiagnosticReportGenerator

        # Get causal effect for sensitivity analysis
        causal_effect = None
        if include_sensitivity and self._causal_effect is not None:
            causal_effect = self._causal_effect
        elif include_sensitivity:
            try:
                causal_effect = self.estimate_ate()
            except Exception:
                if verbose:
                    print(
                        "Warning: Could not estimate causal effect for sensitivity analysis"
                    )
                include_sensitivity = False

        # Generate diagnostic report
        generator = DiagnosticReportGenerator(
            include_balance=include_balance,
            include_overlap=include_overlap,
            include_assumptions=include_assumptions,
            include_specification=include_specification,
            include_sensitivity=include_sensitivity,
        )

        report = generator.generate_comprehensive_report(
            self.treatment_data,
            self.outcome_data,
            self.covariate_data,
            causal_effect=causal_effect,
            verbose=False,
        )

        if verbose:
            generator.print_diagnostic_report(report)

        return report

    def check_assumptions(self, verbose: bool = True) -> dict[str, bool]:
        """Quick assumption check for the fitted estimator.

        Args:
            verbose: Whether to print results

        Returns:
            Dictionary with assumption check results

        Raises:
            EstimationError: If estimator is not fitted
        """
        if not self.is_fitted:
            raise EstimationError(
                "Estimator must be fitted before checking assumptions"
            )

        if (
            self.treatment_data is None
            or self.outcome_data is None
            or self.covariate_data is None
        ):
            raise EstimationError("No data available for assumption checking")

        # Import here to avoid circular imports
        from ..diagnostics.reporting import create_assumption_summary

        assumptions = create_assumption_summary(
            self.treatment_data,
            self.outcome_data,
            self.covariate_data,
        )

        if verbose:
            print("=== Quick Assumption Check ===")
            for assumption, met in assumptions.items():
                status = "✅" if met else "❌"
                print(f"{assumption.replace('_', ' ').title()}: {status}")
            print()

            overall = assumptions.get("overall_assessment", False)
            if overall:
                print("✅ Overall: Key assumptions appear to be met")
            else:
                print(
                    "⚠️ Overall: Some assumptions may be violated - run full diagnostics"
                )

        return assumptions
