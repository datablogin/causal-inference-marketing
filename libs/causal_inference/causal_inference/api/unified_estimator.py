"""Unified CausalAnalysis API for sklearn-style causal inference workflows.

This module provides a unified interface that makes causal inference accessible
through a simple, consistent API similar to sklearn's pattern.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from ..estimators.aipw import AIPWEstimator
from ..estimators.g_computation import GComputationEstimator
from ..estimators.ipw import IPWEstimator
from ..reporting.html_generator import HTMLReportGenerator
from ..sensitivity import generate_sensitivity_report


class CausalAnalysis:
    """Unified API for causal inference analysis with one-call HTML reporting.

    This class provides a sklearn-style interface that unifies all causal inference
    estimators and generates comprehensive HTML reports suitable for business stakeholders.

    The design philosophy is: CausalAnalysis().fit(data).report() should provide
    a complete causal analysis with professional reporting.

    Examples:
        Basic usage - one line complete analysis:

        >>> analysis = CausalAnalysis().fit(data).report()
        >>> print(f"Treatment effect: {analysis.effect.ate:.3f}")

        Custom configuration:

        >>> analysis = CausalAnalysis(
        ...     method='aipw',
        ...     treatment_column='email_campaign',
        ...     outcome_column='revenue',
        ...     covariate_columns=['age', 'segment', 'history']
        ... )
        >>> analysis.fit(data)
        >>> report = analysis.report(output_path='campaign_analysis.html')
    """

    def __init__(
        self,
        method: Literal["auto", "g_computation", "ipw", "aipw"] = "auto",
        treatment_column: str | None = None,
        outcome_column: str | None = None,
        covariate_columns: list[str] | None = None,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 1000,
        random_state: int | None = None,
        **estimator_kwargs: Any,
    ):
        """Initialize the unified causal analysis interface.

        Args:
            method: Estimator method to use ('auto' selects best based on data)
            treatment_column: Name of treatment column (auto-detected if None)
            outcome_column: Name of outcome column (auto-detected if None)
            covariate_columns: List of covariate column names (auto-detected if None)
            confidence_level: Confidence level for intervals
            bootstrap_samples: Number of bootstrap samples for inference
            random_state: Random seed for reproducibility
            **estimator_kwargs: Additional arguments passed to underlying estimator
        """
        self.method = method
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.covariate_columns = covariate_columns
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

        # State
        self.is_fitted_ = False
        self.data_: pd.DataFrame | None = None
        self.treatment_data_: TreatmentData | None = None
        self.outcome_data_: OutcomeData | None = None
        self.covariate_data_: CovariateData | None = None
        self.estimator_: BaseEstimator | None = None
        self.effect_: CausalEffect | None = None

        # Report components
        self.html_report_: str | None = None
        self.report_data_: dict[str, Any] | None = None

    def fit(self, data: pd.DataFrame | str) -> CausalAnalysis:
        """Fit the causal inference model to data.

        Args:
            data: Input data as pandas DataFrame or path to data file

        Returns:
            Self for method chaining

        Examples:
            >>> analysis = CausalAnalysis().fit(data)
            >>> analysis = CausalAnalysis().fit('campaign_data.csv')
        """
        # Load data if path provided
        if isinstance(data, str):
            data = self._load_data(data)

        # Store data
        self.data_ = data.copy()

        # Auto-detect columns if not specified
        self._detect_columns()

        # Validate and prepare data
        self._prepare_data()

        # Select estimator method
        self._select_estimator()

        # Fit the estimator
        self.estimator_.fit(
            self.treatment_data_, self.outcome_data_, self.covariate_data_
        )

        # Estimate treatment effect
        self.effect_ = self.estimator_.estimate_ate()

        # Add p-value if missing and we have standard error
        if not hasattr(self.effect_, "p_value") or self.effect_.p_value is None:
            if hasattr(self.effect_, "ate_se") and self.effect_.ate_se is not None:
                import scipy.stats as stats

                t_stat = self.effect_.ate / self.effect_.ate_se
                # Use conservative df approximation
                df = max(30, len(self.data_) - 10)  # Conservative degrees of freedom
                self.effect_.p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            else:
                # Use simple z-test approximation based on CI
                if (
                    hasattr(self.effect_, "ate_ci_lower")
                    and self.effect_.ate_ci_lower is not None
                    and hasattr(self.effect_, "ate_ci_upper")
                    and self.effect_.ate_ci_upper is not None
                ):
                    # Approximate standard error from CI width
                    se_approx = (
                        self.effect_.ate_ci_upper - self.effect_.ate_ci_lower
                    ) / (2 * 1.96)
                    if se_approx > 0:
                        z_stat = self.effect_.ate / se_approx
                        self.effect_.p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        self.is_fitted_ = True
        return self

    def report(
        self,
        output_path: str | None = None,
        include_sensitivity: bool = True,
        include_diagnostics: bool = True,
        template: str = "executive",
        **report_kwargs: Any,
    ) -> dict[str, Any]:
        """Generate comprehensive HTML report of causal analysis.

        Args:
            output_path: Path to save HTML report (optional)
            include_sensitivity: Whether to include sensitivity analysis
            include_diagnostics: Whether to include diagnostic plots
            template: Report template ('executive', 'technical', 'full')
            **report_kwargs: Additional arguments for report generation

        Returns:
            Dictionary containing report components and metadata

        Examples:
            >>> report = analysis.report()
            >>> report = analysis.report('analysis_results.html', template='full')
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before generating report")

        # Generate report components
        report_generator = HTMLReportGenerator(
            estimator=self.estimator_,
            effect=self.effect_,
            data=self.data_,
            treatment_column=self.treatment_column,
            outcome_column=self.outcome_column,
            covariate_columns=self.covariate_columns,
            method_name=self.method,
            confidence_level=self.confidence_level,
        )

        # Generate HTML report
        self.html_report_ = report_generator.generate_report(
            template=template,
            include_sensitivity=include_sensitivity,
            include_diagnostics=include_diagnostics,
            **report_kwargs,
        )

        # Compile report data
        self.report_data_ = {
            "html_report": self.html_report_,
            "effect": self.effect_,
            "method": self.method,
            "sample_size": len(self.data_),
            "treatment_column": self.treatment_column,
            "outcome_column": self.outcome_column,
            "covariate_columns": self.covariate_columns,
            "confidence_level": self.confidence_level,
            "estimator_diagnostics": getattr(self.estimator_, "diagnostics_", None),
        }

        # Save report if path provided
        if output_path:
            self._save_report(output_path)
            self.report_data_["file_path"] = output_path

        return self.report_data_

    def estimate_ate(self) -> CausalEffect:
        """Estimate average treatment effect.

        Returns:
            CausalEffect object with treatment effect estimates
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before estimating effects")
        return self.effect_

    def predict_ite(self, data: pd.DataFrame | None = None) -> NDArray[np.floating]:
        """Predict individual treatment effects.

        Args:
            data: New data for prediction (uses fitted data if None)

        Returns:
            Array of individual treatment effect predictions
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before prediction")

        if hasattr(self.estimator_, "predict_ite"):
            if data is None:
                data = self.covariate_data_
            else:
                # Convert to CovariateData format
                data = CovariateData(
                    values=data[self.covariate_columns], names=self.covariate_columns
                )
            return self.estimator_.predict_ite(data)
        else:
            warnings.warn(
                f"Estimator {type(self.estimator_)} does not support ITE prediction"
            )
            return np.full(
                len(data) if data is not None else len(self.data_), self.effect_.ate
            )

    def sensitivity_analysis(self, **kwargs: Any) -> dict[str, Any]:
        """Run comprehensive sensitivity analysis.

        Args:
            **kwargs: Arguments passed to sensitivity analysis functions

        Returns:
            Dictionary with sensitivity analysis results
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before sensitivity analysis")

        return generate_sensitivity_report(
            treatment_data=self.treatment_data_.values,
            outcome_data=self.outcome_data_.values,
            covariates_data=self.covariate_data_.values
            if self.covariate_data_
            else None,
            observed_effect=self.effect_.ate,
            ci_lower=self.effect_.ate_ci_lower,
            ci_upper=self.effect_.ate_ci_upper,
            **kwargs,
        )

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from file path."""
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".json"):
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def _detect_columns(self) -> None:
        """Auto-detect treatment, outcome, and covariate columns."""
        if self.data_ is None:
            raise ValueError("No data available for column detection")

        # Auto-detect treatment column
        if self.treatment_column is None:
            binary_cols = []
            for col in self.data_.columns:
                if self.data_[col].nunique() == 2:
                    binary_cols.append(col)

            if len(binary_cols) == 1:
                self.treatment_column = binary_cols[0]
            elif len(binary_cols) > 1:
                # Heuristic: look for common treatment column names
                treatment_candidates = [
                    "treatment",
                    "treated",
                    "group",
                    "condition",
                    "arm",
                    "intervention",
                    "exposed",
                    "campaign",
                ]
                for candidate in treatment_candidates:
                    matches = [
                        col for col in binary_cols if candidate.lower() in col.lower()
                    ]
                    if matches:
                        self.treatment_column = matches[0]
                        break

                if self.treatment_column is None:
                    self.treatment_column = binary_cols[0]  # Use first binary column
                    warnings.warn(
                        f"Multiple binary columns found. Using '{self.treatment_column}' as treatment."
                    )
            else:
                raise ValueError(
                    "No binary treatment column detected. Please specify treatment_column."
                )

        # Auto-detect outcome column
        if self.outcome_column is None:
            numeric_cols = list(self.data_.select_dtypes(include=[np.number]).columns)
            numeric_cols = [col for col in numeric_cols if col != self.treatment_column]

            if len(numeric_cols) >= 1:
                # Heuristic: look for common outcome column names
                outcome_candidates = [
                    "outcome",
                    "y",
                    "target",
                    "response",
                    "revenue",
                    "conversion",
                    "sales",
                    "profit",
                    "value",
                ]
                for candidate in outcome_candidates:
                    matches = [
                        col for col in numeric_cols if candidate.lower() in col.lower()
                    ]
                    if matches:
                        self.outcome_column = matches[0]
                        break

                if self.outcome_column is None:
                    self.outcome_column = numeric_cols[0]  # Use first numeric column
                    warnings.warn(f"Using '{self.outcome_column}' as outcome column.")
            else:
                raise ValueError(
                    "No numeric outcome column detected. Please specify outcome_column."
                )

        # Auto-detect covariate columns
        if self.covariate_columns is None:
            all_cols = set(self.data_.columns)
            used_cols = {self.treatment_column, self.outcome_column}
            self.covariate_columns = list(all_cols - used_cols)

    def _prepare_data(self) -> None:
        """Prepare data objects for estimation."""
        # Create TreatmentData
        treatment_values = self.data_[self.treatment_column]
        treatment_type = "binary" if treatment_values.nunique() == 2 else "continuous"

        self.treatment_data_ = TreatmentData(
            values=treatment_values,
            name=self.treatment_column,
            treatment_type=treatment_type,
        )

        # Create OutcomeData
        outcome_values = self.data_[self.outcome_column]
        outcome_type = (
            "continuous" if outcome_values.dtype in ["float64", "int64"] else "binary"
        )

        self.outcome_data_ = OutcomeData(
            values=outcome_values, name=self.outcome_column, outcome_type=outcome_type
        )

        # Create CovariateData
        if self.covariate_columns:
            covariate_values = self.data_[self.covariate_columns]
            self.covariate_data_ = CovariateData(
                values=covariate_values, names=self.covariate_columns
            )
        else:
            self.covariate_data_ = None

    def _select_estimator(self) -> None:
        """Select and initialize the appropriate estimator."""
        common_params = {
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_level": self.confidence_level,
            "random_state": self.random_state,
            **self.estimator_kwargs,
        }

        if self.method == "auto":
            # Auto-selection logic
            n_samples = len(self.data_)
            n_covariates = len(self.covariate_columns) if self.covariate_columns else 0

            if n_samples > 1000 and n_covariates > 0:
                # Use AIPW for larger samples with covariates (doubly robust)
                self.method = "aipw"
                self.estimator_ = AIPWEstimator(**common_params)
            elif n_covariates > 0:
                # Use G-computation for smaller samples with covariates
                self.method = "g_computation"
                self.estimator_ = GComputationEstimator(**common_params)
            else:
                # Use IPW when no covariates available
                self.method = "ipw"
                self.estimator_ = IPWEstimator(**common_params)

        elif self.method == "g_computation":
            self.estimator_ = GComputationEstimator(**common_params)

        elif self.method == "ipw":
            self.estimator_ = IPWEstimator(**common_params)

        elif self.method == "aipw":
            self.estimator_ = AIPWEstimator(**common_params)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _save_report(self, path: str) -> None:
        """Save HTML report to file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html_report_)

    def __repr__(self) -> str:
        """String representation of the analysis."""
        if self.is_fitted_:
            return (
                f"CausalAnalysis(method='{self.method}', "
                f"ate={self.effect_.ate:.3f}, fitted=True)"
            )
        else:
            return f"CausalAnalysis(method='{self.method}', fitted=False)"
