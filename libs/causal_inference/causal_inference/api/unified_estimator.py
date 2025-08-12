"""Unified CausalAnalysis API for sklearn-style causal inference workflows.

This module provides a unified interface that makes causal inference accessible
through a simple, consistent API similar to sklearn's pattern.
"""

from __future__ import annotations

import warnings
from pathlib import Path
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
        self.bootstrap_samples = self._validate_bootstrap_samples(bootstrap_samples)
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

        # Comprehensive data validation
        self._validate_data()

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

        # Add p-value if missing - with improved statistical rigor
        if not hasattr(self.effect_, "p_value") or self.effect_.p_value is None:
            import warnings

            import scipy.stats as stats

            if hasattr(self.effect_, "ate_se") and self.effect_.ate_se is not None:
                # Calculate appropriate degrees of freedom based on method and sample size
                if self.effect_.ate_se > 0:
                    t_stat = self.effect_.ate / self.effect_.ate_se

                    # More principled degrees of freedom calculation
                    n_total = len(self.data_)
                    n_covariates = (
                        len(self.covariate_columns) if self.covariate_columns else 0
                    )

                    # Conservative df calculation based on method
                    if self.method in ["g_computation", "aipw"]:
                        # Account for model degrees of freedom
                        df = max(10, n_total - n_covariates - 2)
                    else:
                        # For IPW and simple methods
                        df = max(10, n_total - 2)

                    # Add warning about approximation
                    warnings.warn(
                        f"P-value approximated using t-distribution with df={df}. "
                        "For precise inference, use bootstrap-based confidence intervals from the underlying estimator.",
                        UserWarning,
                    )

                    self.effect_.p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                else:
                    warnings.warn(
                        "Standard error is zero or negative, cannot calculate p-value",
                        UserWarning,
                    )
                    self.effect_.p_value = None

            elif (
                hasattr(self.effect_, "ate_ci_lower")
                and self.effect_.ate_ci_lower is not None
                and hasattr(self.effect_, "ate_ci_upper")
                and self.effect_.ate_ci_upper is not None
            ):
                # Use CI-based approximation with warnings
                se_approx = (self.effect_.ate_ci_upper - self.effect_.ate_ci_lower) / (
                    2 * 1.96
                )

                if se_approx > 0:
                    z_stat = self.effect_.ate / se_approx

                    warnings.warn(
                        "P-value approximated from confidence interval width. "
                        "This assumes normal distribution and may be inaccurate for small samples.",
                        UserWarning,
                    )

                    self.effect_.p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                else:
                    warnings.warn(
                        "Confidence interval width is zero or negative, cannot calculate p-value",
                        UserWarning,
                    )
                    self.effect_.p_value = None
            else:
                warnings.warn(
                    "No standard error or confidence interval available for p-value calculation. "
                    "Consider using an estimator that provides bootstrap-based inference.",
                    UserWarning,
                )
                self.effect_.p_value = None

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
        """Load data from file path with validation."""

        # Validate and sanitize file path
        file_path = self._validate_file_path(path)

        try:
            if file_path.suffix.lower() == ".csv":
                return pd.read_csv(file_path)
            elif file_path.suffix.lower() == ".parquet":
                return pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".json":
                return pd.read_json(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}. "
                    f"Supported formats: .csv, .parquet, .json"
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found: {file_path}. Please check the file path and permissions."
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load data from {file_path}: {str(e)}. "
                f"Please check file format and data integrity."
            )

    def _validate_file_path(self, path: str) -> Path:
        """Validate and sanitize file paths for security."""

        if not isinstance(path, str) or not path.strip():
            raise ValueError("File path must be a non-empty string")

        try:
            file_path = Path(path).resolve()
        except Exception as e:
            raise ValueError(f"Invalid file path: {path}. Error: {str(e)}")

        # Check file format first (before checking existence)
        supported_formats = {".csv", ".parquet", ".json"}
        if file_path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        # Basic security checks
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Check file size (limit to 1GB for safety)
        max_size = 1024 * 1024 * 1024  # 1GB
        if file_path.stat().st_size > max_size:
            raise ValueError(
                f"File too large: {file_path.stat().st_size / 1024 / 1024:.1f}MB. "
                f"Maximum allowed: {max_size / 1024 / 1024:.0f}MB"
            )

        return file_path

    def _validate_data(self) -> None:
        """Comprehensive data validation before analysis."""
        if self.data_ is None or self.data_.empty:
            raise ValueError("Data is empty or None")

        # Check minimum sample size
        if len(self.data_) < 10:
            raise ValueError(
                f"Sample size too small: {len(self.data_)}. Minimum 10 observations required."
            )

        # Validate all columns exist
        required_columns = []
        if self.treatment_column:
            required_columns.append(self.treatment_column)
        if self.outcome_column:
            required_columns.append(self.outcome_column)
        if self.covariate_columns:
            required_columns.extend(self.covariate_columns)

        missing_columns = [
            col for col in required_columns if col not in self.data_.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Required columns not found in data: {missing_columns}. "
                f"Available columns: {list(self.data_.columns)}"
            )

        # Check for missing values
        for col in required_columns:
            if self.data_[col].isnull().any():
                missing_count = self.data_[col].isnull().sum()
                missing_pct = missing_count / len(self.data_) * 100
                raise ValueError(
                    f"Column '{col}' contains {missing_count} missing values ({missing_pct:.1f}%). "
                    f"Please handle missing data before analysis."
                )

        # Validate numeric columns for finite values
        numeric_columns = [self.outcome_column] + (self.covariate_columns or [])
        for col in numeric_columns:
            if col in self.data_.columns and self.data_[col].dtype in [
                "float64",
                "int64",
                "float32",
                "int32",
            ]:
                if not np.isfinite(self.data_[col]).all():
                    inf_count = np.isinf(self.data_[col]).sum()
                    nan_count = np.isnan(self.data_[col]).sum()
                    raise ValueError(
                        f"Column '{col}' contains non-finite values: "
                        f"{inf_count} infinite, {nan_count} NaN values. "
                        f"Please clean data before analysis."
                    )

        # Check for constant columns (no variation)
        for col in required_columns:
            if self.data_[col].nunique() == 1:
                unique_val = self.data_[col].iloc[0]
                raise ValueError(
                    f"Column '{col}' has no variation (all values = {unique_val}). "
                    f"Cannot perform causal analysis on constant variables."
                )

        # Additional validation specific to treatment column
        if self.treatment_column and self.treatment_column in self.data_.columns:
            treatment_unique = self.data_[self.treatment_column].nunique()
            if treatment_unique > 10:
                import warnings

                warnings.warn(
                    f"Treatment column '{self.treatment_column}' has {treatment_unique} unique values. "
                    f"This may indicate a continuous treatment or misconfigured column.",
                    UserWarning,
                )

    def _detect_columns(self) -> None:
        """Auto-detect treatment, outcome, and covariate columns."""
        if self.data_ is None:
            raise ValueError("No data available for column detection")

        # Check for empty data first
        if self.data_.empty:
            raise ValueError("Data is empty or None")

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
                # Provide more helpful error message
                available_cols = list(self.data_.columns)
                unique_counts = {
                    col: self.data_[col].nunique() for col in available_cols[:10]
                }  # Show first 10
                raise ValueError(
                    f"No binary treatment column detected. Available columns and their unique value counts: {unique_counts}. "
                    f"Please specify treatment_column or ensure your data contains a binary treatment variable."
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
                # Provide more helpful error message
                available_cols = list(self.data_.columns)
                col_types = {
                    col: str(self.data_[col].dtype) for col in available_cols[:10]
                }  # Show first 10
                raise ValueError(
                    f"No numeric outcome column detected. Available columns and their types: {col_types}. "
                    f"Please specify outcome_column or ensure your data contains a numeric outcome variable."
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
            covariate_values = self.data_[self.covariate_columns].copy()

            # Handle categorical variables by encoding them
            categorical_columns = covariate_values.select_dtypes(
                include=["object", "category"]
            ).columns
            if len(categorical_columns) > 0:
                import warnings

                warnings.warn(
                    f"Categorical columns detected: {list(categorical_columns)}. "
                    f"Converting to numeric using one-hot encoding.",
                    UserWarning,
                )

                # Use pandas get_dummies to encode categorical variables
                covariate_values = pd.get_dummies(
                    covariate_values,
                    columns=categorical_columns,
                    drop_first=True,  # Avoid multicollinearity
                    prefix_sep="_",
                )

                # Update covariate column names
                self.covariate_columns = list(covariate_values.columns)

            self.covariate_data_ = CovariateData(
                values=covariate_values, names=self.covariate_columns
            )
        else:
            self.covariate_data_ = None

    def _validate_selection_assumptions(self) -> None:
        """Validate assumptions for method selection and provide warnings."""
        import warnings

        # Check treatment balance
        treatment_balance = self.data_[self.treatment_column].mean()
        if treatment_balance <= 0.05 or treatment_balance >= 0.95:
            warnings.warn(
                f"Extreme treatment imbalance ({treatment_balance:.1%}) may affect method performance. "
                f"Consider stratified sampling or specialized imbalanced treatment methods.",
                UserWarning,
            )
        elif treatment_balance <= 0.15 or treatment_balance >= 0.85:
            warnings.warn(
                f"Moderate treatment imbalance ({treatment_balance:.1%}) detected. "
                f"Results should be interpreted with caution.",
                UserWarning,
            )

        # Check for adequate covariate coverage
        if self.covariate_columns:
            n_covariates = len(self.covariate_columns)
            n_samples = len(self.data_)
            covariate_ratio = n_samples / n_covariates

            if covariate_ratio < 20:
                warnings.warn(
                    f"High-dimensional data detected: {n_covariates} covariates with {n_samples} samples "
                    f"(ratio: {covariate_ratio:.1f}). Consider dimensionality reduction or regularization.",
                    UserWarning,
                )

        # Check outcome distribution
        if self.outcome_column:
            outcome_var = self.data_[self.outcome_column].var()
            if outcome_var < 1e-10:
                warnings.warn(
                    f"Very low outcome variance detected ({outcome_var:.2e}). "
                    f"This may indicate scaling issues or near-constant outcomes.",
                    UserWarning,
                )

        # Performance warnings for computational requirements
        n_samples = len(self.data_)
        n_covariates = len(self.covariate_columns) if self.covariate_columns else 0

        # Memory usage estimation
        estimated_memory_mb = (n_samples * n_covariates * 8) / (
            1024 * 1024
        )  # 8 bytes per float64
        if estimated_memory_mb > 500:  # 500MB threshold
            warnings.warn(
                f"Large dataset detected: {n_samples:,} samples × {n_covariates} covariates "
                f"(~{estimated_memory_mb:.0f}MB). Consider data reduction or chunked processing for better performance.",
                UserWarning,
            )

        # Computational complexity warnings
        if n_samples > 50000 and self.bootstrap_samples > 1000:
            warnings.warn(
                f"High computational load: {n_samples:,} samples with {self.bootstrap_samples} bootstrap samples. "
                f"Consider reducing bootstrap_samples to 500-1000 or using analytical confidence intervals.",
                UserWarning,
            )

        # High-dimensional data warning
        if n_covariates > n_samples / 10:
            warnings.warn(
                f"High-dimensional data: {n_covariates} covariates with {n_samples} samples "
                f"(ratio {n_samples / n_covariates:.1f}:1). Consider feature selection, PCA, or regularization.",
                UserWarning,
            )

    def _select_estimator(self) -> None:
        """Select and initialize the appropriate estimator with validation."""
        # Validate method selection assumptions
        self._validate_selection_assumptions()

        common_params = {
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_level": self.confidence_level,
            "random_state": self.random_state,
            **self.estimator_kwargs,
        }

        if self.method == "auto":
            # Enhanced auto-selection logic with adaptive thresholds
            n_samples = len(self.data_)
            n_covariates = len(self.covariate_columns) if self.covariate_columns else 0

            # Calculate treatment balance for adaptive selection
            treatment_balance = self.data_[self.treatment_column].mean()

            # Adaptive selection based on multiple factors
            if n_samples >= 1000 and n_covariates > 0:
                # Use AIPW for larger samples with covariates (doubly robust)
                # But consider treatment balance
                if treatment_balance < 0.1 or treatment_balance > 0.9:
                    # Extreme imbalance - AIPW may still be best due to double robustness
                    import warnings

                    warnings.warn(
                        f"Extreme treatment imbalance ({treatment_balance:.1%}) detected. "
                        f"AIPW selected for double robustness, but consider IPW with careful propensity modeling.",
                        UserWarning,
                    )
                self.method = "aipw"
                self.estimator_ = AIPWEstimator(**common_params)

            elif n_covariates > 0:
                # Use G-computation for smaller samples with covariates
                # Unless we have extreme imbalance, then prefer IPW
                if treatment_balance < 0.05 or treatment_balance > 0.95:
                    self.method = "ipw"
                    self.estimator_ = IPWEstimator(**common_params)
                    import warnings

                    warnings.warn(
                        f"Extreme treatment imbalance ({treatment_balance:.1%}) with small sample. "
                        f"Using IPW instead of G-computation for better propensity handling.",
                        UserWarning,
                    )
                else:
                    self.method = "g_computation"
                    self.estimator_ = GComputationEstimator(**common_params)
            else:
                # When no covariates available, use G-computation which can work without covariates
                # (it reduces to simple difference in means)
                import warnings

                warnings.warn(
                    "No covariates available for adjustment. Using G-computation which reduces to simple difference in means.",
                    UserWarning,
                )
                self.method = "g_computation"
                self.estimator_ = GComputationEstimator(**common_params)

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

    def _validate_bootstrap_samples(self, bootstrap_samples: int) -> int:
        """Validate and adjust bootstrap samples for performance."""
        import warnings

        if bootstrap_samples < 0:
            raise ValueError(
                f"Bootstrap samples must be non-negative, got {bootstrap_samples}"
            )

        if bootstrap_samples == 0:
            return 0  # No bootstrap

        # Performance-based recommendations
        if bootstrap_samples < 100:
            warnings.warn(
                f"Low bootstrap samples ({bootstrap_samples}) may provide unreliable confidence intervals. "
                f"Consider using at least 100 samples.",
                UserWarning,
            )

        if bootstrap_samples > 5000:
            warnings.warn(
                f"High bootstrap samples ({bootstrap_samples}) may cause long computation times. "
                f"Consider using 1000-2000 samples for most applications.",
                UserWarning,
            )

        return bootstrap_samples
