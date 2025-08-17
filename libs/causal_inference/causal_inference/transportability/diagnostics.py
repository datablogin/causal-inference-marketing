"""Covariate shift detection and diagnostics for transportability analysis."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class ShiftSeverity(Enum):
    """Enumeration for covariate shift severity levels."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class DistributionDifference:
    """Results from covariate distribution difference analysis.

    Attributes:
        variable_name: Name of the variable analyzed
        standardized_mean_diff: Standardized mean difference (Cohen's d)
        ks_statistic: Kolmogorov-Smirnov test statistic
        ks_pvalue: P-value from KS test
        wasserstein_distance: Wasserstein (Earth Mover's) distance
        severity: Classification of shift severity
        source_mean: Mean value in source population
        target_mean: Mean value in target population
        source_std: Standard deviation in source population
        target_std: Standard deviation in target population
    """

    variable_name: str
    standardized_mean_diff: float
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    severity: ShiftSeverity
    source_mean: float
    target_mean: float
    source_std: float
    target_std: float

    @property
    def is_significant(self) -> bool:
        """Check if distribution difference is statistically significant."""
        return self.ks_pvalue < 0.05

    @property
    def effect_size_interpretation(self) -> str:
        """Provide interpretation of standardized mean difference."""
        abs_smd = abs(self.standardized_mean_diff)
        if abs_smd < 0.2:
            return "negligible"
        elif abs_smd < 0.5:
            return "small"
        elif abs_smd < 0.8:
            return "medium"
        else:
            return "large"


class CovariateShiftDiagnostics:
    """Comprehensive covariate shift detection and analysis.

    This class provides statistical tests and visualizations to assess
    differences in covariate distributions between source and target populations.

    Attributes:
        min_samples: Minimum sample size for reliable tests
        alpha: Significance level for statistical tests
        smd_threshold_mild: SMD threshold for mild shift classification
        smd_threshold_moderate: SMD threshold for moderate shift classification
    """

    def __init__(
        self,
        min_samples: int = 50,
        alpha: float = 0.05,
        smd_threshold_mild: float = 0.1,
        smd_threshold_moderate: float = 0.25,
    ) -> None:
        """Initialize covariate shift diagnostics.

        Args:
            min_samples: Minimum sample size for reliable statistical tests
            alpha: Significance level for hypothesis tests
            smd_threshold_mild: SMD threshold for classifying mild shifts
            smd_threshold_moderate: SMD threshold for classifying moderate shifts
        """
        self.min_samples = min_samples
        self.alpha = alpha
        self.smd_threshold_mild = smd_threshold_mild
        self.smd_threshold_moderate = smd_threshold_moderate

        # Results storage
        self.distribution_differences: list[DistributionDifference] = []
        self.overall_shift_score: float | None = None
        self.discriminative_accuracy: float | None = None

    def analyze_covariate_shift(
        self,
        source_data: pd.DataFrame | NDArray[Any],
        target_data: pd.DataFrame | NDArray[Any],
        variable_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive covariate shift analysis.

        Args:
            source_data: Covariate data from source population
            target_data: Covariate data from target population
            variable_names: Names of variables (inferred if not provided)

        Returns:
            Dictionary containing comprehensive shift analysis results

        Raises:
            ValueError: If data dimensions don't match or sample sizes too small
        """
        # Convert to DataFrame for easier handling
        source_df = self._ensure_dataframe(source_data, variable_names)
        target_df = self._ensure_dataframe(target_data, variable_names)

        # Validate inputs
        self._validate_inputs(source_df, target_df)

        # Analyze each variable
        self.distribution_differences = []
        for col in source_df.columns:
            diff = self._analyze_single_variable(
                source_df[col].values, target_df[col].values, col
            )
            self.distribution_differences.append(diff)

        # Calculate overall shift metrics
        self.overall_shift_score = self._calculate_overall_shift_score()
        self.discriminative_accuracy = self._calculate_discriminative_accuracy(
            source_df, target_df
        )

        # Create comprehensive results
        results = {
            "distribution_differences": self.distribution_differences,
            "overall_shift_score": self.overall_shift_score,
            "discriminative_accuracy": self.discriminative_accuracy,
            "n_variables": len(self.distribution_differences),
            "n_severe_shifts": self._count_shifts_by_severity(ShiftSeverity.SEVERE),
            "n_moderate_shifts": self._count_shifts_by_severity(ShiftSeverity.MODERATE),
            "n_mild_shifts": self._count_shifts_by_severity(ShiftSeverity.MILD),
            "max_smd": max(
                abs(diff.standardized_mean_diff)
                for diff in self.distribution_differences
            ),
            "mean_smd": np.mean(
                [
                    abs(diff.standardized_mean_diff)
                    for diff in self.distribution_differences
                ]
            ),
            "recommendations": self._generate_recommendations(),
        }

        return results

    def _ensure_dataframe(
        self,
        data: pd.DataFrame | NDArray[Any],
        variable_names: list[str] | None,
    ) -> pd.DataFrame:
        """Convert input data to DataFrame with proper column names."""
        if isinstance(data, pd.DataFrame):
            return data

        # Convert numpy array to DataFrame
        if variable_names is None:
            variable_names = [f"var_{i}" for i in range(data.shape[1])]

        if len(variable_names) != data.shape[1]:
            raise ValueError(
                f"Number of variable names ({len(variable_names)}) doesn't match "
                f"number of columns ({data.shape[1]})"
            )

        return pd.DataFrame(data, columns=variable_names)

    def _validate_inputs(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> None:
        """Validate input data for covariate shift analysis."""
        # Check sample sizes
        if len(source_df) < self.min_samples or len(target_df) < self.min_samples:
            warnings.warn(
                f"Sample sizes ({len(source_df)}, {len(target_df)}) below recommended minimum "
                f"({self.min_samples}) for reliable shift detection",
                UserWarning,
            )

        # Check dimensions match
        if source_df.shape[1] != target_df.shape[1]:
            raise ValueError(
                f"Source ({source_df.shape[1]}) and target ({target_df.shape[1]}) "
                "must have same number of variables"
            )

        # Check column names match
        if not source_df.columns.equals(target_df.columns):
            raise ValueError("Source and target data must have matching column names")

    def _analyze_single_variable(
        self,
        source_values: NDArray[Any],
        target_values: NDArray[Any],
        variable_name: str,
    ) -> DistributionDifference:
        """Analyze distribution differences for a single variable."""
        # Remove missing values
        source_clean = source_values[~pd.isna(source_values)]
        target_clean = target_values[~pd.isna(target_values)]

        # Calculate basic statistics
        source_mean = float(np.mean(source_clean))
        target_mean = float(np.mean(target_clean))
        source_std = float(np.std(source_clean, ddof=1))
        target_std = float(np.std(target_clean, ddof=1))

        # Calculate standardized mean difference (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(source_clean) - 1) * source_std**2
                + (len(target_clean) - 1) * target_std**2
            )
            / (len(source_clean) + len(target_clean) - 2)
        )

        if pooled_std == 0:
            smd = 0.0
        else:
            smd = (target_mean - source_mean) / pooled_std

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(source_clean, target_clean)

        # Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(source_clean, target_clean)

        # Classify severity
        severity = self._classify_shift_severity(abs(smd))

        return DistributionDifference(
            variable_name=variable_name,
            standardized_mean_diff=smd,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            wasserstein_distance=float(wasserstein_dist),
            severity=severity,
            source_mean=source_mean,
            target_mean=target_mean,
            source_std=source_std,
            target_std=target_std,
        )

    def _classify_shift_severity(self, abs_smd: float) -> ShiftSeverity:
        """Classify shift severity based on standardized mean difference."""
        if abs_smd <= self.smd_threshold_mild:
            return ShiftSeverity.MILD
        elif abs_smd <= self.smd_threshold_moderate:
            return ShiftSeverity.MODERATE
        else:
            return ShiftSeverity.SEVERE

    def _calculate_overall_shift_score(self) -> float:
        """Calculate overall shift score as mean absolute SMD."""
        if not self.distribution_differences:
            return 0.0

        return float(
            np.mean(
                [
                    abs(diff.standardized_mean_diff)
                    for diff in self.distribution_differences
                ]
            )
        )

    def _calculate_discriminative_accuracy(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> float:
        """Calculate how well a classifier can distinguish source from target.

        High accuracy indicates substantial covariate shift.
        """
        try:
            # Create combined dataset with population labels
            combined_data = pd.concat([source_df, target_df], ignore_index=True)
            labels = np.concatenate(
                [
                    np.zeros(len(source_df)),  # Source = 0
                    np.ones(len(target_df)),  # Target = 1
                ]
            )

            # Handle missing values
            combined_data = combined_data.fillna(combined_data.mean())

            # Use Random Forest for discrimination
            rf = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
            )

            # Cross-validation to get robust estimate
            cv_scores = cross_val_score(
                rf, combined_data, labels, cv=5, scoring="roc_auc", n_jobs=-1
            )

            return float(np.mean(cv_scores))

        except Exception as e:
            warnings.warn(
                f"Could not calculate discriminative accuracy: {str(e)}", UserWarning
            )
            return 0.5  # Random classifier performance

    def _count_shifts_by_severity(self, severity: ShiftSeverity) -> int:
        """Count number of variables with given shift severity."""
        return sum(
            1 for diff in self.distribution_differences if diff.severity == severity
        )

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on shift analysis."""
        recommendations: list[str] = []

        if not self.distribution_differences:
            return recommendations

        n_severe = self._count_shifts_by_severity(ShiftSeverity.SEVERE)
        n_moderate = self._count_shifts_by_severity(ShiftSeverity.MODERATE)
        max_smd = max(
            abs(diff.standardized_mean_diff) for diff in self.distribution_differences
        )

        # Overall assessment
        if n_severe > 0:
            recommendations.append(
                f"CRITICAL: {n_severe} variables show severe covariate shift (SMD > {self.smd_threshold_moderate}). "
                f"Maximum SMD is {max_smd:.3f}. Consider domain adaptation or reweighting methods."
            )
        elif n_moderate > 0:
            recommendations.append(
                f"WARNING: {n_moderate} variables show moderate covariate shift. "
                "Transportability methods recommended."
            )
        else:
            recommendations.append(
                "Good news: No severe covariate shifts detected. "
                "Simple transport may be sufficient."
            )

        # Discriminative accuracy assessment
        if self.discriminative_accuracy is not None:
            if self.discriminative_accuracy > 0.8:
                recommendations.append(
                    f"High discriminative accuracy ({self.discriminative_accuracy:.3f}) indicates "
                    "substantial population differences. Use advanced weighting methods."
                )
            elif self.discriminative_accuracy > 0.7:
                recommendations.append(
                    f"Moderate discriminative accuracy ({self.discriminative_accuracy:.3f}). "
                    "Standard reweighting should be effective."
                )

        # Variable-specific recommendations
        severe_vars = [
            str(diff.variable_name)
            for diff in self.distribution_differences
            if diff.severity == ShiftSeverity.SEVERE
        ]
        if severe_vars:
            recommendations.append(
                f"Focus reweighting on variables with largest shifts: {', '.join(severe_vars[:3])}"
                + ("..." if len(severe_vars) > 3 else "")
            )

        return recommendations

    def create_shift_summary_table(self) -> pd.DataFrame:
        """Create a summary table of covariate shifts.

        Returns:
            DataFrame with shift statistics for each variable
        """
        if not self.distribution_differences:
            return pd.DataFrame()

        rows = []
        for diff in self.distribution_differences:
            rows.append(
                {
                    "Variable": diff.variable_name,
                    "SMD": round(diff.standardized_mean_diff, 3),
                    "Abs_SMD": round(abs(diff.standardized_mean_diff), 3),
                    "KS_Statistic": round(diff.ks_statistic, 3),
                    "KS_P_Value": round(diff.ks_pvalue, 4),
                    "Wasserstein": round(diff.wasserstein_distance, 3),
                    "Severity": diff.severity.value,
                    "Effect_Size": diff.effect_size_interpretation,
                    "Significant": diff.is_significant,
                    "Source_Mean": round(diff.source_mean, 3),
                    "Target_Mean": round(diff.target_mean, 3),
                }
            )

        df = pd.DataFrame(rows)

        # Sort by absolute SMD (largest shifts first)
        df = df.sort_values("Abs_SMD", ascending=False)

        return df

    def print_diagnostic_summary(self) -> None:
        """Print a comprehensive diagnostic summary."""
        if not self.distribution_differences:
            print("No covariate shift analysis has been performed.")
            return

        print("=" * 60)
        print("COVARIATE SHIFT DIAGNOSTIC SUMMARY")
        print("=" * 60)

        # Overall metrics
        print(f"Overall Shift Score (Mean |SMD|): {self.overall_shift_score:.3f}")
        if self.discriminative_accuracy:
            print(f"Discriminative Accuracy: {self.discriminative_accuracy:.3f}")

        print(f"Number of variables analyzed: {len(self.distribution_differences)}")

        # Severity breakdown
        print("\nShift Severity Breakdown:")
        print(
            f"  Severe shifts:   {self._count_shifts_by_severity(ShiftSeverity.SEVERE)}"
        )
        print(
            f"  Moderate shifts: {self._count_shifts_by_severity(ShiftSeverity.MODERATE)}"
        )
        print(
            f"  Mild shifts:     {self._count_shifts_by_severity(ShiftSeverity.MILD)}"
        )

        # Top shifts
        top_shifts = sorted(
            self.distribution_differences,
            key=lambda x: abs(x.standardized_mean_diff),
            reverse=True,
        )[:5]

        print(f"\nTop {min(5, len(top_shifts))} Largest Shifts:")
        for i, diff in enumerate(top_shifts, 1):
            print(
                f"  {i}. {diff.variable_name}: SMD = {diff.standardized_mean_diff:.3f} "
                f"({diff.severity.value})"
            )

        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print("=" * 60)
