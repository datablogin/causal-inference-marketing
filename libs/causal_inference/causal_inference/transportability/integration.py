"""Integration framework for transportability with existing estimators."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.base import (
    BaseEstimator,
    CausalEffect,
)
from .diagnostics import CovariateShiftDiagnostics, ShiftSeverity
from .weighting import (
    DensityRatioEstimator,
    TransportabilityWeightingInterface,
    WeightingResult,
)


class TransportabilityEstimator:
    """Wrapper that adds transportability capabilities to existing estimators.

    This class allows any existing causal inference estimator to be used
    for transportability analysis by automatically applying transport weights
    and adjusting estimates for target populations.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        weighting_method: str = "classification",
        auto_diagnostics: bool = True,
        trim_weights: bool = True,
        max_weight: float = 10.0,
        min_shift_threshold: float = 0.1,
        **weighting_kwargs: Any,
    ) -> None:
        """Initialize transportability wrapper.

        Args:
            base_estimator: Any fitted causal inference estimator
            weighting_method: Method for computing transport weights
            auto_diagnostics: Whether to automatically run shift diagnostics
            trim_weights: Whether to trim extreme weights
            max_weight: Maximum allowed weight value
            min_shift_threshold: Minimum shift to apply transportability
            **weighting_kwargs: Additional arguments for weighting method
        """
        self.base_estimator = base_estimator
        self.weighting_method = weighting_method
        self.auto_diagnostics = auto_diagnostics
        self.trim_weights = trim_weights
        self.max_weight = max_weight
        self.min_shift_threshold = min_shift_threshold
        self.weighting_kwargs = weighting_kwargs

        # State tracking
        self.shift_diagnostics: Optional[CovariateShiftDiagnostics] = None
        self.transport_weights: Optional[NDArray[Any]] = None
        self.weighting_result: Optional[WeightingResult] = None
        self.last_target_data: Optional[pd.DataFrame] = None

    def estimate_transported_effect(
        self,
        target_covariates: pd.DataFrame | NDArray[Any],
        run_diagnostics: bool = True,
        force_transport: bool = False,
    ) -> CausalEffect:
        """Estimate causal effect transported to target population.

        Args:
            target_covariates: Covariate data from target population
            run_diagnostics: Whether to run covariate shift diagnostics
            force_transport: Whether to apply transport even for small shifts

        Returns:
            CausalEffect transported to target population

        Raises:
            ValueError: If base estimator not fitted or invalid inputs
        """
        if not self.base_estimator.is_fitted:
            raise ValueError("Base estimator must be fitted before transport")

        if self.base_estimator.covariate_data is None:
            raise ValueError("Base estimator must have covariate data for transport")

        # Convert target data to DataFrame
        target_df = self._ensure_dataframe(target_covariates)

        # Run covariate shift diagnostics if requested
        if run_diagnostics or self.auto_diagnostics:
            self._run_shift_diagnostics(target_df)

        # Check if transport is needed
        if not force_transport and self._should_skip_transport():
            warnings.warn(
                "Covariate shift below threshold. Using original estimates. "
                "Set force_transport=True to override.",
                UserWarning,
            )
            return self.base_estimator.estimate_ate()

        # Estimate transport weights
        self._estimate_transport_weights(target_df)

        # Apply transport to base estimator
        transported_effect = self._apply_transport_to_estimator()

        # Store target data for validation
        self.last_target_data = target_df

        return transported_effect

    def _ensure_dataframe(
        self, data: Union[pd.DataFrame, NDArray[Any]]
    ) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def _run_shift_diagnostics(self, target_df: pd.DataFrame) -> None:
        """Run covariate shift diagnostics."""
        self.shift_diagnostics = CovariateShiftDiagnostics()

        if self.base_estimator.covariate_data is None:
            raise ValueError("Base estimator must have covariate data")
        source_data = self.base_estimator.covariate_data.values

        # Ensure both source and target have consistent DataFrame format
        source_df = self._ensure_dataframe(source_data)

        # Run analysis
        results = self.shift_diagnostics.analyze_covariate_shift(
            source_data=source_df,
            target_data=target_df,
        )

        if hasattr(self.base_estimator, "verbose") and self.base_estimator.verbose:
            print("=== COVARIATE SHIFT ANALYSIS ===")
            print(f"Overall shift score: {results['overall_shift_score']:.3f}")
            print(f"Discriminative accuracy: {results['discriminative_accuracy']:.3f}")
            print(f"Variables with severe shifts: {results['n_severe_shifts']}")
            print(f"Variables with moderate shifts: {results['n_moderate_shifts']}")

            if results["recommendations"]:
                print("\nRecommendations:")
                for rec in results["recommendations"][:3]:
                    print(f"  • {rec}")

    def _should_skip_transport(self) -> bool:
        """Check if transport should be skipped due to minimal shift."""
        if self.shift_diagnostics is None:
            return False

        # Check overall shift score
        if (
            self.shift_diagnostics.overall_shift_score is not None
            and self.shift_diagnostics.overall_shift_score < self.min_shift_threshold
        ):
            return True

        # Check if no severe or moderate shifts
        n_severe = self.shift_diagnostics._count_shifts_by_severity(
            ShiftSeverity.SEVERE
        )
        n_moderate = self.shift_diagnostics._count_shifts_by_severity(
            ShiftSeverity.MODERATE
        )

        return n_severe == 0 and n_moderate == 0

    def _estimate_transport_weights(self, target_df: pd.DataFrame) -> None:
        """Estimate transport weights for target population."""
        if self.base_estimator.covariate_data is None:
            raise ValueError("Base estimator must have covariate data")
        source_data = self.base_estimator.covariate_data.values

        # Use density ratio estimation
        # Prepare kwargs, avoiding duplicate random_state
        estimator_kwargs = self.weighting_kwargs.copy()
        if "random_state" not in estimator_kwargs:
            estimator_kwargs["random_state"] = getattr(
                self.base_estimator, "random_state", None
            )

        weighting_estimator = DensityRatioEstimator(
            trim_weights=self.trim_weights,
            max_weight=self.max_weight,
            **estimator_kwargs,
        )

        self.weighting_result = weighting_estimator.fit_weights(
            source_data=source_data,
            target_data=target_df,
        )

        self.transport_weights = self.weighting_result.weights

        if hasattr(self.base_estimator, "verbose") and self.base_estimator.verbose:
            print("\n=== TRANSPORT WEIGHTS ===")
            print(
                f"Effective sample size: {self.weighting_result.effective_sample_size:.1f}"
            )
            print(f"Max weight: {self.weighting_result.max_weight:.2f}")
            print(
                f"Weight stability ratio: {self.weighting_result.weight_stability_ratio:.2f}"
            )
            print(
                f"Relative efficiency: {self.weighting_result.relative_efficiency:.3f}"
            )

    def _apply_transport_to_estimator(self) -> CausalEffect:
        """Apply transport weights to base estimator."""
        if self.transport_weights is None:
            raise ValueError("Transport weights not computed")

        # Get base estimator type and apply appropriate transport method
        estimator_name = self.base_estimator.__class__.__name__

        if hasattr(self.base_estimator, "_apply_sample_weights"):
            # Estimator supports sample weights directly
            return self._apply_weighted_estimation()
        elif estimator_name in ["IPWEstimator", "AIPW"]:
            # IPW-based estimators - modify weights
            return self._apply_ipw_transport()
        elif estimator_name in ["GComputationEstimator"]:
            # G-computation - reweight predictions
            return self._apply_gcomp_transport()
        elif estimator_name in ["DoublyRobustML", "TMLE"]:
            # Advanced estimators - weighted targeting
            return self._apply_ml_transport()
        else:
            # Generic approach - reweight outcomes
            return self._apply_generic_transport()

    def _apply_weighted_estimation(self) -> CausalEffect:
        """Apply transport using estimator's built-in sample weight support."""
        # Re-run estimation with transport weights
        original_effect = self.base_estimator.estimate_ate()

        # This is a placeholder - would need estimator-specific implementation
        transported_ate = self._calculate_weighted_ate()

        return CausalEffect(
            ate=transported_ate,
            method=f"{original_effect.method}_transported",
            n_observations=original_effect.n_observations,
            diagnostics=self._create_transport_diagnostics(original_effect),
        )

    def _apply_ipw_transport(self) -> CausalEffect:
        """Apply transport to IPW-based estimators."""
        # Combine transport weights with propensity weights
        original_effect = self.base_estimator.estimate_ate()

        # For IPW estimators, multiply existing weights by transport weights
        if hasattr(self.base_estimator, "propensity_weights"):
            combined_weights = (
                self.base_estimator.propensity_weights * self.transport_weights
            )
        else:
            combined_weights = self.transport_weights

        transported_ate = self._calculate_ipw_transported_ate(combined_weights)

        return CausalEffect(
            ate=transported_ate,
            method=f"{original_effect.method}_transported",
            n_observations=original_effect.n_observations,
            diagnostics=self._create_transport_diagnostics(original_effect),
        )

    def _apply_gcomp_transport(self) -> CausalEffect:
        """Apply transport to G-computation estimators."""
        original_effect = self.base_estimator.estimate_ate()

        # For G-computation, reweight the predicted potential outcomes
        if hasattr(self.base_estimator, "potential_outcomes_0") and hasattr(
            self.base_estimator, "potential_outcomes_1"
        ):
            y0_pred = self.base_estimator.potential_outcomes_0
            y1_pred = self.base_estimator.potential_outcomes_1

            # Transport-weighted means
            transported_y0 = np.average(y0_pred, weights=self.transport_weights)
            transported_y1 = np.average(y1_pred, weights=self.transport_weights)

            transported_ate = transported_y1 - transported_y0
        else:
            # Fallback to generic approach
            transported_ate = self._calculate_weighted_ate()

        return CausalEffect(
            ate=transported_ate,
            method=f"{original_effect.method}_transported",
            n_observations=original_effect.n_observations,
            diagnostics=self._create_transport_diagnostics(original_effect),
        )

    def _apply_ml_transport(self) -> CausalEffect:
        """Apply transport to ML-based estimators (DML, TMLE)."""
        # For advanced ML estimators, use influence function reweighting
        original_effect = self.base_estimator.estimate_ate()

        # This would require access to influence functions or efficient scores
        # For now, use generic weighted approach
        transported_ate = self._calculate_weighted_ate()

        return CausalEffect(
            ate=transported_ate,
            method=f"{original_effect.method}_transported",
            n_observations=original_effect.n_observations,
            diagnostics=self._create_transport_diagnostics(original_effect),
        )

    def _apply_generic_transport(self) -> CausalEffect:
        """Generic transport approach using outcome reweighting."""
        original_effect = self.base_estimator.estimate_ate()
        transported_ate = self._calculate_weighted_ate()

        return CausalEffect(
            ate=transported_ate,
            method=f"{original_effect.method}_transported",
            n_observations=original_effect.n_observations,
            diagnostics=self._create_transport_diagnostics(original_effect),
        )

    def _calculate_weighted_ate(self) -> float:
        """Calculate ATE using transport weights on outcomes."""
        if (
            self.base_estimator.treatment_data is None
            or self.base_estimator.outcome_data is None
        ):
            raise ValueError("No treatment or outcome data available")

        T = np.array(self.base_estimator.treatment_data.values)
        Y = np.array(self.base_estimator.outcome_data.values)

        if self.transport_weights is None:
            raise ValueError("Transport weights not computed")
        weights = self.transport_weights

        # Transport-weighted means by treatment group
        treated_mask = T == 1
        control_mask = T == 0

        if np.sum(treated_mask) == 0 or np.sum(control_mask) == 0:
            raise ValueError("Need both treated and control units for ATE estimation")

        y1_mean = np.average(Y[treated_mask], weights=weights[treated_mask])
        y0_mean = np.average(Y[control_mask], weights=weights[control_mask])

        return float(y1_mean - y0_mean)

    def _calculate_ipw_transported_ate(self, combined_weights: NDArray[Any]) -> float:
        """Calculate IPW ATE with combined transport and propensity weights."""
        if (
            self.base_estimator.treatment_data is None
            or self.base_estimator.outcome_data is None
        ):
            raise ValueError("No treatment or outcome data available")

        T = np.array(self.base_estimator.treatment_data.values)
        Y = np.array(self.base_estimator.outcome_data.values)

        # IPW with transport weights
        treated_contribution = np.sum(combined_weights * T * Y) / np.sum(
            combined_weights * T
        )
        control_contribution = np.sum(combined_weights * (1 - T) * Y) / np.sum(
            combined_weights * (1 - T)
        )

        return float(treated_contribution - control_contribution)

    def _create_transport_diagnostics(
        self, original_effect: CausalEffect
    ) -> dict[str, Any]:
        """Create comprehensive diagnostics for transported estimate."""
        diagnostics = original_effect.diagnostics or {}

        # Add transport-specific diagnostics
        if self.weighting_result:
            diagnostics.update(
                {
                    "transport_effective_sample_size": self.weighting_result.effective_sample_size,
                    "transport_max_weight": self.weighting_result.max_weight,
                    "transport_stability_ratio": self.weighting_result.weight_stability_ratio,
                    "transport_relative_efficiency": self.weighting_result.relative_efficiency,
                }
            )

        if self.shift_diagnostics:
            diagnostics.update(
                {
                    "overall_shift_score": self.shift_diagnostics.overall_shift_score,
                    "discriminative_accuracy": self.shift_diagnostics.discriminative_accuracy,
                    "n_severe_shifts": self.shift_diagnostics._count_shifts_by_severity(
                        ShiftSeverity.SEVERE
                    ),
                    "n_moderate_shifts": self.shift_diagnostics._count_shifts_by_severity(
                        ShiftSeverity.MODERATE
                    ),
                }
            )

        diagnostics["transportability_applied"] = True
        diagnostics["weighting_method"] = self.weighting_method

        return diagnostics

    def validate_transport_quality(self) -> dict[str, Any]:
        """Validate the quality of transportability adjustment.

        Returns:
            Dictionary with validation metrics
        """
        if (
            self.transport_weights is None
            or self.last_target_data is None
            or self.base_estimator.covariate_data is None
        ):
            raise ValueError("Transport estimation must be run first")

        # Use weighting validation
        weighting_estimator = TransportabilityWeightingInterface()
        validation_results = weighting_estimator.validate_weights(
            weights=self.transport_weights,
            source_data=self.base_estimator.covariate_data.values,
            target_data=self.last_target_data,
        )

        # Add shift diagnostics to validation
        if self.shift_diagnostics:
            validation_results["shift_analysis"] = {
                "overall_shift_score": self.shift_diagnostics.overall_shift_score,
                "discriminative_accuracy": self.shift_diagnostics.discriminative_accuracy,
                "recommendations": self.shift_diagnostics._generate_recommendations(),
            }

        return validation_results

    def create_transport_summary(self) -> str:
        """Create a summary report of transportability analysis."""
        if not self.shift_diagnostics:
            return "No transportability analysis has been performed."

        lines = [
            "=" * 50,
            "TRANSPORTABILITY ANALYSIS SUMMARY",
            "=" * 50,
        ]

        # Shift diagnostics
        if self.shift_diagnostics.overall_shift_score is not None:
            lines.append(
                f"Overall Shift Score: {self.shift_diagnostics.overall_shift_score:.3f}"
            )

        if self.shift_diagnostics.discriminative_accuracy is not None:
            lines.append(
                f"Discriminative Accuracy: {self.shift_diagnostics.discriminative_accuracy:.3f}"
            )

        # Severity breakdown
        n_severe = self.shift_diagnostics._count_shifts_by_severity(
            ShiftSeverity.SEVERE
        )
        n_moderate = self.shift_diagnostics._count_shifts_by_severity(
            ShiftSeverity.MODERATE
        )
        n_mild = self.shift_diagnostics._count_shifts_by_severity(ShiftSeverity.MILD)

        lines.extend(
            [
                f"Severe shifts: {n_severe}",
                f"Moderate shifts: {n_moderate}",
                f"Mild shifts: {n_mild}",
            ]
        )

        # Transport weights
        if self.weighting_result:
            lines.extend(
                [
                    "",
                    "Transport Weights:",
                    f"  Effective sample size: {self.weighting_result.effective_sample_size:.1f}",
                    f"  Maximum weight: {self.weighting_result.max_weight:.2f}",
                    f"  Stability ratio: {self.weighting_result.weight_stability_ratio:.2f}",
                    f"  Relative efficiency: {self.weighting_result.relative_efficiency:.3f}",
                ]
            )

        # Recommendations
        recommendations = self.shift_diagnostics._generate_recommendations()
        if recommendations:
            lines.extend(
                [
                    "",
                    "Recommendations:",
                ]
            )
            for i, rec in enumerate(recommendations[:3], 1):
                lines.append(f"  {i}. {rec}")

        lines.append("=" * 50)

        return "\n".join(lines)
