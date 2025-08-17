"""Targeted Maximum Transported Likelihood (TMTL) estimator for transportability."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from .weighting import DensityRatioEstimator, WeightingResult


class TargetedMaximumTransportedLikelihood(BaseEstimator):
    """Targeted Maximum Transported Likelihood (TMTL) estimator.

    Extends the Targeted Maximum Likelihood Estimation (TMLE) framework
    for transportability by incorporating transport weights and target
    population adjustments.

    This estimator provides valid causal effect estimates that generalize
    from a source population to a target population by accounting for
    covariate distribution differences.
    """

    def __init__(
        self,
        outcome_model: SklearnEstimator | None = None,
        treatment_model: SklearnEstimator | None = None,
        transport_weighting_method: str = "classification",
        max_transport_iterations: int = 100,
        transport_tolerance: float = 1e-6,
        trim_weights: bool = True,
        max_weight: float = 10.0,
        cross_fit: bool = True,
        n_folds: int = 5,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize TMTL estimator.

        Args:
            outcome_model: ML model for outcome regression
            treatment_model: ML model for treatment mechanism
            transport_weighting_method: Method for transport weights
            max_transport_iterations: Max iterations for transport targeting
            transport_tolerance: Convergence tolerance for transport targeting
            trim_weights: Whether to trim extreme transport weights
            max_weight: Maximum allowed transport weight
            cross_fit: Whether to use cross-fitting
            n_folds: Number of folds for cross-fitting
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        super().__init__(random_state=random_state, verbose=verbose)

        # Model specifications
        self.outcome_model = outcome_model or RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        self.treatment_model = treatment_model or LogisticRegression(
            random_state=random_state, max_iter=1000
        )

        # Transport weighting parameters
        self.transport_weighting_method = transport_weighting_method
        self.max_transport_iterations = max_transport_iterations
        self.transport_tolerance = transport_tolerance
        self.trim_weights = trim_weights
        self.max_weight = max_weight

        # Cross-fitting parameters
        self.cross_fit = cross_fit
        self.n_folds = n_folds

        # Fitted components
        self.transport_weights: NDArray[Any] | None = None
        self.transport_weighting_result: WeightingResult | None = None
        self.initial_estimates: dict[str, Any] | None = None
        self.targeted_estimates: dict[str, Any] | None = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the TMTL estimator to source data.

        Note: This method fits to source data only. Transport weights
        are computed when target data is provided to estimate_transported_ate.
        """
        if covariates is None:
            raise ValueError("TMTL requires covariates for transportability")

        # Store data for transport estimation
        self.source_treatment = treatment
        self.source_outcome = outcome
        self.source_covariates = covariates

        # Fit initial models on source data
        self._fit_initial_models()

    def _fit_initial_models(self) -> None:
        """Fit initial outcome and treatment models on source data."""
        if self.source_covariates is None:
            raise ValueError("No covariate data available")

        x = np.array(self.source_covariates.values)
        t = np.array(self.source_treatment.values)
        y = np.array(self.source_outcome.values)

        if self.cross_fit:
            self._fit_with_cross_fitting(x, t, y)
        else:
            self._fit_without_cross_fitting(x, t, y)

    def _fit_with_cross_fitting(
        self, x: NDArray[Any], t: NDArray[Any], y: NDArray[Any]
    ) -> None:
        """Fit models using cross-fitting to avoid overfitting bias."""
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Initialize arrays for out-of-fold predictions
        outcome_preds_0 = np.zeros(len(y))
        outcome_preds_1 = np.zeros(len(y))
        treatment_probs = np.zeros(len(t))

        for train_idx, val_idx in kf.split(x):
            # Split data
            x_train, x_val = x[train_idx], x[val_idx]
            t_train = t[train_idx]
            y_train = y[train_idx]

            # Fit outcome model on treated and control separately
            treated_mask_train = t_train == 1
            control_mask_train = t_train == 0

            if np.sum(treated_mask_train) > 0:
                outcome_model_1 = clone(self.outcome_model)
                outcome_model_1.fit(
                    x_train[treated_mask_train], y_train[treated_mask_train]
                )
                outcome_preds_1[val_idx] = outcome_model_1.predict(x_val)

            if np.sum(control_mask_train) > 0:
                outcome_model_0 = clone(self.outcome_model)
                outcome_model_0.fit(
                    x_train[control_mask_train], y_train[control_mask_train]
                )
                outcome_preds_0[val_idx] = outcome_model_0.predict(x_val)

            # Fit treatment model
            treatment_model = clone(self.treatment_model)
            treatment_model.fit(x_train, t_train)
            treatment_probs[val_idx] = treatment_model.predict_proba(x_val)[:, 1]

        # Store initial estimates
        self.initial_estimates = {
            "outcome_model_0": outcome_preds_0,
            "outcome_model_1": outcome_preds_1,
            "treatment_probs": treatment_probs,
        }

    def _fit_without_cross_fitting(
        self, x: NDArray[Any], t: NDArray[Any], y: NDArray[Any]
    ) -> None:
        """Fit models without cross-fitting (less robust but simpler)."""
        # Fit outcome models
        treated_mask = t == 1
        control_mask = t == 0

        outcome_model_1 = self.outcome_model
        outcome_model_0 = self.outcome_model

        if np.sum(treated_mask) > 0:
            from sklearn.base import clone

            outcome_model_1 = clone(self.outcome_model)
            outcome_model_1.fit(x[treated_mask], y[treated_mask])

        if np.sum(control_mask) > 0:
            from sklearn.base import clone

            outcome_model_0 = clone(self.outcome_model)
            outcome_model_0.fit(x[control_mask], y[control_mask])

        # Fit treatment model
        self.treatment_model.fit(x, t)

        # Generate predictions
        outcome_preds_1 = outcome_model_1.predict(x)
        outcome_preds_0 = outcome_model_0.predict(x)
        treatment_probs = self.treatment_model.predict_proba(x)[:, 1]

        self.initial_estimates = {
            "outcome_model_0": outcome_preds_0,
            "outcome_model_1": outcome_preds_1,
            "treatment_probs": treatment_probs,
        }

    def estimate_transported_ate(
        self,
        target_covariates: pd.DataFrame | NDArray[Any],
    ) -> CausalEffect:
        """Estimate transported ATE for target population.

        Args:
            target_covariates: Covariate data from target population

        Returns:
            CausalEffect with transported estimate and diagnostics

        Raises:
            ValueError: If estimator not fitted or invalid inputs
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before transported estimation")

        if self.source_covariates is None or self.initial_estimates is None:
            raise ValueError("No source data available for transport")

        # Estimate transport weights
        self._estimate_transport_weights(target_covariates)

        # Apply transport-specific targeting
        self._apply_transport_targeting()

        # Calculate transported ATE
        transported_ate = self._calculate_transported_ate()

        # Calculate confidence intervals using influence functions
        ate_se, ate_ci_lower, ate_ci_upper = self._calculate_transport_inference(
            transported_ate
        )

        # Create causal effect object
        causal_effect = CausalEffect(
            ate=transported_ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            method="TMTL",
            n_observations=len(self.source_treatment.values),
            diagnostics=self._create_transport_diagnostics(),
        )

        return causal_effect

    def _estimate_transport_weights(
        self, target_covariates: pd.DataFrame | NDArray[Any]
    ) -> None:
        """Estimate transport weights to match target population."""
        # Use density ratio estimation for transport weights
        weighting_estimator = DensityRatioEstimator(
            trim_weights=self.trim_weights,
            max_weight=self.max_weight,
            random_state=self.random_state,
        )

        self.transport_weighting_result = weighting_estimator.fit_weights(
            source_data=self.source_covariates.values,
            target_data=target_covariates,
        )

        self.transport_weights = self.transport_weighting_result.weights

        if self.verbose:
            print(
                f"Computed transport weights with effective sample size: "
                f"{self.transport_weighting_result.effective_sample_size:.1f}"
            )

    def _apply_transport_targeting(self) -> None:
        """Apply transport-specific targeting step."""
        if self.transport_weights is None or self.initial_estimates is None:
            raise ValueError("Transport weights or initial estimates not available")

        t = np.array(self.source_treatment.values)
        y = np.array(self.source_outcome.values)
        weights = self.transport_weights

        # Initial estimates
        q_0 = self.initial_estimates["outcome_model_0"]
        q_1 = self.initial_estimates["outcome_model_1"]
        g = self.initial_estimates["treatment_probs"]

        # Clip propensity scores to avoid extreme values
        g = np.clip(g, 0.01, 0.99)

        # Transport-weighted targeting
        transported_estimates = self._iterative_transport_targeting(
            y, t, q_0, q_1, g, weights
        )

        self.targeted_estimates = transported_estimates

    def _iterative_transport_targeting(
        self,
        y: NDArray[Any],
        t: NDArray[Any],
        q_0: NDArray[Any],
        q_1: NDArray[Any],
        g: NDArray[Any],
        weights: NDArray[Any],
    ) -> dict[str, NDArray[Any] | bool | int]:
        """Perform iterative targeting with transport weights."""
        q_0_updated = q_0.copy()
        q_1_updated = q_1.copy()

        # Initialize convergence variables before loop
        conv_0 = conv_1 = float("inf")

        for iteration in range(self.max_transport_iterations):
            # Calculate transport-weighted clever covariates
            h_0 = weights * (1 - t) / (1 - g)
            h_1 = weights * t / g

            # Calculate targeting parameters
            residuals_0 = y - q_0_updated
            residuals_1 = y - q_1_updated

            # Weighted regression for targeting
            if np.sum(h_0) > 0:
                epsilon_0 = np.sum(h_0 * residuals_0) / np.sum(h_0 * h_0)
            else:
                epsilon_0 = 0.0

            if np.sum(h_1) > 0:
                epsilon_1 = np.sum(h_1 * residuals_1) / np.sum(h_1 * h_1)
            else:
                epsilon_1 = 0.0

            # Update estimates
            q_0_new = q_0_updated + epsilon_0 * h_0
            q_1_new = q_1_updated + epsilon_1 * h_1

            # Check convergence
            conv_0 = np.abs(epsilon_0)
            conv_1 = np.abs(epsilon_1)

            if max(conv_0, conv_1) < self.transport_tolerance:
                if self.verbose:
                    print(
                        f"Transport targeting converged after {iteration + 1} iterations"
                    )
                break

            q_0_updated = q_0_new
            q_1_updated = q_1_new

        return {
            "Q_0_targeted": q_0_updated,
            "Q_1_targeted": q_1_updated,
            "convergence_achieved": max(conv_0, conv_1) < self.transport_tolerance,
            "n_iterations": iteration + 1,
        }

    def _calculate_transported_ate(self) -> float:
        """Calculate the transported ATE estimate."""
        if self.targeted_estimates is None or self.transport_weights is None:
            raise ValueError("Targeted estimates or transport weights not available")

        q_0 = self.targeted_estimates["Q_0_targeted"]
        q_1 = self.targeted_estimates["Q_1_targeted"]
        weights = self.transport_weights

        # Transport-weighted average treatment effect
        weighted_potential_outcome_1 = np.average(q_1, weights=weights)
        weighted_potential_outcome_0 = np.average(q_0, weights=weights)

        transported_ate = weighted_potential_outcome_1 - weighted_potential_outcome_0

        return float(transported_ate)

    def _calculate_transport_inference(
        self, transported_ate: float
    ) -> tuple[float, float, float]:
        """Calculate standard errors and confidence intervals for transported ATE."""
        if (
            self.targeted_estimates is None
            or self.transport_weights is None
            or self.initial_estimates is None
        ):
            warnings.warn(
                "Cannot calculate inference - missing estimates. Using bootstrap approximation.",
                UserWarning,
            )
            return self._bootstrap_transport_inference(transported_ate)

        # Calculate influence function for transported estimate
        influence_function = self._calculate_transport_influence_function()

        # Standard error from influence function
        n = len(influence_function)
        ate_se = float(np.sqrt(np.var(influence_function, ddof=1) / n))

        # Confidence intervals (normal approximation)
        z_alpha = 1.96  # 95% CI
        ate_ci_lower = transported_ate - z_alpha * ate_se
        ate_ci_upper = transported_ate + z_alpha * ate_se

        return ate_se, ate_ci_lower, ate_ci_upper

    def _calculate_transport_influence_function(self) -> NDArray[Any]:
        """Calculate influence function for transported ATE."""
        if (
            self.targeted_estimates is None
            or self.transport_weights is None
            or self.initial_estimates is None
        ):
            raise ValueError("Missing estimates for influence function calculation")

        y = np.array(self.source_outcome.values)
        t = np.array(self.source_treatment.values)
        weights = self.transport_weights
        g = self.initial_estimates["treatment_probs"]

        q_0 = self.targeted_estimates["Q_0_targeted"]
        q_1 = self.targeted_estimates["Q_1_targeted"]

        # Clip propensity scores
        g = np.clip(g, 0.01, 0.99)

        # Transport-weighted potential outcomes
        psi_1 = np.average(q_1, weights=weights)
        psi_0 = np.average(q_0, weights=weights)

        # Influence function components
        # Treatment group component
        if_1 = weights * (t / g) * (y - q_1) + weights * q_1 - psi_1

        # Control group component
        if_0 = weights * ((1 - t) / (1 - g)) * (y - q_0) + weights * q_0 - psi_0

        # Combined influence function
        influence_function = if_1 - if_0

        return np.asarray(influence_function, dtype=np.float64)

    def _bootstrap_transport_inference(
        self, transported_ate: float
    ) -> tuple[float, float, float]:
        """Bootstrap-based inference when influence function fails."""
        # Simple approximation based on transport weight variance
        if self.transport_weights is None:
            return 0.1, transported_ate - 0.2, transported_ate + 0.2

        # Rough standard error based on effective sample size and outcome variance
        eff_n = (np.sum(self.transport_weights) ** 2) / np.sum(
            self.transport_weights**2
        )
        # Base SE on outcome variance and effective sample size
        outcome_var = np.var(self.source_outcome.values, ddof=1)
        rough_se = np.sqrt(outcome_var / eff_n)

        ate_ci_lower = transported_ate - 1.96 * rough_se
        ate_ci_upper = transported_ate + 1.96 * rough_se

        return rough_se, ate_ci_lower, ate_ci_upper

    def _create_transport_diagnostics(self) -> dict[str, Any]:
        """Create diagnostic information for transported estimate."""
        diagnostics = {}

        if self.transport_weighting_result:
            diagnostics.update(
                {
                    "transport_effective_sample_size": self.transport_weighting_result.effective_sample_size,
                    "transport_max_weight": self.transport_weighting_result.max_weight,
                    "transport_weight_stability": self.transport_weighting_result.weight_stability_ratio,
                    "transport_convergence": self.transport_weighting_result.convergence_achieved,
                }
            )

        if self.targeted_estimates:
            diagnostics.update(
                {
                    "targeting_convergence": self.targeted_estimates.get(
                        "convergence_achieved", False
                    ),
                    "targeting_iterations": self.targeted_estimates.get(
                        "n_iterations", 0
                    ),
                }
            )

        return diagnostics

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Standard ATE estimation (not transported).

        This method is required by BaseEstimator but for TMTL we recommend
        using estimate_transported_ate with target population data.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before estimation")

        warnings.warn(
            "Using standard ATE estimation without transport. "
            "Consider using estimate_transported_ate() for transportability.",
            UserWarning,
        )

        # Calculate standard ATE without transport weights
        if self.initial_estimates is None:
            raise ValueError("No initial estimates available")

        q_0 = self.initial_estimates["outcome_model_0"]
        q_1 = self.initial_estimates["outcome_model_1"]

        ate = float(np.mean(q_1) - np.mean(q_0))

        return CausalEffect(
            ate=ate,
            method="TMTL_no_transport",
            n_observations=len(self.source_treatment.values),
        )
