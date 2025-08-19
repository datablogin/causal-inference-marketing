"""Spillover Effect Estimation.

This module provides tools for estimating spillover effects where outcomes
are modeled as a function of both own treatment and neighbor treatment exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from .exposure_mapping import ExposureMapping


@dataclass
class SpilloverResults:
    """Results from spillover effect estimation.

    Contains estimates of direct effects, spillover effects, and total effects.
    """

    # Direct effects (own treatment) - required fields first
    direct_effect: float
    spillover_effect: float  # Moved required field up

    # Optional fields with defaults
    direct_effect_se: float | None = None
    direct_effect_ci_lower: float | None = None
    direct_effect_ci_upper: float | None = None
    direct_effect_pvalue: float | None = None

    # Spillover effects (neighbor treatment)
    spillover_effect_se: float | None = None
    spillover_effect_ci_lower: float | None = None
    spillover_effect_ci_upper: float | None = None
    spillover_effect_pvalue: float | None = None

    # Total effects (direct + spillover) - optional fields with defaults
    total_effect_se: float | None = None
    total_effect_ci_lower: float | None = None
    total_effect_ci_upper: float | None = None

    # Model diagnostics
    model_r_squared: float | None = None
    model_aic: float | None = None
    model_bic: float | None = None
    residual_std: float | None = None

    # Spillover-specific diagnostics
    spillover_mechanism: str = "additive"
    exposure_strength: float = 1.0
    n_spillover_relationships: int = 0
    max_spillover_exposure: float = 0.0
    mean_spillover_exposure: float = 0.0

    @property
    def total_effect(self) -> float:
        """Calculate total effect as sum of direct and spillover effects."""
        return self.direct_effect + self.spillover_effect

    @property
    def spillover_ratio(self) -> float:
        """Ratio of spillover effect to direct effect."""
        if self.direct_effect == 0:
            return float("inf") if self.spillover_effect != 0 else 0
        return abs(self.spillover_effect) / abs(self.direct_effect)

    @property
    def is_spillover_significant(self) -> bool:
        """Check if spillover effect is statistically significant."""
        if self.spillover_effect_pvalue is None:
            return False
        return self.spillover_effect_pvalue < 0.05


class SpilloverModel(BaseModel):
    """Base configuration for spillover estimation models."""

    mechanism: str = Field(
        default="additive",
        description="Spillover mechanism: 'additive', 'multiplicative', or 'threshold'",
    )
    include_interactions: bool = Field(
        default=False, description="Whether to include interaction terms"
    )

    model_config = {"arbitrary_types_allowed": True}


class AdditiveSpilloverModel(SpilloverModel):
    """Additive spillover model: Y = f(own_treatment) + g(neighbor_treatment)."""

    mechanism: str = Field(default="additive", frozen=True)


class MultiplicativeSpilloverModel(SpilloverModel):
    """Multiplicative spillover model: Y = f(own_treatment) * g(neighbor_treatment)."""

    mechanism: str = Field(default="multiplicative", frozen=True)
    base_effect: float = Field(
        default=1.0,
        description="Base effect when no treatment (multiplicative baseline)",
    )


class ThresholdSpilloverModel(SpilloverModel):
    """Threshold spillover model: spillover only above exposure threshold."""

    mechanism: str = Field(default="threshold", frozen=True)
    threshold: float = Field(
        default=0.5, description="Minimum exposure for spillover to occur"
    )


class SpilloverEstimator(BaseEstimator):
    """Estimate spillover effects using exposure mapping and treatment data.

    Models outcomes as a function of both own treatment assignment and
    neighbor treatment exposure, supporting different spillover mechanisms.
    """

    def __init__(
        self,
        spillover_model: SpilloverModel,
        exposure_mapping: ExposureMapping,
        estimator_type: str = "linear",
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize spillover estimator.

        Args:
            spillover_model: Configuration for spillover mechanism
            exposure_mapping: Mapping of spillover exposure relationships
            estimator_type: Type of estimator ('linear', 'forest')
            confidence_level: Confidence level for intervals
            random_state: Random seed
            verbose: Whether to print verbose output
        """
        super().__init__(random_state=random_state, verbose=verbose)

        self.spillover_model = spillover_model
        self.exposure_mapping = exposure_mapping
        self.estimator_type = estimator_type
        self.confidence_level = confidence_level

        # Internal state
        self._fitted_model: Any | None = None
        self._spillover_exposure: NDArray[Any] | None = None
        self._spillover_results: SpilloverResults | None = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit spillover estimation model."""
        # Validate that treatment units match exposure mapping
        self._validate_exposure_alignment(treatment)

        # Calculate spillover exposure for each unit
        self._spillover_exposure = self._calculate_spillover_exposure(treatment)

        # Create design matrix
        X = self._create_design_matrix(treatment, covariates)
        y = np.array(outcome.values)

        # Fit the model
        if self.estimator_type == "linear":
            self._fitted_model = LinearRegression()
        elif self.estimator_type == "forest":
            self._fitted_model = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown estimator type: {self.estimator_type}")

        self._fitted_model.fit(X, y)

        if self.verbose:
            print(f"Fitted {self.estimator_type} spillover model")
            print(f"Spillover mechanism: {self.spillover_model.mechanism}")
            if hasattr(self._fitted_model, "score"):
                r2 = self._fitted_model.score(X, y)
                print(f"Model R²: {r2:.4f}")

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate average treatment effect accounting for spillover."""
        if self._fitted_model is None or self._spillover_exposure is None:
            raise ValueError("Model must be fitted before estimation")

        # Estimate spillover results
        spillover_results = self._estimate_spillover_effects()

        # Store spillover results
        self._spillover_results = spillover_results

        # Create CausalEffect object with spillover information
        causal_effect = CausalEffect(
            ate=spillover_results.total_effect,
            ate_se=spillover_results.total_effect_se,
            ate_ci_lower=spillover_results.total_effect_ci_lower,
            ate_ci_upper=spillover_results.total_effect_ci_upper,
            method=f"SpilloverEstimator_{self.spillover_model.mechanism}",
            n_observations=len(self.treatment_data.values)
            if self.treatment_data
            else None,
            diagnostics={
                "direct_effect": spillover_results.direct_effect,
                "spillover_effect": spillover_results.spillover_effect,
                "spillover_ratio": spillover_results.spillover_ratio,
                "spillover_mechanism": spillover_results.spillover_mechanism,
                "model_r_squared": spillover_results.model_r_squared,
                "exposure_strength": spillover_results.exposure_strength,
                "n_spillover_relationships": spillover_results.n_spillover_relationships,
            },
        )

        return causal_effect

    def _validate_exposure_alignment(self, treatment: TreatmentData) -> None:
        """Validate that treatment data aligns with exposure mapping."""
        n_treatment_units = len(treatment.values)
        n_exposure_units = self.exposure_mapping.n_units

        if n_treatment_units != n_exposure_units:
            raise ValueError(
                f"Number of treatment units ({n_treatment_units}) doesn't match "
                f"exposure mapping units ({n_exposure_units})"
            )

    def _calculate_spillover_exposure(self, treatment: TreatmentData) -> NDArray[Any]:
        """Calculate spillover exposure for each unit."""
        treatment_array = np.array(treatment.values)
        exposure_matrix = self.exposure_mapping.exposure_matrix

        # Spillover exposure = sum of (exposure_weight * neighbor_treatment)
        spillover_exposure = np.dot(exposure_matrix, treatment_array)

        return spillover_exposure

    def _create_design_matrix(
        self, treatment: TreatmentData, covariates: CovariateData | None = None
    ) -> NDArray[Any]:
        """Create design matrix for spillover estimation."""
        features = []

        # Own treatment
        own_treatment = np.array(treatment.values).reshape(-1, 1)
        features.append(own_treatment)

        # Spillover exposure
        spillover_exposure = self._spillover_exposure.reshape(-1, 1)

        # Apply spillover mechanism
        if self.spillover_model.mechanism == "additive":
            features.append(spillover_exposure)
        elif self.spillover_model.mechanism == "multiplicative":
            # Multiplicative: include log(spillover_exposure + 1) or interaction
            if self.spillover_model.include_interactions:
                interaction = own_treatment.flatten() * spillover_exposure.flatten()
                features.append(interaction.reshape(-1, 1))
            else:
                features.append(np.log1p(spillover_exposure))
        elif self.spillover_model.mechanism == "threshold":
            # Threshold: binary indicator above threshold
            threshold_spillover = (
                spillover_exposure > self.spillover_model.threshold
            ).astype(float)
            features.append(threshold_spillover)

        # Include interactions if requested
        if (
            self.spillover_model.include_interactions
            and self.spillover_model.mechanism != "multiplicative"
        ):
            interaction = own_treatment.flatten() * spillover_exposure.flatten()
            features.append(interaction.reshape(-1, 1))

        # Include covariates if provided
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                cov_array = covariates.values.values
            else:
                cov_array = np.array(covariates.values)
            if len(cov_array.shape) == 1:
                cov_array = cov_array.reshape(-1, 1)
            features.append(cov_array)

        # Combine all features
        X = np.hstack(features)

        return X

    def _estimate_spillover_effects(self) -> SpilloverResults:
        """Estimate direct and spillover effects from fitted model."""
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before spillover estimation")

        # For linear models, extract coefficients
        if hasattr(self._fitted_model, "coef_"):
            coefficients = self._fitted_model.coef_

            # Direct effect (own treatment coefficient)
            direct_effect = float(coefficients[0])

            # Spillover effect (spillover exposure coefficient)
            spillover_effect = float(coefficients[1])

            # Calculate standard errors and p-values if linear model
            if self.estimator_type == "linear":
                se_results = self._calculate_linear_standard_errors()
                direct_effect_se = se_results.get("direct_se")
                spillover_effect_se = se_results.get("spillover_se")
                direct_effect_pvalue = se_results.get("direct_pvalue")
                spillover_effect_pvalue = se_results.get("spillover_pvalue")
            else:
                direct_effect_se = None
                spillover_effect_se = None
                direct_effect_pvalue = None
                spillover_effect_pvalue = None

        else:
            # For non-linear models, use prediction-based estimation
            effects = self._estimate_effects_by_prediction()
            direct_effect = effects["direct"]
            spillover_effect = effects["spillover"]
            direct_effect_se = None
            spillover_effect_se = None
            direct_effect_pvalue = None
            spillover_effect_pvalue = None

        # Calculate confidence intervals
        alpha = 1 - self.confidence_level

        if direct_effect_se is not None:
            t_crit = stats.t.ppf(1 - alpha / 2, df=len(self.treatment_data.values) - 2)
            direct_effect_ci_lower = direct_effect - t_crit * direct_effect_se
            direct_effect_ci_upper = direct_effect + t_crit * direct_effect_se
            spillover_effect_ci_lower = spillover_effect - t_crit * spillover_effect_se
            spillover_effect_ci_upper = spillover_effect + t_crit * spillover_effect_se
        else:
            direct_effect_ci_lower = None
            direct_effect_ci_upper = None
            spillover_effect_ci_lower = None
            spillover_effect_ci_upper = None

        # Calculate total effect confidence interval
        if direct_effect_se is not None and spillover_effect_se is not None:
            total_effect_se = np.sqrt(direct_effect_se**2 + spillover_effect_se**2)
            total_effect_ci_lower = (
                direct_effect + spillover_effect
            ) - t_crit * total_effect_se
            total_effect_ci_upper = (
                direct_effect + spillover_effect
            ) + t_crit * total_effect_se
        else:
            total_effect_se = None
            total_effect_ci_lower = None
            total_effect_ci_upper = None

        # Model diagnostics
        model_r_squared = None
        if hasattr(self._fitted_model, "score"):
            X = self._create_design_matrix(self.treatment_data, self.covariate_data)
            y = np.array(self.outcome_data.values)
            model_r_squared = self._fitted_model.score(X, y)

        # Spillover diagnostics
        exposure_matrix = self.exposure_mapping.exposure_matrix
        n_spillover_relationships = int(np.sum(exposure_matrix > 0))
        max_spillover_exposure = float(np.max(self._spillover_exposure))
        mean_spillover_exposure = float(np.mean(self._spillover_exposure))

        return SpilloverResults(
            direct_effect=direct_effect,
            direct_effect_se=direct_effect_se,
            direct_effect_ci_lower=direct_effect_ci_lower,
            direct_effect_ci_upper=direct_effect_ci_upper,
            direct_effect_pvalue=direct_effect_pvalue,
            spillover_effect=spillover_effect,
            spillover_effect_se=spillover_effect_se,
            spillover_effect_ci_lower=spillover_effect_ci_lower,
            spillover_effect_ci_upper=spillover_effect_ci_upper,
            spillover_effect_pvalue=spillover_effect_pvalue,
            total_effect_se=total_effect_se,
            total_effect_ci_lower=total_effect_ci_lower,
            total_effect_ci_upper=total_effect_ci_upper,
            model_r_squared=model_r_squared,
            spillover_mechanism=self.spillover_model.mechanism,
            exposure_strength=1.0,  # Could be made configurable
            n_spillover_relationships=n_spillover_relationships,
            max_spillover_exposure=max_spillover_exposure,
            mean_spillover_exposure=mean_spillover_exposure,
        )

    def _calculate_linear_standard_errors(self) -> dict[str, float]:
        """Calculate standard errors for linear model coefficients."""
        if not hasattr(self._fitted_model, "coef_"):
            return {}

        X = self._create_design_matrix(self.treatment_data, self.covariate_data)
        y = np.array(self.outcome_data.values)

        # Calculate residuals
        y_pred = self._fitted_model.predict(X)
        residuals = y - y_pred
        n = len(y)
        p = X.shape[1]

        # Calculate standard errors
        mse = np.sum(residuals**2) / (n - p)
        var_covar_matrix = mse * np.linalg.inv(X.T @ X)
        standard_errors = np.sqrt(np.diag(var_covar_matrix))

        # Calculate t-statistics and p-values
        t_stats = self._fitted_model.coef_ / standard_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))

        return {
            "direct_se": float(standard_errors[0]),
            "spillover_se": float(standard_errors[1]),
            "direct_pvalue": float(p_values[0]),
            "spillover_pvalue": float(p_values[1]),
        }

    def _estimate_effects_by_prediction(self) -> dict[str, float]:
        """Estimate effects using prediction differences for non-linear models."""
        # Create counterfactual scenarios
        X_base = self._create_design_matrix(self.treatment_data, self.covariate_data)

        # Scenario 1: No treatment, no spillover
        X_no_treatment = X_base.copy()
        X_no_treatment[:, 0] = 0  # Own treatment
        X_no_treatment[:, 1] = 0  # Spillover exposure

        # Scenario 2: Treatment, no spillover
        X_direct_only = X_base.copy()
        X_direct_only[:, 0] = 1  # Own treatment
        X_direct_only[:, 1] = 0  # Spillover exposure

        # Scenario 3: No treatment, spillover
        X_spillover_only = X_base.copy()
        X_spillover_only[:, 0] = 0  # Own treatment
        X_spillover_only[:, 1] = np.mean(self._spillover_exposure)  # Average spillover

        # Make predictions
        pred_no_treatment = self._fitted_model.predict(X_no_treatment)
        pred_direct_only = self._fitted_model.predict(X_direct_only)
        pred_spillover_only = self._fitted_model.predict(X_spillover_only)

        # Calculate effects
        direct_effect = np.mean(pred_direct_only - pred_no_treatment)
        spillover_effect = np.mean(pred_spillover_only - pred_no_treatment)

        return {"direct": float(direct_effect), "spillover": float(spillover_effect)}

    def get_spillover_results(self) -> SpilloverResults | None:
        """Get detailed spillover estimation results."""
        return self._spillover_results

    def predict_spillover_effects(
        self, treatment_scenario: NDArray[Any], covariates: NDArray[Any] | None = None
    ) -> dict[str, NDArray[Any]]:
        """Predict outcomes under different spillover scenarios.

        Args:
            treatment_scenario: Treatment assignment scenario
            covariates: Optional covariate values

        Returns:
            Dictionary with predictions for different spillover levels
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        n_units = len(treatment_scenario)

        # Calculate spillover exposure under this scenario
        spillover_exposure = np.dot(
            self.exposure_mapping.exposure_matrix, treatment_scenario
        )

        # Create design matrices for different scenarios
        scenarios = {}

        # Current scenario
        X_current = self._create_design_matrix_from_arrays(
            treatment_scenario, spillover_exposure, covariates
        )
        scenarios["current"] = self._fitted_model.predict(X_current)

        # No spillover scenario
        X_no_spillover = self._create_design_matrix_from_arrays(
            treatment_scenario, np.zeros(n_units), covariates
        )
        scenarios["no_spillover"] = self._fitted_model.predict(X_no_spillover)

        # Spillover contribution
        scenarios["spillover_contribution"] = (
            scenarios["current"] - scenarios["no_spillover"]
        )

        return scenarios

    def _create_design_matrix_from_arrays(
        self,
        treatment: NDArray[Any],
        spillover_exposure: NDArray[Any],
        covariates: NDArray[Any] | None = None,
    ) -> NDArray[Any]:
        """Create design matrix from numpy arrays."""
        features = [treatment.reshape(-1, 1), spillover_exposure.reshape(-1, 1)]

        if covariates is not None:
            if len(covariates.shape) == 1:
                covariates = covariates.reshape(-1, 1)
            features.append(covariates)

        return np.hstack(features)
