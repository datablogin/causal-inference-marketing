"""IPW estimator for survival analysis.

This module implements Inverse Probability Weighting for survival outcomes,
using propensity scores to create weighted survival analyses that estimate
causal effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from lifelines.utils import restricted_mean_survival_time

    LIFELINES_AVAILABLE = True
except ImportError:
    KaplanMeierFitter = None
    logrank_test = None
    restricted_mean_survival_time = None
    LIFELINES_AVAILABLE = False

from ..core.base import (
    CausalEffect,
    CovariateData,
    EstimationError,
    SurvivalOutcomeData,
    TreatmentData,
)
from .survival import SurvivalEstimator


class SurvivalIPWEstimator(SurvivalEstimator):
    """IPW estimator for survival analysis.

    This estimator implements Inverse Probability Weighting for survival outcomes by:
    1. Estimating propensity scores (probability of treatment given covariates)
    2. Creating inverse probability weights
    3. Using weighted Kaplan-Meier estimators to estimate survival curves
    4. Computing causal effects from weighted survival analyses

    Mathematical Framework:
    ----------------------
    For survival outcome T, treatment A, and covariates X:

    1. Propensity score: π(X) = P(A=1|X)
    2. IPW weights: w_i = A_i/π(X_i) + (1-A_i)/(1-π(X_i))
    3. Weighted survival: Ŝ(t|A=a) = ∏_{t_i ≤ t} [1 - d_a(t_i)/n_a(t_i)]^w_i
    4. Causal effects from weighted curves

    Where:
    - π(X) is the propensity score model
    - w_i are the inverse probability weights
    - d_a(t_i) is weighted number of events at time t_i in group a
    - n_a(t_i) is weighted number at risk at time t_i in group a

    Stabilized weights option: w_i^stab = P(A=a) · w_i / P(A=A_i|X_i)
    """

    def __init__(
        self,
        propensity_model: str = "logistic",
        weight_stabilization: bool = True,
        weight_trimming: float | None = 0.01,
        time_horizon: float | None = None,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize IPW survival estimator.

        Args:
            propensity_model: Type of propensity model ('logistic', 'random_forest')
            weight_stabilization: Whether to use stabilized weights
            weight_trimming: Threshold for trimming extreme weights (None = no trimming)
            time_horizon: Time horizon for RMST calculation
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        if not LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines library is required for survival analysis. "
                "Install with: pip install lifelines"
            )
        super().__init__(
            method="ipw",
            survival_model="kaplan_meier",  # IPW uses non-parametric KM
            time_horizon=time_horizon,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            random_state=random_state,
            verbose=verbose,
        )

        self.propensity_model_type = propensity_model
        self.weight_stabilization = weight_stabilization
        self.weight_trimming = weight_trimming

        # Fitted models and weights
        self.propensity_model: Any = None
        self.propensity_scores: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.stabilized_weights: np.ndarray | None = None

    def _fit_propensity_model(
        self,
        treatment: TreatmentData,
        covariates: CovariateData,
    ) -> None:
        """Fit propensity score model.

        Args:
            treatment: Treatment assignment data
            covariates: Covariate data for propensity modeling
        """
        if covariates is None:
            raise EstimationError("Covariates are required for IPW estimation")

        # Prepare covariate data
        if isinstance(covariates.values, pd.DataFrame):
            X = covariates.values.values
        else:
            X = covariates.values

        y = treatment.values

        # Fit appropriate propensity model
        if self.propensity_model_type == "logistic":
            self.propensity_model = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
        elif self.propensity_model_type == "random_forest":
            self.propensity_model = RandomForestClassifier(
                random_state=self.random_state, n_estimators=100
            )
        else:
            raise EstimationError(
                f"Unsupported propensity model: {self.propensity_model_type}"
            )

        self.propensity_model.fit(X, y)

        # Get propensity scores
        if hasattr(self.propensity_model, "predict_proba"):
            # For binary treatment, get probability of treatment=1
            self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        else:
            raise EstimationError("Propensity model must support predict_proba")

    def _compute_weights(self, treatment: TreatmentData) -> None:
        """Compute inverse probability weights.

        Args:
            treatment: Treatment assignment data
        """
        if self.propensity_scores is None:
            raise EstimationError("Propensity scores must be computed first")

        t = treatment.values
        ps = self.propensity_scores

        # Compute basic IPW weights
        weights = np.where(t == 1, 1 / ps, 1 / (1 - ps))

        # Apply weight trimming if specified
        if self.weight_trimming is not None:
            # Trim extreme propensity scores
            ps_trimmed = np.clip(ps, self.weight_trimming, 1 - self.weight_trimming)
            weights = np.where(t == 1, 1 / ps_trimmed, 1 / (1 - ps_trimmed))

        self.weights = weights

        # Compute stabilized weights if requested
        if self.weight_stabilization:
            # Stabilized weights use marginal treatment probability
            marginal_prob = np.mean(t)
            stabilized_weights = np.where(
                t == 1, marginal_prob / ps, (1 - marginal_prob) / (1 - ps)
            )

            if self.weight_trimming is not None:
                ps_trimmed = np.clip(ps, self.weight_trimming, 1 - self.weight_trimming)
                stabilized_weights = np.where(
                    t == 1,
                    marginal_prob / ps_trimmed,
                    (1 - marginal_prob) / (1 - ps_trimmed),
                )

            self.stabilized_weights = stabilized_weights

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: SurvivalOutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the IPW survival model.

        Args:
            treatment: Treatment assignment data
            outcome: Survival outcome data
            covariates: Covariate data for propensity modeling
        """
        if covariates is None:
            raise EstimationError("Covariates are required for IPW survival estimation")

        # Fit propensity model
        self._fit_propensity_model(treatment, covariates)

        # Compute weights
        self._compute_weights(treatment)

        if self.verbose:
            print(
                f"Propensity score range: [{np.min(self.propensity_scores):.3f}, {np.max(self.propensity_scores):.3f}]"
            )
            weights_to_use = (
                self.stabilized_weights if self.weight_stabilization else self.weights
            )
            print(
                f"Weight range: [{np.min(weights_to_use):.3f}, {np.max(weights_to_use):.3f}]"
            )

    def estimate_survival_curves(self) -> dict[str, pd.DataFrame]:
        """Estimate weighted survival curves using IPW.

        Returns:
            Dictionary with 'treated' and 'control' survival curves
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        if self._survival_curves is not None:
            return self._survival_curves

        # Use stabilized weights if available, otherwise use basic weights
        weights_to_use = (
            self.stabilized_weights if self.weight_stabilization else self.weights
        )

        # Assert that we have the required data
        assert self.treatment_data is not None
        assert self.outcome_data is not None

        # Create separate datasets for treated and control
        treated_mask = self.treatment_data.values == 1
        control_mask = self.treatment_data.values == 0

        # Treated group
        treated_times = self.outcome_data.times[treated_mask]
        treated_events = self.outcome_data.events[treated_mask]
        treated_weights = weights_to_use[treated_mask]

        # Control group
        control_times = self.outcome_data.times[control_mask]
        control_events = self.outcome_data.events[control_mask]
        control_weights = weights_to_use[control_mask]

        # Fit weighted Kaplan-Meier estimators
        kmf_treated = KaplanMeierFitter()
        kmf_control = KaplanMeierFitter()

        # Note: lifelines KM fitter supports weights
        kmf_treated.fit(
            treated_times,
            treated_events,
            weights=treated_weights,
            label="Treated (Weighted)",
        )

        kmf_control.fit(
            control_times,
            control_events,
            weights=control_weights,
            label="Control (Weighted)",
        )

        # Extract survival curves
        treated_curve = pd.DataFrame(
            {
                "timeline": kmf_treated.timeline,
                "survival_prob": kmf_treated.survival_function_.iloc[:, 0],
                "confidence_interval_lower": kmf_treated.confidence_interval_.iloc[
                    :, 0
                ],
                "confidence_interval_upper": kmf_treated.confidence_interval_.iloc[
                    :, 1
                ],
            }
        )

        control_curve = pd.DataFrame(
            {
                "timeline": kmf_control.timeline,
                "survival_prob": kmf_control.survival_function_.iloc[:, 0],
                "confidence_interval_lower": kmf_control.confidence_interval_.iloc[
                    :, 0
                ],
                "confidence_interval_upper": kmf_control.confidence_interval_.iloc[
                    :, 1
                ],
            }
        )

        self._survival_curves = {
            "treated": treated_curve,
            "control": control_curve,
        }

        return self._survival_curves

    def estimate_rmst_difference(self) -> dict[str, float]:
        """Estimate RMST difference using weighted survival curves.

        Returns:
            Dictionary with RMST estimates and difference
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        if self.time_horizon is None:
            raise EstimationError("time_horizon must be set for RMST calculation")

        if self._rmst_results is not None:
            return self._rmst_results

        # Get weighted survival curves
        curves = self.estimate_survival_curves()

        # Calculate RMST for each group using weighted curves
        treated_curve = curves["treated"]
        control_curve = curves["control"]

        # Create proper DataFrames for RMST calculation
        treated_sf = pd.DataFrame(
            treated_curve["survival_prob"].values,
            index=treated_curve["timeline"].values,
            columns=["survival_prob"],
        )
        control_sf = pd.DataFrame(
            control_curve["survival_prob"].values,
            index=control_curve["timeline"].values,
            columns=["survival_prob"],
        )

        rmst_treated = restricted_mean_survival_time(treated_sf, self.time_horizon)
        rmst_control = restricted_mean_survival_time(control_sf, self.time_horizon)

        rmst_difference = rmst_treated - rmst_control

        self._rmst_results = {
            "rmst_treated": rmst_treated,
            "rmst_control": rmst_control,
            "rmst_difference": rmst_difference,
        }

        return self._rmst_results

    def weighted_log_rank_test(self) -> float:
        """Perform weighted log-rank test.

        Returns:
            Weighted log-rank test p-value
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        # Use stabilized weights if available
        weights_to_use = (
            self.stabilized_weights if self.weight_stabilization else self.weights
        )

        # Assert that we have the required data
        assert self.treatment_data is not None
        assert self.outcome_data is not None

        # Create separate datasets
        treated_mask = self.treatment_data.values == 1
        control_mask = self.treatment_data.values == 0

        # Perform weighted log-rank test using the standard logrank_test with weights
        results = logrank_test(
            self.outcome_data.times[treated_mask],
            self.outcome_data.times[control_mask],
            event_observed_A=self.outcome_data.events[treated_mask],
            event_observed_B=self.outcome_data.events[control_mask],
            weights_A=weights_to_use[treated_mask],
            weights_B=weights_to_use[control_mask],
        )

        return float(results.p_value)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate causal effects using IPW for survival outcomes.

        Returns:
            CausalEffect object with IPW survival-specific estimates
        """
        # Assert that we have the required data
        assert self.treatment_data is not None
        assert self.outcome_data is not None

        # Get basic counts
        n_observations = len(self.treatment_data.values)
        n_treated = int(np.sum(self.treatment_data.values == 1))
        n_control = int(np.sum(self.treatment_data.values == 0))

        # Estimate weighted survival curves
        survival_curves = self.estimate_survival_curves()

        # Calculate median survival times from weighted curves
        treated_curve = survival_curves["treated"]
        control_curve = survival_curves["control"]

        median_treated = None
        median_control = None

        treated_below_half = treated_curve[treated_curve["survival_prob"] <= 0.5]
        if len(treated_below_half) > 0:
            median_treated = float(treated_below_half["timeline"].iloc[0])

        control_below_half = control_curve[control_curve["survival_prob"] <= 0.5]
        if len(control_below_half) > 0:
            median_control = float(control_below_half["timeline"].iloc[0])

        # Calculate RMST if time_horizon is set
        rmst_treated = None
        rmst_control = None
        rmst_difference = None

        if self.time_horizon is not None:
            rmst_results = self.estimate_rmst_difference()
            rmst_treated = rmst_results["rmst_treated"]
            rmst_control = rmst_results["rmst_control"]
            rmst_difference = rmst_results["rmst_difference"]

        # Perform weighted log-rank test
        weighted_log_rank_pvalue = self.weighted_log_rank_test()

        # Estimate hazard ratio from weighted data (approximation)
        # This is a simplified approach - could be improved with more sophisticated methods
        hazard_ratio = None
        if median_treated is not None and median_control is not None:
            # Approximate HR from median ratio (rough approximation)
            hazard_ratio = median_control / median_treated

        # For survival outcomes, ATE can be interpreted as RMST difference
        ate = rmst_difference if rmst_difference is not None else 0.0

        # Weight diagnostics
        weights_to_use = (
            self.stabilized_weights if self.weight_stabilization else self.weights
        )

        return CausalEffect(
            ate=ate,
            method="ipw_survival",
            n_observations=n_observations,
            n_treated=n_treated,
            n_control=n_control,
            confidence_level=self.confidence_level,
            # Survival-specific estimates
            hazard_ratio=hazard_ratio,
            rmst_treated=rmst_treated,
            rmst_control=rmst_control,
            rmst_difference=rmst_difference,
            median_survival_treated=median_treated,
            median_survival_control=median_control,
            log_rank_test_pvalue=weighted_log_rank_pvalue,
            survival_curves={
                "treated": treated_curve.to_dict("records"),
                "control": control_curve.to_dict("records"),
            },
            # Diagnostics
            diagnostics={
                "propensity_model": self.propensity_model_type,
                "weight_stabilization": self.weight_stabilization,
                "weight_trimming": self.weight_trimming,
                "time_horizon": self.time_horizon,
                "propensity_score_range": [
                    float(np.min(self.propensity_scores)),
                    float(np.max(self.propensity_scores)),
                ],
                "weight_range": [
                    float(np.min(weights_to_use)),
                    float(np.max(weights_to_use)),
                ],
                "effective_sample_size_treated": float(
                    np.sum(weights_to_use[self.treatment_data.values == 1])
                ),
                "effective_sample_size_control": float(
                    np.sum(weights_to_use[self.treatment_data.values == 0])
                ),
                "events_treated": int(
                    np.sum(
                        (self.treatment_data.values == 1)
                        & (self.outcome_data.events == 1)
                    )
                ),
                "events_control": int(
                    np.sum(
                        (self.treatment_data.values == 0)
                        & (self.outcome_data.events == 1)
                    )
                ),
                "censoring_rate": self.outcome_data.censoring_rate,
            },
        )

    def get_weight_diagnostics(self) -> dict[str, Any]:
        """Get diagnostics for the computed weights.

        Returns:
            Dictionary with weight diagnostic information
        """
        if not self.is_fitted or self.weights is None:
            raise EstimationError("Model must be fitted to get weight diagnostics")

        weights_to_use = (
            self.stabilized_weights if self.weight_stabilization else self.weights
        )

        return {
            "propensity_scores": {
                "min": float(np.min(self.propensity_scores)),
                "max": float(np.max(self.propensity_scores)),
                "mean": float(np.mean(self.propensity_scores)),
                "std": float(np.std(self.propensity_scores)),
            },
            "weights": {
                "min": float(np.min(weights_to_use)),
                "max": float(np.max(weights_to_use)),
                "mean": float(np.mean(weights_to_use)),
                "std": float(np.std(weights_to_use)),
            },
            "effective_sample_sizes": {
                "treated": float(
                    np.sum(weights_to_use[self.treatment_data.values == 1])
                ),
                "control": float(
                    np.sum(weights_to_use[self.treatment_data.values == 0])
                ),
                "total": float(np.sum(weights_to_use)),
            },
            "balance_checks": {
                "weight_stabilization": self.weight_stabilization,
                "weight_trimming": self.weight_trimming,
            },
        }

    def _create_bootstrap_estimator(
        self, random_state: int | None = None
    ) -> SurvivalIPWEstimator:
        """Create a new estimator instance for bootstrap sampling.

        Args:
            random_state: Random state for this bootstrap instance

        Returns:
            New SurvivalIPWEstimator instance configured for bootstrap
        """
        return SurvivalIPWEstimator(
            propensity_model=self.propensity_model_type,
            weight_stabilization=self.weight_stabilization,
            weight_trimming=self.weight_trimming,
            time_horizon=self.time_horizon,
            bootstrap_samples=0,  # No nested bootstrap
            confidence_level=self.confidence_level,
            random_state=random_state,
            verbose=False,  # Reduce verbosity in bootstrap
        )
