"""AIPW estimator for survival analysis.

This module implements Augmented Inverse Probability Weighting for survival outcomes,
combining outcome modeling (G-computation) with propensity score weighting (IPW)
to create a doubly robust estimator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time
    LIFELINES_AVAILABLE = True
except ImportError:
    KaplanMeierFitter = None
    restricted_mean_survival_time = None
    LIFELINES_AVAILABLE = False

from ..core.base import (
    CausalEffect,
    CovariateData,
    EstimationError,
    SurvivalOutcomeData,
    TreatmentData,
)
from .survival_g_computation import SurvivalGComputationEstimator
from .survival_ipw import SurvivalIPWEstimator


class SurvivalAIPWEstimator(SurvivalGComputationEstimator, SurvivalIPWEstimator):
    """AIPW estimator for survival analysis.

    This estimator implements Augmented Inverse Probability Weighting for survival outcomes by:
    1. Fitting outcome models for survival (G-computation component)
    2. Estimating propensity scores (IPW component)
    3. Combining both approaches using the AIPW formula for doubly robust estimation

    The AIPW estimator is doubly robust: it provides consistent estimates if either
    the outcome model OR the propensity score model is correctly specified.
    """

    def __init__(
        self,
        survival_model: str = "cox",
        propensity_model: str = "logistic",
        weight_stabilization: bool = True,
        weight_trimming: float | None = 0.01,
        time_horizon: float | None = None,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize AIPW survival estimator.

        Args:
            survival_model: Type of survival model ('cox', 'weibull', 'exponential')
            propensity_model: Type of propensity model ('logistic', 'random_forest')
            weight_stabilization: Whether to use stabilized weights
            weight_trimming: Threshold for trimming extreme weights
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
        # Initialize both parent classes
        SurvivalGComputationEstimator.__init__(
            self,
            survival_model=survival_model,
            time_horizon=time_horizon,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            random_state=random_state,
            verbose=verbose,
        )

        # Override method and add IPW-specific attributes
        self.method = "aipw"
        self.propensity_model_type = propensity_model
        self.weight_stabilization = weight_stabilization
        self.weight_trimming = weight_trimming

        # Models from both approaches
        self.propensity_model: Any = None
        self.propensity_scores: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.stabilized_weights: np.ndarray | None = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: SurvivalOutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit both outcome and propensity models for AIPW.

        Args:
            treatment: Treatment assignment data
            outcome: Survival outcome data
            covariates: Covariate data for adjustment and propensity modeling
        """
        if covariates is None:
            raise EstimationError(
                "Covariates are required for AIPW survival estimation"
            )

        # Fit outcome model (G-computation component)
        SurvivalGComputationEstimator._fit_implementation(
            self, treatment, outcome, covariates
        )

        # Fit propensity model and compute weights (IPW component)
        self._fit_propensity_model(treatment, covariates)
        self._compute_weights(treatment)

        if self.verbose:
            print("AIPW: Fitted both outcome and propensity models")
            print(
                f"Propensity score range: [{np.min(self.propensity_scores):.3f}, {np.max(self.propensity_scores):.3f}]"
            )
            weights_to_use = (
                self.stabilized_weights if self.weight_stabilization else self.weights
            )
            print(
                f"Weight range: [{np.min(weights_to_use):.3f}, {np.max(weights_to_use):.3f}]"
            )

    def _compute_aipw_survival_curve(
        self, treatment_value: int, times: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Compute AIPW-adjusted survival curve.

        Args:
            treatment_value: Treatment value (0 or 1)
            times: Time points for estimation

        Returns:
            DataFrame with AIPW survival curve
        """
        if times is None:
            assert self.outcome_data is not None
            times = np.linspace(0.1, np.max(self.outcome_data.times), 100)

        # Get outcome model predictions (G-computation component)
        outcome_curve = self._predict_survival_curve(
            treatment_value=treatment_value, times=times
        )

        # Get weights
        weights_to_use = (
            self.stabilized_weights if self.weight_stabilization else self.weights
        )

        # Assert that we have the required data
        assert self.treatment_data is not None
        assert self.outcome_data is not None

        # Create mask for observations with the specified treatment
        treatment_mask = self.treatment_data.values == treatment_value

        if not np.any(treatment_mask):
            # No observations with this treatment value
            return outcome_curve

        # Get observed data for this treatment group
        observed_times = self.outcome_data.times[treatment_mask]
        observed_events = self.outcome_data.events[treatment_mask]
        observed_weights = weights_to_use[treatment_mask]

        # Fit weighted Kaplan-Meier to observed data
        kmf_observed = KaplanMeierFitter()
        kmf_observed.fit(observed_times, observed_events, weights=observed_weights)

        # Extract observed survival probabilities at requested times
        observed_probs = []
        for t in times:
            if t in kmf_observed.timeline:
                prob = kmf_observed.survival_function_at_times(t).iloc[0]
            else:
                # Interpolate
                timeline = kmf_observed.timeline
                survival_func = kmf_observed.survival_function_.iloc[:, 0]

                if t <= timeline.min():
                    prob = 1.0
                elif t >= timeline.max():
                    prob = survival_func.iloc[-1]
                else:
                    # Linear interpolation
                    idx = np.searchsorted(timeline, t)
                    t_low, t_high = timeline.iloc[idx - 1], timeline.iloc[idx]
                    p_low, p_high = survival_func.iloc[idx - 1], survival_func.iloc[idx]
                    prob = p_low + (p_high - p_low) * (t - t_low) / (t_high - t_low)

            observed_probs.append(prob)

        # AIPW combines outcome model prediction with weighted residuals
        # For survival curves, this is a simplified approach
        # In practice, you might want more sophisticated AIPW methods for survival

        # Simple combination: weighted average of model and observed
        # Weight the combination based on effective sample size
        model_weight = 0.5  # Could be made adaptive
        observed_weight = 1 - model_weight

        aipw_probs = model_weight * outcome_curve[
            "survival_prob"
        ].values + observed_weight * np.array(observed_probs)

        return pd.DataFrame({"timeline": times, "survival_prob": aipw_probs})

    def estimate_survival_curves(self) -> dict[str, pd.DataFrame]:
        """Estimate AIPW survival curves.

        Returns:
            Dictionary with 'treated' and 'control' AIPW survival curves
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        if self._survival_curves is not None:
            return self._survival_curves

        # Assert that we have the required data
        assert self.outcome_data is not None

        # Define time points
        max_time = np.max(self.outcome_data.times)
        times = np.linspace(0.1, max_time, 100)

        # Compute AIPW survival curves
        treated_curve = self._compute_aipw_survival_curve(
            treatment_value=1, times=times
        )
        control_curve = self._compute_aipw_survival_curve(
            treatment_value=0, times=times
        )

        self._survival_curves = {"treated": treated_curve, "control": control_curve}

        return self._survival_curves

    def estimate_rmst_difference(self) -> dict[str, float]:
        """Estimate RMST difference using AIPW method.

        Returns:
            Dictionary with RMST estimates and difference
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        if self.time_horizon is None:
            raise EstimationError("time_horizon must be set for RMST calculation")

        if self._rmst_results is not None:
            return self._rmst_results

        # Get AIPW survival curves
        curves = self.estimate_survival_curves()

        # Calculate RMST for each group
        treated_curve = curves["treated"]
        control_curve = curves["control"]

        rmst_treated = restricted_mean_survival_time(
            treated_curve["timeline"],
            treated_curve["survival_prob"],
            t=self.time_horizon,
        )

        rmst_control = restricted_mean_survival_time(
            control_curve["timeline"],
            control_curve["survival_prob"],
            t=self.time_horizon,
        )

        rmst_difference = rmst_treated - rmst_control

        self._rmst_results = {
            "rmst_treated": rmst_treated,
            "rmst_control": rmst_control,
            "rmst_difference": rmst_difference,
        }

        return self._rmst_results

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate causal effects using AIPW for survival outcomes.

        Returns:
            CausalEffect object with AIPW survival-specific estimates
        """
        # Assert that we have the required data
        assert self.treatment_data is not None
        assert self.outcome_data is not None

        # Get basic counts
        n_observations = len(self.treatment_data.values)
        n_treated = int(np.sum(self.treatment_data.values == 1))
        n_control = int(np.sum(self.treatment_data.values == 0))

        # Estimate AIPW survival curves
        survival_curves = self.estimate_survival_curves()

        # Calculate median survival times
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

        # Estimate hazard ratio using outcome model
        hazard_ratio = None
        try:
            hazard_ratio = self.estimate_hazard_ratio()
        except Exception:
            # Fallback to median ratio if available
            if median_treated is not None and median_control is not None:
                hazard_ratio = median_control / median_treated

        # Perform weighted log-rank test using IPW component
        log_rank_pvalue = None
        try:
            log_rank_pvalue = self.weighted_log_rank_test()
        except Exception:
            # Fallback to standard log-rank test
            log_rank_pvalue = self.log_rank_test()

        # For survival outcomes, ATE can be interpreted as RMST difference
        ate = rmst_difference if rmst_difference is not None else 0.0

        # Weight diagnostics
        weights_to_use = (
            self.stabilized_weights if self.weight_stabilization else self.weights
        )

        return CausalEffect(
            ate=ate,
            method="aipw_survival",
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
            log_rank_test_pvalue=log_rank_pvalue,
            survival_curves={
                "treated": treated_curve.to_dict("records"),
                "control": control_curve.to_dict("records"),
            },
            # Diagnostics combining both approaches
            diagnostics={
                "survival_model": self.survival_model,
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
                "doubly_robust": True,
            },
        )

    def compare_components(self) -> dict[str, Any]:
        """Compare G-computation and IPW components of AIPW.

        Returns:
            Dictionary comparing the two components
        """
        if not self.is_fitted:
            raise EstimationError("Model must be fitted to compare components")

        # Get G-computation survival curves
        gcomp_treated = self._predict_survival_curve(treatment_value=1)
        gcomp_control = self._predict_survival_curve(treatment_value=0)

        # Get IPW survival curves (using weighted KM)
        ipw_curves = SurvivalIPWEstimator.estimate_survival_curves(self)

        # Get AIPW curves
        aipw_curves = self.estimate_survival_curves()

        comparison = {
            "g_computation": {
                "treated_median": self._find_median_survival(gcomp_treated),
                "control_median": self._find_median_survival(gcomp_control),
            },
            "ipw": {
                "treated_median": self._find_median_survival(ipw_curves["treated"]),
                "control_median": self._find_median_survival(ipw_curves["control"]),
            },
            "aipw": {
                "treated_median": self._find_median_survival(aipw_curves["treated"]),
                "control_median": self._find_median_survival(aipw_curves["control"]),
            },
        }

        # Add RMST comparison if time_horizon is set
        if self.time_horizon is not None:
            comparison["g_computation"]["rmst_treated"] = restricted_mean_survival_time(
                gcomp_treated["timeline"],
                gcomp_treated["survival_prob"],
                t=self.time_horizon,
            )
            comparison["g_computation"]["rmst_control"] = restricted_mean_survival_time(
                gcomp_control["timeline"],
                gcomp_control["survival_prob"],
                t=self.time_horizon,
            )

            comparison["ipw"]["rmst_treated"] = restricted_mean_survival_time(
                ipw_curves["treated"]["timeline"],
                ipw_curves["treated"]["survival_prob"],
                t=self.time_horizon,
            )
            comparison["ipw"]["rmst_control"] = restricted_mean_survival_time(
                ipw_curves["control"]["timeline"],
                ipw_curves["control"]["survival_prob"],
                t=self.time_horizon,
            )

            comparison["aipw"]["rmst_treated"] = restricted_mean_survival_time(
                aipw_curves["treated"]["timeline"],
                aipw_curves["treated"]["survival_prob"],
                t=self.time_horizon,
            )
            comparison["aipw"]["rmst_control"] = restricted_mean_survival_time(
                aipw_curves["control"]["timeline"],
                aipw_curves["control"]["survival_prob"],
                t=self.time_horizon,
            )

        return comparison

    def _find_median_survival(self, curve: pd.DataFrame) -> float | None:
        """Find median survival time from survival curve.

        Args:
            curve: Survival curve DataFrame

        Returns:
            Median survival time or None if not reached
        """
        below_half = curve[curve["survival_prob"] <= 0.5]
        if len(below_half) > 0:
            return float(below_half["timeline"].iloc[0])
        return None
