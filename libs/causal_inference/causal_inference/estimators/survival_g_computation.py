"""G-computation estimator for survival analysis.

This module implements G-computation (standardization) for survival outcomes,
using parametric survival models to estimate counterfactual survival curves
and causal effects.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter, ExponentialFitter, WeibullFitter

    LIFELINES_AVAILABLE = True
except ImportError:
    CoxPHFitter = None
    ExponentialFitter = None
    WeibullFitter = None
    LIFELINES_AVAILABLE = False

from ..core.base import (
    CausalEffect,
    CovariateData,
    EstimationError,
    SurvivalOutcomeData,
    TreatmentData,
)
from .survival import SurvivalEstimator


class SurvivalGComputationEstimator(SurvivalEstimator):
    """G-computation estimator for survival analysis.

    This estimator implements the G-computation method for survival outcomes by:
    1. Fitting a parametric survival model to the observed data
    2. Using the fitted model to predict counterfactual survival curves under different treatments
    3. Averaging these predictions to estimate causal effects

    Mathematical Framework:
    ----------------------
    For survival outcome T and treatment A, the G-computation estimator:

    1. Fits survival model: S(t|A,X) = P(T > t | A, X)
    2. Estimates counterfactual survival: S(t|A=a) = E[S(t|A=a,X)]
    3. Computes causal hazard ratio: HR = λ(t|A=1) / λ(t|A=0)
    4. Estimates RMST difference: RMST(τ) = ∫₀^τ [S(t|A=1) - S(t|A=0)] dt

    Where:
    - S(t|A,X) is the conditional survival function
    - λ(t|A) is the hazard function for treatment A
    - RMST(τ) is restricted mean survival time up to horizon τ

    The method supports various parametric survival models including Cox, Weibull, and Exponential.
    """

    def __init__(
        self,
        survival_model: str = "cox",
        time_horizon: Optional[float] = None,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize G-computation survival estimator.

        Args:
            survival_model: Type of survival model ('cox', 'weibull', 'exponential')
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
            method="g_computation",
            survival_model=survival_model,
            time_horizon=time_horizon,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            random_state=random_state,
            verbose=verbose,
        )

        # Fitted survival model
        self.fitted_model: Any = None

    def _construct_cox_formula(self, df: pd.DataFrame) -> str:
        """Construct formula string for Cox proportional hazards model.

        Args:
            df: DataFrame with survival data including covariates

        Returns:
            Formula string for Cox model fitting
        """
        # Determine covariates for formula
        covariate_cols = [
            col for col in df.columns if col not in ["T", "E", "event_type"]
        ]

        if len(covariate_cols) == 1:
            # Only treatment
            return "treatment"
        else:
            # Treatment plus other covariates
            other_cols = [col for col in covariate_cols if col != "treatment"]
            return f"treatment + {' + '.join(other_cols)}"

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: SurvivalOutcomeData,
        covariates: Optional[CovariateData] = None,
    ) -> None:
        """Fit the G-computation survival model.

        Args:
            treatment: Treatment assignment data
            outcome: Survival outcome data
            covariates: Optional covariate data for adjustment
        """
        # Create combined dataset
        df = self._create_survival_data(treatment, outcome, covariates)

        # Select and fit appropriate survival model
        if self.survival_model == "cox":
            self.fitted_model = CoxPHFitter()
            formula = self._construct_cox_formula(df)
            self.fitted_model.fit(df, duration_col="T", event_col="E", formula=formula)

        elif self.survival_model == "weibull":
            self.fitted_model = WeibullFitter()

            # For parametric models, we need to fit separate models or use regression
            # Here we'll fit a simple Weibull model to the pooled data
            # In practice, you'd want to include covariates in the fitting
            self.fitted_model.fit(df["T"], df["E"])

        elif self.survival_model == "exponential":
            self.fitted_model = ExponentialFitter()
            self.fitted_model.fit(df["T"], df["E"])

        else:
            raise EstimationError(f"Unsupported survival model: {self.survival_model}")

    def _predict_survival_curve(
        self,
        treatment_value: int,
        covariates_df: Optional[pd.DataFrame] = None,
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Predict survival curve for given treatment and covariates.

        Args:
            treatment_value: Treatment value (0 or 1)
            covariates_df: Covariate values for prediction
            times: Time points for prediction

        Returns:
            DataFrame with predicted survival probabilities
        """
        if not self.is_fitted or self.fitted_model is None:
            raise EstimationError("Model must be fitted before prediction")

        # Default time points if not provided
        if times is None:
            assert self.outcome_data is not None
            times = np.linspace(0.1, self.outcome_data.median_time * 2, 100)

        if self.survival_model == "cox":
            # For Cox model, we need to create a prediction dataset
            if covariates_df is None:
                # Use mean covariate values if not provided
                df = self._create_survival_data(
                    self.treatment_data, self.outcome_data, self.covariate_data
                )

                # Create representative individual with mean covariates
                mean_covariates = df.drop(["T", "E", "treatment"], axis=1).mean()
                pred_df = pd.DataFrame([mean_covariates])
                pred_df["treatment"] = treatment_value
            else:
                pred_df = covariates_df.copy()
                pred_df["treatment"] = treatment_value

            # Predict survival function
            survival_func = self.fitted_model.predict_survival_function(pred_df)

            # Interpolate to desired time points
            survival_probs = []
            for t in times:
                if t in survival_func.index:
                    prob = survival_func.loc[t].iloc[0]
                else:
                    # Linear interpolation
                    idx = np.searchsorted(survival_func.index, t)
                    if idx == 0:
                        prob = 1.0
                    elif idx >= len(survival_func.index):
                        prob = survival_func.iloc[-1, 0]
                    else:
                        t_low, t_high = (
                            survival_func.index[idx - 1],
                            survival_func.index[idx],
                        )
                        p_low, p_high = (
                            survival_func.iloc[idx - 1, 0],
                            survival_func.iloc[idx, 0],
                        )
                        prob = p_low + (p_high - p_low) * (t - t_low) / (t_high - t_low)
                survival_probs.append(prob)

            return pd.DataFrame({"timeline": times, "survival_prob": survival_probs})

        elif self.survival_model in ["weibull", "exponential"]:
            # For parametric models, predict directly
            survival_probs = self.fitted_model.survival_function_at_times(times)

            return pd.DataFrame({"timeline": times, "survival_prob": survival_probs})

        else:
            raise EstimationError(
                f"Prediction not implemented for {self.survival_model}"
            )

    def estimate_survival_curves(self) -> dict[str, pd.DataFrame]:
        """Estimate counterfactual survival curves using G-computation.

        Returns:
            Dictionary with 'treated' and 'control' survival curves
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        if self._survival_curves is not None:
            return self._survival_curves

        # Define time points for prediction
        max_time = np.max(self.outcome_data.times)
        times = np.linspace(0.1, max_time, 100)

        # Predict survival curves for both treatment values
        treated_curve = self._predict_survival_curve(treatment_value=1, times=times)
        control_curve = self._predict_survival_curve(treatment_value=0, times=times)

        self._survival_curves = {"treated": treated_curve, "control": control_curve}

        return self._survival_curves

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate causal effects for survival outcomes.

        Returns:
            CausalEffect object with survival-specific estimates
        """
        # Get basic counts
        n_observations = len(self.treatment_data.values)
        n_treated = int(np.sum(self.treatment_data.values == 1))
        n_control = int(np.sum(self.treatment_data.values == 0))

        # Estimate hazard ratio
        hazard_ratio = self.estimate_hazard_ratio()

        # Estimate survival curves
        survival_curves = self.estimate_survival_curves()

        # Calculate median survival times
        treated_curve = survival_curves["treated"]
        control_curve = survival_curves["control"]

        # Find median survival times (where survival probability = 0.5)
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

        # Perform log-rank test
        log_rank_pvalue = self.log_rank_test()

        # For survival outcomes, ATE can be interpreted as RMST difference
        ate = rmst_difference if rmst_difference is not None else np.log(hazard_ratio)

        return CausalEffect(
            ate=ate,
            method="g_computation_survival",
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
            # Diagnostics
            diagnostics={
                "survival_model": self.survival_model,
                "time_horizon": self.time_horizon,
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

    def predict_individual_survival(
        self,
        treatment_value: int,
        covariates: pd.DataFrame,
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Predict survival curve for specific individuals.

        Args:
            treatment_value: Treatment value (0 or 1)
            covariates: Individual covariate values
            times: Time points for prediction

        Returns:
            DataFrame with predicted survival probabilities for each individual
        """
        if not self.is_fitted:
            raise EstimationError("Model must be fitted before prediction")

        if times is None:
            times = np.linspace(0.1, self.outcome_data.median_time * 2, 100)

        results = []

        for idx, row in covariates.iterrows():
            individual_df = pd.DataFrame([row])
            survival_curve = self._predict_survival_curve(
                treatment_value=treatment_value,
                covariates_df=individual_df,
                times=times,
            )
            survival_curve["individual_id"] = idx
            results.append(survival_curve)

        return pd.concat(results, ignore_index=True)

    def _create_bootstrap_estimator(
        self, random_state: Optional[int] = None
    ) -> SurvivalGComputationEstimator:
        """Create a new estimator instance for bootstrap sampling.

        Args:
            random_state: Random state for this bootstrap instance

        Returns:
            New SurvivalGComputationEstimator instance configured for bootstrap
        """
        return SurvivalGComputationEstimator(
            survival_model=self.survival_model,
            time_horizon=self.time_horizon,
            bootstrap_samples=0,  # No nested bootstrap
            confidence_level=self.confidence_level,
            random_state=random_state,
            verbose=False,  # Reduce verbosity in bootstrap
        )
