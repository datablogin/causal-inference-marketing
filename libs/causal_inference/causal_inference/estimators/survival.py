"""Survival analysis estimators for causal inference.

This module implements causal inference methods for time-to-event (survival) outcomes,
extending standard methods to handle censored data and hazard-based causal effects.
"""

from __future__ import annotations

import abc
import warnings
from typing import Any, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter
    from lifelines.statistics import logrank_test
    from lifelines.utils import restricted_mean_survival_time
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    warnings.warn(
        "lifelines library not available. Install with 'pip install lifelines' "
        "to use survival analysis estimators."
    )

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
    from sksurv.preprocessing import OneHotEncoder
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    warnings.warn(
        "scikit-survival library not available. Install with 'pip install scikit-survival' "
        "to use ML-based survival analysis estimators."
    )

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    SurvivalOutcomeData,
    TreatmentData,
)
from ..core.bootstrap import BootstrapMixin


class SurvivalEstimator(BootstrapMixin, BaseEstimator):
    """Base class for causal survival analysis estimators.
    
    Provides common functionality for survival analysis methods including
    hazard ratio estimation, survival curve estimation, and RMST calculation.
    """
    
    def __init__(
        self,
        method: str = "cox",
        survival_model: str = "cox",
        time_horizon: float | None = None,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize survival estimator.
        
        Args:
            method: Causal inference method ('g_computation', 'ipw', 'aipw')
            survival_model: Underlying survival model ('cox', 'weibull', 'exponential', 'kaplan_meier')
            time_horizon: Time horizon for RMST calculation
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        super().__init__(
            random_state=random_state,
            verbose=verbose,
        )
        
        self.method = method
        self.survival_model = survival_model
        self.time_horizon = time_horizon
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        
        # Fitted models
        self.treated_model: Any = None
        self.control_model: Any = None
        self.pooled_model: Any = None
        self.propensity_model: Any = None
        
        # Cached results
        self._survival_curves: dict[str, Any] | None = None
        self._hazard_ratio: float | None = None
        self._rmst_results: dict[str, float] | None = None
        
        # Validate availability of required libraries
        if not LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines library is required for survival analysis. "
                "Install with: pip install lifelines"
            )

    def _validate_survival_inputs(
        self,
        treatment: TreatmentData,
        outcome: SurvivalOutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Validate inputs specific to survival analysis.
        
        Args:
            treatment: Treatment assignment data
            outcome: Survival outcome data
            covariates: Optional covariate data
            
        Raises:
            EstimationError: If survival-specific validation fails
        """
        # Call parent validation
        super()._validate_inputs(treatment, outcome, covariates)
        
        # Check that we have SurvivalOutcomeData
        if not isinstance(outcome, SurvivalOutcomeData):
            raise EstimationError(
                "SurvivalEstimator requires SurvivalOutcomeData for outcome"
            )
        
        # Check for sufficient events
        if outcome.n_events < 10:
            raise EstimationError(
                f"Insufficient events for survival analysis. "
                f"Got {outcome.n_events} events, need at least 10."
            )
        
        # Check censoring rate
        if outcome.censoring_rate > 0.9:
            warnings.warn(
                f"High censoring rate ({outcome.censoring_rate:.1%}). "
                "Results may be unreliable."
            )
            
        # Check treatment-specific events
        if treatment.treatment_type == "binary":
            treated_events = np.sum(
                (treatment.values == 1) & (outcome.events == 1)
            )
            control_events = np.sum(
                (treatment.values == 0) & (outcome.events == 1)
            )
            
            if treated_events < 5 or control_events < 5:
                raise EstimationError(
                    f"Insufficient events per treatment group. "
                    f"Treated: {treated_events}, Control: {control_events}. "
                    f"Need at least 5 events per group."
                )

    def _create_survival_data(
        self,
        treatment: TreatmentData,
        outcome: SurvivalOutcomeData,
        covariates: CovariateData | None = None,
        treatment_value: int | None = None,
    ) -> pd.DataFrame:
        """Create DataFrame for survival analysis.
        
        Args:
            treatment: Treatment assignment data
            outcome: Survival outcome data
            covariates: Optional covariate data
            treatment_value: If specified, filter to this treatment value
            
        Returns:
            DataFrame with survival data in lifelines format
        """
        # Start with basic survival data
        df = outcome.to_lifelines_format()
        df['treatment'] = treatment.values
        
        # Add covariates if provided
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                for col in covariates.values.columns:
                    df[col] = covariates.values[col].values
            else:
                # Handle numpy array
                covariate_names = covariates.names or [f"X{i}" for i in range(covariates.values.shape[1])]
                for i, name in enumerate(covariate_names):
                    df[name] = covariates.values[:, i]
        
        # Filter to specific treatment if requested
        if treatment_value is not None:
            df = df[df['treatment'] == treatment_value].copy()
            
        return df

    def estimate_survival_curves(self) -> dict[str, pd.DataFrame]:
        """Estimate survival curves for treated and control groups.
        
        Returns:
            Dictionary with 'treated' and 'control' survival curves
            
        Raises:
            EstimationError: If estimator is not fitted
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")
            
        if self._survival_curves is not None:
            return self._survival_curves
            
        # Create data for each group
        treated_data = self._create_survival_data(
            self.treatment_data,
            self.outcome_data,
            self.covariate_data,
            treatment_value=1
        )
        
        control_data = self._create_survival_data(
            self.treatment_data,
            self.outcome_data,
            self.covariate_data,
            treatment_value=0
        )
        
        # Fit Kaplan-Meier estimators
        kmf_treated = KaplanMeierFitter()
        kmf_control = KaplanMeierFitter()
        
        kmf_treated.fit(treated_data['T'], treated_data['E'], label='Treated')
        kmf_control.fit(control_data['T'], control_data['E'], label='Control')
        
        # Extract survival curves
        treated_curve = pd.DataFrame({
            'timeline': kmf_treated.timeline,
            'survival_prob': kmf_treated.survival_function_.iloc[:, 0],
            'confidence_interval_lower': kmf_treated.confidence_interval_.iloc[:, 0],
            'confidence_interval_upper': kmf_treated.confidence_interval_.iloc[:, 1],
        })
        
        control_curve = pd.DataFrame({
            'timeline': kmf_control.timeline,
            'survival_prob': kmf_control.survival_function_.iloc[:, 0],
            'confidence_interval_lower': kmf_control.confidence_interval_.iloc[:, 0],
            'confidence_interval_upper': kmf_control.confidence_interval_.iloc[:, 1],
        })
        
        self._survival_curves = {
            'treated': treated_curve,
            'control': control_curve,
        }
        
        return self._survival_curves

    def estimate_hazard_ratio(self) -> float:
        """Estimate hazard ratio using Cox proportional hazards model.
        
        Returns:
            Estimated hazard ratio
            
        Raises:
            EstimationError: If estimator is not fitted
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")
            
        if self._hazard_ratio is not None:
            return self._hazard_ratio
            
        # Create pooled data
        df = self._create_survival_data(
            self.treatment_data,
            self.outcome_data,
            self.covariate_data
        )
        
        # Fit Cox model
        cph = CoxPHFitter()
        
        # Determine columns for fitting
        covariate_cols = [col for col in df.columns if col not in ['T', 'E', 'event_type']]
        
        cph.fit(df, duration_col='T', event_col='E', formula=f"treatment + {' + '.join([c for c in covariate_cols if c != 'treatment'])}" if len(covariate_cols) > 1 else "treatment")
        
        # Extract hazard ratio for treatment
        self._hazard_ratio = float(np.exp(cph.params_['treatment']))
        
        return self._hazard_ratio

    def estimate_rmst_difference(self) -> dict[str, float]:
        """Estimate restricted mean survival time difference.
        
        Returns:
            Dictionary with RMST estimates and difference
            
        Raises:
            EstimationError: If estimator is not fitted or no time_horizon set
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")
            
        if self.time_horizon is None:
            raise EstimationError("time_horizon must be set for RMST calculation")
            
        if self._rmst_results is not None:
            return self._rmst_results
            
        # Get survival curves
        curves = self.estimate_survival_curves()
        
        # Calculate RMST for each group
        treated_curve = curves['treated']
        control_curve = curves['control']
        
        # Use lifelines RMST function
        rmst_treated = restricted_mean_survival_time(
            treated_curve['timeline'],
            treated_curve['survival_prob'],
            t=self.time_horizon
        )
        
        rmst_control = restricted_mean_survival_time(
            control_curve['timeline'],
            control_curve['survival_prob'],
            t=self.time_horizon
        )
        
        rmst_difference = rmst_treated - rmst_control
        
        self._rmst_results = {
            'rmst_treated': rmst_treated,
            'rmst_control': rmst_control,
            'rmst_difference': rmst_difference,
        }
        
        return self._rmst_results

    def log_rank_test(self) -> float:
        """Perform log-rank test for survival difference.
        
        Returns:
            Log-rank test p-value
            
        Raises:
            EstimationError: If estimator is not fitted
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")
            
        # Create data for each group
        treated_data = self._create_survival_data(
            self.treatment_data,
            self.outcome_data,
            self.covariate_data,
            treatment_value=1
        )
        
        control_data = self._create_survival_data(
            self.treatment_data,
            self.outcome_data,
            self.covariate_data,
            treatment_value=0
        )
        
        # Perform log-rank test
        results = logrank_test(
            treated_data['T'],
            control_data['T'],
            treated_data['E'],
            control_data['E']
        )
        
        return float(results.p_value)

    @abc.abstractmethod
    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: SurvivalOutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Implement the specific fitting logic for this survival estimator.
        
        Args:
            treatment: Treatment assignment data
            outcome: Survival outcome data
            covariates: Optional covariate data for adjustment
        """
        pass

    @abc.abstractmethod
    def _estimate_ate_implementation(self) -> CausalEffect:
        """Implement the specific ATE estimation logic for this survival estimator.
        
        For survival outcomes, this typically estimates hazard ratios and RMST differences.
        
        Returns:
            CausalEffect object with survival-specific estimates
        """
        pass

    def fit(
        self,
        treatment: TreatmentData,
        outcome: Union[SurvivalOutcomeData, Any],
        covariates: CovariateData | None = None,
    ) -> SurvivalEstimator:
        """Fit the survival estimator to data.
        
        Args:
            treatment: Treatment assignment data
            outcome: Survival outcome data
            covariates: Optional covariate data for adjustment
            
        Returns:
            self: The fitted estimator instance
            
        Raises:
            EstimationError: If fitting fails
        """
        # Validate survival-specific inputs
        self._validate_survival_inputs(treatment, outcome, covariates)
        
        # Store data
        self.treatment_data = treatment
        self.outcome_data = outcome
        self.covariate_data = covariates
        
        # Clear cached results
        self._causal_effect = None
        self._survival_curves = None
        self._hazard_ratio = None
        self._rmst_results = None
        
        try:
            # Call the implementation-specific fitting logic
            self._fit_implementation(treatment, outcome, covariates)
            self.is_fitted = True
            
            if self.verbose:
                print(f"Successfully fitted {self.__class__.__name__}")
                print(f"Observations: {len(treatment.values)}")
                print(f"Events: {outcome.n_events}")
                print(f"Censoring rate: {outcome.censoring_rate:.1%}")
                
        except Exception as e:
            raise EstimationError(f"Failed to fit survival estimator: {str(e)}") from e
            
        return self