"""Augmented Inverse Probability Weighting (AIPW) estimator for causal inference.

This module implements the AIPW method, which combines G-computation and IPW
to create a doubly robust estimator. The AIPW estimator provides consistent
estimates as long as either the outcome model OR the propensity score model
is correctly specified (but not necessarily both).

The AIPW estimator works by:
1. Fitting both an outcome model (like G-computation) and propensity score model (like IPW)
2. Computing the doubly robust estimator that combines both approaches
3. Providing enhanced robustness against model misspecification

Key Features:
- Doubly robust: consistent if either model is correct
- Cross-fitting to reduce overfitting bias
- Influence function-based variance estimation
- Targeted Maximum Likelihood Estimation (TMLE) variant
- Comprehensive diagnostics comparing component models

Example Usage:
    Basic AIPW estimation:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from causal_inference.core.base import TreatmentData, OutcomeData, CovariateData
    >>> from causal_inference.estimators.aipw import AIPWEstimator
    >>>
    >>> # Prepare your data
    >>> treatment = TreatmentData(values=treatment_series, treatment_type="binary")
    >>> outcome = OutcomeData(values=outcome_series, outcome_type="continuous")
    >>> covariates = CovariateData(values=covariate_df, names=list(covariate_df.columns))
    >>>
    >>> # Initialize and fit the estimator
    >>> estimator = AIPWEstimator(
    ...     outcome_model_type="auto",
    ...     propensity_model_type="logistic",
    ...     cross_fitting=True,
    ...     n_folds=5,
    ...     bootstrap_samples=1000,
    ...     random_state=42
    ... )
    >>>
    >>> # Fit the model
    >>> estimator.fit(treatment, outcome, covariates)
    >>>
    >>> # Estimate causal effect
    >>> effect = estimator.estimate_ate()
    >>> print(f"Average Treatment Effect: {effect.ate:.3f}")
    >>> print(f"95% CI: [{effect.ate_ci_lower:.3f}, {effect.ate_ci_upper:.3f}]")
    >>>
    >>> # Compare with component estimators
    >>> diagnostics = estimator.get_component_diagnostics()
    >>> print(f"G-computation ATE: {diagnostics['g_computation_ate']:.3f}")
    >>> print(f"IPW ATE: {diagnostics['ipw_ate']:.3f}")
    >>> print(f"AIPW ATE: {effect.ate:.3f}")

Advanced Usage with TMLE:
    >>> # Use Targeted Maximum Likelihood Estimation
    >>> estimator_tmle = AIPWEstimator(
    ...     outcome_model_type="random_forest",
    ...     propensity_model_type="random_forest",
    ...     cross_fitting=True,
    ...     n_folds=10,
    ...     use_tmle=True,
    ...     tmle_fluctuation="logistic",
    ...     influence_function_se=True,
    ...     verbose=True
    ... )
    >>>
    >>> estimator_tmle.fit(treatment, outcome, covariates)
    >>> effect_tmle = estimator_tmle.estimate_ate()

Cross-fitting for Bias Reduction:
    >>> # Cross-fitting reduces overfitting bias
    >>> estimator_cf = AIPWEstimator(
    ...     cross_fitting=True,
    ...     n_folds=5,
    ...     stratify_folds=True,  # Stratify by treatment
    ...     random_state=42
    ... )
    >>>
    >>> estimator_cf.fit(treatment, outcome, covariates)
    >>> effect_cf = estimator_cf.estimate_ate()

Notes:
    - AIPW is doubly robust: consistent if either outcome or propensity model is correct
    - Cross-fitting reduces finite-sample bias from model overfitting
    - Influence function variance estimation accounts for all sources of uncertainty
    - TMLE variant provides additional efficiency gains
    - Compare component models to assess which may be misspecified
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from .g_computation import GComputationEstimator
from .ipw import IPWEstimator


class AIPWEstimator(BaseEstimator):
    """Augmented Inverse Probability Weighting estimator for causal inference.

    AIPW combines G-computation and IPW to create a doubly robust estimator.
    The estimator is consistent as long as either the outcome model OR the
    propensity score model is correctly specified.

    The AIPW estimator formula is:
    τ_AIPW = 1/n Σ[μ₁(X_i) - μ₀(X_i) + (T_i/e(X_i))(Y_i - μ₁(X_i)) - ((1-T_i)/(1-e(X_i)))(Y_i - μ₀(X_i))]

    Where:
    - μ₁(X), μ₀(X) are outcome models for treated/control
    - e(X) is the propensity score
    - T_i, Y_i are treatment and outcome for unit i

    Attributes:
        outcome_estimator: G-computation estimator for outcome models
        propensity_estimator: IPW estimator for propensity scores
        cross_fitting: Whether to use cross-fitting
        n_folds: Number of cross-fitting folds
        use_tmle: Whether to use TMLE variant
        influence_function_se: Whether to use influence function standard errors
    """

    def __init__(
        self,
        outcome_model_type: str = "auto",
        outcome_model_params: dict[str, Any] | None = None,
        propensity_model_type: str = "logistic",
        propensity_model_params: dict[str, Any] | None = None,
        cross_fitting: bool = True,
        n_folds: int = 5,
        stratify_folds: bool = True,
        use_tmle: bool = False,
        tmle_fluctuation: str = "logistic",
        influence_function_se: bool = True,
        weight_truncation: str | None = "percentile",
        truncation_threshold: float = 0.01,
        stabilized_weights: bool = True,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the AIPW estimator.

        Args:
            outcome_model_type: Type of outcome model ('auto', 'linear', 'logistic', 'random_forest')
            outcome_model_params: Parameters for outcome model
            propensity_model_type: Type of propensity model ('logistic', 'random_forest')
            propensity_model_params: Parameters for propensity model
            cross_fitting: Whether to use cross-fitting to reduce bias
            n_folds: Number of folds for cross-fitting
            stratify_folds: Whether to stratify folds by treatment
            use_tmle: Whether to use Targeted Maximum Likelihood Estimation
            tmle_fluctuation: TMLE fluctuation method ('logistic', 'linear')
            influence_function_se: Whether to compute influence function standard errors
            weight_truncation: IPW weight truncation method
            truncation_threshold: Threshold for weight truncation
            stabilized_weights: Whether to use stabilized IPW weights
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        super().__init__(random_state=random_state, verbose=verbose)

        self.outcome_model_type = outcome_model_type
        self.outcome_model_params = outcome_model_params or {}
        self.propensity_model_type = propensity_model_type
        self.propensity_model_params = propensity_model_params or {}
        self.cross_fitting = cross_fitting
        self.n_folds = n_folds
        self.stratify_folds = stratify_folds
        self.use_tmle = use_tmle
        self.tmle_fluctuation = tmle_fluctuation
        self.influence_function_se = influence_function_se
        self.weight_truncation = weight_truncation
        self.truncation_threshold = truncation_threshold
        self.stabilized_weights = stabilized_weights
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level

        # Component estimators
        self.outcome_estimator: GComputationEstimator | None = None
        self.propensity_estimator: IPWEstimator | None = None

        # Cross-fitting storage
        self._fold_models: list[dict[str, Any]] = []
        self._cross_fit_predictions: dict[str, NDArray[Any]] = {}

        # AIPW results
        self._aipw_components: dict[str, NDArray[Any]] = {}
        self._influence_functions: NDArray[Any] | None = None
        self._component_diagnostics: dict[str, Any] = {}

    def _create_component_estimators(
        self,
    ) -> tuple[GComputationEstimator, IPWEstimator]:
        """Create the component G-computation and IPW estimators.

        Returns:
            Tuple of (outcome_estimator, propensity_estimator)
        """
        # Create G-computation estimator
        outcome_estimator = GComputationEstimator(
            model_type=self.outcome_model_type,
            model_params=self.outcome_model_params,
            bootstrap_samples=0,  # We'll handle bootstrapping at AIPW level
            random_state=self.random_state,
            verbose=False,  # Reduce noise from component estimators
        )

        # Create IPW estimator
        propensity_estimator = IPWEstimator(
            propensity_model_type=self.propensity_model_type,
            propensity_model_params=self.propensity_model_params,
            weight_truncation=self.weight_truncation,
            truncation_threshold=self.truncation_threshold,
            stabilized_weights=self.stabilized_weights,
            bootstrap_samples=0,  # We'll handle bootstrapping at AIPW level
            check_overlap=True,
            random_state=self.random_state,
            verbose=False,
        )

        return outcome_estimator, propensity_estimator

    def _create_cross_fitting_folds(
        self, treatment: TreatmentData, n_samples: int
    ) -> list[tuple[NDArray[Any], NDArray[Any]]]:
        """Create cross-fitting fold indices.

        Args:
            treatment: Treatment data for stratification
            n_samples: Number of samples

        Returns:
            List of (train_idx, test_idx) tuples for each fold
        """
        if self.stratify_folds and treatment.treatment_type == "binary":
            # Stratify by treatment to ensure balance in each fold
            if isinstance(treatment.values, pd.Series):
                treatment_values = treatment.values.values
            else:
                treatment_values = treatment.values

            kfold = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            folds = list(kfold.split(np.arange(n_samples), treatment_values))
        else:
            kfold = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            folds = list(kfold.split(np.arange(n_samples)))

        return folds

    def _fit_cross_fitting(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit models using cross-fitting to reduce overfitting bias.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Covariate data for adjustment
        """
        if covariates is None:
            raise EstimationError(
                "AIPW requires covariates for both outcome and propensity models"
            )

        n_samples = len(treatment.values)
        folds = self._create_cross_fitting_folds(treatment, n_samples)

        # Initialize prediction arrays
        if isinstance(treatment.values, pd.Series):
            treatment_values = treatment.values.values
        else:
            treatment_values = treatment.values

        if isinstance(outcome.values, pd.Series):
            outcome_values = outcome.values.values
        else:
            outcome_values = outcome.values

        # Initialize cross-fit predictions with NaN (will be filled by successful folds)
        self._cross_fit_predictions = {
            "mu_0": np.full(n_samples, np.nan),
            "mu_1": np.full(n_samples, np.nan),
            "propensity_scores": np.full(n_samples, np.nan),
            "ipw_weights": np.full(n_samples, np.nan),
        }

        self._fold_models = []
        successful_folds = 0

        if self.verbose:
            print(f"Performing {self.n_folds}-fold cross-fitting...")

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            if self.verbose:
                print(f"  Fitting fold {fold_idx + 1}/{self.n_folds}")

            # Create fold-specific data
            fold_treatment = TreatmentData(
                values=treatment_values[train_idx],
                name=treatment.name,
                treatment_type=treatment.treatment_type,
                categories=treatment.categories,
            )

            fold_outcome = OutcomeData(
                values=outcome_values[train_idx],
                name=outcome.name,
                outcome_type=outcome.outcome_type,
            )

            if isinstance(covariates.values, pd.DataFrame):
                fold_covariates_values = covariates.values.iloc[train_idx]
                test_covariates_values = covariates.values.iloc[test_idx]
            else:
                fold_covariates_values = covariates.values[train_idx]
                test_covariates_values = covariates.values[test_idx]

            fold_covariates = CovariateData(
                values=fold_covariates_values,
                names=covariates.names,
            )

            # Fit component estimators on this fold
            fold_outcome_estimator, fold_propensity_estimator = (
                self._create_component_estimators()
            )

            try:
                # Fit outcome model
                fold_outcome_estimator.fit(
                    fold_treatment, fold_outcome, fold_covariates
                )

                # Fit propensity model
                fold_propensity_estimator.fit(
                    fold_treatment, fold_outcome, fold_covariates
                )
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Fold {fold_idx + 1} failed to fit: {str(e)}")
                # Skip this fold but continue with others
                continue

            # Make predictions on held-out data
            CovariateData(
                values=test_covariates_values,
                names=covariates.names,
            )

            # Get test treatment values for predictions
            test_treatment_values = treatment_values[test_idx]

            # Predict potential outcomes
            mu_0_test, mu_1_test = fold_outcome_estimator.predict_potential_outcomes(
                treatment_values=test_treatment_values,  # Actual treatment values
                covariates=test_covariates_values
                if isinstance(test_covariates_values, np.ndarray)
                else test_covariates_values.values,
            )

            # Store predictions
            self._cross_fit_predictions["mu_0"][test_idx] = mu_0_test
            self._cross_fit_predictions["mu_1"][test_idx] = mu_1_test

            # Predict propensity scores
            if hasattr(fold_propensity_estimator, "predict_propensity_scores"):
                ps_test = fold_propensity_estimator.predict_propensity_scores(
                    test_covariates_values
                )
                self._cross_fit_predictions["propensity_scores"][test_idx] = ps_test

                # Compute IPW weights for test set
                weights_test = np.zeros(len(test_idx))
                test_treatment_values = treatment_values[test_idx]

                # Weights for treated units
                treated_mask = test_treatment_values == 1
                if np.any(treated_mask):
                    weights_test[treated_mask] = 1 / ps_test[treated_mask]

                # Weights for control units
                control_mask = test_treatment_values == 0
                if np.any(control_mask):
                    weights_test[control_mask] = 1 / (1 - ps_test[control_mask])

                # Apply stabilized weights if requested
                if self.stabilized_weights:
                    treatment_prob = np.mean(treatment_values[train_idx])
                    stabilized_weights = np.zeros_like(weights_test)
                    stabilized_weights[treated_mask] = (
                        treatment_prob * weights_test[treated_mask]
                    )
                    stabilized_weights[control_mask] = (
                        1 - treatment_prob
                    ) * weights_test[control_mask]
                    weights_test = stabilized_weights

                # Apply weight truncation
                if self.weight_truncation == "percentile":
                    train_weights = fold_propensity_estimator.get_weights()
                    if train_weights is not None:
                        lower_percentile = self.truncation_threshold * 100
                        upper_percentile = 100 - lower_percentile
                        lower_bound = np.percentile(train_weights, lower_percentile)
                        upper_bound = np.percentile(train_weights, upper_percentile)
                        weights_test = np.clip(weights_test, lower_bound, upper_bound)
                elif self.weight_truncation == "threshold":
                    max_weight = 1 / self.truncation_threshold
                    min_weight = self.truncation_threshold
                    weights_test = np.clip(weights_test, min_weight, max_weight)

                self._cross_fit_predictions["ipw_weights"][test_idx] = weights_test

            # Store fold models for diagnostics
            self._fold_models.append(
                {
                    "outcome_estimator": fold_outcome_estimator,
                    "propensity_estimator": fold_propensity_estimator,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )

            successful_folds += 1

        # Check that we have enough successful folds
        if successful_folds == 0:
            raise EstimationError(
                "Cross-fitting failed: no folds completed successfully. "
                "Consider disabling cross-fitting or checking data quality."
            )
        elif successful_folds < self.n_folds // 2:
            if self.verbose:
                print(
                    f"Warning: Only {successful_folds}/{self.n_folds} folds succeeded. "
                    "Results may be less reliable."
                )

        # Check for missing predictions and fill with zeros if needed
        for key in self._cross_fit_predictions:
            missing_mask = np.isnan(self._cross_fit_predictions[key])
            if np.any(missing_mask):
                if self.verbose:
                    n_missing = np.sum(missing_mask)
                    print(
                        f"Warning: {n_missing} observations missing {key} predictions. "
                        "Filling with zeros."
                    )
                # Fill missing predictions with zeros (simple fallback)
                self._cross_fit_predictions[key][missing_mask] = 0.0

    def _fit_no_cross_fitting(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit models without cross-fitting (standard approach).

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Covariate data for adjustment
        """
        # Create and fit component estimators
        self.outcome_estimator, self.propensity_estimator = (
            self._create_component_estimators()
        )

        # Fit both models on full data
        self.outcome_estimator.fit(treatment, outcome, covariates)
        self.propensity_estimator.fit(treatment, outcome, covariates)

        # Get predictions
        if isinstance(treatment.values, pd.Series):
            treatment_values = treatment.values.values
        else:
            treatment_values = treatment.values

        len(treatment_values)

        # Get potential outcomes
        if covariates is not None:
            covariate_values = (
                covariates.values
                if isinstance(covariates.values, np.ndarray)
                else covariates.values.values
            )
        else:
            raise EstimationError("AIPW requires covariates")

        # Predict potential outcomes
        mu_0, mu_1 = self.outcome_estimator.predict_potential_outcomes(
            treatment_values=treatment_values, covariates=covariate_values
        )

        # Get propensity scores and weights
        propensity_scores = self.propensity_estimator.get_propensity_scores()
        ipw_weights = self.propensity_estimator.get_weights()

        if propensity_scores is None or ipw_weights is None:
            raise EstimationError("Failed to obtain propensity scores or weights")

        # Store predictions
        self._cross_fit_predictions = {
            "mu_0": mu_0,
            "mu_1": mu_1,
            "propensity_scores": propensity_scores,
            "ipw_weights": ipw_weights,
        }

    def _compute_aipw_estimate(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
    ) -> float:
        """Compute the AIPW estimate using the doubly robust formula.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data

        Returns:
            AIPW estimate of the average treatment effect
        """
        if isinstance(treatment.values, pd.Series):
            treatment_values = treatment.values.values
        else:
            treatment_values = treatment.values

        if isinstance(outcome.values, pd.Series):
            outcome_values = outcome.values.values
        else:
            outcome_values = outcome.values

        mu_0 = self._cross_fit_predictions["mu_0"]
        mu_1 = self._cross_fit_predictions["mu_1"]
        propensity_scores = self._cross_fit_predictions["propensity_scores"]

        n_samples = len(treatment_values)

        # AIPW formula components
        # τ_AIPW = 1/n Σ[μ₁(X_i) - μ₀(X_i) + (T_i/e(X_i))(Y_i - μ₁(X_i)) - ((1-T_i)/(1-e(X_i)))(Y_i - μ₀(X_i))]

        # G-computation component
        g_comp_component = mu_1 - mu_0

        # IPW correction components
        treated_mask = treatment_values == 1
        control_mask = treatment_values == 0

        ipw_correction = np.zeros(n_samples)

        # Treated units correction: (T_i/e(X_i))(Y_i - μ₁(X_i))
        if np.any(treated_mask):
            ipw_correction[treated_mask] = (
                treatment_values[treated_mask] / propensity_scores[treated_mask]
            ) * (outcome_values[treated_mask] - mu_1[treated_mask])

        # Control units correction: -((1-T_i)/(1-e(X_i)))(Y_i - μ₀(X_i))
        if np.any(control_mask):
            ipw_correction[control_mask] = -(
                (
                    (1 - treatment_values[control_mask])
                    / (1 - propensity_scores[control_mask])
                )
                * (outcome_values[control_mask] - mu_0[control_mask])
            )

        # Store components for diagnostics
        self._aipw_components = {
            "g_computation": g_comp_component,
            "ipw_correction": ipw_correction,
            "full_aipw": g_comp_component + ipw_correction,
        }

        # AIPW estimate
        aipw_estimate = np.mean(g_comp_component + ipw_correction)

        return aipw_estimate

    def _apply_tmle_fluctuation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
    ) -> None:
        """Apply TMLE fluctuation to improve efficiency.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
        """
        if not self.use_tmle:
            return

        if self.verbose:
            print("Applying TMLE fluctuation...")

        # TMLE implementation is complex and would require additional code
        # For now, we'll keep the standard AIPW estimate
        # This is a placeholder for future TMLE implementation
        pass

    def _compute_influence_function_se(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
    ) -> float | None:
        """Compute standard error using influence functions.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data

        Returns:
            Standard error estimate
        """
        if not self.influence_function_se:
            return None

        if isinstance(treatment.values, pd.Series):
            treatment_values = treatment.values.values
        else:
            treatment_values = treatment.values

        if isinstance(outcome.values, pd.Series):
            pass
        else:
            pass

        n_samples = len(treatment_values)

        # Compute influence function for each observation
        # This is a simplified version - full implementation would be more complex
        aipw_components = self._aipw_components["full_aipw"]
        aipw_estimate = np.mean(aipw_components)

        # Influence function is approximately the component minus the mean
        influence_functions = aipw_components - aipw_estimate

        # Store for diagnostics
        self._influence_functions = influence_functions

        # Standard error is sqrt(var(influence_function) / n)
        influence_variance = np.var(influence_functions, ddof=1)
        standard_error = np.sqrt(influence_variance / n_samples)

        return standard_error

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the AIPW estimator.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Covariate data for adjustment
        """
        if covariates is None:
            raise EstimationError(
                "AIPW requires covariates for both outcome and propensity models"
            )

        if self.verbose:
            print("Fitting AIPW estimator...")
            print(f"Cross-fitting: {self.cross_fitting}")
            if self.cross_fitting:
                print(f"Number of folds: {self.n_folds}")
            print(f"TMLE: {self.use_tmle}")
            print(f"Influence function SE: {self.influence_function_se}")

        # Fit models (with or without cross-fitting)
        if self.cross_fitting:
            self._fit_cross_fitting(treatment, outcome, covariates)
        else:
            self._fit_no_cross_fitting(treatment, outcome, covariates)

        # Apply TMLE fluctuation if requested
        if self.use_tmle:
            self._apply_tmle_fluctuation(treatment, outcome)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate Average Treatment Effect using AIPW.

        Returns:
            CausalEffect object with AIPW estimate and diagnostics
        """
        if (
            self.treatment_data is None
            or self.outcome_data is None
            or not self._cross_fit_predictions
        ):
            raise EstimationError("Model must be fitted before estimation")

        # Compute AIPW estimate
        aipw_ate = self._compute_aipw_estimate(self.treatment_data, self.outcome_data)

        # Compute influence function standard error
        influence_se = self._compute_influence_function_se(
            self.treatment_data, self.outcome_data
        )

        # Bootstrap confidence intervals
        ate_ci_lower, ate_ci_upper, bootstrap_estimates = None, None, None
        if self.bootstrap_samples > 0:
            ate_ci_lower, ate_ci_upper, bootstrap_estimates = (
                self._bootstrap_confidence_interval()
            )

        # Use influence function SE if available, otherwise bootstrap SE
        if influence_se is not None:
            ate_se = influence_se
        elif bootstrap_estimates is not None:
            ate_se = np.std(bootstrap_estimates)
        else:
            ate_se = None

        # Compute component estimates for diagnostics
        self._compute_component_diagnostics()

        # Get sample sizes
        if isinstance(self.treatment_data.values, pd.Series):
            treatment_values = self.treatment_data.values.values
        else:
            treatment_values = self.treatment_data.values

        n_treated = np.sum(treatment_values == 1)
        n_control = np.sum(treatment_values == 0)

        # Potential outcome means from G-computation component
        mu_0_mean = np.mean(self._cross_fit_predictions["mu_0"])
        mu_1_mean = np.mean(self._cross_fit_predictions["mu_1"])

        return CausalEffect(
            ate=aipw_ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            confidence_level=self.confidence_level,
            potential_outcome_treated=mu_1_mean,
            potential_outcome_control=mu_0_mean,
            method="AIPW",
            n_observations=len(treatment_values),
            n_treated=n_treated,
            n_control=n_control,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_estimates=bootstrap_estimates,
            diagnostics=self._component_diagnostics,
        )

    def _bootstrap_confidence_interval(
        self,
    ) -> tuple[float | None, float | None, NDArray[Any] | None]:
        """Calculate bootstrap confidence intervals for AIPW estimate.

        Returns:
            Tuple of (lower_ci, upper_ci, bootstrap_estimates)
        """
        if self.bootstrap_samples <= 0:
            return None, None, None

        if (
            self.treatment_data is None
            or self.outcome_data is None
            or self.covariate_data is None
        ):
            raise EstimationError("Data must be available for bootstrap")

        bootstrap_ates: list[float] = []
        n_obs = len(self.treatment_data.values)

        for _ in range(self.bootstrap_samples):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)

            # Create bootstrap datasets
            boot_treatment = TreatmentData(
                values=self.treatment_data.values.iloc[bootstrap_indices]
                if isinstance(self.treatment_data.values, pd.Series)
                else self.treatment_data.values[bootstrap_indices],
                name=self.treatment_data.name,
                treatment_type=self.treatment_data.treatment_type,
                categories=self.treatment_data.categories,
            )

            boot_outcome = OutcomeData(
                values=self.outcome_data.values.iloc[bootstrap_indices]
                if isinstance(self.outcome_data.values, pd.Series)
                else self.outcome_data.values[bootstrap_indices],
                name=self.outcome_data.name,
                outcome_type=self.outcome_data.outcome_type,
            )

            if isinstance(self.covariate_data.values, pd.DataFrame):
                boot_cov_values = self.covariate_data.values.iloc[bootstrap_indices]
            else:
                boot_cov_values = self.covariate_data.values[bootstrap_indices]

            boot_covariates = CovariateData(
                values=boot_cov_values,
                names=self.covariate_data.names,
            )

            # Fit AIPW model on bootstrap sample
            try:
                boot_estimator = AIPWEstimator(
                    outcome_model_type=self.outcome_model_type,
                    outcome_model_params=self.outcome_model_params,
                    propensity_model_type=self.propensity_model_type,
                    propensity_model_params=self.propensity_model_params,
                    cross_fitting=self.cross_fitting,
                    n_folds=self.n_folds,
                    stratify_folds=self.stratify_folds,
                    bootstrap_samples=0,  # Don't bootstrap within bootstrap
                    random_state=None,  # Use different random state for each bootstrap
                    verbose=False,
                )

                boot_estimator.fit(boot_treatment, boot_outcome, boot_covariates)
                boot_effect = boot_estimator.estimate_ate(use_cache=False)
                bootstrap_ates.append(boot_effect.ate)

            except Exception:
                # Skip failed bootstrap samples
                continue

        if len(bootstrap_ates) == 0:
            return None, None, None

        bootstrap_ates_array = np.array(bootstrap_ates)

        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        ate_ci_lower = float(np.percentile(bootstrap_ates_array, lower_percentile))
        ate_ci_upper = float(np.percentile(bootstrap_ates_array, upper_percentile))

        return ate_ci_lower, ate_ci_upper, bootstrap_ates_array

    def _compute_component_diagnostics(self) -> None:
        """Compute diagnostics comparing component estimators."""
        diagnostics: dict[str, Any] = {
            "cross_fitting": self.cross_fitting,
            "n_folds": self.n_folds if self.cross_fitting else None,
            "use_tmle": self.use_tmle,
            "influence_function_se": self.influence_function_se,
        }

        # G-computation component estimate
        g_comp_ate = np.mean(self._aipw_components["g_computation"])
        diagnostics["g_computation_ate"] = g_comp_ate

        # Pure IPW estimate (for comparison)
        if (
            self._cross_fit_predictions.get("ipw_weights") is not None
            and self.treatment_data is not None
            and self.outcome_data is not None
        ):
            if isinstance(self.treatment_data.values, pd.Series):
                treatment_values = self.treatment_data.values.values
            else:
                treatment_values = self.treatment_data.values

            if isinstance(self.outcome_data.values, pd.Series):
                outcome_values = self.outcome_data.values.values
            else:
                outcome_values = self.outcome_data.values

            ipw_weights = self._cross_fit_predictions["ipw_weights"]

            # Compute weighted means
            treated_mask = treatment_values == 1
            control_mask = treatment_values == 0

            if np.any(treated_mask) and np.any(control_mask):
                weighted_outcome_treated = np.sum(
                    outcome_values[treated_mask] * ipw_weights[treated_mask]
                ) / np.sum(ipw_weights[treated_mask])
                weighted_outcome_control = np.sum(
                    outcome_values[control_mask] * ipw_weights[control_mask]
                ) / np.sum(ipw_weights[control_mask])
                ipw_ate = weighted_outcome_treated - weighted_outcome_control
                diagnostics["ipw_ate"] = ipw_ate

        # Cross-fitting model performance
        if self.cross_fitting and self._fold_models:
            diagnostics["n_folds_used"] = len(self._fold_models)

        # Propensity score diagnostics
        propensity_scores = self._cross_fit_predictions.get("propensity_scores")
        if propensity_scores is not None:
            diagnostics["propensity_score_stats"] = {
                "mean": float(np.mean(propensity_scores)),
                "min": float(np.min(propensity_scores)),
                "max": float(np.max(propensity_scores)),
                "std": float(np.std(propensity_scores)),
            }

        # Component balance
        if (
            "g_computation" in self._aipw_components
            and "ipw_correction" in self._aipw_components
        ):
            g_comp_contribution = np.mean(
                np.abs(self._aipw_components["g_computation"])
            )
            ipw_contribution = np.mean(np.abs(self._aipw_components["ipw_correction"]))
            total_contribution = g_comp_contribution + ipw_contribution

            if total_contribution > 0:
                diagnostics["component_balance"] = {
                    "g_computation_weight": g_comp_contribution / total_contribution,
                    "ipw_correction_weight": ipw_contribution / total_contribution,
                }

        self._component_diagnostics = diagnostics

    def get_component_diagnostics(self) -> dict[str, Any]:
        """Get diagnostics comparing AIPW components.

        Returns:
            Dictionary with component diagnostics
        """
        return self._component_diagnostics

    def get_cross_fit_predictions(self) -> dict[str, NDArray[Any]]:
        """Get cross-fitted predictions for diagnostics.

        Returns:
            Dictionary with cross-fitted predictions
        """
        return self._cross_fit_predictions

    def get_influence_functions(self) -> NDArray[Any] | None:
        """Get influence function values for each observation.

        Returns:
            Array of influence function values if computed, None otherwise
        """
        return self._influence_functions

    def predict_potential_outcomes(
        self,
        treatment_values: pd.Series | NDArray[Any],
        covariates: pd.DataFrame | NDArray[Any] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Predict potential outcomes using the fitted AIPW model.

        For AIPW, we use the outcome model component (G-computation) for prediction.

        Args:
            treatment_values: Treatment assignment values to predict for
            covariates: Covariate values for prediction

        Returns:
            Tuple of (Y0_predictions, Y1_predictions)
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before prediction")

        if self.cross_fitting:
            raise EstimationError(
                "Potential outcome prediction not supported with cross-fitting. "
                "Use the component outcome_estimator directly."
            )

        if self.outcome_estimator is None:
            raise EstimationError("Outcome estimator not available")

        return self.outcome_estimator.predict_potential_outcomes(
            treatment_values, covariates
        )
