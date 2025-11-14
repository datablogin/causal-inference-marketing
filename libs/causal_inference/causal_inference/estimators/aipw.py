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
- Comprehensive diagnostics comparing component models
- Support for multiple machine learning models

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

Advanced Usage with Different Models:
    >>> # Use random forest models for both outcome and propensity
    >>> estimator_rf = AIPWEstimator(
    ...     outcome_model_type="random_forest",
    ...     propensity_model_type="random_forest",
    ...     outcome_model_params={"n_estimators": 100, "max_depth": 5},
    ...     propensity_model_params={"n_estimators": 100, "max_depth": 3},
    ...     cross_fitting=True,
    ...     n_folds=10,
    ...     influence_function_se=True,
    ...     verbose=True
    ... )
    >>>
    >>> estimator_rf.fit(treatment, outcome, covariates)
    >>> effect_rf = estimator_rf.estimate_ate()

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

Component Balance Optimization with Custom Bounds:
    >>> # Optimize component weighting with tighter bounds for more conservative balance
    >>> estimator_opt = AIPWEstimator(
    ...     cross_fitting=False,
    ...     optimize_component_balance=True,
    ...     component_weight_bounds=(0.4, 0.6),  # Tighter bounds than default (0.3, 0.7)
    ...     component_variance_penalty=0.5,
    ...     influence_function_se=False,  # Required with optimization
    ...     bootstrap_samples=1000,  # Use bootstrap for valid CIs
    ...     random_state=42
    ... )
    >>>
    >>> estimator_opt.fit(treatment, outcome, covariates)
    >>> effect_opt = estimator_opt.estimate_ate()
    >>> # Check optimized weights
    >>> opt_diag = estimator_opt.get_optimization_diagnostics()
    >>> print(f"G-computation weight: {opt_diag['optimal_g_computation_weight']:.3f}")
    >>> print(f"IPW weight: {opt_diag['optimal_ipw_weight']:.3f}")

Notes:
    - AIPW is doubly robust: consistent if either outcome or propensity model is correct
    - Cross-fitting reduces finite-sample bias from model overfitting
    - Influence function variance estimation accounts for all sources of uncertainty
    - Propensity scores are automatically bounded to prevent numerical instability
    - Compare component models to assess which may be misspecified
"""

from __future__ import annotations

from typing import Any, Optional

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
from ..core.bootstrap import BootstrapConfig, BootstrapMixin
from ..core.optimization_mixin import OptimizationMixin
from .g_computation import GComputationEstimator
from .ipw import IPWEstimator


class AIPWEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):
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

    Important Note on Component Optimization:
        When optimize_component_balance=True, the optimization happens during
        estimate_ate(), NOT during fit(). This is because component optimization
        requires both G-computation and IPW components, which are computed during
        estimation. Therefore, get_optimization_diagnostics() will return None
        until after estimate_ate() is called.

        Correct usage:
            estimator.fit(treatment, outcome, covariates)  # No optimization yet
            effect = estimator.estimate_ate()             # Optimization happens here
            diagnostics = estimator.get_optimization_diagnostics()  # Now available

    Attributes:
        outcome_estimator: G-computation estimator for outcome models
        propensity_estimator: IPW estimator for propensity scores
        cross_fitting: Whether to use cross-fitting
        n_folds: Number of cross-fitting folds
        influence_function_se: Whether to use influence function standard errors
    """

    def __init__(
        self,
        outcome_model_type: str = "auto",
        outcome_model_params: Optional[dict[str, Any]] = None,
        propensity_model_type: str = "logistic",
        propensity_model_params: Optional[dict[str, Any]] = None,
        cross_fitting: bool = True,
        n_folds: int = 5,
        stratify_folds: bool = True,
        influence_function_se: bool = True,
        weight_truncation: Optional[str] = "percentile",
        truncation_threshold: float = 0.01,
        stabilized_weights: bool = True,
        bootstrap_config: Optional[Any] = None,
        optimization_config: Optional[Any] = None,
        # Component optimization settings
        optimize_component_balance: bool = False,
        component_variance_penalty: float = 0.5,
        component_weight_bounds: tuple[float, float] = (0.3, 0.7),
        # Legacy parameters for backward compatibility
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
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
            influence_function_se: Whether to compute influence function standard errors
            weight_truncation: IPW weight truncation method
            truncation_threshold: Threshold for weight truncation
            stabilized_weights: Whether to use stabilized IPW weights
            bootstrap_config: Configuration for bootstrap confidence intervals
            optimization_config: Configuration for optimization strategies
            optimize_component_balance: Optimize G-computation vs IPW balance. Note: optimization
                happens during estimate_ate(), so diagnostics are only available after calling
                estimate_ate(). This differs from IPW, where optimization happens during fit().
            component_variance_penalty: Penalty for deviating from 50/50 balance
            component_weight_bounds: Bounds for component weight optimization (min, max). Ensures
                both G-computation and IPW components contribute meaningfully. Default (0.3, 0.7)
                means the G-computation weight is constrained between 30% and 70%. Set equal bounds
                like (0.5, 0.5) to force a specific fixed weighting.
            bootstrap_samples: Legacy parameter - number of bootstrap samples (use bootstrap_config instead)
            confidence_level: Legacy parameter - confidence level (use bootstrap_config instead)
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        # Create bootstrap config if not provided (for backward compatibility)
        if bootstrap_config is None:
            bootstrap_config = BootstrapConfig(
                n_samples=bootstrap_samples,
                confidence_level=confidence_level,
                random_state=random_state,
            )

        super().__init__(
            bootstrap_config=bootstrap_config,
            optimization_config=optimization_config,
            random_state=random_state,
            verbose=verbose,
        )

        self.outcome_model_type = outcome_model_type
        self.outcome_model_params = outcome_model_params or {}
        self.propensity_model_type = propensity_model_type
        self.propensity_model_params = propensity_model_params or {}
        self.cross_fitting = cross_fitting
        self.n_folds = n_folds
        self.stratify_folds = stratify_folds
        self.influence_function_se = influence_function_se
        self.weight_truncation = weight_truncation
        self.truncation_threshold = truncation_threshold
        self.stabilized_weights = stabilized_weights
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level

        # Component optimization settings
        self.optimize_component_balance = optimize_component_balance
        self.component_variance_penalty = component_variance_penalty
        self.component_weight_bounds = component_weight_bounds

        # Validate component optimization settings
        if component_variance_penalty < 0:
            raise ValueError(
                f"component_variance_penalty must be non-negative, got {component_variance_penalty}"
            )

        if (
            len(component_weight_bounds) != 2
            or component_weight_bounds[0] > component_weight_bounds[1]
            or component_weight_bounds[0] < 0
            or component_weight_bounds[1] > 1
        ):
            raise ValueError(
                f"component_weight_bounds must be (min, max) with 0 <= min <= max <= 1, "
                f"got {component_weight_bounds}"
            )

        if optimize_component_balance and influence_function_se:
            raise ValueError(
                "Cannot use influence_function_se=True with optimize_component_balance=True. "
                "Influence function standard errors assume a fixed estimator formula (50/50 weighting), "
                "but component optimization data-adaptively chooses weights, invalidating this assumption. "
                "The influence function would need to account for the weight selection process, which is complex. "
                "Either set influence_function_se=False or disable optimization. "
                "Use bootstrap_config to compute valid confidence intervals with optimization."
            )

        # Component estimators
        self.outcome_estimator: Optional[GComputationEstimator] = None
        self.propensity_estimator: Optional[IPWEstimator] = None

        # Cross-fitting storage
        self._fold_models: list[dict[str, Any]] = []
        self._cross_fit_predictions: dict[str, NDArray[Any]] = {}

        # AIPW results
        self._aipw_components: dict[str, NDArray[Any]] = {}
        self._influence_functions: Optional[NDArray[Any]] = None
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

    def _create_bootstrap_estimator(
        self, random_state: Optional[int] = None
    ) -> AIPWEstimator:
        """Create a new estimator instance for bootstrap sampling.

        Args:
            random_state: Random state for this bootstrap instance

        Returns:
            New AIPWEstimator instance configured for bootstrap
        """
        return AIPWEstimator(
            outcome_model_type=self.outcome_model_type,
            outcome_model_params=self.outcome_model_params,
            propensity_model_type=self.propensity_model_type,
            propensity_model_params=self.propensity_model_params,
            cross_fitting=self.cross_fitting,
            n_folds=self.n_folds,
            stratify_folds=self.stratify_folds,
            influence_function_se=self.influence_function_se,
            weight_truncation=self.weight_truncation,
            truncation_threshold=self.truncation_threshold,
            stabilized_weights=self.stabilized_weights,
            bootstrap_config=BootstrapConfig(n_samples=0),  # No nested bootstrap
            optimize_component_balance=self.optimize_component_balance,
            component_variance_penalty=self.component_variance_penalty,
            component_weight_bounds=self.component_weight_bounds,
            random_state=random_state,
            verbose=False,  # Reduce verbosity in bootstrap
        )

    def _ensure_propensity_score_bounds(
        self,
        propensity_scores: NDArray[Any],
        min_bound: float = 1e-6,
        max_bound: float = 1 - 1e-6,
    ) -> NDArray[Any]:
        """Ensure propensity scores are within safe bounds to prevent division by zero.

        Args:
            propensity_scores: Array of propensity scores
            min_bound: Minimum allowed propensity score
            max_bound: Maximum allowed propensity score

        Returns:
            Array of bounded propensity scores
        """
        bounded_scores = np.clip(propensity_scores, min_bound, max_bound)

        # Count and warn about extreme values
        n_clipped_low = np.sum(propensity_scores < min_bound)
        n_clipped_high = np.sum(propensity_scores > max_bound)

        if self.verbose and (n_clipped_low > 0 or n_clipped_high > 0):
            total_clipped = n_clipped_low + n_clipped_high
            print(f"  Clipped {total_clipped} extreme propensity scores to safe bounds")
            if n_clipped_low > 0:
                print(f"    {n_clipped_low} scores < {min_bound} (near 0)")
            if n_clipped_high > 0:
                print(f"    {n_clipped_high} scores > {max_bound} (near 1)")

        return bounded_scores

    def _validate_overlap_assumption(self, propensity_scores: NDArray[Any]) -> None:
        """Validate the overlap/positivity assumption.

        Args:
            propensity_scores: Array of propensity scores

        Raises:
            EstimationError: If overlap assumption is violated
        """
        # Check for extreme propensity scores that violate overlap
        overlap_threshold = 0.05  # Common threshold for practical overlap

        n_extreme_low = np.sum(propensity_scores < overlap_threshold)
        n_extreme_high = np.sum(propensity_scores > (1 - overlap_threshold))
        n_total = len(propensity_scores)

        if n_extreme_low > n_total * 0.05:  # More than 5% have very low PS
            if self.verbose:
                print(
                    f"Warning: {n_extreme_low} observations ({n_extreme_low / n_total:.1%}) have propensity scores < {overlap_threshold}"
                )
                print("This may indicate poor overlap and unreliable AIPW estimates")

        if n_extreme_high > n_total * 0.05:  # More than 5% have very high PS
            if self.verbose:
                print(
                    f"Warning: {n_extreme_high} observations ({n_extreme_high / n_total:.1%}) have propensity scores > {1 - overlap_threshold}"
                )
                print("This may indicate poor overlap and unreliable AIPW estimates")

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
        covariates: Optional[CovariateData] = None,
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
            treatment_values = np.asarray(treatment.values.values)
        else:
            treatment_values = np.asarray(treatment.values)

        if isinstance(outcome.values, pd.Series):
            outcome_values = np.asarray(outcome.values.values)
        else:
            outcome_values = np.asarray(outcome.values)

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

            # Create fold-specific data with validation
            fold_treatment_values = treatment_values[train_idx]
            fold_outcome_values = outcome_values[train_idx]

            # Check for NaN values in training data
            if np.any(np.isnan(fold_treatment_values)) or np.any(
                np.isnan(fold_outcome_values)
            ):
                if self.verbose:
                    print(
                        f"    Warning: Fold {fold_idx + 1} has NaN values in treatment/outcome"
                    )
                continue

            fold_treatment = TreatmentData(
                values=fold_treatment_values,
                name=treatment.name,
                treatment_type=treatment.treatment_type,
                categories=treatment.categories,
            )

            fold_outcome = OutcomeData(
                values=fold_outcome_values,
                name=outcome.name,
                outcome_type=outcome.outcome_type,
            )

            if isinstance(covariates.values, pd.DataFrame):
                fold_covariates_values = covariates.values.iloc[train_idx]
                test_covariates_values = covariates.values.iloc[test_idx]
            else:
                fold_covariates_values = covariates.values[train_idx]
                test_covariates_values = covariates.values[test_idx]

            # Check for NaN values in covariate data
            if isinstance(fold_covariates_values, pd.DataFrame):
                if fold_covariates_values.isna().any().any():
                    if self.verbose:
                        print(
                            f"    Warning: Fold {fold_idx + 1} has NaN values in covariates"
                        )
                    continue
            else:
                if np.any(np.isnan(fold_covariates_values)):
                    if self.verbose:
                        print(
                            f"    Warning: Fold {fold_idx + 1} has NaN values in covariates"
                        )
                    continue

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
                ps_test_raw = fold_propensity_estimator.predict_propensity_scores(
                    test_covariates_values
                )
                # Apply bounds to prevent division by zero
                ps_test = self._ensure_propensity_score_bounds(ps_test_raw)
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
            if self.verbose:
                print(
                    "Warning: Cross-fitting failed completely. Falling back to no cross-fitting."
                )
            # Fallback to no cross-fitting
            self._fit_no_cross_fitting(treatment, outcome, covariates)
            return
        elif successful_folds < self.n_folds // 2:
            if self.verbose:
                print(
                    f"Warning: Only {successful_folds}/{self.n_folds} folds succeeded. "
                    "Results may be less reliable."
                )
            # If too few folds succeeded, raise an error to be handled upstream
            raise EstimationError(
                f"Cross-fitting failed: only {successful_folds}/{self.n_folds} folds completed successfully. "
                "Consider disabling cross-fitting or checking data quality."
            )

        # Check for missing predictions and handle appropriately
        for key in self._cross_fit_predictions:
            missing_mask = np.isnan(self._cross_fit_predictions[key])
            if np.any(missing_mask):
                n_missing = np.sum(missing_mask)
                if self.verbose:
                    print(
                        f"Warning: {n_missing} observations missing {key} predictions."
                    )

                # For critical missing predictions, raise an error
                if n_missing > len(treatment_values) * 0.1:  # More than 10% missing
                    raise EstimationError(
                        f"Too many missing {key} predictions ({n_missing}/{len(treatment_values)}). "
                        "Cross-fitting failed for too many observations. "
                        "Consider reducing n_folds or checking data quality."
                    )

                # For small amounts of missing data, use model-based imputation
                if key in ["mu_0", "mu_1"]:
                    # Use mean imputation for potential outcomes
                    valid_mask = ~missing_mask
                    if np.any(valid_mask):
                        mean_value = np.mean(
                            self._cross_fit_predictions[key][valid_mask]
                        )
                        self._cross_fit_predictions[key][missing_mask] = mean_value
                        if self.verbose:
                            print(
                                f"  Imputed missing {key} with mean: {mean_value:.4f}"
                            )
                    else:
                        raise EstimationError(f"All {key} predictions are missing")

                elif key == "propensity_scores":
                    # Use marginal treatment probability for missing propensity scores
                    marginal_prob = np.mean(treatment_values)
                    self._cross_fit_predictions[key][missing_mask] = marginal_prob
                    if self.verbose:
                        print(
                            f"  Imputed missing propensity scores with marginal probability: {marginal_prob:.4f}"
                        )

                elif key == "ipw_weights":
                    # Recompute weights from imputed propensity scores
                    ps_values = self._cross_fit_predictions["propensity_scores"][
                        missing_mask
                    ]
                    treatment_vals = treatment_values[missing_mask]
                    weights = np.zeros_like(ps_values)

                    treated_mask = treatment_vals == 1
                    control_mask = treatment_vals == 0

                    if np.any(treated_mask):
                        weights[treated_mask] = 1 / ps_values[treated_mask]
                    if np.any(control_mask):
                        weights[control_mask] = 1 / (1 - ps_values[control_mask])

                    self._cross_fit_predictions[key][missing_mask] = weights
                    if self.verbose:
                        print(
                            "  Recomputed missing IPW weights from imputed propensity scores"
                        )

    def _fit_no_cross_fitting(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Optional[CovariateData] = None,
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

        # Fit both models on full data with error handling
        try:
            self.outcome_estimator.fit(treatment, outcome, covariates)
        except Exception as e:
            raise EstimationError(
                f"Failed to fit outcome model in AIPW: {str(e)}"
            ) from e

        try:
            self.propensity_estimator.fit(treatment, outcome, covariates)
        except Exception as e:
            raise EstimationError(
                f"Failed to fit propensity model in AIPW: {str(e)}"
            ) from e

        # Get predictions
        if isinstance(treatment.values, pd.Series):
            treatment_values = np.asarray(treatment.values.values)
        else:
            treatment_values = np.asarray(treatment.values)

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
        propensity_scores_raw = self.propensity_estimator.get_propensity_scores()
        ipw_weights = self.propensity_estimator.get_weights()

        if propensity_scores_raw is None or ipw_weights is None:
            raise EstimationError("Failed to obtain propensity scores or weights")

        # Apply bounds to propensity scores for numerical stability
        propensity_scores = self._ensure_propensity_score_bounds(propensity_scores_raw)

        # Validate overlap assumption
        self._validate_overlap_assumption(propensity_scores)

        # Store predictions
        self._cross_fit_predictions = {
            "mu_0": mu_0,
            "mu_1": mu_1,
            "propensity_scores": propensity_scores,
            "ipw_weights": ipw_weights,
        }

    def _optimize_component_balance(
        self,
        g_comp_components: NDArray[np.floating[Any]],
        ipw_components: NDArray[np.floating[Any]],
    ) -> float:
        """Optimize balance between G-computation and IPW components.

        Optimization Objective:
            Minimizes the estimated variance of the weighted AIPW estimator:

                Var(τ̂_opt) = Var(α·g_comp + (1-α)·ipw) / n

            where g_comp and ipw are the unit-level AIPW components. The weight
            α is constrained to [0.3, 0.7] with a penalty for deviations from 0.5
            to ensure both components contribute meaningfully.

        Mathematical Formulation:
            The weighted AIPW estimator is:

                τ̂_opt = 1/n Σ[α·(μ₁(X_i) - μ₀(X_i)) + (1-α)·IPW_corr_i]

            We minimize:
                objective(α) = Var(components) / n + λ·(α - 0.5)²

            where λ is the component_variance_penalty.

        Theoretical Tradeoffs:
            **Standard AIPW (α=1)** is doubly robust: consistent if EITHER the
            outcome model OR propensity model is correct. The IPW correction
            exactly cancels bias when the outcome model is wrong.

            **Optimized AIPW (α ≠ 1)** trades double robustness for efficiency:
            - Lower variance when both models are reasonably correct
            - Potential bias increase if both models are misspecified
            - Still benefits from having two models, but not fully doubly robust

        When to Use:
            ✓ Both models are well-specified (validated through diagnostics)
            ✓ Sample size is large enough for precise variance estimation
            ✓ Primary concern is efficiency over robustness
            ✓ Using bootstrap for valid confidence intervals

        When NOT to Use:
            ✗ Strong concern about model misspecification
            ✗ Small samples (n < 200)
            ✗ Poor covariate overlap (extreme propensity scores)
            ✗ Relying on influence function standard errors

        Args:
            g_comp_components: G-computation component values (μ₁ - μ₀) for each unit
            ipw_components: IPW correction component values for each unit

        Returns:
            Optimal weight for G-computation component (alpha), bounded by component_weight_bounds

        References:
            - PyRake: https://github.com/rwilson4/PyRake
            - Doubly robust estimation: Bang & Robins (2005), Biometrics
            - Variance-optimal weighting: van der Laan & Rose (2011), Targeted Learning
        """
        from scipy.optimize import OptimizeResult, minimize_scalar

        def objective(alpha: float) -> float:
            """Weighted AIPW estimator variance with balance penalty.

            Minimizes Var(α·ḡ + (1-α)·īpw) = Var(components) / n
            where components = α·g_comp + (1-α)·ipw for each unit.
            """
            # alpha is weight on G-computation (0 to 1)
            # (1 - alpha) is implicit weight on IPW

            combined = alpha * g_comp_components + (1 - alpha) * ipw_components
            n = len(combined)

            # Variance of the mean estimator = Var(components) / n
            estimator_variance = float(np.var(combined, ddof=1) / n)

            # Penalty for extreme weights (force meaningful contribution from both)
            balance_penalty = self.component_variance_penalty * (alpha - 0.5) ** 2

            return estimator_variance + balance_penalty

        result: OptimizeResult = minimize_scalar(
            objective,
            bounds=self.component_weight_bounds,
            method="bounded",
        )

        optimal_alpha: float = float(result.x)

        # Store diagnostics
        n = len(g_comp_components)
        optimized_components = (
            optimal_alpha * g_comp_components + (1 - optimal_alpha) * ipw_components
        )
        standard_components = g_comp_components + ipw_components

        self._optimization_diagnostics = {
            "optimal_g_computation_weight": optimal_alpha,
            "optimal_ipw_weight": 1 - optimal_alpha,
            "optimized_estimator_variance": float(
                np.var(optimized_components, ddof=1) / n
            ),
            "standard_estimator_variance": float(
                np.var(standard_components, ddof=1) / n
            ),
            # Also store component-level variances for reference
            "optimized_component_variance": float(np.var(optimized_components, ddof=1)),
            "standard_component_variance": float(np.var(standard_components, ddof=1)),
        }

        if self.verbose:
            print("\n=== Component Balance Optimization ===")
            print(f"Optimal G-computation weight: {optimal_alpha:.4f}")
            print(f"Optimal IPW weight: {1 - optimal_alpha:.4f}")
            std_var = self._optimization_diagnostics["standard_estimator_variance"]
            opt_var = self._optimization_diagnostics["optimized_estimator_variance"]
            variance_reduction = std_var - opt_var
            rel_reduction = (variance_reduction / std_var * 100) if std_var > 0 else 0.0
            print(
                f"Estimator variance reduction: {variance_reduction:.6f} ({rel_reduction:.2f}%)"
            )
            print(f"Standard estimator variance: {std_var:.6f}")
            print(f"Optimized estimator variance: {opt_var:.6f}")

        return optimal_alpha

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
            treatment_values = np.asarray(treatment.values.values)
        else:
            treatment_values = np.asarray(treatment.values)

        if isinstance(outcome.values, pd.Series):
            outcome_values = np.asarray(outcome.values.values)
        else:
            outcome_values = np.asarray(outcome.values)

        mu_0 = self._cross_fit_predictions["mu_0"]
        mu_1 = self._cross_fit_predictions["mu_1"]
        propensity_scores = self._cross_fit_predictions["propensity_scores"]

        # Ensure propensity scores are still within safe bounds for AIPW computation
        propensity_scores = self._ensure_propensity_score_bounds(propensity_scores)

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
        }

        # Standard AIPW (always computed for reference)
        standard_aipw = g_comp_component + ipw_correction
        self._aipw_components["full_aipw_standard"] = standard_aipw

        # Optimize component balance if enabled
        if self.optimize_component_balance:
            if self.verbose:
                print(
                    "\n⚠️  WARNING: Component optimization may affect double robustness property."
                )
                print(
                    "   Standard AIPW is consistent if either outcome or propensity model is correct."
                )
                print(
                    "   With optimization, bias may increase if both models are misspecified."
                )

            optimal_alpha = self._optimize_component_balance(
                g_comp_component, ipw_correction
            )

            # Use optimized weights
            optimized_aipw = (
                optimal_alpha * g_comp_component + (1 - optimal_alpha) * ipw_correction
            )
            self._aipw_components["full_aipw_optimized"] = optimized_aipw
            self._aipw_components["full_aipw"] = optimized_aipw
            aipw_estimate = float(np.mean(optimized_aipw))
        else:
            # Standard AIPW formula (equal weights)
            self._aipw_components["full_aipw"] = standard_aipw
            aipw_estimate = float(np.mean(standard_aipw))

        return aipw_estimate

    def _compute_influence_function_se(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
    ) -> Optional[float]:
        """Compute standard error using influence functions.

        Note:
            Influence function standard errors are not valid with component
            optimization because they don't account for the data-driven weight
            selection. Use bootstrap confidence intervals instead.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data

        Returns:
            Standard error estimate

        Raises:
            EstimationError: If component optimization is enabled and influence
                function SEs are requested
        """
        if not self.influence_function_se:
            return None

        # Critical: Influence function SEs are invalid with optimization
        if self.optimize_component_balance:
            raise EstimationError(
                "Influence function standard errors are not valid with component "
                "optimization because they don't account for data-driven weight selection. "
                "This would lead to anti-conservative confidence intervals. "
                "Use bootstrap_config with n_samples > 0 to compute valid confidence intervals, "
                "or disable optimization (optimize_component_balance=False) to use influence function SEs."
            )

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

        return float(standard_error)

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Optional[CovariateData] = None,
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
            print(f"Influence function SE: {self.influence_function_se}")

        # Fit models (with or without cross-fitting)
        if self.cross_fitting:
            self._fit_cross_fitting(treatment, outcome, covariates)
        else:
            self._fit_no_cross_fitting(treatment, outcome, covariates)

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
    ) -> Optional[tuple[float, float, NDArray[Any]]]:
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
                    optimize_component_balance=self.optimize_component_balance,
                    component_variance_penalty=self.component_variance_penalty,
                    component_weight_bounds=self.component_weight_bounds,
                    influence_function_se=False,  # Required: influence function SE incompatible with optimization (see __init__ validation)
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
                outcome_values = np.asarray(self.outcome_data.values.values)
            else:
                outcome_values = np.asarray(self.outcome_data.values)

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

    def get_influence_functions(self) -> Optional[NDArray[Any]]:
        """Get influence function values for each observation.

        Returns:
            Array of influence function values if computed, None otherwise
        """
        return self._influence_functions

    def predict_potential_outcomes(
        self,
        treatment_values: pd.Series | NDArray[Any],
        covariates: pd.DataFrame | Optional[NDArray[Any]] = None,
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
