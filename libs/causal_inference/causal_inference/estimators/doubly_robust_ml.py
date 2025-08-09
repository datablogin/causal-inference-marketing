"""Doubly Robust Machine Learning estimator for causal inference.

This module implements doubly robust causal inference methods that combine
machine learning for nuisance parameter estimation with cross-fitting to
achieve √n-consistency and efficient inference.
"""
# ruff: noqa: N803

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from ..ml.cross_fitting import CrossFittingEstimator, ParallelCrossFittingConfig
from ..ml.super_learner import SuperLearner
from .orthogonal_moments import MomentFunctionType, OrthogonalMoments

__all__ = ["DoublyRobustMLEstimator"]


class DoublyRobustMLEstimator(CrossFittingEstimator, BaseEstimator):
    """Doubly Robust Machine Learning estimator for causal inference.

    This estimator implements the Double/Debiased Machine Learning (DML) approach
    of Chernozhukov et al. (2018), which provides:
    - √n-consistent and asymptotically normal estimates
    - Robustness to model misspecification (doubly robust)
    - Cross-fitting to handle overfitting bias from ML methods
    - Efficient influence function-based inference

    The estimator supports multiple treatment types:

    **Binary Treatments**: Traditional ATE between treated (A=1) and control (A=0) groups.

    **Categorical Treatments**: For K categories, ATE compares extreme categories
    (highest vs lowest category values). Use predict_potential_outcomes() for
    pairwise comparisons between specific categories.

    **Continuous Treatments**: ATE represents the average marginal effect across
    the dose range. Use the dose-response function in diagnostics for detailed
    dose-effect relationships.

    Attributes:
        outcome_learner: Machine learning model for outcome regression
        propensity_learner: Machine learning model for propensity score
        cross_fitting: Whether to use cross-fitting
        moment_function: Type of moment function ('aipw' or 'orthogonal')
    """

    def __init__(
        self,
        outcome_learner: SuperLearner | Any = None,
        propensity_learner: SuperLearner | Any = None,
        cross_fitting: bool = True,
        cv_folds: int = 5,
        moment_function: MomentFunctionType = "aipw",
        regularization: bool = True,
        stratified: bool = True,
        compute_diagnostics: bool = True,
        propensity_clip_bounds: tuple[float, float] = (0.01, 0.99),
        performance_config: ParallelCrossFittingConfig | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the Doubly Robust ML estimator.

        Args:
            outcome_learner: ML model for outcome regression E[Y|A,X]
            propensity_learner: ML model for propensity score P(A=1|X)
            cross_fitting: Whether to use cross-fitting for bias reduction
            cv_folds: Number of cross-validation folds
            moment_function: Type of moment function ('aipw', 'orthogonal', 'partialling_out', 'interactive_iv', 'plr', 'pliv', 'auto')
            regularization: Whether to use regularized versions of learners
            stratified: Whether to use stratified cross-validation
            compute_diagnostics: Whether to compute diagnostic information
            propensity_clip_bounds: Bounds for clipping propensity scores
            performance_config: Configuration for parallel processing and performance optimization
            random_state: Random seed for reproducibility
            verbose: Whether to print verbose output
        """
        # Set up performance configuration with sensible defaults for DML
        if performance_config is None:
            performance_config = ParallelCrossFittingConfig(
                n_jobs=-1,  # Use all available cores by default
                parallel_backend="threading",  # Threading is often better for ML models
                max_memory_gb=4.0,
                chunk_size=10000,
                enable_caching=True,
                timeout_per_fold_minutes=10.0,
                enable_gc_per_fold=True,
                memory_monitoring=True,
            )

        super().__init__(
            cv_folds=cv_folds if cross_fitting else 1,
            stratified=stratified,
            parallel_config=performance_config,
            random_state=random_state,
            verbose=verbose,
        )

        # Default learners if not provided
        if outcome_learner is None:
            base_learners = [
                "linear_regression",
                "ridge",
                "lasso",
                "random_forest",
                "gradient_boosting",
            ]
            if regularization:
                base_learners = ["ridge", "lasso", "random_forest"]

            outcome_learner = SuperLearner(
                base_learners=base_learners, task_type="auto"
            )

        if propensity_learner is None:
            base_learners = [
                "logistic_regression",
                "ridge_logistic",
                "lasso_logistic",
                "random_forest",
                "gradient_boosting",
            ]
            if regularization:
                base_learners = ["ridge_logistic", "lasso_logistic", "random_forest"]

            propensity_learner = SuperLearner(
                base_learners=base_learners, task_type="classification"
            )

        self.outcome_learner = outcome_learner
        self.propensity_learner = propensity_learner
        self.cross_fitting = cross_fitting
        self.moment_function = moment_function
        self.regularization = regularization
        self.compute_diagnostics = compute_diagnostics
        self.propensity_clip_bounds = propensity_clip_bounds
        self.performance_config = performance_config

        # Validate moment function
        allowed_moments = set(OrthogonalMoments.get_available_methods()) | {"auto"}
        if moment_function not in allowed_moments:
            raise ValueError(f"moment_function must be one of {allowed_moments}")

        # Storage for auto-selection results
        self._moment_selection_results_: dict[str, Any] = {}

        # Storage for fitted models and estimates
        self.outcome_models_: list[Any] = []
        self.propensity_models_: list[Any] = []
        self.propensity_scores_: NDArray[Any] | None = None
        self.outcome_predictions_: dict[str, NDArray[Any]] = {}
        self.influence_function_: NDArray[Any] | None = None

        # Storage for diagnostic data
        self._diagnostic_data: dict[str, Any] = {}
        self._residuals_: dict[str, NDArray[Any]] = {}
        self._fold_performance_: list[dict[str, float]] = []
        self._cross_fit_residuals_: dict[str, list[NDArray[Any]]] = {}

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the Doubly Robust ML estimator to data."""
        if covariates is None:
            raise EstimationError("DoublyRobustMLEstimator requires covariates")

        # Store treatment information for use in estimation
        self.treatment_type_ = treatment.treatment_type
        if treatment.treatment_type == "categorical":
            self.treatment_categories_ = (
                treatment.categories or treatment.get_unique_values()
            )
            self.n_treatment_categories_ = len(self.treatment_categories_)

            # Warn about memory usage for high-cardinality treatments
            if self.n_treatment_categories_ > 10:
                warnings.warn(
                    f"High-cardinality categorical treatment detected ({self.n_treatment_categories_} categories). "
                    f"This will create {self.n_treatment_categories_} separate outcome models, "
                    f"which may consume significant memory. Consider grouping categories or using continuous treatment.",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.n_treatment_categories_ > 50:
                raise EstimationError(
                    f"Too many treatment categories ({self.n_treatment_categories_}). "
                    f"Maximum supported is 50 categories. Consider using continuous treatment or grouping categories."
                )
        elif treatment.treatment_type == "continuous":
            self.treatment_dose_range_ = treatment.dose_range

        # Validate treatment type is supported
        supported_types = {"binary", "categorical", "continuous"}
        if treatment.treatment_type not in supported_types:
            raise EstimationError(
                f"Treatment type '{treatment.treatment_type}' not supported. "
                f"Supported types: {supported_types}"
            )

        # Convert data to numpy arrays
        X = np.array(covariates.values)
        A = np.array(treatment.values)
        Y = np.array(outcome.values)

        if self.cross_fitting:
            # Perform cross-fitting
            self._perform_cross_fitting(X, Y, A)
            # Aggregate residuals from cross-fitting folds
            if self.compute_diagnostics:
                self._aggregate_cross_fit_residuals()
        else:
            # Fit on full data (no cross-fitting)
            self._fit_full_data(X, Y, A)

    def _fit_nuisance_models(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: NDArray[Any] | None = None,
    ) -> dict[str, Any]:
        """Fit nuisance parameter models on training data for different treatment types."""
        if treatment_train is None:
            raise ValueError("Treatment data required for DoublyRobustML")

        models = {}

        # Fit models based on treatment type
        if self.treatment_type_ == "binary":
            models = self._fit_binary_treatment_models(
                X_train, y_train, treatment_train
            )
        elif self.treatment_type_ == "categorical":
            models = self._fit_categorical_treatment_models(
                X_train, y_train, treatment_train
            )
        elif self.treatment_type_ == "continuous":
            models = self._fit_continuous_treatment_models(
                X_train, y_train, treatment_train
            )
        else:
            raise EstimationError(f"Unsupported treatment type: {self.treatment_type_}")

        return models

    def _fit_binary_treatment_models(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: NDArray[Any],
    ) -> dict[str, Any]:
        """Fit models for binary treatment."""
        # Convert to 0/1 encoding if needed
        unique_vals = np.unique(treatment_train)
        if len(unique_vals) != 2:
            raise ValueError(
                f"Binary treatment must have exactly 2 values, got {len(unique_vals)}"
            )

        # Create binary mask (map to 0/1 if needed)
        if not np.array_equal(sorted(unique_vals), [0, 1]):
            treatment_binary = np.where(treatment_train == unique_vals[0], 0, 1)
        else:
            treatment_binary = treatment_train.copy()

        treated_mask = treatment_binary == 1
        control_mask = treatment_binary == 0

        # Outcome model for treated units: E[Y|A=1,X]
        outcome_model_treated = None
        if np.sum(treated_mask) > 0:
            try:
                outcome_model_treated = clone(self.outcome_learner)
                outcome_model_treated.fit(X_train[treated_mask], y_train[treated_mask])
            except (ValueError, RuntimeError) as e:
                warnings.warn(
                    f"Failed to fit outcome model for treated units: {str(e)}. "
                    f"Using zero predictions for treated outcome model.",
                    UserWarning,
                    stacklevel=2,
                )

        # Outcome model for control units: E[Y|A=0,X]
        outcome_model_control = None
        if np.sum(control_mask) > 0:
            try:
                outcome_model_control = clone(self.outcome_learner)
                outcome_model_control.fit(X_train[control_mask], y_train[control_mask])
            except (ValueError, RuntimeError) as e:
                warnings.warn(
                    f"Failed to fit outcome model for control units: {str(e)}. "
                    f"Using zero predictions for control outcome model.",
                    UserWarning,
                    stacklevel=2,
                )

        # Combined outcome model: E[Y|A,X]
        X_with_treatment = np.column_stack([X_train, treatment_binary])
        outcome_model_combined = clone(self.outcome_learner)
        outcome_model_combined.fit(X_with_treatment, y_train)

        # Propensity score model: P(A=1|X)
        propensity_model = clone(self.propensity_learner)
        propensity_model.fit(X_train, treatment_binary)

        return {
            "outcome_model_treated": outcome_model_treated,
            "outcome_model_control": outcome_model_control,
            "outcome_model_combined": outcome_model_combined,
            "propensity_model": propensity_model,
            "treatment_encoding": treatment_binary,
        }

    def _fit_categorical_treatment_models(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: NDArray[Any],
    ) -> dict[str, Any]:
        """Fit models for categorical (multi-valued) treatment."""
        # Get treatment categories
        unique_treatments = np.unique(treatment_train)
        n_categories = len(unique_treatments)

        if n_categories < 2:
            raise ValueError("Categorical treatment must have at least 2 categories")

        # Fit separate outcome models for each treatment category
        outcome_models_by_category = {}
        for treatment_val in unique_treatments:
            mask = treatment_train == treatment_val
            if np.sum(mask) > 0:
                try:
                    model = clone(self.outcome_learner)
                    model.fit(X_train[mask], y_train[mask])
                    outcome_models_by_category[treatment_val] = model
                except (ValueError, RuntimeError) as e:
                    warnings.warn(
                        f"Failed to fit outcome model for treatment {treatment_val}: {str(e)}. "
                        f"Using zero predictions for this treatment level.",
                        UserWarning,
                        stacklevel=2,
                    )
                    outcome_models_by_category[treatment_val] = None

        # Combined outcome model with treatment as feature
        # Use label encoding for treatment
        treatment_encoded = np.searchsorted(unique_treatments, treatment_train)
        X_with_treatment = np.column_stack([X_train, treatment_encoded])
        outcome_model_combined = clone(self.outcome_learner)
        outcome_model_combined.fit(X_with_treatment, y_train)

        # Multinomial propensity score model: P(A=k|X) for k in categories
        propensity_model = clone(self.propensity_learner)
        # For categorical treatment, propensity learner should be classification-capable
        propensity_model.fit(X_train, treatment_train)

        return {
            "outcome_models_by_category": outcome_models_by_category,
            "outcome_model_combined": outcome_model_combined,
            "propensity_model": propensity_model,
            "treatment_categories": unique_treatments,
            "treatment_encoding": treatment_encoded,
        }

    def _fit_continuous_treatment_models(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: NDArray[Any],
    ) -> dict[str, Any]:
        """Fit models for continuous treatment."""
        # Validate that treatment is indeed continuous/numeric
        try:
            treatment_numeric = pd.to_numeric(treatment_train, errors="coerce")
            if np.any(pd.isna(treatment_numeric)):
                raise ValueError("Continuous treatment contains non-numeric values")
        except (ValueError, TypeError) as e:
            raise EstimationError(f"Invalid continuous treatment values: {e}")

        # For continuous treatments, we need dose-response functions
        # Combined outcome model: E[Y|D=d,X] where D is dose
        X_with_treatment = np.column_stack([X_train, treatment_numeric])
        outcome_model_combined = clone(self.outcome_learner)
        outcome_model_combined.fit(X_with_treatment, y_train)

        # Propensity density model: p(D|X) - density estimation for continuous treatment
        # For simplicity, we'll use a regression approach to model the density
        # In practice, you might want kernel density estimation or other methods
        propensity_model = clone(self.outcome_learner)  # Use regression for density
        propensity_model.fit(X_train, treatment_numeric)

        # Store treatment statistics for dose-response functions
        treatment_min, treatment_max = (
            float(np.min(treatment_numeric)),
            float(np.max(treatment_numeric)),
        )
        treatment_mean, treatment_std = (
            float(np.mean(treatment_numeric)),
            float(np.std(treatment_numeric)),
        )

        return {
            "outcome_model_combined": outcome_model_combined,
            "propensity_model": propensity_model,  # Really a treatment prediction model
            "treatment_range": (treatment_min, treatment_max),
            "treatment_stats": {"mean": treatment_mean, "std": treatment_std},
            "treatment_values": treatment_numeric,
        }

    def _predict_nuisance_parameters(
        self,
        models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: NDArray[Any] | None = None,
        y_val: NDArray[Any] | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Predict nuisance parameters on validation data for different treatment types."""
        if treatment_val is None:
            raise ValueError("Treatment data required for DoublyRobustML predictions")

        # Dispatch based on treatment type
        if self.treatment_type_ == "binary":
            return self._predict_binary_parameters(models, X_val, treatment_val, y_val)
        elif self.treatment_type_ == "categorical":
            return self._predict_categorical_parameters(
                models, X_val, treatment_val, y_val
            )
        elif self.treatment_type_ == "continuous":
            return self._predict_continuous_parameters(
                models, X_val, treatment_val, y_val
            )
        else:
            raise EstimationError(f"Unsupported treatment type: {self.treatment_type_}")

    def _predict_binary_parameters(
        self,
        models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: NDArray[Any],
        y_val: NDArray[Any] | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Predict nuisance parameters for binary treatment."""
        predictions = {}

        # Convert treatment to binary if needed
        treatment_binary = models.get("treatment_encoding", treatment_val)
        if treatment_binary is None:
            # Handle encoding on the fly
            unique_vals = np.unique(treatment_val)
            if not np.array_equal(sorted(unique_vals), [0, 1]):
                treatment_binary = np.where(treatment_val == unique_vals[0], 0, 1)
            else:
                treatment_binary = treatment_val

        # Predict potential outcomes μ₁(X) = E[Y|A=1,X] and μ₀(X) = E[Y|A=0,X]
        if models["outcome_model_treated"] is not None:
            predictions["mu1"] = models["outcome_model_treated"].predict(X_val)
        else:
            predictions["mu1"] = np.zeros(len(X_val))

        if models["outcome_model_control"] is not None:
            predictions["mu0"] = models["outcome_model_control"].predict(X_val)
        else:
            predictions["mu0"] = np.zeros(len(X_val))

        # Predict combined outcome model for orthogonal moments
        X_with_treatment_1 = np.column_stack([X_val, np.ones(len(X_val))])
        X_with_treatment_0 = np.column_stack([X_val, np.zeros(len(X_val))])

        predictions["mu1_combined"] = models["outcome_model_combined"].predict(
            X_with_treatment_1
        )
        predictions["mu0_combined"] = models["outcome_model_combined"].predict(
            X_with_treatment_0
        )

        # Predict observed outcome
        X_with_observed_treatment = np.column_stack([X_val, treatment_binary])
        predictions["mu_observed"] = models["outcome_model_combined"].predict(
            X_with_observed_treatment
        )

        # Predict propensity scores P(A=1|X)
        if hasattr(models["propensity_model"], "predict_proba"):
            propensity_scores = models["propensity_model"].predict_proba(X_val)[:, 1]
        else:
            propensity_scores = models["propensity_model"].predict(X_val)

        # Clip propensity scores to ensure overlap
        predictions["propensity_scores"] = np.clip(
            propensity_scores,
            self.propensity_clip_bounds[0],
            self.propensity_clip_bounds[1],
        )

        # Compute residuals for diagnostic purposes if we have observed outcomes
        if y_val is not None and self.compute_diagnostics:
            self._compute_residuals(predictions, treatment_binary, y_val)

        return predictions

    def _predict_categorical_parameters(
        self,
        models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: NDArray[Any],
        y_val: NDArray[Any] | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Predict nuisance parameters for categorical treatment."""
        predictions = {}
        treatment_categories = models["treatment_categories"]

        # Predict outcome for each treatment category
        outcome_predictions_by_category = {}
        for treatment_cat in treatment_categories:
            if models["outcome_models_by_category"].get(treatment_cat) is not None:
                outcome_predictions_by_category[treatment_cat] = models[
                    "outcome_models_by_category"
                ][treatment_cat].predict(X_val)
            else:
                outcome_predictions_by_category[treatment_cat] = np.zeros(len(X_val))

        # Store predictions for all categories (for pairwise comparisons)
        predictions["outcome_by_category"] = outcome_predictions_by_category

        # Predict observed outcome using combined model
        treatment_encoded = np.searchsorted(treatment_categories, treatment_val)
        X_with_observed_treatment = np.column_stack([X_val, treatment_encoded])
        predictions["mu_observed"] = models["outcome_model_combined"].predict(
            X_with_observed_treatment
        )

        # Predict propensity scores P(A=k|X) for each category
        if hasattr(models["propensity_model"], "predict_proba"):
            propensity_probs = models["propensity_model"].predict_proba(X_val)
            predictions["propensity_by_category"] = {}
            for idx, treatment_cat in enumerate(treatment_categories):
                predictions["propensity_by_category"][treatment_cat] = propensity_probs[
                    :, idx
                ]
        else:
            # Fallback for models without predict_proba
            predictions["propensity_by_category"] = {}
            propensity_pred = models["propensity_model"].predict(X_val)
            for treatment_cat in treatment_categories:
                predictions["propensity_by_category"][treatment_cat] = (
                    propensity_pred == treatment_cat
                ).astype(float)

        # Create propensity scores for observed treatments
        propensity_scores = np.zeros(len(X_val))
        for i, treatment_obs in enumerate(treatment_val):
            if treatment_obs in predictions["propensity_by_category"]:
                propensity_scores[i] = predictions["propensity_by_category"][
                    treatment_obs
                ][i]

        # Clip for stability
        predictions["propensity_scores"] = np.clip(
            propensity_scores,
            self.propensity_clip_bounds[0],
            self.propensity_clip_bounds[1],
        )

        # Store treatment categories and encoding for later use
        predictions["treatment_categories"] = treatment_categories
        predictions["treatment_encoding"] = treatment_encoded

        # Compute residuals for diagnostics
        if y_val is not None and self.compute_diagnostics:
            self._compute_residuals(predictions, treatment_val, y_val)

        return predictions

    def _predict_continuous_parameters(
        self,
        models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: NDArray[Any],
        y_val: NDArray[Any] | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Predict nuisance parameters for continuous treatment."""
        predictions = {}

        # Convert treatment to numeric
        treatment_numeric = pd.to_numeric(treatment_val, errors="coerce")
        if np.any(pd.isna(treatment_numeric)):
            raise EstimationError(
                "Continuous treatment contains non-numeric values during prediction"
            )

        # Predict observed outcome using dose-response function E[Y|D=d,X]
        X_with_observed_treatment = np.column_stack([X_val, treatment_numeric])
        predictions["mu_observed"] = models["outcome_model_combined"].predict(
            X_with_observed_treatment
        )

        # For continuous treatments, we need to estimate dose-response at different levels
        # Create a grid of dose values for dose-response function
        dose_min, dose_max = models["treatment_range"]
        dose_grid = np.linspace(
            dose_min, dose_max, num=min(50, int((dose_max - dose_min) * 10) + 1)
        )

        # Predict outcome at each dose level for ADRF (Average Dose-Response Function)
        dose_response_predictions = {}
        for dose in dose_grid:
            X_with_dose = np.column_stack([X_val, np.full(len(X_val), dose)])
            dose_response_predictions[dose] = models["outcome_model_combined"].predict(
                X_with_dose
            )

        predictions["dose_response_function"] = dose_response_predictions
        predictions["dose_grid"] = dose_grid

        # Propensity density estimation p(D|X)
        # For continuous treatment, this is really treatment prediction, not probability
        propensity_pred = models["propensity_model"].predict(X_val)

        # For continuous treatment, we use treatment residuals instead of propensity scores
        # This represents how far the observed treatment is from the predicted treatment
        treatment_residuals = treatment_numeric - propensity_pred
        predictions["treatment_residuals"] = treatment_residuals
        predictions["propensity_scores"] = propensity_pred  # Treatment predictions

        # Store treatment statistics
        predictions["treatment_range"] = models["treatment_range"]
        predictions["treatment_stats"] = models["treatment_stats"]

        # Compute residuals for diagnostics
        if y_val is not None and self.compute_diagnostics:
            # For continuous treatment, we use treatment residuals directly
            outcome_residuals = y_val - predictions["mu_observed"]

            if self.cross_fitting:
                if "outcome" not in self._cross_fit_residuals_:
                    self._cross_fit_residuals_["outcome"] = []
                if "treatment" not in self._cross_fit_residuals_:
                    self._cross_fit_residuals_["treatment"] = []

                self._cross_fit_residuals_["outcome"].append(outcome_residuals)
                self._cross_fit_residuals_["treatment"].append(treatment_residuals)
            else:
                self._residuals_["outcome"] = outcome_residuals
                self._residuals_["treatment"] = treatment_residuals

        return predictions

    def _compute_residuals(
        self,
        predictions: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
    ) -> None:
        """Compute and store residuals for diagnostic analysis."""
        # Outcome residuals: Y - μ(A,X)
        outcome_residuals = outcome - predictions["mu_observed"]

        # Treatment residuals: A - g(X)
        treatment_residuals = treatment - predictions["propensity_scores"]

        # Store residuals - handle cross-fitting case
        if self.cross_fitting:
            # Store fold-wise residuals for cross-fitting aggregation
            if "outcome" not in self._cross_fit_residuals_:
                self._cross_fit_residuals_["outcome"] = []
            if "treatment" not in self._cross_fit_residuals_:
                self._cross_fit_residuals_["treatment"] = []

            self._cross_fit_residuals_["outcome"].append(outcome_residuals)
            self._cross_fit_residuals_["treatment"].append(treatment_residuals)
        else:
            # Store directly for non-cross-fitting case
            self._residuals_["outcome"] = outcome_residuals
            self._residuals_["treatment"] = treatment_residuals

    def _perform_cross_fitting(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        treatment: NDArray[Any] | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Override to handle residual computation during cross-fitting."""
        # Create cross-fitting splits
        self.cross_fit_data_ = self._create_cross_fit_splits(X, y, treatment)

        # Initialize storage for nuisance estimates
        n_samples = X.shape[0]
        nuisance_estimates = {}

        # Perform cross-fitting for each fold
        for fold_idx in range(self.cv_folds):
            # Get training and validation data for this fold
            X_train = self.cross_fit_data_.X_train_folds[fold_idx]
            y_train = self.cross_fit_data_.y_train_folds[fold_idx]
            X_val = self.cross_fit_data_.X_val_folds[fold_idx]
            y_val = self.cross_fit_data_.y_val_folds[fold_idx]

            treatment_train = None
            treatment_val = None
            if (
                treatment is not None
                and self.cross_fit_data_.treatment_train_folds is not None
            ):
                treatment_train = self.cross_fit_data_.treatment_train_folds[fold_idx]
            if (
                treatment is not None
                and self.cross_fit_data_.treatment_val_folds is not None
            ):
                treatment_val = self.cross_fit_data_.treatment_val_folds[fold_idx]

            # Fit nuisance models on training data
            fitted_models = self._fit_nuisance_models(X_train, y_train, treatment_train)

            # Predict nuisance parameters on validation data (with y_val for residuals)
            fold_predictions = self._predict_nuisance_parameters(
                fitted_models, X_val, treatment_val, y_val
            )

            # Store predictions for validation indices
            val_indices = self.cross_fit_data_.val_indices[fold_idx]
            for param_name, predictions in fold_predictions.items():
                if param_name not in nuisance_estimates:
                    nuisance_estimates[param_name] = np.full(n_samples, np.nan)
                nuisance_estimates[param_name][val_indices] = predictions

        # Check that all samples have nuisance estimates
        for param_name, estimates in nuisance_estimates.items():
            if np.any(np.isnan(estimates)):
                raise ValueError(
                    f"Some samples missing nuisance estimates for {param_name}"
                )

        self.nuisance_estimates_ = nuisance_estimates

        # Store propensity scores and outcome predictions for compatibility
        if "propensity_scores" in nuisance_estimates:
            self.propensity_scores_ = nuisance_estimates["propensity_scores"]
        if "mu1" in nuisance_estimates and "mu0" in nuisance_estimates:
            self.outcome_predictions_ = {
                "mu1": nuisance_estimates["mu1"],
                "mu0": nuisance_estimates["mu0"],
            }

        return nuisance_estimates

    def _aggregate_cross_fit_residuals(self) -> None:
        """Aggregate residuals from cross-fitting folds for diagnostic analysis."""
        if not self._cross_fit_residuals_:
            return

        # Aggregate residuals from all folds using the cross-fit indices
        if self.cross_fit_data_ is not None:
            n_samples = sum(
                len(indices) for indices in self.cross_fit_data_.val_indices
            )

            if "outcome" in self._cross_fit_residuals_:
                outcome_residuals_full = np.full(n_samples, np.nan)
                for fold_idx, residuals in enumerate(
                    self._cross_fit_residuals_["outcome"]
                ):
                    val_indices = self.cross_fit_data_.val_indices[fold_idx]
                    outcome_residuals_full[val_indices] = residuals
                self._residuals_["outcome"] = outcome_residuals_full

            if "treatment" in self._cross_fit_residuals_:
                treatment_residuals_full = np.full(n_samples, np.nan)
                for fold_idx, residuals in enumerate(
                    self._cross_fit_residuals_["treatment"]
                ):
                    val_indices = self.cross_fit_data_.val_indices[fold_idx]
                    treatment_residuals_full[val_indices] = residuals
                self._residuals_["treatment"] = treatment_residuals_full

    def _fit_full_data(self, X: NDArray[Any], Y: NDArray[Any], A: NDArray[Any]) -> None:
        """Fit on full data without cross-fitting."""
        # Fit nuisance models
        models = self._fit_nuisance_models(X, Y, A)
        predictions = self._predict_nuisance_parameters(models, X, A, Y)

        # Store models and predictions
        self.outcome_models_ = [models]
        self.propensity_models_ = [models["propensity_model"]]
        self.nuisance_estimates_ = predictions
        self.propensity_scores_ = predictions["propensity_scores"]
        self.outcome_predictions_ = {
            "mu1": predictions["mu1"],
            "mu0": predictions["mu0"],
        }

    def _estimate_target_parameter(
        self,
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
    ) -> float:
        """Estimate the target causal parameter using doubly robust ML."""
        # Handle automatic method selection
        moment_method = self.moment_function
        if moment_method == "auto":
            # Use stored covariate data from fit process
            covariates_array = None
            if hasattr(self, "covariate_data_") and self.covariate_data_ is not None:
                covariates_array = np.array(self.covariate_data_.values)
            elif hasattr(self, "covariate_data") and self.covariate_data is not None:
                # Fallback for backward compatibility
                covariates_array = np.array(self.covariate_data.values)

            moment_method, selection_results = OrthogonalMoments.select_optimal_method(
                nuisance_estimates, treatment, outcome, covariates_array
            )
            self._moment_selection_results_ = selection_results

            if self.verbose:
                print(f"Auto-selected moment function: {moment_method}")
                print(f"Selection rationale: {selection_results['decision_factors']}")

        # Compute orthogonal scores using selected method
        scores = OrthogonalMoments.compute_scores(
            moment_method,
            nuisance_estimates,
            treatment,
            outcome,
            treatment_type=self.treatment_type_,
        )

        # Store influence function for variance estimation
        ate_estimate = np.mean(scores)
        self.influence_function_ = scores - ate_estimate

        return float(ate_estimate)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate the Average Treatment Effect using Doubly Robust ML."""
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        # Get data
        A = np.array(self.treatment_data.values)  # type: ignore
        Y = np.array(self.outcome_data.values)  # type: ignore

        # Estimate ATE using appropriate moment function
        ate = self._estimate_target_parameter(self.nuisance_estimates_, A, Y)

        # Estimate variance using influence function
        if self.influence_function_ is not None:
            ate_var = np.var(self.influence_function_) / len(A)
            ate_se = np.sqrt(ate_var)
        else:
            # Fallback to bootstrap if influence function not available
            ate_se = None

        # Calculate confidence interval
        if ate_se is not None:
            z_score = 1.96  # For 95% CI
            ate_ci_lower = ate - z_score * ate_se
            ate_ci_upper = ate + z_score * ate_se
        else:
            ate_ci_lower = None
            ate_ci_upper = None

        # Calculate potential outcome means based on treatment type
        potential_outcome_treated = None
        potential_outcome_control = None

        if self.treatment_type_ == "binary":
            if "mu1" in self.nuisance_estimates_ and "mu0" in self.nuisance_estimates_:
                potential_outcome_treated = np.mean(self.nuisance_estimates_["mu1"])
                potential_outcome_control = np.mean(self.nuisance_estimates_["mu0"])
            elif "mu1_combined" in self.nuisance_estimates_:
                potential_outcome_treated = np.mean(
                    self.nuisance_estimates_["mu1_combined"]
                )
                potential_outcome_control = np.mean(
                    self.nuisance_estimates_["mu0_combined"]
                )
        elif self.treatment_type_ == "categorical":
            # For categorical treatments, report outcomes by category
            if "outcome_by_category" in self.nuisance_estimates_:
                potential_outcome_by_category = {}
                for cat, outcomes in self.nuisance_estimates_[
                    "outcome_by_category"
                ].items():
                    potential_outcome_by_category[cat] = np.mean(outcomes)
                # For ATE reporting, use first vs last category as reference
                categories = sorted(
                    self.nuisance_estimates_.get("treatment_categories", [])
                )
                if len(categories) >= 2:
                    potential_outcome_control = potential_outcome_by_category.get(
                        categories[0]
                    )
                    potential_outcome_treated = potential_outcome_by_category.get(
                        categories[-1]
                    )
        elif self.treatment_type_ == "continuous":
            # For continuous treatments, report mean outcome across dose range
            if "dose_response_function" in self.nuisance_estimates_:
                dose_responses = self.nuisance_estimates_["dose_response_function"]
                all_responses = []
                for dose_outcomes in dose_responses.values():
                    all_responses.extend(dose_outcomes)
                potential_outcome_treated = (
                    np.mean(all_responses) if all_responses else None
                )

        # Collect comprehensive diagnostics
        diagnostics = {
            "cross_fitting": self.cross_fitting,
            "moment_function": self.moment_function,
            "regularization": self.regularization,
            "cv_folds": self.cv_folds,
        }

        # Add diagnostic computations if enabled
        if self.compute_diagnostics:
            # Add orthogonality check
            try:
                orthogonality_result = self.check_orthogonality()
                diagnostics["orthogonality_check"] = orthogonality_result
            except EstimationError as e:
                diagnostics["orthogonality_check"] = {"error": str(e)}

            # Add enhanced learner performance
            try:
                learner_performance = self.get_learner_performance()
                diagnostics["learner_performance"] = learner_performance
            except EstimationError as e:
                diagnostics["learner_performance"] = {"error": str(e)}

            # Add residual analysis
            try:
                residual_analysis = self.analyze_residuals()
                diagnostics["residual_analysis"] = residual_analysis
            except EstimationError as e:
                diagnostics["residual_analysis"] = {"error": str(e)}

            # Add cross-fitting validation
            try:
                cf_validation = self.validate_cross_fitting()
                diagnostics["cross_fitting_validation"] = cf_validation
            except EstimationError as e:
                diagnostics["cross_fitting_validation"] = {"error": str(e)}

        # Calculate treatment counts based on treatment type
        n_treated, n_control = self._calculate_treatment_counts(A)

        return CausalEffect(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            potential_outcome_treated=potential_outcome_treated,
            potential_outcome_control=potential_outcome_control,
            method=f"DoublyRobustML_{self._moment_selection_results_.get('selected_method', self.moment_function)}_{self.treatment_type_}",
            n_observations=len(A),
            n_treated=n_treated,
            n_control=n_control,
            diagnostics=diagnostics,
        )

    def _calculate_treatment_counts(self, A: NDArray[Any]) -> tuple[int, int]:
        """Calculate treatment and control counts based on treatment type."""
        if self.treatment_type_ == "binary":
            # Convert to binary if needed
            unique_vals = np.unique(A)
            if len(unique_vals) == 2 and not np.array_equal(
                sorted(unique_vals), [0, 1]
            ):
                A_binary = np.where(A == unique_vals[0], 0, 1)
            else:
                A_binary = A
            n_treated = int(np.sum(A_binary == 1))
            n_control = int(np.sum(A_binary == 0))
        elif self.treatment_type_ == "categorical":
            # For categorical, consider the most common category as "control" and others as "treated"
            unique_vals, counts = np.unique(A, return_counts=True)
            most_common_idx = np.argmax(counts)
            n_control = int(counts[most_common_idx])
            n_treated = int(len(A) - n_control)
        else:  # continuous
            # For continuous, all observations are considered "treated" with different doses
            n_treated = len(A)
            n_control = 0

        return n_treated, n_control

    def predict_potential_outcomes(
        self,
        treatment_values: pd.Series | NDArray[Any],
        covariates: pd.DataFrame | NDArray[Any] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Predict potential outcomes Y(0) and Y(1) for given inputs."""
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before prediction")

        if covariates is None:
            raise EstimationError("Covariates required for DoublyRobustML prediction")

        # Convert to numpy arrays
        if isinstance(covariates, pd.DataFrame):
            X = covariates.values
        else:
            X = np.array(covariates)

        # Use the first set of outcome models for prediction
        models = self.outcome_models_[0]

        # Predict potential outcomes
        if (
            models["outcome_model_control"] is not None
            and models["outcome_model_treated"] is not None
        ):
            # Use separate models for treated and control
            Y0_pred = models["outcome_model_control"].predict(X)
            Y1_pred = models["outcome_model_treated"].predict(X)
        else:
            # Use combined model
            X_with_treatment_0 = np.column_stack([X, np.zeros(len(X))])
            X_with_treatment_1 = np.column_stack([X, np.ones(len(X))])

            Y0_pred = models["outcome_model_combined"].predict(X_with_treatment_0)
            Y1_pred = models["outcome_model_combined"].predict(X_with_treatment_1)

        return Y0_pred, Y1_pred

    def get_variable_importance(self) -> pd.DataFrame | None:
        """Get variable importance from ensemble of learners."""
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        importance_dfs = []

        # Get importance from outcome learner
        if hasattr(self.outcome_learner, "get_variable_importance"):
            outcome_importance = self.outcome_learner.get_variable_importance()
            if outcome_importance is not None:
                outcome_importance["learner_type"] = "outcome"
                importance_dfs.append(outcome_importance)

        # Get importance from propensity learner
        if hasattr(self.propensity_learner, "get_variable_importance"):
            propensity_importance = self.propensity_learner.get_variable_importance()
            if propensity_importance is not None:
                propensity_importance["learner_type"] = "propensity"
                importance_dfs.append(propensity_importance)

        if not importance_dfs:
            return None

        # Combine importance from both learners
        combined_importance = pd.concat(importance_dfs, ignore_index=True)

        # Average importance across learner types
        avg_importance = (
            combined_importance.groupby("feature")["importance"]
            .mean()
            .reset_index()
            .sort_values("importance", ascending=False)
        )

        return avg_importance

    def get_influence_function(self) -> NDArray[Any] | None:
        """Get the influence function for statistical inference."""
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        return self.influence_function_

    def check_orthogonality(self) -> dict[str, Any]:
        """Check orthogonality assumption by computing correlation between residuals.

        Returns:
            Dictionary containing:
            - correlation: Pearson correlation between outcome and treatment residuals
            - p_value: P-value for correlation test
            - is_orthogonal: Boolean indicating if assumption is satisfied (|r| < 0.05)
            - interpretation: String explaining the result

        The orthogonality condition requires that outcome residuals Y - μ(A,X) are
        uncorrelated with treatment residuals A - g(X), indicating proper
        confounding adjustment.
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        if "outcome" not in self._residuals_ or "treatment" not in self._residuals_:
            warnings.warn(
                "Residuals not available for orthogonality check. "
                "This may occur with cross-fitting or if diagnostic data was not stored.",
                UserWarning,
            )
            return {
                "correlation": None,
                "p_value": None,
                "is_orthogonal": None,
                "interpretation": "Residuals not available for analysis",
            }

        outcome_residuals = self._residuals_["outcome"]
        treatment_residuals = self._residuals_["treatment"]

        # Compute Pearson correlation
        correlation, p_value = pearsonr(outcome_residuals, treatment_residuals)

        # Check orthogonality threshold
        threshold = 0.05
        is_orthogonal = abs(correlation) < threshold

        if is_orthogonal:
            interpretation = (
                f"Orthogonality assumption satisfied: |correlation| = {abs(correlation):.4f} < {threshold}. "
                "Outcome and treatment residuals are approximately uncorrelated, suggesting "
                "adequate confounding adjustment."
            )
        else:
            interpretation = (
                f"Orthogonality assumption violated: |correlation| = {abs(correlation):.4f} >= {threshold}. "
                "Strong correlation between residuals suggests remaining confounding or "
                "model misspecification. Consider improving nuisance models."
            )

        result = {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "is_orthogonal": bool(is_orthogonal),
            "interpretation": interpretation,
        }

        if not is_orthogonal:
            warnings.warn(
                "Orthogonality assumption violated. "
                f"Correlation between residuals: {correlation:.4f}. "
                "Consider improving nuisance model specifications.",
                UserWarning,
            )

        return result

    def get_learner_performance(self) -> dict[str, Any]:
        """Get comprehensive performance metrics for nuisance models.

        Enhanced version including R² scores and cross-validation metrics.
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        results = {}

        # Get basic SuperLearner performance
        if hasattr(self.outcome_learner, "get_learner_performance"):
            try:
                results["outcome_learner_performance"] = (
                    self.outcome_learner.get_learner_performance()
                )
            except (ValueError, AttributeError):
                results["outcome_learner_performance"] = {
                    "error": "SuperLearner not properly fitted"
                }

            try:
                results["outcome_ensemble_performance"] = (
                    self.outcome_learner.get_ensemble_performance()
                )
            except (ValueError, AttributeError):
                results["outcome_ensemble_performance"] = {
                    "error": "SuperLearner not properly fitted"
                }

        if hasattr(self.propensity_learner, "get_learner_performance"):
            try:
                results["propensity_learner_performance"] = (
                    self.propensity_learner.get_learner_performance()
                )
            except (ValueError, AttributeError):
                results["propensity_learner_performance"] = {
                    "error": "SuperLearner not properly fitted"
                }

            try:
                results["propensity_ensemble_performance"] = (
                    self.propensity_learner.get_ensemble_performance()
                )
            except (ValueError, AttributeError):
                results["propensity_ensemble_performance"] = {
                    "error": "SuperLearner not properly fitted"
                }

        # Add R² scores if we have stored data
        if hasattr(self, "treatment_data") and hasattr(self, "outcome_data"):
            A = np.array(self.treatment_data.values)  # type: ignore
            Y = np.array(self.outcome_data.values)  # type: ignore

            # Compute R² for outcome models
            if "mu_observed" in self.nuisance_estimates_:
                y_pred = self.nuisance_estimates_["mu_observed"]
                outcome_r2 = r2_score(Y, y_pred)
                outcome_mse = mean_squared_error(Y, y_pred)

                results["outcome_model_r2"] = float(outcome_r2)
                results["outcome_model_mse"] = float(outcome_mse)

            # For propensity model R² (using McFadden's pseudo-R²)
            if self.propensity_scores_ is not None:
                # Compute log-likelihood for null model (intercept only)
                p_null = np.mean(A)
                p_null = np.clip(p_null, 1e-8, 1 - 1e-8)  # Avoid log(0)
                ll_null = np.sum(A * np.log(p_null) + (1 - A) * np.log(1 - p_null))

                # Compute log-likelihood for fitted model
                p_fitted = np.clip(
                    self.propensity_scores_,
                    max(1e-8, self.propensity_clip_bounds[0]),
                    min(1 - 1e-8, self.propensity_clip_bounds[1]),
                )
                ll_fitted = np.sum(
                    A * np.log(p_fitted) + (1 - A) * np.log(1 - p_fitted)
                )

                # McFadden's pseudo-R² (more stable than Nagelkerke)
                if ll_null != 0:
                    mcfadden_r2 = 1 - (ll_fitted / ll_null)
                    # Ensure it's in valid range
                    mcfadden_r2 = np.clip(mcfadden_r2, 0, 1)
                else:
                    mcfadden_r2 = 0

                results["propensity_model_pseudo_r2"] = float(mcfadden_r2)

        # Add cross-validation performance if available
        if self._fold_performance_:
            results["cv_fold_performance"] = self._fold_performance_

            # Compute aggregated CV statistics
            if self._fold_performance_:
                outcome_r2_scores = [
                    fold.get("outcome_r2", 0) for fold in self._fold_performance_
                ]
                propensity_r2_scores = [
                    fold.get("propensity_pseudo_r2", 0)
                    for fold in self._fold_performance_
                ]

                results["cv_outcome_r2_mean"] = float(np.mean(outcome_r2_scores))
                results["cv_outcome_r2_std"] = float(np.std(outcome_r2_scores))
                results["cv_propensity_r2_mean"] = float(np.mean(propensity_r2_scores))
                results["cv_propensity_r2_std"] = float(np.std(propensity_r2_scores))

        return results

    def analyze_residuals(self) -> dict[str, Any]:
        """Comprehensive residual analysis for diagnostic purposes.

        Returns:
            Dictionary containing:
            - residual_statistics: Mean, variance, skewness, kurtosis of residuals
            - heteroscedasticity_test: Tests for non-constant variance
            - normality_test: Tests for residual normality
            - correlation_matrix: Correlations between different residual types
            - residual_plots_data: Data for generating diagnostic plots
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        if not self._residuals_:
            return {
                "error": "No residuals available for analysis. "
                "This may occur with cross-fitting or if diagnostic data was not stored."
            }

        results = {}

        # Basic residual statistics
        residual_stats = {}
        for residual_type, residuals in self._residuals_.items():
            stats = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "variance": float(np.var(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "median": float(np.median(residuals)),
                "q25": float(np.percentile(residuals, 25)),
                "q75": float(np.percentile(residuals, 75)),
            }

            # Add skewness and kurtosis if scipy is available
            try:
                from scipy.stats import kurtosis, skew

                stats["skewness"] = float(skew(residuals))
                stats["kurtosis"] = float(kurtosis(residuals))
            except ImportError:
                pass

            residual_stats[residual_type] = stats

        results["residual_statistics"] = residual_stats

        # Test for heteroscedasticity (Breusch-Pagan test)
        if "outcome" in self._residuals_:
            outcome_residuals = self._residuals_["outcome"]
            if "mu_observed" in self.nuisance_estimates_:
                fitted_values = self.nuisance_estimates_["mu_observed"]
                try:
                    # Breusch-Pagan test implementation
                    squared_residuals = outcome_residuals**2
                    n = len(outcome_residuals)

                    # Regression of squared residuals on fitted values
                    X_het = np.column_stack([np.ones(n), fitted_values])

                    # Compute coefficients using normal equations: (X'X)^(-1)X'y
                    XtX_inv = np.linalg.pinv(X_het.T @ X_het)
                    coeffs = XtX_inv @ X_het.T @ squared_residuals

                    # Predicted values from auxiliary regression
                    y_pred_het = X_het @ coeffs

                    # Sum of squares
                    ss_total = np.sum(
                        (squared_residuals - np.mean(squared_residuals)) ** 2
                    )
                    ss_residual = np.sum((squared_residuals - y_pred_het) ** 2)
                    ss_regression = ss_total - ss_residual

                    # R-squared from auxiliary regression
                    r_squared = ss_regression / ss_total if ss_total > 0 else 0

                    # Lagrange Multiplier (LM) test statistic: n * R²
                    lm_statistic = n * r_squared

                    # Chi-square test with 1 degree of freedom
                    from scipy.stats import chi2

                    p_value = 1 - chi2.cdf(lm_statistic, df=1)

                    results["heteroscedasticity_test"] = {
                        "test": "breusch_pagan",
                        "lm_statistic": float(lm_statistic),
                        "p_value": float(p_value),
                        "r_squared_aux": float(r_squared),
                        "is_homoscedastic": bool(p_value > 0.05),
                        "interpretation": (
                            "Homoscedasticity assumption satisfied (fail to reject null)"
                            if p_value > 0.05
                            else "Evidence of heteroscedasticity detected (reject null)"
                        ),
                    }
                except (ValueError, RuntimeError, ImportError):
                    # Fall back to simple correlation test if scipy unavailable
                    squared_residuals = outcome_residuals**2
                    het_corr, het_p = pearsonr(fitted_values, squared_residuals)
                    results["heteroscedasticity_test"] = {
                        "test": "correlation_fallback",
                        "correlation": float(het_corr),
                        "p_value": float(het_p),
                        "is_homoscedastic": bool(abs(het_corr) < 0.1),
                        "interpretation": (
                            "Homoscedasticity assumption satisfied (simple test)"
                            if abs(het_corr) < 0.1
                            else "Evidence of heteroscedasticity detected (simple test)"
                        ),
                        "note": "Using correlation fallback - install scipy for proper Breusch-Pagan test",
                    }

        # Test for residual normality (if scipy available)
        try:
            from scipy.stats import jarque_bera, shapiro

            if "outcome" in self._residuals_:
                outcome_residuals = self._residuals_["outcome"]

                # Shapiro-Wilk test (for smaller samples)
                if len(outcome_residuals) <= 5000:
                    shapiro_stat, shapiro_p = shapiro(outcome_residuals)
                    results["normality_test_shapiro"] = {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": bool(shapiro_p > 0.05),
                    }

                # Jarque-Bera test (for larger samples)
                jb_stat, jb_p = jarque_bera(outcome_residuals)
                results["normality_test_jarque_bera"] = {
                    "statistic": float(jb_stat),
                    "p_value": float(jb_p),
                    "is_normal": bool(jb_p > 0.05),
                }
        except ImportError:
            results["normality_tests"] = {
                "error": "scipy not available for normality tests"
            }

        # Correlation matrix between residual types
        if len(self._residuals_) > 1:
            residual_keys = list(self._residuals_.keys())
            n_residuals = len(residual_keys)
            corr_matrix = np.eye(n_residuals)

            for i, key1 in enumerate(residual_keys):
                for j, key2 in enumerate(residual_keys):
                    if i != j:
                        try:
                            corr, _ = pearsonr(
                                self._residuals_[key1], self._residuals_[key2]
                            )
                            corr_matrix[i, j] = corr
                        except (ValueError, RuntimeError):
                            corr_matrix[i, j] = np.nan

            results["residual_correlation_matrix"] = {
                "matrix": corr_matrix.tolist(),
                "labels": residual_keys,
            }

        # Prepare data for plotting (bins and counts for histograms)
        plot_data = {}
        for residual_type, residuals in self._residuals_.items():
            hist, bin_edges = np.histogram(residuals, bins=30)
            plot_data[f"{residual_type}_histogram"] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        results["residual_plots_data"] = plot_data

        return results

    def validate_cross_fitting(self) -> dict[str, Any]:
        """Validate cross-fitting implementation and effectiveness.

        Returns:
            Dictionary containing:
            - fold_disjoint_check: Whether train/test splits are disjoint
            - prediction_consistency: Consistency of predictions across folds
            - ate_stability: Stability of ATE estimates across folds
            - fold_sample_sizes: Sample sizes in each fold
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        if not self.cross_fitting:
            return {
                "message": "Cross-fitting was not used in this estimation",
                "cross_fitting_enabled": False,
            }

        results = {"cross_fitting_enabled": True}

        # Check if we have fold-level performance data
        if self._fold_performance_:
            results["fold_performance"] = self._fold_performance_

            # Analyze ATE estimate stability across folds
            ate_estimates = []
            sample_sizes = []

            for i, fold_perf in enumerate(self._fold_performance_):
                if "ate_estimate" in fold_perf:
                    ate_estimates.append(fold_perf["ate_estimate"])
                if "sample_size" in fold_perf:
                    sample_sizes.append(fold_perf["sample_size"])

            if ate_estimates:
                ate_mean = np.mean(ate_estimates)
                ate_std = np.std(ate_estimates)
                ate_cv = ate_std / abs(ate_mean) if ate_mean != 0 else np.inf

                results["ate_stability"] = {
                    "mean_ate": float(ate_mean),
                    "std_ate": float(ate_std),
                    "coefficient_of_variation": float(ate_cv),
                    "is_stable": ate_cv < 0.5,  # Coefficient of variation threshold
                    "interpretation": (
                        "ATE estimates are stable across folds"
                        if ate_cv < 0.5
                        else "ATE estimates show high variability across folds"
                    ),
                }

            if sample_sizes:
                results["fold_sample_sizes"] = {
                    "sizes": sample_sizes,
                    "mean_size": float(np.mean(sample_sizes)),
                    "std_size": float(np.std(sample_sizes)),
                    "balanced": max(sample_sizes) / min(sample_sizes)
                    < 1.5,  # Balance threshold
                }

        # General cross-fitting diagnostics
        results["cv_folds"] = self.cv_folds
        results["stratified"] = self.stratified

        # Check for potential issues
        issues = []
        if self._fold_performance_:
            # Check for convergence failures
            failed_folds = sum(
                1 for fold in self._fold_performance_ if fold.get("fit_failed", False)
            )
            if failed_folds > 0:
                issues.append(f"{failed_folds} folds failed to fit properly")

            # Check for extreme performance variations
            outcome_r2_scores = [
                fold.get("outcome_r2", 0) for fold in self._fold_performance_
            ]
            if (
                outcome_r2_scores
                and (max(outcome_r2_scores) - min(outcome_r2_scores)) > 0.5
            ):
                issues.append("Large variation in model performance across folds")

        results["potential_issues"] = issues

        return results

    def get_moment_selection_results(self) -> dict[str, Any]:
        """Get results from automatic moment function selection.

        Returns:
            Dictionary containing selection criteria, decision factors, and data characteristics

        Raises:
            EstimationError: If moment function was not auto-selected
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        if not self._moment_selection_results_:
            raise EstimationError(
                "No moment selection results available. "
                "Use moment_function='auto' to enable automatic selection."
            )

        return self._moment_selection_results_

    def validate_moment_function_choice(self) -> dict[str, Any]:
        """Validate the chosen moment function using orthogonality tests.

        Returns:
            Dictionary containing orthogonality validation results

        Raises:
            EstimationError: If estimator has not been fitted
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        if self.influence_function_ is None:
            raise EstimationError("Influence function not available for validation")

        # Get the actual method used
        actual_method = self._moment_selection_results_.get(
            "selected_method", self.moment_function
        )

        # Validate orthogonality using the influence function (which contains the scores)
        A = np.array(self.treatment_data.values)  # type: ignore
        validation_results = OrthogonalMoments.validate_orthogonality(
            self.influence_function_, self.nuisance_estimates_, A
        )

        validation_results["moment_method"] = actual_method
        validation_results["validation_passed"] = validation_results["is_orthogonal"]

        return validation_results

    def compare_moment_functions(
        self, candidate_methods: list[str] | None = None, cv_folds: int = 3
    ) -> dict[str, Any]:
        """Compare different moment functions using cross-validation.

        Args:
            candidate_methods: List of methods to compare (default: all available)
            cv_folds: Number of cross-validation folds for comparison

        Returns:
            Dictionary containing comparison results and method rankings

        Raises:
            EstimationError: If estimator has not been fitted
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        if candidate_methods is None:
            candidate_methods = OrthogonalMoments.get_available_methods()

        # Get data
        A = np.array(self.treatment_data.values)  # type: ignore
        Y = np.array(self.outcome_data.values)  # type: ignore

        # Perform cross-validation comparison
        comparison_results = OrthogonalMoments.cross_validate_methods(
            candidate_methods, self.nuisance_estimates_, A, Y, cv_folds=cv_folds
        )

        comparison_results["current_method"] = self._moment_selection_results_.get(
            "selected_method", self.moment_function
        )

        return comparison_results

    def select_moment_function(
        self,
        covariates: CovariateData | None = None,
        instrument: NDArray[Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Select optimal moment function for the current data.

        Args:
            covariates: Covariate data for selection criteria
            instrument: Optional instrumental variable

        Returns:
            Tuple of (selected_method, selection_rationale)

        Raises:
            EstimationError: If estimator has not been fitted
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        # Get data
        A = np.array(self.treatment_data.values)  # type: ignore
        Y = np.array(self.outcome_data.values)  # type: ignore

        covariates_array = None
        if covariates is not None:
            covariates_array = np.array(covariates.values)
        elif hasattr(self, "covariate_data") and self.covariate_data is not None:
            covariates_array = np.array(self.covariate_data.values)

        return OrthogonalMoments.select_optimal_method(
            self.nuisance_estimates_, A, Y, covariates_array, instrument
        )
