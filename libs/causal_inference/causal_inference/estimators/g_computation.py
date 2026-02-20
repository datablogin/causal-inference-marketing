"""G-computation (Standardization) estimator for causal inference.

This module implements the G-computation method, which estimates causal effects
by fitting outcome models and then averaging predicted outcomes under different
treatment scenarios.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error

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
from ..utils.memory_efficient import (
    MemoryMonitor,
    optimize_pandas_dtypes,
)


class GComputationEstimator(OptimizationMixin, BootstrapMixin, BaseEstimator):
    """G-computation (Standardization) estimator for causal inference.

    G-computation estimates causal effects by:
    1. Fitting an outcome model that predicts Y from treatment and covariates
    2. Using the fitted model to predict counterfactual outcomes under different treatments
    3. Averaging these predictions to estimate causal effects

    This method is also known as the G-formula or standardization.

    Attributes:
        outcome_model: The fitted sklearn model for outcome prediction
        model_type: Type of model to use ('linear', 'logistic', 'random_forest')
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        confidence_level: Confidence level for intervals (default 0.95)
    """

    def __init__(
        self,
        model_type: str = "auto",
        model_params: Optional[dict[str, Any]] = None,
        bootstrap_config: Optional[Any] = None,
        optimization_config: Optional[Any] = None,
        # Legacy parameters for backward compatibility
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        verbose: bool = False,
        # Large dataset optimization parameters
        chunk_size: int = 10000,
        memory_efficient: bool = True,
        large_dataset_threshold: int = 100000,
        # Ensemble settings
        use_ensemble: bool = False,
        ensemble_models: Optional[list[str]] = None,
        ensemble_variance_penalty: float = 0.1,
        ensemble_use_cv: bool = True,
        ensemble_cv_folds: int = 5,
        ensemble_model_params: Optional[dict[str, dict[str, Any]]] = None,
        ensemble_min_models: int = 2,
    ) -> None:
        """Initialize the G-computation estimator.

        Args:
            model_type: Model type ('auto', 'linear', 'logistic', 'random_forest')
            model_params: Parameters to pass to the sklearn model (single-model mode only)
            bootstrap_config: Configuration for bootstrap confidence intervals
            optimization_config: Configuration for optimization strategies
            bootstrap_samples: Legacy parameter - number of bootstrap samples (use bootstrap_config instead)
            confidence_level: Legacy parameter - confidence level (use bootstrap_config instead)
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
            chunk_size: Size of chunks for processing large datasets
            memory_efficient: Enable memory optimizations for large datasets
            large_dataset_threshold: Sample size threshold for enabling optimizations
            use_ensemble: Use ensemble of models instead of single model
            ensemble_models: List of model types for ensemble (if use_ensemble=True)
            ensemble_variance_penalty: Penalty on ensemble weight variance
            ensemble_use_cv: Use cross-validation for ensemble weight optimization (super learner approach)
            ensemble_cv_folds: Number of CV folds for ensemble weight optimization
            ensemble_model_params: Per-model hyperparameters for ensemble models.
                Keys are model names (e.g., 'random_forest', 'ridge'), values are
                dicts of parameters for that model. Model-specific params take highest
                priority over defaults.
            ensemble_min_models: Minimum number of models that must fit successfully
                for ensemble mode. If fewer models fit, raises EstimationError.
                Set to 1 to allow single-model fallback.
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

        self.model_type = model_type
        self.model_params = model_params or {}

        # Large dataset optimization parameters
        self.chunk_size = chunk_size
        self.memory_efficient = memory_efficient
        self.large_dataset_threshold = large_dataset_threshold

        # Ensemble settings
        self.use_ensemble = use_ensemble
        self.ensemble_models = ensemble_models or ["linear", "ridge", "random_forest"]
        self.ensemble_variance_penalty = ensemble_variance_penalty
        self.ensemble_use_cv = ensemble_use_cv
        self.ensemble_cv_folds = ensemble_cv_folds
        self.ensemble_model_params = ensemble_model_params or {}
        self.ensemble_min_models = ensemble_min_models

        # Model storage
        self.outcome_model: Optional[SklearnBaseEstimator] = None
        self.ensemble_models_fitted: dict[str, Any] = {}
        self.ensemble_weights: Optional[NDArray[Any]] = None
        self.ensemble_oof_predictions_: Optional[NDArray[Any]] = None
        self._model_features: Optional[list[str]] = None

    def _check_is_fitted(self) -> None:
        """Check if the estimator has been fitted.

        Raises:
            EstimationError: If the estimator is not fitted
        """
        if self.treatment_data is None:
            raise EstimationError("Model must be fitted before prediction/estimation")

        if not self.use_ensemble and self.outcome_model is None:
            raise EstimationError("Model must be fitted before prediction/estimation")

        if self.use_ensemble and not self.ensemble_models_fitted:
            raise EstimationError(
                "Ensemble models must be fitted before prediction/estimation"
            )

    def _create_bootstrap_estimator(
        self, random_state: Optional[int] = None
    ) -> GComputationEstimator:
        """Create a new estimator instance for bootstrap sampling.

        Args:
            random_state: Random state for this bootstrap instance

        Returns:
            New GComputationEstimator instance configured for bootstrap
        """
        # Determine whether to propagate ensemble settings
        propagate = (
            self.use_ensemble
            and self.bootstrap_config is not None
            and getattr(self.bootstrap_config, "propagate_ensemble", True)
        )

        return GComputationEstimator(
            model_type=self.model_type,
            model_params=self.model_params,
            bootstrap_config=BootstrapConfig(n_samples=0),  # No nested bootstrap
            optimization_config=None,  # Disable optimization in bootstrap
            random_state=random_state,
            verbose=False,  # Reduce verbosity in bootstrap
            use_ensemble=propagate,
            ensemble_models=self.ensemble_models if propagate else None,
            ensemble_variance_penalty=self.ensemble_variance_penalty
            if propagate
            else 0.1,
            ensemble_use_cv=False,  # Disable CV in bootstrap sub-estimators
            ensemble_model_params=self.ensemble_model_params if propagate else None,
            ensemble_min_models=self.ensemble_min_models if propagate else 2,
        )

    def _select_model(self, outcome_type: str) -> SklearnBaseEstimator:
        """Select appropriate sklearn model based on outcome type and model_type.

        Args:
            outcome_type: Type of outcome ('continuous', 'binary', 'count')

        Returns:
            Initialized sklearn model
        """
        if self.model_type == "auto":
            if outcome_type == "continuous":
                model_type = "linear"
            elif outcome_type == "binary":
                model_type = "logistic"
            else:  # count
                model_type = "linear"  # Could use Poisson in future
        else:
            model_type = self.model_type

        # Create model based on type
        if model_type == "linear":
            return LinearRegression(**self.model_params)
        elif model_type == "logistic":
            # Add solver and max_iter parameters to handle convergence issues
            default_params = {
                "solver": "liblinear",  # Better for small datasets and binary problems
                "max_iter": 1000,  # Increase max iterations
                "C": 1.0,  # Regularization parameter
            }
            # Merge with user params, giving priority to user params
            merged_params = {**default_params, **self.model_params}
            return LogisticRegression(random_state=self.random_state, **merged_params)
        elif model_type == "random_forest":
            if outcome_type == "continuous":
                return RandomForestRegressor(
                    random_state=self.random_state, **self.model_params
                )
            else:
                return RandomForestClassifier(
                    random_state=self.random_state, **self.model_params
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _fit_ensemble_models(
        self,
        features: pd.DataFrame,
        y: NDArray[Any],
        outcome_type: str,
    ) -> dict[str, Any]:
        """Fit multiple models for ensemble.

        Args:
            features: Feature matrix
            y: Outcome vector
            outcome_type: Type of outcome ('continuous', 'binary')

        Returns:
            Dictionary of fitted models
        """
        models = {}
        self._ensemble_fit_failures: list[str] = []

        for model_name in self.ensemble_models:
            # Get model-specific params (highest priority)
            specific_params = self.ensemble_model_params.get(model_name, {})

            if outcome_type == "continuous":
                if model_name == "linear":
                    model = LinearRegression(**specific_params)
                elif model_name == "ridge":
                    ridge_params = {"alpha": 1.0, **specific_params}
                    model = Ridge(**ridge_params)
                elif model_name == "random_forest":
                    rf_params = {
                        "n_estimators": 100,
                        "random_state": self.random_state,
                        **specific_params,
                    }
                    model = RandomForestRegressor(**rf_params)
                else:
                    continue
            elif outcome_type == "binary":
                if model_name == "linear":
                    # Unregularized logistic regression
                    logistic_params = {
                        "max_iter": 1000,
                        "random_state": self.random_state,
                        **specific_params,
                    }
                    model = LogisticRegression(**logistic_params)
                elif model_name == "ridge":
                    # Regularized logistic regression (L2 penalty)
                    logistic_ridge_params = {
                        "max_iter": 1000,
                        "C": 1.0,  # Inverse of regularization strength
                        "penalty": "l2",
                        "random_state": self.random_state,
                        **specific_params,
                    }
                    model = LogisticRegression(**logistic_ridge_params)
                elif model_name == "random_forest":
                    rf_params = {
                        "n_estimators": 100,
                        "random_state": self.random_state,
                        **specific_params,
                    }
                    model = RandomForestClassifier(**rf_params)
                else:
                    continue
            else:
                continue

            try:
                model.fit(features, y)
                models[model_name] = model
                if self.verbose:
                    print(f"Fitted {model_name} model")
            except Exception as e:
                warning_msg = f"Failed to fit {model_name}: {str(e)}"
                warnings.warn(warning_msg, RuntimeWarning)
                self._ensemble_fit_failures.append(f"{model_name}: {str(e)}")
                if self.verbose:
                    import traceback

                    print(f"{warning_msg}\n{traceback.format_exc()}")

        return models

    def _optimize_ensemble_weights(
        self,
        models: dict[str, Any],
        features: pd.DataFrame,
        y: NDArray[Any],
        oof_predictions: Optional[NDArray[Any]] = None,
    ) -> NDArray[Any]:
        """Optimize ensemble weights with variance penalty.

        Args:
            models: Dictionary of fitted models
            features: Feature matrix
            y: Outcome vector
            oof_predictions: Optional out-of-fold predictions from CV.
                If provided, uses these instead of in-sample predictions
                to avoid overfitting bias in weight optimization.

        Returns:
            Optimized ensemble weights
        """
        from scipy.optimize import minimize

        n_models = len(models)
        model_names = list(models.keys())

        # Get predictions from each model
        if oof_predictions is not None:
            predictions = oof_predictions
        else:
            predictions = np.column_stack(
                [models[name].predict(features) for name in model_names]
            )

        def objective(weights: NDArray[Any]) -> float:
            """MSE with variance penalty."""
            ensemble_pred = predictions @ weights
            mse = float(np.mean((y - ensemble_pred) ** 2))

            # Variance penalty (encourages uniform weights, prevents over-reliance on single model)
            variance_penalty = self.ensemble_variance_penalty * float(np.var(weights))

            return mse + variance_penalty

        # Constraints: weights sum to 1, all non-negative
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]
        bounds = [(0, 1) for _ in range(n_models)]

        # Initial guess: equal weights
        initial_weights = np.ones(n_models) / n_models

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if not result.success and self.verbose:
            print(f"Ensemble optimization warning: {result.message}")

        # Store diagnostics (merge instead of overwriting)
        if not hasattr(self, "_optimization_diagnostics"):
            self._optimization_diagnostics = {}

        self._optimization_diagnostics.update(
            {
                "ensemble_success": result.success,
                "ensemble_objective": result.fun,
                "ensemble_weights": {
                    name: float(w) for name, w in zip(model_names, result.x)
                },
                "ensemble_cv_folds": self.ensemble_cv_folds
                if oof_predictions is not None
                else None,
            }
        )

        return np.asarray(result.x)

    def _check_ensemble_positivity(
        self,
        predictions_treated: NDArray[Any],
        predictions_control: NDArray[Any],
        outcome_type: str,
    ) -> dict[str, Any]:
        """Check for extreme predictions that may indicate positivity violations.

        Ensemble models (especially tree-based) can produce extreme predictions
        in regions with sparse treatment overlap, leading to unreliable causal
        effect estimates.

        Args:
            predictions_treated: Predicted outcomes under treatment
            predictions_control: Predicted outcomes under control
            outcome_type: Type of outcome ('continuous', 'binary')

        Returns:
            Dictionary with positivity diagnostic results
        """
        diagnostics: dict[str, Any] = {
            "positivity_check": True,
            "warnings": [],
        }

        if outcome_type == "binary":
            # For binary outcomes, check for predictions near 0 or 1
            extreme_threshold_low = 0.01
            extreme_threshold_high = 0.99

            for label, preds in [
                ("treated", predictions_treated),
                ("control", predictions_control),
            ]:
                extreme_low = float(np.mean(preds < extreme_threshold_low))
                extreme_high = float(np.mean(preds > extreme_threshold_high))
                extreme_total = extreme_low + extreme_high

                diagnostics[f"{label}_extreme_low_frac"] = extreme_low
                diagnostics[f"{label}_extreme_high_frac"] = extreme_high

                if extreme_total > 0.05:  # More than 5% extreme
                    diagnostics["positivity_check"] = False
                    diagnostics["warnings"].append(
                        f"{extreme_total:.1%} of {label} predictions are extreme "
                        f"(<{extreme_threshold_low} or >{extreme_threshold_high}). "
                        f"This may indicate positivity violations."
                    )

        # For all outcome types, check prediction spread
        effect_estimates = predictions_treated - predictions_control
        diagnostics["effect_range"] = [
            float(np.min(effect_estimates)),
            float(np.max(effect_estimates)),
        ]
        diagnostics["effect_std"] = float(np.std(effect_estimates))

        # Check for extreme individual treatment effects (outliers)
        iqr = float(
            np.percentile(effect_estimates, 75)
            - np.percentile(effect_estimates, 25)
        )
        if iqr > 0:
            outlier_threshold = 3.0 * iqr
            median_effect = float(np.median(effect_estimates))
            n_outliers = int(
                np.sum(np.abs(effect_estimates - median_effect) > outlier_threshold)
            )
            outlier_frac = n_outliers / len(effect_estimates)
            diagnostics["effect_outlier_fraction"] = float(outlier_frac)

            if outlier_frac > 0.05:
                diagnostics["positivity_check"] = False
                diagnostics["warnings"].append(
                    f"{outlier_frac:.1%} of individual treatment effects are outliers "
                    f"(>{outlier_threshold:.2f} from median). Consider trimming or "
                    f"reducing model complexity."
                )

        # Emit warnings
        for warning_msg in diagnostics["warnings"]:
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

        return diagnostics

    def _get_oof_predictions(
        self,
        models_config: dict[str, Any],
        features: pd.DataFrame,
        y: NDArray[Any],
        outcome_type: str,
    ) -> NDArray[Any]:
        """Generate out-of-fold predictions using K-fold CV (super learner approach).

        This implements the key insight from the super learner literature: use
        cross-validated out-of-fold predictions for weight optimization to avoid
        overfitting bias. Models that overfit (e.g., random forest) will not
        receive disproportionately high weight because their OOF predictions
        reflect true generalization performance.

        Args:
            models_config: Dictionary mapping model names to fitted model instances
                (will be cloned to create unfitted copies for each fold)
            features: Feature matrix
            y: Outcome vector
            outcome_type: Type of outcome ('continuous', 'binary')

        Returns:
            Array of shape (n_samples, n_models) with OOF predictions
        """
        from sklearn.base import clone as sklearn_clone
        from sklearn.model_selection import KFold, StratifiedKFold

        n_samples = len(y)
        model_names = list(models_config.keys())
        n_models = len(model_names)

        # Determine number of folds, auto-reduce for small datasets
        n_folds = self.ensemble_cv_folds
        min_samples_per_fold = 5
        if n_samples < n_folds * min_samples_per_fold:
            n_folds = max(2, n_samples // min_samples_per_fold)
            warnings.warn(
                f"Reduced ensemble CV folds from {self.ensemble_cv_folds} to "
                f"{n_folds} due to small dataset size ({n_samples} samples).",
                UserWarning,
                stacklevel=2,
            )

        # Choose fold strategy based on outcome type
        if outcome_type == "binary":
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state
            )
            split_target = y
        else:
            cv = KFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state
            )
            split_target = y

        # Initialize OOF prediction matrix
        oof_predictions = np.full((n_samples, n_models), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(
            cv.split(features, split_target)
        ):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train = y[train_idx]

            for model_idx, model_name in enumerate(model_names):
                try:
                    # Clone the model for this fold (creates unfitted copy)
                    fold_model = sklearn_clone(models_config[model_name])
                    fold_model.fit(X_train, y_train)
                    oof_predictions[val_idx, model_idx] = fold_model.predict(X_val)
                except Exception as e:
                    # Fill with training mean for failed models
                    warnings.warn(
                        f"Model {model_name} failed on fold {fold_idx}: "
                        f"{str(e)}. Using training mean as fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    oof_predictions[val_idx, model_idx] = np.mean(y_train)

        # Check for any remaining NaN values and fill with column mean
        for col in range(n_models):
            nan_mask = np.isnan(oof_predictions[:, col])
            if np.any(nan_mask):
                col_mean = np.nanmean(oof_predictions[:, col])
                oof_predictions[nan_mask, col] = col_mean

        return oof_predictions

    def _prepare_features(
        self,
        treatment: TreatmentData,
        covariates: Optional[CovariateData] = None,
    ) -> pd.DataFrame:
        """Prepare feature matrix for model fitting.

        Args:
            treatment: Treatment data
            covariates: Optional covariate data

        Returns:
            Feature DataFrame with treatment and covariates
        """
        # Start with treatment as first feature
        if isinstance(treatment.values, pd.Series):
            features = pd.DataFrame({treatment.name: treatment.values})
        else:
            features = pd.DataFrame({treatment.name: treatment.values})

        # Add covariates if provided
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                # Use the DataFrame directly and update names if needed
                for col in covariates.values.columns:
                    features[col] = covariates.values[col]
                # Store the actual column names for later use
                if not covariates.names:
                    # Update the covariates object with actual column names
                    covariates.names = list(covariates.values.columns)
            else:
                # Convert array to DataFrame with covariate names
                cov_names = covariates.names or [
                    f"X{i}" for i in range(covariates.values.shape[1])
                ]
                for i, name in enumerate(cov_names):
                    features[name] = covariates.values[:, i]

        return features

    def _prepare_features_efficient(
        self,
        treatment: TreatmentData,
        covariates: Optional[CovariateData] = None,
        n_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Prepare feature matrix with memory optimizations for large datasets.

        Args:
            treatment: Treatment data
            covariates: Optional covariate data
            n_samples: Number of samples (for optimization decisions)

        Returns:
            Optimized feature DataFrame
        """
        if n_samples is None:
            n_samples = len(treatment.values)

        # Start with regular feature preparation
        features = self._prepare_features(treatment, covariates)

        # Apply memory optimizations for large datasets
        if self.memory_efficient and n_samples >= self.large_dataset_threshold:
            if self.verbose:
                memory_before = features.memory_usage(deep=True).sum() / 1024**2
                print(
                    f"Feature matrix memory before optimization: {memory_before:.1f} MB"
                )

            # Optimize data types
            features = optimize_pandas_dtypes(features)

            if self.verbose:
                memory_after = features.memory_usage(deep=True).sum() / 1024**2
                print(
                    f"Feature matrix memory after optimization: {memory_after:.1f} MB"
                )
                print(
                    f"Memory saved: {memory_before - memory_after:.1f} MB ({((memory_before - memory_after) / memory_before * 100):.1f}%)"
                )

        return features

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Optional[CovariateData] = None,
    ) -> None:
        """Fit the outcome model for G-computation.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data for adjustment
        """
        n_samples = len(treatment.values)

        # Use memory monitoring for large datasets (if enabled in bootstrap config)
        monitor_memory = (
            self.memory_efficient
            and n_samples >= self.large_dataset_threshold
            and self.verbose
            and getattr(
                self.bootstrap_config, "enable_memory_monitoring", True
            )  # Default to True for backward compatibility
        )

        with MemoryMonitor("fit_implementation") if monitor_memory else nullcontext():
            # Prepare features with memory optimization
            X = self._prepare_features_efficient(treatment, covariates, n_samples)
            self._model_features = list(X.columns)

            # Prepare outcome
            if isinstance(outcome.values, pd.Series):
                y = np.asarray(outcome.values.values)
            else:
                y = np.asarray(outcome.values)

            # Select and fit model
            if not self.use_ensemble:
                self.outcome_model = self._select_model(outcome.outcome_type)

        try:
            # Check for treatment variation first
            unique_treatments = np.unique(treatment.values)
            if len(unique_treatments) < 2:
                raise EstimationError(
                    f"No treatment variation detected. Treatment values: {unique_treatments}. "
                    "Cannot estimate causal effects without variation in treatment assignment."
                )

            # Ensemble path
            if self.use_ensemble:
                self.ensemble_models_fitted = self._fit_ensemble_models(
                    features=X, y=y, outcome_type=outcome.outcome_type
                )

                # Store fit failure diagnostics
                if not hasattr(self, "_optimization_diagnostics"):
                    self._optimization_diagnostics = {}
                self._optimization_diagnostics["ensemble_fit_failures"] = (
                    self._ensemble_fit_failures
                )
                self._optimization_diagnostics["ensemble_fit_success_rate"] = (
                    len(self.ensemble_models_fitted) / len(self.ensemble_models)
                    if len(self.ensemble_models) > 0
                    else 0.0
                )

                if len(self.ensemble_models_fitted) >= self.ensemble_min_models:
                    # Proceed with ensemble (existing weight optimization logic)
                    if len(self.ensemble_models_fitted) > 1:
                        # Use CV-based OOF predictions if enabled
                        oof_preds = None
                        min_cv_samples = self.ensemble_cv_folds * 5
                        if self.ensemble_use_cv and len(y) >= min_cv_samples:
                            oof_preds = self._get_oof_predictions(
                                models_config=self.ensemble_models_fitted,
                                features=X,
                                y=y,
                                outcome_type=outcome.outcome_type,
                            )
                            self.ensemble_oof_predictions_ = oof_preds

                            # Refit all models on full data after OOF generation
                            self.ensemble_models_fitted = self._fit_ensemble_models(
                                features=X, y=y, outcome_type=outcome.outcome_type
                            )

                            # Update diagnostics after refit
                            self._optimization_diagnostics[
                                "ensemble_fit_failures"
                            ] = self._ensemble_fit_failures
                            self._optimization_diagnostics[
                                "ensemble_fit_success_rate"
                            ] = (
                                len(self.ensemble_models_fitted)
                                / len(self.ensemble_models)
                                if len(self.ensemble_models) > 0
                                else 0.0
                            )
                        elif self.ensemble_use_cv:
                            warnings.warn(
                                f"Dataset too small ({len(y)} samples) for "
                                f"{self.ensemble_cv_folds}-fold CV. "
                                f"Falling back to in-sample weight optimization.",
                                UserWarning,
                                stacklevel=2,
                            )

                        self.ensemble_weights = self._optimize_ensemble_weights(
                            models=self.ensemble_models_fitted,
                            features=X,
                            y=y,
                            oof_predictions=oof_preds,
                        )

                        if self.verbose:
                            print("\n=== Ensemble Weights ===")
                            cv_note = (
                                " (CV-optimized)" if oof_preds is not None else ""
                            )
                            print(f"Weight optimization{cv_note}:")
                            for name, weight in zip(
                                self.ensemble_models_fitted.keys(),
                                self.ensemble_weights,
                            ):
                                print(f"  {name}: {weight:.4f}")
                    else:
                        # Single model met the threshold (ensemble_min_models=1)
                        # Fall back to single model mode
                        warnings.warn(
                            f"Ensemble has only {len(self.ensemble_models_fitted)} "
                            f"model(s) fitted successfully. "
                            f"Falling back to single model mode.",
                            UserWarning,
                            stacklevel=2,
                        )
                        self.use_ensemble = False
                        self.outcome_model = self._select_model(outcome.outcome_type)
                        self.outcome_model.fit(X, y)
                elif len(self.ensemble_models_fitted) > 0:
                    # Below threshold but some models fitted
                    raise EstimationError(
                        f"Ensemble fitting failed: only {len(self.ensemble_models_fitted)}/{len(self.ensemble_models)} "
                        f"models fitted successfully (minimum required: {self.ensemble_min_models}). "
                        f"Failed models may have incompatible parameters or data issues. "
                        f"Set ensemble_min_models=1 to allow single-model fallback, "
                        f"or check ensemble_model_params for compatibility."
                    )
                else:
                    # Zero models fitted
                    raise EstimationError(
                        f"Ensemble fitting failed: no models fitted successfully out of "
                        f"{len(self.ensemble_models)} attempted. Check that your data is "
                        f"compatible with the requested model types."
                    )
            else:
                # Existing single model path
                self.outcome_model.fit(X, y)

            if self.verbose and not self.use_ensemble:
                # Calculate model fit metrics (for single model only)
                y_pred = self.outcome_model.predict(X)
                if outcome.outcome_type == "continuous":
                    mse = mean_squared_error(y, y_pred)
                    print(f"Outcome model MSE: {mse:.4f}")
                elif outcome.outcome_type == "binary":
                    if hasattr(self.outcome_model, "predict_proba"):
                        y_pred_proba = self.outcome_model.predict_proba(X)[:, 1]
                        ll = log_loss(y, y_pred_proba)
                        print(f"Outcome model log-loss: {ll:.4f}")

        except np.linalg.LinAlgError as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["singular", "ill-conditioned"]):
                raise EstimationError(
                    f"Linear algebra error in outcome model: {str(e)}. "
                    "This may be due to multicollinearity or rank deficiency in covariates."
                ) from e
            else:
                raise EstimationError(
                    f"Linear algebra error in outcome model: {str(e)}"
                ) from e
        except ValueError as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["convergence", "separation"]):
                raise EstimationError(
                    f"Model convergence issue: {str(e)}. "
                    "This may be due to perfect separation or numerical instability."
                ) from e
            else:
                raise EstimationError(f"Failed to fit outcome model: {str(e)}") from e
        except Exception as e:
            raise EstimationError(f"Failed to fit outcome model: {str(e)}") from e

    def _predict_counterfactuals(
        self,
        treatment_value: float | int,
        covariates: Optional[CovariateData] = None,
    ) -> NDArray[Any]:
        """Predict counterfactual outcomes for a given treatment value.

        Args:
            treatment_value: Treatment value to set for all units
            covariates: Covariate data (uses fitted data if None)

        Returns:
            Array of predicted counterfactual outcomes
        """
        # Check if model is fitted
        self._check_is_fitted()

        # Use original data if no new covariates provided
        if covariates is None:
            covariates = self.covariate_data
            n_obs = len(self.treatment_data.values)
        else:
            # Use the number of observations in the new covariate data
            if isinstance(covariates.values, pd.DataFrame):
                n_obs = len(covariates.values)
            else:
                n_obs = covariates.values.shape[0]

        # Use chunked prediction for large datasets
        if self.memory_efficient and n_obs >= self.large_dataset_threshold:
            # Record telemetry if enabled
            if getattr(self.bootstrap_config, "enable_telemetry", False):
                from ..core.bootstrap import OptimizationTelemetry

                OptimizationTelemetry.record_optimization("chunked_prediction")
            return self._predict_counterfactuals_chunked(
                treatment_value, covariates, n_obs
            )

        # Regular prediction for smaller datasets
        return self._predict_counterfactuals_regular(treatment_value, covariates, n_obs)

    def _predict_counterfactuals_regular(
        self,
        treatment_value: float | int,
        covariates: Optional[CovariateData],
        n_obs: int,
    ) -> NDArray[Any]:
        """Regular prediction method for smaller datasets."""
        # Check if treatment_data exists
        if self.treatment_data is None:
            raise EstimationError(
                "Treatment data is required for counterfactual prediction"
            )

        # Start with treatment set to specified value
        counterfactual_features = pd.DataFrame(
            {self.treatment_data.name: [treatment_value] * n_obs}
        )

        # Add covariates
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                for col in covariates.values.columns:
                    counterfactual_features[col] = covariates.values[col].values
            else:
                cov_names = covariates.names or [
                    f"X{i}" for i in range(covariates.values.shape[1])
                ]
                for i, name in enumerate(cov_names):
                    counterfactual_features[name] = covariates.values[:, i]

        # Ensure features are in the same order as training
        if self._model_features is not None:
            counterfactual_features = counterfactual_features[self._model_features]

        # Predict counterfactual outcomes
        if self.use_ensemble and self.ensemble_models_fitted:
            # Ensemble prediction
            if self.ensemble_weights is None:
                raise EstimationError(
                    "Ensemble weights must be optimized before prediction"
                )
            predictions = np.column_stack(
                [
                    model.predict(counterfactual_features)
                    for model in self.ensemble_models_fitted.values()
                ]
            )
            return np.asarray(predictions @ self.ensemble_weights)
        else:
            # Single model prediction
            if self.outcome_model is None:
                raise EstimationError("Outcome model must be fitted before prediction")
            return np.asarray(self.outcome_model.predict(counterfactual_features))

    def _predict_counterfactuals_chunked(
        self,
        treatment_value: float | int,
        covariates: Optional[CovariateData],
        n_obs: int,
    ) -> NDArray[Any]:
        """Memory-efficient chunked prediction for large datasets.

        Performance Characteristics:
            - Memory: O(chunk_size * n_features) instead of O(n_obs * n_features)
            - Time: O(n_obs / chunk_size) prediction passes
            - Optimal for: n_obs >> chunk_size (e.g., 1M+ observations)
        """

        def predict_chunk(chunk_slice: slice) -> NDArray[Any]:
            """Predict for a single chunk."""
            chunk_size = chunk_slice.stop - chunk_slice.start

            # Check if treatment_data exists
            if self.treatment_data is None:
                raise EstimationError(
                    "Treatment data is required for chunked prediction"
                )

            # Create chunk features
            chunk_features = pd.DataFrame(
                {self.treatment_data.name: [treatment_value] * chunk_size}
            )

            # Add covariates for this chunk
            if covariates is not None:
                if isinstance(covariates.values, pd.DataFrame):
                    chunk_covariates = covariates.values.iloc[chunk_slice]
                    for col in chunk_covariates.columns:
                        chunk_features[col] = chunk_covariates[col].values
                else:
                    chunk_covariates_values = covariates.values[chunk_slice]
                    cov_names = covariates.names or [
                        f"X{i}" for i in range(chunk_covariates_values.shape[1])
                    ]
                    for i, name in enumerate(cov_names):
                        chunk_features[name] = chunk_covariates_values[:, i]

            # Ensure correct feature order
            if self._model_features is not None:
                chunk_features = chunk_features[self._model_features]

            # Predict using ensemble or single model
            if self.use_ensemble and self.ensemble_models_fitted:
                # Ensemble prediction
                if self.ensemble_weights is None:
                    raise EstimationError(
                        "Ensemble weights must be optimized before prediction"
                    )
                predictions = np.column_stack(
                    [
                        model.predict(chunk_features)
                        for model in self.ensemble_models_fitted.values()
                    ]
                )
                return np.asarray(predictions @ self.ensemble_weights)
            else:
                # Single model prediction
                if self.outcome_model is None:
                    raise EstimationError(
                        "Outcome model must be fitted before prediction"
                    )
                return np.asarray(self.outcome_model.predict(chunk_features))

        # Use chunked operation to predict
        def chunk_operation(chunk_start: int) -> NDArray[Any]:
            chunk_end = min(chunk_start + self.chunk_size, n_obs)
            return predict_chunk(slice(chunk_start, chunk_end))

        # Combine results from all chunks
        chunks = range(0, n_obs, self.chunk_size)
        results = []

        for chunk_start in chunks:
            chunk_result = chunk_operation(chunk_start)
            results.append(chunk_result)

        return np.concatenate(results)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate Average Treatment Effect using G-computation.

        Returns:
            CausalEffect object with ATE estimate and confidence intervals
        """
        # Check if model is fitted
        self._check_is_fitted()

        # Determine treatment values for binary/categorical treatments
        treatment_values: list[Any]
        if self.treatment_data.treatment_type == "binary":
            treatment_values = [0, 1]
        elif self.treatment_data.treatment_type == "categorical":
            if self.treatment_data.categories is None:
                treatment_values = list(np.unique(self.treatment_data.values))
            else:
                treatment_values = list(self.treatment_data.categories)
        else:
            # For continuous treatments, use min/max or some meaningful range
            min_val = np.min(self.treatment_data.values)
            max_val = np.max(self.treatment_data.values)
            treatment_values = [min_val, max_val]

        # Predict counterfactual outcomes
        outcomes = {}
        for treat_val in treatment_values:
            outcomes[treat_val] = self._predict_counterfactuals(treat_val)

        # Calculate ATE (for binary treatment: E[Y(1)] - E[Y(0)])
        if len(treatment_values) == 2:
            y1_mean = np.mean(outcomes[treatment_values[1]])
            y0_mean = np.mean(outcomes[treatment_values[0]])
            ate = y1_mean - y0_mean

            potential_outcome_treated = y1_mean
            potential_outcome_control = y0_mean
        else:
            # For categorical treatments, compare first category to others
            ate = np.mean(outcomes[treatment_values[1]]) - np.mean(
                outcomes[treatment_values[0]]
            )
            potential_outcome_treated = np.mean(outcomes[treatment_values[1]])
            potential_outcome_control = np.mean(outcomes[treatment_values[0]])

        # Run positivity diagnostics for ensemble predictions
        if self.use_ensemble:
            y1_pred = outcomes[treatment_values[1]]
            y0_pred = outcomes[treatment_values[0]]
            self._positivity_diagnostics = self._check_ensemble_positivity(
                predictions_treated=y1_pred,
                predictions_control=y0_pred,
                outcome_type=self.outcome_data.outcome_type,
            )

            # Store in diagnostics
            if not hasattr(self, "_optimization_diagnostics"):
                self._optimization_diagnostics = {}
            self._optimization_diagnostics["positivity"] = (
                self._positivity_diagnostics
            )

        # Calculate analytical standard error approximation when bootstrap is not used
        ate_se = None
        ate_ci_lower = None
        ate_ci_upper = None

        # Provide analytical SE when bootstrap is not used
        # Exception: Don't provide it during explicit "no bootstrap" tests that expect None
        provide_analytical = not self.bootstrap_config or (
            self.bootstrap_config.n_samples == 0
            and not getattr(self, "_disable_analytical_inference", False)
        )
        # Disable analytical SE for ensemble models (not yet supported)
        if provide_analytical:
            if self.use_ensemble:
                warnings.warn(
                    "Analytical standard errors are not supported for ensemble models. "
                    "Use bootstrap_samples > 0 for confidence intervals.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # Simple analytical SE approximation for G-computation
                try:
                    # Predict outcomes for current data to calculate residual variance
                    X = (
                        self.covariate_data.values
                        if self.covariate_data is not None
                        else np.array([]).reshape(len(self.treatment_data.values), 0)
                    )
                    X_with_treatment = (
                        np.column_stack([self.treatment_data.values, X])
                        if X.shape[1] > 0
                        else self.treatment_data.values.reshape(-1, 1)
                    )

                    predicted_outcomes = self.outcome_model.predict(X_with_treatment)
                    residuals = self.outcome_data.values - predicted_outcomes
                    residual_variance = np.var(residuals, ddof=1)

                    # Simple approximation: SE ≈ sqrt(residual_var * (1/n_treated + 1/n_control))
                    n_treated = np.sum(self.treatment_data.values == 1)
                    n_control = np.sum(self.treatment_data.values == 0)

                    if n_treated > 0 and n_control > 0:
                        ate_se = np.sqrt(
                            residual_variance * (1 / n_treated + 1 / n_control)
                        )

                        # Calculate basic confidence intervals using normal approximation
                        import scipy.stats as stats

                        confidence_level = (
                            self.bootstrap_config.confidence_level
                            if self.bootstrap_config
                            else 0.95
                        )
                        z_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                        ate_ci_lower = ate - z_alpha * ate_se
                        ate_ci_upper = ate + z_alpha * ate_se

                except Exception:
                    # If analytical SE calculation fails, continue without it
                    ate_se = None

        # Enhanced bootstrap confidence intervals
        bootstrap_result = None
        bootstrap_estimates = None

        # Additional CI fields for different bootstrap methods
        ate_ci_lower_bca = None
        ate_ci_upper_bca = None
        ate_ci_lower_bias_corrected = None
        ate_ci_upper_bias_corrected = None
        ate_ci_lower_studentized = None
        ate_ci_upper_studentized = None
        bootstrap_method = None
        bootstrap_converged = None
        bootstrap_bias = None
        bootstrap_acceleration = None

        if (
            self.use_ensemble
            and self.bootstrap_config
            and self.bootstrap_config.n_samples > 0
            and getattr(self.bootstrap_config, "propagate_ensemble", True)
        ):
            n_models = (
                len(self.ensemble_models_fitted)
                if self.ensemble_models_fitted
                else len(self.ensemble_models)
            )
            total_fits = n_models * self.bootstrap_config.n_samples
            warnings.warn(
                f"Ensemble + bootstrap will fit {total_fits} models "
                f"({n_models} models x {self.bootstrap_config.n_samples} bootstrap samples). "
                f"Set bootstrap_config.propagate_ensemble=False for faster "
                f"(but less accurate) CIs.",
                UserWarning,
                stacklevel=2,
            )

        if self.bootstrap_config and self.bootstrap_config.n_samples > 0:
            try:
                bootstrap_result = self.compute_bootstrap_confidence_intervals(ate)

                # Extract bootstrap estimates and diagnostics
                bootstrap_estimates = bootstrap_result.bootstrap_estimates
                ate_se = bootstrap_result.bootstrap_se
                bootstrap_method = bootstrap_result.config.method
                bootstrap_converged = bootstrap_result.converged
                bootstrap_bias = bootstrap_result.bias_estimate
                bootstrap_acceleration = bootstrap_result.acceleration_estimate

                # Set confidence intervals based on method
                if bootstrap_result.config.method == "percentile":
                    ate_ci_lower = bootstrap_result.ci_lower_percentile
                    ate_ci_upper = bootstrap_result.ci_upper_percentile
                elif bootstrap_result.config.method == "bias_corrected":
                    ate_ci_lower = bootstrap_result.ci_lower_bias_corrected
                    ate_ci_upper = bootstrap_result.ci_upper_bias_corrected
                elif bootstrap_result.config.method == "bca":
                    ate_ci_lower = bootstrap_result.ci_lower_bca
                    ate_ci_upper = bootstrap_result.ci_upper_bca
                elif bootstrap_result.config.method == "studentized":
                    ate_ci_lower = bootstrap_result.ci_lower_studentized
                    ate_ci_upper = bootstrap_result.ci_upper_studentized

                # Set all available CI methods for comparison
                ate_ci_lower_bca = bootstrap_result.ci_lower_bca
                ate_ci_upper_bca = bootstrap_result.ci_upper_bca
                ate_ci_lower_bias_corrected = bootstrap_result.ci_lower_bias_corrected
                ate_ci_upper_bias_corrected = bootstrap_result.ci_upper_bias_corrected
                ate_ci_lower_studentized = bootstrap_result.ci_lower_studentized
                ate_ci_upper_studentized = bootstrap_result.ci_upper_studentized

            except Exception as e:
                if self.verbose:
                    print(f"Bootstrap confidence intervals failed: {str(e)}")
                # Continue without bootstrap CIs

        # Count treatment/control units
        if self.treatment_data.treatment_type == "binary":
            n_treated = np.sum(self.treatment_data.values == 1)
            n_control = np.sum(self.treatment_data.values == 0)
        else:
            n_treated = len(self.treatment_data.values)
            n_control = 0

        return CausalEffect(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            confidence_level=self.bootstrap_config.confidence_level
            if self.bootstrap_config
            else 0.95,
            potential_outcome_treated=potential_outcome_treated,
            potential_outcome_control=potential_outcome_control,
            method="G-computation",
            n_observations=len(self.treatment_data.values),
            n_treated=n_treated,
            n_control=n_control,
            bootstrap_samples=self.bootstrap_config.n_samples
            if self.bootstrap_config
            else 0,
            bootstrap_estimates=bootstrap_estimates,
            # Enhanced bootstrap confidence intervals
            ate_ci_lower_bca=ate_ci_lower_bca,
            ate_ci_upper_bca=ate_ci_upper_bca,
            ate_ci_lower_bias_corrected=ate_ci_lower_bias_corrected,
            ate_ci_upper_bias_corrected=ate_ci_upper_bias_corrected,
            ate_ci_lower_studentized=ate_ci_lower_studentized,
            ate_ci_upper_studentized=ate_ci_upper_studentized,
            # Bootstrap diagnostics
            bootstrap_method=bootstrap_method,
            bootstrap_converged=bootstrap_converged,
            bootstrap_bias=bootstrap_bias,
            bootstrap_acceleration=bootstrap_acceleration,
        )

    def predict_potential_outcomes(
        self,
        treatment_values: pd.Series | NDArray[Any],
        covariates: pd.DataFrame | Optional[NDArray[Any]] = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Predict potential outcomes Y(0) and Y(1) for given inputs.

        Args:
            treatment_values: Treatment assignment values (not used in G-computation prediction)
            covariates: Covariate values for prediction

        Returns:
            Tuple of (Y0_predictions, Y1_predictions)
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before prediction")

        if self.outcome_model is None:
            raise EstimationError("Outcome model not available")

        # Prepare covariate data with correct feature names
        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                cov_data = CovariateData(
                    values=covariates, names=list(covariates.columns)
                )
            else:
                # Use the same covariate names as the training data
                if self.covariate_data is not None and self.covariate_data.names:
                    cov_names = self.covariate_data.names
                else:
                    # Fall back to generic names matching the number of features in training
                    n_features = (
                        len(self._model_features) - 1
                        if self._model_features
                        else covariates.shape[1]
                    )
                    cov_names = [f"X{i}" for i in range(n_features)]

                # Create DataFrame to ensure correct column ordering
                cov_df = pd.DataFrame(covariates, columns=cov_names)
                cov_data = CovariateData(values=cov_df, names=cov_names)
        else:
            cov_data = None

        # Predict under control (Y(0)) and treatment (Y(1))
        y0_pred = self._predict_counterfactuals(0, cov_data)
        y1_pred = self._predict_counterfactuals(1, cov_data)

        return y0_pred, y1_pred

    @property
    def bootstrap_samples(self) -> int:
        """Number of bootstrap samples for backward compatibility."""
        if self.bootstrap_config:
            return int(self.bootstrap_config.n_samples)
        return 0

    @property
    def confidence_level(self) -> float:
        """Confidence level for backward compatibility."""
        if self.bootstrap_config:
            return float(self.bootstrap_config.confidence_level)
        return 0.95

    def _bootstrap_confidence_interval(
        self,
    ) -> Optional[tuple[float, float, NDArray[Any]]]:
        """Legacy method for backward compatibility with old test API.

        Returns:
            Tuple of (ci_lower, ci_upper, bootstrap_estimates)
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before bootstrap")

        if self.bootstrap_config and self.bootstrap_config.n_samples > 0:
            # First we need to get the point estimate
            effect = self._estimate_ate_implementation()

            # Compute bootstrap confidence intervals
            bootstrap_result = self.compute_bootstrap_confidence_intervals(effect.ate)

            # Return the primary CI method results
            if bootstrap_result.config.method == "percentile":
                ci_lower = bootstrap_result.ci_lower_percentile
                ci_upper = bootstrap_result.ci_upper_percentile
            elif bootstrap_result.config.method == "bias_corrected":
                ci_lower = bootstrap_result.ci_lower_bias_corrected
                ci_upper = bootstrap_result.ci_upper_bias_corrected
            elif bootstrap_result.config.method == "bca":
                ci_lower = bootstrap_result.ci_lower_bca
                ci_upper = bootstrap_result.ci_upper_bca
            elif bootstrap_result.config.method == "studentized":
                ci_lower = bootstrap_result.ci_lower_studentized
                ci_upper = bootstrap_result.ci_upper_studentized
            else:
                ci_lower = bootstrap_result.ci_lower_percentile
                ci_upper = bootstrap_result.ci_upper_percentile

            return ci_lower, ci_upper, bootstrap_result.bootstrap_estimates
        else:
            return None, None, None

    def get_positivity_diagnostics(self) -> dict[str, Any] | None:
        """Get positivity diagnostics from the last ensemble prediction.

        Returns diagnostics about extreme predictions that may indicate
        positivity violations. Only populated when use_ensemble=True.

        Returns:
            Dictionary with positivity diagnostic results, or None
        """
        return getattr(self, "_positivity_diagnostics", None)
