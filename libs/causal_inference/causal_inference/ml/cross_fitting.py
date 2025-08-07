"""Cross-fitting infrastructure for bias reduction in causal inference.

Cross-fitting (also called cross-validation or sample splitting) is used to
reduce overfitting bias when using machine learning methods for nuisance
parameter estimation in causal inference.
"""
# ruff: noqa: N803

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold

__all__ = ["CrossFitData", "CrossFittingEstimator", "create_cross_fit_data"]


@dataclass
class CrossFitData:
    """Data structure for cross-fitting splits.

    Contains training and validation indices for each fold,
    along with corresponding data splits.
    """

    n_folds: int
    train_indices: list[NDArray[Any]]
    val_indices: list[NDArray[Any]]
    X_train_folds: list[NDArray[Any]]
    X_val_folds: list[NDArray[Any]]
    y_train_folds: list[NDArray[Any]]
    y_val_folds: list[NDArray[Any]]
    treatment_train_folds: list[NDArray[Any]] | None = None
    treatment_val_folds: list[NDArray[Any]] | None = None


class NuisanceLearner(Protocol):
    """Protocol for nuisance parameter learners used in cross-fitting."""

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> None:
        """Fit the learner to training data."""
        ...

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Make predictions on new data."""
        ...


class CrossFittingEstimator(ABC):
    """Abstract base class for estimators that use cross-fitting.

    Cross-fitting is essential for machine learning-based causal inference
    methods to avoid overfitting bias in nuisance parameter estimation.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        stratified: bool = True,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize cross-fitting estimator.

        Args:
            cv_folds: Number of cross-validation folds
            stratified: Whether to use stratified cross-validation
            random_state: Random state for reproducibility
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.cv_folds = cv_folds
        self.stratified = stratified
        self.random_state = random_state

        # Storage for cross-fitting results
        self.cross_fit_data_: CrossFitData | None = None
        self.nuisance_estimates_: dict[str, NDArray[Any]] = {}

    @abstractmethod
    def _fit_nuisance_models(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: NDArray[Any] | None = None,
    ) -> dict[str, Any]:
        """Fit nuisance parameter models on training data.

        Args:
            X_train: Training covariates
            y_train: Training outcomes
            treatment_train: Training treatments (if needed)

        Returns:
            Dictionary of fitted nuisance models
        """
        pass

    @abstractmethod
    def _predict_nuisance_parameters(
        self,
        models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: NDArray[Any] | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Predict nuisance parameters on validation data.

        Args:
            models: Fitted nuisance models
            X_val: Validation covariates
            treatment_val: Validation treatments (if needed)

        Returns:
            Dictionary of nuisance parameter predictions
        """
        pass

    @abstractmethod
    def _estimate_target_parameter(
        self,
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
    ) -> float:
        """Estimate the target causal parameter using nuisance estimates.

        Args:
            nuisance_estimates: Cross-fitted nuisance parameter estimates
            treatment: Treatment assignments
            outcome: Outcomes

        Returns:
            Estimated target parameter (e.g., ATE)
        """
        pass

    def _create_cross_fit_splits(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        treatment: NDArray[Any] | None = None,
    ) -> CrossFitData:
        """Create cross-fitting data splits.

        Args:
            X: Covariate matrix
            y: Outcome vector
            treatment: Treatment vector (for stratified splitting)

        Returns:
            CrossFitData object with splits
        """
        # Create cross-validation splitter
        if self.stratified and treatment is not None:
            # Stratify on treatment assignment
            cv_splitter = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(cv_splitter.split(X, treatment))
        else:
            cv_splitter = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(cv_splitter.split(X))

        # Extract indices and data for each fold
        train_indices = []
        val_indices = []
        X_train_folds = []
        X_val_folds = []
        y_train_folds = []
        y_val_folds = []
        treatment_train_folds = []
        treatment_val_folds = []

        for train_idx, val_idx in splits:
            train_indices.append(train_idx)
            val_indices.append(val_idx)

            X_train_folds.append(X[train_idx])
            X_val_folds.append(X[val_idx])
            y_train_folds.append(y[train_idx])
            y_val_folds.append(y[val_idx])

            if treatment is not None:
                treatment_train_folds.append(treatment[train_idx])
                treatment_val_folds.append(treatment[val_idx])

        return CrossFitData(
            n_folds=self.cv_folds,
            train_indices=train_indices,
            val_indices=val_indices,
            X_train_folds=X_train_folds,
            X_val_folds=X_val_folds,
            y_train_folds=y_train_folds,
            y_val_folds=y_val_folds,
            treatment_train_folds=treatment_train_folds
            if treatment is not None
            else None,
            treatment_val_folds=treatment_val_folds if treatment is not None else None,
        )

    def _perform_cross_fitting(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        treatment: NDArray[Any] | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Perform cross-fitting to estimate nuisance parameters.

        Args:
            X: Covariate matrix
            y: Outcome vector
            treatment: Treatment vector

        Returns:
            Dictionary of cross-fitted nuisance parameter estimates
        """
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

            # Predict nuisance parameters on validation data
            fold_predictions = self._predict_nuisance_parameters(
                fitted_models, X_val, treatment_val
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
        return nuisance_estimates

    def get_cross_validation_performance(self) -> dict[str, Any]:
        """Get cross-validation performance metrics.

        Returns:
            Dictionary with performance metrics for each fold
        """
        if self.cross_fit_data_ is None:
            raise ValueError("Must perform cross-fitting first")

        # This is a placeholder - specific implementations should override
        return {
            "n_folds": self.cv_folds,
            "fold_sizes": [
                len(indices) for indices in self.cross_fit_data_.val_indices
            ],
        }


def create_cross_fit_data(
    X: Union[NDArray[Any], pd.DataFrame],
    y: Union[NDArray[Any], pd.Series],
    treatment: Optional[Union[NDArray[Any], pd.Series]] = None,
    cv_folds: int = 5,
    stratified: bool = True,
    random_state: int | None = None,
) -> CrossFitData:
    """Create cross-fitting data splits.

    Args:
        X: Covariate matrix
        y: Outcome vector
        treatment: Treatment vector (for stratified splitting)
        cv_folds: Number of cross-validation folds
        stratified: Whether to use stratified cross-validation
        random_state: Random state for reproducibility

    Returns:
        CrossFitData object with splits
    """
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = np.asarray(y.values)
    if isinstance(treatment, pd.Series):
        treatment = np.asarray(treatment.values)

    X = np.array(X)
    y = np.array(y)
    if treatment is not None:
        treatment = np.array(treatment)

    # Create cross-validation splitter
    if stratified and treatment is not None:
        cv_splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        splits = list(cv_splitter.split(X, treatment))
    else:
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        splits = list(cv_splitter.split(X))

    # Extract indices and data for each fold
    train_indices = []
    val_indices = []
    X_train_folds = []
    X_val_folds = []
    y_train_folds = []
    y_val_folds = []
    treatment_train_folds = []
    treatment_val_folds = []

    for train_idx, val_idx in splits:
        train_indices.append(train_idx)
        val_indices.append(val_idx)

        X_train_folds.append(X[train_idx])
        X_val_folds.append(X[val_idx])
        y_train_folds.append(y[train_idx])
        y_val_folds.append(y[val_idx])

        if treatment is not None:
            treatment_train_folds.append(treatment[train_idx])
            treatment_val_folds.append(treatment[val_idx])

    return CrossFitData(
        n_folds=cv_folds,
        train_indices=train_indices,
        val_indices=val_indices,
        X_train_folds=X_train_folds,
        X_val_folds=X_val_folds,
        y_train_folds=y_train_folds,
        y_val_folds=y_val_folds,
        treatment_train_folds=treatment_train_folds if treatment is not None else None,
        treatment_val_folds=treatment_val_folds if treatment is not None else None,
    )
