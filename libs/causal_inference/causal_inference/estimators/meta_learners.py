"""Meta-learners for heterogeneous treatment effect (CATE) estimation.

This module implements meta-learning algorithms for estimating conditional
average treatment effects (CATE): T-learner, S-learner, X-learner, and R-learner.
These methods leverage standard supervised learning algorithms to estimate
treatment effects that vary with covariates.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from ..utils.validation import validate_input_dimensions

__all__ = [
    "BaseMetaLearner",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "CATEResult",
]


class CATEResult(CausalEffect):
    """Extended causal effect result for CATE estimation.

    Includes individual-level treatment effects and summary statistics.
    """

    def __init__(
        self,
        ate: float,
        confidence_interval: tuple[float, float],
        cate_estimates: NDArray[Any],
        cate_std: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CATE result.

        Args:
            ate: Average treatment effect
            confidence_interval: 95% confidence interval for ATE
            cate_estimates: Individual CATE estimates
            cate_std: Standard deviation of CATE estimates (optional)
            **kwargs: Additional fields for parent class
        """
        super().__init__(
            ate=ate,
            ate_ci_lower=confidence_interval[0],
            ate_ci_upper=confidence_interval[1],
            **kwargs
        )
        self.cate_estimates = cate_estimates
        self.cate_std = cate_std

    def plot_cate_distribution(
        self,
        ax: Any | None = None,
        bins: int = 30,
        kde: bool = True,
    ) -> Any:
        """Plot distribution of CATE estimates.

        Args:
            ax: Matplotlib axis (created if None)
            bins: Number of histogram bins
            kde: Whether to overlay kernel density estimate

        Returns:
            Matplotlib axis object
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot histogram
        ax.hist(
            self.cate_estimates,
            bins=bins,
            density=True,
            alpha=0.7,
            label="CATE distribution",
        )

        # Overlay KDE if requested
        if kde:
            sns.kdeplot(data=self.cate_estimates, ax=ax, color="red", label="KDE")

        # Add ATE line
        ax.axvline(
            self.ate,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"ATE = {self.ate:.3f}",
        )

        ax.set_xlabel("Conditional Average Treatment Effect")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of CATE Estimates")
        ax.legend()

        return ax


class BaseMetaLearner(BaseEstimator):
    """Base class for meta-learning algorithms.

    Provides common functionality for all meta-learners including
    model management, cross-fitting, and CATE estimation.
    """

    def __init__(
        self,
        base_learner: SklearnBaseEstimator | None = None,
        propensity_learner: SklearnBaseEstimator | None = None,
        n_folds: int = 5,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize base meta-learner.

        Args:
            base_learner: Base ML model for outcome/effect estimation
            propensity_learner: Model for propensity scores (if needed)
            n_folds: Number of cross-fitting folds
            random_state: Random seed
            verbose: Verbosity flag
            **kwargs: Additional arguments for parent class
        """
        super().__init__(random_state=random_state, verbose=verbose, **kwargs)

        # Set default learners if not provided
        self.base_learner = (
            base_learner
            if base_learner is not None
            else RandomForestRegressor(n_estimators=100, random_state=random_state)
        )
        self.propensity_learner = (
            propensity_learner
            if propensity_learner is not None
            else RandomForestClassifier(n_estimators=100, random_state=random_state)
        )

        self.n_folds = n_folds
        self._cate_estimates: NDArray[Any] | None = None
        self._cate_std: NDArray[Any] | None = None

    def _check_is_fitted(self) -> None:
        """Check if the estimator is fitted."""
        if not self.is_fitted:
            raise EstimationError("Estimator is not fitted. Call fit() first.")

    def estimate_cate(
        self,
        x: pd.DataFrame | NDArray[Any],
    ) -> NDArray[Any]:
        """Estimate conditional average treatment effects.

        Args:
            x: Covariate matrix for prediction

        Returns:
            Array of CATE estimates
        """
        self._check_is_fitted()

        # Convert to numpy if needed
        if isinstance(x, pd.DataFrame):
            x_array = x.values
        else:
            x_array = x

        return self._estimate_cate_implementation(x_array)

    @abstractmethod
    def _estimate_cate_implementation(self, x: NDArray[Any]) -> NDArray[Any]:
        """Implementation-specific CATE estimation logic.

        Args:
            x: Covariate matrix

        Returns:
            CATE estimates
        """
        pass

    def _prepare_data(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Prepare and validate data for meta-learners.

        Returns:
            Tuple of (treatment_array, outcome_array, covariate_array)
        """
        # Extract arrays
        T = (
            treatment.values
            if hasattr(treatment.values, "values")
            else treatment.values
        )
        Y = outcome.values if hasattr(outcome.values, "values") else outcome.values

        if covariates is not None:
            X = (
                covariates.values
                if hasattr(covariates.values, "values")
                else covariates.values
            )
        else:
            # Create dummy covariates if none provided
            X = np.ones((len(T), 1))

        # Ensure arrays
        T = np.asarray(T).flatten()
        Y = np.asarray(Y).flatten()
        X = np.asarray(X)

        # Validate dimensions
        validate_input_dimensions(T, Y, X)

        # Ensure binary treatment for meta-learners
        unique_treatments = np.unique(T)
        if len(unique_treatments) != 2:
            raise ValueError(
                f"Meta-learners require binary treatment. "
                f"Found {len(unique_treatments)} treatment values: {unique_treatments}"
            )

        # Ensure treatment is 0/1
        if not np.array_equal(sorted(unique_treatments), [0, 1]):
            # Map to 0/1
            T = (T == unique_treatments[1]).astype(int)

        return T, Y, X


class SLearner(BaseMetaLearner):
    """S-Learner (Single learner) for CATE estimation.

    The S-learner uses a single model to estimate E[Y|X,T] and derives
    CATE as the difference in predictions with T=1 vs T=0.
    """

    def __init__(
        self,
        base_learner: SklearnBaseEstimator | None = None,
        include_propensity: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize S-learner.

        Args:
            base_learner: Base ML model
            include_propensity: Whether to include propensity score as feature
            **kwargs: Additional arguments for parent class
        """
        super().__init__(base_learner=base_learner, **kwargs)
        self.include_propensity = include_propensity
        self._outcome_model: SklearnBaseEstimator | None = None
        self._propensity_model: SklearnBaseEstimator | None = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit S-learner model.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Covariate data
        """
        T, Y, X = self._prepare_data(treatment, outcome, covariates)

        # Optionally estimate propensity scores
        if self.include_propensity:
            self._propensity_model = clone(self.propensity_learner)
            self._propensity_model.fit(X, T)
            propensity = self._propensity_model.predict_proba(X)[:, 1].reshape(-1, 1)
            X_augmented = np.hstack([X, propensity])
        else:
            X_augmented = X

        # Create feature matrix with treatment indicator
        X_with_treatment = np.hstack([X_augmented, T.reshape(-1, 1)])

        # Fit outcome model
        self._outcome_model = clone(self.base_learner)
        self._outcome_model.fit(X_with_treatment, Y)

        # Store training data dimensions
        self._n_features = X.shape[1]

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate average treatment effect."""
        # For S-learner, we need the training data to estimate ATE
        # This is a simplified implementation
        ate = np.mean(self._cate_estimates) if self._cate_estimates is not None else 0.0

        return CATEResult(
            ate=ate,
            confidence_interval=(ate - 1.96 * 0.1, ate + 1.96 * 0.1),  # Placeholder CI
            cate_estimates=self._cate_estimates if self._cate_estimates is not None else np.array([]),
            cate_std=self._cate_std,
            method="S-Learner",
        )

    def _estimate_cate_implementation(self, x: NDArray[Any]) -> NDArray[Any]:
        """Estimate CATE using S-learner approach."""
        if self._outcome_model is None:
            raise EstimationError("Model not fitted. Call fit() first.")

        # Add propensity scores if used during training
        if self.include_propensity and self._propensity_model is not None:
            propensity = self._propensity_model.predict_proba(x)[:, 1].reshape(-1, 1)
            x_augmented = np.hstack([x, propensity])
        else:
            x_augmented = x

        # Predict with T=1 and T=0
        x_treated = np.hstack([x_augmented, np.ones((x.shape[0], 1))])
        x_control = np.hstack([x_augmented, np.zeros((x.shape[0], 1))])

        y1_pred = self._outcome_model.predict(x_treated)
        y0_pred = self._outcome_model.predict(x_control)

        return np.asarray(y1_pred - y0_pred)


class TLearner(BaseMetaLearner):
    """T-Learner (Two learners) for CATE estimation.

    The T-learner fits separate models for treated and control groups,
    then estimates CATE as the difference in predictions.
    """

    def __init__(
        self,
        base_learner: SklearnBaseEstimator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize T-learner.

        Args:
            base_learner: Base ML model
            **kwargs: Additional arguments for parent class
        """
        super().__init__(base_learner=base_learner, **kwargs)
        self._model_treated: SklearnBaseEstimator | None = None
        self._model_control: SklearnBaseEstimator | None = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit T-learner models.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Covariate data
        """
        T, Y, X = self._prepare_data(treatment, outcome, covariates)

        # Split data by treatment
        treated_idx = T == 1
        control_idx = T == 0

        # Fit separate models
        self._model_treated = clone(self.base_learner)
        self._model_control = clone(self.base_learner)

        self._model_treated.fit(X[treated_idx], Y[treated_idx])
        self._model_control.fit(X[control_idx], Y[control_idx])

        # Estimate CATE on training data for ATE calculation
        self._cate_estimates = self._estimate_cate_implementation(X)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate average treatment effect."""
        if self._cate_estimates is None:
            raise EstimationError("CATE not estimated during fitting.")

        ate = np.mean(self._cate_estimates)

        # Simple bootstrap CI (placeholder - should use proper bootstrap)
        se = np.std(self._cate_estimates) / np.sqrt(len(self._cate_estimates))
        ci = (ate - 1.96 * se, ate + 1.96 * se)

        return CATEResult(
            ate=ate,
            confidence_interval=ci,
            cate_estimates=self._cate_estimates,
            method="T-Learner",
        )

    def _estimate_cate_implementation(self, x: NDArray[Any]) -> NDArray[Any]:
        """Estimate CATE using T-learner approach."""
        if self._model_treated is None or self._model_control is None:
            raise EstimationError("Models not fitted. Call fit() first.")

        y1_pred = self._model_treated.predict(x)
        y0_pred = self._model_control.predict(x)

        return np.asarray(y1_pred - y0_pred)


class XLearner(BaseMetaLearner):
    """X-Learner for CATE estimation.

    The X-learner extends T-learner by using propensity scores to
    combine estimates from treated and control models, potentially
    improving performance when treatment groups are imbalanced.
    """

    def __init__(
        self,
        base_learner: SklearnBaseEstimator | None = None,
        propensity_learner: SklearnBaseEstimator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize X-learner.

        Args:
            base_learner: Base ML model for outcomes
            propensity_learner: Model for propensity scores
            **kwargs: Additional arguments for parent class
        """
        super().__init__(
            base_learner=base_learner, propensity_learner=propensity_learner, **kwargs
        )
        # First stage models (same as T-learner)
        self._model_treated: SklearnBaseEstimator | None = None
        self._model_control: SklearnBaseEstimator | None = None
        # Second stage models
        self._tau_treated: SklearnBaseEstimator | None = None
        self._tau_control: SklearnBaseEstimator | None = None
        # Propensity model
        self._propensity_model: SklearnBaseEstimator | None = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit X-learner models.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Covariate data
        """
        T, Y, X = self._prepare_data(treatment, outcome, covariates)

        # Split indices
        treated_idx = T == 1
        control_idx = T == 0

        # Stage 1: Fit outcome models (like T-learner)
        self._model_treated = clone(self.base_learner)
        self._model_control = clone(self.base_learner)

        self._model_treated.fit(X[treated_idx], Y[treated_idx])
        self._model_control.fit(X[control_idx], Y[control_idx])

        # Stage 2: Compute imputed treatment effects
        # For treated units: D^1_i = Y_i - \hat{m}_0(X_i)
        D_treated = Y[treated_idx] - self._model_control.predict(X[treated_idx])

        # For control units: D^0_i = \hat{m}_1(X_i) - Y_i
        D_control = self._model_treated.predict(X[control_idx]) - Y[control_idx]

        # Stage 3: Fit CATE models
        self._tau_treated = clone(self.base_learner)
        self._tau_control = clone(self.base_learner)

        self._tau_treated.fit(X[treated_idx], D_treated)
        self._tau_control.fit(X[control_idx], D_control)

        # Fit propensity model
        self._propensity_model = clone(self.propensity_learner)
        self._propensity_model.fit(X, T)

        # Estimate CATE on training data
        self._cate_estimates = self._estimate_cate_implementation(X)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate average treatment effect."""
        if self._cate_estimates is None:
            raise EstimationError("CATE not estimated during fitting.")

        ate = np.mean(self._cate_estimates)

        # Bootstrap CI (simplified)
        se = np.std(self._cate_estimates) / np.sqrt(len(self._cate_estimates))
        ci = (ate - 1.96 * se, ate + 1.96 * se)

        return CATEResult(
            ate=ate,
            confidence_interval=ci,
            cate_estimates=self._cate_estimates,
            method="X-Learner",
        )

    def _estimate_cate_implementation(self, x: NDArray[Any]) -> NDArray[Any]:
        """Estimate CATE using X-learner approach."""
        if (
            self._tau_treated is None
            or self._tau_control is None
            or self._propensity_model is None
        ):
            raise EstimationError("Models not fitted. Call fit() first.")

        # Get propensity scores
        propensity = self._propensity_model.predict_proba(x)[:, 1]

        # Get CATE estimates from both models
        tau_1 = self._tau_treated.predict(x)
        tau_0 = self._tau_control.predict(x)

        # Combine using propensity weighting
        cate = propensity * tau_0 + (1 - propensity) * tau_1

        return np.asarray(cate)


class RLearner(BaseMetaLearner):
    """R-Learner (Robinson learner) for CATE estimation.

    The R-learner uses a two-stage residualization approach based on
    the Robinson transformation to estimate heterogeneous treatment effects.
    This method is particularly robust to regularization bias.
    """

    def __init__(
        self,
        base_learner: SklearnBaseEstimator | None = None,
        outcome_learner: SklearnBaseEstimator | None = None,
        propensity_learner: SklearnBaseEstimator | None = None,
        regularization_param: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Initialize R-learner.

        Args:
            base_learner: Model for final CATE estimation
            outcome_learner: Model for outcome regression
            propensity_learner: Model for propensity scores
            regularization_param: Regularization parameter for residuals
            **kwargs: Additional arguments for parent class
        """
        super().__init__(
            base_learner=base_learner, propensity_learner=propensity_learner, **kwargs
        )
        self.outcome_learner = (
            outcome_learner
            if outcome_learner is not None
            else RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        )
        self.regularization_param = regularization_param

        self._outcome_model: SklearnBaseEstimator | None = None
        self._propensity_model: SklearnBaseEstimator | None = None
        self._cate_model: SklearnBaseEstimator | None = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit R-learner model.

        Args:
            treatment: Treatment data
            outcome: Outcome data
            covariates: Covariate data
        """
        T, Y, X = self._prepare_data(treatment, outcome, covariates)

        # Use cross-fitting to avoid overfitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Initialize residual arrays
        Y_residuals = np.zeros_like(Y)
        T_residuals = np.zeros_like(T, dtype=float)

        # Cross-fitting for outcome and propensity models
        for train_idx, test_idx in kf.split(X):
            # Fit outcome model
            outcome_model = clone(self.outcome_learner)
            outcome_model.fit(X[train_idx], Y[train_idx])
            Y_residuals[test_idx] = Y[test_idx] - outcome_model.predict(X[test_idx])

            # Fit propensity model
            prop_model = clone(self.propensity_learner)
            prop_model.fit(X[train_idx], T[train_idx])
            propensity = prop_model.predict_proba(X[test_idx])[:, 1]
            T_residuals[test_idx] = T[test_idx] - propensity

        # Fit final models on all data for later predictions
        self._outcome_model = clone(self.outcome_learner)
        self._outcome_model.fit(X, Y)

        self._propensity_model = clone(self.propensity_learner)
        self._propensity_model.fit(X, T)

        # Fit CATE model using weighted regression
        # Weight by (T - e(X))^2 to handle heteroskedasticity
        weights = T_residuals**2 + self.regularization_param

        # Create weighted dataset
        # R-learner minimizes: sum_i (Y_res_i - tau(X_i) * T_res_i)^2 / weight_i
        # This is equivalent to regressing Y_res/T_res on X with weights

        # Avoid division by zero
        valid_idx = np.abs(T_residuals) > 1e-6

        if np.sum(valid_idx) < 10:
            raise EstimationError("Not enough variation in treatment residuals.")

        # Target for regression
        target = Y_residuals[valid_idx] / T_residuals[valid_idx]
        features = X[valid_idx]
        sample_weights = weights[valid_idx]

        # Fit CATE model
        self._cate_model = clone(self.base_learner)

        # Check if model supports sample weights
        if hasattr(self._cate_model, "fit"):
            try:
                self._cate_model.fit(features, target, sample_weight=sample_weights)
            except TypeError:
                # Model doesn't support sample weights, fit without
                self._cate_model.fit(features, target)
        else:
            self._cate_model.fit(features, target)

        # Estimate CATE on training data
        self._cate_estimates = self._estimate_cate_implementation(X)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate average treatment effect."""
        if self._cate_estimates is None:
            raise EstimationError("CATE not estimated during fitting.")

        ate = np.mean(self._cate_estimates)

        # Bootstrap CI (simplified)
        se = np.std(self._cate_estimates) / np.sqrt(len(self._cate_estimates))
        ci = (ate - 1.96 * se, ate + 1.96 * se)

        return CATEResult(
            ate=ate,
            confidence_interval=ci,
            cate_estimates=self._cate_estimates,
            method="R-Learner",
        )

    def _estimate_cate_implementation(self, x: NDArray[Any]) -> NDArray[Any]:
        """Estimate CATE using R-learner approach."""
        if self._cate_model is None:
            raise EstimationError("Model not fitted. Call fit() first.")

        return np.asarray(self._cate_model.predict(x))
