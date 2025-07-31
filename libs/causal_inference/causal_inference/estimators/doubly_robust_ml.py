"""Doubly Robust Machine Learning estimator for causal inference.

This module implements doubly robust causal inference methods that combine
machine learning for nuisance parameter estimation with cross-fitting to
achieve √n-consistency and efficient inference.
"""
# ruff: noqa: N803

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from ..ml.cross_fitting import CrossFittingEstimator
from ..ml.super_learner import SuperLearner

__all__ = ["DoublyRobustMLEstimator"]


class DoublyRobustMLEstimator(CrossFittingEstimator, BaseEstimator):
    """Doubly Robust Machine Learning estimator for causal inference.

    This estimator implements the Double/Debiased Machine Learning (DML) approach
    of Chernozhukov et al. (2018), which provides:
    - √n-consistent and asymptotically normal estimates
    - Robustness to model misspecification (doubly robust)
    - Cross-fitting to handle overfitting bias from ML methods
    - Efficient influence function-based inference

    The estimator supports both AIPW-style and orthogonal moment-based estimation.

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
        moment_function: str = "aipw",
        regularization: bool = True,
        stratified: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the Doubly Robust ML estimator.

        Args:
            outcome_learner: ML model for outcome regression E[Y|A,X]
            propensity_learner: ML model for propensity score P(A=1|X)
            cross_fitting: Whether to use cross-fitting for bias reduction
            cv_folds: Number of cross-validation folds
            moment_function: Type of moment function ('aipw' or 'orthogonal')
            regularization: Whether to use regularized versions of learners
            stratified: Whether to use stratified cross-validation
            random_state: Random seed for reproducibility
            verbose: Whether to print verbose output
        """
        super().__init__(
            cv_folds=cv_folds if cross_fitting else 1,
            stratified=stratified,
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

        # Validate moment function
        allowed_moments = {"aipw", "orthogonal"}
        if moment_function not in allowed_moments:
            raise ValueError(f"moment_function must be one of {allowed_moments}")

        # Storage for fitted models and estimates
        self.outcome_models_: list[Any] = []
        self.propensity_models_: list[Any] = []
        self.propensity_scores_: NDArray[Any] | None = None
        self.outcome_predictions_: dict[str, NDArray[Any]] = {}
        self.influence_function_: NDArray[Any] | None = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the Doubly Robust ML estimator to data."""
        if covariates is None:
            raise EstimationError("DoublyRobustMLEstimator requires covariates")

        # Validate binary treatment for now
        if treatment.treatment_type != "binary":
            raise EstimationError(
                "DoublyRobustMLEstimator currently supports only binary treatments"
            )

        # Convert data to numpy arrays
        X = np.array(covariates.values)
        A = np.array(treatment.values)
        Y = np.array(outcome.values)

        if self.cross_fitting:
            # Perform cross-fitting
            self._perform_cross_fitting(X, Y, A)
        else:
            # Fit on full data (no cross-fitting)
            self._fit_full_data(X, Y, A)

    def _fit_nuisance_models(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: NDArray[Any] | None = None,
    ) -> dict[str, Any]:
        """Fit nuisance parameter models on training data."""
        if treatment_train is None:
            raise ValueError("Treatment data required for DoublyRobustML")

        # Fit outcome regression models
        # Separate models for treated and control units (for AIPW-style estimation)
        treated_mask = treatment_train == 1
        control_mask = treatment_train == 0

        # Outcome model for treated units: E[Y|A=1,X]
        if np.sum(treated_mask) > 0:
            try:
                outcome_model_treated = clone(self.outcome_learner)
                outcome_model_treated.fit(X_train[treated_mask], y_train[treated_mask])
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to fit outcome model for treated units: {str(e)}. "
                    f"Using zero predictions for treated outcome model.",
                    UserWarning,
                    stacklevel=2,
                )
                outcome_model_treated = None
        else:
            import warnings

            warnings.warn(
                "No treated units in training data. Using zero predictions for treated outcome model.",
                UserWarning,
                stacklevel=2,
            )
            outcome_model_treated = None

        # Outcome model for control units: E[Y|A=0,X]
        if np.sum(control_mask) > 0:
            try:
                outcome_model_control = clone(self.outcome_learner)
                outcome_model_control.fit(X_train[control_mask], y_train[control_mask])
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to fit outcome model for control units: {str(e)}. "
                    f"Using zero predictions for control outcome model.",
                    UserWarning,
                    stacklevel=2,
                )
                outcome_model_control = None
        else:
            import warnings

            warnings.warn(
                "No control units in training data. Using zero predictions for control outcome model.",
                UserWarning,
                stacklevel=2,
            )
            outcome_model_control = None

        # Combined outcome model: E[Y|A,X] (for orthogonal moments)
        X_with_treatment = np.column_stack([X_train, treatment_train])
        try:
            outcome_model_combined = clone(self.outcome_learner)
            outcome_model_combined.fit(X_with_treatment, y_train)
        except Exception as e:
            raise EstimationError(f"Failed to fit combined outcome model: {str(e)}")

        # Fit propensity score model: P(A=1|X)
        try:
            propensity_model = clone(self.propensity_learner)
            propensity_model.fit(X_train, treatment_train)
        except Exception as e:
            raise EstimationError(f"Failed to fit propensity score model: {str(e)}")

        return {
            "outcome_model_treated": outcome_model_treated,
            "outcome_model_control": outcome_model_control,
            "outcome_model_combined": outcome_model_combined,
            "propensity_model": propensity_model,
        }

    def _predict_nuisance_parameters(
        self,
        models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: NDArray[Any] | None = None,
    ) -> dict[str, NDArray[Any]]:
        """Predict nuisance parameters on validation data."""
        if treatment_val is None:
            raise ValueError("Treatment data required for DoublyRobustML predictions")

        predictions = {}

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
        X_with_observed_treatment = np.column_stack([X_val, treatment_val])
        predictions["mu_observed"] = models["outcome_model_combined"].predict(
            X_with_observed_treatment
        )

        # Predict propensity scores
        if hasattr(models["propensity_model"], "predict_proba"):
            propensity_scores = models["propensity_model"].predict_proba(X_val)[:, 1]
        else:
            propensity_scores = models["propensity_model"].predict(X_val)

        # Clip propensity scores to ensure overlap
        predictions["propensity_scores"] = np.clip(propensity_scores, 0.01, 0.99)

        return predictions

    def _fit_full_data(self, X: NDArray[Any], Y: NDArray[Any], A: NDArray[Any]) -> None:
        """Fit on full data without cross-fitting."""
        # Fit nuisance models
        models = self._fit_nuisance_models(X, Y, A)
        predictions = self._predict_nuisance_parameters(models, X, A)

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
        if self.moment_function == "aipw":
            return self._estimate_ate_aipw(nuisance_estimates, treatment, outcome)
        elif self.moment_function == "orthogonal":
            return self._estimate_ate_orthogonal(nuisance_estimates, treatment, outcome)
        else:
            raise ValueError(f"Unknown moment function: {self.moment_function}")

    def _estimate_ate_aipw(
        self,
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
    ) -> float:
        """Estimate ATE using AIPW moment function."""
        mu1 = nuisance_estimates["mu1"]
        mu0 = nuisance_estimates["mu0"]
        g = nuisance_estimates["propensity_scores"]

        # AIPW estimator: ψ(Y,A,X) = μ₁(X) - μ₀(X) + A(Y-μ₁(X))/g(X) - (1-A)(Y-μ₀(X))/(1-g(X))
        aipw_scores = (
            mu1
            - mu0
            + treatment * (outcome - mu1) / g
            - (1 - treatment) * (outcome - mu0) / (1 - g)
        )

        # Store influence function for variance estimation
        ate_estimate = np.mean(aipw_scores)
        self.influence_function_ = aipw_scores - ate_estimate

        return float(ate_estimate)

    def _estimate_ate_orthogonal(
        self,
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
    ) -> float:
        """Estimate ATE using orthogonal moment function."""
        mu1 = nuisance_estimates["mu1_combined"]
        mu0 = nuisance_estimates["mu0_combined"]
        mu_observed = nuisance_estimates["mu_observed"]
        g = nuisance_estimates["propensity_scores"]

        # Orthogonal moment function (Neyman-orthogonal)
        # ψ(Y,A,X) = (A-g(X))(Y-μ(A,X)) + μ₁(X) - μ₀(X)
        orthogonal_scores = (treatment - g) * (outcome - mu_observed) + mu1 - mu0

        # Store influence function for variance estimation
        ate_estimate = np.mean(orthogonal_scores)
        self.influence_function_ = orthogonal_scores - ate_estimate

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

        # Calculate potential outcome means
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
        else:
            potential_outcome_treated = None
            potential_outcome_control = None

        return CausalEffect(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            potential_outcome_treated=potential_outcome_treated,
            potential_outcome_control=potential_outcome_control,
            method=f"DoublyRobustML_{self.moment_function}",
            n_observations=len(A),
            n_treated=int(np.sum(A == 1)),
            n_control=int(np.sum(A == 0)),
            diagnostics={
                "cross_fitting": self.cross_fitting,
                "moment_function": self.moment_function,
                "regularization": self.regularization,
                "cv_folds": self.cv_folds,
            },
        )

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

    def get_learner_performance(self) -> dict[str, Any]:
        """Get performance of individual learners."""
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted first")

        results = {}

        # Get outcome learner results
        if hasattr(self.outcome_learner, "get_learner_performance"):
            results["outcome_learner_performance"] = (
                self.outcome_learner.get_learner_performance()
            )
            results["outcome_ensemble_performance"] = (
                self.outcome_learner.get_ensemble_performance()
            )

        # Get propensity learner results
        if hasattr(self.propensity_learner, "get_learner_performance"):
            results["propensity_learner_performance"] = (
                self.propensity_learner.get_learner_performance()
            )
            results["propensity_ensemble_performance"] = (
                self.propensity_learner.get_ensemble_performance()
            )

        return results

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
