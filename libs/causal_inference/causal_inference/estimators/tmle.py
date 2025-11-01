"""Targeted Maximum Likelihood Estimation (TMLE) for causal inference.

TMLE is a doubly robust, semi-parametric estimation method that combines
machine learning for nuisance parameter estimation with targeting to
reduce bias for the parameter of interest.
"""
# ruff: noqa: N803

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

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

__all__ = ["TMLEEstimator"]


class TMLEEstimator(CrossFittingEstimator, BaseEstimator):
    """Targeted Maximum Likelihood Estimation (TMLE) for causal inference.

    TMLE is a semi-parametric, doubly robust estimation method that:
    1. Uses machine learning to estimate nuisance parameters (outcome regression and propensity score)
    2. Updates the outcome model using a targeting step to reduce bias
    3. Provides efficient, asymptotically normal estimates of causal effects

    The estimator supports:
    - Cross-fitting to reduce overfitting bias
    - Super Learner ensembles for nuisance parameter estimation
    - Both one-step and iterative TMLE
    - Efficient influence function-based confidence intervals

    Attributes:
        outcome_learner: Learner for outcome regression Q(A,W)
        propensity_learner: Learner for propensity score π(W)
        cross_fitting: Whether to use cross-fitting for bias reduction
        iterative: Whether to use iterative TMLE (vs one-step)
        max_iterations: Maximum iterations for iterative TMLE
        convergence_threshold: Convergence threshold for iterative TMLE
    """

    def __init__(
        self,
        outcome_learner: SuperLearner | Any = None,
        propensity_learner: SuperLearner | Any = None,
        cross_fitting: bool = True,
        cv_folds: int = 5,
        iterative: bool = False,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-6,
        targeted_regularization: bool = True,
        stratified: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the TMLE estimator.

        Args:
            outcome_learner: Learner for outcome regression
            propensity_learner: Learner for propensity score
            cross_fitting: Whether to use cross-fitting
            cv_folds: Number of cross-validation folds
            iterative: Whether to use iterative TMLE
            max_iterations: Maximum iterations for iterative TMLE
            convergence_threshold: Convergence threshold for iterative TMLE
            targeted_regularization: Whether to use targeted regularization
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
            outcome_learner = SuperLearner(
                base_learners=["linear_regression", "lasso", "random_forest"],
                task_type="auto",
            )

        if propensity_learner is None:
            propensity_learner = SuperLearner(
                base_learners=[
                    "logistic_regression",
                    "lasso_logistic",
                    "random_forest",
                ],
                task_type="classification",
            )

        self.outcome_learner = outcome_learner
        self.propensity_learner = propensity_learner
        self.cross_fitting = cross_fitting
        self.iterative = iterative
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.targeted_regularization = targeted_regularization

        # Storage for fitted models and estimates
        self.outcome_models_: list[Any] = []
        self.propensity_models_: list[Any] = []
        self.targeting_models_: list[LogisticRegression] = []
        self.initial_outcome_estimates_: Optional[NDArray[Any]] = None
        self.targeted_outcome_estimates_: Optional[NDArray[Any]] = None
        self.propensity_scores_: Optional[NDArray[Any]] = None
        self.efficient_influence_function_: Optional[NDArray[Any]] = None

        # Convergence tracking for iterative TMLE
        self.convergence_history_: list[dict[str, float]] = []
        self.converged_: bool = False
        self.n_iterations_: int = 0

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Optional[CovariateData] = None,
    ) -> None:
        """Fit the TMLE estimator to data."""
        if covariates is None:
            raise EstimationError("TMLE requires covariates for confounding adjustment")

        # Convert data to numpy arrays
        X = np.array(covariates.values)
        A = np.array(treatment.values)
        Y = np.array(outcome.values)

        if self.cross_fitting:
            # Perform cross-fitting
            self._perform_cross_fitting(X, Y, A)
            # Perform targeting step after cross-fitting
            self.propensity_scores_ = self.nuisance_estimates_["propensity_scores"]
            self._perform_targeting_step(X, Y, A)
        else:
            # Fit on full data (no cross-fitting)
            self._fit_full_data(X, Y, A)

    def _fit_nuisance_models(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: Optional[NDArray[Any]] = None,
    ) -> dict[str, Any]:
        """Fit nuisance parameter models on training data."""
        if treatment_train is None:
            raise ValueError("Treatment data required for TMLE")

        # Fit outcome regression Q(A,W) = E[Y|A,W]
        # Create features including treatment
        X_with_treatment = np.column_stack([X_train, treatment_train])

        outcome_model = clone(self.outcome_learner)
        outcome_model.fit(X_with_treatment, y_train)

        # Fit propensity score π(W) = P(A=1|W)
        propensity_model = clone(self.propensity_learner)
        propensity_model.fit(X_train, treatment_train)

        return {
            "outcome_model": outcome_model,
            "propensity_model": propensity_model,
        }

    def _predict_nuisance_parameters(
        self,
        models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: Optional[NDArray[Any]] = None,
    ) -> dict[str, NDArray[Any]]:
        """Predict nuisance parameters on validation data."""
        if treatment_val is None:
            raise ValueError("Treatment data required for TMLE predictions")

        outcome_model = models["outcome_model"]
        propensity_model = models["propensity_model"]

        # Predict initial outcome under observed treatment
        X_with_treatment = np.column_stack([X_val, treatment_val])
        initial_outcome = outcome_model.predict(X_with_treatment)

        # Predict potential outcomes Q(1,W) and Q(0,W)
        X_with_treatment_1 = np.column_stack([X_val, np.ones(len(X_val))])
        X_with_treatment_0 = np.column_stack([X_val, np.zeros(len(X_val))])

        Q1W = outcome_model.predict(X_with_treatment_1)
        Q0W = outcome_model.predict(X_with_treatment_0)

        # Predict propensity scores
        if hasattr(propensity_model, "predict_proba"):
            propensity_scores = propensity_model.predict_proba(X_val)[:, 1]
        else:
            propensity_scores = propensity_model.predict(X_val)

        # Clip propensity scores to avoid numerical issues
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

        return {
            "initial_outcome": initial_outcome,
            "Q1W": Q1W,
            "Q0W": Q0W,
            "propensity_scores": propensity_scores,
        }

    def _fit_full_data(self, X: NDArray[Any], Y: NDArray[Any], A: NDArray[Any]) -> None:
        """Fit TMLE on full data without cross-fitting."""
        # Fit nuisance models
        models = self._fit_nuisance_models(X, Y, A)
        predictions = self._predict_nuisance_parameters(models, X, A)

        # Store models and predictions
        self.outcome_models_ = [models["outcome_model"]]
        self.propensity_models_ = [models["propensity_model"]]

        # Set nuisance estimates
        self.nuisance_estimates_ = predictions
        self.initial_outcome_estimates_ = predictions["initial_outcome"]
        self.propensity_scores_ = predictions["propensity_scores"]

        # Perform targeting step
        self._perform_targeting_step(X, Y, A)

    def _perform_targeting_step(
        self, X: NDArray[Any], Y: NDArray[Any], A: NDArray[Any]
    ) -> None:
        """Perform the targeting step of TMLE."""
        if self.propensity_scores_ is None:
            raise EstimationError("Propensity scores not estimated")

        # Get initial estimates
        Q1W = self.nuisance_estimates_["Q1W"]
        Q0W = self.nuisance_estimates_["Q0W"]
        g1W = self.propensity_scores_

        # Create clever covariate H(A,W)
        H_AW = A / g1W - (1 - A) / (1 - g1W)

        if self.iterative:
            # Iterative TMLE
            self._iterative_targeting(Y, A, Q1W, Q0W, g1W, H_AW)
        else:
            # One-step TMLE
            self._one_step_targeting(Y, A, Q1W, Q0W, H_AW)

    def _one_step_targeting(
        self,
        Y: NDArray[Any],
        A: NDArray[Any],
        Q1W: NDArray[Any],
        Q0W: NDArray[Any],
        H_AW: NDArray[Any],
    ) -> None:
        """Perform one-step targeting."""
        assert self.propensity_scores_ is not None
        # Create targeted outcome based on observed treatment
        Q_AW = A * Q1W + (1 - A) * Q0W

        # Fit targeting model
        # Use logistic model if outcomes are binary, linear if continuous
        if len(np.unique(Y)) == 2 and set(np.unique(Y)).issubset({0, 1}):
            # Binary outcome - use logistic targeting
            targeting_model = LogisticRegression(fit_intercept=False)
            targeting_model.fit(H_AW.reshape(-1, 1), Y)

            epsilon = targeting_model.coef_[0, 0]

            # Update Q functions
            Q1W_targeted = expit(
                logit(np.clip(Q1W, 0.01, 0.99)) + epsilon / self.propensity_scores_
            )
            Q0W_targeted = expit(
                logit(np.clip(Q0W, 0.01, 0.99))
                - epsilon / (1 - self.propensity_scores_)
            )

        else:
            # Continuous outcome - use linear targeting
            residuals = Y - Q_AW

            # Use OLS for continuous outcomes
            from sklearn.linear_model import LinearRegression

            targeting_model = LinearRegression(fit_intercept=False)
            targeting_model.fit(H_AW.reshape(-1, 1), residuals)

            epsilon = targeting_model.coef_[0]

            # Update Q functions
            Q1W_targeted = Q1W + epsilon / self.propensity_scores_
            Q0W_targeted = Q0W - epsilon / (1 - self.propensity_scores_)

        # Store targeting model and updated estimates
        self.targeting_models_ = [targeting_model]
        self.nuisance_estimates_["Q1W_targeted"] = Q1W_targeted
        self.nuisance_estimates_["Q0W_targeted"] = Q0W_targeted
        self.targeted_outcome_estimates_ = A * Q1W_targeted + (1 - A) * Q0W_targeted

    def _iterative_targeting(
        self,
        Y: NDArray[Any],
        A: NDArray[Any],
        Q1W: NDArray[Any],
        Q0W: NDArray[Any],
        g1W: NDArray[Any],
        H_AW: NDArray[Any],
    ) -> None:
        """Perform iterative targeting until convergence."""
        assert self.propensity_scores_ is not None
        Q1W_current = Q1W.copy()
        Q0W_current = Q0W.copy()

        targeting_models = []
        self.convergence_history_ = []
        self.converged_ = False

        for iteration in range(self.max_iterations):
            # Current targeted outcome
            Q_AW = A * Q1W_current + (1 - A) * Q0W_current

            # Fit targeting model for this iteration
            if len(np.unique(Y)) == 2 and set(np.unique(Y)).issubset({0, 1}):
                # Binary outcome
                targeting_model = LogisticRegression(fit_intercept=False)
                targeting_model.fit(H_AW.reshape(-1, 1), Y)

                epsilon = targeting_model.coef_[0, 0]

                # Update Q functions
                Q1W_new = expit(logit(np.clip(Q1W_current, 0.01, 0.99)) + epsilon / g1W)
                Q0W_new = expit(
                    logit(np.clip(Q0W_current, 0.01, 0.99)) - epsilon / (1 - g1W)
                )

            else:
                # Continuous outcome
                residuals = Y - Q_AW

                from sklearn.linear_model import LinearRegression

                targeting_model = LinearRegression(fit_intercept=False)
                targeting_model.fit(H_AW.reshape(-1, 1), residuals)

                epsilon = targeting_model.coef_[0]

                # Update Q functions
                Q1W_new = Q1W_current + epsilon / g1W
                Q0W_new = Q0W_current - epsilon / (1 - g1W)

            targeting_models.append(targeting_model)

            # Check convergence
            q1_change = np.mean(np.abs(Q1W_new - Q1W_current))
            q0_change = np.mean(np.abs(Q0W_new - Q0W_current))
            max_change = max(q1_change, q0_change)

            # Store convergence history
            convergence_info = {
                "iteration": iteration + 1,
                "q1_change": q1_change,
                "q0_change": q0_change,
                "max_change": max_change,
                "epsilon": epsilon,
            }
            self.convergence_history_.append(convergence_info)

            if self.verbose:
                print(
                    f"Iteration {iteration + 1}: Q1W change = {q1_change:.6f}, Q0W change = {q0_change:.6f}"
                )

            if max_change < self.convergence_threshold:
                self.converged_ = True
                self.n_iterations_ = iteration + 1
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            Q1W_current = Q1W_new
            Q0W_current = Q0W_new

        # Store final iteration count if not converged
        if not self.converged_:
            self.n_iterations_ = self.max_iterations
            if self.verbose:
                print(f"Did not converge after {self.max_iterations} iterations")

        # Store final targeting models and estimates
        self.targeting_models_ = targeting_models
        self.nuisance_estimates_["Q1W_targeted"] = Q1W_current
        self.nuisance_estimates_["Q0W_targeted"] = Q0W_current
        self.targeted_outcome_estimates_ = A * Q1W_current + (1 - A) * Q0W_current

    def _estimate_target_parameter(
        self,
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
    ) -> float:
        """Estimate the target causal parameter (ATE) using TMLE."""
        # Get targeted Q functions
        Q1W_targeted = nuisance_estimates["Q1W_targeted"]
        Q0W_targeted = nuisance_estimates["Q0W_targeted"]

        # TMLE estimate of ATE
        ate_tmle = np.mean(Q1W_targeted - Q0W_targeted)

        return float(ate_tmle)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate the Average Treatment Effect using TMLE."""
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        if self.treatment_data is None or self.outcome_data is None:
            raise EstimationError("Treatment and outcome data must be available")

        # Get data
        A = np.array(self.treatment_data.values)
        Y = np.array(self.outcome_data.values)

        # Estimate ATE
        ate = self._estimate_target_parameter(self.nuisance_estimates_, A, Y)

        # Calculate efficient influence function for variance estimation
        self._calculate_efficient_influence_function(A, Y)

        # Estimate variance using efficient influence function
        if self.efficient_influence_function_ is None:
            raise EstimationError("Efficient influence function not calculated")
        ate_var = np.var(self.efficient_influence_function_) / len(A)
        ate_se = np.sqrt(ate_var)

        # Calculate confidence interval
        z_score = 1.96  # For 95% CI
        ate_ci_lower = ate - z_score * ate_se
        ate_ci_upper = ate + z_score * ate_se

        # Calculate potential outcome means
        Q1W_targeted = self.nuisance_estimates_["Q1W_targeted"]
        Q0W_targeted = self.nuisance_estimates_["Q0W_targeted"]
        potential_outcome_treated = np.mean(Q1W_targeted)
        potential_outcome_control = np.mean(Q0W_targeted)

        return CausalEffect(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            potential_outcome_treated=potential_outcome_treated,
            potential_outcome_control=potential_outcome_control,
            method="TMLE",
            n_observations=len(A),
            n_treated=int(np.sum(A == 1)),
            n_control=int(np.sum(A == 0)),
            diagnostics={
                "cross_fitting": self.cross_fitting,
                "iterative": self.iterative,
                "targeted_regularization": self.targeted_regularization,
                "convergence_achieved": self.converged_ if self.iterative else True,
                "n_iterations": self.n_iterations_ if self.iterative else 1,
                "convergence_threshold": self.convergence_threshold
                if self.iterative
                else None,
                "max_iterations": self.max_iterations if self.iterative else None,
            },
        )

    def _calculate_efficient_influence_function(
        self, A: NDArray[Any], Y: NDArray[Any]
    ) -> None:
        """Calculate the efficient influence function for variance estimation."""
        if self.propensity_scores_ is None:
            raise EstimationError("Propensity scores not estimated")

        # Get nuisance estimates
        Q1W = self.nuisance_estimates_["Q1W_targeted"]
        Q0W = self.nuisance_estimates_["Q0W_targeted"]
        g1W = self.propensity_scores_

        # Calculate efficient influence function components
        # IF = (A/g1W - (1-A)/(1-g1W)) * (Y - Q(A,W)) + Q(1,W) - Q(0,W) - ψ
        Q_AW = A * Q1W + (1 - A) * Q0W
        clever_covariate = A / g1W - (1 - A) / (1 - g1W)
        ate_estimate = np.mean(Q1W - Q0W)

        influence_function = clever_covariate * (Y - Q_AW) + (Q1W - Q0W) - ate_estimate

        self.efficient_influence_function_ = influence_function

    def predict_potential_outcomes(
        self,
        treatment_values: pd.Series | NDArray[Any],
        covariates: pd.DataFrame | Optional[NDArray[Any]] = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Predict potential outcomes Y(0) and Y(1) for given inputs."""
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before prediction")

        if covariates is None:
            raise EstimationError("Covariates required for TMLE prediction")

        # Convert to numpy arrays
        if isinstance(covariates, pd.DataFrame):
            X = covariates.values
        else:
            X = np.array(covariates)

        # Use the first outcome model for prediction (could ensemble across folds)
        outcome_model = self.outcome_models_[0]

        # Predict Q(1,W) and Q(0,W)
        X_with_treatment_1 = np.column_stack([X, np.ones(len(X))])
        X_with_treatment_0 = np.column_stack([X, np.zeros(len(X))])

        Y1_pred = outcome_model.predict(X_with_treatment_1)
        Y0_pred = outcome_model.predict(X_with_treatment_0)

        return Y0_pred, Y1_pred

    def get_super_learner_results(self) -> dict[str, Any]:
        """Get Super Learner performance results."""
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
            results["outcome_learner_weights"] = (
                self.outcome_learner.get_learner_weights()
            )

        # Get propensity learner results
        if hasattr(self.propensity_learner, "get_learner_performance"):
            results["propensity_learner_performance"] = (
                self.propensity_learner.get_learner_performance()
            )
            results["propensity_ensemble_performance"] = (
                self.propensity_learner.get_ensemble_performance()
            )
            results["propensity_learner_weights"] = (
                self.propensity_learner.get_learner_weights()
            )

        return results

    def get_variable_importance(self) -> Optional[pd.DataFrame]:
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

    def get_convergence_info(self) -> Optional[dict[str, Any]]:
        """Get convergence information for iterative TMLE.

        Returns:
            Dictionary with convergence information, or None if not using iterative TMLE
        """
        if not self.iterative or not self.is_fitted:
            return None

        return {
            "converged": self.converged_,
            "n_iterations": self.n_iterations_,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "convergence_history": self.convergence_history_,
        }
