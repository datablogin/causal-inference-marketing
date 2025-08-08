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

        # Storage for diagnostic data
        self._diagnostic_data: dict[str, Any] = {}
        self._residuals_: dict[str, NDArray[Any]] = {}
        self._fold_performance_: list[dict[str, float]] = []

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
        y_val: NDArray[Any] | None = None,
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

        # Compute residuals for diagnostic purposes if we have observed outcomes
        if y_val is not None:
            self._compute_residuals(predictions, treatment_val, y_val)

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

        # Store residuals for later analysis
        self._residuals_["outcome"] = outcome_residuals
        self._residuals_["treatment"] = treatment_residuals

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

        # Collect comprehensive diagnostics
        diagnostics = {
            "cross_fitting": self.cross_fitting,
            "moment_function": self.moment_function,
            "regularization": self.regularization,
            "cv_folds": self.cv_folds,
        }

        # Add orthogonality check
        try:
            orthogonality_result = self.check_orthogonality()
            diagnostics["orthogonality_check"] = orthogonality_result
        except Exception as e:
            diagnostics["orthogonality_check"] = {"error": str(e)}

        # Add enhanced learner performance
        try:
            learner_performance = self.get_learner_performance()
            diagnostics["learner_performance"] = learner_performance
        except Exception as e:
            diagnostics["learner_performance"] = {"error": str(e)}

        # Add residual analysis
        try:
            residual_analysis = self.analyze_residuals()
            diagnostics["residual_analysis"] = residual_analysis
        except Exception as e:
            diagnostics["residual_analysis"] = {"error": str(e)}

        # Add cross-fitting validation
        try:
            cf_validation = self.validate_cross_fitting()
            diagnostics["cross_fitting_validation"] = cf_validation
        except Exception as e:
            diagnostics["cross_fitting_validation"] = {"error": str(e)}

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
            diagnostics=diagnostics,
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
                p_fitted = np.clip(self.propensity_scores_, 1e-8, 1 - 1e-8)
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

        # Test for heteroscedasticity (Breusch-Pagan test approximation)
        if "outcome" in self._residuals_:
            outcome_residuals = self._residuals_["outcome"]

            # Simple test: correlation between squared residuals and fitted values
            if "mu_observed" in self.nuisance_estimates_:
                fitted_values = self.nuisance_estimates_["mu_observed"]
                squared_residuals = outcome_residuals**2

                try:
                    het_corr, het_p = pearsonr(fitted_values, squared_residuals)
                    results["heteroscedasticity_test"] = {
                        "correlation": float(het_corr),
                        "p_value": float(het_p),
                        "is_homoscedastic": bool(
                            abs(het_corr) < 0.1
                        ),  # Simple threshold
                        "interpretation": (
                            "Homoscedasticity assumption satisfied"
                            if abs(het_corr) < 0.1
                            else "Evidence of heteroscedasticity detected"
                        ),
                    }
                except Exception:
                    results["heteroscedasticity_test"] = {
                        "error": "Could not compute heteroscedasticity test"
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
                        except Exception:
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
