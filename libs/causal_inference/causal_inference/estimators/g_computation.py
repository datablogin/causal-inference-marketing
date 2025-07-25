"""G-computation (Standardization) estimator for causal inference.

This module implements the G-computation method, which estimates causal effects
by fitting outcome models and then averaging predicted outcomes under different
treatment scenarios.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)


class GComputationEstimator(BaseEstimator):
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
        model_params: dict[str, Any] | None = None,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the G-computation estimator.

        Args:
            model_type: Model type ('auto', 'linear', 'logistic', 'random_forest')
            model_params: Parameters to pass to the sklearn model
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        super().__init__(random_state=random_state, verbose=verbose)

        self.model_type = model_type
        self.model_params = model_params or {}
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level

        # Model storage
        self.outcome_model: SklearnBaseEstimator | None = None
        self._model_features: list[str] | None = None

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
            return LogisticRegression(
                random_state=self.random_state,
                **self.model_params
            )
        elif model_type == "random_forest":
            if outcome_type == "continuous":
                return RandomForestRegressor(
                    random_state=self.random_state,
                    **self.model_params
                )
            else:
                return RandomForestClassifier(
                    random_state=self.random_state,
                    **self.model_params
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _prepare_features(
        self,
        treatment: TreatmentData,
        covariates: CovariateData | None = None,
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
                # Use the DataFrame directly
                for col in covariates.values.columns:
                    features[col] = covariates.values[col]
            else:
                # Convert array to DataFrame with covariate names
                cov_names = covariates.names or [f"X{i}" for i in range(covariates.values.shape[1])]
                for i, name in enumerate(cov_names):
                    features[name] = covariates.values[:, i]

        return features

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the outcome model for G-computation.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data for adjustment
        """
        # Prepare features
        X = self._prepare_features(treatment, covariates)
        self._model_features = list(X.columns)

        # Prepare outcome
        if isinstance(outcome.values, pd.Series):
            y = outcome.values.values
        else:
            y = outcome.values

        # Select and fit model
        self.outcome_model = self._select_model(outcome.outcome_type)

        try:
            self.outcome_model.fit(X, y)

            if self.verbose:
                # Calculate model fit metrics
                y_pred = self.outcome_model.predict(X)
                if outcome.outcome_type == "continuous":
                    mse = mean_squared_error(y, y_pred)
                    print(f"Outcome model MSE: {mse:.4f}")
                elif outcome.outcome_type == "binary":
                    if hasattr(self.outcome_model, "predict_proba"):
                        y_pred_proba = self.outcome_model.predict_proba(X)[:, 1]
                        ll = log_loss(y, y_pred_proba)
                        print(f"Outcome model log-loss: {ll:.4f}")

        except Exception as e:
            raise EstimationError(f"Failed to fit outcome model: {str(e)}") from e

    def _predict_counterfactuals(
        self,
        treatment_value: float | int,
        covariates: CovariateData | None = None,
    ) -> NDArray[Any]:
        """Predict counterfactual outcomes for a given treatment value.

        Args:
            treatment_value: Treatment value to set for all units
            covariates: Covariate data (uses fitted data if None)

        Returns:
            Array of predicted counterfactual outcomes
        """
        if self.outcome_model is None or self.treatment_data is None:
            raise EstimationError("Model must be fitted before prediction")

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

        # Start with treatment set to specified value
        counterfactual_features = pd.DataFrame({
            self.treatment_data.name: [treatment_value] * n_obs
        })

        # Add covariates
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                for col in covariates.values.columns:
                    counterfactual_features[col] = covariates.values[col].values
            else:
                cov_names = covariates.names or [f"X{i}" for i in range(covariates.values.shape[1])]
                for i, name in enumerate(cov_names):
                    counterfactual_features[name] = covariates.values[:, i]

        # Ensure features are in the same order as training
        if self._model_features is not None:
            counterfactual_features = counterfactual_features[self._model_features]

        # Predict counterfactual outcomes
        return self.outcome_model.predict(counterfactual_features)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate Average Treatment Effect using G-computation.

        Returns:
            CausalEffect object with ATE estimate and confidence intervals
        """
        if self.outcome_model is None or self.treatment_data is None:
            raise EstimationError("Model must be fitted before estimation")

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
            ate = np.mean(outcomes[treatment_values[1]]) - np.mean(outcomes[treatment_values[0]])
            potential_outcome_treated = np.mean(outcomes[treatment_values[1]])
            potential_outcome_control = np.mean(outcomes[treatment_values[0]])

        # Bootstrap confidence intervals
        ate_ci_lower, ate_ci_upper, bootstrap_estimates = self._bootstrap_confidence_interval()

        # Calculate standard error from bootstrap
        ate_se = np.std(bootstrap_estimates) if bootstrap_estimates is not None else None

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
            confidence_level=self.confidence_level,
            potential_outcome_treated=potential_outcome_treated,
            potential_outcome_control=potential_outcome_control,
            method="G-computation",
            n_observations=len(self.treatment_data.values),
            n_treated=n_treated,
            n_control=n_control,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_estimates=bootstrap_estimates,
        )

    def _bootstrap_confidence_interval(self) -> tuple[float | None, float | None, NDArray[Any] | None]:
        """Calculate bootstrap confidence intervals for ATE.

        Returns:
            Tuple of (lower_ci, upper_ci, bootstrap_estimates)
        """
        if self.bootstrap_samples <= 0:
            return None, None, None

        if self.treatment_data is None or self.outcome_data is None:
            raise EstimationError("Data must be available for bootstrap")

        bootstrap_ates: list[float] = []
        n_obs = len(self.treatment_data.values)

        for _ in range(self.bootstrap_samples):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)

            # Create bootstrap datasets
            boot_treatment = TreatmentData(
                values=self.treatment_data.values.iloc[bootstrap_indices] if isinstance(self.treatment_data.values, pd.Series)
                       else self.treatment_data.values[bootstrap_indices],
                name=self.treatment_data.name,
                treatment_type=self.treatment_data.treatment_type,
                categories=self.treatment_data.categories,
            )

            boot_outcome = OutcomeData(
                values=self.outcome_data.values.iloc[bootstrap_indices] if isinstance(self.outcome_data.values, pd.Series)
                       else self.outcome_data.values[bootstrap_indices],
                name=self.outcome_data.name,
                outcome_type=self.outcome_data.outcome_type,
            )

            boot_covariates = None
            if self.covariate_data is not None:
                if isinstance(self.covariate_data.values, pd.DataFrame):
                    boot_cov_values = self.covariate_data.values.iloc[bootstrap_indices]
                else:
                    boot_cov_values = self.covariate_data.values[bootstrap_indices]

                boot_covariates = CovariateData(
                    values=boot_cov_values,
                    names=self.covariate_data.names,
                )

            # Fit model on bootstrap sample
            try:
                boot_estimator = GComputationEstimator(
                    model_type=self.model_type,
                    model_params=self.model_params,
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

    def predict_potential_outcomes(
        self,
        treatment_values: pd.Series | NDArray[Any],
        covariates: pd.DataFrame | NDArray[Any] | None = None,
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
                cov_data = CovariateData(values=covariates, names=list(covariates.columns))
            else:
                # Use the same covariate names as the training data
                if self.covariate_data is not None:
                    cov_names = self.covariate_data.names
                else:
                    # Fall back to generic names matching the number of features in training
                    n_features = len(self._model_features) - 1 if self._model_features else covariates.shape[1]
                    cov_names = [f"X{i+1}" for i in range(n_features)]

                # Create DataFrame to ensure correct column ordering
                cov_df = pd.DataFrame(covariates, columns=cov_names)
                cov_data = CovariateData(values=cov_df, names=cov_names)
        else:
            cov_data = None

        # Predict under control (Y(0)) and treatment (Y(1))
        y0_pred = self._predict_counterfactuals(0, cov_data)
        y1_pred = self._predict_counterfactuals(1, cov_data)

        return y0_pred, y1_pred
