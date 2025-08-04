"""Causal mediation analysis estimator for estimating direct and indirect effects.

This module implements causal mediation analysis to decompose total causal effects
into natural direct effects (NDE) and natural indirect effects (NIE) through mediators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from ..core.bootstrap import BootstrapConfig, BootstrapMixin

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class MediatorData:
    """Data model for mediator variables in mediation analysis.

    Represents the mediator variable(s) that lie on the causal pathway
    between treatment and outcome.
    """

    values: pd.Series | NDArray[Any]
    name: str = "mediator"
    mediator_type: str = "continuous"

    def __post_init__(self) -> None:
        """Validate mediator data."""
        if len(self.values) == 0:
            raise ValueError("Mediator values cannot be empty")


@dataclass
class MediationEffect(CausalEffect):
    """Extended causal effect class for mediation analysis results.

    Includes natural direct effect (NDE), natural indirect effect (NIE),
    and mediated proportion in addition to standard causal effect measures.
    """

    # Mediation-specific estimates
    nde: float | None = None  # Natural Direct Effect
    nie: float | None = None  # Natural Indirect Effect
    mediated_proportion: float | None = None  # Proportion mediated

    # Confidence intervals for mediation effects
    nde_ci_lower: float | None = None
    nde_ci_upper: float | None = None
    nie_ci_lower: float | None = None
    nie_ci_upper: float | None = None
    mediated_prop_ci_lower: float | None = None
    mediated_prop_ci_upper: float | None = None

    # Standard errors for mediation effects
    nde_se: float | None = None
    nie_se: float | None = None
    mediated_prop_se: float | None = None

    def __post_init__(self) -> None:
        """Validate mediation effect estimates."""
        super().__post_init__()

        # Check that NDE + NIE equals total effect (if all are present)
        if self.nde is not None and self.nie is not None:
            total_from_components = self.nde + self.nie
            if abs(total_from_components - self.ate) > 1e-6:
                # Allow small numerical differences
                logger.warning(
                    f"Effect decomposition inconsistency: NDE + NIE ({total_from_components:.6f}) != ATE ({self.ate:.6f})"
                )

    @property
    def is_mediated(self) -> bool:
        """Check if there is evidence of mediation.

        Returns True if the NIE confidence interval doesn't contain zero.
        """
        if self.nie_ci_lower is None or self.nie_ci_upper is None:
            return False
        return self.nie_ci_lower > 0 or self.nie_ci_upper < 0


class MediationEstimator(BootstrapMixin, BaseEstimator):
    """Causal mediation analysis estimator.

    This estimator implements the causal mediation analysis framework to decompose
    total treatment effects into natural direct effects (NDE) and natural indirect
    effects (NIE) through specified mediators.

    The method assumes:
    1. No unmeasured confounding of treatment-outcome relationship
    2. No unmeasured confounding of mediator-outcome relationship
    3. No unmeasured confounding of treatment-mediator relationship
    4. No treatment-induced confounding of mediator-outcome relationship

    Attributes:
        mediator_model: Model for predicting mediator from treatment and covariates
        outcome_model: Model for predicting outcome from treatment, mediator, and covariates
        mediator_model_type: Type of model for mediator ('auto', 'linear', 'logistic', 'random_forest')
        outcome_model_type: Type of model for outcome ('auto', 'linear', 'logistic', 'random_forest')
    """

    def __init__(
        self,
        mediator_model_type: str = "auto",
        outcome_model_type: str = "auto",
        mediator_model_params: dict[str, Any] | None = None,
        outcome_model_params: dict[str, Any] | None = None,
        bootstrap_config: Any | None = None,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the mediation estimator.

        Args:
            mediator_model_type: Model type for mediator ('auto', 'linear', 'logistic', 'random_forest')
            outcome_model_type: Model type for outcome ('auto', 'linear', 'logistic', 'random_forest')
            mediator_model_params: Parameters for mediator model
            outcome_model_params: Parameters for outcome model
            bootstrap_config: Configuration for bootstrap confidence intervals
            bootstrap_samples: Number of bootstrap samples (legacy parameter)
            confidence_level: Confidence level for intervals (legacy parameter)
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        # Create bootstrap config if not provided
        if bootstrap_config is None:
            bootstrap_config = BootstrapConfig(
                n_samples=bootstrap_samples,
                confidence_level=confidence_level,
                random_state=random_state,
            )

        super().__init__(
            bootstrap_config=bootstrap_config,
            random_state=random_state,
            verbose=verbose,
        )

        self.mediator_model_type = mediator_model_type
        self.outcome_model_type = outcome_model_type
        self.mediator_model_params = mediator_model_params or {}
        self.outcome_model_params = outcome_model_params or {}

        # Models will be set during fitting
        self.mediator_model: SklearnBaseEstimator | None = None
        self.outcome_model: SklearnBaseEstimator | None = None

        # Store mediator data
        self.mediator_data: MediatorData | None = None

    def _create_bootstrap_estimator(
        self, random_state: int | None = None
    ) -> MediationEstimator:
        """Create a new estimator instance for bootstrap sampling.

        Args:
            random_state: Random state for this bootstrap instance

        Returns:
            New MediationEstimator instance configured for bootstrap
        """
        return MediationEstimator(
            mediator_model_type=self.mediator_model_type,
            outcome_model_type=self.outcome_model_type,
            mediator_model_params=self.mediator_model_params,
            outcome_model_params=self.outcome_model_params,
            bootstrap_config=BootstrapConfig(n_samples=0),  # No nested bootstrap
            random_state=random_state,
            verbose=False,  # Reduce verbosity in bootstrap
        )

    def _create_model(
        self, model_type: str, outcome_type: str, params: dict[str, Any]
    ) -> SklearnBaseEstimator:
        """Create a sklearn model based on type and parameters.

        Args:
            model_type: Type of model to create
            outcome_type: Type of outcome variable
            params: Parameters for the model

        Returns:
            Fitted sklearn model
        """
        if model_type == "auto":
            if outcome_type in {"binary", "categorical"}:
                model_type = "logistic"
            else:
                model_type = "linear"

        model_params = params.copy()

        if model_type == "linear":
            # LinearRegression doesn't accept random_state
            return LinearRegression(**model_params)
        elif model_type == "logistic":
            if self.random_state is not None:
                model_params["random_state"] = self.random_state
            return LogisticRegression(**model_params)
        elif model_type == "random_forest":
            if self.random_state is not None:
                model_params["random_state"] = self.random_state
            if outcome_type in {"binary", "categorical"}:
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(**model_params)
            else:
                return RandomForestRegressor(**model_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(  # type: ignore[override]
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        mediator: MediatorData,
        covariates: CovariateData | None = None,
    ) -> MediationEstimator:
        """Fit the mediation estimator to data.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            mediator: Mediator variable data
            covariates: Optional covariate data for adjustment

        Returns:
            self: The fitted estimator instance
        """
        # Store mediator data
        self.mediator_data = mediator

        # Call parent fit method for validation and setup
        super().fit(treatment, outcome, covariates)

        return self

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Implement the specific fitting logic for mediation analysis.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data for adjustment
        """
        if self.mediator_data is None:
            raise EstimationError("Mediator data must be provided to fit method")

        # Prepare feature matrices
        X_mediator, X_outcome = self._prepare_feature_matrices(treatment, covariates)

        # Fit mediator model: M ~ T + X
        self.mediator_model = self._create_model(
            self.mediator_model_type,
            self.mediator_data.mediator_type,
            self.mediator_model_params,
        )
        self.mediator_model.fit(X_mediator, self.mediator_data.values)

        # Fit outcome model: Y ~ T + M + X
        self.outcome_model = self._create_model(
            self.outcome_model_type,
            outcome.outcome_type,
            self.outcome_model_params,
        )
        self.outcome_model.fit(X_outcome, outcome.values)

        if self.verbose:
            print("Fitted mediator and outcome models for mediation analysis")

    def _prepare_feature_matrices(
        self,
        treatment: TreatmentData,
        covariates: CovariateData | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Prepare feature matrices for mediator and outcome models.

        Args:
            treatment: Treatment data
            covariates: Covariate data

        Returns:
            Tuple of (mediator_features, outcome_features)
        """
        # Convert treatment to array
        if isinstance(treatment.values, pd.Series):
            T = np.asarray(treatment.values.values).reshape(-1, 1)
        else:
            T = np.asarray(treatment.values).reshape(-1, 1)

        # Convert mediator to array
        if self.mediator_data is None:
            raise EstimationError("Mediator data is required")
        if isinstance(self.mediator_data.values, pd.Series):
            M = np.asarray(self.mediator_data.values.values).reshape(-1, 1)
        else:
            M = np.asarray(self.mediator_data.values).reshape(-1, 1)

        # Add covariates if provided
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                X = covariates.values.values
            else:
                X = np.array(covariates.values)

            # Features for mediator model: [T, X]
            X_mediator = np.hstack([T, X])
            # Features for outcome model: [T, M, X]
            X_outcome = np.hstack([T, M, X])
        else:
            # Features for mediator model: [T]
            X_mediator = T
            # Features for outcome model: [T, M]
            X_outcome = np.hstack([T, M])

        return X_mediator, X_outcome

    def _estimate_ate_implementation(self) -> MediationEffect:
        """Estimate mediation effects using the fitted models.

        Returns:
            MediationEffect object with NDE, NIE, and total effect estimates
        """
        if self.mediator_model is None or self.outcome_model is None:
            raise EstimationError("Models must be fitted before estimation")

        # Check that we have fitted data
        if self.treatment_data is None:
            raise EstimationError("Treatment data is not available")

        # Get original data arrays
        X_mediator, X_outcome = self._prepare_feature_matrices(
            self.treatment_data, self.covariate_data
        )

        n_obs = len(self.treatment_data.values)

        # Create counterfactual datasets
        X_mediator_t1 = X_mediator.copy()
        X_mediator_t0 = X_mediator.copy()
        X_mediator_t1[:, 0] = 1  # Set treatment to 1
        X_mediator_t0[:, 0] = 0  # Set treatment to 0

        # Predict mediator under different treatment conditions
        M_t1 = self.mediator_model.predict(X_mediator_t1)
        M_t0 = self.mediator_model.predict(X_mediator_t0)

        # Prepare outcome prediction features
        if self.covariate_data is not None:
            if isinstance(self.covariate_data.values, pd.DataFrame):
                X_cov = self.covariate_data.values.values
            else:
                X_cov = np.array(self.covariate_data.values)

            # Counterfactual outcome features: [T, M, X]
            X_outcome_t1_m1 = np.hstack(
                [np.ones((n_obs, 1)), M_t1.reshape(-1, 1), X_cov]
            )
            X_outcome_t1_m0 = np.hstack(
                [np.ones((n_obs, 1)), M_t0.reshape(-1, 1), X_cov]
            )
            X_outcome_t0_m0 = np.hstack(
                [np.zeros((n_obs, 1)), M_t0.reshape(-1, 1), X_cov]
            )
        else:
            # Counterfactual outcome features: [T, M]
            X_outcome_t1_m1 = np.hstack([np.ones((n_obs, 1)), M_t1.reshape(-1, 1)])
            X_outcome_t1_m0 = np.hstack([np.ones((n_obs, 1)), M_t0.reshape(-1, 1)])
            X_outcome_t0_m0 = np.hstack([np.zeros((n_obs, 1)), M_t0.reshape(-1, 1)])

        # Predict counterfactual outcomes
        Y_t1_m1 = self.outcome_model.predict(X_outcome_t1_m1)  # Y(1, M(1))
        Y_t1_m0 = self.outcome_model.predict(X_outcome_t1_m0)  # Y(1, M(0))
        Y_t0_m0 = self.outcome_model.predict(X_outcome_t0_m0)  # Y(0, M(0))

        # Calculate mediation effects
        # Natural Direct Effect: E[Y(1, M(0)) - Y(0, M(0))]
        nde = np.mean(Y_t1_m0 - Y_t0_m0)

        # Natural Indirect Effect: E[Y(1, M(1)) - Y(1, M(0))]
        nie = np.mean(Y_t1_m1 - Y_t1_m0)

        # Total Effect: E[Y(1, M(1)) - Y(0, M(0))]
        ate = np.mean(Y_t1_m1 - Y_t0_m0)

        # Mediated proportion
        mediated_proportion = nie / ate if ate != 0 else 0.0

        # Create mediation effect object
        effect = MediationEffect(
            ate=ate,
            nde=nde,
            nie=nie,
            mediated_proportion=mediated_proportion,
            method="mediation_analysis",
            n_observations=n_obs,
        )

        return effect

    def estimate_ate(self, use_cache: bool = True) -> MediationEffect:
        """Estimate the Average Treatment Effect with mediation decomposition.

        Args:
            use_cache: Whether to use cached results if available

        Returns:
            MediationEffect object with mediation analysis results
        """
        if not self.is_fitted:
            raise EstimationError("Estimator must be fitted before estimation")

        # Return cached result if available and requested
        if use_cache and self._causal_effect is not None:
            if isinstance(self._causal_effect, MediationEffect):
                return self._causal_effect
            # If cached result is wrong type, recalculate
            self._causal_effect = None

        try:
            # Get point estimates
            effect = self._estimate_ate_implementation()

            # Add bootstrap confidence intervals if configured
            if (
                self.bootstrap_config is not None
                and self.bootstrap_config.n_samples > 0
            ):
                effect = self._add_bootstrap_confidence_intervals(effect)

            # Cache the result
            self._causal_effect = effect

            if self.verbose:
                print(f"Estimated ATE: {effect.ate:.4f}")
                print(f"Estimated NDE: {effect.nde:.4f}")
                print(f"Estimated NIE: {effect.nie:.4f}")
                print(f"Mediated Proportion: {effect.mediated_proportion:.4f}")
                if effect.ate_ci_lower is not None:
                    print(
                        f"ATE 95% CI: [{effect.ate_ci_lower:.4f}, {effect.ate_ci_upper:.4f}]"
                    )

            return effect

        except Exception as e:
            raise EstimationError(
                f"Failed to estimate mediation effects: {str(e)}"
            ) from e

    def _add_bootstrap_confidence_intervals(
        self, effect: MediationEffect
    ) -> MediationEffect:
        """Add bootstrap confidence intervals to mediation effect.

        Args:
            effect: Original mediation effect estimate

        Returns:
            Updated mediation effect with confidence intervals
        """
        if not self.bootstrap_config or self.bootstrap_config.n_samples <= 0:
            return effect

        try:
            # Perform bootstrap sampling with proper error handling
            bootstrap_results = self._perform_mediation_bootstrap()

            if not bootstrap_results:
                logger.warning("Bootstrap failed - no successful samples")
                return effect

            # Calculate confidence intervals
            confidence_level = self.bootstrap_config.confidence_level
            alpha = 1 - confidence_level
            lower_percentile = 100 * (alpha / 2)
            upper_percentile = 100 * (1 - alpha / 2)

            # Extract bootstrap estimates
            ate_estimates = [r["ate"] for r in bootstrap_results]
            nde_estimates = [r["nde"] for r in bootstrap_results]
            nie_estimates = [r["nie"] for r in bootstrap_results]
            mp_estimates = [r["mediated_proportion"] for r in bootstrap_results]

            # Add confidence intervals
            if ate_estimates:
                effect.ate_ci_lower = float(
                    np.percentile(ate_estimates, lower_percentile)
                )
                effect.ate_ci_upper = float(
                    np.percentile(ate_estimates, upper_percentile)
                )
                effect.ate_se = float(np.std(ate_estimates))
                effect.bootstrap_samples = len(ate_estimates)

            if nde_estimates:
                effect.nde_ci_lower = float(
                    np.percentile(nde_estimates, lower_percentile)
                )
                effect.nde_ci_upper = float(
                    np.percentile(nde_estimates, upper_percentile)
                )
                effect.nde_se = float(np.std(nde_estimates))

            if nie_estimates:
                effect.nie_ci_lower = float(
                    np.percentile(nie_estimates, lower_percentile)
                )
                effect.nie_ci_upper = float(
                    np.percentile(nie_estimates, upper_percentile)
                )
                effect.nie_se = float(np.std(nie_estimates))

            if mp_estimates:
                effect.mediated_prop_ci_lower = float(
                    np.percentile(mp_estimates, lower_percentile)
                )
                effect.mediated_prop_ci_upper = float(
                    np.percentile(mp_estimates, upper_percentile)
                )
                effect.mediated_prop_se = float(np.std(mp_estimates))

            effect.bootstrap_method = "percentile"
            effect.bootstrap_converged = True

        except Exception as e:
            logger.warning(f"Bootstrap confidence intervals failed: {e}")
            effect.bootstrap_converged = False

        return effect

    def _perform_mediation_bootstrap(self) -> list[dict[str, float]]:
        """Perform bootstrap sampling for mediation analysis.

        Returns:
            List of bootstrap results with mediation estimates
        """
        if self.treatment_data is None or self.bootstrap_config is None:
            raise EstimationError("Treatment data and bootstrap config are required")

        n_obs = len(self.treatment_data.values)
        bootstrap_results = []
        failed_samples = 0
        max_failures = int(
            0.5 * self.bootstrap_config.n_samples
        )  # Allow up to 50% failures

        for i in range(self.bootstrap_config.n_samples):
            try:
                # Generate bootstrap indices
                if self.bootstrap_config.random_state is not None:
                    np.random.seed(self.bootstrap_config.random_state + i)

                boot_indices = np.random.choice(n_obs, size=n_obs, replace=True)

                # Check required data
                if (
                    self.treatment_data is None
                    or self.outcome_data is None
                    or self.mediator_data is None
                ):
                    raise EstimationError("Required data not available for bootstrap")

                # Create bootstrap samples
                boot_treatment_values = (
                    self.treatment_data.values.iloc[boot_indices]
                    if isinstance(self.treatment_data.values, pd.Series)
                    else self.treatment_data.values[boot_indices]
                )

                boot_outcome_values = (
                    self.outcome_data.values.iloc[boot_indices]
                    if isinstance(self.outcome_data.values, pd.Series)
                    else self.outcome_data.values[boot_indices]
                )

                boot_mediator_values = (
                    self.mediator_data.values.iloc[boot_indices]
                    if isinstance(self.mediator_data.values, pd.Series)
                    else self.mediator_data.values[boot_indices]
                )

                # Create data objects
                boot_treatment = TreatmentData(
                    values=boot_treatment_values,
                    name=self.treatment_data.name,
                    treatment_type=self.treatment_data.treatment_type,
                )

                boot_outcome = OutcomeData(
                    values=boot_outcome_values,
                    name=self.outcome_data.name,
                    outcome_type=self.outcome_data.outcome_type,
                )

                boot_mediator = MediatorData(
                    values=boot_mediator_values,
                    name=self.mediator_data.name,
                    mediator_type=self.mediator_data.mediator_type,
                )

                boot_covariates = None
                if self.covariate_data is not None:
                    boot_cov_values = (
                        self.covariate_data.values.iloc[boot_indices]
                        if isinstance(self.covariate_data.values, pd.DataFrame)
                        else self.covariate_data.values[boot_indices]
                    )
                    boot_covariates = CovariateData(
                        values=boot_cov_values, names=self.covariate_data.names
                    )

                # Create and fit bootstrap estimator
                boot_estimator = self._create_bootstrap_estimator(
                    random_state=self.bootstrap_config.random_state + i
                    if self.bootstrap_config.random_state
                    else None
                )

                boot_estimator.fit(
                    boot_treatment, boot_outcome, boot_mediator, boot_covariates
                )
                boot_effect = boot_estimator._estimate_ate_implementation()

                # Store results
                bootstrap_results.append(
                    {
                        "ate": float(boot_effect.ate),
                        "nde": float(boot_effect.nde)
                        if boot_effect.nde is not None
                        else 0.0,
                        "nie": float(boot_effect.nie)
                        if boot_effect.nie is not None
                        else 0.0,
                        "mediated_proportion": float(boot_effect.mediated_proportion)
                        if boot_effect.mediated_proportion is not None
                        else 0.0,
                    }
                )

            except Exception as e:
                failed_samples += 1
                if self.verbose:
                    logger.debug(f"Bootstrap sample {i} failed: {e}")

                # Check for too many failures
                if failed_samples > max_failures:
                    logger.error(
                        f"Too many bootstrap failures ({failed_samples}/{i + 1})"
                    )
                    break

        # Check minimum sample requirement
        if len(bootstrap_results) < 10:
            raise EstimationError(
                f"Insufficient successful bootstrap samples: {len(bootstrap_results)}/{self.bootstrap_config.n_samples}"
            )

        success_rate = len(bootstrap_results) / self.bootstrap_config.n_samples
        if success_rate < 0.5:
            logger.warning(f"Low bootstrap success rate: {success_rate:.1%}")

        return bootstrap_results
