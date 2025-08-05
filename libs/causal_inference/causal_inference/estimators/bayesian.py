"""Bayesian Causal Inference estimator using PyMC.

This module implements Bayesian methods for causal inference that provide
full posterior distributions over treatment effects rather than point estimates.
The Bayesian approach offers several advantages:

1. Full uncertainty quantification through posterior distributions
2. Natural incorporation of prior knowledge
3. Credible intervals with Bayesian interpretation
4. Model averaging and sensitivity analysis capabilities

The estimator uses PyMC for probabilistic programming and MCMC sampling.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)

# Set up logger
logger = logging.getLogger(__name__)

# Suppress specific PyMC warnings for cleaner output, but keep convergence warnings
warnings.filterwarnings(
    "ignore",
    message=".*does not provide a compiled.*",
    category=UserWarning,
    module="pymc",
)
warnings.filterwarnings(
    "ignore", message=".*future warning.*", category=UserWarning, module="pymc"
)


@dataclass
class BayesianCausalEffect(CausalEffect):
    """Extended causal effect class for Bayesian results.

    Includes posterior samples, credible intervals, and Bayesian diagnostics
    in addition to standard causal effect measures.
    """

    # Posterior samples and diagnostics
    posterior_samples: NDArray[Any] | None = None
    credible_interval_level: float = 0.95
    effective_sample_size: float | None = None
    r_hat: float | None = None

    # Bayesian-specific intervals
    ate_credible_lower: float | None = None
    ate_credible_upper: float | None = None

    # Model information
    model_summary: dict[str, Any] | None = None
    prior_specification: dict[str, Any] | None = None
    mcmc_diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize Bayesian-specific fields."""
        super().__post_init__()

        # Use credible intervals as the primary CI if available
        if self.ate_credible_lower is not None:
            self.ate_ci_lower = self.ate_credible_lower
        if self.ate_credible_upper is not None:
            self.ate_ci_upper = self.ate_credible_upper


class BayesianEstimator(BaseEstimator):
    """Bayesian causal inference estimator using PyMC.

    This estimator implements Bayesian linear models for estimating
    average treatment effects with full posterior uncertainty quantification.

    The model specification is:
    Y = α + β*T + γ*X + ε

    Where:
    - Y is the outcome
    - T is the binary treatment (0/1)
    - X are covariates
    - β is the average treatment effect (ATE)
    - ε ~ Normal(0, σ²)

    Priors:
    - α ~ Normal(0, 10)  # Intercept prior
    - β ~ Normal(0, 2.5)  # Treatment effect prior (weakly informative)
    - γ ~ Normal(0, 2.5)  # Covariate effects prior
    - σ ~ HalfNormal(2.5)  # Error standard deviation prior

    Attributes:
        prior_intercept_scale: Prior scale for intercept
        prior_treatment_scale: Prior scale for treatment effect
        prior_covariate_scale: Prior scale for covariate effects
        prior_sigma_scale: Prior scale for error standard deviation
        mcmc_draws: Number of MCMC draws
        mcmc_tune: Number of tuning steps
        mcmc_chains: Number of MCMC chains
        credible_level: Credible interval level (default 0.95)
        random_state: Random seed for reproducible results
    """

    def __init__(
        self,
        prior_intercept_scale: float = 10.0,
        prior_treatment_scale: float = 2.5,
        prior_covariate_scale: float = 2.5,
        prior_sigma_scale: float = 2.5,
        mcmc_draws: int = 2000,
        mcmc_tune: int = 1000,
        mcmc_chains: int = 4,
        credible_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Bayesian estimator.

        Args:
            prior_intercept_scale: Scale parameter for intercept prior
            prior_treatment_scale: Scale parameter for treatment effect prior
            prior_covariate_scale: Scale parameter for covariate effects prior
            prior_sigma_scale: Scale parameter for error std prior
            mcmc_draws: Number of MCMC posterior draws
            mcmc_tune: Number of MCMC tuning steps
            mcmc_chains: Number of MCMC chains
            credible_level: Level for credible intervals (e.g., 0.95)
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
            **kwargs: Additional arguments for parent class
        """
        super().__init__(random_state=random_state, verbose=verbose, **kwargs)

        # Validate prior scale parameters
        if prior_intercept_scale <= 0:
            raise ValueError("prior_intercept_scale must be positive")
        if prior_treatment_scale <= 0:
            raise ValueError("prior_treatment_scale must be positive")
        if prior_covariate_scale <= 0:
            raise ValueError("prior_covariate_scale must be positive")
        if prior_sigma_scale <= 0:
            raise ValueError("prior_sigma_scale must be positive")

        # Validate MCMC parameters
        if mcmc_draws <= 0:
            raise ValueError("mcmc_draws must be positive")
        if mcmc_tune < 0:
            raise ValueError("mcmc_tune must be non-negative")
        if mcmc_chains <= 0:
            raise ValueError("mcmc_chains must be positive")
        if not 0 < credible_level < 1:
            raise ValueError("credible_level must be between 0 and 1")

        self.prior_intercept_scale = prior_intercept_scale
        self.prior_treatment_scale = prior_treatment_scale
        self.prior_covariate_scale = prior_covariate_scale
        self.prior_sigma_scale = prior_sigma_scale

        self.mcmc_draws = mcmc_draws
        self.mcmc_tune = mcmc_tune
        self.mcmc_chains = mcmc_chains
        self.credible_level = credible_level

        # Storage for fitted model and results
        self.model_: pm.Model | None = None
        self.trace_: az.InferenceData | None = None

    def _validate_data(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Validate input data for Bayesian estimation.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data
        """
        # Check sample size is sufficient for Bayesian estimation
        n_obs = len(treatment.values)
        if n_obs < 50:
            raise EstimationError(
                f"Insufficient sample size: {n_obs}. "
                f"Bayesian estimation requires at least 50 observations."
            )

        # Check treatment is binary
        if treatment.treatment_type != "binary":
            raise EstimationError(
                f"Bayesian estimator currently only supports binary treatments, "
                f"got: {treatment.treatment_type}"
            )

        # Check outcome is continuous
        if outcome.outcome_type not in ["continuous"]:
            raise EstimationError(
                f"Bayesian estimator currently only supports continuous outcomes, "
                f"got: {outcome.outcome_type}"
            )

        # Check for missing values
        treatment_values = np.asarray(treatment.values)
        outcome_values = np.asarray(outcome.values)

        if np.any(np.isnan(treatment_values)):
            raise EstimationError("Treatment data contains missing values")
        if np.any(np.isnan(outcome_values)):
            raise EstimationError("Outcome data contains missing values")

        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                cov_values = covariates.values.values
            else:
                cov_values = np.asarray(covariates.values)
            if np.any(np.isnan(cov_values)):
                raise EstimationError("Covariate data contains missing values")

    def _prepare_data(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None]:
        """Prepare data arrays for Bayesian modeling.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data

        Returns:
            Tuple of (treatment_array, outcome_array, covariate_array)
        """
        # Convert treatment to array
        treatment_array = np.asarray(treatment.values, dtype=float)

        # Convert outcome to array
        outcome_array = np.asarray(outcome.values, dtype=float)

        # Convert covariates to array if provided
        covariate_array = None
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                covariate_array = covariates.values.values.astype(float)
            else:
                covariate_array = np.asarray(covariates.values, dtype=float)

        return treatment_array, outcome_array, covariate_array

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the Bayesian causal model.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data for adjustment
        """
        # Store data for later use
        self.treatment_data = treatment
        self.outcome_data = outcome
        self.covariate_data = covariates

        # Validate and prepare data
        self._validate_data(treatment, outcome, covariates)
        T, Y, X = self._prepare_data(treatment, outcome, covariates)

        n_obs = len(T)
        n_covariates = X.shape[1] if X is not None else 0

        if self.verbose:
            logger.info(f"Fitting Bayesian model with {n_obs} observations")
            logger.info(f"Number of covariates: {n_covariates}")

        # Build PyMC model
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal("intercept", mu=0, sigma=self.prior_intercept_scale)
            beta = pm.Normal("treatment_effect", mu=0, sigma=self.prior_treatment_scale)

            # Covariate effects if present
            if X is not None:
                gamma = pm.Normal(
                    "covariate_effects",
                    mu=0,
                    sigma=self.prior_covariate_scale,
                    shape=n_covariates,
                )
                mu = alpha + beta * T + pm.math.dot(X, gamma)
            else:
                mu = alpha + beta * T

            # Error standard deviation
            sigma = pm.HalfNormal("error_sd", sigma=self.prior_sigma_scale)

            # Likelihood
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y)

        # Store model
        self.model_ = model

        # Sample from posterior
        if self.verbose:
            logger.info("Starting MCMC sampling...")

        with model:
            self.trace_ = pm.sample(
                draws=self.mcmc_draws,
                tune=self.mcmc_tune,
                chains=self.mcmc_chains,
                random_seed=self.random_state,
                progressbar=self.verbose,
                return_inferencedata=True,
            )

        if self.verbose:
            logger.info("MCMC sampling completed")

    def _estimate_ate_implementation(self) -> BayesianCausalEffect:
        """Estimate the Average Treatment Effect from posterior samples.

        Returns:
            BayesianCausalEffect object with posterior-based estimates
        """
        if self.trace_ is None:
            raise EstimationError("Model must be fitted before estimation")

        # Extract treatment effect posterior samples
        treatment_effect_samples = self.trace_.posterior[  # type: ignore[attr-defined]
            "treatment_effect"
        ].values.flatten()

        # Calculate posterior statistics
        ate_mean = float(np.mean(treatment_effect_samples))
        ate_std = float(np.std(treatment_effect_samples))

        # Calculate credible intervals
        alpha = 1 - self.credible_level
        credible_lower = float(np.percentile(treatment_effect_samples, 100 * alpha / 2))
        credible_upper = float(
            np.percentile(treatment_effect_samples, 100 * (1 - alpha / 2))
        )

        # MCMC diagnostics
        summary = az.summary(self.trace_, var_names=["treatment_effect"])
        ess = float(summary["ess_bulk"].iloc[0])
        r_hat = float(summary["r_hat"].iloc[0])

        if self.verbose:
            logger.info(f"ATE posterior mean: {ate_mean:.4f}")
            logger.info(f"ATE posterior std: {ate_std:.4f}")
            logger.info(
                f"{self.credible_level * 100}% credible interval: [{credible_lower:.4f}, {credible_upper:.4f}]"
            )
            logger.info(f"Effective sample size: {ess:.0f}")
            logger.info(f"R-hat: {r_hat:.4f}")

        # Check convergence with stricter thresholds
        if r_hat > 1.05:
            raise EstimationError(
                f"MCMC did not converge (R-hat={r_hat:.4f} > 1.05). "
                f"Try increasing mcmc_draws or mcmc_tune."
            )
        elif r_hat > 1.02:
            logger.warning(
                f"R-hat = {r_hat:.4f} > 1.02, indicating marginal convergence"
            )

        if ess < 400:
            logger.warning(
                f"Low effective sample size: {ess:.0f} < 400. "
                f"Consider increasing mcmc_draws for better estimates."
            )
        elif ess < 100:
            logger.warning(
                f"Very low effective sample size: {ess:.0f} < 100. "
                f"Results may be unreliable."
            )

        # Create prior specification summary
        prior_spec = {
            "intercept_scale": self.prior_intercept_scale,
            "treatment_scale": self.prior_treatment_scale,
            "covariate_scale": self.prior_covariate_scale,
            "sigma_scale": self.prior_sigma_scale,
        }

        # MCMC diagnostics
        mcmc_diag = {
            "draws": self.mcmc_draws,
            "tune": self.mcmc_tune,
            "chains": self.mcmc_chains,
            "effective_sample_size": ess,
            "r_hat": r_hat,
        }

        # Get number of observations
        n_obs = len(self.treatment_data.values) if self.treatment_data else 0

        return BayesianCausalEffect(
            ate=ate_mean,
            ate_se=ate_std,
            ate_credible_lower=credible_lower,
            ate_credible_upper=credible_upper,
            credible_interval_level=self.credible_level,
            confidence_level=self.credible_level,
            posterior_samples=treatment_effect_samples,
            effective_sample_size=ess,
            r_hat=r_hat,
            method="Bayesian Linear Model",
            n_observations=n_obs,
            prior_specification=prior_spec,
            mcmc_diagnostics=mcmc_diag,
        )

    def plot_posterior(
        self, var_names: list[str] | None = None, figsize: tuple[int, int] = (10, 6)
    ) -> Any:
        """Plot posterior distributions.

        Args:
            var_names: Variables to plot (default: ["treatment_effect"])
            figsize: Figure size

        Returns:
            ArviZ plot object
        """
        if self.trace_ is None:
            raise EstimationError("Model must be fitted before plotting")

        if var_names is None:
            var_names = ["treatment_effect"]

        return az.plot_posterior(  # type: ignore[no-untyped-call]
            self.trace_, var_names=var_names, figsize=figsize, textsize=12
        )

    def plot_trace(
        self, var_names: list[str] | None = None, figsize: tuple[int, int] = (12, 8)
    ) -> Any:
        """Plot MCMC traces for convergence diagnostics.

        Args:
            var_names: Variables to plot (default: ["treatment_effect"])
            figsize: Figure size

        Returns:
            ArviZ plot object
        """
        if self.trace_ is None:
            raise EstimationError("Model must be fitted before plotting")

        if var_names is None:
            var_names = ["treatment_effect"]

        return az.plot_trace(self.trace_, var_names=var_names, figsize=figsize)

    def parameter_summary(self) -> Any:
        """Get summary statistics for all model parameters.

        Returns:
            DataFrame with posterior summary statistics
        """
        if self.trace_ is None:
            raise EstimationError("Model must be fitted before parameter summary")

        return az.summary(self.trace_)

    def posterior_predictive_check(self, n_samples: int = 100) -> Any:
        """Perform posterior predictive checks.

        Args:
            n_samples: Number of posterior predictive samples

        Returns:
            ArviZ plot object
        """
        if self.trace_ is None or self.model_ is None:
            raise EstimationError(
                "Model must be fitted before posterior predictive checks"
            )

        with self.model_:
            pp_trace = pm.sample_posterior_predictive(
                self.trace_,
                samples=n_samples,
                random_seed=self.random_state,
                progressbar=False,
            )

        return az.plot_ppc(  # type: ignore[no-untyped-call]
            pp_trace, group="posterior_predictive", num_pp_samples=n_samples
        )
