"""G-estimation for Structural Nested Models (SNMs) implementation.

This module implements G-estimation, a robust method for estimating parameters
of structural nested models that provides consistent estimates even when the
outcome model is misspecified, as long as the treatment model is correct.

G-estimation works by finding parameter values that make the treatment effect
zero in the counterfactual world where treatment was assigned randomly.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from ..core.bootstrap import BootstrapConfig, BootstrapMixin


class GEstimationEstimator(BootstrapMixin, BaseEstimator):
    """G-estimation for Structural Nested Models.

    G-estimation estimates parameters of structural nested models by finding
    parameter values where the treatment effect is zero after adjusting for
    the structural model. This provides consistent estimates even when the
    outcome model is misspecified, as long as the treatment model is correct.

    The method involves:
    1. Specifying a structural nested mean model (SNMM)
    2. Using optimization to find parameters where treatment is unassociated
       with adjusted outcomes
    3. Computing confidence intervals via bootstrap or asymptotic theory

    Attributes:
        structural_model: Type of structural model ('linear', 'multiplicative', 'general')
        treatment_model: Type of treatment model ('logistic', 'random_forest')
        optimization_method: Optimization approach ('grid_search', 'root_finding', 'gradient')
        parameter_range: Range for parameter search (for grid search)
        n_grid_points: Number of grid points (for grid search)
        optimization_result: Results from the optimization procedure
        estimated_parameters: Final estimated parameters
    """

    def __init__(
        self,
        structural_model: Literal["linear", "multiplicative", "general"] = "linear",
        treatment_model: Literal["logistic", "random_forest"] = "logistic",
        optimization_method: Literal[
            "grid_search", "root_finding", "gradient"
        ] = "grid_search",
        parameter_range: tuple[float, float] = (-10.0, 10.0),
        n_grid_points: int = 1000,
        treatment_model_params: dict[str, Any] | None = None,
        covariates_for_interaction: list[str] | None = None,
        bootstrap_config: BootstrapConfig | None = None,
        # Legacy parameters for backward compatibility
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the G-estimation estimator.

        Args:
            structural_model: Type of structural nested model
            treatment_model: Type of treatment model for propensity scores
            optimization_method: Method for parameter optimization
            parameter_range: Range for parameter search (min, max)
            n_grid_points: Number of grid points for grid search
            treatment_model_params: Parameters for treatment model
            covariates_for_interaction: Covariates to include in structural model interactions
            bootstrap_config: Configuration for bootstrap confidence intervals
            bootstrap_samples: Legacy parameter - number of bootstrap samples
            confidence_level: Legacy parameter - confidence level
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
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
            random_state=random_state,
            verbose=verbose,
        )

        self.structural_model = structural_model
        self.treatment_model = treatment_model
        self.optimization_method = optimization_method
        self.parameter_range = parameter_range
        self.n_grid_points = n_grid_points
        self.treatment_model_params = treatment_model_params or {}
        self.covariates_for_interaction = covariates_for_interaction or []

        # Model and results storage
        self.propensity_model: SklearnBaseEstimator | None = None
        self.propensity_scores: NDArray[Any] | None = None
        self.optimization_result: dict[str, Any] | None = None
        self.estimated_parameters: dict[str, float] | None = None

        # Validation
        if not isinstance(parameter_range, tuple) or len(parameter_range) != 2:
            raise ValueError("parameter_range must be a tuple of (min, max)")
        if parameter_range[0] >= parameter_range[1]:
            raise ValueError("parameter_range min must be less than max")

    def _create_bootstrap_estimator(
        self, random_state: int | None = None
    ) -> GEstimationEstimator:
        """Create a new estimator instance for bootstrap sampling.

        Args:
            random_state: Random state for this bootstrap instance

        Returns:
            New GEstimationEstimator instance configured for bootstrap
        """
        return GEstimationEstimator(
            structural_model=self.structural_model,
            treatment_model=self.treatment_model,
            optimization_method=self.optimization_method,
            parameter_range=self.parameter_range,
            n_grid_points=self.n_grid_points,
            treatment_model_params=self.treatment_model_params,
            covariates_for_interaction=self.covariates_for_interaction,
            bootstrap_config=BootstrapConfig(n_samples=0),  # No nested bootstrap
            random_state=random_state,
            verbose=False,  # Reduce verbosity in bootstrap
        )

    def _create_treatment_model(self) -> SklearnBaseEstimator:
        """Create treatment model for propensity score estimation.

        Returns:
            Initialized sklearn model for treatment prediction
        """
        if self.treatment_model == "logistic":
            default_params = {
                "solver": "liblinear",
                "max_iter": 1000,
                "C": 1.0,
            }
            merged_params = {**default_params, **self.treatment_model_params}
            return LogisticRegression(random_state=self.random_state, **merged_params)
        elif self.treatment_model == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
            }
            merged_params = {**default_params, **self.treatment_model_params}
            return RandomForestClassifier(
                random_state=self.random_state, **merged_params
            )
        else:
            raise ValueError(f"Unknown treatment model: {self.treatment_model}")

    def _fit_propensity_model(
        self, treatment: TreatmentData, covariates: CovariateData | None = None
    ) -> None:
        """Fit propensity score model for treatment prediction.

        Args:
            treatment: Treatment assignment data
            covariates: Covariate data for propensity modeling
        """
        if covariates is None:
            raise EstimationError(
                "G-estimation requires covariates for propensity modeling"
            )

        # Prepare features
        if isinstance(covariates.values, pd.DataFrame):
            X = covariates.values
        else:
            cov_names = covariates.names or [
                f"X{i}" for i in range(covariates.values.shape[1])
            ]
            X = pd.DataFrame(covariates.values, columns=cov_names)

        # Prepare treatment
        y = np.asarray(treatment.values)

        # Create and fit model
        self.propensity_model = self._create_treatment_model()
        self.propensity_model.fit(X, y)

        # Estimate propensity scores
        raw_scores = self.propensity_model.predict_proba(X)[:, 1]

        # Apply propensity score trimming for numerical stability
        self.propensity_scores = np.clip(raw_scores, 0.01, 0.99)

        if self.verbose:
            # Calculate model fit metrics
            from sklearn.metrics import roc_auc_score

            try:
                auc = roc_auc_score(y, raw_scores)
                print(f"Propensity model AUC: {auc:.4f}")
                # Report trimming statistics
                n_trimmed_low = np.sum(raw_scores < 0.01)
                n_trimmed_high = np.sum(raw_scores > 0.99)
                if n_trimmed_low > 0 or n_trimmed_high > 0:
                    print(
                        f"Trimmed {n_trimmed_low} low scores and {n_trimmed_high} high scores"
                    )
            except ValueError:
                print("Could not calculate AUC (possibly no treatment variation)")

    def _create_structural_model_matrix(
        self,
        treatment: TreatmentData,
        covariates: CovariateData | None = None,
    ) -> pd.DataFrame:
        """Create design matrix for structural nested model.

        Args:
            treatment: Treatment assignment data
            covariates: Covariate data

        Returns:
            Design matrix for structural model
        """
        # Start with treatment
        design_matrix = pd.DataFrame({"treatment": np.asarray(treatment.values)})

        # Add interactions if specified
        if self.covariates_for_interaction and covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                cov_df = covariates.values
                cov_names = list(cov_df.columns)
            else:
                cov_names = covariates.names or [
                    f"X{i}" for i in range(covariates.values.shape[1])
                ]
                cov_df = pd.DataFrame(covariates.values, columns=cov_names)

            # Add interaction terms
            for cov_name in self.covariates_for_interaction:
                if cov_name in cov_names:
                    interaction_name = f"treatment_x_{cov_name}"
                    design_matrix[interaction_name] = (
                        design_matrix["treatment"] * cov_df[cov_name]
                    )

        return design_matrix

    def _apply_structural_model(
        self,
        outcome: NDArray[Any],
        treatment: NDArray[Any],
        parameters: dict[str, float],
        covariates: CovariateData | None = None,
    ) -> NDArray[Any]:
        """Apply structural nested model to adjust outcomes.

        Args:
            outcome: Original outcome values
            treatment: Treatment assignment values
            parameters: Model parameters
            covariates: Covariate data

        Returns:
            Adjusted outcome values H(outcome, treatment, parameters)
        """
        if self.structural_model == "linear":
            # Linear SNMM: H = Y - psi * A
            adjustment = parameters.get("main_effect", 0.0) * treatment

            # Add interaction adjustments
            if self.covariates_for_interaction and covariates is not None:
                if isinstance(covariates.values, pd.DataFrame):
                    cov_df = covariates.values
                    cov_names = list(cov_df.columns)
                else:
                    cov_names = covariates.names or [
                        f"X{i}" for i in range(covariates.values.shape[1])
                    ]
                    cov_df = pd.DataFrame(covariates.values, columns=cov_names)

                for cov_name in self.covariates_for_interaction:
                    if cov_name in cov_names:
                        param_name = f"{cov_name}_interaction"
                        if param_name in parameters:
                            adjustment += (
                                parameters[param_name] * treatment * cov_df[cov_name]
                            )

            return np.asarray(outcome - adjustment)

        elif self.structural_model == "multiplicative":
            # Multiplicative SNMM: H = Y / (1 + psi * A)
            multiplier = 1.0 + parameters.get("main_effect", 0.0) * treatment
            # Protect against division by zero and negative multipliers
            # For negative multipliers, use absolute value with minimum threshold
            multiplier = np.where(
                multiplier > 0,
                np.maximum(multiplier, 1e-10),  # Positive case
                np.maximum(
                    np.abs(multiplier), 1e-10
                ),  # Negative case (use absolute value)
            )
            return np.asarray(outcome / multiplier)

        elif self.structural_model == "general":
            # General form - could be extended for more complex models
            # For now, default to linear
            return self._apply_structural_model(
                outcome, treatment, parameters, covariates
            )

        else:
            raise ValueError(f"Unknown structural model: {self.structural_model}")

    def _objective_function(
        self,
        params: NDArray[Any],
        outcome: NDArray[Any],
        treatment: NDArray[Any],
        propensity_scores: NDArray[Any],
        covariates: CovariateData | None = None,
    ) -> float:
        """Objective function for G-estimation optimization.

        The objective is to find parameters where the adjusted outcome
        is unassociated with treatment given covariates (propensity scores).

        Args:
            params: Parameter values to evaluate
            outcome: Outcome values
            treatment: Treatment values
            propensity_scores: Propensity scores
            covariates: Covariate data

        Returns:
            Objective function value (should be minimized to zero)
        """
        # Convert parameters to dictionary with bounds checking
        if len(params) == 0:
            raise ValueError("No parameters provided to objective function")

        parameters = {"main_effect": params[0]}

        # Add interaction parameters safely
        n_interactions = len(self.covariates_for_interaction)
        if n_interactions > 0 and len(params) > 1:
            # Only use as many interaction parameters as we have covariates and parameters
            n_params_to_use = min(n_interactions, len(params) - 1)
            for i in range(n_params_to_use):
                cov_name = self.covariates_for_interaction[i]
                parameters[f"{cov_name}_interaction"] = params[i + 1]

        # Apply structural model
        adjusted_outcome = self._apply_structural_model(
            outcome, treatment, parameters, covariates
        )

        # Calculate test statistic - correlation between adjusted outcome and treatment
        # weighted by inverse propensity scores
        treated_mask = treatment == 1
        control_mask = treatment == 0

        if np.sum(treated_mask) == 0 or np.sum(control_mask) == 0:
            return float("inf")

        # IPW-weighted means with bounds checking for extreme propensity scores
        treated_ps = propensity_scores[treated_mask]
        control_ps = propensity_scores[control_mask]

        # Check for extreme propensity scores and handle infinite weights
        treated_weights = 1.0 / treated_ps
        control_weights = 1.0 / (1.0 - control_ps)

        # Cap weights to prevent infinite values from affecting computation
        max_weight = 100.0  # Reasonable upper bound
        treated_weights = np.minimum(treated_weights, max_weight)
        control_weights = np.minimum(control_weights, max_weight)

        # Check for any remaining infinite or NaN weights
        if not np.all(np.isfinite(treated_weights)) or not np.all(
            np.isfinite(control_weights)
        ):
            return float("inf")

        # Normalize weights
        treated_weights = treated_weights / np.sum(treated_weights)
        control_weights = control_weights / np.sum(control_weights)

        # Weighted means of adjusted outcomes
        treated_mean = np.average(
            adjusted_outcome[treated_mask], weights=treated_weights
        )
        control_mean = np.average(
            adjusted_outcome[control_mask], weights=control_weights
        )

        # Test statistic is the difference in means
        test_statistic = treated_mean - control_mean

        return float(test_statistic**2)

    def _grid_search_optimization(
        self,
        outcome: NDArray[Any],
        treatment: NDArray[Any],
        propensity_scores: NDArray[Any],
        covariates: CovariateData | None = None,
    ) -> dict[str, Any]:
        """Perform grid search optimization.

        Args:
            outcome: Outcome values
            treatment: Treatment values
            propensity_scores: Propensity scores
            covariates: Covariate data

        Returns:
            Optimization results dictionary
        """
        min_param, max_param = self.parameter_range
        param_grid = np.linspace(min_param, max_param, self.n_grid_points)

        best_objective = float("inf")
        best_param = None
        objective_values = []

        for param in param_grid:
            objective_val = self._objective_function(
                np.array([param]), outcome, treatment, propensity_scores, covariates
            )
            objective_values.append(objective_val)

            if objective_val < best_objective:
                best_objective = objective_val
                best_param = param

        return {
            "success": best_param is not None,
            "converged": True,
            "best_parameter": best_param,
            "objective_value": best_objective,
            "param_grid": param_grid,
            "objective_values": np.array(objective_values),
            "method": "grid_search",
        }

    def _root_finding_optimization(
        self,
        outcome: NDArray[Any],
        treatment: NDArray[Any],
        propensity_scores: NDArray[Any],
        covariates: CovariateData | None = None,
    ) -> dict[str, Any]:
        """Perform root finding optimization using Brent's method.

        Args:
            outcome: Outcome values
            treatment: Treatment values
            propensity_scores: Propensity scores
            covariates: Covariate data

        Returns:
            Optimization results dictionary
        """

        def objective_func(param: float) -> float:
            obj_val = self._objective_function(
                np.array([param]), outcome, treatment, propensity_scores, covariates
            )
            # The objective function already returns squared difference,
            # so for root finding we want the difference itself (not square root)
            # Take square root to get the actual difference for root finding
            return float(np.sqrt(obj_val) if obj_val >= 0 else -np.sqrt(-obj_val))

        try:
            min_param, max_param = self.parameter_range

            # Use Brent's method for root finding
            result = optimize.brentq(
                objective_func,
                min_param,
                max_param,
                xtol=1e-6,
                maxiter=1000,
            )

            return {
                "success": True,
                "converged": True,
                "best_parameter": result,
                "objective_value": objective_func(result) ** 2,
                "method": "root_finding",
            }

        except ValueError as e:
            # Root finding failed - fall back to grid search
            if self.verbose:
                print(f"Root finding failed: {e}. Falling back to grid search.")
            return self._grid_search_optimization(
                outcome, treatment, propensity_scores, covariates
            )

    def _gradient_optimization(
        self,
        outcome: NDArray[Any],
        treatment: NDArray[Any],
        propensity_scores: NDArray[Any],
        covariates: CovariateData | None = None,
    ) -> dict[str, Any]:
        """Perform gradient-based optimization.

        Args:
            outcome: Outcome values
            treatment: Treatment values
            propensity_scores: Propensity scores
            covariates: Covariate data

        Returns:
            Optimization results dictionary
        """
        # Determine number of parameters
        n_params = 1 + len(self.covariates_for_interaction)

        # Initial guess - start from center of parameter range
        initial_guess = np.zeros(n_params)
        center = sum(self.parameter_range) / 2
        initial_guess[0] = center

        def objective_func(params: NDArray[Any]) -> float:
            return self._objective_function(
                params, outcome, treatment, propensity_scores, covariates
            )

        try:
            # Use bounded optimization
            bounds = [self.parameter_range] * n_params

            result = optimize.minimize(
                objective_func,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000},
            )

            # Extract parameter names and values
            best_params = {}
            best_params["main_effect"] = result.x[0]
            for i, cov_name in enumerate(self.covariates_for_interaction):
                if i + 1 < len(result.x):
                    best_params[f"{cov_name}_interaction"] = result.x[i + 1]

            return {
                "success": result.success,
                "converged": result.success,
                "best_parameter": result.x[0],  # Main effect for backward compatibility
                "best_parameters": best_params,
                "objective_value": result.fun,
                "method": "gradient",
                "scipy_result": result,
            }

        except Exception as e:
            if self.verbose:
                print(
                    f"Gradient optimization failed: {e}. Falling back to grid search."
                )
            return self._grid_search_optimization(
                outcome, treatment, propensity_scores, covariates
            )

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the G-estimation model.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Covariate data for adjustment
        """
        if treatment.treatment_type != "binary":
            raise EstimationError(
                "G-estimation currently only supports binary treatments"
            )

        # Fit propensity score model
        self._fit_propensity_model(treatment, covariates)

        # Prepare data for optimization
        outcome_values = np.asarray(outcome.values)
        treatment_values = np.asarray(treatment.values)

        if self.propensity_scores is None:
            raise EstimationError("Propensity scores not available")

        # Perform optimization
        if self.optimization_method == "grid_search":
            self.optimization_result = self._grid_search_optimization(
                outcome_values, treatment_values, self.propensity_scores, covariates
            )
        elif self.optimization_method == "root_finding":
            self.optimization_result = self._root_finding_optimization(
                outcome_values, treatment_values, self.propensity_scores, covariates
            )
        elif self.optimization_method == "gradient":
            self.optimization_result = self._gradient_optimization(
                outcome_values, treatment_values, self.propensity_scores, covariates
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

        # Extract estimated parameters
        if self.optimization_result and self.optimization_result["success"]:
            if "best_parameters" in self.optimization_result:
                self.estimated_parameters = self.optimization_result["best_parameters"]
            else:
                self.estimated_parameters = {
                    "main_effect": self.optimization_result["best_parameter"]
                }

            if self.verbose:
                print(
                    f"G-estimation converged: {self.optimization_result['converged']}"
                )
                print(
                    f"Best parameter: {self.optimization_result['best_parameter']:.4f}"
                )
                print(
                    f"Objective value: {self.optimization_result['objective_value']:.8f}"
                )
        else:
            raise EstimationError("G-estimation optimization failed to converge")

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate Average Treatment Effect using G-estimation.

        Returns:
            CausalEffect object with ATE estimate and confidence intervals
        """
        if (
            self.optimization_result is None
            or self.estimated_parameters is None
            or not self.optimization_result["success"]
        ):
            raise EstimationError("G-estimation must converge before ATE estimation")

        # The main effect parameter is the ATE estimate
        ate = self.estimated_parameters["main_effect"]

        # Enhanced bootstrap confidence intervals
        bootstrap_result = None
        ate_se = None
        ate_ci_lower = None
        ate_ci_upper = None
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
        if self.treatment_data and self.treatment_data.treatment_type == "binary":
            n_treated = np.sum(self.treatment_data.values == 1)
            n_control = np.sum(self.treatment_data.values == 0)
        else:
            n_treated = len(self.treatment_data.values) if self.treatment_data else 0
            n_control = 0

        return CausalEffect(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            confidence_level=self.bootstrap_config.confidence_level
            if self.bootstrap_config
            else 0.95,
            method="G-estimation",
            n_observations=len(self.treatment_data.values)
            if self.treatment_data
            else 0,
            n_treated=n_treated,
            n_control=n_control,
            bootstrap_samples=self.bootstrap_config.n_samples
            if self.bootstrap_config
            else 0,
            bootstrap_estimates=bootstrap_estimates,
            diagnostics={
                "optimization_method": self.optimization_method,
                "converged": self.optimization_result["converged"],
                "objective_value": self.optimization_result["objective_value"],
                "structural_model": self.structural_model,
                "estimated_parameters": self.estimated_parameters,
            },
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

    def get_optimization_results(self) -> dict[str, Any] | None:
        """Get optimization results from G-estimation.

        Returns:
            Dictionary containing optimization details
        """
        return self.optimization_result

    def get_estimated_parameters(self) -> dict[str, float] | None:
        """Get estimated parameters from G-estimation.

        Returns:
            Dictionary of parameter names and estimated values
        """
        return self.estimated_parameters

    def rank_preservation_test(
        self, n_bootstrap: int = 1000, alpha: float = 0.05
    ) -> dict[str, Any]:
        """Perform rank preservation test for model validity.

        The rank preservation test checks whether the structural model
        preserves the rank ordering of potential outcomes.

        Args:
            n_bootstrap: Number of bootstrap samples for test
            alpha: Significance level for test

        Returns:
            Dictionary with test results
        """
        if not self.is_fitted or self.estimated_parameters is None:
            raise EstimationError("Model must be fitted before rank preservation test")

        if (
            self.treatment_data is None
            or self.outcome_data is None
            or self.covariate_data is None
        ):
            raise EstimationError("Data required for rank preservation test")

        # Apply structural model to get adjusted outcomes
        outcome_values = np.asarray(self.outcome_data.values)
        treatment_values = np.asarray(self.treatment_data.values)

        adjusted_outcomes = self._apply_structural_model(
            outcome_values,
            treatment_values,
            self.estimated_parameters,
            self.covariate_data,
        )

        # Test statistic: correlation between original and adjusted outcomes
        # within treatment groups
        treated_mask = treatment_values == 1
        control_mask = treatment_values == 0

        if np.sum(treated_mask) < 2 or np.sum(control_mask) < 2:
            return {
                "test_statistic": np.nan,
                "p_value": np.nan,
                "conclusion": "Insufficient data for rank preservation test",
            }

        # Spearman rank correlation within groups
        treated_corr = stats.spearmanr(
            outcome_values[treated_mask], adjusted_outcomes[treated_mask]
        )[0]
        control_corr = stats.spearmanr(
            outcome_values[control_mask], adjusted_outcomes[control_mask]
        )[0]

        # Test statistic is average correlation
        test_statistic = (treated_corr + control_corr) / 2

        # Bootstrap test for significance
        bootstrap_stats: list[float] = []
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(
                len(outcome_values), size=len(outcome_values), replace=True
            )

            boot_outcome = outcome_values[boot_indices]
            boot_treatment = treatment_values[boot_indices]

            # Apply structural model
            boot_adjusted = self._apply_structural_model(
                boot_outcome,
                boot_treatment,
                self.estimated_parameters,
                self.covariate_data,
            )

            # Calculate correlations
            boot_treated_mask = boot_treatment == 1
            boot_control_mask = boot_treatment == 0

            if np.sum(boot_treated_mask) >= 2 and np.sum(boot_control_mask) >= 2:
                boot_treated_corr = stats.spearmanr(
                    boot_outcome[boot_treated_mask], boot_adjusted[boot_treated_mask]
                )[0]
                boot_control_corr = stats.spearmanr(
                    boot_outcome[boot_control_mask], boot_adjusted[boot_control_mask]
                )[0]
                boot_stat = (boot_treated_corr + boot_control_corr) / 2
                bootstrap_stats.append(boot_stat)

        if len(bootstrap_stats) == 0:
            p_value = np.nan
        else:
            # Two-sided test: H0: correlation = 1 (perfect rank preservation)
            bootstrap_stats_array = np.array(bootstrap_stats)
            p_value = 2 * min(
                np.mean(bootstrap_stats_array <= test_statistic),
                np.mean(bootstrap_stats_array >= test_statistic),
            )

        # Conclusion
        if np.isnan(p_value):
            conclusion = "Test could not be completed"
        elif p_value < alpha:
            conclusion = f"Reject rank preservation (p = {p_value:.4f})"
        else:
            conclusion = f"Do not reject rank preservation (p = {p_value:.4f})"

        return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "conclusion": conclusion,
            "treated_correlation": treated_corr,
            "control_correlation": control_corr,
            "n_bootstrap": len(bootstrap_stats),
        }

    def compare_with_other_methods(
        self,
        treatment: TreatmentData | None = None,
        outcome: OutcomeData | None = None,
        covariates: CovariateData | None = None,
        methods: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare G-estimation results with other causal inference methods.

        Args:
            treatment: Treatment data (uses fitted data if None)
            outcome: Outcome data (uses fitted data if None)
            covariates: Covariate data (uses fitted data if None)
            methods: List of methods to compare ('g_computation', 'ipw', 'aipw')

        Returns:
            Dictionary with comparison results for each method
        """
        # Use fitted data if not provided
        treatment = treatment or self.treatment_data
        outcome = outcome or self.outcome_data
        covariates = covariates or self.covariate_data

        if not (treatment and outcome):
            raise EstimationError("Treatment and outcome data required for comparison")

        if methods is None:
            methods = ["g_computation", "ipw", "aipw"]

        results = {}

        # G-estimation results (current model)
        if self.is_fitted:
            g_est_effect = self.estimate_ate()
            results["g_estimation"] = {
                "ate": g_est_effect.ate,
                "ate_se": g_est_effect.ate_se,
                "ate_ci_lower": g_est_effect.ate_ci_lower,
                "ate_ci_upper": g_est_effect.ate_ci_upper,
                "method": g_est_effect.method,
            }

        # Compare with other methods
        for method in methods:
            try:
                if method == "g_computation":
                    from .g_computation import GComputationEstimator

                    g_comp_estimator = GComputationEstimator(
                        bootstrap_samples=0,
                        random_state=self.random_state,
                        verbose=False,
                    )
                    g_comp_estimator.fit(treatment, outcome, covariates)
                    effect = g_comp_estimator.estimate_ate()
                elif method == "ipw":
                    from .ipw import IPWEstimator

                    ipw_estimator = IPWEstimator(
                        bootstrap_samples=0,
                        random_state=self.random_state,
                        verbose=False,
                    )
                    ipw_estimator.fit(treatment, outcome, covariates)
                    effect = ipw_estimator.estimate_ate()
                elif method == "aipw":
                    from .aipw import AIPWEstimator

                    aipw_estimator = AIPWEstimator(
                        bootstrap_samples=0,
                        random_state=self.random_state,
                        verbose=False,
                    )
                    aipw_estimator.fit(treatment, outcome, covariates)
                    effect = aipw_estimator.estimate_ate()
                else:
                    if self.verbose:
                        print(f"Unknown method: {method}")
                    continue

                results[method] = {
                    "ate": effect.ate,
                    "ate_se": effect.ate_se,
                    "ate_ci_lower": effect.ate_ci_lower,
                    "ate_ci_upper": effect.ate_ci_upper,
                    "method": effect.method,
                }

            except Exception as e:
                if self.verbose:
                    print(f"Failed to fit {method}: {str(e)}")
                results[method] = {
                    "ate": np.nan,
                    "error": str(e),
                }

        return results

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
