"""Synthetic Control estimator for causal inference.

This module implements the Synthetic Control method for estimating
causal effects in single-unit or regional marketing interventions using
optimization over weights to create synthetic controls.
"""

from __future__ import annotations
# ruff: noqa: UP007

from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    DataValidationError,
    EstimationError,
    OutcomeData,
    TreatmentData,
)

__all__ = ["SyntheticControlEstimator", "SyntheticControlResult"]


class SyntheticControlResult(CausalEffect):
    """Extended causal effect result for Synthetic Control estimation.

    Includes synthetic control-specific outputs like weights, RMSPE,
    and pre/post trajectory data.
    """

    def __init__(
        self,
        ate: float,
        ate_ci_lower: Union[float, None] = None,
        ate_ci_upper: Union[float, None] = None,
        weights: Union[NDArray[Any], None] = None,
        rmspe_pre: Union[float, None] = None,
        rmspe_post: Union[float, None] = None,
        treated_trajectory: Union[NDArray[Any], None] = None,
        synthetic_trajectory: Union[NDArray[Any], None] = None,
        control_units: Union[list[Any], None] = None,
        intervention_period: Union[int, None] = None,
        optimization_converged: Union[bool, None] = None,
        optimization_objective: Union[float, None] = None,
        optimization_iterations: Union[int, None] = None,
        inference_method: str = "normal",
        **kwargs: Any,
    ) -> None:
        """Initialize Synthetic Control result.

        Args:
            ate: Average treatment effect
            ate_ci_lower: Lower bound of 95% confidence interval
            ate_ci_upper: Upper bound of 95% confidence interval
            weights: Weights assigned to control units for synthetic control
            rmspe_pre: Root Mean Squared Prediction Error in pre-intervention period
            rmspe_post: Root Mean Squared Prediction Error in post-intervention period
            treated_trajectory: Outcome trajectory for treated unit
            synthetic_trajectory: Outcome trajectory for synthetic control
            control_units: Names/identifiers of control units
            intervention_period: Time period when intervention occurred
            optimization_converged: Whether weight optimization converged successfully
            optimization_objective: Final objective function value from optimization
            optimization_iterations: Number of optimization iterations performed
            inference_method: Method used for confidence interval calculation
            **kwargs: Additional fields for parent class
        """
        super().__init__(
            ate=ate,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            method="Synthetic Control",
            **kwargs,
        )
        self.weights = weights
        self.rmspe_pre = rmspe_pre
        self.rmspe_post = rmspe_post
        self.treated_trajectory = treated_trajectory
        self.synthetic_trajectory = synthetic_trajectory
        self.control_units = control_units
        self.intervention_period = intervention_period
        self.optimization_converged = optimization_converged
        self.optimization_objective = optimization_objective
        self.optimization_iterations = optimization_iterations
        self.inference_method = inference_method

    def plot_trajectories(
        self,
        ax: Union[Any, None] = None,
        figsize: tuple[int, int] = (12, 8),
        show_intervention: bool = True,
    ) -> Any:
        """Plot outcome trajectories for treated unit vs synthetic control.

        Args:
            ax: Matplotlib axis (created if None)
            figsize: Figure size if creating new figure
            show_intervention: Whether to show intervention period as vertical line

        Returns:
            Matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if (
            self.treated_trajectory is not None
            and self.synthetic_trajectory is not None
        ):
            time_periods = np.arange(len(self.treated_trajectory))

            # Plot trajectories
            ax.plot(
                time_periods,
                self.treated_trajectory,
                "o-",
                label="Treated Unit",
                color="red",
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                time_periods,
                self.synthetic_trajectory,
                "o--",
                label="Synthetic Control",
                color="blue",
                linewidth=2,
                markersize=6,
                alpha=0.8,
            )

            # Show intervention period
            if show_intervention and self.intervention_period is not None:
                ax.axvline(
                    x=self.intervention_period - 0.5,
                    color="gray",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.7,
                    label="Intervention",
                )

            # Add pre/post shading
            if self.intervention_period is not None:
                ax.axvspan(
                    -0.5,
                    self.intervention_period - 0.5,
                    alpha=0.1,
                    color="blue",
                    label="Pre-Intervention",
                )
                ax.axvspan(
                    self.intervention_period - 0.5,
                    len(self.treated_trajectory) - 0.5,
                    alpha=0.1,
                    color="red",
                    label="Post-Intervention",
                )

        ax.set_xlabel("Time Period")
        ax.set_ylabel("Outcome")
        ax.set_title("Synthetic Control Analysis: Treated vs Synthetic")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_weights(
        self,
        ax: Union[Any, None] = None,
        figsize: tuple[int, int] = (10, 6),
        top_n: Union[int, None] = None,
    ) -> Any:
        """Plot weights assigned to control units.

        Args:
            ax: Matplotlib axis (created if None)
            figsize: Figure size if creating new figure
            top_n: Show only top N weighted units (None for all)

        Returns:
            Matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if self.weights is not None and self.control_units is not None:
            # Sort by weight
            sorted_indices = np.argsort(self.weights)[::-1]
            sorted_weights = self.weights[sorted_indices]
            sorted_units = [self.control_units[i] for i in sorted_indices]

            # Show only top N if specified
            if top_n is not None:
                sorted_weights = sorted_weights[:top_n]
                sorted_units = sorted_units[:top_n]

            # Filter out zero weights for cleaner visualization
            non_zero_mask = sorted_weights > 1e-6
            sorted_weights = sorted_weights[non_zero_mask]
            sorted_units = [
                unit for i, unit in enumerate(sorted_units) if non_zero_mask[i]
            ]

            # Create bar plot
            bars = ax.bar(range(len(sorted_weights)), sorted_weights, alpha=0.7)
            ax.set_xlabel("Control Units")
            ax.set_ylabel("Weight")
            ax.set_title("Synthetic Control Weights")
            ax.set_xticks(range(len(sorted_weights)))
            ax.set_xticklabels([str(unit) for unit in sorted_units], rotation=45)

            # Add weight values on bars
            for i, (bar, weight) in enumerate(zip(bars, sorted_weights)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{weight:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        return ax


class SyntheticControlEstimator(BaseEstimator):
    """Synthetic Control estimator for causal inference.

    The Synthetic Control method creates a synthetic version of the treated unit
    using a weighted combination of control units. The weights are optimized to
    minimize the distance between the treated unit and synthetic control in the
    pre-intervention period.

    Attributes:
        optimization_method: Method for weight optimization
        intervention_period: Time period when intervention occurred
        weights_: Fitted weights for control units
        control_units_: Names/identifiers of control units
        treated_trajectory_: Outcome trajectory for treated unit
        synthetic_trajectory_: Outcome trajectory for synthetic control
    """

    def __init__(
        self,
        intervention_period: int,
        optimization_method: str = "SLSQP",
        weight_penalty: float = 0.0,
        normalize_features: bool = True,
        inference_method: str = "normal",
        n_permutations: int = 1000,
        random_state: Union[int, None] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize Synthetic Control estimator.

        Args:
            intervention_period: Time period when intervention occurred (0-indexed)
            optimization_method: Optimization method for weight calculation
            weight_penalty: L2 penalty on weights to promote sparsity
            normalize_features: Whether to normalize features before optimization.
                Features are normalized across time periods (not units) to preserve
                unit-specific characteristics while standardizing temporal variation.
            inference_method: Method for confidence intervals ('normal', 'permutation')
            n_permutations: Number of permutations for permutation-based inference
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output during estimation
            **kwargs: Additional arguments for parent class
        """
        super().__init__(random_state=random_state, verbose=verbose, **kwargs)
        self.intervention_period = intervention_period
        self.optimization_method = optimization_method
        self.weight_penalty = weight_penalty
        self.normalize_features = normalize_features
        self.inference_method = inference_method
        self.n_permutations = n_permutations

        # Fitted attributes
        self.weights_: Union[NDArray[Any], None] = None
        self.control_units_: Union[list[Any], None] = None
        self.treated_trajectory_: Union[NDArray[Any], None] = None
        self.synthetic_trajectory_: Union[NDArray[Any], None] = None
        self._feature_means: Union[NDArray[Any], None] = None
        self._feature_stds: Union[NDArray[Any], None] = None
        self._optimization_result: Union[Any, None] = None

    def _validate_data(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Union[CovariateData, None] = None,
    ) -> None:
        """Validate input data for synthetic control analysis.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data

        Raises:
            DataValidationError: If data validation fails
        """
        # Check that we have panel data structure
        if not isinstance(outcome.values, pd.DataFrame):
            raise DataValidationError(
                "Synthetic Control requires panel data structure. "
                "Outcome should be a DataFrame with units as rows and time as columns."
            )

        # Check intervention period
        n_periods = outcome.values.shape[1]
        if self.intervention_period >= n_periods:
            raise DataValidationError(
                f"Intervention period ({self.intervention_period}) must be less than "
                f"number of time periods ({n_periods})"
            )

        if self.intervention_period < 1:
            raise DataValidationError(
                "Intervention period must be at least 1 (need pre-intervention data)"
            )

        # Check that we have treated and control units
        if isinstance(treatment.values, pd.Series):
            unique_treatments = treatment.values.unique()
        else:
            unique_treatments = np.unique(treatment.values)

        if len(unique_treatments) != 2:
            raise DataValidationError(
                f"Synthetic Control requires exactly 2 treatment groups. "
                f"Found {len(unique_treatments)} groups: {unique_treatments}"
            )

        # Check that we have exactly one treated unit
        treated_mask = treatment.values == 1
        n_treated = np.sum(treated_mask)
        if n_treated != 1:
            raise DataValidationError(
                f"Synthetic Control requires exactly 1 treated unit. Found {n_treated}"
            )

        # Check that we have multiple control units
        n_control = np.sum(treatment.values == 0)
        if n_control < 2:
            raise DataValidationError(
                f"Synthetic Control requires at least 2 control units. Found {n_control}"
            )

    def _prepare_data(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Union[CovariateData, None] = None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], list[Any]]:
        """Prepare data for synthetic control analysis.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data

        Returns:
            Tuple of (treated_pre, control_pre, treated_trajectory, control_units)
        """
        # Extract treatment assignment
        if isinstance(treatment.values, pd.Series):
            treatment_array = treatment.values.values
        else:
            treatment_array = np.array(treatment.values)

        # Get treated and control unit indices
        treated_idx = np.where(treatment_array == 1)[0]
        control_idx = np.where(treatment_array == 0)[0]

        # Extract outcome data
        outcome_df = outcome.values
        if not isinstance(outcome_df, pd.DataFrame):
            raise DataValidationError("Outcome must be a DataFrame for panel data")

        # Get treated and control outcomes
        treated_trajectory = outcome_df.iloc[treated_idx[0]].values
        control_trajectories = outcome_df.iloc[control_idx].values

        # Pre-intervention period data
        treated_pre = treated_trajectory[: self.intervention_period]
        control_pre = control_trajectories[:, : self.intervention_period]

        # Control unit identifiers
        control_units = outcome_df.index[control_idx].tolist()

        return treated_pre, control_pre, treated_trajectory, control_units

    def _optimize_weights(
        self,
        treated_pre: NDArray[Any],
        control_pre: NDArray[Any],
    ) -> NDArray[Any]:
        """Optimize weights for synthetic control.

        Args:
            treated_pre: Pre-intervention outcomes for treated unit
            control_pre: Pre-intervention outcomes for control units

        Returns:
            Optimized weights array
        """
        n_control = control_pre.shape[0]

        def objective(weights: NDArray[Any]) -> float:
            """Objective function to minimize: squared prediction error."""
            synthetic_pre = np.dot(weights, control_pre)
            mse = float(np.mean((treated_pre - synthetic_pre) ** 2))

            # Add L2 penalty on weights if specified
            if self.weight_penalty > 0:
                mse += float(self.weight_penalty * np.sum(weights**2))

            return mse

        # Constraints: weights sum to 1 and are non-negative
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_control)]

        # Initial guess: equal weights
        initial_weights = np.ones(n_control) / n_control

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method=self.optimization_method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        # Store optimization result for diagnostics
        self._optimization_result = result

        if not result.success:
            warning_msg = (
                f"Optimization did not converge: {result.message}. "
                f"Final objective value: {result.fun:.6f}, "
                f"Iterations: {result.nit}, "
                f"Status: {result.status}"
            )
            if self.verbose:
                print(f"Warning: {warning_msg}")

        return np.asarray(result.x)

    def _calculate_synthetic_trajectory(
        self,
        weights: NDArray[Any],
        control_trajectories: NDArray[Any],
    ) -> NDArray[Any]:
        """Calculate synthetic control trajectory using optimized weights.

        Args:
            weights: Optimized weights
            control_trajectories: Control unit trajectories (n_control x n_periods)

        Returns:
            Synthetic control trajectory
        """
        return np.asarray(np.dot(weights, control_trajectories))

    def _calculate_rmspe(
        self,
        treated: NDArray[Any],
        synthetic: NDArray[Any],
    ) -> float:
        """Calculate Root Mean Squared Prediction Error.

        Args:
            treated: Treated unit outcomes
            synthetic: Synthetic control outcomes

        Returns:
            RMSPE value
        """
        return float(np.sqrt(mean_squared_error(treated, synthetic)))

    def _permutation_inference(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        ate_observed: float,
    ) -> tuple[Union[float, None], Union[float, None]]:
        """Perform permutation-based inference for confidence intervals.

        Following Abadie et al. (2010) approach, this method permutes the treatment
        assignment across control units to generate a null distribution.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            ate_observed: Observed ATE to compare against null distribution

        Returns:
            Tuple of (lower_ci, upper_ci) based on permutation test
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Get control unit indices
        treatment_array = (
            treatment.values.values
            if isinstance(treatment.values, pd.Series)
            else np.array(treatment.values)
        )
        control_indices = np.where(treatment_array == 0)[0]

        # Store original data
        outcome_df = outcome.values
        if not isinstance(outcome_df, pd.DataFrame):
            raise ValueError(
                "Outcome data must be a DataFrame for permutation inference"
            )
        permutation_effects: list[float] = []

        for perm in range(self.n_permutations):
            # Randomly select a control unit to be "treated"
            if perm == 0:
                # First permutation uses original treated unit for comparison
                fake_treated_idx = np.where(treatment_array == 1)[0][0]
            else:
                fake_treated_idx = np.random.choice(control_indices)

            # Create fake treatment assignment
            fake_treatment = np.zeros_like(treatment_array)
            fake_treatment[fake_treated_idx] = 1

            # Get remaining control units
            fake_control_indices = np.where(fake_treatment == 0)[0]

            if len(fake_control_indices) < 2:
                continue  # Need at least 2 control units

            # Extract trajectories
            fake_treated_trajectory = outcome_df.iloc[fake_treated_idx].values
            fake_control_trajectories = outcome_df.iloc[fake_control_indices].values

            # Pre-intervention data for weight optimization
            fake_treated_pre = fake_treated_trajectory[: self.intervention_period]
            fake_control_pre = fake_control_trajectories[:, : self.intervention_period]

            try:
                # Optimize weights for this permutation
                if (
                    self.normalize_features
                    and self._feature_means is not None
                    and self._feature_stds is not None
                ):
                    fake_control_pre_norm = (
                        fake_control_pre - self._feature_means
                    ) / self._feature_stds
                    fake_treated_pre_norm = (
                        fake_treated_pre - self._feature_means.flatten()
                    ) / self._feature_stds.flatten()
                else:
                    fake_control_pre_norm = fake_control_pre
                    fake_treated_pre_norm = fake_treated_pre

                fake_weights = self._optimize_weights(
                    fake_treated_pre_norm, fake_control_pre_norm
                )

                # Calculate fake synthetic trajectory
                fake_synthetic_trajectory = self._calculate_synthetic_trajectory(
                    fake_weights, fake_control_trajectories
                )

                # Calculate fake treatment effect
                fake_post_treated = fake_treated_trajectory[self.intervention_period :]
                fake_post_synthetic = fake_synthetic_trajectory[
                    self.intervention_period :
                ]
                fake_ate = np.mean(fake_post_treated - fake_post_synthetic)
                permutation_effects.append(fake_ate)

            except Exception:
                # Skip failed optimizations
                continue

        if len(permutation_effects) < 100:  # Minimum for reasonable inference
            if self.verbose:
                print("Warning: Too few successful permutations for reliable inference")
            return None, None

        # Calculate p-value and confidence interval
        permutation_effects_array = np.array(permutation_effects)

        # Two-sided confidence interval
        alpha = 0.05  # For 95% CI
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = float(np.percentile(permutation_effects_array, lower_percentile))
        ci_upper = float(np.percentile(permutation_effects_array, upper_percentile))

        return ci_lower, ci_upper

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Union[CovariateData, None] = None,
    ) -> None:
        """Fit the Synthetic Control estimator.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data
        """
        # Validate data
        self._validate_data(treatment, outcome, covariates)

        # Prepare data
        treated_pre, control_pre, treated_trajectory, control_units = (
            self._prepare_data(treatment, outcome, covariates)
        )

        # Normalize features if requested (normalize across time periods, not units)
        if self.normalize_features:
            # Normalize across time periods (axis=1) for each unit
            all_pre_data = np.vstack([treated_pre.reshape(1, -1), control_pre])
            self._feature_means = np.mean(all_pre_data, axis=0, keepdims=True)
            self._feature_stds = np.std(all_pre_data, axis=0, keepdims=True)
            # Avoid division by zero
            self._feature_stds = np.where(
                self._feature_stds == 0, 1, self._feature_stds
            )
            control_pre_norm = (control_pre - self._feature_means) / self._feature_stds
            treated_pre_norm = (
                treated_pre - self._feature_means.flatten()
            ) / self._feature_stds.flatten()
        else:
            control_pre_norm = control_pre
            treated_pre_norm = treated_pre

        # Optimize weights
        weights = self._optimize_weights(treated_pre_norm, control_pre_norm)

        # Calculate synthetic trajectory
        outcome_df = outcome.values
        if not isinstance(outcome_df, pd.DataFrame):
            raise DataValidationError("Outcome must be a DataFrame for panel data")
        control_trajectories = outcome_df.iloc[
            np.where(treatment.values == 0)[0]
        ].values
        synthetic_trajectory = self._calculate_synthetic_trajectory(
            weights, control_trajectories
        )

        # Store results
        self.weights_ = weights
        self.control_units_ = control_units
        self.treated_trajectory_ = treated_trajectory
        self.synthetic_trajectory_ = synthetic_trajectory

        if self.verbose:
            rmspe_pre = self._calculate_rmspe(
                treated_trajectory[: self.intervention_period],
                synthetic_trajectory[: self.intervention_period],
            )
            print(f"Pre-intervention RMSPE: {rmspe_pre:.4f}")
            print(
                f"Number of control units with non-zero weights: {np.sum(weights > 1e-6)}"
            )
            print(f"Maximum weight: {np.max(weights):.4f}")

    def _estimate_ate_implementation(self) -> SyntheticControlResult:
        """Estimate the average treatment effect using synthetic control.

        Returns:
            SyntheticControlResult with causal effect estimates
        """
        if (
            self.weights_ is None
            or self.treated_trajectory_ is None
            or self.synthetic_trajectory_ is None
        ):
            raise EstimationError("Model must be fitted before estimation")

        # Calculate post-intervention effects
        post_treated = self.treated_trajectory_[self.intervention_period :]
        post_synthetic = self.synthetic_trajectory_[self.intervention_period :]
        post_effects = post_treated - post_synthetic

        # Average treatment effect (mean of post-intervention effects)
        ate = np.mean(post_effects)

        # Calculate RMSPEs
        rmspe_pre = self._calculate_rmspe(
            self.treated_trajectory_[: self.intervention_period],
            self.synthetic_trajectory_[: self.intervention_period],
        )
        rmspe_post = self._calculate_rmspe(post_treated, post_synthetic)

        # Calculate confidence intervals based on inference method
        if self.inference_method == "permutation":
            if self.verbose:
                print(
                    f"Computing permutation-based confidence intervals with {self.n_permutations} permutations..."
                )

            if self.treatment_data is not None and self.outcome_data is not None:
                ate_ci_lower, ate_ci_upper = self._permutation_inference(
                    self.treatment_data, self.outcome_data, ate
                )
            else:
                ate_ci_lower, ate_ci_upper = None, None

            # Fallback to normal approximation if permutation fails
            if ate_ci_lower is None or ate_ci_upper is None:
                if self.verbose:
                    print(
                        "Permutation inference failed, falling back to normal approximation"
                    )
                ate_se = np.std(post_effects) / np.sqrt(len(post_effects))
                ate_ci_lower = ate - 1.96 * ate_se
                ate_ci_upper = ate + 1.96 * ate_se
                inference_method = "normal"
            else:
                inference_method = "permutation"
        else:
            # Normal approximation (default)
            ate_se = np.std(post_effects) / np.sqrt(len(post_effects))
            ate_ci_lower = ate - 1.96 * ate_se
            ate_ci_upper = ate + 1.96 * ate_se
            inference_method = "normal"

        # Extract optimization diagnostics
        optimization_converged = (
            bool(self._optimization_result.success)
            if self._optimization_result
            else None
        )
        optimization_objective = (
            self._optimization_result.fun if self._optimization_result else None
        )
        optimization_iterations = (
            self._optimization_result.nit if self._optimization_result else None
        )

        return SyntheticControlResult(
            ate=ate,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            weights=self.weights_,
            rmspe_pre=rmspe_pre,
            rmspe_post=rmspe_post,
            treated_trajectory=self.treated_trajectory_,
            synthetic_trajectory=self.synthetic_trajectory_,
            control_units=self.control_units_,
            intervention_period=self.intervention_period,
            optimization_converged=optimization_converged,
            optimization_objective=optimization_objective,
            optimization_iterations=optimization_iterations,
            inference_method=inference_method,
            n_observations=len(self.treated_trajectory_),
        )

    def predict_counterfactual(
        self,
        new_periods: int,
    ) -> NDArray[Any]:
        """Predict counterfactual outcomes for additional periods.

        Args:
            new_periods: Number of additional periods to predict

        Returns:
            Predicted counterfactual outcomes

        Raises:
            EstimationError: If estimator is not fitted
        """
        if self.weights_ is None:
            raise EstimationError("Model must be fitted before prediction")

        # This is a simplified implementation
        # In practice, you'd need additional control unit data for new periods
        raise NotImplementedError(
            "Counterfactual prediction for future periods requires additional data"
        )
