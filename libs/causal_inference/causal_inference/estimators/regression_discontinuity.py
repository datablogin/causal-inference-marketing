"""Regression Discontinuity Design (RDD) estimator for causal inference.

This module implements RDD, a quasi-experimental design that estimates causal
effects by exploiting discontinuous changes in treatment assignment at a cutoff
threshold of a forcing variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from ..core.bootstrap import BootstrapConfig, BootstrapMixin


@dataclass
class ForcingVariableData:
    """Data model for forcing (running) variables in RDD.

    The forcing variable determines treatment assignment based on a cutoff threshold.
    Units above/below the cutoff receive different treatments.
    """

    values: pd.Series | NDArray[Any]
    name: str = "forcing_variable"
    cutoff: float = 0.0

    def __post_init__(self) -> None:
        """Validate forcing variable data after initialization."""
        if len(self.values) == 0:
            raise ValueError("Forcing variable values cannot be empty")

        # Check for variation around cutoff
        values_array = np.array(self.values)
        n_below = np.sum(values_array < self.cutoff)
        n_above = np.sum(values_array >= self.cutoff)

        if n_below == 0 or n_above == 0:
            raise ValueError(
                f"Forcing variable must have observations both above and below cutoff {self.cutoff}. "
                f"Found {n_below} below and {n_above} above cutoff."
            )

    @property
    def treatment_assignment(self) -> NDArray[Any]:
        """Generate treatment assignment based on cutoff rule."""
        return np.array(self.values >= self.cutoff, dtype=int)

    @property
    def centered_values(self) -> NDArray[Any]:
        """Get forcing variable values centered around the cutoff."""
        return np.array(self.values) - self.cutoff


@dataclass
class RDDResult:
    """Results from RDD estimation."""

    ate: float
    ate_se: Optional[float] = None
    ate_ci_lower: Optional[float] = None
    ate_ci_upper: Optional[float] = None
    confidence_level: float = 0.95

    # RDD-specific diagnostics
    cutoff: float = 0.0
    bandwidth: Optional[float] = None
    n_left: int = 0  # Observations to the left of cutoff
    n_right: int = 0  # Observations to the right of cutoff

    # Model diagnostics
    left_model_r2: Optional[float] = None
    right_model_r2: Optional[float] = None
    placebo_test_pvalue: Optional[float] = None
    density_test_pvalue: Optional[float] = None

    # Polynomial order used
    polynomial_order: int = 1

    def to_causal_effect(self) -> CausalEffect:
        """Convert RDD results to standard CausalEffect format."""
        return CausalEffect(
            ate=self.ate,
            ate_se=self.ate_se,
            ate_ci_lower=self.ate_ci_lower,
            ate_ci_upper=self.ate_ci_upper,
            confidence_level=self.confidence_level,
            method="Regression Discontinuity Design",
            n_observations=self.n_left + self.n_right,
            diagnostics={
                "cutoff": self.cutoff,
                "bandwidth": self.bandwidth,
                "n_left": self.n_left,
                "n_right": self.n_right,
                "left_model_r2": self.left_model_r2,
                "right_model_r2": self.right_model_r2,
                "polynomial_order": self.polynomial_order,
                "placebo_test_pvalue": self.placebo_test_pvalue,
                "density_test_pvalue": self.density_test_pvalue,
            },
        )


class RDDEstimator(BootstrapMixin, BaseEstimator):
    """Regression Discontinuity Design estimator for causal inference.

    RDD estimates causal effects by exploiting discontinuous changes in treatment
    assignment at a cutoff threshold. The method compares outcomes just above and
    below the cutoff, where treatment assignment changes discontinuously.

    Key assumptions:
    1. Units cannot precisely manipulate the forcing variable around the cutoff
    2. All other factors change smoothly through the cutoff
    3. Treatment assignment changes discontinuously at the cutoff

    Attributes:
        cutoff: The threshold value for treatment assignment
        bandwidth: Window around cutoff for local estimation
        polynomial_order: Order of polynomial for flexible regression
        kernel: Kernel type for local regression ('uniform', 'triangular')
    """

    def __init__(
        self,
        cutoff: float = 0.0,
        bandwidth: Optional[float] = None,
        polynomial_order: int = 1,
        kernel: str = "uniform",
        robust_se: str = "HC2",
        bootstrap_config: Optional[BootstrapConfig] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the RDD estimator.

        Args:
            cutoff: Threshold value for treatment assignment
            bandwidth: Window around cutoff for local estimation (auto if None)
            polynomial_order: Order of polynomial for flexible regression
            kernel: Kernel type for local regression
            robust_se: Type of robust standard errors ('HC0', 'HC1', 'HC2', 'HC3')
            bootstrap_config: Configuration for bootstrap confidence intervals
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        super().__init__(
            bootstrap_config=bootstrap_config,
            random_state=random_state,
            verbose=verbose,
        )

        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.polynomial_order = polynomial_order
        self.kernel = kernel
        self.robust_se = robust_se

        # Model storage
        self.forcing_variable_data: Optional[ForcingVariableData] = None
        self.left_model: Optional[LinearRegression] = None
        self.right_model: Optional[LinearRegression] = None
        self._rdd_result: Optional[RDDResult] = None

        # Feature caching for efficiency
        self._cached_left_features: Optional[NDArray[Any]] = None
        self._cached_right_features: Optional[NDArray[Any]] = None
        self._cached_left_x: Optional[NDArray[Any]] = None
        self._cached_right_x: Optional[NDArray[Any]] = None

    def _create_bootstrap_estimator(
        self, random_state: Optional[int] = None
    ) -> RDDEstimator:
        """Create a new estimator instance for bootstrap sampling."""
        return RDDEstimator(
            cutoff=self.cutoff,
            bandwidth=self.bandwidth,
            polynomial_order=self.polynomial_order,
            kernel=self.kernel,
            robust_se=self.robust_se,
            bootstrap_config=BootstrapConfig(n_samples=0),  # No nested bootstrap
            random_state=random_state,
            verbose=False,
        )

    def _optimal_bandwidth_imbens_kalyanaraman(
        self, forcing_var: NDArray[Any], outcome: NDArray[Any]
    ) -> float:
        """Calculate optimal bandwidth using Imbens-Kalyanaraman (IK) method.

        This is a simplified version of the IK bandwidth selector.

        Args:
            forcing_var: Forcing variable values
            outcome: Outcome values

        Returns:
            Optimal bandwidth
        """
        # Center forcing variable around cutoff
        x_centered = forcing_var - self.cutoff

        # Calculate basic statistics
        n = len(x_centered)
        x_range = float(np.max(x_centered) - np.min(x_centered))

        # Use rule-of-thumb bandwidth as starting point
        # IK method uses more sophisticated approach, but this provides reasonable estimate
        h_rot = 1.84 * float(np.std(x_centered)) * (n ** (-1 / 5))

        # Scale based on data range and ensure reasonable bounds
        optimal_bw = max(min(h_rot, x_range / 4), x_range / 20)

        return float(optimal_bw)

    def _apply_kernel_weights(
        self, distances: NDArray[Any], bandwidth: float
    ) -> NDArray[Any]:
        """Apply kernel weights based on distance from cutoff."""
        if self.kernel == "uniform":
            return np.ones_like(distances)
        elif self.kernel == "triangular":
            weights = np.maximum(0, 1 - np.abs(distances) / bandwidth)
            return np.asarray(weights)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")

    def _fit_local_polynomial(
        self,
        x: NDArray[Any],
        y: NDArray[Any],
        weights: Optional[NDArray[Any]] = None,
    ) -> LinearRegression:
        """Fit local polynomial regression.

        Args:
            x: Forcing variable values (centered)
            y: Outcome values
            weights: Sample weights

        Returns:
            Fitted LinearRegression model
        """
        # Create polynomial features
        if self.polynomial_order > 1:
            poly_features = PolynomialFeatures(
                degree=self.polynomial_order, include_bias=True
            )
            X = poly_features.fit_transform(x.reshape(-1, 1))
        else:
            # Linear case - include intercept
            X = np.column_stack([np.ones(len(x)), x])

        # Fit weighted regression
        model = LinearRegression(fit_intercept=False)  # We handle intercept manually
        model.fit(X, y, sample_weight=weights)

        return model

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: Optional[CovariateData] = None,
    ) -> None:
        """Fit the RDD estimator.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data (for robustness checks)
        """
        # RDD requires a forcing variable to be provided as treatment data
        # The actual treatment assignment is derived from the forcing variable and cutoff

        # Validate forcing variable data types
        if isinstance(treatment.values, (pd.Series, np.ndarray)):  # noqa: UP038
            forcing_values = np.array(treatment.values)
        else:
            raise EstimationError(
                f"Forcing variable must be pandas Series or numpy array, got {type(treatment.values)}"
            )

        # Validate forcing variable is numeric
        if not np.issubdtype(forcing_values.dtype, np.number):
            raise EstimationError(
                f"Forcing variable must be numeric, got dtype {forcing_values.dtype}"
            )

        # Check for missing values
        if np.any(np.isnan(forcing_values)):
            raise EstimationError(
                "Forcing variable cannot contain missing values (NaN)"
            )

        # Create forcing variable data structure
        self.forcing_variable_data = ForcingVariableData(
            values=forcing_values,
            name=treatment.name,
            cutoff=self.cutoff,
        )

        # Get outcome values
        y = np.array(outcome.values)

        # Determine bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._optimal_bandwidth_imbens_kalyanaraman(
                forcing_values, y
            )
            if self.verbose:
                print(f"Using optimal bandwidth: {self.bandwidth:.4f}")

        # Center forcing variable around cutoff
        x_centered = self.forcing_variable_data.centered_values

        # Select observations within bandwidth
        within_bandwidth = np.abs(x_centered) <= self.bandwidth
        x_bw = x_centered[within_bandwidth]
        y_bw = y[within_bandwidth]

        if len(x_bw) < 10:
            raise EstimationError(
                f"Too few observations within bandwidth ({len(x_bw)}). "
                f"Consider increasing bandwidth or checking data."
            )

        # Split into left and right of cutoff
        left_mask = x_bw < 0
        right_mask = x_bw >= 0

        x_left = x_bw[left_mask]
        y_left = y_bw[left_mask]
        x_right = x_bw[right_mask]
        y_right = y_bw[right_mask]

        if len(x_left) < 3 or len(x_right) < 3:
            raise EstimationError(
                f"Too few observations on one side of cutoff. "
                f"Left: {len(x_left)}, Right: {len(x_right)}"
            )

        # Apply kernel weights
        weights_left = self._apply_kernel_weights(np.abs(x_left), self.bandwidth)
        weights_right = self._apply_kernel_weights(np.abs(x_right), self.bandwidth)

        # Cache the training data features and fit local polynomial models
        left_features = self._get_cached_features(x_left, "left")
        right_features = self._get_cached_features(x_right, "right")

        # Fit local polynomial models on each side
        try:
            self.left_model = self._fit_local_polynomial(x_left, y_left, weights_left)
            self.right_model = self._fit_local_polynomial(
                x_right, y_right, weights_right
            )
        except Exception as e:
            raise EstimationError(
                f"Failed to fit local polynomial models: {str(e)}"
            ) from e

        if self.verbose:
            left_r2 = self.left_model.score(left_features, y_left, weights_left)
            right_r2 = self.right_model.score(right_features, y_right, weights_right)
            print(f"Left model R²: {left_r2:.4f}")
            print(f"Right model R²: {right_r2:.4f}")

    def _compute_robust_se(
        self,
        design_matrix: NDArray[Any],
        residuals: NDArray[Any],
        model: LinearRegression,
        robust_type: str = "HC2",
    ) -> NDArray[Any]:
        """Compute robust (heteroskedasticity-consistent) standard errors.

        Args:
            design_matrix: Design matrix
            residuals: Model residuals
            model: Fitted linear regression model
            robust_type: Type of robust SE ('HC0', 'HC1', 'HC2', 'HC3')

        Returns:
            Array of robust standard errors for model coefficients
        """
        n, k = design_matrix.shape

        # Get hat matrix diagonal (leverage values)
        try:
            # X(X'X)^(-1)X' diagonal elements
            XTX_inv = np.linalg.inv(design_matrix.T @ design_matrix)
            hat_diag = np.diag(design_matrix @ XTX_inv @ design_matrix.T)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse for singular matrices
            XTX_pinv = np.linalg.pinv(design_matrix.T @ design_matrix)
            hat_diag = np.diag(design_matrix @ XTX_pinv @ design_matrix.T)

        # Compute robust variance adjustment based on type
        if robust_type == "HC0":
            # White's original heteroskedasticity-consistent estimator
            weights = np.ones(n)
        elif robust_type == "HC1":
            # Degrees of freedom adjustment
            weights = np.full(n, n / (n - k))
        elif robust_type == "HC2":
            # MacKinnon & White (1985) - recommended for moderate samples
            weights = 1 / (1 - hat_diag)
        elif robust_type == "HC3":
            # MacKinnon & White (1985) - more conservative, better for small samples
            weights = 1 / ((1 - hat_diag) ** 2)
        else:
            raise ValueError(f"Unknown robust standard error type: {robust_type}")

        # Compute robust covariance matrix
        weighted_residuals = residuals * np.sqrt(weights)
        Omega = np.diag(weighted_residuals**2)

        try:
            robust_cov = XTX_inv @ (design_matrix.T @ Omega @ design_matrix) @ XTX_inv
        except np.linalg.LinAlgError:
            robust_cov = XTX_pinv @ (design_matrix.T @ Omega @ design_matrix) @ XTX_pinv

        # Extract standard errors
        robust_se = np.sqrt(np.diag(robust_cov))

        return np.asarray(robust_se)

    def _get_cached_features(self, x: NDArray[Any], side: str) -> NDArray[Any]:
        """Get cached polynomial features or create and cache them.

        Args:
            x: Input values
            side: 'left' or 'right' to specify which cache to use

        Returns:
            Polynomial features matrix
        """
        if side == "left":
            if (
                self._cached_left_x is not None
                and self._cached_left_features is not None
                and np.array_equal(x, self._cached_left_x)
            ):
                return self._cached_left_features

            features = self._create_polynomial_features(x)
            self._cached_left_x = x.copy()
            self._cached_left_features = features
            return np.asarray(features)

        elif side == "right":
            if (
                self._cached_right_x is not None
                and self._cached_right_features is not None
                and np.array_equal(x, self._cached_right_x)
            ):
                return self._cached_right_features

            features = self._create_polynomial_features(x)
            self._cached_right_x = x.copy()
            self._cached_right_features = features
            return np.asarray(features)

        else:
            # For other cases (like plotting), don't cache
            return np.asarray(self._create_polynomial_features(x))

    def _create_polynomial_features(self, x: NDArray[Any]) -> NDArray[Any]:
        """Create polynomial features for prediction."""
        if self.polynomial_order > 1:
            poly_features = PolynomialFeatures(
                degree=self.polynomial_order, include_bias=True
            )
            features = poly_features.fit_transform(x.reshape(-1, 1))
            return np.asarray(features)
        else:
            features = np.column_stack([np.ones(len(x)), x])
            return np.asarray(features)

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate the treatment effect at the cutoff using RDD."""
        if (
            self.left_model is None
            or self.right_model is None
            or self.forcing_variable_data is None
            or self.outcome_data is None
        ):
            raise EstimationError("Models must be fitted before estimation")

        # Predict outcomes at the cutoff from both sides
        # At cutoff (x=0), we only need the intercept terms
        cutoff_features = self._create_polynomial_features(np.array([0.0]))

        y0_pred = self.left_model.predict(cutoff_features)[0]  # Limit from left
        y1_pred = self.right_model.predict(cutoff_features)[0]  # Limit from right

        # RDD estimate is the discontinuity at cutoff
        ate = y1_pred - y0_pred

        # Calculate standard error using residual-based approach
        x_centered = self.forcing_variable_data.centered_values
        within_bandwidth = np.abs(x_centered) <= self.bandwidth
        x_bw = x_centered[within_bandwidth]
        y_bw = np.array(self.outcome_data.values)[within_bandwidth]

        # Split observations
        left_mask = x_bw < 0
        right_mask = x_bw >= 0

        x_left = x_bw[left_mask]
        y_left = y_bw[left_mask]
        x_right = x_bw[right_mask]
        y_right = y_bw[right_mask]

        # Calculate residuals and standard errors using cached features
        try:
            left_features = self._get_cached_features(x_left, "left")
            right_features = self._get_cached_features(x_right, "right")

            left_pred = self.left_model.predict(left_features)
            right_pred = self.right_model.predict(right_features)

            left_residuals = y_left - left_pred
            right_residuals = y_right - right_pred

            # Compute robust standard errors for both models
            left_robust_se = self._compute_robust_se(
                left_features, left_residuals, self.left_model, self.robust_se
            )
            right_robust_se = self._compute_robust_se(
                right_features, right_residuals, self.right_model, self.robust_se
            )

            # Standard error of RDD estimate at cutoff (intercept term)
            # For RDD, we're interested in the SE of the discontinuity at x=0
            # This is the SE of the difference between intercepts
            left_intercept_se = left_robust_se[0]  # Intercept is first coefficient
            right_intercept_se = right_robust_se[0]

            # SE of difference (assuming independence between left and right models)
            ate_se = np.sqrt(left_intercept_se**2 + right_intercept_se**2)

            # Calculate confidence interval
            alpha = 1 - (
                self.bootstrap_config.confidence_level
                if self.bootstrap_config
                else 0.95
            )
            t_critical = stats.t.ppf(1 - alpha / 2, df=len(x_bw) - 4)
            ate_ci_lower = ate - t_critical * ate_se
            ate_ci_upper = ate + t_critical * ate_se

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not calculate standard errors: {str(e)}")
            ate_se = None
            ate_ci_lower = None
            ate_ci_upper = None

        # Store RDD-specific results using cached features
        left_r2 = None
        right_r2 = None
        try:
            # Reuse features already computed above
            left_r2 = self.left_model.score(left_features, y_left)
            right_r2 = self.right_model.score(right_features, y_right)
        except Exception:
            pass

        self._rdd_result = RDDResult(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            confidence_level=self.bootstrap_config.confidence_level
            if self.bootstrap_config
            else 0.95,
            cutoff=self.cutoff,
            bandwidth=self.bandwidth,
            n_left=len(x_left),
            n_right=len(x_right),
            left_model_r2=left_r2,
            right_model_r2=right_r2,
            polynomial_order=self.polynomial_order,
        )

        causal_effect = self._rdd_result.to_causal_effect()

        # Add bootstrap information if available
        if self.bootstrap_config and self.bootstrap_config.n_samples > 0:
            try:
                bootstrap_result = self.compute_bootstrap_confidence_intervals(ate)

                causal_effect.bootstrap_samples = int(self.bootstrap_config.n_samples)
                causal_effect.bootstrap_estimates = bootstrap_result.bootstrap_estimates
                causal_effect.bootstrap_method = bootstrap_result.config.method
                causal_effect.bootstrap_converged = bootstrap_result.converged
                causal_effect.bootstrap_bias = bootstrap_result.bias_estimate
                causal_effect.bootstrap_acceleration = (
                    bootstrap_result.acceleration_estimate
                )

                # Update confidence intervals with bootstrap results
                if bootstrap_result.config.method == "percentile":
                    causal_effect.ate_ci_lower = bootstrap_result.ci_lower_percentile
                    causal_effect.ate_ci_upper = bootstrap_result.ci_upper_percentile
                elif bootstrap_result.config.method == "bias_corrected":
                    causal_effect.ate_ci_lower = (
                        bootstrap_result.ci_lower_bias_corrected
                    )
                    causal_effect.ate_ci_upper = (
                        bootstrap_result.ci_upper_bias_corrected
                    )
                elif bootstrap_result.config.method == "bca":
                    causal_effect.ate_ci_lower = bootstrap_result.ci_lower_bca
                    causal_effect.ate_ci_upper = bootstrap_result.ci_upper_bca

                # Set all CI variants
                causal_effect.ate_ci_lower_bca = bootstrap_result.ci_lower_bca
                causal_effect.ate_ci_upper_bca = bootstrap_result.ci_upper_bca
                causal_effect.ate_ci_lower_bias_corrected = (
                    bootstrap_result.ci_lower_bias_corrected
                )
                causal_effect.ate_ci_upper_bias_corrected = (
                    bootstrap_result.ci_upper_bias_corrected
                )

                if bootstrap_result.bootstrap_se is not None:
                    causal_effect.ate_se = bootstrap_result.bootstrap_se

            except Exception as e:
                if self.verbose:
                    print(f"Bootstrap confidence intervals failed: {str(e)}")

        return causal_effect

    def plot_rdd(
        self,
        figsize: tuple[int, int] = (10, 6),
        scatter_alpha: float = 0.6,
        line_color: str = "red",
        scatter_color: str = "blue",
    ) -> Any:
        """Plot RDD visualization showing discontinuity at cutoff.

        Args:
            figsize: Figure size tuple
            scatter_alpha: Transparency for scatter points
            line_color: Color for fitted regression lines
            scatter_color: Color for scatter points

        Returns:
            Matplotlib figure object
        """
        if (
            not self.is_fitted
            or self.forcing_variable_data is None
            or self.outcome_data is None
        ):
            raise EstimationError("Estimator must be fitted before plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Get data
        x = self.forcing_variable_data.centered_values
        y = np.array(self.outcome_data.values)

        # Plot all data points
        colors = ["red" if xi >= 0 else "blue" for xi in x]
        ax.scatter(
            x + self.cutoff, y, alpha=scatter_alpha, c=colors, label="Observations"
        )

        # Plot fitted lines within bandwidth
        if self.bandwidth is not None:
            x_plot_left = np.linspace(-self.bandwidth, 0, 100)
            x_plot_right = np.linspace(0, self.bandwidth, 100)

            # Predict using fitted models
            if self.left_model is not None and self.right_model is not None:
                left_features = self._create_polynomial_features(x_plot_left)
                right_features = self._create_polynomial_features(x_plot_right)

                y_pred_left = self.left_model.predict(left_features)
                y_pred_right = self.right_model.predict(right_features)

                ax.plot(
                    x_plot_left + self.cutoff,
                    y_pred_left,
                    color=line_color,
                    linewidth=2,
                    label="Fitted regression",
                )
                ax.plot(
                    x_plot_right + self.cutoff,
                    y_pred_right,
                    color=line_color,
                    linewidth=2,
                )

                # Highlight discontinuity
                cutoff_left = self.left_model.predict(
                    self._create_polynomial_features(np.array([0.0]))
                )[0]
                cutoff_right = self.right_model.predict(
                    self._create_polynomial_features(np.array([0.0]))
                )[0]

                ax.plot(
                    [self.cutoff, self.cutoff],
                    [cutoff_left, cutoff_right],
                    color="green",
                    linewidth=3,
                    label=f"RDD Effect: {cutoff_right - cutoff_left:.3f}",
                )

        # Add vertical line at cutoff
        ax.axvline(
            x=self.cutoff, color="black", linestyle="--", alpha=0.7, label="Cutoff"
        )

        ax.set_xlabel("Forcing Variable")
        ax.set_ylabel("Outcome")
        ax.set_title("Regression Discontinuity Design")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def run_placebo_test(self, placebo_cutoff: float) -> float:
        """Run placebo test at false cutoff to check for spurious discontinuities.

        Args:
            placebo_cutoff: Alternative cutoff point for placebo test

        Returns:
            P-value for placebo test (should be non-significant)
        """
        if not self.is_fitted or self.forcing_variable_data is None:
            raise EstimationError("Estimator must be fitted before placebo test")

        # Temporarily change cutoff and re-estimate
        original_cutoff = self.cutoff
        self.cutoff = placebo_cutoff

        try:
            # Re-fit with placebo cutoff
            if self.treatment_data is not None and self.outcome_data is not None:
                self._fit_implementation(
                    self.treatment_data, self.outcome_data, self.covariate_data
                )
            placebo_result = self._estimate_ate_implementation()

            # Calculate test statistic
            if placebo_result.ate_se is not None and placebo_result.ate_se > 0:
                t_stat = placebo_result.ate / placebo_result.ate_se
                df = (
                    (self._rdd_result.n_left + self._rdd_result.n_right - 4)
                    if self._rdd_result is not None
                    else 100
                )
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            else:
                p_value = 1.0  # Conservative default

        finally:
            # Restore original cutoff and re-fit
            self.cutoff = original_cutoff
            if self.treatment_data is not None and self.outcome_data is not None:
                self._fit_implementation(
                    self.treatment_data, self.outcome_data, self.covariate_data
                )

        return float(p_value)

    def run_mccrary_density_test(self, bins: int = 30) -> float:
        """Run McCrary (2008) density test for manipulation around cutoff.

        Tests for discontinuities in the density of the forcing variable,
        which would suggest manipulation around the cutoff.

        Args:
            bins: Number of bins for histogram-based density estimation

        Returns:
            P-value for density test (should be non-significant if no manipulation)
        """
        if not self.is_fitted or self.forcing_variable_data is None:
            raise EstimationError("Estimator must be fitted before density test")

        forcing_values = np.array(self.forcing_variable_data.values)
        cutoff = self.cutoff

        # Create bins centered around cutoff
        f_min, f_max = np.min(forcing_values), np.max(forcing_values)
        bin_width = (f_max - f_min) / bins
        bin_edges = np.linspace(f_min, f_max, bins + 1)

        # Find bins around cutoff
        cutoff_bin_idx = int(np.searchsorted(bin_edges, cutoff)) - 1
        cutoff_bin_idx = max(0, min(cutoff_bin_idx, bins - 1))

        # Get densities (counts) in bins
        hist, _ = np.histogram(forcing_values, bins=bin_edges)
        densities = hist / (len(forcing_values) * bin_width)  # Normalize to densities

        # Focus on bins around cutoff (within reasonable window)
        window_size = min(5, bins // 4)  # Look at nearby bins
        start_idx = max(0, cutoff_bin_idx - window_size)
        end_idx = min(bins, cutoff_bin_idx + window_size + 1)

        # Split into left and right of cutoff
        left_indices = []
        right_indices = []

        for i in range(start_idx, end_idx):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            if bin_center < cutoff:
                left_indices.append(i)
            else:
                right_indices.append(i)

        if len(left_indices) < 2 or len(right_indices) < 2:
            # Not enough bins on each side for meaningful test
            return 1.0

        # Get densities on each side
        left_densities = densities[left_indices]
        right_densities = densities[right_indices]

        # Simple t-test for difference in mean densities
        # More sophisticated versions would fit smooth densities and test discontinuity
        try:
            from scipy.stats import ttest_ind

            _, p_value = ttest_ind(left_densities, right_densities, equal_var=False)
            return float(p_value)
        except Exception:
            # Fallback to simple comparison
            left_mean = np.mean(left_densities)
            right_mean = np.mean(right_densities)

            # Crude approximation: if densities are very different, suspicious
            relative_diff = abs(left_mean - right_mean) / (
                left_mean + right_mean + 1e-8
            )

            # Convert to rough p-value (this is very approximate)
            if relative_diff > 0.5:  # 50% difference
                return 0.01  # Suspicious
            elif relative_diff > 0.3:  # 30% difference
                return 0.05  # Borderline
            else:
                return 0.2  # Probably OK

        return float(p_value)

    def estimate_rdd(
        self,
        forcing_variable: pd.Series | NDArray[Any],
        outcome: pd.Series | NDArray[Any],
        cutoff: Optional[float] = None,
        covariates: pd.DataFrame | Optional[NDArray[Any]] = None,
    ) -> RDDResult:
        """Generic RDD estimation function as requested in the issue.

        Args:
            forcing_variable: The running/forcing variable
            outcome: Outcome variable
            cutoff: Cutoff threshold (uses instance cutoff if None)
            covariates: Optional covariates for robustness

        Returns:
            RDD estimation results
        """
        # Set cutoff if provided
        if cutoff is not None:
            self.cutoff = cutoff

        # Prepare data in the expected format
        treatment_data = TreatmentData(
            values=forcing_variable,
            name="forcing_variable",
            treatment_type="continuous",
        )

        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")

        covariate_data = None
        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                covariate_data = CovariateData(
                    values=covariates, names=list(covariates.columns)
                )
            else:
                covariate_data = CovariateData(
                    values=covariates,
                    names=[f"X{i}" for i in range(covariates.shape[1])],
                )

        # Fit and estimate
        self.fit(treatment_data, outcome_data, covariate_data)
        causal_effect = self.estimate_ate()

        # Return RDD-specific result format
        if self._rdd_result is not None:
            return self._rdd_result
        else:
            # Fallback to basic result
            return RDDResult(
                ate=causal_effect.ate,
                ate_se=causal_effect.ate_se,
                ate_ci_lower=causal_effect.ate_ci_lower,
                ate_ci_upper=causal_effect.ate_ci_upper,
                confidence_level=causal_effect.confidence_level,
                cutoff=self.cutoff,
            )
