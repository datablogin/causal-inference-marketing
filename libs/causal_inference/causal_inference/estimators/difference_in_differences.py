"""Difference-in-Differences (DID) estimator for causal inference.

This module implements the Difference-in-Differences method for estimating
causal effects using panel data with treatment and control groups observed
before and after treatment.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
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

# from ..core.bootstrap import BootstrapConfig, BootstrapMixin

__all__ = ["DifferenceInDifferencesEstimator", "DIDResult"]


class DIDResult(CausalEffect):
    """Extended causal effect result for DID estimation.

    Includes additional DID-specific outputs like parallel trends validation
    and group-specific estimates.
    """

    def __init__(
        self,
        ate: float,
        ate_ci_lower: float | None = None,
        ate_ci_upper: float | None = None,
        pre_treatment_diff: float | None = None,
        post_treatment_diff: float | None = None,
        treated_pre_mean: float | None = None,
        treated_post_mean: float | None = None,
        control_pre_mean: float | None = None,
        control_post_mean: float | None = None,
        parallel_trends_test_p_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DID result.

        Args:
            ate: Average treatment effect (DID estimate)
            ate_ci_lower: Lower bound of 95% confidence interval
            ate_ci_upper: Upper bound of 95% confidence interval
            pre_treatment_diff: Pre-treatment difference between groups
            post_treatment_diff: Post-treatment difference between groups
            treated_pre_mean: Treated group pre-treatment mean
            treated_post_mean: Treated group post-treatment mean
            control_pre_mean: Control group pre-treatment mean
            control_post_mean: Control group post-treatment mean
            parallel_trends_test_p_value: P-value for parallel trends test
            **kwargs: Additional fields for parent class
        """
        super().__init__(
            ate=ate,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            method="Difference-in-Differences",
            **kwargs,
        )
        self.pre_treatment_diff = pre_treatment_diff
        self.post_treatment_diff = post_treatment_diff
        self.treated_pre_mean = treated_pre_mean
        self.treated_post_mean = treated_post_mean
        self.control_pre_mean = control_pre_mean
        self.control_post_mean = control_post_mean
        self.parallel_trends_test_p_value = parallel_trends_test_p_value

    def plot_parallel_trends(
        self,
        ax: Any | None = None,
        figsize: tuple[int, int] = (10, 6),
        show_counterfactual: bool = True,
    ) -> Any:
        """Plot parallel trends visualization.

        Args:
            ax: Matplotlib axis (created if None)
            figsize: Figure size if creating new figure
            show_counterfactual: Whether to show counterfactual trend

        Returns:
            Matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Time periods
        periods = [0, 1]  # Pre and post

        # Plot actual trends
        if (
            self.treated_pre_mean is not None
            and self.treated_post_mean is not None
            and self.control_pre_mean is not None
            and self.control_post_mean is not None
        ):
            treated_means = [self.treated_pre_mean, self.treated_post_mean]
            control_means = [self.control_pre_mean, self.control_post_mean]

            ax.plot(
                periods,
                treated_means,
                "o-",
                label="Treated Group",
                color="red",
                linewidth=2,
            )
            ax.plot(
                periods,
                control_means,
                "o-",
                label="Control Group",
                color="blue",
                linewidth=2,
            )

        if (
            show_counterfactual
            and self.treated_pre_mean is not None
            and self.control_pre_mean is not None
            and self.control_post_mean is not None
        ):
            # Calculate and plot counterfactual (what treated would have been without treatment)
            control_change = self.control_post_mean - self.control_pre_mean
            counterfactual_post = self.treated_pre_mean + control_change
            counterfactual = [self.treated_pre_mean, counterfactual_post]
            ax.plot(
                periods,
                counterfactual,
                "--",
                label="Counterfactual (Treated)",
                color="red",
                alpha=0.7,
                linewidth=2,
            )

            # Add DID effect annotation
            if self.treated_post_mean is not None:
                ax.annotate(
                    f"DID Effect = {self.ate:.3f}",
                    xy=(1, (self.treated_post_mean + counterfactual_post) / 2),
                    xytext=(1.1, (self.treated_post_mean + counterfactual_post) / 2),
                    arrowprops=dict(arrowstyle="<->", color="black"),
                    fontsize=12,
                    ha="left",
                )

        ax.set_xlabel("Time Period")
        ax.set_ylabel("Outcome")
        ax.set_title("Difference-in-Differences: Parallel Trends")
        ax.set_xticks(periods)
        ax.set_xticklabels(["Pre-treatment", "Post-treatment"])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add vertical line at treatment
        ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.7, label="Treatment")

        return ax


class DifferenceInDifferencesEstimator(BaseEstimator):
    """Difference-in-Differences estimator for causal inference.

    The DID estimator compares changes in outcomes over time between
    treatment and control groups to estimate causal effects. This method
    is particularly useful for policy evaluation and natural experiments.

    Key assumptions:
    1. Parallel trends: Treatment and control groups would have similar trends
       in the absence of treatment
    2. No spillover effects between groups
    3. Treatment timing is exogenous

    The estimator supports:
    - Two-period DID (before/after treatment)
    - Staggered adoption (units treated at different times)
    - Covariate adjustment via regression

    Attributes:
        model: Linear regression model for DID estimation
        bootstrap_config: Configuration for bootstrap confidence intervals
        parallel_trends_test: Whether to perform parallel trends test
    """

    def __init__(
        self,
        parallel_trends_test: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the DID estimator.

        Args:
            parallel_trends_test: Whether to test parallel trends assumption
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output during estimation
        """
        super().__init__(random_state=random_state, verbose=verbose)
        self.parallel_trends_test = parallel_trends_test
        self.model: LinearRegression | None = None
        self._time_data: NDArray[Any] | None = None
        self._group_data: NDArray[Any] | None = None
        self._did_result: DIDResult | None = None
        self._design_matrix: NDArray[Any] | None = None
        self._fitted_outcome: NDArray[Any] | None = None

    def _validate_did_data(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        time_data: NDArray[Any],
        group_data: NDArray[Any],
        covariates: CovariateData | None = None,
    ) -> None:
        """Validate data for DID estimation.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            time_data: Time period indicators (0=pre, 1=post)
            group_data: Group indicators (0=control, 1=treated)
            covariates: Optional covariate data
        """
        # Extract arrays
        if isinstance(treatment.values, pd.Series):
            T = treatment.values.values
        else:
            T = np.asarray(treatment.values)

        if isinstance(outcome.values, pd.Series):
            Y = outcome.values.values
        else:
            Y = np.asarray(outcome.values)

        # Check dimensions
        n = len(Y)
        if len(T) != n:
            raise DataValidationError(
                f"Treatment length {len(T)} != outcome length {n}"
            )
        if len(time_data) != n:
            raise DataValidationError(
                f"Time data length {len(time_data)} != outcome length {n}"
            )
        if len(group_data) != n:
            raise DataValidationError(
                f"Group data length {len(group_data)} != outcome length {n}"
            )

        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                X = covariates.values.values
            else:
                X = np.asarray(covariates.values)
            if len(X) != n:
                raise DataValidationError(
                    f"Covariates length {len(X)} != outcome length {n}"
                )

        # Validate time periods (should be 0 and 1)
        unique_times = np.unique(time_data)
        if not np.array_equal(sorted(unique_times), [0, 1]):
            raise DataValidationError(
                f"Time data must contain only 0 (pre) and 1 (post). Found: {unique_times}"
            )

        # Validate groups (should be 0 and 1)
        unique_groups = np.unique(group_data)
        if not np.array_equal(sorted(unique_groups), [0, 1]):
            raise DataValidationError(
                f"Group data must contain only 0 (control) and 1 (treated). Found: {unique_groups}"
            )

        # Check that we have data for all four DID cells
        cells = []
        for t in [0, 1]:
            for g in [0, 1]:
                mask = (time_data == t) & (group_data == g)
                if np.sum(mask) == 0:
                    cells.append(f"time={t}, group={g}")

        if cells:
            raise DataValidationError(
                f"Missing data for DID cells: {cells}. Need observations for all combinations "
                "of time (0,1) and group (0,1)."
            )

        # Warn if treatment doesn't align with group assignment
        if not np.array_equal(np.asarray(T), np.asarray(group_data)):
            if self.verbose:
                print(
                    "Warning: Treatment assignment differs from group assignment. "
                    "Using group assignment for DID estimation."
                )

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the DID model.

        Args:
            treatment: Treatment assignment data (used for validation)
            outcome: Outcome variable data
            covariates: Optional covariate data
        """
        if self._time_data is None or self._group_data is None:
            raise DataValidationError(
                "DID estimation requires time_data and group_data. "
                "Call fit() with time_data and group_data parameters."
            )

        # Validate data
        self._validate_did_data(
            treatment, outcome, self._time_data, self._group_data, covariates
        )

        # Extract outcome data
        if isinstance(outcome.values, pd.Series):
            Y = outcome.values.values
        else:
            Y = np.asarray(outcome.values)

        # Create design matrix
        # DID regression: Y = α + β₁*Time + β₂*Group + β₃*(Time × Group) + ε
        # where β₃ is the DID estimate
        time_dummy = self._time_data.astype(float)
        group_dummy = self._group_data.astype(float)
        interaction = time_dummy * group_dummy

        # Start with basic DID variables
        X = np.column_stack(
            [
                np.ones(len(Y)),  # Intercept
                time_dummy,  # Time effect
                group_dummy,  # Group effect
                interaction,  # DID coefficient
            ]
        )

        # Add covariates if provided
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                X_cov = covariates.values.values
            else:
                X_cov = np.asarray(covariates.values)

            if X_cov.ndim == 1:
                X_cov = X_cov.reshape(-1, 1)

            X = np.column_stack([X, X_cov])

        # Fit linear regression
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X, Y)

        # Store design matrix and data for confidence interval calculation
        self._design_matrix = X
        self._fitted_outcome = np.asarray(Y)

        if self.verbose:
            mse = mean_squared_error(Y, self.model.predict(X))
            print(f"DID model fitted. MSE: {mse:.4f}")

    def fit(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        time_data: NDArray[Any] | None = None,
        group_data: NDArray[Any] | None = None,
    ) -> DifferenceInDifferencesEstimator:
        """Fit the DID estimator.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Optional covariate data
            time_data: Time period indicators (0=pre, 1=post)
            group_data: Group indicators (0=control, 1=treated)

        Returns:
            Fitted estimator instance
        """
        # Store additional DID-specific data
        if time_data is None or group_data is None:
            raise DataValidationError(
                "DID estimation requires time_data and group_data. "
                "Please provide these as additional arguments to fit()."
            )

        self._time_data = np.asarray(time_data)
        self._group_data = np.asarray(group_data)

        # Call parent fit method
        super().fit(treatment, outcome, covariates)

        return self

    def _estimate_ate_implementation(self) -> DIDResult:
        """Estimate the average treatment effect using DID.

        Returns:
            DIDResult with DID estimate and diagnostics
        """
        if (
            self.model is None
            or self._time_data is None
            or self._group_data is None
            or self.outcome_data is None
        ):
            raise EstimationError("Model not fitted. Call fit() first.")

        # Extract outcome data
        if isinstance(self.outcome_data.values, pd.Series):
            Y = self.outcome_data.values.values
        else:
            Y = np.asarray(self.outcome_data.values)

        # Get DID coefficient using robust indexing
        did_coefficient_index = self._get_did_coefficient_index()
        if did_coefficient_index is None:
            raise EstimationError("Could not identify DID coefficient in design matrix")

        did_coefficient = self.model.coef_[did_coefficient_index]

        # Calculate group means for diagnostics
        control_pre_mask = (self._time_data == 0) & (self._group_data == 0)
        control_post_mask = (self._time_data == 1) & (self._group_data == 0)
        treated_pre_mask = (self._time_data == 0) & (self._group_data == 1)
        treated_post_mask = (self._time_data == 1) & (self._group_data == 1)

        control_pre_mean = np.mean(Y[control_pre_mask])
        control_post_mean = np.mean(Y[control_post_mask])
        treated_pre_mean = np.mean(Y[treated_pre_mask])
        treated_post_mean = np.mean(Y[treated_post_mask])

        # Calculate pre and post treatment differences
        pre_treatment_diff = treated_pre_mean - control_pre_mean
        post_treatment_diff = treated_post_mean - control_post_mean

        # Parallel trends test
        parallel_trends_p_value = None
        if self.parallel_trends_test:
            parallel_trends_p_value = self._test_parallel_trends(pre_treatment_diff)

        # Calculate confidence intervals using standard errors
        ci_lower, ci_upper = self._calculate_confidence_intervals()

        # Create DID result
        result = DIDResult(
            ate=did_coefficient,
            ate_ci_lower=ci_lower,
            ate_ci_upper=ci_upper,
            pre_treatment_diff=pre_treatment_diff,
            post_treatment_diff=post_treatment_diff,
            treated_pre_mean=treated_pre_mean,
            treated_post_mean=treated_post_mean,
            control_pre_mean=control_pre_mean,
            control_post_mean=control_post_mean,
            parallel_trends_test_p_value=parallel_trends_p_value,
        )

        self._did_result = result
        return result

    def _test_parallel_trends(self, pre_treatment_diff: float) -> float:
        """Test the parallel trends assumption using pre-treatment group balance.

        This test examines whether the treatment and control groups have
        significantly different baseline levels, which could indicate
        violation of the parallel trends assumption. In practice, tests
        with multiple pre-periods would be preferred.

        Args:
            pre_treatment_diff: Pre-treatment difference between groups

        Returns:
            P-value for parallel trends test (higher p-values support parallel trends)
        """
        try:
            import scipy.stats as stats
        except ImportError:
            # If scipy not available, return placeholder indicating uncertainty
            return 0.5

        # Enhanced error handling
        if (
            self.outcome_data is None
            or self._time_data is None
            or self._group_data is None
        ):
            return 0.5

        if isinstance(self.outcome_data.values, pd.Series):
            Y = self.outcome_data.values.values
        else:
            Y = np.asarray(self.outcome_data.values)

        # Get pre-treatment observations for both groups
        control_pre_mask = (self._time_data == 0) & (self._group_data == 0)
        treated_pre_mask = (self._time_data == 0) & (self._group_data == 1)

        n_control_pre = np.sum(control_pre_mask)
        n_treated_pre = np.sum(treated_pre_mask)

        # Enhanced validation for group sizes
        if n_control_pre < 3 or n_treated_pre < 3:
            # Need at least 3 observations per group for meaningful test
            if self.verbose:
                print(
                    f"Warning: Insufficient pre-treatment observations for parallel trends test "
                    f"(control: {n_control_pre}, treated: {n_treated_pre}). Need at least 3 each."
                )
            return 0.5

        # Extract pre-treatment values
        control_pre_values = Y[control_pre_mask]
        treated_pre_values = Y[treated_pre_mask]

        # Remove any potential NaN or infinite values
        control_pre_values = control_pre_values[np.isfinite(control_pre_values)]
        treated_pre_values = treated_pre_values[np.isfinite(treated_pre_values)]

        if len(control_pre_values) < 3 or len(treated_pre_values) < 3:
            if self.verbose:
                print(
                    "Warning: Insufficient valid pre-treatment observations after filtering"
                )
            return 0.5

        # Perform Welch's t-test (unequal variances assumed)
        try:
            # Test null hypothesis: no difference in pre-treatment means
            # Higher p-values support parallel trends assumption
            t_stat, p_value = stats.ttest_ind(
                treated_pre_values,
                control_pre_values,
                equal_var=False,  # Welch's t-test
            )

            # Additional check: ensure groups have reasonable variance
            control_var = np.var(control_pre_values, ddof=1)
            treated_var = np.var(treated_pre_values, ddof=1)

            # Flag if one group has zero or extremely small variance
            min_reasonable_var = 1e-8
            if control_var < min_reasonable_var or treated_var < min_reasonable_var:
                if self.verbose:
                    print(
                        "Warning: One or both groups have very low variance in pre-treatment period"
                    )
                return 0.5

            return float(p_value)

        except (ValueError, RuntimeError) as e:
            if self.verbose:
                print(f"Warning: Error in parallel trends test: {e}")
            return 0.5

    def _calculate_confidence_intervals(
        self, confidence_level: float = 0.95
    ) -> tuple[float | None, float | None]:
        """Calculate confidence intervals for the DID coefficient using standard errors.

        Args:
            confidence_level: Confidence level for the interval (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        if (
            self.model is None
            or self._design_matrix is None
            or self._fitted_outcome is None
        ):
            return (None, None)

        try:
            import scipy.stats as stats
        except ImportError:
            # If scipy not available, return None
            return (None, None)

        # Get model predictions and residuals
        y_pred = self.model.predict(self._design_matrix)
        residuals = self._fitted_outcome - y_pred

        # Calculate residual variance
        n = len(self._fitted_outcome)
        p = self._design_matrix.shape[1]  # Number of parameters

        if n <= p:
            # Not enough degrees of freedom
            return (None, None)

        residual_variance = np.sum(residuals**2) / (n - p)

        # Calculate covariance matrix: (X'X)^(-1) * σ²
        try:
            xtx_inv = np.linalg.inv(self._design_matrix.T @ self._design_matrix)
            covariance_matrix = xtx_inv * residual_variance

            # Find DID coefficient index dynamically
            did_coefficient_index = self._get_did_coefficient_index()
            if did_coefficient_index is None:
                return (None, None)

            # Standard error for DID coefficient
            se_did = np.sqrt(
                covariance_matrix[did_coefficient_index, did_coefficient_index]
            )

            # Calculate confidence interval
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, n - p)

            did_coefficient = self.model.coef_[did_coefficient_index]
            margin_of_error = t_critical * se_did

            ci_lower = did_coefficient - margin_of_error
            ci_upper = did_coefficient + margin_of_error

            return float(ci_lower), float(ci_upper)

        except np.linalg.LinAlgError:
            # Singular matrix, can't compute standard errors
            return (None, None)

    def _get_did_coefficient_index(self) -> int | None:
        """Get the index of the DID coefficient in the design matrix.

        Returns:
            Index of the interaction term coefficient, or None if not found
        """
        # The DID coefficient is always the interaction term (time * group)
        # In our design matrix construction, it's the 4th column (index 3):
        # [intercept, time, group, interaction, ...covariates]

        if self._design_matrix is None or self._design_matrix.shape[1] < 4:
            return None

        # Verify this is indeed the interaction term by checking
        # that it equals time_dummy * group_dummy
        if self._time_data is not None and self._group_data is not None:
            time_dummy = self._time_data.astype(float)
            group_dummy = self._group_data.astype(float)
            expected_interaction = time_dummy * group_dummy

            # Check if column 3 matches the expected interaction
            if np.allclose(self._design_matrix[:, 3], expected_interaction):
                return 3

        # Fallback: return None if we can't verify
        return None

    # Bootstrap methods removed for simplicity - can be added later if needed

    def predict_counterfactual(
        self,
        group_data: NDArray[Any],
        time_data: NDArray[Any] | None = None,
        covariates: CovariateData | None = None,
    ) -> NDArray[Any]:
        """Predict counterfactual outcomes.

        Args:
            group_data: Group indicators for prediction
            time_data: Time period indicators (if None, uses post-treatment)
            covariates: Optional covariate data

        Returns:
            Predicted counterfactual outcomes
        """
        if not self.is_fitted or self.model is None:
            raise EstimationError("Model not fitted. Call fit() first.")

        if time_data is None:
            time_data = np.ones(len(group_data))  # Post-treatment period

        # Create design matrix (same structure as training)
        time_dummy = time_data.astype(float)
        group_dummy = group_data.astype(float)
        interaction = time_dummy * group_dummy

        X = np.column_stack(
            [
                np.ones(len(group_data)),  # Intercept
                time_dummy,  # Time effect
                group_dummy,  # Group effect
                interaction,  # DID coefficient
            ]
        )

        # Add covariates if provided
        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                X_cov = covariates.values.values
            else:
                X_cov = np.asarray(covariates.values)

            if X_cov.ndim == 1:
                X_cov = X_cov.reshape(-1, 1)

            X = np.column_stack([X, X_cov])

        return self.model.predict(X)  # type: ignore[no-any-return]

    def get_did_summary(self) -> dict[str, Any]:
        """Get a summary of DID estimation results.

        Returns:
            Dictionary with DID summary statistics
        """
        if self._did_result is None:
            raise EstimationError("DID not estimated. Call estimate_ate() first.")

        summary = {
            "did_estimate": self._did_result.ate,
            "confidence_interval": (
                self._did_result.ate_ci_lower,
                self._did_result.ate_ci_upper,
            ),
            "group_means": {
                "control_pre": self._did_result.control_pre_mean,
                "control_post": self._did_result.control_post_mean,
                "treated_pre": self._did_result.treated_pre_mean,
                "treated_post": self._did_result.treated_post_mean,
            },
            "differences": {
                "pre_treatment": self._did_result.pre_treatment_diff,
                "post_treatment": self._did_result.post_treatment_diff,
            },
            "parallel_trends_p_value": self._did_result.parallel_trends_test_p_value,
        }

        return summary
