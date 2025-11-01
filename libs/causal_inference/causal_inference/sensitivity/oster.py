"""Oster's δ method for assessing omitted variable bias.

This module implements Emily Oster's approach to assess sensitivity of
regression results to omitted variable bias using selection on observables
as a guide for selection on unobservables.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .utils import (
    check_model_assumptions,
    check_treatment_variation,
    format_sensitivity_warnings,
    standardize_input,
    validate_treatment_outcome_lengths,
)


def oster_delta(
    outcome: Union[NDArray[Any], pd.Series],
    treatment: Union[NDArray[Any], pd.Series],
    covariates_restricted: Optional[Union[NDArray[Any], pd.DataFrame]] = None,
    covariates_full: Optional[Union[NDArray[Any], pd.DataFrame]] = None,
    r_max: Optional[float] = None,
    delta: float = 1.0,
    bootstrap_samples: int = 0,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Calculate Oster's δ for assessing omitted variable bias.

    This implements Emily Oster's (2019) method to assess how robust
    treatment effects are to omitted variable bias. The method uses
    the degree of selection on observables as a guide for the degree
    of selection on unobservables.

    Args:
        outcome: Outcome variable
        treatment: Treatment variable
        covariates_restricted: Restricted set of controls (baseline model)
        covariates_full: Full set of available controls
        r_max: Maximum R-squared achievable (if None, estimated as 1.3 * R² from full model)
        delta: Assumed degree of selection on unobservables relative to observables
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        random_state: Random state for reproducibility

    Returns:
        Dictionary containing:
            - beta_restricted: Treatment coefficient from restricted model
            - beta_full: Treatment coefficient from full model
            - r2_restricted: R² from restricted model
            - r2_full: R² from full model
            - r_max_used: Maximum R² used in calculations
            - delta_used: δ parameter used
            - beta_star: Bias-adjusted treatment coefficient
            - oster_bound: Oster bound for β*
            - robustness_ratio: |β*| / |β_full| ratio
            - passes_robustness_test: Whether |β*| / |β_full| ≥ threshold
            - interpretation: Qualitative interpretation
            - bootstrap_results: Bootstrap confidence intervals if requested

    Raises:
        ValueError: If data dimensions don't match or insufficient variation

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from causal_inference.sensitivity import oster_delta
        >>>
        >>> # Simulate data
        >>> n = 1000
        >>> X1 = np.random.normal(0, 1, n)  # Observed confounder
        >>> X2 = np.random.normal(0, 1, n)  # Additional observed control
        >>> U = np.random.normal(0, 1, n)   # Unobserved confounder
        >>> T = 0.5 * X1 + 0.3 * U + np.random.normal(0, 1, n)  # Treatment
        >>> Y = 2.0 * T + X1 + X2 + 0.5 * U + np.random.normal(0, 1, n)  # Outcome
        >>>
        >>> # Oster analysis
        >>> results = oster_delta(
        >>>     outcome=Y,
        >>>     treatment=T,
        >>>     covariates_restricted=X1.reshape(-1, 1),
        >>>     covariates_full=np.column_stack([X1, X2])
        >>> )
        >>> print(f"Bias-adjusted coefficient: {results['beta_star']:.3f}")
        >>> print(f"Robustness ratio: {results['robustness_ratio']:.3f}")

    Notes:
        The method compares a restricted model with a full model that includes
        additional controls. The key insight is that if unobserved confounders
        are similar to observed ones, the change in coefficients and R² when
        adding controls provides information about potential omitted variable bias.

        Common interpretations:
        - δ = 0: No selection on unobservables
        - δ = 1: Equal selection on observables and unobservables
        - δ > 1: Stronger selection on unobservables than observables

        Robustness test: Results are considered robust if |β*| is at least
        some fraction (typically 50-80%) of |β_full|.
    """
    # Configure warning formatting
    format_sensitivity_warnings()

    # Standardize inputs with enhanced validation
    y = standardize_input(outcome, name="outcome", min_length=10)
    t = standardize_input(treatment, name="treatment", min_length=10)

    # Validate consistent lengths
    arrays_to_check = [y, t]
    names_to_check = ["outcome", "treatment"]

    # Handle covariates with standardization
    x_restricted = None
    if covariates_restricted is not None:
        x_restricted = standardize_input(
            covariates_restricted,
            name="covariates_restricted",
            allow_2d=True,
            min_length=len(y),
        )
        arrays_to_check.append(x_restricted)
        names_to_check.append("covariates_restricted")

    x_full = None
    if covariates_full is not None:
        x_full = standardize_input(
            covariates_full, name="covariates_full", allow_2d=True, min_length=len(y)
        )
        arrays_to_check.append(x_full)
        names_to_check.append("covariates_full")

    # Validate all arrays have consistent lengths
    validate_treatment_outcome_lengths(*arrays_to_check, names=names_to_check)

    # Check treatment variation and basic assumptions
    check_treatment_variation(t)
    check_model_assumptions(y, t, x_full)

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Fit restricted model (treatment only or treatment + restricted controls)
    if x_restricted is not None:
        x_restricted_with_treatment = np.column_stack([t, x_restricted])
    else:
        x_restricted_with_treatment = t.reshape(-1, 1)

    model_restricted = LinearRegression()
    model_restricted.fit(x_restricted_with_treatment, y)
    beta_restricted = model_restricted.coef_[0]  # Treatment coefficient
    y_pred_restricted = model_restricted.predict(x_restricted_with_treatment)
    r2_restricted = r2_score(y, y_pred_restricted)

    # Fit full model
    if x_full is not None:
        x_full_with_treatment = np.column_stack([t, x_full])
    else:
        # If no full covariates provided, use restricted + treatment
        x_full_with_treatment = x_restricted_with_treatment

    model_full = LinearRegression()
    model_full.fit(x_full_with_treatment, y)
    beta_full = model_full.coef_[0]  # Treatment coefficient
    y_pred_full = model_full.predict(x_full_with_treatment)
    r2_full = r2_score(y, y_pred_full)

    # Determine R_max
    if r_max is None:
        # Oster suggests using 1.3 * R² from full model as reasonable upper bound
        r_max_used = min(1.3 * r2_full, 1.0)
    else:
        r_max_used = min(r_max, 1.0)  # Cap at 1.0

    # Calculate bias-adjusted coefficient (β*)
    # β* = β_tilde - δ * [β_dot - β_tilde] * [R_max - R_tilde] / [R_tilde - R_dot]

    # Check if there's meaningful improvement (threshold for noise)
    r2_improvement = r2_full - r2_restricted
    meaningful_improvement_threshold = 0.01  # 1% improvement threshold

    if r2_improvement <= meaningful_improvement_threshold:
        # No meaningful improvement from additional controls
        beta_star = beta_full
        oster_bound = beta_full
    else:
        # Calculate Oster bound
        numerator = r_max_used - r2_full
        denominator = r2_improvement

        if denominator < 1e-10:  # Avoid division by zero
            beta_star = beta_full
            oster_bound = beta_full
        else:
            bias_adjustment = (
                delta * (beta_restricted - beta_full) * (numerator / denominator)
            )
            beta_star = beta_full - bias_adjustment
            oster_bound = beta_star

    # Calculate robustness ratio
    robustness_ratio = (
        abs(beta_star) / abs(beta_full) if abs(beta_full) > 1e-10 else 1.0
    )

    # Warning system for edge cases and assumption violations
    if r2_full < 0.1:
        warnings.warn(
            f"Low R² in full model ({r2_full:.3f}). "
            "Oster bounds may be unreliable with poor model fit. "
            "Consider adding more relevant covariates.",
            UserWarning,
            stacklevel=2,
        )

    if r2_improvement < 0.001:
        warnings.warn(
            f"Very small R² improvement ({r2_improvement:.4f}) from additional controls. "
            "This suggests either no meaningful confounders were added "
            "or all relevant confounders were already controlled.",
            UserWarning,
            stacklevel=2,
        )

    if abs(beta_full) < 0.01:
        warnings.warn(
            "Treatment effect is very small. "
            "Oster bounds may be sensitive to numerical precision issues.",
            UserWarning,
            stacklevel=2,
        )

    if robustness_ratio < 0.2:
        warnings.warn(
            f"Low robustness ratio ({robustness_ratio:.3f}). "
            "Results are highly sensitive to unobserved confounding. "
            "Consider collecting additional control variables.",
            UserWarning,
            stacklevel=2,
        )

    # Robustness test (common threshold is 0.5 or 0.8)
    robustness_threshold = 0.5
    passes_robustness_test = robustness_ratio >= robustness_threshold

    # Interpretation
    if passes_robustness_test and robustness_ratio >= 0.8:
        interpretation = (
            "Highly robust - bias-adjusted effect retains most of original magnitude"
        )
    elif passes_robustness_test:
        interpretation = "Moderately robust - some sensitivity to omitted variables"
    elif robustness_ratio >= 0.2:
        interpretation = (
            "Limited robustness - substantial sensitivity to omitted variables"
        )
    else:
        interpretation = (
            "Not robust - effect may be largely due to omitted variable bias"
        )

    results = {
        "beta_restricted": float(beta_restricted),
        "beta_full": float(beta_full),
        "r2_restricted": float(r2_restricted),
        "r2_full": float(r2_full),
        "r_max_used": float(r_max_used),
        "delta_used": float(delta),
        "beta_star": float(beta_star),
        "oster_bound": float(oster_bound),
        "robustness_ratio": float(robustness_ratio),
        "passes_robustness_test": passes_robustness_test,
        "interpretation": interpretation,
    }

    # Bootstrap confidence intervals if requested
    if bootstrap_samples > 0:
        bootstrap_results = _bootstrap_oster(
            y, t, x_restricted, x_full, r_max_used, delta, bootstrap_samples
        )
        results["bootstrap_results"] = bootstrap_results

    return results


def _bootstrap_oster(
    y: NDArray[Any],
    t: NDArray[Any],
    x_restricted: Optional[NDArray[Any]],
    x_full: Optional[NDArray[Any]],
    r_max: float,
    delta: float,
    n_bootstrap: int,
    chunk_size: int = 100,
) -> dict[str, Any]:
    """Bootstrap confidence intervals for Oster bounds with chunked processing.

    Args:
        y: Outcome variable
        t: Treatment variable
        x_restricted: Restricted covariates
        x_full: Full covariates
        r_max: Maximum R-squared
        delta: Selection parameter
        n_bootstrap: Number of bootstrap samples
        chunk_size: Number of bootstrap samples to process in each chunk

    Returns:
        Dictionary with bootstrap confidence intervals
    """
    n = len(y)
    bootstrap_betas = []
    bootstrap_ratios = []

    # Process bootstrap samples in chunks to manage memory
    n_chunks = (n_bootstrap + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_bootstrap)
        chunk_samples = end_idx - start_idx

        # Generate all indices for this chunk at once for efficiency
        chunk_indices = np.random.choice(n, size=(chunk_samples, n), replace=True)

        for i in range(chunk_samples):
            idx = chunk_indices[i]
            y_boot = y[idx]
            t_boot = t[idx]
            x_restricted_boot = x_restricted[idx] if x_restricted is not None else None
            x_full_boot = x_full[idx] if x_full is not None else None

            try:
                # Calculate Oster delta for bootstrap sample
                boot_results = oster_delta(
                    y_boot,
                    t_boot,
                    x_restricted_boot,
                    x_full_boot,
                    r_max=r_max,
                    delta=delta,
                    bootstrap_samples=0,  # No nested bootstrapping
                )
                bootstrap_betas.append(boot_results["beta_star"])
                bootstrap_ratios.append(boot_results["robustness_ratio"])
            except (ValueError, np.linalg.LinAlgError):
                # Skip failed bootstrap samples
                continue

    if len(bootstrap_betas) == 0:
        return {"error": "All bootstrap samples failed"}

    bootstrap_betas = np.array(bootstrap_betas)
    bootstrap_ratios = np.array(bootstrap_ratios)

    return {
        "beta_star_ci": [
            float(np.percentile(bootstrap_betas, 2.5)),
            float(np.percentile(bootstrap_betas, 97.5)),
        ],
        "robustness_ratio_ci": [
            float(np.percentile(bootstrap_ratios, 2.5)),
            float(np.percentile(bootstrap_ratios, 97.5)),
        ],
        "n_successful_bootstraps": len(bootstrap_betas),
        "n_chunks_processed": n_chunks,
    }
