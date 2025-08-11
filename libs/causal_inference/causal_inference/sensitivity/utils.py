"""Utility functions for sensitivity analysis modules.

This module provides common utilities for input validation, standardization,
and error handling across sensitivity analysis functions.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def standardize_input(
    data: NDArray[Any] | pd.Series | pd.DataFrame,
    name: str = "data",
    allow_2d: bool = False,
    min_length: int = 1,
) -> NDArray[Any]:
    """Standardize input to numpy array with proper validation.

    Args:
        data: Input data (numpy array, pandas Series/DataFrame)
        name: Name of the data for error messages
        allow_2d: Whether to allow 2D arrays (for covariates)
        min_length: Minimum required length

    Returns:
        Standardized numpy array

    Raises:
        ValueError: If data validation fails
        TypeError: If data type is not supported
    """
    if data is None:
        raise ValueError(f"{name} cannot be None")

    # Convert to numpy array
    if hasattr(data, "values"):
        # Pandas Series/DataFrame
        if isinstance(data, pd.DataFrame) and allow_2d:
            arr = np.asarray(data.values)
        elif isinstance(data, pd.Series):
            arr = np.asarray(data.values).flatten()
        elif isinstance(data, pd.DataFrame) and not allow_2d:
            if data.shape[1] == 1:
                arr = np.asarray(data.values).flatten()
            else:
                raise ValueError(
                    f"{name} DataFrame has {data.shape[1]} columns but expected 1. "
                    "Use allow_2d=True if multiple columns are intended."
                )
        else:
            arr = np.asarray(data.values)
    else:
        # Numpy array or array-like
        arr = np.asarray(data)

    # Validate dimensions
    if not allow_2d and arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr.flatten()
        else:
            raise ValueError(
                f"{name} has shape {arr.shape} but expected 1D array. "
                "Use allow_2d=True if 2D array is intended."
            )
    elif allow_2d and arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ValueError(f"{name} has {arr.ndim} dimensions but expected at most 2")

    # Check for minimum length
    if len(arr) < min_length:
        raise ValueError(
            f"{name} has length {len(arr)} but requires at least {min_length}"
        )

    # Check for missing values
    if np.any(np.isnan(arr)):
        n_missing = np.sum(np.isnan(arr))
        raise ValueError(
            f"{name} contains {n_missing} missing values. "
            "Please handle missing data before analysis."
        )

    # Check for infinite values
    if np.any(np.isinf(arr)):
        n_infinite = np.sum(np.isinf(arr))
        raise ValueError(
            f"{name} contains {n_infinite} infinite values. "
            "Please handle infinite values before analysis."
        )

    return arr


def validate_treatment_outcome_lengths(
    *arrays: NDArray[Any], names: list[str] | None = None
) -> None:
    """Validate that treatment, outcome, and covariates have consistent lengths.

    Args:
        *arrays: Variable number of arrays to check
        names: Optional names for arrays (for error messages)

    Raises:
        ValueError: If arrays have inconsistent lengths
    """
    if not arrays:
        return

    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]
    elif len(names) != len(arrays):
        raise ValueError("Number of names must match number of arrays")

    lengths = [len(arr) for arr in arrays if arr is not None]
    if not lengths:
        return

    reference_length = lengths[0]
    for i, length in enumerate(lengths):
        if length != reference_length:
            raise ValueError(
                f"{names[i]} has length {length} but expected {reference_length}. "
                "All arrays must have the same length."
            )


def check_treatment_variation(
    treatment: NDArray[Any],
    min_unique_values: int = 2,
    binary_threshold: float = 0.01,
) -> dict[str, Any]:
    """Check treatment variation and provide diagnostics.

    Args:
        treatment: Treatment variable
        min_unique_values: Minimum number of unique values required
        binary_threshold: Minimum proportion for binary treatment groups

    Returns:
        Dictionary with treatment diagnostics

    Raises:
        ValueError: If treatment has insufficient variation
    """
    unique_values = np.unique(treatment)
    n_unique = len(unique_values)

    if n_unique < min_unique_values:
        raise ValueError(
            f"Treatment has only {n_unique} unique values but requires at least {min_unique_values}. "
            "Insufficient treatment variation for analysis."
        )

    # For binary treatment, check group sizes
    if n_unique == 2:
        proportions = []
        for val in unique_values:
            prop = np.mean(treatment == val)
            proportions.append(prop)
            if prop < binary_threshold:
                warnings.warn(
                    f"Treatment group with value {val} has only {prop:.1%} of observations. "
                    f"Consider removing if below {binary_threshold:.1%} threshold.",
                    UserWarning,
                    stacklevel=3,
                )

        return {
            "is_binary": True,
            "unique_values": unique_values.tolist(),
            "proportions": proportions,
            "balanced": min(proportions) >= 0.3,  # Reasonable balance
        }
    else:
        return {
            "is_binary": False,
            "unique_values": unique_values[:10].tolist(),  # First 10 for display
            "n_unique": n_unique,
            "is_continuous": n_unique > 10,
        }


def check_model_assumptions(
    outcome: NDArray[Any],
    treatment: NDArray[Any],
    covariates: NDArray[Any] | None = None,
    check_linearity: bool = True,
    check_multicollinearity: bool = True,
) -> dict[str, Any]:
    """Check basic modeling assumptions for sensitivity analysis.

    Args:
        outcome: Outcome variable
        treatment: Treatment variable
        covariates: Optional covariates
        check_linearity: Whether to check linearity assumption
        check_multicollinearity: Whether to check multicollinearity

    Returns:
        Dictionary with assumption check results
    """
    diagnostics = {}

    # Check outcome distribution
    outcome_skew = float(np.abs(pd.Series(outcome).skew()))
    if outcome_skew > 2.0:
        diagnostics["outcome_skewed"] = True
        diagnostics["outcome_skew"] = outcome_skew
        warnings.warn(
            f"Outcome variable is highly skewed (skewness = {outcome_skew:.2f}). "
            "Consider transformation for better model performance.",
            UserWarning,
            stacklevel=3,
        )
    else:
        diagnostics["outcome_skewed"] = False
        diagnostics["outcome_skew"] = outcome_skew

    # Check treatment-outcome correlation for basic sanity
    treatment_outcome_corr = float(np.corrcoef(treatment, outcome)[0, 1])
    diagnostics["treatment_outcome_correlation"] = treatment_outcome_corr

    if abs(treatment_outcome_corr) < 0.01:
        warnings.warn(
            f"Very weak correlation between treatment and outcome ({treatment_outcome_corr:.3f}). "
            "This may indicate no treatment effect or measurement issues.",
            UserWarning,
            stacklevel=3,
        )

    # Check multicollinearity if covariates provided
    if covariates is not None and check_multicollinearity and covariates.shape[1] > 1:
        try:
            # Calculate correlation matrix for covariates
            cov_corr = np.corrcoef(covariates.T)
            max_corr = np.max(np.abs(cov_corr[np.triu_indices_from(cov_corr, k=1)]))
            diagnostics["max_covariate_correlation"] = float(max_corr)

            if max_corr > 0.9:
                warnings.warn(
                    f"High correlation between covariates detected ({max_corr:.3f}). "
                    "Consider removing redundant variables to avoid multicollinearity.",
                    UserWarning,
                    stacklevel=3,
                )
        except Exception:
            diagnostics["multicollinearity_check_failed"] = True

    return diagnostics


def format_sensitivity_warnings() -> None:
    """Configure warning formatting for sensitivity analysis modules."""
    # Ensure warnings are shown with proper formatting
    warnings.filterwarnings("default", category=UserWarning)

    # Custom warning format for sensitivity analysis
    def custom_warning_format(
        message, category, filename, lineno, file=None, line=None
    ):
        return f"⚠️  Sensitivity Analysis Warning: {message}\n"

    # Set custom format
    warnings.formatwarning = custom_warning_format
