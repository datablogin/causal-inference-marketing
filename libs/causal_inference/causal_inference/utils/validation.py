"""Validation utilities for causal inference estimators.

This module provides common validation functions used across different estimators.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.base import DataValidationError


def validate_input_dimensions(
    treatment: NDArray[Any],
    outcome: NDArray[Any],
    covariates: NDArray[Any] | None = None,
) -> None:
    """Validate dimensions of input arrays.

    Args:
        treatment: Treatment array
        outcome: Outcome array
        covariates: Covariate array (optional)

    Raises:
        DataValidationError: If dimensions don't match
    """
    n_samples_t = len(treatment)
    n_samples_y = len(outcome)

    if n_samples_t != n_samples_y:
        raise DataValidationError(
            f"Treatment and outcome must have same number of samples. "
            f"Got {n_samples_t} and {n_samples_y} respectively."
        )

    if covariates is not None:
        n_samples_x = covariates.shape[0]
        if n_samples_x != n_samples_t:
            raise DataValidationError(
                f"Covariates must have same number of samples as treatment. "
                f"Got {n_samples_x} and {n_samples_t} respectively."
            )

    # Check for NaN values
    if np.any(np.isnan(treatment)):
        raise DataValidationError("Treatment contains NaN values.")

    if np.any(np.isnan(outcome)):
        raise DataValidationError("Outcome contains NaN values.")

    if covariates is not None and np.any(np.isnan(covariates)):
        raise DataValidationError("Covariates contain NaN values.")


def validate_binary_treatment(
    treatment: NDArray[Any] | pd.Series,
) -> NDArray[Any]:
    """Validate and convert treatment to binary 0/1 array.

    Args:
        treatment: Treatment variable

    Returns:
        Binary treatment array with values 0 and 1

    Raises:
        DataValidationError: If treatment is not binary
    """
    # Convert to numpy array
    if isinstance(treatment, pd.Series):
        t_array = treatment.values
    else:
        t_array = np.asarray(treatment)

    # Get unique values
    unique_vals = np.unique(np.asarray(t_array))

    if len(unique_vals) != 2:
        raise DataValidationError(
            f"Treatment must be binary. Found {len(unique_vals)} unique values: {unique_vals}"
        )

    # Map to 0/1
    if not np.array_equal(sorted(unique_vals), [0, 1]):
        # Map the first (lower) value to 0, second to 1
        t_binary = (t_array == unique_vals[1]).astype(int)
    else:
        t_binary = t_array.astype(int)

    return np.asarray(t_binary)


def validate_propensity_scores(
    propensity: NDArray[Any],
    min_prop: float = 0.01,
    max_prop: float = 0.99,
) -> NDArray[Any]:
    """Validate and clip propensity scores to avoid extreme weights.

    Args:
        propensity: Propensity score array
        min_prop: Minimum allowed propensity score
        max_prop: Maximum allowed propensity score

    Returns:
        Clipped propensity scores

    Raises:
        DataValidationError: If propensity scores are invalid
    """
    if np.any(np.isnan(propensity)):
        raise DataValidationError("Propensity scores contain NaN values.")

    if np.any((propensity < 0) | (propensity > 1)):
        raise DataValidationError("Propensity scores must be between 0 and 1.")

    # Clip extreme values
    clipped = np.clip(propensity, min_prop, max_prop)

    # Warn if many values were clipped
    n_clipped = np.sum((propensity < min_prop) | (propensity > max_prop))
    if n_clipped > 0.1 * len(propensity):
        import warnings

        warnings.warn(
            f"Clipped {n_clipped} ({100 * n_clipped / len(propensity):.1f}%) "
            f"propensity scores to [{min_prop}, {max_prop}] range.",
            RuntimeWarning,
        )

    return clipped


def check_common_support(
    propensity: NDArray[Any],
    treatment: NDArray[Any],
    threshold: float = 0.1,
) -> bool:
    """Check for common support between treatment groups.

    Args:
        propensity: Propensity scores
        treatment: Binary treatment indicator
        threshold: Minimum overlap threshold

    Returns:
        True if there is adequate common support
    """
    treated_props = propensity[treatment == 1]
    control_props = propensity[treatment == 0]

    # Check overlap in propensity score ranges
    treated_min, treated_max = np.min(treated_props), np.max(treated_props)
    control_min, control_max = np.min(control_props), np.max(control_props)

    overlap_min = max(treated_min, control_min)
    overlap_max = min(treated_max, control_max)

    # Check if there's meaningful overlap
    if overlap_min >= overlap_max:
        return False

    # Check density in overlap region
    treated_in_overlap = np.sum(
        (treated_props >= overlap_min) & (treated_props <= overlap_max)
    )
    control_in_overlap = np.sum(
        (control_props >= overlap_min) & (control_props <= overlap_max)
    )

    treated_prop_in_overlap = treated_in_overlap / len(treated_props)
    control_prop_in_overlap = control_in_overlap / len(control_props)

    return bool(
        (treated_prop_in_overlap >= threshold)
        and (control_prop_in_overlap >= threshold)
    )
