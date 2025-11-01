"""Transportability weighting methods for population adjustment."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnEstimator
from sklearn.linear_model import LogisticRegression


@dataclass
class WeightingResult:
    """Results from transportability weighting.

    Attributes:
        weights: Transport weights for source population
        density_ratio: Estimated density ratio p(X|target) / p(X|source)
        effective_sample_size: Effective sample size after weighting
        max_weight: Maximum weight value
        weight_stability_ratio: Ratio of max to min weight (stability measure)
        convergence_achieved: Whether estimation converged
        diagnostics: Additional diagnostic information
    """

    weights: NDArray[Any]
    density_ratio: NDArray[Any]
    effective_sample_size: float
    max_weight: float
    weight_stability_ratio: float
    convergence_achieved: bool
    diagnostics: dict[str, Any]

    @property
    def is_stable(self) -> bool:
        """Check if weights are stable (low variance)."""
        return self.weight_stability_ratio < 20  # Common threshold

    @property
    def relative_efficiency(self) -> float:
        """Calculate relative efficiency compared to equal weights."""
        n = len(self.weights)
        return self.effective_sample_size / n


class TransportabilityWeighting(ABC):
    """Abstract base class for transportability weighting methods."""

    def __init__(
        self,
        trim_weights: bool = True,
        max_weight: float = 10.0,
        min_weight: float = 0.1,
        stabilize: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize transportability weighting method.

        Args:
            trim_weights: Whether to trim extreme weights
            max_weight: Maximum allowed weight value
            min_weight: Minimum allowed weight value
            stabilize: Whether to apply weight stabilization
            random_state: Random seed for reproducibility
        """
        self.trim_weights = trim_weights
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.stabilize = stabilize
        self.random_state = random_state

        # Fitted state
        self.is_fitted = False
        self.density_ratio_model: Optional[SklearnEstimator] = None

    @abstractmethod
    def _estimate_density_ratio(
        self,
        source_data: pd.DataFrame | NDArray[Any],
        target_data: pd.DataFrame | NDArray[Any],
    ) -> NDArray[Any]:
        """Estimate density ratio p(X|target) / p(X|source).

        Args:
            source_data: Covariate data from source population
            target_data: Covariate data from target population

        Returns:
            Estimated density ratios for source observations
        """
        pass

    def fit_weights(
        self,
        source_data: pd.DataFrame | NDArray[Any],
        target_data: pd.DataFrame | NDArray[Any],
    ) -> WeightingResult:
        """Fit transportability weights to match target population.

        Args:
            source_data: Covariate data from source population
            target_data: Covariate data from target population

        Returns:
            WeightingResult with weights and diagnostics

        Raises:
            ValueError: If data dimensions don't match
        """
        # Validate inputs
        source_df = self._ensure_dataframe(source_data)
        target_df = self._ensure_dataframe(target_data)
        self._validate_inputs(source_df, target_df)

        # Estimate density ratios
        density_ratios = self._estimate_density_ratio(source_df, target_df)

        # Convert to weights (normalize by source sample size)
        weights = density_ratios * len(source_df) / np.sum(density_ratios)

        # Apply stabilization if requested
        if self.stabilize:
            weights = self._stabilize_weights(weights)

        # Trim extreme weights if requested
        if self.trim_weights:
            weights = self._trim_weights(weights)

        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(weights, source_df, target_df)

        # Create result object
        result = WeightingResult(
            weights=weights,
            density_ratio=density_ratios,
            effective_sample_size=self._calculate_effective_sample_size(weights),
            max_weight=float(np.max(weights)),
            weight_stability_ratio=float(np.max(weights) / np.min(weights)),
            convergence_achieved=True,  # Override in subclasses if needed
            diagnostics=diagnostics,
        )

        self.is_fitted = True
        return result

    def _ensure_dataframe(
        self, data: Union[pd.DataFrame, NDArray[Any]]
    ) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    def _validate_inputs(
        self, source_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> None:
        """Validate input data."""
        if source_df.shape[1] != target_df.shape[1]:
            raise ValueError(
                f"Source ({source_df.shape[1]}) and target ({target_df.shape[1]}) "
                "must have same number of variables"
            )

        if len(source_df) == 0 or len(target_df) == 0:
            raise ValueError("Source and target datasets cannot be empty")

    def _stabilize_weights(self, weights: NDArray[Any]) -> NDArray[Any]:
        """Apply weight stabilization to reduce variance."""
        # Simple truncation-based stabilization
        median_weight = float(np.median(weights))
        stabilized = np.clip(
            weights,
            median_weight / 5,  # Lower bound
            median_weight * 5,  # Upper bound
        )

        # Renormalize
        result = stabilized * len(weights) / np.sum(stabilized)
        return np.asarray(result, dtype=np.float64)

    def _trim_weights(self, weights: NDArray[Any]) -> NDArray[Any]:
        """Trim extreme weights to improve stability."""
        trimmed = np.clip(weights, self.min_weight, self.max_weight)

        # Renormalize to maintain sum, but ensure bounds are still respected
        normalization_factor = len(weights) / np.sum(trimmed)
        result = trimmed * normalization_factor

        # Apply clipping again after renormalization to ensure bounds are respected
        result = np.clip(result, self.min_weight, self.max_weight)
        return np.asarray(result, dtype=np.float64)

    def _calculate_effective_sample_size(self, weights: NDArray[Any]) -> float:
        """Calculate effective sample size after weighting."""
        return float((np.sum(weights) ** 2) / np.sum(weights**2))

    def _calculate_diagnostics(
        self,
        weights: NDArray[Any],
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate diagnostic metrics for weighting quality."""
        return {
            "weight_mean": float(np.mean(weights)),
            "weight_std": float(np.std(weights)),
            "weight_cv": float(np.std(weights) / np.mean(weights)),
            "n_extreme_weights": int(
                np.sum((weights > self.max_weight) | (weights < self.min_weight))
            ),
            "source_sample_size": len(source_df),
            "target_sample_size": len(target_df),
            "weight_entropy": float(
                -np.sum(weights * np.log(weights + 1e-12)) / len(weights)
            ),
        }


class DensityRatioEstimator(TransportabilityWeighting):
    """Density ratio estimation using classification approach.

    Estimates p(X|target) / p(X|source) by training a classifier to distinguish
    between source and target populations, then using predicted probabilities.
    """

    def __init__(
        self,
        classifier: Optional[SklearnEstimator] = None,
        cross_validate: bool = True,
        cv_folds: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize density ratio estimator.

        Args:
            classifier: Sklearn classifier for discrimination task
            cross_validate: Whether to use cross-validation
            cv_folds: Number of CV folds
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)

        self.classifier = classifier or LogisticRegression(
            random_state=self.random_state, max_iter=1000, class_weight="balanced"
        )
        self.cross_validate = cross_validate
        self.cv_folds = cv_folds

    def _estimate_density_ratio(
        self,
        source_data: pd.DataFrame | NDArray[Any],
        target_data: pd.DataFrame | NDArray[Any],
    ) -> NDArray[Any]:
        """Estimate density ratio using classification approach."""
        # Combine data and create labels with consistent column naming
        source_df = pd.DataFrame(source_data)
        target_df = pd.DataFrame(target_data)

        # Ensure consistent column naming (all strings)
        if source_df.shape[1] == target_df.shape[1]:
            column_names = [f"feature_{i}" for i in range(source_df.shape[1])]
            source_df.columns = column_names
            target_df.columns = column_names

        combined_data = pd.concat([source_df, target_df], ignore_index=True)

        labels = np.concatenate(
            [
                np.zeros(len(source_data)),  # Source = 0
                np.ones(len(target_data)),  # Target = 1
            ]
        )

        # Handle missing values
        combined_data = combined_data.fillna(combined_data.mean())

        # Fit classifier
        if self.cross_validate:
            # Use cross-validation for more robust estimates
            from sklearn.model_selection import cross_val_predict

            probabilities = cross_val_predict(
                self.classifier,
                combined_data,
                labels,
                cv=self.cv_folds,
                method="predict_proba",
            )[:, 1]  # Probability of being target
        else:
            self.classifier.fit(combined_data, labels)
            probabilities = self.classifier.predict_proba(combined_data)[:, 1]

        # Extract probabilities for source data
        source_probs = probabilities[: len(source_data)]

        # Convert to density ratios: p(target|X) / p(source|X)
        # Using Bayes rule: p(X|target) / p(X|source) =
        # [p(target|X) / p(source|X)] * [p(source) / p(target)]

        # Avoid division by zero with more robust clipping
        source_probs = np.clip(source_probs, 1e-8, 1 - 1e-8)

        # Prior ratio (assuming equal priors for simplicity)
        prior_ratio = 1.0

        density_ratios = (source_probs / (1 - source_probs)) * prior_ratio

        return np.asarray(density_ratios, dtype=np.float64)


class OptimalTransportWeighting(TransportabilityWeighting):
    """Optimal transport-based weighting for population matching.

    Uses optimal transport theory to find weights that minimize
    the Wasserstein distance between source and target distributions.
    """

    def __init__(
        self,
        reg_param: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Initialize optimal transport weighting.

        Args:
            reg_param: Regularization parameter for entropy regularization
            max_iterations: Maximum iterations for Sinkhorn algorithm
            tolerance: Convergence tolerance
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)

        self.reg_param = reg_param
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def _estimate_density_ratio(
        self,
        source_data: pd.DataFrame | NDArray[Any],
        target_data: pd.DataFrame | NDArray[Any],
    ) -> NDArray[Any]:
        """Estimate density ratio using optimal transport."""
        try:
            import ot  # type: ignore # Python Optimal Transport library
        except ImportError:
            warnings.warn(
                "POT (Python Optimal Transport) library not available. "
                "Falling back to classification-based approach.",
                UserWarning,
            )
            # Fallback to classification approach
            fallback = DensityRatioEstimator(random_state=self.random_state)
            return fallback._estimate_density_ratio(source_data, target_data)

        # Convert to numpy arrays
        source_array = np.array(source_data)
        target_array = np.array(target_data)

        # Calculate pairwise distances (cost matrix)
        # Using Euclidean distance in standardized space
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        source_scaled = scaler.fit_transform(source_array)
        target_scaled = scaler.transform(target_array)

        # Cost matrix: squared Euclidean distances
        cost_matrix = ot.dist(source_scaled, target_scaled, metric="sqeuclidean")

        # Uniform distributions
        source_weights = np.ones(len(source_data)) / len(source_data)
        target_weights = np.ones(len(target_data)) / len(target_data)

        # Solve optimal transport with entropy regularization
        transport_plan = ot.sinkhorn(
            source_weights,
            target_weights,
            cost_matrix,
            reg=self.reg_param,
            numItermax=self.max_iterations,
            stopThr=self.tolerance,
        )

        # Extract density ratios from transport plan
        # Weight for each source point is proportional to total mass transported
        transport_weights = np.sum(transport_plan, axis=1) * len(target_data)

        return np.asarray(transport_weights, dtype=np.float64)


class TransportabilityWeightingInterface:
    """Main interface for transportability weighting.

    Provides a unified interface for different weighting methods
    with automatic method selection and validation.
    """

    def __init__(
        self,
        method: str = "classification",
        auto_select: bool = True,
        **method_kwargs: Any,
    ) -> None:
        """Initialize transportability weighting.

        Args:
            method: Weighting method ('classification', 'optimal_transport')
            auto_select: Whether to automatically select best method
            **method_kwargs: Arguments passed to the specific method
        """
        self.method = method
        self.auto_select = auto_select
        self.method_kwargs = method_kwargs

        # Initialize weighting estimator
        self.estimator = self._create_estimator()

    def _create_estimator(self) -> TransportabilityWeighting:
        """Create the appropriate weighting estimator."""
        if self.method == "classification":
            return DensityRatioEstimator(**self.method_kwargs)
        elif self.method == "optimal_transport":
            return OptimalTransportWeighting(**self.method_kwargs)
        else:
            raise ValueError(f"Unknown weighting method: {self.method}")

    def estimate_weights(
        self,
        source_data: pd.DataFrame | NDArray[Any],
        target_data: pd.DataFrame | NDArray[Any],
    ) -> WeightingResult:
        """Estimate transportability weights.

        Args:
            source_data: Covariate data from source population
            target_data: Covariate data from target population

        Returns:
            WeightingResult with weights and diagnostics
        """
        if self.auto_select:
            # Try multiple methods and select best
            return self._auto_select_method(source_data, target_data)
        else:
            # Use specified method
            return self.estimator.fit_weights(source_data, target_data)

    def _auto_select_method(
        self,
        source_data: pd.DataFrame | NDArray[Any],
        target_data: pd.DataFrame | NDArray[Any],
    ) -> WeightingResult:
        """Automatically select best weighting method."""
        methods = {
            "classification": DensityRatioEstimator(**self.method_kwargs),
            "optimal_transport": OptimalTransportWeighting(**self.method_kwargs),
        }

        best_result = None
        best_score = float("-inf")
        best_method = None

        for method_name, estimator in methods.items():
            try:
                result = estimator.fit_weights(source_data, target_data)

                # Score based on stability and efficiency
                stability_score = 1.0 / (1.0 + result.weight_stability_ratio / 20)
                efficiency_score = result.relative_efficiency

                # Combined score (could be made more sophisticated)
                score = 0.6 * stability_score + 0.4 * efficiency_score

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_method = method_name

            except Exception as e:
                warnings.warn(f"Method {method_name} failed: {str(e)}", UserWarning)
                continue

        if best_result is None:
            raise RuntimeError("All weighting methods failed")

        # Update diagnostics with method selection info
        best_result.diagnostics["selected_method"] = best_method
        best_result.diagnostics["method_selection_score"] = best_score

        return best_result

    def validate_weights(
        self,
        weights: NDArray[Any],
        source_data: pd.DataFrame | NDArray[Any],
        target_data: pd.DataFrame | NDArray[Any],
    ) -> dict[str, Any]:
        """Validate quality of transportability weights.

        Args:
            weights: Estimated transport weights
            source_data: Source population covariates
            target_data: Target population covariates

        Returns:
            Dictionary with validation metrics
        """
        # Convert to DataFrames
        source_df = (
            pd.DataFrame(source_data)
            if not isinstance(source_data, pd.DataFrame)
            else source_data
        )
        target_df = (
            pd.DataFrame(target_data)
            if not isinstance(target_data, pd.DataFrame)
            else target_data
        )

        # Calculate weighted source statistics
        weighted_means = {}
        target_means = {}
        standardized_diffs = {}

        for col in source_df.columns:
            source_col = source_df[col].values
            target_col = target_df[col].values

            # Weighted mean for source
            weighted_mean = np.average(source_col, weights=weights)
            target_mean = np.mean(target_col)

            # Standardized mean difference after weighting
            pooled_std = np.sqrt(
                (np.var(source_col, ddof=1) + np.var(target_col, ddof=1)) / 2
            )

            if pooled_std > 0:
                smd = (weighted_mean - target_mean) / pooled_std
            else:
                smd = 0.0

            weighted_means[col] = weighted_mean
            target_means[col] = target_mean
            standardized_diffs[col] = smd

        # Overall balance assessment
        mean_abs_smd = np.mean([abs(smd) for smd in standardized_diffs.values()])
        max_abs_smd = np.max([abs(smd) for smd in standardized_diffs.values()])

        # Balance thresholds (ensure Python bool, not numpy bool)
        good_balance = bool(mean_abs_smd < 0.1 and max_abs_smd < 0.25)

        return {
            "weighted_means": weighted_means,
            "target_means": target_means,
            "standardized_mean_differences": standardized_diffs,
            "mean_absolute_smd": mean_abs_smd,
            "max_absolute_smd": max_abs_smd,
            "good_balance_achieved": good_balance,
            "n_variables_balanced": sum(
                1 for smd in standardized_diffs.values() if abs(smd) < 0.1
            ),
            "total_variables": len(standardized_diffs),
            "effective_sample_size": (np.sum(weights) ** 2) / np.sum(weights**2),
            "weight_statistics": {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "cv": float(np.std(weights) / np.mean(weights)),
            },
        }
