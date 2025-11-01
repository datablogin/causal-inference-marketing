"""Cross-fitting infrastructure for bias reduction in causal inference.

Cross-fitting (also called cross-validation or sample splitting) is used to
reduce overfitting bias when using machine learning methods for nuisance
parameter estimation in causal inference.
"""
# ruff: noqa: N803

from __future__ import annotations

import gc
import hashlib
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib not available. Parallel processing will be disabled.")

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

__all__ = [
    "CrossFitData",
    "CrossFittingEstimator",
    "create_cross_fit_data",
    "ParallelCrossFittingConfig",
]


@dataclass
class ParallelCrossFittingConfig:
    """Configuration for parallel cross-fitting optimization."""

    n_jobs: int = -1
    parallel_backend: str = "threading"
    max_memory_gb: float = 4.0
    chunk_size: int = 10000
    enable_caching: bool = True
    cache_size: int = 100
    timeout_per_fold_minutes: float = 10.0
    enable_gc_per_fold: bool = True
    memory_monitoring: bool = False


@dataclass
class CrossFitData:
    """Data structure for cross-fitting splits.

    Contains training and validation indices for each fold,
    along with corresponding data splits.
    """

    n_folds: int
    train_indices: list[NDArray[Any]]
    val_indices: list[NDArray[Any]]
    X_train_folds: list[NDArray[Any]]
    X_val_folds: list[NDArray[Any]]
    y_train_folds: list[NDArray[Any]]
    y_val_folds: list[NDArray[Any]]
    treatment_train_folds: Optional[list[NDArray[Any]]] = None
    treatment_val_folds: Optional[list[NDArray[Any]]] = None


class NuisanceLearner(Protocol):
    """Protocol for nuisance parameter learners used in cross-fitting."""

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> None:
        """Fit the learner to training data."""
        ...

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Make predictions on new data."""
        ...


class CrossFittingEstimator(ABC):
    """Abstract base class for estimators that use cross-fitting.

    Cross-fitting is essential for machine learning-based causal inference
    methods to avoid overfitting bias in nuisance parameter estimation.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        stratified: bool = True,
        random_state: Optional[int] = None,
        parallel_config: Optional[ParallelCrossFittingConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize cross-fitting estimator.

        Args:
            cv_folds: Number of cross-validation folds
            stratified: Whether to use stratified cross-validation
            random_state: Random state for reproducibility
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.cv_folds = cv_folds
        self.stratified = stratified
        self.random_state = random_state
        self.parallel_config = parallel_config or ParallelCrossFittingConfig()

        # Storage for cross-fitting results
        self.cross_fit_data_: Optional[CrossFitData] = None
        self.nuisance_estimates_: dict[str, NDArray[Any]] = {}

        # Performance monitoring
        self._fold_timings_: list[float] = []
        self._fold_memory_usage_: list[float] = []
        self._parallel_speedup_: Optional[float] = None

        # Model caching for performance optimization
        self._model_cache_: dict[str, Any] = {}
        self._cache_hits_: int = 0
        self._cache_misses_: int = 0

    @abstractmethod
    def _fit_nuisance_models(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: Optional[NDArray[Any]] = None,
    ) -> dict[str, Any]:
        """Fit nuisance parameter models on training data.

        Args:
            X_train: Training covariates
            y_train: Training outcomes
            treatment_train: Training treatments (if needed)

        Returns:
            Dictionary of fitted nuisance models
        """
        pass

    @abstractmethod
    def _predict_nuisance_parameters(
        self,
        models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: Optional[NDArray[Any]] = None,
    ) -> dict[str, NDArray[Any]]:
        """Predict nuisance parameters on validation data.

        Args:
            models: Fitted nuisance models
            X_val: Validation covariates
            treatment_val: Validation treatments (if needed)

        Returns:
            Dictionary of nuisance parameter predictions
        """
        pass

    @abstractmethod
    def _estimate_target_parameter(
        self,
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
    ) -> float:
        """Estimate the target causal parameter using nuisance estimates.

        Args:
            nuisance_estimates: Cross-fitted nuisance parameter estimates
            treatment: Treatment assignments
            outcome: Outcomes

        Returns:
            Estimated target parameter (e.g., ATE)
        """
        pass

    def _create_cross_fit_splits(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        treatment: Optional[NDArray[Any]] = None,
    ) -> CrossFitData:
        """Create cross-fitting data splits.

        Args:
            X: Covariate matrix
            y: Outcome vector
            treatment: Treatment vector (for stratified splitting)

        Returns:
            CrossFitData object with splits
        """
        # Create cross-validation splitter
        if self.stratified and treatment is not None:
            # Stratify on treatment assignment
            cv_splitter = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(cv_splitter.split(X, treatment))
        else:
            cv_splitter = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            splits = list(cv_splitter.split(X))

        # Extract indices and data for each fold
        train_indices = []
        val_indices = []
        X_train_folds = []
        X_val_folds = []
        y_train_folds = []
        y_val_folds = []
        treatment_train_folds = []
        treatment_val_folds = []

        for train_idx, val_idx in splits:
            train_indices.append(train_idx)
            val_indices.append(val_idx)

            X_train_folds.append(X[train_idx])
            X_val_folds.append(X[val_idx])
            y_train_folds.append(y[train_idx])
            y_val_folds.append(y[val_idx])

            if treatment is not None:
                treatment_train_folds.append(treatment[train_idx])
                treatment_val_folds.append(treatment[val_idx])

        return CrossFitData(
            n_folds=self.cv_folds,
            train_indices=train_indices,
            val_indices=val_indices,
            X_train_folds=X_train_folds,
            X_val_folds=X_val_folds,
            y_train_folds=y_train_folds,
            y_val_folds=y_val_folds,
            treatment_train_folds=treatment_train_folds
            if treatment is not None
            else None,
            treatment_val_folds=treatment_val_folds if treatment is not None else None,
        )

    def _perform_cross_fitting(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        treatment: Optional[NDArray[Any]] = None,
    ) -> dict[str, NDArray[Any]]:
        """Perform cross-fitting to estimate nuisance parameters.

        Supports both sequential and parallel processing based on configuration.

        Args:
            X: Covariate matrix
            y: Outcome vector
            treatment: Treatment vector

        Returns:
            Dictionary of cross-fitted nuisance parameter estimates
        """
        # Create cross-fitting splits
        self.cross_fit_data_ = self._create_cross_fit_splits(X, y, treatment)

        # Check memory constraints and enable chunked processing if needed
        should_chunk = self._should_use_chunked_processing(X, y)
        if should_chunk:
            return self._perform_chunked_cross_fitting(X, y, treatment)

        # Choose parallel vs sequential processing
        use_parallel = (
            JOBLIB_AVAILABLE and self.parallel_config.n_jobs != 1 and self.cv_folds > 1
        )

        if use_parallel:
            return self._perform_parallel_cross_fitting(X, y, treatment)
        else:
            return self._perform_sequential_cross_fitting(X, y, treatment)

    def _perform_sequential_cross_fitting(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        treatment: Optional[NDArray[Any]] = None,
    ) -> dict[str, NDArray[Any]]:
        """Sequential cross-fitting implementation (original approach)."""
        n_samples = X.shape[0]
        nuisance_estimates = {}

        # Initialize timing tracking
        self._fold_timings_ = []
        if self.parallel_config.memory_monitoring and PSUTIL_AVAILABLE:
            self._fold_memory_usage_ = []
            process = psutil.Process()

        # Perform cross-fitting for each fold sequentially
        for fold_idx in range(self.cv_folds):
            fold_start_time = time.perf_counter()

            # Memory monitoring
            fold_start_memory = None
            if self.parallel_config.memory_monitoring and PSUTIL_AVAILABLE:
                try:
                    fold_start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except (OSError, AttributeError) as e:
                    warnings.warn(
                        f"Memory monitoring failed for fold {fold_idx}: {str(e)}"
                    )
                    fold_start_memory = None

            # Get training and validation data for this fold
            X_train = self.cross_fit_data_.X_train_folds[fold_idx]
            y_train = self.cross_fit_data_.y_train_folds[fold_idx]
            X_val = self.cross_fit_data_.X_val_folds[fold_idx]

            treatment_train = None
            treatment_val = None
            if (
                treatment is not None
                and self.cross_fit_data_.treatment_train_folds is not None
            ):
                treatment_train = self.cross_fit_data_.treatment_train_folds[fold_idx]
            if (
                treatment is not None
                and self.cross_fit_data_.treatment_val_folds is not None
            ):
                treatment_val = self.cross_fit_data_.treatment_val_folds[fold_idx]

            # Fit nuisance models on training data
            fitted_models = self._fit_nuisance_models(X_train, y_train, treatment_train)

            # Predict nuisance parameters on validation data
            fold_predictions = self._predict_nuisance_parameters(
                fitted_models, X_val, treatment_val
            )

            # Store predictions for validation indices
            val_indices = self.cross_fit_data_.val_indices[fold_idx]
            for param_name, predictions in fold_predictions.items():
                if param_name not in nuisance_estimates:
                    nuisance_estimates[param_name] = np.full(n_samples, np.nan)
                nuisance_estimates[param_name][val_indices] = predictions

            # Performance tracking
            fold_end_time = time.perf_counter()
            self._fold_timings_.append(fold_end_time - fold_start_time)

            if (
                self.parallel_config.memory_monitoring
                and PSUTIL_AVAILABLE
                and fold_start_memory is not None
            ):
                try:
                    fold_end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    self._fold_memory_usage_.append(fold_end_memory - fold_start_memory)
                except (OSError, AttributeError) as e:
                    warnings.warn(
                        f"Memory monitoring failed at end of fold {fold_idx}: {str(e)}"
                    )
                    # Add zero to maintain list consistency
                    self._fold_memory_usage_.append(0.0)

            # Optional garbage collection per fold
            if self.parallel_config.enable_gc_per_fold:
                gc.collect()

        # Check that all samples have nuisance estimates
        for param_name, estimates in nuisance_estimates.items():
            if np.any(np.isnan(estimates)):
                raise ValueError(
                    f"Some samples missing nuisance estimates for {param_name}"
                )

        self.nuisance_estimates_ = nuisance_estimates
        return nuisance_estimates

    def _perform_parallel_cross_fitting(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        treatment: Optional[NDArray[Any]] = None,
    ) -> dict[str, NDArray[Any]]:
        """Parallel cross-fitting implementation using joblib."""
        n_samples = X.shape[0]

        # Time the parallel processing for speedup calculation
        start_time = time.perf_counter()

        try:
            # Create parallel jobs for each fold
            parallel_jobs = Parallel(
                n_jobs=self.parallel_config.n_jobs,
                backend=self.parallel_config.parallel_backend,
                timeout=self.parallel_config.timeout_per_fold_minutes * 60,
            )

            # Define the delayed function for each fold
            delayed_tasks = []
            for fold_idx in range(self.cv_folds):
                X_train = self.cross_fit_data_.X_train_folds[fold_idx]
                y_train = self.cross_fit_data_.y_train_folds[fold_idx]
                X_val = self.cross_fit_data_.X_val_folds[fold_idx]

                treatment_train = None
                treatment_val = None
                if (
                    treatment is not None
                    and self.cross_fit_data_.treatment_train_folds is not None
                ):
                    treatment_train = self.cross_fit_data_.treatment_train_folds[
                        fold_idx
                    ]
                if (
                    treatment is not None
                    and self.cross_fit_data_.treatment_val_folds is not None
                ):
                    treatment_val = self.cross_fit_data_.treatment_val_folds[fold_idx]

                delayed_tasks.append(
                    delayed(self._fit_single_fold)(
                        fold_idx,
                        X_train,
                        y_train,
                        X_val,
                        treatment_train,
                        treatment_val,
                    )
                )

            # Execute parallel jobs
            fold_results = parallel_jobs(delayed_tasks)

        except Exception as e:
            warnings.warn(
                f"Parallel cross-fitting failed: {str(e)}. Falling back to sequential processing."
            )
            return self._perform_sequential_cross_fitting(X, y, treatment)

        # Process results from parallel execution
        nuisance_estimates = {}
        self._fold_timings_ = []

        for fold_idx, (fold_predictions, fold_timing) in enumerate(fold_results):
            if fold_predictions is None:
                warnings.warn(
                    f"Fold {fold_idx} failed. Falling back to sequential processing."
                )
                return self._perform_sequential_cross_fitting(X, y, treatment)

            val_indices = self.cross_fit_data_.val_indices[fold_idx]
            self._fold_timings_.append(fold_timing)

            for param_name, predictions in fold_predictions.items():
                if param_name not in nuisance_estimates:
                    nuisance_estimates[param_name] = np.full(n_samples, np.nan)
                nuisance_estimates[param_name][val_indices] = predictions

        # Calculate parallel speedup estimate
        parallel_time = time.perf_counter() - start_time
        sequential_estimate = sum(self._fold_timings_)
        self._parallel_speedup_ = (
            sequential_estimate / parallel_time if parallel_time > 0 else 1.0
        )

        # Check that all samples have nuisance estimates
        for param_name, estimates in nuisance_estimates.items():
            if np.any(np.isnan(estimates)):
                raise ValueError(
                    f"Some samples missing nuisance estimates for {param_name}"
                )

        self.nuisance_estimates_ = nuisance_estimates
        return nuisance_estimates

    def _fit_single_fold(
        self,
        fold_idx: int,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        X_val: NDArray[Any],
        treatment_train: Optional[NDArray[Any]] = None,
        treatment_val: Optional[NDArray[Any]] = None,
    ) -> Optional[tuple[dict[str, NDArray[Any]]], float]:
        """Fit a single fold - designed to be called in parallel.

        Args:
            fold_idx: Fold index for identification
            X_train: Training features for this fold
            y_train: Training outcomes for this fold
            X_val: Validation features for this fold
            treatment_train: Training treatment assignments for this fold
            treatment_val: Validation treatment assignments for this fold

        Returns:
            Tuple of (fold_predictions, fold_timing)
        """
        fold_start_time = time.perf_counter()

        try:
            # Fit nuisance models on training data
            fitted_models = self._fit_nuisance_models(X_train, y_train, treatment_train)

            # Predict nuisance parameters on validation data
            fold_predictions = self._predict_nuisance_parameters(
                fitted_models, X_val, treatment_val
            )

            fold_end_time = time.perf_counter()
            fold_timing = fold_end_time - fold_start_time

            return fold_predictions, fold_timing

        except Exception as e:
            warnings.warn(f"Fold {fold_idx} fitting failed: {str(e)}")
            fold_end_time = time.perf_counter()
            fold_timing = fold_end_time - fold_start_time
            return None, fold_timing

    def _should_use_chunked_processing(self, X: NDArray[Any], y: NDArray[Any]) -> bool:
        """Determine if chunked processing should be used for large datasets."""
        if not self.parallel_config.memory_monitoring:
            return False

        n_samples, n_features = X.shape

        # Estimate memory usage (rough heuristic)
        data_size_mb = (n_samples * n_features * 8) / (
            1024 * 1024
        )  # 8 bytes per float64
        estimated_peak_usage = data_size_mb * self.cv_folds * 2  # Conservative estimate

        memory_threshold_mb = self.parallel_config.max_memory_gb * 1024

        return (
            estimated_peak_usage > memory_threshold_mb
            or n_samples > self.parallel_config.chunk_size
        )

    def _perform_chunked_cross_fitting(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        treatment: Optional[NDArray[Any]] = None,
    ) -> dict[str, NDArray[Any]]:
        """Cross-fitting with chunked processing for large datasets.

        This implementation processes data in smaller chunks to manage memory usage
        while still maintaining cross-fitting integrity.
        """
        warnings.warn(
            f"Using chunked processing for large dataset (n_samples={X.shape[0]}). "
            f"This may increase runtime but reduces memory usage."
        )

        n_samples = X.shape[0]
        nuisance_estimates = {}

        # Process each fold with chunking
        for fold_idx in range(self.cv_folds):
            fold_start_time = time.perf_counter()

            fold_predictions = self._process_chunked_fold(fold_idx, treatment)

            self._store_fold_predictions(
                fold_predictions, nuisance_estimates, n_samples, fold_idx
            )

            self._track_fold_performance(fold_start_time)

            # Cleanup after each fold
            if self.parallel_config.enable_gc_per_fold:
                gc.collect()

        self._validate_nuisance_estimates(nuisance_estimates)
        self.nuisance_estimates_ = nuisance_estimates
        return nuisance_estimates

    def _process_chunked_fold(
        self, fold_idx: int, treatment: Optional[NDArray[Any]]
    ) -> dict[str, NDArray[Any]]:
        """Process a single fold using chunked validation data."""
        # Get fold data
        X_train = self.cross_fit_data_.X_train_folds[fold_idx]
        y_train = self.cross_fit_data_.y_train_folds[fold_idx]
        X_val = self.cross_fit_data_.X_val_folds[fold_idx]

        treatment_train = None
        treatment_val = None
        if (
            treatment is not None
            and self.cross_fit_data_.treatment_train_folds is not None
        ):
            treatment_train = self.cross_fit_data_.treatment_train_folds[fold_idx]
        if (
            treatment is not None
            and self.cross_fit_data_.treatment_val_folds is not None
        ):
            treatment_val = self.cross_fit_data_.treatment_val_folds[fold_idx]

        # Fit models on full training data
        fitted_models = self._fit_nuisance_models_cached(
            X_train, y_train, treatment_train, fold_idx
        )

        # Process validation data in chunks
        val_chunks_predictions = self._predict_validation_chunks(
            fitted_models, X_val, treatment_val
        )

        # Concatenate chunk predictions
        return self._concatenate_chunk_predictions(val_chunks_predictions)

    def _predict_validation_chunks(
        self,
        fitted_models: dict[str, Any],
        X_val: NDArray[Any],
        treatment_val: Optional[NDArray[Any]],
    ) -> dict[str, list[NDArray[Any]]]:
        """Predict on validation data in chunks to manage memory."""
        chunk_size = self.parallel_config.chunk_size
        val_chunks_predictions = {}

        for chunk_start in range(0, len(X_val), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(X_val))

            X_val_chunk = X_val[chunk_start:chunk_end]
            treatment_val_chunk = (
                treatment_val[chunk_start:chunk_end]
                if treatment_val is not None
                else None
            )

            # Predict on chunk
            chunk_predictions = self._predict_nuisance_parameters(
                fitted_models, X_val_chunk, treatment_val_chunk
            )

            # Store chunk predictions
            for param_name, predictions in chunk_predictions.items():
                if param_name not in val_chunks_predictions:
                    val_chunks_predictions[param_name] = []
                val_chunks_predictions[param_name].append(predictions)

        return val_chunks_predictions

    def _concatenate_chunk_predictions(
        self, val_chunks_predictions: dict[str, list[NDArray[Any]]]
    ) -> dict[str, NDArray[Any]]:
        """Concatenate predictions from all chunks."""
        fold_predictions = {}
        for param_name, chunk_list in val_chunks_predictions.items():
            fold_predictions[param_name] = np.concatenate(chunk_list)
        return fold_predictions

    def _store_fold_predictions(
        self,
        fold_predictions: dict[str, NDArray[Any]],
        nuisance_estimates: dict[str, NDArray[Any]],
        n_samples: int,
        fold_idx: int,
    ) -> None:
        """Store fold predictions in the global nuisance estimates."""
        val_indices = self.cross_fit_data_.val_indices[fold_idx]

        for param_name, predictions in fold_predictions.items():
            if param_name not in nuisance_estimates:
                nuisance_estimates[param_name] = np.full(n_samples, np.nan)
            nuisance_estimates[param_name][val_indices] = predictions

    def _track_fold_performance(self, fold_start_time: float) -> None:
        """Track performance metrics for the current fold."""
        fold_end_time = time.perf_counter()
        if not hasattr(self, "_fold_timings_"):
            self._fold_timings_ = []
        self._fold_timings_.append(fold_end_time - fold_start_time)

    def _validate_nuisance_estimates(
        self, nuisance_estimates: dict[str, NDArray[Any]]
    ) -> None:
        """Validate that all samples have nuisance estimates."""
        for param_name, estimates in nuisance_estimates.items():
            if np.any(np.isnan(estimates)):
                raise ValueError(
                    f"Some samples missing nuisance estimates for {param_name}"
                )

    def get_cross_validation_performance(self) -> dict[str, Any]:
        """Get cross-validation performance metrics.

        Returns:
            Dictionary with performance metrics for each fold
        """
        if self.cross_fit_data_ is None:
            raise ValueError("Must perform cross-fitting first")

        results = {
            "n_folds": self.cv_folds,
            "fold_sizes": [
                len(indices) for indices in self.cross_fit_data_.val_indices
            ],
            "parallel_config": {
                "n_jobs": self.parallel_config.n_jobs,
                "backend": self.parallel_config.parallel_backend,
                "parallel_enabled": JOBLIB_AVAILABLE
                and self.parallel_config.n_jobs != 1,
            },
        }

        # Add timing information if available
        if self._fold_timings_:
            results["timing"] = {
                "fold_times_seconds": self._fold_timings_,
                "total_time_seconds": sum(self._fold_timings_),
                "mean_fold_time": np.mean(self._fold_timings_),
                "std_fold_time": np.std(self._fold_timings_),
            }

        # Add parallel speedup if available
        if self._parallel_speedup_ is not None:
            results["parallel_speedup"] = self._parallel_speedup_

        # Add memory usage if available
        if self._fold_memory_usage_:
            results["memory_usage"] = {
                "fold_memory_mb": self._fold_memory_usage_,
                "total_memory_mb": sum(self._fold_memory_usage_),
                "peak_memory_mb": max(self._fold_memory_usage_),
            }

        # Add caching statistics
        if self.parallel_config.enable_caching:
            results["caching"] = self.get_cache_statistics()

        return results

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get detailed performance metrics from cross-fitting.

        Returns:
            Dictionary with comprehensive performance analysis
        """
        base_metrics = self.get_cross_validation_performance()

        # Add performance analysis
        performance_analysis = {
            "efficiency_metrics": {
                "parallel_backend_used": self.parallel_config.parallel_backend,
                "jobs_configured": self.parallel_config.n_jobs,
                "actual_speedup": self._parallel_speedup_,
                "theoretical_max_speedup": min(
                    self.cv_folds, self.parallel_config.n_jobs
                )
                if self.parallel_config.n_jobs > 0
                else 1,
            },
            "resource_usage": {
                "gc_per_fold_enabled": self.parallel_config.enable_gc_per_fold,
                "memory_monitoring_enabled": self.parallel_config.memory_monitoring,
                "chunked_processing_used": len(self._fold_timings_) == 0,  # Heuristic
            },
        }

        # Combine all metrics
        base_metrics["performance_analysis"] = performance_analysis
        return base_metrics

    def _compute_data_hash(
        self, X: NDArray[Any], y: NDArray[Any], treatment: Optional[NDArray[Any]] = None
    ) -> str:
        """Compute hash for caching purposes based on input data."""
        if not self.parallel_config.enable_caching:
            return "no_caching"

        # Create a hash based on data shape and a sample of values
        hash_input = f"{X.shape}_{y.shape}"

        if treatment is not None:
            hash_input += f"_{treatment.shape}"

        # Add sample of data values for uniqueness (avoid hashing entire arrays for performance)
        if X.size > 0:
            hash_input += (
                f"_{np.mean(X[: min(100, len(X))])}_{np.std(X[: min(100, len(X))])}"
            )
        if y.size > 0:
            hash_input += (
                f"_{np.mean(y[: min(100, len(y))])}_{np.std(y[: min(100, len(y))])}"
            )

        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _fit_nuisance_models_cached(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        treatment_train: Optional[NDArray[Any]] = None,
        fold_idx: int = 0,
    ) -> dict[str, Any]:
        """Fit nuisance models with caching support."""
        if not self.parallel_config.enable_caching:
            return self._fit_nuisance_models(X_train, y_train, treatment_train)

        # Create cache key
        cache_key = f"{self._compute_data_hash(X_train, y_train, treatment_train)}_fold_{fold_idx}"

        # Check cache first
        if cache_key in self._model_cache_:
            self._cache_hits_ += 1
            # Return cloned models to avoid state interference
            cached_models = self._model_cache_[cache_key]
            return {key: clone(model) for key, model in cached_models.items()}

        # Cache miss - fit models
        self._cache_misses_ += 1
        fitted_models = self._fit_nuisance_models(X_train, y_train, treatment_train)

        # Store in cache (clone to avoid state modification)
        if len(self._model_cache_) < self.parallel_config.cache_size:
            self._model_cache_[cache_key] = {
                key: clone(model) for key, model in fitted_models.items()
            }

        return fitted_models

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get model caching performance statistics."""
        total_requests = self._cache_hits_ + self._cache_misses_
        hit_rate = self._cache_hits_ / total_requests if total_requests > 0 else 0.0

        return {
            "cache_enabled": self.parallel_config.enable_caching,
            "cache_hits": self._cache_hits_,
            "cache_misses": self._cache_misses_,
            "hit_rate": hit_rate,
            "cache_size": len(self._model_cache_),
            "cache_efficiency": "high"
            if hit_rate > 0.7
            else "medium"
            if hit_rate > 0.3
            else "low",
        }


def create_cross_fit_data(
    X: NDArray[Any] | pd.DataFrame,
    y: NDArray[Any] | pd.Series,
    treatment: NDArray[Any] | Optional[pd.Series] = None,
    cv_folds: int = 5,
    stratified: bool = True,
    random_state: Optional[int] = None,
    parallel_config: Optional[ParallelCrossFittingConfig] = None,
) -> CrossFitData:
    """Create cross-fitting data splits.

    Args:
        X: Covariate matrix
        y: Outcome vector
        treatment: Treatment vector (for stratified splitting)
        cv_folds: Number of cross-validation folds
        stratified: Whether to use stratified cross-validation
        random_state: Random state for reproducibility

    Returns:
        CrossFitData object with splits
    """
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = np.asarray(y.values)
    if isinstance(treatment, pd.Series):
        treatment = np.asarray(treatment.values)

    X = np.array(X)
    y = np.array(y)
    if treatment is not None:
        treatment = np.array(treatment)

    # Create cross-validation splitter
    if stratified and treatment is not None:
        cv_splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        splits = list(cv_splitter.split(X, treatment))
    else:
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        splits = list(cv_splitter.split(X))

    # Extract indices and data for each fold
    train_indices = []
    val_indices = []
    X_train_folds = []
    X_val_folds = []
    y_train_folds = []
    y_val_folds = []
    treatment_train_folds = []
    treatment_val_folds = []

    for train_idx, val_idx in splits:
        train_indices.append(train_idx)
        val_indices.append(val_idx)

        X_train_folds.append(X[train_idx])
        X_val_folds.append(X[val_idx])
        y_train_folds.append(y[train_idx])
        y_val_folds.append(y[val_idx])

        if treatment is not None:
            treatment_train_folds.append(treatment[train_idx])
            treatment_val_folds.append(treatment[val_idx])

    return CrossFitData(
        n_folds=cv_folds,
        train_indices=train_indices,
        val_indices=val_indices,
        X_train_folds=X_train_folds,
        X_val_folds=X_val_folds,
        y_train_folds=y_train_folds,
        y_val_folds=y_val_folds,
        treatment_train_folds=treatment_train_folds if treatment is not None else None,
        treatment_val_folds=treatment_val_folds if treatment is not None else None,
    )
