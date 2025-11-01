"""Performance profiling and benchmarking utilities for causal inference estimators.

This module provides tools for monitoring performance, memory usage, and scalability
of causal inference estimators, particularly for DoublyRobustML and other ML-based methods.
"""
# ruff: noqa: N803

from __future__ import annotations

import gc
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Memory profiling will be limited.")

try:
    from memory_profiler import memory_usage

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    warnings.warn("memory_profiler not available. Using basic memory monitoring.")

__all__ = [
    "PerformanceConfig",
    "PerformanceProfiler",
    "PerformanceMetrics",
    "benchmark_estimator",
]


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization features."""

    n_jobs: int = -1
    parallel_backend: str = "threading"
    max_memory_gb: float = 4.0
    chunk_size: int = 10000
    enable_caching: bool = True
    timeout_per_fold_minutes: float = 10.0
    enable_profiling: bool = False
    gc_threshold: int = 2


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    runtime_seconds: float
    peak_memory_mb: float
    memory_usage_profile: list[float]
    cpu_usage_percent: float
    parallel_speedup: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    memory_efficiency: Optional[float] = None


class PerformanceProfiler:
    """Performance profiling and monitoring for causal inference estimators."""

    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize the performance profiler.

        Args:
            config: Performance configuration. Uses defaults if None.
        """
        self.config = config or PerformanceConfig()
        self._baseline_memory: Optional[float] = None
        self._start_time: Optional[float] = None
        self._memory_profile: list[float] = []

    def profile_runtime(
        self,
        estimator: Any,
        X: NDArray[Any],
        Y: NDArray[Any],
        A: NDArray[Any],
        n_runs: int = 3,
    ) -> dict[str, Any]:
        """Profile runtime performance of an estimator.

        Args:
            estimator: The causal inference estimator to profile
            X: Covariate matrix
            Y: Outcome vector
            A: Treatment vector
            n_runs: Number of runs for averaging

        Returns:
            Dictionary containing runtime metrics
        """
        runtimes = []

        for run in range(n_runs):
            # Reset estimator state
            if hasattr(estimator, "_reset"):
                estimator._reset()

            # Time the fitting process
            start_time = time.perf_counter()

            try:
                estimator.fit(
                    treatment=self._create_treatment_data(A),
                    outcome=self._create_outcome_data(Y),
                    covariates=self._create_covariate_data(X),
                )
                end_time = time.perf_counter()
                runtime = end_time - start_time
                runtimes.append(runtime)

            except Exception as e:
                warnings.warn(f"Runtime profiling failed on run {run + 1}: {e}")
                continue

        if not runtimes:
            return {"error": "All profiling runs failed"}

        return {
            "mean_runtime": float(np.mean(runtimes)),
            "std_runtime": float(np.std(runtimes)),
            "min_runtime": float(np.min(runtimes)),
            "max_runtime": float(np.max(runtimes)),
            "median_runtime": float(np.median(runtimes)),
            "n_successful_runs": len(runtimes),
            "n_total_runs": n_runs,
            "runtimes": runtimes,
        }

    def profile_memory(
        self,
        estimator: Any,
        X: NDArray[Any],
        Y: NDArray[Any],
        A: NDArray[Any],
        interval: float = 0.1,
    ) -> dict[str, Any]:
        """Profile memory usage of an estimator.

        Args:
            estimator: The causal inference estimator to profile
            X: Covariate matrix
            Y: Outcome vector
            A: Treatment vector
            interval: Memory sampling interval in seconds

        Returns:
            Dictionary containing memory metrics
        """

        def _fit_estimator():
            """Helper function to fit estimator for memory profiling."""
            estimator.fit(
                treatment=self._create_treatment_data(A),
                outcome=self._create_outcome_data(Y),
                covariates=self._create_covariate_data(X),
            )

        # Get baseline memory
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        else:
            baseline_memory = 0.0

        if MEMORY_PROFILER_AVAILABLE:
            # Use memory_profiler for detailed profiling
            memory_profile = memory_usage(_fit_estimator, interval=interval)
            peak_memory = max(memory_profile)
            memory_growth = peak_memory - baseline_memory

        else:
            # Fallback to basic memory monitoring
            gc.collect()  # Clean up before measurement

            if PSUTIL_AVAILABLE:
                start_memory = process.memory_info().rss / 1024 / 1024
            else:
                start_memory = baseline_memory

            _fit_estimator()

            if PSUTIL_AVAILABLE:
                end_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = end_memory
                memory_growth = end_memory - start_memory
                memory_profile = [start_memory, end_memory]
            else:
                peak_memory = 0.0
                memory_growth = 0.0
                memory_profile = [0.0, 0.0]

        return {
            "baseline_memory_mb": float(baseline_memory),
            "peak_memory_mb": float(peak_memory),
            "memory_growth_mb": float(memory_growth),
            "memory_profile": memory_profile,
            "memory_efficiency": float(memory_growth / len(X)) if len(X) > 0 else 0.0,
            "profiler_used": "memory_profiler"
            if MEMORY_PROFILER_AVAILABLE
            else "basic",
        }

    def benchmark_scalability(
        self,
        estimator_factory: Callable[[], Any],
        sample_sizes: list[int],
        n_features: int = 10,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Benchmark estimator scalability across different sample sizes.

        Args:
            estimator_factory: Function that creates a fresh estimator instance
            sample_sizes: List of sample sizes to test
            n_features: Number of features for synthetic data
            random_state: Random seed for reproducibility

        Returns:
            DataFrame with scalability results
        """
        results = []

        if random_state is not None:
            np.random.seed(random_state)

        for n_samples in sample_sizes:
            print(f"Benchmarking sample size: {n_samples}")

            # Generate synthetic data
            X, Y, A = self._generate_synthetic_data(n_samples, n_features, random_state)

            # Create fresh estimator
            estimator = estimator_factory()

            # Profile runtime
            runtime_metrics = self.profile_runtime(estimator, X, Y, A, n_runs=1)

            # Profile memory
            estimator = estimator_factory()  # Fresh instance
            memory_metrics = self.profile_memory(estimator, X, Y, A)

            # Store results
            result = {
                "n_samples": n_samples,
                "n_features": n_features,
                "runtime_seconds": runtime_metrics.get("mean_runtime", np.nan),
                "peak_memory_mb": memory_metrics.get("peak_memory_mb", np.nan),
                "memory_per_sample_kb": memory_metrics.get("memory_growth_mb", 0)
                * 1024
                / n_samples,
                "runtime_per_sample_ms": runtime_metrics.get("mean_runtime", 0)
                * 1000
                / n_samples,
            }

            results.append(result)

        df = pd.DataFrame(results)

        # Add scalability analysis
        if len(df) > 1:
            # Compute complexity estimates (log-log regression)
            log_n = np.log(df["n_samples"])
            log_time = np.log(df["runtime_seconds"].replace(0, np.nan))
            log_memory = np.log(df["peak_memory_mb"].replace(0, np.nan))

            # Time complexity: T(n) ∝ n^α
            if not log_time.isna().all():
                time_complexity_coef = np.polyfit(
                    log_n[~log_time.isna()], log_time[~log_time.isna()], 1
                )[0]
                df["time_complexity_estimate"] = time_complexity_coef

            # Memory complexity: M(n) ∝ n^β
            if not log_memory.isna().all():
                memory_complexity_coef = np.polyfit(
                    log_n[~log_memory.isna()], log_memory[~log_memory.isna()], 1
                )[0]
                df["memory_complexity_estimate"] = memory_complexity_coef

        return df

    def _create_treatment_data(self, A: NDArray[Any]) -> Any:
        """Create TreatmentData from array."""
        from ..core.base import TreatmentData

        return TreatmentData(values=A, treatment_type="binary")

    def _create_outcome_data(self, Y: NDArray[Any]) -> Any:
        """Create OutcomeData from array."""
        from ..core.base import OutcomeData

        return OutcomeData(values=Y, outcome_type="continuous")

    def _create_covariate_data(self, X: NDArray[Any]) -> Any:
        """Create CovariateData from array."""
        from ..core.base import CovariateData

        return CovariateData(values=X)

    def _generate_synthetic_data(
        self, n_samples: int, n_features: int, random_state: Optional[int] = None
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Generate synthetic data for benchmarking."""
        if random_state is not None:
            np.random.seed(random_state)

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate treatment with confounding
        propensity = 1 / (1 + np.exp(-X[:, 0] - 0.5 * X[:, 1]))
        A = np.random.binomial(1, propensity)

        # Generate outcome with treatment effect
        treatment_effect = 2.0
        Y = (
            X[:, 0]
            + 0.5 * X[:, 1]
            + treatment_effect * A
            + np.random.randn(n_samples) * 0.5
        )

        return X, Y, A


def benchmark_estimator(
    estimator_factory: Callable[[], Any],
    sample_sizes: Optional[list[int]] = None,
    n_features: int = 10,
    performance_config: Optional[PerformanceConfig] = None,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Comprehensive benchmarking of a causal inference estimator.

    Args:
        estimator_factory: Function that creates estimator instances
        sample_sizes: List of sample sizes to test
        n_features: Number of features for synthetic data
        performance_config: Performance configuration
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing comprehensive benchmark results
    """
    if sample_sizes is None:
        sample_sizes = [100, 500, 1000, 2000, 5000]

    profiler = PerformanceProfiler(performance_config)

    # Scalability benchmarks
    scalability_results = profiler.benchmark_scalability(
        estimator_factory, sample_sizes, n_features, random_state
    )

    # Test parallel speedup if configured
    parallel_results = None
    if performance_config and performance_config.n_jobs > 1:
        parallel_results = _benchmark_parallel_speedup(
            estimator_factory, performance_config, random_state
        )

    return {
        "scalability_results": scalability_results,
        "parallel_results": parallel_results,
        "config": performance_config,
        "benchmark_timestamp": time.time(),
    }


def _benchmark_parallel_speedup(
    estimator_factory: Callable[[], Any],
    config: PerformanceConfig,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Benchmark parallel processing speedup."""
    # Generate test data
    profiler = PerformanceProfiler()
    X, Y, A = profiler._generate_synthetic_data(1000, 10, random_state)

    # Test sequential (n_jobs=1)
    sequential_estimator = estimator_factory()
    if hasattr(sequential_estimator, "performance_config"):
        sequential_estimator.performance_config = PerformanceConfig(
            n_jobs=1, parallel_backend=config.parallel_backend
        )

    sequential_time = profiler.profile_runtime(sequential_estimator, X, Y, A, n_runs=1)

    # Test parallel
    parallel_estimator = estimator_factory()
    if hasattr(parallel_estimator, "performance_config"):
        parallel_estimator.performance_config = config

    parallel_time = profiler.profile_runtime(parallel_estimator, X, Y, A, n_runs=1)

    # Calculate speedup
    seq_runtime = sequential_time.get("mean_runtime", 0)
    par_runtime = parallel_time.get("mean_runtime", float("inf"))

    speedup = seq_runtime / par_runtime if par_runtime > 0 else 0.0

    return {
        "sequential_runtime": seq_runtime,
        "parallel_runtime": par_runtime,
        "speedup": speedup,
        "parallel_backend": config.parallel_backend,
        "n_jobs": config.n_jobs,
    }
