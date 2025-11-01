"""Performance benchmarking and profiling utilities for causal inference estimators.

This module provides comprehensive benchmarking tools to measure performance
characteristics and validate scalability to large datasets.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.base import BaseEstimator
from ..data.synthetic import SyntheticDataGenerator


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""

    n_samples: int
    n_features: int
    method_name: str

    # Timing results
    fit_time: float
    predict_time: Optional[float] = None
    ate_time: Optional[float] = None
    total_time: Optional[float] = None

    # Memory results
    peak_memory_mb: Optional[float] = None
    memory_efficiency: Optional[float] = None  # MB per 1K samples

    # Quality metrics
    ate_estimate: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None

    # Scalability metrics
    time_per_sample_ms: Optional[float] = None
    samples_per_second: Optional[float] = None

    # Error information
    error: Optional[str] = None

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        if self.fit_time is not None and self.n_samples > 0:
            self.time_per_sample_ms = (self.fit_time * 1000) / self.n_samples
            self.samples_per_second = self.n_samples / self.fit_time

        if self.fit_time and self.ate_time:
            self.total_time = self.fit_time + self.ate_time

        if self.peak_memory_mb is not None and self.n_samples > 0:
            self.memory_efficiency = self.peak_memory_mb / (self.n_samples / 1000)


@contextmanager
def memory_profiler() -> Iterator[dict[str, float]]:
    """Context manager to profile memory usage."""
    try:
        import os

        import psutil  # type: ignore[import-untyped]

        process = psutil.Process(os.getpid())

        # Force garbage collection before starting
        gc.collect()

        memory_info = {"start_mb": process.memory_info().rss / 1024 / 1024}
        peak_memory = memory_info["start_mb"]

        # Store original memory_info method to track peak
        original_memory_info = process.memory_info

        def track_memory() -> Any:
            nonlocal peak_memory
            info = original_memory_info()
            current_mb = info.rss / 1024 / 1024
            peak_memory = max(peak_memory, current_mb)
            return info

        # Monkey patch to track peak memory
        process.memory_info = track_memory

        yield memory_info

        # Restore original method
        process.memory_info = original_memory_info

        # Final measurements
        gc.collect()
        end_memory = process.memory_info().rss / 1024 / 1024

        memory_info.update(
            {
                "end_mb": end_memory,
                "peak_mb": peak_memory,
                "used_mb": end_memory - memory_info["start_mb"],
                "peak_used_mb": peak_memory - memory_info["start_mb"],
            }
        )

    except ImportError:
        # Fallback if psutil not available
        memory_info = {
            "start_mb": 0,
            "end_mb": 0,
            "peak_mb": 0,
            "used_mb": 0,
            "peak_used_mb": 0,
        }
        yield memory_info


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for causal inference estimators."""

    def __init__(
        self,
        sample_sizes: Optional[list[int]] = None,
        n_features_list: Optional[list[int]] = None,
        n_trials: int = 3,
        random_state: int = 42,
    ):
        """Initialize benchmark suite.

        Args:
            sample_sizes: List of sample sizes to test
            n_features_list: List of feature counts to test
            n_trials: Number of trials per configuration
            random_state: Random seed for reproducibility
        """
        self.sample_sizes = sample_sizes or [1000, 5000, 10000, 50000, 100000]
        self.n_features_list = n_features_list or [5, 10, 20, 50]
        self.n_trials = n_trials
        self.random_state = random_state

        self.results: list[BenchmarkResult] = []

    def benchmark_estimator(
        self,
        estimator_class: type[BaseEstimator],
        estimator_params: Optional[dict[str, Any]] = None,
        dataset_params: Optional[dict[str, Any]] = None,
    ) -> list[BenchmarkResult]:
        """Benchmark an estimator across different configurations.

        Args:
            estimator_class: Estimator class to benchmark
            estimator_params: Parameters for estimator initialization
            dataset_params: Parameters for synthetic data generation

        Returns:
            List of benchmark results
        """
        if estimator_params is None:
            estimator_params = {}
        if dataset_params is None:
            dataset_params = {}

        benchmark_results = []

        for n_samples in self.sample_sizes:
            for n_features in self.n_features_list:
                print(
                    f"Benchmarking {estimator_class.__name__} with "
                    f"n_samples={n_samples}, n_features={n_features}"
                )

                trial_results = []

                for trial in range(self.n_trials):
                    result = self._single_benchmark(
                        estimator_class,
                        estimator_params,
                        n_samples,
                        n_features,
                        trial,
                        dataset_params,
                    )
                    trial_results.append(result)

                # Average results across trials
                avg_result = self._average_results(trial_results)
                benchmark_results.append(avg_result)

        self.results.extend(benchmark_results)
        return benchmark_results

    def _single_benchmark(
        self,
        estimator_class: type[BaseEstimator],
        estimator_params: dict[str, Any],
        n_samples: int,
        n_features: int,
        trial: int,
        dataset_params: dict[str, Any],
    ) -> BenchmarkResult:
        """Run a single benchmark trial."""
        # Generate synthetic data
        np.random.seed(self.random_state + trial)

        try:
            # Generate data with proper parameters
            data_params = {
                "n_samples": n_samples,
                "n_confounders": n_features,
                "treatment_effect": 2.0,
                "random_state": self.random_state + trial,
                **dataset_params,
            }

            generator = SyntheticDataGenerator(
                random_state=data_params.get("random_state", self.random_state)
            )
            treatment_data, outcome_data, covariate_data = (
                generator.generate_linear_binary_treatment(
                    n_samples=n_samples,
                    n_confounders=n_features,
                    treatment_effect=data_params.get("treatment_effect", 2.0),
                )
            )

            # Initialize estimator
            estimator = estimator_class(**estimator_params)

            # Benchmark fitting
            with memory_profiler() as memory_info:
                start_time = time.time()
                estimator.fit(treatment_data, outcome_data, covariate_data)
                fit_time = time.time() - start_time

                # Benchmark ATE estimation
                ate_start = time.time()
                causal_effect = estimator.estimate_ate()
                ate_time = time.time() - ate_start

            # Create result
            result = BenchmarkResult(
                n_samples=n_samples,
                n_features=n_features,
                method_name=estimator_class.__name__,
                fit_time=fit_time,
                ate_time=ate_time,
                peak_memory_mb=memory_info["peak_used_mb"],
                ate_estimate=causal_effect.ate,
                confidence_interval=(
                    causal_effect.ate_ci_lower,
                    causal_effect.ate_ci_upper,
                )
                if causal_effect.ate_ci_lower is not None
                and causal_effect.ate_ci_upper is not None
                else None,
            )

        except Exception as e:
            result = BenchmarkResult(
                n_samples=n_samples,
                n_features=n_features,
                method_name=estimator_class.__name__,
                fit_time=0.0,
                error=str(e),
            )

        return result

    def _average_results(self, results: list[BenchmarkResult]) -> BenchmarkResult:
        """Average results across multiple trials."""
        if not results:
            raise ValueError("No results to average")

        # Filter out error results for averaging
        valid_results = [r for r in results if r.error is None]

        if not valid_results:
            # Return first error result if all failed
            return results[0]

        # Average timing metrics
        avg_fit_time = np.mean([r.fit_time for r in valid_results])
        avg_ate_time = np.mean([r.ate_time for r in valid_results if r.ate_time])
        avg_peak_memory = np.mean(
            [r.peak_memory_mb for r in valid_results if r.peak_memory_mb]
        )

        # Use first valid result as template
        template = valid_results[0]

        return BenchmarkResult(
            n_samples=template.n_samples,
            n_features=template.n_features,
            method_name=template.method_name,
            fit_time=float(avg_fit_time),
            ate_time=float(avg_ate_time) if avg_ate_time else None,
            peak_memory_mb=float(avg_peak_memory) if avg_peak_memory else None,
            ate_estimate=float(
                np.mean([r.ate_estimate for r in valid_results if r.ate_estimate])
            )
            if any(r.ate_estimate for r in valid_results)
            else None,
        )

    def compare_estimators(
        self,
        estimator_configs: list[tuple[type[BaseEstimator], dict[str, Any]]],
        dataset_params: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Compare multiple estimators side by side.

        Args:
            estimator_configs: List of (estimator_class, params) tuples
            dataset_params: Parameters for data generation

        Returns:
            DataFrame with comparison results
        """
        all_results = []

        for estimator_class, estimator_params in estimator_configs:
            results = self.benchmark_estimator(
                estimator_class, estimator_params, dataset_params
            )
            all_results.extend(results)

        # Convert to DataFrame for analysis
        results_data = []
        for result in all_results:
            results_data.append(
                {
                    "method": result.method_name,
                    "n_samples": result.n_samples,
                    "n_features": result.n_features,
                    "fit_time": result.fit_time,
                    "ate_time": result.ate_time,
                    "total_time": result.total_time,
                    "peak_memory_mb": result.peak_memory_mb,
                    "memory_efficiency": result.memory_efficiency,
                    "time_per_sample_ms": result.time_per_sample_ms,
                    "samples_per_second": result.samples_per_second,
                    "ate_estimate": result.ate_estimate,
                    "error": result.error,
                }
            )

        return pd.DataFrame(results_data)

    def scalability_analysis(self, results_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze scalability characteristics from benchmark results.

        Args:
            results_df: DataFrame from compare_estimators

        Returns:
            Dictionary with scalability metrics
        """
        analysis = {}

        for method in results_df["method"].unique():
            method_data = results_df[results_df["method"] == method].copy()
            method_data = method_data[method_data["error"].isna()]  # Filter out errors

            if len(method_data) < 2:
                continue

            # Fit linear models to analyze scaling
            from sklearn.linear_model import LinearRegression

            # Time scaling with sample size
            if not method_data["fit_time"].isna().all():
                X = method_data[["n_samples"]].values
                y = method_data["fit_time"].values

                time_model = LinearRegression().fit(X, y)
                time_scaling = time_model.coef_[0]  # Slope

                # Memory scaling
                if not method_data["peak_memory_mb"].isna().all():
                    y_mem = method_data["peak_memory_mb"].values
                    memory_model = LinearRegression().fit(X, y_mem)
                    memory_scaling = memory_model.coef_[0]
                else:
                    memory_scaling = None

                analysis[method] = {
                    "time_scaling_per_sample": time_scaling,
                    "memory_scaling_per_sample": memory_scaling,
                    "max_samples_tested": method_data["n_samples"].max(),
                    "min_fit_time": method_data["fit_time"].min(),
                    "max_fit_time": method_data["fit_time"].max(),
                    "avg_memory_efficiency": method_data["memory_efficiency"].mean(),
                }

        return analysis

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate a comprehensive benchmark report.

        Args:
            results_df: DataFrame with benchmark results

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "CAUSAL INFERENCE PERFORMANCE BENCHMARK REPORT",
            "=" * 60,
            "",
        ]

        # Summary statistics
        report_lines.extend(
            [
                "SUMMARY STATISTICS",
                "-" * 20,
                f"Total configurations tested: {len(results_df)}",
                f"Methods compared: {', '.join(results_df['method'].unique())}",
                f"Sample sizes: {sorted(results_df['n_samples'].unique())}",
                f"Feature counts: {sorted(results_df['n_features'].unique())}",
                "",
            ]
        )

        # Performance by method
        report_lines.extend(
            [
                "PERFORMANCE BY METHOD",
                "-" * 25,
            ]
        )

        for method in results_df["method"].unique():
            method_data = results_df[results_df["method"] == method]
            valid_data = method_data[method_data["error"].isna()]

            if len(valid_data) == 0:
                report_lines.append(f"{method}: All runs failed")
                continue

            avg_fit_time = valid_data["fit_time"].mean()
            max_samples = valid_data["n_samples"].max()
            avg_memory = valid_data["peak_memory_mb"].mean()

            report_lines.extend(
                [
                    f"{method}:",
                    f"  Average fit time: {avg_fit_time:.3f}s",
                    f"  Max samples tested: {max_samples:,}",
                    f"  Average memory usage: {avg_memory:.1f} MB",
                    f"  Success rate: {len(valid_data) / len(method_data) * 100:.1f}%",
                    "",
                ]
            )

        # Scalability analysis
        scalability = self.scalability_analysis(results_df)
        if scalability:
            report_lines.extend(
                [
                    "SCALABILITY ANALYSIS",
                    "-" * 20,
                ]
            )

            for method, metrics in scalability.items():
                report_lines.extend(
                    [
                        f"{method}:",
                        f"  Time scaling: {metrics['time_scaling_per_sample']:.2e} s/sample",
                        f"  Memory scaling: {metrics.get('memory_scaling_per_sample', 'N/A'):.2e} MB/sample"
                        if metrics.get("memory_scaling_per_sample")
                        else "  Memory scaling: N/A",
                        f"  Memory efficiency: {metrics['avg_memory_efficiency']:.2f} MB/1K samples",
                        "",
                    ]
                )

        # Recommendations
        report_lines.extend(
            [
                "RECOMMENDATIONS",
                "-" * 15,
            ]
        )

        fastest_method = results_df.groupby("method")["fit_time"].mean().idxmin()
        most_memory_efficient = (
            results_df.groupby("method")["memory_efficiency"].mean().idxmin()
        )

        report_lines.extend(
            [
                f"Fastest method overall: {fastest_method}",
                f"Most memory efficient: {most_memory_efficient}",
                "",
                "For 1M+ sample datasets:",
            ]
        )

        # Check which methods can handle large datasets
        large_sample_data = results_df[results_df["n_samples"] >= 50000]
        if not large_sample_data.empty:
            successful_methods = large_sample_data[large_sample_data["error"].isna()][
                "method"
            ].unique()
            report_lines.extend(
                [
                    f"  Recommended methods: {', '.join(successful_methods)}",
                    "  Avoid methods with high memory scaling",
                ]
            )

        return "\n".join(report_lines)


class ScalabilityTester:
    """Test scalability limits and performance targets."""

    def __init__(self, target_memory_gb: float = 8.0, target_time_minutes: float = 5.0):
        """Initialize scalability tester.

        Args:
            target_memory_gb: Memory usage target
            target_time_minutes: Time target for operations
        """
        self.target_memory_gb = target_memory_gb
        self.target_time_minutes = target_time_minutes

    def test_memory_target(
        self,
        estimator: BaseEstimator,
        n_samples: int = 1000000,
    ) -> dict[str, Any]:
        """Test if estimator meets memory usage targets.

        Args:
            estimator: Estimator to test
            n_samples: Number of samples to test with

        Returns:
            Test results dictionary
        """
        # Generate test data
        generator = SyntheticDataGenerator()
        treatment_data, outcome_data, covariate_data = (
            generator.generate_linear_binary_treatment(
                n_samples=n_samples,
                n_confounders=10,
                treatment_effect=2.0,
            )
        )

        # Test memory usage
        with memory_profiler() as memory_info:
            try:
                estimator.fit(treatment_data, outcome_data, covariate_data)
                _ = estimator.estimate_ate()
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)

        memory_used_gb = memory_info["peak_used_mb"] / 1024
        meets_target = memory_used_gb <= self.target_memory_gb

        return {
            "n_samples": n_samples,
            "memory_used_gb": memory_used_gb,
            "memory_target_gb": self.target_memory_gb,
            "meets_memory_target": meets_target,
            "success": success,
            "error": error,
        }

    def test_time_target(
        self,
        estimator: BaseEstimator,
        n_samples: int = 100000,
        operation: str = "bootstrap",
    ) -> dict[str, Any]:
        """Test if estimator meets time targets.

        Args:
            estimator: Estimator to test
            n_samples: Number of samples to test with
            operation: Operation to test ("fit", "ate", "bootstrap")

        Returns:
            Test results dictionary
        """
        # Generate test data
        generator = SyntheticDataGenerator()
        treatment_data, outcome_data, covariate_data = (
            generator.generate_linear_binary_treatment(
                n_samples=n_samples,
                n_confounders=10,
                treatment_effect=2.0,
            )
        )

        try:
            # Always fit first
            start_time = time.time()
            estimator.fit(treatment_data, outcome_data, covariate_data)
            fit_time = time.time() - start_time

            if operation == "fit":
                test_time = fit_time
            elif operation == "ate":
                start_time = time.time()
                _ = estimator.estimate_ate()
                test_time = time.time() - start_time
            elif operation == "bootstrap":
                # Test bootstrap specifically if available
                start_time = time.time()
                estimator.estimate_ate()
                test_time = time.time() - start_time
            else:
                raise ValueError(f"Unknown operation: {operation}")

            success = True
            error = None

        except Exception as e:
            test_time = float("inf")
            success = False
            error = str(e)

        test_time_minutes = test_time / 60
        meets_target = test_time_minutes <= self.target_time_minutes

        return {
            "n_samples": n_samples,
            "operation": operation,
            "time_minutes": test_time_minutes,
            "time_target_minutes": self.target_time_minutes,
            "meets_time_target": meets_target,
            "success": success,
            "error": error,
        }
