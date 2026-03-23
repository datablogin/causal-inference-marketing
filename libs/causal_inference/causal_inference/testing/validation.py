"""Statistical validation framework for Double Machine Learning estimators.

This module implements comprehensive validation tools to test the statistical
properties of DML estimators as specified in the DoubleML framework.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.base import BaseEstimator, CovariateData, OutcomeData, TreatmentData
from .synthetic import generate_synthetic_dml_data

__all__ = ["DMLValidator", "DMLBenchmark"]


class DMLValidator:
    """Statistical validation framework for Double Machine Learning estimators.

    This class implements the validation requirements from the DoubleML specification:
    - Bias validation: Check if |estimated_ate - true_ate| < threshold
    - Coverage validation: Check if 95% CI contains true ATE approximately 95% of time
    - Consistency validation: Check convergence properties across sample sizes

    Example:
        >>> from causal_inference.estimators.doubly_robust_ml import DoublyRobustMLEstimator
        >>> validator = DMLValidator()
        >>> estimator = DoublyRobustMLEstimator()
        >>> bias_results = validator.validate_bias(estimator, n_simulations=50)
        >>> print(f"Mean bias: {bias_results['mean_bias']:.4f}")
    """

    def __init__(self, verbose: bool = True):
        """Initialize the DML validator.

        Args:
            verbose: Whether to print detailed validation progress
        """
        self.verbose = verbose

    def validate_bias(
        self,
        estimator: BaseEstimator,
        n_simulations: int = 100,
        n_samples: int = 1000,
        n_features: int = 5,
        true_ate: float = 2.0,
        bias_threshold: float = 0.05,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Validate bias of DML estimator using synthetic data.

        Args:
            estimator: DML estimator to validate
            n_simulations: Number of simulation runs
            n_samples: Sample size per simulation
            n_features: Number of covariate features
            true_ate: True average treatment effect
            bias_threshold: Maximum allowable bias
            seed: Random seed for reproducibility

        Returns:
            Dictionary with bias validation results:
            - mean_bias: Average bias across simulations
            - bias_se: Standard error of bias estimates
            - abs_bias: Mean absolute bias
            - bias_exceeded_threshold: Number of times bias exceeded threshold
            - bias_pass_rate: Proportion of simulations within bias threshold
            - individual_estimates: List of ATE estimates from each simulation
        """
        # Memory efficiency warning for large simulations
        estimated_memory_mb = (n_simulations * n_samples * n_features * 8) / (
            1024 * 1024
        )  # 8 bytes per float
        if estimated_memory_mb > 1000:  # > 1GB
            import warnings

            warnings.warn(
                f"Large simulation may use ~{estimated_memory_mb:.1f} MB memory. "
                f"Consider reducing n_simulations ({n_simulations}) or n_samples ({n_samples}).",
                UserWarning,
                stacklevel=2,
            )

        if self.verbose:
            print(f"Validating bias with {n_simulations} simulations...")
            print(f"Target bias threshold: < {bias_threshold}")
            if estimated_memory_mb > 100:  # Show memory estimate for moderate usage
                print(f"Estimated memory usage: ~{estimated_memory_mb:.1f} MB")

        ate_estimates: list[float] = []
        np.random.seed(seed)

        for sim in range(n_simulations):
            if self.verbose and (sim + 1) % 20 == 0:
                print(f"  Simulation {sim + 1}/{n_simulations}")

            # Generate synthetic data with known true ATE
            sim_seed = seed + sim
            X, D, Y, _ = generate_synthetic_dml_data(
                n=n_samples,
                n_features=n_features,
                true_ate=true_ate,
                seed=sim_seed,
            )

            # Convert to data objects
            treatment_data = TreatmentData(
                values=pd.Series(D),
                name="treatment",
                treatment_type="binary",
            )
            outcome_data = OutcomeData(
                values=pd.Series(Y),
                name="outcome",
                outcome_type="continuous",
            )
            covariate_data = CovariateData(
                values=pd.DataFrame(
                    X, columns=[f"X{i + 1}" for i in range(n_features)]
                ),
                names=[f"X{i + 1}" for i in range(n_features)],
            )

            try:
                # Fit estimator and get ATE
                estimator.fit(treatment_data, outcome_data, covariate_data)
                causal_effect = estimator.estimate_ate()
                ate_estimates.append(causal_effect.ate)
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Simulation {sim + 1} failed: {e}")
                ate_estimates.append(np.nan)

        # Calculate bias statistics
        ate_estimates_array = np.array(ate_estimates)
        valid_estimates = ate_estimates_array[~np.isnan(ate_estimates_array)]

        if len(valid_estimates) == 0:
            raise ValueError("All simulations failed - cannot validate bias")

        bias_values = valid_estimates - true_ate
        mean_bias = np.mean(bias_values)
        bias_se = np.std(bias_values) / np.sqrt(len(bias_values))
        abs_bias = np.mean(np.abs(bias_values))
        bias_exceeded = np.sum(np.abs(bias_values) > bias_threshold)
        bias_pass_rate = 1 - (bias_exceeded / len(bias_values))

        results = {
            "mean_bias": mean_bias,
            "bias_se": bias_se,
            "abs_bias": abs_bias,
            "bias_exceeded_threshold": bias_exceeded,
            "bias_pass_rate": bias_pass_rate,
            "n_valid_simulations": len(valid_estimates),
            "individual_estimates": valid_estimates.tolist(),
            "true_ate": true_ate,
            "bias_threshold": bias_threshold,
        }

        if self.verbose:
            print("\n=== Bias Validation Results ===")
            print(f"Mean bias: {mean_bias:.4f} (SE: {bias_se:.4f})")
            print(f"Mean absolute bias: {abs_bias:.4f}")
            print(f"Pass rate: {bias_pass_rate:.1%} (threshold: {bias_threshold})")
            print(f"Valid simulations: {len(valid_estimates)}/{n_simulations}")

            if abs_bias <= bias_threshold:
                print("✅ BIAS TEST PASSED")
            else:
                print("❌ BIAS TEST FAILED")

        return results

    def validate_coverage(
        self,
        estimator: BaseEstimator,
        n_simulations: int = 100,
        n_samples: int = 1000,
        n_features: int = 5,
        true_ate: float = 2.0,
        nominal_coverage: float = 0.95,
        coverage_tolerance: float = 0.05,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Validate confidence interval coverage of DML estimator.

        Args:
            estimator: DML estimator to validate
            n_simulations: Number of simulation runs
            n_samples: Sample size per simulation
            n_features: Number of covariate features
            true_ate: True average treatment effect
            nominal_coverage: Target coverage rate (e.g., 0.95 for 95% CI)
            coverage_tolerance: Allowable deviation from nominal coverage
            seed: Random seed

        Returns:
            Dictionary with coverage validation results
        """
        # Memory efficiency warning for large simulations
        estimated_memory_mb = (n_simulations * n_samples * n_features * 8) / (
            1024 * 1024
        )  # 8 bytes per float
        if estimated_memory_mb > 1000:  # > 1GB
            import warnings

            warnings.warn(
                f"Large simulation may use ~{estimated_memory_mb:.1f} MB memory. "
                f"Consider reducing n_simulations ({n_simulations}) or n_samples ({n_samples}).",
                UserWarning,
                stacklevel=2,
            )

        if self.verbose:
            print(f"Validating coverage with {n_simulations} simulations...")
            print(f"Target coverage: {nominal_coverage:.1%} ± {coverage_tolerance:.1%}")
            if estimated_memory_mb > 100:  # Show memory estimate for moderate usage
                print(f"Estimated memory usage: ~{estimated_memory_mb:.1f} MB")

        coverage_results: list[bool | float] = []
        ci_widths: list[float] = []
        np.random.seed(seed)

        for sim in range(n_simulations):
            if self.verbose and (sim + 1) % 20 == 0:
                print(f"  Simulation {sim + 1}/{n_simulations}")

            # Generate synthetic data
            sim_seed = seed + sim
            X, D, Y, _ = generate_synthetic_dml_data(
                n=n_samples,
                n_features=n_features,
                true_ate=true_ate,
                seed=sim_seed,
            )

            # Convert to data objects
            treatment_data = TreatmentData(
                values=pd.Series(D),
                name="treatment",
                treatment_type="binary",
            )
            outcome_data = OutcomeData(
                values=pd.Series(Y),
                name="outcome",
                outcome_type="continuous",
            )
            covariate_data = CovariateData(
                values=pd.DataFrame(
                    X, columns=[f"X{i + 1}" for i in range(n_features)]
                ),
                names=[f"X{i + 1}" for i in range(n_features)],
            )

            try:
                # Fit estimator and get confidence interval
                estimator.fit(treatment_data, outcome_data, covariate_data)
                causal_effect = estimator.estimate_ate()

                # Check if confidence interval contains true ATE
                if (
                    causal_effect.ate_ci_lower is not None
                    and causal_effect.ate_ci_upper is not None
                ):
                    contains_true_ate = (
                        causal_effect.ate_ci_lower
                        <= true_ate
                        <= causal_effect.ate_ci_upper
                    )
                    ci_width = causal_effect.ate_ci_upper - causal_effect.ate_ci_lower

                    coverage_results.append(contains_true_ate)
                    ci_widths.append(ci_width)
                else:
                    # If CI not available, record as missing
                    coverage_results.append(np.nan)
                    ci_widths.append(np.nan)

            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Simulation {sim + 1} failed: {e}")
                coverage_results.append(np.nan)
                ci_widths.append(np.nan)

        # Calculate coverage statistics
        coverage_array = np.array(coverage_results)
        valid_coverage = coverage_array[~np.isnan(coverage_array)]

        if len(valid_coverage) == 0:
            raise ValueError("No valid confidence intervals - cannot validate coverage")

        empirical_coverage = np.mean(valid_coverage)
        coverage_se = np.sqrt(
            empirical_coverage * (1 - empirical_coverage) / len(valid_coverage)
        )

        # Test if coverage is within tolerance
        coverage_diff = abs(empirical_coverage - nominal_coverage)
        coverage_passes = coverage_diff <= coverage_tolerance

        # CI width statistics
        ci_widths_array = np.array(ci_widths)
        valid_widths = ci_widths_array[~np.isnan(ci_widths_array)]
        mean_ci_width = np.mean(valid_widths) if len(valid_widths) > 0 else np.nan

        results = {
            "empirical_coverage": empirical_coverage,
            "coverage_se": coverage_se,
            "nominal_coverage": nominal_coverage,
            "coverage_diff": coverage_diff,
            "coverage_passes": coverage_passes,
            "coverage_tolerance": coverage_tolerance,
            "mean_ci_width": mean_ci_width,
            "n_valid_cis": len(valid_coverage),
            "coverage_results": valid_coverage.tolist(),
        }

        if self.verbose:
            print("\n=== Coverage Validation Results ===")
            print(
                f"Empirical coverage: {empirical_coverage:.1%} (SE: {coverage_se:.3f})"
            )
            print(f"Target coverage: {nominal_coverage:.1%}")
            print(f"Coverage difference: {coverage_diff:.3f}")
            print(f"Mean CI width: {mean_ci_width:.3f}")
            print(f"Valid CIs: {len(valid_coverage)}/{n_simulations}")

            if coverage_passes:
                print("✅ COVERAGE TEST PASSED")
            else:
                print("❌ COVERAGE TEST FAILED")

        return results

    def validate_consistency(
        self,
        estimator: BaseEstimator,
        sample_sizes: Optional[list[int]] = None,
        n_simulations: int = 50,
        n_features: int = 5,
        true_ate: float = 2.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Validate consistency properties across different sample sizes.

        Tests whether estimates become more accurate as sample size increases,
        which is a key theoretical property of consistent estimators.

        Args:
            estimator: DML estimator to validate
            sample_sizes: List of sample sizes to test
            n_simulations: Number of simulations per sample size
            n_features: Number of covariate features
            true_ate: True average treatment effect
            seed: Random seed

        Returns:
            Dictionary with consistency validation results
        """
        if sample_sizes is None:
            sample_sizes = [500, 1000, 2000]

        if self.verbose:
            print(f"Validating consistency across sample sizes: {sample_sizes}")
            print(f"Simulations per size: {n_simulations}")

        consistency_results = {}
        np.random.seed(seed)

        for n_samples in sample_sizes:
            if self.verbose:
                print(f"\n--- Testing sample size: {n_samples} ---")

            # Run bias validation for this sample size
            bias_results = self.validate_bias(
                estimator=estimator,
                n_simulations=n_simulations,
                n_samples=n_samples,
                n_features=n_features,
                true_ate=true_ate,
                seed=seed,
            )

            consistency_results[n_samples] = {
                "mean_bias": bias_results["mean_bias"],
                "abs_bias": bias_results["abs_bias"],
                "bias_se": bias_results["bias_se"],
                "estimates": bias_results["individual_estimates"],
            }

        # Test if bias decreases with sample size (consistency)
        sample_sizes_sorted = sorted(sample_sizes)
        abs_biases = [consistency_results[n]["abs_bias"] for n in sample_sizes_sorted]

        # Simple test: is bias decreasing on average?
        bias_slope = np.polyfit(sample_sizes_sorted, abs_biases, 1)[0]
        consistency_passes = bias_slope < 0  # Negative slope indicates decreasing bias

        results = {
            "sample_sizes": sample_sizes_sorted,
            "abs_biases": abs_biases,
            "bias_slope": bias_slope,
            "consistency_passes": consistency_passes,
            "detailed_results": consistency_results,
        }

        if self.verbose:
            print("\n=== Consistency Validation Results ===")
            for n, abs_bias in zip(sample_sizes_sorted, abs_biases):
                print(f"Sample size {n}: |bias| = {abs_bias:.4f}")
            print(f"Bias slope: {bias_slope:.6f}")

            if consistency_passes:
                print("✅ CONSISTENCY TEST PASSED")
            else:
                print("❌ CONSISTENCY TEST FAILED")

        return results

    def validate_orthogonality(
        self,
        estimator: BaseEstimator,
        n_samples: int = 1000,
        n_features: int = 5,
        true_ate: float = 2.0,
        correlation_threshold: float = 0.05,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Validate orthogonality property of DML estimator.

        Tests whether residuals from outcome and treatment models are approximately
        uncorrelated, which is a key requirement for unbiased DML estimation.

        Args:
            estimator: DML estimator to validate
            n_samples: Number of observations
            n_features: Number of covariate features
            true_ate: True average treatment effect
            correlation_threshold: Maximum allowable correlation between residuals
            seed: Random seed

        Returns:
            Dictionary with orthogonality validation results
        """
        if self.verbose:
            print("Validating orthogonality property...")

        # Generate synthetic data
        X, D, Y, _ = generate_synthetic_dml_data(
            n=n_samples,
            n_features=n_features,
            true_ate=true_ate,
            seed=seed,
        )

        # Convert to data objects
        treatment_data = TreatmentData(
            values=pd.Series(D),
            name="treatment",
            treatment_type="binary",
        )
        outcome_data = OutcomeData(
            values=pd.Series(Y),
            name="outcome",
            outcome_type="continuous",
        )
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(n_features)]),
            names=[f"X{i + 1}" for i in range(n_features)],
        )

        # Fit estimator
        estimator.fit(treatment_data, outcome_data, covariate_data)

        # Check if estimator has method to get influence function (for orthogonality check)
        if hasattr(estimator, "get_influence_function"):
            influence_function = estimator.get_influence_function()
            if influence_function is not None:
                # For DML, orthogonality means influence function should be mean-zero
                # and have low correlation with nuisance function errors
                mean_influence = np.mean(influence_function)
                abs_mean_influence = abs(mean_influence)
                orthogonality_passes = abs_mean_influence < correlation_threshold

                results = {
                    "mean_influence_function": mean_influence,
                    "abs_mean_influence": abs_mean_influence,
                    "correlation_threshold": correlation_threshold,
                    "orthogonality_passes": orthogonality_passes,
                }
            else:
                results = {
                    "error": "Influence function not available",
                    "orthogonality_passes": None,
                }
        else:
            # Fallback: basic residual correlation check if available
            results = {
                "error": "Orthogonality check not implemented for this estimator",
                "orthogonality_passes": None,
            }

        if self.verbose:
            print("\n=== Orthogonality Validation Results ===")
            if results.get("orthogonality_passes") is not None:
                print(
                    f"Mean influence function: {results['mean_influence_function']:.4f}"
                )
                print(f"Threshold: {correlation_threshold}")
                if results["orthogonality_passes"]:
                    print("✅ ORTHOGONALITY TEST PASSED")
                else:
                    print("❌ ORTHOGONALITY TEST FAILED")
            else:
                print("⚠️  ORTHOGONALITY TEST NOT AVAILABLE")

        return results


class DMLBenchmark:
    """Performance benchmarking framework for Double Machine Learning estimators.

    This class implements performance benchmarking as specified in the DoubleML
    requirements, including runtime scaling, memory usage, and parallelization gains.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the DML benchmark framework.

        Args:
            verbose: Whether to print benchmarking progress
        """
        self.verbose = verbose

    def benchmark_runtime(
        self,
        estimator: BaseEstimator,
        sample_sizes: Optional[list[int]] = None,
        n_features: int = 5,
        true_ate: float = 2.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Benchmark runtime scaling with sample size.

        Args:
            estimator: DML estimator to benchmark
            sample_sizes: List of sample sizes to test
            n_features: Number of covariate features
            true_ate: True average treatment effect
            seed: Random seed

        Returns:
            Dictionary with runtime benchmarking results
        """
        if sample_sizes is None:
            sample_sizes = [1000, 5000, 10000]

        if self.verbose:
            print(f"Benchmarking runtime across sample sizes: {sample_sizes}")

        runtime_results: dict[int, dict[str, Any]] = {}

        for n_samples in sample_sizes:
            if self.verbose:
                print(f"  Testing sample size: {n_samples}")

            # Generate data
            X, D, Y, _ = generate_synthetic_dml_data(
                n=n_samples,
                n_features=n_features,
                true_ate=true_ate,
                seed=seed,
            )

            # Convert to data objects
            treatment_data = TreatmentData(
                values=pd.Series(D),
                name="treatment",
                treatment_type="binary",
            )
            outcome_data = OutcomeData(
                values=pd.Series(Y),
                name="outcome",
                outcome_type="continuous",
            )
            covariate_data = CovariateData(
                values=pd.DataFrame(
                    X, columns=[f"X{i + 1}" for i in range(n_features)]
                ),
                names=[f"X{i + 1}" for i in range(n_features)],
            )

            # Benchmark fitting
            start_time = time.time()
            try:
                estimator.fit(treatment_data, outcome_data, covariate_data)
                fit_time = time.time() - start_time

                # Benchmark estimation
                start_time = time.time()
                _ = (
                    estimator.estimate_ate()
                )  # We only care about timing, not the result
                estimate_time = time.time() - start_time

                total_time = fit_time + estimate_time

                runtime_results[n_samples] = {
                    "fit_time": fit_time,
                    "estimate_time": estimate_time,
                    "total_time": total_time,
                    "success": True,
                }

                if self.verbose:
                    print(
                        f"    Total time: {total_time:.2f}s (fit: {fit_time:.2f}s, estimate: {estimate_time:.2f}s)"
                    )

            except Exception as e:
                runtime_results[n_samples] = {
                    "error": str(e),
                    "success": False,
                    "fit_time": 0.0,
                    "estimate_time": 0.0,
                    "total_time": 0.0,
                }
                if self.verbose:
                    print(f"    Failed: {e}")

        # Check if 1000 sample benchmark meets target (<10s)
        target_runtime = 10.0  # seconds
        meets_target = False
        if 1000 in runtime_results and runtime_results[1000]["success"]:
            runtime_1k = runtime_results[1000]["total_time"]
            meets_target = runtime_1k < target_runtime

        results = {
            "sample_sizes": sample_sizes,
            "runtime_results": runtime_results,
            "target_runtime_1k": target_runtime,
            "meets_target": meets_target,
        }

        if self.verbose:
            print("\n=== Runtime Benchmark Results ===")
            for n in sample_sizes:
                result = results["runtime_results"][n]  # type: ignore[index]
                if result["success"]:
                    total_time = result["total_time"]
                    print(f"Sample size {n}: {total_time:.2f}s")
            if meets_target:
                print("✅ RUNTIME TARGET MET")
            else:
                print("❌ RUNTIME TARGET NOT MET")

        return results

    def benchmark_memory(
        self,
        estimator: BaseEstimator,
        sample_size: int = 10000,
        n_features: int = 5,
        true_ate: float = 2.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Benchmark memory usage.

        Note: This is a basic implementation. For production use, consider
        integrating with memory_profiler or psutil for more accurate monitoring.

        Args:
            estimator: DML estimator to benchmark
            sample_size: Sample size for memory test
            n_features: Number of covariate features
            true_ate: True average treatment effect
            seed: Random seed

        Returns:
            Dictionary with memory benchmarking results
        """
        if self.verbose:
            print(f"Benchmarking memory usage with sample size: {sample_size}")

        try:
            import os

            import psutil  # type: ignore[import-untyped]

            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Generate data
            X, D, Y, _ = generate_synthetic_dml_data(
                n=sample_size,
                n_features=n_features,
                true_ate=true_ate,
                seed=seed,
            )

            data_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Convert to data objects and fit
            treatment_data = TreatmentData(
                values=pd.Series(D),
                name="treatment",
                treatment_type="binary",
            )
            outcome_data = OutcomeData(
                values=pd.Series(Y),
                name="outcome",
                outcome_type="continuous",
            )
            covariate_data = CovariateData(
                values=pd.DataFrame(
                    X, columns=[f"X{i + 1}" for i in range(n_features)]
                ),
                names=[f"X{i + 1}" for i in range(n_features)],
            )

            # Fit estimator
            estimator.fit(treatment_data, outcome_data, covariate_data)
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Calculate memory usage
            data_memory_used = data_memory - initial_memory
            model_memory_used = peak_memory - data_memory
            total_memory_used = peak_memory - initial_memory

            # Check if under 1GB target
            memory_target_mb = 1024  # 1GB in MB
            meets_target = total_memory_used < memory_target_mb

            results = {
                "initial_memory_mb": initial_memory,
                "data_memory_mb": data_memory_used,
                "model_memory_mb": model_memory_used,
                "total_memory_mb": total_memory_used,
                "peak_memory_mb": peak_memory,
                "memory_target_mb": memory_target_mb,
                "meets_target": meets_target,
                "success": True,
            }

        except ImportError:
            results = {
                "error": "psutil not available for memory monitoring",
                "success": False,
            }
        except Exception as e:
            results = {
                "error": str(e),
                "success": False,
            }

        if self.verbose:
            print("\n=== Memory Benchmark Results ===")
            if results["success"]:
                print(f"Total memory used: {results['total_memory_mb']:.1f} MB")
                print(f"Data memory: {results['data_memory_mb']:.1f} MB")
                print(f"Model memory: {results['model_memory_mb']:.1f} MB")
                print(f"Target: < {results['memory_target_mb']} MB")
                if results["meets_target"]:
                    print("✅ MEMORY TARGET MET")
                else:
                    print("❌ MEMORY TARGET NOT MET")
            else:
                print(f"❌ MEMORY BENCHMARK FAILED: {results['error']}")

        return results
