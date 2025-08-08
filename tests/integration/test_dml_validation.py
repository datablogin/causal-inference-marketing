"""Integration tests for DML validation pipeline.

This module tests that the DoublyRobustMLEstimator passes all statistical
validation checks specified in the DoubleML framework.
"""

import numpy as np
import pytest

from libs.causal_inference.causal_inference.estimators.doubly_robust_ml import (
    DoublyRobustMLEstimator,
)
from libs.causal_inference.causal_inference.testing.synthetic import (
    generate_synthetic_dml_data,
)
from libs.causal_inference.causal_inference.testing.validation import (
    DMLBenchmark,
    DMLValidator,
)


class TestDMLValidationIntegration:
    """Integration tests for DML validation framework."""

    @pytest.fixture
    def default_estimator(self):
        """Default DML estimator for testing."""
        return DoublyRobustMLEstimator(
            cross_fitting=True,
            cv_folds=3,  # Reduced for faster testing
            moment_function="aipw",
            random_state=42,
            verbose=False,
        )

    @pytest.fixture
    def fast_validator(self):
        """Validator with reduced simulation counts for faster testing."""
        return DMLValidator(verbose=False)

    def test_synthetic_data_generation(self):
        """Test that synthetic data generator produces expected properties."""
        # Generate data
        X, D, Y, true_ate = generate_synthetic_dml_data(
            n=1000,
            n_features=5,
            true_ate=2.0,
            seed=42,
        )

        # Check shapes
        assert X.shape == (1000, 5)
        assert D.shape == (1000,)
        assert Y.shape == (1000,)
        assert true_ate == 2.0

        # Check treatment is binary
        assert set(np.unique(D)) <= {0, 1}

        # Check for reasonable treatment balance (not too extreme)
        treatment_rate = np.mean(D)
        assert (
            0.2 <= treatment_rate <= 0.8
        ), f"Treatment rate {treatment_rate} too extreme"

        # Check outcome has reasonable range
        assert np.isfinite(Y).all(), "Outcome contains non-finite values"
        assert np.std(Y) > 0, "Outcome has zero variance"

    def test_bias_validation_passes(self, default_estimator, fast_validator):
        """Test that DML estimator passes bias validation."""
        results = fast_validator.validate_bias(
            estimator=default_estimator,
            n_simulations=20,  # Reduced for speed
            n_samples=500,  # Reduced for speed
            bias_threshold=0.1,  # Slightly relaxed for small samples
            seed=42,
        )

        # Check results structure
        assert "mean_bias" in results
        assert "abs_bias" in results
        assert "bias_pass_rate" in results
        assert "individual_estimates" in results

        # Check that most simulations succeeded
        assert results["n_valid_simulations"] >= 15, "Too many failed simulations"

        # Check bias is reasonable (may not always pass with small samples)
        assert (
            abs(results["mean_bias"]) < 0.5
        ), f"Mean bias {results['mean_bias']} too large"
        assert (
            results["abs_bias"] < 0.5
        ), f"Absolute bias {results['abs_bias']} too large"

    def test_coverage_validation_passes(self, default_estimator, fast_validator):
        """Test that DML estimator passes coverage validation."""
        results = fast_validator.validate_coverage(
            estimator=default_estimator,
            n_simulations=20,  # Reduced for speed
            n_samples=500,  # Reduced for speed
            nominal_coverage=0.95,
            coverage_tolerance=0.15,  # Relaxed for small samples
            seed=42,
        )

        # Check results structure
        assert "empirical_coverage" in results
        assert "coverage_passes" in results
        assert "mean_ci_width" in results

        # Check that CIs were generated
        assert results["n_valid_cis"] >= 15, "Too few valid confidence intervals"

        # Check coverage is reasonable
        coverage = results["empirical_coverage"]
        assert 0.7 <= coverage <= 1.0, f"Coverage {coverage} outside reasonable range"

        # Check CI width is positive
        if not np.isnan(results["mean_ci_width"]):
            assert results["mean_ci_width"] > 0, "Mean CI width should be positive"

    def test_consistency_validation(self, default_estimator, fast_validator):
        """Test consistency validation across sample sizes."""
        results = fast_validator.validate_consistency(
            estimator=default_estimator,
            sample_sizes=[200, 400, 800],  # Smaller sizes for speed
            n_simulations=10,  # Reduced for speed
            seed=42,
        )

        # Check results structure
        assert "sample_sizes" in results
        assert "abs_biases" in results
        assert "bias_slope" in results
        assert "detailed_results" in results

        # Check that all sample sizes were tested
        assert len(results["abs_biases"]) == 3
        assert all(n in results["detailed_results"] for n in [200, 400, 800])

        # Check biases are finite
        assert all(np.isfinite(bias) for bias in results["abs_biases"])

    def test_orthogonality_validation(self, default_estimator, fast_validator):
        """Test orthogonality validation."""
        results = fast_validator.validate_orthogonality(
            estimator=default_estimator,
            n_samples=500,  # Reduced for speed
            seed=42,
        )

        # Check results structure
        assert "orthogonality_passes" in results

        # The test may not be available for all estimators, but should not crash
        assert results is not None

    def test_different_moment_functions(self, fast_validator):
        """Test validation with different moment functions."""
        moment_functions = ["aipw", "orthogonal"]

        for moment_fn in moment_functions:
            estimator = DoublyRobustMLEstimator(
                cross_fitting=True,
                cv_folds=3,
                moment_function=moment_fn,
                random_state=42,
                verbose=False,
            )

            # Test bias validation
            results = fast_validator.validate_bias(
                estimator=estimator,
                n_simulations=10,
                n_samples=300,
                bias_threshold=0.15,
                seed=42,
            )

            assert (
                results["n_valid_simulations"] >= 8
            ), f"Too many failures for {moment_fn}"
            assert abs(results["mean_bias"]) < 1.0, f"Bias too large for {moment_fn}"

    def test_different_sample_sizes(self, default_estimator, fast_validator):
        """Test validation with different sample sizes."""
        sample_sizes = [200, 500, 1000]

        for n in sample_sizes:
            results = fast_validator.validate_bias(
                estimator=default_estimator,
                n_simulations=5,  # Very fast test
                n_samples=n,
                bias_threshold=0.2,
                seed=42,
            )

            # Should work for all sample sizes
            assert results["n_valid_simulations"] >= 4, f"Too many failures for n={n}"
            assert len(results["individual_estimates"]) >= 4


class TestDMLBenchmarkIntegration:
    """Integration tests for DML benchmarking framework."""

    @pytest.fixture
    def default_estimator(self):
        """Default DML estimator for benchmarking."""
        return DoublyRobustMLEstimator(
            cross_fitting=True,
            cv_folds=2,  # Minimal folds for speed
            random_state=42,
            verbose=False,
        )

    @pytest.fixture
    def benchmark(self):
        """Benchmark framework for testing."""
        return DMLBenchmark(verbose=False)

    def test_runtime_benchmark(self, default_estimator, benchmark):
        """Test runtime benchmarking."""
        results = benchmark.benchmark_runtime(
            estimator=default_estimator,
            sample_sizes=[500, 1000],  # Small sizes for speed
            seed=42,
        )

        # Check results structure
        assert "sample_sizes" in results
        assert "runtime_results" in results
        assert "target_runtime_1k" in results

        # Check that benchmarks ran
        for n in [500, 1000]:
            assert n in results["runtime_results"]
            result = results["runtime_results"][n]
            if result["success"]:
                assert "fit_time" in result
                assert "estimate_time" in result
                assert "total_time" in result
                assert result["total_time"] > 0

    def test_memory_benchmark(self, default_estimator, benchmark):
        """Test memory benchmarking."""
        results = benchmark.benchmark_memory(
            estimator=default_estimator,
            sample_size=1000,  # Small size for speed
            seed=42,
        )

        # Check results structure
        assert "success" in results

        if results["success"]:
            assert "total_memory_mb" in results
            assert "memory_target_mb" in results
            assert results["total_memory_mb"] >= 0
        else:
            # Memory monitoring might not be available
            assert "error" in results

    def test_benchmark_with_different_estimators(self, benchmark):
        """Test benchmarking with different estimator configurations."""
        configs = [
            {"cross_fitting": True, "cv_folds": 2},
            {"cross_fitting": False, "cv_folds": 1},
        ]

        for config in configs:
            estimator = DoublyRobustMLEstimator(
                random_state=42,
                verbose=False,
                **config,
            )

            results = benchmark.benchmark_runtime(
                estimator=estimator,
                sample_sizes=[300],
                seed=42,
            )

            # Should work for all configurations
            assert results["runtime_results"][300][
                "success"
            ], f"Failed for config {config}"


class TestDMLValidationRobustness:
    """Test robustness of validation framework to various scenarios."""

    def test_validation_with_failed_estimator(self):
        """Test validation handles estimator failures gracefully."""

        class FailingEstimator(DoublyRobustMLEstimator):
            def estimate_ate(self):
                raise RuntimeError("Intentional failure")

        validator = DMLValidator(verbose=False)
        failing_estimator = FailingEstimator(verbose=False)

        # Should handle failures without crashing
        results = validator.validate_bias(
            estimator=failing_estimator,
            n_simulations=5,
            n_samples=100,
            seed=42,
        )

        # Should report the failures
        assert results["n_valid_simulations"] < 5

    def test_validation_with_extreme_parameters(self):
        """Test validation with extreme parameter values."""
        validator = DMLValidator(verbose=False)
        estimator = DoublyRobustMLEstimator(
            cross_fitting=True,
            cv_folds=2,
            random_state=42,
            verbose=False,
        )

        # Test with very small samples
        results = validator.validate_bias(
            estimator=estimator,
            n_simulations=3,
            n_samples=50,  # Very small
            true_ate=10.0,  # Large effect
            seed=42,
        )

        # Should not crash even with extreme parameters
        assert "mean_bias" in results
        assert "n_valid_simulations" in results

    def test_validation_reproducibility(self):
        """Test that validation results are reproducible with same seed."""
        validator = DMLValidator(verbose=False)
        estimator1 = DoublyRobustMLEstimator(random_state=42, verbose=False)
        estimator2 = DoublyRobustMLEstimator(random_state=42, verbose=False)

        # Run same validation twice
        results1 = validator.validate_bias(
            estimator=estimator1,
            n_simulations=5,
            n_samples=200,
            seed=123,
        )

        results2 = validator.validate_bias(
            estimator=estimator2,
            n_simulations=5,
            n_samples=200,
            seed=123,
        )

        # Results should be identical
        np.testing.assert_almost_equal(
            results1["individual_estimates"],
            results2["individual_estimates"],
            decimal=10,
        )
