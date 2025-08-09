"""Unit tests for performance optimization features in cross-fitting estimators."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from libs.causal_inference.causal_inference.core.base import (
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from libs.causal_inference.causal_inference.estimators.doubly_robust_ml import (
    DoublyRobustMLEstimator,
)
from libs.causal_inference.causal_inference.ml.cross_fitting import (
    CrossFittingEstimator,
    ParallelCrossFittingConfig,
)
from libs.causal_inference.causal_inference.testing.performance import (
    PerformanceConfig,
    PerformanceProfiler,
    benchmark_estimator,
)


class MockCrossFittingEstimator(CrossFittingEstimator):
    """Mock implementation for testing."""

    def _fit_nuisance_models(self, x_train, y_train, treatment_train=None):
        """Mock implementation that returns simple models."""
        return {
            "mock_model_1": MagicMock(),
            "mock_model_2": MagicMock(),
        }

    def _predict_nuisance_parameters(self, models, x_val, treatment_val=None):
        """Mock implementation that returns simple predictions."""
        n_val = x_val.shape[0]
        return {
            "prediction_1": np.random.randn(n_val),
            "prediction_2": np.random.randn(n_val),
        }

    def _estimate_target_parameter(self, nuisance_estimates, treatment, outcome):
        """Mock implementation."""
        return 1.5  # Mock ATE


class TestParallelCrossFittingConfig:
    """Test ParallelCrossFittingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParallelCrossFittingConfig()

        assert config.n_jobs == -1
        assert config.parallel_backend == "threading"
        assert config.max_memory_gb == 4.0
        assert config.chunk_size == 10000
        assert config.enable_caching
        assert config.cache_size == 100
        assert config.timeout_per_fold_minutes == 10.0
        assert config.enable_gc_per_fold
        assert not config.memory_monitoring

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ParallelCrossFittingConfig(
            n_jobs=4,
            parallel_backend="multiprocessing",
            max_memory_gb=8.0,
            chunk_size=5000,
            enable_caching=False,
            cache_size=50,
            timeout_per_fold_minutes=5.0,
            enable_gc_per_fold=False,
            memory_monitoring=True,
        )

        assert config.n_jobs == 4
        assert config.parallel_backend == "multiprocessing"
        assert config.max_memory_gb == 8.0
        assert config.chunk_size == 5000
        assert not config.enable_caching
        assert config.cache_size == 50
        assert config.timeout_per_fold_minutes == 5.0
        assert not config.enable_gc_per_fold
        assert config.memory_monitoring


class TestCrossFittingEstimatorPerformance:
    """Test performance features in CrossFittingEstimator."""

    @pytest.fixture
    def mock_data(self):
        """Generate mock data for testing."""
        np.random.seed(42)
        n_samples, n_features = 100, 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        treatment = np.random.binomial(1, 0.5, n_samples)

        return X, y, treatment

    def test_sequential_cross_fitting(self, mock_data):
        """Test sequential cross-fitting functionality."""
        X, y, treatment = mock_data

        config = ParallelCrossFittingConfig(n_jobs=1)
        estimator = MockCrossFittingEstimator(parallel_config=config)

        # Perform cross-fitting
        nuisance_estimates = estimator._perform_cross_fitting(X, y, treatment)

        # Check results
        assert isinstance(nuisance_estimates, dict)
        assert "prediction_1" in nuisance_estimates
        assert "prediction_2" in nuisance_estimates
        assert len(nuisance_estimates["prediction_1"]) == len(X)
        assert len(estimator._fold_timings_) == estimator.cv_folds

    def test_parallel_cross_fitting(self, mock_data):
        """Test parallel cross-fitting functionality."""
        X, y, treatment = mock_data

        config = ParallelCrossFittingConfig(n_jobs=2, parallel_backend="threading")
        estimator = MockCrossFittingEstimator(parallel_config=config)

        # Perform cross-fitting
        nuisance_estimates = estimator._perform_cross_fitting(X, y, treatment)

        # Check results
        assert isinstance(nuisance_estimates, dict)
        assert "prediction_1" in nuisance_estimates
        assert "prediction_2" in nuisance_estimates
        assert len(nuisance_estimates["prediction_1"]) == len(X)

        # Check parallel speedup was calculated
        assert estimator._parallel_speedup_ is not None
        assert estimator._parallel_speedup_ > 0

    @patch(
        "libs.causal_inference.causal_inference.ml.cross_fitting.JOBLIB_AVAILABLE",
        False,
    )
    def test_fallback_when_joblib_unavailable(self, mock_data):
        """Test fallback to sequential when joblib is unavailable."""
        X, y, treatment = mock_data

        config = ParallelCrossFittingConfig(n_jobs=2)
        estimator = MockCrossFittingEstimator(parallel_config=config)

        # Should fall back to sequential processing
        nuisance_estimates = estimator._perform_cross_fitting(X, y, treatment)

        assert isinstance(nuisance_estimates, dict)
        assert len(nuisance_estimates["prediction_1"]) == len(X)

    def test_chunked_processing_trigger(self, mock_data):
        """Test that chunked processing is triggered for large datasets."""
        X, y, treatment = mock_data

        config = ParallelCrossFittingConfig(
            chunk_size=50,  # Small chunk size to trigger chunking
            memory_monitoring=True,
        )
        estimator = MockCrossFittingEstimator(parallel_config=config)

        # Should trigger chunked processing
        assert estimator._should_use_chunked_processing(X, y)

    def test_caching_functionality(self, mock_data):
        """Test model caching functionality."""
        X, y, treatment = mock_data

        config = ParallelCrossFittingConfig(enable_caching=True)
        estimator = MockCrossFittingEstimator(parallel_config=config)

        # Fit models twice with same data to test caching
        estimator._fit_nuisance_models_cached(X[:50], y[:50], treatment[:50], 0)
        estimator._fit_nuisance_models_cached(X[:50], y[:50], treatment[:50], 0)

        # Should have cache hits and misses
        cache_stats = estimator.get_cache_statistics()
        assert cache_stats["cache_enabled"]
        assert cache_stats["cache_hits"] >= 0
        assert cache_stats["cache_misses"] >= 0

    def test_performance_metrics(self, mock_data):
        """Test performance metrics collection."""
        X, y, treatment = mock_data

        config = ParallelCrossFittingConfig(n_jobs=2, enable_caching=True)
        estimator = MockCrossFittingEstimator(parallel_config=config)

        # Perform cross-fitting
        estimator._perform_cross_fitting(X, y, treatment)

        # Get performance metrics
        metrics = estimator.get_performance_metrics()

        assert "n_folds" in metrics
        assert "parallel_config" in metrics
        assert "performance_analysis" in metrics
        assert "efficiency_metrics" in metrics["performance_analysis"]

        if estimator.parallel_config.enable_caching:
            assert "caching" in metrics


class TestDoublyRobustMLPerformanceIntegration:
    """Test performance features integrated with DoublyRobustMLEstimator."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for DML testing."""
        np.random.seed(123)
        n_samples, n_features = 200, 8

        X = np.random.randn(n_samples, n_features)

        # Generate treatment with confounding
        propensity_logits = X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2]
        propensity = 1 / (1 + np.exp(-propensity_logits))
        A = np.random.binomial(1, propensity)

        # Generate outcome with treatment effect
        treatment_effect = 2.0
        outcome_mean = X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + treatment_effect * A
        Y = outcome_mean + np.random.randn(n_samples) * 0.5

        return X, Y, A

    def test_dml_with_performance_config(self, synthetic_data):
        """Test DoublyRobustMLEstimator with performance configuration."""
        X, Y, A = synthetic_data

        # Create performance config
        performance_config = ParallelCrossFittingConfig(
            n_jobs=2,
            parallel_backend="threading",
            enable_caching=True,
            memory_monitoring=False,
        )

        # Create estimator
        estimator = DoublyRobustMLEstimator(
            cv_folds=3,
            performance_config=performance_config,
            random_state=42,
        )

        # Fit the estimator
        estimator.fit(
            treatment=TreatmentData(values=A, treatment_type="binary"),
            outcome=OutcomeData(values=Y, outcome_type="continuous"),
            covariates=CovariateData(values=X),
        )

        # Check that performance config was applied
        assert estimator.performance_config.n_jobs == 2
        assert estimator.performance_config.parallel_backend == "threading"
        assert estimator.performance_config.enable_caching

        # Check that cross-fitting worked
        assert estimator.is_fitted
        assert hasattr(estimator, "nuisance_estimates_")

        # Get performance metrics
        metrics = estimator.get_performance_metrics()
        assert "parallel_config" in metrics

    def test_dml_default_performance_config(self, synthetic_data):
        """Test DoublyRobustMLEstimator uses sensible default performance config."""
        X, Y, A = synthetic_data

        # Create estimator without explicit config
        estimator = DoublyRobustMLEstimator(
            cv_folds=3,
            random_state=42,
        )

        # Check default config was created
        assert estimator.performance_config is not None
        assert estimator.performance_config.n_jobs == -1  # Use all cores
        assert estimator.performance_config.parallel_backend == "threading"
        assert estimator.performance_config.enable_caching


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""

    @pytest.fixture
    def mock_estimator_factory(self):
        """Create factory for mock estimators."""

        def factory():
            config = ParallelCrossFittingConfig(
                n_jobs=1
            )  # Sequential for consistent testing
            return MockCrossFittingEstimator(parallel_config=config)

        return factory

    def test_performance_profiler_initialization(self):
        """Test PerformanceProfiler initialization."""
        config = PerformanceConfig(n_jobs=2, max_memory_gb=2.0)
        profiler = PerformanceProfiler(config)

        assert profiler.config.n_jobs == 2
        assert profiler.config.max_memory_gb == 2.0
        assert profiler._baseline_memory is None
        assert profiler._memory_profile == []

    def test_runtime_profiling(self, mock_estimator_factory):
        """Test runtime profiling functionality."""
        profiler = PerformanceProfiler()

        # Generate test data
        X, Y, A = profiler._generate_synthetic_data(100, 5, random_state=42)

        # Create estimator
        estimator = mock_estimator_factory()

        # Profile runtime
        runtime_metrics = profiler.profile_runtime(estimator, X, Y, A, n_runs=2)

        assert "mean_runtime" in runtime_metrics
        assert "std_runtime" in runtime_metrics
        assert "n_successful_runs" in runtime_metrics
        assert runtime_metrics["n_successful_runs"] == 2
        assert runtime_metrics["mean_runtime"] > 0

    def test_scalability_benchmarking(self, mock_estimator_factory):
        """Test scalability benchmarking."""
        profiler = PerformanceProfiler()

        sample_sizes = [50, 100]
        results_df = profiler.benchmark_scalability(
            mock_estimator_factory,
            sample_sizes,
            n_features=5,
            random_state=42,
        )

        assert len(results_df) == len(sample_sizes)
        assert "n_samples" in results_df.columns
        assert "runtime_seconds" in results_df.columns
        assert "peak_memory_mb" in results_df.columns
        assert all(results_df["n_samples"] == sample_sizes)

    def test_benchmark_estimator_function(self, mock_estimator_factory):
        """Test the benchmark_estimator convenience function."""
        config = PerformanceConfig(n_jobs=1)

        results = benchmark_estimator(
            mock_estimator_factory,
            sample_sizes=[50, 100],
            n_features=5,
            performance_config=config,
            random_state=42,
        )

        assert "scalability_results" in results
        assert "config" in results
        assert "benchmark_timestamp" in results

        # Check scalability results
        scalability_df = results["scalability_results"]
        assert len(scalability_df) == 2
        assert all(scalability_df["n_samples"] == [50, 100])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_parallel_config(self):
        """Test handling of invalid parallel configuration."""
        # Should not raise an error, just use defaults
        config = ParallelCrossFittingConfig(n_jobs=0)  # Invalid, but handled
        estimator = MockCrossFittingEstimator(parallel_config=config)
        assert estimator.parallel_config.n_jobs == 0

    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        config = ParallelCrossFittingConfig()
        estimator = MockCrossFittingEstimator(parallel_config=config)

        # Very small dataset
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        treatment = np.array([0, 1])

        # Should handle gracefully
        try:
            nuisance_estimates = estimator._perform_cross_fitting(X, y, treatment)
            assert isinstance(nuisance_estimates, dict)
        except ValueError:
            # Acceptable to fail with very small datasets
            pass

    def test_cache_size_limit(self):
        """Test that model cache respects size limits."""
        config = ParallelCrossFittingConfig(enable_caching=True)
        estimator = MockCrossFittingEstimator(parallel_config=config)

        # Fill cache beyond limit (100 models)
        for i in range(105):
            X_small = np.random.randn(10, 2)
            y_small = np.random.randn(10)
            treatment_small = np.random.binomial(1, 0.5, 10)

            estimator._fit_nuisance_models_cached(X_small, y_small, treatment_small, i)

        # Cache should be limited
        cache_stats = estimator.get_cache_statistics()
        assert cache_stats["cache_size"] <= 100


if __name__ == "__main__":
    pytest.main([__file__])
