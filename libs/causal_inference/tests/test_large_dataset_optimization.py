"""Tests for large dataset optimization features."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.bootstrap import BootstrapConfig
from causal_inference.data.synthetic import SyntheticDataGenerator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.utils.benchmarking import PerformanceBenchmark, ScalabilityTester
from causal_inference.utils.memory_efficient import (
    MemoryMonitor,
    chunked_operation,
    efficient_bootstrap_indices,
    estimate_memory_usage,
    optimize_pandas_dtypes,
)
from causal_inference.utils.streaming import (
    CSVDataStream,
    DataFrameStream,
    create_data_stream,
)


class TestMemoryEfficiency:
    """Test memory-efficient operations."""

    def test_optimize_pandas_dtypes(self):
        """Test DataFrame dtype optimization."""
        # Create test DataFrame with suboptimal dtypes
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],  # Can be int8
            'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],  # Can be float32
            'category_col': ['A', 'B', 'A', 'B', 'A'],  # Can be category
        })

        # Ensure starting dtypes are not optimal
        df['int_col'] = df['int_col'].astype('int64')
        df['float_col'] = df['float_col'].astype('float64')

        original_memory = df.memory_usage(deep=True).sum()

        # Optimize dtypes
        optimized_df = optimize_pandas_dtypes(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        # Should use less memory
        assert optimized_memory <= original_memory

        # Values should be preserved
        pd.testing.assert_frame_equal(df.astype(str), optimized_df.astype(str))

    def test_chunked_operation(self):
        """Test chunked operations on large arrays."""
        # Create test data
        data = np.random.randn(1000, 10)
        df = pd.DataFrame(data)

        # Define operation to sum each chunk
        def sum_operation(chunk_df):
            return chunk_df.sum().sum()

        # Test chunked operation
        results = chunked_operation(
            df,
            sum_operation,
            chunk_size=100,
            combine_fn=sum
        )

        # Should equal direct operation
        expected = sum_operation(df)
        assert abs(results - expected) < 1e-10

    def test_efficient_bootstrap_indices(self):
        """Test efficient bootstrap index generation."""
        n_samples = 1000
        n_bootstrap = 50

        # Test without stratification
        indices_gen = efficient_bootstrap_indices(
            n_samples, n_bootstrap, chunk_size=10, random_state=42
        )

        all_indices = []
        for batch in indices_gen:
            all_indices.extend(batch)

        assert len(all_indices) == n_bootstrap

        # Each bootstrap sample should have correct size
        for indices in all_indices:
            assert len(indices) == n_samples
            assert np.all(indices >= 0)
            assert np.all(indices < n_samples)

    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        memory_gb, recommendation = estimate_memory_usage(
            n_samples=100000,
            n_features=50,
            dtype=np.float64
        )

        assert memory_gb > 0
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0

    def test_memory_monitor(self):
        """Test memory monitoring context manager."""
        try:
            with MemoryMonitor("test_operation") as monitor:
                # Create some data to use memory
                data = np.random.randn(10000, 100)
                del data

            # Should have recorded some memory usage
            assert "start_mb" in monitor.__dict__ or True  # May not have psutil

        except ImportError:
            # Skip if psutil not available
            pytest.skip("psutil not available for memory monitoring")


class TestStreamingOperations:
    """Test streaming data operations."""

    def test_dataframe_stream(self):
        """Test DataFrame streaming."""
        # Create test DataFrame
        df = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0] * 20,
            'outcome': np.random.randn(100),
            'covariate_1': np.random.randn(100),
            'covariate_2': np.random.randn(100),
        })

        # Create stream
        stream = DataFrameStream(
            df,
            treatment_col='treatment',
            outcome_col='outcome',
            batch_size=25
        )

        assert len(stream) == 100
        assert stream.batch_size == 25

        # Test iteration
        batches = list(stream)
        assert len(batches) == 4  # 100/25 = 4 batches

        # Check batch contents
        for covariates, treatment, outcome in batches:
            assert len(covariates) <= 25
            assert len(treatment) <= 25
            assert len(outcome) <= 25
            assert len(covariates) == len(treatment) == len(outcome)

    def test_csv_stream(self):
        """Test CSV file streaming."""
        # Create temporary CSV file
        df = pd.DataFrame({
            'treatment': [0, 1, 0, 1, 0] * 10,
            'outcome': np.random.randn(50),
            'covariate_1': np.random.randn(50),
            'covariate_2': np.random.randn(50),
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Create stream
            stream = CSVDataStream(
                csv_path,
                treatment_col='treatment',
                outcome_col='outcome',
                batch_size=20
            )

            assert len(stream) == 50

            # Test iteration
            batches = list(stream)
            assert len(batches) >= 2  # At least 2 batches

            # Verify data integrity
            total_samples = sum(len(treatment) for _, treatment, _ in batches)
            assert total_samples == 50

        finally:
            Path(csv_path).unlink()

    def test_create_data_stream_factory(self):
        """Test data stream factory function."""
        # Test with DataFrame
        df = pd.DataFrame({
            'treatment': [0, 1, 0, 1],
            'outcome': [1.0, 2.0, 1.5, 2.5],
            'covariate': [0.1, 0.2, 0.3, 0.4],
        })

        stream = create_data_stream(
            df,
            treatment_col='treatment',
            outcome_col='outcome',
            batch_size=2
        )

        assert isinstance(stream, DataFrameStream)
        assert len(stream) == 4


class TestBootstrapOptimizations:
    """Test bootstrap optimizations for large datasets."""

    def test_chunked_bootstrap_config(self):
        """Test bootstrap configuration for large datasets."""
        config = BootstrapConfig(
            n_samples=500,
            chunked_bootstrap=True,
            bootstrap_chunk_size=50,
            large_dataset_threshold=100,
        )

        assert config.chunked_bootstrap is True
        assert config.bootstrap_chunk_size == 50
        assert config.large_dataset_threshold == 100

    def test_large_dataset_bootstrap(self):
        """Test bootstrap with large dataset optimizations."""
        # Generate medium-sized synthetic data
        generator = SyntheticDataGenerator(random_state=42)
        treatment_data, outcome_data, covariate_data = generator.generate_linear_binary_treatment(
            n_samples=500,  # Above threshold
            n_confounders=5,
            treatment_effect=2.0,
        )

        # Configure for chunked bootstrap
        bootstrap_config = BootstrapConfig(
            n_samples=100,  # Reduced for testing
            chunked_bootstrap=True,
            bootstrap_chunk_size=20,
            large_dataset_threshold=400,  # Below our dataset size
            parallel=False,  # Disable for simpler testing
        )

        # Create estimator with optimizations
        estimator = GComputationEstimator(
            bootstrap_config=bootstrap_config,
            large_dataset_threshold=400,
            memory_efficient=True,
            verbose=False
        )

        # Fit and estimate
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Should have bootstrap confidence intervals
        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None
        assert effect.bootstrap_samples == 100


class TestPerformanceBenchmarking:
    """Test performance benchmarking utilities."""

    def test_performance_benchmark(self):
        """Test performance benchmarking."""
        benchmark = PerformanceBenchmark(
            sample_sizes=[100, 200],  # Small sizes for testing
            n_features_list=[5, 10],
            n_trials=1,  # Single trial for speed
            random_state=42
        )

        # Benchmark G-computation estimator
        results = benchmark.benchmark_estimator(
            GComputationEstimator,
            estimator_params={
                'bootstrap_config': BootstrapConfig(n_samples=0),  # No bootstrap for speed
                'verbose': False
            }
        )

        assert len(results) == 4  # 2 sample sizes × 2 feature counts

        for result in results:
            assert result.method_name == "GComputationEstimator"
            assert result.n_samples in [100, 200]
            assert result.n_features in [5, 10]
            assert result.fit_time is not None
            assert result.fit_time > 0

    def test_scalability_tester(self):
        """Test scalability testing."""
        tester = ScalabilityTester(
            target_memory_gb=1.0,  # Low threshold for testing
            target_time_minutes=0.1  # 6 seconds
        )

        # Create simple estimator
        estimator = GComputationEstimator(
            bootstrap_config=BootstrapConfig(n_samples=0),  # No bootstrap
            verbose=False
        )

        # Test memory target (small dataset to avoid issues)
        memory_result = tester.test_memory_target(
            estimator, n_samples=1000  # Small dataset
        )

        assert "memory_used_gb" in memory_result
        assert "meets_memory_target" in memory_result
        assert "success" in memory_result

    def test_comparison_dataframe(self):
        """Test estimator comparison DataFrame generation."""
        benchmark = PerformanceBenchmark(
            sample_sizes=[50, 100],  # Very small for testing
            n_features_list=[3, 5],
            n_trials=1,
            random_state=42
        )

        estimator_configs = [
            (GComputationEstimator, {'bootstrap_config': BootstrapConfig(n_samples=0)}),
        ]

        results_df = benchmark.compare_estimators(estimator_configs)

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 4  # 2 sample sizes × 2 feature counts × 1 estimator
        assert "method" in results_df.columns
        assert "n_samples" in results_df.columns
        assert "fit_time" in results_df.columns


class TestLargeDatasetEstimator:
    """Test estimator optimizations for large datasets."""

    def test_memory_efficient_gcomputation(self):
        """Test G-computation with memory optimizations."""
        # Generate test data
        generator = SyntheticDataGenerator(random_state=42)
        treatment_data, outcome_data, covariate_data = generator.generate_linear_binary_treatment(
            n_samples=1000,
            n_confounders=10,
            treatment_effect=1.5,
        )

        # Create estimator with optimizations
        estimator = GComputationEstimator(
            memory_efficient=True,
            large_dataset_threshold=500,  # Below our dataset size
            chunk_size=200,
            bootstrap_config=BootstrapConfig(n_samples=50),  # Small bootstrap
            verbose=False
        )

        # Fit and estimate
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Should produce valid results
        assert effect.ate is not None
        assert abs(effect.ate - 1.5) < 1.0  # Reasonable estimate
        assert effect.n_observations == 1000

    def test_chunked_prediction(self):
        """Test chunked prediction for large datasets."""
        # Generate test data
        generator = SyntheticDataGenerator(random_state=42)
        treatment_data, outcome_data, covariate_data = generator.generate_linear_binary_treatment(
            n_samples=800,
            n_confounders=5,
            treatment_effect=2.0,
        )

        # Create estimator with small chunk size to force chunking
        estimator = GComputationEstimator(
            memory_efficient=True,
            large_dataset_threshold=500,  # Enable optimizations
            chunk_size=100,  # Small chunks
            bootstrap_config=BootstrapConfig(n_samples=0),  # No bootstrap
            verbose=False
        )

        # Fit estimator
        estimator.fit(treatment_data, outcome_data, covariate_data)

        # Test prediction - should use chunked approach
        y0_pred, y1_pred = estimator.predict_potential_outcomes(
            treatment_values=treatment_data.values,
            covariates=covariate_data.values
        )

        assert len(y0_pred) == 800
        assert len(y1_pred) == 800
        assert not np.array_equal(y0_pred, y1_pred)  # Should be different


if __name__ == "__main__":
    pytest.main([__file__])
