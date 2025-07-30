"""Utility modules for causal inference optimization."""

from .benchmarking import (
    BenchmarkResult,
    PerformanceBenchmark,
    ScalabilityTester,
)
from .memory_efficient import (
    MemoryMonitor,
    chunked_operation,
    create_sparse_features,
    efficient_bootstrap_indices,
    efficient_cross_validation_indices,
    estimate_memory_usage,
    memory_efficient_matmul,
    optimize_pandas_dtypes,
    sparse_safe_operation,
)
from .streaming import (
    CSVDataStream,
    DataFrameStream,
    DataStream,
    ExternalSortMerge,
    ParquetDataStream,
    StreamingEstimator,
    create_data_stream,
)

__all__ = [
    # Benchmarking
    "BenchmarkResult",
    "PerformanceBenchmark",
    "ScalabilityTester",
    # Memory efficiency
    "MemoryMonitor",
    "chunked_operation",
    "create_sparse_features",
    "efficient_bootstrap_indices",
    "efficient_cross_validation_indices",
    "estimate_memory_usage",
    "memory_efficient_matmul",
    "optimize_pandas_dtypes",
    "sparse_safe_operation",
    # Streaming
    "CSVDataStream",
    "DataFrameStream",
    "DataStream",
    "ExternalSortMerge",
    "ParquetDataStream",
    "StreamingEstimator",
    "create_data_stream",
]
