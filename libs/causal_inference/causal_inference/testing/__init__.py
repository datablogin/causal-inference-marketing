"""Testing utilities for causal inference methods.

This module provides tools for testing and validating causal inference estimators:
- Synthetic data generation for DML testing
- Statistical validation frameworks
- Performance benchmarking utilities
"""

from .performance import (
    PerformanceConfig,
    PerformanceMetrics,
    PerformanceProfiler,
    benchmark_estimator,
)
from .synthetic import generate_synthetic_dml_data
from .validation import DMLValidator

__all__ = [
    "generate_synthetic_dml_data",
    "DMLValidator",
    "PerformanceConfig",
    "PerformanceProfiler",
    "PerformanceMetrics",
    "benchmark_estimator",
]
