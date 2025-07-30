"""Memory-efficient operations for large dataset causal inference.

This module provides optimized operations for handling large marketing datasets
(1M+ observations) with efficient memory usage and computational patterns.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse


def optimize_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize pandas DataFrame memory usage by converting to appropriate dtypes.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with optimized dtypes
    """
    df_optimized = df.copy()

    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype

        if col_type != "object":
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)

            elif str(col_type)[:5] == "float":
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df_optimized[col] = df_optimized[col].astype(np.float32)
        else:
            # Convert object columns to category if beneficial
            num_unique_values = len(df_optimized[col].unique())
            num_total_values = len(df_optimized[col])
            if num_unique_values / num_total_values < 0.5:
                df_optimized[col] = df_optimized[col].astype("category")

    return df_optimized


def chunked_operation(
    data: pd.DataFrame | NDArray[Any],
    operation: Callable[[pd.DataFrame | NDArray[Any]], Any],
    chunk_size: int = 10000,
    axis: int = 0,
    combine_fn: Callable[[list[Any]], Any] | None = None,
) -> Any:
    """Apply operation to data in chunks to manage memory usage.

    Performance Characteristics:
        - Memory: O(chunk_size) instead of O(n)
        - Time: O(n/chunk_size) passes over data
        - Space-Time Tradeoff: Smaller chunks = lower memory, more passes

    Args:
        data: Input data (DataFrame or array)
        operation: Function to apply to each chunk
        chunk_size: Size of each chunk
        axis: Axis along which to chunk (0 for rows, 1 for columns)
        combine_fn: Function to combine results from chunks

    Returns:
        Combined result from all chunks
    """
    if isinstance(data, pd.DataFrame):
        n_rows, n_cols = data.shape
        n_chunks = n_rows if axis == 0 else n_cols
    else:
        n_chunks = data.shape[axis]

    results = []

    for start_idx in range(0, n_chunks, chunk_size):
        end_idx = min(start_idx + chunk_size, n_chunks)

        chunk: pd.DataFrame | NDArray[Any]
        if isinstance(data, pd.DataFrame):
            if axis == 0:
                chunk = data.iloc[start_idx:end_idx]
            else:
                chunk = data.iloc[:, start_idx:end_idx]
        else:
            if axis == 0:
                chunk = data[start_idx:end_idx]
            else:
                chunk = data[:, start_idx:end_idx]

        chunk_result = operation(chunk)
        results.append(chunk_result)

        # Force garbage collection to free memory
        del chunk
        gc.collect()

    if combine_fn is not None:
        return combine_fn(results)
    else:
        return results


def efficient_bootstrap_indices(
    n_samples: int,
    n_bootstrap: int,
    stratify_by: NDArray[Any] | None = None,
    chunk_size: int = 1000,
    random_state: int | None = None,
) -> Iterator[NDArray[np.int64]]:
    """Generate bootstrap indices efficiently for large datasets.

    Performance Characteristics:
        - Memory: O(chunk_size * n_samples) instead of O(n_bootstrap * n_samples)
        - Time: O(n_bootstrap/chunk_size) iterations
        - Memory Savings: ~chunk_size/n_bootstrap of peak memory usage

    Args:
        n_samples: Number of samples in original dataset
        n_bootstrap: Number of bootstrap samples to generate
        stratify_by: Array to stratify bootstrap samples by
        chunk_size: Number of bootstrap samples to generate at once
        random_state: Random seed

    Yields:
        Bootstrap indices arrays
    """
    if random_state is not None:
        np.random.seed(random_state)

    if stratify_by is not None:
        # Create stratified bootstrap indices
        unique_strata = np.unique(stratify_by)
        strata_indices = {}
        strata_counts = {}

        for stratum in unique_strata:
            stratum_mask = stratify_by == stratum
            strata_indices[stratum] = np.where(stratum_mask)[0]
            strata_counts[stratum] = np.sum(stratum_mask)

    for batch_start in range(0, n_bootstrap, chunk_size):
        batch_end = min(batch_start + chunk_size, n_bootstrap)
        batch_size = batch_end - batch_start

        batch_indices = []

        for _ in range(batch_size):
            bootstrap_idx: NDArray[np.int64]
            if stratify_by is not None:
                # Generate stratified bootstrap sample
                bootstrap_idx_list = []
                for stratum in unique_strata:
                    stratum_indices = strata_indices[stratum]
                    stratum_count = strata_counts[stratum]

                    # Sample with replacement from this stratum
                    sampled = np.random.choice(
                        stratum_indices, size=stratum_count, replace=True
                    )
                    bootstrap_idx_list.extend(sampled)

                # Shuffle the combined indices
                bootstrap_idx_array = np.array(bootstrap_idx_list, dtype=np.int64)
                np.random.shuffle(bootstrap_idx_array)
                bootstrap_idx = bootstrap_idx_array
            else:
                # Simple bootstrap sampling
                bootstrap_idx = np.random.choice(
                    n_samples, size=n_samples, replace=True
                ).astype(np.int64)

            batch_indices.append(bootstrap_idx)

        # Yield as numpy array for memory efficiency
        yield np.array(batch_indices)


def memory_efficient_matmul(
    a: NDArray[Any] | sparse.spmatrix,
    b: NDArray[Any] | sparse.spmatrix,
    chunk_size: int = 10000,
) -> NDArray[Any]:
    """Perform matrix multiplication in chunks for memory efficiency.

    Performance Characteristics:
        - Memory: O(chunk_size * k + k * n) instead of O(m * k + k * n + m * n)
        - Time: O(m * k * n / chunk_size) operations per chunk
        - Best for: Large matrices where m >> chunk_size

    Args:
        a: Left matrix (m x k)
        b: Right matrix (k x n)
        chunk_size: Size of chunks for computation

    Returns:
        Result of a @ b (m x n)
    """
    if sparse.issparse(a) or sparse.issparse(b):
        # Use sparse matrix operations
        return (a @ b).toarray() if sparse.issparse(a @ b) else a @ b

    m, k = a.shape
    k2, n = b.shape

    if k != k2:
        raise ValueError(f"Incompatible matrix dimensions: {a.shape} @ {b.shape}")

    # Chunk along the first dimension of a
    result = np.zeros((m, n), dtype=np.result_type(a.dtype, b.dtype))

    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        chunk_a = a[i:end_i]
        result[i:end_i] = chunk_a @ b

        # Clean up
        del chunk_a
        gc.collect()

    return result


def sparse_safe_operation(
    data: NDArray[Any] | sparse.spmatrix,
    operation: str,
    axis: int | None = None,
    **kwargs: Any,
) -> NDArray[Any] | float:
    """Perform operations safely on potentially sparse data.

    Args:
        data: Input data (dense or sparse)
        operation: Operation name ('mean', 'std', 'sum', 'var')
        axis: Axis along which to perform operation
        **kwargs: Additional arguments for the operation

    Returns:
        Result of the operation
    """
    if sparse.issparse(data):
        if operation == "mean":
            return np.array(data.mean(axis=axis)).flatten()  # type: ignore[union-attr]
        elif operation == "std":
            # For sparse matrices, compute std manually
            mean_val = data.mean(axis=axis)  # type: ignore[union-attr]
            if axis is None:
                variance = data.multiply(data).mean() - mean_val * mean_val  # type: ignore[union-attr]
            else:
                if sparse.issparse(mean_val):
                    variance = data.multiply(data).mean(axis=axis) - mean_val.multiply(
                        mean_val
                    )  # type: ignore[union-attr]
                else:
                    # mean_val is not sparse, use element-wise multiplication
                    variance = data.multiply(data).mean(axis=axis) - mean_val * mean_val  # type: ignore[union-attr]
            return np.sqrt(np.array(variance)).flatten()
        elif operation == "sum":
            return np.array(data.sum(axis=axis)).flatten()  # type: ignore[union-attr]
        elif operation == "var":
            mean_val = data.mean(axis=axis)  # type: ignore[union-attr]
            if axis is None:
                return data.multiply(data).mean() - mean_val * mean_val  # type: ignore[union-attr]
            else:
                if sparse.issparse(mean_val):
                    variance = data.multiply(data).mean(axis=axis) - mean_val.multiply(
                        mean_val
                    )  # type: ignore[union-attr]
                else:
                    # mean_val is not sparse, use element-wise multiplication
                    variance = data.multiply(data).mean(axis=axis) - mean_val * mean_val  # type: ignore[union-attr]
                return np.array(variance).flatten()
        else:
            raise ValueError(f"Unsupported operation for sparse data: {operation}")
    else:
        # Dense data - use standard numpy operations
        if operation == "mean":
            return np.mean(data, axis=axis, **kwargs)
        elif operation == "std":
            return np.std(data, axis=axis, **kwargs)
        elif operation == "sum":
            return np.sum(data, axis=axis, **kwargs)
        elif operation == "var":
            return np.var(data, axis=axis, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")


def estimate_memory_usage(
    n_samples: int,
    n_features: int,
    dtype: type = np.float64,
    overhead_factor: float = 2.0,
) -> tuple[float, str]:
    """Estimate memory usage for dataset operations.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        dtype: Data type
        overhead_factor: Factor to account for overhead

    Returns:
        Tuple of (memory_gb, recommendation)
    """
    # Calculate base memory for the dataset
    if dtype == np.float64:
        bytes_per_element = 8
    elif dtype == np.float32:
        bytes_per_element = 4
    elif dtype == np.int64:
        bytes_per_element = 8
    elif dtype == np.int32:
        bytes_per_element = 4
    else:
        bytes_per_element = 8  # Default assumption

    base_memory_bytes = n_samples * n_features * bytes_per_element
    total_memory_bytes = base_memory_bytes * overhead_factor
    memory_gb = total_memory_bytes / (1024**3)

    if memory_gb < 1:
        recommendation = "Dataset should fit comfortably in memory"
    elif memory_gb < 4:
        recommendation = "Consider using optimized dtypes and chunking for bootstrap"
    elif memory_gb < 8:
        recommendation = "Use chunked operations and monitor memory usage"
    elif memory_gb < 16:
        recommendation = (
            "Strongly recommend chunked processing and streaming operations"
        )
    else:
        recommendation = (
            "Dataset too large - use streaming operations and external storage"
        )

    return memory_gb, recommendation


def create_sparse_features(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    threshold: float = 0.95,
) -> tuple[sparse.spmatrix, list[str]]:
    """Create sparse feature matrix from DataFrame.

    Args:
        df: Input DataFrame
        categorical_columns: Columns to treat as categorical
        threshold: Sparsity threshold for creating sparse matrix

    Returns:
        Tuple of (sparse_matrix, feature_names)
    """
    if categorical_columns is None:
        categorical_columns = []

    # One-hot encode categorical columns
    encoded_dfs = []
    feature_names = []

    for col in df.columns:
        if (
            col in categorical_columns
            or df[col].dtype == "object"
            or df[col].dtype.name == "category"
        ):
            # One-hot encode
            encoded = pd.get_dummies(df[col], prefix=col, sparse=True)
            encoded_dfs.append(encoded)
            feature_names.extend(encoded.columns.tolist())
        else:
            # Keep numeric columns as-is
            encoded_dfs.append(df[[col]])
            feature_names.append(col)

    # Combine all features
    if encoded_dfs:
        combined_df = pd.concat(encoded_dfs, axis=1)
    else:
        combined_df = df

    # Convert to sparse if beneficial
    sparsity = 1 - (
        combined_df.count().sum() / (combined_df.shape[0] * combined_df.shape[1])
    )

    if sparsity > threshold:
        # Convert to sparse matrix
        sparse_matrix = sparse.csr_matrix(combined_df.values)
        return sparse_matrix, feature_names
    else:
        # Return as dense
        return combined_df.values, feature_names


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""

    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_memory = 0
        self.peak_memory = 0

    def __enter__(self) -> MemoryMonitor:
        import os

        import psutil  # type: ignore[import-untyped]

        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = end_memory - self.start_memory

        print(f"Memory usage for {self.operation_name}:")
        print(f"  Start: {self.start_memory:.1f} MB")
        print(f"  End: {end_memory:.1f} MB")
        print(f"  Used: {memory_used:.1f} MB")

        if memory_used > 1000:  # More than 1GB
            warnings.warn(f"High memory usage detected: {memory_used:.1f} MB")


def efficient_cross_validation_indices(
    n_samples: int,
    n_folds: int = 5,
    stratify_by: NDArray[Any] | None = None,
    random_state: int | None = None,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Generate cross-validation indices efficiently for large datasets.

    Args:
        n_samples: Number of samples
        n_folds: Number of CV folds
        stratify_by: Array to stratify folds by
        random_state: Random seed

    Returns:
        List of (train_indices, val_indices) tuples
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)

    if stratify_by is not None:
        # Stratified CV
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        cv_splits = list(skf.split(indices, stratify_by))
    else:
        # Regular CV
        np.random.shuffle(indices)
        fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
        fold_sizes[: n_samples % n_folds] += 1

        cv_splits = []
        start_idx = 0

        for fold_size in fold_sizes:
            val_indices = indices[start_idx : start_idx + fold_size]
            train_indices = np.concatenate(
                [indices[:start_idx], indices[start_idx + fold_size :]]
            )
            cv_splits.append((train_indices, val_indices))
            start_idx += fold_size

    return cv_splits
