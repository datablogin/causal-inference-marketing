"""Streaming and batch processing for large datasets.

This module provides streaming capabilities for datasets that don't fit in memory,
enabling causal inference on very large marketing datasets.
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
except ImportError:
    pq = None

from ..core.base import CovariateData, OutcomeData, TreatmentData


class DataStream(ABC):
    """Abstract base class for data streaming."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """Iterate over batches of (covariates, treatment, outcome)."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Return batch size."""
        pass


class CSVDataStream(DataStream):
    """Stream data from CSV files in batches."""

    def __init__(
        self,
        file_path: str | Path,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str] | None = None,
        batch_size: int = 10000,
        **pd_kwargs: Any,
    ):
        """Initialize CSV data stream.

        Args:
            file_path: Path to CSV file
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariate_cols: List of covariate column names (None for all others)
            batch_size: Size of each batch
            **pd_kwargs: Additional arguments for pd.read_csv
        """
        self.file_path = Path(file_path)
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariate_cols = covariate_cols
        self._batch_size = batch_size
        self.pd_kwargs = pd_kwargs

        # Get total number of rows and validate columns
        self._validate_and_setup()

    def _validate_and_setup(self) -> None:
        """Validate file and setup streaming parameters."""
        # Read first few rows to validate columns
        sample_df = pd.read_csv(self.file_path, nrows=5, **self.pd_kwargs)

        if self.treatment_col not in sample_df.columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' not found")
        if self.outcome_col not in sample_df.columns:
            raise ValueError(f"Outcome column '{self.outcome_col}' not found")

        if self.covariate_cols is None:
            # Use all columns except treatment and outcome
            self.covariate_cols = [
                col
                for col in sample_df.columns
                if col not in {self.treatment_col, self.outcome_col}
            ]
        else:
            # Validate covariate columns exist
            missing_cols = set(self.covariate_cols) - set(sample_df.columns)
            if missing_cols:
                raise ValueError(f"Covariate columns not found: {missing_cols}")

        # Count total rows
        self._n_samples = sum(1 for _ in open(self.file_path)) - 1  # Subtract header

    def __iter__(self) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """Iterate over batches."""
        reader = pd.read_csv(
            self.file_path, chunksize=self._batch_size, **self.pd_kwargs
        )

        for chunk in reader:
            covariates = chunk[self.covariate_cols]
            treatment = chunk[self.treatment_col]
            outcome = chunk[self.outcome_col]

            yield covariates, treatment, outcome

    def __len__(self) -> int:
        """Return total number of samples."""
        return int(self._n_samples)

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self._batch_size


class ParquetDataStream(DataStream):
    """Stream data from Parquet files in batches."""

    def __init__(
        self,
        file_path: str | Path,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str] | None = None,
        batch_size: int = 10000,
    ):
        """Initialize Parquet data stream.

        Args:
            file_path: Path to Parquet file
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariate_cols: List of covariate column names (None for all others)
            batch_size: Size of each batch
        """
        self.file_path = Path(file_path)
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariate_cols = covariate_cols
        self._batch_size = batch_size

        self._validate_and_setup()

    def _validate_and_setup(self) -> None:
        """Validate file and setup streaming parameters."""
        if pq is None:
            raise ImportError("pyarrow is required for parquet streaming")

        # Read parquet metadata
        parquet_file = pq.ParquetFile(self.file_path)
        self._n_samples = parquet_file.metadata.num_rows

        # Get schema information
        schema = parquet_file.schema
        available_columns = [field.name for field in schema]

        if self.treatment_col not in available_columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' not found")
        if self.outcome_col not in available_columns:
            raise ValueError(f"Outcome column '{self.outcome_col}' not found")

        if self.covariate_cols is None:
            self.covariate_cols = [
                col
                for col in available_columns
                if col not in {self.treatment_col, self.outcome_col}
            ]
        else:
            missing_cols = set(self.covariate_cols) - set(available_columns)
            if missing_cols:
                raise ValueError(f"Covariate columns not found: {missing_cols}")

    def __iter__(self) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """Iterate over batches."""
        if pq is None:
            raise ImportError("pyarrow is required for parquet streaming")

        # Read in batches using pyarrow
        parquet_file = pq.ParquetFile(self.file_path)

        for batch in parquet_file.iter_batches(batch_size=self._batch_size):
            df = batch.to_pandas()

            covariates = df[self.covariate_cols]
            treatment = df[self.treatment_col]
            outcome = df[self.outcome_col]

            yield covariates, treatment, outcome

    def __len__(self) -> int:
        """Return total number of samples."""
        return int(self._n_samples)

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self._batch_size


class DataFrameStream(DataStream):
    """Stream batches from an in-memory DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str] | None = None,
        batch_size: int = 10000,
    ):
        """Initialize DataFrame stream.

        Args:
            df: Input DataFrame
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariate_cols: List of covariate column names (None for all others)
            batch_size: Size of each batch
        """
        self.df = df
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self._batch_size = batch_size

        if covariate_cols is None:
            self.covariate_cols = [
                col for col in df.columns if col not in {treatment_col, outcome_col}
            ]
        else:
            self.covariate_cols = covariate_cols

        self._n_samples = len(df)

    def __iter__(self) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """Iterate over batches."""
        n_batches = (self._n_samples + self._batch_size - 1) // self._batch_size

        for i in range(n_batches):
            start_idx = i * self._batch_size
            end_idx = min(start_idx + self._batch_size, self._n_samples)

            batch_df = self.df.iloc[start_idx:end_idx]

            covariates = batch_df[self.covariate_cols]
            treatment = batch_df[self.treatment_col]
            outcome = batch_df[self.outcome_col]

            yield covariates, treatment, outcome

    def __len__(self) -> int:
        """Return total number of samples."""
        return int(self._n_samples)

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self._batch_size


class StreamingEstimator:
    """Base class for estimators that can work with streaming data."""

    def __init__(self, batch_size: int = 10000):
        """Initialize streaming estimator.

        Args:
            batch_size: Default batch size for processing
        """
        self.batch_size = batch_size
        self.is_fitted = False

        # Accumulators for streaming computation
        self._n_samples_seen = 0
        self._sum_weights = 0.0

    def partial_fit(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        sample_weight: NDArray[Any] | None = None,
    ) -> StreamingEstimator:
        """Incrementally fit the estimator with a batch of data.

        Args:
            treatment: Treatment data batch
            outcome: Outcome data batch
            covariates: Covariate data batch
            sample_weight: Sample weights for this batch

        Returns:
            Self for method chaining
        """
        batch_size = len(treatment.values)

        if sample_weight is None:
            sample_weight = np.ones(batch_size)

        self._n_samples_seen += batch_size
        self._sum_weights += float(np.sum(sample_weight))

        # Subclasses should implement actual fitting logic
        self._partial_fit_implementation(treatment, outcome, covariates, sample_weight)

        self.is_fitted = True
        return self

    def _partial_fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        sample_weight: NDArray[Any] | None = None,
    ) -> None:
        """Implement the actual partial fitting logic.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _partial_fit_implementation"
        )

    def fit_stream(self, data_stream: DataStream) -> StreamingEstimator:
        """Fit estimator using a data stream.

        Args:
            data_stream: Stream of data batches

        Returns:
            Self for method chaining
        """
        self._reset_state()

        for covariates_df, treatment_series, outcome_series in data_stream:
            # Convert to our data types
            treatment_data = TreatmentData(values=treatment_series)
            outcome_data = OutcomeData(values=outcome_series)

            if len(covariates_df.columns) > 0:
                covariate_data = CovariateData(
                    values=covariates_df, names=list(covariates_df.columns)
                )
            else:
                covariate_data = None

            self.partial_fit(treatment_data, outcome_data, covariate_data)

        return self

    def _reset_state(self) -> None:
        """Reset internal state for new fitting."""
        self._n_samples_seen = 0
        self._sum_weights = 0.0
        self.is_fitted = False


class ExternalSortMerge:
    """External sort-merge operations for datasets larger than memory."""

    def __init__(self, temp_dir: str | None = None, chunk_size: int = 100000):
        """Initialize external sort-merge.

        Args:
            temp_dir: Directory for temporary files
            chunk_size: Size of chunks to sort in memory
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.chunk_size = chunk_size
        self.temp_files: list[Path] = []

    def external_sort(
        self,
        data_stream: DataStream,
        sort_key: str,
        ascending: bool = True,
    ) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """Sort streaming data using external sort algorithm.

        Args:
            data_stream: Input data stream
            sort_key: Column name to sort by
            ascending: Sort order

        Yields:
            Sorted batches
        """
        # Phase 1: Sort chunks and write to temporary files
        temp_files = []

        for batch_idx, (covariates, treatment, outcome) in enumerate(data_stream):
            # Combine into single DataFrame for sorting
            combined_df = covariates.copy()
            combined_df[treatment.name] = treatment
            combined_df[outcome.name] = outcome

            # Sort this chunk
            if sort_key in combined_df.columns:
                combined_df = combined_df.sort_values(sort_key, ascending=ascending)

            # Write to temporary file
            temp_file = self.temp_dir / f"sorted_chunk_{batch_idx}.parquet"
            combined_df.to_parquet(temp_file)
            temp_files.append(temp_file)

        self.temp_files = temp_files

        # Phase 2: Merge sorted chunks
        yield from self._merge_sorted_files(temp_files, sort_key, ascending)

        # Cleanup
        self._cleanup()

    def _merge_sorted_files(
        self,
        temp_files: list[Path],
        sort_key: str,
        ascending: bool,
    ) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """Merge sorted temporary files."""
        import heapq

        # Open all files and create iterators
        heap: list[tuple[Any, int, Any, Any, Any]] = []

        for file_idx, temp_file in enumerate(temp_files):
            df = pd.read_parquet(temp_file)
            if len(df) > 0:
                iterator = df.iterrows()
                try:
                    row_idx, row = next(iterator)
                    sort_value = row[sort_key] if not ascending else -row[sort_key]
                    heapq.heappush(heap, (sort_value, file_idx, row_idx, row, iterator))
                except StopIteration:
                    continue

        # Collect sorted rows in batches
        batch_rows = []

        while heap:
            sort_value, file_idx, row_idx, row, iterator = heapq.heappop(heap)
            batch_rows.append(row)

            # Try to get next row from same file
            try:
                next_row_idx, next_row = next(iterator)
                next_sort_value = (
                    next_row[sort_key] if not ascending else -next_row[sort_key]
                )
                heapq.heappush(
                    heap, (next_sort_value, file_idx, next_row_idx, next_row, iterator)
                )
            except StopIteration:
                continue

            # Yield batch when full
            if len(batch_rows) >= self.chunk_size:
                yield self._convert_batch_to_output(batch_rows)
                batch_rows = []

        # Yield final batch
        if batch_rows:
            yield self._convert_batch_to_output(batch_rows)

    def _convert_batch_to_output(
        self, batch_rows: list[pd.Series]
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Convert batch of rows back to expected output format."""
        if not batch_rows:
            raise ValueError("Empty batch")

        # Reconstruct DataFrame
        batch_df = pd.DataFrame(batch_rows)

        # Extract treatment and outcome (assuming standard names)
        treatment_col = None
        outcome_col = None

        for col in batch_df.columns:
            if "treatment" in col.lower():
                treatment_col = col
            elif "outcome" in col.lower():
                outcome_col = col

        if treatment_col is None or outcome_col is None:
            # Fallback: assume last two columns are treatment and outcome
            treatment_col = batch_df.columns[-2]
            outcome_col = batch_df.columns[-1]

        covariates = batch_df.drop([treatment_col, outcome_col], axis=1)
        treatment = batch_df[treatment_col]
        outcome = batch_df[outcome_col]

        return covariates, treatment, outcome

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
        self.temp_files = []


def create_data_stream(
    data: str | Path | pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list[str] | None = None,
    batch_size: int = 10000,
    **kwargs: Any,
) -> DataStream:
    """Factory function to create appropriate data stream.

    Args:
        data: Data source (file path or DataFrame)
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        covariate_cols: Covariate column names
        batch_size: Batch size for streaming
        **kwargs: Additional arguments

    Returns:
        Appropriate DataStream instance
    """
    if isinstance(data, pd.DataFrame):
        return DataFrameStream(
            data, treatment_col, outcome_col, covariate_cols, batch_size
        )

    data_path = Path(data)

    if data_path.suffix.lower() == ".csv":
        return CSVDataStream(
            data_path, treatment_col, outcome_col, covariate_cols, batch_size, **kwargs
        )
    elif data_path.suffix.lower() in {".parquet", ".pq"}:
        return ParquetDataStream(
            data_path, treatment_col, outcome_col, covariate_cols, batch_size
        )
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
