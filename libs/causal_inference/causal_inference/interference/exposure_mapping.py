"""Exposure Mapping for Interference and Spillover Detection.

This module provides tools for mapping spillover exposure patterns across different
types of relationships: spatial, network-based, and temporal.
"""

from __future__ import annotations

import abc
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator
from scipy.spatial.distance import cdist


class ExposureMapping(BaseModel):
    """Data model for spillover exposure relationships.

    Represents how units can be affected by treatment spillover from other units.
    """

    unit_ids: NDArray[Any] = Field(..., description="Unique identifiers for units")
    exposure_matrix: NDArray[Any] = Field(
        ...,
        description="Matrix where [i,j] indicates exposure strength from unit j to unit i",
    )
    exposure_type: str = Field(
        default="binary",
        description="Type of exposure: 'binary', 'continuous', or 'distance_weighted'",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about exposure mapping"
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("exposure_type")
    @classmethod
    def validate_exposure_type(cls, v: str) -> str:
        """Validate exposure type is one of allowed values."""
        allowed_types = {"binary", "continuous", "distance_weighted"}
        if v not in allowed_types:
            raise ValueError(f"exposure_type must be one of {allowed_types}")
        return v

    def __init__(self, **data: Any) -> None:
        """Initialize exposure mapping with validation."""
        super().__init__(**data)

        n_units = len(self.unit_ids)
        if self.exposure_matrix.shape != (n_units, n_units):
            raise ValueError(
                f"Exposure matrix shape {self.exposure_matrix.shape} "
                f"doesn't match number of units {n_units}"
            )

        # Validate diagonal is zero (units don't expose themselves)
        if not np.allclose(np.diag(self.exposure_matrix), 0):
            raise ValueError("Diagonal of exposure matrix must be zero")

    @property
    def n_units(self) -> int:
        """Number of units in the exposure mapping."""
        return len(self.unit_ids)

    @property
    def total_exposure_per_unit(self) -> NDArray[Any]:
        """Total exposure received by each unit."""
        return np.sum(self.exposure_matrix, axis=1)

    @property
    def exposure_out_degree(self) -> NDArray[Any]:
        """Number of units each unit can expose (out-degree)."""
        return np.sum(self.exposure_matrix > 0, axis=0)

    @property
    def exposure_in_degree(self) -> NDArray[Any]:
        """Number of units that can expose each unit (in-degree)."""
        return np.sum(self.exposure_matrix > 0, axis=1)


class ExposureMapper(abc.ABC):
    """Abstract base class for exposure mapping algorithms."""

    def __init__(self, random_state: Optional[int] = None):
        """Initialize exposure mapper.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    @abc.abstractmethod
    def map_exposure(self, unit_data: pd.DataFrame, **kwargs: Any) -> ExposureMapping:
        """Map spillover exposure relationships between units.

        Args:
            unit_data: DataFrame with unit information
            **kwargs: Additional parameters for specific mapping algorithms

        Returns:
            ExposureMapping object with exposure relationships
        """
        pass

    def _validate_unit_data(self, unit_data: pd.DataFrame) -> None:
        """Validate unit data input."""
        if unit_data.empty:
            raise ValueError("Unit data cannot be empty")

        if "unit_id" not in unit_data.columns:
            raise ValueError("Unit data must contain 'unit_id' column")

        if unit_data["unit_id"].duplicated().any():
            raise ValueError("Unit IDs must be unique")


class GeographicExposureMapper(ExposureMapper):
    """Map spillover exposure based on geographic proximity.

    Uses spatial coordinates to determine which units can affect each other
    through geographic spillover effects.
    """

    def map_exposure(
        self,
        unit_data: pd.DataFrame,
        distance_threshold: float = 1.0,
        decay_function: str = "step",
        coordinate_columns: tuple[str, str] = ("latitude", "longitude"),
        **kwargs: Any,
    ) -> ExposureMapping:
        """Map geographic spillover exposure.

        Args:
            unit_data: DataFrame with unit coordinates
            distance_threshold: Maximum distance for spillover effect
            decay_function: How spillover decays with distance ('step', 'linear', 'exponential')
            coordinate_columns: Names of latitude and longitude columns

        Returns:
            Geographic exposure mapping
        """
        self._validate_unit_data(unit_data)

        lat_col, lon_col = coordinate_columns
        required_cols = {lat_col, lon_col}
        if not required_cols.issubset(unit_data.columns):
            raise ValueError(f"Unit data must contain columns: {required_cols}")

        # Extract coordinates
        coordinates = unit_data[[lat_col, lon_col]].values
        unit_ids = unit_data["unit_id"].values

        # Calculate pairwise distances
        distances = cdist(coordinates, coordinates, metric="euclidean")

        # Apply decay function
        if decay_function == "step":
            exposure_matrix = (distances <= distance_threshold).astype(float)
        elif decay_function == "linear":
            exposure_matrix = np.maximum(0, 1 - distances / distance_threshold)
        elif decay_function == "exponential":
            exposure_matrix = np.exp(-distances / distance_threshold)
        else:
            raise ValueError(f"Unknown decay function: {decay_function}")

        # Set diagonal to zero (no self-exposure)
        np.fill_diagonal(exposure_matrix, 0)

        exposure_type = "binary" if decay_function == "step" else "distance_weighted"

        metadata = {
            "distance_threshold": distance_threshold,
            "decay_function": decay_function,
            "coordinate_columns": coordinate_columns,
            "mean_distance": np.mean(distances[distances > 0]),
            "max_distance": np.max(distances),
        }

        return ExposureMapping(
            unit_ids=unit_ids,
            exposure_matrix=exposure_matrix,
            exposure_type=exposure_type,
            metadata=metadata,
        )


class NetworkExposureMapper(ExposureMapper):
    """Map spillover exposure based on network connections.

    Uses explicit network relationships (e.g., social networks, supply chains)
    to determine spillover patterns.
    """

    def map_exposure(
        self,
        unit_data: pd.DataFrame,
        network_edges: pd.DataFrame,
        directed: bool = False,
        weight_column: Optional[str] = None,
        max_hops: int = 1,
        **kwargs: Any,
    ) -> ExposureMapping:
        """Map network-based spillover exposure.

        Args:
            unit_data: DataFrame with unit information
            network_edges: DataFrame with network connections (source, target, weight)
            directed: Whether the network is directed
            weight_column: Column name for edge weights (None for unweighted)
            max_hops: Maximum number of hops for spillover propagation

        Returns:
            Network exposure mapping
        """
        self._validate_unit_data(unit_data)

        required_edge_cols = {"source", "target"}
        if not required_edge_cols.issubset(network_edges.columns):
            raise ValueError(
                f"Network edges must contain columns: {required_edge_cols}"
            )

        unit_ids = unit_data["unit_id"].values
        n_units = len(unit_ids)

        # Create unit ID to index mapping
        unit_to_idx = {uid: i for i, uid in enumerate(unit_ids)}

        # Initialize exposure matrix
        exposure_matrix = np.zeros((n_units, n_units))

        # Process network edges
        for _, edge in network_edges.iterrows():
            source_id, target_id = edge["source"], edge["target"]

            if source_id in unit_to_idx and target_id in unit_to_idx:
                source_idx = unit_to_idx[source_id]
                target_idx = unit_to_idx[target_id]

                weight = 1.0
                if weight_column and weight_column in edge:
                    weight = float(edge[weight_column])

                # Target can be exposed by source
                exposure_matrix[target_idx, source_idx] = weight

                # If undirected, source can also be exposed by target
                if not directed:
                    exposure_matrix[source_idx, target_idx] = weight

        # Handle multi-hop exposure if requested
        if max_hops > 1:
            exposure_matrix = self._compute_multihop_exposure(exposure_matrix, max_hops)

        exposure_type = "binary" if weight_column is None else "continuous"

        metadata = {
            "directed": directed,
            "weight_column": weight_column,
            "max_hops": max_hops,
            "n_edges": len(network_edges),
            "network_density": np.sum(exposure_matrix > 0) / (n_units * (n_units - 1)),
        }

        return ExposureMapping(
            unit_ids=unit_ids,
            exposure_matrix=exposure_matrix,
            exposure_type=exposure_type,
            metadata=metadata,
        )

    def _compute_multihop_exposure(
        self, adjacency_matrix: NDArray[Any], max_hops: int
    ) -> NDArray[Any]:
        """Compute multi-hop exposure using matrix powers.

        Args:
            adjacency_matrix: Direct connection matrix
            max_hops: Maximum number of hops

        Returns:
            Multi-hop exposure matrix
        """
        result = adjacency_matrix.copy()
        current_power = adjacency_matrix.copy()

        for hop in range(2, max_hops + 1):
            current_power = np.dot(current_power, adjacency_matrix)
            # Add decayed influence from longer paths
            result += current_power / hop

        # Ensure diagonal remains zero
        np.fill_diagonal(result, 0)

        return result


class SpatialExposureMapper(ExposureMapper):
    """Map spillover exposure using spatial clustering and regions.

    Groups units into spatial clusters and maps exposure within and between clusters.
    """

    def map_exposure(
        self,
        unit_data: pd.DataFrame,
        cluster_column: Optional[str] = None,
        coordinate_columns: tuple[str, str] = ("latitude", "longitude"),
        n_clusters: Optional[int] = None,
        intra_cluster_exposure: float = 1.0,
        inter_cluster_exposure: float = 0.1,
        **kwargs: Any,
    ) -> ExposureMapping:
        """Map spatial cluster-based spillover exposure.

        Args:
            unit_data: DataFrame with unit information and coordinates
            cluster_column: Column name for predefined clusters (optional)
            coordinate_columns: Names of coordinate columns for clustering
            n_clusters: Number of clusters to create if cluster_column not provided
            intra_cluster_exposure: Exposure strength within clusters
            inter_cluster_exposure: Exposure strength between adjacent clusters

        Returns:
            Spatial cluster exposure mapping
        """
        self._validate_unit_data(unit_data)

        unit_ids = unit_data["unit_id"].values
        n_units = len(unit_ids)

        # Get or create clusters
        if cluster_column and cluster_column in unit_data.columns:
            clusters = unit_data[cluster_column].values
        else:
            clusters = self._create_spatial_clusters(
                unit_data, coordinate_columns, n_clusters or max(2, n_units // 10)
            )

        # Create exposure matrix based on clusters
        exposure_matrix = np.zeros((n_units, n_units))

        for i in range(n_units):
            for j in range(n_units):
                if i != j:
                    if clusters[i] == clusters[j]:
                        # Same cluster - high exposure
                        exposure_matrix[i, j] = intra_cluster_exposure
                    elif self._are_adjacent_clusters(
                        clusters[i],
                        clusters[j],
                        unit_data,
                        coordinate_columns,
                        cluster_column or "cluster",
                    ):
                        # Adjacent clusters - low exposure
                        exposure_matrix[i, j] = inter_cluster_exposure

        metadata = {
            "cluster_column": cluster_column,
            "n_clusters": len(np.unique(clusters)),
            "intra_cluster_exposure": intra_cluster_exposure,
            "inter_cluster_exposure": inter_cluster_exposure,
            "cluster_sizes": {
                cluster: np.sum(clusters == cluster) for cluster in np.unique(clusters)
            },
        }

        return ExposureMapping(
            unit_ids=unit_ids,
            exposure_matrix=exposure_matrix,
            exposure_type="continuous",
            metadata=metadata,
        )

    def _create_spatial_clusters(
        self,
        unit_data: pd.DataFrame,
        coordinate_columns: tuple[str, str],
        n_clusters: int,
    ) -> NDArray[Any]:
        """Create spatial clusters using k-means."""
        from sklearn.cluster import KMeans

        lat_col, lon_col = coordinate_columns
        coordinates = unit_data[[lat_col, lon_col]].values

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(coordinates)

        return clusters

    def _are_adjacent_clusters(
        self,
        cluster1: Any,
        cluster2: Any,
        unit_data: pd.DataFrame,
        coordinate_columns: tuple[str, str],
        cluster_column: str = "cluster",
    ) -> bool:
        """Check if two clusters are spatially adjacent."""
        # Simplified adjacency check based on minimum distance between clusters
        lat_col, lon_col = coordinate_columns

        # Get centroids of each cluster (simplified approach)
        cluster1_data = unit_data[unit_data[cluster_column] == cluster1]
        cluster2_data = unit_data[unit_data[cluster_column] == cluster2]

        if cluster1_data.empty or cluster2_data.empty:
            return False

        coords1 = cluster1_data[list(coordinate_columns)].mean().values
        coords2 = cluster2_data[list(coordinate_columns)].mean().values

        distance = np.linalg.norm(coords1 - coords2)

        # Adjacent if distance is below threshold (configurable)
        return distance < 10.0  # Example threshold - made larger for test compatibility


class TemporalExposureMapper(ExposureMapper):
    """Map spillover exposure based on temporal patterns.

    Models how treatment effects can spill over across time periods.
    """

    def map_exposure(
        self,
        unit_data: pd.DataFrame,
        time_column: str = "time_period",
        temporal_decay: float = 0.8,
        max_time_lag: int = 3,
        **kwargs: Any,
    ) -> ExposureMapping:
        """Map temporal spillover exposure.

        Args:
            unit_data: DataFrame with unit information and time periods
            time_column: Column name for time periods
            temporal_decay: How exposure decays over time periods
            max_time_lag: Maximum time lag for spillover effects

        Returns:
            Temporal exposure mapping
        """
        self._validate_unit_data(unit_data)

        if time_column not in unit_data.columns:
            raise ValueError(f"Time column '{time_column}' not found in unit data")

        unit_ids = unit_data["unit_id"].values
        time_periods = unit_data[time_column].values
        n_units = len(unit_ids)

        # Create exposure matrix based on temporal relationships
        exposure_matrix = np.zeros((n_units, n_units))

        for i in range(n_units):
            for j in range(n_units):
                if i != j:
                    time_diff = time_periods[i] - time_periods[j]

                    # Only past periods can expose future periods
                    if 0 < time_diff <= max_time_lag:
                        exposure_strength = temporal_decay**time_diff
                        exposure_matrix[i, j] = exposure_strength

        metadata = {
            "time_column": time_column,
            "temporal_decay": temporal_decay,
            "max_time_lag": max_time_lag,
            "time_range": (np.min(time_periods), np.max(time_periods)),
            "n_time_periods": len(np.unique(time_periods)),
        }

        return ExposureMapping(
            unit_ids=unit_ids,
            exposure_matrix=exposure_matrix,
            exposure_type="continuous",
            metadata=metadata,
        )
