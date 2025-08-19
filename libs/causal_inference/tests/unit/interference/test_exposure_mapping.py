"""Unit tests for exposure mapping functionality."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.interference.exposure_mapping import (
    ExposureMapping,
    GeographicExposureMapper,
    NetworkExposureMapper,
    SpatialExposureMapper,
    TemporalExposureMapper,
)


class TestExposureMapping:
    """Test ExposureMapping data model."""

    def test_exposure_mapping_validation(self):
        """Test basic validation of exposure mapping."""
        unit_ids = np.array([1, 2, 3])
        exposure_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        mapping = ExposureMapping(
            unit_ids=unit_ids, exposure_matrix=exposure_matrix, exposure_type="binary"
        )

        assert mapping.n_units == 3
        assert np.array_equal(mapping.unit_ids, unit_ids)
        assert mapping.exposure_type == "binary"

    def test_exposure_mapping_diagonal_validation(self):
        """Test that diagonal must be zero."""
        unit_ids = np.array([1, 2])
        exposure_matrix = np.array(
            [
                [1, 1],  # Diagonal should be 0
                [1, 0],
            ]
        )

        with pytest.raises(
            ValueError, match="Diagonal of exposure matrix must be zero"
        ):
            ExposureMapping(
                unit_ids=unit_ids,
                exposure_matrix=exposure_matrix,
                exposure_type="binary",
            )

    def test_exposure_mapping_shape_validation(self):
        """Test shape validation between units and matrix."""
        unit_ids = np.array([1, 2, 3])
        exposure_matrix = np.array([[0, 1], [1, 0]])  # Wrong shape

        with pytest.raises(ValueError, match="doesn't match number of units"):
            ExposureMapping(
                unit_ids=unit_ids,
                exposure_matrix=exposure_matrix,
                exposure_type="binary",
            )

    def test_exposure_mapping_properties(self):
        """Test computed properties of exposure mapping."""
        unit_ids = np.array([1, 2, 3])
        exposure_matrix = np.array([[0, 1, 0.5], [1, 0, 1], [0.5, 1, 0]])

        mapping = ExposureMapping(
            unit_ids=unit_ids,
            exposure_matrix=exposure_matrix,
            exposure_type="continuous",
        )

        # Test total exposure per unit (row sums)
        expected_exposure = np.array([1.5, 2.0, 1.5])
        np.testing.assert_array_almost_equal(
            mapping.total_exposure_per_unit, expected_exposure
        )

        # Test in-degree (number of incoming connections)
        expected_in_degree = np.array([2, 2, 2])
        np.testing.assert_array_equal(mapping.exposure_in_degree, expected_in_degree)

        # Test out-degree (number of outgoing connections)
        expected_out_degree = np.array([2, 2, 2])
        np.testing.assert_array_equal(mapping.exposure_out_degree, expected_out_degree)


class TestGeographicExposureMapper:
    """Test geographic exposure mapping."""

    def setup_method(self):
        """Set up test data."""
        self.mapper = GeographicExposureMapper(random_state=42)
        self.unit_data = pd.DataFrame(
            {
                "unit_id": [1, 2, 3, 4],
                "latitude": [0.0, 1.0, 0.0, 1.0],
                "longitude": [0.0, 0.0, 1.0, 1.0],
            }
        )

    def test_geographic_mapping_step_function(self):
        """Test geographic mapping with step decay function."""
        mapping = self.mapper.map_exposure(
            self.unit_data, distance_threshold=1.5, decay_function="step"
        )

        assert mapping.n_units == 4
        assert mapping.exposure_type == "binary"

        # Units should be connected if distance <= threshold
        # Distance between (0,0) and (1,0) is 1.0, should be connected
        assert mapping.exposure_matrix[0, 1] == 1
        assert mapping.exposure_matrix[1, 0] == 1

        # Distance between (0,0) and (1,1) is sqrt(2) ≈ 1.41, should be connected
        assert mapping.exposure_matrix[0, 3] == 1

    def test_geographic_mapping_linear_decay(self):
        """Test geographic mapping with linear decay."""
        mapping = self.mapper.map_exposure(
            self.unit_data, distance_threshold=2.0, decay_function="linear"
        )

        assert mapping.exposure_type == "distance_weighted"

        # Linear decay: exposure = max(0, 1 - distance/threshold)
        # Distance between (0,0) and (1,0) is 1.0
        # Expected exposure = 1 - 1.0/2.0 = 0.5
        assert abs(mapping.exposure_matrix[0, 1] - 0.5) < 0.01

    def test_geographic_mapping_exponential_decay(self):
        """Test geographic mapping with exponential decay."""
        mapping = self.mapper.map_exposure(
            self.unit_data, distance_threshold=1.0, decay_function="exponential"
        )

        assert mapping.exposure_type == "distance_weighted"

        # Exponential decay: exposure = exp(-distance/threshold)
        # Distance between (0,0) and (1,0) is 1.0
        # Expected exposure = exp(-1.0/1.0) = exp(-1) ≈ 0.368
        expected_exposure = np.exp(-1.0)
        assert abs(mapping.exposure_matrix[0, 1] - expected_exposure) < 0.01

    def test_geographic_mapping_missing_coordinates(self):
        """Test error handling for missing coordinate columns."""
        unit_data_no_coords = pd.DataFrame(
            {
                "unit_id": [1, 2, 3],
                "x": [0, 1, 2],  # Missing latitude/longitude
            }
        )

        with pytest.raises(ValueError, match="must contain columns"):
            self.mapper.map_exposure(unit_data_no_coords)

    def test_geographic_mapping_custom_coordinate_columns(self):
        """Test using custom coordinate column names."""
        unit_data_custom = pd.DataFrame(
            {"unit_id": [1, 2, 3], "x": [0, 1, 0], "y": [0, 0, 1]}
        )

        mapping = self.mapper.map_exposure(
            unit_data_custom, distance_threshold=1.5, coordinate_columns=("x", "y")
        )

        assert mapping.n_units == 3
        assert mapping.exposure_matrix[0, 1] == 1  # Distance = 1.0


class TestNetworkExposureMapper:
    """Test network-based exposure mapping."""

    def setup_method(self):
        """Set up test data."""
        self.mapper = NetworkExposureMapper(random_state=42)
        self.unit_data = pd.DataFrame({"unit_id": [1, 2, 3, 4]})
        self.network_edges = pd.DataFrame(
            {"source": [1, 2, 1, 3], "target": [2, 3, 3, 4]}
        )

    def test_network_mapping_unweighted_undirected(self):
        """Test unweighted undirected network mapping."""
        mapping = self.mapper.map_exposure(
            self.unit_data, self.network_edges, directed=False
        )

        assert mapping.n_units == 4
        assert mapping.exposure_type == "binary"

        # Check symmetric connections for undirected network
        assert mapping.exposure_matrix[0, 1] == 1  # 1->2
        assert mapping.exposure_matrix[1, 0] == 1  # 2->1
        assert mapping.exposure_matrix[1, 2] == 1  # 2->3
        assert mapping.exposure_matrix[2, 1] == 1  # 3->2

    def test_network_mapping_weighted_directed(self):
        """Test weighted directed network mapping."""
        weighted_edges = self.network_edges.copy()
        weighted_edges["weight"] = [0.5, 0.8, 0.3, 0.9]

        mapping = self.mapper.map_exposure(
            self.unit_data, weighted_edges, directed=True, weight_column="weight"
        )

        assert mapping.exposure_type == "continuous"

        # Check directed weighted connections
        assert mapping.exposure_matrix[1, 0] == 0.5  # 1->2 with weight 0.5
        assert mapping.exposure_matrix[2, 1] == 0.8  # 2->3 with weight 0.8
        assert mapping.exposure_matrix[0, 1] == 0.0  # No edge 2->1 in directed graph

    def test_network_mapping_multihop(self):
        """Test multi-hop exposure propagation."""
        mapping = self.mapper.map_exposure(
            self.unit_data, self.network_edges, directed=False, max_hops=2
        )

        # Should have both direct and 2-hop connections
        # 1-2-3 creates a 2-hop connection between 1 and 3
        assert mapping.exposure_matrix[0, 2] > 0  # 1->3 via 2

    def test_network_mapping_missing_edges(self):
        """Test error handling for missing edge columns."""
        bad_edges = pd.DataFrame({"from": [1, 2], "to": [2, 3]})

        with pytest.raises(ValueError, match="must contain columns"):
            self.mapper.map_exposure(self.unit_data, bad_edges)


class TestSpatialExposureMapper:
    """Test spatial cluster-based exposure mapping."""

    def setup_method(self):
        """Set up test data."""
        self.mapper = SpatialExposureMapper(random_state=42)
        self.unit_data = pd.DataFrame(
            {
                "unit_id": [1, 2, 3, 4],
                "latitude": [0.0, 0.1, 5.0, 5.1],  # Two spatial clusters
                "longitude": [0.0, 0.1, 5.0, 5.1],
                "cluster": ["A", "A", "B", "B"],
            }
        )

    def test_spatial_mapping_predefined_clusters(self):
        """Test spatial mapping with predefined clusters."""
        mapping = self.mapper.map_exposure(
            self.unit_data,
            cluster_column="cluster",
            intra_cluster_exposure=1.0,
            inter_cluster_exposure=0.1,
        )

        assert mapping.n_units == 4
        assert mapping.exposure_type == "continuous"

        # Units in same cluster should have high exposure
        assert mapping.exposure_matrix[0, 1] == 1.0  # Both in cluster A
        assert mapping.exposure_matrix[2, 3] == 1.0  # Both in cluster B

        # Units in different clusters should have low exposure
        assert mapping.exposure_matrix[0, 2] == 0.1  # A to B

    def test_spatial_mapping_auto_clustering(self):
        """Test spatial mapping with automatic clustering."""
        mapping = self.mapper.map_exposure(
            self.unit_data, n_clusters=2, coordinate_columns=("latitude", "longitude")
        )

        assert mapping.n_units == 4
        # Should create 2 spatial clusters based on coordinates


class TestTemporalExposureMapper:
    """Test temporal exposure mapping."""

    def setup_method(self):
        """Set up test data."""
        self.mapper = TemporalExposureMapper(random_state=42)
        self.unit_data = pd.DataFrame(
            {"unit_id": [1, 2, 3, 4], "time_period": [1, 2, 3, 4]}
        )

    def test_temporal_mapping_basic(self):
        """Test basic temporal exposure mapping."""
        mapping = self.mapper.map_exposure(
            self.unit_data,
            time_column="time_period",
            temporal_decay=0.8,
            max_time_lag=2,
        )

        assert mapping.n_units == 4
        assert mapping.exposure_type == "continuous"

        # Unit 2 (time=2) should be exposed by unit 1 (time=1) with decay^1
        assert abs(mapping.exposure_matrix[1, 0] - 0.8) < 0.01

        # Unit 3 (time=3) should be exposed by unit 1 (time=1) with decay^2
        assert abs(mapping.exposure_matrix[2, 0] - 0.64) < 0.01

        # Unit 1 should not be exposed by future units
        assert mapping.exposure_matrix[0, 1] == 0

        # Exposure beyond max_time_lag should be zero
        assert mapping.exposure_matrix[3, 0] == 0  # time_lag = 3 > max_time_lag = 2

    def test_temporal_mapping_missing_time_column(self):
        """Test error handling for missing time column."""
        unit_data_no_time = pd.DataFrame(
            {
                "unit_id": [1, 2, 3],
                "period": [1, 2, 3],  # Wrong column name
            }
        )

        with pytest.raises(ValueError, match="Time column.*not found"):
            self.mapper.map_exposure(unit_data_no_time)

    def test_temporal_mapping_custom_parameters(self):
        """Test temporal mapping with custom parameters."""
        mapping = self.mapper.map_exposure(
            self.unit_data,
            time_column="time_period",
            temporal_decay=0.5,
            max_time_lag=1,
        )

        # Only 1-step temporal exposure should exist
        assert abs(mapping.exposure_matrix[1, 0] - 0.5) < 0.01
        assert mapping.exposure_matrix[2, 0] == 0  # Beyond max_time_lag
