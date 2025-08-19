"""Unit tests for interference diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_inference.core.base import OutcomeData, TreatmentData
from causal_inference.interference.diagnostics import (
    InterferenceDiagnostics,
    SpilloverDetectionResults,
    plot_cluster_exposure_balance,
    plot_network_connectivity,
    plot_spillover_detection_power,
)
from causal_inference.interference.exposure_mapping import ExposureMapping


class TestSpilloverDetectionResults:
    """Test spillover detection results data structure."""

    def test_spillover_detection_results_initialization(self):
        """Test basic initialization."""
        results = SpilloverDetectionResults(
            spillover_detected=True,
            detection_confidence=0.8,
            detection_method="comprehensive",
            exposure_imbalance=0.3,
            max_exposure_ratio=2.5,
            exposure_balance_pvalue=0.04,
            network_density=0.15,
            clustering_coefficient=0.6,
        )

        assert results.spillover_detected
        assert results.detection_confidence == 0.8
        assert results.detection_method == "comprehensive"
        assert results.exposure_imbalance == 0.3
        assert results.network_density == 0.15
        assert results.warnings == []  # Should initialize empty
        assert results.recommendations == []  # Should initialize empty


class TestInterferenceDiagnostics:
    """Test interference diagnostics functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_units = 100

        # Create network structure with clusters
        exposure_matrix = np.zeros((n_units, n_units))

        # Create 4 clusters with internal connections
        cluster_size = 25
        for cluster in range(4):
            start = cluster * cluster_size
            end = (cluster + 1) * cluster_size

            # Within-cluster connections
            for i in range(start, end):
                for j in range(start, end):
                    if i != j and np.random.random() < 0.3:
                        exposure_matrix[i, j] = 1.0

            # Some between-cluster connections
            if cluster < 3:
                next_start = (cluster + 1) * cluster_size
                next_end = (cluster + 2) * cluster_size
                for i in range(start, end):
                    for j in range(next_start, next_end):
                        if np.random.random() < 0.05:
                            exposure_matrix[i, j] = 1.0

        self.exposure_mapping = ExposureMapping(
            unit_ids=np.arange(n_units),
            exposure_matrix=exposure_matrix,
            exposure_type="binary",
        )

        # Generate treatment with some imbalance
        self.treatment = TreatmentData(
            values=np.random.binomial(1, 0.4, n_units), treatment_type="binary"
        )

        # Generate outcome with spillover effects
        spillover_exposure = np.dot(exposure_matrix, self.treatment.values)
        self.outcome = OutcomeData(
            values=(
                0.5 * self.treatment.values
                + 0.2 * spillover_exposure
                + np.random.normal(0, 0.5, n_units)
            )
        )

        # Unit data with coordinates
        self.unit_data = pd.DataFrame(
            {
                "unit_id": np.arange(n_units),
                "latitude": np.random.uniform(0, 10, n_units),
                "longitude": np.random.uniform(0, 10, n_units),
                "cluster": np.repeat([0, 1, 2, 3], cluster_size),
            }
        )

        self.diagnostics = InterferenceDiagnostics(
            exposure_mapping=self.exposure_mapping, random_state=42
        )

    def test_diagnostics_initialization(self):
        """Test diagnostics initialization."""
        assert self.diagnostics.exposure_mapping == self.exposure_mapping
        assert self.diagnostics.random_state == 42

    def test_comprehensive_diagnostics(self):
        """Test comprehensive diagnostic analysis."""
        results = self.diagnostics.run_comprehensive_diagnostics(
            treatment=self.treatment, outcome=self.outcome, unit_data=self.unit_data
        )

        assert isinstance(results, SpilloverDetectionResults)
        assert results.detection_method == "comprehensive"
        assert results.exposure_imbalance is not None
        assert results.max_exposure_ratio is not None
        assert results.exposure_balance_pvalue is not None
        assert results.network_density is not None
        assert results.clustering_coefficient is not None
        assert results.estimated_power is not None
        assert results.minimum_detectable_effect is not None
        assert isinstance(results.warnings, list)
        assert isinstance(results.recommendations, list)

    def test_exposure_balance_check(self):
        """Test exposure balance checking."""
        balance_results = self.diagnostics._check_exposure_balance(self.treatment)

        assert "imbalance" in balance_results
        assert "max_ratio" in balance_results
        assert "pvalue" in balance_results
        assert balance_results["imbalance"] >= 0
        assert balance_results["max_ratio"] >= 1.0
        assert 0 <= balance_results["pvalue"] <= 1

    def test_network_connectivity_analysis(self):
        """Test network connectivity analysis."""
        connectivity = self.diagnostics._analyze_network_connectivity()

        assert "density" in connectivity
        assert "clustering" in connectivity
        assert 0 <= connectivity["density"] <= 1
        assert 0 <= connectivity["clustering"] <= 1

    def test_spatial_autocorrelation_tests(self):
        """Test spatial autocorrelation tests."""
        spatial_tests = self.diagnostics._spatial_autocorrelation_tests(
            self.treatment, self.outcome, self.unit_data
        )

        # Should run Moran's I and Geary's C tests
        if "moran_i" in spatial_tests:
            assert "moran_i_pvalue" in spatial_tests
            assert 0 <= spatial_tests["moran_i_pvalue"] <= 1

        if "geary_c" in spatial_tests:
            assert "geary_c_pvalue" in spatial_tests
            assert 0 <= spatial_tests["geary_c_pvalue"] <= 1

    def test_morans_i_calculation(self):
        """Test Moran's I spatial autocorrelation calculation."""
        values = np.array([1, 0, 1, 0])
        coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        result = self.diagnostics._calculate_morans_i(values, coordinates)

        assert "statistic" in result
        assert "pvalue" in result
        assert isinstance(result["statistic"], float)
        assert 0 <= result["pvalue"] <= 1

    def test_geary_c_calculation(self):
        """Test Geary's C spatial autocorrelation calculation."""
        values = np.array([1.0, 2.0, 1.5, 2.5])
        coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        result = self.diagnostics._calculate_geary_c(values, coordinates)

        assert "statistic" in result
        assert "pvalue" in result
        assert isinstance(result["statistic"], float)
        assert 0 <= result["pvalue"] <= 1

    def test_spillover_power_analysis(self):
        """Test spillover power analysis."""
        power_results = self.diagnostics._spillover_power_analysis(
            self.treatment, self.outcome
        )

        assert "power" in power_results
        assert "mde" in power_results
        assert 0 <= power_results["power"] <= 1
        assert power_results["mde"] > 0

    def test_spillover_presence_detection(self):
        """Test spillover presence detection."""
        # Create mock diagnostic results
        results = SpilloverDetectionResults(
            spillover_detected=False,
            detection_confidence=0.0,
            detection_method="test",
            exposure_imbalance=0.1,
            max_exposure_ratio=1.2,
            exposure_balance_pvalue=0.08,
            network_density=0.15,
            clustering_coefficient=0.5,
            moran_i_pvalue=0.06,
            geary_c_pvalue=0.12,
        )

        detection = self.diagnostics._detect_spillover_presence(
            self.treatment, self.outcome, results
        )

        assert "detected" in detection
        assert "confidence" in detection
        assert isinstance(detection["detected"], bool)
        assert 0 <= detection["confidence"] <= 1

    def test_warnings_generation(self):
        """Test warning generation."""
        results = SpilloverDetectionResults(
            spillover_detected=True,
            detection_confidence=0.8,
            detection_method="test",
            exposure_imbalance=2.0,  # High imbalance
            max_exposure_ratio=8.0,  # High ratio
            exposure_balance_pvalue=0.05,
            network_density=0.005,  # Very low density
            clustering_coefficient=0.3,
            estimated_power=0.4,  # Low power
        )

        warnings = self.diagnostics._generate_warnings(results)

        assert isinstance(warnings, list)
        assert len(warnings) > 0  # Should generate warnings for the problematic values

        # Check for specific warnings
        warning_text = " ".join(warnings)
        assert "low network density" in warning_text.lower()
        assert "imbalance" in warning_text.lower()
        assert "power" in warning_text.lower()

    def test_recommendations_generation(self):
        """Test recommendation generation."""
        results = SpilloverDetectionResults(
            spillover_detected=True,
            detection_confidence=0.9,
            detection_method="test",
            exposure_imbalance=0.8,
            max_exposure_ratio=3.0,
            exposure_balance_pvalue=0.02,
            network_density=0.4,  # High density
            clustering_coefficient=0.7,
            estimated_power=0.6,
        )

        recommendations = self.diagnostics._generate_recommendations(results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check for specific recommendations
        rec_text = " ".join(recommendations)
        assert "spillover" in rec_text.lower() or "interference" in rec_text.lower()

    def test_diagnostics_without_spatial_data(self):
        """Test diagnostics without spatial coordinate data."""
        unit_data_no_coords = pd.DataFrame(
            {"unit_id": np.arange(100), "cluster": np.repeat([0, 1, 2, 3], 25)}
        )

        results = self.diagnostics.run_comprehensive_diagnostics(
            treatment=self.treatment,
            outcome=self.outcome,
            unit_data=unit_data_no_coords,
        )

        # Should still work, just without spatial tests
        assert results.moran_i_statistic is None
        assert results.geary_c_statistic is None
        assert results.network_density is not None


class TestDiagnosticPlots:
    """Test diagnostic plotting functions."""

    def setup_method(self):
        """Set up test data for plotting."""
        np.random.seed(42)
        n_units = 50

        # Simple exposure mapping
        exposure_matrix = np.random.random((n_units, n_units)) < 0.1
        np.fill_diagonal(exposure_matrix, 0)

        self.exposure_mapping = ExposureMapping(
            unit_ids=np.arange(n_units),
            exposure_matrix=exposure_matrix.astype(float),
            exposure_type="binary",
        )

        self.treatment = TreatmentData(
            values=np.random.binomial(1, 0.5, n_units), treatment_type="binary"
        )

        self.clusters = np.random.randint(0, 3, n_units)

    def test_plot_cluster_exposure_balance(self):
        """Test cluster exposure balance plotting."""
        fig = plot_cluster_exposure_balance(
            exposure_mapping=self.exposure_mapping,
            treatment=self.treatment,
            clusters=self.clusters,
            figsize=(8, 6),
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Should have 2 subplots
        plt.close(fig)

    def test_plot_cluster_exposure_balance_no_clusters(self):
        """Test plotting without cluster information."""
        fig = plot_cluster_exposure_balance(
            exposure_mapping=self.exposure_mapping,
            treatment=self.treatment,
            clusters=None,
            figsize=(8, 6),
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_network_connectivity_basic(self):
        """Test basic network connectivity plotting."""
        fig = plot_network_connectivity(
            exposure_mapping=self.exposure_mapping,
            treatment=self.treatment,
            layout="spring",
            figsize=(10, 8),
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_network_connectivity_no_treatment(self):
        """Test network plotting without treatment information."""
        fig = plot_network_connectivity(
            exposure_mapping=self.exposure_mapping,
            treatment=None,
            layout="circular",
            figsize=(8, 8),
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_spillover_detection_power(self):
        """Test spillover detection power plotting."""
        # Create mock power analysis results
        power_data = []
        for effect_size in [0.1, 0.3, 0.5]:
            for sample_size in [50, 100, 200]:
                # Mock power calculation
                power = min(0.95, effect_size * 2 + sample_size / 300)
                power_data.append(
                    {
                        "effect_size": effect_size,
                        "sample_size": sample_size,
                        "power": power,
                        "alpha": 0.05,
                    }
                )

        power_results = pd.DataFrame(power_data)

        fig = plot_spillover_detection_power(
            power_analysis_results=power_results, figsize=(12, 6)
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Power vs effect size and power vs sample size
        plt.close(fig)

    def test_plot_power_analysis_single_condition(self):
        """Test power plotting with single condition."""
        power_data = pd.DataFrame(
            {
                "effect_size": [0.3],
                "sample_size": [100],
                "power": [0.8],
                "alpha": [0.05],
            }
        )

        fig = plot_spillover_detection_power(power_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestDiagnosticsEdgeCases:
    """Test edge cases and error handling in diagnostics."""

    def test_diagnostics_empty_network(self):
        """Test diagnostics with empty network."""
        n_units = 10
        empty_exposure = ExposureMapping(
            unit_ids=np.arange(n_units),
            exposure_matrix=np.zeros((n_units, n_units)),
            exposure_type="binary",
        )

        treatment = TreatmentData(
            values=np.random.binomial(1, 0.5, n_units), treatment_type="binary"
        )

        outcome = OutcomeData(values=np.random.normal(0, 1, n_units))

        diagnostics = InterferenceDiagnostics(empty_exposure)
        results = diagnostics.run_comprehensive_diagnostics(treatment, outcome)

        assert results.network_density == 0.0
        assert results.clustering_coefficient == 0.0

    def test_diagnostics_no_treatment_variation(self):
        """Test diagnostics with no treatment variation."""
        n_units = 20
        exposure_matrix = np.random.random((n_units, n_units)) < 0.1
        np.fill_diagonal(exposure_matrix, 0)

        exposure_mapping = ExposureMapping(
            unit_ids=np.arange(n_units),
            exposure_matrix=exposure_matrix.astype(float),
            exposure_type="binary",
        )

        # Mostly treated units (19 treated, 1 control for valid binary treatment)
        treatment_values = np.ones(n_units)
        treatment_values[0] = 0  # Make first unit control
        treatment = TreatmentData(values=treatment_values, treatment_type="binary")

        outcome = OutcomeData(values=np.random.normal(0, 1, n_units))

        diagnostics = InterferenceDiagnostics(exposure_mapping)
        results = diagnostics.run_comprehensive_diagnostics(treatment, outcome)

        # Should handle gracefully
        assert results.exposure_balance_pvalue == 1.0

    def test_diagnostics_minimal_units(self):
        """Test diagnostics with minimal valid units (2 units for binary treatment)."""
        exposure_mapping = ExposureMapping(
            unit_ids=np.array([0, 1]),
            exposure_matrix=np.array([[0, 0], [0, 0]]),
            exposure_type="binary",
        )

        treatment = TreatmentData(values=np.array([1, 0]), treatment_type="binary")

        outcome = OutcomeData(values=np.array([2.0, 1.0]))

        diagnostics = InterferenceDiagnostics(exposure_mapping)
        results = diagnostics.run_comprehensive_diagnostics(treatment, outcome)

        # Should handle edge case gracefully
        assert results.network_density == 0.0
        assert results.exposure_imbalance == 0.0
