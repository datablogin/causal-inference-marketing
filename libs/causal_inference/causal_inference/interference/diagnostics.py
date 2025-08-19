"""Diagnostic Tools for Interference and Spillover Detection.

This module provides comprehensive diagnostics for identifying spillover patterns,
assessing exposure balance, and validating interference assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy import stats

from ..core.base import CovariateData, OutcomeData, TreatmentData
from .exposure_mapping import ExposureMapping


@dataclass
class SpilloverDetectionResults:
    """Results from spillover detection diagnostics."""

    # Detection metrics
    spillover_detected: bool
    detection_confidence: float
    detection_method: str

    # Exposure balance metrics
    exposure_imbalance: float
    max_exposure_ratio: float
    exposure_balance_pvalue: float

    # Network connectivity diagnostics
    network_density: float
    clustering_coefficient: float
    average_path_length: float | None = None
    modularity: float | None = None

    # Statistical tests
    moran_i_statistic: float | None = None
    moran_i_pvalue: float | None = None
    geary_c_statistic: float | None = None
    geary_c_pvalue: float | None = None

    # Power analysis
    estimated_power: float | None = None
    minimum_detectable_effect: float | None = None

    # Warnings and recommendations
    warnings: list[str] = None
    recommendations: list[str] = None

    def __post_init__(self) -> None:
        """Initialize empty lists if None."""
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []


class InterferenceDiagnostics:
    """Comprehensive diagnostics for interference and spillover effects."""

    def __init__(
        self, exposure_mapping: ExposureMapping, random_state: int | None = None
    ):
        """Initialize interference diagnostics.

        Args:
            exposure_mapping: Network exposure relationships
            random_state: Random seed for reproducible results
        """
        self.exposure_mapping = exposure_mapping
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def run_comprehensive_diagnostics(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        unit_data: pd.DataFrame | None = None,
    ) -> SpilloverDetectionResults:
        """Run comprehensive spillover detection diagnostics.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            covariates: Optional covariate data
            unit_data: Optional unit-level data with coordinates/metadata

        Returns:
            Comprehensive spillover detection results
        """
        results = SpilloverDetectionResults(
            spillover_detected=False,
            detection_confidence=0.0,
            detection_method="comprehensive",
            exposure_imbalance=0.0,
            max_exposure_ratio=1.0,
            exposure_balance_pvalue=1.0,
            network_density=0.0,
            clustering_coefficient=0.0,
        )

        # 1. Check exposure balance
        exposure_balance = self._check_exposure_balance(treatment)
        results.exposure_imbalance = exposure_balance["imbalance"]
        results.max_exposure_ratio = exposure_balance["max_ratio"]
        results.exposure_balance_pvalue = exposure_balance["pvalue"]

        # 2. Analyze network connectivity
        connectivity = self._analyze_network_connectivity()
        results.network_density = connectivity["density"]
        results.clustering_coefficient = connectivity["clustering"]
        results.average_path_length = connectivity.get("avg_path_length")
        results.modularity = connectivity.get("modularity")

        # 3. Spatial autocorrelation tests (if applicable)
        if unit_data is not None:
            spatial_tests = self._spatial_autocorrelation_tests(
                treatment, outcome, unit_data
            )
            results.moran_i_statistic = spatial_tests.get("moran_i")
            results.moran_i_pvalue = spatial_tests.get("moran_i_pvalue")
            results.geary_c_statistic = spatial_tests.get("geary_c")
            results.geary_c_pvalue = spatial_tests.get("geary_c_pvalue")

        # 4. Power analysis
        power_analysis = self._spillover_power_analysis(treatment, outcome)
        results.estimated_power = power_analysis["power"]
        results.minimum_detectable_effect = power_analysis["mde"]

        # 5. Overall spillover detection
        detection = self._detect_spillover_presence(treatment, outcome, results)
        results.spillover_detected = detection["detected"]
        results.detection_confidence = detection["confidence"]

        # 6. Generate warnings and recommendations
        results.warnings = self._generate_warnings(results)
        results.recommendations = self._generate_recommendations(results)

        return results

    def _check_exposure_balance(self, treatment: TreatmentData) -> dict[str, float]:
        """Check balance in spillover exposure across treatment groups."""
        treatment_array = np.array(treatment.values)
        exposure_matrix = self.exposure_mapping.exposure_matrix

        # Calculate spillover exposure for each unit
        spillover_exposure = np.dot(exposure_matrix, treatment_array)

        # Split by treatment status
        treated_mask = treatment_array == 1
        control_mask = treatment_array == 0

        if np.sum(treated_mask) == 0 or np.sum(control_mask) == 0:
            return {"imbalance": 0.0, "max_ratio": 1.0, "pvalue": 1.0}

        treated_exposure = spillover_exposure[treated_mask]
        control_exposure = spillover_exposure[control_mask]

        # Calculate imbalance metrics
        treated_mean = np.mean(treated_exposure)
        control_mean = np.mean(control_exposure)

        imbalance = abs(treated_mean - control_mean)
        max_ratio = max(treated_mean, control_mean) / (
            min(treated_mean, control_mean) + 1e-8
        )

        # Statistical test for exposure balance
        if len(treated_exposure) > 1 and len(control_exposure) > 1:
            statistic, pvalue = stats.ttest_ind(treated_exposure, control_exposure)
        else:
            pvalue = 1.0

        return {"imbalance": imbalance, "max_ratio": max_ratio, "pvalue": pvalue}

    def _analyze_network_connectivity(self) -> dict[str, Any]:
        """Analyze network connectivity properties."""
        exposure_matrix = self.exposure_mapping.exposure_matrix
        n_units = exposure_matrix.shape[0]

        # Binary adjacency matrix (any positive exposure = connected)
        adjacency = (exposure_matrix > 0).astype(int)

        # Network density
        possible_edges = n_units * (n_units - 1)
        actual_edges = np.sum(adjacency)
        density = actual_edges / possible_edges if possible_edges > 0 else 0

        # Clustering coefficient (local clustering)
        clustering_coeffs = []
        for i in range(n_units):
            neighbors = np.where(adjacency[i] == 1)[0]
            if len(neighbors) < 2:
                clustering_coeffs.append(0)
                continue

            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2

            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adjacency[neighbors[j], neighbors[k]] == 1:
                        triangles += 1

            clustering_coeff = (
                triangles / possible_triangles if possible_triangles > 0 else 0
            )
            clustering_coeffs.append(clustering_coeff)

        avg_clustering = np.mean(clustering_coeffs)

        results = {"density": density, "clustering": avg_clustering}

        # Average path length (simplified calculation)
        try:
            # Use Floyd-Warshall for small networks
            if n_units <= 100:
                path_lengths = self._calculate_average_path_length(adjacency)
                results["avg_path_length"] = path_lengths
        except Exception:
            pass

        return results

    def _calculate_average_path_length(self, adjacency: NDArray[Any]) -> float | None:
        """Calculate average shortest path length."""
        n = adjacency.shape[0]

        # Initialize distance matrix
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)

        # Set distances for direct connections
        dist[adjacency == 1] = 1

        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

        # Calculate average path length for connected pairs
        finite_distances = dist[np.isfinite(dist) & (dist > 0)]

        if len(finite_distances) > 0:
            return float(np.mean(finite_distances))
        else:
            return None

    def _spatial_autocorrelation_tests(
        self, treatment: TreatmentData, outcome: OutcomeData, unit_data: pd.DataFrame
    ) -> dict[str, float]:
        """Test for spatial autocorrelation in treatment or outcomes."""
        results = {}

        # Check if we have coordinate data
        coord_cols = ["latitude", "longitude", "lat", "lon", "x", "y"]
        available_coords = [col for col in coord_cols if col in unit_data.columns]

        if len(available_coords) >= 2:
            try:
                # Moran's I test for treatment assignment
                moran_result = self._calculate_morans_i(
                    np.array(treatment.values), unit_data[available_coords[:2]].values
                )
                results["moran_i"] = moran_result["statistic"]
                results["moran_i_pvalue"] = moran_result["pvalue"]

                # Geary's C test for outcome variable
                geary_result = self._calculate_geary_c(
                    np.array(outcome.values), unit_data[available_coords[:2]].values
                )
                results["geary_c"] = geary_result["statistic"]
                results["geary_c_pvalue"] = geary_result["pvalue"]

            except Exception:
                # Fall back to network-based autocorrelation
                pass

        return results

    def _calculate_morans_i(
        self, values: NDArray[Any], coordinates: NDArray[Any]
    ) -> dict[str, float]:
        """Calculate Moran's I spatial autocorrelation statistic."""
        n = len(values)

        # Create spatial weights matrix (inverse distance)
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(coordinates))

        # Avoid division by zero
        weights = 1 / (distances + 1e-8)
        np.fill_diagonal(weights, 0)

        # Normalize weights
        row_sums = np.sum(weights, axis=1)
        weights = weights / (row_sums[:, np.newaxis] + 1e-8)

        # Calculate Moran's I
        mean_val = np.mean(values)
        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                numerator += (
                    weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
                )
            denominator += (values[i] - mean_val) ** 2

        W = np.sum(weights)
        morans_i = (n / W) * (numerator / denominator) if denominator > 0 else 0

        # Approximate p-value (simplified)
        expected = -1 / (n - 1)
        variance = 2 / ((n - 1) * (n - 2))  # Simplified
        z_score = (morans_i - expected) / np.sqrt(variance)
        pvalue = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {"statistic": float(morans_i), "pvalue": float(pvalue)}

    def _calculate_geary_c(
        self, values: NDArray[Any], coordinates: NDArray[Any]
    ) -> dict[str, float]:
        """Calculate Geary's C spatial autocorrelation statistic."""
        n = len(values)

        # Create spatial weights matrix
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(coordinates))
        weights = 1 / (distances + 1e-8)
        np.fill_diagonal(weights, 0)

        # Calculate Geary's C
        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - values[j]) ** 2

        mean_val = np.mean(values)
        for i in range(n):
            denominator += (values[i] - mean_val) ** 2

        W = np.sum(weights)
        geary_c = (
            ((n - 1) / (2 * W)) * (numerator / denominator) if denominator > 0 else 1
        )

        # Approximate p-value
        expected = 1.0
        variance = 1.0 / n  # Simplified
        z_score = (geary_c - expected) / np.sqrt(variance)
        pvalue = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {"statistic": float(geary_c), "pvalue": float(pvalue)}

    def _spillover_power_analysis(
        self, treatment: TreatmentData, outcome: OutcomeData
    ) -> dict[str, float]:
        """Estimate power to detect spillover effects."""
        n_units = len(treatment.values)

        # Simple power calculation based on sample size and network density
        network_density = self.exposure_mapping.total_exposure_per_unit.mean()

        # Heuristic power calculation
        base_power = min(0.8, n_units / 100)  # More units = more power
        network_power = min(0.2, network_density / 5)  # More connections = more power
        estimated_power = base_power + network_power

        # Minimum detectable effect (Cohen's d)
        # Simplified calculation based on sample size
        mde = 2.8 / np.sqrt(n_units) if n_units > 0 else 1.0

        return {"power": min(1.0, estimated_power), "mde": mde}

    def _detect_spillover_presence(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        diagnostics: SpilloverDetectionResults,
    ) -> dict[str, Any]:
        """Detect presence of spillover effects using multiple indicators."""
        evidence_count = 0
        total_tests = 0

        # Check exposure imbalance
        total_tests += 1
        if diagnostics.exposure_balance_pvalue < 0.1:
            evidence_count += 1

        # Check spatial autocorrelation
        if diagnostics.moran_i_pvalue is not None:
            total_tests += 1
            if diagnostics.moran_i_pvalue < 0.1:
                evidence_count += 1

        if diagnostics.geary_c_pvalue is not None:
            total_tests += 1
            if diagnostics.geary_c_pvalue < 0.1:
                evidence_count += 1

        # Check network properties
        total_tests += 1
        if diagnostics.network_density > 0.1:
            evidence_count += 1

        # Overall detection
        confidence = evidence_count / total_tests if total_tests > 0 else 0
        detected = confidence > 0.5

        return {"detected": detected, "confidence": confidence}

    def _generate_warnings(self, results: SpilloverDetectionResults) -> list[str]:
        """Generate warnings based on diagnostic results."""
        warnings = []

        if results.network_density < 0.01:
            warnings.append(
                "Very low network density may limit spillover detection power"
            )

        if results.exposure_imbalance > 1.0:
            warnings.append("High exposure imbalance between treatment groups detected")

        if results.estimated_power is not None and results.estimated_power < 0.6:
            warnings.append("Low statistical power to detect spillover effects")

        if results.max_exposure_ratio > 5.0:
            warnings.append("Extreme exposure ratio suggests potential confounding")

        return warnings

    def _generate_recommendations(
        self, results: SpilloverDetectionResults
    ) -> list[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []

        if results.spillover_detected:
            recommendations.append(
                "Spillover effects detected - use interference-aware methods"
            )
            recommendations.append(
                "Consider cluster randomization for future experiments"
            )

        if results.network_density > 0.3:
            recommendations.append(
                "High network connectivity - use network-based inference"
            )

        if results.exposure_imbalance > 0.5:
            recommendations.append(
                "Consider stratified randomization by exposure level"
            )

        if results.estimated_power is not None and results.estimated_power < 0.8:
            recommendations.append("Increase sample size or refine network structure")

        return recommendations


def plot_cluster_exposure_balance(
    exposure_mapping: ExposureMapping,
    treatment: TreatmentData,
    clusters: NDArray[Any] | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot exposure balance across clusters.

    Args:
        exposure_mapping: Network exposure relationships
        treatment: Treatment assignment data
        clusters: Cluster assignments (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Calculate spillover exposure
    treatment_array = np.array(treatment.values)
    spillover_exposure = np.dot(exposure_mapping.exposure_matrix, treatment_array)

    # Plot 1: Exposure distribution by treatment status
    treated_mask = treatment_array == 1
    control_mask = treatment_array == 0

    ax1.hist(spillover_exposure[treated_mask], alpha=0.7, label="Treated", bins=20)
    ax1.hist(spillover_exposure[control_mask], alpha=0.7, label="Control", bins=20)
    ax1.set_xlabel("Spillover Exposure")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Spillover Exposure Distribution")
    ax1.legend()

    # Plot 2: Exposure balance by cluster (if available)
    if clusters is not None:
        cluster_data = []
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            cluster_treatment = np.mean(treatment_array[mask])
            cluster_exposure = np.mean(spillover_exposure[mask])
            cluster_data.append(
                {
                    "cluster": cluster,
                    "treatment_rate": cluster_treatment,
                    "avg_exposure": cluster_exposure,
                }
            )

        cluster_df = pd.DataFrame(cluster_data)
        ax2.scatter(cluster_df["treatment_rate"], cluster_df["avg_exposure"])
        ax2.set_xlabel("Cluster Treatment Rate")
        ax2.set_ylabel("Average Spillover Exposure")
        ax2.set_title("Cluster-Level Exposure Balance")
    else:
        ax2.scatter(treatment_array, spillover_exposure, alpha=0.6)
        ax2.set_xlabel("Treatment Assignment")
        ax2.set_ylabel("Spillover Exposure")
        ax2.set_title("Individual Exposure vs Treatment")

    plt.tight_layout()
    return fig


def plot_network_connectivity(
    exposure_mapping: ExposureMapping,
    treatment: TreatmentData | None = None,
    layout: str = "spring",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot network connectivity and treatment assignments.

    Args:
        exposure_mapping: Network exposure relationships
        treatment: Optional treatment assignments for coloring
        layout: Network layout algorithm
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    try:
        import networkx as nx
    except ImportError:
        # Fallback visualization without networkx
        return _plot_adjacency_matrix(exposure_mapping, treatment, figsize)

    # Create network graph
    G = nx.from_numpy_array(exposure_mapping.exposure_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=42)

    # Node colors based on treatment if available
    if treatment is not None:
        treatment_array = np.array(treatment.values)
        node_colors = ["red" if t == 1 else "blue" for t in treatment_array]
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Treated",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="Control",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    else:
        node_colors = "lightblue"

    # Draw network
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=100,
        edge_color="gray",
        alpha=0.7,
        with_labels=False,
    )

    ax.set_title("Network Connectivity")
    return fig


def _plot_adjacency_matrix(
    exposure_mapping: ExposureMapping,
    treatment: TreatmentData | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Fallback plot using adjacency matrix heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    # Plot adjacency matrix
    sns.heatmap(
        exposure_mapping.exposure_matrix,
        ax=ax,
        cmap="Blues",
        cbar_kws={"label": "Exposure Strength"},
    )

    ax.set_title("Network Adjacency Matrix")
    ax.set_xlabel("Source Unit")
    ax.set_ylabel("Target Unit")

    return fig


def plot_spillover_detection_power(
    power_analysis_results: pd.DataFrame, figsize: tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot power analysis results for spillover detection.

    Args:
        power_analysis_results: DataFrame with power analysis results
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Power vs Effect Size
    for sample_size in power_analysis_results["sample_size"].unique():
        subset = power_analysis_results[
            power_analysis_results["sample_size"] == sample_size
        ]
        ax1.plot(
            subset["effect_size"], subset["power"], marker="o", label=f"N={sample_size}"
        )

    ax1.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, label="80% Power")
    ax1.set_xlabel("Spillover Effect Size")
    ax1.set_ylabel("Statistical Power")
    ax1.set_title("Power vs Effect Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Power vs Sample Size
    for effect_size in power_analysis_results["effect_size"].unique():
        subset = power_analysis_results[
            power_analysis_results["effect_size"] == effect_size
        ]
        ax2.plot(
            subset["sample_size"],
            subset["power"],
            marker="o",
            label=f"Effect={effect_size}",
        )

    ax2.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, label="80% Power")
    ax2.set_xlabel("Sample Size")
    ax2.set_ylabel("Statistical Power")
    ax2.set_title("Power vs Sample Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
