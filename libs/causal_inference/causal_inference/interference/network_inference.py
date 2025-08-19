"""Network-based Inference for Interference Effects.

This module provides inference methods for causal effects under interference,
including two-stage randomization, cluster randomization, and network-based
permutation tests that account for spillover effects.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.cluster import KMeans

from ..core.base import CovariateData, OutcomeData, TreatmentData
from .exposure_mapping import ExposureMapping


@dataclass
class InferenceResults:
    """Results from interference-aware inference procedures."""

    # Test statistics - required fields first
    test_statistic: float
    p_value: float
    estimated_effect: float

    # Optional fields with defaults
    critical_value: float | None = None
    effect_se: float | None = None
    effect_ci_lower: float | None = None
    effect_ci_upper: float | None = None

    # Inference method details
    method: str = "unknown"
    n_permutations: int | None = None
    n_clusters: int | None = None
    alpha: float = 0.05

    # Randomization details
    randomization_scheme: str = "individual"
    interference_structure: str = "network"

    # Diagnostics
    assumptions_met: dict[str, bool] = None
    warnings: list[str] = None

    def __post_init__(self) -> None:
        """Initialize empty containers if None."""
        if self.assumptions_met is None:
            self.assumptions_met = {}
        if self.warnings is None:
            self.warnings = []

    @property
    def is_significant(self) -> bool:
        """Check if the effect is statistically significant."""
        return self.p_value < self.alpha


class NetworkInference(abc.ABC):
    """Abstract base class for network-based inference methods."""

    def __init__(
        self,
        exposure_mapping: ExposureMapping,
        alpha: float = 0.05,
        random_state: int | None = None,
    ):
        """Initialize network inference.

        Args:
            exposure_mapping: Network exposure relationships
            alpha: Significance level
            random_state: Random seed for reproducible results
        """
        self.exposure_mapping = exposure_mapping
        self.alpha = alpha
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    @abc.abstractmethod
    def test_treatment_effect(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        **kwargs: Any,
    ) -> InferenceResults:
        """Test for treatment effects under interference.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            covariates: Optional covariate data
            **kwargs: Additional method-specific parameters

        Returns:
            Inference results with test statistics and p-values
        """
        pass


class TwoStageRandomizationInference(NetworkInference):
    """Two-stage randomization inference for cluster-based interference.

    First stage: randomize clusters
    Second stage: randomize individuals within clusters
    """

    def __init__(
        self,
        exposure_mapping: ExposureMapping,
        cluster_column: str | None = None,
        n_clusters: int | None = None,
        alpha: float = 0.05,
        random_state: int | None = None,
    ):
        """Initialize two-stage randomization inference.

        Args:
            exposure_mapping: Network exposure relationships
            cluster_column: Column name for predefined clusters
            n_clusters: Number of clusters to create if not predefined
            alpha: Significance level
            random_state: Random seed
        """
        super().__init__(exposure_mapping, alpha, random_state)
        self.cluster_column = cluster_column
        self.n_clusters = n_clusters
        self._clusters: NDArray[Any] | None = None

    def test_treatment_effect(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        unit_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> InferenceResults:
        """Test treatment effect using two-stage randomization.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            covariates: Optional covariate data
            unit_data: DataFrame with unit information and cluster assignments
            **kwargs: Additional parameters

        Returns:
            Two-stage randomization inference results
        """
        # Get or create clusters
        if unit_data is not None and self.cluster_column in unit_data.columns:
            self._clusters = unit_data[self.cluster_column].values
        else:
            self._clusters = self._create_clusters(
                treatment, outcome, covariates, unit_data
            )

        n_clusters = len(np.unique(self._clusters))

        # Calculate cluster-level treatment and outcome averages
        cluster_treatment = self._calculate_cluster_averages(
            np.array(treatment.values), self._clusters
        )
        cluster_outcome = self._calculate_cluster_averages(
            np.array(outcome.values), self._clusters
        )

        # Test cluster-level effect
        cluster_effect = np.mean(cluster_outcome[cluster_treatment > 0.5]) - np.mean(
            cluster_outcome[cluster_treatment <= 0.5]
        )

        # Calculate standard error using cluster-robust approach
        treated_clusters = cluster_treatment > 0.5
        control_clusters = cluster_treatment <= 0.5

        if np.sum(treated_clusters) == 0 or np.sum(control_clusters) == 0:
            raise ValueError("Must have both treated and control clusters")

        var_treated = np.var(cluster_outcome[treated_clusters])
        var_control = np.var(cluster_outcome[control_clusters])
        n_treated = np.sum(treated_clusters)
        n_control = np.sum(control_clusters)

        se_effect = np.sqrt(var_treated / n_treated + var_control / n_control)

        # Two-sided t-test
        t_statistic = cluster_effect / se_effect if se_effect > 0 else 0
        df = n_clusters - 2
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))

        # Confidence interval
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = cluster_effect - t_critical * se_effect
        ci_upper = cluster_effect + t_critical * se_effect

        # Check assumptions
        assumptions = self._check_two_stage_assumptions(
            treatment, outcome, self._clusters
        )

        return InferenceResults(
            test_statistic=t_statistic,
            p_value=p_value,
            critical_value=t_critical,
            estimated_effect=cluster_effect,
            effect_se=se_effect,
            effect_ci_lower=ci_lower,
            effect_ci_upper=ci_upper,
            method="two_stage_randomization",
            n_clusters=n_clusters,
            alpha=self.alpha,
            randomization_scheme="cluster",
            interference_structure="within_cluster",
            assumptions_met=assumptions,
        )

    def _create_clusters(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        unit_data: pd.DataFrame | None = None,
    ) -> NDArray[Any]:
        """Create clusters based on network structure or covariates."""
        n_units = len(treatment.values)

        if self.n_clusters is None:
            # Default to square root of sample size
            self.n_clusters = max(2, int(np.sqrt(n_units)))

        # Use network structure for clustering
        adjacency_matrix = self.exposure_mapping.exposure_matrix

        # Spectral clustering based on network structure
        try:
            from sklearn.cluster import SpectralClustering

            clustering = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="precomputed",
                random_state=self.random_state,
            )
            clusters = clustering.fit_predict(adjacency_matrix)

        except ImportError:
            # Fallback to k-means on adjacency matrix
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            clusters = kmeans.fit_predict(adjacency_matrix)

        return clusters

    def _calculate_cluster_averages(
        self, values: NDArray[Any], clusters: NDArray[Any]
    ) -> NDArray[Any]:
        """Calculate average values within each cluster."""
        unique_clusters = np.unique(clusters)
        cluster_averages = np.zeros(len(unique_clusters))

        for i, cluster in enumerate(unique_clusters):
            cluster_mask = clusters == cluster
            cluster_averages[i] = np.mean(values[cluster_mask])

        return cluster_averages

    def _check_two_stage_assumptions(
        self, treatment: TreatmentData, outcome: OutcomeData, clusters: NDArray[Any]
    ) -> dict[str, bool]:
        """Check assumptions for two-stage randomization."""
        assumptions = {}

        # Check cluster balance
        unique_clusters = np.unique(clusters)
        cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
        assumptions["balanced_clusters"] = (
            np.max(cluster_sizes) / np.min(cluster_sizes) < 3
        )

        # Check treatment variation within clusters
        treatment_variation = True
        for cluster in unique_clusters:
            cluster_treatment = np.array(treatment.values)[clusters == cluster]
            if len(np.unique(cluster_treatment)) < 2:
                treatment_variation = False
                break
        assumptions["within_cluster_variation"] = treatment_variation

        # Check for sufficient clusters
        assumptions["sufficient_clusters"] = len(unique_clusters) >= 6

        return assumptions


class ClusterRandomizationInference(NetworkInference):
    """Cluster randomization inference with interference within clusters."""

    def test_treatment_effect(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        clusters: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> InferenceResults:
        """Test treatment effect under cluster randomization.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            covariates: Optional covariate data
            clusters: Cluster assignments for each unit
            **kwargs: Additional parameters

        Returns:
            Cluster randomization inference results
        """
        if clusters is None:
            raise ValueError("Cluster assignments required for cluster randomization")

        # Calculate cluster-level statistics
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)

        cluster_means = []
        cluster_treatments = []
        cluster_sizes = []

        for cluster in unique_clusters:
            mask = clusters == cluster
            cluster_outcome = np.mean(np.array(outcome.values)[mask])
            cluster_treatment = np.mean(np.array(treatment.values)[mask])
            cluster_size = np.sum(mask)

            cluster_means.append(cluster_outcome)
            cluster_treatments.append(cluster_treatment)
            cluster_sizes.append(cluster_size)

        cluster_means = np.array(cluster_means)
        cluster_treatments = np.array(cluster_treatments)
        cluster_sizes = np.array(cluster_sizes)

        # Classify clusters as treated/control based on majority treatment
        treated_clusters = cluster_treatments > 0.5

        if np.sum(treated_clusters) == 0 or np.sum(~treated_clusters) == 0:
            raise ValueError("Must have both treated and control clusters")

        # Calculate cluster-level treatment effect
        treated_mean = np.average(
            cluster_means[treated_clusters], weights=cluster_sizes[treated_clusters]
        )
        control_mean = np.average(
            cluster_means[~treated_clusters], weights=cluster_sizes[~treated_clusters]
        )

        cluster_effect = treated_mean - control_mean

        # Calculate cluster-robust standard error
        treated_var = np.average(
            (cluster_means[treated_clusters] - treated_mean) ** 2,
            weights=cluster_sizes[treated_clusters],
        )
        control_var = np.average(
            (cluster_means[~treated_clusters] - control_mean) ** 2,
            weights=cluster_sizes[~treated_clusters],
        )

        n_treated_clusters = np.sum(treated_clusters)
        n_control_clusters = np.sum(~treated_clusters)

        se_effect = np.sqrt(
            treated_var / n_treated_clusters + control_var / n_control_clusters
        )

        # Test statistic and p-value
        t_statistic = cluster_effect / se_effect if se_effect > 0 else 0
        df = n_clusters - 2
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))

        # Confidence interval
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = cluster_effect - t_critical * se_effect
        ci_upper = cluster_effect + t_critical * se_effect

        return InferenceResults(
            test_statistic=t_statistic,
            p_value=p_value,
            critical_value=t_critical,
            estimated_effect=cluster_effect,
            effect_se=se_effect,
            effect_ci_lower=ci_lower,
            effect_ci_upper=ci_upper,
            method="cluster_randomization",
            n_clusters=n_clusters,
            alpha=self.alpha,
            randomization_scheme="cluster",
            interference_structure="within_cluster",
        )


class NetworkPermutationTest(NetworkInference):
    """Permutation test that respects network structure for interference."""

    def __init__(
        self,
        exposure_mapping: ExposureMapping,
        n_permutations: int = 1000,
        test_statistic: str = "difference_in_means",
        alpha: float = 0.05,
        random_state: int | None = None,
    ):
        """Initialize network permutation test.

        Args:
            exposure_mapping: Network exposure relationships
            n_permutations: Number of permutation samples
            test_statistic: Type of test statistic to use
            alpha: Significance level
            random_state: Random seed
        """
        super().__init__(exposure_mapping, alpha, random_state)
        self.n_permutations = n_permutations
        self.test_statistic = test_statistic

    def test_treatment_effect(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
        **kwargs: Any,
    ) -> InferenceResults:
        """Test treatment effect using network-aware permutation test.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome data
            covariates: Optional covariate data
            **kwargs: Additional parameters

        Returns:
            Network permutation test results
        """
        treatment_array = np.array(treatment.values)
        outcome_array = np.array(outcome.values)

        # Calculate observed test statistic
        observed_statistic = self._calculate_test_statistic(
            treatment_array, outcome_array
        )

        # Generate permutation distribution
        permutation_statistics = []

        for _ in range(self.n_permutations):
            # Network-aware permutation
            permuted_treatment = self._network_aware_permutation(treatment_array)

            # Calculate test statistic for permuted data
            perm_statistic = self._calculate_test_statistic(
                permuted_treatment, outcome_array
            )
            permutation_statistics.append(perm_statistic)

        permutation_statistics = np.array(permutation_statistics)

        # Calculate p-value
        if self.test_statistic == "difference_in_means":
            # Two-sided test
            p_value = np.mean(
                np.abs(permutation_statistics) >= np.abs(observed_statistic)
            )
        else:
            # One-sided test (assuming larger is more extreme)
            p_value = np.mean(permutation_statistics >= observed_statistic)

        # Critical value from permutation distribution
        if self.test_statistic == "difference_in_means":
            critical_value = np.percentile(
                np.abs(permutation_statistics), 100 * (1 - self.alpha)
            )
        else:
            critical_value = np.percentile(
                permutation_statistics, 100 * (1 - self.alpha)
            )

        # Estimate treatment effect (for difference in means)
        if self.test_statistic == "difference_in_means":
            estimated_effect = observed_statistic
        else:
            # For other statistics, estimate using simple difference
            treated_mask = treatment_array == 1
            control_mask = treatment_array == 0
            estimated_effect = np.mean(outcome_array[treated_mask]) - np.mean(
                outcome_array[control_mask]
            )

        return InferenceResults(
            test_statistic=observed_statistic,
            p_value=p_value,
            critical_value=critical_value,
            estimated_effect=estimated_effect,
            method="network_permutation",
            n_permutations=self.n_permutations,
            alpha=self.alpha,
            randomization_scheme="network_aware",
            interference_structure="network",
        )

    def _calculate_test_statistic(
        self, treatment: NDArray[Any], outcome: NDArray[Any]
    ) -> float:
        """Calculate test statistic for given treatment and outcome."""
        if self.test_statistic == "difference_in_means":
            treated_mask = treatment == 1
            control_mask = treatment == 0

            if np.sum(treated_mask) == 0 or np.sum(control_mask) == 0:
                return 0.0

            treated_mean = np.mean(outcome[treated_mask])
            control_mean = np.mean(outcome[control_mask])
            return treated_mean - control_mean

        else:
            raise ValueError(f"Unknown test statistic: {self.test_statistic}")

    def _network_aware_permutation(self, treatment: NDArray[Any]) -> NDArray[Any]:
        """Generate network-aware permutation of treatment assignments.

        This preserves the network structure while permuting treatment assignments.
        """
        n_units = len(treatment)
        n_treated = int(np.sum(treatment))

        # Simple random permutation (could be enhanced with network constraints)
        permuted_indices = np.random.permutation(n_units)
        permuted_treatment = np.zeros(n_units)
        permuted_treatment[permuted_indices[:n_treated]] = 1

        return permuted_treatment

    def _block_permutation(self, treatment: NDArray[Any]) -> NDArray[Any]:
        """Generate block-based permutation respecting network communities."""
        # This is a placeholder for more sophisticated network-aware permutation
        # Could use network community detection to create blocks
        return self._network_aware_permutation(treatment)


def conduct_interference_power_analysis(
    exposure_mapping: ExposureMapping,
    effect_sizes: list[float],
    sample_sizes: list[int],
    n_simulations: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Conduct power analysis for interference detection.

    Args:
        exposure_mapping: Network exposure structure
        effect_sizes: List of spillover effect sizes to test
        sample_sizes: List of sample sizes to test
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level
        random_state: Random seed

    Returns:
        DataFrame with power analysis results
    """
    if random_state is not None:
        np.random.seed(random_state)

    results = []

    for effect_size in effect_sizes:
        for sample_size in sample_sizes:
            power = _estimate_power_for_effect(
                exposure_mapping, effect_size, sample_size, n_simulations, alpha
            )

            results.append(
                {
                    "effect_size": effect_size,
                    "sample_size": sample_size,
                    "power": power,
                    "alpha": alpha,
                }
            )

    return pd.DataFrame(results)


def _estimate_power_for_effect(
    exposure_mapping: ExposureMapping,
    effect_size: float,
    sample_size: int,
    n_simulations: int,
    alpha: float,
) -> float:
    """Estimate power for detecting a specific spillover effect."""
    significant_tests = 0

    for _ in range(n_simulations):
        # Generate synthetic data with known spillover effect
        treatment = np.random.binomial(1, 0.5, sample_size)
        spillover_exposure = np.dot(exposure_mapping.exposure_matrix, treatment)

        # Outcome with direct effect + spillover effect + noise
        outcome = (
            0.5 * treatment  # Direct effect
            + effect_size * spillover_exposure  # Spillover effect
            + np.random.normal(0, 1, sample_size)  # Noise
        )

        # Test for spillover effect using simple regression
        try:
            from sklearn.linear_model import LinearRegression

            X = np.column_stack([treatment, spillover_exposure])
            model = LinearRegression().fit(X, outcome)

            # Simple t-test for spillover coefficient (approximation)
            spillover_coef = model.coef_[1]

            # Rough approximation of standard error
            residuals = outcome - model.predict(X)
            mse = np.mean(residuals**2)

            # Simplified standard error calculation
            se_spillover = np.sqrt(mse / np.var(spillover_exposure))

            t_stat = spillover_coef / se_spillover if se_spillover > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), sample_size - 2))

            if p_value < alpha:
                significant_tests += 1

        except Exception:
            # Skip this simulation if estimation fails
            continue

    return significant_tests / n_simulations
