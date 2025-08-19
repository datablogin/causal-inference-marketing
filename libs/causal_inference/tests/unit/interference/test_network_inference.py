"""Unit tests for network-based inference methods."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import OutcomeData, TreatmentData
from causal_inference.interference.exposure_mapping import ExposureMapping
from causal_inference.interference.network_inference import (
    ClusterRandomizationInference,
    InferenceResults,
    NetworkPermutationTest,
    TwoStageRandomizationInference,
    conduct_interference_power_analysis,
)


class TestInferenceResults:
    """Test inference results data structure."""

    def test_inference_results_initialization(self):
        """Test basic inference results initialization."""
        results = InferenceResults(
            test_statistic=2.5, p_value=0.03, estimated_effect=0.4, method="test_method"
        )

        assert results.test_statistic == 2.5
        assert results.p_value == 0.03
        assert results.estimated_effect == 0.4
        assert results.method == "test_method"
        assert results.is_significant  # p_value < 0.05

    def test_inference_results_not_significant(self):
        """Test non-significant results."""
        results = InferenceResults(
            test_statistic=1.0, p_value=0.15, estimated_effect=0.1, method="test_method"
        )

        assert not results.is_significant


class TestTwoStageRandomizationInference:
    """Test two-stage randomization inference."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_units = 60

        # Create exposure mapping with cluster structure
        self.unit_ids = np.arange(n_units)
        exposure_matrix = np.zeros((n_units, n_units))

        # Create 3 clusters of 20 units each
        for cluster in range(3):
            start_idx = cluster * 20
            end_idx = (cluster + 1) * 20

            # Within-cluster connections
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:
                        exposure_matrix[i, j] = 1.0

        self.exposure_mapping = ExposureMapping(
            unit_ids=self.unit_ids,
            exposure_matrix=exposure_matrix,
            exposure_type="binary",
        )

        # Create cluster assignments
        self.clusters = np.repeat([0, 1, 2], 20)

        # Generate treatment with cluster-level randomization
        cluster_treatments = np.random.binomial(1, 0.5, 3)
        self.treatment = TreatmentData(
            values=np.repeat(cluster_treatments, 20), treatment_type="binary"
        )

        # Generate outcome with cluster effects
        cluster_effects = np.array([0.2, 0.5, 0.8])  # Different cluster baselines
        treatment_effect = 0.3

        outcome_values = (
            np.repeat(cluster_effects, 20)
            + treatment_effect * self.treatment.values
            + np.random.normal(0, 0.2, n_units)
        )

        self.outcome = OutcomeData(values=outcome_values)

        # Unit data with cluster information
        self.unit_data = pd.DataFrame(
            {
                "unit_id": self.unit_ids,
                "cluster": self.clusters,
                "latitude": np.random.uniform(0, 10, n_units),
                "longitude": np.random.uniform(0, 10, n_units),
            }
        )

    def test_two_stage_inference_initialization(self):
        """Test initialization of two-stage inference."""
        inference = TwoStageRandomizationInference(
            exposure_mapping=self.exposure_mapping,
            cluster_column="cluster",
            alpha=0.05,
            random_state=42,
        )

        assert inference.cluster_column == "cluster"
        assert inference.alpha == 0.05
        assert inference.random_state == 42

    def test_two_stage_inference_with_predefined_clusters(self):
        """Test two-stage inference with predefined clusters."""
        inference = TwoStageRandomizationInference(
            exposure_mapping=self.exposure_mapping,
            cluster_column="cluster",
            random_state=42,
        )

        results = inference.test_treatment_effect(
            treatment=self.treatment, outcome=self.outcome, unit_data=self.unit_data
        )

        assert results.method == "two_stage_randomization"
        assert results.n_clusters == 3
        assert results.randomization_scheme == "cluster"
        assert results.interference_structure == "within_cluster"
        assert results.test_statistic is not None
        assert results.p_value is not None
        assert results.estimated_effect is not None

        # Should have reasonable effect estimate (cluster-level analysis may amplify effects)
        assert 0.0 < results.estimated_effect < 1.2

    def test_two_stage_inference_auto_clustering(self):
        """Test two-stage inference with automatic clustering."""
        inference = TwoStageRandomizationInference(
            exposure_mapping=self.exposure_mapping, n_clusters=3, random_state=42
        )

        results = inference.test_treatment_effect(
            treatment=self.treatment, outcome=self.outcome, unit_data=self.unit_data
        )

        assert results.n_clusters == 3
        assert results.test_statistic is not None

    def test_two_stage_inference_assumptions(self):
        """Test assumption checking in two-stage inference."""
        inference = TwoStageRandomizationInference(
            exposure_mapping=self.exposure_mapping,
            cluster_column="cluster",
            random_state=42,
        )

        results = inference.test_treatment_effect(
            treatment=self.treatment, outcome=self.outcome, unit_data=self.unit_data
        )

        assert results.assumptions_met is not None
        assert "balanced_clusters" in results.assumptions_met
        assert "within_cluster_variation" in results.assumptions_met
        assert "sufficient_clusters" in results.assumptions_met

    def test_two_stage_inference_no_variation(self):
        """Test error handling when no treatment variation."""
        # Create treatment with no variation
        no_variation_treatment = TreatmentData(
            values=np.ones(60),  # All treated
            treatment_type="binary",
        )

        inference = TwoStageRandomizationInference(
            exposure_mapping=self.exposure_mapping,
            cluster_column="cluster",
            random_state=42,
        )

        with pytest.raises(ValueError, match="Must have both treated and control"):
            inference.test_treatment_effect(
                treatment=no_variation_treatment,
                outcome=self.outcome,
                unit_data=self.unit_data,
            )


class TestClusterRandomizationInference:
    """Test cluster randomization inference."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_units = 40

        # Simple exposure mapping
        self.exposure_mapping = ExposureMapping(
            unit_ids=np.arange(n_units),
            exposure_matrix=np.eye(n_units),  # Simple identity for testing
            exposure_type="binary",
        )

        # Create clusters (4 clusters of 10 units each)
        self.clusters = np.repeat([0, 1, 2, 3], 10)

        # Cluster-level treatment assignment
        cluster_treatments = [1, 0, 1, 0]  # Alternating treatment
        self.treatment = TreatmentData(
            values=np.repeat(cluster_treatments, 10), treatment_type="binary"
        )

        # Generate outcomes with cluster effects
        cluster_baselines = [0.1, 0.2, 0.15, 0.25]
        treatment_effect = 0.4

        outcome_values = (
            np.repeat(cluster_baselines, 10)
            + treatment_effect * self.treatment.values
            + np.random.normal(0, 0.1, n_units)
        )

        self.outcome = OutcomeData(values=outcome_values)

    def test_cluster_inference_basic(self):
        """Test basic cluster randomization inference."""
        inference = ClusterRandomizationInference(
            exposure_mapping=self.exposure_mapping, alpha=0.05, random_state=42
        )

        results = inference.test_treatment_effect(
            treatment=self.treatment, outcome=self.outcome, clusters=self.clusters
        )

        assert results.method == "cluster_randomization"
        assert results.n_clusters == 4
        assert results.randomization_scheme == "cluster"
        assert results.interference_structure == "within_cluster"

        # Should detect the treatment effect (around 0.4)
        assert 0.2 < results.estimated_effect < 0.6

    def test_cluster_inference_missing_clusters(self):
        """Test error handling when clusters not provided."""
        inference = ClusterRandomizationInference(
            exposure_mapping=self.exposure_mapping, random_state=42
        )

        with pytest.raises(ValueError, match="Cluster assignments required"):
            inference.test_treatment_effect(
                treatment=self.treatment, outcome=self.outcome
            )

    def test_cluster_inference_no_variation(self):
        """Test error handling with no cluster treatment variation."""
        # All clusters treated
        all_treated_treatment = TreatmentData(
            values=np.ones(40), treatment_type="binary"
        )

        inference = ClusterRandomizationInference(
            exposure_mapping=self.exposure_mapping, random_state=42
        )

        with pytest.raises(ValueError, match="Must have both treated and control"):
            inference.test_treatment_effect(
                treatment=all_treated_treatment,
                outcome=self.outcome,
                clusters=self.clusters,
            )


class TestNetworkPermutationTest:
    """Test network-based permutation tests."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_units = 50

        # Create network structure
        exposure_matrix = np.zeros((n_units, n_units))

        # Create random network connections
        for i in range(n_units):
            # Connect each unit to 2-3 random others
            n_connections = np.random.choice([2, 3])
            connections = np.random.choice(
                [j for j in range(n_units) if j != i], size=n_connections, replace=False
            )
            for j in connections:
                exposure_matrix[i, j] = 1.0

        self.exposure_mapping = ExposureMapping(
            unit_ids=np.arange(n_units),
            exposure_matrix=exposure_matrix,
            exposure_type="binary",
        )

        # Generate treatment
        self.treatment = TreatmentData(
            values=np.random.binomial(1, 0.5, n_units), treatment_type="binary"
        )

        # Generate outcome with treatment effect
        treatment_effect = 0.3
        self.outcome = OutcomeData(
            values=(
                treatment_effect * self.treatment.values
                + np.random.normal(0, 0.5, n_units)
            )
        )

    def test_network_permutation_initialization(self):
        """Test initialization of network permutation test."""
        test = NetworkPermutationTest(
            exposure_mapping=self.exposure_mapping,
            n_permutations=100,
            test_statistic="difference_in_means",
            random_state=42,
        )

        assert test.n_permutations == 100
        assert test.test_statistic == "difference_in_means"
        assert test.random_state == 42

    def test_network_permutation_test_basic(self):
        """Test basic network permutation test."""
        test = NetworkPermutationTest(
            exposure_mapping=self.exposure_mapping,
            n_permutations=100,  # Small number for testing speed
            random_state=42,
        )

        results = test.test_treatment_effect(
            treatment=self.treatment, outcome=self.outcome
        )

        assert results.method == "network_permutation"
        assert results.n_permutations == 100
        assert results.randomization_scheme == "network_aware"
        assert results.interference_structure == "network"
        assert results.test_statistic is not None
        assert results.p_value is not None
        assert 0 <= results.p_value <= 1
        assert results.critical_value is not None

    def test_network_permutation_calculate_test_statistic(self):
        """Test test statistic calculation."""
        test = NetworkPermutationTest(
            exposure_mapping=self.exposure_mapping,
            test_statistic="difference_in_means",
            random_state=42,
        )

        treatment_array = np.array([1, 0, 1, 0, 1])
        outcome_array = np.array([2.0, 1.0, 2.5, 1.2, 2.1])

        statistic = test._calculate_test_statistic(treatment_array, outcome_array)

        # Should be difference between treated and control means
        treated_mean = np.mean([2.0, 2.5, 2.1])
        control_mean = np.mean([1.0, 1.2])
        expected = treated_mean - control_mean

        assert abs(statistic - expected) < 0.01

    def test_network_permutation_unknown_statistic(self):
        """Test error handling for unknown test statistic."""
        test = NetworkPermutationTest(
            exposure_mapping=self.exposure_mapping,
            test_statistic="unknown_statistic",
            random_state=42,
        )

        with pytest.raises(ValueError, match="Unknown test statistic"):
            test._calculate_test_statistic(np.array([1, 0]), np.array([2.0, 1.0]))

    def test_network_permutation_no_treatment_variation(self):
        """Test permutation test with no treatment variation."""
        # All units treated
        no_variation_treatment = TreatmentData(
            values=np.ones(50), treatment_type="binary"
        )

        test = NetworkPermutationTest(
            exposure_mapping=self.exposure_mapping, n_permutations=10, random_state=42
        )

        results = test.test_treatment_effect(
            treatment=no_variation_treatment, outcome=self.outcome
        )

        # Should return test statistic of 0 and p-value of 1
        assert results.test_statistic == 0.0
        assert results.estimated_effect == 0.0


class TestInterferencePowerAnalysis:
    """Test power analysis for interference detection."""

    def setup_method(self):
        """Set up test data."""
        n_units = 30

        # Simple network for power analysis
        exposure_matrix = np.zeros((n_units, n_units))
        for i in range(n_units - 1):
            exposure_matrix[i, i + 1] = 1.0
            exposure_matrix[i + 1, i] = 1.0

        self.exposure_mapping = ExposureMapping(
            unit_ids=np.arange(n_units),
            exposure_matrix=exposure_matrix,
            exposure_type="binary",
        )

    def test_power_analysis_basic(self):
        """Test basic power analysis functionality."""
        effect_sizes = [0.1, 0.3, 0.5]
        sample_sizes = [20, 50, 100]

        power_results = conduct_interference_power_analysis(
            exposure_mapping=self.exposure_mapping,
            effect_sizes=effect_sizes,
            sample_sizes=sample_sizes,
            n_simulations=50,  # Small number for testing speed
            random_state=42,
        )

        assert len(power_results) == len(effect_sizes) * len(sample_sizes)
        assert "effect_size" in power_results.columns
        assert "sample_size" in power_results.columns
        assert "power" in power_results.columns
        assert "alpha" in power_results.columns

        # Power should generally increase with effect size and sample size
        power_values = power_results["power"].values
        assert all(0 <= p <= 1 for p in power_values)

    def test_power_analysis_increasing_power(self):
        """Test that power increases with effect size."""
        effect_sizes = [0.1, 0.5]
        sample_sizes = [50]

        power_results = conduct_interference_power_analysis(
            exposure_mapping=self.exposure_mapping,
            effect_sizes=effect_sizes,
            sample_sizes=sample_sizes,
            n_simulations=100,
            random_state=42,
        )

        small_effect_power = power_results[power_results["effect_size"] == 0.1][
            "power"
        ].iloc[0]

        large_effect_power = power_results[power_results["effect_size"] == 0.5][
            "power"
        ].iloc[0]

        # Larger effect should have higher power
        assert large_effect_power >= small_effect_power
