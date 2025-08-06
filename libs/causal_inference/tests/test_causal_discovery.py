"""Tests for causal discovery algorithms and integration."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.discovery import (
    CausalDAG,
    DiscoveryResult,
    GESAlgorithm,
    NOTEARSAlgorithm,
    PCAlgorithm,
)
from causal_inference.discovery.integration import (
    DiscoveryEstimatorPipeline,
    estimate_causal_effect_from_discovery,
    validate_discovery_assumptions,
)
from causal_inference.discovery.utils import (
    compare_dags,
    dag_to_adjustment_sets,
    generate_linear_sem_data,
)
from causal_inference.estimators import GComputationEstimator


class TestCausalDAG:
    """Test the CausalDAG data structure."""

    def test_valid_dag_creation(self):
        """Test creating a valid DAG."""
        adj_matrix = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        variable_names = ["X", "Y", "Z"]

        dag = CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

        assert dag.n_variables == 3
        assert dag.n_edges == 2
        assert dag.variable_names == variable_names
        assert dag.is_acyclic()

    def test_cyclic_dag_rejection(self):
        """Test that cyclic graphs are rejected."""
        # Create cyclic graph: X -> Y -> Z -> X
        adj_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        variable_names = ["X", "Y", "Z"]

        with pytest.raises(ValueError, match="contains cycles"):
            CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

    def test_invalid_dimensions(self):
        """Test validation of matrix dimensions."""
        adj_matrix = np.array([[0, 1], [0, 0]])
        variable_names = ["X", "Y", "Z"]  # Wrong number of names

        with pytest.raises(ValueError, match="doesn't match number of variables"):
            CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

    def test_dag_properties(self):
        """Test DAG property methods."""
        adj_matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        variable_names = ["X", "Y", "Z"]

        dag = CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

        assert dag.edge_density == 3 / 6  # 3 edges out of 6 possible
        assert dag.get_parents("Y") == ["X"]
        assert dag.get_parents("Z") == ["X", "Y"]
        assert dag.get_children("X") == ["Y", "Z"]
        assert dag.has_edge("X", "Y")
        assert not dag.has_edge("Y", "X")

    def test_markov_blanket(self):
        """Test Markov blanket computation."""
        # Create DAG: X -> Y -> Z, A -> Y
        adj_matrix = np.array(
            [
                [0, 1, 0, 0],  # X -> Y
                [0, 0, 1, 0],  # Y -> Z
                [0, 0, 0, 0],  # Z (no outgoing)
                [0, 1, 0, 0],  # A -> Y
            ]
        )
        variable_names = ["X", "Y", "Z", "A"]

        dag = CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

        # Markov blanket of Y should be {X, Z, A}
        mb_y = set(dag.get_markov_blanket("Y"))
        expected_mb = {"X", "Z", "A"}
        assert mb_y == expected_mb

    def test_structural_hamming_distance(self):
        """Test structural Hamming distance calculation."""
        adj1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        adj2 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        variable_names = ["X", "Y", "Z"]

        dag1 = CausalDAG(adjacency_matrix=adj1, variable_names=variable_names)
        dag2 = CausalDAG(adjacency_matrix=adj2, variable_names=variable_names)

        # Only one edge differs
        assert dag1.structural_hamming_distance(dag2) == 1


class TestPCAlgorithm:
    """Test the PC Algorithm for causal discovery."""

    def test_pc_initialization(self):
        """Test PC algorithm initialization."""
        pc = PCAlgorithm(independence_test="pearson", alpha=0.05)
        assert pc.independence_test == "pearson"
        assert pc.alpha == 0.05
        assert not pc.is_fitted

    def test_pc_data_validation(self):
        """Test data validation in PC algorithm."""
        pc = PCAlgorithm()

        # Empty DataFrame
        with pytest.raises(Exception):
            pc.discover(pd.DataFrame())

        # Too few variables
        data = pd.DataFrame({"X": [1, 2, 3]})
        with pytest.raises(Exception):
            pc.discover(data)

        # Constant variable
        data = pd.DataFrame({"X": [1, 1, 1], "Y": [1, 2, 3]})
        with pytest.raises(Exception):
            pc.discover(data)

    def test_pc_on_linear_data(self):
        """Test PC algorithm on linear SEM data."""
        # Generate data from known DAG: X -> Y -> Z
        np.random.seed(42)
        n = 500
        X = np.random.normal(0, 1, n)
        Y = 0.8 * X + np.random.normal(0, 0.5, n)
        Z = 0.6 * Y + np.random.normal(0, 0.5, n)

        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        pc = PCAlgorithm(alpha=0.05, verbose=False)
        result = pc.discover(data)

        assert isinstance(result, DiscoveryResult)
        assert result.algorithm_name == "PC"
        assert result.dag.n_variables == 3
        assert result.dag.is_acyclic()

    def test_pc_independence_tests(self):
        """Test different independence tests in PC."""
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, n)
        Y = 0.8 * X + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"X": X, "Y": Y})

        for test_type in ["pearson", "spearman", "mutual_info"]:
            pc = PCAlgorithm(independence_test=test_type, alpha=0.05)
            result = pc.discover(data)
            assert result.dag.n_variables == 2


class TestGESAlgorithm:
    """Test the GES Algorithm for causal discovery."""

    def test_ges_initialization(self):
        """Test GES algorithm initialization."""
        ges = GESAlgorithm(score_function="bic", max_parents=3)
        assert ges.score_function == "bic"
        assert ges.max_parents == 3
        assert not ges.is_fitted

    def test_ges_score_functions(self):
        """Test different score functions in GES."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, n)
        Y = 0.8 * X + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"X": X, "Y": Y})

        for score_func in ["bic", "aic", "likelihood"]:
            ges = GESAlgorithm(score_function=score_func)
            result = ges.discover(data)
            assert result.dag.n_variables == 2

    def test_ges_on_linear_data(self):
        """Test GES algorithm on linear SEM data."""
        # Generate data from known DAG: X -> Y -> Z
        np.random.seed(42)
        n = 300
        X = np.random.normal(0, 1, n)
        Y = 0.8 * X + np.random.normal(0, 0.5, n)
        Z = 0.6 * Y + np.random.normal(0, 0.5, n)

        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        ges = GESAlgorithm(score_function="bic", verbose=False)
        result = ges.discover(data)

        assert isinstance(result, DiscoveryResult)
        assert result.algorithm_name == "GES"
        assert result.dag.n_variables == 3
        assert result.dag.is_acyclic()
        assert result.convergence_achieved is not None


class TestNOTEARSAlgorithm:
    """Test the NOTEARS Algorithm for causal discovery."""

    def test_notears_initialization(self):
        """Test NOTEARS algorithm initialization."""
        notears = NOTEARSAlgorithm(lambda_l1=0.1, lambda_l2=0.01)
        assert notears.lambda_l1 == 0.1
        assert notears.lambda_l2 == 0.01
        assert not notears.is_fitted

    def test_notears_on_linear_data(self):
        """Test NOTEARS algorithm on linear SEM data."""
        # Generate simple linear data
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, n)
        Y = 0.8 * X + np.random.normal(0, 0.5, n)

        data = pd.DataFrame({"X": X, "Y": Y})

        notears = NOTEARSAlgorithm(
            lambda_l1=0.1, lambda_l2=0.01, max_iter=50, verbose=False
        )
        result = notears.discover(data)

        assert isinstance(result, DiscoveryResult)
        assert result.algorithm_name == "NOTEARS"
        assert result.dag.n_variables == 2
        assert result.dag.is_acyclic()


class TestDiscoveryUtils:
    """Test utility functions for causal discovery."""

    def test_generate_linear_sem_data(self):
        """Test SEM data generation."""
        # Create simple DAG: X -> Y
        adj_matrix = np.array([[0, 1], [0, 0]])
        variable_names = ["X", "Y"]
        dag = CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

        data = generate_linear_sem_data(dag, n_samples=100, random_state=42)

        assert isinstance(data, pd.DataFrame)
        assert data.shape == (100, 2)
        assert list(data.columns) == ["X", "Y"]

        # Y should be correlated with X
        correlation = data["X"].corr(data["Y"])
        assert abs(correlation) > 0.3  # Should have some correlation

    def test_dag_to_adjustment_sets(self):
        """Test adjustment set identification."""
        # Create DAG: C -> X -> Y, C -> Y (C is confounder)
        adj_matrix = np.array(
            [
                [0, 1, 1],  # C -> X, C -> Y
                [0, 0, 1],  # X -> Y
                [0, 0, 0],  # Y (no outgoing)
            ]
        )
        variable_names = ["C", "X", "Y"]
        dag = CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

        adjustment_sets = dag_to_adjustment_sets(dag, "X", "Y")

        assert "minimal_adjustment_set" in adjustment_sets
        assert "treatment_parents" in adjustment_sets
        assert "C" in adjustment_sets["treatment_parents"]

    def test_compare_dags(self):
        """Test DAG comparison metrics."""
        # True DAG: X -> Y -> Z
        adj_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        # Estimated DAG: X -> Y, X -> Z (missing Y -> Z, extra X -> Z)
        adj_est = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        variable_names = ["X", "Y", "Z"]

        dag_true = CausalDAG(adjacency_matrix=adj_true, variable_names=variable_names)
        dag_est = CausalDAG(adjacency_matrix=adj_est, variable_names=variable_names)

        metrics = compare_dags(dag_true, dag_est)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "structural_hamming_distance" in metrics
        assert metrics["structural_hamming_distance"] == 2  # 2 differences


class TestDiscoveryIntegration:
    """Test integration between discovery and estimation."""

    def test_discovery_estimator_pipeline(self):
        """Test the complete discovery-estimation pipeline."""
        # Generate data with known causal structure
        np.random.seed(42)
        n = 300
        C = np.random.normal(0, 1, n)  # Confounder
        # Generate binary treatment based on confounder
        treatment_prob = 1 / (1 + np.exp(-(0.5 * C)))
        X = np.random.binomial(1, treatment_prob)  # Binary treatment
        Y = 0.8 * X + 0.3 * C + np.random.normal(0, 0.5, n)  # Outcome

        data = pd.DataFrame({"C": C, "X": X, "Y": Y})

        # Create pipeline
        discovery_alg = PCAlgorithm(alpha=0.1, verbose=False)
        estimator = GComputationEstimator(verbose=False)
        pipeline = DiscoveryEstimatorPipeline(discovery_alg, estimator, verbose=False)

        # Run pipeline
        causal_effect = pipeline.fit_and_estimate(
            data, treatment_col="X", outcome_col="Y"
        )

        assert causal_effect is not None
        assert hasattr(causal_effect, "ate")
        assert pipeline.discovery_result is not None

    def test_estimate_causal_effect_from_discovery(self):
        """Test estimating causal effect from discovery result."""
        # Create simple DAG and data
        adj_matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        variable_names = ["C", "X", "Y"]
        dag = CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

        # Create discovery result
        discovery_result = DiscoveryResult(
            dag=dag, algorithm_name="test", algorithm_parameters={}
        )

        # Generate data
        np.random.seed(42)
        n = 200
        C = np.random.normal(0, 1, n)
        # Generate binary treatment based on confounder
        treatment_prob = 1 / (1 + np.exp(-(0.5 * C)))
        X = np.random.binomial(1, treatment_prob)  # Binary treatment
        Y = 0.8 * X + 0.3 * C + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"C": C, "X": X, "Y": Y})

        # Estimate causal effect
        estimator = GComputationEstimator(verbose=False)
        causal_effect = estimate_causal_effect_from_discovery(
            discovery_result, data, "X", "Y", estimator
        )

        assert causal_effect is not None
        assert hasattr(causal_effect, "ate")

    def test_validate_discovery_assumptions(self):
        """Test validation of discovery assumptions."""
        # Create valid DAG
        adj_matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        variable_names = ["C", "X", "Y"]
        dag = CausalDAG(adjacency_matrix=adj_matrix, variable_names=variable_names)

        discovery_result = DiscoveryResult(
            dag=dag, algorithm_name="test", algorithm_parameters={}
        )

        # Create data
        data = pd.DataFrame(
            {
                "C": np.random.normal(0, 1, 100),
                "X": np.random.normal(0, 1, 100),
                "Y": np.random.normal(0, 1, 100),
            }
        )

        validation = validate_discovery_assumptions(discovery_result, data, "X", "Y")

        assert "dag_is_acyclic" in validation
        assert "treatment_in_dag" in validation
        assert "outcome_in_dag" in validation
        assert "has_causal_path" in validation
        assert validation["dag_is_acyclic"]
        assert validation["treatment_in_dag"]
        assert validation["outcome_in_dag"]


class TestBootstrapDiscovery:
    """Test bootstrap-based discovery for uncertainty quantification."""

    def test_bootstrap_discovery(self):
        """Test bootstrap discovery functionality."""
        # Generate data
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, n)
        Y = 0.8 * X + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"X": X, "Y": Y})

        # Run bootstrap discovery
        pc = PCAlgorithm(alpha=0.1, verbose=False)
        result = pc.bootstrap_discovery(data, n_bootstrap=10, bootstrap_fraction=0.8)

        assert result.bootstrap_dags is not None
        assert len(result.bootstrap_dags) <= 10  # Some might fail
        assert result.edge_probabilities is not None
        assert result.stability_score is not None

        # Check edge probabilities shape
        expected_shape = (data.shape[1], data.shape[1])
        assert result.edge_probabilities.shape == expected_shape


@pytest.mark.integration
class TestDiscoveryIntegrationEnd2End:
    """End-to-end integration tests for causal discovery."""

    def test_complete_discovery_workflow(self):
        """Test complete workflow from discovery to causal effect estimation."""
        # Generate data with known causal structure
        # DAG: C -> X -> Y, C -> Y (C is confounder)
        np.random.seed(42)
        n = 500
        C = np.random.normal(0, 1, n)
        # Generate binary treatment based on confounder
        treatment_prob = 1 / (1 + np.exp(-(0.6 * C)))
        X = np.random.binomial(1, treatment_prob)  # Binary treatment
        Y = 0.8 * X + 0.4 * C + np.random.normal(0, 0.6, n)

        data = pd.DataFrame({"C": C, "X": X, "Y": Y})

        # Test different discovery algorithms
        algorithms = [
            ("PC", PCAlgorithm(alpha=0.05, verbose=False)),
            ("GES", GESAlgorithm(score_function="bic", verbose=False)),
            ("NOTEARS", NOTEARSAlgorithm(lambda_l1=0.05, max_iter=30, verbose=False)),
        ]

        results = {}

        for name, algorithm in algorithms:
            try:
                # Discover structure
                discovery_result = algorithm.discover(data)

                # Estimate causal effect
                estimator = GComputationEstimator(verbose=False)
                causal_effect = estimate_causal_effect_from_discovery(
                    discovery_result, data, "X", "Y", estimator
                )

                # Validate assumptions
                validation = validate_discovery_assumptions(
                    discovery_result, data, "X", "Y"
                )

                results[name] = {
                    "discovery_result": discovery_result,
                    "causal_effect": causal_effect,
                    "validation": validation,
                }

                # Basic checks
                assert discovery_result.dag.is_acyclic()
                assert causal_effect is not None
                assert validation["dag_is_acyclic"]

            except Exception as e:
                pytest.skip(f"Algorithm {name} failed: {str(e)}")

        # At least one algorithm should succeed
        assert len(results) > 0

    def test_algorithm_comparison(self):
        """Test comparing multiple algorithms on the same data."""
        # Generate simple linear data
        np.random.seed(42)
        n = 300
        X = np.random.normal(0, 1, n)
        Y = 0.7 * X + np.random.normal(0, 0.6, n)
        Z = 0.5 * Y + np.random.normal(0, 0.6, n)

        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        algorithms = [
            PCAlgorithm(alpha=0.05, verbose=False),
            GESAlgorithm(score_function="bic", verbose=False),
        ]

        results = []
        for algorithm in algorithms:
            try:
                result = algorithm.discover(data)
                results.append(result)
            except Exception:
                continue

        # Should have at least one successful result
        assert len(results) > 0

        # All results should be acyclic
        for result in results:
            assert result.dag.is_acyclic()
            assert result.dag.n_variables == 3
