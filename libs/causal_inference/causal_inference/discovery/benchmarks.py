"""Benchmarking and validation utilities for causal discovery algorithms.

This module provides functions to evaluate and compare causal discovery
methods on simulated data with known ground truth structures.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .base import BaseDiscoveryAlgorithm, CausalDAG
from .constraint_based import PCAlgorithm
from .score_based import GESAlgorithm, NOTEARSAlgorithm
from .utils import compare_dags, generate_linear_sem_data

__all__ = [
    "create_benchmark_dags",
    "benchmark_discovery_algorithm",
    "compare_discovery_algorithms",
    "DiscoveryBenchmarkSuite",
]


def create_benchmark_dags() -> dict[str, CausalDAG]:
    """Create a collection of benchmark DAGs for testing discovery algorithms.

    Returns:
        Dictionary mapping DAG names to CausalDAG objects
    """
    dags = {}

    # Simple chain: X -> Y -> Z
    adj_chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    dags["chain_3"] = CausalDAG(
        adjacency_matrix=adj_chain, variable_names=["X", "Y", "Z"]
    )

    # Fork: X <- Y -> Z
    adj_fork = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    dags["fork_3"] = CausalDAG(
        adjacency_matrix=adj_fork, variable_names=["X", "Y", "Z"]
    )

    # Collider: X -> Y <- Z
    adj_collider = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    dags["collider_3"] = CausalDAG(
        adjacency_matrix=adj_collider, variable_names=["X", "Y", "Z"]
    )

    # Diamond: X -> Y -> Z, X -> W -> Z
    adj_diamond = np.array(
        [
            [0, 1, 0, 1],  # X -> Y, X -> W
            [0, 0, 1, 0],  # Y -> Z
            [0, 0, 0, 0],  # Z (no outgoing)
            [0, 0, 1, 0],  # W -> Z
        ]
    )
    dags["diamond_4"] = CausalDAG(
        adjacency_matrix=adj_diamond, variable_names=["X", "Y", "Z", "W"]
    )

    # Complex network with confounders
    adj_complex = np.array(
        [
            [0, 1, 1, 0, 0],  # C1 -> X, C1 -> Y
            [0, 0, 1, 1, 0],  # X -> Y, X -> M
            [0, 0, 0, 0, 1],  # Y -> O
            [0, 0, 0, 0, 1],  # M -> O (mediator)
            [0, 0, 0, 0, 0],  # O (outcome)
        ]
    )
    dags["complex_5"] = CausalDAG(
        adjacency_matrix=adj_complex, variable_names=["C1", "X", "Y", "M", "O"]
    )

    # Sparse network (low density)
    adj_sparse = np.zeros((6, 6))
    adj_sparse[0, 3] = 1  # X1 -> X4
    adj_sparse[1, 4] = 1  # X2 -> X5
    adj_sparse[2, 5] = 1  # X3 -> X6
    dags["sparse_6"] = CausalDAG(
        adjacency_matrix=adj_sparse, variable_names=[f"X{i + 1}" for i in range(6)]
    )

    # Dense network (high density)
    adj_dense = np.array(
        [
            [0, 1, 1, 1],  # X1 -> X2, X3, X4
            [0, 0, 1, 1],  # X2 -> X3, X4
            [0, 0, 0, 1],  # X3 -> X4
            [0, 0, 0, 0],  # X4 (no outgoing)
        ]
    )
    dags["dense_4"] = CausalDAG(
        adjacency_matrix=adj_dense, variable_names=[f"X{i + 1}" for i in range(4)]
    )

    return dags


def benchmark_discovery_algorithm(
    algorithm: BaseDiscoveryAlgorithm,
    true_dag: CausalDAG,
    n_samples: int = 500,
    noise_std: float = 1.0,
    n_trials: int = 10,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Benchmark a discovery algorithm on a known DAG structure.

    Args:
        algorithm: Discovery algorithm to test
        true_dag: True causal DAG
        n_samples: Number of samples to generate
        noise_std: Standard deviation of noise in data generation
        n_trials: Number of independent trials
        random_state: Random seed

    Returns:
        Dictionary with benchmark results
    """
    if random_state is not None:
        np.random.seed(random_state)

    trials: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "algorithm_name": algorithm.__class__.__name__,
        "true_dag": true_dag,
        "n_samples": n_samples,
        "n_trials": n_trials,
        "trials": trials,
        "summary_stats": {},
    }

    # Run multiple trials
    for trial in range(n_trials):
        trial_seed = random_state + trial if random_state is not None else None

        # Generate data
        data = generate_linear_sem_data(
            true_dag, n_samples=n_samples, noise_std=noise_std, random_state=trial_seed
        )

        # Discover structure
        start_time = time.time()
        try:
            discovery_result = algorithm.discover(data)
            computation_time = time.time() - start_time
            success = True
        except Exception as e:
            computation_time = time.time() - start_time
            discovery_result = None
            success = False
            error_msg = str(e)

        # Evaluate if successful
        trial_result: dict[str, Any]
        if success and discovery_result is not None:
            metrics = compare_dags(true_dag, discovery_result.dag)
            trial_result = {
                "trial": trial,
                "success": True,
                "computation_time": computation_time,
                "discovery_result": discovery_result,
                "metrics": metrics,
            }
        else:
            trial_result = {
                "trial": trial,
                "success": False,
                "computation_time": computation_time,
                "error": error_msg,
            }

        trials.append(trial_result)

    # Compute summary statistics
    successful_trials = [t for t in trials if t["success"]]
    n_successful = len(successful_trials)

    if n_successful > 0:
        # Aggregate metrics
        metrics_keys = [
            "precision",
            "recall",
            "f1_score",
            "structural_hamming_distance",
            "jaccard_similarity",
        ]
        summary_metrics = {}

        for key in metrics_keys:
            values = [t["metrics"][key] for t in successful_trials]
            summary_metrics[f"{key}_mean"] = np.mean(values)
            summary_metrics[f"{key}_std"] = np.std(values)
            summary_metrics[f"{key}_min"] = np.min(values)
            summary_metrics[f"{key}_max"] = np.max(values)

        # Timing statistics
        times = [t["computation_time"] for t in successful_trials]
        summary_metrics["computation_time_mean"] = np.mean(times)
        summary_metrics["computation_time_std"] = np.std(times)

        results["summary_stats"] = {
            "success_rate": n_successful / n_trials,
            "n_successful": n_successful,
            **summary_metrics,
        }
    else:
        results["summary_stats"] = {
            "success_rate": 0.0,
            "n_successful": 0,
        }

    return results


def compare_discovery_algorithms(
    algorithms: list[BaseDiscoveryAlgorithm],
    algorithm_names: Optional[list[str]] = None,
    benchmark_dags: Optional[dict[str, CausalDAG]] = None,
    n_samples: int = 500,
    n_trials: int = 10,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compare multiple discovery algorithms on benchmark DAGs.

    Args:
        algorithms: List of discovery algorithms to compare
        algorithm_names: Names for algorithms (optional)
        benchmark_dags: Dictionary of benchmark DAGs (uses defaults if None)
        n_samples: Number of samples per trial
        n_trials: Number of trials per DAG
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Nested dictionary with results: {dag_name: {algorithm_name: results}}
    """
    if algorithm_names is None:
        algorithm_names = [alg.__class__.__name__ for alg in algorithms]

    if len(algorithm_names) != len(algorithms):
        raise ValueError("Number of algorithm names must match number of algorithms")

    if benchmark_dags is None:
        benchmark_dags = create_benchmark_dags()

    results: dict[str, dict[str, Any]] = {}

    for dag_name, true_dag in benchmark_dags.items():
        if verbose:
            print(
                f"Testing DAG: {dag_name} ({true_dag.n_variables} variables, {true_dag.n_edges} edges)"
            )

        results[dag_name] = {}

        for algorithm, alg_name in zip(algorithms, algorithm_names):
            if verbose:
                print(f"  Running {alg_name}...")

            try:
                alg_results = benchmark_discovery_algorithm(
                    algorithm,
                    true_dag,
                    n_samples,
                    n_trials=n_trials,
                    random_state=random_state,
                )
                results[dag_name][alg_name] = alg_results

                if verbose:
                    success_rate = alg_results["summary_stats"]["success_rate"]
                    if success_rate > 0:
                        f1_mean = alg_results["summary_stats"].get("f1_score_mean", 0)
                        print(
                            f"    Success rate: {success_rate:.2f}, F1: {f1_mean:.3f}"
                        )
                    else:
                        print(f"    Success rate: {success_rate:.2f}")

            except Exception as e:
                if verbose:
                    print(f"    Failed: {str(e)}")
                results[dag_name][alg_name] = {"error": str(e)}

    return results


class DiscoveryBenchmarkSuite:
    """Comprehensive benchmark suite for causal discovery algorithms."""

    def __init__(
        self,
        algorithms: Optional[list[BaseDiscoveryAlgorithm]] = None,
        algorithm_names: Optional[list[str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize benchmark suite.

        Args:
            algorithms: List of algorithms to benchmark (uses defaults if None)
            algorithm_names: Names for algorithms
            random_state: Random seed for reproducibility
        """
        if algorithms is None:
            # Default set of algorithms
            algorithms = [
                PCAlgorithm(alpha=0.05, verbose=False),
                GESAlgorithm(score_function="bic", verbose=False),
                NOTEARSAlgorithm(lambda_l1=0.1, max_iter=50, verbose=False),
            ]

        self.algorithms = algorithms
        self.algorithm_names = algorithm_names or [
            alg.__class__.__name__ for alg in algorithms
        ]
        self.random_state = random_state

        # Results storage
        self.results: dict[str, Any] = {}
        self.benchmark_dags = create_benchmark_dags()

    def run_sample_size_analysis(
        self,
        sample_sizes: list[int] = [100, 250, 500, 1000],
        dag_name: str = "chain_3",
        n_trials: int = 10,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Analyze algorithm performance vs sample size.

        Args:
            sample_sizes: List of sample sizes to test
            dag_name: Name of DAG to use for testing
            n_trials: Number of trials per sample size
            verbose: Whether to print progress

        Returns:
            Dictionary with sample size analysis results
        """
        if dag_name not in self.benchmark_dags:
            raise ValueError(f"DAG '{dag_name}' not found in benchmark DAGs")

        true_dag = self.benchmark_dags[dag_name]
        algorithms_results: dict[str, Any] = {}
        results: dict[str, Any] = {"sample_sizes": sample_sizes, "algorithms": algorithms_results}

        if verbose:
            print(f"Sample size analysis on DAG: {dag_name}")

        for alg, alg_name in zip(self.algorithms, self.algorithm_names):
            if verbose:
                print(f"  Algorithm: {alg_name}")

            alg_results: dict[str, list[Any]] = {"sample_sizes": [], "metrics": []}

            for n_samples in sample_sizes:
                if verbose:
                    print(f"    Sample size: {n_samples}")

                benchmark_result = benchmark_discovery_algorithm(
                    alg,
                    true_dag,
                    n_samples=n_samples,
                    n_trials=n_trials,
                    random_state=self.random_state,
                )

                alg_results["sample_sizes"].append(n_samples)
                alg_results["metrics"].append(benchmark_result["summary_stats"])

            results["algorithms"][alg_name] = alg_results

        self.results["sample_size_analysis"] = results
        return results

    def run_noise_sensitivity_analysis(
        self,
        noise_levels: list[float] = [0.5, 1.0, 1.5, 2.0],
        dag_name: str = "chain_3",
        n_samples: int = 500,
        n_trials: int = 10,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Analyze algorithm sensitivity to noise levels.

        Args:
            noise_levels: List of noise standard deviations to test
            dag_name: Name of DAG to use for testing
            n_samples: Number of samples per trial
            n_trials: Number of trials per noise level
            verbose: Whether to print progress

        Returns:
            Dictionary with noise sensitivity analysis results
        """
        if dag_name not in self.benchmark_dags:
            raise ValueError(f"DAG '{dag_name}' not found in benchmark DAGs")

        true_dag = self.benchmark_dags[dag_name]
        algorithms_results: dict[str, Any] = {}
        results: dict[str, Any] = {"noise_levels": noise_levels, "algorithms": algorithms_results}

        if verbose:
            print(f"Noise sensitivity analysis on DAG: {dag_name}")

        for alg, alg_name in zip(self.algorithms, self.algorithm_names):
            if verbose:
                print(f"  Algorithm: {alg_name}")

            alg_results: dict[str, list[Any]] = {"noise_levels": [], "metrics": []}

            for noise_std in noise_levels:
                if verbose:
                    print(f"    Noise level: {noise_std}")

                benchmark_result = benchmark_discovery_algorithm(
                    alg,
                    true_dag,
                    n_samples=n_samples,
                    noise_std=noise_std,
                    n_trials=n_trials,
                    random_state=self.random_state,
                )

                alg_results["noise_levels"].append(noise_std)
                alg_results["metrics"].append(benchmark_result["summary_stats"])

            results["algorithms"][alg_name] = alg_results

        self.results["noise_sensitivity_analysis"] = results
        return results

    def run_scalability_analysis(
        self,
        dag_names: Optional[list[str]] = None,
        n_samples: int = 500,
        n_trials: int = 5,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Analyze algorithm scalability across different DAG sizes.

        Args:
            dag_names: List of DAG names to test (uses all if None)
            n_samples: Number of samples per trial
            n_trials: Number of trials per DAG
            verbose: Whether to print progress

        Returns:
            Dictionary with scalability analysis results
        """
        if dag_names is None:
            dag_names = list(self.benchmark_dags.keys())

        algorithms_results: dict[str, Any] = {}
        results: dict[str, Any] = {"dag_names": dag_names, "algorithms": algorithms_results}

        if verbose:
            print("Scalability analysis")

        for alg, alg_name in zip(self.algorithms, self.algorithm_names):
            if verbose:
                print(f"  Algorithm: {alg_name}")

            alg_results: dict[str, list[Any]] = {"dag_properties": [], "metrics": []}

            for dag_name in dag_names:
                true_dag = self.benchmark_dags[dag_name]

                if verbose:
                    print(
                        f"    DAG: {dag_name} ({true_dag.n_variables}V, {true_dag.n_edges}E)"
                    )

                benchmark_result = benchmark_discovery_algorithm(
                    alg,
                    true_dag,
                    n_samples=n_samples,
                    n_trials=n_trials,
                    random_state=self.random_state,
                )

                dag_properties = {
                    "dag_name": dag_name,
                    "n_variables": true_dag.n_variables,
                    "n_edges": true_dag.n_edges,
                    "edge_density": true_dag.edge_density,
                }

                alg_results["dag_properties"].append(dag_properties)
                alg_results["metrics"].append(benchmark_result["summary_stats"])

            results["algorithms"][alg_name] = alg_results

        self.results["scalability_analysis"] = results
        return results

    def generate_summary_report(self) -> str:
        """Generate a text summary of all benchmark results.

        Returns:
            String with formatted summary report
        """
        lines = ["=== Causal Discovery Benchmark Report ===", ""]

        if not self.results:
            lines.append(
                "No benchmark results available. Run benchmark analyses first."
            )
            return "\n".join(lines)

        # Sample size analysis
        if "sample_size_analysis" in self.results:
            lines.extend(["Sample Size Analysis:", "-" * 20])
            sample_results = self.results["sample_size_analysis"]

            for alg_name in sample_results["algorithms"]:
                lines.append(f"\n{alg_name}:")
                alg_data = sample_results["algorithms"][alg_name]

                for i, n_samples in enumerate(alg_data["sample_sizes"]):
                    metrics = alg_data["metrics"][i]
                    if metrics.get("success_rate", 0) > 0:
                        f1_mean = metrics.get("f1_score_mean", 0)
                        lines.append(
                            f"  N={n_samples}: Success={metrics['success_rate']:.2f}, F1={f1_mean:.3f}"
                        )
                    else:
                        lines.append(
                            f"  N={n_samples}: Success={metrics['success_rate']:.2f}"
                        )

        # Noise sensitivity analysis
        if "noise_sensitivity_analysis" in self.results:
            lines.extend(["", "", "Noise Sensitivity Analysis:", "-" * 25])
            noise_results = self.results["noise_sensitivity_analysis"]

            for alg_name in noise_results["algorithms"]:
                lines.append(f"\n{alg_name}:")
                alg_data = noise_results["algorithms"][alg_name]

                for i, noise_level in enumerate(alg_data["noise_levels"]):
                    metrics = alg_data["metrics"][i]
                    if metrics.get("success_rate", 0) > 0:
                        f1_mean = metrics.get("f1_score_mean", 0)
                        lines.append(
                            f"  Noise={noise_level}: Success={metrics['success_rate']:.2f}, F1={f1_mean:.3f}"
                        )
                    else:
                        lines.append(
                            f"  Noise={noise_level}: Success={metrics['success_rate']:.2f}"
                        )

        # Scalability analysis
        if "scalability_analysis" in self.results:
            lines.extend(["", "", "Scalability Analysis:", "-" * 20])
            scalability_results = self.results["scalability_analysis"]

            for alg_name in scalability_results["algorithms"]:
                lines.append(f"\n{alg_name}:")
                alg_data = scalability_results["algorithms"][alg_name]

                for i, dag_props in enumerate(alg_data["dag_properties"]):
                    metrics = alg_data["metrics"][i]
                    dag_name = dag_props["dag_name"]
                    n_vars = dag_props["n_variables"]
                    n_edges = dag_props["n_edges"]

                    if metrics.get("success_rate", 0) > 0:
                        f1_mean = metrics.get("f1_score_mean", 0)
                        time_mean = metrics.get("computation_time_mean", 0)
                        lines.append(
                            f"  {dag_name} ({n_vars}V,{n_edges}E): Success={metrics['success_rate']:.2f}, "
                            f"F1={f1_mean:.3f}, Time={time_mean:.2f}s"
                        )
                    else:
                        lines.append(
                            f"  {dag_name} ({n_vars}V,{n_edges}E): Success={metrics['success_rate']:.2f}"
                        )

        return "\n".join(lines)
