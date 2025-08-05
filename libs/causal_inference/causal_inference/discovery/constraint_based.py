"""Constraint-based causal discovery algorithms.

This module implements constraint-based methods that use conditional independence
tests to learn causal structure from data.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from .base import BaseDiscoveryAlgorithm, CausalDAG, DiscoveryResult

__all__ = ["PCAlgorithm", "FCI"]


class PCAlgorithm(BaseDiscoveryAlgorithm):
    """PC Algorithm for causal discovery using conditional independence tests.

    The PC algorithm learns causal structure by testing conditional independence
    relationships and applying orientation rules based on causal assumptions.

    Reference: Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation,
    prediction, and search. MIT press.
    """

    def __init__(
        self,
        independence_test: str = "pearson",
        alpha: float = 0.05,
        max_conditioning_set_size: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        max_iterations: int = 1000,
    ) -> None:
        """Initialize PC Algorithm.

        Args:
            independence_test: Type of independence test ('pearson', 'spearman', 'mutual_info')
            alpha: Significance level for independence tests
            max_conditioning_set_size: Maximum size of conditioning sets
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
            max_iterations: Maximum iterations (not used in PC)
        """
        super().__init__(random_state, verbose, max_iterations)

        self.independence_test = independence_test
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size

        # Set independence test function
        self._independence_test_func = self._get_independence_test_function()

        # Algorithm state
        self.separation_sets: dict[tuple[int, int], set[int]] = {}
        self.pc_graph: NDArray[Any] | None = None
        self.oriented_graph: NDArray[Any] | None = None

    def _get_independence_test_function(self) -> Callable:
        """Get the appropriate independence test function."""
        if self.independence_test == "pearson":
            return self._pearson_independence_test
        elif self.independence_test == "spearman":
            return self._spearman_independence_test
        elif self.independence_test == "mutual_info":
            return self._mutual_info_independence_test
        else:
            raise ValueError(f"Unknown independence test: {self.independence_test}")

    def _pearson_independence_test(
        self, data: pd.DataFrame, x: int, y: int, conditioning_set: set[int]
    ) -> tuple[bool, float]:
        """Perform partial correlation test for conditional independence."""
        if not conditioning_set:
            # Simple correlation test
            corr, p_value = stats.pearsonr(data.iloc[:, x], data.iloc[:, y])
            return p_value > self.alpha, p_value

        # Partial correlation test
        try:
            # Select relevant columns
            cols = [x, y] + list(conditioning_set)
            subset_data = data.iloc[:, cols].dropna()

            if len(subset_data) < 10:  # Too few observations
                return False, 1.0

            # Compute partial correlation
            corr_matrix = subset_data.corr().values

            if corr_matrix.shape[0] < 3:
                return False, 1.0

            # Partial correlation between x and y given conditioning set
            precision_matrix = np.linalg.inv(corr_matrix)
            partial_corr = -precision_matrix[0, 1] / np.sqrt(
                precision_matrix[0, 0] * precision_matrix[1, 1]
            )

            # Statistical test
            n = len(subset_data)
            k = len(conditioning_set)
            df = n - k - 2

            if df <= 0:
                return False, 1.0

            t_stat = partial_corr * np.sqrt(df / (1 - partial_corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            return p_value > self.alpha, p_value

        except (np.linalg.LinAlgError, ValueError):
            return False, 1.0

    def _spearman_independence_test(
        self, data: pd.DataFrame, x: int, y: int, conditioning_set: set[int]
    ) -> tuple[bool, float]:
        """Perform Spearman rank correlation test."""
        if not conditioning_set:
            # Simple Spearman correlation
            corr, p_value = stats.spearmanr(data.iloc[:, x], data.iloc[:, y])
            return p_value > self.alpha, p_value

        # For conditioning sets, use Pearson on ranks
        rank_data = data.rank()
        return self._pearson_independence_test(rank_data, x, y, conditioning_set)

    def _mutual_info_independence_test(
        self, data: pd.DataFrame, x: int, y: int, conditioning_set: set[int]
    ) -> tuple[bool, float]:
        """Perform mutual information test for independence."""
        # This is a simplified implementation
        # For proper conditional mutual information, more sophisticated methods needed

        if not conditioning_set:
            # Simple mutual information
            mi = mutual_info_regression(
                data.iloc[:, [x]], data.iloc[:, y], random_state=self.random_state
            )[0]
            # Convert to approximate p-value (rough approximation)
            p_value = np.exp(-mi * len(data) / 10)
            return p_value > self.alpha, p_value

        # Conditional mutual information approximation
        # This is simplified - proper implementation would use more sophisticated methods
        try:
            # Use linear regression residuals as approximation
            from sklearn.linear_model import LinearRegression

            # Regress x on conditioning set
            if len(conditioning_set) > 0:
                X_cond = data.iloc[:, list(conditioning_set)]
                reg_x = LinearRegression().fit(X_cond, data.iloc[:, x])
                residual_x = data.iloc[:, x] - reg_x.predict(X_cond)

                reg_y = LinearRegression().fit(X_cond, data.iloc[:, y])
                residual_y = data.iloc[:, y] - reg_y.predict(X_cond)
            else:
                residual_x = data.iloc[:, x]
                residual_y = data.iloc[:, y]

            # Mutual information on residuals
            mi = mutual_info_regression(
                residual_x.values.reshape(-1, 1),
                residual_y.values,
                random_state=self.random_state,
            )[0]

            p_value = np.exp(-mi * len(data) / 10)
            return p_value > self.alpha, p_value

        except Exception:
            return False, 1.0

    def _discover_implementation(self, data: pd.DataFrame) -> DiscoveryResult:
        """Implement PC algorithm discovery logic."""
        start_time = time.time()

        n_vars = len(data.columns)
        variable_names = list(data.columns)

        # Initialize with complete undirected graph
        graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        self.separation_sets = {}

        if self.verbose:
            print(f"Starting PC algorithm with {n_vars} variables")
            print(f"Initial graph has {np.sum(graph)} edges")

        # Phase 1: Edge removal using conditional independence tests
        max_level = (
            self.max_conditioning_set_size
            if self.max_conditioning_set_size is not None
            else n_vars - 2
        )

        for level in range(max_level + 1):
            if self.verbose:
                print(f"Testing conditioning sets of size {level}")

            edges_removed = 0
            edges_to_check = [
                (i, j)
                for i in range(n_vars)
                for j in range(i + 1, n_vars)
                if graph[i, j] == 1
            ]

            for i, j in edges_to_check:
                # Get potential conditioning sets (neighbors excluding i and j)
                neighbors_i = set(np.where(graph[i, :] == 1)[0]) - {j}
                neighbors_j = set(np.where(graph[j, :] == 1)[0]) - {i}
                potential_conditioning = neighbors_i | neighbors_j

                # Skip if not enough neighbors for this level
                if len(potential_conditioning) < level:
                    continue

                # Test all conditioning sets of current level
                found_independence = False
                for conditioning_set in itertools.combinations(
                    potential_conditioning, level
                ):
                    conditioning_set = set(conditioning_set)

                    # Test conditional independence
                    independent, p_value = self._independence_test_func(
                        data, i, j, conditioning_set
                    )

                    if independent:
                        # Remove edge and store separation set
                        graph[i, j] = graph[j, i] = 0
                        self.separation_sets[(i, j)] = conditioning_set
                        edges_removed += 1
                        found_independence = True
                        break

                if found_independence:
                    continue

            if self.verbose:
                print(f"Level {level}: Removed {edges_removed} edges")

            if edges_removed == 0:
                break

        self.pc_graph = graph.copy()

        # Phase 2: Edge orientation using causal rules
        oriented_graph = self._orient_edges(graph, n_vars)
        self.oriented_graph = oriented_graph

        computation_time = time.time() - start_time

        # Create DAG
        dag = CausalDAG(adjacency_matrix=oriented_graph, variable_names=variable_names)

        # Create result
        result = DiscoveryResult(
            dag=dag,
            algorithm_name="PC",
            algorithm_parameters={
                "independence_test": self.independence_test,
                "alpha": self.alpha,
                "max_conditioning_set_size": self.max_conditioning_set_size,
            },
            computation_time=computation_time,
            algorithm_diagnostics={
                "n_independence_tests": len(self.separation_sets),
                "final_n_edges": dag.n_edges,
                "edge_density": dag.edge_density,
            },
        )

        return result

    def _orient_edges(self, graph: NDArray[Any], n_vars: int) -> NDArray[Any]:
        """Apply orientation rules to create directed graph."""
        # Start with skeleton (undirected graph)
        oriented = np.zeros((n_vars, n_vars))

        # Rule 1: Unshielded colliders (v-structures)
        # If i-j-k is unshielded and j is not in separation set of i,k -> i->j<-k
        for j in range(n_vars):
            neighbors_j = [i for i in range(n_vars) if graph[i, j] == 1]

            for i, k in itertools.combinations(neighbors_j, 2):
                # Check if i-k are not adjacent (unshielded)
                if graph[i, k] == 0:
                    # Check if j is not in separation set of i,k
                    sep_set = self.separation_sets.get((min(i, k), max(i, k)), set())
                    if j not in sep_set:
                        # Orient as i->j<-k
                        oriented[i, j] = 1
                        oriented[k, j] = 1

        # Rule 2: Avoid creating new colliders
        # If i->j-k and i,k not adjacent -> j->k
        changed = True
        while changed:
            changed = False
            for i, j, k in itertools.permutations(range(n_vars), 3):
                if (
                    oriented[i, j] == 1
                    and oriented[j, i] == 0  # i->j
                    and graph[j, k] == 1
                    and oriented[j, k] == 0
                    and oriented[k, j] == 0  # j-k
                    and graph[i, k] == 0
                ):  # i,k not adjacent
                    oriented[j, k] = 1
                    changed = True

        # Rule 3: Orient remaining edges to avoid cycles
        # If i->j->k and i-k -> i->k
        changed = True
        while changed:
            changed = False
            for i, j, k in itertools.permutations(range(n_vars), 3):
                if (
                    oriented[i, j] == 1
                    and oriented[j, i] == 0  # i->j
                    and oriented[j, k] == 1
                    and oriented[k, j] == 0  # j->k
                    and graph[i, k] == 1
                    and oriented[i, k] == 0
                    and oriented[k, i] == 0
                ):  # i-k
                    oriented[i, k] = 1
                    changed = True

        # Orient remaining undirected edges arbitrarily (topological order)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if graph[i, j] == 1 and oriented[i, j] == 0 and oriented[j, i] == 0:
                    oriented[i, j] = 1  # Orient as i->j

        return oriented


class FCI(BaseDiscoveryAlgorithm):
    """Fast Causal Inference (FCI) algorithm for discovery with latent confounders.

    FCI extends PC to handle latent confounders and selection bias by learning
    partial ancestral graphs (PAGs) instead of DAGs.

    This is a simplified implementation focusing on the core FCI principles.
    """

    def __init__(
        self,
        independence_test: str = "pearson",
        alpha: float = 0.05,
        max_conditioning_set_size: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        max_iterations: int = 1000,
    ) -> None:
        """Initialize FCI Algorithm.

        Args:
            independence_test: Type of independence test
            alpha: Significance level for independence tests
            max_conditioning_set_size: Maximum size of conditioning sets
            random_state: Random seed
            verbose: Whether to print verbose output
            max_iterations: Maximum iterations
        """
        super().__init__(random_state, verbose, max_iterations)

        self.independence_test = independence_test
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size

        # Use PC algorithm as a component
        self.pc_algorithm = PCAlgorithm(
            independence_test=independence_test,
            alpha=alpha,
            max_conditioning_set_size=max_conditioning_set_size,
            random_state=random_state,
            verbose=verbose,
        )

    def _discover_implementation(self, data: pd.DataFrame) -> DiscoveryResult:
        """Implement FCI algorithm discovery logic."""
        start_time = time.time()

        # Step 1: Run PC algorithm to get initial structure
        pc_result = self.pc_algorithm._discover_implementation(data)

        if self.verbose:
            print("FCI: Completed PC phase")

        # Step 2: Apply FCI-specific rules (simplified implementation)
        # In full FCI, this would include:
        # - Possible d-separation tests
        # - Discriminating path rules
        # - Additional orientation rules for PAGs

        # For now, return PC result with FCI labeling
        # A full implementation would require more sophisticated graph representations
        computation_time = time.time() - start_time

        result = DiscoveryResult(
            dag=pc_result.dag,
            algorithm_name="FCI",
            algorithm_parameters={
                "independence_test": self.independence_test,
                "alpha": self.alpha,
                "max_conditioning_set_size": self.max_conditioning_set_size,
            },
            computation_time=computation_time,
            algorithm_diagnostics={
                "note": "Simplified FCI implementation - uses PC with additional labeling",
                **pc_result.algorithm_diagnostics,
            },
        )

        return result
