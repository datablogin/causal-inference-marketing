"""Base classes and data structures for causal discovery methods.

This module provides the foundational abstract classes and data models
for causal discovery algorithms that learn DAGs from observational data.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "CausalDAG",
    "DiscoveryResult",
    "BaseDiscoveryAlgorithm",
    "CausalDiscoveryError",
    "DiscoveryDataValidationError",
]


class CausalDAG(BaseModel):
    """Data structure representing a causal directed acyclic graph (DAG).

    This class provides a standardized interface for representing and manipulating
    causal DAGs learned from data discovery algorithms.
    """

    adjacency_matrix: NDArray[Any] = Field(
        ..., description="Adjacency matrix representation of the DAG"
    )
    variable_names: list[str] = Field(
        ..., description="Names of variables corresponding to matrix indices"
    )
    edge_weights: Optional[NDArray[Any]] = Field(
        default=None, description="Edge weights/strengths if applicable"
    )
    confidence_scores: Optional[NDArray[Any]] = Field(
        default=None, description="Confidence scores for each edge"
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("adjacency_matrix")
    @classmethod
    def validate_adjacency_matrix(cls, v: NDArray[Any]) -> NDArray[Any]:
        """Validate adjacency matrix is square and contains valid values."""
        if v.ndim != 2:
            raise ValueError("Adjacency matrix must be 2-dimensional")
        if v.shape[0] != v.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if not np.all(np.isin(v, [0, 1])):
            raise ValueError("Adjacency matrix must contain only 0s and 1s")
        return v

    @field_validator("variable_names")
    @classmethod
    def validate_variable_names(cls, v: list[str]) -> list[str]:
        """Validate variable names are unique and non-empty."""
        if len(v) == 0:
            raise ValueError("Variable names cannot be empty")
        if len(set(v)) != len(v):
            raise ValueError("Variable names must be unique")
        return v

    def __init__(self, **data: Any) -> None:
        """Initialize DAG with validation."""
        super().__init__(**data)

        # Check dimensions match
        n_vars = len(self.variable_names)
        if self.adjacency_matrix.shape != (n_vars, n_vars):
            raise ValueError(
                f"Adjacency matrix shape {self.adjacency_matrix.shape} "
                f"doesn't match number of variables {n_vars}"
            )

        # Check for cycles (DAG constraint)
        if not self.is_acyclic():
            raise ValueError("Graph contains cycles - not a valid DAG")

        # Validate optional arrays
        if self.edge_weights is not None:
            if self.edge_weights.shape != self.adjacency_matrix.shape:
                raise ValueError("Edge weights shape must match adjacency matrix")

        if self.confidence_scores is not None:
            if self.confidence_scores.shape != self.adjacency_matrix.shape:
                raise ValueError("Confidence scores shape must match adjacency matrix")

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic (DAG property)."""
        try:
            # Create NetworkX graph from adjacency matrix
            G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
            return nx.is_directed_acyclic_graph(G)
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            # If we can't check acyclicity due to invalid matrix, assume it's not a DAG
            import warnings

            warnings.warn(f"Could not check DAG acyclicity: {e}", UserWarning)
            return False

    @property
    def n_variables(self) -> int:
        """Number of variables in the DAG."""
        return len(self.variable_names)

    @property
    def n_edges(self) -> int:
        """Number of edges in the DAG."""
        return int(np.sum(self.adjacency_matrix))

    @property
    def edge_density(self) -> float:
        """Edge density of the DAG (edges / possible_edges)."""
        n = self.n_variables
        max_edges = n * (n - 1)  # Maximum edges in DAG
        return self.n_edges / max_edges if max_edges > 0 else 0.0

    def get_parents(self, variable: str | int) -> list[str]:
        """Get parent variables of a given variable."""
        if isinstance(variable, str):
            if variable not in self.variable_names:
                raise ValueError(f"Variable '{variable}' not found in DAG")
            var_idx = self.variable_names.index(variable)
        else:
            var_idx = variable

        parent_indices = np.where(self.adjacency_matrix[:, var_idx] == 1)[0]
        return [self.variable_names[i] for i in parent_indices]

    def get_children(self, variable: str | int) -> list[str]:
        """Get child variables of a given variable."""
        if isinstance(variable, str):
            if variable not in self.variable_names:
                raise ValueError(f"Variable '{variable}' not found in DAG")
            var_idx = self.variable_names.index(variable)
        else:
            var_idx = variable

        child_indices = np.where(self.adjacency_matrix[var_idx, :] == 1)[0]
        return [self.variable_names[i] for i in child_indices]

    def has_edge(self, from_var: str | int, to_var: str | int) -> bool:
        """Check if there's an edge from one variable to another."""
        if isinstance(from_var, str):
            from_idx = self.variable_names.index(from_var)
        else:
            from_idx = from_var

        if isinstance(to_var, str):
            to_idx = self.variable_names.index(to_var)
        else:
            to_idx = to_var

        return bool(self.adjacency_matrix[from_idx, to_idx])

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)

        # Add variable names as node labels
        node_mapping = {i: name for i, name in enumerate(self.variable_names)}
        G = nx.relabel_nodes(G, node_mapping)

        # Add edge weights if available
        if self.edge_weights is not None:
            for i, from_var in enumerate(self.variable_names):
                for j, to_var in enumerate(self.variable_names):
                    if self.adjacency_matrix[i, j] == 1:
                        G[from_var][to_var]["weight"] = self.edge_weights[i, j]

        # Add confidence scores if available
        if self.confidence_scores is not None:
            for i, from_var in enumerate(self.variable_names):
                for j, to_var in enumerate(self.variable_names):
                    if self.adjacency_matrix[i, j] == 1:
                        G[from_var][to_var]["confidence"] = self.confidence_scores[i, j]

        return G

    def get_markov_blanket(self, variable: str | int) -> list[str]:
        """Get the Markov blanket of a variable (parents, children, and co-parents)."""
        if isinstance(variable, str):
            var_name = variable
        else:
            var_name = self.variable_names[variable]

        parents = set(self.get_parents(var_name))
        children = set(self.get_children(var_name))

        # Co-parents: parents of children
        co_parents = set()
        for child in children:
            child_parents = set(self.get_parents(child))
            co_parents.update(child_parents)

        # Remove the variable itself
        markov_blanket = (parents | children | co_parents) - {var_name}
        return list(markov_blanket)

    def structural_hamming_distance(self, other: CausalDAG) -> int:
        """Compute structural Hamming distance between two DAGs."""
        if self.n_variables != other.n_variables:
            raise ValueError("DAGs must have same number of variables")

        return int(np.sum(self.adjacency_matrix != other.adjacency_matrix))


@dataclass
class DiscoveryResult:
    """Result of a causal discovery algorithm.

    Contains the learned DAG along with algorithm-specific diagnostics,
    confidence measures, and performance metrics.
    """

    # Core result
    dag: CausalDAG

    # Algorithm information
    algorithm_name: str
    algorithm_parameters: dict[str, Any]

    # Discovery metrics
    n_iterations: Optional[int] = None
    convergence_achieved: Optional[bool] = None
    computation_time: Optional[float] = None

    # Uncertainty quantification
    bootstrap_dags: Optional[list[CausalDAG]] = None
    edge_probabilities: Optional[NDArray[Any]] = None
    stability_score: Optional[float] = None

    # Validation metrics
    likelihood_score: Optional[float] = None
    bic_score: Optional[float] = None
    aic_score: Optional[float] = None

    # Algorithm-specific diagnostics
    algorithm_diagnostics: Optional[dict[str, Any]] = None

    # Performance on known structure (if available)
    true_dag: Optional[CausalDAG] = None
    structural_hamming_distance: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate discovery result after initialization."""
        if self.bootstrap_dags is not None:
            # Check all bootstrap DAGs have same structure
            n_vars = self.dag.n_variables
            for boot_dag in self.bootstrap_dags:
                if boot_dag.n_variables != n_vars:
                    raise ValueError(
                        "All bootstrap DAGs must have same number of variables"
                    )

        if self.edge_probabilities is not None:
            expected_shape = (self.dag.n_variables, self.dag.n_variables)
            if self.edge_probabilities.shape != expected_shape:
                raise ValueError(
                    f"Edge probabilities shape {self.edge_probabilities.shape} "
                    f"doesn't match DAG shape {expected_shape}"
                )

        # Compute validation metrics if true DAG is provided
        if self.true_dag is not None:
            self._compute_validation_metrics()

    def _compute_validation_metrics(self) -> None:
        """Compute validation metrics against true DAG."""
        if self.true_dag is None:
            return

        # Structural Hamming distance
        self.structural_hamming_distance = self.dag.structural_hamming_distance(
            self.true_dag
        )

        # Precision, recall, F1 for edge detection
        learned_edges = self.dag.adjacency_matrix.flatten()
        true_edges = self.true_dag.adjacency_matrix.flatten()

        true_positives = np.sum((learned_edges == 1) & (true_edges == 1))
        false_positives = np.sum((learned_edges == 1) & (true_edges == 0))
        false_negatives = np.sum((learned_edges == 0) & (true_edges == 1))

        self.precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        self.recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        self.f1_score = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )

    @property
    def summary_stats(self) -> dict[str, Any]:
        """Get summary statistics of the discovery result."""
        stats = {
            "algorithm": self.algorithm_name,
            "n_variables": self.dag.n_variables,
            "n_edges": self.dag.n_edges,
            "edge_density": self.dag.edge_density,
            "converged": self.convergence_achieved,
            "computation_time": self.computation_time,
        }

        if self.structural_hamming_distance is not None:
            stats.update(
                {
                    "structural_hamming_distance": self.structural_hamming_distance,
                    "precision": self.precision,
                    "recall": self.recall,
                    "f1_score": self.f1_score,
                }
            )

        return stats


class CausalDiscoveryError(Exception):
    """Base exception for causal discovery specific errors."""

    pass


class DiscoveryDataValidationError(CausalDiscoveryError):
    """Raised when discovery input data fails validation."""

    pass


class BaseDiscoveryAlgorithm(abc.ABC):
    """Abstract base class for causal discovery algorithms.

    This class establishes the common interface and shared functionality
    for all causal discovery methods in the library.
    """

    def __init__(
        self,
        random_state: int | None = None,
        verbose: bool = False,
        max_iterations: int = 1000,
    ) -> None:
        """Initialize the discovery algorithm.

        Args:
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output during discovery
            max_iterations: Maximum number of iterations for iterative algorithms
        """
        self.random_state = random_state
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.is_fitted = False

        # Data containers
        self.data: pd.DataFrame | None = None
        self.variable_names: list[str] | None = None

        # Results cache
        self._discovery_result: DiscoveryResult | None = None

        # Set random state
        if random_state is not None:
            np.random.seed(random_state)

    @abc.abstractmethod
    def _discover_implementation(self, data: pd.DataFrame) -> DiscoveryResult:
        """Implement the specific discovery logic for this algorithm.

        Args:
            data: Input data as pandas DataFrame

        Returns:
            DiscoveryResult with learned DAG and diagnostics
        """
        pass

    def discover(self, data: pd.DataFrame) -> DiscoveryResult:
        """Discover causal structure from data.

        Args:
            data: Input data as pandas DataFrame with variables as columns

        Returns:
            DiscoveryResult with learned DAG and diagnostics

        Raises:
            DiscoveryDataValidationError: If input data fails validation
        """
        # Validate inputs
        self._validate_data(data)

        # Store data
        self.data = data.copy()
        self.variable_names = list(data.columns)

        # Clear cached results
        self._discovery_result = None

        try:
            # Call the implementation-specific discovery logic
            result = self._discover_implementation(data)
            self.is_fitted = True

            # Cache the result
            self._discovery_result = result

            if self.verbose:
                print(f"Successfully discovered DAG using {self.__class__.__name__}")
                print(
                    f"Found {result.dag.n_edges} edges among {result.dag.n_variables} variables"
                )

            return result

        except Exception as e:
            raise CausalDiscoveryError(
                f"Failed to discover causal structure: {str(e)}"
            ) from e

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data for causal discovery with comprehensive checks.

        Args:
            data: Input data to validate

        Raises:
            DiscoveryDataValidationError: If validation fails
        """
        if not isinstance(data, pd.DataFrame):
            raise DiscoveryDataValidationError("Data must be a pandas DataFrame")

        if data.empty:
            raise DiscoveryDataValidationError("Data cannot be empty")

        n_samples, n_variables = data.shape

        # Basic dimension checks
        if n_variables < 2:
            raise DiscoveryDataValidationError(
                "Need at least 2 variables for causal discovery"
            )

        # Sample size validation based on problem complexity
        min_samples_basic = 10
        min_samples_per_var = 5  # Rule of thumb: 5 samples per variable
        min_samples_for_independence = 30  # For reliable independence tests

        recommended_samples = max(
            min_samples_basic,
            n_variables * min_samples_per_var,
            min_samples_for_independence,
        )

        if n_samples < min_samples_basic:
            raise DiscoveryDataValidationError(
                f"Need at least {min_samples_basic} observations for causal discovery, got {n_samples}"
            )
        elif n_samples < recommended_samples:
            import warnings

            warnings.warn(
                f"Only {n_samples} samples for {n_variables} variables. "
                f"Recommend at least {recommended_samples} samples for reliable results.",
                UserWarning,
            )

        # Check for invalid values
        if data.isnull().any().any():
            missing_cols = data.columns[data.isnull().any()].tolist()
            raise DiscoveryDataValidationError(
                f"Missing values not allowed. Found in columns: {missing_cols}"
            )

        if np.isinf(data.select_dtypes(include=[np.number]).values).any():
            raise DiscoveryDataValidationError("Infinite values not allowed in data")

        # Check for constant variables
        constant_vars = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_vars:
            raise DiscoveryDataValidationError(
                f"Constant variables not allowed: {constant_vars}"
            )

        # Check for highly collinear variables
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            try:
                corr_matrix = numeric_data.corr().abs()
                # Find pairs with correlation > 0.99 (excluding diagonal)
                high_corr_pairs = []
                n_numeric = len(numeric_data.columns)
                for i in range(n_numeric):
                    for j in range(i + 1, n_numeric):
                        if corr_matrix.iloc[i, j] > 0.99:
                            high_corr_pairs.append(
                                (numeric_data.columns[i], numeric_data.columns[j])
                            )

                if high_corr_pairs:
                    import warnings

                    warnings.warn(
                        f"Found highly correlated variable pairs (r > 0.99): {high_corr_pairs}. "
                        "This may cause numerical instability in discovery algorithms.",
                        UserWarning,
                    )
            except Exception:
                # Skip correlation check if it fails
                pass

        # Check data types
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            import warnings

            warnings.warn(
                f"Non-numeric columns detected: {non_numeric_cols.tolist()}. "
                "Most discovery algorithms assume continuous data.",
                UserWarning,
            )

        # Check for extreme values that might cause numerical issues
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            # Check for values that are too large/small
            abs_max = np.abs(numeric_data.values).max()
            if abs_max > 1e10:
                import warnings

                warnings.warn(
                    f"Very large values detected (max absolute value: {abs_max:.2e}). "
                    "Consider scaling data to prevent numerical issues.",
                    UserWarning,
                )

            abs_min = np.abs(numeric_data[numeric_data != 0].values).min()
            if abs_min < 1e-10:
                import warnings

                warnings.warn(
                    f"Very small non-zero values detected (min absolute value: {abs_min:.2e}). "
                    "Consider scaling data to prevent numerical issues.",
                    UserWarning,
                )

        # Algorithm-specific sample size warnings
        algorithm_name = self.__class__.__name__
        if algorithm_name == "PCAlgorithm":
            # PC algorithm needs more samples for higher-order conditioning sets
            max_conditioning_size = getattr(
                self, "max_conditioning_set_size", min(n_variables - 2, 3)
            )
            if max_conditioning_size is not None:
                min_samples_pc = 50 * (
                    2**max_conditioning_size
                )  # Exponential in conditioning set size
            else:
                min_samples_pc = 50 * (2 ** min(n_variables - 2, 3))
            if n_samples < min_samples_pc:
                import warnings

                warnings.warn(
                    f"PC algorithm with conditioning sets up to size {max_conditioning_size} "
                    f"may need {min_samples_pc} samples for reliable results. Got {n_samples}.",
                    UserWarning,
                )
        elif algorithm_name == "NOTEARSAlgorithm":
            # NOTEARS needs more samples for reliable continuous optimization
            min_samples_notears = max(
                100, n_variables * n_variables
            )  # At least p^2 samples
            if n_samples < min_samples_notears:
                import warnings

                warnings.warn(
                    f"NOTEARS algorithm may need at least {min_samples_notears} samples "
                    f"for {n_variables} variables. Got {n_samples}.",
                    UserWarning,
                )

        # Check for excessive missing values
        missing_pct = data.isnull().sum() / len(data)
        high_missing = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing:
            raise DiscoveryDataValidationError(
                f"Variables with >50% missing values: {high_missing}"
            )

    def bootstrap_discovery(
        self,
        data: pd.DataFrame,
        n_bootstrap: int = 100,
        bootstrap_fraction: float = 0.8,
    ) -> DiscoveryResult:
        """Perform bootstrap-based causal discovery for uncertainty quantification.

        Args:
            data: Input data
            n_bootstrap: Number of bootstrap samples
            bootstrap_fraction: Fraction of data to sample in each bootstrap

        Returns:
            DiscoveryResult with bootstrap DAGs and edge probabilities
        """
        if bootstrap_fraction <= 0 or bootstrap_fraction > 1:
            raise ValueError("bootstrap_fraction must be between 0 and 1")

        bootstrap_dags = []
        n_samples = int(len(data) * bootstrap_fraction)

        for i in range(n_bootstrap):
            # Bootstrap sample
            boot_data = data.sample(
                n=n_samples,
                replace=True,
                random_state=self.random_state + i if self.random_state else None,
            )

            try:
                # Discover structure on bootstrap sample
                boot_result = self._discover_implementation(boot_data)
                bootstrap_dags.append(boot_result.dag)

                if self.verbose and (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{n_bootstrap} bootstrap samples")

            except Exception as e:
                if self.verbose:
                    print(f"Bootstrap sample {i + 1} failed: {str(e)}")
                continue

        if not bootstrap_dags:
            raise CausalDiscoveryError("All bootstrap samples failed")

        # Compute consensus DAG and edge probabilities
        consensus_dag, edge_probabilities = self._compute_consensus_dag(bootstrap_dags)

        # Create result with bootstrap information
        result = DiscoveryResult(
            dag=consensus_dag,
            algorithm_name=f"{self.__class__.__name__}_Bootstrap",
            algorithm_parameters={
                "n_bootstrap": n_bootstrap,
                "bootstrap_fraction": bootstrap_fraction,
            },
            bootstrap_dags=bootstrap_dags,
            edge_probabilities=edge_probabilities,
            stability_score=self._compute_stability_score(bootstrap_dags),
        )

        return result

    def _compute_consensus_dag(
        self, bootstrap_dags: list[CausalDAG]
    ) -> tuple[CausalDAG, NDArray[Any]]:
        """Compute consensus DAG from bootstrap samples."""
        if not bootstrap_dags:
            raise ValueError("No bootstrap DAGs provided")

        n_vars = bootstrap_dags[0].n_variables
        variable_names = bootstrap_dags[0].variable_names

        # Compute edge probabilities
        edge_counts = np.zeros((n_vars, n_vars))
        for dag in bootstrap_dags:
            edge_counts += dag.adjacency_matrix

        edge_probabilities = edge_counts / len(bootstrap_dags)

        # Create consensus adjacency matrix (edges with >50% support)
        consensus_adj = (edge_probabilities > 0.5).astype(int)

        # Ensure result is acyclic
        consensus_adj = self._make_acyclic(consensus_adj)

        consensus_dag = CausalDAG(
            adjacency_matrix=consensus_adj,
            variable_names=variable_names,
            confidence_scores=edge_probabilities,
        )

        return consensus_dag, edge_probabilities

    def _make_acyclic(self, adj_matrix: NDArray[Any]) -> NDArray[Any]:
        """Ensure adjacency matrix represents an acyclic graph."""
        # Simple approach: remove edges to break cycles
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

        # Remove edges to break cycles
        while not nx.is_directed_acyclic_graph(G):
            # Find a cycle and remove the edge with lowest weight/probability
            try:
                cycle = nx.find_cycle(G, orientation="original")
                # Remove first edge in cycle
                u, v = cycle[0][:2]
                G.remove_edge(u, v)
                adj_matrix[u, v] = 0
            except nx.NetworkXNoCycle:
                break

        return adj_matrix

    def _compute_stability_score(self, bootstrap_dags: list[CausalDAG]) -> float:
        """Compute stability score across bootstrap samples."""
        if len(bootstrap_dags) < 2:
            return 1.0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(bootstrap_dags)):
            for j in range(i + 1, len(bootstrap_dags)):
                dag1, dag2 = bootstrap_dags[i], bootstrap_dags[j]
                # Jaccard similarity for edges
                adj1, adj2 = (
                    dag1.adjacency_matrix.flatten(),
                    dag2.adjacency_matrix.flatten(),
                )
                intersection = np.sum((adj1 == 1) & (adj2 == 1))
                union = np.sum((adj1 == 1) | (adj2 == 1))
                similarity = intersection / union if union > 0 else 1.0
                similarities.append(similarity)

        return float(np.mean(similarities))
