"""Utility functions for causal discovery algorithms.

This module provides helper functions for visualization, data generation,
and integration with existing causal inference estimators.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core.base import CovariateData
from .base import CausalDAG, DiscoveryResult

__all__ = [
    "plot_dag",
    "plot_discovery_comparison",
    "generate_linear_sem_data",
    "dag_to_adjustment_sets",
    "discover_confounders",
]


def plot_dag(
    dag: CausalDAG,
    layout: str = "spring",
    node_size: int = 1000,
    node_color: str = "lightblue",
    edge_color: str = "gray",
    with_labels: bool = True,
    font_size: int = 12,
    figsize: tuple[int, int] = (10, 8),
    save_path: str | None = None,
) -> Figure:
    """Plot a causal DAG using matplotlib and networkx.

    Args:
        dag: CausalDAG to visualize
        layout: Layout algorithm ('spring', 'circular', 'hierarchical')
        node_size: Size of nodes
        node_color: Color of nodes
        edge_color: Color of edges
        with_labels: Whether to show node labels
        font_size: Font size for labels
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to NetworkX graph
    G = dag.to_networkx()

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "hierarchical":
        # Try to create hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except (ImportError, Exception):
            # Fallback to spring layout
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_color, arrows=True, arrowsize=20, arrowstyle="->", ax=ax
    )

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)

    # Add edge weights if available
    if dag.edge_weights is not None:
        edge_labels = {}
        for i, from_var in enumerate(dag.variable_names):
            for j, to_var in enumerate(dag.variable_names):
                if dag.adjacency_matrix[i, j] == 1:
                    weight = dag.edge_weights[i, j]
                    edge_labels[(from_var, to_var)] = f"{weight:.2f}"

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, font_size=font_size - 2, ax=ax
        )

    ax.set_title(f"Causal DAG ({dag.n_variables} variables, {dag.n_edges} edges)")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_discovery_comparison(
    results: list[DiscoveryResult],
    true_dag: CausalDAG | None = None,
    figsize: tuple[int, int] = (15, 10),
    save_path: str | None = None,
) -> Figure:
    """Plot comparison of multiple causal discovery results.

    Args:
        results: List of DiscoveryResult objects to compare
        true_dag: True DAG for comparison (optional)
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure object
    """
    n_results = len(results)
    n_cols = min(3, n_results + (1 if true_dag else 0))
    n_rows = int(np.ceil((n_results + (1 if true_dag else 0)) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    plot_idx = 0

    # Plot true DAG if provided
    if true_dag is not None:
        ax = axes[plot_idx // n_cols, plot_idx % n_cols]
        G_true = true_dag.to_networkx()
        pos = nx.spring_layout(G_true, seed=42)

        nx.draw_networkx_nodes(
            G_true, pos, node_size=500, node_color="lightgreen", ax=ax
        )
        nx.draw_networkx_edges(
            G_true,
            pos,
            edge_color="black",
            arrows=True,
            arrowsize=15,
            arrowstyle="->",
            ax=ax,
        )
        nx.draw_networkx_labels(G_true, pos, font_size=8, ax=ax)

        ax.set_title("True DAG")
        ax.axis("off")
        plot_idx += 1

    # Plot discovered DAGs
    for i, result in enumerate(results):
        if plot_idx >= n_rows * n_cols:
            break

        ax = axes[plot_idx // n_cols, plot_idx % n_cols]
        G = result.dag.to_networkx()

        # Use same layout as true DAG if available
        if true_dag is not None:
            try:
                G_true = true_dag.to_networkx()
                pos = nx.spring_layout(G_true, seed=42)
            except Exception:
                pos = nx.spring_layout(G, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue", ax=ax)
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=15, arrowstyle="->", ax=ax
        )
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        # Add performance metrics to title if available
        title = result.algorithm_name
        if result.f1_score is not None:
            title += f"\nF1: {result.f1_score:.3f}"
        if result.precision is not None and result.recall is not None:
            title += f"\nP: {result.precision:.3f}, R: {result.recall:.3f}"

        ax.set_title(title)
        ax.axis("off")
        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def generate_linear_sem_data(
    dag: CausalDAG,
    n_samples: int = 1000,
    noise_std: float = 1.0,
    edge_weight_range: tuple[float, float] = (0.5, 2.0),
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate data from a linear structural equation model (SEM).

    Args:
        dag: CausalDAG structure to generate data from
        n_samples: Number of samples to generate
        noise_std: Standard deviation of noise terms
        edge_weight_range: Range for edge weights
        random_state: Random seed

    Returns:
        DataFrame with generated data
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_vars = dag.n_variables
    variable_names = dag.variable_names

    # Generate edge weights if not provided
    if dag.edge_weights is None:
        edge_weights = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                if dag.adjacency_matrix[i, j] == 1:
                    weight = np.random.uniform(
                        edge_weight_range[0], edge_weight_range[1]
                    )
                    if np.random.random() < 0.5:  # Random sign
                        weight *= -1
                    edge_weights[i, j] = weight
    else:
        edge_weights = dag.edge_weights

    # Generate data using topological ordering
    G = dag.to_networkx()
    topological_order = list(nx.topological_sort(G))
    var_to_idx = {var: i for i, var in enumerate(variable_names)}

    # Initialize data
    data = np.zeros((n_samples, n_vars))

    # Generate data in topological order
    for var in topological_order:
        j = var_to_idx[var]

        # Get parents
        parents = [var_to_idx[parent] for parent in dag.get_parents(var)]

        if not parents:
            # Root node - generate from noise
            data[:, j] = np.random.normal(0, noise_std, n_samples)
        else:
            # Linear combination of parents plus noise
            linear_combination = np.zeros(n_samples)
            for parent_idx in parents:
                weight = edge_weights[parent_idx, j]
                linear_combination += weight * data[:, parent_idx]

            noise = np.random.normal(0, noise_std, n_samples)
            data[:, j] = linear_combination + noise

    return pd.DataFrame(data, columns=variable_names)


def dag_to_adjustment_sets(
    dag: CausalDAG, treatment: str, outcome: str
) -> dict[str, Any]:
    """Find adjustment sets for causal effect identification from a DAG.

    Args:
        dag: CausalDAG representing causal structure
        treatment: Name of treatment variable
        outcome: Name of outcome variable

    Returns:
        Dictionary with different types of adjustment sets
    """
    if treatment not in dag.variable_names:
        raise ValueError(f"Treatment '{treatment}' not found in DAG")
    if outcome not in dag.variable_names:
        raise ValueError(f"Outcome '{outcome}' not found in DAG")

    G = dag.to_networkx()

    # Find all paths from treatment to outcome
    try:
        all_paths = list(nx.all_simple_paths(G, treatment, outcome))
    except nx.NetworkXNoPath:
        all_paths = []

    # Backdoor criterion: find confounders
    # Simplified implementation - find common causes
    treatment_ancestors = set(nx.ancestors(G, treatment)) | {treatment}
    outcome_ancestors = set(nx.ancestors(G, outcome)) | {outcome}

    # Variables that are ancestors of both treatment and outcome (potential confounders)
    potential_confounders = (treatment_ancestors & outcome_ancestors) - {
        treatment,
        outcome,
    }

    # Remove descendants of treatment (colliders/mediators)
    treatment_descendants = set(nx.descendants(G, treatment))
    backdoor_confounders = potential_confounders - treatment_descendants

    # Minimal adjustment set (simplified)
    # In practice, would need full backdoor criterion algorithm
    minimal_set = list(backdoor_confounders)

    # Parents of treatment (direct confounders)
    treatment_parents = list(dag.get_parents(treatment))

    # All potential confounders
    all_confounders = list(potential_confounders)

    return {
        "minimal_adjustment_set": minimal_set,
        "treatment_parents": treatment_parents,
        "all_potential_confounders": all_confounders,
        "backdoor_confounders": list(backdoor_confounders),
        "causal_paths": all_paths,
    }


def discover_confounders(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    data: pd.DataFrame,
) -> CovariateData:
    """Discover confounders from DAG and return as CovariateData for estimators.

    Args:
        dag: Discovered causal DAG
        treatment: Treatment variable name
        outcome: Outcome variable name
        data: Original data

    Returns:
        CovariateData with discovered confounders
    """
    adjustment_sets = dag_to_adjustment_sets(dag, treatment, outcome)

    # Use minimal adjustment set as confounders
    confounder_names = adjustment_sets["minimal_adjustment_set"]

    if not confounder_names:
        # No confounders found, return empty CovariateData
        empty_df = pd.DataFrame(index=data.index)
        return CovariateData(values=empty_df, names=[])

    # Extract confounder data
    confounder_data = data[confounder_names]

    return CovariateData(values=confounder_data, names=confounder_names)


def compare_dags(dag1: CausalDAG, dag2: CausalDAG) -> dict[str, Any]:
    """Compare two causal DAGs and compute similarity metrics.

    Args:
        dag1: First DAG
        dag2: Second DAG

    Returns:
        Dictionary with comparison metrics
    """
    if dag1.n_variables != dag2.n_variables:
        raise ValueError("DAGs must have same number of variables")

    # Adjacency matrices
    adj1 = dag1.adjacency_matrix
    adj2 = dag2.adjacency_matrix

    # Edge-level metrics
    adj1_flat = adj1.flatten()
    adj2_flat = adj2.flatten()

    true_positives = np.sum((adj1_flat == 1) & (adj2_flat == 1))
    false_positives = np.sum((adj1_flat == 0) & (adj2_flat == 1))
    false_negatives = np.sum((adj1_flat == 1) & (adj2_flat == 0))
    true_negatives = np.sum((adj1_flat == 0) & (adj2_flat == 0))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Structural Hamming Distance
    shd = dag1.structural_hamming_distance(dag2)

    # Jaccard similarity
    intersection = np.sum((adj1_flat == 1) & (adj2_flat == 1))
    union = np.sum((adj1_flat == 1) | (adj2_flat == 1))
    jaccard = intersection / union if union > 0 else 1.0

    return {
        "structural_hamming_distance": shd,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "jaccard_similarity": jaccard,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }
