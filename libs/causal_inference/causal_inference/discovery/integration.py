"""Integration between causal discovery and causal inference estimators.

This module provides functions to seamlessly integrate discovered causal
structures with existing causal inference estimators in the library.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    OutcomeData,
    TreatmentData,
)
from .base import DiscoveryResult
from .utils import discover_confounders

__all__ = [
    "DiscoveryEstimatorPipeline",
    "estimate_causal_effect_from_discovery",
    "validate_discovery_assumptions",
]


class DiscoveryEstimatorPipeline:
    """Pipeline that combines causal discovery with causal effect estimation.

    This class provides an end-to-end workflow that:
    1. Discovers causal structure from data
    2. Identifies appropriate adjustment sets
    3. Estimates causal effects using discovered structure
    """

    def __init__(
        self,
        discovery_algorithm: Any,
        estimator: BaseEstimator,
        verbose: bool = False,
    ) -> None:
        """Initialize the discovery-estimation pipeline.

        Args:
            discovery_algorithm: Causal discovery algorithm instance
            estimator: Causal inference estimator instance
            verbose: Whether to print verbose output
        """
        self.discovery_algorithm = discovery_algorithm
        self.estimator = estimator
        self.verbose = verbose

        # State
        self.discovery_result: Optional[DiscoveryResult] = None
        self.causal_effect: Optional[CausalEffect] = None
        self._is_fitted = False

    def fit_and_estimate(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        treatment_type: str = "binary",
        outcome_type: str = "continuous",
    ) -> CausalEffect:
        """Discover causal structure and estimate causal effect.

        Args:
            data: Input data with all variables
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            treatment_type: Type of treatment variable
            outcome_type: Type of outcome variable

        Returns:
            CausalEffect with estimated treatment effect
        """
        if self.verbose:
            print("Step 1: Discovering causal structure...")

        # Step 1: Discover causal structure
        self.discovery_result = self.discovery_algorithm.discover(data)

        if self.verbose:
            print(f"Discovered DAG with {self.discovery_result.dag.n_edges} edges")
            print("Step 2: Identifying confounders...")

        # Step 2: Identify confounders using discovered structure
        confounders = discover_confounders(
            self.discovery_result.dag, treatment_col, outcome_col, data
        )

        if self.verbose:
            print(f"Found {len(confounders.names)} confounders: {confounders.names}")
            print("Step 3: Estimating causal effect...")

        # Step 3: Prepare data for estimation
        treatment_data = TreatmentData(
            values=data[treatment_col],
            name=treatment_col,
            treatment_type=treatment_type,
        )

        outcome_data = OutcomeData(
            values=data[outcome_col],
            name=outcome_col,
            outcome_type=outcome_type,
        )

        # Step 4: Fit estimator and estimate causal effect
        self.estimator.fit(treatment_data, outcome_data, confounders)
        self.causal_effect = self.estimator.estimate_ate()
        self._is_fitted = True

        if self.verbose:
            print(f"Estimated ATE: {self.causal_effect.ate:.4f}")
            if self.causal_effect.ate_ci_lower is not None:
                print(
                    f"95% CI: [{self.causal_effect.ate_ci_lower:.4f}, {self.causal_effect.ate_ci_upper:.4f}]"
                )

        return self.causal_effect

    def get_discovery_result(self) -> Optional[DiscoveryResult]:
        """Get the causal discovery result."""
        return self.discovery_result

    def get_causal_effect(self) -> Optional[CausalEffect]:
        """Get the estimated causal effect."""
        return self.causal_effect

    def summary(self) -> str:
        """Get a summary of the pipeline results."""
        if not self._is_fitted:
            return "DiscoveryEstimatorPipeline (not fitted)"

        lines = [
            "=== Discovery-Estimation Pipeline Summary ===",
            "",
            "Discovery Results:",
            f"  Algorithm: {self.discovery_result.algorithm_name}",
            f"  Variables: {self.discovery_result.dag.n_variables}",
            f"  Edges: {self.discovery_result.dag.n_edges}",
            f"  Edge Density: {self.discovery_result.dag.edge_density:.3f}",
            f"  Computation Time: {self.discovery_result.computation_time:.2f}s",
            "",
            "Estimation Results:",
            f"  Estimator: {self.estimator.__class__.__name__}",
            f"  ATE: {self.causal_effect.ate:.4f}",
        ]

        if self.causal_effect.ate_ci_lower is not None:
            lines.append(
                f"  95% CI: [{self.causal_effect.ate_ci_lower:.4f}, {self.causal_effect.ate_ci_upper:.4f}]"
            )
            lines.append(f"  Significant: {self.causal_effect.is_significant}")

        return "\n".join(lines)


def estimate_causal_effect_from_discovery(
    discovery_result: DiscoveryResult,
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    estimator: BaseEstimator,
    treatment_type: str = "binary",
    outcome_type: str = "continuous",
    verbose: bool = False,
) -> CausalEffect:
    """Estimate causal effect using a discovered causal structure.

    Args:
        discovery_result: Result from causal discovery algorithm
        data: Original data
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        estimator: Causal inference estimator to use
        treatment_type: Type of treatment variable
        outcome_type: Type of outcome variable
        verbose: Whether to print verbose output

    Returns:
        CausalEffect with estimated treatment effect
    """
    if verbose:
        print("Identifying confounders from discovered DAG...")

    # Identify confounders using discovered structure
    confounders = discover_confounders(
        discovery_result.dag, treatment_col, outcome_col, data
    )

    if verbose:
        print(f"Found {len(confounders.names)} confounders: {confounders.names}")
        print("Fitting estimator...")

    # Prepare data
    treatment_data = TreatmentData(
        values=data[treatment_col],
        name=treatment_col,
        treatment_type=treatment_type,
    )

    outcome_data = OutcomeData(
        values=data[outcome_col],
        name=outcome_col,
        outcome_type=outcome_type,
    )

    # Fit estimator and estimate effect
    estimator.fit(treatment_data, outcome_data, confounders)
    causal_effect = estimator.estimate_ate()

    if verbose:
        print(f"Estimated ATE: {causal_effect.ate:.4f}")
        if causal_effect.ate_ci_lower is not None:
            print(
                f"95% CI: [{causal_effect.ate_ci_lower:.4f}, {causal_effect.ate_ci_upper:.4f}]"
            )

    return causal_effect


def validate_discovery_assumptions(
    discovery_result: DiscoveryResult,
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> dict[str, Any]:
    """Validate assumptions for using discovered structure in causal inference.

    Args:
        discovery_result: Result from causal discovery
        data: Original data
        treatment_col: Treatment variable name
        outcome_col: Outcome variable name

    Returns:
        Dictionary with validation results
    """
    dag = discovery_result.dag

    validation_results = {
        "dag_is_acyclic": dag.is_acyclic(),
        "treatment_in_dag": treatment_col in dag.variable_names,
        "outcome_in_dag": outcome_col in dag.variable_names,
        "has_causal_path": False,
        "confounders_identified": False,
        "warnings": [],
        "recommendations": [],
    }

    # Check if treatment and outcome are in the DAG
    if not validation_results["treatment_in_dag"]:
        validation_results["warnings"].append(
            f"Treatment '{treatment_col}' not found in discovered DAG"
        )
        validation_results["recommendations"].append(
            "Include treatment variable in discovery data"
        )
        return validation_results

    if not validation_results["outcome_in_dag"]:
        validation_results["warnings"].append(
            f"Outcome '{outcome_col}' not found in discovered DAG"
        )
        validation_results["recommendations"].append(
            "Include outcome variable in discovery data"
        )
        return validation_results

    # Check for causal path from treatment to outcome
    G = dag.to_networkx()
    try:
        import networkx as nx

        paths = list(nx.all_simple_paths(G, treatment_col, outcome_col))
        validation_results["has_causal_path"] = len(paths) > 0
        validation_results["causal_paths"] = paths

        if not validation_results["has_causal_path"]:
            validation_results["warnings"].append(
                "No causal path from treatment to outcome in discovered DAG"
            )
            validation_results["recommendations"].append(
                "Check discovery algorithm parameters or data quality"
            )

    except Exception:
        validation_results["warnings"].append("Could not analyze causal paths")

    # Check for confounders
    try:
        confounders = discover_confounders(dag, treatment_col, outcome_col, data)
        validation_results["confounders_identified"] = len(confounders.names) > 0
        validation_results["n_confounders"] = len(confounders.names)
        validation_results["confounder_names"] = confounders.names

        if not validation_results["confounders_identified"]:
            validation_results["warnings"].append(
                "No confounders identified - causal effect may be biased"
            )
            validation_results["recommendations"].append(
                "Consider including more variables in discovery or using instrumental variables"
            )

    except Exception as e:
        validation_results["warnings"].append(
            f"Could not identify confounders: {str(e)}"
        )

    # Check discovery quality
    if discovery_result.convergence_achieved is False:
        validation_results["warnings"].append("Discovery algorithm did not converge")
        validation_results["recommendations"].append(
            "Increase max_iterations or check algorithm parameters"
        )

    if discovery_result.dag.edge_density > 0.5:
        validation_results["warnings"].append(
            "High edge density - DAG may be overly complex"
        )
        validation_results["recommendations"].append(
            "Consider increasing regularization or sparsity constraints"
        )

    if discovery_result.dag.edge_density < 0.1:
        validation_results["warnings"].append(
            "Very low edge density - important relationships may be missing"
        )
        validation_results["recommendations"].append(
            "Consider decreasing regularization or significance thresholds"
        )

    # Overall assessment
    validation_results["overall_assessment"] = (
        validation_results["dag_is_acyclic"]
        and validation_results["treatment_in_dag"]
        and validation_results["outcome_in_dag"]
        and validation_results["has_causal_path"]
        and len(validation_results["warnings"]) <= 1
    )

    return validation_results


def compare_estimators_on_discovery(
    discovery_result: DiscoveryResult,
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    estimators: list[BaseEstimator],
    estimator_names: Optional[list[str]] = None,
    verbose: bool = False,
) -> dict[str, CausalEffect]:
    """Compare multiple estimators using the same discovered causal structure.

    Args:
        discovery_result: Result from causal discovery
        data: Original data
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        estimators: List of estimators to compare
        estimator_names: Names for estimators (optional)
        verbose: Whether to print verbose output

    Returns:
        Dictionary mapping estimator names to causal effects
    """
    if estimator_names is None:
        estimator_names = [est.__class__.__name__ for est in estimators]

    if len(estimator_names) != len(estimators):
        raise ValueError("Number of estimator names must match number of estimators")

    results = {}

    for i, (estimator, name) in enumerate(zip(estimators, estimator_names)):
        if verbose:
            print(f"Running estimator {i + 1}/{len(estimators)}: {name}")

        try:
            causal_effect = estimate_causal_effect_from_discovery(
                discovery_result,
                data,
                treatment_col,
                outcome_col,
                estimator,
                verbose=False,
            )
            results[name] = causal_effect

            if verbose:
                print(f"  ATE: {causal_effect.ate:.4f}")
                if causal_effect.ate_ci_lower is not None:
                    print(
                        f"  95% CI: [{causal_effect.ate_ci_lower:.4f}, {causal_effect.ate_ci_upper:.4f}]"
                    )

        except Exception as e:
            if verbose:
                print(f"  Failed: {str(e)}")
            results[name] = None

    return results
