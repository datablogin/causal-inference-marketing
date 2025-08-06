"""Score-based causal discovery algorithms.

This module implements score-based methods that search over the space of
causal graphs by optimizing scoring functions like BIC, AIC, or likelihood.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from .base import BaseDiscoveryAlgorithm, CausalDAG, DiscoveryResult

__all__ = ["GESAlgorithm", "NOTEARSAlgorithm"]


class GESAlgorithm(BaseDiscoveryAlgorithm):
    """Greedy Equivalence Search (GES) algorithm for causal discovery.

    GES is a score-based algorithm that greedily searches over the space of
    equivalence classes of DAGs by optimizing a scoring function.

    Reference: Chickering, D. M. (2002). Optimal structure identification
    with greedy search. Journal of machine learning research, 3, 507-554.
    """

    def __init__(
        self,
        score_function: str = "bic",
        max_parents: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        max_iterations: int = 1000,
    ) -> None:
        """Initialize GES Algorithm.

        Args:
            score_function: Scoring function ('bic', 'aic', 'likelihood')
            max_parents: Maximum number of parents per node
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
            max_iterations: Maximum number of iterations
        """
        super().__init__(random_state, verbose, max_iterations)

        self.score_function = score_function
        self.max_parents = max_parents

        # Set scoring function
        self._score_func = self._get_score_function()

        # Algorithm state
        self.current_graph: NDArray[Any] | None = None
        self.best_score: float = -np.inf
        self.score_history: list[float] = []
        self._node_scores_cache: dict[
            int, float
        ] = {}  # Cache for individual node scores
        self._data_cache: pd.DataFrame | None = None

    def _get_score_function(self) -> Callable:
        """Get the appropriate scoring function."""
        if self.score_function == "bic":
            return self._bic_score
        elif self.score_function == "aic":
            return self._aic_score
        elif self.score_function == "likelihood":
            return self._likelihood_score
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")

    def _bic_score(self, data: pd.DataFrame, graph: NDArray[Any]) -> float:
        """Compute BIC score for a given graph structure."""
        n_samples, n_vars = data.shape
        total_score = 0.0

        for j in range(n_vars):
            # Find parents of node j
            parents = np.where(graph[:, j] == 1)[0]

            if len(parents) == 0:
                # No parents - just compute variance
                var_j = np.var(data.iloc[:, j])
                if var_j <= 0:
                    var_j = 1e-6
                score_j = -0.5 * n_samples * np.log(2 * np.pi * var_j) - 0.5 * n_samples
                # BIC penalty: -0.5 * k * log(n) where k=1 (intercept only)
                score_j -= 0.5 * 1 * np.log(n_samples)
            else:
                # Linear regression of j on its parents
                X = data.iloc[:, parents].values
                y = data.iloc[:, j].values

                try:
                    # Add intercept
                    np.column_stack([np.ones(n_samples), X])

                    # Fit linear regression
                    reg = LinearRegression().fit(X, y)
                    y_pred = reg.predict(X)
                    mse = mean_squared_error(y, y_pred)

                    if mse <= 0:
                        mse = 1e-6

                    # Log-likelihood
                    log_likelihood = (
                        -0.5 * n_samples * np.log(2 * np.pi * mse) - 0.5 * n_samples
                    )

                    # BIC penalty: -0.5 * k * log(n) where k = number of parameters
                    n_params = len(parents) + 1  # coefficients + intercept
                    bic_penalty = -0.5 * n_params * np.log(n_samples)

                    score_j = log_likelihood + bic_penalty

                except (np.linalg.LinAlgError, ValueError):
                    score_j = -np.inf

            total_score += score_j

        return total_score

    def _compute_node_score(
        self, data: pd.DataFrame, node: int, graph: NDArray[Any]
    ) -> float:
        """Compute score for a single node given its parents in the graph."""
        n_samples = len(data)
        parents = np.where(graph[:, node] == 1)[0]

        if len(parents) == 0:
            # No parents - just compute variance
            var_node = np.var(data.iloc[:, node])
            if var_node <= 0:
                var_node = 1e-6
            score = -0.5 * n_samples * np.log(2 * np.pi * var_node) - 0.5 * n_samples

            # Add penalty based on score function
            if self.score_function == "bic":
                score -= 0.5 * 1 * np.log(n_samples)  # BIC penalty for intercept
            elif self.score_function == "aic":
                score -= 1  # AIC penalty for intercept

        else:
            # Linear regression of node on its parents
            X = data.iloc[:, parents].values
            y = data.iloc[:, node].values

            try:
                reg = LinearRegression().fit(X, y)
                y_pred = reg.predict(X)
                mse = mean_squared_error(y, y_pred)

                if mse <= 0:
                    mse = 1e-6

                # Log-likelihood
                log_likelihood = (
                    -0.5 * n_samples * np.log(2 * np.pi * mse) - 0.5 * n_samples
                )

                # Add penalty based on score function
                n_params = len(parents) + 1  # coefficients + intercept
                if self.score_function == "bic":
                    penalty = -0.5 * n_params * np.log(n_samples)
                elif self.score_function == "aic":
                    penalty = -n_params
                else:  # likelihood
                    penalty = 0

                score = log_likelihood + penalty

            except (np.linalg.LinAlgError, ValueError):
                score = -np.inf

        return score

    def _compute_score_change_add_edge(
        self, data: pd.DataFrame, parent: int, child: int, current_graph: NDArray[Any]
    ) -> float:
        """Compute the change in score when adding an edge parent -> child."""
        # Only the child node's score changes when adding a parent
        old_score = self._node_scores_cache.get(
            child, self._compute_node_score(data, child, current_graph)
        )

        # Create test graph with new edge
        test_graph = current_graph.copy()
        test_graph[parent, child] = 1

        new_score = self._compute_node_score(data, child, test_graph)

        return new_score - old_score

    def _compute_score_change_remove_edge(
        self, data: pd.DataFrame, parent: int, child: int, current_graph: NDArray[Any]
    ) -> float:
        """Compute the change in score when removing an edge parent -> child."""
        # Only the child node's score changes when removing a parent
        old_score = self._node_scores_cache.get(
            child, self._compute_node_score(data, child, current_graph)
        )

        # Create test graph without edge
        test_graph = current_graph.copy()
        test_graph[parent, child] = 0

        new_score = self._compute_node_score(data, child, test_graph)

        return new_score - old_score

    def _update_node_score_cache(
        self, data: pd.DataFrame, node: int, graph: NDArray[Any]
    ) -> None:
        """Update the cached score for a specific node."""
        self._node_scores_cache[node] = self._compute_node_score(data, node, graph)

    def _initialize_score_cache(self, data: pd.DataFrame, graph: NDArray[Any]) -> None:
        """Initialize the score cache for all nodes."""
        n_vars = graph.shape[0]
        self._node_scores_cache.clear()
        for node in range(n_vars):
            self._node_scores_cache[node] = self._compute_node_score(data, node, graph)

    def _aic_score(self, data: pd.DataFrame, graph: NDArray[Any]) -> float:
        """Compute AIC score for a given graph structure."""
        n_samples, n_vars = data.shape
        total_score = 0.0

        for j in range(n_vars):
            parents = np.where(graph[:, j] == 1)[0]

            if len(parents) == 0:
                var_j = np.var(data.iloc[:, j])
                if var_j <= 0:
                    var_j = 1e-6
                score_j = -0.5 * n_samples * np.log(2 * np.pi * var_j) - 0.5 * n_samples
                # AIC penalty: -k where k=1
                score_j -= 1
            else:
                X = data.iloc[:, parents].values
                y = data.iloc[:, j].values

                try:
                    reg = LinearRegression().fit(X, y)
                    y_pred = reg.predict(X)
                    mse = mean_squared_error(y, y_pred)

                    if mse <= 0:
                        mse = 1e-6

                    log_likelihood = (
                        -0.5 * n_samples * np.log(2 * np.pi * mse) - 0.5 * n_samples
                    )

                    # AIC penalty: -k where k = number of parameters
                    n_params = len(parents) + 1
                    aic_penalty = -n_params

                    score_j = log_likelihood + aic_penalty

                except (np.linalg.LinAlgError, ValueError):
                    score_j = -np.inf

            total_score += score_j

        return total_score

    def _likelihood_score(self, data: pd.DataFrame, graph: NDArray[Any]) -> float:
        """Compute log-likelihood score for a given graph structure."""
        n_samples, n_vars = data.shape
        total_score = 0.0

        for j in range(n_vars):
            parents = np.where(graph[:, j] == 1)[0]

            if len(parents) == 0:
                var_j = np.var(data.iloc[:, j])
                if var_j <= 0:
                    var_j = 1e-6
                score_j = -0.5 * n_samples * np.log(2 * np.pi * var_j) - 0.5 * n_samples
            else:
                X = data.iloc[:, parents].values
                y = data.iloc[:, j].values

                try:
                    reg = LinearRegression().fit(X, y)
                    y_pred = reg.predict(X)
                    mse = mean_squared_error(y, y_pred)

                    if mse <= 0:
                        mse = 1e-6

                    score_j = (
                        -0.5 * n_samples * np.log(2 * np.pi * mse) - 0.5 * n_samples
                    )

                except (np.linalg.LinAlgError, ValueError):
                    score_j = -np.inf

            total_score += score_j

        return total_score

    def _discover_implementation(self, data: pd.DataFrame) -> DiscoveryResult:
        """Implement GES algorithm discovery logic."""
        start_time = time.time()

        n_vars = len(data.columns)
        variable_names = list(data.columns)

        # Initialize with empty graph
        self.current_graph = np.zeros((n_vars, n_vars))
        self._data_cache = data
        self._initialize_score_cache(data, self.current_graph)
        self.best_score = sum(self._node_scores_cache.values())
        self.score_history = [self.best_score]

        if self.verbose:
            print(f"Starting GES with {n_vars} variables")
            print(f"Initial score: {self.best_score:.2f}")

        # Phase 1: Forward search (add edges)
        improved = True
        iteration = 0

        while improved and iteration < self.max_iterations:
            improved = False
            best_addition = None
            best_addition_score = self.best_score

            # Try adding each possible edge
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j or self.current_graph[i, j] == 1:
                        continue

                    # Check parent limit
                    if (
                        self.max_parents is not None
                        and np.sum(self.current_graph[:, j]) >= self.max_parents
                    ):
                        continue

                    # Try adding edge i -> j
                    test_graph = self.current_graph.copy()
                    test_graph[i, j] = 1

                    # Check if result is acyclic
                    if not self._is_acyclic(test_graph):
                        continue

                    # Compute score change incrementally
                    score_change = self._compute_score_change_add_edge(
                        data, i, j, self.current_graph
                    )
                    new_score = self.best_score + score_change

                    if new_score > best_addition_score:
                        best_addition = (i, j)
                        best_addition_score = new_score
                        improved = True

            # Apply best addition
            if improved:
                i, j = best_addition
                self.current_graph[i, j] = 1
                self.best_score = best_addition_score
                self.score_history.append(self.best_score)
                # Update cache for the affected node
                self._update_node_score_cache(data, j, self.current_graph)

                if self.verbose:
                    print(
                        f"Iteration {iteration}: Added edge {variable_names[i]} -> {variable_names[j]}, "
                        f"Score: {self.best_score:.2f}"
                    )

            iteration += 1

        if self.verbose:
            print(f"Forward phase completed after {iteration} iterations")

        # Phase 2: Backward search (remove edges)
        improved = True

        while improved and iteration < self.max_iterations:
            improved = False
            best_removal = None
            best_removal_score = self.best_score

            # Try removing each existing edge
            edges = [
                (i, j)
                for i in range(n_vars)
                for j in range(n_vars)
                if self.current_graph[i, j] == 1
            ]

            for i, j in edges:
                # Compute score change incrementally
                score_change = self._compute_score_change_remove_edge(
                    data, i, j, self.current_graph
                )
                new_score = self.best_score + score_change

                if new_score > best_removal_score:
                    best_removal = (i, j)
                    best_removal_score = new_score
                    improved = True

            # Apply best removal
            if improved:
                i, j = best_removal
                self.current_graph[i, j] = 0
                self.best_score = best_removal_score
                self.score_history.append(self.best_score)
                # Update cache for the affected node
                self._update_node_score_cache(data, j, self.current_graph)

                if self.verbose:
                    print(
                        f"Iteration {iteration}: Removed edge {variable_names[i]} -> {variable_names[j]}, "
                        f"Score: {self.best_score:.2f}"
                    )

            iteration += 1

        if self.verbose:
            print(f"Backward phase completed after {iteration} total iterations")

        computation_time = time.time() - start_time

        # Create DAG
        dag = CausalDAG(
            adjacency_matrix=self.current_graph, variable_names=variable_names
        )

        # Create result
        result = DiscoveryResult(
            dag=dag,
            algorithm_name="GES",
            algorithm_parameters={
                "score_function": self.score_function,
                "max_parents": self.max_parents,
            },
            n_iterations=iteration,
            convergence_achieved=not improved,
            computation_time=computation_time,
            likelihood_score=self.best_score
            if self.score_function == "likelihood"
            else None,
            bic_score=self.best_score if self.score_function == "bic" else None,
            aic_score=self.best_score if self.score_function == "aic" else None,
            algorithm_diagnostics={
                "score_history": self.score_history,
                "final_score": self.best_score,
                "n_edges": dag.n_edges,
                "edge_density": dag.edge_density,
            },
        )

        return result

    def _is_acyclic(self, graph: NDArray[Any]) -> bool:
        """Check if a directed graph is acyclic using DFS."""
        n_vars = graph.shape[0]
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = np.zeros(n_vars, dtype=int)

        def dfs(node: int) -> bool:
            colors[node] = GRAY

            for neighbor in range(n_vars):
                if graph[node, neighbor] == 1:
                    if colors[neighbor] == GRAY:  # Back edge - cycle detected
                        return False
                    if colors[neighbor] == WHITE and not dfs(neighbor):
                        return False

            colors[node] = BLACK
            return True

        for node in range(n_vars):
            if colors[node] == WHITE:
                if not dfs(node):
                    return False

        return True


class NOTEARSAlgorithm(BaseDiscoveryAlgorithm):
    """NOTEARS algorithm for causal discovery via continuous optimization.

    NOTEARS formulates causal discovery as a smooth optimization problem
    by using a continuous relaxation of the acyclicity constraint.

    Reference: Zheng, X. et al. (2018). DAGs with NO TEARS: Continuous
    optimization for structure learning. NeurIPS.
    """

    def __init__(
        self,
        lambda_l1: float = 0.1,
        lambda_l2: float = 0.01,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
        random_state: int | None = None,
        verbose: bool = False,
        max_iterations: int = 1000,
    ) -> None:
        """Initialize NOTEARS Algorithm.

        Args:
            lambda_l1: L1 regularization parameter for sparsity
            lambda_l2: L2 regularization parameter
            max_iter: Maximum iterations for augmented Lagrangian
            h_tol: Tolerance for acyclicity constraint
            rho_max: Maximum penalty coefficient
            w_threshold: Threshold for edge weights to determine final graph
            random_state: Random seed
            verbose: Whether to print verbose output
            max_iterations: Maximum iterations (used as max_iter here)
        """
        super().__init__(random_state, verbose, max_iterations)

        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

        # Algorithm state
        self.W_est: NDArray[Any] | None = None
        self.optimization_history: list[dict[str, Any]] = []
        self.max_history_size = 100  # Limit history storage

    def _discover_implementation(self, data: pd.DataFrame) -> DiscoveryResult:
        """Implement NOTEARS algorithm discovery logic."""
        start_time = time.time()

        X = data.values
        n, d = X.shape
        variable_names = list(data.columns)

        if self.verbose:
            print(f"Starting NOTEARS with {d} variables and {n} samples")

        # Initialize weight matrix
        W_est = np.zeros((d, d))

        # Augmented Lagrangian parameters
        rho, alpha, h = 1.0, 0.0, np.inf

        for iteration in range(self.max_iter):
            # Solve the optimization subproblem
            W_new, h_new = self._solve_subproblem(X, W_est, rho, alpha)

            # Check convergence
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break

            # Update dual variable
            alpha += rho * h_new

            # Update primal variable
            W_est = W_new
            h = h_new

            # Store optimization history with size limit
            history_entry = {
                "iteration": iteration,
                "h_value": h,
                "rho": rho,
                "alpha": alpha,
                "nnz": np.sum(np.abs(W_est) > self.w_threshold),
            }
            self.optimization_history.append(history_entry)

            # Limit history size to prevent memory issues
            if len(self.optimization_history) > self.max_history_size:
                # Keep first few and last entries
                keep_first = self.max_history_size // 4
                keep_last = self.max_history_size - keep_first
                self.optimization_history = (
                    self.optimization_history[:keep_first]
                    + self.optimization_history[-keep_last:]
                )

            if self.verbose and iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: h = {h:.6f}, nnz = {np.sum(np.abs(W_est) > self.w_threshold)}"
                )

            # Check convergence
            if h <= self.h_tol or rho >= self.rho_max:
                break

        self.W_est = W_est

        # Threshold weights to get final adjacency matrix
        W_thresh = np.abs(W_est) * (np.abs(W_est) > self.w_threshold)
        adj_matrix = (W_thresh > 0).astype(int)

        computation_time = time.time() - start_time

        if self.verbose:
            print(f"NOTEARS completed in {iteration + 1} iterations")
            print(f"Final h value: {h:.6f}")
            print(f"Found {np.sum(adj_matrix)} edges")

        # Create DAG
        dag = CausalDAG(
            adjacency_matrix=adj_matrix,
            variable_names=variable_names,
            edge_weights=W_thresh,
        )

        # Create result
        result = DiscoveryResult(
            dag=dag,
            algorithm_name="NOTEARS",
            algorithm_parameters={
                "lambda_l1": self.lambda_l1,
                "lambda_l2": self.lambda_l2,
                "w_threshold": self.w_threshold,
                "h_tol": self.h_tol,
            },
            n_iterations=iteration + 1,
            convergence_achieved=h <= self.h_tol,
            computation_time=computation_time,
            algorithm_diagnostics={
                "final_h_value": h,
                "final_rho": rho,
                "optimization_history": self.optimization_history,
                "n_edges": dag.n_edges,
                "edge_density": dag.edge_density,
            },
        )

        return result

    def _solve_subproblem(
        self, x: NDArray[Any], w: NDArray[Any], rho: float, alpha: float
    ) -> tuple[NDArray[Any], float]:
        """Solve the optimization subproblem for given penalty parameters."""

        def objective(w_vec: NDArray[Any]) -> float:
            """Objective function for optimization."""
            w_mat = w_vec.reshape(w.shape)

            # Likelihood term (least squares)
            residual = x - x @ w_mat
            likelihood = 0.5 / x.shape[0] * np.sum(residual**2)

            # Regularization terms
            l1_reg = self.lambda_l1 * np.sum(np.abs(w_mat))
            l2_reg = self.lambda_l2 * np.sum(w_mat**2)

            # Acyclicity constraint (augmented Lagrangian)
            h = self._h_func(w_mat)
            augmented_lagrangian = alpha * h + 0.5 * rho * h**2

            return likelihood + l1_reg + l2_reg + augmented_lagrangian

        def gradient(w_vec: NDArray[Any]) -> NDArray[Any]:
            """Gradient of the objective function."""
            w_mat = w_vec.reshape(w.shape)
            n, d = x.shape

            # Likelihood gradient
            grad_likelihood = -1.0 / n * x.T @ (x - x @ w_mat)

            # L2 gradient
            grad_l2 = 2 * self.lambda_l2 * w_mat

            # Acyclicity gradient
            h = self._h_func(w_mat)
            grad_h = self._grad_h_func(w_mat)
            grad_augmented = (alpha + rho * h) * grad_h

            # L1 is handled by the optimization method
            total_grad = grad_likelihood + grad_l2 + grad_augmented

            return total_grad.flatten()

        # Optimize using L-BFGS-B (handles L1 via proximal operator approximation)
        w_init = w.flatten()

        try:
            result = optimize.minimize(
                objective,
                w_init,
                method="L-BFGS-B",
                jac=gradient,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

            w_new = result.x.reshape(w.shape)

            # Apply soft thresholding for L1 regularization
            w_new = self._soft_threshold(w_new, self.lambda_l1)

        except (ValueError, np.linalg.LinAlgError, optimize.OptimizeWarning) as e:
            # Fallback to gradient descent if L-BFGS-B fails
            if self.verbose:
                print(
                    f"L-BFGS-B optimization failed ({type(e).__name__}: {e}), falling back to gradient descent"
                )
            w_new = self._gradient_descent_step(x, w, rho, alpha)

        # Remove self-loops
        np.fill_diagonal(w_new, 0)

        h_new = self._h_func(w_new)

        return w_new, h_new

    def _h_func(self, w: NDArray[Any]) -> float:
        """Acyclicity constraint function h(W) = tr(e^(W*W)) - d."""
        d = w.shape[0]
        M = w * w

        # Check for numerical stability
        max_eigenval = np.max(np.real(np.linalg.eigvals(M)))
        if max_eigenval > 50:  # Prevent overflow in matrix exponential
            # Use scaling and squaring for large matrices
            from scipy.linalg import expm

            return np.trace(expm(M)) - d

        try:
            # Use scipy's robust matrix exponential when available
            from scipy.linalg import expm

            return np.trace(expm(M)) - d
        except ImportError:
            # Fallback to series expansion with convergence check
            E = np.eye(d)
            M_power = M.copy()

            for i in range(1, min(d, 20)):  # Limit iterations
                E += M_power / np.math.factorial(i)
                M_power = M_power @ M

                # Check convergence
                if np.max(np.abs(M_power)) / np.math.factorial(i + 1) < 1e-12:
                    break

            return np.trace(E) - d

    def _grad_h_func(self, w: NDArray[Any]) -> NDArray[Any]:
        """Gradient of the acyclicity constraint function."""
        d = w.shape[0]
        M = w * w

        try:
            # Use scipy's robust matrix exponential when available
            from scipy.linalg import expm

            E = expm(M)
        except ImportError:
            # Fallback to series expansion with improved convergence
            E = np.eye(d)
            M_power = M.copy()

            for i in range(1, min(d, 15)):  # Limit iterations
                E += M_power / np.math.factorial(i)
                M_power_next = M_power @ M

                # Check convergence
                if np.max(np.abs(M_power_next)) / np.math.factorial(i + 1) < 1e-12:
                    break
                M_power = M_power_next

        # Gradient: 2 * W * (e^(W*W))^T
        return 2 * w * E.T

    def _soft_threshold(self, w: NDArray[Any], threshold: float) -> NDArray[Any]:
        """Apply soft thresholding for L1 regularization."""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

    def _gradient_descent_step(
        self, x: NDArray[Any], w: NDArray[Any], rho: float, alpha: float
    ) -> NDArray[Any]:
        """Simple gradient descent step as fallback."""
        n, d = x.shape

        # Compute gradients
        residual = x - x @ w
        grad_likelihood = -1.0 / n * x.T @ residual
        grad_l2 = 2 * self.lambda_l2 * w

        h = self._h_func(w)
        grad_h = self._grad_h_func(w)
        grad_augmented = (alpha + rho * h) * grad_h

        total_grad = grad_likelihood + grad_l2 + grad_augmented

        # Gradient step with fixed step size
        step_size = 0.01
        w_new = w - step_size * total_grad

        # Apply soft thresholding
        w_new = self._soft_threshold(w_new, step_size * self.lambda_l1)

        return w_new
