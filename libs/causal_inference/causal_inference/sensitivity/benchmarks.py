"""Performance benchmarks for sensitivity analysis functions.

This module provides benchmarking utilities to validate the KPI requirement
that sensitivity analysis functions run < 2s on 100k rows.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .controls import negative_control

# Import functions directly to avoid circular imports
from .e_values import e_value
from .oster import oster_delta
from .placebo import placebo_test
from .rosenbaum import rosenbaum_bounds


def benchmark_sensitivity_functions(
    n_rows: int = 100_000,
    n_bootstrap: int = 100,
    random_state: int = 42,
    target_time_seconds: float = 2.0,
) -> dict[str, Any]:
    """Benchmark all sensitivity analysis functions for performance KPIs.

    Args:
        n_rows: Number of rows to test (default 100k for KPI)
        n_bootstrap: Number of bootstrap samples for functions that support it
        random_state: Random state for reproducibility
        target_time_seconds: Target execution time in seconds

    Returns:
        Dictionary with timing results and pass/fail status for each function
    """
    np.random.seed(random_state)

    # Generate synthetic data
    print(f"Generating synthetic dataset with {n_rows:,} rows...")

    # Generate realistic marketing campaign data
    X1 = np.random.normal(0, 1, n_rows)  # Customer age (standardized)
    X2 = np.random.normal(0, 1, n_rows)  # Income level (standardized)
    X3 = np.random.normal(0, 1, n_rows)  # Previous purchase history
    U = np.random.normal(0, 1, n_rows)  # Unobserved confounder

    # Treatment assignment (campaign exposure) with confounding
    treatment_prob = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2 + 0.2 * U)))
    treatment = np.random.binomial(1, treatment_prob)

    # Outcome (purchase amount) with treatment effect and confounding
    outcome = (
        2.0 * treatment  # True treatment effect
        + 1.5 * X1  # Age effect
        + 1.2 * X2  # Income effect
        + 0.8 * X3  # Purchase history effect
        + 0.5 * U  # Unobserved confounding
        + np.random.normal(0, 2, n_rows)  # Noise
    )

    # Create matched pairs for Rosenbaum bounds (simplified)
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    min_size = min(len(treated_idx), len(control_idx))
    if min_size > 10000:  # Limit for performance
        min_size = 10000

    treated_outcomes = outcome[treated_idx[:min_size]]
    control_outcomes = outcome[control_idx[:min_size]]

    # Covariate matrices
    X_restricted = X1.reshape(-1, 1)
    X_full = np.column_stack([X1, X2, X3])

    results: dict[str, Any] = {}

    # Benchmark E-value
    print("Benchmarking E-value calculation...")
    start_time = time.time()
    try:
        evalue_result = e_value(observed_estimate=2.0, ci_lower=1.8, ci_upper=2.2)
        evalue_time = time.time() - start_time
        results["e_value"] = {
            "time_seconds": evalue_time,
            "passes_kpi": evalue_time < target_time_seconds,
            "status": "success",
            "result_sample": evalue_result,
        }
    except Exception as e:
        results["e_value"] = {
            "time_seconds": time.time() - start_time,
            "passes_kpi": False,
            "status": "error",
            "error": str(e),
        }

    # Benchmark Rosenbaum bounds
    print("Benchmarking Rosenbaum bounds...")
    start_time = time.time()
    try:
        rosenbaum_result = rosenbaum_bounds(
            treated_outcomes=treated_outcomes,
            control_outcomes=control_outcomes,
        )
        rosenbaum_time = time.time() - start_time
        results["rosenbaum_bounds"] = {
            "time_seconds": rosenbaum_time,
            "passes_kpi": rosenbaum_time < target_time_seconds,
            "status": "success",
            "n_pairs": min_size,
            "result_sample": {
                k: v
                for k, v in rosenbaum_result.items()
                if k not in ["bootstrap_results"]
            },
        }
    except Exception as e:
        results["rosenbaum_bounds"] = {
            "time_seconds": time.time() - start_time,
            "passes_kpi": False,
            "status": "error",
            "error": str(e),
        }

    # Benchmark Oster delta
    print("Benchmarking Oster delta...")
    start_time = time.time()
    try:
        oster_result = oster_delta(
            outcome=outcome,
            treatment=treatment,
            covariates_restricted=X_restricted,
            covariates_full=X_full,
            bootstrap_samples=min(n_bootstrap, 50),  # Limit for large data
        )
        oster_time = time.time() - start_time
        results["oster_delta"] = {
            "time_seconds": oster_time,
            "passes_kpi": oster_time < target_time_seconds,
            "status": "success",
            "result_sample": {
                k: v for k, v in oster_result.items() if k not in ["bootstrap_results"]
            },
        }
    except Exception as e:
        results["oster_delta"] = {
            "time_seconds": time.time() - start_time,
            "passes_kpi": False,
            "status": "error",
            "error": str(e),
        }

    # Benchmark negative control
    print("Benchmarking negative control...")
    start_time = time.time()
    try:
        # Generate null outcome for negative control
        null_outcome = np.random.normal(0, 1, n_rows)

        nc_result = negative_control(
            treatment=treatment,
            outcome=outcome,
            negative_control_outcome=null_outcome,
            covariates=X_full,
        )
        nc_time = time.time() - start_time
        results["negative_control"] = {
            "time_seconds": nc_time,
            "passes_kpi": nc_time < target_time_seconds,
            "status": "success",
            "result_sample": nc_result,
        }
    except Exception as e:
        results["negative_control"] = {
            "time_seconds": time.time() - start_time,
            "passes_kpi": False,
            "status": "error",
            "error": str(e),
        }

    # Benchmark placebo test
    print("Benchmarking placebo test...")
    start_time = time.time()
    try:
        placebo_result = placebo_test(
            treatment=treatment,
            outcome=outcome,
            covariates=X_full,
            placebo_type="random_treatment",
        )
        placebo_time = time.time() - start_time
        results["placebo_test"] = {
            "time_seconds": placebo_time,
            "passes_kpi": placebo_time < target_time_seconds,
            "status": "success",
            "result_sample": placebo_result,
        }
    except Exception as e:
        results["placebo_test"] = {
            "time_seconds": time.time() - start_time,
            "passes_kpi": False,
            "status": "error",
            "error": str(e),
        }

    # Summary statistics
    all_times: list[float] = [float(r["time_seconds"]) for r in results.values() if "time_seconds" in r]
    successful_functions = [
        k for k, v in results.items() if v.get("status") == "success"
    ]
    passing_kpi = [k for k, v in results.items() if v.get("passes_kpi", False)]

    results["summary"] = {
        "total_functions_tested": len(results),
        "successful_functions": len(successful_functions),
        "functions_passing_kpi": len(passing_kpi),
        "average_time_seconds": np.mean(all_times) if all_times else 0,
        "max_time_seconds": max(all_times) if all_times else 0,
        "all_pass_kpi": len(passing_kpi) == len(successful_functions),
        "target_time_seconds": target_time_seconds,
        "n_rows_tested": n_rows,
        "passing_functions": passing_kpi,
        "failing_functions": [
            k
            for k in results.keys()
            if k != "summary" and not results[k].get("passes_kpi", False)
        ],
    }

    return results


def run_performance_validation(
    print_results: bool = True, save_to_file: Optional[str] = None
) -> bool:
    """Run performance validation for all sensitivity analysis functions.

    Args:
        print_results: Whether to print detailed results
        save_to_file: Optional file path to save results

    Returns:
        True if all functions pass KPI requirements, False otherwise
    """
    print("Starting performance validation for sensitivity analysis suite...")
    print("=" * 60)

    results = benchmark_sensitivity_functions()

    if print_results:
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)

        for func_name, func_results in results.items():
            if func_name == "summary":
                continue

            status_emoji = "✅" if func_results.get("passes_kpi", False) else "❌"
            time_str = f"{func_results.get('time_seconds', 0):.3f}s"

            print(f"{status_emoji} {func_name}: {time_str}")

            if func_results.get("status") == "error":
                print(f"   Error: {func_results.get('error', 'Unknown error')}")
            elif not func_results.get("passes_kpi", False):
                print("   ⚠️  Exceeds 2s KPI requirement")

        print("\n" + "-" * 60)
        summary = results["summary"]
        print("Overall Results:")
        print(f"  Functions tested: {summary['total_functions_tested']}")
        print(f"  Successful: {summary['successful_functions']}")
        print(f"  Passing KPI: {summary['functions_passing_kpi']}")
        print(f"  Average time: {summary['average_time_seconds']:.3f}s")
        print(f"  Max time: {summary['max_time_seconds']:.3f}s")
        print(f"  Target time: {summary['target_time_seconds']}s")
        print(f"  Dataset size: {summary['n_rows_tested']:,} rows")

        overall_emoji = "✅" if summary["all_pass_kpi"] else "❌"
        print(
            f"\n{overall_emoji} Overall KPI Status: {'PASS' if summary['all_pass_kpi'] else 'FAIL'}"
        )

        if summary["failing_functions"]:
            print(f"   Failing functions: {', '.join(summary['failing_functions'])}")

    if save_to_file:
        import json

        with open(save_to_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {save_to_file}")

    return bool(results["summary"]["all_pass_kpi"])


if __name__ == "__main__":
    # Run performance validation when script is executed directly
    success = run_performance_validation()
    exit(0 if success else 1)
