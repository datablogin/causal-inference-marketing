"""Integration tests for orthogonal moment functions in DoublyRobustML estimator.

This module provides comprehensive integration tests to verify that the enhanced
DoublyRobustML estimator with new orthogonal moment functions works correctly
end-to-end with various data scenarios and settings.
"""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import (
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.doubly_robust_ml import DoublyRobustMLEstimator
from causal_inference.estimators.orthogonal_moments import OrthogonalMoments
from causal_inference.ml.super_learner import SuperLearner


class TestOrthogonalMomentsIntegration:
    """Integration tests for orthogonal moment functions."""

    @pytest.fixture
    def synthetic_scenarios(self):
        """Generate multiple synthetic data scenarios for comprehensive testing."""
        scenarios = {}

        # Scenario 1: Well-behaved, balanced treatment
        np.random.seed(123)
        n1 = 300
        X1 = np.random.randn(n1, 6)
        treatment_coef1 = np.array([0.4, -0.3, 0.2, 0.1, -0.2, 0.0])
        logit_p1 = X1 @ treatment_coef1
        treatment_prob1 = 1 / (1 + np.exp(-logit_p1))
        treatment1 = np.random.binomial(1, treatment_prob1)

        outcome_coef1 = np.array([0.5, -0.2, 0.3, -0.1, 0.4, 0.2])
        true_ate1 = 2.0
        outcome1 = (
            X1 @ outcome_coef1 + true_ate1 * treatment1 + np.random.normal(0, 0.8, n1)
        )

        scenarios["balanced"] = {
            "X": X1,
            "treatment": treatment1,
            "outcome": outcome1,
            "true_ate": true_ate1,
            "description": "Balanced treatment, good overlap",
        }

        # Scenario 2: Small sample, high dimensional (challenging case)
        np.random.seed(456)
        n2 = 80
        p2 = 20
        X2 = np.random.randn(n2, p2)

        # Only first 5 variables matter for treatment
        treatment_coef2 = np.zeros(p2)
        treatment_coef2[:5] = [0.6, -0.4, 0.3, -0.2, 0.5]
        logit_p2 = X2 @ treatment_coef2
        treatment_prob2 = 1 / (1 + np.exp(-logit_p2))
        treatment2 = np.random.binomial(1, treatment_prob2)

        # Only different 5 variables matter for outcome
        outcome_coef2 = np.zeros(p2)
        outcome_coef2[3:8] = [0.4, -0.3, 0.5, -0.2, 0.3]
        true_ate2 = 1.5
        outcome2 = (
            X2 @ outcome_coef2 + true_ate2 * treatment2 + np.random.normal(0, 0.6, n2)
        )

        scenarios["high_dim_small"] = {
            "X": X2,
            "treatment": treatment2,
            "outcome": outcome2,
            "true_ate": true_ate2,
            "description": "High dimensional, small sample",
        }

        # Scenario 3: Poor overlap (challenging for standard methods)
        np.random.seed(789)
        n3 = 200
        X3 = np.random.randn(n3, 4)

        # Create poor overlap by using extreme coefficients
        treatment_coef3 = np.array([2.0, -1.5, 1.0, -0.8])
        logit_p3 = X3 @ treatment_coef3
        treatment_prob3 = 1 / (1 + np.exp(-logit_p3))
        treatment3 = np.random.binomial(1, treatment_prob3)

        outcome_coef3 = np.array([0.3, -0.4, 0.2, 0.5])
        true_ate3 = 2.5
        outcome3 = (
            X3 @ outcome_coef3 + true_ate3 * treatment3 + np.random.normal(0, 0.7, n3)
        )

        scenarios["poor_overlap"] = {
            "X": X3,
            "treatment": treatment3,
            "outcome": outcome3,
            "true_ate": true_ate3,
            "description": "Poor propensity score overlap",
        }

        # Scenario 4: With instrumental variable
        np.random.seed(101112)
        n4 = 250
        X4 = np.random.randn(n4, 5)

        # Generate instrument (affects treatment but not outcome directly)
        Z4 = (
            np.random.binomial(1, 0.4, n4)
            + 0.2 * X4[:, 0]
            + np.random.normal(0, 0.3, n4)
        )

        # Unobserved confounder
        U4 = np.random.randn(n4)

        # Treatment (endogenous - affected by instrument and unobserved confounder)
        treatment_logit4 = 0.8 * Z4 + 0.4 * X4[:, 0] - 0.3 * X4[:, 1] + 0.5 * U4
        treatment_prob4 = 1 / (1 + np.exp(-treatment_logit4))
        treatment4 = np.random.binomial(1, treatment_prob4)

        # Outcome (affected by treatment and unobserved confounder, but not instrument directly)
        true_ate4 = 1.8
        outcome4 = (
            true_ate4 * treatment4
            + 0.4 * X4[:, 0]
            + 0.3 * X4[:, 1]
            - 0.2 * X4[:, 2]
            + 0.6 * U4  # Creates endogeneity
            + np.random.normal(0, 0.5, n4)
        )

        scenarios["with_instrument"] = {
            "X": X4,
            "treatment": treatment4,
            "outcome": outcome4,
            "true_ate": true_ate4,
            "instrument": Z4,
            "description": "Endogenous treatment with instrument",
        }

        return scenarios

    def test_all_moment_functions_basic(self, synthetic_scenarios):
        """Test that all moment functions work on basic balanced scenario."""
        scenario = synthetic_scenarios["balanced"]
        X, treatment, outcome, true_ate = (
            scenario["X"],
            scenario["treatment"],
            scenario["outcome"],
            scenario["true_ate"],
        )

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X), names=[f"X{i}" for i in range(X.shape[1])]
        )

        # Test all available moment functions
        moment_functions = OrthogonalMoments.get_available_methods()

        results = {}
        for moment_function in moment_functions:
            estimator = DoublyRobustMLEstimator(
                outcome_learner=SuperLearner(
                    ["linear_regression", "ridge", "random_forest"]
                ),
                propensity_learner=SuperLearner(
                    ["logistic_regression", "ridge_logistic"]
                ),
                cross_fitting=True,
                cv_folds=3,
                moment_function=moment_function,
                random_state=42,
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            results[moment_function] = {
                "ate": effect.ate,
                "ate_se": effect.ate_se,
                "ci_width": effect.ate_ci_upper - effect.ate_ci_lower
                if effect.ate_ci_upper
                else None,
                "method": effect.method,
                "estimator": estimator,
            }

            # Basic sanity checks
            assert isinstance(effect.ate, float)
            assert np.isfinite(effect.ate)
            assert effect.method.startswith("DoublyRobustML_")

            # Should be reasonably close to true ATE
            assert abs(effect.ate - true_ate) < 2.0

        # Compare performance across methods
        ate_errors = {
            method: abs(results[method]["ate"] - true_ate)
            for method in moment_functions
        }

        print(f"\nMethod comparison for {scenario['description']}:")
        for method, error in sorted(ate_errors.items(), key=lambda x: x[1]):
            print(f"  {method}: ATE error = {error:.3f}")

        assert len(results) == len(moment_functions)

    def test_auto_selection_performance(self, synthetic_scenarios):
        """Test automatic method selection across different scenarios."""
        auto_results = {}

        for scenario_name, scenario in synthetic_scenarios.items():
            X, treatment, outcome, true_ate = (
                scenario["X"],
                scenario["treatment"],
                scenario["outcome"],
                scenario["true_ate"],
            )

            # Prepare data
            treatment_data = TreatmentData(values=treatment, treatment_type="binary")
            outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
            covariate_data = CovariateData(values=pd.DataFrame(X))

            # Test auto selection
            estimator = DoublyRobustMLEstimator(
                cross_fitting=True,
                cv_folds=3,
                moment_function="auto",
                verbose=False,
                random_state=42,
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            # Get selection results
            selection_results = estimator.get_moment_selection_results()

            auto_results[scenario_name] = {
                "selected_method": selection_results["selected_method"],
                "ate": effect.ate,
                "ate_error": abs(effect.ate - true_ate),
                "decision_factors": selection_results["decision_factors"],
                "data_characteristics": selection_results["data_characteristics"],
            }

            # Validate selection results structure
            assert "selected_method" in selection_results
            assert "decision_factors" in selection_results
            assert "data_characteristics" in selection_results
            assert (
                selection_results["selected_method"]
                in OrthogonalMoments.get_available_methods()
            )

        # Print auto-selection results
        print("\nAutomatic method selection results:")
        for scenario_name, result in auto_results.items():
            print(
                f"\n{scenario_name} ({synthetic_scenarios[scenario_name]['description']}):"
            )
            print(f"  Selected: {result['selected_method']}")
            print(f"  ATE error: {result['ate_error']:.3f}")
            print(f"  Reasoning: {'; '.join(result['decision_factors'])}")

        # Test expected selections for specific scenarios
        assert auto_results["balanced"]["selected_method"] == "aipw"
        assert auto_results["high_dim_small"]["selected_method"] == "partialling_out"
        assert auto_results["poor_overlap"]["selected_method"] == "plr"

    def test_method_comparison_validation(self, synthetic_scenarios):
        """Test cross-validation method comparison functionality."""
        scenario = synthetic_scenarios["balanced"]
        X, treatment, outcome = (
            scenario["X"],
            scenario["treatment"],
            scenario["outcome"],
        )

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        # Fit estimator with one method
        estimator = DoublyRobustMLEstimator(
            cross_fitting=True,
            cv_folds=3,
            moment_function="aipw",
            random_state=42,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        estimator.estimate_ate()

        # Compare multiple methods
        candidate_methods = ["aipw", "orthogonal", "partialling_out", "plr"]
        comparison_results = estimator.compare_moment_functions(
            candidate_methods=candidate_methods, cv_folds=3
        )

        # Validate structure
        assert "method_performance" in comparison_results
        assert "rankings" in comparison_results
        assert "recommended_method" in comparison_results
        assert "current_method" in comparison_results

        assert comparison_results["current_method"] == "aipw"
        assert comparison_results["recommended_method"] in candidate_methods

        # Check all methods were evaluated
        for method in candidate_methods:
            assert method in comparison_results["method_performance"]
            perf = comparison_results["method_performance"][method]
            assert "orthogonality_scores" in perf
            assert "ate_estimates" in perf
            assert "score_variance" in perf

        # Check rankings
        rankings = comparison_results["rankings"]["by_combined_score"]
        assert len(rankings) == len(candidate_methods)

        # Should be sorted by combined score (descending)
        scores = [rank["combined_score"] for rank in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_orthogonality_validation(self, synthetic_scenarios):
        """Test orthogonality validation functionality."""
        scenario = synthetic_scenarios["balanced"]
        X, treatment, outcome = (
            scenario["X"],
            scenario["treatment"],
            scenario["outcome"],
        )

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        # Test different moment functions for orthogonality
        moment_functions = ["aipw", "orthogonal", "partialling_out"]

        for moment_function in moment_functions:
            estimator = DoublyRobustMLEstimator(
                cross_fitting=False,  # Simpler for validation testing
                moment_function=moment_function,
                random_state=42,
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            estimator.estimate_ate()

            # Validate orthogonality
            validation_results = estimator.validate_moment_function_choice()

            assert "moment_method" in validation_results
            assert "validation_passed" in validation_results
            assert "is_orthogonal" in validation_results
            assert validation_results["moment_method"] == moment_function

            # For well-behaved data, orthogonality should generally be satisfied
            if validation_results["is_orthogonal"]:
                print(f"{moment_function}: Orthogonality validated ✓")
            else:
                print(
                    f"{moment_function}: Orthogonality concern - {validation_results.get('interpretation', 'No details')}"
                )

    def test_performance_across_scenarios(self, synthetic_scenarios):
        """Test performance of different methods across all scenarios."""
        performance_matrix = {}

        # Test key methods across all scenarios
        test_methods = ["aipw", "orthogonal", "partialling_out", "plr", "auto"]

        for scenario_name, scenario in synthetic_scenarios.items():
            X, treatment, outcome, true_ate = (
                scenario["X"],
                scenario["treatment"],
                scenario["outcome"],
                scenario["true_ate"],
            )

            performance_matrix[scenario_name] = {}

            # Skip instrument-based methods for non-IV scenarios
            scenario_methods = test_methods.copy()
            if "instrument" not in scenario:
                # For non-IV scenarios, test all methods
                pass

            for method in scenario_methods:
                try:
                    # Prepare data
                    treatment_data = TreatmentData(
                        values=treatment, treatment_type="binary"
                    )
                    outcome_data = OutcomeData(
                        values=outcome, outcome_type="continuous"
                    )
                    covariate_data = CovariateData(values=pd.DataFrame(X))

                    estimator = DoublyRobustMLEstimator(
                        cross_fitting=True,
                        cv_folds=3,
                        moment_function=method,
                        random_state=42,
                    )

                    estimator.fit(treatment_data, outcome_data, covariate_data)
                    effect = estimator.estimate_ate()

                    performance_matrix[scenario_name][method] = {
                        "ate": effect.ate,
                        "ate_error": abs(effect.ate - true_ate),
                        "ate_se": effect.ate_se,
                        "ci_width": (
                            effect.ate_ci_upper - effect.ate_ci_lower
                            if effect.ate_ci_upper
                            else None
                        ),
                        "success": True,
                    }

                except Exception as e:
                    performance_matrix[scenario_name][method] = {
                        "ate": np.nan,
                        "ate_error": np.inf,
                        "ate_se": None,
                        "ci_width": None,
                        "success": False,
                        "error": str(e),
                    }

        # Print performance summary
        print("\nPerformance across scenarios:")
        print("=" * 80)

        for scenario_name, results in performance_matrix.items():
            print(
                f"\n{scenario_name} ({synthetic_scenarios[scenario_name]['description']}):"
            )
            print(f"True ATE: {synthetic_scenarios[scenario_name]['true_ate']:.2f}")

            # Sort methods by performance
            successful_methods = {
                method: result
                for method, result in results.items()
                if result.get("success", False)
            }
            sorted_methods = sorted(
                successful_methods.items(), key=lambda x: x[1]["ate_error"]
            )

            for method, result in sorted_methods:
                ate_error = result["ate_error"]
                ci_width = result["ci_width"]
                print(
                    f"  {method:15s}: ATE error = {ate_error:.3f}, "
                    f"CI width = {ci_width:.3f if ci_width else 'N/A'}"
                )

            # Print failed methods if any
            failed_methods = [
                method
                for method, result in results.items()
                if not result.get("success", False)
            ]
            if failed_methods:
                print(f"  Failed methods: {', '.join(failed_methods)}")

        # Validate that at least some methods work for each scenario
        for scenario_name, results in performance_matrix.items():
            successful_count = sum(
                1 for result in results.values() if result.get("success", False)
            )
            assert successful_count >= 3, (
                f"Too few methods successful for {scenario_name}"
            )

    def test_iv_methods_with_instruments(self, synthetic_scenarios):
        """Test IV-based methods when instruments are available."""
        scenario = synthetic_scenarios["with_instrument"]
        X, treatment, outcome, true_ate = (
            scenario["X"],
            scenario["treatment"],
            scenario["outcome"],
            scenario["true_ate"],
        )
        instrument = scenario["instrument"]

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        # Test IV methods manually (since DoublyRobustML doesn't directly support instruments yet)
        # This tests the OrthogonalMoments functionality
        from causal_inference.estimators.doubly_robust_ml import DoublyRobustMLEstimator

        # Fit a basic estimator to get nuisance estimates
        estimator = DoublyRobustMLEstimator(
            cross_fitting=False,
            moment_function="aipw",
            random_state=42,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        estimator.estimate_ate()

        # Get nuisance estimates
        nuisance_estimates = estimator.nuisance_estimates_

        # Test IV methods directly
        iv_methods = ["interactive_iv", "pliv"]

        for method in iv_methods:
            scores = OrthogonalMoments.compute_scores(
                method, nuisance_estimates, treatment, outcome, instrument=instrument
            )

            ate_estimate = np.mean(scores)

            # Should produce finite estimates
            assert np.isfinite(ate_estimate)

            # Should be reasonably close to true ATE (IV estimation can be noisier)
            assert abs(ate_estimate - true_ate) < 3.0

            print(
                f"{method} with instrument: ATE = {ate_estimate:.3f}, "
                f"Error = {abs(ate_estimate - true_ate):.3f}"
            )

    def test_edge_cases_and_robustness(self, synthetic_scenarios):
        """Test edge cases and robustness of the implementation."""
        scenario = synthetic_scenarios["balanced"]
        X, treatment, outcome = (
            scenario["X"],
            scenario["treatment"],
            scenario["outcome"],
        )

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        # Test with very few cross-validation folds
        estimator_min_cv = DoublyRobustMLEstimator(
            cross_fitting=True,
            cv_folds=2,
            moment_function="aipw",
            random_state=42,
        )

        estimator_min_cv.fit(treatment_data, outcome_data, covariate_data)
        effect_min_cv = estimator_min_cv.estimate_ate()

        assert np.isfinite(effect_min_cv.ate)

        # Test with no cross-fitting
        estimator_no_cv = DoublyRobustMLEstimator(
            cross_fitting=False,
            moment_function="partialling_out",
            random_state=42,
        )

        estimator_no_cv.fit(treatment_data, outcome_data, covariate_data)
        effect_no_cv = estimator_no_cv.estimate_ate()

        assert np.isfinite(effect_no_cv.ate)

        # Test error handling for method selection on unfitted estimator
        unfitted_estimator = DoublyRobustMLEstimator()

        with pytest.raises(Exception):  # Should raise EstimationError
            unfitted_estimator.get_moment_selection_results()

        with pytest.raises(Exception):  # Should raise EstimationError
            unfitted_estimator.validate_moment_function_choice()

    def test_consistency_across_random_seeds(self, synthetic_scenarios):
        """Test that results are consistent across different random seeds."""
        scenario = synthetic_scenarios["balanced"]
        X, treatment, outcome = (
            scenario["X"],
            scenario["treatment"],
            scenario["outcome"],
        )

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        # Test consistency for deterministic methods
        method = "aipw"
        random_seeds = [42, 123, 456]
        estimates = []

        for seed in random_seeds:
            estimator = DoublyRobustMLEstimator(
                cross_fitting=True,
                cv_folds=3,
                moment_function=method,
                random_state=seed,
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()
            estimates.append(effect.ate)

        # Estimates should be reasonably consistent (allowing for CV randomness)
        max_diff = max(estimates) - min(estimates)
        assert max_diff < 0.5, f"Estimates vary too much across seeds: {estimates}"

        print(
            f"Consistency check for {method}: estimates = {estimates}, max_diff = {max_diff:.4f}"
        )


class TestDocumentationAndExamples:
    """Test functionality that would be used in documentation examples."""

    def test_basic_usage_example(self):
        """Test basic usage that would appear in documentation."""
        # Generate simple example data
        np.random.seed(42)
        n = 200

        # Generate covariates
        age = np.random.normal(45, 15, n)
        income = np.random.exponential(50000, n)
        education = np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.3, 0.3, 0.2])

        X = np.column_stack([age, income / 1000, education])  # Scale income

        # Treatment assignment (marketing campaign)
        treatment_prob = 1 / (
            1
            + np.exp(
                -(
                    0.02 * (age - 45)
                    + 0.01 * (income / 1000 - 50)
                    + 0.3 * education
                    - 0.5
                )
            )
        )
        treatment = np.random.binomial(1, treatment_prob)

        # Outcome (sales in thousands)
        true_effect = 5.0  # $5k increase in sales
        outcome = (
            20
            + 0.1 * age
            + 0.05 * income / 1000
            + 2 * education
            + true_effect * treatment
            + np.random.normal(0, 8, n)
        )

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=["age", "income_k", "education"]),
            names=["age", "income_k", "education"],
        )

        # Example 1: Basic usage with automatic method selection
        estimator_auto = DoublyRobustMLEstimator(
            moment_function="auto", cross_fitting=True, cv_folds=5, random_state=42
        )

        estimator_auto.fit(treatment_data, outcome_data, covariate_data)
        effect_auto = estimator_auto.estimate_ate()

        print("\nBasic Example Results:")
        print(f"True effect: ${true_effect:.1f}k")
        print(f"Estimated effect: ${effect_auto.ate:.1f}k")
        print(
            f"95% CI: [${effect_auto.ate_ci_lower:.1f}k, ${effect_auto.ate_ci_upper:.1f}k]"
        )
        print(
            f"Method selected: {estimator_auto.get_moment_selection_results()['selected_method']}"
        )

        # Should be reasonably close
        assert abs(effect_auto.ate - true_effect) < 2.0

        # Example 2: Compare multiple methods
        methods = ["aipw", "orthogonal", "partialling_out", "plr"]
        method_results = {}

        for method in methods:
            estimator = DoublyRobustMLEstimator(
                moment_function=method, cross_fitting=True, cv_folds=3, random_state=42
            )

            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            method_results[method] = {
                "ate": effect.ate,
                "ci_width": effect.ate_ci_upper - effect.ate_ci_lower,
                "error": abs(effect.ate - true_effect),
            }

        print("\nMethod Comparison:")
        for method, result in sorted(
            method_results.items(), key=lambda x: x[1]["error"]
        ):
            print(
                f"{method:15s}: ${result['ate']:5.1f}k (±{result['ci_width'] / 2:4.1f}k), "
                f"error = ${result['error']:4.1f}k"
            )

        # All methods should provide reasonable estimates
        for method, result in method_results.items():
            assert result["error"] < 3.0, (
                f"Method {method} error too large: {result['error']}"
            )

    def test_advanced_usage_example(self):
        """Test advanced usage with method selection and validation."""
        # Generate more challenging scenario
        np.random.seed(123)
        n = 150

        # High-dimensional confounders
        X = np.random.randn(n, 12)

        # Complex treatment assignment
        treatment_coef = np.array([0.3, -0.2, 0.4, -0.1, 0.2, 0, 0, 0, 0, 0, 0, 0])
        treatment_prob = 1 / (1 + np.exp(-(X @ treatment_coef)))
        treatment = np.random.binomial(1, treatment_prob)

        # Outcome with different confounders
        outcome_coef = np.array([0.1, 0.2, 0, 0, -0.3, 0.4, -0.2, 0.3, 0, 0, 0, 0])
        true_effect = 2.5
        outcome = X @ outcome_coef + true_effect * treatment + np.random.normal(0, 1, n)

        # Prepare data
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame(X))

        # Advanced example: Method selection with validation
        print("\nAdvanced Example - Method Selection and Validation:")

        # Step 1: Use automatic selection
        estimator = DoublyRobustMLEstimator(
            moment_function="auto", cross_fitting=True, cv_folds=4, random_state=42
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Step 2: Get selection rationale
        selection_results = estimator.get_moment_selection_results()
        print(f"Auto-selected method: {selection_results['selected_method']}")
        print(
            f"Selection reasoning: {'; '.join(selection_results['decision_factors'])}"
        )

        # Step 3: Validate the choice
        validation_results = estimator.validate_moment_function_choice()
        print(f"Orthogonality validated: {validation_results['validation_passed']}")

        # Step 4: Compare with other methods
        comparison_results = estimator.compare_moment_functions(cv_folds=3)
        print(f"CV recommended method: {comparison_results['recommended_method']}")

        best_methods = comparison_results["rankings"]["by_combined_score"][:3]
        print("Top 3 methods by CV:")
        for i, method_result in enumerate(best_methods, 1):
            print(
                f"  {i}. {method_result['method']} (score: {method_result['combined_score']:.3f})"
            )

        # Validate results
        assert isinstance(selection_results["selected_method"], str)
        assert (
            selection_results["selected_method"]
            in OrthogonalMoments.get_available_methods()
        )
        assert isinstance(validation_results["validation_passed"], bool)
        assert (
            comparison_results["recommended_method"]
            in OrthogonalMoments.get_available_methods()
        )

        # Effect should be reasonable
        assert abs(effect.ate - true_effect) < 1.5

        print(f"Final estimate: {effect.ate:.2f} (true: {true_effect})")
