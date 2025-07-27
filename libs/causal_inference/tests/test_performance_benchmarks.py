"""Performance benchmark tests for causal inference estimators.

This module provides performance benchmarks to establish baseline
performance metrics and catch performance regressions.
"""

import time

import numpy as np
import pandas as pd
import pytest

from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator

# Performance test constants
PERFORMANCE_BOOTSTRAP_SAMPLES = 10  # Reduced bootstrap for performance tests
SCALING_BOOTSTRAP_SAMPLES = 50  # Moderate bootstrap for scaling tests
MAX_PERFORMANCE_TIME = 30.0  # Maximum acceptable time for performance tests


@pytest.mark.benchmark
class TestEstimatorPerformance:
    """Benchmark performance of causal inference estimators."""

    def test_g_computation_performance_small(self, benchmark, simple_binary_data):
        """Benchmark G-computation on small dataset."""

        def run_g_computation():
            estimator = GComputationEstimator()
            estimator.fit(
                simple_binary_data["treatment"],
                simple_binary_data["outcome"],
                simple_binary_data["covariates"],
            )
            return estimator.estimate_ate()

        result = benchmark(run_g_computation)
        assert result.ate is not None

    def test_ipw_performance_small(self, benchmark, simple_binary_data):
        """Benchmark IPW on small dataset."""

        def run_ipw():
            estimator = IPWEstimator()
            estimator.fit(
                simple_binary_data["treatment"],
                simple_binary_data["outcome"],
                simple_binary_data["covariates"],
            )
            return estimator.estimate_ate()

        result = benchmark(run_ipw)
        assert result.ate is not None

    def test_aipw_performance_small(self, benchmark, simple_binary_data):
        """Benchmark AIPW on small dataset."""

        def run_aipw():
            estimator = AIPWEstimator(cross_fitting=False, bootstrap_samples=0)
            estimator.fit(
                simple_binary_data["treatment"],
                simple_binary_data["outcome"],
                simple_binary_data["covariates"],
            )
            return estimator.estimate_ate()

        result = benchmark(run_aipw)
        assert result.ate is not None

    @pytest.mark.slow
    def test_g_computation_performance_medium(
        self, benchmark, medium_sample_size, random_state
    ):
        """Benchmark G-computation on medium dataset."""
        # Generate medium-sized dataset
        np.random.seed(random_state)
        n_features = 10

        X = np.random.normal(0, 1, (medium_sample_size, n_features))
        propensity_weights = np.random.normal(0, 0.2, n_features)
        outcome_weights = np.random.normal(0, 0.3, n_features)

        linear_pred = X @ propensity_weights
        propensity = 1 / (1 + np.exp(-linear_pred))
        treatment = np.random.binomial(1, propensity, medium_sample_size)
        outcome = (
            X @ outcome_weights
            + 2 * treatment
            + np.random.normal(0, 1, medium_sample_size)
        )

        from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=[f"X{i}" for i in range(n_features)])
        )

        def run_g_computation():
            estimator = GComputationEstimator()
            estimator.fit(treatment_data, outcome_data, covariate_data)
            return estimator.estimate_ate()

        result = benchmark(run_g_computation)
        assert result.ate is not None

    @pytest.mark.slow
    def test_performance_scaling(self, performance_benchmark_data):
        """Test performance scaling with dataset size."""
        results = {}

        for estimator_name, estimator_class in [
            ("g_computation", GComputationEstimator),
            ("ipw", IPWEstimator),
            ("aipw", lambda: AIPWEstimator(cross_fitting=False, bootstrap_samples=0)),
        ]:
            start_time = time.time()

            estimator = estimator_class()
            estimator.fit(
                performance_benchmark_data["treatment"],
                performance_benchmark_data["outcome"],
                performance_benchmark_data["covariates"],
            )
            effect = estimator.estimate_ate()

            elapsed_time = time.time() - start_time
            results[estimator_name] = {
                "time": elapsed_time,
                "ate": effect.ate,
                "n_samples": len(performance_benchmark_data["treatment"].values),
                "n_features": performance_benchmark_data["covariates"].values.shape[1],
            }

        # Basic performance expectations (adjust as needed)
        for estimator_name, result in results.items():
            # Should complete within reasonable time for 1000 samples
            assert result["time"] < MAX_PERFORMANCE_TIME, (
                f"{estimator_name} took too long: {result['time']:.2f}s"
            )
            # Should produce finite estimates
            assert np.isfinite(result["ate"])

    def test_bootstrap_performance_scaling(self, simple_binary_data):
        """Test how bootstrap samples affect performance."""
        bootstrap_sizes = [10, 50, 100]
        results = {}

        for n_bootstrap in bootstrap_sizes:
            start_time = time.time()

            estimator = GComputationEstimator(bootstrap_samples=n_bootstrap)
            estimator.fit(
                simple_binary_data["treatment"],
                simple_binary_data["outcome"],
                simple_binary_data["covariates"],
            )
            effect = estimator.estimate_ate()

            elapsed_time = time.time() - start_time
            results[n_bootstrap] = {
                "time": elapsed_time,
                "ci_width": effect.confidence_interval[1]
                - effect.confidence_interval[0],
            }

        # Time should scale roughly linearly with bootstrap samples
        times = [results[n]["time"] for n in bootstrap_sizes]

        # More bootstrap samples should take more time
        assert times[2] > times[1] > times[0]

        # Should still be reasonable even with 100 bootstrap samples
        assert times[2] < 10.0

    @pytest.mark.slow
    def test_memory_usage_large_dataset(self, large_sample_size, random_state):
        """Test memory usage with large datasets."""
        np.random.seed(random_state)

        # Create relatively large dataset
        n_features = 50
        X = np.random.normal(0, 1, (large_sample_size, n_features))

        # Simple treatment and outcome generation
        treatment = np.random.binomial(1, 0.5, large_sample_size)
        outcome = np.random.normal(0, 1, large_sample_size) + treatment

        from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=[f"X{i}" for i in range(n_features)])
        )

        # This should not run out of memory or crash
        estimator = GComputationEstimator(
            bootstrap_samples=PERFORMANCE_BOOTSTRAP_SAMPLES
        )
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        assert np.isfinite(effect.ate)


@pytest.mark.benchmark
class TestDataProcessingPerformance:
    """Benchmark data processing and validation performance."""

    def test_data_validation_performance(self, benchmark, performance_benchmark_data):
        """Benchmark data validation performance."""
        from causal_inference.data.validation import validate_causal_data

        def run_validation():
            return validate_causal_data(
                performance_benchmark_data["treatment"],
                performance_benchmark_data["outcome"],
                performance_benchmark_data["covariates"],
            )

        result = benchmark(run_validation)
        assert result is not None

    def test_synthetic_data_generation_performance(
        self, benchmark, large_sample_size, random_state
    ):
        """Benchmark synthetic data generation performance."""
        from causal_inference.data.synthetic import SyntheticDataGenerator

        generator = SyntheticDataGenerator(random_state=random_state)

        def run_data_generation():
            return generator.generate_linear_binary_treatment(
                n_samples=large_sample_size, n_confounders=10, treatment_effect=2.0
            )

        result = benchmark(run_data_generation)
        assert len(result[0].values) == large_sample_size

    def test_missing_data_handling_performance(
        self, benchmark, performance_benchmark_data
    ):
        """Benchmark missing data handling performance."""
        from causal_inference.data.missing_data import MissingDataHandler

        # Introduce some missing values
        covariates_df = performance_benchmark_data["covariates"].values.copy()
        n_samples = len(covariates_df)
        missing_mask = np.random.random((n_samples, len(covariates_df.columns))) < 0.1

        for col in covariates_df.columns:
            covariates_df.loc[missing_mask[:, 0], col] = np.nan

        handler = MissingDataHandler(strategy="mean")

        def run_missing_data_handling():
            # Use the fit_transform method which exists on MissingDataHandler
            dummy_treatment = TreatmentData(values=np.ones(len(covariates_df)), treatment_type="binary")
            dummy_outcome = OutcomeData(values=np.ones(len(covariates_df)), outcome_type="continuous")
            dummy_covariates = CovariateData(values=covariates_df)
            return handler.fit_transform(dummy_treatment, dummy_outcome, dummy_covariates)

        result = benchmark(run_missing_data_handling)
        assert result is not None


@pytest.mark.benchmark
class TestDiagnosticsPerformance:
    """Benchmark diagnostic functions performance."""

    def test_balance_diagnostics_performance(
        self, benchmark, performance_benchmark_data
    ):
        """Benchmark balance diagnostics performance."""
        from causal_inference.diagnostics.balance import check_covariate_balance

        def run_balance_diagnostics():
            return check_covariate_balance(
                performance_benchmark_data["treatment"],
                performance_benchmark_data["covariates"],
            )

        result = benchmark(run_balance_diagnostics)
        assert result is not None

    def test_overlap_diagnostics_performance(
        self, benchmark, performance_benchmark_data
    ):
        """Benchmark overlap diagnostics performance."""
        from causal_inference.diagnostics.overlap import assess_positivity

        def run_overlap_diagnostics():
            return assess_positivity(
                performance_benchmark_data["treatment"],
                performance_benchmark_data["covariates"],
            )

        result = benchmark(run_overlap_diagnostics)
        assert result is not None


class TestPerformanceRegression:
    """Tests to catch performance regressions."""

    def test_g_computation_performance_regression(
        self, medium_sample_size, random_state
    ):
        """Ensure G-computation performance doesn't regress."""
        # Generate consistent test data
        np.random.seed(random_state)
        X = np.random.normal(0, 1, (medium_sample_size, 5))
        treatment = np.random.binomial(1, 0.5, medium_sample_size)
        outcome = (
            np.sum(X, axis=1)
            + 2 * treatment
            + np.random.normal(0, 1, medium_sample_size)
        )

        from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=[f"X{i}" for i in range(5)])
        )

        # Time the operation
        start_time = time.time()

        estimator = GComputationEstimator(bootstrap_samples=SCALING_BOOTSTRAP_SAMPLES)
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        elapsed_time = time.time() - start_time

        # Performance expectations (adjust based on baseline measurements)
        assert elapsed_time < 5.0, f"G-computation took too long: {elapsed_time:.2f}s"
        assert np.isfinite(effect.ate)

    @pytest.mark.parametrize("n_features", [5, 10, 20])
    def test_performance_vs_feature_count(
        self, n_features, small_sample_size, random_state
    ):
        """Test how performance scales with number of features."""
        np.random.seed(random_state)

        X = np.random.normal(0, 1, (small_sample_size, n_features))
        treatment = np.random.binomial(1, 0.5, small_sample_size)
        outcome = (
            np.sum(X[:, : min(3, n_features)], axis=1)
            + treatment
            + np.random.normal(0, 0.5, small_sample_size)
        )

        from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(X, columns=[f"X{i}" for i in range(n_features)])
        )

        start_time = time.time()

        estimator = GComputationEstimator(
            bootstrap_samples=PERFORMANCE_BOOTSTRAP_SAMPLES
        )
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        elapsed_time = time.time() - start_time

        # Time should scale reasonably with feature count
        max_expected_time = 0.5 + 0.1 * n_features  # Base time + feature scaling
        assert elapsed_time < max_expected_time, (
            f"Too slow for {n_features} features: {elapsed_time:.2f}s"
        )
        assert np.isfinite(effect.ate)
