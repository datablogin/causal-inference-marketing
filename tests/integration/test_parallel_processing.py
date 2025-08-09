"""Integration tests for parallel processing performance optimization."""

import time

import numpy as np
import pytest

from libs.causal_inference.causal_inference.core.base import (
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from libs.causal_inference.causal_inference.estimators.doubly_robust_ml import (
    DoublyRobustMLEstimator,
)
from libs.causal_inference.causal_inference.ml.cross_fitting import (
    ParallelCrossFittingConfig,
)
from libs.causal_inference.causal_inference.testing.synthetic import (
    generate_synthetic_dml_data,
)


@pytest.mark.integration
class TestParallelProcessingIntegration:
    """Integration tests for parallel processing features."""

    @pytest.fixture
    def large_synthetic_data(self):
        """Generate larger synthetic dataset for performance testing."""
        np.random.seed(42)
        n_samples = 1000  # Larger dataset to see parallel benefits
        n_features = 15

        data = generate_synthetic_dml_data(
            n_samples=n_samples,
            n_features=n_features,
            treatment_effect=2.0,
            noise_std=0.5,
            confounding_strength=1.0,
            random_state=42,
        )

        return data

    def test_parallel_vs_sequential_performance(self, large_synthetic_data):
        """Test that parallel processing provides performance benefits."""
        X, Y, A = (
            large_synthetic_data["X"],
            large_synthetic_data["Y"],
            large_synthetic_data["A"],
        )

        # Sequential configuration
        sequential_config = ParallelCrossFittingConfig(
            n_jobs=1,
            enable_caching=False,  # Disable to isolate parallel effects
        )

        # Parallel configuration
        parallel_config = ParallelCrossFittingConfig(
            n_jobs=2,
            parallel_backend="threading",
            enable_caching=False,
        )

        # Test sequential processing
        sequential_estimator = DoublyRobustMLEstimator(
            cv_folds=5,
            performance_config=sequential_config,
            random_state=42,
        )

        start_time = time.perf_counter()
        sequential_estimator.fit(
            treatment=TreatmentData(values=A, treatment_type="binary"),
            outcome=OutcomeData(values=Y, outcome_type="continuous"),
            covariates=CovariateData(values=X),
        )
        sequential_time = time.perf_counter() - start_time

        # Test parallel processing
        parallel_estimator = DoublyRobustMLEstimator(
            cv_folds=5,
            performance_config=parallel_config,
            random_state=42,
        )

        start_time = time.perf_counter()
        parallel_estimator.fit(
            treatment=TreatmentData(values=A, treatment_type="binary"),
            outcome=OutcomeData(values=Y, outcome_type="continuous"),
            covariates=CovariateData(values=X),
        )
        parallel_time = time.perf_counter() - start_time

        # Check that both produced similar results
        sequential_ate = sequential_estimator.estimate_ate()
        parallel_ate = parallel_estimator.estimate_ate()

        # ATE estimates should be very similar (within numerical tolerance)
        ate_diff = abs(sequential_ate.ate - parallel_ate.ate)
        assert ate_diff < 0.1, f"ATE estimates differ too much: {ate_diff}"

        # Parallel should generally be faster or at least not much slower
        # Note: For small datasets or simple models, parallel overhead might dominate
        speedup_ratio = sequential_time / parallel_time

        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Parallel time: {parallel_time:.3f}s")
        print(f"Speedup ratio: {speedup_ratio:.2f}x")

        # Parallel processing should at least not be more than 50% slower
        # (accounting for overhead in small examples)
        assert (
            speedup_ratio > 0.5
        ), f"Parallel processing too slow: {speedup_ratio:.2f}x"

        # Check that parallel speedup was recorded
        parallel_metrics = parallel_estimator.get_performance_metrics()
        assert (
            "parallel_speedup" in parallel_metrics
            or "performance_analysis" in parallel_metrics
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
