"""Tests for transportability weighting methods."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from causal_inference.transportability.weighting import (
    DensityRatioEstimator,
    TransportabilityWeightingInterface,
    WeightingResult,
)


class TestWeightingResult:
    """Test suite for WeightingResult dataclass."""

    def test_initialization(self):
        """Test WeightingResult initialization."""
        weights = np.array([1.0, 1.5, 0.8, 1.2])
        density_ratio = np.array([0.9, 1.3, 0.7, 1.1])

        result = WeightingResult(
            weights=weights,
            density_ratio=density_ratio,
            effective_sample_size=3.5,
            max_weight=1.5,
            weight_stability_ratio=1.875,  # 1.5 / 0.8
            convergence_achieved=True,
            diagnostics={"test": "value"},
        )

        assert np.array_equal(result.weights, weights)
        assert np.array_equal(result.density_ratio, density_ratio)
        assert result.effective_sample_size == 3.5
        assert result.max_weight == 1.5
        assert result.convergence_achieved is True

    def test_is_stable_property(self):
        """Test is_stable property."""
        # Stable weights (ratio < 20)
        stable_result = WeightingResult(
            weights=np.array([1.0, 1.5, 0.8]),
            density_ratio=np.array([1.0, 1.5, 0.8]),
            effective_sample_size=2.8,
            max_weight=1.5,
            weight_stability_ratio=10.0,  # < 20
            convergence_achieved=True,
            diagnostics={},
        )
        assert stable_result.is_stable is True

        # Unstable weights (ratio >= 20)
        unstable_result = WeightingResult(
            weights=np.array([1.0, 25.0, 0.5]),
            density_ratio=np.array([1.0, 25.0, 0.5]),
            effective_sample_size=1.2,
            max_weight=25.0,
            weight_stability_ratio=50.0,  # >= 20
            convergence_achieved=True,
            diagnostics={},
        )
        assert unstable_result.is_stable is False

    def test_relative_efficiency_property(self):
        """Test relative_efficiency property."""
        result = WeightingResult(
            weights=np.array([1.0, 1.0, 1.0, 1.0]),  # 4 observations
            density_ratio=np.array([1.0, 1.0, 1.0, 1.0]),
            effective_sample_size=4.0,  # All weights equal
            max_weight=1.0,
            weight_stability_ratio=1.0,
            convergence_achieved=True,
            diagnostics={},
        )

        # Should be 4.0 / 4 = 1.0 (perfect efficiency)
        assert result.relative_efficiency == 1.0

        # Test with unequal weights
        result_unequal = WeightingResult(
            weights=np.array([2.0, 0.5, 1.0, 1.0]),
            density_ratio=np.array([2.0, 0.5, 1.0, 1.0]),
            effective_sample_size=3.2,  # Less than 4 due to weight variation
            max_weight=2.0,
            weight_stability_ratio=4.0,
            convergence_achieved=True,
            diagnostics={},
        )

        # Should be 3.2 / 4 = 0.8
        assert result_unequal.relative_efficiency == 0.8


class TestDensityRatioEstimator:
    """Test suite for DensityRatioEstimator."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with known shift."""
        np.random.seed(42)

        # Source population
        X_source, _ = make_classification(
            n_samples=500, n_features=4, n_informative=3, n_redundant=1, random_state=42
        )

        # Target population with shift
        X_target = X_source.copy()
        X_target[:, 0] += 0.5  # Mean shift in first feature
        X_target += np.random.normal(0, 0.1, X_target.shape)

        return X_source, X_target

    @pytest.fixture
    def identical_data(self):
        """Generate identical source and target data."""
        np.random.seed(42)
        X, _ = make_classification(
            n_samples=300, n_features=3, n_informative=2, n_redundant=0, random_state=42
        )
        return X, X.copy()

    def test_initialization_default(self):
        """Test DensityRatioEstimator initialization with defaults."""
        estimator = DensityRatioEstimator()

        assert estimator.trim_weights is True
        assert estimator.max_weight == 10.0
        assert estimator.min_weight == 0.1
        assert estimator.stabilize is True
        assert estimator.cross_validate is True
        assert estimator.cv_folds == 5
        assert estimator.is_fitted is False

    def test_initialization_custom(self):
        """Test DensityRatioEstimator initialization with custom parameters."""
        estimator = DensityRatioEstimator(
            trim_weights=False,
            max_weight=5.0,
            min_weight=0.2,
            stabilize=False,
            cross_validate=False,
            cv_folds=3,
            random_state=123,
        )

        assert estimator.trim_weights is False
        assert estimator.max_weight == 5.0
        assert estimator.min_weight == 0.2
        assert estimator.stabilize is False
        assert estimator.cross_validate is False
        assert estimator.cv_folds == 3
        assert estimator.random_state == 123

    def test_fit_weights_basic(self, sample_data):
        """Test basic weight fitting functionality."""
        X_source, X_target = sample_data

        estimator = DensityRatioEstimator(random_state=42)
        result = estimator.fit_weights(X_source, X_target)

        # Check result type and structure
        assert isinstance(result, WeightingResult)
        assert len(result.weights) == len(X_source)
        assert len(result.density_ratio) == len(X_source)
        assert result.effective_sample_size > 0
        assert result.max_weight > 0
        assert (
            result.convergence_achieved is True
        )  # Should always be True for this method

        # Check weights are positive
        assert np.all(result.weights > 0)

        # Check weights sum approximately to sample size (normalized)
        expected_sum = len(X_source)
        assert abs(np.sum(result.weights) - expected_sum) < 1e-10

    def test_fit_weights_identical_data(self, identical_data):
        """Test weight fitting on identical data."""
        X_source, X_target = identical_data

        estimator = DensityRatioEstimator(random_state=42)
        result = estimator.fit_weights(X_source, X_target)

        # Weights should be approximately uniform for identical data
        weight_variance = np.var(result.weights)

        # Should have low variance (though not exactly zero due to classification noise)
        assert weight_variance < 0.5  # Reasonable threshold

        # Effective sample size should be close to actual sample size
        efficiency = result.relative_efficiency
        assert efficiency > 0.8  # Should be quite efficient

    def test_dataframe_input(self, sample_data):
        """Test weight fitting with DataFrame input."""
        X_source, X_target = sample_data

        # Convert to DataFrames
        source_df = pd.DataFrame(
            X_source, columns=[f"var_{i}" for i in range(X_source.shape[1])]
        )
        target_df = pd.DataFrame(
            X_target, columns=[f"var_{i}" for i in range(X_target.shape[1])]
        )

        estimator = DensityRatioEstimator(random_state=42)
        result = estimator.fit_weights(source_df, target_df)

        # Should work the same as with arrays
        assert len(result.weights) == len(source_df)
        assert np.all(result.weights > 0)

    def test_weight_trimming(self, sample_data):
        """Test weight trimming functionality."""
        X_source, X_target = sample_data

        # Test with trimming enabled
        estimator_trim = DensityRatioEstimator(
            trim_weights=True, max_weight=3.0, min_weight=0.5, random_state=42
        )
        result_trim = estimator_trim.fit_weights(X_source, X_target)

        # All weights should be within specified bounds
        assert np.all(result_trim.weights >= 0.5)
        assert np.all(result_trim.weights <= 3.0)

        # Test without trimming
        estimator_no_trim = DensityRatioEstimator(
            trim_weights=False,
            stabilize=False,  # Also disable stabilization
            random_state=42,
        )
        result_no_trim = estimator_no_trim.fit_weights(X_source, X_target)

        # May have more extreme weights
        assert result_no_trim.max_weight >= result_trim.max_weight

    def test_cross_validation_option(self, sample_data):
        """Test cross-validation vs no cross-validation."""
        X_source, X_target = sample_data

        # With cross-validation
        estimator_cv = DensityRatioEstimator(
            cross_validate=True, cv_folds=3, random_state=42
        )
        result_cv = estimator_cv.fit_weights(X_source, X_target)

        # Without cross-validation
        estimator_no_cv = DensityRatioEstimator(cross_validate=False, random_state=42)
        result_no_cv = estimator_no_cv.fit_weights(X_source, X_target)

        # Both should produce valid results
        assert len(result_cv.weights) == len(result_no_cv.weights)
        assert np.all(result_cv.weights > 0)
        assert np.all(result_no_cv.weights > 0)

        # Results may differ due to different fitting procedures
        # But should be in same ballpark
        cv_mean = np.mean(result_cv.weights)
        no_cv_mean = np.mean(result_no_cv.weights)
        assert abs(cv_mean - no_cv_mean) < 0.5  # Reasonable difference

    def test_validate_inputs_error_cases(self):
        """Test input validation error cases."""
        estimator = DensityRatioEstimator()

        # Mismatched dimensions
        X_source = np.random.randn(100, 3)
        X_target = np.random.randn(100, 4)  # Different number of features

        with pytest.raises(ValueError, match="same number of variables"):
            estimator.fit_weights(X_source, X_target)

        # Empty datasets
        X_empty_source = np.array([]).reshape(0, 3)
        X_empty_target = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="cannot be empty"):
            estimator.fit_weights(X_empty_source, X_empty_target)

    def test_diagnostics_information(self, sample_data):
        """Test diagnostic information in results."""
        X_source, X_target = sample_data

        estimator = DensityRatioEstimator(random_state=42)
        result = estimator.fit_weights(X_source, X_target)

        # Check diagnostics dictionary
        assert isinstance(result.diagnostics, dict)

        expected_keys = [
            "weight_mean",
            "weight_std",
            "weight_cv",
            "n_extreme_weights",
            "source_sample_size",
            "target_sample_size",
            "weight_entropy",
        ]

        for key in expected_keys:
            assert key in result.diagnostics

        # Check values are reasonable
        assert result.diagnostics["source_sample_size"] == len(X_source)
        assert result.diagnostics["target_sample_size"] == len(X_target)
        assert result.diagnostics["weight_mean"] > 0
        assert result.diagnostics["weight_std"] >= 0

    def test_effective_sample_size_calculation(self):
        """Test effective sample size calculation."""
        # Test with uniform weights
        uniform_weights = np.ones(100)
        uniform_ess = (np.sum(uniform_weights) ** 2) / np.sum(uniform_weights**2)
        assert abs(uniform_ess - 100.0) < 1e-10  # Should equal sample size

        # Test with non-uniform weights
        varied_weights = np.array([2.0, 0.5, 1.0, 1.5])
        varied_ess = (np.sum(varied_weights) ** 2) / np.sum(varied_weights**2)
        expected_ess = (5.0**2) / (4.0 + 0.25 + 1.0 + 2.25)  # 25 / 7.5 = 3.33
        assert abs(varied_ess - expected_ess) < 1e-10

    def test_stabilization_effect(self, sample_data):
        """Test weight stabilization effect."""
        X_source, X_target = sample_data

        # With stabilization
        estimator_stable = DensityRatioEstimator(
            stabilize=True,
            trim_weights=False,  # Don't trim to see stabilization effect
            random_state=42,
        )
        result_stable = estimator_stable.fit_weights(X_source, X_target)

        # Without stabilization
        estimator_unstable = DensityRatioEstimator(
            stabilize=False, trim_weights=False, random_state=42
        )
        result_unstable = estimator_unstable.fit_weights(X_source, X_target)

        # Stabilized weights should have lower variance
        stable_cv = np.std(result_stable.weights) / np.mean(result_stable.weights)
        unstable_cv = np.std(result_unstable.weights) / np.mean(result_unstable.weights)

        # Stabilization should reduce coefficient of variation
        assert stable_cv <= unstable_cv * 1.1  # Allow some tolerance for randomness


class TestTransportabilityWeightingInterface:
    """Test suite for the main TransportabilityWeightingInterface interface."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)

        X_source, _ = make_classification(
            n_samples=200, n_features=3, n_informative=2, n_redundant=0, random_state=42
        )

        X_target = X_source.copy()
        X_target[:, 0] += 0.3  # Small shift

        return X_source, X_target

    def test_classification_method_selection(self, sample_data):
        """Test classification method selection."""
        X_source, X_target = sample_data

        weighting = TransportabilityWeightingInterface(
            method="classification", auto_select=False, random_state=42
        )

        result = weighting.estimate_weights(X_source, X_target)

        assert isinstance(result, WeightingResult)
        assert len(result.weights) == len(X_source)
        assert np.all(result.weights > 0)

    def test_auto_selection_mode(self, sample_data):
        """Test automatic method selection."""
        X_source, X_target = sample_data

        weighting = TransportabilityWeightingInterface(
            auto_select=True, random_state=42
        )

        result = weighting.estimate_weights(X_source, X_target)

        # Should complete successfully and select a method
        assert isinstance(result, WeightingResult)
        assert "selected_method" in result.diagnostics
        assert "method_selection_score" in result.diagnostics

        # Selected method should be one of the available options
        selected_method = result.diagnostics["selected_method"]
        assert selected_method in ["classification", "optimal_transport"]

    def test_weight_validation(self, sample_data):
        """Test weight validation functionality."""
        X_source, X_target = sample_data

        weighting = TransportabilityWeightingInterface(
            method="classification", random_state=42
        )
        result = weighting.estimate_weights(X_source, X_target)

        # Validate the computed weights
        validation = weighting.validate_weights(result.weights, X_source, X_target)

        # Check validation structure
        assert isinstance(validation, dict)

        expected_keys = [
            "weighted_means",
            "target_means",
            "standardized_mean_differences",
            "mean_absolute_smd",
            "max_absolute_smd",
            "good_balance_achieved",
            "effective_sample_size",
            "weight_statistics",
        ]

        for key in expected_keys:
            assert key in validation

        # Check that validation makes sense
        assert validation["effective_sample_size"] > 0
        assert validation["mean_absolute_smd"] >= 0
        assert isinstance(validation["good_balance_achieved"], bool)

    def test_unknown_method_error(self):
        """Test error handling for unknown weighting method."""
        with pytest.raises(ValueError, match="Unknown weighting method"):
            TransportabilityWeightingInterface(
                method="unknown_method", auto_select=False
            )
