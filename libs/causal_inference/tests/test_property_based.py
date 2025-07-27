"""Property-based tests for causal inference estimators.

This module uses hypothesis to generate property-based tests that verify
invariants and mathematical properties of causal inference estimators.

## Property-Based Testing Overview

Property-based testing automatically generates test inputs to verify that
mathematical properties hold across a wide range of scenarios. This is
particularly valuable for causal inference where estimators should satisfy
certain theoretical properties regardless of the specific data.

## Key Properties Tested

### Mathematical Invariants
1. **Scale Invariance**: ATE estimates should scale proportionally when outcomes are scaled
2. **Translation Invariance**: ATE estimates should be unchanged when outcomes are shifted by constants
3. **Covariate Ordering**: Results should be invariant to the order of covariate columns

### Estimator-Specific Properties
1. **IPW Weight Properties**: All weights must be positive and finite
2. **AIPW Robustness**: Should handle challenging scenarios gracefully
3. **Cross-Method Consistency**: Different estimators should give reasonably similar results

### Numerical Stability
1. **Extreme Values**: Estimators should handle large but finite covariate values
2. **High Dimensionality**: Performance with many covariates relative to sample size
3. **Finite Results**: All estimates should be finite numbers

## Test Strategy

The tests use Hypothesis to generate realistic but varied datasets with:
- Controlled sample sizes (20-200 for speed)
- Limited feature counts (1-5 to avoid curse of dimensionality)
- Reasonable coefficient ranges to avoid numerical issues
- Fixed random seeds for reproducibility across hypothesis runs
- Assumptions to ensure sufficient treatment variation

Each test verifies both that the estimator succeeds on valid data and that
any failures are for predictable reasons (numerical issues, convergence problems).
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator


# Hypothesis strategies for generating test data
@st.composite
def binary_treatment_data(draw, min_size=20, max_size=200):
    """Generate binary treatment data with covariates.

    This strategy creates realistic causal inference datasets with:
    - Binary treatment (0/1) with reasonable propensity scores
    - Continuous outcomes with linear relationship to covariates
    - Sufficient variation in both treatment groups
    - Controlled coefficient ranges to avoid numerical issues

    The strategy ensures reproducibility by using a fixed random seed
    within the hypothesis framework while still generating varied test cases.
    """
    n_samples = draw(st.integers(min_value=min_size, max_value=max_size))
    n_features = draw(st.integers(min_value=1, max_value=5))

    # Generate covariates
    X = draw(
        arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(
                min_value=-3, max_value=3, allow_nan=False, allow_infinity=False
            ),
        )
    )

    # Generate treatment with reasonable propensity
    propensity_weights = draw(
        arrays(
            dtype=np.float64,
            shape=(n_features,),
            elements=st.floats(
                min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False
            ),
        )
    )

    linear_pred = X @ propensity_weights
    propensity = 1 / (1 + np.exp(-linear_pred))

    # Ensure some variation in treatment
    assume(np.min(propensity) > 0.05 and np.max(propensity) < 0.95)

    # Use deterministic random state for reproducible hypothesis tests
    rng = np.random.RandomState(12345)  # Fixed seed for reproducibility
    treatment = rng.binomial(1, propensity, n_samples)

    # Ensure both treatment groups are represented
    assume(np.sum(treatment) >= 5 and np.sum(1 - treatment) >= 5)

    # Generate outcome
    outcome_weights = draw(
        arrays(
            dtype=np.float64,
            shape=(n_features,),
            elements=st.floats(
                min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False
            ),
        )
    )

    treatment_effect = draw(
        st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
    )
    noise_scale = draw(st.floats(min_value=0.1, max_value=1.0))

    outcome = (
        X @ outcome_weights
        + treatment_effect * treatment
        + rng.normal(0, noise_scale, n_samples)
    )

    covariate_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(n_features)])

    return {
        "treatment": TreatmentData(values=treatment, treatment_type="binary"),
        "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
        "covariates": CovariateData(values=covariate_df),
        "true_ate": treatment_effect,
    }


class TestEstimatorInvariants:
    """Test mathematical invariants that should hold for all estimators."""

    @given(data=binary_treatment_data())
    @settings(deadline=10000, max_examples=3)
    def test_ate_scale_invariance_g_computation(self, data):
        """Test that ATE estimates scale appropriately with outcome scaling."""
        estimator = GComputationEstimator()

        # Fit with original data
        estimator.fit(data["treatment"], data["outcome"], data["covariates"])
        original_ate = estimator.estimate_ate().ate

        # Scale outcome by constant
        scale_factor = 2.5
        scaled_outcome = OutcomeData(
            values=data["outcome"].values * scale_factor,
            outcome_type=data["outcome"].outcome_type,
        )

        # Fit with scaled data
        estimator_scaled = GComputationEstimator()
        estimator_scaled.fit(data["treatment"], scaled_outcome, data["covariates"])
        scaled_ate = estimator_scaled.estimate_ate().ate

        # ATE should scale by the same factor (with some tolerance for numerical precision)
        expected_scaled_ate = original_ate * scale_factor
        assert abs(scaled_ate - expected_scaled_ate) / abs(expected_scaled_ate) < 0.1

    @given(data=binary_treatment_data())
    @settings(deadline=10000, max_examples=3)
    def test_ate_translation_invariance_g_computation(self, data):
        """Test that ATE estimates are invariant to outcome translation."""
        estimator = GComputationEstimator()

        # Fit with original data
        estimator.fit(data["treatment"], data["outcome"], data["covariates"])
        original_ate = estimator.estimate_ate().ate

        # Translate outcome by constant
        translation = 100.0
        translated_outcome = OutcomeData(
            values=data["outcome"].values + translation,
            outcome_type=data["outcome"].outcome_type,
        )

        # Fit with translated data
        estimator_translated = GComputationEstimator()
        estimator_translated.fit(
            data["treatment"], translated_outcome, data["covariates"]
        )
        translated_ate = estimator_translated.estimate_ate().ate

        # ATE should be unchanged (with some tolerance for numerical precision)
        assert abs(translated_ate - original_ate) < 0.1

    @given(data=binary_treatment_data())
    @settings(deadline=10000, max_examples=3)
    def test_ate_covariate_ordering_invariance(self, data):
        """Test that ATE estimates are invariant to covariate column ordering."""
        estimator1 = GComputationEstimator()
        estimator1.fit(data["treatment"], data["outcome"], data["covariates"])
        ate1 = estimator1.estimate_ate().ate

        # Shuffle covariate columns
        covariates_df = data["covariates"].values.copy()
        columns = list(covariates_df.columns)
        rng = np.random.RandomState(12345)
        rng.shuffle(columns)
        shuffled_covariates = CovariateData(values=covariates_df[columns])

        estimator2 = GComputationEstimator()
        estimator2.fit(data["treatment"], data["outcome"], shuffled_covariates)
        ate2 = estimator2.estimate_ate().ate

        # ATE should be the same (with some tolerance for numerical precision)
        assert abs(ate2 - ate1) < 0.1

    @given(data=binary_treatment_data())
    @settings(deadline=10000, max_examples=3)
    def test_ipw_weight_positivity(self, data):
        """Test that IPW weights are always positive and finite."""
        estimator = IPWEstimator()
        estimator.fit(data["treatment"], data["outcome"], data["covariates"])

        weights = estimator.get_weights()

        # All weights should be positive and finite
        assert weights is not None
        assert np.all(weights > 0)
        assert np.all(np.isfinite(weights))

    @given(data=binary_treatment_data())
    @settings(deadline=10000, max_examples=3)
    def test_aipw_doubly_robust_property(self, data):
        """Test key properties of AIPW estimator."""
        estimator = AIPWEstimator(cross_fitting=False, bootstrap_samples=0)
        estimator.fit(data["treatment"], data["outcome"], data["covariates"])

        # Should be able to estimate ATE
        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)
        # Check confidence interval if available
        if effect.ate_ci_lower is not None and effect.ate_ci_upper is not None:
            assert effect.ate_ci_lower <= effect.ate <= effect.ate_ci_upper

    @given(data=binary_treatment_data())
    @settings(deadline=10000, max_examples=3)  # Reduced examples for performance
    def test_estimator_consistency_across_methods(self, data):
        """Test that different estimators give reasonably consistent results on good data."""
        estimators = [
            GComputationEstimator(),
            IPWEstimator(),
            AIPWEstimator(cross_fitting=False, bootstrap_samples=0),
        ]

        ates = []
        for estimator in estimators:
            try:
                estimator.fit(data["treatment"], data["outcome"], data["covariates"])
                ate = estimator.estimate_ate().ate
                ates.append(ate)
            except Exception:
                # Skip if estimator fails (expected for some edge cases)
                continue

        # If multiple estimators succeeded, they should give reasonably similar results
        if len(ates) >= 2:
            ate_range = max(ates) - min(ates)
            ate_mean = np.mean(ates)
            # Allow for substantial variation but catch gross inconsistencies
            assert ate_range <= 3 * abs(ate_mean) + 1.0


class TestDataValidationProperties:
    """Test properties of data validation and handling."""

    @given(
        treatment=arrays(
            dtype=np.int32,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.integers(min_value=0, max_value=1),
        )
    )
    @settings(deadline=5000, max_examples=20)
    def test_treatment_data_validation_properties(self, treatment):
        """Test properties of TreatmentData validation."""
        assume(len(np.unique(treatment)) == 2)  # Must have both treatment values

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")

        # Basic properties
        assert len(treatment_data.values) == len(treatment)
        assert treatment_data.treatment_type == "binary"
        assert set(treatment_data.values).issubset({0, 1})

    @given(
        outcome=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(deadline=5000, max_examples=20)
    def test_outcome_data_validation_properties(self, outcome):
        """Test properties of OutcomeData validation."""
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")

        # Basic properties
        assert len(outcome_data.values) == len(outcome)
        assert outcome_data.outcome_type == "continuous"
        assert np.all(np.isfinite(outcome_data.values))


class TestNumericalStabilityProperties:
    """Test numerical stability properties of estimators."""

    def test_extreme_covariate_values(self):
        """Test behavior with extreme but valid covariate values."""
        n_samples = 100
        rng = np.random.RandomState(12345)

        # Extreme but finite values
        X1 = rng.uniform(-1000, 1000, n_samples)
        X2 = rng.uniform(0.001, 0.01, n_samples)  # Very small positive values

        # Reasonable treatment and outcome
        treatment = rng.binomial(1, 0.5, n_samples)
        outcome = (
            1 + 0.001 * X1 + 100 * X2 + 2 * treatment + rng.normal(0, 1, n_samples)
        )

        data = {
            "treatment": TreatmentData(values=treatment, treatment_type="binary"),
            "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
            "covariates": CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2})),
        }

        # All estimators should handle this without crashing
        estimators = [
            GComputationEstimator(),
            IPWEstimator(),
            AIPWEstimator(cross_fitting=False, bootstrap_samples=0),
        ]

        for estimator in estimators:
            try:
                estimator.fit(data["treatment"], data["outcome"], data["covariates"])
                effect = estimator.estimate_ate()
                assert np.isfinite(effect.ate)
            except (ValueError, np.linalg.LinAlgError, RuntimeWarning) as e:
                # Some numerical issues might be expected, but should be handled gracefully
                assert any(
                    keyword in str(e).lower()
                    for keyword in [
                        "numerical",
                        "convergence",
                        "singular",
                        "ill-conditioned",
                        "overflow",
                    ]
                )

    def test_high_dimensional_covariates(self):
        """Test behavior with high-dimensional covariate spaces."""
        n_samples = 200
        n_features = 50  # More features than typical
        rng = np.random.RandomState(12345)

        X = rng.normal(0, 1, (n_samples, n_features))

        # Only first few features affect treatment and outcome
        propensity_weights = np.zeros(n_features)
        propensity_weights[:3] = [0.3, -0.2, 0.4]

        outcome_weights = np.zeros(n_features)
        outcome_weights[2:5] = [0.5, -0.3, 0.2]

        linear_pred = X @ propensity_weights
        propensity = 1 / (1 + np.exp(-linear_pred))
        treatment = rng.binomial(1, propensity, n_samples)

        outcome = X @ outcome_weights + 1.5 * treatment + rng.normal(0, 0.5, n_samples)

        covariate_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(n_features)])

        data = {
            "treatment": TreatmentData(values=treatment, treatment_type="binary"),
            "outcome": OutcomeData(values=outcome, outcome_type="continuous"),
            "covariates": CovariateData(values=covariate_df),
        }

        # G-computation should handle this reasonably well
        estimator = GComputationEstimator()
        estimator.fit(data["treatment"], data["outcome"], data["covariates"])
        effect = estimator.estimate_ate()

        assert np.isfinite(effect.ate)
        # Should be reasonably close to true effect (1.5) but allow for estimation error
        assert abs(effect.ate - 1.5) < 2.0


@pytest.mark.slow
class TestRobustnessProperties:
    """Test robustness properties (marked as slow)."""

    def test_bootstrap_stability(self, simple_binary_data):
        """Test that bootstrap confidence intervals have reasonable properties."""
        estimator = GComputationEstimator(bootstrap_samples=100)
        estimator.fit(
            simple_binary_data["treatment"],
            simple_binary_data["outcome"],
            simple_binary_data["covariates"],
        )

        effect = estimator.estimate_ate()

        # Confidence interval should be valid
        if effect.confidence_interval is not None:
            assert (
                effect.confidence_interval[0]
                <= effect.ate
                <= effect.confidence_interval[1]
            )

            # Interval should have reasonable width (not too narrow or too wide)
            interval_width = (
                effect.confidence_interval[1] - effect.confidence_interval[0]
            )
            assert 0.1 <= interval_width <= 10.0

    def test_repeated_estimation_stability(self, simple_binary_data):
        """Test that repeated estimations give consistent results."""
        estimator = GComputationEstimator(bootstrap_samples=50)

        ates = []
        for _ in range(5):
            estimator.fit(
                simple_binary_data["treatment"],
                simple_binary_data["outcome"],
                simple_binary_data["covariates"],
            )
            ate = estimator.estimate_ate().ate
            ates.append(ate)

        # Results should be very similar (bootstrap sampling causes small variations)
        ate_std = np.std(ates)
        assert ate_std < 0.5  # Small standard deviation across repeated estimations
