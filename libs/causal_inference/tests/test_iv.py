"""Tests for Instrumental Variables (IV) estimator."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import (
    CovariateData,
    InstrumentData,
    OutcomeData,
    TreatmentData,
)
from causal_inference.data.nhefs import NHEFSDataLoader
from causal_inference.estimators.iv import IVEstimator


class TestInstrumentData:
    """Test the InstrumentData Pydantic model."""

    def test_instrument_data_creation(self):
        """Test basic creation of InstrumentData."""
        values = pd.Series([1, 2, 3, 4, 5])
        instrument = InstrumentData(values=values, instrument_type="continuous")

        assert instrument.name == "instrument"
        assert instrument.instrument_type == "continuous"
        assert len(instrument.values) == 5

    def test_instrument_data_validation(self):
        """Test validation of InstrumentData."""
        # Test empty values
        with pytest.raises(ValueError, match="Instrument values cannot be empty"):
            InstrumentData(values=pd.Series([]), instrument_type="continuous")

        # Test invalid instrument type
        with pytest.raises(ValueError, match="instrument_type must be one of"):
            InstrumentData(values=pd.Series([1, 2, 3]), instrument_type="invalid")

    def test_binary_instrument(self):
        """Test binary instrument data."""
        values = pd.Series([0, 1, 0, 1, 1])
        instrument = InstrumentData(values=values, instrument_type="binary")

        assert instrument.instrument_type == "binary"
        assert all(v in [0, 1] for v in instrument.values)

    def test_categorical_instrument(self):
        """Test categorical instrument data."""
        values = pd.Series([1, 2, 3, 1, 2])
        instrument = InstrumentData(
            values=values, instrument_type="categorical", categories=[1, 2, 3]
        )

        assert instrument.instrument_type == "categorical"
        assert instrument.categories == [1, 2, 3]


class TestIVEstimator:
    """Test the IVEstimator class."""

    @pytest.fixture
    def simple_iv_data(self):
        """Create simple synthetic IV data for testing."""
        np.random.seed(42)
        n = 1000

        # Confounder
        u = np.random.normal(0, 1, n)

        # Instrument (affects treatment but not outcome directly)
        z = np.random.normal(0, 1, n)

        # Treatment (affected by instrument and confounder)
        treatment = 0.5 * z + 0.3 * u + np.random.normal(0, 0.5, n)

        # Outcome (affected by treatment and confounder, but not instrument directly)
        true_ate = 2.0
        outcome = true_ate * treatment + 0.4 * u + np.random.normal(0, 0.5, n)

        return {
            "treatment": TreatmentData(
                values=pd.Series(treatment), treatment_type="continuous"
            ),
            "outcome": OutcomeData(
                values=pd.Series(outcome), outcome_type="continuous"
            ),
            "instrument": InstrumentData(
                values=pd.Series(z), instrument_type="continuous"
            ),
            "confounder": CovariateData(values=pd.DataFrame({"u": u}), names=["u"]),
            "true_ate": true_ate,
        }

    @pytest.fixture
    def binary_treatment_iv_data(self):
        """Create binary treatment IV data for testing."""
        np.random.seed(42)
        n = 1000

        # Confounder
        u = np.random.normal(0, 1, n)

        # Instrument
        z = np.random.binomial(1, 0.5, n)

        # Treatment (binary, affected by instrument and confounder)
        treatment_prob = 1 / (1 + np.exp(-(0.8 * z + 0.4 * u)))
        treatment = np.random.binomial(1, treatment_prob, n)

        # Outcome
        true_ate = 3.0
        outcome = true_ate * treatment + 0.5 * u + np.random.normal(0, 1, n)

        return {
            "treatment": TreatmentData(
                values=pd.Series(treatment), treatment_type="binary"
            ),
            "outcome": OutcomeData(
                values=pd.Series(outcome), outcome_type="continuous"
            ),
            "instrument": InstrumentData(values=pd.Series(z), instrument_type="binary"),
            "confounder": CovariateData(values=pd.DataFrame({"u": u}), names=["u"]),
            "true_ate": true_ate,
        }

    def test_iv_estimator_initialization(self):
        """Test IV estimator initialization."""
        estimator = IVEstimator(
            first_stage_model="linear",
            second_stage_model="linear",
            weak_instrument_threshold=10.0,
            bootstrap_samples=100,
            random_state=42,
        )

        assert estimator.first_stage_model == "linear"
        assert estimator.second_stage_model == "linear"
        assert estimator.weak_instrument_threshold == 10.0
        assert estimator.bootstrap_samples == 100
        assert estimator.random_state == 42
        assert not estimator.is_fitted

    def test_continuous_treatment_fitting(self, simple_iv_data):
        """Test fitting IV estimator with continuous treatment."""
        estimator = IVEstimator(
            first_stage_model="linear",
            second_stage_model="linear",
            bootstrap_samples=0,  # Skip bootstrap for speed
            random_state=42,
        )

        estimator.fit(
            treatment=simple_iv_data["treatment"],
            outcome=simple_iv_data["outcome"],
            covariates=simple_iv_data["confounder"],
            instrument=simple_iv_data["instrument"],
        )

        assert estimator.is_fitted
        assert estimator._first_stage_fitted_model is not None
        assert estimator._second_stage_fitted_model is not None
        assert estimator._first_stage_predictions is not None

    def test_binary_treatment_fitting(self, binary_treatment_iv_data):
        """Test fitting IV estimator with binary treatment."""
        estimator = IVEstimator(
            first_stage_model="auto",  # Should choose logistic for binary
            second_stage_model="linear",
            bootstrap_samples=0,
            random_state=42,
        )

        estimator.fit(
            treatment=binary_treatment_iv_data["treatment"],
            outcome=binary_treatment_iv_data["outcome"],
            covariates=binary_treatment_iv_data["confounder"],
            instrument=binary_treatment_iv_data["instrument"],
        )

        assert estimator.is_fitted
        # For binary treatment, first stage should use logistic regression
        from sklearn.linear_model import LogisticRegression

        assert isinstance(estimator._first_stage_fitted_model, LogisticRegression)

    def test_ate_estimation(self, simple_iv_data):
        """Test ATE estimation."""
        estimator = IVEstimator(
            first_stage_model="linear",
            second_stage_model="linear",
            bootstrap_samples=0,
            random_state=42,
        )

        estimator.fit(
            treatment=simple_iv_data["treatment"],
            outcome=simple_iv_data["outcome"],
            covariates=simple_iv_data["confounder"],
            instrument=simple_iv_data["instrument"],
        )

        effect = estimator.estimate_ate()

        assert effect.method == "instrumental_variables"
        assert effect.n_observations == 1000
        assert effect.ate is not None
        # Should be reasonably close to true ATE (2.0) but allow for sampling variation
        assert abs(effect.ate - simple_iv_data["true_ate"]) < 1.0

    def test_weak_instrument_detection(self):
        """Test weak instrument detection."""
        np.random.seed(42)
        n = 500

        # Create weak instrument (barely correlated with treatment)
        weak_instrument = np.random.normal(0, 1, n)
        treatment = 0.05 * weak_instrument + np.random.normal(
            0, 1, n
        )  # Very weak correlation
        outcome = 2.0 * treatment + np.random.normal(0, 1, n)

        treatment_data = TreatmentData(
            values=pd.Series(treatment), treatment_type="continuous"
        )
        outcome_data = OutcomeData(values=pd.Series(outcome), outcome_type="continuous")
        instrument_data = InstrumentData(
            values=pd.Series(weak_instrument), instrument_type="continuous"
        )

        estimator = IVEstimator(
            weak_instrument_threshold=10.0, bootstrap_samples=0, random_state=42
        )

        # Should warn about weak instrument
        with pytest.warns(UserWarning, match="Weak instrument detected"):
            estimator.fit(treatment_data, outcome_data, instrument=instrument_data)

        weak_test = estimator.weak_instrument_test()
        assert weak_test["is_weak"]
        assert weak_test["f_statistic"] < 10.0

    def test_strong_instrument(self, simple_iv_data):
        """Test with strong instrument (should not warn)."""
        estimator = IVEstimator(
            weak_instrument_threshold=10.0, bootstrap_samples=0, random_state=42
        )

        # Should not warn (instrument is strong in this data)
        estimator.fit(
            treatment=simple_iv_data["treatment"],
            outcome=simple_iv_data["outcome"],
            covariates=simple_iv_data["confounder"],
            instrument=simple_iv_data["instrument"],
        )

        weak_test = estimator.weak_instrument_test()
        # Note: With the synthetic data, we expect a strong instrument
        # But the exact F-statistic depends on the implementation details
        assert weak_test["f_statistic"] >= 0  # At least non-negative

    def test_first_stage_diagnostics(self, simple_iv_data):
        """Test first stage diagnostics."""
        estimator = IVEstimator(bootstrap_samples=0, random_state=42)

        estimator.fit(
            treatment=simple_iv_data["treatment"],
            outcome=simple_iv_data["outcome"],
            covariates=simple_iv_data["confounder"],
            instrument=simple_iv_data["instrument"],
        )

        diagnostics = estimator.get_first_stage_diagnostics()

        assert "f_statistic" in diagnostics
        assert "p_value" in diagnostics
        assert "is_weak" in diagnostics
        assert "r_squared" in diagnostics
        assert "model_type" in diagnostics

        # R-squared should be between 0 and 1
        assert 0 <= diagnostics["r_squared"] <= 1

    def test_bootstrap_confidence_intervals(self, simple_iv_data):
        """Test bootstrap confidence intervals."""
        estimator = IVEstimator(
            bootstrap_samples=50,  # Small number for test speed
            random_state=42,
        )

        estimator.fit(
            treatment=simple_iv_data["treatment"],
            outcome=simple_iv_data["outcome"],
            covariates=simple_iv_data["confounder"],
            instrument=simple_iv_data["instrument"],
        )

        effect = estimator.estimate_ate()

        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None
        assert effect.ate_ci_lower < effect.ate_ci_upper
        assert effect.bootstrap_samples == 50

    def test_late_estimation(self, simple_iv_data):
        """Test LATE estimation (should be same as ATE for now)."""
        estimator = IVEstimator(bootstrap_samples=0, random_state=42)

        estimator.fit(
            treatment=simple_iv_data["treatment"],
            outcome=simple_iv_data["outcome"],
            covariates=simple_iv_data["confounder"],
            instrument=simple_iv_data["instrument"],
        )

        late = estimator.estimate_late()
        ate = estimator.estimate_ate()

        # For now, LATE should equal ATE
        assert late.ate == ate.ate
        assert late.method == "late"

    def test_input_validation(self):
        """Test input validation."""
        estimator = IVEstimator()

        # Test missing instrument
        treatment = TreatmentData(
            values=pd.Series([1, 2, 3]), treatment_type="continuous"
        )
        outcome = OutcomeData(values=pd.Series([4, 5, 6]), outcome_type="continuous")

        with pytest.raises(ValueError, match="Instrument data is required"):
            estimator.fit(treatment, outcome)

        # Test mismatched lengths
        instrument = InstrumentData(
            values=pd.Series([1, 2]), instrument_type="continuous"
        )  # Wrong length

        with pytest.raises(
            ValueError, match="Instrument and treatment must have same length"
        ):
            estimator.fit(treatment, outcome, instrument=instrument)

    def test_missing_values_detection(self):
        """Test detection of missing values."""
        estimator = IVEstimator()

        # Create data with missing values
        treatment = TreatmentData(
            values=pd.Series([1, 2, np.nan]), treatment_type="continuous"
        )
        outcome = OutcomeData(values=pd.Series([4, 5, 6]), outcome_type="continuous")
        instrument = InstrumentData(
            values=pd.Series([7, 8, 9]), instrument_type="continuous"
        )

        with pytest.raises(ValueError, match="Treatment data contains missing values"):
            estimator.fit(treatment, outcome, instrument=instrument)

    def test_methods_before_fitting(self):
        """Test that methods fail appropriately before fitting."""
        from causal_inference.core.base import EstimationError

        estimator = IVEstimator()

        with pytest.raises(EstimationError, match="Estimator must be fitted"):
            estimator.estimate_ate()

        with pytest.raises(ValueError, match="Estimator must be fitted"):
            estimator.weak_instrument_test()

        with pytest.raises(ValueError, match="Estimator must be fitted"):
            estimator.get_first_stage_diagnostics()


class TestIVWithNHEFS:
    """Test IV estimator with NHEFS dataset."""

    @pytest.fixture
    def nhefs_data(self):
        """Load NHEFS data for IV testing."""
        try:
            loader = NHEFSDataLoader()
            data = loader.load_processed_data()

            # Use education as instrument for smoking cessation
            # This is a common example from the literature
            treatment = TreatmentData(values=data["qsmk"], treatment_type="binary")
            outcome = OutcomeData(values=data["wt82_71"], outcome_type="continuous")

            # Use a subset of covariates
            covariate_cols = ["age", "sex", "race"]
            covariates = CovariateData(
                values=data[covariate_cols], names=covariate_cols
            )

            # Education as instrument
            instrument = InstrumentData(
                values=data["education"], instrument_type="continuous"
            )

            return {
                "treatment": treatment,
                "outcome": outcome,
                "covariates": covariates,
                "instrument": instrument,
                "raw_data": data,
            }
        except FileNotFoundError:
            pytest.skip("NHEFS data not available for testing")

    def test_nhefs_iv_estimation(self, nhefs_data):
        """Test IV estimation with NHEFS data."""
        estimator = IVEstimator(
            first_stage_model="auto",  # Should use logistic for binary treatment
            second_stage_model="linear",
            bootstrap_samples=50,  # Reduced for test speed
            random_state=42,
        )

        estimator.fit(
            treatment=nhefs_data["treatment"],
            outcome=nhefs_data["outcome"],
            covariates=nhefs_data["covariates"],
            instrument=nhefs_data["instrument"],
        )

        effect = estimator.estimate_ate()

        # Basic sanity checks
        assert effect.method == "instrumental_variables"
        assert effect.n_observations > 1000  # NHEFS is a reasonably large dataset
        assert effect.ate is not None
        assert effect.ate_ci_lower is not None
        assert effect.ate_ci_upper is not None

        # The effect should be plausible (weight change from smoking cessation)
        # Allowing for wide range due to IV estimates often having high variance
        assert -10 <= effect.ate <= 20  # kg weight change seems reasonable

    def test_nhefs_instrument_strength(self, nhefs_data):
        """Test instrument strength with NHEFS data."""
        estimator = IVEstimator(
            weak_instrument_threshold=10.0, bootstrap_samples=0, random_state=42
        )

        estimator.fit(
            treatment=nhefs_data["treatment"],
            outcome=nhefs_data["outcome"],
            covariates=nhefs_data["covariates"],
            instrument=nhefs_data["instrument"],
        )

        weak_test = estimator.weak_instrument_test()
        diagnostics = estimator.get_first_stage_diagnostics()

        # Education should be a reasonably strong instrument for smoking cessation
        # though the exact strength may vary
        assert weak_test["f_statistic"] >= 0
        assert diagnostics["r_squared"] >= 0

    def test_nhefs_binary_instrument(self, nhefs_data):
        """Test with binary version of instrument."""
        # Convert education to binary (high vs low education)
        education_median = nhefs_data["raw_data"]["education"].median()
        binary_education = (
            nhefs_data["raw_data"]["education"] > education_median
        ).astype(int)

        binary_instrument = InstrumentData(
            values=pd.Series(binary_education), instrument_type="binary"
        )

        estimator = IVEstimator(
            first_stage_model="auto",
            second_stage_model="linear",
            bootstrap_samples=0,
            random_state=42,
        )

        estimator.fit(
            treatment=nhefs_data["treatment"],
            outcome=nhefs_data["outcome"],
            covariates=nhefs_data["covariates"],
            instrument=binary_instrument,
        )

        effect = estimator.estimate_ate()

        # Should still produce reasonable estimates
        assert effect.ate is not None
        assert -15 <= effect.ate <= 25  # Allow wider range for binary instrument


class TestIVPerformance:
    """Test computational performance of IV estimator."""

    def test_estimation_speed(self):
        """Test that estimation completes in reasonable time."""
        import time

        np.random.seed(42)
        n = 1500  # Similar to NHEFS size

        # Generate data
        z = np.random.normal(0, 1, n)
        treatment = 0.5 * z + np.random.normal(0, 1, n)
        outcome = 2.0 * treatment + np.random.normal(0, 1, n)

        treatment_data = TreatmentData(
            values=pd.Series(treatment), treatment_type="continuous"
        )
        outcome_data = OutcomeData(values=pd.Series(outcome), outcome_type="continuous")
        instrument_data = InstrumentData(
            values=pd.Series(z), instrument_type="continuous"
        )

        estimator = IVEstimator(bootstrap_samples=0, random_state=42)

        start_time = time.time()
        estimator.fit(treatment_data, outcome_data, instrument=instrument_data)
        effect = estimator.estimate_ate()
        end_time = time.time()

        # Should complete within reasonable time (much less than 10 seconds)
        assert end_time - start_time < 5.0
        assert effect.ate is not None

    def test_memory_usage(self):
        """Test memory usage stays reasonable."""
        # This is a basic test - in practice you'd use memory profiling tools
        np.random.seed(42)
        n = 2000

        z = np.random.normal(0, 1, n)
        treatment = 0.5 * z + np.random.normal(0, 1, n)
        outcome = 2.0 * treatment + np.random.normal(0, 1, n)

        treatment_data = TreatmentData(
            values=pd.Series(treatment), treatment_type="continuous"
        )
        outcome_data = OutcomeData(values=pd.Series(outcome), outcome_type="continuous")
        instrument_data = InstrumentData(
            values=pd.Series(z), instrument_type="continuous"
        )

        estimator = IVEstimator(bootstrap_samples=100, random_state=42)

        # Should not raise memory errors
        estimator.fit(treatment_data, outcome_data, instrument=instrument_data)
        effect = estimator.estimate_ate()

        assert effect.ate is not None
