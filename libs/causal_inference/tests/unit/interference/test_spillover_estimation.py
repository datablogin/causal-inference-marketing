"""Unit tests for spillover effect estimation."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.interference.exposure_mapping import ExposureMapping
from causal_inference.interference.spillover_estimation import (
    AdditiveSpilloverModel,
    MultiplicativeSpilloverModel,
    SpilloverEstimator,
    SpilloverResults,
    ThresholdSpilloverModel,
)


class TestSpilloverModels:
    """Test spillover model configurations."""

    def test_additive_spillover_model(self):
        """Test additive spillover model."""
        model = AdditiveSpilloverModel()
        assert model.mechanism == "additive"
        assert not model.include_interactions

    def test_multiplicative_spillover_model(self):
        """Test multiplicative spillover model."""
        model = MultiplicativeSpilloverModel(base_effect=1.5)
        assert model.mechanism == "multiplicative"
        assert model.base_effect == 1.5

    def test_threshold_spillover_model(self):
        """Test threshold spillover model."""
        model = ThresholdSpilloverModel(threshold=0.3)
        assert model.mechanism == "threshold"
        assert model.threshold == 0.3


class TestSpilloverResults:
    """Test spillover results data structure."""

    def test_spillover_results_initialization(self):
        """Test basic spillover results initialization."""
        results = SpilloverResults(direct_effect=0.5, spillover_effect=0.2)

        assert results.direct_effect == 0.5
        assert results.spillover_effect == 0.2
        assert results.total_effect == 0.7  # Should auto-calculate

    def test_spillover_ratio_calculation(self):
        """Test spillover ratio property."""
        results = SpilloverResults(direct_effect=0.4, spillover_effect=0.2)

        assert abs(results.spillover_ratio - 0.5) < 0.01

    def test_spillover_ratio_zero_direct_effect(self):
        """Test spillover ratio with zero direct effect."""
        results = SpilloverResults(direct_effect=0.0, spillover_effect=0.2)

        assert results.spillover_ratio == float("inf")

    def test_spillover_significance(self):
        """Test spillover significance detection."""
        results = SpilloverResults(
            direct_effect=0.5, spillover_effect=0.2, spillover_effect_pvalue=0.03
        )

        assert results.is_spillover_significant

        results.spillover_effect_pvalue = 0.08
        assert not results.is_spillover_significant


class TestSpilloverEstimator:
    """Test spillover effect estimator."""

    def setup_method(self):
        """Set up test data for spillover estimation."""
        # Create simple network structure
        n_units = 100
        np.random.seed(42)

        # Create exposure mapping (each unit connected to 2-3 neighbors)
        self.unit_ids = np.arange(n_units)
        exposure_matrix = np.zeros((n_units, n_units))

        for i in range(n_units):
            # Connect to next 2 units (circular)
            neighbors = [(i + 1) % n_units, (i + 2) % n_units]
            for j in neighbors:
                exposure_matrix[i, j] = 1.0

        self.exposure_mapping = ExposureMapping(
            unit_ids=self.unit_ids,
            exposure_matrix=exposure_matrix,
            exposure_type="binary",
        )

        # Generate treatment assignment
        self.treatment = TreatmentData(
            values=np.random.binomial(1, 0.5, n_units), treatment_type="binary"
        )

        # Generate outcome with known spillover effect
        direct_effect = 0.5
        spillover_effect = 0.3

        spillover_exposure = np.dot(exposure_matrix, self.treatment.values)

        self.outcome = OutcomeData(
            values=(
                direct_effect * self.treatment.values
                + spillover_effect * spillover_exposure
                + np.random.normal(0, 0.5, n_units)
            )
        )

        # Add some covariates
        self.covariates = CovariateData(
            values=pd.DataFrame(
                {
                    "age": np.random.normal(40, 10, n_units),
                    "income": np.random.normal(50000, 15000, n_units),
                }
            ),
            names=["age", "income"],
        )

    def test_spillover_estimator_initialization(self):
        """Test spillover estimator initialization."""
        model = AdditiveSpilloverModel()
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        assert estimator.spillover_model.mechanism == "additive"
        assert estimator.estimator_type == "linear"
        assert not estimator.is_fitted

    def test_spillover_estimator_fit_additive(self):
        """Test fitting spillover estimator with additive model."""
        model = AdditiveSpilloverModel()
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        # Fit the estimator
        estimator.fit(self.treatment, self.outcome, self.covariates)

        assert estimator.is_fitted
        assert estimator._fitted_model is not None
        assert estimator._spillover_exposure is not None

    def test_spillover_estimator_ate_estimation(self):
        """Test ATE estimation with spillover effects."""
        model = AdditiveSpilloverModel()
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        estimator.fit(self.treatment, self.outcome, self.covariates)
        causal_effect = estimator.estimate_ate()

        assert causal_effect.method.startswith("SpilloverEstimator")
        assert causal_effect.ate is not None
        assert "direct_effect" in causal_effect.diagnostics
        assert "spillover_effect" in causal_effect.diagnostics

        # Should recover something close to true total effect (0.5 + 0.3 = 0.8)
        # Allow for some estimation error due to noise
        assert 0.3 < causal_effect.ate < 1.3

    def test_spillover_estimator_get_spillover_results(self):
        """Test getting detailed spillover results."""
        model = AdditiveSpilloverModel()
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        estimator.fit(self.treatment, self.outcome, self.covariates)
        estimator.estimate_ate()

        spillover_results = estimator.get_spillover_results()

        assert spillover_results is not None
        assert spillover_results.direct_effect is not None
        assert spillover_results.spillover_effect is not None
        assert spillover_results.spillover_mechanism == "additive"

        # Should have reasonable estimates (within broad bounds)
        assert -1.0 < spillover_results.direct_effect < 2.0
        assert -1.0 < spillover_results.spillover_effect < 2.0

    def test_spillover_estimator_multiplicative_model(self):
        """Test spillover estimator with multiplicative model."""
        model = MultiplicativeSpilloverModel(include_interactions=True)
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        estimator.fit(self.treatment, self.outcome, self.covariates)
        causal_effect = estimator.estimate_ate()

        assert causal_effect.diagnostics["spillover_mechanism"] == "multiplicative"

    def test_spillover_estimator_threshold_model(self):
        """Test spillover estimator with threshold model."""
        model = ThresholdSpilloverModel(threshold=0.5)
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        estimator.fit(self.treatment, self.outcome, self.covariates)
        causal_effect = estimator.estimate_ate()

        assert causal_effect.diagnostics["spillover_mechanism"] == "threshold"

    def test_spillover_estimator_forest_estimator(self):
        """Test spillover estimator with random forest."""
        model = AdditiveSpilloverModel()
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            estimator_type="forest",
            random_state=42,
        )

        estimator.fit(self.treatment, self.outcome, self.covariates)
        causal_effect = estimator.estimate_ate()

        assert causal_effect.ate is not None

    def test_spillover_estimator_prediction(self):
        """Test spillover effect prediction."""
        model = AdditiveSpilloverModel()
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        estimator.fit(self.treatment, self.outcome, self.covariates)

        # Create a test scenario
        n_units = len(self.treatment.values)
        test_treatment = np.random.binomial(1, 0.3, n_units)

        # Need to provide covariates that match the fitted model's feature count
        test_covariates = (
            self.covariates.values.values
            if isinstance(self.covariates.values, pd.DataFrame)
            else np.array(self.covariates.values)
        )

        predictions = estimator.predict_spillover_effects(
            test_treatment, covariates=test_covariates
        )

        assert "current" in predictions
        assert "no_spillover" in predictions
        assert "spillover_contribution" in predictions
        assert len(predictions["current"]) == n_units

    def test_spillover_estimator_validation_errors(self):
        """Test validation error handling."""
        model = AdditiveSpilloverModel()

        # Wrong exposure mapping size
        wrong_exposure = ExposureMapping(
            unit_ids=np.array([1, 2]),  # Only 2 units
            exposure_matrix=np.array([[0, 1], [1, 0]]),
            exposure_type="binary",
        )

        estimator = SpilloverEstimator(
            spillover_model=model, exposure_mapping=wrong_exposure, random_state=42
        )

        with pytest.raises(ValueError, match="doesn't match"):
            estimator.fit(self.treatment, self.outcome)

    def test_spillover_estimator_estimate_before_fit(self):
        """Test error when estimating before fitting."""
        model = AdditiveSpilloverModel()
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        with pytest.raises(ValueError, match="must be fitted"):
            estimator.estimate_ate()

    def test_spillover_estimator_design_matrix_creation(self):
        """Test design matrix creation for different models."""
        model = AdditiveSpilloverModel(include_interactions=True)
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        estimator.fit(self.treatment, self.outcome, self.covariates)

        # Design matrix should include:
        # - own treatment
        # - spillover exposure
        # - interaction term
        # - covariates
        X = estimator._create_design_matrix(self.treatment, self.covariates)

        # Should have at least 5 columns: treatment + spillover + interaction + 2 covariates
        assert X.shape[1] >= 5
        assert X.shape[0] == len(self.treatment.values)

    def test_spillover_estimator_no_spillover_scenario(self):
        """Test estimator with no actual spillover effects."""
        # Create outcome without spillover effects
        no_spillover_outcome = OutcomeData(
            values=(
                0.5 * self.treatment.values  # Only direct effect
                + np.random.normal(0, 0.5, len(self.treatment.values))
            )
        )

        model = AdditiveSpilloverModel()
        estimator = SpilloverEstimator(
            spillover_model=model,
            exposure_mapping=self.exposure_mapping,
            random_state=42,
        )

        estimator.fit(self.treatment, no_spillover_outcome, self.covariates)
        estimator.estimate_ate()
        spillover_results = estimator.get_spillover_results()

        # Spillover effect should be close to zero
        assert abs(spillover_results.spillover_effect) < 0.3
        # Direct effect should be close to 0.5
        assert 0.2 < spillover_results.direct_effect < 0.8
