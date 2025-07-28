"""Tests for survival analysis estimators.

This module contains comprehensive tests for causal survival analysis methods
including G-computation, IPW, and AIPW for time-to-event outcomes.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from causal_inference.core.base import (
    CovariateData,
    SurvivalOutcomeData,
    TreatmentData,
)
from causal_inference.estimators.survival_aipw import SurvivalAIPWEstimator
from causal_inference.estimators.survival_g_computation import (
    SurvivalGComputationEstimator,
)
from causal_inference.estimators.survival_ipw import SurvivalIPWEstimator


class TestSurvivalOutcomeData:
    """Tests for SurvivalOutcomeData model."""

    def test_basic_survival_data_creation(self):
        """Test basic creation of survival outcome data."""
        times = np.array([10, 20, 30, 15, 25])
        events = np.array([1, 0, 1, 1, 0])

        survival_data = SurvivalOutcomeData(times=times, events=events)

        assert len(survival_data.times) == 5
        assert len(survival_data.events) == 5
        assert survival_data.n_events == 3
        assert survival_data.n_censored == 2
        assert survival_data.censoring_rate == 0.4
        assert survival_data.outcome_type == "time_to_event"

    def test_survival_data_validation(self):
        """Test validation of survival outcome data."""
        # Test negative times
        with pytest.raises(ValueError, match="Survival times must be positive"):
            SurvivalOutcomeData(times=np.array([-1, 5, 10]), events=np.array([1, 0, 1]))

        # Test mismatched lengths
        with pytest.raises(
            ValueError, match="Times and events must have the same length"
        ):
            SurvivalOutcomeData(times=np.array([10, 20]), events=np.array([1, 0, 1]))

        # Test invalid event indicators
        with pytest.raises(ValueError, match="Event indicators must be 0.*or 1"):
            SurvivalOutcomeData(times=np.array([10, 20]), events=np.array([1, 2]))

        # Test empty data
        with pytest.raises(ValueError, match="Survival times cannot be empty"):
            SurvivalOutcomeData(times=np.array([]), events=np.array([]))

    def test_survival_data_with_competing_risks(self):
        """Test survival data with competing risks."""
        times = np.array([10, 20, 30, 15, 25])
        events = np.array([1, 1, 0, 1, 1])
        event_types = np.array(
            [1, 2, 0, 1, 2]
        )  # 0=censored, 1=event of interest, 2=competing

        survival_data = SurvivalOutcomeData(
            times=times,
            events=events,
            event_types=event_types,
            outcome_type="competing_risks",
        )

        assert survival_data.outcome_type == "competing_risks"
        assert len(survival_data.event_types) == 5

    def test_lifelines_format_conversion(self):
        """Test conversion to lifelines format."""
        times = np.array([10, 20, 30])
        events = np.array([1, 0, 1])

        survival_data = SurvivalOutcomeData(times=times, events=events)
        df = survival_data.to_lifelines_format()

        assert list(df.columns) == ["T", "E"]
        assert_array_equal(df["T"].values, times)
        assert_array_equal(df["E"].values, events)


class TestSurvivalDataGeneration:
    """Tests for survival data generation utilities."""

    @staticmethod
    def generate_survival_data(
        n: int = 1000,
        treatment_effect_hr: float = 0.7,
        censoring_rate: float = 0.3,
        random_state: int = 42,
    ) -> tuple[TreatmentData, SurvivalOutcomeData, CovariateData]:
        """Generate simulated survival data for testing.

        Args:
            n: Sample size
            treatment_effect_hr: True hazard ratio (treatment effect)
            censoring_rate: Proportion of observations to censor
            random_state: Random seed

        Returns:
            Tuple of (treatment, outcome, covariates)
        """
        np.random.seed(random_state)

        # Generate covariates
        age = np.random.normal(50, 15, n)
        sex = np.random.binomial(1, 0.5, n)
        covariates_df = pd.DataFrame({"age": age, "sex": sex})

        # Generate treatment with confounding
        logit_p = -2 + 0.05 * (age - 50) + 0.3 * sex
        treatment_prob = 1 / (1 + np.exp(-logit_p))
        treatment = np.random.binomial(1, treatment_prob, n)

        # Generate survival times with treatment effect
        # Base hazard with covariate effects
        lambda_base = 0.01
        hazard_multiplier = np.exp(0.02 * (age - 50) + 0.2 * sex)

        # Apply treatment effect
        hazard_multiplier = np.where(
            treatment == 1,
            hazard_multiplier / treatment_effect_hr,  # Protective treatment
            hazard_multiplier,
        )

        # Generate exponential survival times
        survival_times = np.random.exponential(1 / (lambda_base * hazard_multiplier))

        # Add administrative censoring
        censoring_times = np.random.exponential(
            np.percentile(survival_times, (1 - censoring_rate) * 100)
        )

        observed_times = np.minimum(survival_times, censoring_times)
        events = (survival_times <= censoring_times).astype(int)

        return (
            TreatmentData(values=treatment, treatment_type="binary"),
            SurvivalOutcomeData(times=observed_times, events=events),
            CovariateData(values=covariates_df, names=["age", "sex"]),
        )


class TestSurvivalGComputationEstimator:
    """Tests for G-computation survival estimator."""

    def test_gcomputation_estimator_initialization(self):
        """Test G-computation estimator initialization."""
        estimator = SurvivalGComputationEstimator(
            survival_model="cox", time_horizon=100, random_state=42
        )

        assert estimator.method == "g_computation"
        assert estimator.survival_model == "cox"
        assert estimator.time_horizon == 100
        assert not estimator.is_fitted

    def test_gcomputation_fitting_and_estimation(self):
        """Test G-computation fitting and causal effect estimation."""
        # Generate test data
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(
                n=500, treatment_effect_hr=0.7, random_state=42
            )
        )

        # Initialize and fit estimator
        estimator = SurvivalGComputationEstimator(
            survival_model="cox", time_horizon=50, random_state=42
        )

        estimator.fit(treatment, outcome, covariates)
        assert estimator.is_fitted

        # Estimate causal effect
        causal_effect = estimator.estimate_ate()

        assert causal_effect.method == "g_computation_survival"
        assert causal_effect.n_observations == 500
        assert causal_effect.hazard_ratio is not None
        assert causal_effect.survival_curves is not None

        # Check that hazard ratio is in reasonable range
        # (should be close to true HR of 0.7)
        assert 0.5 < causal_effect.hazard_ratio < 1.2

    def test_gcomputation_survival_curves(self):
        """Test survival curve estimation."""
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(n=300, random_state=42)
        )

        estimator = SurvivalGComputationEstimator(survival_model="cox")
        estimator.fit(treatment, outcome, covariates)

        # Get survival curves
        curves = estimator.estimate_survival_curves()

        assert "treated" in curves
        assert "control" in curves

        treated_curve = curves["treated"]
        control_curve = curves["control"]

        # Check that curves are properly formatted
        assert "timeline" in treated_curve.columns
        assert "survival_prob" in treated_curve.columns

        # Check that survival probabilities are monotonically decreasing
        treated_probs = treated_curve["survival_prob"].values
        control_probs = control_curve["survival_prob"].values

        assert np.all(np.diff(treated_probs) <= 1e-10)  # Allow for numerical precision
        assert np.all(np.diff(control_probs) <= 1e-10)

        # Check that survival starts at 1
        assert treated_probs[0] == approx(1.0, abs=1e-3)
        assert control_probs[0] == approx(1.0, abs=1e-3)

    def test_gcomputation_rmst_calculation(self):
        """Test RMST calculation."""
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(
                n=400, treatment_effect_hr=0.6, random_state=42
            )
        )

        estimator = SurvivalGComputationEstimator(survival_model="cox", time_horizon=30)
        estimator.fit(treatment, outcome, covariates)

        # Calculate RMST
        rmst_results = estimator.estimate_rmst_difference()

        assert "rmst_treated" in rmst_results
        assert "rmst_control" in rmst_results
        assert "rmst_difference" in rmst_results

        # Treatment should improve survival (positive RMST difference)
        assert rmst_results["rmst_difference"] > 0
        assert rmst_results["rmst_treated"] > rmst_results["rmst_control"]


class TestSurvivalIPWEstimator:
    """Tests for IPW survival estimator."""

    def test_ipw_estimator_initialization(self):
        """Test IPW estimator initialization."""
        estimator = SurvivalIPWEstimator(
            propensity_model="logistic",
            weight_stabilization=True,
            weight_trimming=0.05,
            random_state=42,
        )

        assert estimator.method == "ipw"
        assert estimator.propensity_model_type == "logistic"
        assert estimator.weight_stabilization is True
        assert estimator.weight_trimming == 0.05

    def test_ipw_fitting_and_estimation(self):
        """Test IPW fitting and causal effect estimation."""
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(
                n=400, treatment_effect_hr=0.8, random_state=42
            )
        )

        estimator = SurvivalIPWEstimator(
            propensity_model="logistic", time_horizon=40, random_state=42
        )

        estimator.fit(treatment, outcome, covariates)
        assert estimator.is_fitted

        # Check propensity scores
        assert estimator.propensity_scores is not None
        assert len(estimator.propensity_scores) == 400
        assert np.all(
            (estimator.propensity_scores >= 0) & (estimator.propensity_scores <= 1)
        )

        # Check weights
        assert estimator.weights is not None
        assert estimator.stabilized_weights is not None

        # Estimate causal effect
        causal_effect = estimator.estimate_ate()

        assert causal_effect.method == "ipw_survival"
        assert causal_effect.rmst_difference is not None

    def test_ipw_weight_diagnostics(self):
        """Test IPW weight diagnostics."""
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(n=300, random_state=42)
        )

        estimator = SurvivalIPWEstimator(propensity_model="logistic")
        estimator.fit(treatment, outcome, covariates)

        # Get weight diagnostics
        diagnostics = estimator.get_weight_diagnostics()

        assert "propensity_scores" in diagnostics
        assert "weights" in diagnostics
        assert "effective_sample_sizes" in diagnostics

        # Check that effective sample sizes are reasonable
        ess = diagnostics["effective_sample_sizes"]
        assert ess["treated"] > 0
        assert ess["control"] > 0
        assert ess["total"] > 0

    def test_ipw_weighted_survival_curves(self):
        """Test weighted survival curve estimation."""
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(n=350, random_state=42)
        )

        estimator = SurvivalIPWEstimator(propensity_model="random_forest")
        estimator.fit(treatment, outcome, covariates)

        # Get weighted survival curves
        curves = estimator.estimate_survival_curves()

        assert "treated" in curves
        assert "control" in curves

        # Curves should be properly formatted
        for group_name, curve in curves.items():
            assert "timeline" in curve.columns
            assert "survival_prob" in curve.columns
            assert len(curve) > 0


class TestSurvivalAIPWEstimator:
    """Tests for AIPW survival estimator."""

    def test_aipw_estimator_initialization(self):
        """Test AIPW estimator initialization."""
        estimator = SurvivalAIPWEstimator(
            survival_model="cox",
            propensity_model="logistic",
            time_horizon=50,
            random_state=42,
        )

        assert estimator.method == "aipw"
        assert estimator.survival_model == "cox"
        assert estimator.propensity_model_type == "logistic"

    def test_aipw_fitting_and_estimation(self):
        """Test AIPW fitting and causal effect estimation."""
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(
                n=450, treatment_effect_hr=0.75, random_state=42
            )
        )

        estimator = SurvivalAIPWEstimator(
            survival_model="cox",
            propensity_model="logistic",
            time_horizon=35,
            random_state=42,
        )

        estimator.fit(treatment, outcome, covariates)
        assert estimator.is_fitted

        # Check that both models are fitted
        assert estimator.fitted_model is not None
        assert estimator.propensity_model is not None

        # Estimate causal effect
        causal_effect = estimator.estimate_ate()

        assert causal_effect.method == "aipw_survival"
        assert causal_effect.diagnostics["doubly_robust"] is True

        # Should have both HR and RMST estimates
        assert causal_effect.hazard_ratio is not None
        assert causal_effect.rmst_difference is not None

    def test_aipw_component_comparison(self):
        """Test comparison of AIPW components."""
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(
                n=300, treatment_effect_hr=0.7, random_state=42
            )
        )

        estimator = SurvivalAIPWEstimator(
            survival_model="cox",
            propensity_model="logistic",
            time_horizon=30,
            random_state=42,
        )

        estimator.fit(treatment, outcome, covariates)

        # Compare G-computation, IPW, and AIPW components
        comparison = estimator.compare_components()

        assert "g_computation" in comparison
        assert "ipw" in comparison
        assert "aipw" in comparison

        # Each should have RMST estimates
        for method in ["g_computation", "ipw", "aipw"]:
            assert "rmst_treated" in comparison[method]
            assert "rmst_control" in comparison[method]


class TestSurvivalEstimatorEdgeCases:
    """Tests for edge cases in survival estimators."""

    def test_insufficient_events_error(self):
        """Test error when insufficient events for survival analysis."""
        # Create data with very few events
        times = np.array([10, 20, 30, 40, 50])
        events = np.array([1, 0, 0, 0, 0])  # Only 1 event
        treatment = np.array([1, 0, 1, 0, 1])
        covariates_df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = SurvivalOutcomeData(times=times, events=events)
        covariate_data = CovariateData(values=covariates_df, names=["x"])

        estimator = SurvivalGComputationEstimator()

        with pytest.raises(
            Exception
        ):  # Should raise EstimationError about insufficient events
            estimator.fit(treatment_data, outcome_data, covariate_data)

    def test_high_censoring_warning(self):
        """Test warning for high censoring rates."""
        # Create data with high censoring rate
        times = np.random.exponential(10, 100)
        events = np.random.binomial(1, 0.05, 100)  # 95% censored
        treatment = np.random.binomial(1, 0.5, 100)
        covariates_df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = SurvivalOutcomeData(times=times, events=events)
        covariate_data = CovariateData(values=covariates_df, names=["x"])

        estimator = SurvivalGComputationEstimator()

        # Should warn about high censoring rate but still fit
        with pytest.warns(UserWarning, match="High censoring rate"):
            estimator.fit(treatment_data, outcome_data, covariate_data)

    def test_no_covariates_ipw_error(self):
        """Test that IPW requires covariates."""
        times = np.random.exponential(10, 100)
        events = np.random.binomial(1, 0.3, 100)
        treatment = np.random.binomial(1, 0.5, 100)

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = SurvivalOutcomeData(times=times, events=events)

        estimator = SurvivalIPWEstimator()

        with pytest.raises(Exception, match="Covariates are required"):
            estimator.fit(treatment_data, outcome_data)

    def test_rmst_without_time_horizon_error(self):
        """Test RMST calculation requires time_horizon."""
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(n=200, random_state=42)
        )

        # Don't set time_horizon
        estimator = SurvivalGComputationEstimator(survival_model="cox")
        estimator.fit(treatment, outcome, covariates)

        with pytest.raises(Exception, match="time_horizon must be set"):
            estimator.estimate_rmst_difference()


class TestSurvivalEstimatorIntegration:
    """Integration tests comparing different survival estimators."""

    def test_estimator_consistency(self):
        """Test that different estimators give consistent results on the same data."""
        # Generate data with clear treatment effect
        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(
                n=800, treatment_effect_hr=0.6, random_state=42
            )
        )

        # Fit all three estimators
        gcomp = SurvivalGComputationEstimator(
            survival_model="cox", time_horizon=40, random_state=42
        )
        ipw = SurvivalIPWEstimator(
            propensity_model="logistic", time_horizon=40, random_state=42
        )
        aipw = SurvivalAIPWEstimator(
            survival_model="cox",
            propensity_model="logistic",
            time_horizon=40,
            random_state=42,
        )

        gcomp.fit(treatment, outcome, covariates)
        ipw.fit(treatment, outcome, covariates)
        aipw.fit(treatment, outcome, covariates)

        # Get causal effects
        gcomp_effect = gcomp.estimate_ate()
        ipw_effect = ipw.estimate_ate()
        aipw_effect = aipw.estimate_ate()

        # All should detect protective treatment effect (positive RMST difference)
        assert gcomp_effect.rmst_difference > 0
        assert ipw_effect.rmst_difference > 0
        assert aipw_effect.rmst_difference > 0

        # Effects should be reasonably close (within 50% of each other)
        rmst_diffs = [
            gcomp_effect.rmst_difference,
            ipw_effect.rmst_difference,
            aipw_effect.rmst_difference,
        ]

        mean_diff = np.mean(rmst_diffs)
        for diff in rmst_diffs:
            assert abs(diff - mean_diff) / mean_diff < 0.5  # Within 50%

    def test_method_robustness(self):
        """Test robustness of methods to model misspecification."""
        # This is a simplified test - in practice you'd want more sophisticated
        # tests of double robustness for AIPW

        treatment, outcome, covariates = (
            TestSurvivalDataGeneration.generate_survival_data(
                n=600, treatment_effect_hr=0.7, random_state=42
            )
        )

        # Test different model specifications
        estimators = [
            SurvivalGComputationEstimator(survival_model="cox", time_horizon=35),
            SurvivalIPWEstimator(propensity_model="logistic", time_horizon=35),
            SurvivalIPWEstimator(propensity_model="random_forest", time_horizon=35),
            SurvivalAIPWEstimator(
                survival_model="cox", propensity_model="logistic", time_horizon=35
            ),
        ]

        effects = []
        for estimator in estimators:
            estimator.fit(treatment, outcome, covariates)
            effect = estimator.estimate_ate()
            effects.append(effect.rmst_difference)

        # All methods should give positive treatment effects
        for effect in effects:
            assert effect > 0

        # Variation shouldn't be too extreme
        assert np.std(effects) / np.mean(effects) < 0.5
