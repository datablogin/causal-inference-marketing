"""Tests for Bayesian causal inference estimator."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import (
    CovariateData,
    DataValidationError,
    EstimationError,
    OutcomeData,
    TreatmentData,
)

# Skip all tests if Bayesian dependencies are not available
pytest_bayesian = pytest.importorskip("causal_inference.estimators.bayesian")
BayesianCausalEffect = pytest_bayesian.BayesianCausalEffect
BayesianEstimator = pytest_bayesian.BayesianEstimator


class TestBayesianCausalEffect:
    """Test the BayesianCausalEffect data class."""

    def test_creation_with_basic_params(self):
        """Test creating BayesianCausalEffect with basic parameters."""
        effect = BayesianCausalEffect(
            ate=2.5,
            ate_credible_lower=1.0,
            ate_credible_upper=4.0,
            credible_interval_level=0.95,
            method="Bayesian Linear Model",
            n_observations=1000,
        )

        assert effect.ate == 2.5
        assert effect.ate_credible_lower == 1.0
        assert effect.ate_credible_upper == 4.0
        assert effect.credible_interval_level == 0.95
        assert effect.method == "Bayesian Linear Model"

        # Check that credible intervals are used as primary CI
        assert effect.ate_ci_lower == 1.0
        assert effect.ate_ci_upper == 4.0

    def test_creation_with_posterior_samples(self):
        """Test creating BayesianCausalEffect with posterior samples."""
        samples = np.random.normal(2.5, 0.5, 1000)

        effect = BayesianCausalEffect(
            ate=2.5,
            posterior_samples=samples,
            effective_sample_size=950.0,
            r_hat=1.01,
            method="Bayesian Linear Model",
            n_observations=500,
        )

        assert effect.posterior_samples is not None
        assert len(effect.posterior_samples) == 1000
        assert effect.effective_sample_size == 950.0
        assert effect.r_hat == 1.01


class TestBayesianEstimator:
    """Test the BayesianEstimator class."""

    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data for testing."""
        np.random.seed(42)
        n = 200

        # Generate covariates
        age = np.random.normal(50, 15, n)
        income = np.random.normal(50000, 20000, n)

        # Generate treatment (depends on covariates)
        treatment_logit = -2 + 0.02 * age + 0.00002 * income + np.random.normal(0, 1, n)
        treatment = (treatment_logit > 0).astype(int)

        # Generate outcome (true ATE = 5.0)
        outcome = (
            100
            + 5.0 * treatment
            + 0.5 * age
            + 0.0001 * income
            + np.random.normal(0, 10, n)
        )

        # Create data objects
        treatment_data = TreatmentData(
            values=pd.Series(treatment), treatment_type="binary"
        )
        outcome_data = OutcomeData(values=pd.Series(outcome), outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame({"age": age, "income": income}), names=["age", "income"]
        )

        return treatment_data, outcome_data, covariate_data

    @pytest.fixture
    def nhefs_subset_data(self):
        """Create a subset of NHEFS-like data for testing."""
        np.random.seed(123)
        n = 150

        # Generate covariates similar to NHEFS
        age = np.random.uniform(25, 75, n)
        sex = np.random.binomial(1, 0.5, n)  # 1 = female
        education = np.random.randint(1, 5, n)  # 1-5 education levels
        income = np.random.uniform(1, 5, n)  # Income quintiles

        # Generate treatment (qsmk: quit smoking)
        qsmk_logit = -1.5 + 0.02 * age + 0.3 * sex - 0.2 * education + 0.1 * income
        qsmk = np.random.binomial(1, 1 / (1 + np.exp(-qsmk_logit)))

        # Generate weight change outcome (true ATE ≈ 3.5 kg)
        wt82_71 = (
            -2
            + 3.5 * qsmk
            + 0.1 * age
            - 1.5 * sex
            + 0.5 * education
            + np.random.normal(0, 8, n)
        )

        treatment_data = TreatmentData(values=pd.Series(qsmk), treatment_type="binary")
        outcome_data = OutcomeData(values=pd.Series(wt82_71), outcome_type="continuous")
        covariate_data = CovariateData(
            values=pd.DataFrame(
                {"age": age, "sex": sex, "education": education, "income": income}
            ),
            names=["age", "sex", "education", "income"],
        )

        return treatment_data, outcome_data, covariate_data

    def test_estimator_initialization(self):
        """Test BayesianEstimator initialization with default parameters."""
        estimator = BayesianEstimator()

        assert estimator.prior_treatment_scale == 2.5
        assert estimator.mcmc_draws == 2000
        assert estimator.mcmc_chains == 4
        assert estimator.credible_level == 0.95

    def test_estimator_initialization_custom(self):
        """Test BayesianEstimator initialization with custom parameters."""
        estimator = BayesianEstimator(
            prior_treatment_scale=1.0,
            mcmc_draws=1000,
            mcmc_chains=2,
            credible_level=0.90,
            random_state=42,
            verbose=True,
        )

        assert estimator.prior_treatment_scale == 1.0
        assert estimator.mcmc_draws == 1000
        assert estimator.mcmc_chains == 2
        assert estimator.credible_level == 0.90
        assert estimator.random_state == 42
        assert estimator.verbose is True

    def test_data_validation_non_binary_treatment(self, simple_data):
        """Test that non-binary treatment raises error."""
        treatment_data, outcome_data, covariate_data = simple_data

        # Change to continuous treatment
        treatment_data.treatment_type = "continuous"

        estimator = BayesianEstimator()

        with pytest.raises(Exception, match="only supports binary treatments"):
            estimator.fit(treatment_data, outcome_data, covariate_data)

    def test_data_validation_non_continuous_outcome(self, simple_data):
        """Test that non-continuous outcome raises error."""
        treatment_data, outcome_data, covariate_data = simple_data

        # Change to binary outcome
        outcome_data.outcome_type = "binary"

        estimator = BayesianEstimator()

        with pytest.raises(Exception, match="only supports continuous outcomes"):
            estimator.fit(treatment_data, outcome_data, covariate_data)

    def test_data_validation_missing_values(self):
        """Test that missing values raise error."""
        # Data with missing values
        treatment = TreatmentData(
            values=pd.Series([1, 0, np.nan, 1]), treatment_type="binary"
        )
        outcome = OutcomeData(
            values=pd.Series([10.0, 8.0, 12.0, 15.0]), outcome_type="continuous"
        )

        estimator = BayesianEstimator()

        with pytest.raises(Exception, match="contains missing values"):
            estimator.fit(treatment, outcome)

    def test_fitting_simple_data(self, simple_data):
        """Test fitting the Bayesian estimator on simple synthetic data."""
        treatment_data, outcome_data, covariate_data = simple_data

        # Use smaller MCMC settings for faster testing
        estimator = BayesianEstimator(
            mcmc_draws=500, mcmc_tune=200, mcmc_chains=2, random_state=42, verbose=False
        )

        # Should not raise any exceptions
        estimator.fit(treatment_data, outcome_data, covariate_data)

        assert estimator.is_fitted
        assert estimator.model_ is not None
        assert estimator.trace_ is not None

    def test_estimation_simple_data(self, simple_data):
        """Test ATE estimation on simple synthetic data."""
        treatment_data, outcome_data, covariate_data = simple_data

        estimator = BayesianEstimator(
            mcmc_draws=500, mcmc_tune=200, mcmc_chains=2, random_state=42, verbose=False
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Check result type and properties
        assert isinstance(effect, BayesianCausalEffect)
        assert effect.ate is not None
        assert effect.ate_credible_lower is not None
        assert effect.ate_credible_upper is not None
        assert effect.posterior_samples is not None
        assert effect.effective_sample_size is not None
        assert effect.r_hat is not None

        # Check credible interval makes sense
        assert effect.ate_credible_lower < effect.ate < effect.ate_credible_upper

        # True ATE is 5.0, should be reasonably close
        assert abs(effect.ate - 5.0) < 2.0, (
            f"Estimated ATE {effect.ate} too far from true ATE 5.0"
        )

    def test_estimation_nhefs_subset(self, nhefs_subset_data):
        """Test ATE estimation on NHEFS-like data."""
        treatment_data, outcome_data, covariate_data = nhefs_subset_data

        estimator = BayesianEstimator(
            mcmc_draws=600,
            mcmc_tune=300,
            mcmc_chains=2,
            random_state=123,
            verbose=False,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        assert isinstance(effect, BayesianCausalEffect)
        assert effect.posterior_samples is not None
        assert len(effect.posterior_samples) == 600 * 2  # draws * chains

        # True ATE is 3.5, should be reasonably close
        assert abs(effect.ate - 3.5) < 2.0, (
            f"Estimated ATE {effect.ate} too far from true ATE 3.5"
        )

        # Check effective sample size is reasonable
        assert effect.effective_sample_size > 50, (
            f"ESS too low: {effect.effective_sample_size}"
        )

        # Check R-hat indicates convergence
        assert effect.r_hat < 1.2, f"R-hat too high: {effect.r_hat}"

    def test_estimation_without_covariates(self):
        """Test estimation without covariates."""
        np.random.seed(42)
        n = 100

        treatment = np.random.binomial(1, 0.5, n)
        outcome = 10 + 3.0 * treatment + np.random.normal(0, 2, n)

        treatment_data = TreatmentData(
            values=pd.Series(treatment), treatment_type="binary"
        )
        outcome_data = OutcomeData(values=pd.Series(outcome), outcome_type="continuous")

        estimator = BayesianEstimator(
            mcmc_draws=400, mcmc_tune=200, mcmc_chains=2, random_state=42, verbose=False
        )

        estimator.fit(treatment_data, outcome_data)
        effect = estimator.estimate_ate()

        assert isinstance(effect, BayesianCausalEffect)
        # True ATE is 3.0
        assert abs(effect.ate - 3.0) < 1.5, (
            f"Estimated ATE {effect.ate} too far from true ATE 3.0"
        )

    def test_summary_statistics(self, simple_data):
        """Test summary statistics method."""
        treatment_data, outcome_data, covariate_data = simple_data

        estimator = BayesianEstimator(
            mcmc_draws=400, mcmc_tune=200, mcmc_chains=2, random_state=42, verbose=False
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        summary = estimator.parameter_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "treatment_effect" in summary.index
        assert "mean" in summary.columns
        assert "hdi_3%" in summary.columns
        assert "hdi_97%" in summary.columns

    def test_plotting_methods(self, simple_data):
        """Test plotting methods don't raise errors."""
        treatment_data, outcome_data, covariate_data = simple_data

        estimator = BayesianEstimator(
            mcmc_draws=300, mcmc_tune=150, mcmc_chains=2, random_state=42, verbose=False
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)

        # These should not raise exceptions
        # (We can't easily test the actual plots in unit tests)
        try:
            estimator.plot_posterior()
            estimator.plot_trace()
            estimator.posterior_predictive_check(n_samples=50)
        except Exception as e:
            pytest.fail(f"Plotting methods should not raise exceptions: {e}")

    def test_error_before_fitting(self):
        """Test that methods raise errors when called before fitting."""
        estimator = BayesianEstimator()

        with pytest.raises(Exception, match="must be fitted"):
            estimator.estimate_ate()

        with pytest.raises(Exception, match="must be fitted"):
            estimator.parameter_summary()

        with pytest.raises(Exception, match="must be fitted"):
            estimator.plot_posterior()

    def test_different_priors(self, simple_data):
        """Test estimation with different prior specifications."""
        treatment_data, outcome_data, covariate_data = simple_data

        # Test with informative priors
        estimator = BayesianEstimator(
            prior_treatment_scale=0.5,  # More informative prior
            prior_covariate_scale=1.0,
            mcmc_draws=400,
            mcmc_tune=200,
            mcmc_chains=2,
            random_state=42,
            verbose=False,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        assert isinstance(effect, BayesianCausalEffect)
        assert effect.prior_specification["treatment_scale"] == 0.5
        assert effect.prior_specification["covariate_scale"] == 1.0

    def test_credible_interval_levels(self, simple_data):
        """Test different credible interval levels."""
        treatment_data, outcome_data, covariate_data = simple_data

        # Test with 90% credible intervals
        estimator = BayesianEstimator(
            credible_level=0.90,
            mcmc_draws=400,
            mcmc_tune=200,
            mcmc_chains=2,
            random_state=42,
            verbose=False,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        assert effect.credible_interval_level == 0.90
        assert effect.confidence_level == 0.90

        # 90% interval should be narrower than 95%
        interval_width = effect.ate_credible_upper - effect.ate_credible_lower
        assert interval_width > 0
        assert interval_width < 10  # Reasonable width for our test data

    def test_convergence_diagnostics(self, simple_data):
        """Test convergence diagnostics and warnings."""
        treatment_data, outcome_data, covariate_data = simple_data

        # Use very few draws to potentially trigger convergence warnings
        estimator = BayesianEstimator(
            mcmc_draws=50,  # Very small for fast testing
            mcmc_tune=25,
            mcmc_chains=2,
            random_state=42,
            verbose=False,
        )

        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        assert effect.mcmc_diagnostics is not None
        assert "effective_sample_size" in effect.mcmc_diagnostics
        assert "r_hat" in effect.mcmc_diagnostics
        assert effect.mcmc_diagnostics["draws"] == 50
        assert effect.mcmc_diagnostics["chains"] == 2

    def test_reproducibility(self, simple_data):
        """Test that results are reproducible with same random seed."""
        treatment_data, outcome_data, covariate_data = simple_data

        # First run
        estimator1 = BayesianEstimator(
            mcmc_draws=300, mcmc_tune=150, mcmc_chains=2, random_state=42, verbose=False
        )
        estimator1.fit(treatment_data, outcome_data, covariate_data)
        effect1 = estimator1.estimate_ate()

        # Second run with same seed
        estimator2 = BayesianEstimator(
            mcmc_draws=300, mcmc_tune=150, mcmc_chains=2, random_state=42, verbose=False
        )
        estimator2.fit(treatment_data, outcome_data, covariate_data)
        effect2 = estimator2.estimate_ate()

        # Results should be very close (allowing for small numerical differences)
        assert abs(effect1.ate - effect2.ate) < 0.01, "Results should be reproducible"

    def test_parameter_validation(self):
        """Test parameter validation in __init__."""
        # Test negative prior scales
        with pytest.raises(ValueError, match="prior_treatment_scale must be positive"):
            BayesianEstimator(prior_treatment_scale=-1.0)

        with pytest.raises(ValueError, match="prior_intercept_scale must be positive"):
            BayesianEstimator(prior_intercept_scale=0.0)

        # Test invalid MCMC parameters
        with pytest.raises(ValueError, match="mcmc_draws must be positive"):
            BayesianEstimator(mcmc_draws=0)

        with pytest.raises(ValueError, match="mcmc_chains must be positive"):
            BayesianEstimator(mcmc_chains=-1)

        # Test invalid credible level
        with pytest.raises(ValueError, match="credible_level must be between 0 and 1"):
            BayesianEstimator(credible_level=1.5)

        with pytest.raises(ValueError, match="credible_level must be between 0 and 1"):
            BayesianEstimator(credible_level=0.0)

    def test_insufficient_sample_size(self):
        """Test error with insufficient sample size."""
        # Test base class validation (< 10 observations)
        very_small_treatment = TreatmentData(
            values=pd.Series([1, 0, 1, 0, 1]), treatment_type="binary"
        )
        very_small_outcome = OutcomeData(
            values=pd.Series([10.0, 8.0, 12.0, 9.0, 11.0]), outcome_type="continuous"
        )

        estimator = BayesianEstimator()

        with pytest.raises(DataValidationError, match="Minimum sample size of 10"):
            estimator.fit(very_small_treatment, very_small_outcome)

        # Test Bayesian-specific validation (< 50 observations but > 10)
        np.random.seed(42)
        medium_treatment = TreatmentData(
            values=pd.Series(np.random.binomial(1, 0.5, 30)), treatment_type="binary"
        )
        medium_outcome = OutcomeData(
            values=pd.Series(np.random.normal(10, 2, 30)), outcome_type="continuous"
        )

        with pytest.raises(EstimationError, match="Insufficient sample size.*50 observations"):
            estimator.fit(medium_treatment, medium_outcome)

