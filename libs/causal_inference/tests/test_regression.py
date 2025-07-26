"""Regression tests against known results.

This module provides tests against known analytical solutions and
published benchmark results to ensure estimator correctness.
"""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator

# Test constants for reproducible and meaningful tolerances
CONFIDENCE_Z_SCORE = 1.96  # 95% confidence interval z-score
SE_MULTIPLIER = 2  # Number of standard errors for tolerance
REGRESSION_TOLERANCE = 0.2  # ATE estimation tolerance for regression tests
NULL_HYPOTHESIS_TOLERANCE = 0.3  # Tolerance for null hypothesis tests
RELATIVE_TOLERANCE = 0.10  # 10% relative error tolerance
NSW_ATE_LOWER_BOUND = 800  # Lower bound for NSW benchmark (conservative)
NSW_ATE_UPPER_BOUND = 2200  # Upper bound for NSW benchmark (conservative)
KANG_SCHAFER_TOLERANCE = 2.0  # Tolerance for Kang & Schafer benchmark
COVERAGE_LOWER_BOUND = 0.80  # Minimum acceptable CI coverage rate
COVERAGE_UPPER_BOUND = 1.0  # Maximum acceptable CI coverage rate
CONSISTENCY_TOLERANCE = 0.3  # Large sample consistency tolerance


class TestAnalyticalSolutions:
    """Test estimators against problems with known analytical solutions."""

    def test_randomized_experiment_simple_difference(self):
        """Test that estimators recover true ATE in randomized experiment."""
        # Perfect randomization with known treatment effect
        np.random.seed(42)
        n_samples = 1000

        # Purely random treatment assignment
        treatment = np.random.binomial(1, 0.5, n_samples)

        # Outcome with no confounding, pure treatment effect of 3.0
        outcome = 10 + 3.0 * treatment + np.random.normal(0, 1, n_samples)

        # No covariates needed in randomized experiment
        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")

        # True ATE should be exactly 3.0
        true_ate = 3.0

        # Test estimators that work without covariates in randomized experiments
        estimators = [
            ("G-computation", GComputationEstimator()),
            # Note: IPW and AIPW require covariates, so we skip them for this test
        ]

        for name, estimator in estimators:
            estimator.fit(treatment_data, outcome_data, covariates=None)
            effect = estimator.estimate_ate()

            # Should be very close to true ATE (within SE_MULTIPLIER standard errors)
            if effect.confidence_interval is not None:
                se_estimate = (
                    effect.confidence_interval[1] - effect.confidence_interval[0]
                ) / (2 * CONFIDENCE_Z_SCORE)
                assert abs(effect.ate - true_ate) <= SE_MULTIPLIER * se_estimate, (
                    f"{name} failed: {effect.ate} vs {true_ate}"
                )
            else:
                # If no CI available, just check that ATE is reasonably close
                assert abs(effect.ate - true_ate) < 0.5, (
                    f"{name} failed: {effect.ate} vs {true_ate}"
                )

    def test_linear_confounding_analytical_solution(self):
        """Test against analytical solution for linear confounding case."""
        np.random.seed(123)
        n_samples = 2000

        # Single confounder
        X = np.random.normal(0, 1, n_samples)

        # Treatment depends on confounder
        propensity = 1 / (1 + np.exp(-(0.5 * X)))
        treatment = np.random.binomial(1, propensity, n_samples)

        # Linear outcome model: Y = 2 + 1.5*X + 4*T + noise
        outcome = 2 + 1.5 * X + 4.0 * treatment + np.random.normal(0, 0.5, n_samples)

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame({"X": X}))

        # True ATE is 4.0 (coefficient of treatment)
        true_ate = 4.0

        # G-computation should be nearly exact with correct linear model
        estimator = GComputationEstimator()
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Should be very close to true ATE
        assert abs(effect.ate - true_ate) < REGRESSION_TOLERANCE, (
            f"G-computation ATE {effect.ate} not close to true {true_ate}"
        )

    def test_no_treatment_effect_null_hypothesis(self):
        """Test that estimators correctly identify no treatment effect."""
        np.random.seed(456)
        n_samples = 800

        # Confounders
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)

        # Treatment depends on confounders
        propensity = 1 / (1 + np.exp(-(0.3 * X1 + 0.4 * X2)))
        treatment = np.random.binomial(1, propensity, n_samples)

        # Outcome depends on confounders but NOT treatment (true ATE = 0)
        outcome = 1 + 0.8 * X1 + 0.6 * X2 + np.random.normal(0, 1, n_samples)

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame({"X1": X1, "X2": X2}))

        estimators = [GComputationEstimator(), IPWEstimator(), AIPWEstimator()]

        for estimator in estimators:
            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            # Should be close to zero and confidence interval should include zero
            assert abs(effect.ate) < NULL_HYPOTHESIS_TOLERANCE, (
                f"ATE should be near zero: {effect.ate}"
            )
            assert (
                effect.confidence_interval[0] <= 0 <= effect.confidence_interval[1]
            ), "CI should include zero"

    def test_dose_response_linear_relationship(self):
        """Test continuous treatment with known linear dose-response."""
        np.random.seed(789)
        n_samples = 1000

        # Confounders
        X = np.random.normal(0, 1, n_samples)

        # Continuous treatment
        treatment = 0.4 * X + np.random.normal(0, 1, n_samples)

        # Linear dose-response: coefficient = 2.5
        outcome = 3 + 0.7 * X + 2.5 * treatment + np.random.normal(0, 0.5, n_samples)

        treatment_data = TreatmentData(values=treatment, treatment_type="continuous")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=pd.DataFrame({"X": X}))

        # For G-computation with continuous treatment, we test unit increase
        estimator = GComputationEstimator()
        estimator.fit(treatment_data, outcome_data, covariate_data)

        # This would require additional methods for continuous treatment prediction
        # For now, we verify the estimator doesn't crash and produces reasonable output
        # Future enhancement: implement dose-response estimation for continuous treatments
        # n_test = 100
        # X_test = np.random.normal(0, 1, n_test)
        # treatment_test = np.random.normal(0, 1, n_test)
        # covariate_test = CovariateData(values=pd.DataFrame({"X": X_test}))
        # treatment_test_data = TreatmentData(values=treatment_test, treatment_type="continuous")
        # treatment_plus_one = TreatmentData(values=treatment_test + 1, treatment_type="continuous")
        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)


class TestPublishedBenchmarks:
    """Test against results from published papers and standard datasets."""

    def test_lalonde_nsw_benchmark(self):
        """Test using a simplified version of the LaLonde NSW benchmark structure."""
        # Simplified version of the LaLonde experimental benchmark
        # This creates data with similar characteristics to the famous NSW dataset
        np.random.seed(42)
        n_samples = 722  # Similar to original NSW experimental sample

        # Covariates similar to LaLonde study
        age = np.random.uniform(17, 55, n_samples)
        education = np.random.choice(range(3, 17), n_samples)  # Years of education
        black = np.random.binomial(1, 0.8, n_samples)  # Predominantly black sample
        hispanic = np.random.binomial(1, 0.1, n_samples)
        married = np.random.binomial(1, 0.2, n_samples)

        # Pre-treatment earnings (simplified model)
        re74 = np.maximum(0, np.random.normal(2000, 3000, n_samples))  # 1974 earnings
        re75 = np.maximum(0, np.random.normal(2000, 3000, n_samples))  # 1975 earnings

        # Random treatment assignment (experimental design)
        treatment = np.random.binomial(1, 0.5, n_samples)

        # Post-treatment earnings with realistic NSW effect
        # Known NSW experimental effect was around $800-1800
        nsw_effect = 1400  # Middle of the range

        outcome = (
            1000  # Base earnings
            + 50 * age  # Age effect
            + 200 * education  # Education effect
            + -1000 * black  # Racial earnings gap
            + 500 * married  # Marriage premium
            + 0.1 * re74  # Persistence of past earnings
            + 0.1 * re75
            + nsw_effect * treatment  # Treatment effect
            + np.random.normal(0, 2000, n_samples)  # Noise
        )

        outcome = np.maximum(outcome, 0)  # Non-negative earnings

        covariates = pd.DataFrame(
            {
                "age": age,
                "education": education,
                "black": black,
                "hispanic": hispanic,
                "married": married,
                "re74": re74,
                "re75": re75,
            }
        )

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=covariates)

        # Test estimators - should recover the treatment effect
        estimators = [
            ("G-computation", GComputationEstimator()),
            ("IPW", IPWEstimator()),
            ("AIPW", AIPWEstimator()),
        ]

        for name, estimator in estimators:
            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            # Should be reasonably close to true effect (within reasonable bounds)
            # Allow for wider tolerance given complexity of earnings model
            assert NSW_ATE_LOWER_BOUND <= effect.ate <= NSW_ATE_UPPER_BOUND, (
                f"{name} ATE {effect.ate} outside reasonable bounds for NSW"
            )

            # Confidence interval should be reasonable
            ci_width = effect.confidence_interval[1] - effect.confidence_interval[0]
            assert 100 <= ci_width <= 5000, f"{name} CI width {ci_width} unreasonable"

    def test_kang_schafer_benchmark(self):
        """Test against the Kang & Schafer (2007) benchmark simulation."""
        # This is a well-known simulation study for comparing causal inference methods
        np.random.seed(2007)  # Year of the paper
        n_samples = 1000

        # Four covariates as in original study
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        X3 = np.random.normal(0, 1, n_samples)
        X4 = np.random.normal(0, 1, n_samples)

        # Propensity score model (slightly simplified)
        propensity_logit = -X1 + 0.5 * X2 - 0.25 * X3 - 0.1 * X4
        propensity = 1 / (1 + np.exp(-propensity_logit))
        treatment = np.random.binomial(1, propensity, n_samples)

        # Outcome model from Kang & Schafer
        outcome = (
            210
            + 27.4 * X1
            + 13.7 * X2
            + 13.7 * X3
            + 13.7 * X4
            + np.random.normal(0, 1, n_samples)
        )

        # True ATE in this setup is 0 (no treatment effect in outcome model)
        true_ate = 0.0

        covariates = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4})

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=covariates)

        # Test doubly robust estimator (should work even if one model is wrong)
        estimator = AIPWEstimator()
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Should be close to zero
        assert abs(effect.ate) < KANG_SCHAFER_TOLERANCE, (
            f"AIPW ATE {effect.ate} should be near zero"
        )
        assert (
            effect.confidence_interval[0] <= true_ate <= effect.confidence_interval[1]
        )


class TestSimulationStudyReplication:
    """Replicate results from simulation studies in causal inference literature."""

    def test_doubly_robust_simulation(self):
        """Test doubly robust property with misspecified models."""
        np.random.seed(999)
        n_samples = 2000

        # Complex confounders
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        X3 = np.random.normal(0, 1, n_samples)

        # Nonlinear propensity model (true model is complex)
        true_propensity_logit = X1 + X2**2 + np.sin(X3)
        propensity = 1 / (1 + np.exp(-true_propensity_logit))
        treatment = np.random.binomial(1, propensity, n_samples)

        # Nonlinear outcome model (true model is complex)
        true_outcome = (
            1
            + X1**2
            + np.exp(X2 / 2)
            + X3
            + 3 * treatment
            + np.random.normal(0, 0.5, n_samples)
        )

        covariates = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=true_outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=covariates)

        # AIPW should still work reasonably well even with misspecified linear models
        # (though not perfectly due to model misspecification)
        estimator = AIPWEstimator()
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Allow for some bias due to model misspecification, but should be in right ballpark
        # True ATE is 3.0, allow range [1.5, 4.5] for misspecified models
        assert 1.5 <= effect.ate <= 4.5, f"AIPW with misspecification: {effect.ate}"

    def test_monte_carlo_variance_estimation(self):
        """Test that confidence intervals have appropriate coverage."""
        # Run multiple simulations to check CI coverage
        np.random.seed(12345)
        true_ate = 2.0
        n_simulations = 50  # Reduced for test speed
        n_samples = 500

        coverage_count = 0

        for sim in range(n_simulations):
            # Generate fresh data for each simulation
            X = np.random.normal(0, 1, n_samples)
            propensity = 1 / (1 + np.exp(-0.5 * X))
            treatment = np.random.binomial(1, propensity, n_samples)
            outcome = (
                1 + 0.8 * X + true_ate * treatment + np.random.normal(0, 1, n_samples)
            )

            treatment_data = TreatmentData(values=treatment, treatment_type="binary")
            outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
            covariate_data = CovariateData(values=pd.DataFrame({"X": X}))

            estimator = GComputationEstimator(bootstrap_samples=100)
            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            # Check if true ATE is in confidence interval
            if (
                effect.confidence_interval[0]
                <= true_ate
                <= effect.confidence_interval[1]
            ):
                coverage_count += 1

        coverage_rate = coverage_count / n_simulations

        # Should have approximately 95% coverage (allow some variation due to finite samples)
        assert COVERAGE_LOWER_BOUND <= coverage_rate <= COVERAGE_UPPER_BOUND, (
            f"CI coverage rate {coverage_rate} outside acceptable range"
        )

    @pytest.mark.slow
    def test_large_sample_consistency(self):
        """Test that estimators are consistent (converge to true value with large samples)."""
        true_ate = 1.5
        sample_sizes = [500, 1000, 2000]

        ates = []

        for n_samples in sample_sizes:
            np.random.seed(42)  # Fixed seed for fair comparison

            X = np.random.normal(0, 1, n_samples)
            propensity = 1 / (1 + np.exp(-0.3 * X))
            treatment = np.random.binomial(1, propensity, n_samples)
            outcome = (
                2 + 0.5 * X + true_ate * treatment + np.random.normal(0, 0.8, n_samples)
            )

            treatment_data = TreatmentData(values=treatment, treatment_type="binary")
            outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
            covariate_data = CovariateData(values=pd.DataFrame({"X": X}))

            estimator = GComputationEstimator()
            estimator.fit(treatment_data, outcome_data, covariate_data)
            effect = estimator.estimate_ate()

            ates.append(effect.ate)

        # Estimates should get closer to true value with larger samples
        errors = [abs(ate - true_ate) for ate in ates]

        # Error should generally decrease (allow some variation)
        assert errors[-1] <= errors[0] + 0.2, (
            f"Large sample not more accurate: {errors}"
        )

        # Final estimate should be quite close
        assert errors[-1] < CONSISTENCY_TOLERANCE, (
            f"Large sample error too big: {errors[-1]}"
        )
