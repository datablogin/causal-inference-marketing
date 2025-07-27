"""Smoke tests for all main functionality.

Smoke tests are simple tests that verify basic functionality works
without errors. They don't test correctness in detail but catch
major regressions and import issues.
"""

import numpy as np
import pandas as pd

from causal_inference.core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    OutcomeData,
    TreatmentData,
)


class TestCoreModuleImports:
    """Test that core modules can be imported without errors."""

    def test_import_base_classes(self):
        """Test importing base classes."""
        from causal_inference.core.base import (
            CausalEffect,
            CovariateData,
            OutcomeData,
            TreatmentData,
        )

        # Should not raise any import errors
        assert BaseEstimator is not None
        assert CausalEffect is not None
        assert TreatmentData is not None
        assert OutcomeData is not None
        assert CovariateData is not None

    def test_import_estimators(self):
        """Test importing all estimators."""
        from causal_inference.estimators.aipw import AIPWEstimator
        from causal_inference.estimators.g_computation import GComputationEstimator
        from causal_inference.estimators.ipw import IPWEstimator

        assert GComputationEstimator is not None
        assert IPWEstimator is not None
        assert AIPWEstimator is not None

    def test_import_data_utilities(self):
        """Test importing data utilities."""
        from causal_inference.data.missing_data import MissingDataHandler
        from causal_inference.data.nhefs import NHEFSDataLoader
        from causal_inference.data.synthetic import SyntheticDataGenerator
        from causal_inference.data.validation import CausalDataValidator

        assert SyntheticDataGenerator is not None
        assert NHEFSDataLoader is not None
        assert CausalDataValidator is not None
        assert MissingDataHandler is not None

    def test_import_diagnostics(self):
        """Test importing diagnostic modules."""
        from causal_inference.diagnostics import (
            assumptions,
            balance,
            falsification,
            overlap,
            reporting,
            sensitivity,
            specification,
            visualization,
        )

        # Should not raise import errors
        assert assumptions is not None
        assert balance is not None
        assert overlap is not None
        assert sensitivity is not None
        assert specification is not None
        assert reporting is not None
        assert visualization is not None
        assert falsification is not None


class TestDataModelCreation:
    """Test that data models can be created without errors."""

    def test_treatment_data_creation(self):
        """Test creating TreatmentData objects."""
        # Binary treatment
        binary_treatment = TreatmentData(
            values=np.array([0, 1, 1, 0]), treatment_type="binary"
        )
        assert binary_treatment.treatment_type == "binary"
        assert len(binary_treatment.values) == 4

        # Categorical treatment
        categorical_treatment = TreatmentData(
            values=np.array([0, 1, 2, 1]), treatment_type="categorical"
        )
        assert categorical_treatment.treatment_type == "categorical"

        # Continuous treatment
        continuous_treatment = TreatmentData(
            values=np.array([0.1, 1.5, 2.3, 0.8]), treatment_type="continuous"
        )
        assert continuous_treatment.treatment_type == "continuous"

    def test_outcome_data_creation(self):
        """Test creating OutcomeData objects."""
        # Continuous outcome
        continuous_outcome = OutcomeData(
            values=np.array([1.2, 3.4, 2.1, 4.5]), outcome_type="continuous"
        )
        assert continuous_outcome.outcome_type == "continuous"

        # Binary outcome
        binary_outcome = OutcomeData(
            values=np.array([0, 1, 1, 0]), outcome_type="binary"
        )
        assert binary_outcome.outcome_type == "binary"

    def test_covariate_data_creation(self):
        """Test creating CovariateData objects."""
        # DataFrame input
        df = pd.DataFrame(
            {"age": [25, 30, 35, 40], "income": [50000, 60000, 70000, 80000]}
        )
        covariate_data = CovariateData(values=df)
        assert covariate_data.values.shape == (4, 2)

        # Array input
        array = np.array([[1, 2], [3, 4], [5, 6]])
        covariate_data_array = CovariateData(values=array)
        assert covariate_data_array.values.shape == (3, 2)

    def test_causal_effect_creation(self):
        """Test creating CausalEffect objects."""
        effect = CausalEffect(
            ate=2.5, ate_ci_lower=1.2, ate_ci_upper=3.8, confidence_level=0.95
        )
        assert effect.ate == 2.5
        assert effect.ate_ci_lower == 1.2
        assert effect.ate_ci_upper == 3.8
        assert effect.confidence_level == 0.95
        assert effect.is_significant


class TestEstimatorBasicFunctionality:
    """Test basic functionality of all estimators."""

    def test_g_computation_smoke(self, simple_binary_data):
        """Smoke test for G-computation estimator."""
        from causal_inference.estimators.g_computation import GComputationEstimator

        estimator = GComputationEstimator()

        # Should be able to create, fit, and estimate
        estimator.fit(
            simple_binary_data["treatment"],
            simple_binary_data["outcome"],
            simple_binary_data["covariates"],
        )

        effect = estimator.estimate_ate()
        assert isinstance(effect, CausalEffect)
        assert np.isfinite(effect.ate)

        # Should be able to get summary
        summary = estimator.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_ipw_smoke(self, simple_binary_data):
        """Smoke test for IPW estimator."""
        from causal_inference.estimators.ipw import IPWEstimator

        estimator = IPWEstimator()

        estimator.fit(
            simple_binary_data["treatment"],
            simple_binary_data["outcome"],
            simple_binary_data["covariates"],
        )

        effect = estimator.estimate_ate()
        assert isinstance(effect, CausalEffect)
        assert np.isfinite(effect.ate)

        summary = estimator.summary()
        assert isinstance(summary, str)

    def test_aipw_smoke(self, simple_binary_data):
        """Smoke test for AIPW estimator."""
        from causal_inference.estimators.aipw import AIPWEstimator

        estimator = AIPWEstimator()

        estimator.fit(
            simple_binary_data["treatment"],
            simple_binary_data["outcome"],
            simple_binary_data["covariates"],
        )

        effect = estimator.estimate_ate()
        assert isinstance(effect, CausalEffect)
        assert np.isfinite(effect.ate)

        summary = estimator.summary()
        assert isinstance(summary, str)

    def test_estimator_with_categorical_treatment(self, simple_categorical_data):
        """Test estimators with categorical treatment."""
        from causal_inference.estimators.g_computation import GComputationEstimator

        estimator = GComputationEstimator()

        # Should handle categorical treatment
        estimator.fit(
            simple_categorical_data["treatment"],
            simple_categorical_data["outcome"],
            simple_categorical_data["covariates"],
        )

        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)

    def test_estimator_with_continuous_treatment(
        self, simple_continuous_treatment_data
    ):
        """Test estimators with continuous treatment."""
        from causal_inference.estimators.g_computation import GComputationEstimator

        estimator = GComputationEstimator()

        # Should handle continuous treatment
        estimator.fit(
            simple_continuous_treatment_data["treatment"],
            simple_continuous_treatment_data["outcome"],
            simple_continuous_treatment_data["covariates"],
        )

        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)

    def test_estimator_with_binary_outcome(self, binary_outcome_data):
        """Test estimators with binary outcome."""
        from causal_inference.estimators.g_computation import GComputationEstimator

        estimator = GComputationEstimator()

        # Should handle binary outcome
        estimator.fit(
            binary_outcome_data["treatment"],
            binary_outcome_data["outcome"],
            binary_outcome_data["covariates"],
        )

        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)


class TestDataUtilitiesSmoke:
    """Smoke tests for data utilities."""

    def test_synthetic_data_generator_smoke(self):
        """Smoke test for synthetic data generator."""
        from causal_inference.data.synthetic import SyntheticDataGenerator

        generator = SyntheticDataGenerator(random_state=42)

        # Binary treatment
        treatment, outcome, covariates = generator.generate_linear_binary_treatment(
            n_samples=100, n_confounders=3, treatment_effect=2.0
        )

        assert len(treatment.values) == 100
        assert len(outcome.values) == 100
        assert covariates.values.shape == (100, 3)

        # Continuous treatment
        treatment, outcome, covariates = (
            generator.generate_nonlinear_continuous_treatment(
                n_samples=100, n_confounders=2
            )
        )

        assert len(treatment.values) == 100
        assert treatment.treatment_type == "continuous"

    def test_data_validation_smoke(self, simple_binary_data):
        """Smoke test for data validation."""
        from causal_inference.data.validation import validate_causal_data

        # Should not raise errors for valid data
        result = validate_causal_data(
            simple_binary_data["treatment"],
            simple_binary_data["outcome"],
            simple_binary_data["covariates"],
        )

        assert result is not None

    def test_missing_data_handler_smoke(self):
        """Smoke test for missing data handler."""
        from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
        from causal_inference.data.missing_data import handle_missing_data

        # Create data with missing values
        treatment_values = np.array([0, 1, 0, 1, 1])
        outcome_values = np.array([1, 2, np.nan, 4, 5])
        covariate_df = pd.DataFrame(
            {"X1": [1, 2, np.nan, 4, 5], "X2": [10, np.nan, 30, 40, 50]}
        )

        treatment_data = TreatmentData(values=treatment_values, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome_values, outcome_type="continuous")
        covariate_data = CovariateData(values=covariate_df)

        # Should handle missing data without errors
        clean_treatment, clean_outcome, clean_covariates = handle_missing_data(
            treatment_data, outcome_data, covariate_data, strategy="mean"
        )
        assert clean_treatment is not None
        assert clean_outcome is not None
        assert clean_covariates is not None
        # Check that missing values are handled
        assert not pd.isna(clean_outcome.values).any()
        assert not clean_covariates.values.isnull().any().any()

    def test_nhefs_data_loader_smoke(self):
        """Smoke test for NHEFS data loader."""
        from causal_inference.data.nhefs import NHEFSDataLoader

        # Should be able to create loader (even if file doesn't exist)
        try:
            loader = NHEFSDataLoader()
            assert loader is not None
        except FileNotFoundError:
            # Expected if NHEFS data file is not available
            pass


class TestDiagnosticsSmoke:
    """Smoke tests for diagnostic functions."""

    def test_balance_diagnostics_smoke(self, simple_binary_data):
        """Smoke test for balance diagnostics."""
        from causal_inference.diagnostics.balance import check_covariate_balance

        result = check_covariate_balance(
            simple_binary_data["treatment"], simple_binary_data["covariates"]
        )

        assert result is not None
        assert "balance_table" in result

    def test_overlap_diagnostics_smoke(self, simple_binary_data):
        """Smoke test for overlap diagnostics."""
        from causal_inference.diagnostics.overlap import assess_positivity

        result = assess_positivity(
            simple_binary_data["treatment"], simple_binary_data["covariates"]
        )

        assert result is not None
        assert "common_support" in result

    def test_sensitivity_analysis_smoke(self):
        """Smoke test for sensitivity analysis."""
        from causal_inference.diagnostics.sensitivity import calculate_evalue

        # Should handle basic E-value calculation
        evalue = calculate_evalue(effect_size=2.0)
        assert np.isfinite(evalue)
        assert evalue > 0

    def test_specification_tests_smoke(self, simple_binary_data):
        """Smoke test for specification tests."""
        from causal_inference.diagnostics.specification import test_functional_form

        result = test_functional_form(
            simple_binary_data["covariates"], simple_binary_data["outcome"]
        )

        assert result is not None

    def test_diagnostic_reporting_smoke(self, simple_binary_data):
        """Smoke test for diagnostic reporting."""
        from causal_inference.diagnostics.reporting import generate_diagnostic_report

        report = generate_diagnostic_report(
            treatment=simple_binary_data["treatment"],
            outcome=simple_binary_data["outcome"],
            covariates=simple_binary_data["covariates"],
        )

        assert isinstance(report, str)
        assert len(report) > 0


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_analysis_workflow(self):
        """Test a complete analysis from data generation to results."""
        # Generate synthetic data
        from causal_inference.data.synthetic import SyntheticDataGenerator

        generator = SyntheticDataGenerator(random_state=42)
        treatment, outcome, covariates = generator.generate_linear_binary_treatment(
            n_samples=200, n_confounders=3, treatment_effect=1.5
        )

        # Validate data
        from causal_inference.data.validation import validate_causal_data

        validate_causal_data(treatment, outcome, covariates)

        # Run analysis with multiple estimators
        from causal_inference.estimators.aipw import AIPWEstimator
        from causal_inference.estimators.g_computation import GComputationEstimator
        from causal_inference.estimators.ipw import IPWEstimator

        estimators = [GComputationEstimator(), IPWEstimator(), AIPWEstimator()]

        results = {}
        for estimator in estimators:
            estimator.fit(treatment, outcome, covariates)
            effect = estimator.estimate_ate()
            results[estimator.__class__.__name__] = effect.ate

        # All should produce finite results
        for name, ate in results.items():
            assert np.isfinite(ate), f"{name} produced non-finite ATE: {ate}"

        # Run diagnostics
        from causal_inference.diagnostics.balance import check_covariate_balance
        from causal_inference.diagnostics.overlap import assess_positivity

        balance_result = check_covariate_balance(treatment, covariates)
        overlap_result = assess_positivity(treatment, covariates)

        assert balance_result is not None
        assert overlap_result is not None

    def test_workflow_with_missing_data(self):
        """Test workflow when data has missing values."""
        # Create data with missing values
        np.random.seed(42)
        n_samples = 100

        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)

        # Introduce missing values
        X1[10:15] = np.nan
        X2[20:25] = np.nan

        covariates_df = pd.DataFrame({"X1": X1, "X2": X2})

        treatment = np.random.binomial(1, 0.5, n_samples)
        outcome = (
            1
            + 0.5 * np.nanmean([X1, X2], axis=0)
            + 2 * treatment
            + np.random.normal(0, 1, n_samples)
        )

        # Handle missing data
        from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
        from causal_inference.data.missing_data import handle_missing_data

        # Create dummy data objects for handle_missing_data function
        dummy_treatment = TreatmentData(values=treatment, treatment_type="binary")
        dummy_outcome = OutcomeData(values=outcome, outcome_type="continuous")
        dummy_covariates = CovariateData(values=covariates_df)

        clean_treatment, clean_outcome, clean_covariates_data = handle_missing_data(
            dummy_treatment, dummy_outcome, dummy_covariates, strategy="mean"
        )
        clean_covariates = clean_covariates_data.values if clean_covariates_data else pd.DataFrame()

        # Continue with analysis
        from causal_inference.estimators.g_computation import GComputationEstimator

        treatment_data = TreatmentData(values=treatment, treatment_type="binary")
        outcome_data = OutcomeData(values=outcome, outcome_type="continuous")
        covariate_data = CovariateData(values=clean_covariates)

        estimator = GComputationEstimator()
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        assert np.isfinite(effect.ate)

    def test_large_dataset_workflow(self, performance_benchmark_data):
        """Test workflow scales to larger datasets."""
        from causal_inference.diagnostics.balance import check_covariate_balance
        from causal_inference.estimators.g_computation import GComputationEstimator

        # Should handle larger datasets without issues
        estimator = GComputationEstimator(bootstrap_samples=10)  # Reduced for speed
        estimator.fit(
            performance_benchmark_data["treatment"],
            performance_benchmark_data["outcome"],
            performance_benchmark_data["covariates"],
        )

        effect = estimator.estimate_ate()
        assert np.isfinite(effect.ate)

        # Diagnostics should also work
        balance_result = check_covariate_balance(
            performance_benchmark_data["treatment"],
            performance_benchmark_data["covariates"],
        )
        assert balance_result is not None
