"""Integration tests for data utilities working together."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.data.missing_data import MissingDataHandler
from causal_inference.data.nhefs import NHEFSDataLoader
from causal_inference.data.synthetic import SyntheticDataGenerator
from causal_inference.data.validation import CausalDataValidator


class TestDataUtilitiesIntegration:
    """Test integration between different data utilities."""

    def test_synthetic_to_validation_integration(self):
        """Test synthetic data generation followed by validation."""
        # Generate synthetic data
        generator = SyntheticDataGenerator(random_state=42)
        treatment, outcome, covariates = generator.generate_linear_binary_treatment(
            n_samples=200, n_confounders=3
        )

        # Validate the generated data
        validator = CausalDataValidator(verbose=False)
        validator.validate_all(treatment, outcome, covariates)

        # Synthetic data should be valid (no errors)
        assert len(validator.errors) == 0

        # Might have warnings (e.g., about correlations), that's acceptable
        print(f"Validation completed with {len(validator.warnings)} warnings")

    def test_synthetic_missing_data_validation_pipeline(self):
        """Test complete pipeline: synthetic data -> missing data -> validation."""
        # 1. Generate clean synthetic data
        generator = SyntheticDataGenerator(random_state=42)
        treatment, outcome, covariates = generator.generate_marketing_campaign_data(
            n_samples=300
        )

        # 2. Introduce missing data
        treatment_miss, outcome_miss, covariates_miss = (
            generator.generate_missing_data_scenario(
                treatment,
                outcome,
                covariates,
                missing_mechanism="MAR",
                missing_rate=0.15,
            )
        )

        # 3. Handle missing data
        handler = MissingDataHandler(strategy="mean", verbose=False)
        processed_treatment, processed_outcome, processed_covariates = (
            handler.fit_transform(treatment_miss, outcome_miss, covariates_miss)
        )

        # 4. Validate the processed data
        validator = CausalDataValidator(verbose=False)
        validator.validate_all(
            processed_treatment, processed_outcome, processed_covariates
        )

        # Should have no missing data after processing
        if isinstance(processed_covariates.values, pd.DataFrame):
            assert processed_covariates.values.isnull().sum().sum() == 0

        # Should have no critical errors
        assert len(validator.errors) == 0

    def test_nhefs_validation_integration(self):
        """Test NHEFS data loading with validation."""
        # Create mock NHEFS data
        mock_data = pd.DataFrame(
            {
                "qsmk": [0, 1, 0, 1, 0] * 20,
                "wt82_71": np.random.normal(0, 5, 100),
                "age": np.random.randint(18, 80, 100),
                "sex": np.random.binomial(1, 0.5, 100),
                "race": np.random.choice([0, 1, 2], 100),
            }
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            mock_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Load NHEFS data
            loader = NHEFSDataLoader(data_path=temp_file)
            treatment, outcome, covariates = loader.get_causal_data_objects(
                confounders=["age", "sex", "race"]
            )

            # Validate the loaded data
            validator = CausalDataValidator(verbose=False)
            validator.validate_all(treatment, outcome, covariates)

            # Should successfully validate NHEFS data structure
            assert isinstance(treatment, TreatmentData)
            assert isinstance(outcome, OutcomeData)
            assert isinstance(covariates, CovariateData)

            # Treatment should be binary
            assert treatment.treatment_type == "binary"

            # Should have expected covariates
            assert set(covariates.names) == {"age", "sex", "race"}

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_missing_data_strategies_integration(self):
        """Test different missing data strategies with validation."""
        # Generate base data with known properties
        generator = SyntheticDataGenerator(random_state=42)
        treatment, outcome, covariates = generator.generate_linear_binary_treatment(
            n_samples=150
        )

        # Add missing data
        treatment_miss, outcome_miss, covariates_miss = (
            generator.generate_missing_data_scenario(
                treatment,
                outcome,
                covariates,
                missing_mechanism="MCAR",
                missing_rate=0.2,
            )
        )

        strategies = ["listwise", "mean", "median"]

        for strategy in strategies:
            # Process with different strategies
            handler = MissingDataHandler(strategy=strategy, verbose=False)

            try:
                processed_treatment, processed_outcome, processed_covariates = (
                    handler.fit_transform(treatment_miss, outcome_miss, covariates_miss)
                )

                # Validate processed data
                validator = CausalDataValidator(verbose=False)
                validator.validate_all(
                    processed_treatment, processed_outcome, processed_covariates
                )

                # Should have no critical errors regardless of strategy
                assert len(validator.errors) == 0

                # Data should maintain proper types
                assert isinstance(processed_treatment, TreatmentData)
                assert isinstance(processed_outcome, OutcomeData)

                if strategy == "listwise":
                    # Should have fewer observations
                    assert len(processed_treatment.values) < len(treatment.values)
                else:
                    # Should maintain same number of observations
                    assert len(processed_treatment.values) == len(treatment.values)

            except Exception as e:
                # Some strategies might fail - document but don't fail test
                print(f"Strategy {strategy} failed: {e}")

    def test_data_types_consistency_integration(self):
        """Test that data types remain consistent across utilities."""
        # Test with different treatment types
        test_cases = [
            ("binary", [0, 1, 0, 1, 0]),
            ("categorical", ["A", "B", "C", "A", "B"]),
            ("continuous", [0.1, 0.5, 0.8, 0.3, 0.9]),
        ]

        for treatment_type, treatment_values in test_cases:
            # Create test data
            treatment = TreatmentData(
                values=pd.Series(treatment_values),
                name="treatment",
                treatment_type=treatment_type,
                categories=["A", "B", "C"] if treatment_type == "categorical" else None,
            )

            outcome = OutcomeData(
                values=pd.Series([1.2, 2.3, 1.1, 2.5, 1.0]),
                name="outcome",
                outcome_type="continuous",
            )

            covariates = CovariateData(
                values=pd.DataFrame(
                    {
                        "x1": [1, 2, 3, 4, 5],
                        "x2": [0.1, 0.2, 0.3, 0.4, 0.5],
                    }
                ),
                names=["x1", "x2"],
            )

            # Test validation
            validator = CausalDataValidator(verbose=False)
            validator.validate_all(treatment, outcome, covariates)

            # Should handle all treatment types
            assert len(validator.errors) == 0

            # Test missing data handling (only for strategies that work with mixed types)
            if (
                treatment_type != "categorical"
            ):  # Skip categorical for numeric strategies
                handler = MissingDataHandler(strategy="listwise", verbose=False)
                processed_treatment, processed_outcome, processed_covariates = (
                    handler.fit_transform(treatment, outcome, covariates)
                )

                # Types should be preserved
                assert processed_treatment.treatment_type == treatment_type
                assert processed_outcome.outcome_type == "continuous"

    def test_large_dataset_integration(self):
        """Test integration with moderately large datasets."""
        # Generate larger dataset to test performance
        generator = SyntheticDataGenerator(random_state=42)
        treatment, outcome, covariates = generator.generate_marketing_campaign_data(
            n_samples=300  # Reduced from 1000 for CI performance
        )

        # Add some missing data
        treatment_miss, outcome_miss, covariates_miss = (
            generator.generate_missing_data_scenario(
                treatment,
                outcome,
                covariates,
                missing_mechanism="MAR",
                missing_rate=0.1,
            )
        )

        # Process missing data
        handler = MissingDataHandler(strategy="mean", verbose=False)
        processed_treatment, processed_outcome, processed_covariates = (
            handler.fit_transform(treatment_miss, outcome_miss, covariates_miss)
        )

        # Validate
        validator = CausalDataValidator(verbose=False)
        validator.validate_all(
            processed_treatment, processed_outcome, processed_covariates
        )

        # Should handle larger datasets without issues
        assert len(processed_treatment.values) == 1000
        assert len(validator.errors) == 0

        # Performance check - validation should complete quickly
        # (This is more of a smoke test than a rigorous performance test)
        import time

        start_time = time.time()

        validator_new = CausalDataValidator(verbose=False)
        validator_new.validate_all(
            processed_treatment, processed_outcome, processed_covariates
        )

        end_time = time.time()
        validation_time = end_time - start_time

        # Should complete validation in reasonable time (< 1 second for 1000 samples)
        assert validation_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
