"""Tests for NHEFS dataset utilities."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.data.nhefs import NHEFSDataLoader, load_nhefs


class TestNHEFSDataLoader:
    """Test cases for NHEFS data loader."""

    def setup_method(self):
        """Set up test data."""
        # Create a small mock NHEFS dataset for testing
        self.mock_data = pd.DataFrame({
            'seqn': [1, 2, 3, 4, 5],
            'qsmk': [0, 1, 0, 1, 0],
            'wt82_71': [2.5, -1.2, 0.8, 3.1, -0.5],
            'sex': [0, 1, 0, 1, 1],
            'age': [45, 32, 58, 41, 39],
            'race': [1, 0, 1, 0, 1],
            'education': [3, 4, 2, 4, 3],
            'smokeintensity': [20, 15, 25, 10, 30],
            'smokeyrs': [25, 12, 35, 8, 20],
            'exercise': [1, 2, 0, 2, 1],
            'active': [1, 1, 0, 1, 0],
            'wt71': [70.2, 65.8, 85.3, 62.1, 78.9],
            'asthma': [0, 0, 1, 0, 0],
            'bronch': [0, 1, 0, 0, 0],
        })

        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.mock_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_initialization_with_path(self):
        """Test loader initialization with explicit path."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        assert loader.data_path == Path(self.temp_file.name)

    def test_initialization_without_path_file_not_found(self):
        """Test loader initialization when file is not found."""
        # Change to a directory where nhefs.csv doesn't exist
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                with pytest.raises(FileNotFoundError):
                    NHEFSDataLoader()
            finally:
                os.chdir(original_cwd)

    def test_load_raw_data(self):
        """Test loading raw data."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        raw_data = loader.load_raw_data()

        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) == 5
        assert 'qsmk' in raw_data.columns
        assert 'wt82_71' in raw_data.columns

        # Test that data is copied (not referenced)
        raw_data['new_col'] = 1
        raw_data2 = loader.load_raw_data()
        assert 'new_col' not in raw_data2.columns

    def test_load_processed_data_default(self):
        """Test loading processed data with default parameters."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        processed_data = loader.load_processed_data()

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == 5  # No missing data in mock
        assert 'qsmk' in processed_data.columns
        assert 'wt82_71' in processed_data.columns

        # Check that standard confounders are included
        expected_confounders = ['sex', 'age', 'race', 'education', 'smokeintensity',
                              'smokeyrs', 'exercise', 'active', 'wt71', 'asthma', 'bronch']
        for conf in expected_confounders:
            assert conf in processed_data.columns

    def test_load_processed_data_custom_confounders(self):
        """Test loading processed data with custom confounders."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        custom_confounders = ['age', 'sex', 'race']

        processed_data = loader.load_processed_data(confounders=custom_confounders)

        expected_cols = ['wt82_71', 'qsmk'] + custom_confounders
        assert set(processed_data.columns) == set(expected_cols)

    def test_load_processed_data_missing_columns(self):
        """Test handling of missing columns in confounders list."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        confounders_with_missing = ['age', 'sex', 'nonexistent_column']

        # Should not raise error, just warn and exclude missing columns
        processed_data = loader.load_processed_data(confounders=confounders_with_missing)

        assert 'age' in processed_data.columns
        assert 'sex' in processed_data.columns
        assert 'nonexistent_column' not in processed_data.columns

    def test_get_causal_data_objects(self):
        """Test conversion to causal data objects."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        treatment, outcome, covariates = loader.get_causal_data_objects()

        # Check types
        assert isinstance(treatment, TreatmentData)
        assert isinstance(outcome, OutcomeData)
        assert isinstance(covariates, CovariateData)

        # Check properties
        assert treatment.name == 'qsmk'
        assert treatment.treatment_type == 'binary'
        assert len(treatment.values) == 5

        assert outcome.name == 'wt82_71'
        assert outcome.outcome_type == 'continuous'
        assert len(outcome.values) == 5

        assert isinstance(covariates.values, pd.DataFrame)
        assert len(covariates.names) > 0

    def test_get_causal_data_objects_custom_params(self):
        """Test causal data objects with custom parameters."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        treatment, outcome, covariates = loader.get_causal_data_objects(
            outcome='age',
            treatment='sex',
            confounders=['qsmk', 'race']
        )

        assert treatment.name == 'sex'
        assert outcome.name == 'age'
        assert set(covariates.names) == {'qsmk', 'race'}

    def test_get_dataset_info(self):
        """Test dataset information extraction."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        info = loader.get_dataset_info()

        assert info['n_observations'] == 5
        assert info['n_variables'] == 14
        assert 'variables' in info
        assert 'missing_data' in info
        assert 'treatment_distribution' in info
        assert 'outcome_statistics' in info

        # Check treatment distribution
        treatment_dist = info['treatment_distribution']
        assert treatment_dist[0] == 3  # 3 control
        assert treatment_dist[1] == 2  # 2 treated

    def test_print_dataset_summary(self, capsys):
        """Test dataset summary printing."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        loader.print_dataset_summary()

        captured = capsys.readouterr()
        assert "NHEFS Dataset Summary" in captured.out
        assert "Observations: 5" in captured.out
        assert "Variables: 14" in captured.out
        assert "Treatment Distribution" in captured.out


class TestLoadNHEFSConvenienceFunction:
    """Test cases for the load_nhefs convenience function."""

    def setup_method(self):
        """Set up test data."""
        # Create mock data
        self.mock_data = pd.DataFrame({
            'qsmk': [0, 1, 0, 1],
            'wt82_71': [2.5, -1.2, 0.8, 3.1],
            'age': [45, 32, 58, 41],
            'sex': [0, 1, 0, 1],
        })

        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.mock_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_load_nhefs_return_objects(self):
        """Test load_nhefs returning causal data objects."""
        treatment, outcome, covariates = load_nhefs(
            data_path=self.temp_file.name,
            confounders=['age', 'sex'],
            return_objects=True
        )

        assert isinstance(treatment, TreatmentData)
        assert isinstance(outcome, OutcomeData)
        assert isinstance(covariates, CovariateData)

        assert len(treatment.values) == 4
        assert len(outcome.values) == 4
        assert len(covariates.names) == 2

    def test_load_nhefs_return_dataframe(self):
        """Test load_nhefs returning DataFrame."""
        df = load_nhefs(
            data_path=self.temp_file.name,
            confounders=['age', 'sex'],
            return_objects=False
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert set(df.columns) == {'qsmk', 'wt82_71', 'age', 'sex'}


class TestNHEFSWithMissingData:
    """Test NHEFS loader with missing data scenarios."""

    def setup_method(self):
        """Set up test data with missing values."""
        self.mock_data_missing = pd.DataFrame({
            'qsmk': [0, 1, np.nan, 1, 0],
            'wt82_71': [2.5, -1.2, 0.8, np.nan, -0.5],
            'age': [45, 32, 58, 41, np.nan],
            'sex': [0, 1, 0, 1, 1],
        })

        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.mock_data_missing.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_exclude_missing_outcome(self):
        """Test exclusion of missing outcome observations."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        processed_data = loader.load_processed_data(
            confounders=['age', 'sex'],
            exclude_missing_outcome=True,
            exclude_missing_treatment=False  # Don't exclude missing treatment
        )

        # Should exclude row with missing outcome but keep missing treatment
        assert len(processed_data) == 4
        assert not processed_data['wt82_71'].isnull().any()

    def test_exclude_missing_treatment(self):
        """Test exclusion of missing treatment observations."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        processed_data = loader.load_processed_data(
            confounders=['age', 'sex'],
            exclude_missing_treatment=True,
            exclude_missing_outcome=False
        )

        # Should exclude row with missing treatment
        assert len(processed_data) == 4
        assert not processed_data['qsmk'].isnull().any()

    def test_include_missing_data(self):
        """Test keeping missing data when exclusion is disabled."""
        loader = NHEFSDataLoader(data_path=self.temp_file.name)
        processed_data = loader.load_processed_data(
            confounders=['age', 'sex'],
            exclude_missing_outcome=False,
            exclude_missing_treatment=False
        )

        # Should keep all rows
        assert len(processed_data) == 5
        assert processed_data['wt82_71'].isnull().sum() == 1
        assert processed_data['qsmk'].isnull().sum() == 1


if __name__ == "__main__":
    pytest.main([__file__])
