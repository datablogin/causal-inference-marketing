"""Data preprocessing and simulation utilities for causal inference.

This module provides comprehensive data handling utilities including:
- NHEFS dataset loader with proper preprocessing
- Data validation tools with informative error messages
- Missing data handling strategies (listwise deletion, imputation)
- Synthetic data generation for testing and examples
"""

from .missing_data import (
    MissingDataHandler,
    diagnose_missing_data,
    handle_missing_data,
    print_missing_data_report,
)
from .nhefs import NHEFSDataLoader, load_nhefs
from .synthetic import (
    SyntheticDataGenerator,
    generate_confounded_observational,
    generate_simple_rct,
)
from .validation import CausalDataValidator, validate_causal_data

__all__ = [
    # NHEFS dataset utilities
    "NHEFSDataLoader",
    "load_nhefs",
    # Data validation
    "CausalDataValidator",
    "validate_causal_data",
    # Missing data handling
    "MissingDataHandler",
    "diagnose_missing_data",
    "handle_missing_data",
    "print_missing_data_report",
    # Synthetic data generation
    "SyntheticDataGenerator",
    "generate_simple_rct",
    "generate_confounded_observational",
]
