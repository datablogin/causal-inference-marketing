"""Unified API for causal inference estimators.

This module provides a sklearn-style unified interface across all estimators
and comprehensive HTML report generation for complete causal analysis workflows.
"""

from .unified_estimator import CausalAnalysis

__all__ = ["CausalAnalysis"]
