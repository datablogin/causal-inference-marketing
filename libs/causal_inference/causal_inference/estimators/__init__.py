"""Causal inference estimators module.

This module provides implementations of various causal inference estimators
including G-computation, IPW, and doubly robust methods.
"""

from .g_computation import GComputationEstimator

__all__ = ["GComputationEstimator"]
