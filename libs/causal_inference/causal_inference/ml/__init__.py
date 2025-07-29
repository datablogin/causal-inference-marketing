"""Machine learning module for advanced causal inference methods.

This module provides machine learning infrastructure for causal inference,
including Super Learner ensemble methods, cross-fitting utilities, and
advanced ML-based estimators like TMLE.
"""

from .cross_fitting import CrossFittingEstimator, create_cross_fit_data
from .super_learner import SuperLearner

__all__ = [
    "SuperLearner",
    "CrossFittingEstimator",
    "create_cross_fit_data",
]

