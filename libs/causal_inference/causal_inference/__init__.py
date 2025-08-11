"""Causal Inference Library for Marketing Applications.

A comprehensive library for causal inference methods applied to marketing analytics,
following monorepo-compatible patterns for future integration.
"""

__version__ = "0.1.0"
__author__ = "Robert Welborn"

# Unified API
from .api import CausalAnalysis
from .core import *
from .data import *
from .discovery import (
    BaseDiscoveryAlgorithm,
    CausalDAG,
    DiscoveryEstimatorPipeline,
    DiscoveryResult,
    GESAlgorithm,
    NOTEARSAlgorithm,
    PCAlgorithm,
)
from .estimators import (
    AIPWEstimator,
    DoublyRobustMLEstimator,
    GComputationEstimator,
    IPWEstimator,
    IVEstimator,
    TMLEEstimator,
)
from .ml import SuperLearner
from .target_trial import TargetTrialEmulator, TargetTrialProtocol, TargetTrialResults
from .utils import *

__all__ = [
    "__version__",
    "__author__",
    "AIPWEstimator",
    "BaseDiscoveryAlgorithm",
    "CausalAnalysis",
    "CausalDAG",
    "DiscoveryEstimatorPipeline",
    "DiscoveryResult",
    "DoublyRobustMLEstimator",
    "GComputationEstimator",
    "GESAlgorithm",
    "IPWEstimator",
    "IVEstimator",
    "NOTEARSAlgorithm",
    "PCAlgorithm",
    "SuperLearner",
    "TargetTrialEmulator",
    "TargetTrialProtocol",
    "TargetTrialResults",
    "TMLEEstimator",
]
