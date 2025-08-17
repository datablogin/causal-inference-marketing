"""Transportability and Target Population Weighting for Causal Inference.

This module provides tools to assess and adjust for covariate distribution
differences when transporting causal estimates across populations.
"""

from .diagnostics import (
    CovariateShiftDiagnostics,
    DistributionDifference,
    ShiftSeverity,
)
from .integration import TransportabilityEstimator
from .tmtl import TargetedMaximumTransportedLikelihood
from .weighting import (
    DensityRatioEstimator,
    OptimalTransportWeighting,
    TransportabilityWeightingInterface,
)

__all__ = [
    "CovariateShiftDiagnostics",
    "DistributionDifference",
    "ShiftSeverity",
    "TransportabilityWeightingInterface",
    "DensityRatioEstimator",
    "OptimalTransportWeighting",
    "TargetedMaximumTransportedLikelihood",
    "TransportabilityEstimator",
]
