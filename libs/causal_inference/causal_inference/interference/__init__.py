"""Interference and Spillover Detection Tools.

This module provides tools for detecting and quantifying spillover effects
in marketing campaigns where control group members may be exposed to treatment
through indirect channels, violating the SUTVA assumption.

Core Capabilities:
- Exposure mapping by spatial, network, and temporal patterns
- Spillover effect estimation with various mechanisms
- Two-stage randomization inference for interference-robust analysis
- Comprehensive diagnostics for spillover detection

Marketing Applications:
- Store campaigns with geographic spillover
- Social media viral effects
- Loyalty program household spillover
- Geographic marketing campaigns
"""

from .diagnostics import (
    InterferenceDiagnostics,
    SpilloverDetectionResults,
    plot_cluster_exposure_balance,
    plot_network_connectivity,
    plot_spillover_detection_power,
)
from .exposure_mapping import (
    ExposureMapper,
    GeographicExposureMapper,
    NetworkExposureMapper,
    SpatialExposureMapper,
    TemporalExposureMapper,
)
from .network_inference import (
    ClusterRandomizationInference,
    NetworkPermutationTest,
    TwoStageRandomizationInference,
)
from .spillover_estimation import (
    AdditiveSpilloverModel,
    MultiplicativeSpilloverModel,
    SpilloverEstimator,
    SpilloverResults,
    ThresholdSpilloverModel,
)

__all__ = [
    # Exposure Mapping
    "ExposureMapper",
    "GeographicExposureMapper",
    "NetworkExposureMapper",
    "SpatialExposureMapper",
    "TemporalExposureMapper",
    # Spillover Estimation
    "SpilloverEstimator",
    "SpilloverResults",
    "AdditiveSpilloverModel",
    "MultiplicativeSpilloverModel",
    "ThresholdSpilloverModel",
    # Network Inference
    "TwoStageRandomizationInference",
    "ClusterRandomizationInference",
    "NetworkPermutationTest",
    # Diagnostics
    "InterferenceDiagnostics",
    "SpilloverDetectionResults",
    "plot_cluster_exposure_balance",
    "plot_network_connectivity",
    "plot_spillover_detection_power",
]
