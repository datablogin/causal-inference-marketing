"""Causal inference estimators module.

This module provides implementations of various causal inference estimators
including G-computation, IPW, propensity score methods, G-estimation, doubly robust methods,
meta-learners for CATE, and survival analysis estimators.
"""

from .aipw import AIPWEstimator
from .doubly_robust_ml import DoublyRobustMLEstimator
from .g_computation import GComputationEstimator
from .g_estimation import GEstimationEstimator
from .ipw import IPWEstimator
from .iv import IVEstimator
from .meta_learners import (
    BaseMetaLearner,
    CATEResult,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
)
from .propensity_score import PropensityScoreEstimator
from .time_varying import StrategyComparison, StrategyOutcome, TimeVaryingEstimator
from .tmle import TMLEEstimator

__all__ = [
    "AIPWEstimator",
    "DoublyRobustMLEstimator",
    "GComputationEstimator",
    "GEstimationEstimator",
    "IPWEstimator",
    "IVEstimator",
    "PropensityScoreEstimator",
    "TimeVaryingEstimator",
    "TMLEEstimator",
    "StrategyComparison",
    "StrategyOutcome",
    # Meta-learners for CATE
    "BaseMetaLearner",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "CATEResult",
]

# Optional survival analysis imports
try:
    from .survival import SurvivalEstimator
    from .survival_aipw import SurvivalAIPWEstimator
    from .survival_g_computation import SurvivalGComputationEstimator
    from .survival_ipw import SurvivalIPWEstimator

    __all__.extend(
        [
            "SurvivalEstimator",
            "SurvivalAIPWEstimator",
            "SurvivalGComputationEstimator",
            "SurvivalIPWEstimator",
        ]
    )
except ImportError:
    # Survival analysis libraries not available
    pass
