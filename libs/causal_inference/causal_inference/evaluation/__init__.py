"""Evaluation modules for causal inference estimators.

This package provides evaluation metrics and diagnostic tools for
assessing the performance of causal inference methods, particularly
for heterogeneous treatment effect estimation.
"""

from .hte_metrics import (
    HTEEvaluator,
    ate_preservation_score,
    calibration_score,
    cate_r2_score,
    overlap_weighted_mse,
    pehe_score,
    policy_value,
    qini_score,
    rank_weighted_ate,
)

__all__ = [
    "HTEEvaluator",
    "pehe_score",
    "policy_value",
    "overlap_weighted_mse",
    "ate_preservation_score",
    "cate_r2_score",
    "rank_weighted_ate",
    "qini_score",
    "calibration_score",
]
