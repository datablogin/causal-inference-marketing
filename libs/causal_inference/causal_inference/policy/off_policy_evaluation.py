"""Off-policy evaluation for treatment assignment policies.

This module provides methods to evaluate the performance of treatment assignment
policies using historical data, without running new experiments.

Methods implemented:
- Inverse Propensity Scoring (IPS)
- Doubly Robust (DR) estimation
- Doubly Robust with Self-Normalized weights (DR-SN)
- Confidence intervals using influence functions

Classes:
    OffPolicyEvaluator: Main off-policy evaluation engine
    OPEResult: Results container for off-policy evaluation

Functions:
    ips_estimator: Inverse propensity scoring
    dr_estimator: Doubly robust estimation
    dr_sn_estimator: Self-normalized doubly robust
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

__all__ = [
    "OffPolicyEvaluator",
    "OPEResult",
    "ips_estimator",
    "dr_estimator",
    "dr_sn_estimator",
]


@dataclass
class OPEResult:
    """Results container for off-policy evaluation.

    Attributes:
        policy_value: Estimated value of the policy
        policy_value_se: Standard error of the policy value estimate
        confidence_interval: 95% confidence interval for policy value
        method: Evaluation method used
        bias_estimate: Estimated bias (if available)
        variance_estimate: Estimated variance
        evaluation_info: Additional evaluation details
        individual_weights: Importance weights for each individual
        individual_predictions: Outcome predictions (for DR methods)
    """

    policy_value: float
    policy_value_se: float
    confidence_interval: tuple[float, float]
    method: str
    bias_estimate: Optional[float] = None
    variance_estimate: Optional[float] = None
    evaluation_info: dict[str, Any] = field(default_factory=dict)
    individual_weights: Optional[NDArray[np.floating[Any]]] = None
    individual_predictions: Optional[NDArray[np.floating[Any]]] = None

    @property
    def ci_lower(self) -> float:
        """Lower bound of confidence interval."""
        return self.confidence_interval[0]

    @property
    def ci_upper(self) -> float:
        """Upper bound of confidence interval."""
        return self.confidence_interval[1]

    def get_evaluation_summary(self) -> dict[str, Any]:
        """Get summary statistics for the evaluation."""
        return {
            "method": self.method,
            "policy_value": self.policy_value,
            "policy_value_se": self.policy_value_se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "bias_estimate": self.bias_estimate,
            "variance_estimate": self.variance_estimate,
        }


class OffPolicyEvaluator:
    """Off-policy evaluation engine.

    This class provides methods to evaluate treatment assignment policies
    using historical data without conducting new experiments.

    Args:
        method: Evaluation method ('ips', 'dr', 'dr_sn')
        propensity_model: Model for propensity score estimation
        outcome_model: Model for outcome prediction (for DR methods)
        clip_weights: Whether to clip importance weights
        weight_clip_threshold: Threshold for weight clipping
        bootstrap_samples: Number of bootstrap samples for CI
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        method: str = "dr",
        propensity_model: Optional[SklearnBaseEstimator] = None,
        outcome_model: Optional[SklearnBaseEstimator] = None,
        clip_weights: bool = True,
        weight_clip_threshold: float = 10.0,
        bootstrap_samples: int = 200,
        random_state: Optional[int] = None,
    ):
        self.method = method
        self.propensity_model = (
            propensity_model if propensity_model is not None else LogisticRegression()
        )
        self.outcome_model = (
            outcome_model if outcome_model is not None else RandomForestRegressor()
        )
        self.clip_weights = clip_weights
        self.weight_clip_threshold = weight_clip_threshold
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Validate method
        valid_methods = ["ips", "dr", "dr_sn"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

    def evaluate_policy(
        self,
        policy_assignment: NDArray[np.bool_],
        historical_treatments: NDArray[np.bool_],
        historical_outcomes: NDArray[np.floating[Any]],
        features: Optional[NDArray[np.floating[Any]]] = None,
        propensity_scores: Optional[NDArray[np.floating[Any]]] = None,
    ) -> OPEResult:
        """Evaluate policy using off-policy methods.

        Args:
            policy_assignment: Treatment assignment under new policy
            historical_treatments: Historical treatment assignments
            historical_outcomes: Historical outcomes
            features: Individual features for modeling
            propensity_scores: Known propensity scores (optional)

        Returns:
            OPEResult: Policy evaluation results
        """
        n_individuals = len(policy_assignment)

        # Validate inputs
        if len(historical_treatments) != n_individuals:
            raise ValueError("All inputs must have same length")
        if len(historical_outcomes) != n_individuals:
            raise ValueError("All inputs must have same length")

        # Estimate propensity scores if not provided
        if propensity_scores is None:
            propensity_scores = self._estimate_propensity_scores(
                historical_treatments, features
            )

        # Apply evaluation method
        if self.method == "ips":
            return self._evaluate_ips(
                policy_assignment,
                historical_treatments,
                historical_outcomes,
                propensity_scores,
            )
        elif self.method == "dr":
            return self._evaluate_dr(
                policy_assignment,
                historical_treatments,
                historical_outcomes,
                propensity_scores,
                features,
            )
        elif self.method == "dr_sn":
            return self._evaluate_dr_sn(
                policy_assignment,
                historical_treatments,
                historical_outcomes,
                propensity_scores,
                features,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _estimate_propensity_scores(
        self,
        treatments: NDArray[np.bool_],
        features: Optional[NDArray[np.floating[Any]]] = None,
    ) -> NDArray[np.floating[Any]]:
        """Estimate propensity scores using features."""
        if features is None:
            # Use marginal treatment probability
            return np.full(len(treatments), np.mean(treatments))

        # Fit propensity model with cross-validation to avoid overfitting
        try:
            propensity_scores = cross_val_predict(
                self.propensity_model,
                features,
                treatments,
                cv=5,
                method="predict_proba",
            )[:, 1]  # Probability of treatment
        except Exception:
            # Fallback to simple fit
            self.propensity_model.fit(features, treatments)
            propensity_scores = self.propensity_model.predict_proba(features)[:, 1]

        # Clip propensity scores to avoid extreme weights
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

        return np.asarray(propensity_scores, dtype=np.float64)

    def _evaluate_ips(
        self,
        policy_assignment: NDArray[np.bool_],
        historical_treatments: NDArray[np.bool_],
        historical_outcomes: NDArray[np.floating[Any]],
        propensity_scores: NDArray[np.floating[Any]],
    ) -> OPEResult:
        """Inverse Propensity Scoring evaluation."""
        # Calculate importance weights
        weights = np.zeros(len(policy_assignment))

        # Treatment weights
        treated_mask = historical_treatments
        weights[treated_mask & policy_assignment] = (
            1.0 / propensity_scores[treated_mask & policy_assignment]
        )

        # Control weights
        control_mask = ~historical_treatments
        weights[control_mask & ~policy_assignment] = 1.0 / (
            1.0 - propensity_scores[control_mask & ~policy_assignment]
        )

        # Clip weights if requested
        if self.clip_weights:
            weights = np.clip(weights, 0, self.weight_clip_threshold)

        # Calculate policy value
        policy_value = float(np.mean(weights * historical_outcomes))

        # Calculate standard error using influence functions
        influence_scores = weights * historical_outcomes - policy_value
        policy_value_se = float(np.std(influence_scores) / np.sqrt(len(influence_scores)))

        # Confidence interval
        ci_lower = float(policy_value - 1.96 * policy_value_se)
        ci_upper = float(policy_value + 1.96 * policy_value_se)

        return OPEResult(
            policy_value=policy_value,
            policy_value_se=policy_value_se,
            confidence_interval=(ci_lower, ci_upper),
            method="ips",
            variance_estimate=float(policy_value_se**2),
            evaluation_info={
                "effective_sample_size": np.sum(weights > 0),
                "max_weight": np.max(weights),
                "weight_cv": np.std(weights) / (np.mean(weights) + 1e-8),
            },
            individual_weights=weights,
        )

    def _evaluate_dr(
        self,
        policy_assignment: NDArray[np.bool_],
        historical_treatments: NDArray[np.bool_],
        historical_outcomes: NDArray[np.floating[Any]],
        propensity_scores: NDArray[np.floating[Any]],
        features: Optional[NDArray[np.floating[Any]]] = None,
    ) -> OPEResult:
        """Doubly Robust evaluation."""
        # Estimate outcome models
        mu_0, mu_1 = self._estimate_outcome_models(
            historical_treatments, historical_outcomes, features
        )

        # Calculate importance weights (same as IPS)
        weights = np.zeros(len(policy_assignment))

        treated_mask = historical_treatments
        weights[treated_mask] = (
            policy_assignment[treated_mask] / propensity_scores[treated_mask]
        )

        control_mask = ~historical_treatments
        weights[control_mask] = (1 - policy_assignment[control_mask]) / (
            1.0 - propensity_scores[control_mask]
        )

        # Clip weights if requested
        if self.clip_weights:
            weights = np.clip(weights, 0, self.weight_clip_threshold)

        # Doubly robust estimator
        # E[Y^π] = E[μ₁(X)π(X) + μ₀(X)(1-π(X))] + E[W(Y - μ_T(X))]

        # Direct method component
        direct_component = mu_1 * policy_assignment + mu_0 * (1 - policy_assignment)

        # Bias correction component
        outcome_predictions = np.where(historical_treatments, mu_1, mu_0)
        bias_correction = weights * (historical_outcomes - outcome_predictions)

        # Combined estimator
        dr_scores = direct_component + bias_correction
        policy_value = float(np.mean(dr_scores))

        # Standard error
        policy_value_se = float(np.std(dr_scores) / np.sqrt(len(dr_scores)))

        # Confidence interval
        ci_lower = float(policy_value - 1.96 * policy_value_se)
        ci_upper = float(policy_value + 1.96 * policy_value_se)

        return OPEResult(
            policy_value=policy_value,
            policy_value_se=policy_value_se,
            confidence_interval=(ci_lower, ci_upper),
            method="dr",
            variance_estimate=float(policy_value_se**2),
            evaluation_info={
                "direct_component": np.mean(direct_component),
                "bias_correction": np.mean(bias_correction),
                "effective_sample_size": np.sum(weights > 0),
                "max_weight": np.max(weights),
            },
            individual_weights=weights,
            individual_predictions=outcome_predictions,
        )

    def _evaluate_dr_sn(
        self,
        policy_assignment: NDArray[np.bool_],
        historical_treatments: NDArray[np.bool_],
        historical_outcomes: NDArray[np.floating[Any]],
        propensity_scores: NDArray[np.floating[Any]],
        features: Optional[NDArray[np.floating[Any]]] = None,
    ) -> OPEResult:
        """Doubly Robust with Self-Normalized weights."""
        # Get regular DR result first
        dr_result = self._evaluate_dr(
            policy_assignment,
            historical_treatments,
            historical_outcomes,
            propensity_scores,
            features,
        )

        # Self-normalize the weights
        assert dr_result.individual_weights is not None
        raw_weights = dr_result.individual_weights
        normalized_weights = raw_weights / (np.mean(raw_weights) + 1e-8)

        # Recalculate with normalized weights
        # Get the separate μ₁ and μ₀ predictions from the DR result
        mu_0, mu_1 = self._estimate_outcome_models(
            historical_treatments,
            historical_outcomes,
            features,
        )

        # Correct direct component using separate predictions for treatment/control
        direct_component = mu_1 * policy_assignment + mu_0 * (1 - policy_assignment)

        # Use the original outcome predictions for bias correction
        assert dr_result.individual_predictions is not None
        outcome_predictions = dr_result.individual_predictions
        bias_correction = normalized_weights * (
            historical_outcomes - outcome_predictions
        )

        dr_sn_scores = direct_component + bias_correction
        policy_value = float(np.mean(dr_sn_scores))

        # Standard error (adjusted for self-normalization)
        policy_value_se = float(np.std(dr_sn_scores) / np.sqrt(len(dr_sn_scores)))

        # Confidence interval
        ci_lower = float(policy_value - 1.96 * policy_value_se)
        ci_upper = float(policy_value + 1.96 * policy_value_se)

        return OPEResult(
            policy_value=policy_value,
            policy_value_se=policy_value_se,
            confidence_interval=(ci_lower, ci_upper),
            method="dr_sn",
            variance_estimate=float(policy_value_se**2),
            evaluation_info={
                "direct_component": np.mean(direct_component),
                "bias_correction": np.mean(bias_correction),
                "normalization_factor": np.mean(raw_weights),
                "effective_sample_size": np.sum(normalized_weights > 0),
            },
            individual_weights=normalized_weights,
            individual_predictions=outcome_predictions,
        )

    def _estimate_outcome_models(
        self,
        treatments: NDArray[np.bool_],
        outcomes: NDArray[np.floating[Any]],
        features: Optional[NDArray[np.floating[Any]]] = None,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Estimate outcome models for treated and control groups."""
        n_individuals = len(treatments)

        if features is None:
            # Use marginal means
            mu_0 = np.full(n_individuals, np.mean(outcomes[~treatments]))
            mu_1 = np.full(n_individuals, np.mean(outcomes[treatments]))
            return mu_0, mu_1

        # Fit separate models for treated and control
        treated_mask = treatments
        control_mask = ~treatments

        # Control group model
        if np.sum(control_mask) > 0:
            try:
                # Use cross-validation to avoid overfitting
                if np.sum(control_mask) >= 10:  # Enough samples for modeling
                    # Fit model on all control data for predictions on full dataset
                    outcome_model_control = RandomForestRegressor(
                        random_state=self.random_state
                    )
                    outcome_model_control.fit(
                        features[control_mask], outcomes[control_mask]
                    )
                    mu_0 = outcome_model_control.predict(features)
                else:
                    # Not enough samples for modeling, use marginal mean
                    mu_0 = np.full(n_individuals, np.mean(outcomes[control_mask]))
            except Exception:
                mu_0 = np.full(n_individuals, np.mean(outcomes[control_mask]))
        else:
            mu_0 = np.full(n_individuals, np.mean(outcomes))

        # Treated group model
        if np.sum(treated_mask) > 0:
            try:
                mu_1 = np.full(n_individuals, np.mean(outcomes[treated_mask]))
                if np.sum(treated_mask) >= 10:  # Enough samples for modeling
                    outcome_model_treated = RandomForestRegressor(
                        random_state=self.random_state
                    )
                    outcome_model_treated.fit(
                        features[treated_mask], outcomes[treated_mask]
                    )
                    mu_1 = outcome_model_treated.predict(features)
            except Exception:
                mu_1 = np.full(n_individuals, np.mean(outcomes[treated_mask]))
        else:
            mu_1 = np.full(n_individuals, np.mean(outcomes))

        return mu_0, mu_1

    def compare_policies(
        self,
        policy1_assignment: NDArray[np.bool_],
        policy2_assignment: NDArray[np.bool_],
        historical_treatments: NDArray[np.bool_],
        historical_outcomes: NDArray[np.floating[Any]],
        features: Optional[NDArray[np.floating[Any]]] = None,
    ) -> dict[str, Any]:
        """Compare two policies using off-policy evaluation.

        Returns:
            Dictionary with comparison results including confidence interval for difference
        """
        # Evaluate both policies
        result1 = self.evaluate_policy(
            policy1_assignment, historical_treatments, historical_outcomes, features
        )
        result2 = self.evaluate_policy(
            policy2_assignment, historical_treatments, historical_outcomes, features
        )

        # Calculate difference
        diff = result1.policy_value - result2.policy_value
        diff_se = np.sqrt(result1.policy_value_se**2 + result2.policy_value_se**2)

        # Confidence interval for difference
        diff_ci_lower = diff - 1.96 * diff_se
        diff_ci_upper = diff + 1.96 * diff_se

        # Statistical significance test
        t_stat = diff / (diff_se + 1e-8)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            "policy1_value": result1.policy_value,
            "policy2_value": result2.policy_value,
            "difference": diff,
            "difference_se": diff_se,
            "difference_ci": (diff_ci_lower, diff_ci_upper),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
            "policy1_better": bool(diff > 0 and p_value < 0.05),
        }

    def check_positivity(
        self,
        treatments: NDArray[np.bool_],
        features: Optional[NDArray[np.floating[Any]]] = None,
        propensity_scores: Optional[NDArray[np.floating[Any]]] = None,
        min_propensity: float = 0.01,
        max_propensity: float = 0.99,
    ) -> dict[str, Any]:
        """Check positivity assumption: 0 < e(x) < 1 for all x.

        Args:
            treatments: Historical treatment assignments
            features: Individual features (optional)
            propensity_scores: Known propensity scores (optional)
            min_propensity: Minimum acceptable propensity score
            max_propensity: Maximum acceptable propensity score

        Returns:
            Dictionary with positivity diagnostics
        """
        if propensity_scores is None:
            propensity_scores = self._estimate_propensity_scores(treatments, features)

        # Check for violations
        too_low = propensity_scores < min_propensity
        too_high = propensity_scores > max_propensity

        return {
            "n_violations_low": np.sum(too_low),
            "n_violations_high": np.sum(too_high),
            "violation_rate": (np.sum(too_low) + np.sum(too_high))
            / len(propensity_scores),
            "min_propensity": np.min(propensity_scores),
            "max_propensity": np.max(propensity_scores),
            "mean_propensity": np.mean(propensity_scores),
            "std_propensity": np.std(propensity_scores),
            "positivity_ok": not np.any(too_low) and not np.any(too_high),
            "extreme_indices": np.where(too_low | too_high)[0],
        }

    def check_overlap(
        self,
        treatments: NDArray[np.bool_],
        features: NDArray[np.floating[Any]],
    ) -> dict[str, Any]:
        """Check covariate overlap between treatment groups.

        Args:
            treatments: Historical treatment assignments
            features: Individual features

        Returns:
            Dictionary with overlap diagnostics
        """
        # Handle empty arrays
        if len(treatments) == 0 or len(features) == 0:
            raise ValueError("Treatments and features cannot be empty")

        treated_mask = treatments
        control_mask = ~treatments

        treated_features = features[treated_mask]
        control_features = features[control_mask]

        if len(treated_features) == 0 or len(control_features) == 0:
            return {
                "overlap_ok": False,
                "reason": "No overlap - missing treatment or control group",
                "n_treated": len(treated_features),
                "n_control": len(control_features),
            }

        # Calculate mean and std for each group
        treated_mean = np.mean(treated_features, axis=0)
        control_mean = np.mean(control_features, axis=0)
        treated_std = np.std(treated_features, axis=0)
        control_std = np.std(control_features, axis=0)

        # Standardized mean differences
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        standardized_diff = np.abs(treated_mean - control_mean) / (pooled_std + 1e-8)

        # Check for large differences (> 0.25 is concerning, > 0.5 is problematic)
        concerning_features = standardized_diff > 0.25
        problematic_features = standardized_diff > 0.5

        return {
            "overlap_ok": not np.any(problematic_features),
            "n_treated": len(treated_features),
            "n_control": len(control_features),
            "standardized_diffs": standardized_diff,
            "max_std_diff": np.max(standardized_diff),
            "mean_std_diff": np.mean(standardized_diff),
            "concerning_features": np.where(concerning_features)[0],
            "problematic_features": np.where(problematic_features)[0],
            "n_concerning": np.sum(concerning_features),
            "n_problematic": np.sum(problematic_features),
        }

    def diagnose_assumptions(
        self,
        treatments: NDArray[np.bool_],
        features: NDArray[np.floating[Any]],
        propensity_scores: Optional[NDArray[np.floating[Any]]] = None,
    ) -> dict[str, Any]:
        """Comprehensive diagnostic check for causal inference assumptions.

        Args:
            treatments: Historical treatment assignments
            features: Individual features
            propensity_scores: Known propensity scores (optional)

        Returns:
            Dictionary with all diagnostic results
        """
        positivity_check = self.check_positivity(
            treatments, features, propensity_scores
        )
        overlap_check = self.check_overlap(treatments, features)

        # Overall assessment
        assumptions_ok = (
            positivity_check["positivity_ok"] and overlap_check["overlap_ok"]
        )

        warnings = []
        if not positivity_check["positivity_ok"]:
            warnings.append(
                f"Positivity violations: {positivity_check['violation_rate']:.1%} of observations"
            )
        if not overlap_check["overlap_ok"]:
            warnings.append(
                f"Poor covariate overlap: {overlap_check['n_problematic']} features have large differences"
            )

        return {
            "assumptions_ok": assumptions_ok,
            "warnings": warnings,
            "positivity": positivity_check,
            "overlap": overlap_check,
            "recommendation": "Review problematic features and consider sensitivity analysis"
            if not assumptions_ok
            else "Assumptions appear satisfied for causal inference",
        }


def ips_estimator(
    policy_assignment: NDArray[np.bool_],
    historical_treatments: NDArray[np.bool_],
    historical_outcomes: NDArray[np.floating[Any]],
    propensity_scores: NDArray[np.floating[Any]],
) -> float:
    """Simple IPS estimator function.

    Args:
        policy_assignment: Treatment assignment under new policy
        historical_treatments: Historical treatment assignments
        historical_outcomes: Historical outcomes
        propensity_scores: Propensity scores for treatment

    Returns:
        Estimated policy value
    """
    evaluator = OffPolicyEvaluator(method="ips")
    result = evaluator.evaluate_policy(
        policy_assignment,
        historical_treatments,
        historical_outcomes,
        propensity_scores=propensity_scores,
    )
    return result.policy_value


def dr_estimator(
    policy_assignment: NDArray[np.bool_],
    historical_treatments: NDArray[np.bool_],
    historical_outcomes: NDArray[np.floating[Any]],
    features: NDArray[np.floating[Any]],
    propensity_scores: Optional[NDArray[np.floating[Any]]] = None,
) -> float:
    """Simple DR estimator function.

    Args:
        policy_assignment: Treatment assignment under new policy
        historical_treatments: Historical treatment assignments
        historical_outcomes: Historical outcomes
        features: Individual features
        propensity_scores: Propensity scores (optional)

    Returns:
        Estimated policy value
    """
    evaluator = OffPolicyEvaluator(method="dr")
    result = evaluator.evaluate_policy(
        policy_assignment,
        historical_treatments,
        historical_outcomes,
        features,
        propensity_scores,
    )
    return result.policy_value


def dr_sn_estimator(
    policy_assignment: NDArray[np.bool_],
    historical_treatments: NDArray[np.bool_],
    historical_outcomes: NDArray[np.floating[Any]],
    features: NDArray[np.floating[Any]],
    propensity_scores: Optional[NDArray[np.floating[Any]]] = None,
) -> float:
    """Simple DR-SN estimator function.

    Args:
        policy_assignment: Treatment assignment under new policy
        historical_treatments: Historical treatment assignments
        historical_outcomes: Historical outcomes
        features: Individual features
        propensity_scores: Propensity scores (optional)

    Returns:
        Estimated policy value
    """
    evaluator = OffPolicyEvaluator(method="dr_sn")
    result = evaluator.evaluate_policy(
        policy_assignment,
        historical_treatments,
        historical_outcomes,
        features,
        propensity_scores,
    )
    return result.policy_value
