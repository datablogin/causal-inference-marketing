"""Propensity Score Stratification and Matching estimator for causal inference.

This module implements propensity score methods beyond IPW including:
- Propensity score stratification (quintiles, deciles, custom strata)
- Nearest neighbor matching with calipers and k:1 matching ratios
- Balance diagnostics and common support visualization

The PropensityScoreEstimator provides alternative approaches to propensity score
adjustment that can be more robust when weight distributions are problematic or
when direct matching is preferred.

Example Usage:
    Stratification approach:

    >>> from causal_inference.core.base import TreatmentData, OutcomeData, CovariateData
    >>> from causal_inference.estimators.propensity_score import PropensityScoreEstimator
    >>>
    >>> # Prepare your data
    >>> treatment = TreatmentData(values=treatment_series, treatment_type="binary")
    >>> outcome = OutcomeData(values=outcome_series, outcome_type="continuous")
    >>> covariates = CovariateData(values=covariate_df, names=list(covariate_df.columns))
    >>>
    >>> # Stratification approach
    >>> estimator_strat = PropensityScoreEstimator(
    ...     method="stratification",
    ...     n_strata=5,
    ...     propensity_model="logistic",
    ...     balance_threshold=0.1
    ... )
    >>> estimator_strat.fit(treatment, outcome, covariates)
    >>> effect_strat = estimator_strat.estimate_ate()

    Matching approach:
    >>> # Matching approach
    >>> estimator_match = PropensityScoreEstimator(
    ...     method="matching",
    ...     matching_type="nearest_neighbor",
    ...     n_neighbors=1,
    ...     caliper=0.1,
    ...     replacement=False
    ... )
    >>> estimator_match.fit(treatment, outcome, covariates)
    >>> effect_match = estimator_match.estimate_ate()
    >>>
    >>> # Get diagnostics
    >>> balance_diag = estimator_match.get_balance_diagnostics()
    >>> support_diag = estimator_match.get_common_support_diagnostics()
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from ..core.bootstrap import BootstrapConfig, BootstrapMixin


class PropensityScoreEstimator(BootstrapMixin, BaseEstimator):
    """Propensity Score Stratification and Matching estimator for causal inference.

    This estimator implements multiple propensity score adjustment methods:
    - Stratification: Divides units into strata based on propensity scores
    - Matching: Matches treated and control units with similar propensity scores

    The method provides alternatives to IPW that can be more robust when
    weight distributions are problematic or when direct matching is preferred.

    Attributes:
        method: Primary estimation method ('stratification' or 'matching')
        propensity_model: Fitted sklearn model for propensity score estimation
        propensity_scores: Estimated propensity scores for each unit
        strata_assignments: Stratum assignment for each unit (stratification)
        matched_pairs: Matched unit pairs (matching)
        balance_diagnostics: Covariate balance assessment results
        common_support_diagnostics: Common support assessment results
    """

    def __init__(
        self,
        method: Literal["stratification", "matching"] = "stratification",
        # Stratification parameters
        n_strata: int = 5,
        stratification_method: str = "quantile",
        balance_threshold: float = 0.1,
        # Matching parameters
        matching_type: str = "nearest_neighbor",
        n_neighbors: int = 1,
        caliper: float | None = None,
        replacement: bool = False,
        # Propensity model parameters
        propensity_model_type: str = "logistic",
        propensity_model_params: dict[str, Any] | None = None,
        # Bootstrap and diagnostics
        bootstrap_config: Any | None = None,
        check_overlap: bool = True,
        overlap_threshold: float = 0.1,
        # Legacy parameters for backward compatibility
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the PropensityScoreEstimator.

        Args:
            method: Primary method ('stratification' or 'matching')
            n_strata: Number of strata for stratification (default: 5)
            stratification_method: Method for creating strata ('quantile', 'fixed')
            balance_threshold: Maximum standardized mean difference for balance
            matching_type: Type of matching ('nearest_neighbor')
            n_neighbors: Number of neighbors to match (1:k matching)
            caliper: Maximum propensity score distance for matching
            replacement: Whether to match with replacement
            propensity_model_type: Model type ('logistic', 'random_forest')
            propensity_model_params: Parameters for the propensity model
            bootstrap_config: Configuration for bootstrap confidence intervals
            check_overlap: Whether to check common support assumption
            overlap_threshold: Minimum propensity score for overlap check
            bootstrap_samples: Legacy parameter - number of bootstrap samples
            confidence_level: Legacy parameter - confidence level
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        # Create bootstrap config if not provided (for backward compatibility)
        if bootstrap_config is None:
            bootstrap_config = BootstrapConfig(
                n_samples=bootstrap_samples,
                confidence_level=confidence_level,
                random_state=random_state,
            )

        super().__init__(
            bootstrap_config=bootstrap_config,
            random_state=random_state,
            verbose=verbose,
        )

        # Core method parameters
        self.method = method

        # Stratification parameters
        self.n_strata = n_strata
        self.stratification_method = stratification_method
        self.balance_threshold = balance_threshold

        # Matching parameters
        self.matching_type = matching_type
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.replacement = replacement

        # Propensity model parameters
        self.propensity_model_type = propensity_model_type
        self.propensity_model_params = propensity_model_params or {}

        # Diagnostics parameters
        self.check_overlap = check_overlap
        self.overlap_threshold = overlap_threshold

        # Model storage
        self.propensity_model: SklearnBaseEstimator | None = None
        self.propensity_scores: NDArray[Any] | None = None
        self._propensity_features: list[str] | None = None

        # Method-specific results
        self.strata_assignments: NDArray[Any] | None = None
        self.strata_boundaries: NDArray[Any] | None = None
        self.matched_pairs: list[tuple[int, list[int]]] | None = None
        self.matched_indices: dict[str, NDArray[Any]] | None = None

        # Diagnostics
        self.balance_diagnostics: dict[str, Any] | None = None
        self.common_support_diagnostics: dict[str, Any] | None = None

    def _create_bootstrap_estimator(
        self, random_state: int | None = None
    ) -> PropensityScoreEstimator:
        """Create a new estimator instance for bootstrap sampling.

        Args:
            random_state: Random state for this bootstrap instance

        Returns:
            New PropensityScoreEstimator instance configured for bootstrap
        """
        return PropensityScoreEstimator(
            method=self.method,
            n_strata=self.n_strata,
            stratification_method=self.stratification_method,
            balance_threshold=self.balance_threshold,
            matching_type=self.matching_type,
            n_neighbors=self.n_neighbors,
            caliper=self.caliper,
            replacement=self.replacement,
            propensity_model_type=self.propensity_model_type,
            propensity_model_params=self.propensity_model_params,
            bootstrap_config=BootstrapConfig(n_samples=0),  # No nested bootstrap
            check_overlap=False,  # Skip overlap checks in bootstrap
            overlap_threshold=self.overlap_threshold,
            random_state=random_state,
            verbose=False,  # Reduce verbosity in bootstrap
        )

    def _create_propensity_model(self) -> SklearnBaseEstimator:
        """Create propensity score model based on model type.

        Returns:
            Initialized sklearn model for propensity score estimation
        """
        if self.propensity_model_type == "logistic":
            default_params = {
                "solver": "liblinear",
                "max_iter": 1000,
                "C": 1.0,
                "penalty": "l2",
            }
            merged_params = {**default_params, **self.propensity_model_params}
            return LogisticRegression(random_state=self.random_state, **merged_params)
        elif self.propensity_model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_state, **self.propensity_model_params
            )
        else:
            raise ValueError(
                f"Unknown propensity model type: {self.propensity_model_type}"
            )

    def _prepare_propensity_features(
        self, covariates: CovariateData | None = None
    ) -> pd.DataFrame:
        """Prepare feature matrix for propensity score estimation.

        Args:
            covariates: Covariate data for propensity score model

        Returns:
            Feature DataFrame for propensity score estimation

        Raises:
            EstimationError: If no covariates provided (required for propensity scores)
        """
        if covariates is None:
            raise EstimationError(
                "Propensity score methods require covariates for estimation. "
                "Without covariates, these methods reduce to simple difference in means."
            )

        if isinstance(covariates.values, pd.DataFrame):
            features = covariates.values.copy()
        else:
            cov_names = covariates.names or [
                f"X{i}" for i in range(covariates.values.shape[1])
            ]
            features = pd.DataFrame(covariates.values, columns=cov_names)

        return features

    def _fit_propensity_model(
        self,
        treatment: TreatmentData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the propensity score model.

        Args:
            treatment: Treatment assignment data
            covariates: Covariate data for propensity score model
        """
        # Prepare features
        X = self._prepare_propensity_features(covariates)
        self._propensity_features = list(X.columns)

        # Prepare treatment labels
        if isinstance(treatment.values, pd.Series):
            y = treatment.values.values
        else:
            y = treatment.values

        y = np.asarray(y)

        # Create and fit propensity model
        self.propensity_model = self._create_propensity_model()

        try:
            # Check for treatment variation
            unique_treatments = np.unique(y)
            if len(unique_treatments) < 2:
                raise EstimationError(
                    f"No treatment variation detected. Treatment values: {unique_treatments}. "
                    "Cannot estimate propensity scores without variation in treatment assignment."
                )

            self.propensity_model.fit(X, y)

            if self.verbose:
                # Calculate model performance metrics
                if hasattr(self.propensity_model, "predict_proba"):
                    y_pred_proba = self.propensity_model.predict_proba(X)[:, 1]
                    try:
                        auc = roc_auc_score(y, y_pred_proba)
                        print(f"Propensity model AUC: {auc:.4f}")
                    except ValueError:
                        pass  # Skip AUC if only one class present

        except Exception as e:
            raise EstimationError(f"Failed to fit propensity model: {str(e)}") from e

    def _estimate_propensity_scores(self) -> NDArray[Any]:
        """Estimate propensity scores for all units.

        Returns:
            Array of propensity scores (probability of treatment)
        """
        if self.propensity_model is None or self.covariate_data is None:
            raise EstimationError("Propensity model must be fitted before estimation")

        X = self._prepare_propensity_features(self.covariate_data)

        # Ensure features are in the same order as training
        if self._propensity_features is not None:
            X = X[self._propensity_features]

        # Get propensity scores
        if hasattr(self.propensity_model, "predict_proba"):
            propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        else:
            # Fallback for models without predict_proba
            propensity_scores = self.propensity_model.decision_function(X)
            propensity_scores = 1 / (1 + np.exp(-propensity_scores))

        return np.asarray(propensity_scores)

    def _check_common_support(self, propensity_scores: NDArray[Any]) -> dict[str, Any]:
        """Check common support assumption by examining propensity score distribution.

        Args:
            propensity_scores: Array of propensity scores

        Returns:
            Dictionary with common support diagnostics
        """
        if self.treatment_data is None:
            raise EstimationError("Treatment data required for common support check")

        treatment_values = np.asarray(self.treatment_data.values)
        treated_mask = treatment_values == 1
        control_mask = treatment_values == 0

        treated_ps = propensity_scores[treated_mask]
        control_ps = propensity_scores[control_mask]

        # Calculate overlap statistics
        min_treated_ps = float(np.min(treated_ps))
        max_treated_ps = float(np.max(treated_ps))
        min_control_ps = float(np.min(control_ps))
        max_control_ps = float(np.max(control_ps))

        # Common support region
        common_support_min = max(min_treated_ps, min_control_ps)
        common_support_max = min(max_treated_ps, max_control_ps)

        # Calculate overlap percentage
        in_support_treated = np.sum(
            (treated_ps >= common_support_min) & (treated_ps <= common_support_max)
        )
        in_support_control = np.sum(
            (control_ps >= common_support_min) & (control_ps <= common_support_max)
        )

        total_treated = len(treated_ps)
        total_control = len(control_ps)

        overlap_percentage = (in_support_treated + in_support_control) / (
            total_treated + total_control
        )

        violations = []
        if overlap_percentage < 0.9:
            violations.append(
                f"Only {overlap_percentage:.1%} of units have common support"
            )

        if common_support_max <= common_support_min:
            violations.append("No common support region exists")

        diagnostics = {
            "overlap_satisfied": len(violations) == 0,
            "violations": violations,
            "overlap_percentage": float(overlap_percentage),
            "common_support_min": float(common_support_min),
            "common_support_max": float(common_support_max),
            "min_treated_ps": min_treated_ps,
            "max_treated_ps": max_treated_ps,
            "min_control_ps": min_control_ps,
            "max_control_ps": max_control_ps,
            "treated_in_support": int(in_support_treated),
            "control_in_support": int(in_support_control),
            "total_treated": total_treated,
            "total_control": total_control,
        }

        return diagnostics

    def _create_strata(
        self, propensity_scores: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Create strata based on propensity scores.

        Args:
            propensity_scores: Array of propensity scores

        Returns:
            Tuple of (strata_assignments, strata_boundaries)
        """
        if self.stratification_method == "quantile":
            # Create strata based on quantiles of propensity scores
            boundaries = np.quantile(
                propensity_scores, np.linspace(0, 1, self.n_strata + 1)
            )
            # Ensure boundaries are unique
            boundaries = np.unique(boundaries)
            if len(boundaries) < self.n_strata + 1:
                effective_n_strata = len(boundaries) - 1
                if self.verbose:
                    print(
                        f"Warning: Reducing strata from {self.n_strata} to {effective_n_strata} due to tied propensity scores"
                    )

        elif self.stratification_method == "fixed":
            # Create fixed-width strata
            min_ps = np.min(propensity_scores)
            max_ps = np.max(propensity_scores)
            boundaries = np.linspace(min_ps, max_ps, self.n_strata + 1)
            # For fixed method, check for identical min/max (all same propensity scores)
            if min_ps == max_ps:
                if self.verbose:
                    print(
                        "Warning: All propensity scores are identical, creating single stratum"
                    )
                # Create a minimal boundary to avoid errors
                boundaries = np.array([min_ps - 1e-10, max_ps + 1e-10])
        else:
            raise ValueError(
                f"Unknown stratification method: {self.stratification_method}"
            )

        # Assign units to strata
        strata_assignments = np.digitize(propensity_scores, boundaries) - 1

        # Handle edge case where units fall exactly on the maximum boundary
        max_stratum = len(boundaries) - 2
        strata_assignments = np.clip(strata_assignments, 0, max_stratum)

        return strata_assignments, boundaries

    def _perform_matching(
        self, propensity_scores: NDArray[Any], treatment: TreatmentData
    ) -> tuple[list[tuple[int, list[int]]], dict[str, NDArray[Any]]]:
        """Perform propensity score matching.

        Args:
            propensity_scores: Array of propensity scores
            treatment: Treatment assignment data

        Returns:
            Tuple of (matched_pairs, matched_indices)
        """
        treatment_values = np.asarray(treatment.values)
        treated_indices = np.where(treatment_values == 1)[0]
        control_indices = np.where(treatment_values == 0)[0]

        if len(treated_indices) == 0 or len(control_indices) == 0:
            raise EstimationError("Need both treated and control units for matching")

        matched_pairs = []
        used_controls = set()
        matched_treated = []
        matched_controls = []

        if self.matching_type == "nearest_neighbor":
            for treated_idx in treated_indices:
                treated_ps = propensity_scores[treated_idx]

                # Find available control units
                available_controls = control_indices
                if not self.replacement:
                    available_controls = np.array(
                        [idx for idx in control_indices if idx not in used_controls]
                    )

                if len(available_controls) == 0:
                    continue  # No available controls for this treated unit

                # Calculate distances
                control_ps = propensity_scores[available_controls]
                distances = np.abs(control_ps - treated_ps)

                # Apply caliper constraint if specified
                if self.caliper is not None:
                    valid_matches = distances <= self.caliper
                    if not np.any(valid_matches):
                        continue  # No matches within caliper

                    available_controls = np.array(available_controls)[valid_matches]
                    distances = distances[valid_matches]

                # Find k nearest neighbors
                k = min(self.n_neighbors, len(available_controls))
                nearest_indices = np.argsort(distances)[:k]
                matched_control_indices = [
                    available_controls[i] for i in nearest_indices
                ]

                # Store the match
                matched_pairs.append((treated_idx, matched_control_indices))
                matched_treated.append(treated_idx)
                matched_controls.extend(matched_control_indices)

                # Mark controls as used if matching without replacement
                if not self.replacement:
                    used_controls.update(matched_control_indices)

        else:
            raise ValueError(f"Unknown matching type: {self.matching_type}")

        # Create matched indices dictionary
        matched_indices = {
            "treated": np.array(matched_treated),
            "control": np.array(matched_controls),
        }

        # Warn about low match rates
        total_treated = len(treated_indices)
        n_matched = len(matched_pairs)
        match_rate = n_matched / total_treated if total_treated > 0 else 0.0

        if match_rate < 0.5 and self.verbose:
            print(
                f"Warning: Low match rate ({match_rate:.1%}). Only {n_matched}/{total_treated} treated units matched."
            )
            if self.caliper is not None:
                print(
                    f"Consider relaxing the caliper constraint (current: {self.caliper}) or using matching with replacement."
                )
        elif match_rate < 0.8 and self.verbose:
            print(
                f"Note: Moderate match rate ({match_rate:.1%}). {n_matched}/{total_treated} treated units matched."
            )

        return matched_pairs, matched_indices

    def _compute_standardized_mean_difference(
        self,
        covariates: pd.DataFrame,
        treatment: NDArray[Any],
        weights: NDArray[Any] | None = None,
        strata: int | None = None,
    ) -> dict[str, float]:
        """Compute standardized mean differences for covariate balance.

        Args:
            covariates: Covariate DataFrame
            treatment: Treatment assignment array
            weights: Optional weights for calculation
            strata: Optional stratum number for stratified calculation

        Returns:
            Dictionary of standardized mean differences by covariate
        """
        treated_mask = treatment == 1
        control_mask = treatment == 0

        smd_dict = {}

        for col in covariates.columns:
            treated_values = covariates.loc[treated_mask, col]
            control_values = covariates.loc[control_mask, col]

            if weights is not None:
                treated_weights = weights[treated_mask]
                control_weights = weights[control_mask]

                # Weighted means
                treated_mean = np.average(treated_values, weights=treated_weights)
                control_mean = np.average(control_values, weights=control_weights)

                # Weighted variances
                treated_var = np.average(
                    (treated_values - treated_mean) ** 2, weights=treated_weights
                )
                control_var = np.average(
                    (control_values - control_mean) ** 2, weights=control_weights
                )
            else:
                treated_mean = np.mean(treated_values)
                control_mean = np.mean(control_values)
                treated_var = np.var(treated_values, ddof=1)
                control_var = np.var(control_values, ddof=1)

            # Pooled standard deviation
            pooled_std = np.sqrt((treated_var + control_var) / 2)

            # Standardized mean difference
            if pooled_std > 0:
                smd = (treated_mean - control_mean) / pooled_std
            else:
                smd = 0.0

            smd_dict[col] = smd

        return smd_dict

    def _compute_balance_diagnostics(self) -> dict[str, Any]:
        """Compute covariate balance diagnostics.

        Returns:
            Dictionary with balance diagnostics
        """
        if (
            self.covariate_data is None
            or self.treatment_data is None
            or self.propensity_scores is None
        ):
            raise EstimationError("Data required for balance diagnostics")

        covariates = self._prepare_propensity_features(self.covariate_data)
        treatment = np.asarray(self.treatment_data.values)

        # Before adjustment balance
        before_smd = self._compute_standardized_mean_difference(covariates, treatment)

        diagnostics = {
            "before_adjustment": {
                "standardized_mean_differences": before_smd,
                "max_smd": max(abs(smd) for smd in before_smd.values()),
                "balance_achieved": max(abs(smd) for smd in before_smd.values())
                < self.balance_threshold,
            }
        }

        # Method-specific balance calculations
        if self.method == "stratification" and self.strata_assignments is not None:
            strata_balance = {}
            for stratum in np.unique(self.strata_assignments):
                stratum_mask = self.strata_assignments == stratum
                if np.sum(stratum_mask) > 0:
                    stratum_covariates = covariates.loc[stratum_mask]
                    stratum_treatment = treatment[stratum_mask]

                    # Check if both treatment groups are present in this stratum
                    if len(np.unique(stratum_treatment)) > 1:
                        stratum_smd = self._compute_standardized_mean_difference(
                            stratum_covariates, stratum_treatment
                        )
                        strata_balance[f"stratum_{stratum}"] = {
                            "standardized_mean_differences": stratum_smd,
                            "max_smd": max(abs(smd) for smd in stratum_smd.values()),
                            "n_units": np.sum(stratum_mask),
                            "n_treated": np.sum(stratum_treatment == 1),
                            "n_control": np.sum(stratum_treatment == 0),
                        }

            diagnostics["after_stratification"] = {
                "strata": strata_balance,
                "overall_max_smd": max(
                    strata["max_smd"] for strata in strata_balance.values()
                )
                if strata_balance
                else float("inf"),
                "balance_achieved": max(
                    strata["max_smd"] for strata in strata_balance.values()
                )
                < self.balance_threshold
                if strata_balance
                else False,
            }

        elif self.method == "matching" and self.matched_indices is not None:
            # Balance for matched sample
            matched_mask = np.concatenate(
                [self.matched_indices["treated"], self.matched_indices["control"]]
            )

            matched_covariates = covariates.iloc[matched_mask]
            matched_treatment = treatment[matched_mask]

            after_smd = self._compute_standardized_mean_difference(
                matched_covariates, matched_treatment
            )

            diagnostics["after_matching"] = {
                "standardized_mean_differences": after_smd,
                "max_smd": max(abs(smd) for smd in after_smd.values()),
                "balance_achieved": max(abs(smd) for smd in after_smd.values())
                < self.balance_threshold,
                "match_rate": len(self.matched_indices["treated"])
                / np.sum(treatment == 1),
                "n_matched_treated": len(self.matched_indices["treated"]),
                "n_matched_control": len(self.matched_indices["control"]),
            }

        return diagnostics

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the PropensityScoreEstimator.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Covariate data for propensity score estimation
        """
        # Fit propensity score model
        self._fit_propensity_model(treatment, covariates)

        # Estimate propensity scores
        self.propensity_scores = self._estimate_propensity_scores()

        # Check common support
        if self.check_overlap:
            self.common_support_diagnostics = self._check_common_support(
                self.propensity_scores
            )

            if not self.common_support_diagnostics["overlap_satisfied"]:
                if self.verbose:
                    print("Warning: Common support violations detected:")
                    for violation in self.common_support_diagnostics["violations"]:
                        print(f"  - {violation}")

        # Apply method-specific procedures
        if self.method == "stratification":
            self.strata_assignments, self.strata_boundaries = self._create_strata(
                self.propensity_scores
            )

            if self.verbose:
                n_strata = len(np.unique(self.strata_assignments))
                print(f"Created {n_strata} strata for analysis")

        elif self.method == "matching":
            self.matched_pairs, self.matched_indices = self._perform_matching(
                self.propensity_scores, treatment
            )

            if self.verbose:
                n_matched = len(self.matched_pairs)
                total_treated = np.sum(treatment.values == 1)
                match_rate = n_matched / total_treated
                print(
                    f"Matched {n_matched}/{total_treated} treated units ({match_rate:.1%})"
                )

        # Compute balance diagnostics
        self.balance_diagnostics = self._compute_balance_diagnostics()

        if self.verbose:
            if self.method == "stratification":
                overall_balance = self.balance_diagnostics.get(
                    "after_stratification", {}
                ).get("balance_achieved", False)
                max_smd = self.balance_diagnostics.get("after_stratification", {}).get(
                    "overall_max_smd", float("inf")
                )
            else:
                overall_balance = self.balance_diagnostics.get(
                    "after_matching", {}
                ).get("balance_achieved", False)
                max_smd = self.balance_diagnostics.get("after_matching", {}).get(
                    "max_smd", float("inf")
                )

            print(f"Balance achieved: {overall_balance} (max SMD: {max_smd:.3f})")

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate Average Treatment Effect using propensity score methods.

        Returns:
            CausalEffect object with ATE estimate and confidence intervals
        """
        if (
            self.treatment_data is None
            or self.outcome_data is None
            or self.propensity_scores is None
        ):
            raise EstimationError("Model must be fitted before estimation")

        treatment_values = np.asarray(self.treatment_data.values)
        outcome_values = np.asarray(self.outcome_data.values)

        if self.method == "stratification":
            ate = self._estimate_ate_stratification(treatment_values, outcome_values)
        elif self.method == "matching":
            ate = self._estimate_ate_matching(treatment_values, outcome_values)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Bootstrap confidence intervals
        bootstrap_result = None
        ate_se = None
        ate_ci_lower = None
        ate_ci_upper = None
        bootstrap_estimates = None

        # Additional CI fields for different bootstrap methods
        ate_ci_lower_bca = None
        ate_ci_upper_bca = None
        ate_ci_lower_bias_corrected = None
        ate_ci_upper_bias_corrected = None
        ate_ci_lower_studentized = None
        ate_ci_upper_studentized = None
        bootstrap_method = None
        bootstrap_converged = None
        bootstrap_bias = None
        bootstrap_acceleration = None

        if self.bootstrap_config and self.bootstrap_config.n_samples > 0:
            try:
                bootstrap_result = self.compute_bootstrap_confidence_intervals(ate)

                # Extract bootstrap estimates and diagnostics
                bootstrap_estimates = bootstrap_result.bootstrap_estimates
                ate_se = bootstrap_result.bootstrap_se
                bootstrap_method = bootstrap_result.config.method
                bootstrap_converged = bootstrap_result.converged
                bootstrap_bias = bootstrap_result.bias_estimate
                bootstrap_acceleration = bootstrap_result.acceleration_estimate

                # Set confidence intervals based on method
                if bootstrap_result.config.method == "percentile":
                    ate_ci_lower = bootstrap_result.ci_lower_percentile
                    ate_ci_upper = bootstrap_result.ci_upper_percentile
                elif bootstrap_result.config.method == "bias_corrected":
                    ate_ci_lower = bootstrap_result.ci_lower_bias_corrected
                    ate_ci_upper = bootstrap_result.ci_upper_bias_corrected
                elif bootstrap_result.config.method == "bca":
                    ate_ci_lower = bootstrap_result.ci_lower_bca
                    ate_ci_upper = bootstrap_result.ci_upper_bca
                elif bootstrap_result.config.method == "studentized":
                    ate_ci_lower = bootstrap_result.ci_lower_studentized
                    ate_ci_upper = bootstrap_result.ci_upper_studentized

                # Set all available CI methods for comparison
                ate_ci_lower_bca = bootstrap_result.ci_lower_bca
                ate_ci_upper_bca = bootstrap_result.ci_upper_bca
                ate_ci_lower_bias_corrected = bootstrap_result.ci_lower_bias_corrected
                ate_ci_upper_bias_corrected = bootstrap_result.ci_upper_bias_corrected
                ate_ci_lower_studentized = bootstrap_result.ci_lower_studentized
                ate_ci_upper_studentized = bootstrap_result.ci_upper_studentized

            except Exception as e:
                if self.verbose:
                    print(f"Bootstrap confidence intervals failed: {str(e)}")

        # Count treatment/control units
        n_treated = np.sum(treatment_values == 1)
        n_control = np.sum(treatment_values == 0)

        # Compile diagnostics
        diagnostics = {}
        if self.common_support_diagnostics is not None:
            diagnostics["common_support"] = self.common_support_diagnostics
        if self.balance_diagnostics is not None:
            diagnostics["balance"] = self.balance_diagnostics

        return CausalEffect(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            confidence_level=self.bootstrap_config.confidence_level
            if self.bootstrap_config
            else 0.95,
            method=f"Propensity Score {self.method.title()}",
            n_observations=len(treatment_values),
            n_treated=n_treated,
            n_control=n_control,
            bootstrap_samples=self.bootstrap_config.n_samples
            if self.bootstrap_config
            else 0,
            bootstrap_estimates=bootstrap_estimates,
            # Enhanced bootstrap confidence intervals
            ate_ci_lower_bca=ate_ci_lower_bca,
            ate_ci_upper_bca=ate_ci_upper_bca,
            ate_ci_lower_bias_corrected=ate_ci_lower_bias_corrected,
            ate_ci_upper_bias_corrected=ate_ci_upper_bias_corrected,
            ate_ci_lower_studentized=ate_ci_lower_studentized,
            ate_ci_upper_studentized=ate_ci_upper_studentized,
            # Bootstrap diagnostics
            bootstrap_method=bootstrap_method,
            bootstrap_converged=bootstrap_converged,
            bootstrap_bias=bootstrap_bias,
            bootstrap_acceleration=bootstrap_acceleration,
            diagnostics=diagnostics,
        )

    def _estimate_ate_stratification(
        self, treatment_values: NDArray[Any], outcome_values: NDArray[Any]
    ) -> float:
        """Estimate ATE using stratification method.

        Args:
            treatment_values: Treatment assignment array
            outcome_values: Outcome values array

        Returns:
            Average treatment effect estimate
        """
        if self.strata_assignments is None:
            raise EstimationError("Strata assignments not available")

        stratum_effects = []
        stratum_weights = []

        for stratum in np.unique(self.strata_assignments):
            stratum_mask = self.strata_assignments == stratum
            stratum_treatment = treatment_values[stratum_mask]
            stratum_outcome = outcome_values[stratum_mask]

            # Check if both treatment groups are present
            unique_treatments = np.unique(stratum_treatment)
            if len(unique_treatments) < 2:
                continue  # Skip strata with only one treatment group

            # Calculate treatment effect within stratum
            treated_mask = stratum_treatment == 1
            control_mask = stratum_treatment == 0

            treated_outcome = np.mean(stratum_outcome[treated_mask])
            control_outcome = np.mean(stratum_outcome[control_mask])
            stratum_effect = treated_outcome - control_outcome

            # Weight by stratum size
            stratum_weight = np.sum(stratum_mask)

            stratum_effects.append(stratum_effect)
            stratum_weights.append(stratum_weight)

        if not stratum_effects:
            raise EstimationError("No valid strata found for ATE estimation")

        # Weighted average of stratum effects
        stratum_weights_array = np.array(stratum_weights)
        stratum_effects_array = np.array(stratum_effects)

        ate = np.average(stratum_effects_array, weights=stratum_weights_array)
        return float(ate)

    def _estimate_ate_matching(
        self, treatment_values: NDArray[Any], outcome_values: NDArray[Any]
    ) -> float:
        """Estimate ATE using matching method.

        Args:
            treatment_values: Treatment assignment array
            outcome_values: Outcome values array

        Returns:
            Average treatment effect estimate
        """
        if self.matched_pairs is None:
            raise EstimationError("Matched pairs not available")

        pair_effects = []

        for treated_idx, control_indices in self.matched_pairs:
            treated_outcome = outcome_values[treated_idx]

            # Average outcome across matched controls
            control_outcomes = outcome_values[control_indices]
            control_outcome = np.mean(control_outcomes)

            pair_effect = treated_outcome - control_outcome
            pair_effects.append(pair_effect)

        if not pair_effects:
            raise EstimationError("No matched pairs found for ATE estimation")

        # Average treatment effect across matched pairs
        ate = np.mean(pair_effects)
        return float(ate)

    # Public diagnostic methods
    def get_propensity_scores(self) -> NDArray[Any] | None:
        """Get the estimated propensity scores.

        Returns:
            Array of propensity scores if fitted, None otherwise
        """
        return self.propensity_scores

    def get_balance_diagnostics(self) -> dict[str, Any] | None:
        """Get covariate balance diagnostics.

        Returns:
            Dictionary with balance diagnostics if fitted, None otherwise
        """
        return self.balance_diagnostics

    def get_common_support_diagnostics(self) -> dict[str, Any] | None:
        """Get common support diagnostics.

        Returns:
            Dictionary with common support diagnostics if fitted, None otherwise
        """
        return self.common_support_diagnostics

    def get_matching_diagnostics(self) -> dict[str, Any] | None:
        """Get matching-specific diagnostics.

        Returns:
            Dictionary with matching diagnostics if fitted and method is matching
        """
        if self.method != "matching" or self.matched_pairs is None:
            return None

        if self.propensity_scores is None or self.treatment_data is None:
            return None

        treatment_values = np.asarray(self.treatment_data.values)
        total_treated = np.sum(treatment_values == 1)
        n_matched = len(self.matched_pairs)

        # Calculate average propensity score distance for matched pairs
        distances = []
        for treated_idx, control_indices in self.matched_pairs:
            treated_ps = self.propensity_scores[treated_idx]
            for control_idx in control_indices:
                control_ps = self.propensity_scores[control_idx]
                distances.append(abs(treated_ps - control_ps))

        return {
            "match_rate": n_matched / total_treated,
            "n_matched_pairs": n_matched,
            "total_treated": total_treated,
            "average_distance": np.mean(distances) if distances else 0.0,
            "max_distance": np.max(distances) if distances else 0.0,
            "min_distance": np.min(distances) if distances else 0.0,
        }

    def plot_propensity_distributions(self) -> Any:
        """Plot propensity score distributions by treatment group.

        Returns:
            Matplotlib figure object
        """
        if self.propensity_scores is None or self.treatment_data is None:
            raise EstimationError("Estimator must be fitted before plotting")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise EstimationError("matplotlib required for plotting")

        treatment_values = np.asarray(self.treatment_data.values)
        treated_ps = self.propensity_scores[treatment_values == 1]
        control_ps = self.propensity_scores[treatment_values == 0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.hist(control_ps, bins=30, alpha=0.7, label="Control", density=True)
        ax.hist(treated_ps, bins=30, alpha=0.7, label="Treated", density=True)

        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Density")
        ax.set_title("Propensity Score Distributions by Treatment Group")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add common support region if available
        if self.common_support_diagnostics is not None:
            support_min = self.common_support_diagnostics.get("common_support_min", 0)
            support_max = self.common_support_diagnostics.get("common_support_max", 1)
            ax.axvspan(
                support_min,
                support_max,
                alpha=0.2,
                color="green",
                label=f"Common Support ({support_min:.3f}-{support_max:.3f})",
            )
            ax.legend()

        plt.tight_layout()
        return fig
