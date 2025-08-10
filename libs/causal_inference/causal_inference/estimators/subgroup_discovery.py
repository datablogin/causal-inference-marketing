"""Subgroup discovery methods for heterogeneous treatment effects.

This module implements algorithms for identifying subgroups with
differential treatment effects, including optimal policy trees
and virtual twins methodology.

These methods help discover meaningful patient segments or subpopulations
that respond differently to treatment, enabling personalized interventions.

IMPLEMENTATION LIMITATIONS:
- Virtual Twins uses simplified regression models rather than advanced ML techniques
  that could better capture complex treatment-outcome relationships
- OptimalPolicyTree is a basic placeholder using standard decision trees rather than
  specialized policy learning algorithms with welfare maximization objectives
- SIDES clustering relies on k-means which assumes spherical clusters and may miss
  complex subgroup structures; silhouette analysis may not always select optimal k
- All methods assume binary treatments and may not generalize to continuous/multi-valued
- Statistical significance testing uses basic t-tests without multiple comparison adjustments
- Subgroup characterization is limited to simple decision rules rather than rich descriptions

Production applications should consider more sophisticated methods like causal trees,
policy learning with doubly robust estimation, or advanced clustering techniques
designed specifically for treatment effect heterogeneity.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from ..utils.validation import validate_input_dimensions

__all__ = [
    "Subgroup",
    "SubgroupResult",
    "VirtualTwins",
    "OptimalPolicyTree",
    "SIDES",
]


class Subgroup(NamedTuple):
    """Represents a discovered subgroup with treatment effect."""

    rule: str  # Human-readable description of subgroup rule
    indices: NDArray[Any]  # Indices of observations in subgroup
    treatment_effect: float  # Estimated treatment effect in subgroup
    treatment_effect_se: float  # Standard error of treatment effect
    size: int  # Number of observations in subgroup
    p_value: float  # P-value for treatment effect significance


class SubgroupResult:
    """Result object for subgroup discovery methods."""

    def __init__(
        self,
        subgroups: list[Subgroup],
        overall_ate: float,
        method: str,
    ):
        """Initialize subgroup result.

        Args:
            subgroups: List of discovered subgroups
            overall_ate: Overall average treatment effect
            method: Method used for subgroup discovery
        """
        self.subgroups = subgroups
        self.overall_ate = overall_ate
        self.method = method

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of discovered subgroups."""
        if not self.subgroups:
            return pd.DataFrame()

        data = []
        for i, subgroup in enumerate(self.subgroups):
            data.append(
                {
                    "subgroup_id": i,
                    "rule": subgroup.rule,
                    "size": subgroup.size,
                    "treatment_effect": subgroup.treatment_effect,
                    "se": subgroup.treatment_effect_se,
                    "p_value": subgroup.p_value,
                    "significant": subgroup.p_value < 0.05,
                    "effect_vs_overall": subgroup.treatment_effect - self.overall_ate,
                }
            )

        return pd.DataFrame(data)

    def significant_subgroups(self, alpha: float = 0.05) -> list[Subgroup]:
        """Get subgroups with statistically significant treatment effects."""
        return [s for s in self.subgroups if s.p_value < alpha]


class VirtualTwins(BaseEstimator):
    """Virtual Twins method for subgroup identification.

    Identifies patients with enhanced treatment response by first
    training a model to predict treatment benefit, then using
    a second model to identify high-benefit subgroups.

    References:
        Foster, J. C., Taylor, J. M., & Ruberg, S. J. (2011). Subgroup
        identification from randomized clinical trial data. Statistics
        in medicine, 30(24), 2867-2880.
    """

    def __init__(
        self,
        benefit_learner: SklearnBaseEstimator | None = None,
        subgroup_learner: SklearnBaseEstimator | None = None,
        min_subgroup_size: int = 30,
        significance_level: float = 0.05,
        random_state: int | None = None,
        **kwargs: Any,
    ):
        """Initialize Virtual Twins.

        Args:
            benefit_learner: Model to predict treatment benefit
            subgroup_learner: Model to identify subgroups
            min_subgroup_size: Minimum subgroup size
            significance_level: Significance level for testing
            random_state: Random seed
            **kwargs: Additional arguments for parent class
        """
        super().__init__(random_state=random_state, **kwargs)

        self.benefit_learner = (
            benefit_learner
            if benefit_learner is not None
            else RandomForestRegressor(n_estimators=100, random_state=random_state)
        )
        self.subgroup_learner = (
            subgroup_learner
            if subgroup_learner is not None
            else DecisionTreeClassifier(
                min_samples_leaf=min_subgroup_size, random_state=random_state
            )
        )
        self.min_subgroup_size = min_subgroup_size
        self.significance_level = significance_level

        self._benefit_model = None
        self._subgroup_model = None
        self._training_data = None

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit Virtual Twins model.

        Args:
            treatment: Treatment assignments
            outcome: Outcomes
            covariates: Covariates for subgroup identification
        """
        T, Y, X = self._prepare_data(treatment, outcome, covariates)

        # Step 1: Estimate individual treatment benefits
        # The Virtual Twins algorithm estimates treatment benefit by
        # fitting a model that can predict outcomes with T=1 vs T=0

        # Fit benefit model on original data
        X_with_T = np.hstack([X, T.reshape(-1, 1)])
        self._benefit_model = clone(self.benefit_learner)
        self._benefit_model.fit(X_with_T, Y)

        # Step 2: Predict treatment benefits for each patient
        X_treated = np.hstack([X, np.ones((len(X), 1))])
        X_control = np.hstack([X, np.zeros((len(X), 1))])

        Y1_pred = self._benefit_model.predict(X_treated)
        Y0_pred = self._benefit_model.predict(X_control)
        benefits = Y1_pred - Y0_pred

        # Step 3: Identify high-benefit vs low-benefit groups
        # Use median split or more sophisticated approach
        benefit_threshold = np.median(benefits)
        high_benefit = (benefits > benefit_threshold).astype(int)

        # Step 4: Learn subgroup identification rules
        self._subgroup_model = clone(self.subgroup_learner)
        self._subgroup_model.fit(X, high_benefit)

        # Store training data
        self._training_data = {
            "X": X,
            "Y": Y,
            "T": T,
            "benefits": benefits,
            "benefit_threshold": benefit_threshold,
        }

    def _prepare_data(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Prepare data for Virtual Twins."""
        # Extract arrays
        if isinstance(treatment.values, pd.Series):
            T = treatment.values.values
        else:
            T = np.asarray(treatment.values).flatten()

        if isinstance(outcome.values, pd.Series):
            Y = outcome.values.values
        else:
            Y = np.asarray(outcome.values).flatten()

        if covariates is not None:
            if isinstance(covariates.values, pd.DataFrame):
                X = covariates.values.values
            else:
                X = np.asarray(covariates.values)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
        else:
            X = np.ones((len(T), 1))

        # Validate
        validate_input_dimensions(T, Y)
        validate_input_dimensions(T, X)

        # Ensure binary treatment
        unique_treatments = np.unique(T)
        if len(unique_treatments) != 2:
            raise ValueError("Virtual Twins requires binary treatment")

        # Map to 0/1 if needed
        if not np.array_equal(sorted(unique_treatments), [0, 1]):
            T = (T == unique_treatments[1]).astype(int)

        return T, Y, X

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate overall ATE (placeholder implementation)."""
        if self._training_data is None:
            raise EstimationError("Model not fitted.")

        T = self._training_data["T"]
        Y = self._training_data["Y"]

        treated_mean = np.mean(Y[T == 1])
        control_mean = np.mean(Y[T == 0])
        ate = treated_mean - control_mean

        # Simple standard error
        treated_se = np.std(Y[T == 1]) / np.sqrt(np.sum(T == 1))
        control_se = np.std(Y[T == 0]) / np.sqrt(np.sum(T == 0))
        se = np.sqrt(treated_se**2 + control_se**2)

        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return CausalEffect(
            ate=ate,
            ate_ci_lower=ci_lower,
            ate_ci_upper=ci_upper,
            method="Virtual Twins",
        )

    def discover_subgroups(
        self, covariates: pd.DataFrame | NDArray[Any] | None = None
    ) -> SubgroupResult:
        """Discover subgroups with differential treatment effects.

        Args:
            covariates: Covariate data for subgroup identification

        Returns:
            SubgroupResult with discovered subgroups
        """
        if not self.is_fitted:
            raise EstimationError("Model not fitted. Call fit() first.")

        if self._training_data is None or self._subgroup_model is None:
            raise EstimationError("No training data available.")

        X = self._training_data["X"]
        Y = self._training_data["Y"]
        T = self._training_data["T"]

        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                X_predict = covariates.values
            else:
                X_predict = np.asarray(covariates)
        else:
            X_predict = X

        # Predict subgroup membership
        subgroup_pred = self._subgroup_model.predict(X_predict)

        # Extract subgroups
        subgroups = []
        overall_ate = self._estimate_ate_implementation().ate

        for subgroup_id in np.unique(subgroup_pred):
            indices = np.where(subgroup_pred == subgroup_id)[0]

            if len(indices) < self.min_subgroup_size:
                continue  # Skip small subgroups

            # Compute treatment effect in subgroup
            subgroup_T = T[indices]
            subgroup_Y = Y[indices]

            treated_mask = subgroup_T == 1
            control_mask = subgroup_T == 0

            if not (np.sum(treated_mask) > 0 and np.sum(control_mask) > 0):
                continue  # Need both groups

            treated_mean = np.mean(subgroup_Y[treated_mask])
            control_mean = np.mean(subgroup_Y[control_mask])
            te = treated_mean - control_mean

            # Standard error
            n_treated = np.sum(treated_mask)
            n_control = np.sum(control_mask)

            if n_treated > 1 and n_control > 1:
                treated_var = np.var(subgroup_Y[treated_mask], ddof=1)
                control_var = np.var(subgroup_Y[control_mask], ddof=1)
                se = np.sqrt(treated_var / n_treated + control_var / n_control)
            else:
                se = 1.0

            # P-value (simple t-test approximation)
            if se > 0:
                t_stat = te / se
                df = n_treated + n_control - 2
                from scipy import stats

                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
            else:
                p_value = 1.0

            # Create rule description (simplified)
            rule = f"Subgroup {subgroup_id} (size={len(indices)})"

            subgroup = Subgroup(
                rule=rule,
                indices=indices,
                treatment_effect=te,
                treatment_effect_se=se,
                size=len(indices),
                p_value=p_value,
            )
            subgroups.append(subgroup)

        return SubgroupResult(
            subgroups=subgroups, overall_ate=overall_ate, method="Virtual Twins"
        )


class OptimalPolicyTree:
    """Placeholder for Optimal Policy Tree implementation.

    This would implement the optimal policy tree algorithm for
    learning interpretable treatment assignment rules.

    Note: This is a simplified placeholder. A full implementation
    would require solving a mixed-integer optimization problem.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        **kwargs: Any,
    ):
        """Initialize Optimal Policy Tree.

        Args:
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples in leaf nodes
            **kwargs: Additional arguments
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(
        self,
        x: NDArray[Any],
        outcomes: NDArray[Any],
        treatments: NDArray[Any],
        cate_estimates: NDArray[Any],
    ) -> OptimalPolicyTree:
        """Fit optimal policy tree (placeholder).

        Args:
            x: Covariates
            outcomes: Observed outcomes
            treatments: Treatment assignments
            cate_estimates: Estimated individual treatment effects

        Returns:
            Self for chaining
        """
        # Placeholder implementation using simple decision tree
        # on CATE estimates
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
        )
        self.tree_.fit(x, cate_estimates)
        return self

    def predict_policy(self, x: NDArray[Any]) -> NDArray[Any]:
        """Predict treatment policy (treat if CATE > 0).

        Args:
            x: Covariates for prediction

        Returns:
            Binary treatment recommendations
        """
        cate_pred = self.tree_.predict(x)
        return (cate_pred > 0).astype(int)


class SIDES:
    """Subgroup Identification for Differential Effect Size (SIDES).

    Placeholder for SIDES methodology that identifies subgroups
    with significantly different treatment effects from the overall
    population.

    Note: This is a simplified implementation focused on the
    core concept of identifying differential effects.
    """

    def __init__(
        self,
        base_learner: SklearnBaseEstimator | None = None,
        min_subgroup_size: int = 50,
        significance_level: float = 0.05,
        **kwargs: Any,
    ):
        """Initialize SIDES.

        Args:
            base_learner: Base model for CATE estimation
            min_subgroup_size: Minimum subgroup size
            significance_level: Significance level for testing
            **kwargs: Additional arguments
        """
        self.base_learner = (
            base_learner
            if base_learner is not None
            else RandomForestRegressor(n_estimators=100)
        )
        self.min_subgroup_size = min_subgroup_size
        self.significance_level = significance_level

    def discover_subgroups(
        self,
        x: NDArray[Any],
        outcomes: NDArray[Any],
        treatments: NDArray[Any],
        cate_estimates: NDArray[Any],
    ) -> SubgroupResult:
        """Discover subgroups with differential treatment effects.

        Args:
            X: Covariates
            outcomes: Observed outcomes
            treatments: Treatment assignments
            cate_estimates: Individual treatment effect estimates

        Returns:
            SubgroupResult with discovered subgroups
        """
        # Improved implementation: use k-means clustering on CATE estimates
        # with proper cluster selection

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        # Cluster observations based on CATE estimates and covariates
        features = np.column_stack([x, cate_estimates.reshape(-1, 1)])

        # Standardize features for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Select optimal number of clusters using silhouette analysis
        max_clusters = min(10, len(features) // (2 * self.min_subgroup_size))
        max_clusters = max(2, max_clusters)  # At least 2 clusters

        best_k = 2
        best_score = -1

        for k in range(2, max_clusters + 1):
            try:
                kmeans_test = KMeans(n_clusters=k, random_state=42)
                labels_test = kmeans_test.fit_predict(features_scaled)

                # Check if all clusters have minimum size
                cluster_sizes = [np.sum(labels_test == i) for i in range(k)]
                if min(cluster_sizes) < self.min_subgroup_size:
                    continue

                # Compute silhouette score
                score = silhouette_score(features_scaled, labels_test)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        # Fit final clustering model
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Create subgroups
        subgroups = []
        overall_ate = np.mean(cate_estimates)

        for k in range(best_k):
            indices = np.where(cluster_labels == k)[0]

            if len(indices) < self.min_subgroup_size:
                continue

            # Compute treatment effect in subgroup
            subgroup_outcomes = outcomes[indices]
            subgroup_treatments = treatments[indices]

            treated_mask = subgroup_treatments == 1
            control_mask = subgroup_treatments == 0

            if not (np.sum(treated_mask) > 0 and np.sum(control_mask) > 0):
                continue

            te = np.mean(subgroup_outcomes[treated_mask]) - np.mean(
                subgroup_outcomes[control_mask]
            )

            # Simple standard error calculation
            n_treated = np.sum(treated_mask)
            n_control = np.sum(control_mask)
            se = np.sqrt(np.var(subgroup_outcomes) * (1 / n_treated + 1 / n_control))

            # Test for difference from overall ATE
            diff = te - overall_ate
            t_stat = diff / se if se > 0 else 0
            from scipy import stats

            p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

            rule = f"Cluster {k} (n={len(indices)})"

            subgroup = Subgroup(
                rule=rule,
                indices=indices,
                treatment_effect=te,
                treatment_effect_se=se,
                size=len(indices),
                p_value=p_value,
            )
            subgroups.append(subgroup)

        return SubgroupResult(
            subgroups=subgroups, overall_ate=overall_ate, method="SIDES"
        )
