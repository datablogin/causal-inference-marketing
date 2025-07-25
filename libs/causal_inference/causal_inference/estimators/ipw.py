"""Inverse Probability Weighting (IPW) estimator for causal inference.

This module implements the IPW method, which estimates causal effects by weighting
observations by the inverse of their propensity scores (treatment assignment probabilities).

The IPW estimator is particularly useful when you have measured confounders and want to
adjust for selection bias in treatment assignment. It works by:

1. Estimating propensity scores (probability of treatment given covariates)
2. Computing inverse probability weights for each unit
3. Using these weights to estimate causal effects

Example Usage:
    Basic IPW estimation with logistic regression propensity model:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from causal_inference.core.base import TreatmentData, OutcomeData, CovariateData
    >>> from causal_inference.estimators.ipw import IPWEstimator
    >>> 
    >>> # Prepare your data
    >>> treatment = TreatmentData(values=treatment_series, treatment_type="binary")
    >>> outcome = OutcomeData(values=outcome_series, outcome_type="continuous")
    >>> covariates = CovariateData(values=covariate_df, names=list(covariate_df.columns))
    >>> 
    >>> # Initialize and fit the estimator
    >>> estimator = IPWEstimator(
    ...     propensity_model_type="logistic",
    ...     weight_truncation="percentile",
    ...     truncation_threshold=0.05,
    ...     stabilized_weights=True,
    ...     bootstrap_samples=1000,
    ...     random_state=42
    ... )
    >>> 
    >>> # Fit the model
    >>> estimator.fit(treatment, outcome, covariates)
    >>> 
    >>> # Estimate causal effect
    >>> effect = estimator.estimate_ate()
    >>> print(f"Average Treatment Effect: {effect.ate:.3f}")
    >>> print(f"95% CI: [{effect.ate_ci_lower:.3f}, {effect.ate_ci_upper:.3f}]")
    >>> 
    >>> # Check diagnostics
    >>> overlap_diag = estimator.get_overlap_diagnostics()
    >>> weight_diag = estimator.get_weight_diagnostics()
    >>> print(f"Overlap satisfied: {overlap_diag['overlap_satisfied']}")
    >>> print(f"Effective sample size: {weight_diag['effective_sample_size']:.1f}")

Advanced Usage with Random Forest and Custom Options:
    >>> # Use random forest for propensity score estimation
    >>> estimator_rf = IPWEstimator(
    ...     propensity_model_type="random_forest",
    ...     propensity_model_params={"n_estimators": 100, "max_depth": 5},
    ...     weight_truncation="threshold",
    ...     truncation_threshold=0.01,  # Truncate weights outside [0.01, 100]
    ...     stabilized_weights=True,
    ...     check_overlap=True,
    ...     overlap_threshold=0.05,
    ...     verbose=True
    ... )
    >>> 
    >>> estimator_rf.fit(treatment, outcome, covariates)
    >>> effect_rf = estimator_rf.estimate_ate()

Propensity Score Diagnostics:
    >>> # Get propensity scores for diagnostic purposes
    >>> propensity_scores = estimator.get_propensity_scores()
    >>> 
    >>> # Check overlap assumption
    >>> overlap_diagnostics = estimator.get_overlap_diagnostics()
    >>> if not overlap_diagnostics["overlap_satisfied"]:
    ...     print("Warning: Overlap assumption violated!")
    ...     for violation in overlap_diagnostics["violations"]:
    ...         print(f"  - {violation}")
    >>> 
    >>> # Examine weight distribution
    >>> weight_diagnostics = estimator.get_weight_diagnostics()
    >>> print(f"Weight range: [{weight_diagnostics['min_weight']:.3f}, {weight_diagnostics['max_weight']:.3f}]")
    >>> print(f"Mean weight: {weight_diagnostics['mean_weight']:.3f}")
    >>> print(f"% extreme weights: {weight_diagnostics['extreme_weights_pct']:.1f}%")

Notes:
    - IPW requires the "no unmeasured confounders" assumption
    - Overlap/positivity is crucial - all units must have non-zero probability of each treatment
    - Weight truncation helps handle extreme weights from poor overlap
    - Stabilized weights can improve finite-sample properties
    - Bootstrap confidence intervals account for propensity score estimation uncertainty
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    EstimationError,
    OutcomeData,
    TreatmentData,
)


class IPWEstimator(BaseEstimator):
    """Inverse Probability Weighting estimator for causal inference.

    IPW estimates causal effects by:
    1. Fitting a propensity score model that predicts treatment assignment from covariates
    2. Computing inverse probability weights for each unit
    3. Using these weights to estimate causal effects

    This method requires the "no unmeasured confounders" assumption and positivity
    (overlap) to provide unbiased estimates.

    Attributes:
        propensity_model: The fitted sklearn model for propensity score estimation
        propensity_model_type: Type of model to use ('logistic', 'random_forest')
        propensity_scores: Estimated propensity scores for each unit
        weights: IPW weights for each unit
        weight_truncation: Truncation method for extreme weights
        truncation_threshold: Threshold for weight truncation
        stabilized_weights: Whether to use stabilized weights
    """

    def __init__(
        self,
        propensity_model_type: str = "logistic",
        propensity_model_params: dict[str, Any] | None = None,
        weight_truncation: str | None = None,
        truncation_threshold: float = 0.01,
        stabilized_weights: bool = False,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        check_overlap: bool = True,
        overlap_threshold: float = 0.1,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the IPW estimator.

        Args:
            propensity_model_type: Model type ('logistic', 'random_forest')
            propensity_model_params: Parameters to pass to the sklearn model
            weight_truncation: Truncation method ('percentile', 'threshold', None)
            truncation_threshold: Threshold for truncation (0.01 = 1st/99th percentile)
            stabilized_weights: Whether to use stabilized weights
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals
            check_overlap: Whether to check overlap assumption
            overlap_threshold: Minimum propensity score for overlap check
            random_state: Random seed for reproducible results
            verbose: Whether to print verbose output
        """
        super().__init__(random_state=random_state, verbose=verbose)

        self.propensity_model_type = propensity_model_type
        self.propensity_model_params = propensity_model_params or {}
        self.weight_truncation = weight_truncation
        self.truncation_threshold = truncation_threshold
        self.stabilized_weights = stabilized_weights
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.check_overlap = check_overlap
        self.overlap_threshold = overlap_threshold

        # Model storage
        self.propensity_model: SklearnBaseEstimator | None = None
        self.propensity_scores: NDArray[Any] | None = None
        self.weights: NDArray[Any] | None = None
        self._propensity_features: list[str] | None = None

        # Diagnostics
        self._overlap_diagnostics: dict[str, Any] | None = None
        self._weight_diagnostics: dict[str, Any] | None = None

    def _create_propensity_model(self) -> SklearnBaseEstimator:
        """Create propensity score model based on model type.

        Returns:
            Initialized sklearn model for propensity score estimation
        """
        if self.propensity_model_type == "logistic":
            return LogisticRegression(
                random_state=self.random_state, **self.propensity_model_params
            )
        elif self.propensity_model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_state, **self.propensity_model_params
            )
        else:
            raise ValueError(f"Unknown propensity model type: {self.propensity_model_type}")

    def _prepare_propensity_features(
        self, covariates: CovariateData | None = None
    ) -> pd.DataFrame:
        """Prepare feature matrix for propensity score estimation.

        Args:
            covariates: Covariate data for propensity score model

        Returns:
            Feature DataFrame for propensity score estimation

        Raises:
            EstimationError: If no covariates provided (required for IPW)
        """
        if covariates is None:
            raise EstimationError(
                "IPW requires covariates for propensity score estimation. "
                "Without covariates, IPW reduces to simple difference in means."
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

        # Create and fit propensity model
        self.propensity_model = self._create_propensity_model()

        try:
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

                    ll = log_loss(y, y_pred_proba)
                    print(f"Propensity model log-loss: {ll:.4f}")

                # Cross-validation score
                try:
                    cv_scores = cross_val_score(
                        self.propensity_model, X, y, cv=5, scoring="roc_auc"
                    )
                    print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                except ValueError:
                    pass  # Skip if cross-validation fails

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
            # Convert to probabilities using sigmoid
            propensity_scores = 1 / (1 + np.exp(-propensity_scores))

        return propensity_scores

    def _check_overlap(self, propensity_scores: NDArray[Any]) -> dict[str, Any]:
        """Check overlap assumption by examining propensity score distribution.

        Args:
            propensity_scores: Array of propensity scores

        Returns:
            Dictionary with overlap diagnostics
        """
        violations: list[str] = []
        warnings: list[str] = []
        
        diagnostics = {
            "overlap_satisfied": True,
            "violations": violations,
            "warnings": warnings,
            "min_propensity": float(np.min(propensity_scores)),
            "max_propensity": float(np.max(propensity_scores)),
            "mean_propensity": float(np.mean(propensity_scores)),
            "propensity_range": float(np.max(propensity_scores) - np.min(propensity_scores)),
        }

        # Check for extreme propensity scores
        if np.min(propensity_scores) < self.overlap_threshold:
            diagnostics["overlap_satisfied"] = False
            violations.append(
                f"Minimum propensity score ({np.min(propensity_scores):.4f}) "
                f"below threshold ({self.overlap_threshold})"
            )

        if np.max(propensity_scores) > (1 - self.overlap_threshold):
            diagnostics["overlap_satisfied"] = False
            violations.append(
                f"Maximum propensity score ({np.max(propensity_scores):.4f}) "
                f"above threshold ({1 - self.overlap_threshold})"
            )

        # Additional warnings for poor overlap
        very_low = np.sum(propensity_scores < 0.05)
        very_high = np.sum(propensity_scores > 0.95)
        total = len(propensity_scores)

        if very_low > 0.05 * total:  # More than 5% of units have very low PS
            warnings.append(
                f"{very_low} units ({100 * very_low / total:.1f}%) have propensity scores < 0.05"
            )

        if very_high > 0.05 * total:  # More than 5% of units have very high PS
            warnings.append(
                f"{very_high} units ({100 * very_high / total:.1f}%) have propensity scores > 0.95"
            )

        return diagnostics

    def _truncate_weights(self, weights: NDArray[Any]) -> NDArray[Any]:
        """Apply weight truncation to handle extreme weights.

        Args:
            weights: Array of IPW weights

        Returns:
            Array of truncated weights
        """
        if self.weight_truncation is None:
            return weights

        if self.weight_truncation == "percentile":
            # Truncate at specified percentiles
            lower_percentile = self.truncation_threshold * 100
            upper_percentile = 100 - lower_percentile

            lower_bound = np.percentile(weights, lower_percentile)
            upper_bound = np.percentile(weights, upper_percentile)

            truncated_weights = np.clip(weights, lower_bound, upper_bound)

        elif self.weight_truncation == "threshold":
            # Truncate weights outside [threshold, 1/threshold]
            max_weight = 1 / self.truncation_threshold
            min_weight = self.truncation_threshold

            truncated_weights = np.clip(weights, min_weight, max_weight)

        else:
            raise ValueError(f"Unknown weight truncation method: {self.weight_truncation}")

        if self.verbose:
            n_truncated = np.sum(weights != truncated_weights)
            if n_truncated > 0:
                print(f"Truncated {n_truncated} extreme weights ({100 * n_truncated / len(weights):.1f}%)")

        return truncated_weights

    def _compute_weights(
        self,
        treatment: TreatmentData,
        propensity_scores: NDArray[Any],
    ) -> NDArray[Any]:
        """Compute IPW weights from propensity scores.

        Args:
            treatment: Treatment assignment data
            propensity_scores: Array of propensity scores

        Returns:
            Array of IPW weights
        """
        if isinstance(treatment.values, pd.Series):
            treatment_values = treatment.values.values
        else:
            treatment_values = treatment.values

        # Basic IPW weights: W_i = T_i / e_i + (1 - T_i) / (1 - e_i)
        weights = np.zeros_like(propensity_scores)

        # Weights for treated units
        treated_mask = treatment_values == 1
        weights[treated_mask] = 1 / propensity_scores[treated_mask]

        # Weights for control units
        control_mask = treatment_values == 0
        weights[control_mask] = 1 / (1 - propensity_scores[control_mask])

        # Apply stabilized weights if requested
        if self.stabilized_weights:
            # Stabilized weights multiply by marginal treatment probability
            treatment_prob = np.mean(treatment_values)

            # SW_i = P(T=1) * T_i / e_i + P(T=0) * (1 - T_i) / (1 - e_i)
            stabilized_weights = np.zeros_like(weights)
            stabilized_weights[treated_mask] = treatment_prob * weights[treated_mask]
            stabilized_weights[control_mask] = (1 - treatment_prob) * weights[control_mask]

            weights = stabilized_weights

        # Apply weight truncation
        weights = self._truncate_weights(weights)

        return weights

    def _compute_weight_diagnostics(self, weights: NDArray[Any]) -> dict[str, Any]:
        """Compute diagnostics for IPW weights.

        Args:
            weights: Array of IPW weights

        Returns:
            Dictionary with weight diagnostics
        """
        return {
            "mean_weight": float(np.mean(weights)),
            "median_weight": float(np.median(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "weight_std": float(np.std(weights)),
            "weight_variance": float(np.var(weights)),
            "effective_sample_size": float(np.sum(weights) ** 2 / np.sum(weights ** 2)),
            "weight_ratio": float(np.max(weights) / np.min(weights)),
            "extreme_weights_pct": float(100 * np.sum((weights > 10) | (weights < 0.1)) / len(weights)),
        }

    def _fit_implementation(
        self,
        treatment: TreatmentData,
        outcome: OutcomeData,
        covariates: CovariateData | None = None,
    ) -> None:
        """Fit the IPW estimator.

        Args:
            treatment: Treatment assignment data
            outcome: Outcome variable data
            covariates: Covariate data for propensity score estimation
        """
        # Fit propensity score model
        self._fit_propensity_model(treatment, covariates)

        # Estimate propensity scores
        self.propensity_scores = self._estimate_propensity_scores()

        # Check overlap assumption
        if self.check_overlap:
            self._overlap_diagnostics = self._check_overlap(self.propensity_scores)

            if not self._overlap_diagnostics["overlap_satisfied"]:
                if self.verbose:
                    print("Warning: Overlap assumption violations detected:")
                    for violation in self._overlap_diagnostics["violations"]:
                        print(f"  - {violation}")

                # Could raise AssumptionViolationError here if strict checking desired
                # For now, just warn

        # Compute IPW weights
        self.weights = self._compute_weights(treatment, self.propensity_scores)

        # Compute weight diagnostics
        self._weight_diagnostics = self._compute_weight_diagnostics(self.weights)

        if self.verbose:
            print(f"Mean weight: {self._weight_diagnostics['mean_weight']:.4f}")
            print(f"Weight range: [{self._weight_diagnostics['min_weight']:.4f}, {self._weight_diagnostics['max_weight']:.4f}]")
            print(f"Effective sample size: {self._weight_diagnostics['effective_sample_size']:.1f}")

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate Average Treatment Effect using IPW.

        Returns:
            CausalEffect object with ATE estimate and confidence intervals
        """
        if (
            self.weights is None
            or self.treatment_data is None
            or self.outcome_data is None
        ):
            raise EstimationError("Model must be fitted before estimation")

        # Get treatment and outcome values
        if isinstance(self.treatment_data.values, pd.Series):
            treatment_values = self.treatment_data.values.values
        else:
            treatment_values = self.treatment_data.values

        if isinstance(self.outcome_data.values, pd.Series):
            outcome_values = self.outcome_data.values.values
        else:
            outcome_values = self.outcome_data.values

        # Compute weighted means for treated and control groups
        treated_mask = treatment_values == 1
        control_mask = treatment_values == 0

        # Weighted mean outcomes
        weighted_outcome_treated = np.sum(
            outcome_values[treated_mask] * self.weights[treated_mask]
        ) / np.sum(self.weights[treated_mask])

        weighted_outcome_control = np.sum(
            outcome_values[control_mask] * self.weights[control_mask]
        ) / np.sum(self.weights[control_mask])

        # Average Treatment Effect
        ate = weighted_outcome_treated - weighted_outcome_control

        # Bootstrap confidence intervals
        ate_ci_lower, ate_ci_upper, bootstrap_estimates = (
            self._bootstrap_confidence_interval()
        )

        # Calculate standard error from bootstrap
        ate_se = (
            np.std(bootstrap_estimates) if bootstrap_estimates is not None else None
        )

        # Count treatment/control units
        n_treated = np.sum(treated_mask)
        n_control = np.sum(control_mask)

        # Compile diagnostics
        diagnostics = {}
        if self._overlap_diagnostics is not None:
            diagnostics["overlap"] = self._overlap_diagnostics
        if self._weight_diagnostics is not None:
            diagnostics["weights"] = self._weight_diagnostics

        return CausalEffect(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            confidence_level=self.confidence_level,
            potential_outcome_treated=weighted_outcome_treated,
            potential_outcome_control=weighted_outcome_control,
            method="IPW",
            n_observations=len(treatment_values),
            n_treated=n_treated,
            n_control=n_control,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_estimates=bootstrap_estimates,
            diagnostics=diagnostics,
        )

    def _bootstrap_confidence_interval(
        self,
    ) -> tuple[float | None, float | None, NDArray[Any] | None]:
        """Calculate bootstrap confidence intervals for ATE.

        Returns:
            Tuple of (lower_ci, upper_ci, bootstrap_estimates)
        """
        if self.bootstrap_samples <= 0:
            return None, None, None

        if (
            self.treatment_data is None
            or self.outcome_data is None
            or self.covariate_data is None
        ):
            raise EstimationError("Data must be available for bootstrap")

        bootstrap_ates: list[float] = []
        n_obs = len(self.treatment_data.values)

        for _ in range(self.bootstrap_samples):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)

            # Create bootstrap datasets
            boot_treatment = TreatmentData(
                values=self.treatment_data.values.iloc[bootstrap_indices]
                if isinstance(self.treatment_data.values, pd.Series)
                else self.treatment_data.values[bootstrap_indices],
                name=self.treatment_data.name,
                treatment_type=self.treatment_data.treatment_type,
                categories=self.treatment_data.categories,
            )

            boot_outcome = OutcomeData(
                values=self.outcome_data.values.iloc[bootstrap_indices]
                if isinstance(self.outcome_data.values, pd.Series)
                else self.outcome_data.values[bootstrap_indices],
                name=self.outcome_data.name,
                outcome_type=self.outcome_data.outcome_type,
            )

            if isinstance(self.covariate_data.values, pd.DataFrame):
                boot_cov_values = self.covariate_data.values.iloc[bootstrap_indices]
            else:
                boot_cov_values = self.covariate_data.values[bootstrap_indices]

            boot_covariates = CovariateData(
                values=boot_cov_values,
                names=self.covariate_data.names,
            )

            # Fit model on bootstrap sample
            try:
                boot_estimator = IPWEstimator(
                    propensity_model_type=self.propensity_model_type,
                    propensity_model_params=self.propensity_model_params,
                    weight_truncation=self.weight_truncation,
                    truncation_threshold=self.truncation_threshold,
                    stabilized_weights=self.stabilized_weights,
                    bootstrap_samples=0,  # Don't bootstrap within bootstrap
                    check_overlap=False,  # Skip overlap checks in bootstrap
                    random_state=None,  # Use different random state for each bootstrap
                    verbose=False,
                )

                boot_estimator.fit(boot_treatment, boot_outcome, boot_covariates)
                boot_effect = boot_estimator.estimate_ate(use_cache=False)
                bootstrap_ates.append(boot_effect.ate)

            except Exception:
                # Skip failed bootstrap samples
                continue

        if len(bootstrap_ates) == 0:
            return None, None, None

        bootstrap_ates_array = np.array(bootstrap_ates)

        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        ate_ci_lower = float(np.percentile(bootstrap_ates_array, lower_percentile))
        ate_ci_upper = float(np.percentile(bootstrap_ates_array, upper_percentile))

        return ate_ci_lower, ate_ci_upper, bootstrap_ates_array

    def get_propensity_scores(self) -> NDArray[Any] | None:
        """Get the estimated propensity scores.

        Returns:
            Array of propensity scores if fitted, None otherwise
        """
        return self.propensity_scores

    def get_weights(self) -> NDArray[Any] | None:
        """Get the computed IPW weights.

        Returns:
            Array of IPW weights if fitted, None otherwise
        """
        return self.weights

    def get_overlap_diagnostics(self) -> dict[str, Any] | None:
        """Get overlap assumption diagnostics.

        Returns:
            Dictionary with overlap diagnostics if fitted, None otherwise
        """
        return self._overlap_diagnostics

    def get_weight_diagnostics(self) -> dict[str, Any] | None:
        """Get weight distribution diagnostics.

        Returns:
            Dictionary with weight diagnostics if fitted, None otherwise
        """
        return self._weight_diagnostics

    def predict_propensity_scores(
        self, covariates: pd.DataFrame | NDArray[Any]
    ) -> NDArray[Any]:
        """Predict propensity scores for new covariate data.

        Args:
            covariates: New covariate data for prediction

        Returns:
            Array of predicted propensity scores

        Raises:
            EstimationError: If estimator is not fitted
        """
        if not self.is_fitted or self.propensity_model is None:
            raise EstimationError("Estimator must be fitted before prediction")

        # Prepare covariate data
        if isinstance(covariates, pd.DataFrame):
            cov_data = CovariateData(
                values=covariates, names=list(covariates.columns)
            )
        else:
            # Use the same covariate names as the training data
            if self.covariate_data is not None:
                cov_names = self.covariate_data.names
            else:
                cov_names = [f"X{i}" for i in range(covariates.shape[1])]

            cov_df = pd.DataFrame(covariates, columns=cov_names)
            cov_data = CovariateData(values=cov_df, names=cov_names)

        # Prepare features and predict
        X = self._prepare_propensity_features(cov_data)

        # Ensure features are in the same order as training
        if self._propensity_features is not None:
            X = X[self._propensity_features]

        if hasattr(self.propensity_model, "predict_proba"):
            return self.propensity_model.predict_proba(X)[:, 1]
        else:
            # Fallback for models without predict_proba
            scores = self.propensity_model.decision_function(X)
            return 1 / (1 + np.exp(-scores))
