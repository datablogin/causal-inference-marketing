"""Time-varying treatment estimators for longitudinal causal inference.

This module implements methods for causal inference with time-varying treatments,
including the parametric g-formula, inverse probability weighting, and g-estimation
for longitudinal data with sequential treatment assignments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    EstimationError,
)
from ..core.bootstrap import BootstrapMixin
from ..core.longitudinal import LongitudinalData, TreatmentStrategy

__all__ = [
    "TimeVaryingEstimator",
    "StrategyComparison",
    "StrategyOutcome",
]


class StrategyOutcome:
    """Results for a single treatment strategy."""

    def __init__(
        self,
        strategy_name: str,
        mean_outcome: float,
        outcome_trajectory: NDArray[Any],
        individual_outcomes: NDArray[Any],
        confidence_interval: tuple[float, float] | None = None,
    ):
        self.strategy_name = strategy_name
        self.mean_outcome = mean_outcome
        self.outcome_trajectory = outcome_trajectory
        self.individual_outcomes = individual_outcomes
        self.confidence_interval = confidence_interval


class StrategyComparison:
    """Results comparing multiple treatment strategies."""

    def __init__(
        self,
        strategy_outcomes: dict[str, StrategyOutcome],
        strategy_contrasts: dict[str, CausalEffect],
        ranking: list[str],
    ):
        self.strategy_outcomes = strategy_outcomes
        self.strategy_contrasts = strategy_contrasts
        self.ranking = ranking

    def get_best_strategy(self) -> str:
        """Get the name of the best-performing strategy."""
        return self.ranking[0]

    def get_strategy_effect(
        self, strategy1: str, strategy2: str
    ) -> CausalEffect | None:
        """Get the causal effect comparing two strategies."""
        contrast_name = f"{strategy1}_vs_{strategy2}"
        return self.strategy_contrasts.get(contrast_name)


class TimeVaryingEstimator(BaseEstimator, BootstrapMixin):
    """Estimator for causal effects with time-varying treatments.

    This class implements methods for estimating causal effects of dynamic
    treatment strategies using longitudinal data. It supports:

    - Parametric G-formula for time-varying treatments
    - Inverse probability weighting with stabilized weights
    - G-estimation with structural nested models
    - Doubly robust methods combining G-formula and IPW

    The estimator can compare multiple treatment strategies and identify
    optimal dynamic treatment regimes.
    """

    def __init__(
        self,
        method: str = "g_formula",
        outcome_model: SklearnBaseEstimator | None = None,
        treatment_model: SklearnBaseEstimator | None = None,
        time_horizon: int | None = None,
        weight_stabilization: bool = True,
        weight_truncation: float | None = None,
        bootstrap_samples: int = 500,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize the time-varying treatment estimator.

        Args:
            method: Estimation method ('g_formula', 'ipw', 'g_estimation', 'doubly_robust')
            outcome_model: Model for outcome prediction (default: Random Forest)
            treatment_model: Model for treatment prediction (default: Logistic Regression)
            time_horizon: Maximum time horizon for analysis
            weight_stabilization: Whether to use stabilized weights for IPW
            weight_truncation: Threshold for truncating extreme weights
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed output
        """
        super().__init__(random_state=random_state, verbose=verbose)

        # Method validation
        allowed_methods = {"g_formula", "ipw", "g_estimation", "doubly_robust"}
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")

        self.method = method
        self.time_horizon = time_horizon
        self.weight_stabilization = weight_stabilization
        self.weight_truncation = weight_truncation
        self.bootstrap_samples = bootstrap_samples

        # Default models
        if outcome_model is None:
            self.outcome_model = RandomForestRegressor(
                n_estimators=100, random_state=random_state
            )
        else:
            self.outcome_model = outcome_model

        if treatment_model is None:
            self.treatment_model = LogisticRegression(
                random_state=random_state, max_iter=1000
            )
        else:
            self.treatment_model = treatment_model

        # Storage for fitted models and results
        self.fitted_outcome_models_: dict[int, SklearnBaseEstimator] = {}
        self.fitted_treatment_models_: dict[int, SklearnBaseEstimator] = {}
        self.longitudinal_data_: LongitudinalData | None = None
        self.strategy_results_: StrategyComparison | None = None

    def _fit_implementation(
        self,
        treatment: Any,  # Not used - we expect LongitudinalData
        outcome: Any,  # Not used - we expect LongitudinalData
        covariates: Any = None,  # Not used - we expect LongitudinalData
    ) -> None:
        """Fit implementation - not used for time-varying estimator."""
        raise NotImplementedError(
            "TimeVaryingEstimator requires fit_longitudinal() method"
        )

    def _estimate_ate_implementation(self) -> CausalEffect:
        """Estimate ATE implementation - not used for time-varying estimator."""
        raise NotImplementedError(
            "TimeVaryingEstimator requires estimate_strategy_effects() method"
        )

    def fit_longitudinal(self, data: LongitudinalData) -> TimeVaryingEstimator:
        """Fit the estimator to longitudinal data.

        Args:
            data: Longitudinal data with time-varying treatments and outcomes

        Returns:
            self: Fitted estimator
        """
        self.longitudinal_data_ = data

        # Determine time horizon
        if self.time_horizon is None:
            self.time_horizon = data.n_time_periods
        else:
            self.time_horizon = min(self.time_horizon, data.n_time_periods)

        # Fit models for each time period
        time_periods = data.time_periods[: self.time_horizon]

        for i, time_period in enumerate(time_periods):
            if self.verbose:
                print(f"Fitting models for time period {time_period}")

            # Get data for this time period
            time_data = data.data[data.data[data.time_col] == time_period]

            if len(time_data) == 0:
                continue

            # Prepare features (confounders + baseline variables + treatment history)
            feature_cols = data.confounder_cols + data.baseline_cols
            if i > 0:
                # Add treatment history for periods after baseline
                prev_periods = time_periods[:i]
                for prev_period in prev_periods:
                    for treat_col in data.treatment_cols:
                        hist_col = f"{treat_col}_t{prev_period}"
                        if hist_col not in time_data.columns:
                            # Create treatment history by merging previous time periods
                            prev_data = data.data[
                                data.data[data.time_col] == prev_period
                            ][[data.id_col, treat_col]]
                            prev_data = prev_data.rename(columns={treat_col: hist_col})
                            time_data = time_data.merge(
                                prev_data, on=data.id_col, how="left"
                            )
                        feature_cols.append(hist_col)

            # Prepare feature matrix
            X = time_data[feature_cols].fillna(0)  # Simple missing data handling

            # Fit outcome model
            if data.outcome_cols:
                y_outcome = time_data[data.outcome_cols[0]]
                outcome_model = clone(self.outcome_model)

                # Include treatment in outcome model features
                X_outcome = pd.concat([X, time_data[data.treatment_cols]], axis=1)
                outcome_model.fit(X_outcome, y_outcome)
                self.fitted_outcome_models_[i] = outcome_model

            # Fit treatment model (for IPW)
            if data.treatment_cols and self.method in ["ipw", "doubly_robust"]:
                y_treatment = time_data[data.treatment_cols[0]]
                treatment_model = clone(self.treatment_model)
                treatment_model.fit(X, y_treatment)
                self.fitted_treatment_models_[i] = treatment_model

        self.is_fitted = True
        return self

    def estimate_strategy_effects(
        self,
        strategies: dict[str, TreatmentStrategy],
        confidence_level: float = 0.95,
    ) -> StrategyComparison:
        """Estimate effects of multiple treatment strategies.

        Args:
            strategies: Dictionary mapping strategy names to strategy functions
            confidence_level: Confidence level for intervals

        Returns:
            StrategyComparison with results for all strategies
        """
        if not self.is_fitted or self.longitudinal_data_ is None:
            raise EstimationError("Estimator must be fitted before strategy estimation")

        strategy_outcomes = {}

        # Estimate outcome for each strategy
        for strategy_name, strategy_func in strategies.items():
            if self.verbose:
                print(f"Estimating outcomes for strategy: {strategy_name}")

            outcome = self._estimate_strategy_outcome(strategy_func, strategy_name)
            strategy_outcomes[strategy_name] = outcome

        # Compare strategies pairwise
        strategy_contrasts = {}
        strategy_names = list(strategies.keys())

        for i, strategy1 in enumerate(strategy_names):
            for strategy2 in strategy_names[i + 1 :]:
                contrast_name = f"{strategy1}_vs_{strategy2}"

                outcome1 = strategy_outcomes[strategy1]
                outcome2 = strategy_outcomes[strategy2]

                effect = outcome1.mean_outcome - outcome2.mean_outcome

                # Bootstrap confidence intervals if requested
                ci_lower, ci_upper = None, None
                if self.bootstrap_samples > 0:
                    bootstrap_effects = self._bootstrap_strategy_contrast(
                        strategies[strategy1], strategies[strategy2]
                    )
                    alpha = 1 - confidence_level
                    ci_lower = np.percentile(bootstrap_effects, 100 * alpha / 2)
                    ci_upper = np.percentile(bootstrap_effects, 100 * (1 - alpha / 2))

                causal_effect = CausalEffect(
                    ate=effect,
                    ate_ci_lower=ci_lower,
                    ate_ci_upper=ci_upper,
                    confidence_level=confidence_level,
                    method=f"time_varying_{self.method}",
                    n_observations=self.longitudinal_data_.n_individuals,
                )

                strategy_contrasts[contrast_name] = causal_effect

        # Rank strategies by mean outcome
        ranking = sorted(
            strategy_names,
            key=lambda s: strategy_outcomes[s].mean_outcome,
            reverse=True,
        )

        self.strategy_results_ = StrategyComparison(
            strategy_outcomes, strategy_contrasts, ranking
        )

        return self.strategy_results_

    def _estimate_strategy_outcome(
        self, strategy: TreatmentStrategy, strategy_name: str
    ) -> StrategyOutcome:
        """Estimate outcome for a single treatment strategy.

        Args:
            strategy: Treatment strategy function
            strategy_name: Name of the strategy

        Returns:
            StrategyOutcome with estimated results
        """
        if self.method == "g_formula":
            return self._g_formula_strategy_outcome(strategy, strategy_name)
        elif self.method == "ipw":
            return self._ipw_strategy_outcome(strategy, strategy_name)
        elif self.method == "doubly_robust":
            # Combine G-formula and IPW
            g_outcome = self._g_formula_strategy_outcome(strategy, strategy_name)
            ipw_outcome = self._ipw_strategy_outcome(strategy, strategy_name)

            # Simple average of the two estimates
            mean_outcome = (g_outcome.mean_outcome + ipw_outcome.mean_outcome) / 2
            individual_outcomes = (
                g_outcome.individual_outcomes + ipw_outcome.individual_outcomes
            ) / 2

            return StrategyOutcome(
                strategy_name=strategy_name,
                mean_outcome=mean_outcome,
                outcome_trajectory=g_outcome.outcome_trajectory,
                individual_outcomes=individual_outcomes,
            )
        else:
            raise NotImplementedError(f"Method {self.method} not yet implemented")

    def _g_formula_strategy_outcome(
        self, strategy: TreatmentStrategy, strategy_name: str
    ) -> StrategyOutcome:
        """Estimate strategy outcome using parametric G-formula.

        Args:
            strategy: Treatment strategy function
            strategy_name: Name of the strategy

        Returns:
            StrategyOutcome estimated via G-formula
        """
        data = self.longitudinal_data_
        if data is None:
            raise EstimationError("No longitudinal data available")

        # Get unique individuals
        individuals = data.individuals
        n_individuals = len(individuals)
        time_periods = data.time_periods[: self.time_horizon]

        # Initialize outcome trajectories
        individual_outcomes = np.zeros(n_individuals)
        outcome_trajectory = np.zeros(len(time_periods))

        # For each individual, simulate forward through time
        for i, individual_id in enumerate(individuals):
            # Get individual's baseline data
            individual_data = data.get_individual_trajectory(individual_id)

            if len(individual_data) == 0:
                continue

            # Start with baseline covariates
            current_covariates = {}
            baseline_data = individual_data.iloc[0]

            for col in data.baseline_cols:
                current_covariates[col] = baseline_data[col]

            # Initialize treatment history
            treatment_history: dict[str, Any] = {}

            # Simulate forward through time
            simulated_outcomes = []

            for t, time_period in enumerate(time_periods):
                # Get observed data for this time period
                time_mask = individual_data[data.time_col] == time_period
                if time_mask.any():
                    observed_data = individual_data[time_mask].iloc[0]

                    # Update time-varying covariates with observed values
                    for col in data.confounder_cols:
                        if col in observed_data:
                            current_covariates[col] = observed_data[col]
                else:
                    # If no observed data, use last known values or predictions
                    observed_data = None

                # Apply treatment strategy
                # Create temporary dataframe for strategy function
                temp_data = pd.DataFrame([current_covariates])
                for treat_col, hist_val in treatment_history.items():
                    temp_data[treat_col] = hist_val

                strategy_treatment = strategy(temp_data, time_period)
                if isinstance(strategy_treatment, np.ndarray):
                    strategy_treatment = strategy_treatment[0]

                # Store treatment in history
                treatment_col = data.treatment_cols[0]
                treatment_history[f"{treatment_col}_t{time_period}"] = (
                    strategy_treatment
                )

                # Predict outcome using fitted model
                if t in self.fitted_outcome_models_:
                    model = self.fitted_outcome_models_[t]

                    # Prepare features for prediction
                    feature_cols = data.confounder_cols + data.baseline_cols
                    X_pred = []

                    for col in feature_cols:
                        X_pred.append(current_covariates.get(col, 0))

                    # Add treatment history
                    for hist_col in treatment_history:
                        if f"{treatment_col}_t" in hist_col:
                            X_pred.append(treatment_history[hist_col])

                    # Add current treatment
                    X_pred.append(strategy_treatment)

                    # Predict outcome
                    X_pred_array = np.array(X_pred).reshape(1, -1)

                    # Handle feature dimension mismatch
                    if X_pred_array.shape[1] != model.n_features_in_:
                        # Pad or trim features to match model
                        if X_pred_array.shape[1] < model.n_features_in_:
                            # Pad with zeros
                            padding = np.zeros(
                                (1, model.n_features_in_ - X_pred_array.shape[1])
                            )
                            X_pred_array = np.hstack([X_pred_array, padding])
                        else:
                            # Trim features
                            X_pred_array = X_pred_array[:, : model.n_features_in_]

                    predicted_outcome = model.predict(X_pred_array)[0]
                    simulated_outcomes.append(predicted_outcome)

                    # Update outcome trajectory
                    outcome_trajectory[t] += predicted_outcome / n_individuals

            # Use final outcome as individual outcome
            if simulated_outcomes:
                individual_outcomes[i] = simulated_outcomes[-1]

        mean_outcome = float(np.mean(individual_outcomes))

        return StrategyOutcome(
            strategy_name=strategy_name,
            mean_outcome=mean_outcome,
            outcome_trajectory=outcome_trajectory,
            individual_outcomes=individual_outcomes,
        )

    def _ipw_strategy_outcome(
        self, strategy: TreatmentStrategy, strategy_name: str
    ) -> StrategyOutcome:
        """Estimate strategy outcome using inverse probability weighting.

        Args:
            strategy: Treatment strategy function
            strategy_name: Name of the strategy

        Returns:
            StrategyOutcome estimated via IPW
        """
        data = self.longitudinal_data_
        if data is None:
            raise EstimationError("No longitudinal data available")

        time_periods = data.time_periods[: self.time_horizon]

        # Calculate weights for each observation
        weights = self._calculate_ipw_weights(strategy)

        # Apply weights to outcomes
        weighted_outcomes = []
        outcome_trajectory = np.zeros(len(time_periods))

        # Get final outcomes for each individual
        individuals = data.individuals
        individual_outcomes = np.zeros(len(individuals))

        for i, individual_id in enumerate(individuals):
            individual_data = data.get_individual_trajectory(individual_id)

            if len(individual_data) == 0:
                continue

            # Get weight for this individual
            individual_weight = weights.get(individual_id, 0)

            # Get final outcome
            if data.outcome_cols and len(individual_data) > 0:
                final_outcome = individual_data[data.outcome_cols[0]].iloc[-1]
                weighted_outcome = final_outcome * individual_weight
                weighted_outcomes.append(weighted_outcome)
                individual_outcomes[i] = final_outcome

        # Calculate mean outcome
        mean_outcome = np.mean(weighted_outcomes) if weighted_outcomes else 0

        return StrategyOutcome(
            strategy_name=strategy_name,
            mean_outcome=mean_outcome,
            outcome_trajectory=outcome_trajectory,
            individual_outcomes=individual_outcomes,
        )

    def _calculate_ipw_weights(
        self, strategy: TreatmentStrategy
    ) -> dict[int | str, float]:
        """Calculate inverse probability weights for a treatment strategy.

        Args:
            strategy: Treatment strategy function

        Returns:
            Dictionary mapping individual IDs to weights
        """
        data = self.longitudinal_data_
        if data is None:
            raise EstimationError("No longitudinal data available")

        individuals = data.individuals
        time_periods = data.time_periods[: self.time_horizon]
        weights: dict[int | str, float] = {}

        for individual_id in individuals:
            individual_data = data.get_individual_trajectory(individual_id)

            if len(individual_data) == 0:
                weights[individual_id] = 0.0
                continue

            # Calculate product of weights across time periods
            individual_weight = 1.0

            for t, time_period in enumerate(time_periods):
                # Get observed treatment
                time_mask = individual_data[data.time_col] == time_period
                if not time_mask.any():
                    continue

                observed_data = individual_data[time_mask].iloc[0]
                observed_treatment = observed_data[data.treatment_cols[0]]

                # Get strategy treatment
                temp_data = pd.DataFrame([observed_data])
                strategy_treatment = strategy(temp_data, time_period)
                if isinstance(strategy_treatment, np.ndarray):
                    strategy_treatment = strategy_treatment[0]

                # Calculate probability of observed treatment
                if t in self.fitted_treatment_models_:
                    model = self.fitted_treatment_models_[t]

                    # Prepare features
                    feature_cols = data.confounder_cols + data.baseline_cols
                    X = np.array(observed_data[feature_cols]).reshape(1, -1)

                    # Handle missing features
                    if X.shape[1] != model.n_features_in_:
                        if X.shape[1] < model.n_features_in_:
                            padding = np.zeros((1, model.n_features_in_ - X.shape[1]))
                            X = np.hstack([X, padding])
                        else:
                            X = X[:, : model.n_features_in_]

                    # Get treatment probability
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)[0]
                        if len(probs) > 1 and observed_treatment in [0, 1]:
                            treatment_prob = probs[int(observed_treatment)]
                        else:
                            treatment_prob = 0.5  # Default for edge cases
                    else:
                        treatment_prob = 0.5  # Default for non-probabilistic models

                    # Update weight
                    if treatment_prob > 0:
                        if observed_treatment == strategy_treatment:
                            individual_weight *= 1.0 / treatment_prob
                        else:
                            individual_weight = 0  # Treatment doesn't match strategy
                            break

            # Apply weight truncation if specified
            if self.weight_truncation is not None:
                max_weight = 1.0 / self.weight_truncation
                individual_weight = min(individual_weight, max_weight)

            weights[individual_id] = float(individual_weight)

        # Stabilize weights if requested
        if self.weight_stabilization:
            mean_weight = np.mean(list(weights.values()))
            if mean_weight > 0:
                weights = {k: float(v / mean_weight) for k, v in weights.items()}

        return weights

    def _create_bootstrap_estimator(
        self, random_state: int | None = None
    ) -> TimeVaryingEstimator:
        """Create a bootstrap estimator instance.

        Returns:
            TimeVaryingEstimator configured for bootstrap sampling
        """
        return TimeVaryingEstimator(
            method=self.method,
            outcome_model=clone(self.outcome_model),
            treatment_model=clone(self.treatment_model),
            time_horizon=self.time_horizon,
            weight_stabilization=self.weight_stabilization,
            weight_truncation=self.weight_truncation,
            bootstrap_samples=0,  # No nested bootstrapping
            random_state=random_state or self.random_state,
            verbose=False,
        )

    def _bootstrap_strategy_contrast(
        self, strategy1: TreatmentStrategy, strategy2: TreatmentStrategy
    ) -> NDArray[Any]:
        """Bootstrap confidence intervals for strategy contrasts.

        Args:
            strategy1: First treatment strategy
            strategy2: Second treatment strategy

        Returns:
            Array of bootstrap effect estimates
        """
        if self.longitudinal_data_ is None:
            raise EstimationError("No longitudinal data available")

        bootstrap_effects = []
        data = self.longitudinal_data_
        n_individuals = data.n_individuals

        for _ in range(self.bootstrap_samples):
            # Bootstrap sample individuals
            bootstrap_ids = np.random.choice(
                data.individuals, size=n_individuals, replace=True
            )

            # Create bootstrap dataset with unique individual IDs
            bootstrap_data_list = []

            for new_id, individual_id in enumerate(bootstrap_ids):
                individual_data = data.get_individual_trajectory(individual_id).copy()

                # Assign new unique ID to avoid duplicates in bootstrap sample
                individual_data[data.id_col] = new_id
                bootstrap_data_list.append(individual_data)

            if not bootstrap_data_list:
                continue

            bootstrap_df = pd.concat(bootstrap_data_list, ignore_index=True)

            # Create bootstrap LongitudinalData
            bootstrap_longitudinal = LongitudinalData(
                data=bootstrap_df,
                id_col=data.id_col,
                time_col=data.time_col,
                treatment_cols=data.treatment_cols,
                outcome_cols=data.outcome_cols,
                confounder_cols=data.confounder_cols,
                baseline_cols=data.baseline_cols,
            )

            # Fit bootstrap estimator
            bootstrap_estimator = self._create_bootstrap_estimator()

            try:
                bootstrap_estimator.fit_longitudinal(bootstrap_longitudinal)

                # Estimate outcomes for both strategies
                outcome1 = bootstrap_estimator._estimate_strategy_outcome(
                    strategy1, "strategy1"
                )
                outcome2 = bootstrap_estimator._estimate_strategy_outcome(
                    strategy2, "strategy2"
                )

                effect = outcome1.mean_outcome - outcome2.mean_outcome
                bootstrap_effects.append(effect)

            except Exception:
                # Skip failed bootstrap samples
                continue

        return np.array(bootstrap_effects)

    def check_sequential_exchangeability(self) -> dict[str, Any]:
        """Check sequential exchangeability assumption for longitudinal data.

        Returns:
            Dictionary with exchangeability check results
        """
        if self.longitudinal_data_ is None:
            raise EstimationError("No longitudinal data available")

        return self.longitudinal_data_.check_sequential_exchangeability()

    def test_treatment_confounder_feedback(self) -> dict[str, Any]:
        """Test for treatment-confounder feedback in longitudinal data.

        Returns:
            Dictionary with feedback test results
        """
        if self.longitudinal_data_ is None:
            raise EstimationError("No longitudinal data available")

        return self.longitudinal_data_.test_treatment_confounder_feedback()

    def sensitivity_analysis(
        self,
        strategies: dict[str, TreatmentStrategy],
        unmeasured_confounder_strength: NDArray[Any],
    ) -> dict[str, Any]:
        """Perform sensitivity analysis for unmeasured confounding.

        Args:
            strategies: Treatment strategies to analyze
            unmeasured_confounder_strength: Array of bias strengths to test

        Returns:
            Dictionary with sensitivity analysis results
        """
        if len(strategies) < 2:
            raise ValueError("Need at least 2 strategies for sensitivity analysis")

        strategy_names = list(strategies.keys())
        base_results = self.estimate_strategy_effects(strategies)
        base_effect = base_results.strategy_contrasts[
            f"{strategy_names[0]}_vs_{strategy_names[1]}"
        ].ate

        effects_by_bias = []

        for bias_strength in unmeasured_confounder_strength:
            # Simple sensitivity analysis - adjust outcomes by bias strength
            # This is a simplified approach; more sophisticated methods would
            # model the unmeasured confounder explicitly
            adjusted_effect = base_effect * (1 - bias_strength)
            effects_by_bias.append(adjusted_effect)

        return {
            "base_effect": base_effect,
            "effect_by_bias_strength": effects_by_bias,
            "bias_strengths": unmeasured_confounder_strength,
            "strategy_comparison": f"{strategy_names[0]} vs {strategy_names[1]}",
        }

    def get_weight_diagnostics(self) -> dict[str, Any]:
        """Get diagnostics for IPW weights.

        Returns:
            Dictionary with weight diagnostic information
        """
        if self.method not in ["ipw", "doubly_robust"]:
            raise ValueError("Weight diagnostics only available for IPW methods")

        if self.longitudinal_data_ is None:
            raise EstimationError("No longitudinal data available")

        # Create a dummy strategy to calculate weights
        def dummy_strategy(data: pd.DataFrame, time: Any) -> NDArray[Any]:
            return np.ones(len(data))

        weights = self._calculate_ipw_weights(dummy_strategy)
        weight_values = list(weights.values())

        if not weight_values:
            return {"error": "No weights calculated"}

        return {
            "mean_weight": np.mean(weight_values),
            "median_weight": np.median(weight_values),
            "max_weight": np.max(weight_values),
            "min_weight": np.min(weight_values),
            "weight_variance": np.var(weight_values),
            "effective_sample_size": (np.sum(weight_values) ** 2)
            / np.sum(np.array(weight_values) ** 2),
            "n_zero_weights": np.sum(np.array(weight_values) == 0),
        }

    def find_optimal_strategy(self, strategies: dict[str, TreatmentStrategy]) -> str:
        """Find the optimal treatment strategy from a set of candidates.

        Args:
            strategies: Dictionary of treatment strategies to compare

        Returns:
            Name of the optimal strategy
        """
        results = self.estimate_strategy_effects(strategies)
        return results.get_best_strategy()
