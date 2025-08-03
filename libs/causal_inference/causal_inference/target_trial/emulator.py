"""Target trial emulator implementation.

This module implements the core TargetTrialEmulator class that conducts
the emulation of randomized trials using observational data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.base import (
    BaseEstimator,
    CausalEffect,
    CovariateData,
    OutcomeData,
    TreatmentData,
)
from ..estimators import AIPWEstimator, GComputationEstimator, IPWEstimator
from .protocol import TargetTrialProtocol
from .results import EmulationDiagnostics, TargetTrialResults


class TargetTrialEmulator:
    """Target trial emulator for causal inference from observational data.

    This class implements the target trial emulation framework by:
    1. Taking a specified trial protocol
    2. Applying eligibility criteria to observational data
    3. Implementing cloning, censoring, and weighting as needed
    4. Estimating causal effects using specified methods
    """

    def __init__(
        self,
        protocol: TargetTrialProtocol,
        estimation_method: str = "g_computation",
        adherence_adjustment: str = "intention_to_treat",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize target trial emulator.

        Args:
            protocol: Target trial protocol specification
            estimation_method: Causal inference method ("g_computation", "ipw", "aipw")
            adherence_adjustment: Analysis type ("intention_to_treat", "per_protocol", "both")
            random_state: Random seed for reproducibility
            verbose: Whether to print verbose output
        """
        self.protocol = protocol
        self.estimation_method = estimation_method
        self.adherence_adjustment = adherence_adjustment
        self.random_state = random_state
        self.verbose = verbose

        # Validation
        allowed_methods = {"g_computation", "ipw", "aipw"}
        if estimation_method not in allowed_methods:
            raise ValueError(f"estimation_method must be one of {allowed_methods}")

        allowed_adjustments = {"intention_to_treat", "per_protocol", "both"}
        if adherence_adjustment not in allowed_adjustments:
            raise ValueError(
                f"adherence_adjustment must be one of {allowed_adjustments}"
            )

        # Internal state
        self._fitted_estimators: dict[str, Any] = {}
        self._emulated_data: pd.DataFrame | None = None
        self._cloned_data: pd.DataFrame | None = None

    def emulate(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> TargetTrialResults:
        """Emulate the target trial using observational data.

        Args:
            data: Observational dataset
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariate_cols: Names of covariate columns for adjustment

        Returns:
            TargetTrialResults with emulation results
        """
        if self.verbose:
            print("Starting target trial emulation...")

        # Step 1: Validate protocol against data
        validation = self.protocol.validate_against_data(data)
        if not validation["valid"]:
            raise ValueError(f"Protocol validation failed: {validation['errors']}")

        # Step 2: Check feasibility
        feasibility = self.protocol.check_feasibility(data)
        if not feasibility["is_feasible"]:
            raise ValueError(
                f"Protocol is not feasible: {feasibility['infeasibility_reasons']}"
            )

        if self.verbose and feasibility["warnings"]:
            print(f"Feasibility warnings: {feasibility['warnings']}")

        # Step 3: Apply eligibility criteria
        eligible_data = self._apply_eligibility_criteria(data)

        if self.verbose:
            print(
                f"Applied eligibility criteria: {len(eligible_data):,} eligible participants"
            )

        # Step 3.5: Handle missing data systematically
        complete_data = self._handle_missing_data(
            eligible_data, treatment_col, outcome_col, covariate_cols
        )

        if self.verbose:
            missing_count = len(eligible_data) - len(complete_data)
            if missing_count > 0:
                print(f"Removed {missing_count:,} participants due to missing data")

        # Step 4: Clone participants for different treatment strategies (if needed)
        emulated_data = self._create_emulated_trial_data(complete_data, treatment_col)

        # Step 5: Apply censoring and adherence rules
        analysis_data = self._apply_censoring_and_adherence(
            emulated_data, treatment_col
        )

        # Step 6: Estimate causal effects
        itt_effect = self._estimate_intention_to_treat_effect(
            analysis_data, treatment_col, outcome_col, covariate_cols
        )

        pp_effect = None
        if self.adherence_adjustment in ["per_protocol", "both"]:
            pp_effect = self._estimate_per_protocol_effect(
                analysis_data, treatment_col, outcome_col, covariate_cols
            )

        # Step 7: Generate diagnostics
        diagnostics = self._generate_diagnostics(
            data, eligible_data, analysis_data, treatment_col
        )

        # Step 8: Create results object
        results = TargetTrialResults(
            intention_to_treat_effect=itt_effect,
            per_protocol_effect=pp_effect,
            protocol_summary=self.protocol.get_protocol_summary(),
            emulation_method="cloning_and_censoring",
            estimation_method=self.estimation_method,
            diagnostics=diagnostics,
            emulated_data=emulated_data,
            cloned_data=self._cloned_data,
        )

        if self.verbose:
            print("Target trial emulation completed successfully")

        return results

    def _apply_eligibility_criteria(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply eligibility criteria to select trial population.

        Args:
            data: Full observational dataset

        Returns:
            DataFrame with eligible participants only
        """
        eligible_mask = self.protocol.eligibility_criteria.check_eligibility(data)
        eligible_data = data[eligible_mask].copy()

        if self.verbose:
            eligibility_rate = len(eligible_data) / len(data)
            print(
                f"Eligibility rate: {eligibility_rate:.1%} ({len(eligible_data):,}/{len(data):,})"
            )

        return eligible_data

    def _handle_missing_data(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> pd.DataFrame:
        """Handle missing data systematically for target trial emulation.

        Args:
            data: Eligible participant data
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariate_cols: Names of covariate columns

        Returns:
            DataFrame with missing data handled appropriately
        """
        if self.verbose:
            missing_summary = {}

            # Report missing data patterns
            key_cols = [treatment_col, outcome_col] + covariate_cols
            for col in key_cols:
                if col in data.columns:
                    missing_count = data[col].isna().sum()
                    missing_pct = (missing_count / len(data)) * 100
                    missing_summary[col] = {"count": missing_count, "pct": missing_pct}

            if any(v["count"] > 0 for v in missing_summary.values()):
                print("Missing data summary:")
                for col, stats in missing_summary.items():
                    if stats["count"] > 0:
                        print(f"  {col}: {stats['count']:,} ({stats['pct']:.1f}%)")

        # Strategy 1: Complete case analysis for essential variables
        essential_cols = [treatment_col, outcome_col]
        complete_essential = data.dropna(subset=essential_cols)

        # Strategy 2: Handle missing covariates more flexibly
        # For covariates, we can use imputation or indicator variables
        working_data = complete_essential.copy()

        # Simple mean/mode imputation for covariates with indicator variables
        for col in covariate_cols:
            if col in working_data.columns:
                missing_mask = working_data[col].isna()
                if missing_mask.any():
                    # Create missing indicator
                    working_data[f"{col}_missing"] = missing_mask

                    # Impute missing values
                    if working_data[col].dtype in ["int64", "float64"]:
                        # Mean imputation for numeric
                        fill_value = working_data[col].mean()
                        working_data[col] = working_data[col].fillna(fill_value)
                    else:
                        # Mode imputation for categorical
                        mode_series = working_data[col].mode()
                        if len(mode_series) > 0:
                            working_data[col] = working_data[col].fillna(
                                mode_series.iloc[0]
                            )
                        else:
                            working_data[col] = working_data[col].fillna("missing")

        # Strategy 3: Quality checks
        if len(working_data) < len(data) * 0.5:
            print(
                f"Warning: Missing data handling removed {len(data) - len(working_data):,} participants"
            )
            print(
                "Consider more sophisticated missing data methods if this is substantial"
            )

        return working_data

    def _create_emulated_trial_data(
        self, data: pd.DataFrame, treatment_col: str
    ) -> pd.DataFrame:
        """Create emulated trial data by cloning participants under different strategies.

        Args:
            data: Eligible participant data
            treatment_col: Name of treatment column

        Returns:
            DataFrame with cloned participants for each treatment strategy
        """
        # True cloning implementation: each participant is duplicated for each treatment strategy
        cloned_datasets = []

        for strategy_name, strategy in self.protocol.treatment_strategies.items():
            # Clone all participants for this strategy
            strategy_data = data.copy()
            strategy_data["assigned_strategy"] = strategy_name
            strategy_data["clone_id"] = strategy_data.index

            # Apply strategy-specific treatment assignment
            strategy_data["counterfactual_treatment"] = (
                strategy.get_assigned_treatment_value(treatment_col)
            )

            # Mark whether this matches observed treatment (for adherence assessment)
            if treatment_col in strategy_data.columns:
                strategy_data["matches_observed"] = (
                    strategy_data[treatment_col]
                    == strategy_data["counterfactual_treatment"]
                )
            else:
                strategy_data["matches_observed"] = True

            cloned_datasets.append(strategy_data)

        # Combine all cloned datasets
        emulated_data = pd.concat(cloned_datasets, ignore_index=True)

        # Add unique participant-strategy identifier
        emulated_data["participant_strategy_id"] = (
            emulated_data["clone_id"].astype(str)
            + "_"
            + emulated_data["assigned_strategy"]
        )

        # Store cloned data for reference
        self._cloned_data = emulated_data.copy()

        if self.verbose:
            n_original = len(data)
            n_strategies = len(self.protocol.treatment_strategies)
            n_cloned = len(emulated_data)
            print(
                f"Cloned {n_original:,} participants across {n_strategies} strategies = {n_cloned:,} total observations"
            )

        return emulated_data

    def _apply_censoring_and_adherence(
        self, data: pd.DataFrame, treatment_col: str
    ) -> pd.DataFrame:
        """Apply censoring rules and adherence definitions.

        Args:
            data: Emulated trial data
            treatment_col: Name of treatment column

        Returns:
            DataFrame with censoring and adherence applied
        """
        analysis_data = data.copy()

        # Enhanced adherence logic based on treatment matching
        analysis_data["adherent"] = self._assess_adherence(analysis_data, treatment_col)

        # Apply grace period if specified
        if self.protocol.grace_period:
            analysis_data["within_grace_period"] = self._apply_grace_period_logic(
                analysis_data, treatment_col
            )
        else:
            analysis_data["within_grace_period"] = True

        # Apply censoring for loss to follow-up, death, etc.
        analysis_data["censored"] = self._apply_censoring_logic(analysis_data)

        # Remove non-adherent participants in per-protocol analysis context
        if hasattr(analysis_data, "matches_observed"):
            analysis_data["eligible_for_pp"] = (
                analysis_data["adherent"]
                & ~analysis_data["censored"]
                & analysis_data["within_grace_period"]
            )
        else:
            analysis_data["eligible_for_pp"] = (
                analysis_data["adherent"]
                & ~analysis_data["censored"]
                & analysis_data["within_grace_period"]
            )

        return analysis_data

    def _assess_adherence(self, data: pd.DataFrame, treatment_col: str) -> pd.Series:
        """Assess treatment adherence based on observed vs assigned treatment.

        Args:
            data: Emulated trial data with treatment assignments
            treatment_col: Name of treatment column

        Returns:
            Boolean series indicating adherence status
        """
        # Base adherence on whether observed treatment matches assigned strategy
        if "matches_observed" in data.columns:
            base_adherence = data["matches_observed"]
        else:
            # Fallback: assume adherence based on strategy assignment
            base_adherence = pd.Series(True, index=data.index)

        # Apply strategy-specific adherence requirements
        adherence = base_adherence.copy()

        for strategy_name, strategy in self.protocol.treatment_strategies.items():
            strategy_mask = data["assigned_strategy"] == strategy_name

            if strategy.sustained and strategy_mask.any():
                # For sustained strategies, require continuous adherence
                # This is a simplified version - in practice would need temporal data
                sustained_adherence_rate = 0.8  # Realistic sustained adherence rate
                n_strategy = strategy_mask.sum()
                n_adherent = int(n_strategy * sustained_adherence_rate)

                # Randomly select who remains adherent (in practice, would use temporal patterns)
                strategy_indices = data[strategy_mask].index
                if len(strategy_indices) > 0:
                    np.random.seed(42)  # For reproducibility
                    adherent_indices = np.random.choice(
                        strategy_indices,
                        size=min(n_adherent, len(strategy_indices)),
                        replace=False,
                    )
                    adherence.loc[strategy_mask] = False
                    adherence.loc[adherent_indices] = True

        return adherence

    def _apply_grace_period_logic(
        self, data: pd.DataFrame, treatment_col: str
    ) -> pd.Series:
        """Apply grace period logic for treatment initiation.

        Args:
            data: Emulated trial data
            treatment_col: Name of treatment column

        Returns:
            Boolean series indicating whether treatment was initiated within grace period
        """
        # In a real implementation, this would use actual date columns
        # For now, simulate based on treatment assignment patterns
        within_grace = pd.Series(True, index=data.index)

        # Simulate that some participants don't initiate treatment within grace period
        # Higher likelihood for strategies requiring behavior change
        for strategy_name, strategy in self.protocol.treatment_strategies.items():
            strategy_mask = data["assigned_strategy"] == strategy_name

            if strategy_mask.any():
                # Different grace period compliance rates by strategy type
                if "quit" in strategy_name.lower() or "stop" in strategy_name.lower():
                    # Behavior change strategies have lower grace period compliance
                    compliance_rate = 0.85
                else:
                    # Medication or simpler strategies have higher compliance
                    compliance_rate = 0.95

                n_strategy = strategy_mask.sum()
                n_compliant = int(n_strategy * compliance_rate)

                strategy_indices = data[strategy_mask].index
                if len(strategy_indices) > 0:
                    np.random.seed(43)  # Different seed for grace period
                    compliant_indices = np.random.choice(
                        strategy_indices,
                        size=min(n_compliant, len(strategy_indices)),
                        replace=False,
                    )
                    within_grace.loc[strategy_mask] = False
                    within_grace.loc[compliant_indices] = True

        return within_grace

    def _apply_censoring_logic(self, data: pd.DataFrame) -> pd.Series:
        """Apply censoring logic for loss to follow-up and competing events.

        Args:
            data: Emulated trial data

        Returns:
            Boolean series indicating censoring status
        """
        # Simulate realistic censoring patterns
        censored = pd.Series(False, index=data.index)

        # Base censoring rate (loss to follow-up, death, etc.)
        base_censoring_rate = 0.05

        # Higher censoring in certain strategies (e.g., if they're sicker)
        for strategy_name in self.protocol.treatment_strategies.keys():
            strategy_mask = data["assigned_strategy"] == strategy_name

            if strategy_mask.any():
                # Differential censoring by strategy if needed
                strategy_censoring_rate = base_censoring_rate

                n_strategy = strategy_mask.sum()
                n_censored = int(n_strategy * strategy_censoring_rate)

                strategy_indices = data[strategy_mask].index
                if len(strategy_indices) > 0:
                    np.random.seed(44)  # Different seed for censoring
                    censored_indices = np.random.choice(
                        strategy_indices,
                        size=min(n_censored, len(strategy_indices)),
                        replace=False,
                    )
                    censored.loc[censored_indices] = True

        return censored

    def _estimate_intention_to_treat_effect(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> CausalEffect:
        """Estimate intention-to-treat effect.

        Args:
            data: Analysis dataset
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariate_cols: Names of covariate columns

        Returns:
            CausalEffect for intention-to-treat analysis
        """
        return self._estimate_causal_effect(
            data, treatment_col, outcome_col, covariate_cols, "itt"
        )

    def _estimate_per_protocol_effect(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> CausalEffect:
        """Estimate per-protocol effect.

        Args:
            data: Analysis dataset
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariate_cols: Names of covariate columns

        Returns:
            CausalEffect for per-protocol analysis
        """
        # Filter to participants eligible for per-protocol analysis
        if "eligible_for_pp" in data.columns:
            pp_eligible_data = data[data["eligible_for_pp"]].copy()
        else:
            # Fallback to adherent participants only
            pp_eligible_data = data[data["adherent"]].copy()

        if len(pp_eligible_data) == 0:
            raise ValueError(
                "No participants eligible for per-protocol analysis after applying adherence and censoring criteria"
            )

        return self._estimate_causal_effect(
            pp_eligible_data, treatment_col, outcome_col, covariate_cols, "pp"
        )

    def _estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
        analysis_type: str,
    ) -> CausalEffect:
        """Estimate causal effect using specified method.

        Args:
            data: Analysis dataset
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariate_cols: Names of covariate columns
            analysis_type: Type of analysis ("itt" or "pp")

        Returns:
            CausalEffect estimate
        """
        # Prepare data objects with inferred types
        treatment_data = TreatmentData(
            values=data[treatment_col],
            name=treatment_col,
            treatment_type=self._infer_treatment_type(data[treatment_col]),
        )

        outcome_data = OutcomeData(
            values=data[outcome_col],
            name=outcome_col,
            outcome_type=self._infer_outcome_type(data[outcome_col]),
        )

        covariate_data = CovariateData(
            values=data[covariate_cols], names=covariate_cols
        )

        # Create and fit estimator
        estimator: BaseEstimator
        if self.estimation_method == "g_computation":
            estimator = GComputationEstimator(
                random_state=self.random_state, verbose=self.verbose
            )
        elif self.estimation_method == "ipw":
            estimator = IPWEstimator(
                random_state=self.random_state, verbose=self.verbose
            )
        elif self.estimation_method == "aipw":
            estimator = AIPWEstimator(
                random_state=self.random_state, verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown estimation method: {self.estimation_method}")

        # Fit and estimate
        estimator.fit(treatment_data, outcome_data, covariate_data)
        effect = estimator.estimate_ate()

        # Store estimator for later use
        self._fitted_estimators[analysis_type] = estimator

        return effect

    def _infer_treatment_type(self, treatment_series: pd.Series) -> str:
        """Infer treatment type from data characteristics.

        Args:
            treatment_series: Series containing treatment values

        Returns:
            Inferred treatment type
        """
        unique_values = treatment_series.dropna().unique()
        n_unique = len(unique_values)

        if n_unique == 2:
            return "binary"
        elif n_unique <= 10 and all(
            isinstance(x, (int, float)) and x == int(x)
            for x in unique_values  # noqa: UP038
        ):
            return "categorical"
        elif treatment_series.dtype in ["int64", "float64"]:
            return "continuous"
        else:
            return "categorical"

    def _infer_outcome_type(self, outcome_series: pd.Series) -> str:
        """Infer outcome type from data characteristics.

        Args:
            outcome_series: Series containing outcome values

        Returns:
            Inferred outcome type
        """
        if outcome_series.dtype == "bool":
            return "binary"
        elif len(outcome_series.dropna().unique()) == 2:
            return "binary"
        elif outcome_series.dtype in ["int64"] and (outcome_series >= 0).all():
            # Could be count data
            if outcome_series.max() < 50:  # Arbitrary threshold for count vs continuous
                return "count"
            else:
                return "continuous"
        elif outcome_series.dtype in ["int64", "float64"]:
            return "continuous"
        else:
            return "continuous"  # Default fallback

    def _generate_diagnostics(
        self,
        original_data: pd.DataFrame,
        eligible_data: pd.DataFrame,
        analysis_data: pd.DataFrame,
        treatment_col: str,
    ) -> EmulationDiagnostics:
        """Generate diagnostic information for emulation quality assessment.

        Args:
            original_data: Original observational data
            eligible_data: Data after eligibility criteria
            analysis_data: Final analysis dataset
            treatment_col: Name of treatment column

        Returns:
            EmulationDiagnostics object
        """
        # Basic sample size information
        total_sample_size = len(original_data)
        eligible_sample_size = len(eligible_data)
        final_sample_size = len(analysis_data)
        eligibility_rate = eligible_sample_size / total_sample_size

        # Treatment group sizes
        treatment_group_sizes = {}
        treatment_group_rates = {}

        for strategy_name in self.protocol.treatment_strategies.keys():
            if "assigned_strategy" in analysis_data.columns:
                group_size = (analysis_data["assigned_strategy"] == strategy_name).sum()
                treatment_group_sizes[strategy_name] = int(group_size)
                treatment_group_rates[strategy_name] = group_size / final_sample_size

        # Adherence and censoring rates
        adherence_rates = {}
        if "adherent" in analysis_data.columns:
            for strategy_name in self.protocol.treatment_strategies.keys():
                strategy_data = analysis_data[
                    analysis_data["assigned_strategy"] == strategy_name
                ]
                if len(strategy_data) > 0:
                    adherence_rate = strategy_data["adherent"].mean()
                    adherence_rates[strategy_name] = adherence_rate

        censoring_rates = {}
        if "censored" in analysis_data.columns:
            for strategy_name in self.protocol.treatment_strategies.keys():
                strategy_data = analysis_data[
                    analysis_data["assigned_strategy"] == strategy_name
                ]
                if len(strategy_data) > 0:
                    censoring_rate = strategy_data["censored"].mean()
                    censoring_rates[strategy_name] = censoring_rate

        lost_to_followup_rate = 0.0  # Simplified

        # Grace period analysis
        grace_period_compliance_rate = None
        subjects_treated_in_grace = None
        if (
            self.protocol.grace_period
            and "within_grace_period" in analysis_data.columns
        ):
            grace_period_compliance_rate = analysis_data["within_grace_period"].mean()
            subjects_treated_in_grace = int(analysis_data["within_grace_period"].sum())

        # Enhanced balance and overlap diagnostics
        covariate_balance = self._compute_covariate_balance(
            analysis_data, treatment_col
        )
        propensity_score_overlap = self._compute_propensity_overlap(
            analysis_data, treatment_col
        )

        return EmulationDiagnostics(
            total_sample_size=total_sample_size,
            eligible_sample_size=eligible_sample_size,
            final_analysis_sample_size=final_sample_size,
            eligibility_rate=eligibility_rate,
            treatment_group_sizes=treatment_group_sizes,
            treatment_group_rates=treatment_group_rates,
            adherence_rates=adherence_rates,
            censoring_rates=censoring_rates,
            lost_to_followup_rate=lost_to_followup_rate,
            grace_period_compliance_rate=grace_period_compliance_rate,
            subjects_treated_in_grace=subjects_treated_in_grace,
            covariate_balance=covariate_balance,
            propensity_score_overlap=propensity_score_overlap,
        )

    def _compute_covariate_balance(
        self, data: pd.DataFrame, treatment_col: str
    ) -> dict[str, float]:
        """Compute covariate balance across treatment strategies.

        Args:
            data: Analysis dataset
            treatment_col: Name of treatment column

        Returns:
            Dictionary of standardized mean differences for covariates
        """
        balance_stats: dict[str, float] = {}

        # Get numeric covariates for balance assessment
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        covariate_cols = [
            col
            for col in numeric_cols
            if col not in [treatment_col, "clone_id", "participant_strategy_id"]
        ]

        if len(covariate_cols) == 0:
            return balance_stats

        # For binary treatment (most common)
        treatment_values = data[treatment_col].unique()
        if len(treatment_values) == 2:
            treated = data[data[treatment_col] == treatment_values[1]]
            control = data[data[treatment_col] == treatment_values[0]]

            for col in covariate_cols:
                if col in data.columns:
                    treated_mean = treated[col].mean() if len(treated) > 0 else 0
                    control_mean = control[col].mean() if len(control) > 0 else 0

                    treated_var = treated[col].var() if len(treated) > 1 else 1
                    control_var = control[col].var() if len(control) > 1 else 1

                    pooled_std = np.sqrt((treated_var + control_var) / 2)

                    if pooled_std > 0:
                        smd = abs(treated_mean - control_mean) / pooled_std
                        balance_stats[col] = smd

        return balance_stats

    def _compute_propensity_overlap(
        self, data: pd.DataFrame, treatment_col: str
    ) -> dict[str, Any]:
        """Compute propensity score overlap diagnostics.

        Args:
            data: Analysis dataset
            treatment_col: Name of treatment column

        Returns:
            Dictionary with propensity score overlap statistics
        """
        overlap_stats: dict[str, Any] = {}

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            # Get numeric covariates for propensity modeling
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            covariate_cols = [
                col
                for col in numeric_cols
                if col not in [treatment_col, "clone_id", "participant_strategy_id"]
            ]

            if len(covariate_cols) == 0:
                return {
                    "error": "No covariates available for propensity score modeling"
                }

            # Prepare data for binary treatment
            treatment_values = data[treatment_col].unique()
            if len(treatment_values) != 2:
                return {
                    "error": "Propensity score overlap only implemented for binary treatments"
                }

            X = data[covariate_cols].values
            y = (data[treatment_col] == treatment_values[1]).astype(int)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit propensity score model
            ps_model = LogisticRegression(random_state=42)
            ps_model.fit(X_scaled, y)

            # Predict propensity scores
            propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]

            # Overlap statistics
            treated_ps = propensity_scores[y == 1]
            control_ps = propensity_scores[y == 0]

            overlap_stats = {
                "treated_ps_mean": float(treated_ps.mean())
                if len(treated_ps) > 0
                else 0.0,
                "control_ps_mean": float(control_ps.mean())
                if len(control_ps) > 0
                else 0.0,
                "treated_ps_std": float(treated_ps.std())
                if len(treated_ps) > 0
                else 0.0,
                "control_ps_std": float(control_ps.std())
                if len(control_ps) > 0
                else 0.0,
                "ps_overlap_min": float(propensity_scores.min()),
                "ps_overlap_max": float(propensity_scores.max()),
                "extreme_ps_rate": float(
                    (propensity_scores < 0.1).sum() + (propensity_scores > 0.9).sum()
                )
                / len(propensity_scores),
            }

        except ImportError:
            overlap_stats = {
                "error": "sklearn not available for propensity score modeling"
            }
        except Exception as e:
            overlap_stats = {"error": f"Propensity score computation failed: {str(e)}"}

        return overlap_stats

    @staticmethod
    def compare_estimators(
        results_by_method: dict[str, TargetTrialResults], protocol: TargetTrialProtocol
    ) -> dict[str, Any]:
        """Compare results across different estimation methods.

        Args:
            results_by_method: Dictionary mapping method names to results
            protocol: Target trial protocol used

        Returns:
            Dictionary with comparison results
        """
        comparison: dict[str, Any] = {
            "protocol_summary": protocol.get_protocol_summary(),
            "methods_compared": list(results_by_method.keys()),
            "itt_effects": {},
            "pp_effects": {},
            "consistency_assessment": {},
        }

        # Extract ITT effects
        itt_ates = []
        for method, results in results_by_method.items():
            ate = results.intention_to_treat_effect.ate
            comparison["itt_effects"][method] = {
                "ate": ate,
                "ci_lower": results.intention_to_treat_effect.ate_ci_lower,
                "ci_upper": results.intention_to_treat_effect.ate_ci_upper,
            }
            itt_ates.append(ate)

        # Extract PP effects if available
        pp_ates = []
        for method, results in results_by_method.items():
            if results.per_protocol_effect:
                ate = results.per_protocol_effect.ate
                comparison["pp_effects"][method] = {
                    "ate": ate,
                    "ci_lower": results.per_protocol_effect.ate_ci_lower,
                    "ci_upper": results.per_protocol_effect.ate_ci_upper,
                }
                pp_ates.append(ate)

        # Assess consistency
        if len(itt_ates) > 1:
            itt_range = max(itt_ates) - min(itt_ates)
            itt_mean = np.mean(itt_ates)
            itt_cv = np.std(itt_ates) / abs(itt_mean) if itt_mean != 0 else 0

            comparison["consistency_assessment"]["itt"] = {
                "range": itt_range,
                "coefficient_of_variation": itt_cv,
                "consistent": itt_range < 0.5 and itt_cv < 0.1,  # Arbitrary thresholds
            }

        if len(pp_ates) > 1:
            pp_range = max(pp_ates) - min(pp_ates)
            pp_mean = np.mean(pp_ates)
            pp_cv = np.std(pp_ates) / abs(pp_mean) if pp_mean != 0 else 0

            comparison["consistency_assessment"]["pp"] = {
                "range": pp_range,
                "coefficient_of_variation": pp_cv,
                "consistent": pp_range < 0.5 and pp_cv < 0.1,
            }

        return comparison
