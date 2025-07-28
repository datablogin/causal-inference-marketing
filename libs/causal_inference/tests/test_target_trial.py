"""Tests for target trial emulation framework.

This module contains comprehensive tests for the target trial emulation framework,
covering protocol specification, emulation methods, and results reporting.
"""

import numpy as np
import pandas as pd
import pytest

from causal_inference.target_trial import (
    TargetTrialEmulator,
    TargetTrialProtocol,
    TargetTrialResults,
)
from causal_inference.target_trial.protocol import (
    EligibilityCriteria,
    FollowUpPeriod,
    GracePeriod,
    TreatmentStrategy,
)


class TestEligibilityCriteria:
    """Test cases for EligibilityCriteria."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200

        self.test_data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n),
                "smoker": np.random.choice([True, False], n),
                "wt82_71": np.random.normal(2.5, 5.0, n),
                "education": np.random.randint(8, 20, n),
                "sex": np.random.choice([0, 1], n),
            }
        )

        # Add some missing values
        missing_indices = np.random.choice(n, size=20, replace=False)
        self.test_data.loc[missing_indices, "wt82_71"] = np.nan

    def test_age_criteria(self):
        """Test age-based eligibility criteria."""
        criteria = EligibilityCriteria(age_min=25, age_max=65)
        eligible = criteria.check_eligibility(self.test_data)

        assert eligible.dtype == bool
        assert eligible.sum() > 0

        # Check that age criteria are correctly applied
        eligible_ages = self.test_data.loc[eligible, "age"]
        assert eligible_ages.min() >= 25
        assert eligible_ages.max() <= 65

    def test_smoking_criteria(self):
        """Test smoking status eligibility criteria."""
        criteria = EligibilityCriteria(baseline_smoker=True)
        eligible = criteria.check_eligibility(self.test_data)

        # All eligible participants should be smokers
        eligible_smoking = self.test_data.loc[eligible, "smoker"]
        assert eligible_smoking.all()

    def test_no_missing_weight_criteria(self):
        """Test missing weight data criteria."""
        criteria = EligibilityCriteria(no_missing_weight=True)
        eligible = criteria.check_eligibility(self.test_data)

        # All eligible participants should have weight data
        eligible_weights = self.test_data.loc[eligible, "wt82_71"]
        assert eligible_weights.notna().all()

    def test_custom_criteria(self):
        """Test custom eligibility criteria."""
        criteria = EligibilityCriteria(
            custom_criteria={"education": {"in": [12, 16, 20]}, "sex": 1}
        )
        eligible = criteria.check_eligibility(self.test_data)

        eligible_education = self.test_data.loc[eligible, "education"]
        eligible_sex = self.test_data.loc[eligible, "sex"]

        assert eligible_education.isin([12, 16, 20]).all()
        assert (eligible_sex == 1).all()

    def test_combined_criteria(self):
        """Test multiple eligibility criteria combined."""
        criteria = EligibilityCriteria(
            age_min=30,
            age_max=70,
            baseline_smoker=True,
            no_missing_weight=True,
            custom_criteria={"education": (12, 18)},
        )
        eligible = criteria.check_eligibility(self.test_data)

        if eligible.sum() > 0:
            eligible_data = self.test_data[eligible]
            assert (eligible_data["age"] >= 30).all()
            assert (eligible_data["age"] <= 70).all()
            assert eligible_data["smoker"].all()
            assert eligible_data["wt82_71"].notna().all()
            assert (eligible_data["education"] >= 12).all()
            assert (eligible_data["education"] <= 18).all()


class TestTreatmentStrategy:
    """Test cases for TreatmentStrategy."""

    def setup_method(self):
        """Set up test data."""
        self.test_data = pd.DataFrame(
            {"qsmk": [0, 1, 0, 1, 1, 0], "other_treatment": [1, 0, 1, 0, 1, 0]}
        )

    def test_treatment_assignment(self):
        """Test treatment strategy assignment."""
        strategy = TreatmentStrategy(treatment_assignment={"qsmk": 1})

        assigned = strategy.apply_strategy(self.test_data, "qsmk")
        expected = pd.Series([False, True, False, True, True, False])

        pd.testing.assert_series_equal(assigned, expected, check_names=False)

    def test_sustained_treatment(self):
        """Test sustained treatment strategy."""
        strategy = TreatmentStrategy(treatment_assignment={"qsmk": 1}, sustained=True)

        # Basic functionality test - more complex sustained logic would be in emulator
        assigned = strategy.apply_strategy(self.test_data, "qsmk")
        assert assigned.sum() == 3  # Three participants with qsmk=1


class TestFollowUpPeriod:
    """Test cases for FollowUpPeriod."""

    def test_follow_up_period_validation(self):
        """Test follow-up period validation."""
        period = FollowUpPeriod(duration=10, unit="years")
        assert period.duration == 10
        assert period.unit == "years"

    def test_invalid_unit(self):
        """Test invalid time unit raises error."""
        with pytest.raises(ValueError, match="unit must be one of"):
            FollowUpPeriod(duration=5, unit="decades")

    def test_to_days_conversion(self):
        """Test conversion to days."""
        years_period = FollowUpPeriod(duration=2, unit="years")
        assert years_period.to_days() == 730

        months_period = FollowUpPeriod(duration=6, unit="months")
        assert months_period.to_days() == 180

        days_period = FollowUpPeriod(duration=30, unit="days")
        assert days_period.to_days() == 30


class TestGracePeriod:
    """Test cases for GracePeriod."""

    def test_grace_period_validation(self):
        """Test grace period validation."""
        grace = GracePeriod(duration=6, unit="months")
        assert grace.duration == 6
        assert grace.unit == "months"

    def test_grace_period_to_days(self):
        """Test grace period conversion to days."""
        grace = GracePeriod(duration=3, unit="months")
        assert grace.to_days() == 90


class TestTargetTrialProtocol:
    """Test cases for TargetTrialProtocol."""

    def setup_method(self):
        """Set up test data and protocol."""
        np.random.seed(42)
        n = 500

        self.test_data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n),
                "smoker": np.random.choice([True, False], n, p=[0.3, 0.7]),
                "qsmk": np.random.choice([0, 1], n, p=[0.7, 0.3]),
                "wt82_71": np.random.normal(2.5, 5.0, n),
                "education": np.random.randint(8, 20, n),
                "sex": np.random.choice([0, 1], n),
            }
        )

        self.protocol = TargetTrialProtocol(
            eligibility_criteria=EligibilityCriteria(
                age_min=25, age_max=75, baseline_smoker=True
            ),
            treatment_strategies={
                "quit_smoking": TreatmentStrategy(treatment_assignment={"qsmk": 1}),
                "continue_smoking": TreatmentStrategy(treatment_assignment={"qsmk": 0}),
            },
            follow_up_period=FollowUpPeriod(duration=10, unit="years"),
            primary_outcome="wt82_71",
        )

    def test_protocol_initialization(self):
        """Test protocol initialization."""
        assert len(self.protocol.treatment_strategies) == 2
        assert self.protocol.primary_outcome == "wt82_71"
        assert self.protocol.follow_up_period.duration == 10

    def test_treatment_strategies_validation(self):
        """Test treatment strategies validation."""
        with pytest.raises(
            ValueError, match="Must specify at least 2 treatment strategies"
        ):
            TargetTrialProtocol(
                treatment_strategies={
                    "only_one": TreatmentStrategy(treatment_assignment={"qsmk": 1})
                },
                follow_up_period=FollowUpPeriod(duration=5, unit="years"),
                primary_outcome="outcome",
            )

    def test_assignment_procedure_validation(self):
        """Test assignment procedure validation."""
        with pytest.raises(ValueError, match="assignment_procedure must be one of"):
            TargetTrialProtocol(
                treatment_strategies={
                    "a": TreatmentStrategy(treatment_assignment={"qsmk": 1}),
                    "b": TreatmentStrategy(treatment_assignment={"qsmk": 0}),
                },
                assignment_procedure="invalid_procedure",
                follow_up_period=FollowUpPeriod(duration=5, unit="years"),
                primary_outcome="outcome",
            )

    def test_feasibility_check(self):
        """Test protocol feasibility check."""
        feasibility = self.protocol.check_feasibility(self.test_data)

        assert isinstance(feasibility, dict)
        assert "is_feasible" in feasibility
        assert "eligible_sample_size" in feasibility
        assert "treatment_group_sizes" in feasibility

        # Should be feasible with sufficient sample size
        assert feasibility["eligible_sample_size"] > 0

    def test_infeasible_protocol(self):
        """Test infeasible protocol detection."""
        infeasible_protocol = TargetTrialProtocol(
            eligibility_criteria=EligibilityCriteria(
                age_min=20,
                age_max=25,  # Very narrow age range
                baseline_smoker=True,
                custom_criteria={"education": (19, 20)},  # Unrealistic education
            ),
            treatment_strategies={
                "quit": TreatmentStrategy(treatment_assignment={"qsmk": 1}),
                "continue": TreatmentStrategy(treatment_assignment={"qsmk": 0}),
            },
            follow_up_period=FollowUpPeriod(duration=5, unit="years"),
            primary_outcome="wt82_71",
        )

        feasibility = infeasible_protocol.check_feasibility(self.test_data)
        # Might be infeasible due to strict criteria
        if not feasibility["is_feasible"]:
            assert len(feasibility["infeasibility_reasons"]) > 0

    def test_protocol_summary(self):
        """Test protocol summary generation."""
        summary = self.protocol.get_protocol_summary()

        assert isinstance(summary, str)
        assert "Target Trial Protocol Summary" in summary
        assert "wt82_71" in summary
        assert "quit_smoking" in summary
        assert "continue_smoking" in summary

    def test_validate_against_data(self):
        """Test protocol validation against data."""
        validation = self.protocol.validate_against_data(self.test_data)

        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation

        # Should be valid with our test data
        assert validation["valid"] is True

    def test_validate_missing_variables(self):
        """Test validation with missing variables."""
        incomplete_data = self.test_data.drop(columns=["wt82_71"])
        validation = self.protocol.validate_against_data(incomplete_data)

        assert validation["valid"] is False
        assert "wt82_71" in validation["missing_variables"]


class TestTargetTrialEmulator:
    """Test cases for TargetTrialEmulator."""

    def setup_method(self):
        """Set up test data and emulator."""
        np.random.seed(42)
        n = 300

        # Create NHEFS-like synthetic data
        age = np.random.normal(40, 10, n)
        sex = np.random.binomial(1, 0.5, n)
        race = np.random.binomial(1, 0.3, n)
        education = np.random.poisson(12, n)

        # Generate treatment with confounding
        treatment_logits = -1.0 + 0.02 * age + 0.3 * sex - 0.2 * race + 0.05 * education
        treatment_probs = 1 / (1 + np.exp(-treatment_logits))
        qsmk = np.random.binomial(1, treatment_probs)

        # Generate outcome with treatment effect
        true_ate = 3.0
        wt82_71 = (
            0.1 * age
            + 1.5 * sex
            + 1.0 * race
            - 0.1 * education
            + true_ate * qsmk
            + np.random.normal(0, 2, n)
        )

        self.test_data = pd.DataFrame(
            {
                "age": age,
                "sex": sex,
                "race": race,
                "education": education,
                "qsmk": qsmk,
                "wt82_71": wt82_71,
                "smoker": True,  # Assume all are baseline smokers for simplicity
            }
        )

        self.protocol = TargetTrialProtocol(
            eligibility_criteria=EligibilityCriteria(
                age_min=25, age_max=75, baseline_smoker=True
            ),
            treatment_strategies={
                "quit_smoking": TreatmentStrategy(treatment_assignment={"qsmk": 1}),
                "continue_smoking": TreatmentStrategy(treatment_assignment={"qsmk": 0}),
            },
            follow_up_period=FollowUpPeriod(duration=10, unit="years"),
            primary_outcome="wt82_71",
            grace_period=GracePeriod(duration=6, unit="months"),
        )

        self.true_ate = true_ate

    def test_emulator_initialization(self):
        """Test emulator initialization."""
        emulator = TargetTrialEmulator(
            protocol=self.protocol, estimation_method="g_computation"
        )

        assert emulator.protocol == self.protocol
        assert emulator.estimation_method == "g_computation"
        assert emulator.adherence_adjustment == "intention_to_treat"

    def test_invalid_estimation_method(self):
        """Test invalid estimation method raises error."""
        with pytest.raises(ValueError, match="estimation_method must be one of"):
            TargetTrialEmulator(
                protocol=self.protocol, estimation_method="invalid_method"
            )

    def test_invalid_adherence_adjustment(self):
        """Test invalid adherence adjustment raises error."""
        with pytest.raises(ValueError, match="adherence_adjustment must be one of"):
            TargetTrialEmulator(
                protocol=self.protocol, adherence_adjustment="invalid_adjustment"
            )

    def test_basic_emulation_g_computation(self):
        """Test basic emulation with G-computation."""
        emulator = TargetTrialEmulator(
            protocol=self.protocol,
            estimation_method="g_computation",
            random_state=42,
            verbose=False,
        )

        results = emulator.emulate(
            data=self.test_data,
            treatment_col="qsmk",
            outcome_col="wt82_71",
            covariate_cols=["age", "sex", "race", "education"],
        )

        assert isinstance(results, TargetTrialResults)
        assert results.intention_to_treat_effect is not None
        assert isinstance(results.intention_to_treat_effect.ate, float)

        # Should recover treatment effect within reasonable bounds
        assert abs(results.intention_to_treat_effect.ate - self.true_ate) < 2.0

    def test_basic_emulation_ipw(self):
        """Test basic emulation with IPW."""
        emulator = TargetTrialEmulator(
            protocol=self.protocol,
            estimation_method="ipw",
            random_state=42,
            verbose=False,
        )

        results = emulator.emulate(
            data=self.test_data,
            treatment_col="qsmk",
            outcome_col="wt82_71",
            covariate_cols=["age", "sex", "race", "education"],
        )

        assert isinstance(results, TargetTrialResults)
        assert results.intention_to_treat_effect is not None
        # Should provide reasonable estimate
        assert abs(results.intention_to_treat_effect.ate - self.true_ate) < 3.0

    def test_basic_emulation_aipw(self):
        """Test basic emulation with AIPW."""
        emulator = TargetTrialEmulator(
            protocol=self.protocol,
            estimation_method="aipw",
            random_state=42,
            verbose=False,
        )

        results = emulator.emulate(
            data=self.test_data,
            treatment_col="qsmk",
            outcome_col="wt82_71",
            covariate_cols=["age", "sex", "race", "education"],
        )

        assert isinstance(results, TargetTrialResults)
        assert results.intention_to_treat_effect is not None
        # Should provide reasonable estimate
        assert abs(results.intention_to_treat_effect.ate - self.true_ate) < 3.0

    def test_per_protocol_analysis(self):
        """Test per-protocol analysis."""
        emulator = TargetTrialEmulator(
            protocol=self.protocol,
            estimation_method="g_computation",
            adherence_adjustment="both",
            random_state=42,
            verbose=False,
        )

        results = emulator.emulate(
            data=self.test_data,
            treatment_col="qsmk",
            outcome_col="wt82_71",
            covariate_cols=["age", "sex", "race", "education"],
        )

        assert results.per_protocol_effect is not None
        assert isinstance(results.per_protocol_effect.ate, float)

        # Per-protocol effect might be different from ITT
        # (though in our simplified implementation they'll be similar)

    def test_emulation_diagnostics(self):
        """Test emulation diagnostics generation."""
        emulator = TargetTrialEmulator(
            protocol=self.protocol, estimation_method="g_computation", random_state=42
        )

        results = emulator.emulate(
            data=self.test_data,
            treatment_col="qsmk",
            outcome_col="wt82_71",
            covariate_cols=["age", "sex", "race", "education"],
        )

        assert results.diagnostics is not None
        assert results.diagnostics.total_sample_size == len(self.test_data)
        assert results.diagnostics.eligible_sample_size > 0
        assert len(results.diagnostics.treatment_group_sizes) > 0

    def test_infeasible_protocol_raises_error(self):
        """Test that infeasible protocol raises error."""
        infeasible_protocol = TargetTrialProtocol(
            eligibility_criteria=EligibilityCriteria(
                age_min=90,  # No one will meet this
                age_max=95,
            ),
            treatment_strategies={
                "quit": TreatmentStrategy(treatment_assignment={"qsmk": 1}),
                "continue": TreatmentStrategy(treatment_assignment={"qsmk": 0}),
            },
            follow_up_period=FollowUpPeriod(duration=5, unit="years"),
            primary_outcome="wt82_71",
        )

        emulator = TargetTrialEmulator(
            protocol=infeasible_protocol, estimation_method="g_computation"
        )

        with pytest.raises(ValueError, match="Protocol is not feasible"):
            emulator.emulate(
                data=self.test_data,
                treatment_col="qsmk",
                outcome_col="wt82_71",
                covariate_cols=["age", "sex", "race", "education"],
            )

    def test_invalid_protocol_raises_error(self):
        """Test that invalid protocol raises error."""
        invalid_protocol = TargetTrialProtocol(
            treatment_strategies={
                "quit": TreatmentStrategy(treatment_assignment={"qsmk": 1}),
                "continue": TreatmentStrategy(treatment_assignment={"qsmk": 0}),
            },
            follow_up_period=FollowUpPeriod(duration=5, unit="years"),
            primary_outcome="missing_outcome",  # Not in data
        )

        emulator = TargetTrialEmulator(
            protocol=invalid_protocol, estimation_method="g_computation"
        )

        with pytest.raises(ValueError, match="Protocol validation failed"):
            emulator.emulate(
                data=self.test_data,
                treatment_col="qsmk",
                outcome_col="wt82_71",
                covariate_cols=["age", "sex", "race", "education"],
            )


class TestTargetTrialResults:
    """Test cases for TargetTrialResults."""

    def setup_method(self):
        """Set up test results."""
        from causal_inference.core.base import CausalEffect
        from causal_inference.target_trial.results import EmulationDiagnostics

        self.itt_effect = CausalEffect(
            ate=3.2,
            ate_se=0.5,
            ate_ci_lower=2.2,
            ate_ci_upper=4.2,
            method="G-computation",
        )

        self.pp_effect = CausalEffect(
            ate=3.8,
            ate_se=0.6,
            ate_ci_lower=2.6,
            ate_ci_upper=5.0,
            method="G-computation",
        )

        self.diagnostics = EmulationDiagnostics(
            total_sample_size=1000,
            eligible_sample_size=800,
            final_analysis_sample_size=750,
            eligibility_rate=0.8,
            treatment_group_sizes={"quit": 250, "continue": 500},
            treatment_group_rates={"quit": 0.33, "continue": 0.67},
            adherence_rates={"quit": 0.85, "continue": 0.95},
            censoring_rates={"quit": 0.10, "continue": 0.05},
            lost_to_followup_rate=0.05,
        )

        self.results = TargetTrialResults(
            intention_to_treat_effect=self.itt_effect,
            per_protocol_effect=self.pp_effect,
            protocol_summary="Test protocol summary",
            estimation_method="g_computation",
            diagnostics=self.diagnostics,
        )

    def test_results_initialization(self):
        """Test results initialization."""
        assert self.results.intention_to_treat_effect.ate == 3.2
        assert self.results.per_protocol_effect.ate == 3.8
        assert self.results.estimation_method == "g_computation"

    def test_generate_report(self):
        """Test report generation."""
        report = self.results.generate_report()

        assert hasattr(report, "sections")
        assert "emulation_results" in report.sections
        assert isinstance(report.to_string(), str)

    def test_compare_itt_vs_pp(self):
        """Test ITT vs per-protocol comparison."""
        comparison = self.results.compare_itt_vs_pp()

        assert comparison["itt_ate"] == 3.2
        assert comparison["pp_ate"] == 3.8
        assert (
            abs(comparison["difference"] - 0.6) < 1e-10
        )  # Account for floating point precision
        assert "interpretation" in comparison

    def test_compare_itt_vs_pp_no_pp(self):
        """Test comparison when no per-protocol effect available."""
        results_no_pp = TargetTrialResults(intention_to_treat_effect=self.itt_effect)

        comparison = results_no_pp.compare_itt_vs_pp()
        assert "error" in comparison


class TestTargetTrialComparison:
    """Test cases for comparing multiple estimators."""

    def setup_method(self):
        """Set up comparison test data."""
        from causal_inference.core.base import CausalEffect

        # Mock results from different methods
        self.g_comp_effect = CausalEffect(ate=3.2, method="G-computation")
        self.ipw_effect = CausalEffect(ate=3.0, method="IPW")
        self.aipw_effect = CausalEffect(ate=3.1, method="AIPW")

        self.results_by_method = {
            "g_computation": TargetTrialResults(
                intention_to_treat_effect=self.g_comp_effect
            ),
            "ipw": TargetTrialResults(intention_to_treat_effect=self.ipw_effect),
            "aipw": TargetTrialResults(intention_to_treat_effect=self.aipw_effect),
        }

        self.protocol = TargetTrialProtocol(
            treatment_strategies={
                "quit": TreatmentStrategy(treatment_assignment={"qsmk": 1}),
                "continue": TreatmentStrategy(treatment_assignment={"qsmk": 0}),
            },
            follow_up_period=FollowUpPeriod(duration=10, unit="years"),
            primary_outcome="wt82_71",
        )

    def test_compare_estimators(self):
        """Test estimator comparison functionality."""
        comparison = TargetTrialEmulator.compare_estimators(
            self.results_by_method, self.protocol
        )

        assert "methods_compared" in comparison
        assert "itt_effects" in comparison
        assert "consistency_assessment" in comparison

        assert len(comparison["methods_compared"]) == 3
        assert "g_computation" in comparison["itt_effects"]
        assert "ipw" in comparison["itt_effects"]
        assert "aipw" in comparison["itt_effects"]


if __name__ == "__main__":
    pytest.main([__file__])
