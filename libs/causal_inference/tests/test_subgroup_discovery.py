"""Tests for subgroup discovery methods."""

import numpy as np
import pandas as pd
import pytest

from causal_inference.core.base import CovariateData, EstimationError, OutcomeData, TreatmentData
from causal_inference.estimators.subgroup_discovery import (
    SIDES,
    OptimalPolicyTree,
    Subgroup,
    SubgroupResult,
    VirtualTwins,
)


class TestSubgroup:
    """Test cases for Subgroup namedtuple."""

    def test_subgroup_creation(self):
        """Test Subgroup creation and properties."""
        indices = np.array([0, 1, 5, 10])
        subgroup = Subgroup(
            rule="X1 > 0.5",
            indices=indices,
            treatment_effect=1.5,
            treatment_effect_se=0.3,
            size=4,
            p_value=0.02,
        )

        assert subgroup.rule == "X1 > 0.5"
        assert np.array_equal(subgroup.indices, indices)
        assert subgroup.treatment_effect == 1.5
        assert subgroup.treatment_effect_se == 0.3
        assert subgroup.size == 4
        assert subgroup.p_value == 0.02


class TestSubgroupResult:
    """Test cases for SubgroupResult class."""

    def test_subgroup_result_creation(self):
        """Test SubgroupResult creation."""
        subgroups = [
            Subgroup("Rule 1", np.array([0, 1]), 1.0, 0.2, 2, 0.01),
            Subgroup("Rule 2", np.array([2, 3]), 0.5, 0.3, 2, 0.10),
        ]

        result = SubgroupResult(subgroups, overall_ate=0.8, method="Test")

        assert len(result.subgroups) == 2
        assert result.overall_ate == 0.8
        assert result.method == "Test"

    def test_subgroup_result_summary(self):
        """Test summary DataFrame creation."""
        subgroups = [
            Subgroup("Rule 1", np.array([0, 1]), 1.0, 0.2, 2, 0.01),
            Subgroup("Rule 2", np.array([2, 3]), 0.5, 0.3, 2, 0.10),
        ]

        result = SubgroupResult(subgroups, overall_ate=0.6, method="Test")
        summary_df = result.summary()

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 2
        assert "rule" in summary_df.columns
        assert "treatment_effect" in summary_df.columns
        assert "p_value" in summary_df.columns
        assert "significant" in summary_df.columns
        assert "effect_vs_overall" in summary_df.columns

        # Check calculated values
        assert summary_df.iloc[0]["effect_vs_overall"] == pytest.approx(
            0.4
        )  # 1.0 - 0.6
        assert summary_df.iloc[1]["effect_vs_overall"] == pytest.approx(
            -0.1
        )  # 0.5 - 0.6

    def test_subgroup_result_empty(self):
        """Test with empty subgroups list."""
        result = SubgroupResult([], overall_ate=1.0, method="Empty")
        summary_df = result.summary()

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 0

    def test_significant_subgroups(self):
        """Test filtering for significant subgroups."""
        subgroups = [
            Subgroup("Sig 1", np.array([0, 1]), 1.0, 0.2, 2, 0.01),  # Significant
            Subgroup("Not sig", np.array([2, 3]), 0.5, 0.3, 2, 0.10),  # Not significant
            Subgroup("Sig 2", np.array([4, 5]), 2.0, 0.1, 2, 0.001),  # Significant
        ]

        result = SubgroupResult(subgroups, overall_ate=0.8, method="Test")

        significant = result.significant_subgroups(alpha=0.05)
        assert len(significant) == 2
        assert significant[0].rule == "Sig 1"
        assert significant[1].rule == "Sig 2"

        # Test with different alpha
        significant_strict = result.significant_subgroups(alpha=0.01)
        assert len(significant_strict) == 2  # Both should still be significant

        significant_very_strict = result.significant_subgroups(alpha=0.001)
        assert len(significant_very_strict) == 1
        assert significant_very_strict[0].rule == "Sig 2"


class TestVirtualTwins:
    """Test cases for Virtual Twins implementation."""

    def test_virtual_twins_init(self):
        """Test Virtual Twins initialization."""
        vt = VirtualTwins(min_subgroup_size=20, random_state=42)
        assert vt.min_subgroup_size == 20
        assert vt.random_state == 42
        assert not vt.is_fitted

    def test_prepare_data(self):
        """Test data preparation."""
        np.random.seed(42)
        n = 100

        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.randn(n)
        X = np.random.randn(n, 3)

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        vt = VirtualTwins()
        T, Y, X_out = vt._prepare_data(treatment_data, outcome_data, covariate_data)

        assert T.shape == (n,)
        assert Y.shape == (n,)
        assert X_out.shape == (n, 3)
        assert np.all((T == 0) | (T == 1))

    def test_prepare_data_non_binary_treatment(self):
        """Test error with non-binary treatment."""
        treatment_data = TreatmentData(values=np.array([0, 1, 2]))
        outcome_data = OutcomeData(values=np.array([1.0, 2.0, 3.0]))

        vt = VirtualTwins()
        with pytest.raises(ValueError, match="binary treatment"):
            vt._prepare_data(treatment_data, outcome_data, None)

    def test_fit_basic(self):
        """Test basic Virtual Twins fitting."""
        np.random.seed(42)
        n = 100

        # Create synthetic data
        X = np.random.randn(n, 3)
        treatment = np.random.binomial(1, 0.5, n)
        # Heterogeneous effect based on X[0]
        effect = 0.5 + 1.0 * X[:, 0]
        outcome = X.sum(axis=1) + treatment * effect + np.random.randn(n) * 0.3

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        vt = VirtualTwins(random_state=42)
        vt.fit(treatment_data, outcome_data, covariate_data)

        assert vt.is_fitted
        assert vt._training_data is not None
        assert vt._benefit_model is not None
        assert vt._subgroup_model is not None

    def test_discover_subgroups(self):
        """Test subgroup discovery."""
        np.random.seed(42)
        n = 100

        X = np.random.randn(n, 2)
        treatment = np.random.binomial(1, 0.5, n)
        effect = np.where(X[:, 0] > 0, 2.0, 0.5)  # Clear subgroup structure
        outcome = X.sum(axis=1) + treatment * effect + np.random.randn(n) * 0.3

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        vt = VirtualTwins(min_subgroup_size=10, random_state=42)
        vt.fit(treatment_data, outcome_data, covariate_data)

        result = vt.discover_subgroups()

        assert isinstance(result, SubgroupResult)
        assert result.method == "Virtual Twins"
        assert len(result.subgroups) >= 0  # May find 0 or more subgroups

        # Each subgroup should have required properties
        for subgroup in result.subgroups:
            assert len(subgroup.indices) >= vt.min_subgroup_size
            assert isinstance(subgroup.treatment_effect, float)
            assert isinstance(subgroup.p_value, float)
            assert 0 <= subgroup.p_value <= 1

    def test_estimate_ate(self):
        """Test ATE estimation."""
        np.random.seed(42)
        n = 80

        X = np.random.randn(n, 2)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = X.sum(axis=1) + treatment * 1.5 + np.random.randn(n) * 0.3

        treatment_data = TreatmentData(values=treatment)
        outcome_data = OutcomeData(values=outcome)
        covariate_data = CovariateData(values=X)

        vt = VirtualTwins(random_state=42)
        vt.fit(treatment_data, outcome_data, covariate_data)

        ate_result = vt.estimate_ate()

        assert hasattr(ate_result, "ate")
        assert ate_result.method == "Virtual Twins"
        # Should be reasonably close to true ATE of 1.5
        assert abs(ate_result.ate - 1.5) < 1.0

    def test_not_fitted_error(self):
        """Test error when discovering subgroups before fitting."""
        vt = VirtualTwins()

        with pytest.raises(EstimationError, match="not fitted"):
            vt.discover_subgroups()


class TestOptimalPolicyTree:
    """Test cases for Optimal Policy Tree (placeholder implementation)."""

    def test_optimal_policy_tree_init(self):
        """Test OptimalPolicyTree initialization."""
        opt = OptimalPolicyTree(max_depth=2, min_samples_leaf=15)
        assert opt.max_depth == 2
        assert opt.min_samples_leaf == 15

    def test_fit_and_predict(self):
        """Test basic fitting and prediction."""
        np.random.seed(42)
        n = 100

        X = np.random.randn(n, 3)
        outcomes = np.random.randn(n)
        treatments = np.random.binomial(1, 0.5, n)
        cate_estimates = np.random.randn(n)

        opt = OptimalPolicyTree(max_depth=3)
        opt.fit(X, outcomes, treatments, cate_estimates)

        # Test prediction
        X_test = np.random.randn(20, 3)
        policy = opt.predict_policy(X_test)

        assert policy.shape == (20,)
        assert np.all((policy == 0) | (policy == 1))  # Binary policy

    def test_policy_prediction_logic(self):
        """Test that policy recommends treatment when CATE > 0."""
        np.random.seed(42)
        n = 50

        X = np.random.randn(n, 2)
        outcomes = np.random.randn(n)
        treatments = np.random.binomial(1, 0.5, n)

        # Clear positive and negative effects
        cate_estimates = np.array([2.0] * 25 + [-2.0] * 25)

        opt = OptimalPolicyTree()
        opt.fit(X, outcomes, treatments, cate_estimates)

        # Test on training data
        policy = opt.predict_policy(X)

        # Should recommend treatment for positive CATE estimates
        # (This is probabilistic due to tree splits, so we allow some flexibility)
        positive_cate_indices = np.where(cate_estimates > 0)[0]
        recommended_rate = np.mean(policy[positive_cate_indices])
        assert (
            recommended_rate > 0.3
        )  # Should recommend treatment for most positive cases


class TestSIDES:
    """Test cases for SIDES (placeholder implementation)."""

    def test_sides_init(self):
        """Test SIDES initialization."""
        sides = SIDES(min_subgroup_size=25, significance_level=0.01)
        assert sides.min_subgroup_size == 25
        assert sides.significance_level == 0.01

    def test_discover_subgroups_basic(self):
        """Test basic subgroup discovery with SIDES."""
        np.random.seed(42)
        n = 100

        X = np.random.randn(n, 3)
        treatments = np.random.binomial(1, 0.5, n)
        outcomes = np.random.randn(n)
        cate_estimates = np.random.randn(n)

        sides = SIDES(min_subgroup_size=20)
        result = sides.discover_subgroups(X, outcomes, treatments, cate_estimates)

        assert isinstance(result, SubgroupResult)
        assert result.method == "SIDES"
        assert len(result.subgroups) >= 0

        # Check subgroup properties
        for subgroup in result.subgroups:
            assert len(subgroup.indices) >= sides.min_subgroup_size
            assert isinstance(subgroup.treatment_effect, float)
            assert 0 <= subgroup.p_value <= 1

    def test_discover_subgroups_with_clear_clusters(self):
        """Test SIDES with data that has clear subgroup structure."""
        np.random.seed(42)
        n = 120

        # Create two clear subgroups
        X1 = np.random.randn(60, 2) + np.array([2, 2])  # Cluster 1
        X2 = np.random.randn(60, 2) + np.array([-2, -2])  # Cluster 2
        X = np.vstack([X1, X2])

        treatments = np.random.binomial(1, 0.5, n)

        # Different treatment effects for each cluster
        cate_true = np.array([2.0] * 60 + [0.5] * 60)
        outcomes = np.random.randn(n) + treatments * cate_true
        cate_estimates = cate_true + np.random.randn(n) * 0.1  # Slightly noisy

        sides = SIDES(min_subgroup_size=25)
        result = sides.discover_subgroups(X, outcomes, treatments, cate_estimates)

        assert len(result.subgroups) >= 1  # Should find at least one subgroup

        # Check that subgroups have different treatment effects
        if len(result.subgroups) > 1:
            effects = [s.treatment_effect for s in result.subgroups]
            assert max(effects) - min(effects) > 0.5  # Should detect difference


@pytest.fixture
def synthetic_subgroup_data():
    """Create synthetic data with clear subgroup structure."""
    np.random.seed(789)
    n = 200

    # Create two groups based on X[0]
    X = np.random.randn(n, 3)
    group_indicator = X[:, 0] > 0

    # Different treatment effects for each group
    treatment = np.random.binomial(1, 0.5, n)
    effect_high = 2.0  # High effect group
    effect_low = 0.2  # Low effect group

    true_effect = np.where(group_indicator, effect_high, effect_low)
    outcome = X.sum(axis=1) + treatment * true_effect + np.random.randn(n) * 0.4

    return {
        "X": X,
        "treatment": treatment,
        "outcome": outcome,
        "group_indicator": group_indicator,
        "true_effect": true_effect,
        "effect_high": effect_high,
        "effect_low": effect_low,
    }


class TestSubgroupDiscoveryIntegration:
    """Integration tests for subgroup discovery methods."""

    def test_virtual_twins_finds_subgroups(self, synthetic_subgroup_data):
        """Test that Virtual Twins can identify meaningful subgroups."""
        data = synthetic_subgroup_data

        treatment_data = TreatmentData(values=data["treatment"])
        outcome_data = OutcomeData(values=data["outcome"])
        covariate_data = CovariateData(values=data["X"])

        vt = VirtualTwins(min_subgroup_size=30, random_state=42)
        vt.fit(treatment_data, outcome_data, covariate_data)

        result = vt.discover_subgroups()

        # Should find some subgroups
        assert len(result.subgroups) > 0

        # Check that subgroups have reasonable treatment effects
        for subgroup in result.subgroups:
            assert subgroup.size >= vt.min_subgroup_size
            # Treatment effect should be between the two true effects
            assert (
                data["effect_low"] - 0.5
                <= subgroup.treatment_effect
                <= data["effect_high"] + 0.5
            )

    def test_sides_clustering_approach(self, synthetic_subgroup_data):
        """Test SIDES clustering-based approach."""
        data = synthetic_subgroup_data

        # Estimate CATE (use true effects with some noise)
        cate_estimates = (
            data["true_effect"] + np.random.randn(len(data["true_effect"])) * 0.1
        )

        sides = SIDES(min_subgroup_size=40)
        result = sides.discover_subgroups(
            data["X"], data["outcome"], data["treatment"], cate_estimates
        )

        # Should identify subgroups
        assert len(result.subgroups) >= 1

        # Subgroups should have different treatment effects
        if len(result.subgroups) > 1:
            effects = [s.treatment_effect for s in result.subgroups]
            effect_range = max(effects) - min(effects)
            assert effect_range > 0.3  # Should detect meaningful differences

    def test_subgroup_result_workflow(self, synthetic_subgroup_data):
        """Test complete subgroup analysis workflow."""
        data = synthetic_subgroup_data

        # Use Virtual Twins for discovery
        treatment_data = TreatmentData(values=data["treatment"])
        outcome_data = OutcomeData(values=data["outcome"])
        covariate_data = CovariateData(values=data["X"])

        vt = VirtualTwins(min_subgroup_size=25, random_state=42)
        vt.fit(treatment_data, outcome_data, covariate_data)

        result = vt.discover_subgroups()

        # Analyze results
        summary = result.summary()
        # Lenient for small sample

        # Should have some structure
        if len(summary) > 0:
            assert "treatment_effect" in summary.columns
            assert "p_value" in summary.columns
            assert "significant" in summary.columns

            # Check that effect vs overall is computed correctly
            for i, row in summary.iterrows():
                expected_diff = row["treatment_effect"] - result.overall_ate
                assert abs(row["effect_vs_overall"] - expected_diff) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])

