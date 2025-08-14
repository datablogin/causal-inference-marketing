"""Tests for HTE evaluation metrics."""

import numpy as np
import pytest

from causal_inference.evaluation.hte_metrics import (
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


class TestPEHEScore:
    """Test cases for PEHE (Precision in Estimation of Heterogeneous Effect)."""

    def test_pehe_perfect_prediction(self):
        """Test PEHE with perfect predictions."""
        y_true = np.array([1.0, 2.0, -0.5, 3.0])
        y_pred = np.array([1.0, 2.0, -0.5, 3.0])

        pehe = pehe_score(y_true, y_pred)
        assert pehe == pytest.approx(0.0, abs=1e-10)

        pehe_sqrt = pehe_score(y_true, y_pred, squared=False)
        assert pehe_sqrt == pytest.approx(0.0, abs=1e-10)

    def test_pehe_basic_calculation(self):
        """Test PEHE calculation with known values."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 3.5])

        # MSE = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
        expected_pehe = 0.25
        pehe = pehe_score(y_true, y_pred)
        assert pehe == pytest.approx(expected_pehe)

        pehe_sqrt = pehe_score(y_true, y_pred, squared=False)
        assert pehe_sqrt == pytest.approx(0.5)

    def test_pehe_single_value(self):
        """Test PEHE with single prediction."""
        y_true = np.array([2.0])
        y_pred = np.array([1.5])

        expected_pehe = 0.25  # (2-1.5)^2
        pehe = pehe_score(y_true, y_pred)
        assert pehe == pytest.approx(expected_pehe)


class TestPolicyValue:
    """Test cases for policy value computation."""

    def test_policy_value_all_treat(self):
        """Test policy value when policy treats everyone."""
        outcomes = np.array([2.0, 3.0, 1.5, 4.0])
        treatments = np.array([1, 0, 1, 0])
        cate_estimates = np.array([1.0, 2.0, 3.0, 1.5])  # All positive

        pv = policy_value(outcomes, treatments, cate_estimates, policy_threshold=0.0)
        assert isinstance(pv, float)
        assert not np.isnan(pv)

    def test_policy_value_no_treat(self):
        """Test policy value when policy treats no one."""
        outcomes = np.array([2.0, 3.0, 1.5, 4.0])
        treatments = np.array([1, 0, 1, 0])
        cate_estimates = np.array([-1.0, -2.0, -0.5, -1.5])  # All negative

        pv = policy_value(outcomes, treatments, cate_estimates, policy_threshold=0.0)
        assert isinstance(pv, float)
        assert not np.isnan(pv)

    def test_policy_value_mixed(self):
        """Test policy value with mixed treatment recommendations."""
        outcomes = np.array([2.0, 3.0, 1.5, 4.0, 2.5, 3.5])
        treatments = np.array([1, 0, 1, 0, 1, 0])
        cate_estimates = np.array([1.0, -0.5, 2.0, -1.0, 0.5, -2.0])

        pv = policy_value(outcomes, treatments, cate_estimates, policy_threshold=0.0)
        assert isinstance(pv, float)
        assert not np.isnan(pv)


class TestOverlapWeightedMSE:
    """Test cases for overlap-weighted MSE."""

    def test_overlap_weighted_mse_uniform_propensity(self):
        """Test with uniform propensity scores (e=0.5)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 3.5])
        propensity_scores = np.array([0.5, 0.5, 0.5, 0.5])

        # With e=0.5 everywhere, overlap weights are all 1.0
        # So this should equal regular MSE
        regular_mse = np.mean((y_true - y_pred) ** 2)
        overlap_mse = overlap_weighted_mse(y_true, y_pred, propensity_scores)

        assert overlap_mse == pytest.approx(regular_mse)

    def test_overlap_weighted_mse_varying_propensity(self):
        """Test with varying propensity scores."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 3.5])
        propensity_scores = np.array([0.1, 0.3, 0.7, 0.9])

        overlap_mse = overlap_weighted_mse(y_true, y_pred, propensity_scores)
        assert isinstance(overlap_mse, float)
        assert overlap_mse > 0


class TestATEPreservationScore:
    """Test cases for ATE preservation score."""

    def test_ate_preservation_perfect(self):
        """Test perfect ATE preservation."""
        true_ate = 1.5
        cate_estimates = np.array([1.5, 1.5, 1.5, 1.5])

        score = ate_preservation_score(true_ate, cate_estimates)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_ate_preservation_deviation(self):
        """Test ATE preservation with deviation."""
        true_ate = 2.0
        cate_estimates = np.array([1.0, 2.0, 3.0, 2.0])  # Mean = 2.0

        score = ate_preservation_score(true_ate, cate_estimates)
        assert score == pytest.approx(0.0)

    def test_ate_preservation_bias(self):
        """Test ATE preservation with bias."""
        true_ate = 1.0
        cate_estimates = np.array([2.0, 2.5, 1.5, 2.0])  # Mean = 2.0

        score = ate_preservation_score(true_ate, cate_estimates)
        assert score == pytest.approx(1.0)  # |1.0 - 2.0| = 1.0


class TestCATER2Score:
    """Test cases for CATE R² score."""

    def test_cate_r2_perfect(self):
        """Test R² with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        r2 = cate_r2_score(y_true, y_pred)
        assert r2 == pytest.approx(1.0)

    def test_cate_r2_no_prediction(self):
        """Test R² when predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5])  # Mean prediction

        r2 = cate_r2_score(y_true, y_pred)
        assert r2 == pytest.approx(0.0, abs=1e-10)

    def test_cate_r2_worse_than_mean(self):
        """Test R² when predictions are worse than mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 1.0, 4.0, 1.0])  # Poor predictions

        r2 = cate_r2_score(y_true, y_pred)
        assert r2 < 0  # Negative R² indicates worse than mean


class TestRankWeightedATE:
    """Test cases for rank-weighted ATE."""

    def test_rank_weighted_ate_basic(self):
        """Test basic rank-weighted ATE calculation."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        treatments = np.array([0, 1, 0, 1, 0, 1])
        cate_estimates = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        # Top 33% should be indices [4, 5] (highest CATE estimates)
        ate_top = rank_weighted_ate(
            outcomes, treatments, cate_estimates, top_k_fraction=0.33
        )
        assert isinstance(ate_top, float)

    def test_rank_weighted_ate_no_variation(self):
        """Test when top-k group has no treatment variation."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0])
        treatments = np.array([1, 1, 0, 0])  # All treated in top group
        cate_estimates = np.array([3.0, 2.0, 1.0, 0.5])

        ate_top = rank_weighted_ate(
            outcomes, treatments, cate_estimates, top_k_fraction=0.5
        )
        assert ate_top == pytest.approx(0.0)  # Should return 0 when no variation


class TestQiniScore:
    """Test cases for Qini score."""

    def test_qini_score_basic(self):
        """Test basic Qini score calculation."""
        np.random.seed(42)
        n = 100
        outcomes = np.random.randn(n)
        treatments = np.random.binomial(1, 0.5, n)
        cate_estimates = np.random.randn(n)

        qini = qini_score(outcomes, treatments, cate_estimates)
        assert isinstance(qini, float)
        assert not np.isnan(qini)

    def test_qini_score_perfect_ranking(self):
        """Test Qini score with perfect treatment effect ranking."""
        # Create data where higher CATE estimates correspond to higher actual effects
        n = 50
        np.random.seed(123)

        cate_estimates = np.linspace(-1, 1, n)  # Increasing
        treatments = np.random.binomial(1, 0.5, n)

        # Outcomes that align with CATE estimates
        outcomes = cate_estimates * treatments + np.random.randn(n) * 0.1

        qini = qini_score(outcomes, treatments, cate_estimates)
        assert isinstance(qini, float)


class TestCalibrationScore:
    """Test cases for calibration score."""

    def test_calibration_score_basic(self):
        """Test basic calibration score calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])

        cal_metrics = calibration_score(y_true, y_pred)

        assert "calibration_error" in cal_metrics
        assert isinstance(cal_metrics["calibration_error"], (float, type(np.nan)))  # noqa: UP038

    def test_calibration_score_with_ci(self):
        """Test calibration score with confidence intervals."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8])
        confidence_intervals = np.array(
            [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]
        )

        cal_metrics = calibration_score(y_true, y_pred, confidence_intervals)

        assert "coverage_probability" in cal_metrics
        assert "mean_interval_width" in cal_metrics

        # All true values should be in intervals (perfect coverage)
        assert cal_metrics["coverage_probability"] == pytest.approx(1.0)

    def test_calibration_score_no_coverage(self):
        """Test calibration with no coverage."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8])
        confidence_intervals = np.array(
            [
                [2.0, 2.5],  # Doesn't contain 1.0
                [3.0, 3.5],  # Doesn't contain 2.0
                [4.0, 4.5],  # Doesn't contain 3.0
                [5.0, 5.5],  # Doesn't contain 4.0
            ]
        )

        cal_metrics = calibration_score(y_true, y_pred, confidence_intervals)
        assert cal_metrics["coverage_probability"] == pytest.approx(0.0)


class TestHTEEvaluator:
    """Test cases for HTEEvaluator class."""

    def test_hte_evaluator_init(self):
        """Test HTEEvaluator initialization."""
        evaluator = HTEEvaluator(confidence_level=0.90, n_bootstrap=50)
        assert evaluator.confidence_level == 0.90
        assert evaluator.n_bootstrap == 50

    def test_hte_evaluator_evaluate_basic(self):
        """Test basic evaluation functionality."""
        np.random.seed(42)
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.randn(50) * 0.1  # Add some noise

        evaluator = HTEEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)

        # Check that key metrics are present
        assert "pehe" in metrics
        assert "cate_r2" in metrics
        assert "cate_mse" in metrics
        assert "rank_correlation" in metrics

        # Values should be reasonable
        assert metrics["pehe"] >= 0
        assert -1 <= metrics["cate_r2"] <= 1
        assert metrics["cate_mse"] >= 0

    def test_hte_evaluator_evaluate_full(self):
        """Test evaluation with all optional parameters."""
        np.random.seed(42)
        n = 100

        y_true = np.random.randn(n)
        y_pred = y_true + np.random.randn(n) * 0.2
        outcomes = np.random.randn(n)
        treatments = np.random.binomial(1, 0.5, n)
        propensity_scores = np.random.beta(2, 2, n)  # Between 0 and 1
        confidence_intervals = np.column_stack([y_pred - 0.5, y_pred + 0.5])
        true_ate = 1.5

        evaluator = HTEEvaluator()
        metrics = evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            outcomes=outcomes,
            treatments=treatments,
            propensity_scores=propensity_scores,
            confidence_intervals=confidence_intervals,
            true_ate=true_ate,
        )

        # Check that all metrics are computed
        expected_keys = [
            "pehe",
            "cate_r2",
            "cate_mse",
            "ate_preservation",
            "overlap_weighted_mse",
            "policy_value",
            "rank_weighted_ate",
            "qini_score",
            "calibration_error",
            "coverage_probability",
            "mean_interval_width",
            "cate_variance_ratio",
            "rank_correlation",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_hte_evaluator_bootstrap_ci(self):
        """Test bootstrap confidence interval computation."""
        np.random.seed(42)
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.randn(50) * 0.1

        evaluator = HTEEvaluator(n_bootstrap=20, random_state=42)

        def metric_func(yt, yp):
            return np.mean((yt - yp) ** 2)

        lower, upper = evaluator.bootstrap_ci(y_true, y_pred, metric_func)

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper

    def test_hte_evaluator_plot_cate_scatter(self):
        """Test CATE scatter plot creation."""
        np.random.seed(42)
        y_true = np.random.randn(30)
        y_pred = y_true + np.random.randn(30) * 0.1

        evaluator = HTEEvaluator()

        try:
            import matplotlib.pyplot as plt

            ax = evaluator.plot_cate_scatter(y_true, y_pred)
            assert ax is not None
            plt.close("all")  # Clean up
        except ImportError:
            pytest.skip("Matplotlib not available")

    def test_hte_evaluator_plot_with_ci(self):
        """Test CATE scatter plot with confidence intervals."""
        np.random.seed(42)
        y_true = np.random.randn(30)
        y_pred = y_true + np.random.randn(30) * 0.1
        confidence_intervals = np.column_stack([y_pred - 0.3, y_pred + 0.3])

        evaluator = HTEEvaluator()

        try:
            import matplotlib.pyplot as plt

            ax = evaluator.plot_cate_scatter(y_true, y_pred, confidence_intervals)
            assert ax is not None
            plt.close("all")  # Clean up
        except ImportError:
            pytest.skip("Matplotlib not available")


@pytest.fixture
def synthetic_evaluation_data():
    """Create synthetic data for evaluation testing."""
    np.random.seed(456)
    n = 200

    # True treatment effects
    y_true = np.random.randn(n)

    # Predictions with some noise and bias
    y_pred = 0.8 * y_true + 0.2 + np.random.randn(n) * 0.3

    # Other data
    outcomes = np.random.randn(n)
    treatments = np.random.binomial(1, 0.5, n)
    propensity_scores = np.random.beta(2, 2, n)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "outcomes": outcomes,
        "treatments": treatments,
        "propensity_scores": propensity_scores,
    }


class TestHTEMetricsIntegration:
    """Integration tests for HTE evaluation metrics."""

    def test_full_evaluation_pipeline(self, synthetic_evaluation_data):
        """Test complete evaluation pipeline."""
        data = synthetic_evaluation_data

        evaluator = HTEEvaluator(confidence_level=0.95, random_state=42)

        metrics = evaluator.evaluate(
            y_true=data["y_true"],
            y_pred=data["y_pred"],
            outcomes=data["outcomes"],
            treatments=data["treatments"],
            propensity_scores=data["propensity_scores"],
        )

        # Verify all metrics are computed and reasonable
        assert metrics["pehe"] > 0
        assert -1 <= metrics["cate_r2"] <= 1
        assert (
            0 <= metrics["rank_correlation"] <= 1
        )  # Should be positive due to construction
        assert not np.isnan(metrics["policy_value"])
        assert not np.isnan(metrics["qini_score"])

    def test_metric_consistency(self, synthetic_evaluation_data):
        """Test consistency between related metrics."""
        data = synthetic_evaluation_data

        # Perfect predictions should give optimal metrics
        y_true = data["y_true"]
        y_pred_perfect = y_true.copy()

        evaluator = HTEEvaluator()

        metrics_perfect = evaluator.evaluate(y_true, y_pred_perfect)
        metrics_noisy = evaluator.evaluate(y_true, data["y_pred"])

        # Perfect predictions should have better metrics
        assert metrics_perfect["pehe"] < metrics_noisy["pehe"]
        assert metrics_perfect["cate_r2"] > metrics_noisy["cate_r2"]
        assert metrics_perfect["cate_mse"] < metrics_noisy["cate_mse"]


if __name__ == "__main__":
    pytest.main([__file__])
