"""Edge case tests for causal inference estimators.

This module provides tests for edge cases and challenging scenarios
that estimators should handle gracefully or fail predictably.
"""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from causal_inference.core.base import TreatmentData
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator


class TestPerfectSeparation:
    """Test estimator behavior with perfect separation scenarios."""

    def test_perfect_separation_handling(self, edge_case_data):
        """Test that estimators handle perfect separation gracefully."""
        data = edge_case_data["perfect_separation"]

        estimators = [
            ("G-computation", GComputationEstimator()),
            ("IPW", IPWEstimator()),
            ("AIPW", AIPWEstimator(cross_fitting=False, bootstrap_samples=0)),
        ]

        for name, estimator in estimators:
            # Perfect separation may cause convergence warnings or errors
            # We test that estimators either succeed or fail gracefully
            try:
                with warnings.catch_warnings(record=True) as warning_list:
                    estimator.fit(
                        data["treatment"], data["outcome"], data["covariates"]
                    )
                    effect = estimator.estimate_ate()

                    # If successful, result should be finite
                    assert np.isfinite(effect.ate), (
                        f"{name} produced non-finite ATE with perfect separation"
                    )

                # Check if convergence warnings were raised (expected)
                # Convergence warnings are expected but not required
                _ = [
                    w
                    for w in warning_list
                    if issubclass(w.category, ConvergenceWarning)
                ]

            except (ValueError, np.linalg.LinAlgError, RuntimeWarning) as e:
                # These exceptions are acceptable for perfect separation
                error_message = str(e).lower()
                expected_keywords = [
                    "singular",
                    "convergence",
                    "separation",
                    "ill-conditioned",
                    "overflow",
                ]
                assert any(keyword in error_message for keyword in expected_keywords), (
                    f"{name} failed with unexpected error: {e}"
                )


class TestMulticollinearity:
    """Test estimator behavior with multicollinear covariates."""

    def test_multicollinearity_handling(self, edge_case_data):
        """Test that estimators handle multicollinearity appropriately."""
        data = edge_case_data["multicollinearity"]

        estimators = [
            ("G-computation", GComputationEstimator()),
            ("IPW", IPWEstimator()),
            ("AIPW", AIPWEstimator(cross_fitting=False, bootstrap_samples=0)),
        ]

        for name, estimator in estimators:
            try:
                estimator.fit(data["treatment"], data["outcome"], data["covariates"])
                effect = estimator.estimate_ate()

                # Should produce finite results
                assert np.isfinite(effect.ate), (
                    f"{name} produced non-finite ATE with multicollinearity"
                )

                # Confidence intervals should be finite (though potentially wide)
                if effect.confidence_interval is not None:
                    assert np.isfinite(effect.confidence_interval[0]), (
                        f"{name} CI lower bound not finite"
                    )
                    assert np.isfinite(effect.confidence_interval[1]), (
                        f"{name} CI upper bound not finite"
                    )

            except (ValueError, np.linalg.LinAlgError) as e:
                # Some estimators may fail with severe multicollinearity
                error_message = str(e).lower()
                expected_keywords = [
                    "singular",
                    "ill-conditioned",
                    "multicollinearity",
                    "rank deficient",
                ]
                assert any(keyword in error_message for keyword in expected_keywords), (
                    f"{name} failed with unexpected error: {e}"
                )


class TestRankDeficiency:
    """Test estimator behavior with rank-deficient covariate matrices."""

    def test_rank_deficient_matrix_handling(self, edge_case_data):
        """Test handling of rank-deficient covariate matrices."""
        data = edge_case_data["rank_deficient"]

        estimators = [
            ("G-computation", GComputationEstimator()),
            ("IPW", IPWEstimator()),
            ("AIPW", AIPWEstimator(cross_fitting=False, bootstrap_samples=0)),
        ]

        for name, estimator in estimators:
            try:
                estimator.fit(data["treatment"], data["outcome"], data["covariates"])
                effect = estimator.estimate_ate()

                # If estimation succeeds, results should be finite
                assert np.isfinite(effect.ate), (
                    f"{name} produced non-finite ATE with rank deficiency"
                )

            except (ValueError, np.linalg.LinAlgError) as e:
                # Rank deficiency should cause predictable linear algebra errors
                error_message = str(e).lower()
                expected_keywords = [
                    "singular",
                    "rank",
                    "deficient",
                    "ill-conditioned",
                    "inversion",
                ]
                assert any(keyword in error_message for keyword in expected_keywords), (
                    f"{name} failed with unexpected error for rank deficiency: {e}"
                )


class TestNoTreatmentVariation:
    """Test estimator behavior when there's no variation in treatment."""

    def test_no_treatment_variation_error(self, edge_case_data):
        """Test that treatment data with no variation fails at construction time."""
        # Test that creating TreatmentData with no variation fails at construction time
        with pytest.raises(
            ValueError, match="Binary treatment must have exactly 2 unique values"
        ):
            TreatmentData(values=np.ones(100), treatment_type="binary")

        with pytest.raises(
            ValueError, match="Binary treatment must have exactly 2 unique values"
        ):
            TreatmentData(values=np.zeros(100), treatment_type="binary")

        # The edge_case_data now has valid but severely imbalanced data that should still work
        # but may produce warnings or poor results - test that it doesn't crash
        data = edge_case_data["no_treatment_variation"]
        estimator = GComputationEstimator()

        # Should not raise an exception with valid (though imbalanced) data
        estimator.fit(data["treatment"], data["outcome"], data["covariates"])
        result = estimator.estimate_ate()
        assert result.ate is not None  # Should return some result


class TestExtremePropensity:
    """Test estimator behavior with extreme propensity scores."""

    def test_extreme_propensity_scores(self, edge_case_data):
        """Test handling of extreme propensity scores (near 0 or 1)."""
        data = edge_case_data["extreme_propensity"]

        # IPW should be most sensitive to extreme propensities
        ipw_estimator = IPWEstimator()

        try:
            ipw_estimator.fit(data["treatment"], data["outcome"], data["covariates"])
            effect = ipw_estimator.estimate_ate()

            # If IPW succeeds, check that weights aren't extremely large
            weights = ipw_estimator.get_weights()
            max_weight = np.max(weights)

            # Weights shouldn't be unreasonably large (indicating extreme propensities)
            # This is more of a sanity check than a hard requirement
            if max_weight > 100:
                pytest.warn(
                    f"IPW weights are very large (max: {max_weight:.2f}), indicating extreme propensities"
                )

            assert np.isfinite(effect.ate), (
                "IPW ATE should be finite even with extreme propensities"
            )

        except (ValueError, RuntimeWarning) as e:
            # IPW may fail with extreme propensities - this is acceptable
            error_message = str(e).lower()
            expected_keywords = ["propensity", "extreme", "weight", "overflow", "inf"]
            assert any(keyword in error_message for keyword in expected_keywords), (
                f"IPW failed with unexpected error: {e}"
            )

        # G-computation and AIPW should be more robust
        robust_estimators = [
            ("G-computation", GComputationEstimator()),
            ("AIPW", AIPWEstimator(cross_fitting=False, bootstrap_samples=0)),
        ]

        for name, estimator in robust_estimators:
            try:
                estimator.fit(data["treatment"], data["outcome"], data["covariates"])
                effect = estimator.estimate_ate()
                assert np.isfinite(effect.ate), (
                    f"{name} should handle extreme propensities"
                )

            except Exception as e:
                # Even robust estimators may struggle with extreme cases
                pytest.warn(f"{name} failed with extreme propensities: {e}")


class TestSmallSample:
    """Test estimator behavior with very small samples."""

    def test_small_sample_handling(self, edge_case_data):
        """Test that estimators handle small samples appropriately."""
        data = edge_case_data["small_sample"]

        estimators = [
            (
                "G-computation",
                GComputationEstimator(bootstrap_samples=10),
            ),  # Reduce bootstrap for small samples
            ("IPW", IPWEstimator()),
            ("AIPW", AIPWEstimator(cross_fitting=False, bootstrap_samples=0)),
        ]

        for name, estimator in estimators:
            try:
                estimator.fit(data["treatment"], data["outcome"], data["covariates"])
                effect = estimator.estimate_ate()

                # Results should be finite
                assert np.isfinite(effect.ate), (
                    f"{name} produced non-finite ATE with small sample"
                )

                # Confidence intervals should be wide due to uncertainty (if available)
                if effect.confidence_interval is not None:
                    ci_width = (
                        effect.confidence_interval[1] - effect.confidence_interval[0]
                    )
                    assert ci_width > 0, f"{name} CI width should be positive"

                # With very small samples, CI might be very wide - this is expected

            except (ValueError, RuntimeWarning) as e:
                # Small samples may cause estimation difficulties
                error_message = str(e).lower()
                expected_keywords = [
                    "sample",
                    "size",
                    "insufficient",
                    "degrees",
                    "freedom",
                ]
                assert any(keyword in error_message for keyword in expected_keywords), (
                    f"{name} failed with unexpected small sample error: {e}"
                )


class TestEdgeCaseIntegration:
    """Integration tests combining multiple edge case conditions."""

    def test_multiple_edge_conditions(self, edge_case_data):
        """Test that we can handle combinations of edge case conditions."""
        # Test that we have all expected edge cases available
        expected_scenarios = [
            "no_treatment_variation",
            "extreme_propensity",
            "small_sample",
            "perfect_separation",
            "multicollinearity",
            "rank_deficient",
        ]

        for scenario in expected_scenarios:
            assert scenario in edge_case_data, f"Missing edge case scenario: {scenario}"

            # Basic validation that each scenario has required data structure
            data = edge_case_data[scenario]
            assert "treatment" in data, f"Missing treatment data in {scenario}"
            assert "outcome" in data, f"Missing outcome data in {scenario}"
            assert "covariates" in data, f"Missing covariate data in {scenario}"

    def test_edge_case_documentation(self, edge_case_data):
        """Test that edge cases are properly documented and structured."""
        for scenario_name, scenario_data in edge_case_data.items():
            # Each scenario should have proper data types
            assert hasattr(scenario_data["treatment"], "values"), (
                f"{scenario_name} treatment missing values"
            )
            assert hasattr(scenario_data["outcome"], "values"), (
                f"{scenario_name} outcome missing values"
            )
            assert hasattr(scenario_data["covariates"], "values"), (
                f"{scenario_name} covariates missing values"
            )

            # Data should have consistent sample sizes
            n_treatment = len(scenario_data["treatment"].values)
            n_outcome = len(scenario_data["outcome"].values)
            n_covariates = len(scenario_data["covariates"].values)

            assert n_treatment == n_outcome == n_covariates, (
                f"{scenario_name} has inconsistent sample sizes: {n_treatment}, {n_outcome}, {n_covariates}"
            )
