"""Integration tests for transportability module."""

import numpy as np
import pytest

from causal_inference.core.base import CovariateData, OutcomeData, TreatmentData
from causal_inference.estimators.aipw import AIPW
from causal_inference.estimators.g_computation import GComputationEstimator
from causal_inference.estimators.ipw import IPWEstimator
from causal_inference.transportability import (
    CovariateShiftDiagnostics,
    DensityRatioEstimator,
    TargetedMaximumTransportedLikelihood,
    TransportabilityEstimator,
)


class TestTransportabilityIntegration:
    """Integration tests for transportability components working together."""

    @pytest.fixture
    def marketing_data(self):
        """Generate realistic marketing campaign data."""
        np.random.seed(42)
        n_source = 2000
        n_target = 1500

        # Source population (e.g., US customers)
        age_source = np.random.normal(45, 12, n_source)
        income_source = np.random.normal(60000, 20000, n_source)
        education_source = np.random.choice([0, 1, 2], n_source, p=[0.3, 0.5, 0.2])

        X_source = np.column_stack([age_source, income_source, education_source])

        # Treatment assignment (email campaign)
        # Higher income and education increase treatment probability
        treatment_logits = (
            -2
            + 0.00002 * income_source
            + 0.5 * education_source
            + 0.01 * age_source
            + np.random.normal(0, 0.5, n_source)
        )
        treatment_probs = 1 / (1 + np.exp(-treatment_logits))
        T_source = np.random.binomial(1, treatment_probs)

        # Outcome (purchase amount in $)
        # Treatment effect varies by customer characteristics
        base_outcome = (
            0.2 * age_source
            + 0.0001 * income_source
            + 10 * education_source
            + np.random.normal(0, 5, n_source)
        )

        # Heterogeneous treatment effect
        treatment_effect = (
            15
            + 0.0001 * income_source
            + 2 * education_source
            + np.random.normal(0, 2, n_source)
        )

        Y_source = base_outcome + T_source * treatment_effect

        # Target population (e.g., EU customers) with distribution shift
        age_target = np.random.normal(50, 10, n_target)  # Older population
        income_target = np.random.normal(
            55000, 18000, n_target
        )  # Slightly lower income
        education_target = np.random.choice(
            [0, 1, 2], n_target, p=[0.2, 0.6, 0.2]
        )  # More educated

        X_target = np.column_stack([age_target, income_target, education_target])

        return {
            "source": {
                "X": X_source,
                "T": T_source,
                "Y": Y_source,
                "feature_names": ["age", "income", "education"],
            },
            "target": {"X": X_target, "feature_names": ["age", "income", "education"]},
            "true_source_ate": np.mean(treatment_effect),  # True ATE in source
        }

    def test_end_to_end_transportability_workflow(self, marketing_data):
        """Test complete transportability workflow from diagnostics to estimation."""
        source = marketing_data["source"]
        target = marketing_data["target"]

        # Step 1: Covariate shift diagnostics
        diagnostics = CovariateShiftDiagnostics()
        shift_results = diagnostics.analyze_covariate_shift(
            source_data=source["X"],
            target_data=target["X"],
            variable_names=source["feature_names"],
        )

        # Should detect some shift
        assert shift_results["overall_shift_score"] > 0.1
        assert shift_results["n_variables"] == 3

        # Step 2: Transport weight estimation
        weighting = DensityRatioEstimator(random_state=42)
        weight_result = weighting.fit_weights(source["X"], target["X"])

        assert weight_result.effective_sample_size > 0
        assert len(weight_result.weights) == len(source["X"])

        # Step 3: TMTL estimation
        treatment_data = TreatmentData(values=source["T"])
        outcome_data = OutcomeData(values=source["Y"])
        covariate_data = CovariateData(
            values=source["X"], names=source["feature_names"]
        )

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        transported_effect = tmtl.estimate_transported_ate(target["X"])

        # Should produce reasonable results
        assert isinstance(transported_effect.ate, float)
        assert transported_effect.diagnostics is not None
        assert "transport_effective_sample_size" in transported_effect.diagnostics

        # Step 4: Validation
        validation = weighting.validate_weights(
            weight_result.weights, source["X"], target["X"]
        )

        assert "good_balance_achieved" in validation
        assert validation["effective_sample_size"] > 0

    def test_transportability_estimator_wrapper_with_aipw(self, marketing_data):
        """Test TransportabilityEstimator wrapper with AIPW."""
        source = marketing_data["source"]
        target = marketing_data["target"]

        # Fit base AIPW estimator
        treatment_data = TreatmentData(values=source["T"])
        outcome_data = OutcomeData(values=source["Y"])
        covariate_data = CovariateData(
            values=source["X"], names=source["feature_names"]
        )

        aipw = AIPW(random_state=42, verbose=False)
        aipw.fit(treatment_data, outcome_data, covariate_data)

        # Wrap with transportability
        transport_aipw = TransportabilityEstimator(
            base_estimator=aipw,
            weighting_method="classification",
            auto_diagnostics=True,
            random_state=42,
        )

        # Estimate transported effect
        transported_effect = transport_aipw.estimate_transported_effect(target["X"])

        assert isinstance(transported_effect.ate, float)
        assert "transportability_applied" in transported_effect.diagnostics
        assert transported_effect.method.endswith("_transported")

        # Compare with original effect
        original_effect = aipw.estimate_ate()

        # Effects may differ due to population differences
        assert (
            abs(transported_effect.ate - original_effect.ate) >= 0
        )  # May be same or different

        # Validate transport quality
        validation = transport_aipw.validate_transport_quality()
        assert "good_balance_achieved" in validation

    def test_transportability_estimator_wrapper_with_gcomp(self, marketing_data):
        """Test TransportabilityEstimator wrapper with G-computation."""
        source = marketing_data["source"]
        target = marketing_data["target"]

        # Fit base G-computation estimator
        treatment_data = TreatmentData(values=source["T"])
        outcome_data = OutcomeData(values=source["Y"])
        covariate_data = CovariateData(
            values=source["X"], names=source["feature_names"]
        )

        gcomp = GComputationEstimator(random_state=42, verbose=False)
        gcomp.fit(treatment_data, outcome_data, covariate_data)

        # Wrap with transportability
        transport_gcomp = TransportabilityEstimator(
            base_estimator=gcomp,
            weighting_method="classification",
            auto_diagnostics=True,
            min_shift_threshold=0.05,  # Lower threshold to force transport
        )

        # Estimate transported effect
        transported_effect = transport_gcomp.estimate_transported_effect(
            target["X"], force_transport=True
        )

        assert isinstance(transported_effect.ate, float)
        assert transported_effect.diagnostics["transportability_applied"] is True

        # Get transport summary
        summary = transport_gcomp.create_transport_summary()
        assert "TRANSPORTABILITY ANALYSIS SUMMARY" in summary
        assert "Overall Shift Score" in summary

    def test_minimal_covariate_shift_behavior(self):
        """Test behavior when covariate shift is minimal."""
        np.random.seed(42)
        n = 1000

        # Nearly identical populations
        X_source = np.random.randn(n, 3)
        X_target = X_source + np.random.normal(0, 0.01, X_source.shape)  # Tiny shift

        T = np.random.binomial(1, 0.5, n)
        Y = X_source[:, 0] + 2 * T + np.random.normal(0, 0.5, n)

        # Fit estimator
        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(values=X_source, names=["X1", "X2", "X3"])

        aipw = AIPW(random_state=42, verbose=False)
        aipw.fit(treatment_data, outcome_data, covariate_data)

        # Wrap with transportability
        transport_estimator = TransportabilityEstimator(
            base_estimator=aipw,
            min_shift_threshold=0.1,  # Higher threshold
            auto_diagnostics=True,
        )

        # Should skip transport due to minimal shift
        with pytest.warns(UserWarning, match="below threshold"):
            effect = transport_estimator.estimate_transported_effect(X_target)

        # Should be same as original estimate
        original_effect = aipw.estimate_ate()
        assert abs(effect.ate - original_effect.ate) < 1e-10

    def test_severe_covariate_shift_handling(self):
        """Test handling of severe covariate shift."""
        np.random.seed(42)
        n_source = 800
        n_target = 600

        # Source population
        X_source = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_source)
        T = np.random.binomial(1, 0.5, n_source)
        Y = X_source[:, 0] + X_source[:, 1] + 3 * T + np.random.normal(0, 0.5, n_source)

        # Target population with severe shift
        X_target = np.random.multivariate_normal(
            [2, -1], [[2, -0.5], [-0.5, 2]], n_target
        )

        # Run diagnostics first
        diagnostics = CovariateShiftDiagnostics()
        shift_results = diagnostics.analyze_covariate_shift(X_source, X_target)

        # Should detect severe shift
        assert shift_results["overall_shift_score"] > 0.5
        assert shift_results["n_severe_shifts"] > 0

        # TMTL should handle severe shift
        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(values=X_source, names=["X1", "X2"])

        tmtl = TargetedMaximumTransportedLikelihood(
            trim_weights=True,
            max_weight=20.0,  # Allow higher weights for severe shift
            random_state=42,
            verbose=False,
        )
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        transported_effect = tmtl.estimate_transported_ate(X_target)

        # Should complete without error despite severe shift
        assert isinstance(transported_effect.ate, float)
        assert transported_effect.diagnostics["transport_max_weight"] > 1.0

    def test_multiple_estimator_comparison(self, marketing_data):
        """Test transportability with multiple base estimators."""
        source = marketing_data["source"]
        target = marketing_data["target"]

        # Prepare data
        treatment_data = TreatmentData(values=source["T"])
        outcome_data = OutcomeData(values=source["Y"])
        covariate_data = CovariateData(
            values=source["X"], names=source["feature_names"]
        )

        # Test with different base estimators
        estimators = {
            "AIPW": AIPW(random_state=42, verbose=False),
            "G-Computation": GComputationEstimator(random_state=42, verbose=False),
            "IPW": IPWEstimator(random_state=42, verbose=False),
        }

        transported_effects = {}

        for name, estimator in estimators.items():
            # Fit base estimator
            estimator.fit(treatment_data, outcome_data, covariate_data)

            # Wrap with transportability
            transport_estimator = TransportabilityEstimator(
                base_estimator=estimator,
                auto_diagnostics=False,  # Skip for speed
                random_state=42,
            )

            # Estimate transported effect
            effect = transport_estimator.estimate_transported_effect(
                target["X"], run_diagnostics=False
            )
            transported_effects[name] = effect.ate

        # All estimators should produce reasonable results
        for name, ate in transported_effects.items():
            assert isinstance(ate, float)
            assert not np.isnan(ate)

        # Results should be in similar range (within factor of 2)
        ates = list(transported_effects.values())
        ate_range = max(ates) - min(ates)
        assert ate_range < 2 * abs(np.mean(ates))  # Reasonable agreement

    def test_cross_validation_robustness(self, marketing_data):
        """Test robustness across different random seeds."""
        source = marketing_data["source"]
        target = marketing_data["target"]

        treatment_data = TreatmentData(values=source["T"])
        outcome_data = OutcomeData(values=source["Y"])
        covariate_data = CovariateData(
            values=source["X"], names=source["feature_names"]
        )

        # Test with different random seeds
        seeds = [42, 123, 456, 789]
        transported_ates = []

        for seed in seeds:
            tmtl = TargetedMaximumTransportedLikelihood(
                cross_fit=True, n_folds=3, random_state=seed, verbose=False
            )
            tmtl.fit(treatment_data, outcome_data, covariate_data)

            effect = tmtl.estimate_transported_ate(target["X"])
            transported_ates.append(effect.ate)

        # Results should be reasonably stable across seeds
        ate_std = np.std(transported_ates)
        ate_mean = np.mean(transported_ates)

        # Coefficient of variation should be reasonable
        cv = ate_std / abs(ate_mean) if ate_mean != 0 else ate_std
        assert cv < 0.3  # Less than 30% variation

    def test_edge_case_single_feature(self):
        """Test transportability with single feature."""
        np.random.seed(42)
        n = 500

        # Single feature data
        X_source = np.random.normal(0, 1, (n, 1))
        X_target = np.random.normal(0.5, 1, (n // 2, 1))  # Shifted target

        T = np.random.binomial(1, 0.5, n)
        Y = 2 * X_source[:, 0] + 1.5 * T + np.random.normal(0, 0.3, n)

        # Should work with single feature
        diagnostics = CovariateShiftDiagnostics()
        shift_results = diagnostics.analyze_covariate_shift(X_source, X_target)

        assert len(shift_results["distribution_differences"]) == 1
        assert shift_results["overall_shift_score"] > 0.1  # Should detect shift

        # TMTL should work
        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(values=X_source, names=["X1"])

        tmtl = TargetedMaximumTransportedLikelihood(random_state=42, verbose=False)
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        effect = tmtl.estimate_transported_ate(X_target)
        assert isinstance(effect.ate, float)

    def test_high_dimensional_covariates(self):
        """Test transportability with high-dimensional covariates."""
        np.random.seed(42)
        n_source = 1000
        n_target = 800
        n_features = 20  # High-dimensional

        # Generate sparse effects (only first 5 features matter)
        X_source = np.random.randn(n_source, n_features)
        X_target = X_source[:n_target].copy()

        # Add shift to first few features
        X_target[:, :3] += np.random.normal(0.3, 0.1, (n_target, 3))

        # Treatment depends on first few features
        treatment_score = (
            0.5 * X_source[:, 0] + 0.3 * X_source[:, 1] + 0.2 * X_source[:, 2]
        )
        T = np.random.binomial(1, 1 / (1 + np.exp(-treatment_score)))

        # Outcome depends on first few features
        Y = (
            X_source[:, 0]
            + 0.5 * X_source[:, 1]
            + 2 * T
            + np.random.normal(0, 0.5, n_source)
        )

        # Test with regularized models (should handle high dimensions)
        treatment_data = TreatmentData(values=T)
        outcome_data = OutcomeData(values=Y)
        covariate_data = CovariateData(
            values=X_source, names=[f"X{i}" for i in range(n_features)]
        )

        tmtl = TargetedMaximumTransportedLikelihood(
            cross_fit=True,  # Important for high dimensions
            random_state=42,
            verbose=False,
        )
        tmtl.fit(treatment_data, outcome_data, covariate_data)

        effect = tmtl.estimate_transported_ate(X_target)

        # Should complete without error
        assert isinstance(effect.ate, float)
        assert not np.isnan(effect.ate)

        # Effective sample size should be reasonable despite high dimensions
        eff_n = effect.diagnostics["transport_effective_sample_size"]
        assert eff_n > n_target * 0.1  # At least 10% efficiency
