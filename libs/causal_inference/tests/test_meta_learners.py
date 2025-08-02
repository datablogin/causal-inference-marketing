"""Tests for meta-learners (T/S/X/R-learner) for CATE estimation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_inference.core.base import (
    CovariateData,
    DataValidationError,
    EstimationError,
    OutcomeData,
    TreatmentData,
)
from causal_inference.estimators.meta_learners import (
    CATEResult,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
)


class TestMetaLearnersCommon:
    """Common tests for all meta-learners."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with heterogeneous treatment effects."""
        np.random.seed(42)
        n = 1000

        # Covariates
        X = np.random.randn(n, 5)

        # Propensity score (depends on X)
        propensity = 1 / (1 + np.exp(-0.5 * X[:, 0] - 0.3 * X[:, 1]))
        T = np.random.binomial(1, propensity)

        # Heterogeneous treatment effect (CATE depends on X)
        tau = 1 + 0.5 * X[:, 0] + 0.3 * X[:, 2] + 0.2 * X[:, 0] * X[:, 2]

        # Outcome
        Y0 = 2 + X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5
        Y1 = Y0 + tau
        Y = T * Y1 + (1 - T) * Y0

        return {
            "X": X,
            "T": T,
            "Y": Y,
            "tau": tau,
            "propensity": propensity,
        }

    @pytest.fixture
    def prepared_data(self, synthetic_data):
        """Prepare data in the format expected by estimators."""
        return (
            TreatmentData(values=synthetic_data["T"], name="treatment"),
            OutcomeData(values=synthetic_data["Y"], name="outcome"),
            CovariateData(
                values=synthetic_data["X"], names=[f"X{i}" for i in range(5)]
            ),
        )

    def test_binary_treatment_validation(self, prepared_data):
        """Test that meta-learners require binary treatment."""
        treatment, outcome, covariates = prepared_data

        # Create multi-valued treatment
        treatment_multi = TreatmentData(
            values=np.random.randint(0, 3, size=len(treatment.values)),
            name="treatment",
        )

        # All meta-learners should raise error
        for LearnerClass in [SLearner, TLearner, XLearner, RLearner]:
            learner = LearnerClass()
            with pytest.raises(ValueError, match="binary treatment"):
                learner.fit(treatment_multi, outcome, covariates)

    def test_data_validation(self, prepared_data):
        """Test input data validation."""
        treatment, outcome, covariates = prepared_data

        # Mismatched dimensions
        outcome_short = OutcomeData(values=outcome.values[:-10])

        for LearnerClass in [SLearner, TLearner, XLearner, RLearner]:
            learner = LearnerClass()
            with pytest.raises(DataValidationError):
                learner.fit(treatment, outcome_short, covariates)

    def test_estimate_before_fit(self):
        """Test that estimation fails before fitting."""
        X = np.random.randn(10, 3)

        for LearnerClass in [SLearner, TLearner, XLearner, RLearner]:
            learner = LearnerClass()
            with pytest.raises(EstimationError, match="not fitted"):
                learner.estimate_cate(X)


class TestSLearner:
    """Tests specific to S-learner."""

    def test_slearner_basic(self, prepared_data, synthetic_data):
        """Test basic S-learner functionality."""
        treatment, outcome, covariates = prepared_data

        # Fit S-learner
        slearner = SLearner(
            base_learner=RandomForestRegressor(n_estimators=50, random_state=42)
        )
        slearner.fit(treatment, outcome, covariates)

        # Estimate CATE
        cate_estimates = slearner.estimate_cate(synthetic_data["X"])

        # Check shape
        assert cate_estimates.shape == (len(synthetic_data["X"]),)

        # Check ATE
        result = slearner.estimate_ate()
        assert isinstance(result, CATEResult)
        assert hasattr(result, "ate")
        assert hasattr(result, "confidence_interval")

    def test_slearner_with_propensity(self, prepared_data, synthetic_data):
        """Test S-learner with propensity score feature."""
        treatment, outcome, covariates = prepared_data

        # Fit with propensity scores
        slearner = SLearner(
            base_learner=RandomForestRegressor(n_estimators=50, random_state=42),
            include_propensity=True,
        )
        slearner.fit(treatment, outcome, covariates)

        # Should have fitted propensity model
        assert slearner._propensity_model is not None

        # Estimate CATE
        cate_estimates = slearner.estimate_cate(synthetic_data["X"])
        assert cate_estimates.shape == (len(synthetic_data["X"]),)

    def test_slearner_different_base_learners(self, prepared_data):
        """Test S-learner with different base learners."""
        treatment, outcome, covariates = prepared_data

        base_learners = [
            LinearRegression(),
            GradientBoostingRegressor(n_estimators=20, random_state=42),
        ]

        for base_learner in base_learners:
            slearner = SLearner(base_learner=base_learner)
            slearner.fit(treatment, outcome, covariates)
            result = slearner.estimate_ate()
            assert isinstance(result.ate, int | float)


class TestTLearner:
    """Tests specific to T-learner."""

    def test_tlearner_basic(self, prepared_data, synthetic_data):
        """Test basic T-learner functionality."""
        treatment, outcome, covariates = prepared_data

        # Fit T-learner
        tlearner = TLearner(
            base_learner=RandomForestRegressor(n_estimators=50, random_state=42)
        )
        tlearner.fit(treatment, outcome, covariates)

        # Check that two models were fitted
        assert tlearner._model_treated is not None
        assert tlearner._model_control is not None

        # Estimate CATE
        cate_estimates = tlearner.estimate_cate(synthetic_data["X"])
        assert cate_estimates.shape == (len(synthetic_data["X"]),)

        # Check ATE
        result = tlearner.estimate_ate()
        assert isinstance(result, CATEResult)

    def test_tlearner_treatment_imbalance(self, synthetic_data):
        """Test T-learner with imbalanced treatment groups."""
        # Create highly imbalanced treatment
        n = len(synthetic_data["T"])
        T_imbalanced = np.zeros(n)
        T_imbalanced[:50] = 1  # Only 5% treated

        treatment = TreatmentData(values=T_imbalanced)
        outcome = OutcomeData(values=synthetic_data["Y"])
        covariates = CovariateData(values=synthetic_data["X"])

        tlearner = TLearner()
        tlearner.fit(treatment, outcome, covariates)

        # Should still work
        cate_estimates = tlearner.estimate_cate(synthetic_data["X"])
        assert cate_estimates.shape == (n,)


class TestXLearner:
    """Tests specific to X-learner."""

    def test_xlearner_basic(self, prepared_data, synthetic_data):
        """Test basic X-learner functionality."""
        treatment, outcome, covariates = prepared_data

        # Fit X-learner
        xlearner = XLearner(
            base_learner=RandomForestRegressor(n_estimators=50, random_state=42),
            propensity_learner=LogisticRegression(random_state=42),
        )
        xlearner.fit(treatment, outcome, covariates)

        # Check that all models were fitted
        assert xlearner._model_treated is not None
        assert xlearner._model_control is not None
        assert xlearner._tau_treated is not None
        assert xlearner._tau_control is not None
        assert xlearner._propensity_model is not None

        # Estimate CATE
        cate_estimates = xlearner.estimate_cate(synthetic_data["X"])
        assert cate_estimates.shape == (len(synthetic_data["X"]),)

        # Check ATE
        result = xlearner.estimate_ate()
        assert isinstance(result, CATEResult)

    def test_xlearner_propensity_weighting(self, prepared_data, synthetic_data):
        """Test that X-learner uses propensity weighting correctly."""
        treatment, outcome, covariates = prepared_data

        xlearner = XLearner()
        xlearner.fit(treatment, outcome, covariates)

        # Get CATE estimates
        cate_estimates = xlearner.estimate_cate(synthetic_data["X"])

        # Manually compute to verify
        propensity = xlearner._propensity_model.predict_proba(synthetic_data["X"])[:, 1]
        tau_1 = xlearner._tau_treated.predict(synthetic_data["X"])
        tau_0 = xlearner._tau_control.predict(synthetic_data["X"])
        expected_cate = propensity * tau_0 + (1 - propensity) * tau_1

        np.testing.assert_allclose(cate_estimates, expected_cate, rtol=1e-10)


class TestRLearner:
    """Tests specific to R-learner."""

    def test_rlearner_basic(self, prepared_data, synthetic_data):
        """Test basic R-learner functionality."""
        treatment, outcome, covariates = prepared_data

        # Fit R-learner
        rlearner = RLearner(
            base_learner=RandomForestRegressor(n_estimators=50, random_state=42),
            n_folds=3,  # Fewer folds for faster test
        )
        rlearner.fit(treatment, outcome, covariates)

        # Check models were fitted
        assert rlearner._outcome_model is not None
        assert rlearner._propensity_model is not None
        assert rlearner._cate_model is not None

        # Estimate CATE
        cate_estimates = rlearner.estimate_cate(synthetic_data["X"])
        assert cate_estimates.shape == (len(synthetic_data["X"]),)

        # Check ATE
        result = rlearner.estimate_ate()
        assert isinstance(result, CATEResult)

    def test_rlearner_regularization(self, prepared_data):
        """Test R-learner with different regularization parameters."""
        treatment, outcome, covariates = prepared_data

        reg_params = [0.001, 0.01, 0.1]

        for reg_param in reg_params:
            rlearner = RLearner(
                regularization_param=reg_param,
                n_folds=2,
            )
            rlearner.fit(treatment, outcome, covariates)
            result = rlearner.estimate_ate()
            assert isinstance(result.ate, int | float)

    def test_rlearner_insufficient_variation(self, synthetic_data):
        """Test R-learner with insufficient treatment variation."""
        # Create almost deterministic treatment
        T_deterministic = (synthetic_data["X"][:, 0] > 0).astype(int)

        treatment = TreatmentData(values=T_deterministic)
        outcome = OutcomeData(values=synthetic_data["Y"])
        covariates = CovariateData(values=synthetic_data["X"])

        rlearner = RLearner(n_folds=2)

        # Should raise error due to insufficient variation
        with pytest.raises(EstimationError, match="variation in treatment residuals"):
            rlearner.fit(treatment, outcome, covariates)


class TestCATEResult:
    """Tests for CATEResult class."""

    def test_cate_result_initialization(self):
        """Test CATEResult initialization."""
        cate_estimates = np.random.randn(100)
        ate = np.mean(cate_estimates)
        ci = (ate - 0.1, ate + 0.1)

        result = CATEResult(
            ate=ate,
            confidence_interval=ci,
            cate_estimates=cate_estimates,
            method="Test",
        )

        assert result.ate == ate
        assert result.confidence_interval == ci
        assert np.array_equal(result.cate_estimates, cate_estimates)
        assert result.method == "Test"

    def test_cate_result_plotting(self):
        """Test CATE distribution plotting."""
        cate_estimates = np.random.randn(1000)
        ate = np.mean(cate_estimates)

        result = CATEResult(
            ate=ate,
            confidence_interval=(ate - 0.1, ate + 0.1),
            cate_estimates=cate_estimates,
        )

        # Test that plot method exists and runs without error
        try:
            import matplotlib.pyplot as plt

            ax = result.plot_cate_distribution(kde=True)
            assert ax is not None
            plt.close()
        except ImportError:
            # Skip plotting test if matplotlib not available
            pass


class TestMetaLearnersIntegration:
    """Integration tests comparing meta-learners."""

    def test_all_learners_consistent_ate(self, prepared_data):
        """Test that all learners give reasonably consistent ATE estimates."""
        treatment, outcome, covariates = prepared_data

        learners = [
            SLearner(
                base_learner=RandomForestRegressor(n_estimators=50, random_state=42)
            ),
            TLearner(
                base_learner=RandomForestRegressor(n_estimators=50, random_state=42)
            ),
            XLearner(
                base_learner=RandomForestRegressor(n_estimators=50, random_state=42)
            ),
            RLearner(
                base_learner=RandomForestRegressor(n_estimators=50, random_state=42),
                n_folds=3,
            ),
        ]

        ates = []
        for learner in learners:
            learner.fit(treatment, outcome, covariates)
            result = learner.estimate_ate()
            ates.append(result.ate)

        # ATEs should be reasonably close (within 0.5 of each other)
        ate_range = max(ates) - min(ates)
        assert ate_range < 0.5, f"ATE estimates vary too much: {ates}"

    def test_learners_with_pandas_input(self, synthetic_data):
        """Test that learners work with pandas DataFrames."""
        # Create pandas data
        df = pd.DataFrame(synthetic_data["X"], columns=[f"X{i}" for i in range(5)])
        df["treatment"] = synthetic_data["T"]
        df["outcome"] = synthetic_data["Y"]

        treatment = TreatmentData(values=df["treatment"])
        outcome = OutcomeData(values=df["outcome"])
        covariates = CovariateData(values=df[[f"X{i}" for i in range(5)]])

        # Test each learner
        for LearnerClass in [SLearner, TLearner, XLearner, RLearner]:
            learner = LearnerClass()
            learner.fit(treatment, outcome, covariates)

            # Estimate on DataFrame
            cate_df = learner.estimate_cate(df[[f"X{i}" for i in range(5)]])
            assert len(cate_df) == len(df)

    def test_heterogeneous_effects_recovery(self, synthetic_data):
        """Test that learners can recover heterogeneous treatment effects."""
        # Create data with strong heterogeneity
        n = 2000
        X = np.random.randn(n, 3)

        # Treatment assignment
        propensity = 0.5
        T = np.random.binomial(1, propensity, n)

        # Strong heterogeneous effect: positive for X[:, 0] > 0, negative otherwise
        tau = np.where(X[:, 0] > 0, 2.0, -1.0)

        # Outcome
        Y = 1 + X[:, 1] + T * tau + np.random.randn(n) * 0.5

        treatment = TreatmentData(values=T)
        outcome = OutcomeData(values=Y)
        covariates = CovariateData(values=X)

        # Fit learners
        learners = {
            "S-Learner": SLearner(base_learner=RandomForestRegressor(n_estimators=100)),
            "T-Learner": TLearner(base_learner=RandomForestRegressor(n_estimators=100)),
            "X-Learner": XLearner(base_learner=RandomForestRegressor(n_estimators=100)),
            "R-Learner": RLearner(
                base_learner=RandomForestRegressor(n_estimators=100), n_folds=3
            ),
        }

        for name, learner in learners.items():
            learner.fit(treatment, outcome, covariates)
            cate_estimates = learner.estimate_cate(X)

            # Check that CATE estimates capture heterogeneity
            cate_positive = cate_estimates[X[:, 0] > 0].mean()
            cate_negative = cate_estimates[X[:, 0] <= 0].mean()

            # Should recover that effect is positive for X[:, 0] > 0
            assert cate_positive > 0.5, f"{name} failed to recover positive effect"
            assert cate_negative < 0.5, f"{name} failed to recover negative effect"
            assert cate_positive > cate_negative, (
                f"{name} failed to capture heterogeneity"
            )
