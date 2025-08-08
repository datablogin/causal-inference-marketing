"""Neyman-orthogonal score functions for Double Machine Learning.

This module implements various orthogonal moment conditions following
Chernozhukov et al. (2018) Double ML theory, providing enhanced
finite-sample properties and alternative estimation strategies.
"""
# ruff: noqa: N803

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr


class OrthogonalMoments:
    """Factory class for Neyman-orthogonal score functions in Double ML.

    This class provides various orthogonal moment functions that can be used
    in the DoublyRobustMLEstimator, each offering different theoretical
    properties and suitability for different data scenarios.

    The orthogonal moment functions implemented include:
    - AIPW: Augmented Inverse Probability Weighting
    - Orthogonal: Basic Neyman-orthogonal scores
    - Partialling Out: Interactive moment conditions with partialling out
    - Interactive IV: Instrumental variable orthogonal scores
    - PLR: Partially Linear Regression orthogonal moments
    - PLIV: Partially Linear IV orthogonal moments

    Each moment function ensures the orthogonality condition:
    E[ψ(W,θ,η̂(W)) | η(W)] ≈ 0
    where W are observed variables, θ is the target parameter,
    and η̂ are estimated nuisance functions.
    """

    @staticmethod
    def aipw(
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Compute AIPW orthogonal scores.

        The AIPW (Augmented Inverse Probability Weighting) score function:
        ψ_AIPW(Y,A,X) = μ₁(X) - μ₀(X) + A(Y-μ₁(X))/g(X) - (1-A)(Y-μ₀(X))/(1-g(X))

        This is doubly robust: consistent if either outcome model OR propensity model
        is correctly specified.

        Args:
            nuisance_estimates: Dict containing 'mu1', 'mu0', 'propensity_scores'
            treatment: Binary treatment vector (0/1)
            outcome: Continuous outcome vector
            **kwargs: Additional arguments (unused)

        Returns:
            Array of AIPW scores for each observation
        """
        mu1 = nuisance_estimates["mu1"]
        mu0 = nuisance_estimates["mu0"]
        g = nuisance_estimates["propensity_scores"]

        # AIPW estimator
        aipw_scores = (
            mu1
            - mu0
            + treatment * (outcome - mu1) / g
            - (1 - treatment) * (outcome - mu0) / (1 - g)
        )

        return aipw_scores

    @staticmethod
    def orthogonal(
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Compute basic orthogonal (Neyman-orthogonal) scores.

        The basic orthogonal score function:
        ψ(Y,A,X) = (A-g(X))(Y-μ(A,X)) + μ₁(X) - μ₀(X)

        This provides robustness to misspecification in nuisance functions
        through the orthogonality condition.

        Args:
            nuisance_estimates: Dict containing combined model predictions
            treatment: Binary treatment vector (0/1)
            outcome: Continuous outcome vector
            **kwargs: Additional arguments (unused)

        Returns:
            Array of orthogonal scores for each observation
        """
        mu1 = nuisance_estimates["mu1_combined"]
        mu0 = nuisance_estimates["mu0_combined"]
        mu_observed = nuisance_estimates["mu_observed"]
        g = nuisance_estimates["propensity_scores"]

        # Orthogonal moment function
        orthogonal_scores = (treatment - g) * (outcome - mu_observed) + mu1 - mu0

        return orthogonal_scores

    @staticmethod
    def partialling_out(
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Compute partialling out orthogonal scores.

        The partialling out score function:
        ψ_PO(Y,D,X) = θ(D - m(X)) + (Y - l(X)) - θ(D - m(X))
        where m(X) = E[D|X], l(X) = E[Y|X]

        This approach partials out the effect of X on both Y and D,
        then estimates the direct effect of D on the residualized Y.

        Args:
            nuisance_estimates: Dict containing propensity scores and outcome predictions
            treatment: Binary treatment vector (0/1)
            outcome: Continuous outcome vector
            **kwargs: Additional arguments (unused)

        Returns:
            Array of partialling out scores for each observation
        """
        # Extract nuisance parameters
        g = nuisance_estimates["propensity_scores"]  # m(X) = E[D|X]
        mu_observed = nuisance_estimates["mu_observed"]  # l(X) = E[Y|X]

        # Residualize treatment and outcome
        treatment_residual = treatment - g
        outcome_residual = outcome - mu_observed

        # Simple method: estimate θ using residualized variables
        # θ̂ = E[(Y - l(X))(D - m(X))] / E[(D - m(X))²]
        denominator = np.mean(treatment_residual**2)

        if abs(denominator) < 1e-8:
            warnings.warn(
                "Weak treatment variation after partialling out. "
                "Partialling out method may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Partialling out scores: θ̂(D - m(X)) + (Y - l(X)) - θ̂(D - m(X))
        # This simplifies to: (Y - l(X))
        partialling_out_scores = outcome_residual

        return partialling_out_scores

    @staticmethod
    def interactive_iv(
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        instrument: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Compute interactive IV orthogonal scores.

        The interactive IV score function (when instrument Z is available):
        ψ_IIV(Y,D,X,Z) = (Z - r(X))(Y - μ(D,X)) + (D - m(X))(Y - g(D,X))
        where r(X) = E[Z|X]

        This approach uses instrumental variables for identification when available.
        If no instrument is provided, falls back to treatment as its own instrument.

        Args:
            nuisance_estimates: Dict containing various model predictions
            treatment: Binary treatment vector (0/1)
            outcome: Continuous outcome vector
            instrument: Instrumental variable (optional)
            **kwargs: Additional arguments (unused)

        Returns:
            Array of interactive IV scores for each observation
        """
        if instrument is None:
            # Fallback: use treatment as its own instrument (reduced to orthogonal)
            warnings.warn(
                "No instrument provided for interactive_iv. "
                "Falling back to basic orthogonal moment function.",
                UserWarning,
                stacklevel=2,
            )
            return OrthogonalMoments.orthogonal(nuisance_estimates, treatment, outcome)

        # Extract nuisance parameters
        g = nuisance_estimates["propensity_scores"]  # m(X) = E[D|X]
        mu_observed = nuisance_estimates["mu_observed"]  # μ(D,X) = E[Y|D,X]

        # For instrument residualization, we'd need E[Z|X]
        # As approximation, use mean of instrument by treatment/covariate groups
        # This is a simplified implementation - in practice would need proper modeling
        r_x = np.mean(instrument)  # Simple approximation: r(X) ≈ E[Z]

        # Interactive IV scores
        instrument_residual = instrument - r_x
        treatment_residual = treatment - g
        outcome_mu_residual = outcome - mu_observed

        # Combined score: (Z - r(X))(Y - μ(D,X)) + (D - m(X))(Y - μ(D,X))
        interactive_iv_scores = (
            instrument_residual * outcome_mu_residual
            + treatment_residual * outcome_mu_residual
        )

        return interactive_iv_scores

    @staticmethod
    def plr(
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Compute Partially Linear Regression orthogonal scores.

        The PLR score function for the partially linear model:
        Y = θD + g(X) + ε

        ψ_PLR(Y,D,X) = (Y - μ₀(X) - θ(D - m(X))) * (D - m(X))
        where m(X) = E[D|X], μ₀(X) = E[Y|D=0,X]

        This is suitable when the outcome model is partially linear in treatment.

        Args:
            nuisance_estimates: Dict containing model predictions
            treatment: Binary treatment vector (0/1)
            outcome: Continuous outcome vector
            **kwargs: Additional arguments (unused)

        Returns:
            Array of PLR scores for each observation
        """
        # Extract nuisance parameters
        g = nuisance_estimates["propensity_scores"]  # m(X) = E[D|X]
        mu0 = nuisance_estimates["mu0"]  # μ₀(X) = E[Y|D=0,X]

        # Residualize treatment
        treatment_residual = treatment - g

        # Estimate θ using the partially linear structure
        # θ̂ = E[(Y - μ₀(X))(D - m(X))] / E[(D - m(X))²]
        y_residual = outcome - mu0
        numerator = np.mean(y_residual * treatment_residual)
        denominator = np.mean(treatment_residual**2)

        if abs(denominator) < 1e-8:
            warnings.warn(
                "Weak treatment variation for PLR method. Estimate may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            theta_hat = 0.0
        else:
            theta_hat = numerator / denominator

        # PLR scores: (Y - μ₀(X) - θ̂(D - m(X))) * (D - m(X))
        plr_residual = outcome - mu0 - theta_hat * treatment_residual
        plr_scores = plr_residual * treatment_residual

        return plr_scores

    @staticmethod
    def pliv(
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        instrument: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Compute Partially Linear IV orthogonal scores.

        The PLIV score function for the partially linear IV model:
        Y = θD + g(X) + ε, with instrument Z

        ψ_PLIV(Y,D,X,Z) = (Y - μ₀(X) - θ(D - m(X))) * (Z - r(X))
        where r(X) = E[Z|X]

        This extends PLR to handle endogenous treatment using instruments.

        Args:
            nuisance_estimates: Dict containing model predictions
            treatment: Binary treatment vector (0/1)
            outcome: Continuous outcome vector
            instrument: Instrumental variable (optional)
            **kwargs: Additional arguments (unused)

        Returns:
            Array of PLIV scores for each observation
        """
        if instrument is None:
            # Fallback to PLR when no instrument available
            warnings.warn(
                "No instrument provided for PLIV. Falling back to PLR method.",
                UserWarning,
                stacklevel=2,
            )
            return OrthogonalMoments.plr(nuisance_estimates, treatment, outcome)

        # Extract nuisance parameters
        g = nuisance_estimates["propensity_scores"]  # m(X) = E[D|X]
        mu0 = nuisance_estimates["mu0"]  # μ₀(X) = E[Y|D=0,X]

        # Simple approximation for r(X) = E[Z|X]
        r_x = np.mean(instrument)  # Would need proper modeling in practice

        # Residualize treatment and instrument
        treatment_residual = treatment - g
        instrument_residual = instrument - r_x

        # Two-stage estimation for θ using IV
        # First stage: D - m(X) = γ(Z - r(X)) + v
        if np.var(instrument_residual) < 1e-8:
            warnings.warn(
                "Weak instrument for PLIV method. Falling back to PLR.",
                UserWarning,
                stacklevel=2,
            )
            return OrthogonalMoments.plr(nuisance_estimates, treatment, outcome)

        # First stage coefficient
        gamma_hat = np.cov(treatment_residual, instrument_residual)[0, 1] / np.var(
            instrument_residual
        )

        # Second stage: (Y - μ₀(X)) = θ * γ̂(Z - r(X)) + u
        y_residual = outcome - mu0
        if abs(gamma_hat) < 1e-8:
            warnings.warn(
                "Weak first stage for PLIV method. Falling back to PLR.",
                UserWarning,
                stacklevel=2,
            )
            return OrthogonalMoments.plr(nuisance_estimates, treatment, outcome)

        theta_hat = np.cov(y_residual, instrument_residual)[0, 1] / (
            gamma_hat * np.var(instrument_residual)
        )

        # PLIV scores: (Y - μ₀(X) - θ̂(D - m(X))) * (Z - r(X))
        pliv_residual = outcome - mu0 - theta_hat * treatment_residual
        pliv_scores = pliv_residual * instrument_residual

        return pliv_scores

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """Get list of available orthogonal moment methods.

        Returns:
            List of method names that can be used
        """
        return [
            "aipw",
            "orthogonal",
            "partialling_out",
            "interactive_iv",
            "plr",
            "pliv",
        ]

    @classmethod
    def compute_scores(
        cls,
        method: str,
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        instrument: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Compute orthogonal scores using specified method.

        Args:
            method: Name of orthogonal moment method
            nuisance_estimates: Dictionary of nuisance parameter estimates
            treatment: Binary treatment vector
            outcome: Continuous outcome vector
            instrument: Optional instrumental variable
            **kwargs: Additional method-specific arguments

        Returns:
            Array of orthogonal scores

        Raises:
            ValueError: If method is not recognized
        """
        method_map = {
            "aipw": cls.aipw,
            "orthogonal": cls.orthogonal,
            "partialling_out": cls.partialling_out,
            "interactive_iv": cls.interactive_iv,
            "plr": cls.plr,
            "pliv": cls.pliv,
        }

        if method not in method_map:
            available = list(method_map.keys())
            raise ValueError(
                f"Unknown method '{method}'. Available methods: {available}"
            )

        return method_map[method](
            nuisance_estimates, treatment, outcome, instrument=instrument, **kwargs
        )

    @classmethod
    def validate_orthogonality(
        cls,
        scores: NDArray[Any],
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Validate orthogonality condition for computed scores.

        Checks that the scores are approximately orthogonal to nuisance functions
        by computing correlations with residuals.

        Args:
            scores: Computed orthogonal scores
            nuisance_estimates: Dictionary of nuisance estimates
            treatment: Binary treatment vector
            threshold: Correlation threshold for orthogonality (default 0.05)

        Returns:
            Dictionary containing orthogonality validation results
        """
        results = {"is_orthogonal": True, "correlations": {}, "threshold": threshold}

        # Check correlation with treatment residuals (A - g(X))
        if "propensity_scores" in nuisance_estimates:
            treatment_residual = treatment - nuisance_estimates["propensity_scores"]
            corr_treatment, p_val_treatment = pearsonr(scores, treatment_residual)

            results["correlations"]["treatment_residual"] = {
                "correlation": float(corr_treatment),
                "p_value": float(p_val_treatment),
                "is_orthogonal": abs(corr_treatment) < threshold,
            }

            if abs(corr_treatment) >= threshold:
                results["is_orthogonal"] = False

        # Check correlation with outcome residuals if available
        if "mu_observed" in nuisance_estimates:
            # We'd need the actual outcome to compute residuals
            # This is a placeholder for more comprehensive validation
            pass

        # Overall orthogonality assessment
        failed_checks = []
        for key, check in results["correlations"].items():
            if not check["is_orthogonal"]:
                failed_checks.append(key)

        results["failed_checks"] = failed_checks
        results["interpretation"] = (
            "Orthogonality condition satisfied"
            if results["is_orthogonal"]
            else f"Orthogonality condition violated for: {', '.join(failed_checks)}"
        )

        return results

    @classmethod
    def select_optimal_method(
        cls,
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        covariates: NDArray[Any] | None = None,
        instrument: NDArray[Any] | None = None,
        sample_size_threshold: int = 200,
        dimensionality_threshold: int = 20,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Automatically select optimal orthogonal moment function.

        Selects the most appropriate moment function based on:
        - Data characteristics (sample size, dimensionality)
        - Treatment balance and overlap
        - Availability of instruments
        - Theoretical efficiency considerations

        Args:
            nuisance_estimates: Dictionary of nuisance estimates
            treatment: Binary treatment vector
            outcome: Continuous outcome vector
            covariates: Covariate matrix (optional)
            instrument: Instrumental variable (optional)
            sample_size_threshold: Threshold for small vs large samples
            dimensionality_threshold: Threshold for low vs high dimensionality
            **kwargs: Additional selection criteria

        Returns:
            Tuple of (selected_method, selection_rationale)
        """
        n = len(treatment)
        selection_rationale = {"criteria": {}, "decision_factors": []}

        # Analyze data characteristics
        is_small_sample = n < sample_size_threshold
        selection_rationale["criteria"]["small_sample"] = is_small_sample

        if covariates is not None:
            p = covariates.shape[1] if len(covariates.shape) > 1 else 1
            is_high_dimensional = p > dimensionality_threshold
        else:
            p = 0
            is_high_dimensional = False
        selection_rationale["criteria"]["high_dimensional"] = is_high_dimensional

        # Check treatment balance
        treatment_balance = min(np.mean(treatment), 1 - np.mean(treatment))
        is_balanced = treatment_balance > 0.1
        selection_rationale["criteria"]["balanced_treatment"] = is_balanced

        # Check propensity score overlap
        if "propensity_scores" in nuisance_estimates:
            g = nuisance_estimates["propensity_scores"]
            overlap_quality = min(np.min(g), 1 - np.max(g))
            has_good_overlap = overlap_quality > 0.05
        else:
            has_good_overlap = True
        selection_rationale["criteria"]["good_overlap"] = has_good_overlap

        # Instrument availability
        has_instrument = instrument is not None
        selection_rationale["criteria"]["has_instrument"] = has_instrument

        # Selection logic
        if has_instrument and (not is_balanced or not has_good_overlap):
            # Use IV methods when treatment assignment is problematic
            if is_small_sample:
                selected_method = "pliv"
                selection_rationale["decision_factors"].append(
                    "PLIV selected: instrument available, sample size small"
                )
            else:
                selected_method = "interactive_iv"
                selection_rationale["decision_factors"].append(
                    "Interactive IV selected: instrument available, large sample"
                )
        elif is_high_dimensional or is_small_sample:
            # Use partialling out for high-dimensional or small samples
            selected_method = "partialling_out"
            selection_rationale["decision_factors"].append(
                f"Partialling out selected: {'high-dimensional' if is_high_dimensional else 'small sample'} setting"
            )
        elif not has_good_overlap:
            # Use PLR when overlap is poor but no instrument
            selected_method = "plr"
            selection_rationale["decision_factors"].append(
                "PLR selected: poor overlap, no instrument available"
            )
        else:
            # Default to AIPW for well-behaved settings
            selected_method = "aipw"
            selection_rationale["decision_factors"].append(
                "AIPW selected: balanced treatment, good overlap, moderate dimensionality"
            )

        selection_rationale["selected_method"] = selected_method
        selection_rationale["data_characteristics"] = {
            "n": n,
            "p": p,
            "treatment_balance": float(treatment_balance),
            "overlap_quality": float(overlap_quality) if has_good_overlap else None,
        }

        return selected_method, selection_rationale

    @classmethod
    def cross_validate_methods(
        cls,
        candidate_methods: list[str],
        nuisance_estimates: dict[str, NDArray[Any]],
        treatment: NDArray[Any],
        outcome: NDArray[Any],
        cv_folds: int = 5,
        instrument: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Cross-validate different orthogonal moment methods.

        Compares methods based on out-of-sample orthogonality properties
        and estimation stability.

        Args:
            candidate_methods: List of methods to compare
            nuisance_estimates: Dictionary of nuisance estimates
            treatment: Binary treatment vector
            outcome: Continuous outcome vector
            cv_folds: Number of cross-validation folds
            instrument: Optional instrumental variable
            **kwargs: Additional arguments

        Returns:
            Dictionary containing cross-validation results and rankings
        """
        from sklearn.model_selection import KFold

        cv_results = {"method_performance": {}, "rankings": {}}
        n = len(treatment)

        # Initialize results storage
        for method in candidate_methods:
            cv_results["method_performance"][method] = {
                "orthogonality_scores": [],
                "ate_estimates": [],
                "score_variance": [],
                "convergence_issues": 0,
            }

        # Cross-validation loop
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(range(n)):
            # Extract validation data
            val_treatment = treatment[val_idx]
            val_outcome = outcome[val_idx]
            val_instrument = instrument[val_idx] if instrument is not None else None

            # Extract validation nuisance estimates
            val_nuisance = {}
            for key, estimates in nuisance_estimates.items():
                val_nuisance[key] = estimates[val_idx]

            # Evaluate each method on validation fold
            for method in candidate_methods:
                try:
                    # Compute scores
                    scores = cls.compute_scores(
                        method,
                        val_nuisance,
                        val_treatment,
                        val_outcome,
                        instrument=val_instrument,
                        **kwargs,
                    )

                    # Assess orthogonality
                    orthogonality_results = cls.validate_orthogonality(
                        scores, val_nuisance, val_treatment
                    )

                    # Store results
                    perf = cv_results["method_performance"][method]
                    perf["orthogonality_scores"].append(
                        1.0 if orthogonality_results["is_orthogonal"] else 0.0
                    )
                    perf["ate_estimates"].append(float(np.mean(scores)))
                    perf["score_variance"].append(float(np.var(scores)))

                except (ValueError, RuntimeWarning):
                    # Handle convergence issues
                    cv_results["method_performance"][method]["convergence_issues"] += 1
                    cv_results["method_performance"][method][
                        "orthogonality_scores"
                    ].append(0.0)
                    cv_results["method_performance"][method]["ate_estimates"].append(
                        np.nan
                    )
                    cv_results["method_performance"][method]["score_variance"].append(
                        np.nan
                    )

        # Aggregate results and rank methods
        method_scores = {}
        for method in candidate_methods:
            perf = cv_results["method_performance"][method]

            # Compute aggregate metrics
            avg_orthogonality = np.mean(perf["orthogonality_scores"])
            ate_stability = 1.0 / (1.0 + np.nanstd(perf["ate_estimates"]))
            avg_variance = np.nanmean(perf["score_variance"])
            reliability = 1.0 - (perf["convergence_issues"] / cv_folds)

            # Combined score (weighted sum)
            combined_score = (
                0.4 * avg_orthogonality  # Orthogonality most important
                + 0.3 * ate_stability  # Stability important
                + 0.2 * reliability  # Reliability matters
                + 0.1 * (1.0 / (1.0 + avg_variance))  # Lower variance better
            )

            method_scores[method] = {
                "combined_score": combined_score,
                "orthogonality": avg_orthogonality,
                "stability": ate_stability,
                "reliability": reliability,
                "avg_variance": avg_variance,
            }

        # Rank methods by combined score
        ranked_methods = sorted(
            method_scores.items(), key=lambda x: x[1]["combined_score"], reverse=True
        )

        cv_results["rankings"]["by_combined_score"] = [
            {"method": method, **scores} for method, scores in ranked_methods
        ]

        cv_results["recommended_method"] = (
            ranked_methods[0][0] if ranked_methods else "aipw"
        )

        return cv_results


MomentFunctionType = Literal[
    "aipw", "orthogonal", "partialling_out", "interactive_iv", "plr", "pliv", "auto"
]
