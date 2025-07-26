"""Visualization tools for causal inference diagnostics.

This module provides comprehensive visualization capabilities for all diagnostic
tools including balance plots, overlap assessments, sensitivity analysis, and more.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.base import TreatmentData
from .balance import BalanceResults
from .overlap import OverlapResults
from .sensitivity import SensitivityResults
from .specification import SpecificationResults

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class DiagnosticVisualizer:
    """Main class for creating diagnostic visualizations."""

    def __init__(self, style: str = "whitegrid", figsize: tuple[int, int] = (10, 6)):
        """Initialize the visualizer.

        Args:
            style: Seaborn style for plots
            figsize: Default figure size for plots
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting libraries not available. Please install matplotlib and seaborn: "
                "pip install matplotlib seaborn"
            )

        self.style = style
        self.figsize = figsize
        sns.set_style(style)

    def plot_balance_diagnostics(
        self,
        balance_results: BalanceResults,
        threshold: float = 0.1,
        save_path: str | None = None,
    ) -> Any:
        """Create comprehensive balance diagnostic plots.

        Args:
            balance_results: Results from balance diagnostics
            threshold: SMD threshold for balance assessment
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Covariate Balance Diagnostics", fontsize=16)

        # 1. Standardized Mean Differences
        ax1 = axes[0, 0]
        variables = list(balance_results.standardized_mean_differences.keys())
        smds = list(balance_results.standardized_mean_differences.values())

        colors = ["red" if abs(smd) > threshold else "green" for smd in smds]
        ax1.barh(variables, smds, color=colors, alpha=0.7)
        ax1.axvline(
            threshold,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Threshold: ±{threshold}",
        )
        ax1.axvline(-threshold, color="red", linestyle="--", alpha=0.8)
        ax1.axvline(0, color="black", linestyle="-", alpha=0.5)
        ax1.set_xlabel("Standardized Mean Difference")
        ax1.set_title("Standardized Mean Differences")
        ax1.legend()

        # 2. Variance Ratios
        ax2 = axes[0, 1]
        var_ratios = list(balance_results.variance_ratios.values())
        colors = ["red" if vr < 0.5 or vr > 2.0 else "green" for vr in var_ratios]
        ax2.barh(variables, var_ratios, color=colors, alpha=0.7)
        ax2.axvline(
            0.5,
            color="red",
            linestyle="--",
            alpha=0.8,
            label="Acceptable range: 0.5-2.0",
        )
        ax2.axvline(2.0, color="red", linestyle="--", alpha=0.8)
        ax2.axvline(1.0, color="black", linestyle="-", alpha=0.5)
        ax2.set_xlabel("Variance Ratio")
        ax2.set_title("Variance Ratios")
        ax2.legend()

        # 3. Balance Assessment Summary
        ax3 = axes[1, 0]
        imbalanced_vars = set(balance_results.imbalanced_covariates)
        balanced_count = len(variables) - len(imbalanced_vars)
        imbalanced_count = len(imbalanced_vars)

        ax3.pie(
            [balanced_count, imbalanced_count],
            labels=["Balanced", "Imbalanced"],
            colors=["green", "red"],
            autopct="%1.1f%%",
        )
        ax3.set_title("Overall Balance Assessment")

        # 4. Love Plot (SMD distribution)
        ax4 = axes[1, 1]
        colors = ["red" if var in imbalanced_vars else "green" for var in variables]
        ax4.scatter(smds, range(len(smds)), c=colors, alpha=0.7, s=60)
        ax4.axvline(threshold, color="red", linestyle="--", alpha=0.8)
        ax4.axvline(-threshold, color="red", linestyle="--", alpha=0.8)
        ax4.axvline(0, color="black", linestyle="-", alpha=0.5)
        ax4.set_xlabel("Standardized Mean Difference")
        ax4.set_ylabel("Variables")
        ax4.set_title("Love Plot (SMD Distribution)")
        ax4.set_yticks(range(len(variables)))
        ax4.set_yticklabels(variables)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_overlap_diagnostics(
        self,
        overlap_results: OverlapResults,
        treatment: TreatmentData,
        save_path: str | None = None,
    ) -> Any:
        """Create overlap and positivity diagnostic plots.

        Args:
            overlap_results: Results from overlap diagnostics
            treatment: Treatment data for context
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Overlap and Positivity Diagnostics", fontsize=16)

        # 1. Propensity Score Distribution
        ax1 = axes[0, 0]
        if overlap_results.propensity_scores is not None:
            treatment_vals = np.asarray(treatment.values)
            ps_scores = overlap_results.propensity_scores

            # Plot histograms for treated and control
            ax1.hist(
                ps_scores[treatment_vals == 0],
                alpha=0.7,
                label="Control",
                bins=30,
                color="blue",
            )
            ax1.hist(
                ps_scores[treatment_vals == 1],
                alpha=0.7,
                label="Treated",
                bins=30,
                color="red",
            )
            ax1.set_xlabel("Propensity Score")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Propensity Score Distribution")
            ax1.legend()
            ax1.axvline(
                0.1, color="red", linestyle="--", alpha=0.8, label="Common support"
            )
            ax1.axvline(0.9, color="red", linestyle="--", alpha=0.8)

        # 2. Common Support Region
        ax2 = axes[0, 1]
        if overlap_results.common_support_range is not None:
            lower, upper = overlap_results.common_support_range
            support_width = upper - lower

            ax2.barh(
                ["Common Support"],
                [support_width],
                left=[lower],
                color="green",
                alpha=0.7,
            )
            ax2.barh(
                ["Outside Support"],
                [lower + (1 - upper)],
                left=[0],
                color="red",
                alpha=0.7,
            )
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Propensity Score Range")
            ax2.set_title(f"Common Support: [{lower:.3f}, {upper:.3f}]")

        # 3. Overlap Quality Metrics
        ax3 = axes[1, 0]
        # Calculate common support percentage
        common_support_pct = (
            overlap_results.units_in_common_support / overlap_results.total_units
        )

        metrics = {
            "Positivity Met": 1.0 if overlap_results.overall_positivity_met else 0.0,
            "Common Support %": common_support_pct,
        }

        colors = [
            "green" if v > 0.8 else "orange" if v > 0.6 else "red"
            for v in metrics.values()
        ]
        bars = ax3.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7)
        ax3.set_ylabel("Score/Percentage")
        ax3.set_title("Overlap Quality Metrics")
        ax3.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # 4. Positivity Assessment
        ax4 = axes[1, 1]
        if overlap_results.violations:
            regions_text = "Violations Detected:\n"
            regions_text += "\n".join(
                [
                    f"• {violation.get('description', 'Violation')}"
                    for violation in overlap_results.violations[:5]
                ]
            )
            if len(overlap_results.violations) > 5:
                regions_text += f"\n... and {len(overlap_results.violations) - 5} more"
        else:
            regions_text = "✅ No Violations\nDetected"

        ax4.text(
            0.5,
            0.5,
            regions_text,
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )
        ax4.set_title("Positivity Assessment")
        ax4.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_sensitivity_analysis(
        self,
        sensitivity_results: SensitivityResults,
        save_path: str | None = None,
    ) -> Any:
        """Create sensitivity analysis visualization.

        Args:
            sensitivity_results: Results from sensitivity analysis
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Sensitivity Analysis", fontsize=16)

        # 1. E-value interpretation
        ax1 = axes[0, 0]
        if sensitivity_results.evalue is not None:
            evalue = sensitivity_results.evalue

            # Create a gauge-like visualization
            categories = [
                "Weak\n(< 1.5)",
                "Moderate\n(1.5-3)",
                "Strong\n(3-5)",
                "Very Strong\n(> 5)",
            ]
            thresholds = [1.5, 3, 5, 10]
            colors = ["red", "orange", "yellow", "green"]

            for i, (cat, thresh, color) in enumerate(
                zip(categories, thresholds, colors)
            ):
                ax1.barh([i], [1], color=color, alpha=0.3, edgecolor="black")
                ax1.text(0.5, i, cat, ha="center", va="center", fontweight="bold")

            # Mark the E-value position
            if evalue < 1.5:
                position = 0
            elif evalue < 3:
                position = 1
            elif evalue < 5:
                position = 2
            else:
                position = 3

            ax1.scatter(
                [0.8],
                [position],
                color="red",
                s=200,
                marker=">",
                label=f"E-value: {evalue:.2f}",
            )
            ax1.set_xlim(0, 1)
            ax1.set_ylim(-0.5, 3.5)
            ax1.set_title("E-value Robustness Assessment")
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.legend()

        # 2. Confounding Strength Analysis
        ax2 = axes[0, 1]
        if sensitivity_results.sensitivity_plots_data:
            plot_data = sensitivity_results.sensitivity_plots_data
            if "results" in plot_data:
                strengths = [r["strength"] for r in plot_data["results"]]
                biases = [r["bias"] for r in plot_data["results"]]

                ax2.plot(strengths, biases, "o-", color="blue", alpha=0.7)
                ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
                ax2.set_xlabel("Confounder Strength")
                ax2.set_ylabel("Bias in Effect Estimate")
                ax2.set_title("Unmeasured Confounding Impact")
                ax2.grid(True, alpha=0.3)

        # 3. Robustness Summary
        ax3 = axes[1, 0]
        robustness_text = sensitivity_results.robustness_assessment
        robustness_color = "green" if "robust" in robustness_text.lower() else "red"

        ax3.text(
            0.5,
            0.5,
            robustness_text,
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=robustness_color, alpha=0.3),
        )
        ax3.set_title("Robustness Assessment")
        ax3.axis("off")

        # 4. Recommendations
        ax4 = axes[1, 1]
        recommendations_text = "\n".join(
            [f"• {rec}" for rec in sensitivity_results.recommendations[:5]]
        )
        if len(sensitivity_results.recommendations) > 5:
            recommendations_text += (
                f"\n... and {len(sensitivity_results.recommendations) - 5} more"
            )

        ax4.text(
            0.05,
            0.95,
            "Recommendations:",
            ha="left",
            va="top",
            transform=ax4.transAxes,
            fontsize=12,
            fontweight="bold",
        )
        ax4.text(
            0.05,
            0.85,
            recommendations_text,
            ha="left",
            va="top",
            transform=ax4.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
        )
        ax4.set_title("Analysis Recommendations")
        ax4.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_specification_tests(
        self,
        spec_results: SpecificationResults,
        save_path: str | None = None,
    ) -> Any:
        """Create model specification diagnostic plots.

        Args:
            spec_results: Results from specification tests
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Specification Diagnostics", fontsize=16)

        # 1. Linearity Test Results
        ax1 = axes[0, 0]
        if spec_results.linearity_tests:
            variables = list(spec_results.linearity_tests.keys())
            p_values = [
                test.get("p_value", 1.0)
                for test in spec_results.linearity_tests.values()
            ]

            colors = ["red" if p < 0.05 else "green" for p in p_values]
            bars = ax1.barh(
                variables, [-np.log10(p) for p in p_values], color=colors, alpha=0.7
            )
            ax1.axvline(
                -np.log10(0.05),
                color="red",
                linestyle="--",
                alpha=0.8,
                label="p = 0.05",
            )
            ax1.set_xlabel("-log10(p-value)")
            ax1.set_title("Linearity Tests")
            ax1.legend()

        # 2. Functional Form Assessment
        ax2 = axes[0, 1]
        if spec_results.functional_form_tests:
            form_tests = spec_results.functional_form_tests
            test_names = list(form_tests.keys())
            test_stats = [test.get("test_statistic", 0) for test in form_tests.values()]

            ax2.bar(test_names, test_stats, alpha=0.7, color="skyblue")
            ax2.set_ylabel("Test Statistic")
            ax2.set_title("Functional Form Tests")
            ax2.tick_params(axis="x", rotation=45)

        # 3. Overall Specification Assessment
        ax3 = axes[1, 0]
        # Create a summary of specification issues
        issues = []
        if spec_results.linearity_tests:
            linearity_issues = sum(
                1
                for test in spec_results.linearity_tests.values()
                if test.get("p_value", 1.0) < 0.05
            )
            if linearity_issues > 0:
                issues.append(f"Linearity violations: {linearity_issues}")

        if spec_results.interaction_tests:
            interaction_issues = sum(
                1
                for test in spec_results.interaction_tests.values()
                if test.get("p_value", 1.0) < 0.05
            )
            if interaction_issues > 0:
                issues.append(f"Missing interactions: {interaction_issues}")

        if not issues:
            status_text = "✅ No Major Specification\nIssues Detected"
            status_color = "green"
        else:
            status_text = "⚠️ Specification Issues:\n" + "\n".join(issues)
            status_color = "orange"

        ax3.text(
            0.5,
            0.5,
            status_text,
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.3),
        )
        ax3.set_title("Overall Specification Assessment")
        ax3.axis("off")

        # 4. Model Comparison (if available)
        ax4 = axes[1, 1]
        if spec_results.model_comparison:
            comparison = spec_results.model_comparison
            models = list(comparison.keys())
            metrics = [comp.get("aic", 0) for comp in comparison.values()]

            bars = ax4.bar(models, metrics, alpha=0.7, color="lightcoral")
            ax4.set_ylabel("AIC Score")
            ax4.set_title("Model Comparison (Lower AIC = Better)")
            ax4.tick_params(axis="x", rotation=45)

            # Highlight best model
            best_idx = np.argmin(metrics)
            bars[best_idx].set_color("green")
        else:
            ax4.text(
                0.5,
                0.5,
                "No Model Comparison\nData Available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
            ax4.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_diagnostic_dashboard(
        self,
        balance_results: BalanceResults | None = None,
        overlap_results: OverlapResults | None = None,
        sensitivity_results: SensitivityResults | None = None,
        spec_results: SpecificationResults | None = None,
        treatment: TreatmentData | None = None,
        save_path: str | None = None,
    ) -> list[Any]:
        """Create a comprehensive diagnostic dashboard.

        Args:
            balance_results: Balance diagnostic results
            overlap_results: Overlap diagnostic results
            sensitivity_results: Sensitivity analysis results
            spec_results: Specification test results
            treatment: Treatment data for context
            save_path: Base path to save plots (will append suffixes)

        Returns:
            List of matplotlib figure objects
        """
        figures = []

        if balance_results:
            fig = self.plot_balance_diagnostics(
                balance_results,
                save_path=f"{save_path}_balance.png" if save_path else None,
            )
            figures.append(fig)

        if overlap_results and treatment:
            fig = self.plot_overlap_diagnostics(
                overlap_results,
                treatment,
                save_path=f"{save_path}_overlap.png" if save_path else None,
            )
            figures.append(fig)

        if sensitivity_results:
            fig = self.plot_sensitivity_analysis(
                sensitivity_results,
                save_path=f"{save_path}_sensitivity.png" if save_path else None,
            )
            figures.append(fig)

        if spec_results:
            fig = self.plot_specification_tests(
                spec_results,
                save_path=f"{save_path}_specification.png" if save_path else None,
            )
            figures.append(fig)

        return figures


# Convenience functions
def plot_balance_diagnostics(
    balance_results: BalanceResults,
    threshold: float = 0.1,
    save_path: str | None = None,
) -> Any:
    """Convenience function for balance diagnostic plots."""
    visualizer = DiagnosticVisualizer()
    return visualizer.plot_balance_diagnostics(balance_results, threshold, save_path)


def plot_overlap_diagnostics(
    overlap_results: OverlapResults,
    treatment: TreatmentData,
    save_path: str | None = None,
) -> Any:
    """Convenience function for overlap diagnostic plots."""
    visualizer = DiagnosticVisualizer()
    return visualizer.plot_overlap_diagnostics(overlap_results, treatment, save_path)


def plot_sensitivity_analysis(
    sensitivity_results: SensitivityResults,
    save_path: str | None = None,
) -> Any:
    """Convenience function for sensitivity analysis plots."""
    visualizer = DiagnosticVisualizer()
    return visualizer.plot_sensitivity_analysis(sensitivity_results, save_path)


def plot_specification_tests(
    spec_results: SpecificationResults,
    save_path: str | None = None,
) -> Any:
    """Convenience function for specification test plots."""
    visualizer = DiagnosticVisualizer()
    return visualizer.plot_specification_tests(spec_results, save_path)
