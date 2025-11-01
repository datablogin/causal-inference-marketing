"""Propensity score visualization and overlap assessment tools.

This module provides comprehensive visualization capabilities for propensity scores,
including overlap assessment, common support evaluation, and calibration plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

# Import plotting libraries with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..core.base import TreatmentData


@dataclass
class PropensityOverlapResult:
    """Results from propensity score overlap analysis."""

    propensity_scores: NDArray[np.floating[Any]]
    treatment_values: NDArray[Any]
    overlap_percentage: float
    common_support_range: tuple[float, float]
    positivity_violations: int
    auc_score: float
    brier_score: float
    calibration_slope: float
    calibration_intercept: float
    recommended_trimming: Optional[tuple[float, float]]


class PropensityPlotGenerator:
    """Generator for propensity score visualizations."""

    def __init__(
        self,
        overlap_threshold: float = 0.1,
        positivity_threshold: float = 0.01,
        figsize: tuple[int, int] = (15, 10),
        style: str = "whitegrid",
    ):
        """Initialize propensity plot generator.

        Args:
            overlap_threshold: Minimum overlap for common support
            positivity_threshold: Minimum propensity for positivity
            figsize: Figure size for plots
            style: Plotting style
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting libraries not available. Install with: "
                "pip install matplotlib seaborn"
            )

        self.overlap_threshold = overlap_threshold
        self.positivity_threshold = positivity_threshold
        self.figsize = figsize
        self.style = style

        sns.set_style(style)

    def analyze_propensity_overlap(
        self,
        propensity_scores: NDArray[np.floating[Any]],
        treatment: TreatmentData,
    ) -> PropensityOverlapResult:
        """Analyze propensity score overlap and common support.

        Args:
            propensity_scores: Estimated propensity scores
            treatment: Treatment assignment data

        Returns:
            PropensityOverlapResult with overlap analysis
        """
        ps = np.asarray(propensity_scores)
        treatment_values = treatment.values

        # Separate propensity scores by treatment group
        treated_ps = ps[treatment_values == 1]
        control_ps = ps[treatment_values == 0]

        # Calculate overlap percentage
        treated_range = (np.min(treated_ps), np.max(treated_ps))
        control_range = (np.min(control_ps), np.max(control_ps))

        overlap_min = max(treated_range[0], control_range[0])
        overlap_max = min(treated_range[1], control_range[1])

        if overlap_max > overlap_min:
            overlap_percentage = (
                np.sum((ps >= overlap_min) & (ps <= overlap_max)) / len(ps) * 100
            )
            common_support_range = (overlap_min, overlap_max)
        else:
            overlap_percentage = 0.0
            common_support_range = (0.0, 0.0)

        # Check for positivity violations
        positivity_violations = np.sum(ps < self.positivity_threshold) + np.sum(
            ps > 1 - self.positivity_threshold
        )

        # Calculate discrimination metrics
        auc_score = roc_auc_score(treatment_values, ps)
        brier_score = brier_score_loss(treatment_values, ps)

        # Calibration assessment
        prob_true, prob_pred = calibration_curve(
            treatment_values, ps, n_bins=10, strategy="uniform"
        )

        # Linear regression for calibration slope and intercept
        if len(prob_true) > 1:
            slope, intercept, _, _, _ = stats.linregress(prob_pred, prob_true)
        else:
            slope, intercept = 1.0, 0.0

        # Trimming recommendations
        recommended_trimming = None
        if positivity_violations > 0:
            lower_trim = np.percentile(ps, 2.5)
            upper_trim = np.percentile(ps, 97.5)
            if (
                lower_trim < self.positivity_threshold
                or upper_trim > 1 - self.positivity_threshold
            ):
                recommended_trimming = (lower_trim, upper_trim)

        return PropensityOverlapResult(
            propensity_scores=ps,
            treatment_values=treatment_values,
            overlap_percentage=overlap_percentage,
            common_support_range=common_support_range,
            positivity_violations=positivity_violations,
            auc_score=auc_score,
            brier_score=brier_score,
            calibration_slope=slope,
            calibration_intercept=intercept,
            recommended_trimming=recommended_trimming,
        )

    def create_propensity_plots(
        self,
        overlap_result: PropensityOverlapResult,
        title: str = "Propensity Score Diagnostics",
        save_path: Optional[str] = None,
        interactive: bool = False,
    ) -> Union[plt.Figure, go.Figure]:
        """Create comprehensive propensity score plots.

        Args:
            overlap_result: Propensity overlap analysis results
            title: Main title for the plots
            save_path: Path to save the plot
            interactive: Whether to create interactive plots

        Returns:
            Matplotlib or Plotly figure
        """
        if interactive and not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available for interactive plots")

        if interactive:
            return self._create_interactive_propensity_plots(
                overlap_result, title, save_path
            )
        else:
            return self._create_static_propensity_plots(
                overlap_result, title, save_path
            )

    def _create_static_propensity_plots(
        self,
        result: PropensityOverlapResult,
        title: str,
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create static matplotlib propensity plots."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Overlap histogram
        ax1 = fig.add_subplot(gs[0, :2])

        treated_ps = result.propensity_scores[result.treatment_values == 1]
        control_ps = result.propensity_scores[result.treatment_values == 0]

        bins = np.linspace(0, 1, 31)

        ax1.hist(
            control_ps,
            bins=bins,
            alpha=0.7,
            label="Control",
            color="skyblue",
            density=True,
        )
        ax1.hist(
            treated_ps,
            bins=bins,
            alpha=0.7,
            label="Treated",
            color="orange",
            density=True,
        )

        # Highlight common support region
        if result.common_support_range[1] > result.common_support_range[0]:
            ax1.axvspan(
                result.common_support_range[0],
                result.common_support_range[1],
                alpha=0.2,
                color="green",
                label="Common Support",
            )

        # Mark positivity violations
        ax1.axvline(
            self.positivity_threshold,
            color="red",
            linestyle="--",
            label=f"Positivity Threshold ({self.positivity_threshold})",
        )
        ax1.axvline(1 - self.positivity_threshold, color="red", linestyle="--")

        ax1.set_xlabel("Propensity Score")
        ax1.set_ylabel("Density")
        ax1.set_title("Propensity Score Overlap")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plots by treatment group
        ax2 = fig.add_subplot(gs[0, 2])

        box_data = [control_ps, treated_ps]
        box_plot = ax2.boxplot(
            box_data, labels=["Control", "Treated"], patch_artist=True
        )
        box_plot["boxes"][0].set_facecolor("skyblue")
        box_plot["boxes"][1].set_facecolor("orange")

        ax2.set_ylabel("Propensity Score")
        ax2.set_title("Distribution by\nTreatment Group")
        ax2.grid(True, alpha=0.3)

        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[1, 0])

        fpr, tpr, _ = roc_curve(result.treatment_values, result.propensity_scores)
        ax3.plot(
            fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {result.auc_score:.3f})"
        )
        ax3.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")

        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curve")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Calibration plot
        ax4 = fig.add_subplot(gs[1, 1])

        prob_true, prob_pred = calibration_curve(
            result.treatment_values,
            result.propensity_scores,
            n_bins=10,
            strategy="uniform",
        )

        ax4.plot(
            prob_pred, prob_true, "bo-", linewidth=2, markersize=6, label="Calibration"
        )
        ax4.plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfect Calibration")

        ax4.set_xlabel("Mean Predicted Probability")
        ax4.set_ylabel("Fraction of Positives")
        ax4.set_title(f"Calibration Plot\n(Slope: {result.calibration_slope:.3f})")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Propensity score vs residuals (if treatment is binary)
        ax5 = fig.add_subplot(gs[1, 2])

        if len(np.unique(result.treatment_values)) == 2:
            # Calculate residuals (observed - predicted)
            residuals = result.treatment_values - result.propensity_scores

            ax5.scatter(result.propensity_scores, residuals, alpha=0.5, s=20)
            ax5.axhline(0, color="red", linestyle="--")
            ax5.set_xlabel("Propensity Score")
            ax5.set_ylabel("Residuals")
            ax5.set_title("Residuals vs\nPropensity Score")
        else:
            # Alternative: density plot
            ax5.hist(result.propensity_scores, bins=30, alpha=0.7, color="lightgreen")
            ax5.set_xlabel("Propensity Score")
            ax5.set_ylabel("Frequency")
            ax5.set_title("Overall Distribution")
        ax5.grid(True, alpha=0.3)

        # 6. Summary statistics and recommendations
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")

        # Create summary table
        summary_data = [
            ["Metric", "Value", "Assessment"],
            [
                "Overlap Percentage",
                f"{result.overlap_percentage:.1f}%",
                "✅ Good"
                if result.overlap_percentage > 80
                else "⚠️ Moderate"
                if result.overlap_percentage > 50
                else "❌ Poor",
            ],
            [
                "AUC Score",
                f"{result.auc_score:.3f}",
                "✅ Good"
                if 0.6 <= result.auc_score <= 0.8
                else "⚠️ Check"
                if result.auc_score < 0.6 or result.auc_score > 0.9
                else "❌ Poor",
            ],
            [
                "Brier Score",
                f"{result.brier_score:.3f}",
                "✅ Good"
                if result.brier_score < 0.2
                else "⚠️ Moderate"
                if result.brier_score < 0.3
                else "❌ Poor",
            ],
            [
                "Positivity Violations",
                f"{result.positivity_violations}",
                "✅ None"
                if result.positivity_violations == 0
                else "⚠️ Some"
                if result.positivity_violations < 10
                else "❌ Many",
            ],
            [
                "Calibration Slope",
                f"{result.calibration_slope:.3f}",
                "✅ Good" if 0.8 <= result.calibration_slope <= 1.2 else "⚠️ Check",
            ],
        ]

        table = ax6.table(
            cellText=summary_data,
            cellLoc="center",
            loc="center",
            bbox=[0.1, 0.0, 0.8, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style table
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        for i in range(1, len(summary_data)):
            for j in range(len(summary_data[0])):
                if j == 2:  # Assessment column
                    cell_text = summary_data[i][j]
                    if "✅" in cell_text:
                        table[(i, j)].set_facecolor("#d4edda")
                    elif "⚠️" in cell_text:
                        table[(i, j)].set_facecolor("#fff3cd")
                    elif "❌" in cell_text:
                        table[(i, j)].set_facecolor("#f8d7da")
                else:
                    table[(i, j)].set_facecolor("#f8f9fa")

        plt.suptitle(title, fontsize=16, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _create_interactive_propensity_plots(
        self,
        result: PropensityOverlapResult,
        title: str,
        save_path: Optional[str],
    ) -> go.Figure:
        """Create interactive Plotly propensity plots."""
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Propensity Score Overlap",
                "ROC Curve",
                "Calibration Plot",
                "Distribution by Treatment",
                "Residuals Plot",
                "Summary Metrics",
            ],
            specs=[
                [
                    {"secondary_y": False},
                    {"secondary_y": False},
                    {"secondary_y": False},
                ],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}],
            ],
        )

        # Separate data by treatment group
        treated_ps = result.propensity_scores[result.treatment_values == 1]
        control_ps = result.propensity_scores[result.treatment_values == 0]

        # 1. Overlap histogram with enhanced hover information
        fig.add_trace(
            go.Histogram(
                x=control_ps,
                name="Control",
                opacity=0.7,
                nbinsx=30,
                marker_color="skyblue",
                hovertemplate=(
                    "<b>Control Group</b><br>"
                    "Propensity Score: %{x:.3f}<br>"
                    "Count: %{y}<br>"
                    "Density: %{y}/%{meta}<br>"
                    "<extra></extra>"
                ),
                meta=len(control_ps),
                histnorm="probability density",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=treated_ps,
                name="Treated",
                opacity=0.7,
                nbinsx=30,
                marker_color="orange",
                histnorm="probability density",
                hovertemplate=(
                    "<b>Treated Group</b><br>"
                    "Propensity Score: %{x:.3f}<br>"
                    "Count: %{y}<br>"
                    "Density: %{y}/%{meta}<br>"
                    "<extra></extra>"
                ),
                meta=len(treated_ps),
            ),
            row=1,
            col=1,
        )

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(result.treatment_values, result.propensity_scores)

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines+markers",
                name=f"ROC (AUC = {result.auc_score:.3f})",
                line=dict(color="blue", width=2),
                marker=dict(size=4, opacity=0.6),
                hovertemplate=(
                    "<b>ROC Curve</b><br>"
                    "False Positive Rate: %{x:.3f}<br>"
                    "True Positive Rate: %{y:.3f}<br>"
                    "Sensitivity: %{y:.3f}<br>"
                    "1 - Specificity: %{x:.3f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=2,
        )

        # 3. Calibration plot
        prob_true, prob_pred = calibration_curve(
            result.treatment_values,
            result.propensity_scores,
            n_bins=10,
            strategy="uniform",
        )

        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="Calibration",
                line=dict(color="blue", width=2),
                marker=dict(size=8),
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=3,
        )

        # 4. Box plots
        fig.add_trace(
            go.Box(y=control_ps, name="Control", marker_color="skyblue"), row=2, col=1
        )

        fig.add_trace(
            go.Box(y=treated_ps, name="Treated", marker_color="orange"), row=2, col=1
        )

        # 5. Residuals scatter plot
        residuals = result.treatment_values - result.propensity_scores

        fig.add_trace(
            go.Scatter(
                x=result.propensity_scores,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(size=5, opacity=0.6),
                hovertemplate="PS: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        # 6. Summary table
        summary_data = [
            ["Overlap Percentage", f"{result.overlap_percentage:.1f}%"],
            ["AUC Score", f"{result.auc_score:.3f}"],
            ["Brier Score", f"{result.brier_score:.3f}"],
            ["Positivity Violations", f"{result.positivity_violations}"],
            ["Calibration Slope", f"{result.calibration_slope:.3f}"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"], fill_color="lightblue", align="left"
                ),
                cells=dict(
                    values=list(zip(*summary_data)),
                    fill_color="lightgray",
                    align="left",
                ),
            ),
            row=2,
            col=3,
        )

        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=800,
            # Enhanced interactivity
            hovermode="closest",
            dragmode="zoom",  # Default to zoom mode
            # Add range selector buttons for better navigation
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=[{"dragmode": "zoom"}],
                                label="Zoom",
                                method="relayout",
                            ),
                            dict(
                                args=[{"dragmode": "pan"}],
                                label="Pan",
                                method="relayout",
                            ),
                            dict(
                                args=[{"dragmode": "select"}],
                                label="Select",
                                method="relayout",
                            ),
                            dict(
                                args=[{"dragmode": "lasso"}],
                                label="Lasso",
                                method="relayout",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top",
                ),
            ],
        )

        # Update axes for better interaction
        fig.update_xaxes(
            showspikes=True, spikecolor="green", spikesnap="cursor", spikemode="across"
        )
        fig.update_yaxes(
            showspikes=True, spikecolor="orange", spikesnap="cursor", spikemode="across"
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def generate_recommendations(
        self,
        overlap_result: PropensityOverlapResult,
    ) -> list[str]:
        """Generate recommendations based on propensity score analysis.

        Args:
            overlap_result: Results from propensity overlap analysis

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Overlap assessment
        if overlap_result.overlap_percentage > 80:
            recommendations.append(
                "✅ Good propensity score overlap between treatment groups."
            )
        elif overlap_result.overlap_percentage > 50:
            recommendations.append(
                f"⚠️ Moderate overlap ({overlap_result.overlap_percentage:.1f}%). "
                "Consider trimming extreme propensity scores."
            )
        else:
            recommendations.append(
                f"❌ Poor overlap ({overlap_result.overlap_percentage:.1f}%). "
                "Strong evidence of limited common support. Consider different approach."
            )

        # Positivity violations
        if overlap_result.positivity_violations == 0:
            recommendations.append("✅ No positivity violations detected.")
        elif overlap_result.positivity_violations < 10:
            recommendations.append(
                f"⚠️ {overlap_result.positivity_violations} positivity violations detected. "
                "Consider trimming extreme values."
            )
        else:
            recommendations.append(
                f"❌ Many positivity violations ({overlap_result.positivity_violations}). "
                "Serious overlap issues detected."
            )

        # Discrimination assessment
        if 0.6 <= overlap_result.auc_score <= 0.8:
            recommendations.append(
                f"✅ Good propensity model discrimination (AUC: {overlap_result.auc_score:.3f})."
            )
        elif overlap_result.auc_score < 0.6:
            recommendations.append(
                f"⚠️ Low discrimination (AUC: {overlap_result.auc_score:.3f}). "
                "Consider adding more predictive covariates."
            )
        else:
            recommendations.append(
                f"⚠️ Very high discrimination (AUC: {overlap_result.auc_score:.3f}). "
                "May indicate near-perfect separation. Check model specification."
            )

        # Calibration assessment
        if 0.8 <= overlap_result.calibration_slope <= 1.2:
            recommendations.append("✅ Good propensity score calibration.")
        else:
            recommendations.append(
                f"⚠️ Poor calibration (slope: {overlap_result.calibration_slope:.3f}). "
                "Consider recalibrating propensity scores."
            )

        # Trimming recommendations
        if overlap_result.recommended_trimming:
            lower, upper = overlap_result.recommended_trimming
            recommendations.append(
                f"💡 Consider trimming propensity scores to range [{lower:.3f}, {upper:.3f}] "
                "to improve overlap and reduce extreme weights."
            )

        return recommendations


def create_propensity_plots(
    propensity_scores: NDArray[np.floating[Any]],
    treatment: TreatmentData,
    title: str = "Propensity Score Diagnostics",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> tuple[Union[plt.Figure, go.Figure], PropensityOverlapResult, list[str]]:
    """Convenience function to create propensity score diagnostic plots.

    Args:
        propensity_scores: Estimated propensity scores
        treatment: Treatment assignment data
        title: Plot title
        save_path: Path to save the plot
        interactive: Whether to create interactive plot

    Returns:
        Tuple of (figure, overlap_result, recommendations)
    """
    generator = PropensityPlotGenerator()
    overlap_result = generator.analyze_propensity_overlap(propensity_scores, treatment)
    figure = generator.create_propensity_plots(
        overlap_result, title, save_path, interactive
    )
    recommendations = generator.generate_recommendations(overlap_result)

    return figure, overlap_result, recommendations
