"""Weight distribution diagnostics and visualization tools.

This module provides tools for analyzing and visualizing weight distributions
in causal inference analyses, including extreme weight detection and trimming
recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

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


@dataclass
class WeightDiagnosticsResult:
    """Results from weight distribution diagnostics."""

    weights: NDArray[np.floating[Any]]
    n_observations: int
    min_weight: float
    max_weight: float
    mean_weight: float
    median_weight: float
    std_weight: float
    skewness: float
    kurtosis: float
    extreme_weight_count: int
    extreme_weight_percentage: float
    recommended_trimming_threshold: float | None
    effective_sample_size: float
    weight_summary: dict[str, float]


class WeightDiagnostics:
    """Analyzer for weight distribution diagnostics."""

    def __init__(
        self,
        extreme_weight_threshold: float = 10.0,
        trimming_percentile: float = 99.0,
        figsize: tuple[int, int] = (12, 8),
        style: str = "whitegrid",
    ):
        """Initialize weight diagnostics analyzer.

        Args:
            extreme_weight_threshold: Threshold for flagging extreme weights
            trimming_percentile: Percentile for trimming recommendations
            figsize: Figure size for plots
            style: Plotting style
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting libraries not available. Install with: "
                "pip install matplotlib seaborn"
            )

        self.extreme_weight_threshold = extreme_weight_threshold
        self.trimming_percentile = trimming_percentile
        self.figsize = figsize
        self.style = style

        sns.set_style(style)

    def analyze_weights(
        self,
        weights: NDArray[np.floating[Any]],
    ) -> WeightDiagnosticsResult:
        """Analyze weight distribution and provide diagnostics.

        Args:
            weights: Array of weights to analyze

        Returns:
            WeightDiagnosticsResult with diagnostic information
        """
        weights = np.asarray(weights)
        n_obs = len(weights)

        # Basic statistics
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        mean_weight = np.mean(weights)
        median_weight = np.median(weights)
        std_weight = np.std(weights)

        # Higher order moments
        skewness = stats.skew(weights)
        kurtosis = stats.kurtosis(weights)

        # Extreme weight analysis
        extreme_mask = weights > self.extreme_weight_threshold
        extreme_count = np.sum(extreme_mask)
        extreme_percentage = extreme_count / n_obs * 100

        # Trimming recommendations
        trimming_threshold = np.percentile(weights, self.trimming_percentile)

        # Effective sample size calculation
        # ESS = (sum of weights)^2 / sum of weights^2
        ess = (np.sum(weights) ** 2) / np.sum(weights**2)

        # Weight summary statistics
        weight_summary = {
            "p5": np.percentile(weights, 5),
            "p25": np.percentile(weights, 25),
            "p50": median_weight,
            "p75": np.percentile(weights, 75),
            "p95": np.percentile(weights, 95),
            "p99": np.percentile(weights, 99),
        }

        return WeightDiagnosticsResult(
            weights=weights,
            n_observations=n_obs,
            min_weight=min_weight,
            max_weight=max_weight,
            mean_weight=mean_weight,
            median_weight=median_weight,
            std_weight=std_weight,
            skewness=skewness,
            kurtosis=kurtosis,
            extreme_weight_count=extreme_count,
            extreme_weight_percentage=extreme_percentage,
            recommended_trimming_threshold=trimming_threshold
            if extreme_count > 0
            else None,
            effective_sample_size=ess,
            weight_summary=weight_summary,
        )

    def create_weight_plots(
        self,
        diagnostics_result: WeightDiagnosticsResult,
        title: str = "Weight Distribution Diagnostics",
        save_path: str | None = None,
        interactive: bool = False,
    ) -> plt.Figure | go.Figure:
        """Create comprehensive weight distribution plots.

        Args:
            diagnostics_result: Weight diagnostics results
            title: Main title for the plots
            save_path: Path to save the plot
            interactive: Whether to create interactive plots

        Returns:
            Matplotlib or Plotly figure
        """
        if interactive and not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available for interactive plots")

        if interactive:
            return self._create_interactive_weight_plots(
                diagnostics_result, title, save_path
            )
        else:
            return self._create_static_weight_plots(
                diagnostics_result, title, save_path
            )

    def _create_static_weight_plots(
        self,
        result: WeightDiagnosticsResult,
        title: str,
        save_path: str | None,
    ) -> plt.Figure:
        """Create static matplotlib weight plots."""
        fig = plt.figure(figsize=(16, 12))

        # Create subplot grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Histogram with density
        ax1 = fig.add_subplot(gs[0, :2])

        # Regular histogram
        n_bins = min(50, max(10, int(np.sqrt(result.n_observations))))
        counts, bins, patches = ax1.hist(
            result.weights,
            bins=n_bins,
            alpha=0.7,
            density=True,
            color="skyblue",
            edgecolor="black",
        )

        # Color extreme weights differently
        extreme_threshold = self.extreme_weight_threshold
        for i, patch in enumerate(patches):
            if bins[i] > extreme_threshold:
                patch.set_facecolor("red")
                patch.set_alpha(0.8)

        # Add vertical lines for key statistics
        ax1.axvline(
            result.mean_weight,
            color="red",
            linestyle="-",
            label=f"Mean: {result.mean_weight:.2f}",
        )
        ax1.axvline(
            result.median_weight,
            color="green",
            linestyle="--",
            label=f"Median: {result.median_weight:.2f}",
        )

        if result.recommended_trimming_threshold:
            ax1.axvline(
                result.recommended_trimming_threshold,
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"Trim Threshold: {result.recommended_trimming_threshold:.2f}",
            )

        ax1.set_xlabel("Weight Value")
        ax1.set_ylabel("Density")
        ax1.set_title("Weight Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plot
        ax2 = fig.add_subplot(gs[0, 2])
        box_plot = ax2.boxplot(result.weights, vert=True, patch_artist=True)
        box_plot["boxes"][0].set_facecolor("lightblue")

        # Highlight extreme values
        outliers = result.weights[result.weights > extreme_threshold]
        if len(outliers) > 0:
            ax2.scatter([1] * len(outliers), outliers, color="red", alpha=0.6, s=30)

        ax2.set_ylabel("Weight Value")
        ax2.set_title("Weight Distribution\n(Box Plot)")
        ax2.set_xticklabels(["Weights"])

        # 3. Q-Q plot for normality assessment
        ax3 = fig.add_subplot(gs[1, 0])
        stats.probplot(result.weights, dist="norm", plot=ax3)
        ax3.set_title("Q-Q Plot\n(Normality Check)")
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative distribution
        ax4 = fig.add_subplot(gs[1, 1])
        sorted_weights = np.sort(result.weights)
        cumulative_prob = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)

        ax4.plot(sorted_weights, cumulative_prob, "b-", linewidth=2)
        ax4.axvline(
            extreme_threshold,
            color="red",
            linestyle="--",
            label=f"Extreme Threshold: {extreme_threshold}",
        )
        ax4.set_xlabel("Weight Value")
        ax4.set_ylabel("Cumulative Probability")
        ax4.set_title("Cumulative Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Log scale histogram (if weights span large range)
        ax5 = fig.add_subplot(gs[1, 2])
        if result.max_weight / result.min_weight > 100:  # Large range
            log_weights = np.log10(
                result.weights + 1e-10
            )  # Add small constant to avoid log(0)
            ax5.hist(
                log_weights,
                bins=n_bins,
                alpha=0.7,
                color="lightgreen",
                edgecolor="black",
            )
            ax5.set_xlabel("Log10(Weight)")
            ax5.set_ylabel("Frequency")
            ax5.set_title("Log-Scale Distribution")
        else:
            # Alternative: scatter plot of weights vs. observation index
            ax5.scatter(range(len(result.weights)), result.weights, alpha=0.5, s=20)
            ax5.axhline(extreme_threshold, color="red", linestyle="--")
            ax5.set_xlabel("Observation Index")
            ax5.set_ylabel("Weight Value")
            ax5.set_title("Weight by Observation")
        ax5.grid(True, alpha=0.3)

        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")

        # Create summary table
        summary_data = [
            ["Statistic", "Value"],
            ["N Observations", f"{result.n_observations:,}"],
            ["Mean Weight", f"{result.mean_weight:.3f}"],
            ["Median Weight", f"{result.median_weight:.3f}"],
            ["Std Dev", f"{result.std_weight:.3f}"],
            ["Min Weight", f"{result.min_weight:.3f}"],
            ["Max Weight", f"{result.max_weight:.3f}"],
            ["Skewness", f"{result.skewness:.3f}"],
            ["Kurtosis", f"{result.kurtosis:.3f}"],
            [
                "Extreme Weights",
                f"{result.extreme_weight_count} ({result.extreme_weight_percentage:.1f}%)",
            ],
            ["Effective Sample Size", f"{result.effective_sample_size:.1f}"],
        ]

        # Split into two columns
        mid_point = len(summary_data) // 2 + 1
        col1_data = summary_data[:mid_point]
        col2_data = summary_data[mid_point:]

        # Create table
        table1 = ax6.table(
            cellText=col1_data,
            cellLoc="left",
            loc="center left",
            bbox=[0.0, 0.0, 0.45, 1.0],
        )
        table2 = ax6.table(
            cellText=col2_data,
            cellLoc="left",
            loc="center right",
            bbox=[0.55, 0.0, 0.45, 1.0],
        )

        # Style tables
        for table in [table1, table2]:
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # Color header row
            for i in range(len(table._cells)):
                if i < 2:  # Header row
                    table._cells[i].set_facecolor("#40466e")
                    table._cells[i].set_text_props(weight="bold", color="white")
                else:
                    table._cells[i].set_facecolor("#f1f1f2")

        plt.suptitle(title, fontsize=16, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _create_interactive_weight_plots(
        self,
        result: WeightDiagnosticsResult,
        title: str,
        save_path: str | None,
    ) -> go.Figure:
        """Create interactive Plotly weight plots."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Weight Distribution",
                "Box Plot",
                "Cumulative Distribution",
                "Weight by Observation",
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 1. Histogram
        fig.add_trace(
            go.Histogram(
                x=result.weights,
                nbinsx=min(50, max(10, int(np.sqrt(result.n_observations)))),
                name="Weight Distribution",
                opacity=0.7,
                marker_color="skyblue",
            ),
            row=1,
            col=1,
        )

        # Add vertical lines for statistics
        fig.add_vline(
            x=result.mean_weight,
            line_dash="solid",
            line_color="red",
            annotation_text=f"Mean: {result.mean_weight:.2f}",
            row=1,
            col=1,
        )

        # 2. Box plot
        fig.add_trace(
            go.Box(
                y=result.weights,
                name="Weights",
                boxpoints="outliers",
                marker_color="lightblue",
            ),
            row=1,
            col=2,
        )

        # 3. Cumulative distribution
        sorted_weights = np.sort(result.weights)
        cumulative_prob = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)

        fig.add_trace(
            go.Scatter(
                x=sorted_weights,
                y=cumulative_prob,
                mode="lines",
                name="Cumulative Distribution",
                line=dict(color="blue", width=2),
            ),
            row=2,
            col=1,
        )

        # 4. Scatter plot of weights
        fig.add_trace(
            go.Scatter(
                x=list(range(len(result.weights))),
                y=result.weights,
                mode="markers",
                name="Weights by Observation",
                marker=dict(size=5, opacity=0.6),
                hovertemplate="Observation: %{x}<br>Weight: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=800,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def generate_recommendations(
        self,
        diagnostics_result: WeightDiagnosticsResult,
    ) -> list[str]:
        """Generate actionable recommendations based on weight diagnostics.

        Args:
            diagnostics_result: Results from weight analysis

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for extreme weights
        if diagnostics_result.extreme_weight_percentage > 5:
            recommendations.append(
                f"⚠️ High proportion of extreme weights ({diagnostics_result.extreme_weight_percentage:.1f}%). "
                f"Consider trimming weights above {diagnostics_result.recommended_trimming_threshold:.2f}."
            )
        elif diagnostics_result.extreme_weight_percentage > 1:
            recommendations.append(
                f"⚠️ Some extreme weights detected ({diagnostics_result.extreme_weight_percentage:.1f}%). "
                "Monitor for impact on variance."
            )
        else:
            recommendations.append(
                "✅ Weight distribution appears reasonable with few extreme values."
            )

        # Check effective sample size
        ess_ratio = (
            diagnostics_result.effective_sample_size / diagnostics_result.n_observations
        )
        if ess_ratio < 0.5:
            recommendations.append(
                f"⚠️ Low effective sample size ({diagnostics_result.effective_sample_size:.0f}, "
                f"{ess_ratio:.1%} of original). High weight variability detected."
            )
        elif ess_ratio < 0.8:
            recommendations.append(
                f"⚠️ Moderate effective sample size ({diagnostics_result.effective_sample_size:.0f}, "
                f"{ess_ratio:.1%} of original). Some precision loss expected."
            )
        else:
            recommendations.append(
                f"✅ Good effective sample size ({diagnostics_result.effective_sample_size:.0f}, "
                f"{ess_ratio:.1%} of original)."
            )

        # Check for heavy tails
        if diagnostics_result.kurtosis > 10:
            recommendations.append(
                "⚠️ Very heavy-tailed weight distribution detected. "
                "Consider weight stabilization or trimming."
            )
        elif diagnostics_result.kurtosis > 5:
            recommendations.append(
                "⚠️ Heavy-tailed weight distribution. Monitor for outlier influence."
            )

        # Check for skewness
        if abs(diagnostics_result.skewness) > 2:
            recommendations.append(
                f"⚠️ Highly skewed weight distribution (skewness: {diagnostics_result.skewness:.2f}). "
                "Consider log transformation or alternative weighting approach."
            )

        return recommendations


def create_weight_plots(
    weights: NDArray[np.floating[Any]],
    extreme_weight_threshold: float = 10.0,
    title: str = "Weight Distribution Diagnostics",
    save_path: str | None = None,
    interactive: bool = False,
) -> tuple[plt.Figure | go.Figure, WeightDiagnosticsResult, list[str]]:
    """Convenience function to create weight diagnostic plots.

    Args:
        weights: Array of weights to analyze
        extreme_weight_threshold: Threshold for flagging extreme weights
        title: Plot title
        save_path: Path to save the plot
        interactive: Whether to create interactive plot

    Returns:
        Tuple of (figure, diagnostics_result, recommendations)
    """
    analyzer = WeightDiagnostics(extreme_weight_threshold=extreme_weight_threshold)
    diagnostics_result = analyzer.analyze_weights(weights)
    figure = analyzer.create_weight_plots(
        diagnostics_result, title, save_path, interactive
    )
    recommendations = analyzer.generate_recommendations(diagnostics_result)

    return figure, diagnostics_result, recommendations
