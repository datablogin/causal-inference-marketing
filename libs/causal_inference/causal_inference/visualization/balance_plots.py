"""Love plots and covariate balance visualizations.

This module implements Love plots and other balance visualization tools for
assessing covariate balance before and after adjustment in causal inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

# Import plotting libraries with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..core.base import CovariateData, TreatmentData
from ..diagnostics.balance import calculate_standardized_mean_difference


@dataclass
class LovePlotData:
    """Data structure for Love plot visualization."""

    covariate_names: list[str]
    smd_before: NDArray[np.floating[Any]]
    smd_after: Optional[NDArray[np.floating[Any]]] = None
    balance_threshold: float = 0.1
    poor_balance_threshold: float = 0.25


class LovePlotGenerator:
    """Generator for Love plots (standardized mean difference plots)."""

    def __init__(
        self,
        balance_threshold: float = 0.1,
        poor_balance_threshold: float = 0.25,
        style: str = "whitegrid",
        figsize: tuple[int, int] = (10, 8),
    ):
        """Initialize Love plot generator.

        Args:
            balance_threshold: SMD threshold for "good balance" (default 0.1)
            poor_balance_threshold: SMD threshold for "poor balance" (default 0.25)
            style: Plotting style for matplotlib/seaborn
            figsize: Figure size for plots
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting libraries not available. Install with: "
                "pip install matplotlib seaborn"
            )

        self.balance_threshold = balance_threshold
        self.poor_balance_threshold = poor_balance_threshold
        self.style = style
        self.figsize = figsize

        # Set plotting style
        sns.set_style(style)

    def calculate_balance_data(
        self,
        covariates: CovariateData,
        treatment: TreatmentData,
        weights_before: Optional[NDArray[np.floating[Any]]] = None,
        weights_after: Optional[NDArray[np.floating[Any]]] = None,
    ) -> LovePlotData:
        """Calculate standardized mean differences for Love plot.

        Args:
            covariates: Covariate data
            treatment: Treatment assignment data
            weights_before: Weights for before-adjustment calculation
            weights_after: Weights for after-adjustment calculation

        Returns:
            LovePlotData with SMD calculations
        """
        covariate_names = covariates.names
        n_covariates = len(covariate_names)

        smd_before = np.zeros(n_covariates)
        smd_after = np.zeros(n_covariates) if weights_after is not None else None

        for i, name in enumerate(covariate_names):
            covariate_values = covariates.values[:, i]

            # Calculate SMD before adjustment
            smd_before[i] = calculate_standardized_mean_difference(
                covariate_values, treatment.values
            )

            # Calculate SMD after adjustment if weights provided
            if weights_after is not None and smd_after is not None:
                # For weighted SMD, we need to modify the calculation
                # This is a simplified version - full implementation would use proper weighted SMD
                smd_after[i] = self._calculate_weighted_smd(
                    covariate_values, treatment.values, weights_after
                )

        return LovePlotData(
            covariate_names=covariate_names,
            smd_before=smd_before,
            smd_after=smd_after,
            balance_threshold=self.balance_threshold,
            poor_balance_threshold=self.poor_balance_threshold,
        )

    def _calculate_weighted_smd(
        self,
        covariate_values: NDArray[Any],
        treatment_values: NDArray[Any],
        weights: NDArray[np.floating[Any]],
    ) -> float:
        """Calculate weighted standardized mean difference."""
        treated_mask = treatment_values == 1
        control_mask = treatment_values == 0

        # Weighted means
        treated_weights = weights[treated_mask]
        control_weights = weights[control_mask]

        if np.sum(treated_weights) == 0 or np.sum(control_weights) == 0:
            return np.nan

        treated_mean = np.average(
            covariate_values[treated_mask], weights=treated_weights
        )
        control_mean = np.average(
            covariate_values[control_mask], weights=control_weights
        )

        # Weighted variances for pooled standard deviation
        treated_var = np.average(
            (covariate_values[treated_mask] - treated_mean) ** 2,
            weights=treated_weights,
        )
        control_var = np.average(
            (covariate_values[control_mask] - control_mean) ** 2,
            weights=control_weights,
        )

        pooled_std = np.sqrt((treated_var + control_var) / 2)

        if pooled_std == 0:
            return 0.0

        return (treated_mean - control_mean) / pooled_std

    def create_love_plot(
        self,
        love_plot_data: LovePlotData,
        title: str = "Covariate Balance (Love Plot)",
        save_path: Optional[str] = None,
        interactive: bool = False,
    ) -> Union[plt.Figure, go.Figure]:
        """Create a Love plot visualization.

        Args:
            love_plot_data: Data for the Love plot
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive Plotly plot

        Returns:
            Matplotlib or Plotly figure
        """
        if interactive and not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available for interactive plots")

        if interactive:
            return self._create_interactive_love_plot(love_plot_data, title, save_path)
        else:
            return self._create_static_love_plot(love_plot_data, title, save_path)

    def _create_static_love_plot(
        self,
        data: LovePlotData,
        title: str,
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create static matplotlib Love plot."""
        fig, ax = plt.subplots(figsize=self.figsize)

        y_pos = np.arange(len(data.covariate_names))

        # Plot before adjustment
        colors_before = []
        for smd in data.smd_before:
            abs_smd = abs(smd)
            if abs_smd <= data.balance_threshold:
                colors_before.append("#2E8B57")  # Good balance - green
            elif abs_smd <= data.poor_balance_threshold:
                colors_before.append("#FF8C00")  # Moderate imbalance - orange
            else:
                colors_before.append("#DC143C")  # Poor balance - red

        ax.scatter(
            data.smd_before,
            y_pos,
            c=colors_before,
            s=100,
            alpha=0.7,
            label="Before Adjustment",
            marker="o",
        )

        # Plot after adjustment if available
        if data.smd_after is not None:
            colors_after = []
            for smd in data.smd_after:
                abs_smd = abs(smd)
                if abs_smd <= data.balance_threshold:
                    colors_after.append("#228B22")  # Good balance - darker green
                elif abs_smd <= data.poor_balance_threshold:
                    colors_after.append("#FF4500")  # Moderate imbalance - red-orange
                else:
                    colors_after.append("#B22222")  # Poor balance - dark red

            ax.scatter(
                data.smd_after,
                y_pos,
                c=colors_after,
                s=100,
                alpha=0.7,
                label="After Adjustment",
                marker="s",
            )

        # Add threshold lines
        ax.axvline(
            data.balance_threshold,
            color="green",
            linestyle="--",
            alpha=0.5,
            label=f"Balance Threshold (±{data.balance_threshold})",
        )
        ax.axvline(-data.balance_threshold, color="green", linestyle="--", alpha=0.5)

        ax.axvline(
            data.poor_balance_threshold,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Poor Balance Threshold (±{data.poor_balance_threshold})",
        )
        ax.axvline(-data.poor_balance_threshold, color="red", linestyle="--", alpha=0.5)

        # Customize plot
        ax.set_xlabel("Standardized Mean Difference")
        ax.set_ylabel("Covariates")
        ax.set_title(title)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data.covariate_names)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add balance summary
        if data.smd_after is not None:
            balanced_before = np.sum(np.abs(data.smd_before) <= data.balance_threshold)
            balanced_after = np.sum(np.abs(data.smd_after) <= data.balance_threshold)
            total = len(data.covariate_names)

            summary_text = (
                f"Balanced Covariates:\n"
                f"Before: {balanced_before}/{total} ({balanced_before / total:.1%})\n"
                f"After: {balanced_after}/{total} ({balanced_after / total:.1%})"
            )
        else:
            balanced = np.sum(np.abs(data.smd_before) <= data.balance_threshold)
            total = len(data.covariate_names)
            summary_text = f"Balanced: {balanced}/{total} ({balanced / total:.1%})"

        ax.text(
            0.02,
            0.98,
            summary_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _create_interactive_love_plot(
        self,
        data: LovePlotData,
        title: str,
        save_path: Optional[str],
    ) -> go.Figure:
        """Create interactive Plotly Love plot."""
        fig = go.Figure()

        # Create color mappings
        def get_color(smd: float) -> str:
            abs_smd = abs(smd)
            if abs_smd <= data.balance_threshold:
                return "#2E8B57"  # Good balance
            elif abs_smd <= data.poor_balance_threshold:
                return "#FF8C00"  # Moderate imbalance
            else:
                return "#DC143C"  # Poor balance

        # Before adjustment
        colors_before = [get_color(smd) for smd in data.smd_before]

        fig.add_trace(
            go.Scatter(
                x=data.smd_before,
                y=data.covariate_names,
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors_before,
                    symbol="circle",
                    line=dict(width=1, color="black"),
                ),
                name="Before Adjustment",
                hovertemplate=("<b>%{y}</b><br>SMD: %{x:.3f}<br><extra></extra>"),
            )
        )

        # After adjustment if available
        if data.smd_after is not None:
            colors_after = [get_color(smd) for smd in data.smd_after]

            fig.add_trace(
                go.Scatter(
                    x=data.smd_after,
                    y=data.covariate_names,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=colors_after,
                        symbol="square",
                        line=dict(width=1, color="black"),
                    ),
                    name="After Adjustment",
                    hovertemplate=("<b>%{y}</b><br>SMD: %{x:.3f}<br><extra></extra>"),
                )
            )

        # Add threshold lines
        fig.add_vline(
            x=data.balance_threshold,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Balance Threshold (+{data.balance_threshold})",
        )
        fig.add_vline(
            x=-data.balance_threshold,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Balance Threshold (-{data.balance_threshold})",
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Standardized Mean Difference",
            yaxis_title="Covariates",
            hovermode="closest",
            showlegend=True,
            height=max(400, len(data.covariate_names) * 30),
        )

        if save_path:
            fig.write_html(save_path)

        return fig


def create_love_plot(
    covariates: CovariateData,
    treatment: TreatmentData,
    weights_before: Optional[NDArray[np.floating[Any]]] = None,
    weights_after: Optional[NDArray[np.floating[Any]]] = None,
    balance_threshold: float = 0.1,
    title: str = "Covariate Balance (Love Plot)",
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> Union[plt.Figure, go.Figure]:
    """Convenience function to create a Love plot.

    Args:
        covariates: Covariate data
        treatment: Treatment assignment data
        weights_before: Weights for before-adjustment calculation
        weights_after: Weights for after-adjustment calculation
        balance_threshold: SMD threshold for good balance
        title: Plot title
        save_path: Path to save the plot
        interactive: Whether to create interactive plot

    Returns:
        Matplotlib or Plotly figure
    """
    generator = LovePlotGenerator(balance_threshold=balance_threshold)
    love_data = generator.calculate_balance_data(
        covariates, treatment, weights_before, weights_after
    )
    return generator.create_love_plot(love_data, title, save_path, interactive)
