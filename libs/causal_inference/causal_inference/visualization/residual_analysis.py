"""Residual analysis and model diagnostic tools.

This module provides comprehensive residual analysis capabilities for outcome
models, including normality tests, heteroscedasticity detection, and outlier
identification.
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
class ResidualAnalysisResult:
    """Results from residual analysis."""

    residuals: NDArray[np.floating[Any]]
    fitted_values: NDArray[np.floating[Any]]
    standardized_residuals: NDArray[np.floating[Any]]
    studentized_residuals: NDArray[np.floating[Any]]
    leverage_values: NDArray[np.floating[Any]] | None
    cooks_distance: NDArray[np.floating[Any]] | None

    # Normality tests
    shapiro_stat: float
    shapiro_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float

    # Heteroscedasticity tests
    breusch_pagan_stat: float
    breusch_pagan_pvalue: float
    white_stat: float | None
    white_pvalue: float | None

    # Summary statistics
    outlier_count: int
    high_leverage_count: int
    influential_count: int

    # Recommendations
    normality_assumption_met: bool
    homoscedasticity_assumption_met: bool
    outliers_detected: bool


class ResidualAnalyzer:
    """Analyzer for residual diagnostics and model validation."""

    def __init__(
        self,
        outlier_threshold: float = 2.5,
        leverage_threshold: float | None = None,
        influence_threshold: float = 1.0,
        figsize: tuple[int, int] = (15, 12),
        style: str = "whitegrid",
    ):
        """Initialize residual analyzer.

        Args:
            outlier_threshold: Z-score threshold for outlier detection
            leverage_threshold: Leverage threshold (if None, uses 2*p/n rule)
            influence_threshold: Cook's distance threshold for influential points
            figsize: Figure size for plots
            style: Plotting style
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting libraries not available. Install with: "
                "pip install matplotlib seaborn"
            )

        self.outlier_threshold = outlier_threshold
        self.leverage_threshold = leverage_threshold
        self.influence_threshold = influence_threshold
        self.figsize = figsize
        self.style = style

        sns.set_style(style)

    def analyze_residuals(
        self,
        residuals: NDArray[np.floating[Any]],
        fitted_values: NDArray[np.floating[Any]],
        design_matrix: NDArray[Any] | None = None,
    ) -> ResidualAnalysisResult:
        """Perform comprehensive residual analysis.

        Args:
            residuals: Model residuals
            fitted_values: Model fitted values
            design_matrix: Design matrix for leverage calculation

        Returns:
            ResidualAnalysisResult with diagnostic information
        """
        residuals = np.asarray(residuals)
        fitted_values = np.asarray(fitted_values)
        n = len(residuals)

        # Standardized residuals
        residual_std = np.std(residuals, ddof=1)
        standardized_residuals = (
            residuals / residual_std if residual_std > 0 else residuals
        )

        # Studentized residuals (simplified version)
        studentized_residuals = standardized_residuals

        # Leverage and influence (if design matrix provided)
        leverage_values = None
        cooks_distance = None

        if design_matrix is not None:
            try:
                X = np.asarray(design_matrix)
                # Calculate hat matrix diagonal (leverage)
                XtX_inv = np.linalg.pinv(X.T @ X)
                leverage_values = np.diag(X @ XtX_inv @ X.T)

                # Cook's distance approximation
                p = X.shape[1]
                mse = np.sum(residuals**2) / (n - p)
                cooks_distance = (standardized_residuals**2 * leverage_values) / (
                    p * mse
                )

            except (np.linalg.LinAlgError, ValueError):
                # If matrix operations fail, continue without leverage/influence
                pass

        # Normality tests
        shapiro_stat, shapiro_pvalue = stats.shapiro(
            residuals[:5000]
        )  # Limit for large samples
        jarque_bera_stat, jarque_bera_pvalue = stats.jarque_bera(residuals)

        # Heteroscedasticity tests
        breusch_pagan_stat, breusch_pagan_pvalue = self._breusch_pagan_test(
            residuals, fitted_values
        )

        white_stat, white_pvalue = None, None
        if design_matrix is not None:
            try:
                white_stat, white_pvalue = self._white_test(residuals, design_matrix)
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Outlier detection
        outlier_mask = np.abs(standardized_residuals) > self.outlier_threshold
        outlier_count = np.sum(outlier_mask)

        # High leverage detection
        high_leverage_count = 0
        if leverage_values is not None:
            if self.leverage_threshold is None:
                p = design_matrix.shape[1] if design_matrix is not None else 1
                leverage_threshold = 2 * p / n
            else:
                leverage_threshold = self.leverage_threshold

            high_leverage_mask = leverage_values > leverage_threshold
            high_leverage_count = np.sum(high_leverage_mask)

        # Influential points
        influential_count = 0
        if cooks_distance is not None:
            influential_mask = cooks_distance > self.influence_threshold
            influential_count = np.sum(influential_mask)

        # Assumption checks
        normality_met = shapiro_pvalue > 0.05 and jarque_bera_pvalue > 0.05
        homoscedasticity_met = breusch_pagan_pvalue > 0.05
        outliers_detected = outlier_count > 0

        return ResidualAnalysisResult(
            residuals=residuals,
            fitted_values=fitted_values,
            standardized_residuals=standardized_residuals,
            studentized_residuals=studentized_residuals,
            leverage_values=leverage_values,
            cooks_distance=cooks_distance,
            shapiro_stat=shapiro_stat,
            shapiro_pvalue=shapiro_pvalue,
            jarque_bera_stat=jarque_bera_stat,
            jarque_bera_pvalue=jarque_bera_pvalue,
            breusch_pagan_stat=breusch_pagan_stat,
            breusch_pagan_pvalue=breusch_pagan_pvalue,
            white_stat=white_stat,
            white_pvalue=white_pvalue,
            outlier_count=outlier_count,
            high_leverage_count=high_leverage_count,
            influential_count=influential_count,
            normality_assumption_met=normality_met,
            homoscedasticity_assumption_met=homoscedasticity_met,
            outliers_detected=outliers_detected,
        )

    def _breusch_pagan_test(
        self,
        residuals: NDArray[np.floating[Any]],
        fitted_values: NDArray[np.floating[Any]],
    ) -> tuple[float, float]:
        """Perform Breusch-Pagan test for heteroscedasticity."""
        squared_residuals = residuals**2

        # Simple regression of squared residuals on fitted values
        X = np.column_stack([np.ones(len(fitted_values)), fitted_values])

        try:
            beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
            fitted_squared = X @ beta

            # Calculate test statistic
            tss = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)
            ess = np.sum((fitted_squared - np.mean(squared_residuals)) ** 2)

            if tss > 0:
                r_squared = ess / tss
                n = len(residuals)
                lm_stat = n * r_squared
                p_value = 1 - stats.chi2.cdf(lm_stat, 1)

                return lm_stat, p_value
            else:
                return 0.0, 1.0

        except np.linalg.LinAlgError:
            return 0.0, 1.0

    def _white_test(
        self,
        residuals: NDArray[np.floating[Any]],
        design_matrix: NDArray[Any],
    ) -> tuple[float, float]:
        """Perform White test for heteroscedasticity."""
        X = np.asarray(design_matrix)
        squared_residuals = residuals**2

        # Create expanded matrix with squares and cross-products
        n, p = X.shape
        expanded_X = []

        # Add original variables
        expanded_X.append(X)

        # Add squared terms
        for i in range(p):
            expanded_X.append((X[:, i] ** 2).reshape(-1, 1))

        # Add interaction terms (simplified - only first few to avoid explosion)
        max_interactions = min(5, p * (p - 1) // 2)
        interaction_count = 0

        for i in range(p):
            for j in range(i + 1, p):
                if interaction_count < max_interactions:
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    expanded_X.append(interaction)
                    interaction_count += 1

        # Combine all terms
        X_expanded = np.hstack(expanded_X)

        try:
            # Regression of squared residuals on expanded matrix
            beta = np.linalg.lstsq(X_expanded, squared_residuals, rcond=None)[0]
            fitted_squared = X_expanded @ beta

            # Calculate R-squared
            tss = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)
            ess = np.sum((fitted_squared - np.mean(squared_residuals)) ** 2)

            if tss > 0:
                r_squared = ess / tss
                lm_stat = n * r_squared
                df = X_expanded.shape[1] - 1
                p_value = 1 - stats.chi2.cdf(lm_stat, df)

                return lm_stat, p_value
            else:
                return 0.0, 1.0

        except np.linalg.LinAlgError:
            return 0.0, 1.0

    def create_residual_plots(
        self,
        analysis_result: ResidualAnalysisResult,
        title: str = "Residual Analysis",
        save_path: str | None = None,
        interactive: bool = False,
    ) -> plt.Figure | go.Figure:
        """Create comprehensive residual diagnostic plots.

        Args:
            analysis_result: Residual analysis results
            title: Main title for the plots
            save_path: Path to save the plot
            interactive: Whether to create interactive plots

        Returns:
            Matplotlib or Plotly figure
        """
        if interactive and not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available for interactive plots")

        if interactive:
            return self._create_interactive_residual_plots(
                analysis_result, title, save_path
            )
        else:
            return self._create_static_residual_plots(analysis_result, title, save_path)

    def _create_static_residual_plots(
        self,
        result: ResidualAnalysisResult,
        title: str,
        save_path: str | None,
    ) -> plt.Figure:
        """Create static matplotlib residual plots."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Residuals vs Fitted
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.scatter(result.fitted_values, result.residuals, alpha=0.6, s=30)
        ax1.axhline(0, color="red", linestyle="--", linewidth=2)

        # Add LOWESS smoother
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(result.residuals, result.fitted_values, frac=0.3)
            ax1.plot(smoothed[:, 0], smoothed[:, 1], "r-", linewidth=2, alpha=0.8)
        except ImportError:
            pass

        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted")
        ax1.grid(True, alpha=0.3)

        # 2. Q-Q plot for normality
        ax2 = fig.add_subplot(gs[0, 1])
        stats.probplot(result.residuals, dist="norm", plot=ax2)
        ax2.set_title(f"Q-Q Plot\n(Shapiro p={result.shapiro_pvalue:.3f})")
        ax2.grid(True, alpha=0.3)

        # 3. Scale-Location plot
        ax3 = fig.add_subplot(gs[0, 2])

        sqrt_abs_residuals = np.sqrt(np.abs(result.standardized_residuals))
        ax3.scatter(result.fitted_values, sqrt_abs_residuals, alpha=0.6, s=30)

        # Add smoothed line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(sqrt_abs_residuals, result.fitted_values, frac=0.3)
            ax3.plot(smoothed[:, 0], smoothed[:, 1], "r-", linewidth=2, alpha=0.8)
        except ImportError:
            pass

        ax3.set_xlabel("Fitted Values")
        ax3.set_ylabel("√|Standardized Residuals|")
        ax3.set_title(f"Scale-Location\n(BP p={result.breusch_pagan_pvalue:.3f})")
        ax3.grid(True, alpha=0.3)

        # 4. Histogram of residuals
        ax4 = fig.add_subplot(gs[1, 0])

        n_bins = min(50, max(10, int(np.sqrt(len(result.residuals)))))
        counts, bins, patches = ax4.hist(
            result.residuals,
            bins=n_bins,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            density=True,
        )

        # Overlay normal distribution
        x_norm = np.linspace(result.residuals.min(), result.residuals.max(), 100)
        y_norm = stats.norm.pdf(
            x_norm, np.mean(result.residuals), np.std(result.residuals)
        )
        ax4.plot(x_norm, y_norm, "r-", linewidth=2, label="Normal")

        ax4.set_xlabel("Residuals")
        ax4.set_ylabel("Density")
        ax4.set_title("Residual Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Leverage plot (if available)
        ax5 = fig.add_subplot(gs[1, 1])

        if result.leverage_values is not None:
            ax5.scatter(
                range(len(result.leverage_values)),
                result.leverage_values,
                alpha=0.6,
                s=30,
            )

            # Add threshold line
            p = 1  # Simplified - would need actual number of parameters
            n = len(result.leverage_values)
            threshold = 2 * p / n
            ax5.axhline(
                threshold,
                color="red",
                linestyle="--",
                label=f"Threshold (2p/n = {threshold:.3f})",
            )

            ax5.set_xlabel("Observation Index")
            ax5.set_ylabel("Leverage")
            ax5.set_title("Leverage Values")
            ax5.legend()
        else:
            # Alternative: autocorrelation plot
            ax5.acorr(result.residuals, maxlags=20, alpha=0.6)
            ax5.set_title("Residual Autocorrelation")

        ax5.grid(True, alpha=0.3)

        # 6. Cook's Distance (if available)
        ax6 = fig.add_subplot(gs[1, 2])

        if result.cooks_distance is not None:
            ax6.scatter(
                range(len(result.cooks_distance)),
                result.cooks_distance,
                alpha=0.6,
                s=30,
            )
            ax6.axhline(
                self.influence_threshold,
                color="red",
                linestyle="--",
                label=f"Threshold ({self.influence_threshold})",
            )

            # Highlight influential points
            influential_mask = result.cooks_distance > self.influence_threshold
            if np.any(influential_mask):
                influential_indices = np.where(influential_mask)[0]
                ax6.scatter(
                    influential_indices,
                    result.cooks_distance[influential_mask],
                    color="red",
                    s=50,
                    alpha=0.8,
                    label="Influential",
                )

            ax6.set_xlabel("Observation Index")
            ax6.set_ylabel("Cook's Distance")
            ax6.set_title("Cook's Distance")
            ax6.legend()
        else:
            # Alternative: residuals vs index
            ax6.scatter(range(len(result.residuals)), result.residuals, alpha=0.6, s=30)
            ax6.axhline(0, color="red", linestyle="--")
            ax6.set_xlabel("Observation Index")
            ax6.set_ylabel("Residuals")
            ax6.set_title("Residuals vs Index")

        ax6.grid(True, alpha=0.3)

        # 7. Summary statistics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis("off")

        # Create comprehensive summary
        summary_data = [
            ["Test/Metric", "Value", "P-value", "Assessment"],
            [
                "Shapiro-Wilk (Normality)",
                f"{result.shapiro_stat:.4f}",
                f"{result.shapiro_pvalue:.4f}",
                "✅ Normal" if result.shapiro_pvalue > 0.05 else "❌ Non-normal",
            ],
            [
                "Jarque-Bera (Normality)",
                f"{result.jarque_bera_stat:.4f}",
                f"{result.jarque_bera_pvalue:.4f}",
                "✅ Normal" if result.jarque_bera_pvalue > 0.05 else "❌ Non-normal",
            ],
            [
                "Breusch-Pagan (Homoscedasticity)",
                f"{result.breusch_pagan_stat:.4f}",
                f"{result.breusch_pagan_pvalue:.4f}",
                "✅ Homoscedastic"
                if result.breusch_pagan_pvalue > 0.05
                else "❌ Heteroscedastic",
            ],
        ]

        if result.white_stat is not None:
            summary_data.append(
                [
                    "White Test (Homoscedasticity)",
                    f"{result.white_stat:.4f}",
                    f"{result.white_pvalue:.4f}",
                    "✅ Homoscedastic"
                    if result.white_pvalue > 0.05
                    else "❌ Heteroscedastic",
                ]
            )

        summary_data.extend(
            [
                [
                    "Outliers Detected",
                    f"{result.outlier_count}",
                    "-",
                    "✅ None"
                    if result.outlier_count == 0
                    else "⚠️ Some"
                    if result.outlier_count < 10
                    else "❌ Many",
                ],
                [
                    "High Leverage Points",
                    f"{result.high_leverage_count}",
                    "-",
                    "✅ None"
                    if result.high_leverage_count == 0
                    else "⚠️ Some"
                    if result.high_leverage_count < 5
                    else "❌ Many",
                ],
                [
                    "Influential Points",
                    f"{result.influential_count}",
                    "-",
                    "✅ None"
                    if result.influential_count == 0
                    else "⚠️ Some"
                    if result.influential_count < 5
                    else "❌ Many",
                ],
            ]
        )

        table = ax7.table(
            cellText=summary_data,
            cellLoc="center",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style table
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        for i in range(1, len(summary_data)):
            for j in range(len(summary_data[0])):
                if j == 3:  # Assessment column
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

    def _create_interactive_residual_plots(
        self,
        result: ResidualAnalysisResult,
        title: str,
        save_path: str | None,
    ) -> go.Figure:
        """Create interactive Plotly residual plots."""
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Residuals vs Fitted",
                "Q-Q Plot",
                "Scale-Location",
                "Residual Distribution",
                "Leverage (if available)",
                "Cook's Distance (if available)",
            ],
        )

        # 1. Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=result.fitted_values,
                y=result.residuals,
                mode="markers",
                name="Residuals",
                marker=dict(size=5, opacity=0.6),
                hovertemplate=(
                    "<b>Residuals vs Fitted</b><br>"
                    "Fitted Value: %{x:.3f}<br>"
                    "Residual: %{y:.3f}<br>"
                    "Observation: %{pointNumber}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # 2. Q-Q Plot (simplified)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(result.residuals))
        )
        sample_quantiles = np.sort(result.residuals)

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                name="Q-Q Plot",
                marker=dict(size=5, opacity=0.6),
            ),
            row=1,
            col=2,
        )

        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[sample_quantiles.min(), sample_quantiles.max()],
                mode="lines",
                name="Reference",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=2,
        )

        # 3. Scale-Location
        sqrt_abs_residuals = np.sqrt(np.abs(result.standardized_residuals))

        fig.add_trace(
            go.Scatter(
                x=result.fitted_values,
                y=sqrt_abs_residuals,
                mode="markers",
                name="Scale-Location",
                marker=dict(size=5, opacity=0.6),
            ),
            row=1,
            col=3,
        )

        # 4. Histogram
        fig.add_trace(
            go.Histogram(
                x=result.residuals,
                name="Residual Distribution",
                nbinsx=30,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        # 5. Leverage or alternative
        if result.leverage_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.leverage_values))),
                    y=result.leverage_values,
                    mode="markers",
                    name="Leverage",
                    marker=dict(size=5, opacity=0.6),
                ),
                row=2,
                col=2,
            )
        else:
            # Alternative: residuals by index
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.residuals))),
                    y=result.residuals,
                    mode="markers",
                    name="Residuals by Index",
                    marker=dict(size=5, opacity=0.6),
                ),
                row=2,
                col=2,
            )

        # 6. Cook's Distance or alternative
        if result.cooks_distance is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.cooks_distance))),
                    y=result.cooks_distance,
                    mode="markers",
                    name="Cook's Distance",
                    marker=dict(size=5, opacity=0.6),
                ),
                row=2,
                col=3,
            )
        else:
            # Alternative: standardized residuals
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.standardized_residuals))),
                    y=result.standardized_residuals,
                    mode="markers",
                    name="Standardized Residuals",
                    marker=dict(size=5, opacity=0.6),
                ),
                row=2,
                col=3,
            )

        # Update layout with enhanced interactivity
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=800,
            # Enhanced interactivity
            hovermode="closest",
            dragmode="zoom",
            # Add interactive controls
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=[{"dragmode": "zoom"}],
                                label="🔍 Zoom",
                                method="relayout",
                            ),
                            dict(
                                args=[{"dragmode": "pan"}],
                                label="✋ Pan",
                                method="relayout",
                            ),
                            dict(
                                args=[{"dragmode": "select"}],
                                label="📦 Select",
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

        # Enhanced axes with crossfilter spikes
        fig.update_xaxes(
            showspikes=True,
            spikecolor="blue",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1,
        )
        fig.update_yaxes(
            showspikes=True,
            spikecolor="red",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def generate_recommendations(
        self,
        analysis_result: ResidualAnalysisResult,
    ) -> list[str]:
        """Generate recommendations based on residual analysis.

        Args:
            analysis_result: Results from residual analysis

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Normality assessment
        if analysis_result.normality_assumption_met:
            recommendations.append("✅ Residuals appear normally distributed.")
        else:
            recommendations.append(
                "⚠️ Non-normal residuals detected. Consider robust standard errors "
                "or transformation of outcome variable."
            )

        # Homoscedasticity assessment
        if analysis_result.homoscedasticity_assumption_met:
            recommendations.append("✅ No evidence of heteroscedasticity.")
        else:
            recommendations.append(
                "⚠️ Heteroscedasticity detected. Consider robust standard errors "
                "or weighted least squares."
            )

        # Outlier assessment
        if not analysis_result.outliers_detected:
            recommendations.append("✅ No outliers detected.")
        elif analysis_result.outlier_count < 10:
            recommendations.append(
                f"⚠️ {analysis_result.outlier_count} outliers detected. "
                "Consider investigating these observations."
            )
        else:
            recommendations.append(
                f"❌ Many outliers detected ({analysis_result.outlier_count}). "
                "Consider robust estimation methods."
            )

        # Leverage assessment
        if analysis_result.high_leverage_count == 0:
            recommendations.append("✅ No high leverage points detected.")
        elif analysis_result.high_leverage_count < 5:
            recommendations.append(
                f"⚠️ {analysis_result.high_leverage_count} high leverage points detected. "
                "Monitor for influence on results."
            )
        else:
            recommendations.append(
                f"❌ Many high leverage points ({analysis_result.high_leverage_count}). "
                "Consider model specification or robust methods."
            )

        # Influence assessment
        if analysis_result.influential_count == 0:
            recommendations.append("✅ No influential points detected.")
        elif analysis_result.influential_count < 5:
            recommendations.append(
                f"⚠️ {analysis_result.influential_count} influential points detected. "
                "Consider sensitivity analysis without these points."
            )
        else:
            recommendations.append(
                f"❌ Many influential points ({analysis_result.influential_count}). "
                "Results may be driven by few observations."
            )

        return recommendations


def create_residual_plots(
    residuals: NDArray[np.floating[Any]],
    fitted_values: NDArray[np.floating[Any]],
    design_matrix: NDArray[Any] | None = None,
    title: str = "Residual Analysis",
    save_path: str | None = None,
    interactive: bool = False,
) -> tuple[plt.Figure | go.Figure, ResidualAnalysisResult, list[str]]:
    """Convenience function to create residual diagnostic plots.

    Args:
        residuals: Model residuals
        fitted_values: Model fitted values
        design_matrix: Design matrix for leverage calculation
        title: Plot title
        save_path: Path to save the plot
        interactive: Whether to create interactive plot

    Returns:
        Tuple of (figure, analysis_result, recommendations)
    """
    analyzer = ResidualAnalyzer()
    analysis_result = analyzer.analyze_residuals(
        residuals, fitted_values, design_matrix
    )
    figure = analyzer.create_residual_plots(
        analysis_result, title, save_path, interactive
    )
    recommendations = analyzer.generate_recommendations(analysis_result)

    return figure, analysis_result, recommendations
