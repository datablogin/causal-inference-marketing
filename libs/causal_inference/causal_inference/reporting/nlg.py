"""Natural Language Generation for business insights and recommendations.

This module translates technical causal inference results into business-friendly
language that stakeholders can understand and act upon.
"""

from __future__ import annotations

import pandas as pd

from ..core.base import CausalEffect


class BusinessInsightsGenerator:
    """Generate natural language insights and recommendations from causal analysis."""

    def __init__(
        self,
        effect: CausalEffect,
        data: pd.DataFrame,
        treatment_column: str,
        outcome_column: str,
        method_name: str = "causal_analysis",
    ):
        """Initialize the insights generator.

        Args:
            effect: Estimated causal effect
            data: Analysis data
            treatment_column: Name of treatment column
            outcome_column: Name of outcome column
            method_name: Name of estimation method
        """
        self.effect = effect
        self.data = data
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.method_name = method_name

    def generate_executive_summary(self) -> str:
        """Generate executive summary in business language."""

        # Determine effect direction and magnitude
        effect_direction = "positive" if self.effect.ate > 0 else "negative"
        effect_magnitude = self._classify_effect_magnitude()
        p_value = getattr(self.effect, "p_value", None)
        significance = p_value is not None and p_value < 0.05

        # Get sample characteristics
        n_treated = sum(self.data[self.treatment_column])
        treatment_rate = n_treated / len(self.data)

        # Build executive summary
        summary_parts = []

        # Opening statement
        treatment_name = self.treatment_column.replace("_", " ").title()
        outcome_name = self.outcome_column.replace("_", " ").title()

        if significance:
            summary_parts.append(
                f"<p><strong>Key Finding:</strong> The analysis reveals a statistically significant "
                f"{effect_direction} effect of {treatment_name} on {outcome_name}. "
                f"The treatment shows a <strong>{effect_magnitude}</strong> impact with high confidence.</p>"
            )
        else:
            summary_parts.append(
                f"<p><strong>Key Finding:</strong> The analysis does not show a statistically significant "
                f"effect of {treatment_name} on {outcome_name}. Any observed differences "
                f"may be due to random variation rather than a true causal effect.</p>"
            )

        # Sample description
        summary_parts.append(
            f"<p><strong>Analysis Overview:</strong> This analysis examined {len(self.data):,} observations, "
            f"with {n_treated:,} individuals receiving the treatment ({treatment_rate:.1%} treatment rate). "
            f"We used {self._get_method_description()} to estimate causal effects.</p>"
        )

        # Effect size interpretation
        if significance:
            if self.effect.ate > 0:
                base_text = (
                    f"<p><strong>Business Impact:</strong> On average, the treatment increases "
                    f"{outcome_name} by {self.effect.ate:.3f} units."
                )
                if (
                    self.effect.ate_ci_lower is not None
                    and self.effect.ate_ci_upper is not None
                ):
                    base_text += (
                        f" Based on our confidence interval "
                        f"[{self.effect.ate_ci_lower:.3f}, {self.effect.ate_ci_upper:.3f}], the true effect "
                        f"is likely between these values with 95% confidence."
                    )
                base_text += "</p>"
                summary_parts.append(base_text)
            else:
                summary_parts.append(
                    f"<p><strong>Business Impact:</strong> On average, the treatment decreases "
                    f"{outcome_name} by {abs(self.effect.ate):.3f} units. This suggests the treatment "
                    f"may have unintended negative consequences that should be investigated further.</p>"
                )

        return "\\n".join(summary_parts)

    def interpret_effect_size(self) -> str:
        """Provide interpretation of the effect size."""
        magnitude = self._classify_effect_magnitude()
        direction = "increase" if self.effect.ate > 0 else "decrease"

        outcome_std = self.data[self.outcome_column].std()
        if outcome_std > 0:
            standardized_effect = abs(self.effect.ate) / outcome_std

            if standardized_effect < 0.2:
                size_desc = "very small"
            elif standardized_effect < 0.5:
                size_desc = "small"
            elif standardized_effect < 0.8:
                size_desc = "medium"
            else:
                size_desc = "large"

            return (
                f"The effect represents a {size_desc} {direction} "
                f"({standardized_effect:.2f} standard deviations) in {self.outcome_column.replace('_', ' ')}."
            )
        else:
            return f"The effect represents a {magnitude} {direction} in {self.outcome_column.replace('_', ' ')}."

    def generate_recommendations(self) -> list[str]:
        """Generate business recommendations based on results."""
        recommendations = []

        p_value = getattr(self.effect, "p_value", None)
        significance = p_value is not None and p_value < 0.05

        if significance and self.effect.ate > 0:
            # Positive significant effect
            recommendations.extend(
                [
                    "✅ **Implement the treatment**: The analysis shows clear evidence of positive impact. "
                    "Consider scaling this intervention to a broader population.",
                    f"📊 **Monitor key metrics**: Track {self.outcome_column.replace('_', ' ')} closely "
                    f"during rollout to confirm these results hold in practice.",
                    "🎯 **Optimize treatment delivery**: Investigate ways to enhance the treatment "
                    "to potentially increase the observed effect size.",
                ]
            )

            # Add ROI consideration if outcome appears monetary
            if any(
                word in self.outcome_column.lower()
                for word in ["revenue", "profit", "value", "sales"]
            ):
                recommendations.append(
                    f"💰 **Calculate ROI**: With an average increase of {self.effect.ate:.2f} "
                    f"per treated individual, estimate the return on investment for full deployment."
                )

        elif significance and self.effect.ate < 0:
            # Negative significant effect
            recommendations.extend(
                [
                    "⚠️ **Reconsider the intervention**: The treatment appears to have negative effects. "
                    "Investigate why this is happening before proceeding.",
                    "🔍 **Analyze subgroups**: The overall negative effect might mask positive effects "
                    "in certain segments. Consider subgroup analysis.",
                    "🛠️ **Modify the treatment**: Consider adjusting the intervention design "
                    "to address the factors causing negative outcomes.",
                ]
            )

        else:
            # Non-significant results
            recommendations.extend(
                [
                    "🤔 **Inconclusive results**: No statistically significant effect detected. "
                    "Consider whether the treatment needs modification or more time to show impact.",
                    "📈 **Increase sample size**: If feasible, collect more data to improve statistical power "
                    "and detect smaller effects that might be practically meaningful.",
                    "⏱️ **Extend observation period**: Some effects may take time to manifest. "
                    "Consider a longer follow-up period if appropriate.",
                ]
            )

        # Add confidence interval considerations
        if (
            self.effect.ate_ci_upper is not None
            and self.effect.ate_ci_lower is not None
        ):
            ci_width = self.effect.ate_ci_upper - self.effect.ate_ci_lower
            if ci_width > abs(self.effect.ate) * 2:  # Wide confidence interval
                recommendations.append(
                    "📊 **Note uncertainty**: The confidence interval is relatively wide, "
                    "indicating substantial uncertainty about the true effect size."
                )
        else:
            recommendations.append(
                "📊 **Note uncertainty**: No confidence intervals available. "
                "Consider using an estimator with bootstrap-based inference for uncertainty quantification."
            )

        return recommendations

    def suggest_next_steps(self) -> list[str]:
        """Suggest follow-up actions and next steps."""
        next_steps = []

        # Always suggest validation
        next_steps.extend(
            [
                "🔄 **Validate results**: Consider replicating this analysis with new data or a different time period",
                "📋 **Document assumptions**: Clearly document the key assumptions underlying this analysis for future reference",
            ]
        )

        # Method-specific suggestions
        if self.method_name == "ipw":
            next_steps.append(
                "⚖️ **Check propensity scores**: Review propensity score distribution and consider trimming extreme weights"
            )
        elif self.method_name == "g_computation":
            next_steps.append(
                "🎯 **Validate outcome model**: Check outcome model fit and consider alternative model specifications"
            )
        elif self.method_name == "aipw":
            next_steps.append(
                "🔍 **Review model diagnostics**: Check both propensity and outcome model performance"
            )

        # Sample-specific suggestions
        if len(self.data) < 1000:
            next_steps.append(
                "📊 **Consider larger sample**: Small sample size may limit statistical power and generalizability"
            )

        # Suggest sensitivity analysis if not already done
        next_steps.append(
            "🛡️ **Conduct sensitivity analysis**: Test robustness of results to unmeasured confounding and model assumptions"
        )

        return next_steps

    def list_caveats(self) -> list[str]:
        """List important caveats and limitations."""
        caveats = []

        # General causal inference caveats
        caveats.extend(
            [
                "🔮 **Causal assumptions**: This analysis assumes no unmeasured confounders affect both treatment and outcome",
                "📊 **Statistical assumptions**: Results depend on correct model specification and standard statistical assumptions",
            ]
        )

        # Sample-specific caveats
        sample_size = len(self.data)
        if sample_size < 500:
            caveats.append(
                f"📉 **Small sample size**: With only {sample_size:,} observations, results should be interpreted cautiously"
            )

        # Treatment balance
        treatment_balance = self.data[self.treatment_column].mean()
        if treatment_balance < 0.1 or treatment_balance > 0.9:
            caveats.append(
                f"⚖️ **Imbalanced treatment**: {treatment_balance:.1%} treatment rate may limit analysis power"
            )

        # Confidence interval width
        if (
            hasattr(self.effect, "ate_ci_lower")
            and hasattr(self.effect, "ate_ci_upper")
            and self.effect.ate_ci_lower is not None
            and self.effect.ate_ci_upper is not None
        ):
            ci_width = self.effect.ate_ci_upper - self.effect.ate_ci_lower
            if ci_width > abs(self.effect.ate):
                caveats.append(
                    "📊 **Wide confidence interval**: Substantial uncertainty around the point estimate"
                )

        # Method-specific limitations
        method_caveats = {
            "g_computation": "🎯 **Model dependence**: Results are sensitive to outcome model specification",
            "ipw": "⚖️ **Weight sensitivity**: Results can be sensitive to extreme propensity score weights",
            "aipw": "🔧 **Model complexity**: Requires careful specification of both propensity and outcome models",
        }

        if self.method_name in method_caveats:
            caveats.append(method_caveats[self.method_name])

        return caveats

    def _classify_effect_magnitude(self) -> str:
        """Classify effect magnitude for business communication."""
        abs_effect = abs(self.effect.ate)

        # Get outcome scale for context
        outcome_range = (
            self.data[self.outcome_column].max() - self.data[self.outcome_column].min()
        )

        if outcome_range > 0:
            relative_effect = abs_effect / outcome_range

            if relative_effect < 0.01:
                return "very small"
            elif relative_effect < 0.05:
                return "small"
            elif relative_effect < 0.15:
                return "moderate"
            elif relative_effect < 0.30:
                return "large"
            else:
                return "very large"
        else:
            return "moderate"

    def _get_method_description(self) -> str:
        """Get business-friendly description of analysis method."""
        descriptions = {
            "g_computation": "outcome modeling (G-computation)",
            "ipw": "propensity score weighting (IPW)",
            "aipw": "doubly robust estimation (AIPW)",
            "auto": "automated method selection",
        }

        return descriptions.get(self.method_name, f"{self.method_name} analysis")
