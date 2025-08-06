import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducible results
np.random.seed(42)


def smooth_s_curve(
    x, max_revenue=1400000, growth_rate=0.000006, inflection=350000, baseline=80000
):
    """Smooth logistic S-curve for marketing spend vs revenue"""
    return baseline + (max_revenue - baseline) / (
        1 + np.exp(-growth_rate * (x - inflection))
    )


# Generate realistic spend data
spend = np.sort(np.random.lognormal(mean=12.5, sigma=0.8, size=150))
spend = np.clip(spend, 0, 1500000)

# Create smooth trend line
spend_smooth = np.linspace(0, 1500000, 500)
revenue_smooth = smooth_s_curve(spend_smooth)

# Add realistic noise to data points
revenue_data = smooth_s_curve(spend) + np.random.normal(0, 25000, len(spend))

# Create clean plot
plt.figure(figsize=(10, 6))
plt.scatter(spend / 1000, revenue_data / 1000, alpha=0.6, color="steelblue", s=30)
plt.plot(spend_smooth / 1000, revenue_smooth / 1000, color="crimson", linewidth=3)

plt.title("Marketing S-Curve: Spend vs Revenue", fontsize=14, fontweight="bold")
plt.xlabel("Ad Spend ($000s)")
plt.ylabel("Revenue ($000s)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
