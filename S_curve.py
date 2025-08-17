import matplotlib.pyplot as plt
import numpy as np


def logistic_curve(x, max_revenue, k, x0, b):
    """Enhanced logistic function with baseline
    max_revenue: Maximum revenue (upper asymptote)
    k: Growth rate (steepness of the curve)
    x0: Midpoint spend where growth accelerates
    b: Baseline revenue (lower asymptote)
    """
    return b + (max_revenue - b) / (1 + np.exp(-k * (x - x0)))


def adstock_transform(spend, adstock_rate=0.5):
    """Apply adstock (carryover effect) to spending"""
    adstocked = np.zeros_like(spend)
    for i in range(len(spend)):
        if i == 0:
            adstocked[i] = spend[i]
        else:
            adstocked[i] = spend[i] + adstock_rate * adstocked[i - 1]
    return adstocked


def hill_saturation(x, alpha=2.0, gamma=0.5):
    """Hill saturation curve for more realistic diminishing returns
    alpha: Shape parameter (higher = steeper saturation)
    gamma: Half-saturation point
    """
    return (x**alpha) / (gamma**alpha + x**alpha)


# Set random seed for reproducibility
np.random.seed(42)

# Parameters for more realistic S-curve
L = 1000000  # Maximum revenue (upper asymptote)
k = 0.000008  # Much smaller growth rate for smoother curve
x0 = 400000  # Midpoint spend
baseline = 50000  # Baseline revenue (what you get with zero spend)

# Generate advertising spend data with more realistic distribution
# More data points for smoother curve
n_points = 200
spend = np.sort(np.random.lognormal(mean=12.5, sigma=0.8, size=n_points))
spend = np.clip(spend, 0, 1500000)  # Cap at $1.2M

# Apply adstock transformation for carryover effects
spend_adstocked = adstock_transform(spend, adstock_rate=0.3)

# Create multiple curve variations for comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Original steep curve
revenue_steep = L / (1 + np.exp(-0.001 * (spend - x0))) + np.random.normal(
    0, 15000, len(spend)
)
ax1.scatter(spend / 1000, revenue_steep / 1000, color="blue", alpha=0.6, s=20)
ax1.plot(
    spend / 1000,
    L / (1 + np.exp(-0.001 * (spend - x0))) / 1000,
    color="red",
    linewidth=2,
)
ax1.set_title("Original Steep S-Curve", fontsize=12, fontweight="bold")
ax1.set_xlabel("Advertising Spend ($000s)")
ax1.set_ylabel("Revenue ($000s)")
ax1.grid(True, alpha=0.3)

# 2. Smoother logistic curve
revenue_smooth = logistic_curve(spend, L, k, x0, baseline) + np.random.normal(
    0, 1000, len(spend)
)
ax2.scatter(spend / 1000, revenue_smooth / 1000, color="blue", alpha=0.6, s=20)
ax2.plot(
    spend / 1000,
    logistic_curve(spend, L, k, x0, baseline) / 1000,
    color="green",
    linewidth=2,
)
ax2.set_title("Smooth Logistic S-Curve", fontsize=12, fontweight="bold")
ax2.set_xlabel("Advertising Spend ($000s)")
ax2.set_ylabel("Revenue ($000s)")
ax2.grid(True, alpha=0.3)

# 3. Hill saturation curve (very smooth)
spend_normalized = spend / np.max(spend)
hill_response = hill_saturation(spend_normalized, alpha=1.5, gamma=0.4)
revenue_hill = (
    baseline + (L - baseline) * hill_response + np.random.normal(0, 10000, len(spend))
)
ax3.scatter(spend / 1000, revenue_hill / 1000, color="blue", alpha=0.6, s=20)
ax3.plot(
    spend / 1000,
    (baseline + (L - baseline) * hill_response) / 1000,
    color="purple",
    linewidth=2,
)
ax3.set_title("Hill Saturation Curve (Very Smooth)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Advertising Spend ($000s)")
ax3.set_ylabel("Revenue ($000s)")
ax3.grid(True, alpha=0.3)

# 4. Adstocked smooth curve (most realistic)
revenue_adstock = logistic_curve(
    spend_adstocked, L * 0.9, k * 0.8, x0 * 0.7, baseline
) + np.random.normal(0, 8000, len(spend))
ax4.scatter(spend / 1000, revenue_adstock / 1000, color="blue", alpha=0.6, s=20)
ax4.plot(
    spend / 1000,
    logistic_curve(spend_adstocked, L * 0.9, k * 0.8, x0 * 0.7, baseline) / 1000,
    color="orange",
    linewidth=2,
)
ax4.set_title("Adstocked S-Curve (Most Realistic)", fontsize=12, fontweight="bold")
ax4.set_xlabel("Advertising Spend ($000s)")
ax4.set_ylabel("Revenue ($000s)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create a final "best" smooth S-curve
plt.figure(figsize=(12, 8))

# Use the smoothest, most realistic parameters
L_final = 1400000  # Increased maximum revenue to $1.4M
k_final = 0.000006
x0_final = 350000
baseline_final = 80000

# Generate clean spend data - extend range to show full curve
spend_clean = np.linspace(0, 1500000, 1000)
revenue_clean = logistic_curve(spend_clean, L_final, k_final, x0_final, baseline_final)

# Add realistic noise with heteroscedasticity (more noise at higher spends)
noise_std = 5000 + 0.01 * spend
revenue_noisy = logistic_curve(
    spend, L_final, k_final, x0_final, baseline_final
) + np.random.normal(0, noise_std)

# Plot the final smooth curve
plt.scatter(
    spend / 1000,
    revenue_noisy / 1000,
    color="steelblue",
    alpha=0.5,
    s=25,
    label="Observed Data",
)
plt.plot(
    spend_clean / 1000,
    revenue_clean / 1000,
    color="crimson",
    linewidth=3,
    label="Smooth S-Curve",
)

# Add annotations for key points
plt.axvline(
    x=x0_final / 1000,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Inflection Point: ${x0_final / 1000:.0f}K",
)
plt.axhline(
    y=baseline_final / 1000,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Baseline: ${baseline_final / 1000:.0f}K",
)
plt.axhline(
    y=L_final / 1000,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Saturation: ${L_final / 1000:.0f}K",
)

plt.title(
    "Realistic Smooth S-Curve: Marketing Spend vs Revenue",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
plt.xlabel("Advertising Spend ($000s)", fontsize=12)
plt.ylabel("Revenue ($000s)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add text box with curve characteristics
textstr = f"""Curve Characteristics:
• Baseline Revenue: ${baseline_final:,.0f}
• Saturation Point: ${L_final:,.0f}
• Inflection Point: ${x0_final:,.0f}
• Growth Rate: {k_final:.6f}
• Smooth Factor: High"""

props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
plt.text(
    0.02,
    0.98,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=props,
)

plt.tight_layout()
plt.show()

print("S-Curve Analysis:")
print(f"Baseline Revenue: ${baseline_final:,.0f}")
print(f"Maximum Revenue: ${L_final:,.0f}")
print(f"Revenue Lift: {((L_final - baseline_final) / baseline_final) * 100:.1f}%")
print(f"Inflection Point: ${x0_final:,.0f}")
print(f"Growth Rate: {k_final:.6f} (lower = smoother)")

# Calculate marginal ROI at different spend levels
spend_levels = [50000, 200000, 400000, 600000, 800000]
print("\nMarginal ROI at different spend levels:")
for spend_level in spend_levels:
    # Calculate derivative (marginal revenue)
    delta = 1000
    rev1 = logistic_curve(spend_level, L_final, k_final, x0_final, baseline_final)
    rev2 = logistic_curve(
        spend_level + delta, L_final, k_final, x0_final, baseline_final
    )
    marginal_revenue = rev2 - rev1
    marginal_roi = marginal_revenue / delta
    print(f"  ${spend_level / 1000:3.0f}K spend: {marginal_roi:.2f}x ROI")
