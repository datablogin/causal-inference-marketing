import matplotlib.pyplot as plt
import numpy as np

# Step 0: Simulate marketing channels as multi-armed bandits
# Each 'arm' represents a digital channel (e.g., Facebook Ads, Google Search) with true unknown reward means (e.g., ROI lifts)
num_arms = 5  # Number of channels
true_means = np.array(
    [0.1, 0.2, 0.3, 0.25, 0.15]
)  # True reward probabilities (e.g., conversion rates or lifts)
num_rounds = 1000  # Total budget/iterations for testing
initial_budget = num_rounds  # Fixed budget for best-arm identification

# Priors for Thompson Sampling (Beta distribution: alpha=1, beta=1 for uniform prior)
alphas = np.ones(num_arms)
betas = np.ones(num_arms)

# Track cumulative regret (lost opportunity from not choosing the best arm)
best_arm = np.argmax(true_means)
regret = np.zeros(num_rounds)
cumulative_regret = 0

# Track allocations and pulls
pulls = np.zeros(num_arms)
rewards = np.zeros(num_arms)

# TS-EBA Implementation
for t in range(num_rounds):
    # Step 1: Exploration via Thompson Sampling (TS)
    # Sample from Bayesian priors (Beta distribution) for each arm
    samples = np.random.beta(alphas, betas)
    chosen_arm = np.argmax(
        samples
    )  # Select arm with highest sample (random exploration based on priors)

    # Pull the arm (simulate reward from Bernoulli distribution based on true mean)
    reward = np.random.binomial(1, true_means[chosen_arm])

    # Update pulls and rewards
    pulls[chosen_arm] += 1
    rewards[chosen_arm] += reward

    # Update Beta posteriors (Bayesian update)
    alphas[chosen_arm] += reward
    betas[chosen_arm] += 1 - reward

    # Calculate regret for this round
    regret[t] = true_means[best_arm] - true_means[chosen_arm]
    cumulative_regret += regret[t]

    # Step 2: EBA Refinement - Empirically adjust allocations
    # After initial pulls (e.g., after 20% of budget), focus on high-variance arms
    if t > 0.2 * num_rounds:
        # Compute empirical variances (high-variance = uncertain/promising)
        variances = (
            (rewards / pulls) * (1 - rewards / pulls) / pulls
        )  # Sample variance for Bernoulli
        variances[np.isnan(variances)] = 1.0  # Handle unp pulled arms

        # Adjust sampling: Weight towards high-variance arms (e.g., multiply samples by variance factor)
        samples *= (
            1 + variances
        )  # Refinement: Boost uncertain arms for better exploration

# Outcome: Plot cumulative regret to show lower regret over time
plt.plot(np.cumsum(regret))
plt.title("Cumulative Regret in TS-EBA for Marketing Channels")
plt.xlabel("Rounds (Budget Spent)")
plt.ylabel("Cumulative Regret")
plt.show()

# Print final results
print(f"Best arm (channel): {best_arm}")
print(f"Pulls per arm: {pulls}")
print(f"Estimated means: {rewards / pulls}")
print(f"Total regret: {cumulative_regret}")
