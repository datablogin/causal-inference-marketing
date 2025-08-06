import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Step 1: Simulate observational data for PSM example
np.random.seed(42)
n_samples = 100
user_behavior = np.random.normal(50, 10, n_samples)  # Confounder: User behavior score
high_icac_ads = (user_behavior > 55).astype(int) + np.random.binomial(
    1, 0.1, n_samples
)  # Treatment: Exposed to high-iCAC ads (influenced by behavior)
iclv = (
    100 + 30 * high_icac_ads + 2 * user_behavior + np.random.normal(0, 10, n_samples)
)  # Outcome: iCLV, causally affected by treatment and confounder

data = pd.DataFrame(
    {"High_iCAC_Ads": high_icac_ads, "User_Behavior": user_behavior, "iCLV": iclv}
)

# Step 2: Build the Causal Graph with networkx
G = nx.DiGraph()
G.add_node("High_iCAC_Ads")  # Treatment
G.add_node("User_Behavior")  # Confounder
G.add_node("iCLV")  # Outcome
G.add_edge("User_Behavior", "High_iCAC_Ads")  # Confounder → Treatment
G.add_edge("User_Behavior", "iCLV")  # Confounder → Outcome
G.add_edge("High_iCAC_Ads", "iCLV")  # Treatment → Outcome

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=3000,
    font_size=10,
    font_weight="bold",
    arrows=True,
    arrowstyle="->",
    arrowsize=20,
)
plt.title("Causal DAG for High iCAC Ads Impact on iCLV")
plt.show()  # In a full environment, this renders the graph; save with plt.savefig('dag_example3.png')

# Step 3: Propensity Score Matching (PSM) for causal estimation
# Fit propensity model (logistic regression on confounder to predict treatment)
X = data[["User_Behavior"]]
y = data["High_iCAC_Ads"]
propensity_model = LogisticRegression().fit(X, y)
data["propensity_score"] = propensity_model.predict_proba(X)[:, 1]

# Nearest neighbor matching on propensity scores
treated = data[data["High_iCAC_Ads"] == 1]
control = data[data["High_iCAC_Ads"] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity_score"]])
distances, indices = nn.kneighbors(treated[["propensity_score"]])

# Create matched dataset
matched_control = control.iloc[indices.flatten()]
matched_data = pd.concat(
    [treated.reset_index(drop=True), matched_control.reset_index(drop=True)]
)

# Estimate Average Treatment Effect (ATE) on iCLV
ate = (
    matched_data[matched_data["High_iCAC_Ads"] == 1]["iCLV"].mean()
    - matched_data[matched_data["High_iCAC_Ads"] == 0]["iCLV"].mean()
)
print(f"Estimated ATE (Causal iCLV Lift from High iCAC Ads): {ate}")

# Optional: OLS on matched data for refined estimation (controlling for confounder)
X_matched = sm.add_constant(matched_data[["High_iCAC_Ads", "User_Behavior"]])
y_matched = matched_data["iCLV"]
model_matched = sm.OLS(y_matched, X_matched).fit()
print(model_matched.summary())
