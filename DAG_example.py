import matplotlib.pyplot as plt
import networkx as nx

# Create the DAG
G = nx.DiGraph()

# Add nodes
G.add_node("Facebook Ads Budget")
G.add_node("Demographics")
G.add_node("iCAC")
G.add_node("iCLV")

# Add edges (causal relationships)
G.add_edge("Facebook Ads Budget", "iCAC")  # Budget affects acquisition costs
G.add_edge("Demographics", "iCAC")  # Confounder affects costs
G.add_edge("Demographics", "iCLV")  # Confounder affects lifetime value
G.add_edge("iCAC", "iCLV")  # Costs affect lifetime value

# Draw the graph
pos = nx.spring_layout(G)  # Layout for visualization
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=3000,
    font_size=8,
    font_weight="bold",
    arrows=True,
    arrowstyle="->",
    arrowsize=20,
)
plt.title("Causal DAG for iCLV vs iCAC in Digital Campaigns")
plt.show()
