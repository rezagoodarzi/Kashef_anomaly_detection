import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

# 1. Load RTT/Location Data
esfahan_df = pd.read_csv("Esfehan_RSSI_merged.csv")
'''

esfahan_grouped = esfahan_df.groupby('Mapped_ID').agg({
    'LATITUDE': 'mean',
    'LONGITUDE': 'mean',
    'AZIMUTH': 'mean'
}).reset_index()

'''
esfahan_grouped = esfahan_df.drop_duplicates(subset='Mapped_ID')[['Mapped_ID', 'LATITUDE', 'LONGITUDE', 'AZIMUTH']]

sector_loc = esfahan_grouped.set_index('Mapped_ID')[['LATITUDE', 'LONGITUDE', 'AZIMUTH']].to_dict(orient='index')

# 2. Load anomaly detection results
with open("daily_analysis_results.pkl", "rb") as f:
    results = pickle.load(f)

anomaly_sectors = list(results['propagation_analysis'].keys())
affected_nodes = set(results['affected_sectors'])

# 3. Build graph
G = nx.DiGraph()
node_colors = []
edge_colors = []

# 3.1 Add nodes
for node in affected_nodes.union(anomaly_sectors):
    pos = sector_loc.get(node)
    if pos:
        G.add_node(node, pos=(pos['LONGITUDE'], pos['LATITUDE']), azimuth=pos['AZIMUTH'])

# 3.2 Add edges
for anomaly_sector in anomaly_sectors:
    propagation = results['propagation_analysis'][anomaly_sector]
    primary = set(propagation['primary_affected'])
    secondary = set(propagation['secondary_affected'])

    for neighbor in primary:
        if neighbor in G.nodes and anomaly_sector in G.nodes:
            G.add_edge(anomaly_sector, neighbor)
            edge_colors.append('red')

    for neighbor in secondary:
        if neighbor in G.nodes and anomaly_sector in G.nodes:
            G.add_edge(anomaly_sector, neighbor)
            edge_colors.append('orange')

# 3.3 Assign colors to nodes
for node in G.nodes:
    if node in anomaly_sectors:
        node_colors.append('red')  # Anomalous
    elif node in affected_nodes:
        node_colors.append('skyblue')  # Affected
    else:
        node_colors.append('gray')  # Not used

# 4. Draw the graph
plt.figure(figsize=(14, 10))
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.95)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, width=1.8)
nx.draw_networkx_labels(G, pos, font_size=7)

#plt.title("Sector Co-behavior & Propagation Connectivity Graph")
#plt.axis("off")
#plt.tight_layout()
#plt.savefig("sector_connectivity_graph.png", dpi=300)
#plt.show()
