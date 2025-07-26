import folium
from folium import plugins
import random
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

# Simulated minimal sector map data (replace with actual Esfehan_RSSI_merged.csv)
esfahan_df = pd.read_csv("Esfehan_RSSI_merged.csv")
esfahan_df['AZIMUTH'] = pd.to_numeric(esfahan_df['AZIMUTH'], errors='coerce')

esfahan_df.dropna(subset=['LATITUDE', 'LONGITUDE', 'AZIMUTH'], inplace=True)

# Use only one row per sector
esfahan_grouped = esfahan_df.drop_duplicates(subset='Mapped_ID')[['Mapped_ID', 'LATITUDE', 'LONGITUDE', 'AZIMUTH']]
position_dict = esfahan_grouped.set_index('Mapped_ID')[['LATITUDE', 'LONGITUDE', 'AZIMUTH']].to_dict('index')

# Simulated results for testing
with open("daily_analysis_results0.9_2.pkl", "rb") as f:
    results = pickle.load(f)

anomaly_sectors = list(results['propagation_analysis'].keys())
primary_edges = []
secondary_edges = []
affected_nodes = set()
connected_nodes = set()

# Build edge lists and collect all affected sectors
for anomaly in anomaly_sectors:
    primary = results['propagation_analysis'][anomaly]['primary_affected']
    secondary = results['propagation_analysis'][anomaly]['secondary_affected']
    for tgt in primary:
        primary_edges.append((anomaly, tgt))
        affected_nodes.add(tgt)
        connected_nodes.add(anomaly)
        connected_nodes.add(tgt)
    for tgt in secondary:
        secondary_edges.append((anomaly, tgt))
        affected_nodes.add(tgt)
        connected_nodes.add(anomaly)
        connected_nodes.add(tgt)

connected_nodes.update(anomaly_sectors)
position_dict = {k: v for k, v in position_dict.items() if k in connected_nodes}

# Build map centered at mean location
mean_lat = np.mean([v['LATITUDE'] for v in position_dict.values()])
mean_lon = np.mean([v['LONGITUDE'] for v in position_dict.values()])
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

# Draw nodes
for node, data in position_dict.items():
    color = 'red' if node in anomaly_sectors else 'blue' if node in affected_nodes else 'gray'
    folium.CircleMarker(
        location=[data['LATITUDE'], data['LONGITUDE']],
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=f"{node} - Azimuth: {data['AZIMUTH']}"
    ).add_to(m)

    # Draw azimuth direction
    azimuth_rad = np.deg2rad(data['AZIMUTH'])
    dx = 0.003 * np.cos(azimuth_rad)
    dy = 0.003 * np.sin(azimuth_rad)
    end_lat = data['LATITUDE'] + dy
    end_lon = data['LONGITUDE'] + dx

    folium.PolyLine(
        locations=[[data['LATITUDE'], data['LONGITUDE']], [end_lat, end_lon]],
        color=color,
        weight=2,
        opacity=0.5
    ).add_to(m)

# Draw edges
def draw_edges(edge_list, color):
    for src, dst in edge_list:
        if src in position_dict and dst in position_dict:
            latlngs = [
                [position_dict[src]['LATITUDE'], position_dict[src]['LONGITUDE']],
                [position_dict[dst]['LATITUDE'], position_dict[dst]['LONGITUDE']]
            ]
            folium.PolyLine(
                locations=latlngs,
                color=color,
                weight=3,
                opacity=0.6
            ).add_to(m)

draw_edges(primary_edges, 'red')
draw_edges(secondary_edges, 'orange')

m.save("anomaly_propagation_map_130_2.html")
