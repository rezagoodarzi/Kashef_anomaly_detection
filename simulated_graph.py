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
from clustering_Louvain import cluster_and_visualise

# Simulated minimal sector map data 
def data_preprocess(RSSI = "Esfehan_RSSI_merged.csv",daily_results = "daily_analysis_results_2025_07_01.pkl" ):
    esfahan_df = pd.read_csv(RSSI)
    esfahan_df['AZIMUTH'] = pd.to_numeric(esfahan_df['AZIMUTH'], errors='coerce')

    esfahan_df.dropna(subset=['LATITUDE', 'LONGITUDE', 'AZIMUTH'], inplace=True)

    # Use only one row per sector
    esfahan_grouped = esfahan_df.drop_duplicates(subset='Mapped_ID')[['Mapped_ID', 'LATITUDE', 'LONGITUDE', 'AZIMUTH']]
    position_dict = esfahan_grouped.set_index('Mapped_ID')[['LATITUDE', 'LONGITUDE', 'AZIMUTH']].to_dict('index')   
    with open(daily_results, "rb") as f:
        results = pickle.load(f)    
    return position_dict , results

def simulate_sector_map_data(position_dict,results,cluster_output ="cluster_importance_map_2025_07_01.html" ,graph_output ="anomaly_propagation_map_2025_07_01.html" ):

    # Step: Keep only nodes with known positions
    valid_nodes = set(position_dict.keys())

    # Optional: log what’s missing
    all_nodes_in_results = set(results['similarity_matrix'].keys())
    for sector in all_nodes_in_results:
        neighbors = results['similarity_matrix'][sector].keys()
        for n in neighbors:
            if n not in valid_nodes:
                print(f"⚠️ Warning: Skipping {n} (no position data)")

    # Filter out bad entries from similarity_matrix and causality_results
    for sector in list(results['similarity_matrix'].keys()):
        results['similarity_matrix'][sector] = {
            n: v for n, v in results['similarity_matrix'][sector].items() if n in valid_nodes
        }
        if not results['similarity_matrix'][sector]:
            del results['similarity_matrix'][sector]

    for sector in list(results['causality_results'].keys()):
        results['causality_results'][sector] = {
            n: v for n, v in results['causality_results'][sector].items() if n in valid_nodes
        }
        if not results['causality_results'][sector]:
            print(f"⚠️ Warning: casulaity {n} (no position data)")
            del results['causality_results'][sector]

    # position_dict built exactly as in your Folium script
    ranked_clusters, partition = cluster_and_visualise(
            results=results,
            position_dict=position_dict,
            output_html=cluster_output
    )
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
        arrow_scale = 0.003  # adjust arrow length as you wish

        # Draw azimuth direction
        theta = np.deg2rad(data['AZIMUTH'])          # 0° = North
        dlat  =  arrow_scale * np.cos(theta)         # Δlat north–south
        # correct longitude step by cos(lat) to keep a consistent ground distance
        dlon  = (arrow_scale * np.sin(theta) /
                np.cos(np.deg2rad(data['LATITUDE'])))

        end_lat = data['LATITUDE'] + dlat
        end_lon = data['LONGITUDE'] + dlon

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

    m.save(graph_output)
