# ==========================================================
# file: cluster_rank_visualise.py
# ==========================================================
import networkx as nx
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import folium
import community  
import random
from matplotlib.colors import to_hex  # ADD THIS at the top of your file

# ---------- 1. Build weighted graph -----------------------------------------
def build_similarity_graph(results, sim_threshold=0.65, causal_bonus=0.2):
    G = nx.Graph()
    for anom, nbrs in results['similarity_matrix'].items():
        for nbr, vals in nbrs.items():
            w = vals['weighted_similarity']
            if w < sim_threshold:
                continue
            # add causal bonus if any direction is_causal
            if results['causality_results'][anom][nbr]['is_causal']:
                w += causal_bonus
            if G.has_edge(anom, nbr):
                G[anom][nbr]['weight'] = max(G[anom][nbr]['weight'], w)
            else:
                G.add_edge(anom, nbr, weight=w)
    return G

# ---------- 2. Louvain clustering  ------------------------------------------
def louvain_partition(G):
    # weights are considered automatically
    partition = community.best_partition(G, weight='weight')
    # partition: dict {node: community_id}
    return partition

# ---------- 3. Rank clusters  ----------------------------------------------
def rank_clusters(G, partition, anomaly_nodes, alpha=1.5, beta=1.0, gamma=0.5):
    # build cluster→nodes
    clusters = {}
    for node, cid in partition.items():
        clusters.setdefault(cid, []).append(node)

    ranked = []
    for cid, nodes in clusters.items():
        sub = G.subgraph(nodes)
        intra_weights = [d['weight'] for _, _, d in sub.edges(data=True)]
        mean_w = np.mean(intra_weights) if intra_weights else 0
        n_anom = sum(1 for n in nodes if n in anomaly_nodes)
        score = alpha * n_anom + beta * mean_w + gamma * len(nodes)
        ranked.append({
            'cluster_id': cid,
            'nodes': nodes,
            'size': len(nodes),
            'n_anomaly': n_anom,
            'mean_sim': round(mean_w, 3),
            'score': round(score, 2)
        })
    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked

# ---------- 4. Folium map ---------------------------------------------------
def folium_cluster_map(position_dict, ranked_clusters, anomaly_nodes,
                       outfile="cluster_map.html"):
    # continuous colour palette
    palette = cm.get_cmap('tab20', len(ranked_clusters))


    cluster_color = {
        cl['cluster_id']: to_hex(palette(i)[:3])
        for i, cl in enumerate(ranked_clusters)
    }

    # map centre
    lat = np.mean([v['LATITUDE'] for v in position_dict.values()])
    lon = np.mean([v['LONGITUDE'] for v in position_dict.values()])
    fmap = folium.Map([lat, lon], zoom_start=12)

    # plot nodes
    for cl in ranked_clusters:
        cid = cl['cluster_id']
        color = cluster_color[cid]
        for node in cl['nodes']:
            if node not in position_dict:
                continue  # skip missing location info
            data = position_dict[node]
            popup = (f"<b>{node}</b><br>"
                    f"Cluster {cid}<br>"
                    f"Mean_sim: {cl['mean_sim']}<br>"
                    f"Score: {cl['score']}")
            folium.CircleMarker(
                [data['LATITUDE'], data['LONGITUDE']],
                radius=6,
                color='red' if node in anomaly_nodes else color,
                fill=True, fill_color=color,
                popup=popup
            ).add_to(fmap)
    # save
    fmap.save(outfile)
    print(f"Map saved to {outfile}")

# ---------- 5. Glue function -------------------------------------------------
def cluster_and_visualise(results, position_dict, output_html="cluster_map.html"):
    anomaly_nodes = set(results['propagation_analysis'].keys())

    G = build_similarity_graph(results)
    partition = louvain_partition(G)
    ranked = rank_clusters(G, partition, anomaly_nodes)

    # print ranking table
    print("=== Cluster Ranking ===")
    for i, cl in enumerate(ranked, 1):
        print(f"{i}. cid={cl['cluster_id']}  "
              f"score={cl['score']}  "
              f"size={cl['size']}  "
              f"n_anom={cl['n_anomaly']}  "
              f"mean_sim={cl['mean_sim']}")

    m =folium_cluster_map(position_dict, ranked, anomaly_nodes,
                       outfile=output_html)
   # m.save("cluster_map.html")

    return ranked, partition
