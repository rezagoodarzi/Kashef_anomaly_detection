import pickle
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import folium
from folium import plugins
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import os

def load_prepared_data(prepared_data_path):
    """
    Load rtwp_data, path_loss_matrix, anomaly_sectors, and sector_ids from a pickle.
    """
    with open(prepared_data_path, 'rb') as f:
        data = pickle.load(f)
    return (
        data['rtwp_data'],
        data['path_loss_matrix'],
        data['anomaly_sectors'],
        data['sector_ids'],
    )


def load_position_dict(rssi_csv):
    """
    Load Mapped_ID, LATITUDE, LONGITUDE from RSSI CSV into a dict.
    """
    df = pd.read_csv(rssi_csv)
    df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    mapping = df.drop_duplicates(subset='Mapped_ID')[['Mapped_ID', 'LATITUDE', 'LONGITUDE']]
    return mapping.set_index('Mapped_ID')[['LATITUDE', 'LONGITUDE']].to_dict('index')


def build_affinity_matrix(results, sector_ids, metric='weighted_similarity'):
    """
    Build a symmetric NxN affinity matrix for the given metric.
    """
    N = len(sector_ids)
    idx = {sec: i for i, sec in enumerate(sector_ids)}
    W = np.zeros((N, N))
    for anomaly, sims in results['similarity_matrix'].items():
        if anomaly not in idx: continue
        i = idx[anomaly]
        for nbr, simvals in sims.items():
            if nbr not in idx: continue
            j = idx[nbr]
            W[i, j] = simvals.get(metric, 0.0)
    W = np.maximum(W, W.T)
    maxv = W.max()
    if maxv > 0: W /= maxv
    return W


def build_feature_matrix(W, sector_ids, anomaly_sectors):
    """
    Treat each node's similarity to each anomaly as a feature vector.
    Returns F of shape (N, M) and list of anomalies used.
    """
    N = len(sector_ids)
    idx = {sec: i for i, sec in enumerate(sector_ids)}
    # Only keep anomalies present in sector_ids
    anomalies = [a for a in anomaly_sectors if a in idx]
    M = len(anomalies)
    F = np.zeros((N, M))
    for j, a in enumerate(anomalies):
        ai = idx[a]
        F[:, j] = W[:, ai]
    return F, anomalies


def run_density_clustering(F, method='hdbscan', **kwargs):
    """
    Cluster rows of F using HDBSCAN or DBSCAN.
    Returns labels array of length N.
    """
    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=kwargs.get('min_cluster_size', 5),
            min_samples=kwargs.get('min_samples', None),
            metric='euclidean'
        )
        labels = clusterer.fit_predict(F)
    else:
        labels = DBSCAN(
            eps=kwargs.get('eps', 0.5),
            min_samples=kwargs.get('min_samples', 5),
            metric='euclidean'
        ).fit_predict(F)
    return labels


def plot_feature_heatmap(F, sector_ids, anomalies, labels, output_png):
    """
    Plot a heatmap of F (nodes x anomalies), ordered by cluster labels.
    Saves heatmap to output_png.
    """
    # Handle noise label
    unique_labels = set(labels)
    max_label = max(l for l in unique_labels if l >= 0) if any(l>=0 for l in labels) else 0
    # Map noise (-1) to bottom
    label_for_sort = np.array([lab if lab>=0 else max_label+1 for lab in labels])
    order = np.argsort(label_for_sort)

    df = pd.DataFrame(F, index=sector_ids, columns=anomalies)
    df_ord = df.iloc[order]

    plt.figure(figsize=(10, 6))
    plt.imshow(df_ord.values, aspect='auto')
    plt.colorbar(label='Similarity')
    plt.xlabel('Anomaly sector')
    plt.ylabel('Sectors (ordered by cluster)')
    plt.title('Feature heatmap (similarity to anomalies)')
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"Heatmap saved to {output_png}")


def generate_cluster_map(position_dict, sector_ids, labels, output_html):
    """
    Display clusters on a real map with Folium, colored by cluster label.
    """
    # Determine center
    lats = []
    lons = []
    for sec in sector_ids:
        if sec in position_dict:
            loc = position_dict[sec]
            lats.append(loc['LATITUDE']); lons.append(loc['LONGITUDE'])
    center = [np.mean(lats), np.mean(lons)] if lats else [0, 0]
    m = folium.Map(location=center, zoom_start=12)

    # Color palette
    unique_labels = sorted(set(labels))
    n_colors = len(unique_labels)
    cmap = cm.get_cmap('tab20', n_colors)
    label_color = {lab: mcolors.to_hex(cmap(i)) for i, lab in enumerate(unique_labels)}
    label_color[-1] = '#000000'  # noise as black

    for i, sec in enumerate(sector_ids):
        if sec not in position_dict: continue
        loc = position_dict[sec]
        lab = labels[i]
        color = label_color.get(lab, '#333333')
        folium.CircleMarker(
            location=[loc['LATITUDE'], loc['LONGITUDE']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{sec}: cluster {lab}"
        ).add_to(m)

    m.save(output_html)
    print(f"Cluster map saved to {output_html}")


if __name__ == '__main__':
    # === Default parameters (overrideable) ===
    rssi_csv = 'Esfehan_RSSI_merged.csv'
    metric = 'weighted_similarity'
    method = 'hdbscan'            # 'hdbscan' or 'dbscan'
    prepared_data_path = r'outputs\20250702\prepared_data_20250702.pkl'
    daily_result = r'outputs\20250702\daily_analysis_results_20250702.pkl'
    date = '20250702' 
    output_dir = os.path.join('outputs', date)

    # DBSCAN params
    eps = 0.5
    min_samples = 5
    # HDBSCAN params
    min_cluster_size = 10

    heatmap_png = 'density_features_heatmap.png'
    cluster_map_html = f'density_clusters_map_eps{eps}_minsample{min_samples}_size{min_cluster_size}.html'

    # Load data
    rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids = load_prepared_data(prepared_data_path)
    position_dict = load_position_dict(rssi_csv)
    from anomaly_detection import run_daily_analysis
    results, _ = run_daily_analysis(rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids)

    # Build affinity + features
    W = build_affinity_matrix(results, sector_ids, metric)
    F, anomalies = build_feature_matrix(W, sector_ids, anomaly_sectors)

    # Cluster
    labels = run_density_clustering(
        F,
        method=method,
        eps=eps,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size
    )

    output_heatmap = os.path.join(output_dir,cluster_map_html)

    # Heatmap of features
    plot_feature_heatmap(F, sector_ids, anomalies, labels, heatmap_png)

    # Map clusters
    generate_cluster_map(position_dict, sector_ids, labels, output_heatmap)
