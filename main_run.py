import os
import pickle
from data_prep import prepare_inputs_for_date
from anomaly_detection import run_daily_analysis
from clustering_Louvain import cluster_and_visualise
from connectivity_graph import plt, nx, pd, np
from simulated_graph import simulate_sector_map_data, data_preprocess

# ==== Configuration ======
config = {
    "date": "20250701",
    "rtwp_csv": "Esfehan_RSSI_merged_avg_carriers_2U_filled.csv",
    "RSSI": "Esfehan_RSSI_merged.csv",
    "pathloss_csv": "Data_bridge_ESFAHAN.Neighbors_3G_3G.csv",
    "anomaly_csv": "1.csv",
    "base_output_dir": "outputs"
}

# ==== Helper Functions ======
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def save_txt(text, filename):
    with open(filename, "w") as f:
        f.write(text)

# ==== Main Pipeline ======
def main():
    date = config["date"]
    output_dir = os.path.join(config["base_output_dir"], date)
    ensure_dir(output_dir)

    # Step 1: Prepare data
    print("📦 Preparing input data...")
    rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids = prepare_inputs_for_date(
        rtwp_csv=config["rtwp_csv"],
        pathloss_csv=config["pathloss_csv"],
        anomaly_csv=config["anomaly_csv"],
        target_date=date
    )

    prep_file = os.path.join(output_dir, f"prepared_data_{date}.pkl")
    save_pickle({
        "rtwp_data": rtwp_data,
        "path_loss_matrix": path_loss_matrix,
        "anomaly_sectors": anomaly_sectors,
        "sector_ids": sector_ids
    }, prep_file)

    # Step 2: Run anomaly + propagation analysis
    print("🔍 Running anomaly detection & co-behavior analysis...")
    results, report = run_daily_analysis(
        rtwp_data=rtwp_data,
        path_loss_matrix=path_loss_matrix,
        anomaly_sectors=anomaly_sectors,
        sector_ids=sector_ids,
        path_loss_threshold=130,
        similarity_threshold=0.64,
        causality_threshold=0.05,
        min_cluster_size=2
    )

    report_txt = os.path.join(output_dir, f"daily_analysis_report_{date}.txt")
    results_pkl = os.path.join(output_dir, f"daily_analysis_results_{date}.pkl")
    save_txt(report, report_txt)
    save_pickle(results, results_pkl)

    # Step 3: Clustering & Map
    print("🗺️ Generating cluster map...")
    esfahan_df = pd.read_csv("Esfehan_RSSI_merged.csv")
    esfahan_grouped = esfahan_df.drop_duplicates(subset='Mapped_ID')[['Mapped_ID', 'LATITUDE', 'LONGITUDE', 'AZIMUTH']]
    position_dict = esfahan_grouped.set_index('Mapped_ID')[['LATITUDE', 'LONGITUDE', 'AZIMUTH']].to_dict(orient='index')

    cluster_html = os.path.join(output_dir, f"cluster_importance_map_{date}.html")
    cluster_and_visualise(results, position_dict, output_html=cluster_html)

    position_dict, _ = data_preprocess(config["RSSI"], results_pkl)

    cluster_html = os.path.join(output_dir, f"cluster_importance_map_{date}.html")
    graph_html = os.path.join(output_dir, f"anomaly_propagation_map_{date}.html")

    simulate_sector_map_data(position_dict, results, cluster_output=cluster_html, graph_output=graph_html)


# ==== Entry point ======
if __name__ == "__main__":
    main()
