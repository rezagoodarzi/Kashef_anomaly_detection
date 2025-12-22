import pickle
from anomaly_detection import run_daily_analysis
from simulated_graph import cluster_and_visualise
import simulated_graph as sg
def run_connectivity_graph(RSSI = "Esfehan_RSSI_merged.csv",daily_results = "daily_analysis_results_2025_07_01.pkl",
                           cluster_output ="cluster_importance_map_2025_07_01.html",
                           graph_output = "anomaly_propagation_map_2025_07_01.html"):
    sg.simulate_sector_map_data(sg.data_preprocess(RSSI,daily_results),cluster_output,graph_output)

def run_pickle(prepared_data = "prepared_data_2025_07_01.pkl",daily_analysis_output =  "daily_analysis_report_2025_07_01"):
    with open(prepared_data, "rb") as f:
        data = pickle.load(f)

    rtwp_data = data["rtwp_data"]
    path_loss_matrix = data["path_loss_matrix"]
    anomaly_sectors = data["anomaly_sectors"]
    sector_ids = data["sector_ids"]

    # Run analysis
    results, report = run_daily_analysis(
        rtwp_data=rtwp_data,
        path_loss_matrix=path_loss_matrix,
        anomaly_sectors=anomaly_sectors,
        sector_ids=sector_ids,
        path_loss_threshold=130,  # dB
        similarity_threshold=0.64,
        causality_threshold=0.05,
        min_cluster_size=2
    )

    print("📊 Analysis report:\n")
    print(report)
    # Optionally save report & results
    with open(f'{daily_analysis_output}.txt', "w") as f:
        f.write(report)
    with open(f'{daily_analysis_output}.pkl', "wb") as f:
        pickle.dump(results, f)


run_connectivity_graph()