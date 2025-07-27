import pickle
from anomaly_detection import run_daily_analysis

# Load prepared data
with open("prepared_data.pkl", "rb") as f:
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
    sector_ids=sector_ids
)

print("📊 Analysis report:\n")
print(report)
# Optionally save report & results
with open("daily_analysis_report_0.9_2_0.7.txt", "w") as f:
    f.write(report)
with open("daily_analysis_results0.9_2_0.7.pkl", "wb") as f:
    pickle.dump(results, f)
