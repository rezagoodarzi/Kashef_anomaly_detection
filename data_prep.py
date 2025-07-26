import numpy as np
import pandas as pd
from run_anomaly import run_daily_analysis
import pandas as pd
import numpy as np

def merge_carriers(rtwp_df):
    """
    Merge carriers in RTWP data by averaging values over all carriers for each sector.
    
    Expects:
    - rtwp_df: DataFrame with columns including 'Mapped_ID', 'DATETIME', 'VALUE'
    
    Returns:
    - merged_rtwp_df: DataFrame with carriers merged (one row per sector+time)
    """
    # Extract the base sector ID without carrier (last character is carrier digit)
    rtwp_df['Base_ID'] = rtwp_df['Mapped_ID'].str[:-1]
    
    # Group by Base_ID and DATETIME and average VALUE
    merged_rtwp_df = (
        rtwp_df.groupby(['Base_ID', 'DATETIME'], as_index=False)
               .agg({'VALUE': 'mean'})
    )
    
    # Rename Base_ID back to Mapped_ID
    merged_rtwp_df.rename(columns={'Base_ID': 'Mapped_ID'}, inplace=True)
    
    return merged_rtwp_df

def prepare_inputs_for_date(rtwp_csv, pathloss_csv, anomaly_csv, target_date):
    """
    Load and prepare inputs for a specific date, using only Mapped_ID as identifiers.
    - rtwp_data: dict {mapped_id: hourly VALUE array for target_date}
    - path_loss_matrix: numpy array (M x M)
    - anomaly_sectors: list of mapped_id strings for anomalies
    - sector_ids: list of mapped_id strings that have data & neighbors
    """
    # 1) Load raw DataFrames
    rtwp_df = pd.read_csv(rtwp_csv, low_memory=False)
    pathloss_df = pd.read_csv(pathloss_csv)
    anom_df = pd.read_csv(anomaly_csv, parse_dates=['DATE_KEY'])
    
    # 2) Compute Mapped_ID
    anom_df['Mapped_ID'] = anom_df['name'].str[:2] + anom_df['name'].str[4:-1]
    
    # 3) Extract anomaly mapped IDs
    anomaly_sectors = anom_df['Mapped_ID'].unique().tolist()

    print("Anomaly DF columns:", anom_df.columns.tolist())
    print("Mapped anomalies:", anomaly_sectors)
    # 4) Build DATETIME and filter to target_date
    # filter by date
    target_date_dt = pd.to_datetime(target_date, format="%Y%m%d").date()
    rtwp_df['DATETIME'] = pd.to_datetime(rtwp_df['DATETIME'], errors='coerce')

    rtwp_date = rtwp_df[rtwp_df['DATETIME'].dt.date == target_date_dt]
    print("rtwp_date :" )
    print(rtwp_date)
    # 5) Filter pathloss to only anomaly mapped sectors, keep rank ≤ 30
    rel_pl = pathloss_df[pathloss_df['Sector'].isin(anomaly_sectors)]
    print("rel_pl = ",rel_pl)    
    # 6) Sector_ids initial from rel_pl
    candidate_ids = pd.unique(rel_pl[['Sector', 'Neighbor']].values.ravel()).tolist()
    print("Candidate IDs from pathloss:", candidate_ids)

    # 7) Keep only those candidate_ids present in rtwp_date
    available_ids = rtwp_date['Mapped_ID'].unique().tolist()
    sector_ids = [mid for mid in candidate_ids if mid in available_ids]
    print("Final sector_ids:", sector_ids)

    # 8) Re-filter rel_pl to rows where both Sector and Neighbor in sector_ids
    rel_pl = rel_pl[
        rel_pl['Sector'].isin(sector_ids) &
        rel_pl['Neighbor'].isin(sector_ids)
    ]
    print("Final sector_ids 8):", sector_ids)
    print("rel_pl version 2= ",rel_pl)    

    # 9) Build path loss matrix
    M = len(sector_ids)
    pl_mat = np.full((M, M), np.inf)
    for _, row in rel_pl.iterrows():
        i = sector_ids.index(row['Sector'])
        j = sector_ids.index(row['Neighbor'])
        pl_mat[i, j] = row['pathloss']
    pl_mat = np.minimum(pl_mat, pl_mat.T)
    np.fill_diagonal(pl_mat, 0.0)
    print("Final sector_ids 9):", sector_ids)

    # 10) Build rtwp_data dict for target_date
    rtwp_data = {}
    for mid in sector_ids:
        vals = (rtwp_date[rtwp_date['Mapped_ID'] == mid]
                .sort_values('DATETIME')['VALUE']
                .to_numpy(dtype=float))
        # Only include if full 24 hours present
       # if len(vals) == 24:
        rtwp_data[mid] = vals
    print("Final sector_ids 10):", sector_ids)

    # 11) Drop any sector_ids without 24-value series
    sector_ids = list(rtwp_data.keys())
    print("Final sector_ids 11):", sector_ids)

    return rtwp_data, pl_mat, anomaly_sectors, sector_ids


rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids = prepare_inputs_for_date(
    rtwp_csv='Esfehan_RSSI_merged_avg_carriers_2U_filled.csv',
    pathloss_csv='Data_bridge_ESFAHAN.Neighbors_3G_3G.csv',
    anomaly_csv='2.csv',
    target_date='20250702'
)
print("RTWP Data:", rtwp_data)
print("Path Loss Matrix:", path_loss_matrix)
print("Anomaly Sectors:", anomaly_sectors)
print("Sector IDs:", sector_ids)
'''
    print("RTWP Data:", rtwp_data)
    print("Path Loss Matrix:", path_loss_matrix)
    print("Anomaly Sectors:", anomaly_sectors)
    print("Sector IDs:", sector_ids)
'''
    # Now you can pass these into run_daily_analysis(...)
    # results, report = run_daily_analysis(rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids)
'''
    results, report = run_daily_analysis(
        rtwp_data=rtwp_data,
        path_loss_matrix=path_loss_matrix,
        anomaly_sectors=anomaly_sectors,
        sector_ids=sector_ids
    )
    print("Results:", results)
    print("Report:", report)
    with open("daily_analysis_report.txt", "w") as f:
        f.write(str(report))
    with open("daily_analysis_results.txt", "w") as f:
        f.write(str(results))
'''

import pickle

with open("prepared_data.pkl", "wb") as f:
    pickle.dump({
        "rtwp_data": rtwp_data,
        "path_loss_matrix": path_loss_matrix,
        "anomaly_sectors": anomaly_sectors,
        "sector_ids": sector_ids
    }, f)

