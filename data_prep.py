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
    Optimized version with better memory usage and vectorized operations.
    """
    print(f"Loading data for {target_date}...")
    
    # 1) Load raw DataFrames with optimized data types
    print("Loading RTWP data...")
    rtwp_df = pd.read_csv(rtwp_csv, low_memory=False, dtype={'VALUE': 'float32'})
    
    print("Loading path loss data...")
    pathloss_df = pd.read_csv(pathloss_csv, dtype={'pathloss': 'float32'})
    
    print("Loading anomaly data...")
    anom_df = pd.read_csv(anomaly_csv, parse_dates=['DATE_KEY'])
    
    # 2) Compute Mapped_ID vectorized
    anom_df['Mapped_ID'] = anom_df['name'].str[:2] + anom_df['name'].str[4:-1]
    
    # 3) Extract anomaly mapped IDs
    anomaly_sectors = anom_df['Mapped_ID'].unique().tolist()

    print("Anomaly DF columns:", anom_df.columns.tolist())
    print("Mapped anomalies:", anomaly_sectors)
    
    # 4) Build DATETIME and filter to target_date - optimized
    target_date_dt = pd.to_datetime(target_date, format="%Y%m%d").date()
    rtwp_df['DATETIME'] = pd.to_datetime(rtwp_df['DATETIME'], errors='coerce')
    
    # Use query for better performance on large datasets
    rtwp_date = rtwp_df.query('DATETIME.dt.date == @target_date_dt')
    print(f"Filtered RTWP data: {len(rtwp_date)} rows")
    
    # 5) Filter pathloss to only anomaly mapped sectors
    rel_pl = pathloss_df[pathloss_df['Sector'].isin(anomaly_sectors)].copy()
    print(f"Relevant path loss entries: {len(rel_pl)}")    
    
    # 6) Sector_ids initial from rel_pl
    candidate_ids = pd.unique(rel_pl[['Sector', 'Neighbor']].values.ravel()).tolist()
    print("Candidate IDs from pathloss:", len(candidate_ids))

    # 7) Keep only those candidate_ids present in rtwp_date
    available_ids = set(rtwp_date['Mapped_ID'].unique())
    sector_ids = [mid for mid in candidate_ids if mid in available_ids]
    print("Final sector_ids:", len(sector_ids))

    # 8) Re-filter rel_pl to rows where both Sector and Neighbor in sector_ids
    sector_ids_set = set(sector_ids)
    rel_pl = rel_pl[
        rel_pl['Sector'].isin(sector_ids_set) &
        rel_pl['Neighbor'].isin(sector_ids_set)
    ].copy()
    print(f"Final path loss matrix entries: {len(rel_pl)}")

    # 9) Build path loss matrix - vectorized approach
    M = len(sector_ids)
    sector_to_idx = {sector: idx for idx, sector in enumerate(sector_ids)}
    
    # Vectorized matrix building
    pl_mat = np.full((M, M), np.inf, dtype=np.float32)
    
    # Use vectorized indexing instead of iterrows()
    if len(rel_pl) > 0:
        sector_indices = [sector_to_idx[s] for s in rel_pl['Sector']]
        neighbor_indices = [sector_to_idx[n] for n in rel_pl['Neighbor']]
        pl_mat[sector_indices, neighbor_indices] = rel_pl['pathloss'].values
    
    # Make symmetric
    pl_mat = np.minimum(pl_mat, pl_mat.T)
    np.fill_diagonal(pl_mat, 0.0)

    # 10) Build rtwp_data dict for target_date - optimized
    print("Building RTWP time series data...")
    rtwp_data = {}
    
    # Group by Mapped_ID once and convert to dict for faster access
    rtwp_grouped = (rtwp_date.groupby('Mapped_ID')
                    .apply(lambda x: x.sort_values('DATETIME')['VALUE'].values.astype(np.float32))
                    .to_dict())
    
    for mid in sector_ids:
        if mid in rtwp_grouped:
            rtwp_data[mid] = rtwp_grouped[mid]

    # 11) Update sector_ids to only include those with data
    sector_ids = list(rtwp_data.keys())
    print(f"Final sector count with data: {len(sector_ids)}")

    return rtwp_data, pl_mat, anomaly_sectors, sector_ids

# Remove the execution code from the module
if __name__ == "__main__":
    # This should only run when the script is executed directly
    rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids = prepare_inputs_for_date(
        rtwp_csv='Esfehan_RSSI_merged_avg_carriers_2U_filled.csv',
        pathloss_csv='Data_bridge_ESFAHAN.Neighbors_3G_3G.csv',
        anomaly_csv='1.csv',
        target_date='20250701'
    )
    print("RTWP Data keys:", len(rtwp_data))
    print("Path Loss Matrix shape:", path_loss_matrix.shape)
    print("Anomaly Sectors count:", len(anomaly_sectors))
    print("Sector IDs count:", len(sector_ids))
    
    # Save prepared data
    import pickle
    with open("prepared_data_2025_07_01.pkl", "wb") as f:
        pickle.dump({
            "rtwp_data": rtwp_data,
            "path_loss_matrix": path_loss_matrix,
            "anomaly_sectors": anomaly_sectors,
            "sector_ids": sector_ids
        }, f)
    print("Data saved to prepared_data_2025_07_01.pkl")

