import numpy as np
import pandas as pd
from run_anomaly import run_daily_analysis
def prepare_inputs(rtwp_csv, pathloss_csv, anomaly_csv):
    """
    Load and prepare:
      - rtwp_data: dict {NE_NAME: hourly VALUE array}
      - path_loss_matrix: numpy array of shape (M,M)
      - anomaly_sectors: list of 3G NE_NAME strings
      - sector_ids: list of mapped IDs (Sector + Neighbor keys)
    """
    # 1) Load raw DataFrames (no parse_dates!)
    rtwp_df = pd.read_csv(rtwp_csv, low_memory=False)
    pathloss_df = pd.read_csv(pathloss_csv)
    anom_df = pd.read_csv(anomaly_csv, parse_dates=['DATE_KEY'])
    
    # 2) Extract anomaly sectors (NE_NAME)
    anomaly_sectors = anom_df['name'].unique().tolist()
    
    # 3) Build Mapped_ID in RTWP to match pathloss table
    #    e.g. "ES2U1304B2" -> "ES1304B"
    rtwp_df['Mapped_ID'] = (
        rtwp_df['NE_NAME'].str.slice(0,2) +
        rtwp_df['NE_NAME'].str.slice(4, -1)
    )
    
    # 4) Build a proper datetime column from DATE_KEY (YYYYMMDD) + HOUR_KEY
    #    Ensure both are strings, pad hour to two digits
    rtwp_df['DATE_KEY'] = rtwp_df['DATE_KEY'].astype(str)
    rtwp_df['HOUR_KEY'] = rtwp_df['HOUR_KEY'].astype(int).astype(str).str.zfill(2)
    dt_series = rtwp_df['DATE_KEY'] + rtwp_df['HOUR_KEY']
    rtwp_df['DATETIME'] = pd.to_datetime(dt_series, format='%Y%m%d%H')
    
    # 5) Filter pathloss rows: only those where Sector is a mapped anomaly
    mapped_anom = [s[:2] + s[4:-1] for s in anomaly_sectors]
    # keep only rank ≤ 30 (should already be, but just in case)
    pathloss_df = pathloss_df[pathloss_df['rank'] <= 30]
    rel_pl = pathloss_df[pathloss_df['Sector'].isin(mapped_anom)]
    
    # 6) Build the full list of IDs: every mapped anomaly Sector + all its Neighbors
    sector_ids = pd.unique(
        rel_pl[['Sector','Neighbor']].values.ravel()
    ).tolist()
    
    # 7) Build path‑loss matrix of shape (M,M)
    M = len(sector_ids)
    pl_mat = np.full((M, M), np.inf)
    # fill entries
    for _, row in rel_pl.iterrows():
        i = sector_ids.index(row['Sector'])
        j = sector_ids.index(row['Neighbor'])
        pl_mat[i, j] = row['pathloss']
    # mirror to make symmetric
    pl_mat = np.minimum(pl_mat, pl_mat.T)
    np.fill_diagonal(pl_mat, 0.0)
    
    # 8) Build rtwp_data dict: for each NE_NAME in this set, grab its full hourly series
    sel = rtwp_df[rtwp_df['Mapped_ID'].isin(sector_ids)]
    rtwp_data = {}
    for ne in sel['NE_NAME'].unique():
        arr = ( sel[sel['NE_NAME'] == ne]
                .sort_values('DATETIME')['VALUE']
                .to_numpy(dtype=float) )
        rtwp_data[ne] = arr
    
    return rtwp_data, pl_mat, anomaly_sectors, sector_ids


if __name__ == "__main__":
    rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids = prepare_inputs(
        rtwp_csv='Esfehan_RSSI_merged.csv',
        pathloss_csv='Data_bridge_ESFAHAN.Neighbors.csv',
        anomaly_csv='2.csv'
    )
    '''
    '''
    print("RTWP Data:", rtwp_data)
    print("Path Loss Matrix:", path_loss_matrix)
    print("Anomaly Sectors:", anomaly_sectors)
    print("Sector IDs:", sector_ids)
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
