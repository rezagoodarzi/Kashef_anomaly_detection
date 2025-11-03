import pandas as pd
import re

# Read the CSV files
df1 = pd.read_csv(r'C:\Users\Reza\Documents\GitHub\Kashef_anomaly\Kashef_anomaly_detection\data\3g_data_2100\3g_data_2100_merged.csv')


rssi_col = 'uplink_average_RSSI(dbm)(UCell_Eric)(CRA)'
df1['Anomaly'] = df1[rssi_col] > -93
df1.to_csv("TH_ne_site_anomaly.csv", index=False)