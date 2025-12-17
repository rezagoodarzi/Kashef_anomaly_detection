import pandas as pd
import re

# Read the CSV files
df1 = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_ne_site_anomaly_no_cordinate.csv')


rssi_col = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
df1['Anomaly'] = df1[rssi_col] > -93
df1.to_csv("KJ_ne_site_anomaly_no_cordinate.csv", index=False)