import pandas as pd
import re

df1 = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged_averaged.csv')
df2 = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\raw\Alborz_EP.csv')

# Select only the columns we need from first dataset
df2_filtered = df2[['Site', 'Latitude','Longitude']].copy()
df2_filtered.to_csv("site_lat_lon.csv", index=False)

def transform_ne_to_site(ne_value):
    """
    Transform NE format (e.g., KJ3L0002A) to Site format (e.g., KJ0002)
    1. Remove last character if it's a letter
    2. Remove characters between KJ and the numbers (like 3L)
    """
    # Remove last character if it's a letter
    if ne_value[-1].isalpha():
        ne_value = ne_value[:-1]
    
    # Keep the first 2 characters, skip the next 2, and keep the rest
    ne_value = ne_value[:2] + ne_value[4:]

    print(ne_value)
    return ne_value

df1_filtered = df1[['Date', 'NE', 'RSSI_PUCCH(EUCell_Eric)(CRA)']].copy()
df1_filtered['Site'] = df1_filtered['NE'].apply(transform_ne_to_site)

df1_filtered.to_csv("KJ_ne_site_anomaly_no_cordinate.csv", index=False)