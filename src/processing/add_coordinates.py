import pandas as pd

# Read the CSV files
df1 = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_ne_site_anomaly_no_cordinate.csv')
df2 = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\site_lat_lon.csv')

# Create a dictionary mapping site to coordinates from df2
# Each site has unique lat/long coordinates
site_coords = {}
for _, row in df2.iterrows():
    site = row['Site']
    if site not in site_coords:
        site_coords[site] = {
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'TowerHeight': row['TowerHeight'],
            'AntennaHeight': row['AntennaHeight'],
            'Azimuth': row['Azimuth'],
            'MTilt': row['MTilt'],
            'ETilt': row['ETilt']

        }

# Add Latitude and Longitude columns to df1 by iterating through rows
df1['Latitude'] = None
df1['Longitude'] = None
df1['TowerHeight'] = None
df1['AntennaHeight'] = None
df1['Azimuth'] = None
df1['MTilt'] = None
df1['ETilt'] = None

for idx, row in df1.iterrows():
    site = row['Site']
    if site in site_coords:
        df1.at[idx, 'Latitude'] = site_coords[site]['Latitude']
        df1.at[idx, 'Longitude'] = site_coords[site]['Longitude']
        df1.at[idx, 'TowerHeight'] = site_coords[site]['TowerHeight']
        df1.at[idx, 'AntennaHeight'] = site_coords[site]['AntennaHeight']
        df1.at[idx, 'Azimuth'] = site_coords[site]['Azimuth']
        df1.at[idx, 'MTilt'] = site_coords[site]['MTilt']
        df1.at[idx, 'ETilt'] = site_coords[site]['ETilt']


# Save the result
df1.to_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged_with_coordinates_degree.csv', index=False)

# Display results
print("Successfully added coordinates to df1")
print(f"Total rows in df1: {len(df1)}")
print(f"Rows with coordinates: {df1['Latitude'].notna().sum()}")
print(f"Rows without coordinates: {df1['Latitude'].isna().sum()}")
print("\nFirst few rows:")
print(df1.head())
