import pandas as pd
import re

# Read the CSV files
df1 = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged_averaged.csv')
df2 = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\raw\Alborz_EP.csv')

# Select only the columns we need from first dataset
df1_filtered = df1[['Date', 'NE', 'RSSI_PUCCH(EUCell_Eric)(CRA)']].copy()

# Select only the columns we need from second dataset
df2_filtered = df2[['Site', 'Latitude', 'Longitude']].copy()
df2_filtered = df2_filtered.drop_duplicates(subset='Site', keep='first')

# Function to transform NE to Site format
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

# Apply the transformation to create a matching column
df1_filtered['Site'] = df1_filtered['NE'].apply(transform_ne_to_site)

# Merge the dataframes based on Site
merged_df = df1_filtered.merge(
    df2_filtered,
    on='Site',
    how='left'
)

# Reorder columns for better readability
merged_df = merged_df[['Date', 'NE', 'Site', 'Latitude', 'Longitude', 'RSSI_PUCCH(EUCell_Eric)(CRA)']]

# Save the merged dataset
merged_df.to_csv('merged_output.csv', index=False)

# Display first few rows to verify
print("Merged dataset preview:")
print(merged_df.head(15))
print(f"\nTotal rows: {len(merged_df)}")
print(f"Rows with matching coordinates: {merged_df['Latitude'].notna().sum()}")
print(f"Rows without matching coordinates: {merged_df['Latitude'].isna().sum()}")