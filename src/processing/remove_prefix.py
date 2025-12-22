import pandas as pd

# Load the CSV file
df = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged_averaged.csv')

# Remove "KJ | " prefix from NE column
df['NE'] = df['NE'].str.replace('KJ | ', '', regex=False)

# Save the file
df.to_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged_averaged.csv', index=False)

print("Done! Prefix removed.")
