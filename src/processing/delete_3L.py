import pandas as pd

# Load the CSV file
file_path = r"C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged_with_coordinates_degree.csv"
df = pd.read_csv(file_path)

# Remove '3L' from the 'NE' column
df['NE'] = df['NE'].str.replace('3L', '', regex=False)

# Save the modified CSV (overwrite or create a new file)
new_file_path = r"C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_merged_with_coordinates_degree.csv"
df.to_csv(new_file_path, index=False)

print(f"All '3L' strings removed from 'NE' column and saved to {new_file_path}")