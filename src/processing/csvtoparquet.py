import pandas as pd
import os

def csv_to_parquet(csv_path, parquet_path=None):
    """
    Convert a CSV file to Parquet format
    
    Args:
        csv_path (str): Path to the input CSV file
        parquet_path (str): Path for the output Parquet file. If None, will use same name as CSV
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # If no parquet_path is provided, create one from csv_path
        if parquet_path is None:
            parquet_path = os.path.splitext(csv_path)[0] + '.parquet'
        
        # Convert to parquet
        df.to_parquet(parquet_path, index=False)
        print(f"Successfully converted {csv_path} to {parquet_path}")
        
    except Exception as e:
        print(f"Error converting file: {str(e)}")

if __name__ == "__main__":
    # Example usage
    csv_file =r"C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_anomaly_with_cordinate.csv"  # Replace with your CSV file path
    csv_to_parquet(csv_file)