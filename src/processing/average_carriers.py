import pandas as pd
import numpy as np
import os

def average_carriers(input_csv, output_csv="KJ3L_merged_averaged.csv"):
    """
    Average RSSI values across all carriers (1-5) for each sector at each timestamp.

    For example:
    - KJ | KJ3L0021D1, KJ | KJ3L0021D2, ... KJ | KJ3L0021D5
    Will become:
    - KJ | KJ3L0021D (with averaged RSSI values)

    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
    """
    print("="*70)
    print("CARRIER AVERAGING PROCESS")
    print("="*70)

    # Read the CSV file
    print(f"\n[1/5] Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"      Loaded {len(df):,} rows")
    print(f"      Columns: {list(df.columns)}")

    # Show sample data
    print(f"\n[2/5] Sample data before processing:")
    print(df.head(10))

    # Extract base sector name (remove the carrier number at the end)
    print(f"\n[3/5] Extracting base sector names (removing carrier numbers)...")
    # The pattern is: "KJ | KJ3L0021D1" -> "KJ | KJ3L0021D"
    # We remove the last character if it's a digit (1-5)
    df['Base_Sector'] = df['NE'].str.rstrip('12345')

    # Show unique carriers per sector (for verification)
    sample_sectors = df['Base_Sector'].unique()[:5]
    print(f"\n      Sample base sectors created:")
    for sector in sample_sectors:
        carriers = df[df['Base_Sector'] == sector]['NE'].unique()
        print(f"      {sector} -> {len(carriers)} carriers: {list(carriers)[:5]}")

    # Convert RSSI column to numeric (handle empty values)
    print(f"\n[4/5] Converting RSSI values to numeric...")
    rssi_col = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
    df[rssi_col] = pd.to_numeric(df[rssi_col], errors='coerce')

    null_count = df[rssi_col].isna().sum()
    print(f"      Found {null_count:,} null/empty RSSI values ({null_count/len(df)*100:.2f}%)")

    # Group by Date and Base_Sector, then average the RSSI values
    print(f"\n[5/5] Averaging RSSI values across carriers for each sector at each timestamp...")
    averaged_df = df.groupby(['Date', 'Base_Sector'], as_index=False).agg({
        rssi_col: 'mean'  # Average the RSSI values
    })

    # Rename Base_Sector back to NE
    averaged_df = averaged_df.rename(columns={'Base_Sector': 'NE'})

    # Sort by Date and NE
    averaged_df = averaged_df.sort_values(['Date', 'NE']).reset_index(drop=True)

    # Calculate statistics
    original_rows = len(df)
    averaged_rows = len(averaged_df)
    reduction_pct = (1 - averaged_rows / original_rows) * 100

    unique_sectors_before = df['NE'].nunique()
    unique_sectors_after = averaged_df['NE'].nunique()

    # Save to CSV
    output_path = os.path.join(os.path.dirname(input_csv), output_csv)
    averaged_df.to_csv(output_path, index=False)

    # Print results
    print("\n" + "="*70)
    print("AVERAGING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nInput file:  {os.path.basename(input_csv)}")
    print(f"Output file: {output_csv}")
    print(f"\nRows before: {original_rows:,}")
    print(f"Rows after:  {averaged_rows:,}")
    print(f"Reduction:   {reduction_pct:.1f}%")
    print(f"\nUnique sectors before: {unique_sectors_before:,} (with carrier numbers)")
    print(f"Unique sectors after:  {unique_sectors_after:,} (base sectors only)")
    print(f"Average carriers per sector: {unique_sectors_before / unique_sectors_after:.1f}")

    print(f"\n[Sample of averaged data]:")
    print(averaged_df.head(15))

    print(f"\n[Date range]:")
    print(f"Start: {averaged_df['Date'].min()}")
    print(f"End:   {averaged_df['Date'].max()}")

    print(f"\n[RSSI Statistics]:")
    print(f"Mean:   {averaged_df[rssi_col].mean():.3f}")
    print(f"Median: {averaged_df[rssi_col].median():.3f}")
    print(f"Min:    {averaged_df[rssi_col].min():.3f}")
    print(f"Max:    {averaged_df[rssi_col].max():.3f}")
    print(f"Nulls:  {averaged_df[rssi_col].isna().sum():,}")

    print("\n" + "="*70)
    print(f"File saved to: {output_path}")
    print("="*70)

    return averaged_df

if __name__ == "__main__":
    # Input file path
    input_file = r"C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged.csv"

    # Output file name
    output_file = "KJ3L_merged_averaged.csv"

    # Run the averaging process
    try:
        result_df = average_carriers(input_file, output_file)
        print("\nProcess completed successfully!")
    except FileNotFoundError:
        print(f"\nError: Input file not found at: {input_file}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
