import pandas as pd
import os

def merge_csv_files(file_list, output_filename="KJ3L_merged.csv"):
    """
    Merge multiple CSV files into a single CSV file.

    Args:
        file_list (list): List of CSV filenames to merge
        output_filename (str): Name of the output merged CSV file
    """
    print("Starting CSV merge process...\n")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List to store dataframes
    dfs = []

    # Read each CSV file
    for csv_file in file_list:
        file_path = os.path.join(script_dir, csv_file)

        if not os.path.exists(file_path):
            print(f"[X] File not found: {csv_file}")
            continue

        try:
            df = pd.read_csv(file_path)
            print(f"[OK] Loaded {csv_file}")
            print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}\n")
            dfs.append(df)
        except Exception as e:
            print(f"[X] Error reading {csv_file}: {str(e)}\n")

    if not dfs:
        print("No files were successfully loaded. Merge aborted.")
        return

    # Concatenate all dataframes
    print("Merging dataframes...")
    merged_df = pd.concat(dfs, ignore_index=True)

    # Sort by Date
    print("Sorting by date...")
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)

    # Remove duplicates if any
    initial_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['Date', 'NE'], keep='first')
    duplicates_removed = initial_rows - len(merged_df)

    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")

    # Save merged file
    output_path = os.path.join(script_dir, output_filename)
    merged_df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "="*60)
    print("MERGE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Output file: {output_filename}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")
    print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"Unique NE values: {merged_df['NE'].nunique()}")
    print("="*60)

if __name__ == "__main__":
    # List of CSV files to merge (in the order you specified)
    files_to_merge = [
        "KJ3L_1.csv",
        "KJ3L_3.csv",
        "KJ3L_2.csv"
    ]

    # Output filename
    output_file = "KJ3L_merged.csv"

    # Merge the files
    merge_csv_files(files_to_merge, output_file)
