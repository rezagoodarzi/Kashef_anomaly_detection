import pandas as pd
import os
import glob

def convert_excel_to_csv(directory_path, sheet_name="Interference-Sources"):
    """
    Convert specific sheet from Excel files in the specified directory to CSV format.

    Args:
        directory_path (str): Path to the directory containing Excel files
        sheet_name (str): Name of the sheet to extract (default: "Interference-Sources")
    """
    # Change to the target directory
    os.chdir(directory_path)

    # Find all Excel files (both .xlsx and .xls)
    excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")

    if not excel_files:
        print("No Excel files found in the directory.")
        return

    print(f"Found {len(excel_files)} Excel file(s):")
    print(f"Extracting sheet: '{sheet_name}'\n")

    for excel_file in excel_files:
        print(f"Processing: {excel_file}")

        try:
            # Read all sheet names first to check if the target sheet exists
            excel_data = pd.ExcelFile(excel_file)
            print(f"  Available sheets: {excel_data.sheet_names}")

            if sheet_name not in excel_data.sheet_names:
                print(f"  ✗ Sheet '{sheet_name}' not found in {excel_file}")
                continue

            # Read only the specified sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Create CSV filename with sheet name
            base_filename = os.path.splitext(excel_file)[0]
            csv_filename = f"{base_filename}_{sheet_name}.csv"

            # Convert to CSV
            df.to_csv(csv_filename, index=False)

            print(f"  ✓ Converted {excel_file} (sheet: {sheet_name}) → {csv_filename}")
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}\n")

        except Exception as e:
            print(f"  ✗ Error converting {excel_file}: {str(e)}\n")

    print("Conversion completed!")

if __name__ == "__main__":
    # Directory path containing the Excel files
    directory_path = r"C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\raw"

    # Convert Excel files to CSV
    convert_excel_to_csv(directory_path)