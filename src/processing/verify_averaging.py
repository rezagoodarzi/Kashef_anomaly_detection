import pandas as pd

# Load both files
print("Loading files...")
df_orig = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged.csv')
df_avg = pd.read_csv(r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ3L_merged_averaged.csv')

# Add base sector column to original
df_orig['Base'] = df_orig['NE'].str.rstrip('12345')

# Find sectors with multiple carriers
bases = df_orig.groupby('Base')['NE'].nunique()
multi_carrier = bases[bases > 1].head(10)

print("\nSectors with multiple carriers:")
print(multi_carrier)

if len(multi_carrier) > 0:
    # Check first example
    sample_base = multi_carrier.index[0]
    sample_data = df_orig[(df_orig['Base'] == sample_base) & (df_orig['Date'] == '2025-09-07 00:00:00')]

    print(f"\n\nExample verification for: {sample_base}")
    print("="*70)
    print("\nOriginal data (with carriers):")
    print(sample_data[['Date', 'NE', 'RSSI_PUCCH(EUCell_Eric)(CRA)']].to_string(index=False))

    # Calculate average
    avg_val = pd.to_numeric(sample_data['RSSI_PUCCH(EUCell_Eric)(CRA)'], errors='coerce').mean()
    print(f"\nCalculated average: {avg_val:.3f}")

    # Check averaged file
    result = df_avg[(df_avg['NE'] == sample_base) & (df_avg['Date'] == '2025-09-07 00:00:00')]
    print("\nAveraged data (from output file):")
    print(result[['Date', 'NE', 'RSSI_PUCCH(EUCell_Eric)(CRA)']].to_string(index=False))

    print("\n" + "="*70)
    if not result.empty:
        file_avg = result['RSSI_PUCCH(EUCell_Eric)(CRA)'].values[0]
        print(f"Verification: {avg_val:.3f} == {file_avg:.3f} ? {abs(avg_val - file_avg) < 0.001}")
else:
    print("\nNo sectors found with multiple carriers.")
    print("This means each sector only has one carrier number.")
