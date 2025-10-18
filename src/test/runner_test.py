#!/usr/bin/env python3
"""
Simple script to run the anomaly detection system with your data
"""

import pandas as pd
import numpy as np
from pathlib import Path
rssi_kj_path = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_merged_with_coordinates_degree.csv'
neighbor_kj_path =  r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\raw\Data_bridge_ALBORZ.Neighbors.csv'
# First, install required packages if not already installed
def install_requirements():
    """Install required packages"""
    requirements = [
        'numpy',
        'pandas', 
        'scipy',
        'scikit-learn',
        'ruptures',
        'matplotlib',
        'seaborn',
        'plotly',
        'folium'
    ]
    
    import subprocess
    import sys
    
    for package in requirements:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Create sample data files if they don't exist (for testing)
def create_sample_data():
    """Create sample data files for testing"""
    
    # Check if files already exist
    if Path('telecom_data.csv').exists() and Path('neighbor_data.csv').exists():
        print("Data files found. Using existing files.")
        return
    
    print("Creating sample data files for demonstration...")
    
    # Create main data
    np.random.seed(42)
    dates = pd.date_range('2025-09-01', '2025-09-14', freq='h')
    
    sectors = ['KJ0002A', 'KJ0002B', 'KJ0002C', 'KJ0003A', 'KJ0003B', 'KJ0003C']
    data = []
    
    for date in dates:
        for sector in sectors:
            # Base RSSI with hourly pattern
            hour = date.hour
            base_rssi = -110 + 5 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
            
            # Add anomalies
            is_anomaly = False
            if sector in ['KJ0002A', 'KJ0002B'] and date.day in [7, 8, 9]:
                if 10 <= hour <= 14:  # Anomaly during specific hours
                    base_rssi = np.random.uniform(-85, -75)
                    is_anomaly = True
            
            data.append({
                'Date': date,
                'NE': sector,
                'RSSI_PUCCH(EUCell_Eric)(CRA)': base_rssi,
                'Site': sector[:-1],
                'Anomaly': is_anomaly,
                'Latitude': 35.82481 + np.random.uniform(-0.01, 0.01),
                'Longitude': 50.97884 + np.random.uniform(-0.01, 0.01),
                'TowerHeight': 45,
                'AntennaHeight': 26.5,
                'Azimuth': [0, 120, 240][ord(sector[-1]) - ord('A')],
                'MTilt': 0,
                'ETilt': np.random.choice([0, 2, 4, 'Unavailable Ret'])
            })
    
    df = pd.DataFrame(data)
    df.to_csv('telecom_data.csv', index=False)
    print(f"Created telecom_data.csv with {len(df)} records")
    
    # Create neighbor data
    neighbor_data = []
    for sector in sectors:
        for i, neighbor in enumerate(sectors):
            if sector != neighbor:
                distance = np.random.uniform(100, 5000)
                neighbor_data.append({
                    '_id': f'id_{sector}_{neighbor}',
                    'Sector': sector,
                    'Neighbor': neighbor,
                    'rank': i + 1,
                    'distance': distance,
                    'pathloss': 100 + 20 * np.log10(distance / 100),
                    'ring': min(int(distance / 1000), 3),
                    'type': '3G_3G'
                })
    
    neighbor_df = pd.DataFrame(neighbor_data)
    neighbor_df.to_csv('neighbor_data.csv', index=False)
    print(f"Created neighbor_data.csv with {len(neighbor_df)} records")

# Quick data validation
def validate_data():
    """Validate the data files"""
    try:
        # Load main data
        df = pd.read_csv(rssi_kj_path)
        print(f"\n✓ Main data loaded: {len(df)} records")
        print(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  - Sectors: {df['NE'].nunique()}")
        print(f"  - Anomalies: {df['Anomaly'].sum()} ({df['Anomaly'].sum()/len(df)*100:.1f}%)")
        
        # Check required columns
        required_cols = ['Date', 'NE', 'RSSI_PUCCH(EUCell_Eric)(CRA)', 'Anomaly', 
                        'Latitude', 'Longitude', 'Azimuth']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  ⚠ Missing columns: {missing}")
        else:
            print("  ✓ All required columns present")
        
        # Load neighbor data
        neighbors = pd.read_csv(neighbor_kj_path)
        print(f"\n✓ Neighbor data loaded: {len(neighbors)} relationships")
        print(f"  - Unique sectors: {neighbors['Sector'].nunique()}")
        
        # Check data quality
        print("\n📊 Data Quality Check:")
        print(f"  - RSSI range: {df['RSSI_PUCCH(EUCell_Eric)(CRA)'].min():.1f} to {df['RSSI_PUCCH(EUCell_Eric)(CRA)'].max():.1f} dBm")
        print(f"  - Missing RSSI values: {df['RSSI_PUCCH(EUCell_Eric)(CRA)'].isna().sum()}")
        print(f"  - Missing locations: {df['Latitude'].isna().sum()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error validating data: {e}")
        return False

def run_analysis():
    """Run the complete analysis"""
    print("\n" + "="*80)
    print("STARTING ANOMALY SOURCE LOCALIZATION ANALYSIS")
    print("="*80)
    
    # Import the main module (assuming it's saved as anomaly_localization.py)
    import sys
    import importlib.util
    
    # Load the main module dynamically
    spec = importlib.util.spec_from_file_location("anomaly_system", "Source_Localization.py")
    if spec is None:
        print("Error: Could not find anomaly_localization.py")
        print("Please ensure the main implementation file is saved as 'anomaly_localization.py'")
        return
        
    anomaly_system = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anomaly_system)
    
    # Run the analysis
    try:
        # Load data
                # Load data
        '''
        loader = anomaly_system.DataLoader(rssi_kj_path,neighbor_kj_path)
        data_df, neighbor_df = loader.load_data()
        data_df = loader.preprocess_data(data_df)
        '''

        loader = anomaly_system.DataLoader(rssi_kj_path, neighbor_kj_path)
        data_df, neighbor_df = loader.load_data()
        
        # Sample 100 unique sectors
        unique_sectors = data_df['NE'].unique()
        sampled_sectors = np.random.choice(unique_sectors, min(800, len(unique_sectors)), replace=False)
        
        # Filter for sampled sectors and last 10 days
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        last_date = data_df['Date'].max()
        start_date = last_date - pd.Timedelta(days=10)
        
        data_df = data_df[
            (data_df['NE'].isin(sampled_sectors)) & 
            (data_df['Date'] >= start_date)
        ]
        # Drop rows with missing latitude or longitude
        data_df = data_df.dropna(subset=['Latitude', 'Longitude'])
        
        data_df = loader.preprocess_data(data_df)
        # Run pipeline
        pipeline = anomaly_system.AnomalyLocalizationPipeline()
        results = pipeline.process_data(data_df, neighbor_df)
        
        # Create visualizations
        visualizer = anomaly_system.Visualizer()
        visualizer.create_all_visualizations(data_df, results, neighbor_df)
        
        # Print results summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        if results:
            print(f"\n📍 Detected {len(results)} anomaly source(s)")
            
            for i, result in enumerate(results, 1):
                if result['source_location']:
                    loc = result['source_location']
                    print(f"\n  Source {i}:")
                    print(f"    - Location: ({loc['latitude']:.6f}, {loc['longitude']:.6f})")
                    print(f"    - Uncertainty: {loc['uncertainty_meters']:.0f} meters")
                    print(f"    - Pattern: {result['temporal_pattern']}")
                    print(f"    - Confidence: {result['pattern_confidence']:.1%}")
                    print(f"    - Duration: {result['duration_hours']:.1f} hours")
                    print(f"    - Affected sectors: {result['num_anomaly_sectors']}")
        else:
            print("\nNo anomaly sources detected in the current data.")
        
        print("\n📁 Output Files Created:")
        output_files = [
            ('anomaly_results/localization_results.json', 'Detailed results'),
            ('anomaly_results/localization_summary.csv', 'Summary table'),
            ('anomaly_results/anomaly_map.html', 'Interactive map'),
            ('anomaly_results/rssi_analysis.html', 'RSSI analysis plots'),
            ('anomaly_results/temporal_analysis.png', 'Temporal patterns'),
            ('anomaly_results/localization_analysis.png', 'Accuracy metrics'),
            ('anomaly_results/network_topology.html', 'Network visualization')
        ]
        
        for file, desc in output_files:
            if Path(file).exists():
                print(f"  ✓ {file} - {desc}")
        
        print("\n💡 Next Steps:")
        print("  1. Open 'anomaly_results/anomaly_map.html' in a browser to see the interactive map")
        print("  2. Review 'anomaly_results/localization_summary.csv' for source details")
        print("  3. Check the visualization plots for patterns and insights")
        
        # Open the map automatically if possible
        import webbrowser
        map_path = Path('anomaly_results/anomaly_map.html').absolute()
        if map_path.exists():
            print(f"\n🌐 Opening interactive map in browser...")
            webbrowser.open(f'file://{map_path}')
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution"""
    print("Telecom Anomaly Source Localization System")
    print("-" * 40)
    
    # Install requirements
    print("\n1. Checking required packages...")
    #install_requirements()
    
    # Create sample data if needed
    print("\n2. Checking data files...")
    #create_sample_data()
    
    # Validate data
    print("\n3. Validating data...")
    if not validate_data():
        print("Please fix data issues before proceeding.")
        return
    
    # Save the main implementation
    print("\n4. Preparing analysis system...")
    
    # Check if main implementation exists
    if not Path('Source_Localization.py').exists():
        print("⚠ Warning: anomaly_localization.py not found!")
        print("Please save the main implementation code as 'anomaly_localization.py'")
        print("You can copy it from the first artifact provided.")
        return
     
    # Run analysis
    print("\n5. Running analysis...")
    run_analysis()

if __name__ == "__main__":
    main()