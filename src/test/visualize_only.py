#!/usr/bin/env python3
"""
Simple script to regenerate visualizations from saved results.
This is useful when you already have results and just want to create new visualizations.
"""

import sys
import importlib.util
from pathlib import Path

# Data file paths - UPDATE THESE TO MATCH YOUR DATA
rssi_kj_path = r'C:\Users\Reza\Documents\GitHub\Kashef_anomaly\Kashef_anomaly_detection\data\KJ_ne_site_anomaly_no_cordinate.csv'
neighbor_kj_path = r'C:\Users\Reza\Documents\GitHub\Kashef_anomaly\Kashef_anomaly_detection\data\Data_bridge_ALBORZ.Neighbors.csv'

# Results directory containing localization_results.json (in src/test/)
# Will look for: src/test/anomaly_results/
results_directory = 'anomaly_results'

def main():
    """Run visualization only"""
    print("Visualization-Only Mode")
    print("-" * 40)
    
    # Get the correct path to Source_Localization.py (in the same directory as this script)
    script_dir = Path(__file__).parent
    source_loc_path = script_dir / "Source_Localization.py"
    
    if not source_loc_path.exists():
        print(f"❌ Error: Could not find Source_Localization.py at {source_loc_path}")
        print("Please ensure the file is in src/test/ directory.")
        return
    
    # Load the main module
    spec = importlib.util.spec_from_file_location("anomaly_system", str(source_loc_path))
    if spec is None:
        print("Error: Could not load Source_Localization.py")
        return
        
    anomaly_system = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anomaly_system)
    
    # Check if results exist
    results_path = Path(results_directory) / 'localization_results.json'
    if not results_path.exists():
        print(f"\n❌ Error: Results file not found: {results_path}")
        print("\nYou need to run the full analysis first using runner_test.py")
        print("This will generate the results that can be visualized.")
        return
    
    # Run visualization
    print("\nRegenerating visualizations from saved results...\n")
    anomaly_system.visualize_from_saved_results(
        data_file=rssi_kj_path,
        neighbor_file=neighbor_kj_path,
        results_dir=results_directory
    )

if __name__ == "__main__":
    main()

