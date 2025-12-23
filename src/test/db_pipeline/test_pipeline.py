#!/usr/bin/env python3
"""
Test Script for Database-Driven Behavioral Analysis Pipeline
============================================================

Quick test to verify the pipeline works with mock data.
Run this to ensure everything is set up correctly.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from db_pipeline import run_pipeline

def test_mock_pipeline():
    """Test pipeline with mock CSV data."""
    
    print("=" * 70)
    print("TESTING BEHAVIORAL ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Use existing CSV file path
    mock_data_path = r"C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv"
    
    if not Path(mock_data_path).exists():
        print(f"ERROR: Mock data file not found at: {mock_data_path}")
        print("Please update the path in this script.")
        return False
    
    print(f"\nUsing mock data: {mock_data_path}")
    
    # Run pipeline
    try:
        result = run_pipeline(
            kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
            city="Karaj",
            technology="4g",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config_dir=str(Path(__file__).parent / "config"),
            output_dir=str(Path(__file__).parent / "output"),
            mock_data_path=mock_data_path
        )
        
        # Print results
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Success: {result.success}")
        
        if result.success:
            print(f"✓ Connections found: {result.connections_count}")
            print(f"✓ Change points found: {result.change_points_count}")
            print(f"✓ Sectors analyzed: {result.sectors_analyzed}")
            print(f"✓ Execution time: {result.execution_time_seconds:.2f} seconds")
            
            if result.output_files:
                print("\nOutput files:")
                for name, path in result.output_files.items():
                    print(f"  - {name}: {path}")
            
            print("\n✓ TEST PASSED!")
            return True
        else:
            print(f"✗ TEST FAILED: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mock_pipeline()
    sys.exit(0 if success else 1)

