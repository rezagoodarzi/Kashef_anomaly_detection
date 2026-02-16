#!/usr/bin/env python3
"""
Plot RSSI time series for a sector and all its behaviorally correlated neighbors.
Uses behavioral_connections.csv to find correlated sectors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SectorCompositeScorePlotter:
    """Plot RSSI composite_scores for sectors"""
    
    def __init__(self, data_file, connections_file, change_points_file=None, results_dir='anomaly_results'):
        """
        Initialize the plotter
        
        Args:
            data_file: Path to main data CSV (with RSSI values)
            connections_file: Path to behavioral_connections.csv
            change_points_file: Path to change_points.csv or .json (optional)
            results_dir: Directory containing analysis results
        """
        self.data_file = data_file
        self.connections_file = connections_file
        self.results_dir = Path(results_dir)
        
        # Load data
        print("Loading data...")
        self.data_df = pd.read_csv(data_file)
        self.data_df['Date'] = pd.to_datetime(self.data_df['Date'])
        
        # Load behavioral connections
        connections_path = self.results_dir / connections_file
        if not connections_path.exists():
            # Try absolute path if relative fails
            connections_path = Path(connections_file)
            if not connections_path.exists():
                print(f"❌ Error: {connections_path} not found!")
                print("Please run the analysis first to generate behavioral_connections.csv")
                sys.exit(1)
            
        self.connections_df = pd.read_csv(connections_path)
        print(f"✓ Loaded {len(self.connections_df)} behavioral connections")
        print(f"✓ Loaded {len(self.data_df)} data records")
        
        # Load change points if provided
        self.change_points_data = {}
        if change_points_file:
            # Check if absolute or relative
            cp_path = Path(change_points_file)
            if not cp_path.exists():
                cp_path = self.results_dir / change_points_file
            
            if cp_path.exists():
                print(f"Loading change points from {cp_path}...")
                
                if cp_path.suffix.lower() == '.json':
                    with open(cp_path, 'r', encoding='utf-8') as f:
                        cp_data = json.load(f)
                        # Extract by_sector data for quick lookup
                        if 'by_sector' in cp_data:
                            for sector, sector_data in cp_data['by_sector'].items():
                                self.change_points_data[sector] = sector_data.get('change_points', [])
                                
                elif cp_path.suffix.lower() == '.csv':
                    # Load CSV and convert to dictionary of lists
                    cp_df = pd.read_csv(cp_path)
                    # Ensure required columns exist
                    required = ['sector', 'timestamp', 'direction']
                    if not all(col in cp_df.columns for col in required):
                         print(f"⚠ Warning: CSV missing required columns: {required}")
                    else:
                        # Group by sector
                        for sector, group in cp_df.groupby('sector'):
                            # Convert group to list of dicts
                            records = group.to_dict('records')
                            # Map columns to match what plotting expects if needed
                            # Plotting expects: timestamp, value_at_change, direction
                            # CSV has: timestamp, value_after, direction. 
                            # We'll use value_after as value_at_change
                            for r in records:
                                if 'value_at_change' not in r and 'value_after' in r:
                                    r['value_at_change'] = r['value_after']
                            
                            self.change_points_data[sector] = records
                            
                print(f"✓ Loaded change points for {len(self.change_points_data)} sectors")
            else:
                print(f"⚠ Warning: Change points file not found: {change_points_file}")
        
    def find_correlated_sectors(self, sector_name):
        """
        Find all sectors correlated with the given sector
        
        Args:
            sector_name: Name of the sector to query
            
        Returns:
            DataFrame with correlated sectors and their info
        """
        # Find connections where sector is either sector_1 or sector_2
        connections = self.connections_df[
            (self.connections_df['sector_1'] == sector_name) | 
            (self.connections_df['sector_2'] == sector_name)
        ].copy()
        
        if len(connections) == 0:
            return None
        
        # Create a list of correlated sectors
        correlated = []
        for _, row in connections.iterrows():
            if row['sector_1'] == sector_name:
                correlated.append({
                    'sector': row['sector_2'],
                    'composite_score': row['composite_score'],
                    'has_anomaly': row['sector_2_has_anomaly'],
                    'avg_rssi': row['sector_2_avg_rssi']
                })
            else:
                correlated.append({
                    'sector': row['sector_1'],
                    'composite_score': row['composite_score'],
                    'has_anomaly': row['sector_1_has_anomaly'],
                    'avg_rssi': row['sector_1_avg_rssi']
                })
        
        return pd.DataFrame(correlated).sort_values('composite_score', ascending=False)
    
    def plot_sector_and_composite_scores(self, sector_name, date_range=None, top_n=10, save_path=None):
        """
        Plot RSSI time series for a sector and its correlated neighbors
        
        Args:
            sector_name: Name of the sector to plot
            date_range: Tuple of (start_date, end_date) or None for all dates
            top_n: Number of top correlated sectors to show (default: 10)
            save_path: Path to save the plot (optional)
        """
        # Check if sector exists in data
        if sector_name not in self.data_df['NE'].values:
            print(f"❌ Error: Sector '{sector_name}' not found in data!")
            available_sectors = self.data_df['NE'].unique()[:20]
            print(f"Available sectors (first 20): {', '.join(available_sectors)}")
            return
        
        # Find correlated sectors
        print(f"\n🔍 Finding sectors correlated with '{sector_name}'...")
        correlated_df = self.find_correlated_sectors(sector_name)
        
        if correlated_df is None or len(correlated_df) == 0:
            print(f"⚠ No correlated sectors found for '{sector_name}'")
            print("This sector may not have strong behavioral connections (composite_score > 0.7)")
            return
        
        print(f"✓ Found {len(correlated_df)} correlated sectors")
        
        # Limit to top N
        correlated_df = correlated_df.head(top_n)
        
        # Get RSSI data for main sector
        sector_data = self.data_df[self.data_df['NE'] == sector_name].copy()
        
        # Filter by date range if provided
        if date_range:
            start_date, end_date = date_range
            sector_data = sector_data[
                (sector_data['Date'] >= start_date) & 
                (sector_data['Date'] <= end_date)
            ]
        
        sector_data = sector_data.sort_values('Date')
        
        # Create figure with subplots
        n_plots = len(correlated_df) + 1  # +1 for main sector
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot main sector (first plot)
        ax = axes[0]
        has_anomaly = sector_data['Anomaly'].any() if 'Anomaly' in sector_data.columns else False
        
        # Plot RSSI
        ax.plot(sector_data['Date'], sector_data['RSSI_PUCCH(EUCell_Eric)(CRA)'], 
                linewidth=2, color='darkblue', label='RSSI')
        
        # Plot change points if available
        if sector_name in self.change_points_data:
            cp_list = self.change_points_data[sector_name]
            if cp_list:
                # Filter change points by date range if provided
                cp_timestamps = []
                cp_values = []
                cp_directions = []
                
                for cp in cp_list:
                    cp_time = pd.to_datetime(cp['timestamp'])
                    if date_range:
                        start_date, end_date = date_range
                        if cp_time >= pd.to_datetime(start_date) and cp_time <= pd.to_datetime(end_date):
                            cp_timestamps.append(cp_time)
                            cp_values.append(cp.get('value_at_change', cp.get('value_after', 0)))
                            cp_directions.append(cp['direction'])
                    else:
                        cp_timestamps.append(cp_time)
                        cp_values.append(cp.get('value_at_change', cp.get('value_after', 0)))
                        cp_directions.append(cp['direction'])
                
                # Plot UP change points in green
                up_times = [t for t, d in zip(cp_timestamps, cp_directions) if d == 'UP']
                up_values = [v for v, d in zip(cp_values, cp_directions) if d == 'UP']
                if up_times:
                    ax.scatter(up_times, up_values, color='lime', s=120, marker='^', 
                              alpha=0.9, label='Change Point (UP)', zorder=10, edgecolors='black', linewidths=1)
                
                # Plot DOWN change points in red
                down_times = [t for t, d in zip(cp_timestamps, cp_directions) if d == 'DOWN']
                down_values = [v for v, d in zip(cp_values, cp_directions) if d == 'DOWN']
                if down_times:
                    ax.scatter(down_times, down_values, color='magenta', s=120, marker='v', 
                              alpha=0.9, label='Change Point (DOWN)', zorder=10, edgecolors='black', linewidths=1)
        
        # Highlight anomalies if present
        if 'Anomaly' in sector_data.columns and sector_data['Anomaly'].any():
            anomaly_data = sector_data[sector_data['Anomaly'] == True]
            ax.scatter(anomaly_data['Date'], anomaly_data['RSSI_PUCCH(EUCell_Eric)(CRA)'],
                      color='red', s=50, alpha=0.3, label='Anomaly', zorder=5)
        
        # Styling
        ax.set_title(f"🎯 TARGET SECTOR: {sector_name}\n" + 
                    f"{'⚠ HAS ANOMALIES' if has_anomaly else '✓ No anomalies'}", 
                    fontsize=12, fontweight='bold', color='darkblue')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('RSSI (dBm)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add stats text
        stats_text = f"Avg: {sector_data['RSSI_PUCCH(EUCell_Eric)(CRA)'].mean():.1f} dBm\n"
        stats_text += f"Min: {sector_data['RSSI_PUCCH(EUCell_Eric)(CRA)'].min():.1f} dBm\n"
        stats_text += f"Max: {sector_data['RSSI_PUCCH(EUCell_Eric)(CRA)'].max():.1f} dBm"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot correlated sectors
        for idx, (_, corr_info) in enumerate(correlated_df.iterrows(), start=1):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            corr_sector = corr_info['sector']
            composite_score = corr_info['composite_score']
            
            # Get data for this correlated sector
            corr_data = self.data_df[self.data_df['NE'] == corr_sector].copy()
            
            # Filter by date range if provided
            if date_range:
                corr_data = corr_data[
                    (corr_data['Date'] >= start_date) & 
                    (corr_data['Date'] <= end_date)
                ]
            
            corr_data = corr_data.sort_values('Date')
            
            if len(corr_data) == 0:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{corr_sector}\ncomposite_score: {composite_score:.3f}")
                continue
            
            # Determine color based on composite_score
            if composite_score > 0.85:
                color = 'darkgreen'
                strength = 'Very Strong'
            elif composite_score > 0.75:
                color = 'green'
                strength = 'Strong'
            else:
                color = 'orange'
                strength = 'Moderate'
            
            # Plot RSSI
            ax.plot(corr_data['Date'], corr_data['RSSI_PUCCH(EUCell_Eric)(CRA)'],
                   linewidth=2, color=color, label='RSSI', alpha=0.8)
            
            # Plot change points if available
            if corr_sector in self.change_points_data:
                cp_list = self.change_points_data[corr_sector]
                if cp_list:
                    # Filter change points by date range if provided
                    cp_timestamps = []
                    cp_values = []
                    cp_directions = []
                    
                    for cp in cp_list:
                        cp_time = pd.to_datetime(cp['timestamp'])
                        if date_range:
                            start_date, end_date = date_range
                            if cp_time >= pd.to_datetime(start_date) and cp_time <= pd.to_datetime(end_date):
                                cp_timestamps.append(cp_time)
                                cp_values.append(cp.get('value_at_change', cp.get('value_after', 0)))
                                cp_directions.append(cp['direction'])
                        else:
                            cp_timestamps.append(cp_time)
                            cp_values.append(cp.get('value_at_change', cp.get('value_after', 0)))
                            cp_directions.append(cp['direction'])
                    
                    # Plot UP change points in green
                    up_times = [t for t, d in zip(cp_timestamps, cp_directions) if d == 'UP']
                    up_values = [v for v, d in zip(cp_values, cp_directions) if d == 'UP']
                    if up_times:
                        ax.scatter(up_times, up_values, color='lime', s=100, marker='^', 
                                  alpha=0.7, label='Change Point (UP)', zorder=10, edgecolors='black', linewidths=0.5)
                    
                    # Plot DOWN change points in red
                    down_times = [t for t, d in zip(cp_timestamps, cp_directions) if d == 'DOWN']
                    down_values = [v for v, d in zip(cp_values, cp_directions) if d == 'DOWN']
                    if down_times:
                        ax.scatter(down_times, down_values, color='magenta', s=100, marker='v', 
                                  alpha=0.7, label='Change Point (DOWN)', zorder=10, edgecolors='black', linewidths=0.5)
            
            # Highlight anomalies if present
            if 'Anomaly' in corr_data.columns and corr_data['Anomaly'].any():
                anomaly_data = corr_data[corr_data['Anomaly'] == True]
                ax.scatter(anomaly_data['Date'], anomaly_data['RSSI_PUCCH(EUCell_Eric)(CRA)'],
                          color='red', s=50, alpha=0.3, label='Anomaly', zorder=5)
            
            # Styling
            has_anomaly = corr_info['has_anomaly']
            ax.set_title(f"Sector: {corr_sector}\n" + 
                        f"composite_score: {composite_score:.3f} ({strength}) " +
                        f"{'⚠' if has_anomaly else '✓'}", 
                        fontsize=11, color=color)
            ax.set_xlabel('Date', fontsize=9)
            ax.set_ylabel('RSSI (dBm)', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add stats
            avg_rssi = corr_data['RSSI_PUCCH(EUCell_Eric)(CRA)'].mean()
            stats_text = f"Avg: {avg_rssi:.1f} dBm"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle(f'Behavioral composite_score Analysis: {sector_name}\n' + 
                    f'Showing top {len(correlated_df)} correlated sectors',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        # Show plot
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print(f"composite_score SUMMARY FOR: {sector_name}")
        print("="*60)
        for idx, row in correlated_df.iterrows():
            status = "⚠ ANOMALY" if row['has_anomaly'] else "✓ Normal"
            print(f"  {row['sector']:15} | Corr: {row['composite_score']:.3f} | "
                  f"Avg RSSI: {row['avg_rssi']:.1f} dBm | {status}")
        print("="*60)

def main():
    """Main function"""
    print("="*60)
    print("SECTOR BEHAVIORAL composite_score PLOTTER")
    print("="*60)
    
    # Configuration - UPDATE THESE PATHS
    data_file = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv'
    connections_file = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline\behavioral_connections.csv'
    
    # Updated to use the correct CSV file path
    change_points_file = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline_changepoint\change_points.csv'
    
    results_dir = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline'
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print("Please update the data_file path in the script.")
        return
    
    # Initialize plotter with change points
    plotter = SectorCompositeScorePlotter(
        data_file, 
        connections_file, 
        change_points_file=change_points_file,
        results_dir=results_dir
    )
    
    # Specify the sector you want to analyze
    sector_name = "KJ0031A"  # Change this to your target sector

    if not sector_name:
        print("No sector specified!")
        return
    
    # Optional: Limit date range (uncomment to use)
    # date_range = ('2024-01-01', '2024-01-31')
    date_range = None
    
    # Plot
    print(f"\n📈 Plotting composite_scores for: {sector_name}")
    
    # Optional: Save to file
    save_dir = Path(results_dir)
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'composite_score_plot_{sector_name}.png'
    
    plotter.plot_sector_and_composite_scores(
        sector_name=sector_name,
        date_range=("2025-09-13", "2025-09-30"),
        top_n=15,  # Show top 15 correlated sectors
        save_path=save_path
    )

if __name__ == "__main__":
    main()
