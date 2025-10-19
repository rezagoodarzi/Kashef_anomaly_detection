#!/usr/bin/env python3
"""
Telecom Anomaly Source Localization System
Detects and localizes interference sources based on RSSI changes and network topology
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

# Data processing and ML
from scipy.stats import pearsonr, median_abs_deviation
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Change point detection
import ruptures as rpt

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import folium.plugins as plugins

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DataLoader:
    """Load and preprocess the telecom data"""
    
    def __init__(self, data_path, neighbor_path):
        self.data_path = data_path
        self.neighbor_path = neighbor_path
        
    def load_data(self):
        """Load main dataset and neighbor data"""
        # Load main data
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle missing values
        df['ETilt'] = df['ETilt'].replace('Unavailable Ret', np.nan)
        df['ETilt'] = pd.to_numeric(df['ETilt'], errors='coerce')
        
        # Load neighbor data
        neighbors = pd.read_csv(self.neighbor_path)
        
        print(f"Loaded {len(df)} records with {df['Anomaly'].sum()} anomalies")
        print(f"Loaded {len(neighbors)} neighbor relationships")
        
        return df, neighbors
    
    def preprocess_data(self, df):
        """Add derived features"""
        df['hour'] = df['Date'].dt.hour
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([3, 4])
        
        # Add sector ID without suffix
        df['SectorBase'] = df['NE'].str[:-1]
        
        return df

class BaselineEstimator:
    """Establish baseline RSSI patterns and detect anomalies"""
    
    def __init__(self, window_days=7):
        self.window_days = window_days
        self.baselines = {}
        
    def establish_baseline(self, sector_data):
        """Create hourly baseline for each sector"""
        baseline = {}
        
        # Filter non-anomaly data for baseline
        normal_data = sector_data[sector_data['Anomaly'] == False].copy()
        
        if len(normal_data) < 24:  # Need at least one day of data
            return None
            
        for hour in range(24):
            hour_data = normal_data[normal_data['hour'] == hour]['RSSI_PUCCH(EUCell_Eric)(CRA)']
            
            if len(hour_data) > 0:
                baseline[hour] = {
                    'median': np.median(hour_data),
                    'mean': np.mean(hour_data),
                    'std': np.std(hour_data),
                    'mad': median_abs_deviation(hour_data),
                    'q25': np.percentile(hour_data, 25),
                    'q75': np.percentile(hour_data, 75),
                    'min': np.min(hour_data),
                    'max': np.max(hour_data)
                }
            else:
                baseline[hour] = None
                
        return baseline
    
    def detect_anomaly_enhanced(self, rssi_value, hour, baseline):
        """Enhanced anomaly detection"""
        if baseline is None or baseline.get(hour) is None:
            return rssi_value > -94.5, rssi_value + 110  # Simple threshold
            
        stats = baseline[hour]
        
        # Multiple criteria for anomaly detection
        criteria = []
        
        # 1. Absolute threshold
        criteria.append(rssi_value > -94.5)
        
        # 2. Statistical deviation (IQR method)
        iqr = stats['q75'] - stats['q25']
        upper_bound = stats['q75'] + 1.5 * iqr
        criteria.append(rssi_value > upper_bound)
        
        # 3. Z-score method
        if stats['std'] > 0:
            z_score = (rssi_value - stats['mean']) / stats['std']
            criteria.append(z_score > 3)
        
        # 4. MAD method (more robust)
        if stats['mad'] > 0:
            mad_score = (rssi_value - stats['median']) / stats['mad']
            criteria.append(mad_score > 3)
            
        # Anomaly if any criteria met
        is_anomaly = any(criteria)
        anomaly_score = rssi_value - stats['median'] if is_anomaly else 0
        
        return is_anomaly, anomaly_score

class NeighborAnalyzer:
    """Analyze neighbor correlations and find affected sectors"""
    
    def __init__(self, neighbor_df):
        self.neighbors = neighbor_df
        self.neighbor_graph = self._build_graph()
        
    def _build_graph(self):
        """Build neighbor graph for quick lookup"""
        graph = {}
        for _, row in self.neighbors.iterrows():
            if row['Sector'] not in graph:
                graph[row['Sector']] = []
            graph[row['Sector']].append({
                'neighbor': row['Neighbor'],
                'rank': row['rank'],
                'distance': row['distance'],
                'pathloss': row['pathloss'],
                'ring': row['ring']
            })
        return graph
    
    def find_changepoints_pelt(self, signal, penalty=3):
        """Detect change points using PELT algorithm"""
        if len(signal) < 5:
            return []
        
        try:
            # Normalize signal
            signal = np.array(signal).flatten()
            if np.std(signal) > 0:
                signal = (signal - np.mean(signal)) / np.std(signal)
            
            # PELT algorithm
            algo = rpt.Pelt(model="rbf", min_size=3).fit(signal)
            change_points = algo.predict(pen=penalty)
            return change_points[:-1]  # Remove last point (end of signal)
        except:
            return []
    
    def calculate_correlation_features(self, series1, series2):
        """Calculate multiple correlation features"""
        if len(series1) != len(series2) or len(series1) < 3:
            return None
            
        features = {}
        
        # Pearson correlation
        try:
            corr, p_value = pearsonr(series1, series2)
            features['pearson_corr'] = corr
            features['p_value'] = p_value
        except:
            features['pearson_corr'] = 0
            features['p_value'] = 1
        
        # Difference correlation
        diff1 = np.diff(series1)
        diff2 = np.diff(series2)
        if len(diff1) > 0:
            try:
                diff_corr, _ = pearsonr(diff1, diff2)
                features['diff_corr'] = diff_corr
            except:
                features['diff_corr'] = 0
        else:
            features['diff_corr'] = 0
            
        # Peak alignment
        peaks1, _ = find_peaks(series1, height=-94)
        peaks2, _ = find_peaks(series2, height=-94)
        
        changes1 = self.find_changepoints_pelt(series1)
        changes2 = self.find_changepoints_pelt(series2)
        
        if len(changes1) > 0 and len(changes2) > 0:
            # Check if change points occur at similar times
            change_overlap = len(set(changes1) & set(changes2)) / max(len(changes1), len(changes2))
            features['change_alignment'] = change_overlap
        else:
            features['change_alignment'] = 0

        if len(peaks1) > 0 and len(peaks2) > 0:
            # Check if peaks occur at similar times
            peak_overlap = len(set(peaks1) & set(peaks2)) / max(len(peaks1), len(peaks2))
            features['peak_alignment'] = peak_overlap
        else:
            features['peak_alignment'] = 0
            
        return features
    
    def find_correlated_neighbors(self, anomaly_sector, data_df, time_window, threshold=0.6, baseline=None):
        """Find neighbors with correlated RSSI patterns"""
        if anomaly_sector not in self.neighbor_graph:
            return []
            
        # Get anomaly sector data
        anomaly_data = data_df[
            (data_df['NE'] == anomaly_sector) & 
            (data_df['Date'] >= time_window[0]) & 
            (data_df['Date'] <= time_window[1])
        ].sort_values('Date')
        
        if len(anomaly_data) < 3:
            return []
            
        anomaly_rssi = anomaly_data['RSSI_PUCCH(EUCell_Eric)(CRA)'].values
        
        correlated = []
        
        # Check each neighbor
        for neighbor_info in self.neighbor_graph[anomaly_sector]:
            neighbor_id = neighbor_info['neighbor']
            
            # Get neighbor data
            neighbor_data = data_df[
                (data_df['NE'] == neighbor_id) & 
                (data_df['Date'] >= time_window[0]) & 
                (data_df['Date'] <= time_window[1])
            ].sort_values('Date')
            
            if len(neighbor_data) != len(anomaly_data):
                continue
                
            neighbor_rssi = neighbor_data['RSSI_PUCCH(EUCell_Eric)(CRA)'].values
            
            # Calculate correlation features
            corr_features = self.calculate_correlation_features(anomaly_rssi, neighbor_rssi)
            
            if corr_features is None:
                continue
                
            # Determine if correlated
            score = (
                corr_features['pearson_corr'] * 0.4 +
                corr_features['diff_corr'] * 0.2 +
                corr_features['peak_alignment'] * 0.3 + 
                corr_features['change_alignment'] * 0.5
            )
            
            if score > threshold or corr_features['pearson_corr'] > 0.7 or corr_features['change_alignment'] > 0.6:
                # Get sector info
                sector_info = neighbor_data.iloc[-1]
                
                # Calculate baseline median for this neighbor
                # Option 1: Use historical data before anomaly
                historical_data = data_df[
                    (data_df['NE'] == neighbor_id) & 
                    (data_df['Date'] < time_window[0]) &
                    (data_df['Anomaly'] == False)
                ]['RSSI_PUCCH(EUCell_Eric)(CRA)']
                
                if len(historical_data) > 0:
                    baseline_median = historical_data.median()
                else:
                    # Option 2: Use percentile of current data as baseline
                    baseline_median = np.percentile(neighbor_rssi, 25)  # Use 25th percentile
                
                # For capped values, estimate relative change
                max_rssi = np.max(neighbor_rssi)
                if max_rssi >= -93:  # Likely capped
                    # Use percentage of values at cap as intensity indicator
                    capped_ratio = np.sum(neighbor_rssi >= -93) / len(neighbor_rssi)
                    rssi_change = (max_rssi - baseline_median) * (1 + capped_ratio)
                else:
                    rssi_change = max_rssi - baseline_median
                
                correlated.append({
                    'sector': neighbor_id,
                    'correlation_score': score,
                    'pearson_corr': corr_features['pearson_corr'],
                    'distance': neighbor_info['distance'],
                    'pathloss': neighbor_info['pathloss'],
                    'rank': neighbor_info['rank'],
                    'rssi_change': rssi_change,
                    'max_rssi': max_rssi,
                    'baseline_median': baseline_median,  # Store for reference
                    'latitude': sector_info['Latitude'],
                    'longitude': sector_info['Longitude'],
                    'azimuth': sector_info['Azimuth'],
                    'etilt': sector_info['ETilt']
                })
        
        # Sort by correlation score
        correlated.sort(key=lambda x: x['correlation_score'], reverse=True)
        
        return correlated

class SourceLocalizer:
    """Localize interference source using multilateration and optimization"""
    
    def __init__(self, path_loss_exponent=3.5):
        self.n = path_loss_exponent  # Path loss exponent for urban environment
        self.freq_mhz = 2100  # 4G frequency
        
    def rssi_to_distance(self, rssi_dbm, tx_power_dbm=40):
        """Convert RSSI to estimated distance"""
        # Free space path loss: RSSI = Tx_power - 20*log10(f_MHz) - 20*log10(d_km) + 27.55
        # Simplified: d = 10^((Tx_power - RSSI - 20*log10(f) + 27.55) / (10*n))
        if rssi_dbm >= -93:  # Capped value detected
        # Assume linear distribution above cap
            estimated_actual = rssi_dbm + np.random.uniform(0, 10)  # -92 could be -92 to -72
            rssi_dbm = min(estimated_actual, -60)

        # Using simplified model
        if rssi_dbm >= -60:  # Very close
            return 50  # meters
        elif rssi_dbm <= -120:  # Very far
            return 5000  # meters
        
        # Log-distance path loss model
        reference_rssi = -60  # RSSI at 50m
        reference_distance = 50
        
        path_loss_db = reference_rssi - rssi_dbm
        distance = reference_distance * (10 ** (path_loss_db / (10 * self.n)))
        
        return min(max(distance, 10), 10000)  # Limit between 10m and 10km
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing from point 1 to point 2 in degrees"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        bearing = np.arctan2(x, y)
        bearing = np.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def antenna_gain_factor(self, source_bearing, sector_azimuth, beamwidth=65):
        """Calculate antenna gain based on angle difference"""
        angle_diff = abs(source_bearing - sector_azimuth)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        if angle_diff <= beamwidth/2:
            return 1.0
        elif angle_diff <= beamwidth:
            # Cosine rolloff
            return 0.5 * (1 + np.cos(np.pi * (angle_diff - beamwidth/2) / (beamwidth/2)))
        else:
            return 0.1  # Back/side lobe
    
    def objective_function(self, source_location, sectors_data):
        """Objective function for optimization"""
        source_lat, source_lon = source_location
        total_error = 0
        
        for sector in sectors_data:
            # Skip if missing critical data
            if pd.isna(sector.get('latitude')) or pd.isna(sector.get('longitude')):
                continue
                
            # Estimated distance from RSSI
            if 'max_rssi' in sector:
                estimated_dist = self.rssi_to_distance(sector['max_rssi'])
            else:
                estimated_dist = self.rssi_to_distance(sector.get('rssi_change', -100))
            
            # Actual distance
            actual_dist = self.haversine_distance(
                source_lat, source_lon,
                sector['latitude'], sector['longitude']
            )
            
            # Calculate bearing to source
            bearing = self.calculate_bearing(
                sector['latitude'], sector['longitude'],
                source_lat, source_lon
            )
            
            # Antenna pattern weight
            if not pd.isna(sector.get('azimuth')):
                ant_weight = self.antenna_gain_factor(bearing, sector['azimuth'])
            else:
                ant_weight = 1.0
            
            # Path loss weight (inverse - higher path loss = lower weight)
            pl_weight = 1.0 / (1 + sector.get('pathloss', 100) / 100)
            
            # Calculate weighted error
            distance_error = (actual_dist - estimated_dist) ** 2
            error = distance_error * ant_weight * pl_weight
            
            # Add RSSI consistency term
            if 'rssi_change' in sector:
                rssi_penalty = abs(sector['rssi_change']) / 100
                error *= (1 + rssi_penalty)
            
            total_error += error
            
        return total_error
    
    def localize(self, anomaly_sectors, correlated_neighbors, baselines=None):
        """Main localization algorithm"""
        # Combine all affected sectors
        all_sectors = []

        # Add anomaly sectors
        for _, sector in anomaly_sectors.iterrows():
            if baselines and sector['NE'] in baselines and sector['hour'] in baselines[sector['NE']]:
                baseline_val = baselines[sector['NE']][sector['hour']].get('median', -110)
            else:
                baseline_val = -110  # Default baseline value

            all_sectors.append({
                'latitude': sector['Latitude'],
                'longitude': sector['Longitude'],
                'azimuth': sector['Azimuth'],
                'rssi_change': sector['RSSI_PUCCH(EUCell_Eric)(CRA)'] - baseline_val,
                'max_rssi': sector['RSSI_PUCCH(EUCell_Eric)(CRA)'],
                'pathloss': 100,  # Default if not in neighbor data
                'weight': 1.5  # Higher weight for anomaly sectors
            })
        
        # Add correlated neighbors
        for neighbor in correlated_neighbors:
            all_sectors.append(neighbor)
            
        if len(all_sectors) < 2:
            # Need at least 2 sectors for localization
            return None
            
        # Initial estimate: weighted centroid
        weights = []
        lats = []
        lons = []
        
        for sector in all_sectors:
            if pd.isna(sector.get('latitude')) or pd.isna(sector.get('longitude')):
                continue
                
            # Weight based on RSSI change and correlation
            weight = abs(sector.get('rssi_change', 10)) * sector.get('weight', 1.0)
            weights.append(weight)
            lats.append(sector['latitude'])
            lons.append(sector['longitude'])
        
        if len(weights) == 0:
            return None
            
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        initial_lat = np.average(lats, weights=weights)
        initial_lon = np.average(lons, weights=weights)
        
        # Bounds for optimization (within reasonable range)
        lat_margin = 0.05  # ~5.5km
        lon_margin = 0.05
        
        bounds = [
            (min(lats) - lat_margin, max(lats) + lat_margin),
            (min(lons) - lon_margin, max(lons) + lon_margin)
        ]
        
        # Optimize
        try:
            result = minimize(
                self.objective_function,
                [initial_lat, initial_lon],
                args=(all_sectors,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                final_location = result.x
            else:
                final_location = [initial_lat, initial_lon]
        except:
            final_location = [initial_lat, initial_lon]
            
        # Calculate uncertainty
        distances = []
        for sector in all_sectors:
            if pd.isna(sector.get('latitude')) or pd.isna(sector.get('longitude')):
                continue
            dist = self.haversine_distance(
                final_location[0], final_location[1],
                sector['latitude'], sector['longitude']
            )
            distances.append(dist)
            
        uncertainty = np.std(distances) if len(distances) > 0 else 1000
        
        return {
            'latitude': final_location[0],
            'longitude': final_location[1],
            'initial_lat': initial_lat,
            'initial_lon': initial_lon,
            'uncertainty_meters': uncertainty,
            'num_sectors': len(all_sectors),
            'convergence': result.success if 'result' in locals() else False
        }

class TemporalPatternClassifier:
    """Classify temporal patterns of interference sources"""
    
    def __init__(self):
        self.patterns = {
            'intermittent_daily': 'Daily pattern (e.g., 4 hours/day)',
            'continuous_commercial': 'Continuous commercial (weeks)',
            'military': 'Military/Government (months)',
            'sporadic': 'Sporadic/Random'
        }
        
    def extract_temporal_features(self, anomaly_data):
        """Extract temporal features from anomaly occurrences"""
        features = {}
        
        if len(anomaly_data) < 2:
            return features
            
        # Sort by date
        anomaly_data = anomaly_data.sort_values('Date')
        
        # Duration analysis
        anomaly_periods = []
        current_period_start = None
        last_date = None
        
        for _, row in anomaly_data.iterrows():
            if last_date is None:
                current_period_start = row['Date']
            elif (row['Date'] - last_date).total_seconds() / 3600 > 2:  # Gap > 2 hours
                # End of period
                anomaly_periods.append({
                    'start': current_period_start,
                    'end': last_date,
                    'duration_hours': (last_date - current_period_start).total_seconds() / 3600
                })
                current_period_start = row['Date']
            
            last_date = row['Date']
            
        # Add last period
        if current_period_start is not None:
            anomaly_periods.append({
                'start': current_period_start,
                'end': last_date,
                'duration_hours': (last_date - current_period_start).total_seconds() / 3600
            })
        
        if len(anomaly_periods) > 0:
            durations = [p['duration_hours'] for p in anomaly_periods]
            features['avg_duration_hours'] = np.mean(durations)
            features['max_duration_hours'] = np.max(durations)
            features['num_periods'] = len(anomaly_periods)
            
            # Inter-arrival times
            if len(anomaly_periods) > 1:
                inter_arrivals = []
                for i in range(1, len(anomaly_periods)):
                    gap = (anomaly_periods[i]['start'] - anomaly_periods[i-1]['end']).total_seconds() / 3600
                    inter_arrivals.append(gap)
                features['avg_gap_hours'] = np.mean(inter_arrivals)
                features['gap_regularity'] = 1 / (1 + np.std(inter_arrivals)) if len(inter_arrivals) > 1 else 0
        
        # Hour of day distribution
        hours = anomaly_data['Date'].dt.hour.value_counts()
        features['active_hours'] = len(hours)
        features['peak_hour'] = hours.idxmax() if len(hours) > 0 else -1
        
        # Day pattern
        features['unique_days'] = anomaly_data['Date'].dt.date.nunique()
        features['span_days'] = (anomaly_data['Date'].max() - anomaly_data['Date'].min()).days
        
        return features
    
    def classify_pattern(self, features):
        """Classify the temporal pattern based on features"""
        if not features:
            return 'sporadic', 0.3
            
        # Decision tree for classification
        avg_duration = features.get('avg_duration_hours', 0)
        active_hours = features.get('active_hours', 0)
        gap_regularity = features.get('gap_regularity', 0)
        
        confidence = 0.5
        
        if avg_duration < 6 and active_hours < 8 and gap_regularity > 0.5:
            pattern = 'intermittent_daily'
            confidence = 0.8 + gap_regularity * 0.2
        elif avg_duration > 24 * 30:  # More than a month
            pattern = 'military'
            confidence = 0.85
        elif avg_duration > 24 * 7:  # More than a week
            pattern = 'continuous_commercial'
            confidence = 0.75
        else:
            pattern = 'sporadic'
            confidence = 0.4
            
        return pattern, confidence

class AnomalyLocalizationPipeline:
    """Main pipeline for anomaly source localization"""
    
    def __init__(self, output_dir='anomaly_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.baseline_estimator = BaselineEstimator()
        self.neighbor_analyzer = None
        self.source_localizer = SourceLocalizer()
        self.pattern_classifier = TemporalPatternClassifier()
        
        # Results storage
        self.results = []
        self.baselines = {}
        
    def process_data(self, data_df, neighbor_df):
        """Main processing pipeline"""
        print("Starting Anomaly Localization Pipeline...")
        
        # Initialize neighbor analyzer
        self.neighbor_analyzer = NeighborAnalyzer(neighbor_df)
        
        # 1. Establish baselines for all sectors
        print("\n1. Establishing baselines for all sectors...")
        unique_sectors = data_df['NE'].unique()
        
        for sector in unique_sectors:
            sector_data = data_df[data_df['NE'] == sector]
            baseline = self.baseline_estimator.establish_baseline(sector_data)
            if baseline:
                self.baselines[sector] = baseline
        
        print(f"   Baselines established for {len(self.baselines)} sectors")
        
        # 2. Group anomalies by time and location
        print("\n2. Grouping anomaly events...")
        anomaly_data = data_df[data_df['Anomaly'] == True].copy()
        
        # Group anomalies that occur close in time and space
        anomaly_groups = self._group_anomalies(anomaly_data)
        print(f"   Found {len(anomaly_groups)} anomaly groups")
        
        # 3. Process each anomaly group
        print("\n3. Processing anomaly groups...")
        
        for idx, group in enumerate(anomaly_groups):
            print(f"\n   Processing group {idx+1}/{len(anomaly_groups)}...")
            
            # Find correlated neighbors
            correlated_neighbors = []
            for _, anomaly_sector in group['sectors'].iterrows():
                time_window = (
                    group['start_time'] - timedelta(hours=72),
                    group['end_time']
                )
                
                neighbors = self.neighbor_analyzer.find_correlated_neighbors(
                    anomaly_sector['NE'],
                    data_df,
                    time_window
                )
                correlated_neighbors.extend(neighbors)
            
            # Remove duplicates
            seen = set()
            unique_neighbors = []
            for n in correlated_neighbors:
                if n['sector'] not in seen:
                    seen.add(n['sector'])
                    unique_neighbors.append(n)
            
            # Localize source
            location = self.source_localizer.localize(group['sectors'], unique_neighbors, self.baselines)

            print(f"      Found {len(unique_neighbors)} correlated neighbors")
            
            if location:
                # Extract temporal features
                group_anomaly_data = data_df[
                    (data_df['NE'].isin(group['sectors']['NE'])) &
                    (data_df['Date'] >= group['start_time']) &
                    (data_df['Date'] <= group['end_time']) &
                    (data_df['Anomaly'] == True)
                ]
                
                temporal_features = self.pattern_classifier.extract_temporal_features(group_anomaly_data)
                pattern, pattern_confidence = self.pattern_classifier.classify_pattern(temporal_features)
                
                # Store results
                result = {
                    'group_id': idx,
                    'start_time': group['start_time'].isoformat(),
                    'end_time': group['end_time'].isoformat(),
                    'duration_hours': group['duration_hours'],
                    'num_anomaly_sectors': len(group['sectors']),
                    'num_correlated_neighbors': len(unique_neighbors),
                    'source_location': location,
                    'temporal_pattern': pattern,
                    'pattern_confidence': pattern_confidence,
                    'temporal_features': temporal_features,
                    'anomaly_sectors': group['sectors'][['NE', 'Latitude', 'Longitude', 'RSSI_PUCCH(EUCell_Eric)(CRA)']].to_dict('records'),
                    'correlated_neighbors': unique_neighbors[:10]  # Top 10
                }
                
                self.results.append(result)
                print(f"      Source localized at: ({location['latitude']:.6f}, {location['longitude']:.6f})")
                print(f"      Pattern: {pattern} (confidence: {pattern_confidence:.2f})")
        
        # 4. Save results
        self._save_results()
        
        return self.results
    
    def _group_anomalies(self, anomaly_data, time_threshold_hours=6, distance_threshold_km=5):
        """Group anomalies that occur close in time and space"""
        if len(anomaly_data) == 0:
            return []
            
        anomaly_data = anomaly_data.sort_values('Date')
        
        groups = []
        current_group = None
        
        for _, row in anomaly_data.iterrows():
            if current_group is None:
                # Start new group
                current_group = {
                    'sectors': pd.DataFrame([row]),
                    'start_time': row['Date'],
                    'end_time': row['Date'],
                    'center_lat': row['Latitude'],
                    'center_lon': row['Longitude']
                }
            else:
                # Check if this anomaly belongs to current group
                time_diff_hours = (row['Date'] - current_group['end_time']).total_seconds() / 3600
                
                distance_km = self.source_localizer.haversine_distance(
                    row['Latitude'], row['Longitude'],
                    current_group['center_lat'], current_group['center_lon']
                ) / 1000
                
                if time_diff_hours <= time_threshold_hours and distance_km <= distance_threshold_km:
                    # Add to current group
                    current_group['sectors'] = pd.concat([current_group['sectors'], pd.DataFrame([row])], ignore_index=True)
                    current_group['end_time'] = row['Date']
                    
                    # Update center
                    current_group['center_lat'] = current_group['sectors']['Latitude'].mean()
                    current_group['center_lon'] = current_group['sectors']['Longitude'].mean()
                else:
                    # Save current group and start new one
                    current_group['duration_hours'] = (
                        current_group['end_time'] - current_group['start_time']
                    ).total_seconds() / 3600
                    groups.append(current_group)
                    
                    current_group = {
                        'sectors': pd.DataFrame([row]),
                        'start_time': row['Date'],
                        'end_time': row['Date'],
                        'center_lat': row['Latitude'],
                        'center_lon': row['Longitude']
                    }
        
        # Add last group
        if current_group is not None:
            current_group['duration_hours'] = (
                current_group['end_time'] - current_group['start_time']
            ).total_seconds() / 3600
            groups.append(current_group)
        
        return groups
    
    def _save_results(self):
        """Save all results to files"""
        # Save as JSON
        json_path = self.output_dir / 'localization_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n   Results saved to {json_path}")
        
        # Save as CSV for easy analysis
        if self.results:
            flat_results = []
            for r in self.results:
                flat_result = {
                    'group_id': r['group_id'],
                    'start_time': r['start_time'],
                    'end_time': r['end_time'],
                    'duration_hours': r['duration_hours'],
                    'num_anomaly_sectors': r['num_anomaly_sectors'],
                    'num_correlated_neighbors': r['num_correlated_neighbors'],
                    'source_lat': r['source_location']['latitude'],
                    'source_lon': r['source_location']['longitude'],
                    'uncertainty_meters': r['source_location']['uncertainty_meters'],
                    'temporal_pattern': r['temporal_pattern'],
                    'pattern_confidence': r['pattern_confidence']
                }
                flat_results.append(flat_result)
            
            csv_df = pd.DataFrame(flat_results)
            csv_path = self.output_dir / 'localization_summary.csv'
            csv_df.to_csv(csv_path, index=False)
            print(f"   Summary saved to {csv_path}")
        
        # Save baselines
        baseline_path = self.output_dir / 'sector_baselines.pkl'
        with open(baseline_path, 'wb') as f:
            pickle.dump(self.baselines, f)
        print(f"   Baselines saved to {baseline_path}")

class Visualizer:
    """Create comprehensive visualizations"""
    
    def __init__(self, output_dir='anomaly_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_all_visualizations(self, data_df, results, neighbor_df):
        """Create all visualizations"""
        print("\n4. Creating visualizations...")
        
        # 1. Interactive map
        self.create_interactive_map(data_df, results)
        
        # 2. RSSI analysis plots
        self.create_rssi_plots(data_df)
        
        # 3. Temporal pattern analysis
        self.create_temporal_plots(data_df)
        
        # 4. Source localization accuracy
        self.create_localization_plots(results)
        
        # 5. Network topology visualization
        self.create_network_topology(data_df, neighbor_df, results)
        
        print("   All visualizations created!")
        

    def create_interactive_map(self, data_df, results):
        """Create interactive Folium map"""
        # Calculate map center
        center_lat = data_df['Latitude'].mean()
        center_lon = data_df['Longitude'].mean()
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            control_scale=True
        )
                
        # Add tile layers (built-in) — NO attribution argument
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name='Stamen Terrain',
            overlay=True,
            control=True
        ).add_to(m)

        # Add CartoDB Dark Matter properly
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='CartoDB dark_matter',
            overlay=True,
            control=True
        ).add_to(m)


        folium.LayerControl().add_to(m)
        
        # 1. Add sectors layer
        # ... rest of your code

        # 1. Add sectors layer
        sector_group = folium.FeatureGroup(name='Sectors')
        
        for _, sector in data_df.drop_duplicates('NE').iterrows():
            color = 'red' if sector['Anomaly'] else 'green'
            
            # Create sector marker
            folium.CircleMarker(
                location=[sector['Latitude'], sector['Longitude']],
                radius=5,
                popup=folium.Popup(
                    f"<b>Sector:</b> {sector['NE']}<br>"
                    f"<b>Site:</b> {sector['Site']}<br>"
                    f"<b>Azimuth:</b> {sector['Azimuth']}°<br>"
                    f"<b>Height:</b> {sector['AntennaHeight']}m",
                    max_width=200
                ),
                tooltip=f"Sector: {sector['NE']}",
                color=color,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(sector_group)
            
            # Add azimuth line
            if not pd.isna(sector['Azimuth']):
                end_point = self._calculate_endpoint(
                    sector['Latitude'], 
                    sector['Longitude'],
                    sector['Azimuth'], 
                    500  # 500m line
                )
                
                folium.PolyLine(
                    locations=[
                        [sector['Latitude'], sector['Longitude']],
                        end_point
                    ],
                    color=color,
                    weight=1,
                    opacity=0.5
                ).add_to(sector_group)
        
        sector_group.add_to(m)
        
        # 2. Add anomaly sources layer
        source_group = folium.FeatureGroup(name='Detected Sources')
        
        for result in results:
            if result['source_location']:
                loc = result['source_location']
                
                # Source marker
                folium.Marker(
                    location=[loc['latitude'], loc['longitude']],
                    popup=folium.Popup(
                        f"<b>Source ID:</b> {result['group_id']}<br>"
                        f"<b>Pattern:</b> {result['temporal_pattern']}<br>"
                        f"<b>Confidence:</b> {result['pattern_confidence']:.2%}<br>"
                        f"<b>Uncertainty:</b> {loc['uncertainty_meters']:.0f}m<br>"
                        f"<b>Duration:</b> {result['duration_hours']:.1f}h<br>"
                        f"<b>Affected Sectors:</b> {result['num_anomaly_sectors']}",
                        max_width=250
                    ),
                    tooltip=f"Source {result['group_id']}: {result['temporal_pattern']}",
                    icon=folium.Icon(color='orange', icon='warning', prefix='fa')
                ).add_to(source_group)
                
                # Uncertainty circle
                folium.Circle(
                    location=[loc['latitude'], loc['longitude']],
                    radius=loc['uncertainty_meters'],
                    color='orange',
                    fill=True,
                    fillOpacity=0.2,
                    weight=2,
                    popup=f"Uncertainty: {loc['uncertainty_meters']:.0f}m"
                ).add_to(source_group)
                
                # Draw lines to affected sectors
                for sector in result['anomaly_sectors']:
                    folium.PolyLine(
                        locations=[
                            [loc['latitude'], loc['longitude']],
                            [sector['Latitude'], sector['Longitude']]
                        ],
                        color='red',
                        weight=1,
                        opacity=0.3,
                        dash_array='5'
                    ).add_to(source_group)
        
        source_group.add_to(m)
        
        # 3. Add heatmap of anomalies
        anomaly_data = data_df[data_df['Anomaly'] == True]
        if len(anomaly_data) > 0:
            heat_data = anomaly_data[['Latitude', 'Longitude', 'RSSI_PUCCH(EUCell_Eric)(CRA)']].values.tolist()
            
            # Normalize RSSI for heatmap intensity
            for point in heat_data:
                point[2] = (point[2] + 120) / 60  # Normalize to 0-1
            
            HeatMap(
                heat_data,
                name='Anomaly Heatmap',
                min_opacity=0.3,
                max_zoom=18,
                radius=25,
                blur=15,
                gradient={0.4: 'blue', 0.65: 'yellow', 0.85: 'orange', 1: 'red'}
            ).add_to(m)
        
        # Add minimap
        minimap = plugins.MiniMap(tile_layer='CartoDB dark_matter')
        m.add_child(minimap)
        
        # Add measurement tool
        plugins.MeasureControl(position='topright').add_to(m)
        
        # Save map
        map_path = self.output_dir / 'anomaly_map.html'
        m.save(str(map_path))
        print(f"   Interactive map saved to {map_path}")
    
    def _calculate_endpoint(self, lat, lon, bearing, distance_m):
        """Calculate endpoint given start point, bearing, and distance"""
        R = 6371000  # Earth radius in meters
        
        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        bearing_rad = np.radians(bearing)
        
        lat2 = np.arcsin(
            np.sin(lat1) * np.cos(distance_m/R) +
            np.cos(lat1) * np.sin(distance_m/R) * np.cos(bearing_rad)
        )
        
        lon2 = lon1 + np.arctan2(
            np.sin(bearing_rad) * np.sin(distance_m/R) * np.cos(lat1),
            np.cos(distance_m/R) - np.sin(lat1) * np.sin(lat2)
        )
        
        return [np.degrees(lat2), np.degrees(lon2)]
    
    def create_rssi_plots(self, data_df):
        """Create RSSI analysis plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'RSSI Distribution (Normal vs Anomaly)',
                'RSSI Time Series by Sector',
                'RSSI Hourly Pattern',
                'RSSI Correlation Matrix'
            ],
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # 1. RSSI Distribution
        normal_rssi = data_df[data_df['Anomaly'] == False]['RSSI_PUCCH(EUCell_Eric)(CRA)']
        anomaly_rssi = data_df[data_df['Anomaly'] == True]['RSSI_PUCCH(EUCell_Eric)(CRA)']
        
        fig.add_trace(
            go.Histogram(x=normal_rssi, name='Normal', opacity=0.6, marker_color='green'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=anomaly_rssi, name='Anomaly', opacity=0.6, marker_color='red'),
            row=1, col=1
        )
        
        # 2. Time Series
        sample_sectors = data_df['NE'].unique()[:5]  # First 5 sectors
        for sector in sample_sectors:
            sector_data = data_df[data_df['NE'] == sector].sort_values('Date')
            fig.add_trace(
                go.Scatter(
                    x=sector_data['Date'],
                    y=sector_data['RSSI_PUCCH(EUCell_Eric)(CRA)'],
                    mode='lines',
                    name=sector,
                    line=dict(width=1)
                ),
                row=1, col=2
            )
        
        # 3. Hourly Pattern
        hourly_avg = data_df.groupby(['hour', 'Anomaly'])['RSSI_PUCCH(EUCell_Eric)(CRA)'].mean().reset_index()
        
        for anomaly_status in [False, True]:
            data = hourly_avg[hourly_avg['Anomaly'] == anomaly_status]
            fig.add_trace(
                go.Scatter(
                    x=data['hour'],
                    y=data['RSSI_PUCCH(EUCell_Eric)(CRA)'],
                    mode='lines+markers',
                    name='Anomaly' if anomaly_status else 'Normal',
                    marker=dict(color='red' if anomaly_status else 'green')
                ),
                row=2, col=1
            )
        
        # 4. Correlation Matrix
        # Pivot data for correlation
        pivot_data = data_df.pivot_table(
            index='Date',
            columns='NE',
            values='RSSI_PUCCH(EUCell_Eric)(CRA)',
            aggfunc='mean'
        )
        
        if len(pivot_data.columns) > 1:
            corr_matrix = pivot_data.corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns[:10],  # Limit to 10 sectors for readability
                    y=corr_matrix.index[:10],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values[:10, :10], 2),
                    texttemplate='%{text}',
                    textfont={"size": 8}
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="RSSI Analysis Dashboard",
            title_font_size=20
        )
        
        # Update axes
        fig.update_xaxes(title_text="RSSI (dBm)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="RSSI (dBm)", row=1, col=2)
        fig.update_yaxes(title_text="RSSI (dBm)", row=2, col=1)
        
        # Save
        rssi_plot_path = self.output_dir / 'rssi_analysis.html'
        fig.write_html(str(rssi_plot_path))
        print(f"   RSSI analysis saved to {rssi_plot_path}")
    
    def create_temporal_plots(self, data_df):
        """Create temporal pattern analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Anomaly occurrences over time
        ax1 = axes[0, 0]
        anomaly_counts = data_df[data_df['Anomaly'] == True].groupby(
            data_df['Date'].dt.date
        ).size()
        
        if len(anomaly_counts) > 0:
            ax1.bar(anomaly_counts.index, anomaly_counts.values, color='red', alpha=0.7)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Anomaly Count')
            ax1.set_title('Daily Anomaly Occurrences')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Hour of day distribution
        ax2 = axes[0, 1]
        hourly_anomalies = data_df[data_df['Anomaly'] == True]['hour'].value_counts().sort_index()
        
        if len(hourly_anomalies) > 0:
            ax2.bar(hourly_anomalies.index, hourly_anomalies.values, color='orange', alpha=0.7)
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Anomaly Count')
            ax2.set_title('Anomaly Distribution by Hour')
            ax2.set_xticks(range(0, 24, 2))
        
        # 3. Weekday vs Weekend
        ax3 = axes[1, 0]
        weekend_data = data_df.groupby(['is_weekend', 'Anomaly']).size().unstack(fill_value=0)
        
        if len(weekend_data) > 0:
            weekend_data.plot(kind='bar', ax=ax3, color=['green', 'red'], alpha=0.7)
            ax3.set_xlabel('Weekend')
            ax3.set_ylabel('Count')
            ax3.set_title('Anomaly Distribution: Weekday vs Weekend')
            ax3.set_xticklabels(['Weekday', 'Weekend'], rotation=0)
            ax3.legend(['Normal', 'Anomaly'])
        
        # 4. Sector anomaly frequency
        ax4 = axes[1, 1]
        sector_anomalies = data_df[data_df['Anomaly'] == True]['NE'].value_counts().head(10)
        
        if len(sector_anomalies) > 0:
            ax4.barh(range(len(sector_anomalies)), sector_anomalies.values, color='purple', alpha=0.7)
            ax4.set_yticks(range(len(sector_anomalies)))
            ax4.set_yticklabels(sector_anomalies.index)
            ax4.set_xlabel('Anomaly Count')
            ax4.set_title('Top 10 Sectors by Anomaly Frequency')
        
        plt.suptitle('Temporal Pattern Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save
        temporal_plot_path = self.output_dir / 'temporal_analysis.png'
        plt.savefig(temporal_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Temporal analysis saved to {temporal_plot_path}")
    
    def create_localization_plots(self, results):
        """Create source localization accuracy plots"""
        if not results:
            print("   No localization results to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data from results
        uncertainties = [r['source_location']['uncertainty_meters'] for r in results if r['source_location']]
        num_sectors = [r['num_anomaly_sectors'] + r['num_correlated_neighbors'] for r in results]
        patterns = [r['temporal_pattern'] for r in results]
        confidences = [r['pattern_confidence'] for r in results]
        
        # 1. Uncertainty distribution
        ax1 = axes[0, 0]
        if uncertainties:
            ax1.hist(uncertainties, bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax1.axvline(np.mean(uncertainties), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(uncertainties):.0f}m')
            ax1.set_xlabel('Uncertainty (meters)')
            ax1.set_ylabel('Count')
            ax1.set_title('Localization Uncertainty Distribution')
            ax1.legend()
        
        # 2. Uncertainty vs Number of Sectors
        ax2 = axes[0, 1]
        if uncertainties and num_sectors:
            ax2.scatter(num_sectors, uncertainties, alpha=0.6, s=50)
            ax2.set_xlabel('Number of Affected Sectors')
            ax2.set_ylabel('Uncertainty (meters)')
            ax2.set_title('Uncertainty vs Sector Count')
            
            # Add trend line
            z = np.polyfit(num_sectors, uncertainties, 1)
            p = np.poly1d(z)
            ax2.plot(sorted(num_sectors), p(sorted(num_sectors)), 
                    "r--", alpha=0.5, label=f'Trend')
            ax2.legend()
        
        # 3. Pattern distribution
        ax3 = axes[1, 0]
        if patterns:
            pattern_counts = pd.Series(patterns).value_counts()
            ax3.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%',
                   colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
            ax3.set_title('Temporal Pattern Distribution')
        
        # 4. Confidence scores
        ax4 = axes[1, 1]
        if confidences:
            ax4.hist(confidences, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(confidences), color='red', linestyle='--',
                       label=f'Mean: {np.mean(confidences):.2f}')
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Count')
            ax4.set_title('Pattern Classification Confidence')
            ax4.legend()
        
        plt.suptitle('Source Localization Performance', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save
        localization_plot_path = self.output_dir / 'localization_analysis.png'
        plt.savefig(localization_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Localization analysis saved to {localization_plot_path}")
    
    def create_network_topology(self, data_df, neighbor_df, results):
        """Create network topology visualization"""
        # Create interactive network graph
        fig = go.Figure()
        
        # Get unique sectors
        sectors = data_df.drop_duplicates('NE')
        
        # Add sector nodes
        for _, sector in sectors.iterrows():
            color = 'red' if sector['Anomaly'] else 'lightgreen'
            
            fig.add_trace(go.Scattermapbox(
                mode='markers+text',
                lon=[sector['Longitude']],
                lat=[sector['Latitude']],
                marker={'size': 10, 'color': color},
                text=sector['NE'],
                textposition='top center',
                name=sector['NE'],
                showlegend=False,
                hovertemplate=f"<b>{sector['NE']}</b><br>" +
                             f"Site: {sector['Site']}<br>" +
                             f"Azimuth: {sector['Azimuth']}°<br>" +
                             f"<extra></extra>"
            ))
        
        # Add neighbor connections (sample for performance)
        sample_neighbors = neighbor_df[neighbor_df['rank'] <= 3].sample(
            min(500, len(neighbor_df[neighbor_df['rank'] <= 3]))
        )
        
        for _, neighbor in sample_neighbors.iterrows():
            # Get coordinates
            sector_data = sectors[sectors['NE'] == neighbor['Sector']]
            neighbor_data = sectors[sectors['NE'] == neighbor['Neighbor']]
            
            if len(sector_data) > 0 and len(neighbor_data) > 0:
                fig.add_trace(go.Scattermapbox(
                    mode='lines',
                    lon=[sector_data.iloc[0]['Longitude'], neighbor_data.iloc[0]['Longitude']],
                    lat=[sector_data.iloc[0]['Latitude'], neighbor_data.iloc[0]['Latitude']],
                    line={'width': 1, 'color': 'blue'},
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add detected sources
        for result in results:
            if result['source_location']:
                loc = result['source_location']
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lon=[loc['longitude']],
                    lat=[loc['latitude']],
                    marker={'size': 15, 'color': 'orange', 'symbol': 'star'},
                    text=f"Source {result['group_id']}",
                    name=f"Source {result['group_id']}",
                    showlegend=True,
                    hovertemplate=f"<b>Source {result['group_id']}</b><br>" +
                                 f"Pattern: {result['temporal_pattern']}<br>" +
                                 f"Uncertainty: {loc['uncertainty_meters']:.0f}m<br>" +
                                 f"<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=sectors['Latitude'].mean(),
                    lon=sectors['Longitude'].mean()
                ),
                zoom=11
            ),
            height=800,
            title="Network Topology and Detected Sources",
            showlegend=True
        )
        
        # Save
        topology_path = self.output_dir / 'network_topology.html'
        fig.write_html(str(topology_path))
        print(f"   Network topology saved to {topology_path}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("TELECOM ANOMALY SOURCE LOCALIZATION SYSTEM")
    print("=" * 80)
    
    # File paths
    data_file = 'telecom_data.csv'  # Your main data file
    neighbor_file = 'neighbor_data.csv'  # Your neighbor data file
    
    try:
        # Load data
        loader = DataLoader(data_file, neighbor_file)
        data_df, neighbor_df = loader.load_data()
        data_df = loader.preprocess_data(data_df)
        
        # Run pipeline
        pipeline = AnomalyLocalizationPipeline()
        results = pipeline.process_data(data_df, neighbor_df)
        
        # Create visualizations
        visualizer = Visualizer()
        visualizer.create_all_visualizations(data_df, results, neighbor_df)
        
        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total anomaly groups processed: {len(results)}")
        
        if results:
            avg_uncertainty = np.mean([r['source_location']['uncertainty_meters'] 
                                      for r in results if r['source_location']])
            print(f"Average localization uncertainty: {avg_uncertainty:.0f} meters")
            
            pattern_counts = pd.Series([r['temporal_pattern'] for r in results]).value_counts()
            print("\nDetected patterns:")
            for pattern, count in pattern_counts.items():
                print(f"  - {pattern}: {count} sources")
        
        print(f"\nResults saved in: anomaly_results/")
        print("\nVisualization files:")
        print("  - anomaly_map.html (Interactive map)")
        print("  - rssi_analysis.html (RSSI analysis)")
        print("  - temporal_analysis.png (Temporal patterns)")
        print("  - localization_analysis.png (Localization accuracy)")
        print("  - network_topology.html (Network visualization)")
        
    except FileNotFoundError as e:
        print(f"\nError: Could not find data files. Please ensure these files exist:")
        print(f"  - {data_file}")
        print(f"  - {neighbor_file}")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()