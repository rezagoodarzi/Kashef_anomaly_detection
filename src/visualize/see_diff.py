def find_correlated_neighbors(self, anomaly_sector, data_by_sector, time_window, threshold=0.6, baseline=None):
    """OPTIMIZED: Find neighbors with correlated RSSI patterns"""
    if anomaly_sector not in self.neighbor_graph:
        return []
    
    # Get anomaly sector data from pre-filtered dict
    if anomaly_sector not in data_by_sector:
        return []
    
    anomaly_data = data_by_sector[anomaly_sector]
    anomaly_data = anomaly_data[
        (anomaly_data['Date'] >= time_window[0]) & 
        (anomaly_data['Date'] <= time_window[1])
    ]
    
    if len(anomaly_data) < 3:
        return []
    
    anomaly_rssi = anomaly_data['RSSI_PUCCH(EUCell_Eric)(CRA)'].values
    anomaly_dates = anomaly_data['Date'].values
    
    correlated = []
    
    # Check each neighbor
    for neighbor_info in self.neighbor_graph[anomaly_sector]:
        neighbor_id = neighbor_info['neighbor']
        
        # Get neighbor data from pre-filtered dict
        if neighbor_id not in data_by_sector:
            continue
        
        neighbor_data = data_by_sector[neighbor_id]
        neighbor_data = neighbor_data[
            (neighbor_data['Date'] >= time_window[0]) & 
            (neighbor_data['Date'] <= time_window[1])
        ]
        
        # Align timestamps using merge instead of checking length
        if len(neighbor_data) < 3:
            continue
        
        # Merge on Date to align data points
        merged = pd.merge(
            anomaly_data[['Date', 'RSSI_PUCCH(EUCell_Eric)(CRA)']],
            neighbor_data[['Date', 'RSSI_PUCCH(EUCell_Eric)(CRA)', 'Latitude', 'Longitude', 'Azimuth', 'ETilt']],
            on='Date',
            suffixes=('_anomaly', '_neighbor')
        )
        
        if len(merged) < 3:
            continue
        
        neighbor_rssi = merged['RSSI_PUCCH(EUCell_Eric)(CRA)_neighbor'].values
        
        # Calculate correlation features
        corr_features = self.calculate_correlation_features(
            merged['RSSI_PUCCH(EUCell_Eric)(CRA)_anomaly'].values,
            neighbor_rssi
        )
        
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
            sector_info = merged.iloc[-1]
            
            # Calculate baseline median
            if baseline and neighbor_id in baseline:
                historical_median = np.median([baseline[neighbor_id][h].get('median', -110) 
                                              for h in baseline[neighbor_id] if baseline[neighbor_id][h]])
            else:
                historical_median = np.percentile(neighbor_rssi, 25)
            
            max_rssi = np.max(neighbor_rssi)
            
            # Handle capped values
            if max_rssi >= -93:
                capped_ratio = np.sum(neighbor_rssi >= -93) / len(neighbor_rssi)
                rssi_change = (max_rssi - historical_median) * (1 + capped_ratio)
            else:
                rssi_change = max_rssi - historical_median
            
            correlated.append({
                'sector': neighbor_id,
                'correlation_score': score,
                'pearson_corr': corr_features['pearson_corr'],
                'distance': neighbor_info['distance'],
                'pathloss': neighbor_info['pathloss'],
                'rank': neighbor_info['rank'],
                'rssi_change': rssi_change,
                'max_rssi': max_rssi,
                'baseline_median': historical_median,
                'latitude': sector_info['Latitude'],
                'longitude': sector_info['Longitude'],
                'azimuth': sector_info['Azimuth'],
                'etilt': sector_info['ETilt'],
                'weight': 1.0
            })
    
    correlated.sort(key=lambda x: x['correlation_score'], reverse=True)
    return correlated