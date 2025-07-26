import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw
import pelt_copy
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

class PathLossWeightedCoehaviorAnalyzer:
    def __init__(self, path_loss_threshold=130, similarity_threshold=0.7, 
                 causality_threshold=0.05, min_cluster_size=2):
        """
        Initialize the analyzer with configurable thresholds
        
        Parameters:
        - path_loss_threshold: Maximum path loss (dB) to consider sectors as connected
        - similarity_threshold: Minimum similarity score for clustering
        - causality_threshold: P-value threshold for Granger causality
        - min_cluster_size: Minimum sectors in a cluster
        """
        self.path_loss_threshold = path_loss_threshold
        self.similarity_threshold = similarity_threshold
        self.causality_threshold = causality_threshold
        self.min_cluster_size = min_cluster_size
        
    def calculate_influence_weights(self, path_loss_matrix):
        """
        Convert path loss to influence weights
        Lower path loss = higher influence
        """
        valid_connections = path_loss_matrix <= self.path_loss_threshold
        
        influence_weights = np.zeros_like(path_loss_matrix)
        
        mask = valid_connections & (path_loss_matrix > 0)
        influence_weights[mask] = np.exp(-path_loss_matrix[mask] / 40)  # 40dB decay constant
        
        np.fill_diagonal(influence_weights, 0)
        
        return influence_weights, valid_connections
    
    def calculate_rtwp_changes(self, rtwp_timeseries):
        """
        Calculate various change metrics for RTWP time series
        """
        hourly_changes = np.diff(rtwp_timeseries)
        
        acceleration = np.diff(hourly_changes)
        
        volatility = pd.Series(rtwp_timeseries).rolling(window=6).std().fillna(0).values
        
        daily_mean = np.mean(rtwp_timeseries)
        cumulative_deviation = np.cumsum(rtwp_timeseries - daily_mean)
        
        return {
            'hourly_changes': hourly_changes,
            'acceleration': acceleration,
            'volatility': volatility,
            'cumulative_deviation': cumulative_deviation,
            'daily_mean': daily_mean
        }
    
    def calculate_weighted_similarity(self, sector_data, neighbor_data, influence_weight):
        """
        Calculate weighted similarity between sector and neighbor RTWP patterns
        """
        min_len = min(len(sector_data), len(neighbor_data))
        sector_data = sector_data[:min_len]
        neighbor_data = neighbor_data[:min_len]
        
        similarities = {}
        
        if len(sector_data) > 3:
            pearson_corr, p_value = stats.pearsonr(sector_data, neighbor_data)
            similarities['pearson'] = abs(pearson_corr) if not np.isnan(pearson_corr) else 0
        else:
            similarities['pearson'] = 0
        
        if len(sector_data) > 3:
            spearman_corr, _ = stats.spearmanr(sector_data, neighbor_data)
            similarities['spearman'] = abs(spearman_corr) if not np.isnan(spearman_corr) else 0
        else:
            similarities['spearman'] = 0
        
        try:
            dtw_distance = dtw.distance(sector_data, neighbor_data)
            max_possible_dtw = len(sector_data) * max(np.std(sector_data), np.std(neighbor_data))
            dtw_similarity = 1 - min(dtw_distance / max_possible_dtw, 1) if max_possible_dtw > 0 else 0
            similarities['dtw'] = dtw_similarity
        except:
            similarities['dtw'] = 0
        
        sector_changes = np.diff(sector_data)
        neighbor_changes = np.diff(neighbor_data)
        #print(neighbor_changes)
        if len(sector_changes) > 1:
            change_corr, _ = stats.pearsonr(sector_changes, neighbor_changes)
            similarities['change_pattern'] = abs(change_corr) if not np.isnan(change_corr) else 0
        else:
            similarities['change_pattern'] = 0
        
        # 5. Synchronization of extreme events
        sector_extremes = np.abs(sector_data - np.mean(sector_data)) > 2 * np.std(sector_data)
        neighbor_extremes = np.abs(neighbor_data - np.mean(neighbor_data)) > 2 * np.std(neighbor_data)
        if np.any(sector_extremes) or np.any(neighbor_extremes):
            sync_score = np.mean(sector_extremes == neighbor_extremes)
            similarities['synchronization'] = sync_score
        else:
            similarities['synchronization'] = 0
        
        weights = {
            'pearson': 0.4,
            'spearman': 0.01,
            'dtw': 0.05,
            'change_pattern': 0.8,
            'synchronization': 0.04
        }
        
        combined_similarity = sum(similarities[key] * weights[key] for key in weights)
        
        # influence_weight temporary deleted 
        # weighted_similarity = combined_similarity * influence_weight

        weighted_similarity = combined_similarity 
        
        return weighted_similarity, similarities
    
    def detect_change_points(self, rtwp_data):
        """
        Detect change points in RTWP time series
        """
        try:
            algo = pelt_copy.Pelt(model="rbf").fit(rtwp_data)
            change_points = algo.predict(pen=10)
            return change_points[:-1]  
        except:
            return []
    
    def test_granger_causality(self, cause_series, effect_series, max_lag=6):
        """
        Test Granger causality between two time series
        """
        try:
            data = np.column_stack([effect_series, cause_series])
            
            if len(data) < 2 * max_lag + 1:
                return False, 1.0
            
            gc_result = grangercausalitytests(data, max_lag, verbose=False)
            
            p_values = [gc_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            min_p_value = min(p_values)
            
            is_causal = min_p_value < self.causality_threshold
            
            return is_causal, min_p_value
            
        except Exception as e:
            return False, 1.0
    
    def analyze_daily_anomalies(self, rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids):
        """
        Main analysis function for daily anomaly detection
        
        Parameters:
        - rtwp_data: dict {sector_id: hourly_rtwp_values}
        - path_loss_matrix: 2D array of path loss values between sectors
        - anomaly_sectors: list of sector IDs detected as anomalous
        - sector_ids: list of all sector IDs in order matching path_loss_matrix
        
        Returns:
        - Analysis results with clusters and propagation patterns
        """
        # Calculate influence weights
        influence_weights, valid_connections = self.calculate_influence_weights(path_loss_matrix)
        
        # Create sector ID to index mapping
        sector_to_idx = {sector_id: idx for idx, sector_id in enumerate(sector_ids)}
        
        results = {
            'clusters': {},
            'propagation_analysis': {},
            'similarity_matrix': {},
            'change_points': {} ,  
            'causality_results': {},
            'affected_sectors': set()
        }
        
        # For each anomalous sector, analyze its influence network
        for anomaly_sector in anomaly_sectors:
            if anomaly_sector not in sector_to_idx:
                continue
                
            anomaly_idx = sector_to_idx[anomaly_sector]
            anomaly_data = rtwp_data[anomaly_sector]
            '''
            cp_anomaly = self.detect_change_points(anomaly_data)
            results['change_points'][anomaly_sector] = {
                'self': cp_anomaly,
                'neighbors': {}      # store neighbors too
            }
            '''
            # Find sectors within path loss threshold
            connected_sectors = []
            for i, sector_id in enumerate(sector_ids):
                if (i != anomaly_idx and 
                    valid_connections[anomaly_idx, i] and 
                    sector_id in rtwp_data):
                    connected_sectors.append((sector_id, i))
            
            # Calculate similarities with connected sectors
            similarities = {}
            causality_results = {}
            
            for sector_id, sector_idx in connected_sectors:
                neighbor_data = rtwp_data[sector_id]
               # cp_neighbor = self.detect_change_points(neighbor_data)
               # results['change_points'][anomaly_sector]['neighbors'][sector_id] = cp_neighbor
                influence_weight = influence_weights[anomaly_idx, sector_idx]
                
                # Calculate weighted similarity
                weighted_sim, detailed_sims = self.calculate_weighted_similarity(
                    anomaly_data, neighbor_data, influence_weight
                )
                
                similarities[sector_id] = {
                    'weighted_similarity': weighted_sim,
                    'raw_similarity': detailed_sims,
                    'influence_weight': influence_weight,
                    'path_loss': path_loss_matrix[anomaly_idx, sector_idx]
                }
                
                # Test Granger causality
                is_causal, p_value = self.test_granger_causality(
                    anomaly_data, neighbor_data
                )
                causality_results[sector_id] = {
                    'is_causal': is_causal,
                    'p_value': p_value
                }
            
            # Identify affected sectors (high similarity or causal relationship)
            affected_sectors = []
            for sector_id in similarities:
                sim_score = similarities[sector_id]['weighted_similarity']
                is_causal = causality_results[sector_id]['is_causal']
                
                # Sector is affected if:
                # 1. High similarity score, OR
                # 2. Significant causal relationship, OR
                # 3. Moderate similarity with strong path loss connection``
                if (sim_score >= self.similarity_threshold or 
                    is_causal or 
                    (sim_score >= 0.5 and similarities[sector_id]['influence_weight'] >= 0.5)):
                    affected_sectors.append(sector_id)
                    results['affected_sectors'].add(sector_id)
            
            results['clusters'][anomaly_sector] = affected_sectors
            results['similarity_matrix'][anomaly_sector] = similarities
            results['causality_results'][anomaly_sector] = causality_results
            
            # Propagation analysis
            propagation_analysis = {
                'primary_affected': [s for s in affected_sectors 
                                   if similarities[s]['weighted_similarity'] >= 0.8],
                'secondary_affected': [s for s in affected_sectors 
                                     if 0.5 <= similarities[s]['weighted_similarity'] < 0.8],
                'causal_affected': [s for s in affected_sectors 
                                  if causality_results[s]['is_causal']],
                'total_affected_count': len(affected_sectors),
                'max_similarity': max([similarities[s]['weighted_similarity'] 
                                     for s in similarities] + [0]),
                'avg_path_loss': np.mean([similarities[s]['path_loss'] 
                                        for s in affected_sectors] + [0])
            }
            
            results['propagation_analysis'][anomaly_sector] = propagation_analysis
        
        return results
    
    def generate_report(self, results, anomaly_sectors):
        """
        Generate human-readable analysis report
        """
        report = []
        report.append("=== DAILY ANOMALY CO-BEHAVIOR ANALYSIS REPORT ===\n")
            
        total_affected = len(results['affected_sectors'])
        report.append(f"Total sectors analyzed: {len(anomaly_sectors)}")
        report.append(f"Total additional sectors affected: {total_affected}")
        report.append(f"Average affected sectors per anomaly: {total_affected/len(anomaly_sectors):.1f}\n")
        
        for anomaly_sector in anomaly_sectors:
            if anomaly_sector not in results['clusters']:
                continue
                
            cluster = results['clusters'][anomaly_sector]
            propagation = results['propagation_analysis'][anomaly_sector]
            
            report.append(f"\n--- ANOMALY SECTOR: {anomaly_sector} ---")
            report.append(f"Affected sectors: {len(cluster)}")
            report.append(f"Primary affected: {len(propagation['primary_affected'])}")
            report.append(f"Secondary affected: {len(propagation['secondary_affected'])}")
            report.append(f"Causal relationships: {len(propagation['causal_affected'])}")
            report.append(f"Max similarity score: {propagation['max_similarity']:.3f}")
            report.append(f"Avg path loss to affected: {propagation['avg_path_loss']:.1f} dB")
            '''
            cps_self = results['change_points'].get(anomaly_sector, {}).get('self', [])
            cps_self_fmt = ', '.join(map(str, cps_self)) if cps_self else 'none'
            report.append(f"Change-points detected in anomaly sector: {cps_self_fmt} "
                    f"(total: {len(cps_self)})")            
            cps_neighbors = results['change_points'].get(anomaly_sector, {}).get('neighbors', {})
            for nbr, cps in cps_neighbors.items():
                cps_fmt = ', '.join(map(str, cps)) if cps else 'none'
                report.append(f"  Neighbor {nbr} change-points: {cps_fmt}")    
            '''
            
            if cluster:
                report.append(f"Affected sector list: {', '.join(map(str, cluster))}")
        
        return "\n".join(report)

# Example usage function
def run_daily_analysis(rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids):
    """
    Run the complete daily anomaly analysis
    
    Parameters:
    - rtwp_data: dict {sector_id: [24 hourly RTWP values]}
    - path_loss_matrix: 2D numpy array (30x30) of path loss values
    - anomaly_sectors: list of sector IDs detected as anomalous
    - sector_ids: list of 30 sector IDs matching path_loss_matrix order
    
    Returns:
    - results: detailed analysis results
    - report: human-readable report
    """
    analyzer = PathLossWeightedCoehaviorAnalyzer(
        path_loss_threshold=130,  # dB
        similarity_threshold=0.5,
        causality_threshold=0.05,
        min_cluster_size=2
    )
    
    results = analyzer.analyze_daily_anomalies(
        rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids
    )
    
    report = analyzer.generate_report(results, anomaly_sectors)
    
    return results, report

# Example data structure for testing
def create_example_data():
    """
    Create example data structure for testing
    """
    np.random.seed(42)
    rtwp_data = {}
    
    for i in range(30):
        sector_id = f"sector_{i:02d}"
        base_rtwp = np.random.normal(-95, 10, 24)
        
        if i < 5:  # First 5 sectors have correlated anomalies
            anomaly_pattern = np.sin(np.linspace(0, 4*np.pi, 24)) * 5
            base_rtwp += anomaly_pattern
        
        rtwp_data[sector_id] = base_rtwp
    
    path_loss_matrix = np.random.uniform(80, 140, (30, 30))
    np.fill_diagonal(path_loss_matrix, 0)  # No self path loss
    
    path_loss_matrix = (path_loss_matrix + path_loss_matrix.T) / 2
    
    sector_ids = [f"sector_{i:02d}" for i in range(30)]
    
    anomaly_sectors = ["sector_00", "sector_01", "sector_02"]
    
    return rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids

'''
if __name__ == "__main__":
    #rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids = create_example_data()
    anamoly_df = pd.read_csv('28.csv')
    anamoly_sectors = anamoly_df['name'].tolist()

    #print(f"rtwp_data:{rtwp_data}, path_loss_matrix:{path_loss_matrix}, anomaly_sectors:{anomaly_sectors}, sector_ids:{sector_ids}")
    #results, report = run_daily_analysis( rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids)
    #print(results,report)
'''
    