import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw
import ruptures as rpt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import time

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
        
        # Add caching for expensive operations
        self._similarity_cache = {}
        self._causality_cache = {}
        
    @lru_cache(maxsize=128)
    def _cached_dtw_distance(self, s1_tuple, s2_tuple):
        """Cached DTW distance calculation"""
        s1 = np.array(s1_tuple, dtype=np.float32)
        s2 = np.array(s2_tuple, dtype=np.float32)
        return dtw.distance(s1, s2)
        
    def calculate_influence_weights(self, path_loss_matrix):
        """
        Convert path loss to influence weights - vectorized
        Lower path loss = higher influence
        """
        # Vectorized operations for better performance
        valid_connections = path_loss_matrix <= self.path_loss_threshold
        
        # Initialize with zeros using the same dtype as input
        influence_weights = np.zeros_like(path_loss_matrix, dtype=np.float32)
        
        # Vectorized exponential decay calculation
        mask = valid_connections & (path_loss_matrix > 0)
        influence_weights[mask] = np.exp(-path_loss_matrix[mask] / 40.0, dtype=np.float32)
        
        # Set diagonal to 0 (sector doesn't influence itself)
        np.fill_diagonal(influence_weights, 0)
        
        return influence_weights, valid_connections
    
    def calculate_rtwp_changes(self, rtwp_timeseries):
        """
        Calculate various change metrics for RTWP time series - optimized
        """
        # Convert to numpy array for vectorized operations
        data = np.array(rtwp_timeseries, dtype=np.float32)
        
        if len(data) < 2:
            return {
                'total_variation': 0.0,
                'max_change': 0.0,
                'std_dev': 0.0,
                'change_points': []
            }
        
        # Vectorized difference calculation
        diffs = np.diff(data)
        
        return {
            'total_variation': float(np.sum(np.abs(diffs))),
            'max_change': float(np.max(np.abs(diffs))),
            'std_dev': float(np.std(data)),
            'change_points': self.detect_change_points(data)
        }
    
    def calculate_weighted_similarity(self, ts1, ts2, influence_weight):
        """
        Calculate multiple similarity metrics between time series - optimized
        """
        # Convert to numpy arrays for vectorized operations
        ts1 = np.array(ts1, dtype=np.float32)
        ts2 = np.array(ts2, dtype=np.float32)
        
        if len(ts1) != len(ts2) or len(ts1) < 2:
            return 0.0, {}
        
        # Create cache key
        cache_key = (tuple(ts1), tuple(ts2), influence_weight)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Vectorized similarity calculations
        # 1. Pearson correlation (vectorized)
        correlation = np.corrcoef(ts1, ts2)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # 2. Cosine similarity (using sklearn for efficiency)
        cosine_sim = cosine_similarity(ts1.reshape(1, -1), ts2.reshape(1, -1))[0, 0]
        
        # 3. DTW distance (cached)
        try:
            dtw_dist = self._cached_dtw_distance(tuple(ts1), tuple(ts2))
            # Normalize DTW distance to similarity (0-1 scale)
            max_possible_dtw = np.sqrt(len(ts1)) * np.max([np.std(ts1), np.std(ts2)])
            dtw_similarity = max(0, 1 - (dtw_dist / max_possible_dtw)) if max_possible_dtw > 0 else 0
        except:
            dtw_similarity = 0.0
        
        # 4. Euclidean similarity (vectorized)
        euclidean_dist = np.linalg.norm(ts1 - ts2)
        max_possible_euclidean = np.linalg.norm(ts1) + np.linalg.norm(ts2)
        euclidean_similarity = max(0, 1 - (euclidean_dist / max_possible_euclidean)) if max_possible_euclidean > 0 else 0
        
        # Combine similarities with weights
        similarities = {
            'correlation': float(correlation),
            'cosine': float(cosine_sim),
            'dtw': float(dtw_similarity),
            'euclidean': float(euclidean_similarity)
        }
        
        # Weighted average (you can adjust these weights)
        weights = [0.3, 0.3, 0.25, 0.15]  # correlation, cosine, dtw, euclidean
        combined_similarity = (
            weights[0] * abs(correlation) +
            weights[1] * cosine_sim +
            weights[2] * dtw_similarity +
            weights[3] * euclidean_similarity
        )
        
        # Apply influence weight
        weighted_similarity = combined_similarity * influence_weight
        
        # Cache the result
        result = (float(weighted_similarity), similarities)
        self._similarity_cache[cache_key] = result
        
        return result
    
    def detect_change_points(self, rtwp_data):
        """
        Detect change points in RTWP time series using optimized Pelt
        """
        if len(rtwp_data) < 10:  # Minimum length for change point detection
            return []
            
        try:
            # Use optimized Pelt algorithm with caching
            data = np.array(rtwp_data, dtype=np.float32)
            algo = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(data)
            change_points = algo.predict(pen=10)
            return change_points[:-1]  # Remove last point (end of series)
        except:
            return []
    
    def test_granger_causality(self, cause_series, effect_series, max_lag=6):
        """
        Test Granger causality between two time series with caching
        """
        # Create cache key
        cache_key = (tuple(cause_series), tuple(effect_series), max_lag)
        if cache_key in self._causality_cache:
            return self._causality_cache[cache_key]
            
        try:
            # Convert to numpy arrays
            cause_series = np.array(cause_series, dtype=np.float32)
            effect_series = np.array(effect_series, dtype=np.float32)
            
            # Prepare data (effect, cause)
            data = np.column_stack([effect_series, cause_series])
            
            # Ensure minimum length
            if len(data) < 2 * max_lag + 1:
                result = (False, 1.0)
                self._causality_cache[cache_key] = result
                return result
            
            # Test causality
            gc_result = grangercausalitytests(data, max_lag, verbose=False)
            
            # Get minimum p-value across all lags (vectorized)
            p_values = [gc_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            min_p_value = min(p_values)
            
            is_causal = min_p_value < self.causality_threshold
            
            result = (is_causal, float(min_p_value))
            self._causality_cache[cache_key] = result
            return result
            
        except Exception as e:
            result = (False, 1.0)
            self._causality_cache[cache_key] = result
            return result
    
    def analyze_daily_anomalies(self, rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids):
        """
        Main analysis function for daily anomaly detection - optimized version
        """
        start_time = time.time()
        print(f"Starting analysis of {len(anomaly_sectors)} anomalous sectors...")
        
        # Calculate influence weights (vectorized)
        influence_weights, valid_connections = self.calculate_influence_weights(path_loss_matrix)
        
        # Create sector ID to index mapping
        sector_to_idx = {sector_id: idx for idx, sector_id in enumerate(sector_ids)}
        
        results = {
            'clusters': {},
            'propagation_analysis': {},
            'similarity_matrix': {},
            'causality_results': {},
            'affected_sectors': set(),
            'performance_stats': {}
        }
        
        # Batch process anomalous sectors for better performance
        valid_anomaly_sectors = [s for s in anomaly_sectors if s in sector_to_idx and s in rtwp_data]
        print(f"Processing {len(valid_anomaly_sectors)} valid anomalous sectors...")
        
        for anomaly_sector in valid_anomaly_sectors:
            if anomaly_sector not in sector_to_idx:
                continue
                
            anomaly_idx = sector_to_idx[anomaly_sector]
            anomaly_data = rtwp_data[anomaly_sector]
            
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
                # 3. Moderate similarity with strong path loss connection
                if (sim_score >= self.similarity_threshold or 
                    is_causal or 
                    (sim_score >= 0.5 and similarities[sector_id]['influence_weight'] >= 0.5)):
                    affected_sectors.append(sector_id)
                    results['affected_sectors'].add(sector_id)
            
            # Store results for this anomaly sector
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
        
        end_time = time.time()
        results['performance_stats']['total_analysis_time'] = end_time - start_time
        print(f"Analysis completed in {results['performance_stats']['total_analysis_time']:.2f} seconds.")
        
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
    # Initialize analyzer
    analyzer = PathLossWeightedCoehaviorAnalyzer(
        path_loss_threshold=130,  # dB
        similarity_threshold=0.5,
        causality_threshold=0.05,
        min_cluster_size=2
    )
    
    # Run analysis
    results = analyzer.analyze_daily_anomalies(
        rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids
    )
    
    # Generate report
    report = analyzer.generate_report(results, anomaly_sectors)
    
    return results, report

# Example data structure for testing
def create_example_data():
    """
    Create example data structure for testing
    """
    # 30 sectors with 24 hourly RTWP values each
    np.random.seed(42)
    rtwp_data = {}
    
    for i in range(30):
        sector_id = f"sector_{i:02d}"
        # Generate realistic RTWP values (-120 to -70 dBm)
        base_rtwp = np.random.normal(-95, 10, 24)
        
        # Add some correlation structure for testing
        if i < 5:  # First 5 sectors have correlated anomalies
            anomaly_pattern = np.sin(np.linspace(0, 4*np.pi, 24)) * 5
            base_rtwp += anomaly_pattern
        
        rtwp_data[sector_id] = base_rtwp
    
    # Path loss matrix (30x30) - realistic path loss values
    path_loss_matrix = np.random.uniform(80, 140, (30, 30))
    np.fill_diagonal(path_loss_matrix, 0)  # No self path loss
    
    # Make matrix symmetric (path loss A->B = B->A)
    path_loss_matrix = (path_loss_matrix + path_loss_matrix.T) / 2
    
    # Sector IDs
    sector_ids = [f"sector_{i:02d}" for i in range(30)]
    
    # Example anomalous sectors
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
    