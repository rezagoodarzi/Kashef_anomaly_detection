#!/usr/bin/env python3
"""
Behavioral Connections Generator (Enhanced Version)
Detects sector pairs with correlated behavior using multiple algorithms:
1. Pearson Correlation - Overall pattern similarity
2. Difference Correlation - Rate of change similarity  
3. PELT Change Point Detection - Simultaneous sudden changes
4. Direction Alignment - Same direction changes

Outputs:
- behavioral_connections.csv/json - Sector pairs with aggregated correlation scores
- change_points.csv/json - All detected change points with correlated sectors
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Import ruptures for change point detection
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: 'ruptures' not installed. Install with: pip install ruptures")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for behavioral connectivity analysis."""
    
    # Correlation thresholds
    PEARSON_THRESHOLD: float = 0.85
    COMPOSITE_THRESHOLD: float = 0.7  # Lower threshold for composite score
    
    # Change point detection
    PELT_PENALTY: float = 5.0  # Higher = fewer change points
    MIN_SIGNAL_LENGTH: int = 6
    
    # Weights for composite score
    WEIGHT_PEARSON: float = 0.45
    WEIGHT_DIFF_CORR: float = 0.0
    WEIGHT_CHANGE_ALIGN: float = 0.5  # Higher weight - important for telecom
    WEIGHT_DIRECTION: float = 0.05
    
    # Data requirements
    MIN_SECTOR_DATA_POINTS: int = 10
    MAX_SECTORS: int = 200
    ANOMALY_SECTOR_PRIORITY: int = 50
    DEFAULT_RSSI_FILL: float = -110.0
    
    # Deseasonalization (remove daily patterns to avoid false correlations)
    REMOVE_DAILY_PATTERN: bool = False
    
    # Output filenames
    OUTPUT_CONNECTIONS_CSV: str = 'behavioral_connections.csv'
    OUTPUT_CONNECTIONS_JSON: str = 'behavioral_connections.json'
    OUTPUT_CHANGEPOINTS_CSV: str = 'change_points.csv'
    OUTPUT_CHANGEPOINTS_JSON: str = 'change_points.json'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the telecom data."""
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    required_cols = ['Date', 'NE', 'RSSI_PUCCH(EUCell_Eric)(CRA)', 'Anomaly']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"Loaded {len(df):,} records with {df['NE'].nunique()} unique sectors")
    return df


# =============================================================================
# SECTOR SELECTION
# =============================================================================

def select_valid_sectors(
    df: pd.DataFrame,
    config: Config
) -> List[str]:
    """Select sectors with sufficient data, prioritizing anomaly sectors."""
    sector_counts = df.groupby('NE').size()
    valid_sectors = sector_counts[sector_counts >= config.MIN_SECTOR_DATA_POINTS].index.tolist()
    
    if len(valid_sectors) <= config.MAX_SECTORS:
        return valid_sectors
    
    anomaly_sectors = df[df['Anomaly'] == True]['NE'].unique().tolist()
    normal_sectors = df[df['Anomaly'] == False]['NE'].unique().tolist()
    
    anomaly_sectors = [s for s in anomaly_sectors if s in valid_sectors]
    normal_sectors = [s for s in normal_sectors if s in valid_sectors]
    
    selected = anomaly_sectors[:config.ANOMALY_SECTOR_PRIORITY]
    remaining_slots = config.MAX_SECTORS - len(selected)
    
    if remaining_slots > 0 and normal_sectors:
        np.random.seed(42)
        normal_sample = np.random.choice(
            normal_sectors,
            size=min(remaining_slots, len(normal_sectors)),
            replace=False
        ).tolist()
        selected.extend(normal_sample)
    
    return selected[:config.MAX_SECTORS]


# =============================================================================
# CHANGE POINT DETECTION (PELT Algorithm)
# =============================================================================

def detect_change_points_pelt(
    signal: np.ndarray,
    penalty: float = Config.PELT_PENALTY,
    min_size: int = 3
) -> List[int]:
    """
    Detect change points using PELT (Pruned Exact Linear Time) algorithm.
    
    PELT Algorithm:
    - Finds points where the statistical properties of the signal change
    - Uses dynamic programming with pruning for efficiency
    - Cost function: measures how well a segment fits a model (RBF kernel)
    
    Mathematical basis:
    - Minimizes: sum(Cost(segment)) + penalty * num_changepoints
    - RBF model assumes Gaussian distribution within segments
    
    Args:
        signal: 1D numpy array of values
        penalty: Penalty for adding change points (higher = fewer points)
        min_size: Minimum segment size between change points
        
    Returns:
        List of change point indices
    """
    if not RUPTURES_AVAILABLE:
        return []
    
    if len(signal) < Config.MIN_SIGNAL_LENGTH:
        return []
    
    try:
        signal = np.array(signal).flatten()
        
        # Normalize signal for better detection
        if np.std(signal) > 0:
            signal_norm = (signal - np.mean(signal)) / np.std(signal)
        else:
            return []
        
        # PELT with RBF (Radial Basis Function) kernel
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal_norm)
        change_points = algo.predict(pen=penalty)
        
        # Remove last point (end of signal)
        return change_points[:-1] if change_points else []
        
    except Exception:
        return []


def get_change_magnitude_and_direction(
    signal: np.ndarray,
    change_point: int,
    window: int = 3
) -> Tuple[float, int]:
    """
    Calculate the magnitude and direction of change at a change point.
    
    Args:
        signal: The time series
        change_point: Index of the change point
        window: Number of points before/after to average
        
    Returns:
        (magnitude, direction) where direction is +1 (increase) or -1 (decrease)
    """
    before_start = max(0, change_point - window)
    after_end = min(len(signal), change_point + window)
    
    before_mean = np.mean(signal[before_start:change_point])
    after_mean = np.mean(signal[change_point:after_end])
    
    magnitude = abs(after_mean - before_mean)
    direction = 1 if after_mean > before_mean else -1
    
    return magnitude, direction


# =============================================================================
# DESEASONALIZATION (Remove Daily Patterns)
# =============================================================================

def remove_daily_pattern(
    signal: np.ndarray,
    timestamps: List,
    min_samples_per_hour: int = 2
) -> np.ndarray:
    """
    Remove daily pattern from signal to avoid spurious correlations.
    
    Method:
    1. Calculate median RSSI for each hour of day (0-23) from historical data
    2. Subtract hourly baseline from signal
    3. Return residuals (deviations from normal pattern)
    
    This way, normal daily traffic patterns don't create false correlations.
    Only ANOMALOUS deviations (interference) will correlate.
    
    Args:
        signal: Raw RSSI time series
        timestamps: List of datetime timestamps
        min_samples_per_hour: Minimum samples needed to calculate hourly baseline
        
    Returns:
        Residual signal (original - hourly_baseline)
    """
    if len(signal) != len(timestamps):
        return signal
    
    # Extract hour from each timestamp
    try:
        hours = np.array([t.hour if hasattr(t, 'hour') else pd.Timestamp(t).hour 
                         for t in timestamps])
    except:
        return signal  # Can't extract hours, return original
    
    # Calculate median for each hour (0-23)
    hourly_baseline = {}
    for hour in range(24):
        hour_mask = hours == hour
        hour_values = signal[hour_mask]
        
        if len(hour_values) >= min_samples_per_hour:
            hourly_baseline[hour] = np.median(hour_values)
        else:
            hourly_baseline[hour] = None
    
    # Fill missing hours with overall median
    overall_median = np.median(signal)
    for hour in range(24):
        if hourly_baseline[hour] is None:
            hourly_baseline[hour] = overall_median
    
    # Calculate residuals
    residuals = np.zeros_like(signal, dtype=float)
    for i, (value, hour) in enumerate(zip(signal, hours)):
        residuals[i] = value - hourly_baseline[hour]
    
    return residuals


# =============================================================================
# CORRELATION ALGORITHMS
# =============================================================================

def calculate_pearson_correlation(series1: np.ndarray, series2: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Formula: r = Σ(x-x̄)(y-ȳ) / √[Σ(x-x̄)² × Σ(y-ȳ)²]
    
    Returns:
        (correlation, p_value)
    """
    if len(series1) != len(series2) or len(series1) < 3:
        return 0.0, 1.0
    
    try:
        corr, p_value = pearsonr(series1, series2)
        return float(corr) if not np.isnan(corr) else 0.0, float(p_value)
    except:
        return 0.0, 1.0


def calculate_difference_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Calculate correlation of first differences (rate of change).
    
    This measures if sectors change at the same RATE, not just same level.
    
    Formula: 
    - diff1[i] = series1[i+1] - series1[i]
    - diff2[i] = series2[i+1] - series2[i]  
    - r_diff = pearson(diff1, diff2)
    
    Returns:
        Correlation of differences
    """
    if len(series1) != len(series2) or len(series1) < 4:
        return 0.0
    
    try:
        diff1 = np.diff(series1)
        diff2 = np.diff(series2)
        
        if np.std(diff1) == 0 or np.std(diff2) == 0:
            return 0.0
            
        corr, _ = pearsonr(diff1, diff2)
        return float(corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0


def calculate_change_point_alignment(
    cp1: List[int],
    cp2: List[int],
    tolerance: int = 2
) -> float:
    """
    Calculate how well change points align between two sectors.
    
    Two change points "align" if they occur within 'tolerance' time steps.
    
    Formula:
    - For each CP in sector1, check if there's a matching CP in sector2 within ±tolerance
    - alignment_score = 2 * num_matched / (len(cp1) + len(cp2))
    
    Args:
        cp1: Change points for sector 1
        cp2: Change points for sector 2
        tolerance: Maximum time difference for alignment
        
    Returns:
        Alignment score (0 to 1)
    """
    if not cp1 or not cp2:
        return 0.0
    
    matched = 0
    cp2_set = set(cp2)
    
    for cp in cp1:
        # Check if any CP in sector2 is within tolerance
        for offset in range(-tolerance, tolerance + 1):
            if (cp + offset) in cp2_set:
                matched += 1
                break
    
    # Jaccard-like score
    score = (2 * matched) / (len(cp1) + len(cp2))
    return min(1.0, score)


def calculate_direction_alignment(
    signal1: np.ndarray,
    signal2: np.ndarray,
    cp1: List[int],
    cp2: List[int],
    tolerance: int = 2
) -> float:
    """
    Calculate if aligned change points have the SAME DIRECTION.
    
    This is crucial for telecom: if two sectors both spike UP at the same time,
    it indicates a common interference source.
    
    Returns:
        Proportion of aligned changes with same direction (0 to 1)
    """
    if not cp1 or not cp2:
        return 0.0
    
    same_direction_count = 0
    total_aligned = 0
    
    for cp in cp1:
        # Find closest CP in sector2
        for offset in range(-tolerance, tolerance + 1):
            if (cp + offset) in cp2:
                total_aligned += 1
                
                # Get directions
                _, dir1 = get_change_magnitude_and_direction(signal1, cp)
                _, dir2 = get_change_magnitude_and_direction(signal2, cp + offset)
                
                if dir1 == dir2:
                    same_direction_count += 1
                break
    
    return same_direction_count / total_aligned if total_aligned > 0 else 0.0


def calculate_composite_score(
    pearson: float,
    diff_corr: float,
    change_align: float,
    direction_align: float,
    config: Config
) -> float:
    """
    Calculate weighted composite correlation score.
    
    Formula:
    score = w1*pearson + w2*diff_corr + w3*change_align + w4*direction_align
    
    Where weights sum to 1.0
    """
    score = (
        config.WEIGHT_PEARSON * max(0, pearson) +
        config.WEIGHT_DIFF_CORR * max(0, diff_corr) +
        config.WEIGHT_CHANGE_ALIGN * change_align +
        config.WEIGHT_DIRECTION * direction_align
    )
    return min(1.0, score)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_sector_pair(
    sector1_data: Dict[str, Any],
    sector2_data: Dict[str, Any],
    config: Config
) -> Optional[Dict[str, Any]]:
    """
    Analyze a pair of sectors using all algorithms.
    
    Uses:
    - signal (deseasonalized) for correlation calculations
    - signal_raw for direction alignment (actual spike direction)
    
    Returns:
        Dictionary with all correlation metrics, or None if not correlated
    """
    # Use deseasonalized signal for correlation
    signal1 = sector1_data['signal']
    signal2 = sector2_data['signal']
    
    # Use raw signal for direction analysis (actual spike direction matters)
    signal1_raw = sector1_data.get('signal_raw', signal1)
    signal2_raw = sector2_data.get('signal_raw', signal2)
    
    cp1 = sector1_data['change_points']
    cp2 = sector2_data['change_points']
    
    # Calculate all metrics
    pearson, p_value = calculate_pearson_correlation(signal1, signal2)
    diff_corr = calculate_difference_correlation(signal1, signal2)
    change_align = calculate_change_point_alignment(cp1, cp2)
    # Use RAW signal for direction (we want actual UP/DOWN, not residual direction)
    direction_align = calculate_direction_alignment(signal1_raw, signal2_raw, cp1, cp2)
    
    # Calculate composite score
    composite = calculate_composite_score(
        pearson, diff_corr, change_align, direction_align, config
    )
    
    # Check if meets threshold
    if composite < config.COMPOSITE_THRESHOLD and pearson < config.PEARSON_THRESHOLD:
        return None
    
    return {
        'sector_1': sector1_data['sector_id'],
        'sector_2': sector2_data['sector_id'],
        'composite_score': round(composite, 4),
        'pearson_correlation': round(pearson, 4),
        'pearson_p_value': round(p_value, 6),
        'difference_correlation': round(diff_corr, 4),
        'change_point_alignment': round(change_align, 4),
        'direction_alignment': round(direction_align, 4),
        'sector_1_change_points': len(cp1),
        'sector_2_change_points': len(cp2),
        'sector_1_has_anomaly': sector1_data['has_anomaly'],
        'sector_2_has_anomaly': sector2_data['has_anomaly'],
        'sector_1_avg_rssi': round(sector1_data['avg_rssi'], 2),
        'sector_2_avg_rssi': round(sector2_data['avg_rssi'], 2),
        'both_anomaly': sector1_data['has_anomaly'] and sector2_data['has_anomaly']
    }


def build_sector_data(
    df: pd.DataFrame,
    sectors: List[str],
    config: Config
) -> Dict[str, Dict[str, Any]]:
    """
    Build sector data including time series and change points.
    
    If REMOVE_DAILY_PATTERN is True, calculates residuals (signal - hourly baseline)
    to avoid spurious correlations from normal daily traffic patterns.
    
    Returns:
        Dictionary mapping sector_id to sector data
    """
    rssi_col = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
    
    # Create pivot table
    pivot = df[df['NE'].isin(sectors)].pivot_table(
        index='Date',
        columns='NE',
        values=rssi_col,
        aggfunc='mean'
    ).sort_index()
    
    # Fill missing values
    pivot = pivot.ffill().bfill().fillna(config.DEFAULT_RSSI_FILL)
    
    # Get timestamps
    timestamps = pivot.index.tolist()
    
    # Pre-compute sector statistics
    sector_stats = df.groupby('NE').agg({
        rssi_col: 'mean',
        'Anomaly': 'any'
    })
    
    sector_data = {}
    
    if config.REMOVE_DAILY_PATTERN:
        print(f"      Removing daily patterns (deseasonalization enabled)...")
    print(f"      Detecting change points for {len(sectors)} sectors...")
    
    for sector in sectors:
        if sector not in pivot.columns:
            continue
            
        signal_raw = pivot[sector].values
        
        # Apply deseasonalization if enabled
        if config.REMOVE_DAILY_PATTERN:
            # Use residuals for correlation (removes daily pattern)
            signal_for_correlation = remove_daily_pattern(signal_raw, timestamps)
        else:
            signal_for_correlation = signal_raw
        
        # Detect change points on RAW signal (we want to detect actual spikes)
        change_points = detect_change_points_pelt(signal_raw, config.PELT_PENALTY)
        
        # Get sector stats
        avg_rssi = float(sector_stats.loc[sector, rssi_col]) if sector in sector_stats.index else -110.0
        has_anomaly = bool(sector_stats.loc[sector, 'Anomaly']) if sector in sector_stats.index else False
        
        sector_data[sector] = {
            'sector_id': sector,
            'signal': signal_for_correlation,  # Used for correlation
            'signal_raw': signal_raw,          # Used for change point analysis
            'timestamps': timestamps,
            'change_points': change_points,
            'avg_rssi': avg_rssi,
            'has_anomaly': has_anomaly
        }
    
    total_cps = sum(len(sd['change_points']) for sd in sector_data.values())
    print(f"      Found {total_cps} total change points across all sectors")
    
    return sector_data


def find_all_connections(
    sector_data: Dict[str, Dict[str, Any]],
    config: Config
) -> List[Dict[str, Any]]:
    """
    Find all correlated sector pairs using multiple algorithms.
    """
    connections = []
    sectors = list(sector_data.keys())
    n = len(sectors)
    
    total_pairs = n * (n - 1) // 2
    print(f"      Analyzing {total_pairs} sector pairs...")
    
    for i in range(n):
        for j in range(i + 1, n):
            result = analyze_sector_pair(
                sector_data[sectors[i]],
                sector_data[sectors[j]],
                config
            )
            if result:
                connections.append(result)
    
    # Sort by composite score
    connections.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return connections


# =============================================================================
# CHANGE POINTS OUTPUT
# =============================================================================

def build_change_points_output(
    sector_data: Dict[str, Dict[str, Any]],
    connections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build detailed change points data with correlated sectors.
    
    Returns:
        List of change point records with correlated sectors
    """
    # Build lookup for correlated sectors
    correlated_lookup = {}
    for conn in connections:
        s1, s2 = conn['sector_1'], conn['sector_2']
        if s1 not in correlated_lookup:
            correlated_lookup[s1] = []
        if s2 not in correlated_lookup:
            correlated_lookup[s2] = []
        
        correlated_lookup[s1].append({
            'sector': s2,
            'alignment': conn['change_point_alignment'],
            'composite': conn['composite_score']
        })
        correlated_lookup[s2].append({
            'sector': s1,
            'alignment': conn['change_point_alignment'],
            'composite': conn['composite_score']
        })
    
    change_points_output = []
    
    for sector_id, data in sector_data.items():
        # Use RAW signal for actual RSSI values (not deseasonalized)
        signal = data.get('signal_raw', data['signal'])
        timestamps = data['timestamps']
        change_points = data['change_points']
        
        for cp_idx in change_points:
            if cp_idx >= len(timestamps):
                continue
                
            timestamp = timestamps[cp_idx]
            value_at_cp = float(signal[cp_idx])
            magnitude, direction = get_change_magnitude_and_direction(signal, cp_idx)
            
            # Get value before and after
            before_idx = max(0, cp_idx - 1)
            after_idx = min(len(signal) - 1, cp_idx + 1)
            value_before = float(signal[before_idx])
            value_after = float(signal[after_idx])
            
            # Find correlated sectors with aligned change points
            aligned_sectors = []
            if sector_id in correlated_lookup:
                for corr_info in correlated_lookup[sector_id]:
                    corr_sector = corr_info['sector']
                    if corr_sector in sector_data:
                        corr_cps = sector_data[corr_sector]['change_points']
                        # Check if this sector has a change point within ±2 of current
                        for corr_cp in corr_cps:
                            if abs(corr_cp - cp_idx) <= 2:
                                # Use RAW signal for actual RSSI values
                                corr_signal = sector_data[corr_sector].get('signal_raw', sector_data[corr_sector]['signal'])
                                corr_mag, corr_dir = get_change_magnitude_and_direction(corr_signal, corr_cp)
                                aligned_sectors.append({
                                    'sector': corr_sector,
                                    'change_point_index': corr_cp,
                                    'direction': 'UP' if corr_dir > 0 else 'DOWN',
                                    'magnitude': round(corr_mag, 2),
                                    'same_direction': corr_dir == direction
                                })
                                break
            
            change_points_output.append({
                'sector': sector_id,
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'change_point_index': cp_idx,
                'value_at_change': round(value_at_cp, 2),
                'value_before': round(value_before, 2),
                'value_after': round(value_after, 2),
                'magnitude': round(magnitude, 2),
                'direction': 'UP' if direction > 0 else 'DOWN',
                'has_anomaly': data['has_anomaly'],
                'num_aligned_sectors': len(aligned_sectors),
                'aligned_sectors': aligned_sectors
            })
    
    # Sort by timestamp
    change_points_output.sort(key=lambda x: x['timestamp'])
    
    return change_points_output


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def create_connections_json(
    connections: List[Dict],
    sector_data: Dict[str, Dict],
    config: Config
) -> Dict:
    """Create structured JSON for connections."""
    
    summary = {
        'total_connections': len(connections),
        'total_sectors_analyzed': len(sector_data),
        'thresholds': {
            'composite': config.COMPOSITE_THRESHOLD,
            'pearson': config.PEARSON_THRESHOLD
        },
        'weights': {
            'pearson': config.WEIGHT_PEARSON,
            'difference_correlation': config.WEIGHT_DIFF_CORR,
            'change_point_alignment': config.WEIGHT_CHANGE_ALIGN,
            'direction_alignment': config.WEIGHT_DIRECTION
        }
    }
    
    if connections:
        summary['avg_composite_score'] = round(np.mean([c['composite_score'] for c in connections]), 4)
        summary['avg_change_alignment'] = round(np.mean([c['change_point_alignment'] for c in connections]), 4)
        summary['connections_with_anomaly'] = sum(1 for c in connections if c['sector_1_has_anomaly'] or c['sector_2_has_anomaly'])
        summary['high_change_alignment'] = sum(1 for c in connections if c['change_point_alignment'] > 0.5)
    
    # Sector statistics
    sectors_info = []
    for sector_id, data in sector_data.items():
        conn_count = sum(1 for c in connections if c['sector_1'] == sector_id or c['sector_2'] == sector_id)
        sectors_info.append({
            'sector_id': sector_id,
            'has_anomaly': data['has_anomaly'],
            'avg_rssi': round(data['avg_rssi'], 2),
            'num_change_points': len(data['change_points']),
            'num_connections': conn_count
        })
    
    sectors_info.sort(key=lambda x: x['num_connections'], reverse=True)
    
    return {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'algorithms': [
                'Pearson Correlation',
                'Difference Correlation',
                'PELT Change Point Detection',
                'Direction Alignment'
            ],
            'config': {
                'pelt_penalty': config.PELT_PENALTY,
                'composite_threshold': config.COMPOSITE_THRESHOLD,
                'pearson_threshold': config.PEARSON_THRESHOLD
            }
        },
        'summary': summary,
        'sectors': sectors_info,
        'connections': connections
    }


def create_changepoints_json(
    change_points: List[Dict],
    sector_data: Dict[str, Dict],
    config: Config
) -> Dict:
    """Create structured JSON for change points."""
    
    # Group by sector
    by_sector = {}
    for cp in change_points:
        sector = cp['sector']
        if sector not in by_sector:
            by_sector[sector] = []
        by_sector[sector].append(cp)
    
    # Summary statistics
    total_cps = len(change_points)
    cps_with_alignment = sum(1 for cp in change_points if cp['num_aligned_sectors'] > 0)
    
    direction_counts = {'UP': 0, 'DOWN': 0}
    for cp in change_points:
        direction_counts[cp['direction']] += 1
    
    return {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'algorithm': 'PELT (Pruned Exact Linear Time)',
            'config': {
                'penalty': config.PELT_PENALTY,
                'model': 'RBF (Radial Basis Function)'
            }
        },
        'summary': {
            'total_change_points': total_cps,
            'change_points_with_aligned_sectors': cps_with_alignment,
            'alignment_rate': round(cps_with_alignment / total_cps, 4) if total_cps > 0 else 0,
            'direction_distribution': direction_counts,
            'sectors_with_change_points': len(by_sector)
        },
        'by_sector': {
            sector: {
                'count': len(cps),
                'has_anomaly': sector_data[sector]['has_anomaly'] if sector in sector_data else False,
                'change_points': cps
            }
            for sector, cps in by_sector.items()
        },
        'all_change_points': change_points
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_behavioral_connections(
    data_path: str,
    output_dir: str = 'anomaly_results',
    config: Optional[Config] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    """
    Main function to generate behavioral connections and change points.
    
    Uses multiple algorithms:
    1. Pearson Correlation - Overall pattern similarity
    2. Difference Correlation - Rate of change similarity
    3. PELT Change Point Detection - Simultaneous sudden changes
    4. Direction Alignment - Same direction changes
    
    Args:
        data_path: Path to input CSV file
        output_dir: Directory to save output files
        config: Optional configuration object
        
    Returns:
        Tuple of (connections_df, change_points_df, output_files_dict)
    """
    if config is None:
        config = Config()
    
    print("=" * 70)
    print("BEHAVIORAL CONNECTIONS GENERATOR (Enhanced Multi-Algorithm)")
    print("=" * 70)
    print("\nAlgorithms:")
    print("  1. Pearson Correlation (weight: {:.0%})".format(config.WEIGHT_PEARSON))
    print("  2. Difference Correlation (weight: {:.0%})".format(config.WEIGHT_DIFF_CORR))
    print("  3. PELT Change Point Alignment (weight: {:.0%})".format(config.WEIGHT_CHANGE_ALIGN))
    print("  4. Direction Alignment (weight: {:.0%})".format(config.WEIGHT_DIRECTION))
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df = load_data(data_path)
    
    # Step 2: Select sectors
    print("\n[2/6] Selecting sectors...")
    sectors = select_valid_sectors(df, config)
    print(f"      Selected {len(sectors)} sectors for analysis")
    
    if len(sectors) < 2:
        raise ValueError("Need at least 2 valid sectors")
    
    # Step 3: Build sector data with change points
    print("\n[3/6] Building sector data and detecting change points...")
    sector_data = build_sector_data(df, sectors, config)
    
    # Step 4: Find connections
    print("\n[4/6] Finding correlated sector pairs...")
    connections = find_all_connections(sector_data, config)
    print(f"      Found {len(connections)} correlated pairs")
    
    # Step 5: Build change points output
    print("\n[5/6] Building change points output...")
    change_points = build_change_points_output(sector_data, connections)
    print(f"      Documented {len(change_points)} change points")
    
    # Step 6: Save outputs
    print("\n[6/6] Saving outputs...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Connections CSV
    connections_df = pd.DataFrame(connections)
    csv_conn_path = output_path / config.OUTPUT_CONNECTIONS_CSV
    connections_df.to_csv(csv_conn_path, index=False)
    
    # Connections JSON
    json_conn_path = output_path / config.OUTPUT_CONNECTIONS_JSON
    conn_json = create_connections_json(connections, sector_data, config)
    with open(json_conn_path, 'w', encoding='utf-8') as f:
        json.dump(conn_json, f, indent=2, ensure_ascii=False)
    
    # Change points CSV (flatten aligned_sectors for CSV)
    cp_for_csv = []
    for cp in change_points:
        cp_flat = {k: v for k, v in cp.items() if k != 'aligned_sectors'}
        cp_flat['aligned_sectors'] = ';'.join([a['sector'] for a in cp['aligned_sectors']])
        cp_flat['aligned_same_direction'] = sum(1 for a in cp['aligned_sectors'] if a['same_direction'])
        cp_for_csv.append(cp_flat)
    
    change_points_df = pd.DataFrame(cp_for_csv)
    csv_cp_path = output_path / config.OUTPUT_CHANGEPOINTS_CSV
    change_points_df.to_csv(csv_cp_path, index=False)
    
    # Change points JSON
    json_cp_path = output_path / config.OUTPUT_CHANGEPOINTS_JSON
    cp_json = create_changepoints_json(change_points, sector_data, config)
    with open(json_cp_path, 'w', encoding='utf-8') as f:
        json.dump(cp_json, f, indent=2, ensure_ascii=False)
    
    output_files = {
        'connections_csv': csv_conn_path,
        'connections_json': json_conn_path,
        'changepoints_csv': csv_cp_path,
        'changepoints_json': json_cp_path
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    
    print("\n📊 CONNECTIONS SUMMARY:")
    print(f"   Total correlated pairs: {len(connections)}")
    if connections:
        high_align = sum(1 for c in connections if c['change_point_alignment'] > 0.5)
        print(f"   High change-point alignment (>0.5): {high_align}")
        print(f"   Avg composite score: {np.mean([c['composite_score'] for c in connections]):.3f}")
        print(f"   Avg change alignment: {np.mean([c['change_point_alignment'] for c in connections]):.3f}")
    
    print("\n📍 CHANGE POINTS SUMMARY:")
    print(f"   Total change points: {len(change_points)}")
    with_aligned = sum(1 for cp in change_points if cp['num_aligned_sectors'] > 0)
    print(f"   With aligned sectors: {with_aligned} ({100*with_aligned/len(change_points):.1f}%)" if change_points else "   With aligned sectors: 0")
    
    print("\n📁 OUTPUT FILES:")
    for name, path in output_files.items():
        print(f"   {name}: {path}")
    
    return connections_df, change_points_df, output_files


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv'
    OUTPUT_DIR = r'C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\src\test\anomaly_results_pipeline'
    
    try:
        connections_df, change_points_df, output_files = generate_behavioral_connections(
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR
        )
        print("\n✓ Successfully created all outputs!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
