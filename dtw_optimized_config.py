"""
DTW Optimization Configuration
Centralized settings for optimizing Dynamic Time Warping performance.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DTWOptimizer:
    """Centralized DTW optimization configuration and utilities."""
    
    def __init__(self):
        self.optimization_profiles = {
            'fast': {
                'window': 10,                    # Limited warping window
                'use_pruning': True,             # Enable pruning
                'max_dist': None,                # No distance limit
                'max_step': None,                # No step limit
                'max_length_diff': 5,            # Strict length difference
                'penalty': 0,                    # No penalty
                'psi': None,                     # No psi relaxation
                'use_c': True,                   # Use C implementation
                'inner_dist': 'squared euclidean'  # Fastest distance metric
            },
            'balanced': {
                'window': None,                  # Full warping path
                'use_pruning': True,             # Enable pruning
                'max_dist': None,                # No distance limit
                'max_step': None,                # No step limit
                'max_length_diff': 10,           # Moderate length difference
                'penalty': 0,                    # No penalty
                'psi': None,                     # No psi relaxation
                'use_c': True,                   # Use C implementation
                'inner_dist': 'squared euclidean'  # Good balance of speed/accuracy
            },
            'accurate': {
                'window': None,                  # Full warping path
                'use_pruning': False,            # Disable pruning for accuracy
                'max_dist': None,                # No distance limit
                'max_step': None,                # No step limit
                'max_length_diff': None,         # No length limit
                'penalty': 0,                    # No penalty
                'psi': None,                     # No psi relaxation
                'use_c': True,                   # Use C implementation
                'inner_dist': 'euclidean'       # Most accurate distance metric
            }
        }
    
    def get_optimal_settings(self, 
                           series_length: int, 
                           num_comparisons: int,
                           available_memory_gb: float = 2.0,
                           target_speed: str = 'balanced') -> Dict[str, Any]:
        """
        Get optimal DTW settings based on data characteristics and constraints.
        
        Args:
            series_length: Length of time series
            num_comparisons: Number of DTW comparisons to perform
            available_memory_gb: Available memory in GB
            target_speed: 'fast', 'balanced', or 'accurate'
        
        Returns:
            Optimized DTW settings dictionary
        """
        base_settings = self.optimization_profiles[target_speed].copy()
        
        # Adjust settings based on data characteristics
        if series_length > 100:
            # For long series, use more aggressive optimizations
            base_settings['use_pruning'] = True
            if target_speed == 'fast':
                base_settings['window'] = min(20, series_length // 5)
        
        if num_comparisons > 1000:
            # For many comparisons, prioritize speed
            base_settings['use_pruning'] = True
            base_settings['inner_dist'] = 'squared euclidean'
        
        # Memory-based adjustments
        estimated_memory_per_comparison = self._estimate_memory_usage(
            series_length, base_settings
        )
        total_estimated_memory = (estimated_memory_per_comparison * num_comparisons) / 1024**3
        
        if total_estimated_memory > available_memory_gb * 0.8:
            logger.warning(f"High memory usage expected: {total_estimated_memory:.2f}GB")
            # Use more aggressive optimizations
            base_settings['use_pruning'] = True
            if base_settings['window'] is None:
                base_settings['window'] = max(10, series_length // 10)
        
        logger.info(f"DTW settings optimized for {target_speed} profile:")
        logger.info(f"  Series length: {series_length}")
        logger.info(f"  Comparisons: {num_comparisons}")
        logger.info(f"  Estimated memory: {total_estimated_memory:.2f}GB")
        
        return base_settings
    
    def _estimate_memory_usage(self, series_length: int, settings: Dict[str, Any]) -> int:
        """Estimate memory usage in bytes for a single DTW computation."""
        # Base memory for cumulative cost matrix
        if settings.get('window') is not None:
            # Banded matrix
            window = settings['window']
            matrix_size = series_length * (2 * window + 1)
        else:
            # Full matrix
            matrix_size = series_length * series_length
        
        # 4 bytes per float32 value
        memory_bytes = matrix_size * 4
        
        # Add overhead for intermediate calculations
        memory_bytes *= 1.5
        
        return int(memory_bytes)
    
    def enable_c_acceleration(self) -> bool:
        """
        Check if C-based DTW acceleration is available and enable it.
        
        Returns:
            True if C acceleration is available, False otherwise
        """
        try:
            from dtaidistance import dtw_cc
            if dtw_cc is not None:
                logger.info("DTW C acceleration is available and enabled")
                return True
        except ImportError:
            logger.warning("DTW C acceleration not available - falling back to Python")
        
        return False
    
    def benchmark_settings(self, 
                          test_series1: np.ndarray, 
                          test_series2: np.ndarray,
                          settings_list: list = None) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different DTW settings on test data.
        
        Args:
            test_series1: First test time series
            test_series2: Second test time series
            settings_list: List of settings to benchmark (uses default profiles if None)
        
        Returns:
            Dictionary with benchmark results
        """
        import time
        from dtaidistance import dtw
        
        if settings_list is None:
            settings_list = list(self.optimization_profiles.keys())
        
        results = {}
        
        for profile_name in settings_list:
            if isinstance(profile_name, str):
                settings = self.optimization_profiles[profile_name]
                name = profile_name
            else:
                settings = profile_name
                name = "custom"
            
            # Warm up
            try:
                dtw.distance(test_series1, test_series2, **settings)
            except:
                results[name] = {'time': float('inf'), 'error': 'Failed'}
                continue
            
            # Benchmark
            start_time = time.time()
            try:
                distance = dtw.distance(test_series1, test_series2, **settings)
                end_time = time.time()
                
                results[name] = {
                    'time': end_time - start_time,
                    'distance': distance,
                    'error': None
                }
            except Exception as e:
                results[name] = {
                    'time': float('inf'),
                    'distance': None,
                    'error': str(e)
                }
        
        # Sort by execution time
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['time']))
        
        logger.info("DTW Benchmark Results:")
        for name, result in sorted_results.items():
            if result['error'] is None:
                logger.info(f"  {name}: {result['time']:.4f}s (distance: {result['distance']:.4f})")
            else:
                logger.info(f"  {name}: FAILED - {result['error']}")
        
        return sorted_results
    
    def get_recommended_batch_size(self, 
                                 series_length: int,
                                 total_comparisons: int,
                                 available_memory_gb: float = 2.0) -> int:
        """
        Calculate recommended batch size for DTW computations.
        
        Args:
            series_length: Length of time series
            total_comparisons: Total number of DTW comparisons
            available_memory_gb: Available memory in GB
        
        Returns:
            Recommended batch size
        """
        # Estimate memory per comparison
        memory_per_comparison = self._estimate_memory_usage(
            series_length, self.optimization_profiles['balanced']
        )
        
        # Use 50% of available memory for safety
        usable_memory_bytes = available_memory_gb * 0.5 * 1024**3
        
        # Calculate batch size
        batch_size = int(usable_memory_bytes / memory_per_comparison)
        
        # Ensure reasonable bounds
        batch_size = max(1, min(batch_size, total_comparisons))
        
        logger.info(f"Recommended DTW batch size: {batch_size}")
        return batch_size

# Global instance for easy access
dtw_optimizer = DTWOptimizer()

# Convenience functions
def get_fast_dtw_settings(series_length: int = 24) -> Dict[str, Any]:
    """Get fast DTW settings optimized for speed."""
    return dtw_optimizer.get_optimal_settings(
        series_length=series_length,
        num_comparisons=100,
        target_speed='fast'
    )

def get_balanced_dtw_settings(series_length: int = 24) -> Dict[str, Any]:
    """Get balanced DTW settings for speed/accuracy trade-off."""
    return dtw_optimizer.get_optimal_settings(
        series_length=series_length,
        num_comparisons=100,
        target_speed='balanced'
    )

def get_accurate_dtw_settings(series_length: int = 24) -> Dict[str, Any]:
    """Get accurate DTW settings optimized for accuracy."""
    return dtw_optimizer.get_optimal_settings(
        series_length=series_length,
        num_comparisons=100,
        target_speed='accurate'
    )

def enable_dtw_acceleration() -> bool:
    """Enable DTW C-based acceleration if available."""
    return dtw_optimizer.enable_c_acceleration()

if __name__ == "__main__":
    # Example usage and benchmarking
    import numpy as np
    
    # Create test data
    test_series1 = np.random.randn(24).astype(np.float32)
    test_series2 = np.random.randn(24).astype(np.float32)
    
    # Check C acceleration
    c_available = enable_dtw_acceleration()
    print(f"DTW C acceleration available: {c_available}")
    
    # Get optimized settings
    fast_settings = get_fast_dtw_settings()
    balanced_settings = get_balanced_dtw_settings()
    accurate_settings = get_accurate_dtw_settings()
    
    print(f"Fast settings: {fast_settings}")
    print(f"Balanced settings: {balanced_settings}")
    print(f"Accurate settings: {accurate_settings}")
    
    # Benchmark if dtaidistance is available
    try:
        results = dtw_optimizer.benchmark_settings(test_series1, test_series2)
        print("Benchmark completed - see logs for results")
    except ImportError:
        print("dtaidistance not available for benchmarking")