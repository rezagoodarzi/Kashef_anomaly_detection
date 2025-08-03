"""
Performance monitoring and optimization utilities for the anomaly detection system.
"""

import time
import psutil
import os
import functools
import numpy as np
from typing import Callable, Any, Dict, List
import logging
from contextlib import contextmanager
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitoring utilities for tracking timing and memory usage."""
    
    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process(os.getpid())
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            elapsed_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.metrics[operation_name] = {
                'elapsed_time': elapsed_time,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                'memory_delta_mb': memory_delta,
                'timestamp': start_time
            }
            
            logger.info(f"{operation_name}: {elapsed_time:.3f}s, Memory: {memory_delta:+.1f}MB")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator for profiling function performance."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            with self.timer(func_name):
                return func(*args, **kwargs)
        return wrapper
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system performance information."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / 1024**3,
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'process_memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'process_cpu_percent': self.process.cpu_percent()
        }
    
    def print_report(self):
        """Print a performance report."""
        print("\n=== PERFORMANCE REPORT ===")
        system_info = self.get_system_info()
        
        print(f"System CPU Usage: {system_info['cpu_percent']:.1f}%")
        print(f"System Memory Usage: {system_info['memory_percent']:.1f}%")
        print(f"Process Memory Usage: {system_info['process_memory_mb']:.1f}MB")
        
        if self.metrics:
            print("\nOperation Timings:")
            for operation, metrics in self.metrics.items():
                print(f"  {operation}: {metrics['elapsed_time']:.3f}s "
                      f"(Memory: {metrics['memory_delta_mb']:+.1f}MB)")
        print("=" * 30)

class OptimizationUtils:
    """Utilities for data and computation optimization."""
    
    @staticmethod
    def optimize_dtypes(df):
        """Optimize pandas DataFrame data types for memory efficiency."""
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        
        # Optimize categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        new_memory = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Memory usage reduced from {original_memory:.2f}MB to {new_memory:.2f}MB "
                    f"({(original_memory - new_memory) / original_memory * 100:.1f}% reduction)")
        
        return df
    
    @staticmethod
    def batch_process(data: List, batch_size: int, process_func: Callable) -> List:
        """Process data in batches to reduce memory usage."""
        results = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} (size: {len(batch)})")
            
            batch_result = process_func(batch)
            results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            
            # Force garbage collection after each batch
            gc.collect()
        
        return results
    
    @staticmethod
    def optimize_numpy_arrays(*arrays):
        """Optimize numpy arrays for memory and performance."""
        optimized = []
        for arr in arrays:
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            elif arr.dtype == np.int64:
                if arr.min() >= np.iinfo(np.int32).min and arr.max() <= np.iinfo(np.int32).max:
                    arr = arr.astype(np.int32)
            
            # Ensure C-contiguous arrays for better performance
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            
            optimized.append(arr)
        
        return optimized if len(optimized) > 1 else optimized[0]

class CacheManager:
    """Advanced caching utilities for expensive computations."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.cache = {}
        self.cache_info = {}
    
    def cached_computation(self, cache_key: str, computation_func: Callable, *args, **kwargs):
        """Cache the result of expensive computations."""
        if cache_key in self.cache:
            self.cache_info[cache_key]['hits'] += 1
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        
        # Check memory usage before adding new cache entry
        current_memory = sum(self._estimate_memory(value) for value in self.cache.values())
        if current_memory > self.max_memory_mb * 1024 * 1024:  # Convert to bytes
            self._cleanup_cache()
        
        result = computation_func(*args, **kwargs)
        self.cache[cache_key] = result
        self.cache_info[cache_key] = {'hits': 0, 'created': time.time()}
        
        logger.debug(f"Cache miss for {cache_key}, result cached")
        return result
    
    def _estimate_memory(self, obj) -> int:
        """Rough estimate of object memory usage in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_memory(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_memory(k) + self._estimate_memory(v) for k, v in obj.items())
        else:
            return 1024  # Default 1KB estimate for other objects
    
    def _cleanup_cache(self):
        """Remove least recently used cache entries."""
        # Sort by hit count and creation time
        sorted_keys = sorted(self.cache_info.keys(), 
                           key=lambda k: (self.cache_info[k]['hits'], self.cache_info[k]['created']))
        
        # Remove bottom 25% of cache entries
        keys_to_remove = sorted_keys[:len(sorted_keys) // 4]
        for key in keys_to_remove:
            del self.cache[key]
            del self.cache_info[key]
        
        logger.info(f"Cache cleanup: removed {len(keys_to_remove)} entries")
    
    def clear_cache(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.cache_info.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(info['hits'] for info in self.cache_info.values())
        total_entries = len(self.cache)
        memory_usage = sum(self._estimate_memory(value) for value in self.cache.values())
        
        return {
            'total_entries': total_entries,
            'total_hits': total_hits,
            'memory_usage_mb': memory_usage / 1024 / 1024,
            'hit_rate': total_hits / max(total_entries, 1)
        }

# Global instances
monitor = PerformanceMonitor()
cache_manager = CacheManager()
optimization_utils = OptimizationUtils()

# Convenience decorators
def profile_performance(func):
    """Decorator for easy performance profiling."""
    return monitor.profile_function(func)

def optimize_dtw_settings():
    """Get optimized DTW settings for better performance."""
    return {
        'window': None,  # Use full warping path for accuracy
        'use_pruning': True,  # Enable pruning for speed
        'max_dist': None,  # No distance limit
        'max_step': None,  # No step limit
        'max_length_diff': 10,  # Limit length difference
        'penalty': 0,  # No penalty
        'psi': None,  # No psi relaxation
        'use_c': True,  # Use C implementation if available
        'inner_dist': 'squared euclidean'  # Faster than euclidean
    }

def get_optimal_batch_size(data_size: int, available_memory_gb: float = None) -> int:
    """Calculate optimal batch size based on data size and available memory."""
    if available_memory_gb is None:
        available_memory_gb = psutil.virtual_memory().available / 1024**3
    
    # Conservative estimate: use 25% of available memory
    usable_memory_gb = available_memory_gb * 0.25
    
    # Estimate memory per item (rough approximation)
    memory_per_item_mb = 1.0  # 1MB per data item
    
    # Calculate batch size
    batch_size = int((usable_memory_gb * 1024) / memory_per_item_mb)
    
    # Ensure reasonable bounds
    batch_size = max(10, min(batch_size, data_size // 4))
    
    logger.info(f"Calculated optimal batch size: {batch_size} "
                f"(data_size: {data_size}, available_memory: {available_memory_gb:.1f}GB)")
    
    return batch_size