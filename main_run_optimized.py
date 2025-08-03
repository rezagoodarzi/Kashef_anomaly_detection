"""
Optimized main runner for the anomaly detection pipeline.
Includes performance monitoring, parallelization, and memory optimization.
"""

import os
import pickle
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from typing import Dict, List, Tuple, Any

# Import optimized modules
from data_prep import prepare_inputs_for_date
from anomaly_detection import run_daily_analysis
from clustering_Louvain import cluster_and_visualise
from connectivity_graph import plt, nx, pd, np
from simulated_graph import simulate_sector_map_data, data_preprocess
from performance_utils import (
    monitor, optimization_utils, cache_manager, 
    profile_performance, get_optimal_batch_size
)

# ==== Configuration ======
config = {
    "date": "20250701",
    "rtwp_csv": "Esfehan_RSSI_merged_avg_carriers_2U_filled.csv",
    "RSSI": "Esfehan_RSSI_merged.csv",
    "pathloss_csv": "Data_bridge_ESFAHAN.Neighbors_3G_3G.csv",
    "anomaly_csv": "1.csv",
    "base_output_dir": "outputs",
    "max_workers": min(mp.cpu_count(), 4),  # Limit parallel workers
    "memory_limit_gb": 2.0,  # Memory limit for processing
    "enable_caching": True,
    "optimization_level": "high"  # "low", "medium", "high"
}

# ==== Helper Functions ======
def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

@profile_performance
def save_pickle(obj, filename):
    """Save object to pickle file with optimization."""
    # Optimize numpy arrays before saving
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                obj[key] = optimization_utils.optimize_numpy_arrays(value)
    
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_txt(text, filename):
    """Save text to file."""
    with open(filename, "w", encoding='utf-8') as f:
        f.write(text)

@profile_performance 
def parallel_sector_analysis(sector_batch: List[str], rtwp_data: Dict, 
                           path_loss_matrix: np.ndarray, sector_ids: List[str],
                           analyzer_params: Dict) -> Dict:
    """Analyze a batch of sectors in parallel."""
    from run_anomaly import PathLossWeightedCoehaviorAnalyzer
    
    # Create analyzer instance for this process
    analyzer = PathLossWeightedCoehaviorAnalyzer(**analyzer_params)
    
    # Process this batch of sectors
    batch_results = analyzer.analyze_daily_anomalies(
        rtwp_data, path_loss_matrix, sector_batch, sector_ids
    )
    
    # Clean up memory
    gc.collect()
    
    return batch_results

def merge_analysis_results(results_list: List[Dict]) -> Dict:
    """Merge results from parallel processing."""
    merged = {
        'clusters': {},
        'propagation_analysis': {},
        'similarity_matrix': {},
        'causality_results': {},
        'affected_sectors': set(),
        'performance_stats': {}
    }
    
    total_time = 0
    for results in results_list:
        if results:
            merged['clusters'].update(results.get('clusters', {}))
            merged['propagation_analysis'].update(results.get('propagation_analysis', {}))
            merged['similarity_matrix'].update(results.get('similarity_matrix', {}))
            merged['causality_results'].update(results.get('causality_results', {}))
            merged['affected_sectors'].update(results.get('affected_sectors', set()))
            total_time += results.get('performance_stats', {}).get('total_analysis_time', 0)
    
    merged['performance_stats']['total_analysis_time'] = total_time
    merged['performance_stats']['parallel_efficiency'] = len(results_list)
    
    return merged

@profile_performance
def optimized_data_preparation(config: Dict) -> Tuple[Dict, np.ndarray, List[str], List[str]]:
    """Optimized data preparation with memory monitoring."""
    with monitor.timer("Data Preparation"):
        print("📦 Preparing input data with optimizations...")
        
        # Load and prepare data
        rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids = prepare_inputs_for_date(
            rtwp_csv=config["rtwp_csv"],
            pathloss_csv=config["pathloss_csv"],
            anomaly_csv=config["anomaly_csv"],
            target_date=config["date"]
        )
        
        # Optimize data types
        path_loss_matrix = optimization_utils.optimize_numpy_arrays(path_loss_matrix)
        
        # Optimize RTWP data arrays
        for sector_id in rtwp_data:
            rtwp_data[sector_id] = optimization_utils.optimize_numpy_arrays(rtwp_data[sector_id])
        
        print(f"✅ Data prepared: {len(sector_ids)} sectors, {len(anomaly_sectors)} anomalies")
        return rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids

@profile_performance
def optimized_anomaly_analysis(rtwp_data: Dict, path_loss_matrix: np.ndarray, 
                             anomaly_sectors: List[str], sector_ids: List[str],
                             config: Dict) -> Tuple[Dict, str]:
    """Optimized anomaly analysis with parallelization."""
    with monitor.timer("Anomaly Analysis"):
        print("🔍 Running optimized anomaly detection & co-behavior analysis...")
        
        # Determine if we should use parallel processing
        use_parallel = (len(anomaly_sectors) > 5 and 
                       config["max_workers"] > 1 and 
                       config["optimization_level"] in ["medium", "high"])
        
        analyzer_params = {
            "path_loss_threshold": 130,
            "similarity_threshold": 0.64,
            "causality_threshold": 0.05,
            "min_cluster_size": 2
        }
        
        if use_parallel:
            print(f"📊 Using parallel processing with {config['max_workers']} workers")
            
            # Calculate optimal batch size
            batch_size = get_optimal_batch_size(
                len(anomaly_sectors), 
                config.get("memory_limit_gb", 2.0)
            )
            
            # Split anomaly sectors into batches
            sector_batches = [
                anomaly_sectors[i:i + batch_size]
                for i in range(0, len(anomaly_sectors), batch_size)
            ]
            
            # Process batches in parallel
            results_list = []
            with ProcessPoolExecutor(max_workers=config["max_workers"]) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(
                        parallel_sector_analysis,
                        batch, rtwp_data, path_loss_matrix, sector_ids, analyzer_params
                    ): batch for batch in sector_batches
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        result = future.result()
                        results_list.append(result)
                        print(f"✅ Completed batch with {len(batch)} sectors")
                    except Exception as exc:
                        print(f"❌ Batch generated an exception: {exc}")
                        results_list.append({})
            
            # Merge results from all processes
            results = merge_analysis_results(results_list)
            
        else:
            print("📊 Using single-threaded processing")
            # Use original single-threaded approach
            results, report = run_daily_analysis(
                rtwp_data=rtwp_data,
                path_loss_matrix=path_loss_matrix,
                anomaly_sectors=anomaly_sectors,
                sector_ids=sector_ids,
                **analyzer_params
            )
            
            return results, report
        
        # Generate report for parallel results
        from run_anomaly import PathLossWeightedCoehaviorAnalyzer
        analyzer = PathLossWeightedCoehaviorAnalyzer(**analyzer_params)
        report = analyzer.generate_report(results, anomaly_sectors)
        
        print(f"✅ Analysis completed: {len(results['affected_sectors'])} affected sectors")
        return results, report

@profile_performance
def optimized_visualization(results: Dict, config: Dict, output_dir: str) -> None:
    """Optimized visualization with memory management."""
    with monitor.timer("Visualization"):
        print("🗺️ Generating optimized cluster maps...")
        
        # Load and optimize position data
        esfahan_df = pd.read_csv(config["RSSI"])
        esfahan_df = optimization_utils.optimize_dtypes(esfahan_df)
        
        # Generate unique positions
        esfahan_grouped = esfahan_df.drop_duplicates(subset='Mapped_ID')[
            ['Mapped_ID', 'LATITUDE', 'LONGITUDE', 'AZIMUTH']
        ]
        position_dict = esfahan_grouped.set_index('Mapped_ID')[
            ['LATITUDE', 'LONGITUDE', 'AZIMUTH']
        ].to_dict(orient='index')
        
        # Generate cluster map
        cluster_html = os.path.join(output_dir, f"cluster_importance_map_{config['date']}.html")
        cluster_and_visualise(results, position_dict, output_html=cluster_html)
        
        # Generate propagation map
        graph_html = os.path.join(output_dir, f"anomaly_propagation_map_{config['date']}.html")
        simulate_sector_map_data(
            position_dict, results, 
            cluster_output=cluster_html, 
            graph_output=graph_html
        )
        
        print(f"✅ Visualizations saved to {output_dir}")
        
        # Clean up memory
        del esfahan_df, esfahan_grouped, position_dict
        gc.collect()

def cleanup_resources():
    """Clean up resources and print performance report."""
    # Clear caches
    if config["enable_caching"]:
        cache_stats = cache_manager.get_cache_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"  Entries: {cache_stats['total_entries']}")
        print(f"  Hits: {cache_stats['total_hits']}")
        print(f"  Memory: {cache_stats['memory_usage_mb']:.1f}MB")
        print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
        
        cache_manager.clear_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Print performance report
    monitor.print_report()

# ==== Main Pipeline ======
@profile_performance
def main():
    """Optimized main pipeline with performance monitoring."""
    date = config["date"]
    output_dir = os.path.join(config["base_output_dir"], date)
    ensure_dir(output_dir)
    
    print(f"🚀 Starting optimized anomaly detection pipeline for {date}")
    print(f"⚙️  Optimization level: {config['optimization_level']}")
    print(f"🔧 Max workers: {config['max_workers']}")
    print(f"💾 Memory limit: {config['memory_limit_gb']:.1f}GB")
    
    try:
        # Step 1: Optimized data preparation
        rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids = optimized_data_preparation(config)
        
        # Save prepared data
        prep_file = os.path.join(output_dir, f"prepared_data_{date}.pkl")
        save_pickle({
            "rtwp_data": rtwp_data,
            "path_loss_matrix": path_loss_matrix,
            "anomaly_sectors": anomaly_sectors,
            "sector_ids": sector_ids
        }, prep_file)
        
        # Step 2: Optimized anomaly analysis
        results, report = optimized_anomaly_analysis(
            rtwp_data, path_loss_matrix, anomaly_sectors, sector_ids, config
        )
        
        # Save results
        report_txt = os.path.join(output_dir, f"daily_analysis_report_{date}.txt")
        results_pkl = os.path.join(output_dir, f"daily_analysis_results_{date}.pkl")
        save_txt(report, report_txt)
        save_pickle(results, results_pkl)
        
        # Step 3: Optimized visualization
        optimized_visualization(results, config, output_dir)
        
        print(f"🎉 Pipeline completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        raise
    
    finally:
        # Always cleanup resources
        cleanup_resources()

# ==== Entry point ======
if __name__ == "__main__":
    main()