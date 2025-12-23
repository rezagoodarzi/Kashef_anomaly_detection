#!/usr/bin/env python3
"""
Pipeline Orchestrator
=====================

Main entry point for the database-driven behavioral analysis pipeline.
Coordinates all layers: database interface, data service, and analysis.

Responsibilities:
- Orchestrate end-to-end pipeline execution
- Handle configuration loading
- Manage execution flow and error handling
- Write results back to database

Design Principles:
- Single entry point for pipeline execution
- Clear step-by-step workflow
- Comprehensive logging
- Restart-safe operations
"""

import logging
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

# Import pipeline components
try:
    # Try relative imports (when used as module)
    from .db_interface import (
        IDatabaseInterface,
        DatabaseConfig,
        QueryParams,
        DatabaseConnectionManager,
        create_database_interface
    )
    from .data_service import (
        ConfigurationManager,
        DataService,
        AnalysisAdapter,
        create_data_service
    )
except ImportError:
    # Fall back to absolute imports (when run standalone)
    from db_interface import (
        IDatabaseInterface,
        DatabaseConfig,
        QueryParams,
        DatabaseConnectionManager,
        create_database_interface
    )
    from data_service import (
        ConfigurationManager,
        DataService,
        AnalysisAdapter,
        create_data_service
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE INPUT PARAMETERS
# =============================================================================

@dataclass
class PipelineParams:
    """
    Input parameters for the analysis pipeline.
    
    These are the required inputs specified in the technical design.
    """
    kpi_name: str
    start_datetime: datetime
    end_datetime: datetime
    city: str
    technology: str = "4g"
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.start_datetime >= self.end_datetime:
            raise ValueError("start_datetime must be before end_datetime")
        
        if not self.kpi_name:
            raise ValueError("kpi_name is required")
    
    @classmethod
    def for_yesterday(cls, kpi_name: str, city: str, technology: str = "4g"):
        """Create params for yesterday's data (default daily run)."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        return cls(
            kpi_name=kpi_name,
            start_datetime=yesterday,
            end_datetime=today,
            city=city,
            technology=technology
        )


# =============================================================================
# PIPELINE RESULT
# =============================================================================

@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    run_date: datetime
    params: PipelineParams
    connections_count: int = 0
    change_points_count: int = 0
    sectors_analyzed: int = 0
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    output_files: Dict[str, Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'success': self.success,
            'run_date': self.run_date.isoformat(),
            'kpi_name': self.params.kpi_name,
            'city': self.params.city,
            'technology': self.params.technology,
            'start_datetime': self.params.start_datetime.isoformat(),
            'end_datetime': self.params.end_datetime.isoformat(),
            'connections_count': self.connections_count,
            'change_points_count': self.change_points_count,
            'sectors_analyzed': self.sectors_analyzed,
            'execution_time_seconds': self.execution_time_seconds,
            'error_message': self.error_message
        }


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class PipelineOrchestrator:
    """
    Main orchestrator for the behavioral analysis pipeline.
    
    Coordinates all components and manages the complete workflow.
    """
    
    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        config_dir: str = "config",
        output_dir: str = "output",
        mock_data_path: Optional[str] = None
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            db_config: Database configuration (None for mock mode)
            config_dir: Path to configuration directory
            output_dir: Path to output directory
            mock_data_path: Path to CSV for mock database (testing)
        """
        self.db_config = db_config
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.mock_data_path = mock_data_path
        
        # Initialize components (lazy loading)
        self._config_manager: Optional[ConfigurationManager] = None
        self._data_service: Optional[DataService] = None
        self._analysis_adapter: Optional[AnalysisAdapter] = None
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # Load configuration
        self._config_manager = ConfigurationManager(str(self.config_dir))
        self._config_manager.load_all()
        
        # Create data service
        self._data_service = DataService(self._config_manager)
        
        # Create analysis adapter
        self._analysis_adapter = AnalysisAdapter(
            self._data_service, 
            self._config_manager
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Pipeline components initialized")
    
    def _get_city_prefixes(self, city: str) -> List[str]:
        """Get cell name prefixes for city filtering."""
        prefixes = self._config_manager.get_city_prefixes(city)
        if not prefixes:
            logger.warning(f"No prefixes found for city '{city}', no city filtering applied")
        return prefixes
    
    def _fetch_data(
        self, 
        db: IDatabaseInterface, 
        params: PipelineParams
    ) -> pd.DataFrame:
        """
        Fetch data from database.
        
        Step 3-4 of workflow: Resolve inputs and retrieve data.
        """
        logger.info(f"Fetching data for KPI={params.kpi_name}, city={params.city}")
        
        city_prefixes = self._get_city_prefixes(params.city)
        
        query_params = QueryParams(
            kpi_name=params.kpi_name,
            start_datetime=params.start_datetime,
            end_datetime=params.end_datetime,
            city_prefixes=city_prefixes,
            technology=params.technology
        )
        
        df = db.fetch_kpi_data(query_params)
        logger.info(f"Fetched {len(df)} records from database")
        
        return df
    
    def _transform_data(
        self, 
        raw_data: pd.DataFrame, 
        params: PipelineParams
    ) -> pd.DataFrame:
        """
        Transform and prepare data for analysis.
        
        Step 5-6 of workflow: Aggregate and detect anomalies.
        """
        logger.info("Transforming data for analysis...")
        
        # Transform to analysis format
        df = self._data_service.transform_for_analysis(
            raw_data,
            kpi_name=params.kpi_name,
            technology=params.technology
        )
        
        # Prepare for behavioral analysis
        df = self._analysis_adapter.prepare_for_behavioral_analysis(df)
        
        logger.info(f"Transformed {len(df)} records, {df['NE'].nunique()} unique sectors")
        
        return df
    
    def _run_analysis(
        self, 
        df: pd.DataFrame,
        params: PipelineParams
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
        """
        Run behavioral and change point analysis.
        
        Step 7 of workflow: Execute analysis.
        """
        logger.info("Running behavioral analysis...")
        
        # Import the analysis module
        import sys
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from behavioral_connections import (
            select_valid_sectors,
            build_sector_data,
            find_all_connections,
            build_change_points_output,
            create_connections_json,
            create_changepoints_json
        )
        
        # Create compatible config object
        config = self._analysis_adapter.create_config_object()
        
        # Step 1: Select valid sectors
        sectors = select_valid_sectors(df, config)
        logger.info(f"Selected {len(sectors)} sectors for analysis")
        
        if len(sectors) < 2:
            raise ValueError("Need at least 2 valid sectors for analysis")
        
        # Step 2: Build sector data with change points
        sector_data = build_sector_data(df, sectors, config)
        
        # Step 3: Find connections
        connections = find_all_connections(sector_data, config)
        logger.info(f"Found {len(connections)} correlated sector pairs")
        
        # Step 4: Build change points output
        change_points = build_change_points_output(sector_data, connections)
        logger.info(f"Documented {len(change_points)} change points")
        
        # Step 5: Save outputs
        run_date_str = params.start_datetime.strftime('%Y%m%d')
        output_subdir = self.output_dir / f"run_{run_date_str}_{params.city}_{params.technology}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Connections CSV
        connections_df = pd.DataFrame(connections)
        csv_conn_path = output_subdir / config.OUTPUT_CONNECTIONS_CSV
        connections_df.to_csv(csv_conn_path, index=False)
        
        # Connections JSON
        json_conn_path = output_subdir / config.OUTPUT_CONNECTIONS_JSON
        conn_json = create_connections_json(connections, sector_data, config)
        with open(json_conn_path, 'w', encoding='utf-8') as f:
            json.dump(conn_json, f, indent=2, ensure_ascii=False)
        
        # Change points CSV (flatten aligned_sectors)
        cp_for_csv = []
        for cp in change_points:
            cp_flat = {k: v for k, v in cp.items() if k != 'aligned_sectors'}
            cp_flat['aligned_sectors'] = ';'.join([a['sector'] for a in cp['aligned_sectors']])
            cp_flat['aligned_same_direction'] = sum(1 for a in cp['aligned_sectors'] if a['same_direction'])
            cp_for_csv.append(cp_flat)
        
        change_points_df = pd.DataFrame(cp_for_csv)
        csv_cp_path = output_subdir / config.OUTPUT_CHANGEPOINTS_CSV
        change_points_df.to_csv(csv_cp_path, index=False)
        
        # Change points JSON
        json_cp_path = output_subdir / config.OUTPUT_CHANGEPOINTS_JSON
        cp_json = create_changepoints_json(change_points, sector_data, config)
        with open(json_cp_path, 'w', encoding='utf-8') as f:
            json.dump(cp_json, f, indent=2, ensure_ascii=False)
        
        output_files = {
            'connections_csv': csv_conn_path,
            'connections_json': json_conn_path,
            'changepoints_csv': csv_cp_path,
            'changepoints_json': json_cp_path
        }
        
        logger.info(f"Analysis complete. Outputs saved to {output_subdir}")
        
        return connections_df, change_points_df, output_files
    
    def _persist_results(
        self,
        db: IDatabaseInterface,
        connections_df: pd.DataFrame,
        change_points_df: pd.DataFrame,
        params: PipelineParams
    ) -> None:
        """
        Persist results back to database.
        
        Step 8 of workflow: Write to database.
        """
        logger.info("Persisting results to database...")
        
        run_date = params.start_datetime.date()
        
        # Write behavioral connections
        conn_count = db.write_behavioral_connections(connections_df, datetime.combine(run_date, datetime.min.time()))
        logger.info(f"Wrote {conn_count} behavioral connections")
        
        # Write change points
        cp_count = db.write_change_points(change_points_df, datetime.combine(run_date, datetime.min.time()))
        logger.info(f"Wrote {cp_count} change points")
    
    def run(self, params: PipelineParams) -> PipelineResult:
        """
        Execute the complete pipeline.
        
        This is the main entry point for running the analysis.
        
        Args:
            params: Pipeline input parameters
            
        Returns:
            PipelineResult with execution details
        """
        start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("BEHAVIORAL ANALYSIS PIPELINE - STARTING")
        logger.info("=" * 70)
        logger.info(f"KPI: {params.kpi_name}")
        logger.info(f"City: {params.city}")
        logger.info(f"Technology: {params.technology}")
        logger.info(f"Time range: {params.start_datetime} to {params.end_datetime}")
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Connect to database
            with DatabaseConnectionManager(
                config=self.db_config,
                mock_data_path=self.mock_data_path
            ) as db:
                # Check idempotency
                run_date = params.start_datetime.date()
                if hasattr(db, 'check_run_exists'):
                    if db.check_run_exists(
                        datetime.combine(run_date, datetime.min.time()),
                        params.kpi_name,
                        params.city
                    ):
                        logger.warning(f"Run already exists for {run_date}, {params.kpi_name}, {params.city}")
                        # Could skip or re-run based on config
                
                # Step 1: Fetch data
                raw_data = self._fetch_data(db, params)
                
                if raw_data.empty:
                    raise ValueError("No data found for given parameters")
                
                # Step 2: Transform data
                transformed_data = self._transform_data(raw_data, params)
                
                # Step 3: Run analysis
                connections_df, change_points_df, output_files = self._run_analysis(
                    transformed_data, params
                )
                
                # Step 4: Persist to database (if real database)
                if self.db_config is not None:
                    self._persist_results(db, connections_df, change_points_df, params)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = PipelineResult(
                success=True,
                run_date=datetime.combine(run_date, datetime.min.time()),
                params=params,
                connections_count=len(connections_df),
                change_points_count=len(change_points_df),
                sectors_analyzed=transformed_data['NE'].nunique(),
                execution_time_seconds=execution_time,
                output_files=output_files
            )
            
            logger.info("=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Connections: {result.connections_count}")
            logger.info(f"Change points: {result.change_points_count}")
            logger.info(f"Sectors: {result.sectors_analyzed}")
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return PipelineResult(
                success=False,
                run_date=datetime.now(),
                params=params,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def run_from_cli():
    """Run pipeline from command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Database-Driven Behavioral Analysis Pipeline'
    )
    
    parser.add_argument(
        '--kpi', '-k',
        required=True,
        help='KPI name to analyze'
    )
    parser.add_argument(
        '--city', '-c',
        required=True,
        help='City name (must match config)'
    )
    parser.add_argument(
        '--technology', '-t',
        default='4g',
        choices=['2g', '3g', '4g', '5g'],
        help='Technology (default: 4g)'
    )
    parser.add_argument(
        '--start-date', '-s',
        help='Start date (YYYY-MM-DD). Default: yesterday'
    )
    parser.add_argument(
        '--end-date', '-e',
        help='End date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--config-dir',
        default='config',
        help='Configuration directory path'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory path'
    )
    parser.add_argument(
        '--mock-data',
        help='Path to CSV file for mock database (testing)'
    )
    parser.add_argument(
        '--db-host',
        default='localhost',
        help='Database host'
    )
    parser.add_argument(
        '--db-port',
        type=int,
        default=5432,
        help='Database port'
    )
    parser.add_argument(
        '--db-name',
        default='telecom_kpi',
        help='Database name'
    )
    parser.add_argument(
        '--db-user',
        default='postgres',
        help='Database user'
    )
    parser.add_argument(
        '--db-password',
        default='',
        help='Database password'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Create pipeline parameters
    params = PipelineParams(
        kpi_name=args.kpi,
        start_datetime=start_dt,
        end_datetime=end_dt,
        city=args.city,
        technology=args.technology
    )
    
    # Create database config (if not using mock)
    db_config = None
    if not args.mock_data:
        db_config = DatabaseConfig(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password
        )
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(
        db_config=db_config,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        mock_data_path=args.mock_data
    )
    
    result = orchestrator.run(params)
    
    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example: Run with mock data for testing
    config_dir = Path(__file__).parent / "config"
    output_dir = Path(__file__).parent / "output"
    mock_data = r"C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv"
    
    # Create pipeline params
    params = PipelineParams(
        kpi_name="RSSI_PUCCH(EUCell_Eric)(CRA)",
        start_datetime=datetime(2024, 1, 1),
        end_datetime=datetime(2024, 12, 31),
        city="Karaj",
        technology="4g"
    )
    
    # Run orchestrator
    orchestrator = PipelineOrchestrator(
        config_dir=str(config_dir),
        output_dir=str(output_dir),
        mock_data_path=mock_data
    )
    
    result = orchestrator.run(params)
    
    # Print result
    print("\n" + "=" * 50)
    print("RESULT SUMMARY")
    print("=" * 50)
    print(json.dumps(result.to_dict(), indent=2, default=str))

