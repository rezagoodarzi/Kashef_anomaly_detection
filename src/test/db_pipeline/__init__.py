"""
Database-Driven Behavioral Analysis Pipeline
=============================================

A clean, scalable system for detecting behavioral patterns in telecom KPI data.

Architecture:
- Configuration Layer: YAML-based configs for technologies, cities, and analysis
- Database Interface: PostgreSQL connection management and queries
- Data Service: Data transformation and anomaly detection
- Pipeline Orchestrator: Main entry point coordinating all components
- Scheduler: Automated daily execution with retry logic

Usage:
    # Run single analysis
    from db_pipeline import run_pipeline
    result = run_pipeline(
        kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
        city="Karaj",
        technology="4g"
    )
    
    # Run batch analysis
    from db_pipeline import run_batch_pipeline
    results = run_batch_pipeline(
        cities=["Karaj", "Tehran"],
        kpis=["RSSI_PUCCH(EUCell_Eric)(CRA)"],
        technologies=["4g"]
    )

For detailed documentation, see README.md
"""

__version__ = "1.0.0"
__author__ = "Kashef Team"

# Core components
from .db_interface import (
    IDatabaseInterface,
    PostgreSQLInterface,
    MockDatabaseInterface,
    DatabaseConfig,
    QueryParams,
    DatabaseConnectionManager,
    create_database_interface
)

from .data_service import (
    ConfigurationManager,
    DataService,
    AnalysisAdapter,
    KPIConfig,
    TechnologyConfig,
    AnalysisConfig,
    create_data_service
)

from .pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineParams,
    PipelineResult
)

from .scheduler import (
    SchedulerConfig,
    SchedulerDaemon,
    JobExecutor,
    JobResult,
    run_once,
    run_batch
)


# Convenience functions
def run_pipeline(
    kpi: str,
    city: str,
    technology: str = "4g",
    start_date=None,
    end_date=None,
    config_dir: str = "config",
    output_dir: str = "output",
    db_config=None,
    mock_data_path: str = None
):
    """
    Run behavioral analysis pipeline.
    
    Args:
        kpi: KPI name to analyze
        city: City name (must match config)
        technology: Technology (2g, 3g, 4g, 5g)
        start_date: Start datetime
        end_date: End datetime
        config_dir: Path to configuration directory
        output_dir: Path to output directory
        db_config: DatabaseConfig for PostgreSQL (None for mock mode)
        mock_data_path: Path to CSV for mock database
        
    Returns:
        PipelineResult with execution details
    """
    from datetime import datetime, timedelta
    
    # Default dates
    if start_date is None:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = today - timedelta(days=1)
    if end_date is None:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    params = PipelineParams(
        kpi_name=kpi,
        start_datetime=start_date,
        end_datetime=end_date,
        city=city,
        technology=technology
    )
    
    orchestrator = PipelineOrchestrator(
        db_config=db_config,
        config_dir=config_dir,
        output_dir=output_dir,
        mock_data_path=mock_data_path
    )
    
    return orchestrator.run(params)


def run_batch_pipeline(
    cities=None,
    kpis=None,
    technologies=None,
    start_date=None,
    end_date=None,
    config_dir: str = "config",
    output_dir: str = "output",
    db_config=None,
    mock_data_path: str = None
):
    """
    Run batch behavioral analysis for multiple cities/KPIs.
    
    Args:
        cities: List of city names
        kpis: List of KPI names
        technologies: List of technologies
        start_date: Start datetime
        end_date: End datetime
        config_dir: Path to configuration directory
        output_dir: Path to output directory
        db_config: DatabaseConfig for PostgreSQL
        mock_data_path: Path to CSV for mock database
        
    Returns:
        JobResult with all execution details
    """
    config = SchedulerConfig(
        cities=cities or ["Karaj"],
        kpis=kpis or ["RSSI_PUCCH(EUCell_Eric)(CRA)"],
        technologies=technologies or ["4g"],
        config_dir=config_dir,
        output_dir=output_dir,
        db_config=db_config,
        mock_data_path=mock_data_path
    )
    
    return run_batch(config, start_date, end_date)


__all__ = [
    # Version
    '__version__',
    
    # Database Interface
    'IDatabaseInterface',
    'PostgreSQLInterface',
    'MockDatabaseInterface',
    'DatabaseConfig',
    'QueryParams',
    'DatabaseConnectionManager',
    'create_database_interface',
    
    # Data Service
    'ConfigurationManager',
    'DataService',
    'AnalysisAdapter',
    'KPIConfig',
    'TechnologyConfig',
    'AnalysisConfig',
    'create_data_service',
    
    # Pipeline
    'PipelineOrchestrator',
    'PipelineParams',
    'PipelineResult',
    
    # Scheduler
    'SchedulerConfig',
    'SchedulerDaemon',
    'JobExecutor',
    'JobResult',
    'run_once',
    'run_batch',
    
    # Convenience
    'run_pipeline',
    'run_batch_pipeline',
]

