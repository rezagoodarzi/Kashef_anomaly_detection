#!/usr/bin/env python3
"""
Pipeline Scheduler
==================

Automated scheduling for the behavioral analysis pipeline.
Supports multiple execution methods:
- Standalone daemon mode
- Integration with external schedulers (cron, Airflow, etc.)

Responsibilities:
- Daily automated execution
- Retry logic for failures
- Logging and monitoring
- Multi-city/KPI batch processing

Design Principles:
- Restart-safe (idempotent operations)
- No duplicate records
- Configurable execution schedule
"""

import logging
import time
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json
import threading

# Import pipeline components
from pipeline_orchestrator import PipelineOrchestrator, PipelineParams, PipelineResult
from db_interface import DatabaseConfig
from data_service import ConfigurationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEDULER CONFIGURATION
# =============================================================================

@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""
    # Execution schedule
    run_hour: int = 2  # 2 AM
    run_minute: int = 0
    
    # Processing parameters
    lookback_days: int = 1
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 300  # 5 minutes
    
    # Batch processing
    cities: List[str] = None
    kpis: List[str] = None
    technologies: List[str] = None
    
    # Paths
    config_dir: str = "config"
    output_dir: str = "output"
    log_dir: str = "logs"
    
    # Database (None for mock mode)
    db_config: Optional[DatabaseConfig] = None
    mock_data_path: Optional[str] = None
    
    def __post_init__(self):
        if self.cities is None:
            self.cities = ["Karaj"]
        if self.kpis is None:
            self.kpis = ["RSSI_PUCCH(EUCell_Eric)(CRA)"]
        if self.technologies is None:
            self.technologies = ["4g"]


# =============================================================================
# JOB EXECUTION
# =============================================================================

@dataclass
class JobResult:
    """Result of a scheduled job execution."""
    job_id: str
    scheduled_time: datetime
    actual_start: datetime
    actual_end: datetime
    params: List[Dict[str, Any]]
    results: List[PipelineResult]
    total_success: int
    total_failed: int
    
    def to_dict(self) -> Dict:
        return {
            'job_id': self.job_id,
            'scheduled_time': self.scheduled_time.isoformat(),
            'actual_start': self.actual_start.isoformat(),
            'actual_end': self.actual_end.isoformat(),
            'total_runs': len(self.results),
            'success': self.total_success,
            'failed': self.total_failed,
            'results': [r.to_dict() for r in self.results]
        }


class JobExecutor:
    """
    Executes scheduled pipeline jobs.
    
    Handles retry logic and batch processing.
    """
    
    def __init__(self, scheduler_config: SchedulerConfig):
        self.config = scheduler_config
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup file logging for job execution."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        logger.addHandler(file_handler)
    
    def execute_single(
        self,
        kpi: str,
        city: str,
        technology: str,
        start_date: datetime,
        end_date: datetime
    ) -> PipelineResult:
        """
        Execute a single pipeline run with retry logic.
        """
        params = PipelineParams(
            kpi_name=kpi,
            start_datetime=start_date,
            end_datetime=end_date,
            city=city,
            technology=technology
        )
        
        orchestrator = PipelineOrchestrator(
            db_config=self.config.db_config,
            config_dir=self.config.config_dir,
            output_dir=self.config.output_dir,
            mock_data_path=self.config.mock_data_path
        )
        
        # Retry logic
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            logger.info(f"Attempt {attempt}/{self.config.max_retries} for {kpi}/{city}/{technology}")
            
            try:
                result = orchestrator.run(params)
                
                if result.success:
                    return result
                
                last_error = result.error_message
                logger.warning(f"Attempt {attempt} failed: {last_error}")
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt} raised exception: {e}")
            
            if attempt < self.config.max_retries:
                logger.info(f"Waiting {self.config.retry_delay_seconds}s before retry...")
                time.sleep(self.config.retry_delay_seconds)
        
        # All retries exhausted
        return PipelineResult(
            success=False,
            run_date=datetime.now(),
            params=params,
            error_message=f"All {self.config.max_retries} retries failed. Last error: {last_error}"
        )
    
    def execute_batch(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> JobResult:
        """
        Execute batch of pipeline runs for all configured KPIs/cities.
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        actual_start = datetime.now()
        
        # Default to yesterday's data
        if start_date is None:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = today - timedelta(days=self.config.lookback_days)
        if end_date is None:
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        logger.info("=" * 70)
        logger.info(f"STARTING BATCH JOB: {job_id}")
        logger.info("=" * 70)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Cities: {self.config.cities}")
        logger.info(f"KPIs: {self.config.kpis}")
        logger.info(f"Technologies: {self.config.technologies}")
        
        results = []
        params_list = []
        
        # Execute for all combinations
        total_runs = len(self.config.cities) * len(self.config.kpis) * len(self.config.technologies)
        current_run = 0
        
        for city in self.config.cities:
            for kpi in self.config.kpis:
                for tech in self.config.technologies:
                    current_run += 1
                    logger.info(f"\n[{current_run}/{total_runs}] Processing: {city}/{kpi}/{tech}")
                    
                    params_list.append({
                        'city': city,
                        'kpi': kpi,
                        'technology': tech,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    })
                    
                    result = self.execute_single(kpi, city, tech, start_date, end_date)
                    results.append(result)
                    
                    status = "✓ SUCCESS" if result.success else "✗ FAILED"
                    logger.info(f"[{current_run}/{total_runs}] {status}")
        
        actual_end = datetime.now()
        total_success = sum(1 for r in results if r.success)
        total_failed = len(results) - total_success
        
        job_result = JobResult(
            job_id=job_id,
            scheduled_time=start_date,
            actual_start=actual_start,
            actual_end=actual_end,
            params=params_list,
            results=results,
            total_success=total_success,
            total_failed=total_failed
        )
        
        # Log summary
        logger.info("\n" + "=" * 70)
        logger.info("BATCH JOB COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total runs: {len(results)}")
        logger.info(f"Successful: {total_success}")
        logger.info(f"Failed: {total_failed}")
        logger.info(f"Duration: {(actual_end - actual_start).total_seconds():.1f} seconds")
        
        # Save job result to file
        self._save_job_result(job_result)
        
        return job_result
    
    def _save_job_result(self, job_result: JobResult):
        """Save job result to JSON file."""
        log_dir = Path(self.config.log_dir)
        result_file = log_dir / f"{job_result.job_id}_result.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(job_result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Job result saved to: {result_file}")


# =============================================================================
# SCHEDULER DAEMON
# =============================================================================

class SchedulerDaemon:
    """
    Daemon process for scheduled pipeline execution.
    
    Runs continuously and triggers jobs at configured times.
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.executor = JobExecutor(config)
        self._running = False
        self._shutdown_event = threading.Event()
    
    def _should_run_now(self) -> bool:
        """Check if current time matches scheduled run time."""
        now = datetime.now()
        return now.hour == self.config.run_hour and now.minute == self.config.run_minute
    
    def _seconds_until_next_run(self) -> int:
        """Calculate seconds until next scheduled run."""
        now = datetime.now()
        next_run = now.replace(
            hour=self.config.run_hour,
            minute=self.config.run_minute,
            second=0,
            microsecond=0
        )
        
        if next_run <= now:
            next_run += timedelta(days=1)
        
        return int((next_run - now).total_seconds())
    
    def start(self):
        """Start the scheduler daemon."""
        logger.info("=" * 70)
        logger.info("SCHEDULER DAEMON STARTING")
        logger.info("=" * 70)
        logger.info(f"Scheduled run time: {self.config.run_hour:02d}:{self.config.run_minute:02d}")
        
        self._running = True
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        while self._running:
            if self._should_run_now():
                logger.info("Scheduled run time reached. Starting batch job...")
                
                try:
                    self.executor.execute_batch()
                except Exception as e:
                    logger.error(f"Batch job failed with exception: {e}")
                
                # Wait 60 seconds to avoid re-triggering in same minute
                self._shutdown_event.wait(60)
            else:
                # Check every 30 seconds
                seconds_until = self._seconds_until_next_run()
                logger.debug(f"Next run in {seconds_until} seconds")
                self._shutdown_event.wait(30)
        
        logger.info("Scheduler daemon stopped")
    
    def stop(self):
        """Stop the scheduler daemon."""
        logger.info("Stopping scheduler daemon...")
        self._running = False
        self._shutdown_event.set()
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def run_once(
    kpi: str,
    city: str,
    technology: str = "4g",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    config: Optional[SchedulerConfig] = None
) -> PipelineResult:
    """
    Run pipeline once (for external scheduler integration).
    
    Can be called from cron, Airflow, or any external scheduler.
    
    Args:
        kpi: KPI name to analyze
        city: City name
        technology: Technology (2g, 3g, 4g, 5g)
        start_date: Start datetime (default: yesterday)
        end_date: End datetime (default: today)
        config: Scheduler configuration
        
    Returns:
        PipelineResult
    """
    if config is None:
        config = SchedulerConfig()
    
    executor = JobExecutor(config)
    
    # Default dates
    if start_date is None:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = today - timedelta(days=config.lookback_days)
    if end_date is None:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    return executor.execute_single(kpi, city, technology, start_date, end_date)


def run_batch(
    config: Optional[SchedulerConfig] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> JobResult:
    """
    Run batch pipeline (for external scheduler integration).
    
    Processes all configured KPIs, cities, and technologies.
    
    Args:
        config: Scheduler configuration
        start_date: Start datetime
        end_date: End datetime
        
    Returns:
        JobResult with all execution details
    """
    if config is None:
        config = SchedulerConfig()
    
    executor = JobExecutor(config)
    return executor.execute_batch(start_date, end_date)


# =============================================================================
# AIRFLOW DAG TEMPLATE
# =============================================================================

AIRFLOW_DAG_TEMPLATE = '''
"""
Airflow DAG for Behavioral Analysis Pipeline
============================================

Place this file in your Airflow dags folder.
Adjust paths and configuration as needed.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Add pipeline to path
import sys
sys.path.append('/path/to/db_pipeline')

from scheduler import run_once, SchedulerConfig

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'behavioral_analysis_pipeline',
    default_args=default_args,
    description='Daily behavioral analysis for telecom KPIs',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['telecom', 'analysis'],
)

def run_analysis(**context):
    config = SchedulerConfig(
        config_dir='/path/to/config',
        output_dir='/path/to/output',
        db_config=DatabaseConfig(
            host='your-db-host',
            database='telecom_kpi',
            user='pipeline_user',
            password='{{ var.value.db_password }}'
        )
    )
    
    execution_date = context['execution_date']
    start_date = execution_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1)
    
    result = run_once(
        kpi='RSSI_PUCCH(EUCell_Eric)(CRA)',
        city='Karaj',
        technology='4g',
        start_date=start_date,
        end_date=end_date,
        config=config
    )
    
    if not result.success:
        raise Exception(f"Pipeline failed: {result.error_message}")
    
    return result.to_dict()

analysis_task = PythonOperator(
    task_id='run_behavioral_analysis',
    python_callable=run_analysis,
    dag=dag,
)
'''


# =============================================================================
# CLI INTERFACE
# =============================================================================

def run_cli():
    """Run scheduler from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Behavioral Analysis Pipeline Scheduler'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Daemon mode
    daemon_parser = subparsers.add_parser('daemon', help='Run as daemon')
    daemon_parser.add_argument('--hour', type=int, default=2, help='Run hour (0-23)')
    daemon_parser.add_argument('--minute', type=int, default=0, help='Run minute (0-59)')
    
    # Single run mode
    run_parser = subparsers.add_parser('run', help='Run once')
    run_parser.add_argument('--kpi', required=True, help='KPI name')
    run_parser.add_argument('--city', required=True, help='City name')
    run_parser.add_argument('--technology', default='4g', help='Technology')
    run_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    run_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Run batch')
    batch_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    batch_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    # Common arguments
    for p in [daemon_parser, run_parser, batch_parser]:
        p.add_argument('--config-dir', default='config', help='Config directory')
        p.add_argument('--output-dir', default='output', help='Output directory')
        p.add_argument('--mock-data', help='Mock data CSV path')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Create config
    config = SchedulerConfig(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        mock_data_path=getattr(args, 'mock_data', None)
    )
    
    if args.command == 'daemon':
        config.run_hour = args.hour
        config.run_minute = args.minute
        daemon = SchedulerDaemon(config)
        daemon.start()
    
    elif args.command == 'run':
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
        
        result = run_once(
            kpi=args.kpi,
            city=args.city,
            technology=args.technology,
            start_date=start_date,
            end_date=end_date,
            config=config
        )
        
        print(json.dumps(result.to_dict(), indent=2, default=str))
        sys.exit(0 if result.success else 1)
    
    elif args.command == 'batch':
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
        
        result = run_batch(config, start_date, end_date)
        
        print(json.dumps(result.to_dict(), indent=2, default=str))
        sys.exit(0 if result.total_failed == 0 else 1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_cli()

