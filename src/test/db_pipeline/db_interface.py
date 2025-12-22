#!/usr/bin/env python3
"""
Database Interface Layer
========================

Provides a clean abstraction for all database operations.
No business logic directly accesses PostgreSQL - all operations go through this interface.

Responsibilities:
- Manage database connections with connection pooling
- Fetch KPI data with SQL-level filtering (time, city, KPI)
- Write processed results back to database
- Handle batching for large datasets

Design Principles:
- Interface-driven design
- Clear separation from business logic
- Configurable and extensible
"""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Generator, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# PostgreSQL driver
try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor, execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

# SQLAlchemy for ORM-style operations (optional)
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "telecom_kpi"
    user: str = "postgres"
    password: str = ""
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: int = 30
    query_timeout: int = 300

    def get_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def get_dsn(self) -> str:
        """Generate DSN for psycopg2."""
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"


@dataclass
class QueryParams:
    """Parameters for KPI data queries."""
    kpi_name: str
    start_datetime: datetime
    end_datetime: datetime
    city_prefixes: List[str] = field(default_factory=list)
    technology: str = "4g"
    aggregation_type: str = "HOURLY"


# =============================================================================
# ABSTRACT DATABASE INTERFACE
# =============================================================================

class IDatabaseInterface(ABC):
    """
    Abstract interface for database operations.
    
    Implementations can support different databases or ORMs
    while maintaining the same contract.
    """
    
    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    def fetch_kpi_data(self, params: QueryParams) -> pd.DataFrame:
        """
        Fetch KPI data from database.
        
        Args:
            params: Query parameters
            
        Returns:
            DataFrame with columns: cell_name, date_time, kpi_value
        """
        pass
    
    @abstractmethod
    def fetch_kpi_data_batched(
        self, 
        params: QueryParams, 
        batch_size: int = 10000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Fetch KPI data in batches for large datasets.
        
        Yields:
            DataFrames with batch_size rows each
        """
        pass
    
    @abstractmethod
    def write_behavioral_connections(
        self, 
        data: pd.DataFrame,
        run_date: datetime
    ) -> int:
        """
        Write behavioral connections to database.
        
        Args:
            data: DataFrame with connection data
            run_date: Date of the analysis run
            
        Returns:
            Number of rows inserted
        """
        pass
    
    @abstractmethod
    def write_change_points(
        self, 
        data: pd.DataFrame,
        run_date: datetime
    ) -> int:
        """
        Write change points to database.
        
        Args:
            data: DataFrame with change point data
            run_date: Date of the analysis run
            
        Returns:
            Number of rows inserted
        """
        pass
    
    @abstractmethod
    def check_run_exists(self, run_date: datetime, kpi_name: str, city: str) -> bool:
        """
        Check if a run already exists for given parameters (idempotency).
        
        Args:
            run_date: Date of the run
            kpi_name: KPI analyzed
            city: City analyzed
            
        Returns:
            True if run exists, False otherwise
        """
        pass


# =============================================================================
# POSTGRESQL IMPLEMENTATION
# =============================================================================

class PostgreSQLInterface(IDatabaseInterface):
    """
    PostgreSQL implementation of database interface.
    
    Uses connection pooling for efficient resource management.
    Supports both raw psycopg2 and SQLAlchemy engines.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize PostgreSQL interface.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._engine = None
        
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            )
    
    def connect(self) -> None:
        """Establish connection pool."""
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                connect_timeout=self.config.connection_timeout
            )
            logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}")
            
            # Optionally create SQLAlchemy engine for pandas operations
            if SQLALCHEMY_AVAILABLE:
                self._engine = create_engine(
                    self.config.get_connection_string(),
                    poolclass=QueuePool,
                    pool_size=self.config.max_connections,
                    pool_pre_ping=True
                )
                
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Disconnected from PostgreSQL")
        
        if self._engine:
            self._engine.dispose()
            self._engine = None
    
    @contextmanager
    def _get_connection(self):
        """Context manager for connection handling."""
        if not self._pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)
    
    def _build_city_filter(self, prefixes: List[str], column: str = "cell_name") -> Tuple[str, List[str]]:
        """
        Build SQL WHERE clause for city filtering.
        
        Args:
            prefixes: List of cell name prefixes
            column: Column name to filter
            
        Returns:
            Tuple of (SQL clause, parameters)
        """
        if not prefixes:
            return "", []
        
        conditions = []
        params = []
        for prefix in prefixes:
            conditions.append(f"{column} LIKE %s")
            params.append(f"{prefix}%")
        
        clause = f"AND ({' OR '.join(conditions)})"
        return clause, params
    
    def fetch_kpi_data(self, params: QueryParams) -> pd.DataFrame:
        """
        Fetch KPI data from PostgreSQL.
        
        Applies SQL-level filtering for efficiency.
        """
        # Determine table based on technology
        table_mapping = {
            "2g": "pm_2g",
            "3g": "pm_3g", 
            "4g": "pm_4g",
            "5g": "pm_5g"
        }
        table_name = table_mapping.get(params.technology.lower(), "pm_4g")
        
        # Build city filter
        city_clause, city_params = self._build_city_filter(params.city_prefixes)
        
        # Build query
        query = f"""
            SELECT 
                cell_name,
                date_time,
                kpi_name,
                kpi_value,
                kpi_category
            FROM {table_name}
            WHERE kpi_name = %s
              AND date_time >= %s
              AND date_time < %s
              AND agg_type = %s
              {city_clause}
            ORDER BY date_time, cell_name
        """
        
        # Combine parameters
        query_params = [
            params.kpi_name,
            params.start_datetime,
            params.end_datetime,
            params.aggregation_type
        ] + city_params
        
        # Execute query
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=query_params)
        
        logger.info(f"Fetched {len(df)} records from {table_name}")
        return df
    
    def fetch_kpi_data_batched(
        self, 
        params: QueryParams, 
        batch_size: int = 10000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Fetch KPI data in batches using server-side cursor.
        
        Efficient for large datasets - doesn't load all data into memory.
        """
        table_mapping = {
            "2g": "pm_2g",
            "3g": "pm_3g",
            "4g": "pm_4g",
            "5g": "pm_5g"
        }
        table_name = table_mapping.get(params.technology.lower(), "pm_4g")
        
        city_clause, city_params = self._build_city_filter(params.city_prefixes)
        
        query = f"""
            SELECT 
                cell_name,
                date_time,
                kpi_name,
                kpi_value,
                kpi_category
            FROM {table_name}
            WHERE kpi_name = %s
              AND date_time >= %s
              AND date_time < %s
              AND agg_type = %s
              {city_clause}
            ORDER BY date_time, cell_name
        """
        
        query_params = [
            params.kpi_name,
            params.start_datetime,
            params.end_datetime,
            params.aggregation_type
        ] + city_params
        
        with self._get_connection() as conn:
            # Use server-side cursor for memory efficiency
            cursor_name = f"fetch_kpi_{datetime.now().timestamp()}"
            with conn.cursor(name=cursor_name) as cursor:
                cursor.execute(query, query_params)
                
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    
                    df = pd.DataFrame(
                        rows,
                        columns=['cell_name', 'date_time', 'kpi_name', 'kpi_value', 'kpi_category']
                    )
                    yield df
    
    def write_behavioral_connections(
        self, 
        data: pd.DataFrame,
        run_date: datetime
    ) -> int:
        """
        Write behavioral connections to database.
        
        Uses bulk insert for efficiency.
        Idempotent: deletes existing data for same run_date before inserting.
        """
        if data.empty:
            logger.warning("No behavioral connections to write")
            return 0
        
        # Add run metadata
        data = data.copy()
        data['run_date'] = run_date
        data['created_at'] = datetime.now()
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Delete existing data for this run (idempotency)
                cursor.execute(
                    "DELETE FROM behavioral_connections WHERE run_date = %s",
                    (run_date,)
                )
                
                # Prepare columns and values
                columns = [
                    'run_date', 'sector_1', 'sector_2', 'composite_score',
                    'pearson_correlation', 'pearson_p_value', 'difference_correlation',
                    'change_point_alignment', 'direction_alignment',
                    'sector_1_change_points', 'sector_2_change_points',
                    'sector_1_has_anomaly', 'sector_2_has_anomaly',
                    'sector_1_avg_rssi', 'sector_2_avg_rssi', 'both_anomaly',
                    'created_at'
                ]
                
                # Filter to existing columns
                existing_cols = [c for c in columns if c in data.columns]
                
                # Build insert query
                insert_query = f"""
                    INSERT INTO behavioral_connections ({', '.join(existing_cols)})
                    VALUES %s
                """
                
                # Prepare values
                values = [tuple(row[c] for c in existing_cols) for _, row in data.iterrows()]
                
                # Bulk insert
                execute_values(cursor, insert_query, values, page_size=1000)
                
        logger.info(f"Wrote {len(data)} behavioral connections for run_date={run_date}")
        return len(data)
    
    def write_change_points(
        self, 
        data: pd.DataFrame,
        run_date: datetime
    ) -> int:
        """
        Write change points to database.
        
        Uses bulk insert for efficiency.
        Idempotent: deletes existing data for same run_date before inserting.
        """
        if data.empty:
            logger.warning("No change points to write")
            return 0
        
        data = data.copy()
        data['run_date'] = run_date
        data['created_at'] = datetime.now()
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Delete existing data for this run (idempotency)
                cursor.execute(
                    "DELETE FROM change_points WHERE run_date = %s",
                    (run_date,)
                )
                
                columns = [
                    'run_date', 'sector', 'timestamp', 'change_point_index',
                    'value_at_change', 'value_before', 'value_after',
                    'magnitude', 'direction', 'has_anomaly',
                    'num_aligned_sectors', 'aligned_sectors', 'aligned_same_direction',
                    'created_at'
                ]
                
                existing_cols = [c for c in columns if c in data.columns]
                
                insert_query = f"""
                    INSERT INTO change_points ({', '.join(existing_cols)})
                    VALUES %s
                """
                
                values = [tuple(row[c] for c in existing_cols) for _, row in data.iterrows()]
                execute_values(cursor, insert_query, values, page_size=1000)
                
        logger.info(f"Wrote {len(data)} change points for run_date={run_date}")
        return len(data)
    
    def check_run_exists(self, run_date: datetime, kpi_name: str, city: str) -> bool:
        """Check if analysis already ran for given parameters."""
        query = """
            SELECT COUNT(*) 
            FROM pipeline_runs 
            WHERE run_date = %s AND kpi_name = %s AND city = %s
        """
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (run_date, kpi_name, city))
                count = cursor.fetchone()[0]
        
        return count > 0
    
    def record_run(
        self, 
        run_date: datetime, 
        kpi_name: str, 
        city: str,
        status: str = "completed",
        metrics: Optional[Dict] = None
    ) -> None:
        """Record a pipeline run for idempotency tracking."""
        query = """
            INSERT INTO pipeline_runs (run_date, kpi_name, city, status, metrics, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_date, kpi_name, city) 
            DO UPDATE SET status = %s, metrics = %s, updated_at = %s
        """
        
        import json
        metrics_json = json.dumps(metrics) if metrics else None
        now = datetime.now()
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (
                    run_date, kpi_name, city, status, metrics_json, now,
                    status, metrics_json, now
                ))


# =============================================================================
# MOCK INTERFACE FOR TESTING
# =============================================================================

class MockDatabaseInterface(IDatabaseInterface):
    """
    Mock database interface for testing without a real database.
    
    Loads data from CSV files to simulate database queries.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize mock interface with path to CSV data.
        
        Args:
            data_path: Path to CSV file with KPI data
        """
        self.data_path = data_path
        self._data: Optional[pd.DataFrame] = None
        self._connections_written: List[pd.DataFrame] = []
        self._changepoints_written: List[pd.DataFrame] = []
    
    def connect(self) -> None:
        """Load CSV data."""
        self._data = pd.read_csv(self.data_path)
        if 'Date' in self._data.columns:
            self._data['date_time'] = pd.to_datetime(self._data['Date'])
        if 'NE' in self._data.columns:
            self._data['cell_name'] = self._data['NE']
        logger.info(f"MockDB: Loaded {len(self._data)} records from {self.data_path}")
    
    def disconnect(self) -> None:
        """Clear loaded data."""
        self._data = None
        logger.info("MockDB: Disconnected")
    
    def fetch_kpi_data(self, params: QueryParams) -> pd.DataFrame:
        """Filter CSV data based on parameters."""
        if self._data is None:
            raise RuntimeError("Database not connected")
        
        df = self._data.copy()
        
        # Apply time filter
        if 'date_time' in df.columns:
            df = df[
                (df['date_time'] >= params.start_datetime) &
                (df['date_time'] < params.end_datetime)
            ]
        
        # Apply city filter
        if params.city_prefixes and 'cell_name' in df.columns:
            mask = df['cell_name'].str.startswith(tuple(params.city_prefixes))
            df = df[mask]
        
        return df
    
    def fetch_kpi_data_batched(
        self, 
        params: QueryParams, 
        batch_size: int = 10000
    ) -> Generator[pd.DataFrame, None, None]:
        """Yield data in batches."""
        df = self.fetch_kpi_data(params)
        for i in range(0, len(df), batch_size):
            yield df.iloc[i:i+batch_size]
    
    def write_behavioral_connections(
        self, 
        data: pd.DataFrame,
        run_date: datetime
    ) -> int:
        """Store connections in memory."""
        self._connections_written.append(data)
        logger.info(f"MockDB: Stored {len(data)} connections")
        return len(data)
    
    def write_change_points(
        self, 
        data: pd.DataFrame,
        run_date: datetime
    ) -> int:
        """Store change points in memory."""
        self._changepoints_written.append(data)
        logger.info(f"MockDB: Stored {len(data)} change points")
        return len(data)
    
    def check_run_exists(self, run_date: datetime, kpi_name: str, city: str) -> bool:
        """Always return False for testing."""
        return False


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_database_interface(
    config: Optional[DatabaseConfig] = None,
    mock_data_path: Optional[str] = None
) -> IDatabaseInterface:
    """
    Factory function to create appropriate database interface.
    
    Args:
        config: Database configuration (for PostgreSQL)
        mock_data_path: Path to CSV file (for mock interface)
        
    Returns:
        Database interface instance
    """
    if mock_data_path:
        return MockDatabaseInterface(mock_data_path)
    
    if config is None:
        config = DatabaseConfig()
    
    return PostgreSQLInterface(config)


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

class DatabaseConnectionManager:
    """
    Context manager for database connections.
    
    Usage:
        with DatabaseConnectionManager(config) as db:
            data = db.fetch_kpi_data(params)
    """
    
    def __init__(
        self, 
        config: Optional[DatabaseConfig] = None,
        mock_data_path: Optional[str] = None
    ):
        self.config = config
        self.mock_data_path = mock_data_path
        self._interface: Optional[IDatabaseInterface] = None
    
    def __enter__(self) -> IDatabaseInterface:
        self._interface = create_database_interface(
            config=self.config,
            mock_data_path=self.mock_data_path
        )
        self._interface.connect()
        return self._interface
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._interface:
            self._interface.disconnect()
        return False


if __name__ == "__main__":
    # Test with mock interface
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    mock_path = r"C:\Users\USER\Desktop\kasheftest\Git_hub\Kashef_anomaly_detection\data\process\KJ_finalData_102.csv"
    
    with DatabaseConnectionManager(mock_data_path=mock_path) as db:
        params = QueryParams(
            kpi_name="RSSI_PUCCH(EUCell_Eric)(CRA)",
            start_datetime=datetime(2024, 1, 1),
            end_datetime=datetime(2024, 12, 31),
            city_prefixes=["KJ"],
            technology="4g"
        )
        
        data = db.fetch_kpi_data(params)
        print(f"Fetched {len(data)} records")
        print(data.head())

