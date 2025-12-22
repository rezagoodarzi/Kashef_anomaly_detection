#!/usr/bin/env python3
"""
Data Service Layer
==================

Handles data filtering, aggregation, and anomaly detection.
Acts as intermediary between raw database data and analysis algorithms.

Responsibilities:
- Transform raw KPI data into analysis-ready format
- Apply anomaly detection using configurable thresholds
- Aggregate data when needed
- Validate data quality

Design Principles:
- Configuration-driven behavior
- No hard-coded thresholds
- Extensible for new KPIs
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

@dataclass
class KPIConfig:
    """Configuration for a specific KPI."""
    name: str
    category: str
    anomaly_direction: str  # 'higher_worse' or 'lower_worse'
    threshold: float
    aggregation: str = "mean"
    unit: str = ""


@dataclass
class TechnologyConfig:
    """Configuration for a technology (2G, 3G, 4G, 5G)."""
    table_name: str
    description: str
    kpis: Dict[str, KPIConfig] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """Analysis parameters configuration."""
    # Correlation
    pearson_threshold: float = 0.85
    composite_threshold: float = 0.7
    
    # Change point
    pelt_penalty: float = 5.0
    min_signal_length: int = 6
    min_segment_size: int = 3
    
    # Weights
    weight_pearson: float = 0.45
    weight_diff_corr: float = 0.0
    weight_change_align: float = 0.50
    weight_direction: float = 0.05
    
    # Data processing
    min_data_points: int = 10
    max_sectors: int = 200
    anomaly_sector_priority: int = 50
    default_rssi_fill: float = -110.0
    remove_daily_pattern: bool = False
    alignment_tolerance: int = 2


class ConfigurationManager:
    """
    Loads and manages all configuration files.
    
    Provides typed access to configuration values.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing YAML config files
        """
        self.config_dir = Path(config_dir)
        self._technology_configs: Dict[str, TechnologyConfig] = {}
        self._city_configs: Dict[str, Dict] = {}
        self._analysis_config: Optional[AnalysisConfig] = None
    
    def load_all(self) -> None:
        """Load all configuration files."""
        self._load_technology_config()
        self._load_city_config()
        self._load_analysis_config()
        logger.info("All configurations loaded successfully")
    
    def _load_technology_config(self) -> None:
        """Load technology configuration from YAML."""
        config_path = self.config_dir / "technology_config.yaml"
        
        if not config_path.exists():
            logger.warning(f"Technology config not found at {config_path}, using defaults")
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Parse technologies (skip non-technology keys)
        skip_keys = {'category_mappings'}
        
        for tech_name, tech_data in raw_config.items():
            if tech_name in skip_keys or not isinstance(tech_data, dict):
                continue
            
            if 'table_name' not in tech_data:
                continue
            
            kpis = {}
            for kpi_data in tech_data.get('kpis', []):
                kpi = KPIConfig(
                    name=kpi_data['name'],
                    category=kpi_data['category'],
                    anomaly_direction=kpi_data['anomaly_direction'],
                    threshold=kpi_data['threshold'],
                    aggregation=kpi_data.get('aggregation', 'mean'),
                    unit=kpi_data.get('unit', '')
                )
                kpis[kpi.name] = kpi
            
            self._technology_configs[tech_name] = TechnologyConfig(
                table_name=tech_data['table_name'],
                description=tech_data.get('description', ''),
                kpis=kpis
            )
        
        logger.info(f"Loaded {len(self._technology_configs)} technology configurations")
    
    def _load_city_config(self) -> None:
        """Load city configuration from YAML."""
        config_path = self.config_dir / "city_config.yaml"
        
        if not config_path.exists():
            logger.warning(f"City config not found at {config_path}, using defaults")
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Parse cities (skip special keys)
        skip_keys = {'city_groups'}
        
        for city_name, city_data in raw_config.items():
            if city_name in skip_keys or not isinstance(city_data, dict):
                continue
            
            if 'prefixes' in city_data:
                self._city_configs[city_name] = city_data
        
        logger.info(f"Loaded {len(self._city_configs)} city configurations")
    
    def _load_analysis_config(self) -> None:
        """Load analysis configuration from YAML."""
        config_path = self.config_dir / "analysis_config.yaml"
        
        if not config_path.exists():
            logger.warning(f"Analysis config not found at {config_path}, using defaults")
            self._analysis_config = AnalysisConfig()
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Map YAML structure to AnalysisConfig
        corr = raw_config.get('correlation', {})
        cp = raw_config.get('change_point', {})
        weights = raw_config.get('weights', {})
        data_proc = raw_config.get('data_processing', {})
        
        self._analysis_config = AnalysisConfig(
            pearson_threshold=corr.get('pearson_threshold', 0.85),
            composite_threshold=corr.get('composite_threshold', 0.7),
            pelt_penalty=cp.get('pelt_penalty', 5.0),
            min_signal_length=cp.get('min_signal_length', 6),
            min_segment_size=cp.get('min_segment_size', 3),
            weight_pearson=weights.get('pearson', 0.45),
            weight_diff_corr=weights.get('difference_correlation', 0.0),
            weight_change_align=weights.get('change_point_alignment', 0.50),
            weight_direction=weights.get('direction_alignment', 0.05),
            min_data_points=data_proc.get('min_data_points', 10),
            max_sectors=data_proc.get('max_sectors', 200),
            anomaly_sector_priority=data_proc.get('anomaly_sector_priority', 50),
            default_rssi_fill=data_proc.get('default_rssi_fill', -110.0),
            remove_daily_pattern=data_proc.get('remove_daily_pattern', False),
            alignment_tolerance=data_proc.get('alignment_tolerance', 2)
        )
        
        logger.info("Analysis configuration loaded")
    
    def get_technology_config(self, technology: str) -> Optional[TechnologyConfig]:
        """Get configuration for a specific technology."""
        return self._technology_configs.get(technology.lower())
    
    def get_kpi_config(self, technology: str, kpi_name: str) -> Optional[KPIConfig]:
        """Get configuration for a specific KPI."""
        tech_config = self.get_technology_config(technology)
        if tech_config:
            return tech_config.kpis.get(kpi_name)
        return None
    
    def get_city_prefixes(self, city: str) -> List[str]:
        """Get cell name prefixes for a city."""
        city_data = self._city_configs.get(city, {})
        return city_data.get('prefixes', [])
    
    def get_all_cities(self) -> List[str]:
        """Get list of all configured cities."""
        return list(self._city_configs.keys())
    
    @property
    def analysis(self) -> AnalysisConfig:
        """Get analysis configuration."""
        if self._analysis_config is None:
            self._analysis_config = AnalysisConfig()
        return self._analysis_config


# =============================================================================
# DATA SERVICE
# =============================================================================

class DataService:
    """
    Service layer for data processing.
    
    Transforms raw database data into analysis-ready format.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize data service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
    
    def transform_for_analysis(
        self,
        raw_data: pd.DataFrame,
        kpi_name: str,
        technology: str = "4g"
    ) -> pd.DataFrame:
        """
        Transform raw database data into analysis-ready format.
        
        Performs:
        - Column renaming and standardization
        - Data type conversions
        - Anomaly detection based on thresholds
        
        Args:
            raw_data: Raw data from database
            kpi_name: KPI being analyzed
            technology: Technology (2g, 3g, 4g, 5g)
            
        Returns:
            Transformed DataFrame with standard columns
        """
        if raw_data.empty:
            logger.warning("Empty dataset received for transformation")
            return pd.DataFrame()
        
        df = raw_data.copy()
        
        # Standardize column names
        column_mapping = {
            'cell_name': 'NE',
            'date_time': 'Date',
            'kpi_value': 'RSSI_PUCCH(EUCell_Eric)(CRA)'  # Standard analysis column
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure datetime type
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Detect anomalies
        df = self._detect_anomalies(df, kpi_name, technology)
        
        logger.info(f"Transformed {len(df)} records for analysis")
        return df
    
    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        kpi_name: str,
        technology: str
    ) -> pd.DataFrame:
        """
        Detect anomalies based on configurable thresholds.
        
        Thresholds are KPI-specific and technology-specific.
        """
        kpi_config = self.config.get_kpi_config(technology, kpi_name)
        
        if kpi_config is None:
            logger.warning(f"No KPI config found for {kpi_name} in {technology}, using defaults")
            # Use default threshold logic
            kpi_value_col = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
            if kpi_value_col in df.columns:
                # Default: higher values indicate anomaly for RSSI
                threshold = -105.0
                df['Anomaly'] = df[kpi_value_col] > threshold
            else:
                df['Anomaly'] = False
            return df
        
        # Get the value column
        kpi_value_col = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
        if kpi_value_col not in df.columns:
            df['Anomaly'] = False
            return df
        
        # Apply threshold based on direction
        if kpi_config.anomaly_direction == 'higher_worse':
            df['Anomaly'] = df[kpi_value_col] > kpi_config.threshold
        else:  # lower_worse
            df['Anomaly'] = df[kpi_value_col] < kpi_config.threshold
        
        anomaly_count = df['Anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies ({100*anomaly_count/len(df):.1f}%) using threshold {kpi_config.threshold}")
        
        return df
    
    def aggregate_hourly(
        self,
        df: pd.DataFrame,
        value_column: str,
        cell_column: str = 'NE',
        time_column: str = 'Date'
    ) -> pd.DataFrame:
        """
        Aggregate data to hourly granularity.
        
        Args:
            df: Input DataFrame
            value_column: Column to aggregate
            cell_column: Cell identifier column
            time_column: Timestamp column
            
        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        df['hour'] = df[time_column].dt.floor('H')
        
        # Aggregate using configuration
        analysis_config = self.config.analysis
        
        aggregated = df.groupby([cell_column, 'hour']).agg({
            value_column: 'mean',
            'Anomaly': 'any'  # Any anomaly in the hour
        }).reset_index()
        
        aggregated = aggregated.rename(columns={'hour': time_column})
        
        logger.info(f"Aggregated {len(df)} records to {len(aggregated)} hourly records")
        return aggregated
    
    def validate_data(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check required columns
        if required_columns is None:
            required_columns = ['NE', 'Date', 'RSSI_PUCCH(EUCell_Eric)(CRA)', 'Anomaly']
        
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for nulls in critical columns
        for col in ['NE', 'Date']:
            if col in df.columns and df[col].isna().any():
                null_count = df[col].isna().sum()
                issues.append(f"Column '{col}' has {null_count} null values")
        
        # Check for sufficient data
        min_records = self.config.analysis.min_data_points
        sector_counts = df.groupby('NE').size() if 'NE' in df.columns else pd.Series()
        insufficient = sector_counts[sector_counts < min_records]
        if len(insufficient) > 0:
            issues.append(f"{len(insufficient)} sectors have < {min_records} data points")
        
        is_valid = len([i for i in issues if 'Missing' in i]) == 0
        
        return is_valid, issues
    
    def get_sector_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for each sector.
        
        Returns DataFrame with sector-level metrics.
        """
        if df.empty:
            return pd.DataFrame()
        
        rssi_col = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
        
        summary = df.groupby('NE').agg({
            rssi_col: ['mean', 'std', 'min', 'max', 'count'],
            'Anomaly': ['sum', 'mean'],
            'Date': ['min', 'max']
        }).round(4)
        
        # Flatten column names
        summary.columns = [
            'avg_rssi', 'std_rssi', 'min_rssi', 'max_rssi', 'record_count',
            'anomaly_count', 'anomaly_rate',
            'first_date', 'last_date'
        ]
        
        summary = summary.reset_index()
        
        return summary


# =============================================================================
# ANALYSIS ADAPTER
# =============================================================================

class AnalysisAdapter:
    """
    Adapter that connects data service to existing behavioral_connections module.
    
    Translates between database format and the format expected by 
    the existing analysis code.
    """
    
    def __init__(
        self,
        data_service: DataService,
        config_manager: ConfigurationManager
    ):
        self.data_service = data_service
        self.config = config_manager
    
    def prepare_for_behavioral_analysis(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare data for behavioral_connections.py analysis.
        
        Ensures all required columns and formats are correct.
        """
        # Validate
        is_valid, issues = self.data_service.validate_data(df)
        if not is_valid:
            logger.error(f"Data validation failed: {issues}")
            raise ValueError(f"Invalid data: {issues}")
        
        # Ensure proper column names and types
        df = df.copy()
        
        # Sort by date for time series analysis
        df = df.sort_values(['NE', 'Date'])
        
        # Fill any remaining NaN values
        rssi_col = 'RSSI_PUCCH(EUCell_Eric)(CRA)'
        if rssi_col in df.columns:
            df[rssi_col] = df[rssi_col].fillna(self.config.analysis.default_rssi_fill)
        
        return df
    
    def create_config_object(self):
        """
        Create a Config object compatible with behavioral_connections.py.
        
        Maps configuration manager settings to the expected Config class.
        """
        # Import from behavioral_connections or create compatible object
        from dataclasses import dataclass
        
        @dataclass
        class CompatibleConfig:
            """Config compatible with behavioral_connections.py"""
            PEARSON_THRESHOLD: float
            COMPOSITE_THRESHOLD: float
            PELT_PENALTY: float
            MIN_SIGNAL_LENGTH: int
            WEIGHT_PEARSON: float
            WEIGHT_DIFF_CORR: float
            WEIGHT_CHANGE_ALIGN: float
            WEIGHT_DIRECTION: float
            MIN_SECTOR_DATA_POINTS: int
            MAX_SECTORS: int
            ANOMALY_SECTOR_PRIORITY: int
            DEFAULT_RSSI_FILL: float
            REMOVE_DAILY_PATTERN: bool
            OUTPUT_CONNECTIONS_CSV: str = 'behavioral_connections.csv'
            OUTPUT_CONNECTIONS_JSON: str = 'behavioral_connections.json'
            OUTPUT_CHANGEPOINTS_CSV: str = 'change_points.csv'
            OUTPUT_CHANGEPOINTS_JSON: str = 'change_points.json'
        
        analysis = self.config.analysis
        
        return CompatibleConfig(
            PEARSON_THRESHOLD=analysis.pearson_threshold,
            COMPOSITE_THRESHOLD=analysis.composite_threshold,
            PELT_PENALTY=analysis.pelt_penalty,
            MIN_SIGNAL_LENGTH=analysis.min_signal_length,
            WEIGHT_PEARSON=analysis.weight_pearson,
            WEIGHT_DIFF_CORR=analysis.weight_diff_corr,
            WEIGHT_CHANGE_ALIGN=analysis.weight_change_align,
            WEIGHT_DIRECTION=analysis.weight_direction,
            MIN_SECTOR_DATA_POINTS=analysis.min_data_points,
            MAX_SECTORS=analysis.max_sectors,
            ANOMALY_SECTOR_PRIORITY=analysis.anomaly_sector_priority,
            DEFAULT_RSSI_FILL=analysis.default_rssi_fill,
            REMOVE_DAILY_PATTERN=analysis.remove_daily_pattern
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_data_service(config_dir: str = "config") -> Tuple[DataService, ConfigurationManager]:
    """
    Factory function to create data service with configuration.
    
    Args:
        config_dir: Path to configuration directory
        
    Returns:
        Tuple of (DataService, ConfigurationManager)
    """
    config_manager = ConfigurationManager(config_dir)
    config_manager.load_all()
    
    data_service = DataService(config_manager)
    
    return data_service, config_manager


if __name__ == "__main__":
    # Test configuration loading
    logging.basicConfig(level=logging.INFO)
    
    config_dir = Path(__file__).parent / "config"
    
    try:
        data_service, config = create_data_service(str(config_dir))
        
        print("\n=== Technology Configurations ===")
        for tech in ['2g', '3g', '4g', '5g']:
            tech_config = config.get_technology_config(tech)
            if tech_config:
                print(f"\n{tech.upper()}: {tech_config.table_name}")
                for kpi_name, kpi in tech_config.kpis.items():
                    print(f"  - {kpi_name}: threshold={kpi.threshold}, direction={kpi.anomaly_direction}")
        
        print("\n=== City Configurations ===")
        for city in config.get_all_cities():
            prefixes = config.get_city_prefixes(city)
            print(f"  {city}: {prefixes}")
        
        print("\n=== Analysis Configuration ===")
        analysis = config.analysis
        print(f"  Pearson threshold: {analysis.pearson_threshold}")
        print(f"  Composite threshold: {analysis.composite_threshold}")
        print(f"  PELT penalty: {analysis.pelt_penalty}")
        print(f"  Weights: pearson={analysis.weight_pearson}, change_align={analysis.weight_change_align}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

