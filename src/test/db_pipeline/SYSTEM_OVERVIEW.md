# Database-Driven Behavioral Analysis Pipeline - System Overview

## 📦 Complete System Structure

```
db_pipeline/
├── __init__.py                 # Package initialization with convenience functions
├── db_interface.py            # Database abstraction layer (PostgreSQL)
├── data_service.py             # Data transformation and anomaly detection
├── pipeline_orchestrator.py   # Main pipeline coordinator
├── scheduler.py                # Automated scheduling and batch processing
├── test_pipeline.py            # Test script for verification
│
├── config/                     # Configuration files (YAML)
│   ├── technology_config.yaml  # Per-technology KPI definitions
│   ├── city_config.yaml        # City to cell prefix mappings
│   └── analysis_config.yaml    # Analysis parameters
│
├── sql/                        # Database schema
│   ├── create_tables.sql       # Output table definitions
│   └── useful_queries.sql     # Analytical queries
│
└── docs/                       # Documentation
    ├── README.md               # Complete documentation
    ├── QUICK_START.md          # 5-minute setup guide
    ├── IMPLEMENTATION_GUIDE.md # Step-by-step implementation
    └── SYSTEM_OVERVIEW.md     # This file
```

## 🎯 Core Components

### 1. Database Interface (`db_interface.py`)

**Purpose:** Clean abstraction for all database operations

**Key Classes:**
- `IDatabaseInterface` - Abstract interface
- `PostgreSQLInterface` - PostgreSQL implementation
- `MockDatabaseInterface` - Testing without database
- `DatabaseConnectionManager` - Context manager for connections

**Features:**
- Connection pooling
- SQL-level filtering (time, city, KPI)
- Bulk inserts for efficiency
- Batch reading for large datasets

### 2. Data Service (`data_service.py`)

**Purpose:** Transform raw data into analysis-ready format

**Key Classes:**
- `ConfigurationManager` - Loads and manages all YAML configs
- `DataService` - Data transformation and validation
- `AnalysisAdapter` - Bridges to existing behavioral_connections.py

**Features:**
- Configuration-driven anomaly detection
- Data validation
- Hourly aggregation
- Sector summary statistics

### 3. Pipeline Orchestrator (`pipeline_orchestrator.py`)

**Purpose:** Main entry point coordinating all components

**Key Classes:**
- `PipelineOrchestrator` - Main orchestrator
- `PipelineParams` - Input parameters
- `PipelineResult` - Execution results

**Workflow:**
1. Load configuration
2. Connect to database
3. Fetch KPI data
4. Transform data
5. Run behavioral analysis
6. Persist results to database

### 4. Scheduler (`scheduler.py`)

**Purpose:** Automated daily execution

**Key Classes:**
- `SchedulerConfig` - Scheduler configuration
- `SchedulerDaemon` - Background daemon process
- `JobExecutor` - Executes pipeline jobs with retry logic

**Features:**
- Daily scheduled runs
- Retry logic for failures
- Batch processing (multiple cities/KPIs)
- Integration with cron, Airflow, Kubernetes

## 🔄 Data Flow

```
┌─────────────────┐
│  PostgreSQL     │
│  (pm_2g/3g/4g/5g)│
└────────┬────────┘
         │
         │ SQL Query (filtered by time, city, KPI)
         ▼
┌─────────────────┐
│ Database        │
│ Interface       │
└────────┬────────┘
         │
         │ Raw DataFrame
         ▼
┌─────────────────┐
│ Data Service    │
│ - Transform     │
│ - Detect        │
│   Anomalies     │
└────────┬────────┘
         │
         │ Analysis-Ready DataFrame
         ▼
┌─────────────────┐
│ Behavioral      │
│ Connections     │
│ Analysis        │
│ (existing code) │
└────────┬────────┘
         │
         │ Results (connections + change points)
         ▼
┌─────────────────┐
│ Database        │
│ Interface       │
│ (write results) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PostgreSQL     │
│  (behavioral_   │
│   connections,   │
│   change_points)│
└─────────────────┘
```

## 📊 Output Tables

### 1. `behavioral_connections`
- Stores correlated sector pairs
- One row per sector pair per run
- Includes all correlation metrics

### 2. `change_points`
- Stores detected change points
- One row per change point per sector
- Includes aligned sectors information

### 3. `pipeline_runs`
- Tracks pipeline execution
- Enables idempotency
- Stores execution metrics

## ⚙️ Configuration System

### Technology Config
- Defines KPIs per technology
- Sets anomaly thresholds
- Maps KPI names to categories

### City Config
- Maps city names to cell prefixes
- Enables SQL-level filtering
- Supports timezone configuration

### Analysis Config
- Correlation thresholds
- Change point detection parameters
- Algorithm weights
- Data processing settings

## 🚀 Usage Patterns

### Pattern 1: Single Analysis (Python)

```python
from db_pipeline import run_pipeline, DatabaseConfig

result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    db_config=DatabaseConfig(...)
)
```

### Pattern 2: Batch Processing (Python)

```python
from db_pipeline import run_batch_pipeline, SchedulerConfig

config = SchedulerConfig(
    cities=["Karaj", "Tehran"],
    kpis=["RSSI_PUCCH(EUCell_Eric)(CRA)"],
    technologies=["4g"]
)

result = run_batch_pipeline(config=config)
```

### Pattern 3: Command Line

```bash
python pipeline_orchestrator.py \
    --kpi "RSSI_PUCCH(EUCell_Eric)(CRA)" \
    --city "Karaj" \
    --technology "4g"
```

### Pattern 4: Scheduled (Cron)

```bash
0 2 * * * cd /path/to/db_pipeline && python scheduler.py batch
```

## 🔧 Extension Points

### Adding New Technology

1. Add entry to `config/technology_config.yaml`
2. Ensure database table exists
3. No code changes needed!

### Adding New City

1. Add entry to `config/city_config.yaml`
2. Define cell name prefixes
3. No code changes needed!

### Adding New KPI

1. Add KPI to technology config
2. Set threshold and direction
3. Ensure data exists in database
4. No code changes needed!

### Custom Anomaly Detection

Modify `DataService._detect_anomalies()` method to add custom logic.

### Custom Analysis Algorithms

Extend `behavioral_connections.py` and integrate via `AnalysisAdapter`.

## 📈 Performance Characteristics

- **Data Volume:** Handles millions of records efficiently
- **Batch Processing:** Processes multiple cities/KPIs in parallel
- **Database:** Uses connection pooling and bulk inserts
- **Memory:** Processes data in batches for large datasets
- **Execution Time:** ~5-30 minutes per city/KPI (depends on data size)

## 🔒 Security Considerations

- Database credentials stored securely (not in code)
- SQL injection protection via parameterized queries
- Connection pooling limits resource usage
- Idempotent operations prevent duplicate data

## 📝 Key Design Decisions

1. **Configuration-Driven:** All variability in YAML files
2. **Interface-Based:** Clean separation of concerns
3. **Idempotent:** Safe to re-run without duplicates
4. **Extensible:** Easy to add new features
5. **Testable:** Mock interface for testing without database

## 🎓 Learning Path

1. **Start:** Read `QUICK_START.md` (5 minutes)
2. **Understand:** Read `README.md` (30 minutes)
3. **Implement:** Follow `IMPLEMENTATION_GUIDE.md` (5 days)
4. **Customize:** Edit configuration files
5. **Extend:** Add new features as needed

## ✅ Verification Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Database schema created (`sql/create_tables.sql`)
- [ ] Configuration files present and valid
- [ ] Test with mock data passes (`python test_pipeline.py`)
- [ ] Database connection works
- [ ] Single run completes successfully
- [ ] Results appear in database
- [ ] Scheduler configured (if using automation)

## 🎯 Success Criteria

The system is successfully implemented when:

1. ✅ Pipeline runs without errors
2. ✅ Results appear in database tables
3. ✅ Configuration changes take effect
4. ✅ Daily runs complete automatically
5. ✅ Queries return expected results
6. ✅ New KPIs/cities can be added via config only

---

**System Status:** ✅ Complete and Ready for Implementation

**Next Steps:** Follow `IMPLEMENTATION_GUIDE.md` for step-by-step setup.

