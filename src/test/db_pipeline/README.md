# Database-Driven Behavioral Analysis Pipeline

A clean, scalable, and production-ready system for detecting behavioral patterns in telecom KPI data from PostgreSQL databases.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [Scheduling](#scheduling)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This pipeline connects to PostgreSQL databases, retrieves KPI data, detects anomalies, performs behavioral and change-point analysis, and stores results back into the database. It's designed to run **once per day** in an automated manner.

### Key Features

- ✅ **Configuration-Driven**: All behavior controlled via YAML files
- ✅ **Interface-Based Design**: Clean separation between database and business logic
- ✅ **Multi-Algorithm Analysis**: Pearson correlation, change point detection, direction alignment
- ✅ **Idempotent**: Safe to re-run without duplicates
- ✅ **Extensible**: Easy to add new KPIs, cities, and technologies
- ✅ **Production-Ready**: Comprehensive logging, error handling, retry logic

---

## 🏗️ Architecture

```
┌────────────────────────────┐
│        Scheduler            │  (Daily execution)
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│   Pipeline Orchestrator    │  (Main entry point)
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│      Data Service Layer    │  (Filtering, aggregation, anomaly detection)
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│   Database Interface Layer │  (PostgreSQL connection & queries)
└────────────────────────────┘
```

### Components

1. **Configuration Layer** (`config/`)
   - Technology configurations (2G, 3G, 4G, 5G)
   - City mappings (cell name prefixes)
   - Analysis parameters (thresholds, weights)

2. **Database Interface** (`db_interface.py`)
   - PostgreSQL connection pooling
   - SQL-level filtering (time, city, KPI)
   - Bulk inserts for results

3. **Data Service** (`data_service.py`)
   - Data transformation
   - Anomaly detection (configurable thresholds)
   - Data validation

4. **Pipeline Orchestrator** (`pipeline_orchestrator.py`)
   - Coordinates all components
   - Manages execution flow
   - Error handling

5. **Scheduler** (`scheduler.py`)
   - Daily automated execution
   - Retry logic
   - Batch processing

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Database Schema

```bash
psql -U postgres -d telecom_kpi -f sql/create_tables.sql
```

### 3. Configure System

Edit configuration files in `config/`:

- `technology_config.yaml` - Define KPIs and thresholds per technology
- `city_config.yaml` - Map cities to cell name prefixes
- `analysis_config.yaml` - Set analysis parameters

### 4. Run Pipeline

**Single Run (Mock Mode - for testing):**
```python
from db_pipeline import run_pipeline
from datetime import datetime, timedelta

result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    mock_data_path="path/to/data.csv"
)
```

**Single Run (Database Mode):**
```python
from db_pipeline import run_pipeline, DatabaseConfig

db_config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="telecom_kpi",
    user="pipeline_user",
    password="your_password"
)

result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    db_config=db_config
)
```

**Command Line:**
```bash
python pipeline_orchestrator.py \
    --kpi "RSSI_PUCCH(EUCell_Eric)(CRA)" \
    --city "Karaj" \
    --technology "4g" \
    --start-date "2024-01-01" \
    --end-date "2024-01-02"
```

---

## ⚙️ Configuration

### Technology Configuration

File: `config/technology_config.yaml`

Defines per-technology behavior:

```yaml
4g:
  table_name: "pm_4g"
  description: "4G LTE technology metrics"
  kpis:
    - name: "RSSI_PUCCH(EUCell_Eric)"
      category: "PUCCH RSSI"
      anomaly_direction: "higher_worse"  # or "lower_worse"
      threshold: -105.0
      aggregation: "mean"
      unit: "dBm"
```

### City Configuration

File: `config/city_config.yaml`

Maps city names to cell name prefixes:

```yaml
Karaj:
  prefixes:
    - "KJ"
    - "KRJ"
  timezone: "Asia/Tehran"
  region: "Alborz"
```

### Analysis Configuration

File: `config/analysis_config.yaml`

Core analysis parameters:

```yaml
correlation:
  pearson_threshold: 0.85
  composite_threshold: 0.7

change_point:
  pelt_penalty: 5.0
  min_signal_length: 6

weights:
  pearson: 0.45
  change_point_alignment: 0.50
  direction_alignment: 0.05
```

---

## 📖 Usage

### Python API

#### Single Analysis

```python
from db_pipeline import run_pipeline, DatabaseConfig
from datetime import datetime, timedelta

# With database
db_config = DatabaseConfig(
    host="localhost",
    database="telecom_kpi",
    user="user",
    password="pass"
)

result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    db_config=db_config
)

if result.success:
    print(f"Found {result.connections_count} connections")
    print(f"Found {result.change_points_count} change points")
else:
    print(f"Error: {result.error_message}")
```

#### Batch Processing

```python
from db_pipeline import run_batch_pipeline, DatabaseConfig

result = run_batch_pipeline(
    cities=["Karaj", "Tehran"],
    kpis=["RSSI_PUCCH(EUCell_Eric)(CRA)"],
    technologies=["4g"],
    db_config=db_config
)

print(f"Total runs: {len(result.results)}")
print(f"Successful: {result.total_success}")
print(f"Failed: {result.total_failed}")
```

### Command Line

#### Single Run

```bash
python pipeline_orchestrator.py \
    --kpi "RSSI_PUCCH(EUCell_Eric)(CRA)" \
    --city "Karaj" \
    --technology "4g" \
    --start-date "2024-01-01" \
    --end-date "2024-01-02" \
    --db-host "localhost" \
    --db-name "telecom_kpi" \
    --db-user "user" \
    --db-password "pass"
```

#### Scheduler Commands

```bash
# Run once
python scheduler.py run \
    --kpi "RSSI_PUCCH(EUCell_Eric)(CRA)" \
    --city "Karaj" \
    --technology "4g"

# Run batch
python scheduler.py batch

# Run as daemon (scheduled daily at 2 AM)
python scheduler.py daemon --hour 2 --minute 0
```

---

## 🗄️ Database Schema

### Input Tables

The pipeline reads from technology-specific tables:

- `pm_2g`
- `pm_3g`
- `pm_4g`
- `pm_5g`

**Common Schema:**
```sql
CREATE TABLE pm_4g (
    id INT4 PRIMARY KEY,
    cell_name VARCHAR,
    date_time TIMESTAMP,
    agg_type VARCHAR(50),
    kpi_name VARCHAR(100),
    kpi_value FLOAT8,
    kpi_category VARCHAR(20)
);
```

### Output Tables

#### 1. `behavioral_connections`

Stores correlated sector pairs:

```sql
CREATE TABLE behavioral_connections (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    sector_1 VARCHAR(100) NOT NULL,
    sector_2 VARCHAR(100) NOT NULL,
    composite_score FLOAT8 NOT NULL,
    pearson_correlation FLOAT8,
    change_point_alignment FLOAT8,
    direction_alignment FLOAT8,
    -- ... more columns
    CONSTRAINT uq_behavioral_connections_pair 
        UNIQUE (run_date, sector_1, sector_2)
);
```

#### 2. `change_points`

Stores detected change points:

```sql
CREATE TABLE change_points (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    sector VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    change_point_index INT NOT NULL,
    value_at_change FLOAT8,
    magnitude FLOAT8,
    direction VARCHAR(10),
    num_aligned_sectors INT,
    -- ... more columns
);
```

#### 3. `pipeline_runs`

Tracks pipeline execution (idempotency):

```sql
CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    kpi_name VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    status VARCHAR(20),
    metrics JSONB,
    -- ... more columns
    CONSTRAINT uq_pipeline_runs 
        UNIQUE (run_date, kpi_name, city)
);
```

**See `sql/create_tables.sql` for complete schema.**

---

## ⏰ Scheduling

### Option 1: Cron Job

```bash
# Edit crontab
crontab -e

# Add line (runs daily at 2 AM)
0 2 * * * cd /path/to/db_pipeline && python scheduler.py batch
```

### Option 2: Scheduler Daemon

```bash
# Run as background daemon
python scheduler.py daemon --hour 2 --minute 0 &

# Or use systemd service
sudo systemctl start behavioral-pipeline
```

### Option 3: Airflow DAG

See `scheduler.py` for Airflow DAG template. Place in your Airflow `dags/` folder.

### Option 4: Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: behavioral-analysis
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: pipeline
            image: your-image:latest
            command: ["python", "scheduler.py", "batch"]
          restartPolicy: OnFailure
```

---

## 🔍 Querying Results

### Useful Queries

See `sql/useful_queries.sql` for comprehensive examples.

**Top Correlated Pairs:**
```sql
SELECT sector_1, sector_2, composite_score, change_point_alignment
FROM behavioral_connections
WHERE run_date = CURRENT_DATE - INTERVAL '1 day'
ORDER BY composite_score DESC
LIMIT 20;
```

**Synchronized Change Points:**
```sql
SELECT timestamp, direction, COUNT(DISTINCT sector) as affected_sectors
FROM change_points
WHERE run_date = CURRENT_DATE - INTERVAL '1 day'
  AND num_aligned_sectors > 2
GROUP BY timestamp, direction
ORDER BY affected_sectors DESC;
```

**Pipeline Execution History:**
```sql
SELECT run_date, kpi_name, city, status, 
       metrics->>'connections_count' as connections
FROM pipeline_runs
ORDER BY run_date DESC
LIMIT 10;
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Database Connection Error

**Error:** `psycopg2.OperationalError: could not connect to server`

**Solution:**
- Check database is running: `pg_isready -h localhost`
- Verify credentials in `DatabaseConfig`
- Check firewall/network settings

#### 2. No Data Found

**Error:** `No data found for given parameters`

**Solution:**
- Verify KPI name matches database exactly
- Check date range has data
- Verify city prefixes in `city_config.yaml` match cell names

#### 3. Permission Denied

**Error:** `Permission denied: 'change_points.csv'`

**Solution:**
- Close any programs with file open (Excel, VS Code)
- Check file permissions
- Delete output directory and re-run

#### 4. Configuration Not Found

**Error:** `Technology config not found`

**Solution:**
- Ensure `config/` directory exists
- Check YAML syntax is valid
- Verify file paths in code

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing with Mock Data

Use mock database interface for testing:

```python
result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    mock_data_path="path/to/test_data.csv"
)
```

---

## 📚 Additional Documentation

- **Technical Design**: See the technical design document for architecture details
- **SQL Queries**: `sql/useful_queries.sql` for analytical queries
- **API Reference**: See docstrings in each module

---

## 🔧 Extending the System

### Adding a New Technology

1. Add entry to `config/technology_config.yaml`:
```yaml
6g:
  table_name: "pm_6g"
  kpis:
    - name: "NEW_KPI"
      threshold: 100.0
      # ...
```

2. Ensure database table exists with correct schema

### Adding a New City

1. Add entry to `config/city_config.yaml`:
```yaml
NewCity:
  prefixes:
    - "NC"
  timezone: "Asia/Tehran"
```

### Adding a New KPI

1. Add KPI to technology config
2. Ensure data exists in database
3. Run pipeline with new KPI name

---

## 📝 License

[Your License Here]

---

## 👥 Contributors

[Your Team Here]

---

## 📧 Support

For issues or questions, contact: [Your Contact]

---

**Last Updated:** December 2024

