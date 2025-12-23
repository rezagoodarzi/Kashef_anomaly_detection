# Quick Start Guide

Get the database-driven behavioral analysis pipeline running in 5 minutes.

## Prerequisites

- Python 3.8+
- PostgreSQL 12+ (optional - can use mock mode for testing)
- Access to KPI data (CSV or database)

## Step 1: Install Dependencies

```bash
cd src/test/db_pipeline
pip install -r ../../requirements.txt
```

## Step 2: Test with Mock Data (No Database Required)

```python
# test_pipeline.py
from db_pipeline import run_pipeline
from datetime import datetime, timedelta

# Use your existing CSV file
result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    mock_data_path=r"C:\path\to\KJ_finalData_102.csv"
)

print(f"Success: {result.success}")
print(f"Connections: {result.connections_count}")
print(f"Change Points: {result.change_points_count}")
```

Run:
```bash
python test_pipeline.py
```

## Step 3: Set Up Database (Optional)

### 3.1 Create Database

```sql
CREATE DATABASE telecom_kpi;
```

### 3.2 Create Tables

```bash
psql -U postgres -d telecom_kpi -f sql/create_tables.sql
```

### 3.3 Run with Database

```python
from db_pipeline import run_pipeline, DatabaseConfig

db_config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="telecom_kpi",
    user="postgres",
    password="your_password"
)

result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    db_config=db_config
)
```

## Step 4: Configure for Your Data

### Edit City Configuration

Edit `config/city_config.yaml`:

```yaml
YourCity:
  prefixes:
    - "YC"  # Your cell name prefix
  timezone: "Asia/Tehran"
```

### Edit Technology Configuration

Edit `config/technology_config.yaml` to match your KPIs:

```yaml
4g:
  kpis:
    - name: "YOUR_KPI_NAME"
      threshold: -105.0
      anomaly_direction: "higher_worse"
```

## Step 5: Schedule Daily Runs

### Option A: Cron

```bash
# Add to crontab
0 2 * * * cd /path/to/db_pipeline && python scheduler.py batch
```

### Option B: Python Script

```python
# daily_run.py
from db_pipeline import run_batch_pipeline, DatabaseConfig, SchedulerConfig

config = SchedulerConfig(
    cities=["Karaj", "Tehran"],
    kpis=["RSSI_PUCCH(EUCell_Eric)(CRA)"],
    technologies=["4g"],
    db_config=DatabaseConfig(...)
)

result = run_batch_pipeline(config=config)
```

## Common Commands

```bash
# Run single analysis
python pipeline_orchestrator.py \
    --kpi "RSSI_PUCCH(EUCell_Eric)(CRA)" \
    --city "Karaj" \
    --technology "4g" \
    --mock-data "path/to/data.csv"

# Run scheduler once
python scheduler.py run \
    --kpi "RSSI_PUCCH(EUCell_Eric)(CRA)" \
    --city "Karaj"

# Run batch
python scheduler.py batch
```

## Next Steps

- Read full [README.md](README.md) for detailed documentation
- Check `sql/useful_queries.sql` for querying results
- Review configuration files in `config/` directory

## Troubleshooting

**Import errors?**
```bash
pip install psycopg2-binary pyyaml
```

**No data found?**
- Check KPI name matches database exactly
- Verify city prefixes in config
- Check date range has data

**Permission errors?**
- Close files in Excel/VS Code
- Check file permissions
- Delete output directory

For more help, see [README.md](README.md#troubleshooting)

