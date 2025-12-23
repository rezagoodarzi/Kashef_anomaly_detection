# Implementation Guide

Step-by-step guide for implementing the database-driven behavioral analysis pipeline in your environment.

## Phase 1: Setup (Day 1)

### 1.1 Install Dependencies

```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import psycopg2; import yaml; print('✓ Dependencies OK')"
```

### 1.2 Database Setup

**Create Database:**
```sql
CREATE DATABASE telecom_kpi;
```

**Create Tables:**
```bash
psql -U postgres -d telecom_kpi -f sql/create_tables.sql
```

**Verify Tables:**
```sql
\dt  -- Should show: behavioral_connections, change_points, pipeline_runs
```

### 1.3 Configuration Setup

**Verify Configuration Files Exist:**
- `config/technology_config.yaml`
- `config/city_config.yaml`
- `config/analysis_config.yaml`

**Test Configuration Loading:**
```python
from db_pipeline import ConfigurationManager

config = ConfigurationManager("config")
config.load_all()

print(f"Technologies: {list(config._technology_configs.keys())}")
print(f"Cities: {config.get_all_cities()}")
```

## Phase 2: Testing (Day 2)

### 2.1 Test with Mock Data

```bash
python test_pipeline.py
```

Expected output:
```
✓ Connections found: 1671
✓ Change points found: 1338
✓ TEST PASSED!
```

### 2.2 Test Database Connection

```python
from db_pipeline import DatabaseConfig, DatabaseConnectionManager

db_config = DatabaseConfig(
    host="localhost",
    database="telecom_kpi",
    user="postgres",
    password="your_password"
)

with DatabaseConnectionManager(config=db_config) as db:
    # Test query
    from db_interface import QueryParams
    from datetime import datetime
    
    params = QueryParams(
        kpi_name="RSSI_PUCCH(EUCell_Eric)(CRA)",
        start_datetime=datetime(2024, 1, 1),
        end_datetime=datetime(2024, 1, 2),
        city_prefixes=["KJ"],
        technology="4g"
    )
    
    data = db.fetch_kpi_data(params)
    print(f"Fetched {len(data)} records")
```

### 2.3 Test Full Pipeline (Mock Mode)

```python
from db_pipeline import run_pipeline

result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    mock_data_path="path/to/test_data.csv"
)

assert result.success, f"Pipeline failed: {result.error_message}"
print("✓ Pipeline test passed")
```

## Phase 3: Production Configuration (Day 3)

### 3.1 Configure Cities

Edit `config/city_config.yaml` with your cities:

```yaml
YourCity:
  prefixes:
    - "YC"
    - "YCT"
  timezone: "Asia/Tehran"
```

### 3.2 Configure Technologies

Edit `config/technology_config.yaml` with your KPIs:

```yaml
4g:
  kpis:
    - name: "YOUR_KPI_NAME"
      category: "Your Category"
      anomaly_direction: "higher_worse"
      threshold: -105.0
```

### 3.3 Configure Analysis Parameters

Edit `config/analysis_config.yaml`:

```yaml
correlation:
  pearson_threshold: 0.85
  composite_threshold: 0.7

change_point:
  pelt_penalty: 5.0
```

### 3.4 Database Credentials

Create `db_config.py` (DO NOT COMMIT):

```python
from db_pipeline import DatabaseConfig

PROD_DB_CONFIG = DatabaseConfig(
    host="your-db-host",
    port=5432,
    database="telecom_kpi",
    user="pipeline_user",
    password="your_secure_password"
)
```

## Phase 4: First Production Run (Day 4)

### 4.1 Dry Run (No Database Write)

Modify `pipeline_orchestrator.py` temporarily to skip database writes:

```python
# In _persist_results method, comment out:
# db.write_behavioral_connections(...)
# db.write_change_points(...)
```

### 4.2 Run Single Analysis

```python
from db_pipeline import run_pipeline
from db_config import PROD_DB_CONFIG

result = run_pipeline(
    kpi="RSSI_PUCCH(EUCell_Eric)(CRA)",
    city="Karaj",
    technology="4g",
    db_config=PROD_DB_CONFIG
)

print(f"Success: {result.success}")
print(f"Connections: {result.connections_count}")
```

### 4.3 Verify Database Output

```sql
-- Check behavioral connections
SELECT COUNT(*) FROM behavioral_connections 
WHERE run_date = CURRENT_DATE;

-- Check change points
SELECT COUNT(*) FROM change_points 
WHERE run_date = CURRENT_DATE;

-- Check pipeline run
SELECT * FROM pipeline_runs 
ORDER BY started_at DESC LIMIT 1;
```

## Phase 5: Automation (Day 5)

### 5.1 Set Up Scheduler

**Option A: Cron**

```bash
# Edit crontab
crontab -e

# Add (runs daily at 2 AM)
0 2 * * * cd /path/to/db_pipeline && /usr/bin/python3 scheduler.py batch >> /var/log/pipeline.log 2>&1
```

**Option B: Systemd Service**

Create `/etc/systemd/system/behavioral-pipeline.service`:

```ini
[Unit]
Description=Behavioral Analysis Pipeline
After=network.target postgresql.service

[Service]
Type=simple
User=pipeline
WorkingDirectory=/path/to/db_pipeline
ExecStart=/usr/bin/python3 scheduler.py daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable behavioral-pipeline
sudo systemctl start behavioral-pipeline
```

**Option C: Airflow DAG**

Copy Airflow DAG template from `scheduler.py` to your Airflow `dags/` folder.

### 5.2 Monitoring

**Check Logs:**
```bash
tail -f logs/scheduler_$(date +%Y%m%d).log
```

**Check Pipeline Status:**
```sql
SELECT run_date, kpi_name, city, status, 
       metrics->>'connections_count' as connections
FROM pipeline_runs
ORDER BY started_at DESC
LIMIT 10;
```

**Set Up Alerts:**
```python
# alert_on_failure.py
from db_pipeline import DatabaseConnectionManager, DatabaseConfig
import smtplib

def check_failed_runs():
    with DatabaseConnectionManager(config=PROD_DB_CONFIG) as db:
        # Query for recent failures
        # Send email alert if failures found
        pass
```

## Phase 6: Optimization (Week 2)

### 6.1 Performance Tuning

**Database Indexes:**
```sql
-- Already created in create_tables.sql
-- Verify they exist:
\d behavioral_connections
\d change_points
```

**Connection Pooling:**
```python
db_config = DatabaseConfig(
    max_connections=10,  # Adjust based on load
    min_connections=2
)
```

**Batch Size:**
```python
# In analysis_config.yaml
database:
  batch_size: 10000  # Adjust based on memory
```

### 6.2 Error Handling

**Retry Configuration:**
```python
# In scheduler.py
config = SchedulerConfig(
    max_retries=3,
    retry_delay_seconds=300
)
```

**Error Notifications:**
```python
# Add to pipeline_orchestrator.py
if not result.success:
    send_alert_email(result.error_message)
```

## Phase 7: Maintenance (Ongoing)

### 7.1 Daily Checks

**Morning Checklist:**
- [ ] Check pipeline logs for errors
- [ ] Verify yesterday's run completed
- [ ] Check database for new records
- [ ] Review any failed runs

**Weekly Checklist:**
- [ ] Review configuration changes
- [ ] Check database disk space
- [ ] Review performance metrics
- [ ] Update documentation

### 7.2 Adding New KPIs

1. Add to `config/technology_config.yaml`
2. Verify data exists in database
3. Test with single run
4. Add to batch schedule

### 7.3 Adding New Cities

1. Add to `config/city_config.yaml`
2. Verify cell name prefixes
3. Test with single run
4. Add to batch schedule

## Troubleshooting Checklist

- [ ] Database connection working?
- [ ] Configuration files valid YAML?
- [ ] KPI names match database exactly?
- [ ] City prefixes match cell names?
- [ ] Date range has data?
- [ ] Sufficient disk space?
- [ ] Permissions correct?
- [ ] Dependencies installed?

## Support

For issues:
1. Check logs in `logs/` directory
2. Review error messages in `pipeline_runs` table
3. Test with mock data first
4. Check database connectivity
5. Verify configuration syntax

---

**Estimated Timeline:**
- Phase 1-2: 2 days (setup and testing)
- Phase 3-4: 2 days (configuration and first run)
- Phase 5: 1 day (automation)
- Phase 6: Ongoing (optimization)
- Phase 7: Ongoing (maintenance)

**Total Initial Setup: ~5 days**

