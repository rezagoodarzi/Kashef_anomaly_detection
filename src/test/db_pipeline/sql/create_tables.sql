-- =============================================================================
-- DATABASE SCHEMA FOR BEHAVIORAL ANALYSIS PIPELINE
-- =============================================================================
-- This script creates all required tables for storing pipeline outputs.
-- Run this script once to set up the database schema.
--
-- Tables:
--   1. behavioral_connections - Stores correlated sector pairs
--   2. change_points - Stores detected change points per sector
--   3. pipeline_runs - Tracks pipeline execution for idempotency
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. BEHAVIORAL CONNECTIONS TABLE
-- -----------------------------------------------------------------------------
-- Stores detected behavioral correlations between sector pairs.
-- Each row represents a pair of sectors that show correlated behavior.

DROP TABLE IF EXISTS behavioral_connections CASCADE;

CREATE TABLE behavioral_connections (
    id SERIAL PRIMARY KEY,
    
    -- Run metadata
    run_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Sector identifiers
    sector_1 VARCHAR(100) NOT NULL,
    sector_2 VARCHAR(100) NOT NULL,
    
    -- Composite correlation score (0.0 to 1.0)
    composite_score FLOAT8 NOT NULL,
    
    -- Individual correlation metrics
    pearson_correlation FLOAT8,
    pearson_p_value FLOAT8,
    difference_correlation FLOAT8,
    change_point_alignment FLOAT8,
    direction_alignment FLOAT8,
    
    -- Change point counts
    sector_1_change_points INT,
    sector_2_change_points INT,
    
    -- Anomaly flags
    sector_1_has_anomaly BOOLEAN DEFAULT FALSE,
    sector_2_has_anomaly BOOLEAN DEFAULT FALSE,
    both_anomaly BOOLEAN DEFAULT FALSE,
    
    -- Average RSSI values
    sector_1_avg_rssi FLOAT8,
    sector_2_avg_rssi FLOAT8,
    
    -- Ensure unique pairs per run
    CONSTRAINT uq_behavioral_connections_pair 
        UNIQUE (run_date, sector_1, sector_2)
);

-- Indexes for common queries
CREATE INDEX idx_bc_run_date ON behavioral_connections(run_date);
CREATE INDEX idx_bc_sector_1 ON behavioral_connections(sector_1);
CREATE INDEX idx_bc_sector_2 ON behavioral_connections(sector_2);
CREATE INDEX idx_bc_composite_score ON behavioral_connections(composite_score DESC);
CREATE INDEX idx_bc_both_anomaly ON behavioral_connections(both_anomaly) WHERE both_anomaly = TRUE;

-- Comment on table
COMMENT ON TABLE behavioral_connections IS 
    'Stores detected behavioral correlations between sector pairs from the analysis pipeline';


-- -----------------------------------------------------------------------------
-- 2. CHANGE POINTS TABLE
-- -----------------------------------------------------------------------------
-- Stores detected change points (sudden shifts) in KPI time series.
-- Each row represents a single change point for a sector.

DROP TABLE IF EXISTS change_points CASCADE;

CREATE TABLE change_points (
    id SERIAL PRIMARY KEY,
    
    -- Run metadata
    run_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Sector and time identification
    sector VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    change_point_index INT NOT NULL,
    
    -- Values around the change point
    value_at_change FLOAT8,
    value_before FLOAT8,
    value_after FLOAT8,
    
    -- Change characteristics
    magnitude FLOAT8,
    direction VARCHAR(10) CHECK (direction IN ('UP', 'DOWN')),
    
    -- Anomaly flag
    has_anomaly BOOLEAN DEFAULT FALSE,
    
    -- Alignment with other sectors
    num_aligned_sectors INT DEFAULT 0,
    aligned_sectors TEXT,  -- Semicolon-separated list
    aligned_same_direction INT DEFAULT 0,
    
    -- Ensure unique change points per run
    CONSTRAINT uq_change_points 
        UNIQUE (run_date, sector, timestamp, change_point_index)
);

-- Indexes for common queries
CREATE INDEX idx_cp_run_date ON change_points(run_date);
CREATE INDEX idx_cp_sector ON change_points(sector);
CREATE INDEX idx_cp_timestamp ON change_points(timestamp);
CREATE INDEX idx_cp_direction ON change_points(direction);
CREATE INDEX idx_cp_has_anomaly ON change_points(has_anomaly) WHERE has_anomaly = TRUE;
CREATE INDEX idx_cp_aligned ON change_points(num_aligned_sectors DESC) 
    WHERE num_aligned_sectors > 0;

-- Comment on table
COMMENT ON TABLE change_points IS 
    'Stores detected change points (sudden shifts) in KPI time series per sector';


-- -----------------------------------------------------------------------------
-- 3. PIPELINE RUNS TABLE
-- -----------------------------------------------------------------------------
-- Tracks pipeline execution for idempotency and auditing.
-- Prevents duplicate processing of the same data.

DROP TABLE IF EXISTS pipeline_runs CASCADE;

CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    
    -- Run identification
    run_date DATE NOT NULL,
    kpi_name VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    technology VARCHAR(10) DEFAULT '4g',
    
    -- Execution metadata
    status VARCHAR(20) DEFAULT 'running' 
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Results summary
    metrics JSONB,  -- Stores execution metrics as JSON
    error_message TEXT,
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique runs per day/kpi/city combination
    CONSTRAINT uq_pipeline_runs 
        UNIQUE (run_date, kpi_name, city)
);

-- Indexes for common queries
CREATE INDEX idx_pr_run_date ON pipeline_runs(run_date);
CREATE INDEX idx_pr_status ON pipeline_runs(status);
CREATE INDEX idx_pr_kpi_city ON pipeline_runs(kpi_name, city);

-- Comment on table
COMMENT ON TABLE pipeline_runs IS 
    'Tracks pipeline execution for idempotency and auditing';


-- -----------------------------------------------------------------------------
-- 4. HELPER VIEWS
-- -----------------------------------------------------------------------------

-- View: Latest behavioral connections (most recent run)
CREATE OR REPLACE VIEW v_latest_behavioral_connections AS
SELECT bc.*
FROM behavioral_connections bc
INNER JOIN (
    SELECT MAX(run_date) as max_date FROM behavioral_connections
) latest ON bc.run_date = latest.max_date;

-- View: Latest change points (most recent run)
CREATE OR REPLACE VIEW v_latest_change_points AS
SELECT cp.*
FROM change_points cp
INNER JOIN (
    SELECT MAX(run_date) as max_date FROM change_points
) latest ON cp.run_date = latest.max_date;

-- View: Sector connection summary
CREATE OR REPLACE VIEW v_sector_connection_summary AS
SELECT 
    sector,
    COUNT(*) as total_connections,
    AVG(composite_score) as avg_composite_score,
    SUM(CASE WHEN both_anomaly THEN 1 ELSE 0 END) as anomaly_connections,
    MAX(run_date) as last_run_date
FROM (
    SELECT sector_1 as sector, composite_score, both_anomaly, run_date 
    FROM behavioral_connections
    UNION ALL
    SELECT sector_2 as sector, composite_score, both_anomaly, run_date 
    FROM behavioral_connections
) combined
GROUP BY sector
ORDER BY total_connections DESC;

-- View: Change point clusters (multiple sectors changing together)
CREATE OR REPLACE VIEW v_change_point_clusters AS
SELECT 
    run_date,
    timestamp,
    direction,
    COUNT(*) as sector_count,
    ARRAY_AGG(sector) as sectors
FROM change_points
WHERE num_aligned_sectors > 0
GROUP BY run_date, timestamp, direction
HAVING COUNT(*) > 1
ORDER BY run_date DESC, sector_count DESC;


-- -----------------------------------------------------------------------------
-- 5. SAMPLE DATA QUERIES
-- -----------------------------------------------------------------------------
-- These are example queries for common analysis tasks.
-- Uncomment and run as needed.

/*
-- Find top correlated sector pairs
SELECT sector_1, sector_2, composite_score, change_point_alignment
FROM behavioral_connections
WHERE run_date = CURRENT_DATE - INTERVAL '1 day'
ORDER BY composite_score DESC
LIMIT 20;

-- Find sectors with most change points
SELECT sector, COUNT(*) as cp_count, AVG(magnitude) as avg_magnitude
FROM change_points
WHERE run_date = CURRENT_DATE - INTERVAL '1 day'
GROUP BY sector
ORDER BY cp_count DESC
LIMIT 20;

-- Find synchronized change points (potential interference events)
SELECT cp.timestamp, cp.direction, COUNT(DISTINCT cp.sector) as affected_sectors
FROM change_points cp
WHERE cp.run_date = CURRENT_DATE - INTERVAL '1 day'
  AND cp.num_aligned_sectors > 2
GROUP BY cp.timestamp, cp.direction
ORDER BY affected_sectors DESC;

-- Pipeline execution history
SELECT run_date, kpi_name, city, status, 
       metrics->>'connections_count' as connections,
       metrics->>'execution_time_seconds' as exec_time
FROM pipeline_runs
ORDER BY run_date DESC
LIMIT 10;
*/


-- -----------------------------------------------------------------------------
-- GRANT PERMISSIONS (adjust as needed for your environment)
-- -----------------------------------------------------------------------------
-- GRANT SELECT, INSERT, UPDATE, DELETE ON behavioral_connections TO app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON change_points TO app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON pipeline_runs TO app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;


-- =============================================================================
-- END OF SCHEMA
-- =============================================================================

