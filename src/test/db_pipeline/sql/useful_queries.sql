-- =============================================================================
-- USEFUL QUERIES FOR BEHAVIORAL ANALYSIS PIPELINE
-- =============================================================================
-- Collection of analytical queries for exploring pipeline outputs.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- BEHAVIORAL CONNECTIONS ANALYSIS
-- -----------------------------------------------------------------------------

-- 1. Top correlated sector pairs by composite score
SELECT 
    sector_1,
    sector_2,
    composite_score,
    pearson_correlation,
    change_point_alignment,
    direction_alignment,
    both_anomaly
FROM behavioral_connections
WHERE run_date = (SELECT MAX(run_date) FROM behavioral_connections)
ORDER BY composite_score DESC
LIMIT 50;


-- 2. Sectors with most connections
SELECT 
    sector,
    COUNT(*) as connection_count,
    ROUND(AVG(composite_score)::numeric, 4) as avg_score,
    SUM(CASE WHEN both_anomaly THEN 1 ELSE 0 END) as anomaly_connections
FROM (
    SELECT sector_1 as sector, composite_score, both_anomaly 
    FROM behavioral_connections
    WHERE run_date = (SELECT MAX(run_date) FROM behavioral_connections)
    UNION ALL
    SELECT sector_2 as sector, composite_score, both_anomaly 
    FROM behavioral_connections
    WHERE run_date = (SELECT MAX(run_date) FROM behavioral_connections)
) combined
GROUP BY sector
ORDER BY connection_count DESC
LIMIT 30;


-- 3. High correlation pairs with high change point alignment
-- (Strong candidates for common interference source)
SELECT 
    sector_1,
    sector_2,
    composite_score,
    pearson_correlation,
    change_point_alignment,
    sector_1_change_points + sector_2_change_points as total_change_points
FROM behavioral_connections
WHERE run_date = (SELECT MAX(run_date) FROM behavioral_connections)
  AND change_point_alignment > 0.5
  AND pearson_correlation > 0.8
ORDER BY change_point_alignment DESC, composite_score DESC;


-- 4. Correlation trends over time
SELECT 
    run_date,
    COUNT(*) as total_connections,
    ROUND(AVG(composite_score)::numeric, 4) as avg_composite,
    ROUND(AVG(change_point_alignment)::numeric, 4) as avg_cp_align,
    SUM(CASE WHEN both_anomaly THEN 1 ELSE 0 END) as anomaly_pairs
FROM behavioral_connections
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 30;


-- -----------------------------------------------------------------------------
-- CHANGE POINT ANALYSIS
-- -----------------------------------------------------------------------------

-- 5. Most active change point timestamps (potential interference events)
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    direction,
    COUNT(*) as change_point_count,
    COUNT(DISTINCT sector) as affected_sectors,
    ROUND(AVG(magnitude)::numeric, 2) as avg_magnitude
FROM change_points
WHERE run_date = (SELECT MAX(run_date) FROM change_points)
GROUP BY DATE_TRUNC('hour', timestamp), direction
HAVING COUNT(*) > 3
ORDER BY change_point_count DESC;


-- 6. Sectors with most change points
SELECT 
    sector,
    COUNT(*) as cp_count,
    SUM(CASE WHEN direction = 'UP' THEN 1 ELSE 0 END) as up_count,
    SUM(CASE WHEN direction = 'DOWN' THEN 1 ELSE 0 END) as down_count,
    ROUND(AVG(magnitude)::numeric, 2) as avg_magnitude,
    MAX(magnitude) as max_magnitude
FROM change_points
WHERE run_date = (SELECT MAX(run_date) FROM change_points)
GROUP BY sector
ORDER BY cp_count DESC
LIMIT 30;


-- 7. Change points with many aligned sectors (synchronized events)
SELECT 
    sector,
    timestamp,
    direction,
    magnitude,
    num_aligned_sectors,
    aligned_sectors,
    aligned_same_direction
FROM change_points
WHERE run_date = (SELECT MAX(run_date) FROM change_points)
  AND num_aligned_sectors >= 3
ORDER BY num_aligned_sectors DESC, timestamp;


-- 8. Hourly distribution of change points
SELECT 
    EXTRACT(HOUR FROM timestamp) as hour_of_day,
    COUNT(*) as cp_count,
    SUM(CASE WHEN direction = 'UP' THEN 1 ELSE 0 END) as up_count,
    SUM(CASE WHEN direction = 'DOWN' THEN 1 ELSE 0 END) as down_count
FROM change_points
WHERE run_date = (SELECT MAX(run_date) FROM change_points)
GROUP BY EXTRACT(HOUR FROM timestamp)
ORDER BY hour_of_day;


-- -----------------------------------------------------------------------------
-- INTERFERENCE SOURCE IDENTIFICATION
-- -----------------------------------------------------------------------------

-- 9. Find potential interference sources
-- Sectors that appear in many high-correlation pairs with aligned change points
WITH sector_scores AS (
    SELECT 
        sector,
        SUM(composite_score) as total_score,
        SUM(change_point_alignment) as total_alignment,
        COUNT(*) as connection_count
    FROM (
        SELECT sector_1 as sector, composite_score, change_point_alignment 
        FROM behavioral_connections
        WHERE run_date = (SELECT MAX(run_date) FROM behavioral_connections)
          AND change_point_alignment > 0.3
        UNION ALL
        SELECT sector_2 as sector, composite_score, change_point_alignment 
        FROM behavioral_connections
        WHERE run_date = (SELECT MAX(run_date) FROM behavioral_connections)
          AND change_point_alignment > 0.3
    ) combined
    GROUP BY sector
)
SELECT 
    ss.sector,
    ss.connection_count,
    ROUND(ss.total_score::numeric, 2) as total_score,
    ROUND(ss.total_alignment::numeric, 2) as total_alignment,
    cp.cp_count,
    cp.avg_magnitude
FROM sector_scores ss
LEFT JOIN (
    SELECT sector, COUNT(*) as cp_count, ROUND(AVG(magnitude)::numeric, 2) as avg_magnitude
    FROM change_points
    WHERE run_date = (SELECT MAX(run_date) FROM change_points)
    GROUP BY sector
) cp ON ss.sector = cp.sector
ORDER BY ss.connection_count DESC, ss.total_score DESC
LIMIT 20;


-- 10. Temporal clustering of interference events
WITH cp_clusters AS (
    SELECT 
        DATE_TRUNC('hour', timestamp) as event_hour,
        ARRAY_AGG(DISTINCT sector) as affected_sectors,
        COUNT(DISTINCT sector) as sector_count,
        MAX(magnitude) as max_magnitude
    FROM change_points
    WHERE run_date = (SELECT MAX(run_date) FROM change_points)
      AND num_aligned_sectors >= 2
    GROUP BY DATE_TRUNC('hour', timestamp)
    HAVING COUNT(DISTINCT sector) >= 3
)
SELECT 
    event_hour,
    sector_count,
    max_magnitude,
    affected_sectors
FROM cp_clusters
ORDER BY sector_count DESC, event_hour;


-- -----------------------------------------------------------------------------
-- PIPELINE MONITORING
-- -----------------------------------------------------------------------------

-- 11. Recent pipeline runs
SELECT 
    run_date,
    kpi_name,
    city,
    technology,
    status,
    started_at,
    completed_at,
    EXTRACT(EPOCH FROM (completed_at - started_at)) as duration_seconds,
    metrics->>'connections_count' as connections,
    metrics->>'change_points_count' as change_points,
    error_message
FROM pipeline_runs
ORDER BY started_at DESC
LIMIT 20;


-- 12. Pipeline success rate by city
SELECT 
    city,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    ROUND(
        100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*),
        1
    ) as success_rate
FROM pipeline_runs
GROUP BY city
ORDER BY total_runs DESC;


-- 13. Average execution time by technology
SELECT 
    technology,
    COUNT(*) as run_count,
    ROUND(AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))::numeric, 2) as avg_seconds,
    ROUND(AVG((metrics->>'connections_count')::int)::numeric, 0) as avg_connections,
    ROUND(AVG((metrics->>'change_points_count')::int)::numeric, 0) as avg_change_points
FROM pipeline_runs
WHERE status = 'completed'
GROUP BY technology
ORDER BY run_count DESC;


-- =============================================================================
-- END OF QUERIES
-- =============================================================================

