-- Core views for Venom Analysis
-- Assumes table raw exists.

CREATE OR REPLACE VIEW v_team_summary AS
SELECT
  team,
  AVG(auto_score)    AS auto_avg,
  AVG(tele_score)    AS tele_avg,
  AVG(endgame_score) AS endgame_avg,
  AVG(total_score)   AS total_avg,
  COUNT(*)           AS rows
FROM raw
GROUP BY team;

CREATE OR REPLACE VIEW v_team_consistency AS
SELECT
  team,
  AVG(total_score) AS avg,
  STDDEV_POP(total_score) AS stdev,
  QUANTILE_CONT(total_score, 0.25) AS p25,
  QUANTILE_CONT(total_score, 0.75) AS p75,
  COUNT(*) AS rows
FROM raw
GROUP BY team;

CREATE OR REPLACE VIEW v_audit_match_counts AS
WITH counts AS (
  SELECT team, COUNT(DISTINCT match_num) AS matches_observed
  FROM raw
  WHERE match_num IS NOT NULL
  GROUP BY team
),
expected AS (
  SELECT CAST(QUANTILE_CONT(matches_observed, 0.5) AS INT) AS expected_matches
  FROM counts
)
SELECT
  c.team,
  c.matches_observed,
  e.expected_matches,
  (c.matches_observed = e.expected_matches) AS pass
FROM counts c
CROSS JOIN expected e;

CREATE OR REPLACE VIEW v_audit_duplicates AS
SELECT
  team,
  match_num,
  COUNT(*) AS rows
FROM raw
WHERE team IS NOT NULL AND match_num IS NOT NULL
GROUP BY team, match_num
HAVING COUNT(*) > 1;

-- 2026 REBUILT aggregates (calculated metrics)
CREATE OR REPLACE VIEW v_team_base_2026 AS
WITH rows AS (
  SELECT
    team,
    COALESCE(auto_score, 0) AS auto_score,
    COALESCE(active_cycles, 0) AS active_cycles,
    COALESCE(active_shoot_est, active_shoot_est_mid, active_shooting_est, 0) AS active_shoot_value,
    COALESCE(active_miss_heavy, 0) AS active_miss_heavy,
    COALESCE(ferry_count_est, ferry_est_mid, inactive_ferry_est, 0) AS ferry_value,
    COALESCE(inactive_prepped_to_shoot, 0) AS inactive_prepped_to_shoot,
    COALESCE(endgame_score, 0) AS endgame_score,
    COALESCE(climb_level, 0) AS climb_level,
    COALESCE(climb_failed, 0) AS climb_failed,
    COALESCE(active_shoot_est, active_shoot_est_mid, active_shooting_est, 0)
      + COALESCE(active_cycles, 0)
      + COALESCE(endgame_score, 0) AS points_proxy
  FROM raw
  WHERE team IS NOT NULL
)
SELECT
  team,
  COUNT(*) AS matches_played,
  AVG(auto_score) AS avg_auto_score,
  AVG(active_cycles) AS avg_active_cycles,
  AVG(active_shoot_value) AS avg_active_shooting_est,
  AVG(ferry_value) AS avg_inactive_ferry,
  AVG(inactive_prepped_to_shoot) AS prep_rate,
  AVG(endgame_score) AS avg_endgame_score,
  AVG(points_proxy) AS avg_points_proxy,
  STDDEV_SAMP(points_proxy) AS std_points_proxy,
  QUANTILE_CONT(points_proxy, 0.25) AS p25_points_proxy,
  QUANTILE_CONT(points_proxy, 0.75) AS p75_points_proxy,
  AVG(active_miss_heavy) AS miss_heavy_rate,
  AVG(CASE WHEN climb_failed = 1 THEN 1.0 ELSE 0.0 END) AS climb_fail_rate,
  AVG(CASE WHEN climb_level > 0 AND climb_failed = 0 THEN 1.0 ELSE 0.0 END) AS climb_success_rate
FROM rows
GROUP BY team;

CREATE OR REPLACE VIEW v_team_percentiles_2026 AS
SELECT
  team,
  matches_played,
  avg_auto_score,
  avg_active_cycles,
  avg_active_shooting_est,
  avg_inactive_ferry,
  prep_rate,
  avg_endgame_score,
  avg_points_proxy,
  std_points_proxy,
  p25_points_proxy,
  p75_points_proxy,
  miss_heavy_rate,
  climb_fail_rate,
  climb_success_rate,
  PERCENT_RANK() OVER (ORDER BY avg_active_cycles) AS p_active_cycles,
  PERCENT_RANK() OVER (ORDER BY avg_active_shooting_est) AS p_active_shoot,
  PERCENT_RANK() OVER (ORDER BY avg_endgame_score) AS p_endgame,
  PERCENT_RANK() OVER (ORDER BY avg_inactive_ferry) AS p_ferry,
  PERCENT_RANK() OVER (ORDER BY prep_rate) AS p_prep
FROM v_team_base_2026;

CREATE OR REPLACE VIEW v_team_scores_2026 AS
WITH scored AS (
  SELECT
    team,
    matches_played,
    avg_auto_score,
    avg_active_cycles,
    avg_active_shooting_est,
    avg_inactive_ferry,
    prep_rate,
    avg_endgame_score,
    avg_points_proxy,
    std_points_proxy,
    p25_points_proxy,
    p75_points_proxy,
    miss_heavy_rate,
    climb_fail_rate,
    climb_success_rate,
    p_active_cycles,
    p_active_shoot,
    p_endgame,
    p_ferry,
    p_prep,
    100.0 * (
      0.30 * p_active_cycles +
      0.25 * p_active_shoot +
      0.25 * p_endgame +
      0.10 * p_ferry +
      0.10 * p_prep
    ) AS RPScore_raw,
    100.0 * (
      0.35 * p_active_cycles +
      0.35 * p_active_shoot +
      0.20 * p_endgame +
      0.10 * p_ferry
    ) AS PointsScore_raw
  FROM v_team_percentiles_2026
),
shrunk AS (
  SELECT
    *,
    50.0 + (matches_played / (matches_played + 5.0)) * (RPScore_raw - 50.0) AS RPScore,
    50.0 + (matches_played / (matches_played + 5.0)) * (PointsScore_raw - 50.0) AS PointsScore
  FROM scored
)
SELECT
  team,
  matches_played,
  avg_auto_score,
  avg_active_cycles,
  avg_active_shooting_est,
  avg_inactive_ferry,
  prep_rate,
  avg_endgame_score,
  avg_points_proxy,
  std_points_proxy,
  p25_points_proxy,
  p75_points_proxy,
  miss_heavy_rate,
  climb_fail_rate,
  climb_success_rate,
  p_active_cycles,
  p_active_shoot,
  p_endgame,
  p_ferry,
  p_prep,
  RPScore_raw,
  PointsScore_raw,
  RPScore,
  PointsScore,
  PointsScore + 0.30 * RPScore AS QualsScore,
  CASE
    WHEN p_endgame >= 0.65 AND p_active_cycles < 0.50 THEN 'Endgame Anchor'
    WHEN p_active_cycles >= 0.65 AND (p_ferry < 0.45 AND p_prep < 0.45) THEN 'Active Scorer'
    WHEN (p_ferry >= 0.65 OR p_prep >= 0.65) AND p_active_cycles BETWEEN 0.35 AND 0.65 THEN 'Support'
    ELSE 'Balanced'
  END AS role_tag
FROM shrunk;

CREATE OR REPLACE VIEW v_team_confidence_2026 AS
WITH b AS (
  SELECT
    team,
    matches_played,
    std_points_proxy,
    miss_heavy_rate,
    climb_fail_rate
  FROM v_team_base_2026
),
th AS (
  SELECT
    AVG(COALESCE(std_points_proxy, 0)) + STDDEV_SAMP(COALESCE(std_points_proxy, 0)) AS vol_threshold
  FROM b
)
SELECT
  b.team,
  b.matches_played,
  CASE WHEN b.matches_played < 3 THEN 1 ELSE 0 END AS low_data_flag,
  CASE WHEN COALESCE(b.std_points_proxy, 0) >= COALESCE(th.vol_threshold, 0) THEN 1 ELSE 0 END AS volatility_flag,
  CASE WHEN COALESCE(b.miss_heavy_rate, 0) >= 0.35 THEN 1 ELSE 0 END AS miss_risk_flag,
  CASE WHEN COALESCE(b.climb_fail_rate, 0) >= 0.25 THEN 1 ELSE 0 END AS endgame_risk_flag,
  CASE
    WHEN b.matches_played < 3
      OR (
        COALESCE(b.std_points_proxy, 0) >= COALESCE(th.vol_threshold, 0)
        AND (COALESCE(b.miss_heavy_rate, 0) >= 0.35 OR COALESCE(b.climb_fail_rate, 0) >= 0.25)
      )
      THEN 'Houston, We''ve Got a Problem'
    WHEN (b.matches_played BETWEEN 3 AND 5)
      OR (
        b.matches_played >= 3
        AND (
          COALESCE(b.std_points_proxy, 0) >= COALESCE(th.vol_threshold, 0)
          OR COALESCE(b.miss_heavy_rate, 0) >= 0.35
          OR COALESCE(b.climb_fail_rate, 0) >= 0.25
        )
      )
      THEN 'Loose Fit'
    ELSE 'Clear to Launch'
  END AS TeamConfidenceTier
FROM b
CROSS JOIN th;

-- Backward compatibility with earlier naming.
CREATE OR REPLACE VIEW v_scores_2026 AS
SELECT * FROM v_team_scores_2026;

CREATE OR REPLACE VIEW v_metric_catalog_2026 AS
SELECT * FROM (
  VALUES
    ('QualsScore', 'Quals Score'),
    ('PointsScore', 'Points Score'),
    ('RPScore', 'RP Score'),
    ('avg_active_cycles', 'Avg Active Cycles'),
    ('avg_active_shooting_est', 'Avg Active Shooting'),
    ('avg_endgame_score', 'Avg Endgame Score'),
    ('prep_rate', 'Prep Rate'),
    ('avg_inactive_ferry', 'Avg Inactive Ferry'),
    ('climb_success_rate', 'Climb Success Rate'),
    ('avg_auto_score', 'Avg Auto Score'),
    ('miss_heavy_rate', 'Miss Heavy Rate')
) AS t(metric_key, metric_label);

CREATE OR REPLACE VIEW v_match_teams AS
SELECT event_key, match_number, comp_level, 'red' AS alliance, 1 AS slot, red1 AS team FROM schedule
UNION ALL
SELECT event_key, match_number, comp_level, 'red', 2, red2 FROM schedule
UNION ALL
SELECT event_key, match_number, comp_level, 'red', 3, red3 FROM schedule
UNION ALL
SELECT event_key, match_number, comp_level, 'blue', 1, blue1 FROM schedule
UNION ALL
SELECT event_key, match_number, comp_level, 'blue', 2, blue2 FROM schedule
UNION ALL
SELECT event_key, match_number, comp_level, 'blue', 3, blue3 FROM schedule;

CREATE OR REPLACE VIEW v_prematch_sixpack AS
SELECT
  m.event_key,
  m.match_number,
  m.comp_level,
  m.alliance,
  m.slot,
  m.team,
  s.QualsScore,
  s.PointsScore,
  s.RPScore,
  s.avg_active_cycles,
  s.avg_active_shooting_est,
  s.avg_endgame_score,
  s.avg_inactive_ferry,
  s.prep_rate,
  s.matches_played,
  s.role_tag,
  c.TeamConfidenceTier,
  c.low_data_flag,
  c.volatility_flag,
  c.miss_risk_flag,
  c.endgame_risk_flag
FROM v_match_teams m
LEFT JOIN v_team_scores_2026 s ON s.team = m.team
LEFT JOIN v_team_confidence_2026 c ON c.team = m.team
WHERE m.team IS NOT NULL;

CREATE OR REPLACE VIEW v_prematch_sixpack_adjusted AS
WITH latest_override AS (
  SELECT
    event_key,
    match_number,
    team,
    delta,
    reason
  FROM (
    SELECT
      event_key,
      match_number,
      team,
      COALESCE(delta, 0.0) AS delta,
      COALESCE(reason, '') AS reason,
      created_at,
      ROW_NUMBER() OVER (
        PARTITION BY event_key, match_number, team
        ORDER BY created_at DESC
      ) AS rn
    FROM coach_overrides
    WHERE scope = 'total'
  ) x
  WHERE rn = 1
)
SELECT
  s.*,
  COALESCE(o.delta, 0.0) AS delta,
  COALESCE(o.reason, '') AS override_reason,
  s.QualsScore + COALESCE(o.delta, 0.0) AS QualsScore_adj
FROM v_prematch_sixpack s
LEFT JOIN latest_override o
  ON s.event_key = o.event_key
 AND s.match_number = o.match_number
 AND s.team = o.team;

CREATE OR REPLACE VIEW v_match_confidence AS
WITH x AS (
  SELECT
    event_key,
    match_number,
    SUM(CASE WHEN matches_played < 3 THEN 1 ELSE 0 END) AS low_data_count,
    SUM(CASE WHEN volatility_flag = 1 THEN 1 ELSE 0 END) AS volatile_count,
    SUM(CASE WHEN TeamConfidenceTier = 'Loose Fit' THEN 1 ELSE 0 END) AS loose_fit_count,
    SUM(CASE WHEN miss_risk_flag = 1 THEN 1 ELSE 0 END) AS miss_risk_count,
    SUM(CASE WHEN endgame_risk_flag = 1 THEN 1 ELSE 0 END) AS endgame_risk_count,
    MAX(CASE WHEN ABS(COALESCE(delta, 0)) > 0 THEN 1 ELSE 0 END) AS override_active
  FROM v_prematch_sixpack_adjusted
  GROUP BY event_key, match_number
),
scored AS (
  SELECT
    *,
    100
      - 15 * low_data_count
      - 8 * volatile_count
      - 5 * loose_fit_count
      - 10 * override_active
      - 5 * miss_risk_count
      - 5 * endgame_risk_count AS confidence_score
  FROM x
)
SELECT
  event_key,
  match_number,
  confidence_score,
  low_data_count,
  volatile_count,
  loose_fit_count,
  override_active,
  miss_risk_count,
  endgame_risk_count,
  CASE
    WHEN confidence_score >= 80 THEN 'Clear to Launch'
    WHEN confidence_score >= 50 THEN 'Loose Fit'
    ELSE 'Houston, We''ve Got a Problem'
  END AS MissionConfidenceTier
FROM scored;

-- Default picklist view used by UI/import bootstrap.
-- TBA-enriched variants can still replace this view later in builder code.
CREATE OR REPLACE VIEW v_picklist_2026 AS
SELECT
  team,
  matches_played,
  PointsScore,
  RPScore,
  QualsScore,
  CAST(NULL AS DOUBLE) AS team_rp_avg,
  0 AS carried_risk,
  avg_auto_score,
  avg_active_shooting_est,
  avg_active_cycles,
  avg_inactive_ferry,
  prep_rate,
  avg_endgame_score
FROM v_team_scores_2026;

-- Stable alias for UI to avoid year-specific hardcoding.
CREATE OR REPLACE VIEW v_picklist_current AS
SELECT * FROM v_picklist_2026;
