from __future__ import annotations

import io
import json
import zipfile
from typing import Dict, List, Tuple

import duckdb
import pandas as pd

from file_detect import detect_csv_kind, normalize_headers
from schema_2026 import (
    normalize_matchrobot_df_with_meta,
    normalize_pit_df,
    normalize_schedule_df,
    normalize_superscout_df,
)


def _coalesce_bool_int(col: str) -> str:
    return f"CAST(COALESCE({col}, FALSE) AS INTEGER)"


def ensure_ingest_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_match (
          event_key TEXT,
          match_number INTEGER,
          team INTEGER,
          alliance TEXT,
          station INTEGER,
          scout_name TEXT,
          timestamp TIMESTAMP,
          robot_status TEXT,
          notes TEXT,
          start_position TEXT,
          auto_preload_result TEXT,
          auto_leave_alliance_zone BOOLEAN,
          auto_enter_neutral_zone BOOLEAN,
          auto_alliance_zone_pickups TEXT,
          auto_score INTEGER,
          auto_win BOOLEAN,
          active_cycles INTEGER,
          active_shoot_est INTEGER,
          miss_heavy BOOLEAN,
          ferried BOOLEAN,
          ferry_count_est INTEGER,
          inactive_prepped_to_shoot BOOLEAN,
          played_defense BOOLEAN,
          defense_effective TEXT,
          climb_level INTEGER,
          climb_failed BOOLEAN,
          climb_time_bucket TEXT,
          endgame_kept_shooting BOOLEAN,
          endgame_score INTEGER
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS super_scout (
          event_key TEXT,
          match_number INTEGER,
          alliance TEXT,
          team INTEGER,
          start_position_confirmed TEXT,
          auto_path_tag TEXT,
          auto_path_image_ref TEXT,
          auto_notes TEXT,
          driver_skill TEXT,
          defense_rating TEXT,
          cycle_speed_rating TEXT,
          want_on_alliance BOOLEAN,
          role_preference TEXT,
          reliability_flag BOOLEAN,
          reliability_notes TEXT,
          robot_status TEXT,
          auto_payload TEXT,
          shift_comments TEXT,
          endgame_payload TEXT,
          created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute("ALTER TABLE raw_match ADD COLUMN IF NOT EXISTS robot_status TEXT")
    con.execute("ALTER TABLE super_scout ADD COLUMN IF NOT EXISTS robot_status TEXT")
    con.execute("ALTER TABLE super_scout ADD COLUMN IF NOT EXISTS auto_payload TEXT")
    con.execute("ALTER TABLE super_scout ADD COLUMN IF NOT EXISTS shift_comments TEXT")
    con.execute("ALTER TABLE super_scout ADD COLUMN IF NOT EXISTS endgame_payload TEXT")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS pit (
          event_key TEXT,
          team INTEGER,
          drivebase TEXT,
          intake TEXT,
          can_ferry BOOLEAN,
          can_play_defense BOOLEAN,
          climb_claim TEXT,
          auto_claim TEXT,
          preferred_role TEXT,
          known_issues TEXT,
          updated_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule (
          event_key TEXT,
          match_number INTEGER,
          comp_level TEXT DEFAULT 'qm',
          red1 INTEGER,
          red2 INTEGER,
          red3 INTEGER,
          blue1 INTEGER,
          blue2 INTEGER,
          blue3 INTEGER,
          updated_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute("ALTER TABLE schedule ADD COLUMN IF NOT EXISTS comp_level TEXT DEFAULT 'qm'")


def refresh_compat_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw AS
        SELECT
          event_key,
          CAST(team AS BIGINT) AS team,
          CAST(match_number AS BIGINT) AS match_num,
          COALESCE(alliance, '') AS alliance,
          COALESCE(start_position, '') AS start_position,
          COALESCE(auto_preload_result, 'none') AS auto_preload_result,
          {_coalesce_bool_int('auto_leave_alliance_zone')} AS auto_leave_alliance_zone,
          {_coalesce_bool_int('auto_enter_neutral_zone')} AS auto_enter_neutral_zone,
          COALESCE(TRY_CAST(auto_alliance_zone_pickups AS INTEGER), 0) AS auto_alliance_zone_pickups,
          COALESCE(auto_score, 0) AS auto_score,
          {_coalesce_bool_int('auto_win')} AS auto_win,
          COALESCE(active_shoot_est, 0) AS active_shoot_est_mid,
          COALESCE(active_shoot_est, 0) AS active_shoot_est,
          COALESCE(active_shoot_est, 0) AS active_shooting_est,
          COALESCE(active_cycles, 0) AS active_cycles,
          {_coalesce_bool_int('miss_heavy')} AS active_miss_heavy,
          0 AS active_defense_level,
          COALESCE(ferry_count_est, 0) AS ferry_est_mid,
          COALESCE(ferry_count_est, 0) AS ferry_count_est,
          COALESCE(ferry_count_est, 0) AS inactive_ferry_est,
          {_coalesce_bool_int('ferried')} AS ferried,
          CASE
            WHEN COALESCE(ferry_count_est, 0) <= 0 THEN ''
            WHEN COALESCE(ferry_count_est, 0) <= 5 THEN '0-5'
            WHEN COALESCE(ferry_count_est, 0) <= 12 THEN '6-12'
            WHEN COALESCE(ferry_count_est, 0) <= 20 THEN '13-20'
            ELSE '21+'
          END AS ferry_bucket,
          0 AS inactive_defense_level,
          {_coalesce_bool_int('inactive_prepped_to_shoot')} AS inactive_prepped_to_shoot,
          {_coalesce_bool_int('played_defense')} AS played_defense,
          COALESCE(defense_effective, '') AS defense_effective,
          COALESCE(climb_level, 0) AS climb_level,
          {_coalesce_bool_int('climb_failed')} AS climb_failed,
          COALESCE(climb_time_bucket, 'NA') AS climb_time_bucket,
          {_coalesce_bool_int('endgame_kept_shooting')} AS endgame_kept_shooting,
          COALESCE(endgame_score,
            CASE COALESCE(climb_level, 0) WHEN 1 THEN 10 WHEN 2 THEN 20 WHEN 3 THEN 30 ELSE 0 END
          ) AS endgame_score,
          COALESCE(active_shoot_est, 0) AS tele_score,
          COALESCE(auto_score, 0)
            + COALESCE(active_shoot_est, 0)
            + COALESCE(endgame_score, CASE COALESCE(climb_level, 0) WHEN 1 THEN 10 WHEN 2 THEN 20 WHEN 3 THEN 30 ELSE 0 END)
            AS total_score,
          COALESCE(notes, '') AS notes,
          0 AS tele_l1, 0 AS tele_l2, 0 AS tele_l3, 0 AS tele_l4,
          0 AS auto_l1, 0 AS auto_l2, 0 AS auto_l3, 0 AS auto_l4,
          0 AS auto_net, 0 AS tele_net, 0 AS was_defended,
          '{{}}' AS raw_json
        FROM raw_match
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE raw_2026 AS
        SELECT
          event_key,
          team,
          match_num AS match_number,
          alliance,
          start_position,
          auto_preload_result,
          auto_leave_alliance_zone,
          auto_enter_neutral_zone,
          auto_alliance_zone_pickups,
          auto_score,
          auto_win,
          CASE
            WHEN active_shooting_est <= 0 THEN '0-5'
            WHEN active_shooting_est <= 5 THEN '0-5'
            WHEN active_shooting_est <= 12 THEN '6-12'
            WHEN active_shooting_est <= 20 THEN '13-20'
            ELSE '21+'
          END AS active_shoot_bucket,
          active_shoot_est_mid,
          active_shooting_est,
          active_cycles,
          active_miss_heavy,
          active_defense_level,
          ferried,
          ferry_bucket,
          ferry_est_mid,
          inactive_ferry_est,
          played_defense,
          defense_effective,
          inactive_defense_level,
          inactive_prepped_to_shoot,
          climb_level,
          climb_failed,
          climb_time_bucket,
          endgame_kept_shooting,
          endgame_score,
          notes
        FROM raw
        """
    )


def bootstrap_from_legacy_raw(con: duckdb.DuckDBPyConnection) -> None:
    ensure_ingest_tables(con)
    has_raw = "raw" in set(con.execute("SHOW TABLES").df()["name"].astype(str).tolist())
    current = int(con.execute("SELECT COUNT(*) FROM raw_match").fetchone()[0])
    if (not has_raw) or current > 0:
        return
    legacy_cnt = int(con.execute("SELECT COUNT(*) FROM raw").fetchone()[0])
    if legacy_cnt <= 0:
        return
    raw_cols = set(
        con.execute("PRAGMA table_info('raw')").df()["name"].astype(str).str.lower().tolist()
    )

    def pick(cols: List[str], cast: str, default: str = "NULL") -> str:
        for c in cols:
            if c in raw_cols:
                return f"CAST({c} AS {cast})"
        return default

    event_expr = pick(["event_key"], "TEXT", "'2026TEST'")
    match_expr = pick(["match_num", "match_number"], "INTEGER", "0")
    team_expr = pick(["team"], "INTEGER", "NULL")
    alliance_expr = pick(["alliance"], "TEXT", "''")
    notes_expr = pick(["notes"], "TEXT", "''")
    start_expr = pick(["start_position"], "TEXT", "''")
    preload_expr = pick(["auto_preload_result"], "TEXT", "''")
    leave_expr = pick(["auto_leave_alliance_zone"], "INTEGER", "0")
    enter_expr = pick(["auto_enter_neutral_zone"], "INTEGER", "0")
    pickups_expr = pick(["auto_alliance_zone_pickups"], "TEXT", "'0'")
    auto_score_expr = pick(["auto_score"], "INTEGER", "0")
    auto_win_expr = pick(["auto_win"], "INTEGER", "0")
    active_cycles_expr = pick(["active_cycles"], "INTEGER", "0")
    active_shoot_expr = pick(["active_shoot_est", "active_shoot_est_mid", "active_shooting_est"], "INTEGER", "0")
    miss_heavy_expr = pick(["active_miss_heavy"], "INTEGER", "0")
    ferried_expr = pick(["ferried"], "INTEGER", "0")
    ferry_expr = pick(["ferry_count_est", "ferry_est_mid", "inactive_ferry_est"], "INTEGER", "0")
    prep_expr = pick(["inactive_prepped_to_shoot"], "INTEGER", "0")
    played_def_expr = pick(["played_defense"], "INTEGER", "0")
    defense_eff_expr = pick(["defense_effective"], "TEXT", "''")
    climb_level_expr = pick(["climb_level"], "INTEGER", "0")
    climb_failed_expr = pick(["climb_failed"], "INTEGER", "0")
    climb_time_expr = pick(["climb_time_bucket"], "TEXT", "'NA'")
    end_keep_expr = pick(["endgame_kept_shooting"], "INTEGER", "0")
    endgame_score_expr = pick(["endgame_score"], "INTEGER", "0")

    con.execute(
        f"""
        INSERT INTO raw_match (
          event_key, match_number, team, alliance, station, scout_name, timestamp, robot_status, notes,
          start_position, auto_preload_result, auto_leave_alliance_zone, auto_enter_neutral_zone,
          auto_alliance_zone_pickups, auto_score, auto_win, active_cycles, active_shoot_est, miss_heavy,
          ferried, ferry_count_est, inactive_prepped_to_shoot, played_defense, defense_effective,
          climb_level, climb_failed, climb_time_bucket, endgame_kept_shooting, endgame_score
        )
        SELECT
          COALESCE({event_expr}, '2026TEST') AS event_key,
          COALESCE({match_expr}, 0) AS match_number,
          {team_expr} AS team,
          COALESCE({alliance_expr}, '') AS alliance,
          CAST(NULL AS INTEGER) AS station,
          CAST(NULL AS TEXT) AS scout_name,
          CAST(NULL AS TIMESTAMP) AS timestamp,
          CAST(NULL AS TEXT) AS robot_status,
          COALESCE({notes_expr}, '') AS notes,
          COALESCE({start_expr}, '') AS start_position,
          COALESCE({preload_expr}, '') AS auto_preload_result,
          CAST(COALESCE({leave_expr}, 0) > 0 AS BOOLEAN) AS auto_leave_alliance_zone,
          CAST(COALESCE({enter_expr}, 0) > 0 AS BOOLEAN) AS auto_enter_neutral_zone,
          COALESCE({pickups_expr}, '0') AS auto_alliance_zone_pickups,
          COALESCE({auto_score_expr}, 0) AS auto_score,
          CAST(COALESCE({auto_win_expr}, 0) > 0 AS BOOLEAN) AS auto_win,
          COALESCE({active_cycles_expr}, 0) AS active_cycles,
          COALESCE({active_shoot_expr}, 0) AS active_shoot_est,
          CAST(COALESCE({miss_heavy_expr}, 0) > 0 AS BOOLEAN) AS miss_heavy,
          CAST(COALESCE({ferried_expr}, 0) > 0 AS BOOLEAN) AS ferried,
          COALESCE({ferry_expr}, 0) AS ferry_count_est,
          CAST(COALESCE({prep_expr}, 0) > 0 AS BOOLEAN) AS inactive_prepped_to_shoot,
          CAST(COALESCE({played_def_expr}, 0) > 0 AS BOOLEAN) AS played_defense,
          COALESCE({defense_eff_expr}, '') AS defense_effective,
          COALESCE({climb_level_expr}, 0) AS climb_level,
          CAST(COALESCE({climb_failed_expr}, 0) > 0 AS BOOLEAN) AS climb_failed,
          COALESCE({climb_time_expr}, 'NA') AS climb_time_bucket,
          CAST(COALESCE({end_keep_expr}, 0) > 0 AS BOOLEAN) AS endgame_kept_shooting,
          COALESCE({endgame_score_expr}, 0) AS endgame_score
        FROM raw
        WHERE team IS NOT NULL
        """
    )


def _delete_insert(con: duckdb.DuckDBPyConnection, target: str, stage: str, join_cond: str) -> None:
    con.execute(f"DELETE FROM {target} t USING {stage} s WHERE {join_cond}")
    con.execute(f"INSERT INTO {target} SELECT * FROM {stage}")


def _to_bool(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _bucket_mid(v) -> int:
    s = str(v or "").strip().lower()
    mapping = {
        "1-10": 5,
        "10-20": 15,
        "20-40": 30,
        "40-60": 50,
        "60-80": 70,
        "0-5": 3,
        "6-12": 9,
        "13-20": 17,
        "21+": 24,
    }
    return mapping.get(s, 0)


def _acc_frac(v) -> float:
    s = str(v or "").strip().lower()
    mapping = {
        "<25%": 0.125,
        "25%": 0.25,
        "50%": 0.5,
        "75%": 0.75,
        "100%": 1.0,
    }
    return mapping.get(s, 0.0)


def _standardize_df_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = normalize_headers([str(c) for c in out.columns])
    return out


def _is_match_export(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    req = {"composite_key", "event_code", "match_num", "alliance", "station", "team_num"}
    return req.issubset(cols)


def _is_super_export(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return {"hub_status", "driver_ranking_1st"}.issubset(cols) and any(
        c in cols for c in {"r1_team_num", "r2_team_num", "r3_team_num"}
    )


def normalize_match_csv(df: pd.DataFrame, default_event_key: str = "2026TEST") -> pd.DataFrame:
    d = _standardize_df_headers(df)
    rows = []
    for _, r in d.iterrows():
        event_key = str(r.get("event_code") or default_event_key).strip()
        match_number = pd.to_numeric(r.get("match_num"), errors="coerce")
        team = pd.to_numeric(r.get("team_num"), errors="coerce")
        if pd.isna(match_number) or pd.isna(team):
            continue
        active_cycles = 0
        inactive_prepped = False
        ferry_count_est = 0
        active_shoot_est = 0.0
        for i in range(1, 5):
            status = str(r.get(f"shift{i}_status", "")).strip().lower()
            scoring_cycles = pd.to_numeric(r.get(f"shift{i}_scoring_cycles"), errors="coerce")
            scoring_cycles = int(scoring_cycles) if not pd.isna(scoring_cycles) else 0
            if status == "active":
                active_cycles += scoring_cycles
                sj = r.get(f"shift{i}_scoring_json", "")
                if isinstance(sj, str) and sj.strip():
                    try:
                        arr = json.loads(sj)
                    except Exception:
                        arr = []
                    if isinstance(arr, list):
                        for item in arr:
                            if not isinstance(item, dict):
                                continue
                            active_shoot_est += _bucket_mid(item.get("bucket")) * _acc_frac(item.get("accuracy"))
            if status == "inactive":
                inactive_prepped = inactive_prepped or _to_bool(r.get(f"shift{i}_prepared"))
                activity = str(r.get(f"shift{i}_inactive_activity", "")).lower()
                if "passing" in activity or "ferry" in activity:
                    ferry_count_est += _bucket_mid(r.get(f"shift{i}_pass_estimate"))

        climb_ach = str(r.get("end_climb_achieved", "")).strip().lower()
        climb_level = 0
        if "level 3" in climb_ach or climb_ach.endswith("3"):
            climb_level = 3
        elif "level 2" in climb_ach or climb_ach.endswith("2"):
            climb_level = 2
        elif "level 1" in climb_ach or climb_ach.endswith("1"):
            climb_level = 1
        climb_result = str(r.get("end_climb_result", "")).strip().lower()
        climb_failed = climb_result.startswith("failed")
        climb_start = str(r.get("end_climb_start", "")).strip()
        climb_time_bucket = "NA"
        if climb_start:
            lower = climb_start.lower()
            if "early" in lower:
                climb_time_bucket = "EARLY"
            elif "mid" in lower:
                climb_time_bucket = "MID"
            elif "late" in lower:
                climb_time_bucket = "LATE"
        endgame_score = {0: 0, 1: 10, 2: 20, 3: 30}.get(climb_level, climb_level * 10)

        rows.append(
            {
                "event_key": event_key,
                "match_number": int(match_number),
                "team": int(team),
                "alliance": str(r.get("alliance", "")).strip().lower(),
                "station": pd.to_numeric(r.get("station"), errors="coerce"),
                "scout_name": str(r.get("scouter_name", "")).strip(),
                "timestamp": pd.to_datetime(r.get("timestamp"), errors="coerce"),
                "robot_status": str(r.get("robot_status", "")).strip(),
                "notes": str(r.get("post_comments", "")).strip(),
                "start_position": str(r.get("start_position", "")).strip(),
                "auto_preload_result": str(r.get("auto_preload_result", "")).strip(),
                "auto_leave_alliance_zone": _to_bool(r.get("auto_leave_alliance_zone")),
                "auto_enter_neutral_zone": _to_bool(r.get("auto_enter_neutral_zone")),
                "auto_alliance_zone_pickups": str(r.get("auto_alliance_zone_pickups", "")).strip(),
                "auto_score": pd.to_numeric(r.get("auto_score"), errors="coerce"),
                "auto_win": _to_bool(r.get("auto_win")),
                "active_cycles": int(active_cycles),
                "active_shoot_est": int(round(active_shoot_est)),
                "miss_heavy": False,
                "ferried": ferry_count_est > 0,
                "ferry_count_est": int(ferry_count_est),
                "inactive_prepped_to_shoot": bool(inactive_prepped),
                "played_defense": False,
                "defense_effective": "",
                "climb_level": int(climb_level),
                "climb_failed": bool(climb_failed),
                "climb_time_bucket": climb_time_bucket,
                "endgame_kept_shooting": False,
                "endgame_score": int(endgame_score),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["auto_score"] = pd.to_numeric(out["auto_score"], errors="coerce").fillna(0).astype("Int64")
        out["station"] = pd.to_numeric(out["station"], errors="coerce").astype("Int64")
    return out


def normalize_super_csv(df: pd.DataFrame, default_event_key: str = "2026TEST") -> pd.DataFrame:
    d = _standardize_df_headers(df)
    rows = []
    for _, r in d.iterrows():
        event_key = str(r.get("event_code") or default_event_key).strip()
        match_number = pd.to_numeric(r.get("match_num"), errors="coerce")
        if pd.isna(match_number):
            continue
        alliance = str(r.get("alliance", "")).strip().lower()
        for n in (1, 2, 3):
            team = pd.to_numeric(r.get(f"r{n}_team_num"), errors="coerce")
            if pd.isna(team):
                continue
            auto_payload = {
                k: r.get(k)
                for k in d.columns
                if k.startswith(f"r{n}_auto_")
            }
            shift_comments = {
                k: r.get(k)
                for k in d.columns
                if k.startswith(f"r{n}_shift") and "comment" in k
            }
            endgame_payload = {
                k: r.get(k)
                for k in d.columns
                if k.startswith(f"r{n}_end")
            }
            rows.append(
                {
                    "event_key": event_key,
                    "match_number": int(match_number),
                    "alliance": alliance,
                    "team": int(team),
                    "start_position_confirmed": str(r.get(f"r{n}_start_position", "")).strip(),
                    "auto_path_tag": str(r.get(f"r{n}_auto_path_tag", "")).strip(),
                    "auto_path_image_ref": str(r.get(f"r{n}_auto_path_image_ref", "")).strip(),
                    "auto_notes": str(r.get(f"r{n}_auto_notes", "")).strip(),
                    "driver_skill": str(r.get("driver_ranking_1st", "")).strip(),
                    "defense_rating": str(r.get("defense_rating", "")).strip(),
                    "cycle_speed_rating": str(r.get("cycle_speed_rating", "")).strip(),
                    "want_on_alliance": _to_bool(r.get("want_on_alliance")),
                    "role_preference": str(r.get("role_preference", "")).strip(),
                    "reliability_flag": _to_bool(r.get("reliability_flag")),
                    "reliability_notes": str(r.get("reliability_notes", "")).strip(),
                    "robot_status": str(r.get(f"r{n}_status", "")).strip(),
                    "auto_payload": json.dumps(auto_payload),
                    "shift_comments": json.dumps(shift_comments),
                    "endgame_payload": json.dumps(endgame_payload),
                    "created_at": pd.Timestamp.utcnow(),
                }
            )
    return pd.DataFrame(rows)


def _health_matchrobot(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []
    n = len(df)
    if n == 0:
        return warnings
    dup = int(df.duplicated(subset=["event_key", "match_number", "team"]).sum())
    if dup > 0:
        warnings.append(f"duplicate key rows: {dup}")
    ferried_col = df["ferried"] if "ferried" in df.columns else pd.Series([False] * n, index=df.index)
    ferry_count_col = pd.to_numeric(df["ferry_count_est"], errors="coerce").fillna(0) if "ferry_count_est" in df.columns else pd.Series([0] * n, index=df.index)
    ferry_bad = int(((ferried_col.fillna(False)) & (ferry_count_col <= 0)).sum())
    if ferry_bad > 0:
        warnings.append(f"ferried=true but ferry_count_est<=0: {ferry_bad} ({(100.0*ferry_bad/n):.1f}%)")
    played_col = df["played_defense"] if "played_defense" in df.columns else pd.Series([False] * n, index=df.index)
    eff_col = df["defense_effective"] if "defense_effective" in df.columns else pd.Series([""] * n, index=df.index)
    def_bad = int(((played_col.fillna(False)) & (eff_col.fillna("").astype(str).str.strip() == "")).sum())
    if def_bad > 0:
        warnings.append(f"played_defense=true but defense_effective missing: {def_bad} ({(100.0*def_bad/n):.1f}%)")
    return warnings


def _dedupe_by_key(df: pd.DataFrame, key_cols: List[str], timestamp_col: str | None = None, label: str = "rows") -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if df.empty:
        return df, warnings
    work = df.copy()
    missing_keys = [c for c in key_cols if c not in work.columns]
    if missing_keys:
        return work, [f"{label}: dedupe skipped, missing key columns {missing_keys}"]
    dup_count = int(work.duplicated(subset=key_cols, keep=False).sum())
    if dup_count == 0:
        return work, warnings
    if timestamp_col and timestamp_col in work.columns:
        ts = pd.to_datetime(work[timestamp_col], errors="coerce")
        order = ts.fillna(pd.Timestamp.min)
        work = work.assign(_ts_order=order).sort_values("_ts_order", ascending=False).drop(columns=["_ts_order"])
        deduped = work.drop_duplicates(subset=key_cols, keep="first")
        warnings.append(f"{label}: deduped {dup_count} duplicate rows by latest {timestamp_col}.")
    else:
        deduped = work.drop_duplicates(subset=key_cols, keep="first")
        warnings.append(f"{label}: deduped {dup_count} duplicate rows (kept first).")
    return deduped, warnings


def ingest_csv_dataframe(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    filename: str,
    default_event_key: str = "2026TEST",
) -> Tuple[str, int, List[str]]:
    ensure_ingest_tables(con)
    kind = detect_csv_kind(set(df.columns))
    warnings: List[str] = []

    if kind == "matchrobot":
        std = _standardize_df_headers(df)
        if _is_match_export(std):
            clean = normalize_match_csv(std, default_event_key=default_event_key)
            w = []
        else:
            clean, _, w = normalize_matchrobot_df_with_meta(df, default_event_key=default_event_key)
        warnings.extend(w)
        warnings.extend(_health_matchrobot(clean))
        def col(name: str, default):
            if name in clean.columns:
                return clean[name]
            return pd.Series([default] * len(clean), index=clean.index)
        out = pd.DataFrame(
            {
                "event_key": clean["event_key"].astype(str),
                "match_number": pd.to_numeric(clean["match_number"], errors="coerce").astype("Int64"),
                "team": pd.to_numeric(clean["team"], errors="coerce").astype("Int64"),
                "alliance": col("alliance", "").astype("string"),
                "station": pd.to_numeric(col("station", pd.NA), errors="coerce").astype("Int64"),
                "scout_name": col("scout_name", "").astype("string"),
                "timestamp": pd.to_datetime(col("timestamp", pd.NaT), errors="coerce"),
                "robot_status": col("robot_status", "").astype("string"),
                "notes": col("notes", "").astype("string"),
                "start_position": col("start_position", "").astype("string"),
                "auto_preload_result": col("auto_preload_result", "").astype("string"),
                "auto_leave_alliance_zone": col("auto_leave_alliance_zone", False).map(_to_bool),
                "auto_enter_neutral_zone": col("auto_enter_neutral_zone", False).map(_to_bool),
                "auto_alliance_zone_pickups": col("auto_alliance_zone_pickups", "").astype("string"),
                "auto_score": pd.to_numeric(col("auto_score", 0), errors="coerce").fillna(0).astype("Int64"),
                "auto_win": col("auto_win", False).map(_to_bool),
                "active_cycles": pd.to_numeric(col("active_cycles", 0), errors="coerce").fillna(0).astype("Int64"),
                "active_shoot_est": pd.to_numeric(col("active_shoot_est", 0), errors="coerce").fillna(0).astype("Int64"),
                "miss_heavy": col("miss_heavy", False).map(_to_bool),
                "ferried": col("ferried", False).map(_to_bool),
                "ferry_count_est": pd.to_numeric(col("ferry_count_est", 0), errors="coerce").fillna(0).astype("Int64"),
                "inactive_prepped_to_shoot": col("inactive_prepped_to_shoot", False).map(_to_bool),
                "played_defense": col("played_defense", False).map(_to_bool),
                "defense_effective": col("defense_effective", "").astype("string"),
                "climb_level": pd.to_numeric(col("climb_level", 0), errors="coerce").fillna(0).astype("Int64"),
                "climb_failed": col("climb_failed", False).map(_to_bool),
                "climb_time_bucket": col("climb_time_bucket", "NA").astype("string"),
                "endgame_kept_shooting": col("endgame_kept_shooting", False).map(_to_bool),
                "endgame_score": pd.to_numeric(col("endgame_score", 0), errors="coerce").fillna(0).astype("Int64"),
            }
        )
        out = out.dropna(subset=["event_key", "match_number", "team"])
        out, dedupe_w = _dedupe_by_key(
            out,
            ["event_key", "match_number", "team"],
            timestamp_col="timestamp",
            label="MatchRobot",
        )
        warnings.extend(dedupe_w)
        con.register("stage_match", out)
        _delete_insert(
            con,
            "raw_match",
            "stage_match",
            "t.event_key = s.event_key AND t.match_number = s.match_number AND t.team = s.team",
        )
        count = len(out)
    elif kind == "superscout":
        std = _standardize_df_headers(df)
        if _is_super_export(std):
            clean = normalize_super_csv(std, default_event_key=default_event_key)
            w = []
        else:
            result = normalize_superscout_df(df, default_event_key=default_event_key)
            clean = result.df
            w = result.warnings
        warnings.extend(w)
        def col(name: str, default):
            if name in clean.columns:
                return clean[name]
            return pd.Series([default] * len(clean), index=clean.index)
        out = pd.DataFrame(
            {
                "event_key": clean["event_key"].astype(str),
                "match_number": pd.to_numeric(clean["match_number"], errors="coerce").astype("Int64"),
                "alliance": col("alliance", pd.NA).astype("string"),
                "team": pd.to_numeric(clean["team"], errors="coerce").astype("Int64"),
                "start_position_confirmed": col("start_position_confirmed", "").astype("string"),
                "auto_path_tag": col("auto_path_tag", "").astype("string"),
                "auto_path_image_ref": col("auto_path_image_ref", "").astype("string"),
                "auto_notes": col("auto_notes", "").astype("string"),
                "driver_skill": col("driver_skill", "").astype("string"),
                "defense_rating": col("defense_rating", "").astype("string"),
                "cycle_speed_rating": col("cycle_speed_rating", "").astype("string"),
                "want_on_alliance": col("want_on_alliance", False).map(_to_bool),
                "role_preference": col("role_preference", "").astype("string"),
                "reliability_flag": col("reliability_flag", False).map(_to_bool),
                "reliability_notes": col("reliability_notes", "").astype("string"),
                "robot_status": col("robot_status", "").astype("string"),
                "auto_payload": col("auto_payload", "").astype("string"),
                "shift_comments": col("shift_comments", "").astype("string"),
                "endgame_payload": col("endgame_payload", "").astype("string"),
                "created_at": pd.Timestamp.utcnow(),
            }
        )
        out = out.dropna(subset=["event_key", "match_number", "team"])
        out, dedupe_w = _dedupe_by_key(
            out,
            ["event_key", "match_number", "alliance", "team"],
            timestamp_col="created_at",
            label="Super Scout",
        )
        warnings.extend(dedupe_w)
        con.register("stage_super", out)
        _delete_insert(
            con,
            "super_scout",
            "stage_super",
            "t.event_key = s.event_key AND t.match_number = s.match_number AND COALESCE(t.alliance,'') = COALESCE(s.alliance,'') AND t.team = s.team",
        )
        count = len(out)
    elif kind == "pit":
        result = normalize_pit_df(df, default_event_key=default_event_key)
        clean = result.df
        warnings.extend(result.warnings)
        def col(name: str, default):
            if name in clean.columns:
                return clean[name]
            return pd.Series([default] * len(clean), index=clean.index)
        out = pd.DataFrame(
            {
                "event_key": col("event_key", pd.NA).astype("string"),
                "team": pd.to_numeric(clean["team"], errors="coerce").astype("Int64"),
                "drivebase": col("drivebase", "").astype("string"),
                "intake": col("intake", "").astype("string"),
                "can_ferry": col("can_ferry", False).map(_to_bool),
                "can_play_defense": col("can_play_defense", False).map(_to_bool),
                "climb_claim": col("climb_claim", "").astype("string"),
                "auto_claim": col("auto_claim", "").astype("string"),
                "preferred_role": col("preferred_role", "").astype("string"),
                "known_issues": col("known_issues", "").astype("string"),
                "updated_at": pd.Timestamp.utcnow(),
            }
        )
        out = out.dropna(subset=["team"])
        out, dedupe_w = _dedupe_by_key(
            out,
            ["event_key", "team"],
            timestamp_col="updated_at",
            label="Pit",
        )
        warnings.extend(dedupe_w)
        con.register("stage_pit", out)
        _delete_insert(
            con,
            "pit",
            "stage_pit",
            "COALESCE(t.event_key,'') = COALESCE(s.event_key,'') AND t.team = s.team",
        )
        count = len(out)
    elif kind == "schedule":
        try:
            result = normalize_schedule_df(df, default_event_key=default_event_key)
            clean = result.df
            warnings.extend(result.warnings)
            out = clean[["event_key", "match_number", "red1", "red2", "red3", "blue1", "blue2", "blue3", "updated_at"]].copy()
            out["comp_level"] = "qm"
            out = out[
                [
                    "event_key",
                    "match_number",
                    "comp_level",
                    "red1",
                    "red2",
                    "red3",
                    "blue1",
                    "blue2",
                    "blue3",
                    "updated_at",
                ]
            ]
            out = out.dropna(subset=["event_key", "match_number"])
            out, dedupe_w = _dedupe_by_key(
                out,
                ["event_key", "match_number", "comp_level"],
                timestamp_col="updated_at",
                label="Schedule",
            )
            warnings.extend(dedupe_w)
            con.register("stage_sched", out)
            con.execute(
                """
                DELETE FROM schedule
                WHERE event_key IN (SELECT DISTINCT event_key FROM stage_sched)
                """
            )
            _delete_insert(
                con,
                "schedule",
                "stage_sched",
                "t.event_key = s.event_key AND t.match_number = s.match_number AND COALESCE(t.comp_level,'qm') = COALESCE(s.comp_level,'qm')",
            )
            count = len(out)
        except ValueError as exc:
            # Some MatchRobot exports include columns that look "schedule-ish".
            # Retry as match data before failing hard.
            warnings.append(f"{filename}: schedule parse failed ({exc}); retried as matchrobot.")
            clean, _, w = normalize_matchrobot_df_with_meta(df, default_event_key=default_event_key)
            warnings.extend(w)
            warnings.extend(_health_matchrobot(clean))
            def col(name: str, default):
                if name in clean.columns:
                    return clean[name]
                return pd.Series([default] * len(clean), index=clean.index)
            out = pd.DataFrame(
                {
                    "event_key": clean["event_key"].astype(str),
                    "match_number": pd.to_numeric(clean["match_number"], errors="coerce").astype("Int64"),
                    "team": pd.to_numeric(clean["team"], errors="coerce").astype("Int64"),
                    "alliance": col("alliance", "").astype("string"),
                    "station": pd.to_numeric(col("station", pd.NA), errors="coerce").astype("Int64"),
                    "scout_name": col("scout_name", "").astype("string"),
                    "timestamp": pd.to_datetime(col("timestamp", pd.NaT), errors="coerce"),
                    "robot_status": col("robot_status", "").astype("string"),
                    "notes": col("notes", "").astype("string"),
                    "start_position": col("start_position", "").astype("string"),
                    "auto_preload_result": col("auto_preload_result", "").astype("string"),
                    "auto_leave_alliance_zone": col("auto_leave_alliance_zone", False).map(_to_bool),
                    "auto_enter_neutral_zone": col("auto_enter_neutral_zone", False).map(_to_bool),
                    "auto_alliance_zone_pickups": col("auto_alliance_zone_pickups", "").astype("string"),
                    "auto_score": pd.to_numeric(col("auto_score", 0), errors="coerce").fillna(0).astype("Int64"),
                    "auto_win": col("auto_win", False).map(_to_bool),
                    "active_cycles": pd.to_numeric(col("active_cycles", 0), errors="coerce").fillna(0).astype("Int64"),
                    "active_shoot_est": pd.to_numeric(col("active_shoot_est", 0), errors="coerce").fillna(0).astype("Int64"),
                    "miss_heavy": col("miss_heavy", False).map(_to_bool),
                    "ferried": col("ferried", False).map(_to_bool),
                    "ferry_count_est": pd.to_numeric(col("ferry_count_est", 0), errors="coerce").fillna(0).astype("Int64"),
                    "inactive_prepped_to_shoot": col("inactive_prepped_to_shoot", False).map(_to_bool),
                    "played_defense": col("played_defense", False).map(_to_bool),
                    "defense_effective": col("defense_effective", "").astype("string"),
                    "climb_level": pd.to_numeric(col("climb_level", 0), errors="coerce").fillna(0).astype("Int64"),
                    "climb_failed": col("climb_failed", False).map(_to_bool),
                    "climb_time_bucket": col("climb_time_bucket", "NA").astype("string"),
                    "endgame_kept_shooting": col("endgame_kept_shooting", False).map(_to_bool),
                    "endgame_score": pd.to_numeric(col("endgame_score", 0), errors="coerce").fillna(0).astype("Int64"),
                }
            )
            out = out.dropna(subset=["event_key", "match_number", "team"])
            out, dedupe_w = _dedupe_by_key(
                out,
                ["event_key", "match_number", "team"],
                timestamp_col="timestamp",
                label="MatchRobot",
            )
            warnings.extend(dedupe_w)
            con.register("stage_match", out)
            _delete_insert(
                con,
                "raw_match",
                "stage_match",
                "t.event_key = s.event_key AND t.match_number = s.match_number AND t.team = s.team",
            )
            kind = "matchrobot"
            count = len(out)
    else:
        return "unknown", 0, [f"{filename}: unknown CSV kind, skipped"]

    refresh_compat_views(con)
    return kind, count, warnings


def ingest_upload_bytes(
    con: duckdb.DuckDBPyConnection,
    payload: bytes,
    filename: str,
    default_event_key: str = "2026TEST",
) -> Dict[str, object]:
    summary = {"matchrobot": 0, "superscout": 0, "pit": 0, "schedule": 0, "skipped": 0, "warnings": []}

    def _one(csv_bytes: bytes, name: str) -> None:
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as exc:
            summary["skipped"] += 1
            summary["warnings"].append(f"{name}: read failed ({exc})")
            return
        kind, count, warnings = ingest_csv_dataframe(con, df, name, default_event_key=default_event_key)
        if kind == "unknown":
            summary["skipped"] += 1
        else:
            summary[kind] += int(count)
        summary["warnings"].extend(warnings)

    if filename.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                for zname in zf.namelist():
                    if zname.lower().endswith(".csv"):
                        _one(zf.read(zname), zname)
        except Exception as exc:
            summary["skipped"] += 1
            summary["warnings"].append(f"{filename}: zip read failed ({exc})")
    else:
        _one(payload, filename)

    return summary
