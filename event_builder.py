import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import numpy as np
import requests
from schema_2026 import bucket_mid_series, normalize_bool_series, validate_appsheet_csv
from ingest_backend import ensure_ingest_tables, ingest_upload_bytes, refresh_compat_views

HERE = Path(__file__).parent
# 2026 REBUILT point assumptions from Table 6-4.
# Climb/tower levels in TELEOP: L1=10, L2=20, L3=30.
CLIMB_POINTS_2026 = {0: 0, 1: 10, 2: 20, 3: 30}
FUEL_POINTS_AUTO = 1
FUEL_POINTS_TELE = 1

def load_mapping(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in low:
            return low[key]
    # fuzzy contains
    for c in candidates:
        key = str(c).strip().lower()
        for lc, orig in low.items():
            if key and key in lc:
                return orig
    return None

def coerce_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def coerce_bool01(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="Int64")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0, upper=1).astype("Int64")
    parsed = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "t", "yes", "y", "made"])
        .astype("Int64")
    )
    return parsed

def coerce_text(s: pd.Series, upper: bool = False, lower: bool = False) -> pd.Series:
    if s is None:
        return pd.Series(dtype="string")
    out = s.fillna("").astype(str).str.strip()
    if upper:
        out = out.str.upper()
    if lower:
        out = out.str.lower()
    return out.astype("string")

def series_or_default(df: pd.DataFrame, col: Optional[str], default, dtype: str):
    if col is None:
        return pd.Series([default] * len(df), dtype=dtype)
    if dtype == "Int64":
        return coerce_int(df[col]).fillna(default).astype("Int64")
    if dtype == "string":
        return coerce_text(df[col])
    return pd.Series(df[col], dtype=dtype)

def build_raw(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    validated = validate_appsheet_csv(df, "MatchRobot")
    df = validated.df
    for w in validated.warnings:
        print(f"⚠ {w}")

    event_key_c = find_col(df, mapping.get("event_key", []))
    team_c = find_col(df, mapping["team"])
    match_c = find_col(df, mapping["match"])
    alliance_c = find_col(df, mapping.get("alliance", []))
    start_position_c = find_col(df, mapping.get("start_position", []))
    auto_preload_c = find_col(df, mapping.get("auto_preload_result", []))
    auto_leave_c = find_col(df, mapping.get("auto_leave_alliance_zone", []))
    auto_enter_c = find_col(df, mapping.get("auto_enter_neutral_zone", []))
    auto_pickups_c = find_col(df, mapping.get("auto_alliance_zone_pickups", []))
    auto_score_c = find_col(df, mapping.get("auto_score", []))
    auto_win_c = find_col(df, mapping.get("auto_win", []))
    active_shoot_c = find_col(df, mapping.get("active_shooting_est", []))
    active_shoot_bucket_c = find_col(df, mapping.get("active_shoot_bucket", []))
    active_cycles_c = find_col(df, mapping.get("active_cycles", []))
    active_miss_c = find_col(df, mapping.get("active_miss_heavy", []))
    active_def_c = find_col(df, mapping.get("active_defense_level", []))
    inactive_ferry_c = find_col(df, mapping.get("inactive_ferry_est", []))
    ferried_c = find_col(df, mapping.get("ferried", []))
    ferry_bucket_c = find_col(df, mapping.get("ferry_bucket", []))
    played_defense_c = find_col(df, mapping.get("played_defense", []))
    defense_effective_c = find_col(df, mapping.get("defense_effective", []))
    inactive_def_c = find_col(df, mapping.get("inactive_defense_level", []))
    inactive_prep_c = find_col(df, mapping.get("inactive_prepped_to_shoot", []))
    climb_level_c = find_col(df, mapping.get("climb_level", []))
    climb_failed_c = find_col(df, mapping.get("climb_failed", []))
    climb_time_c = find_col(df, mapping.get("climb_time_bucket", []))
    endgame_kept_shooting_c = find_col(df, mapping.get("endgame_kept_shooting", []))
    endgame_score_c = find_col(df, mapping.get("endgame_score", []))
    tele_score_c = find_col(df, mapping.get("tele_score", []))
    total_c = find_col(df, mapping.get("total_score", []))
    notes_c = find_col(df, mapping.get("notes", []))

    if team_c is None:
        raise ValueError("Could not find TEAM column. Edit mapping.json (raw.team).")

    out = pd.DataFrame()
    out["event_key"] = series_or_default(df, event_key_c, "", "string")
    out["team"] = coerce_int(df[team_c])
    out["match_num"] = series_or_default(df, match_c, pd.NA, "Int64")
    out["alliance"] = series_or_default(df, alliance_c, "", "string")
    out["start_position"] = series_or_default(df, start_position_c, "", "string")

    out["auto_preload_result"] = series_or_default(df, auto_preload_c, "none", "string").str.lower()
    auto_preload_made = out["auto_preload_result"].isin(["made", "score", "scored", "yes", "true", "1"]).astype("Int64")
    out["auto_leave_alliance_zone"] = (
        coerce_bool01(df[auto_leave_c]) if auto_leave_c else pd.Series([0] * len(df), dtype="Int64")
    )
    out["auto_enter_neutral_zone"] = (
        coerce_bool01(df[auto_enter_c]) if auto_enter_c else pd.Series([0] * len(df), dtype="Int64")
    )
    if auto_pickups_c:
        pickup_s = df[auto_pickups_c].fillna("").astype(str).str.strip()
        out["auto_alliance_zone_pickups"] = (
            pickup_s.replace({"2+": "2"}).replace("", "0").astype("Int64")
        )
    else:
        out["auto_alliance_zone_pickups"] = pd.Series([0] * len(df), dtype="Int64")
    if auto_score_c:
        out["auto_score"] = series_or_default(df, auto_score_c, 0, "Int64")
    else:
        # Fallback when CSV does not provide explicit auto points.
        out["auto_score"] = (auto_preload_made * FUEL_POINTS_AUTO).astype("Int64")
    out["auto_win"] = (
        coerce_bool01(df[auto_win_c]).astype("Int64")
        if auto_win_c
        else pd.Series([pd.NA] * len(df), dtype="Int64")
    )

    out["active_shoot_bucket"] = series_or_default(df, active_shoot_bucket_c, "", "string")
    out["active_shoot_est_mid"] = (
        bucket_mid_series(out["active_shoot_bucket"]) if active_shoot_bucket_c else pd.Series([0] * len(df), dtype="Int64")
    )
    if active_shoot_c:
        out["active_shooting_est"] = series_or_default(df, active_shoot_c, 0, "Int64")
    else:
        out["active_shooting_est"] = out["active_shoot_est_mid"].astype("Int64")
    out["active_cycles"] = series_or_default(df, active_cycles_c, 0, "Int64")
    out["active_miss_heavy"] = (
        coerce_bool01(df[active_miss_c]) if active_miss_c else pd.Series([0] * len(df), dtype="Int64")
    )
    out["active_defense_level"] = series_or_default(df, active_def_c, 0, "Int64")
    out["ferried"] = (
        normalize_bool_series(df[ferried_c]) if ferried_c else pd.Series([False] * len(df), dtype="boolean")
    )
    out["ferry_bucket"] = series_or_default(df, ferry_bucket_c, "", "string")
    out["ferry_est_mid"] = bucket_mid_series(out["ferry_bucket"])
    out.loc[~out["ferried"].fillna(False), "ferry_est_mid"] = 0
    if inactive_ferry_c:
        out["inactive_ferry_est"] = series_or_default(df, inactive_ferry_c, 0, "Int64")
    else:
        out["inactive_ferry_est"] = out["ferry_est_mid"].astype("Int64")
    out["inactive_defense_level"] = series_or_default(df, inactive_def_c, 0, "Int64")
    out["inactive_prepped_to_shoot"] = (
        coerce_bool01(df[inactive_prep_c]) if inactive_prep_c else pd.Series([0] * len(df), dtype="Int64")
    )
    out["played_defense"] = (
        normalize_bool_series(df[played_defense_c]) if played_defense_c else pd.Series([False] * len(df), dtype="boolean")
    )
    out["defense_effective"] = series_or_default(df, defense_effective_c, "", "string")
    out.loc[~out["played_defense"].fillna(False), "defense_effective"] = ""

    out["climb_level"] = series_or_default(df, climb_level_c, 0, "Int64")
    out["climb_failed"] = (
        coerce_bool01(df[climb_failed_c]) if climb_failed_c else pd.Series([0] * len(df), dtype="Int64")
    )
    out["climb_time_bucket"] = series_or_default(df, climb_time_c, "NA", "string").str.upper()
    out["endgame_kept_shooting"] = (
        coerce_bool01(df[endgame_kept_shooting_c])
        if endgame_kept_shooting_c
        else pd.Series([0] * len(df), dtype="Int64")
    )

    if endgame_score_c:
        provided_endgame = coerce_int(df[endgame_score_c])
        derived_endgame = out["climb_level"].map(CLIMB_POINTS_2026).astype("Int64")
        out["endgame_score"] = provided_endgame.fillna(derived_endgame).fillna(0).astype("Int64")
    else:
        out["endgame_score"] = out["climb_level"].map(CLIMB_POINTS_2026).fillna(0).astype("Int64")

    if tele_score_c:
        out["tele_score"] = series_or_default(df, tele_score_c, 0, "Int64")
    else:
        # Active scored FUEL in TELEOP is worth 1 point each.
        out["tele_score"] = (out["active_shooting_est"] * FUEL_POINTS_TELE).astype("Int64")
    if total_c:
        out["total_score"] = coerce_int(df[total_c]).fillna(0).astype("Int64")
    else:
        out["total_score"] = (out["auto_score"] + out["tele_score"] + out["endgame_score"]).astype("Int64")
    out["notes"] = series_or_default(df, notes_c, "", "string")

    # Legacy compatibility columns (kept so existing app tabs and queries do not crash).
    for legacy_col in [
        "tele_l1",
        "tele_l2",
        "tele_l3",
        "tele_l4",
        "auto_l1",
        "auto_l2",
        "auto_l3",
        "auto_l4",
        "auto_net",
        "tele_net",
        "was_defended",
    ]:
        out[legacy_col] = pd.Series([0] * len(df), dtype="Int64")

    # keep original row as JSON string for game-specific fields
    out["raw_json"] = df.fillna("").to_dict(orient="records")
    out["raw_json"] = out["raw_json"].apply(json.dumps)

    return out

def build_schedule(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    match_c = find_col(df, mapping["match"])
    if match_c is None:
        raise ValueError("Could not find schedule MATCH column. Edit mapping.json (schedule.match).")
    out = pd.DataFrame()
    out["match_num"] = coerce_int(df[match_c])
    for slot in ["red1","red2","red3","blue1","blue2","blue3"]:
        c = find_col(df, mapping.get(slot, []))
        out[slot] = coerce_int(df[c]) if c else pd.Series([pd.NA]*len(df), dtype="Int64")
    return out

def connect_db(out_path: Path) -> duckdb.DuckDBPyConnection:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(out_path))

def create_views(con: duckdb.DuckDBPyConnection, has_schedule: bool):
    # Ensure dependent prematch tables exist before creating views.sql views.
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
          updated_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prematch_plan (
          event_key TEXT,
          match_number INTEGER,
          alliance TEXT DEFAULT 'both',
          plan_text TEXT,
          assignments_json TEXT,
          created_at TIMESTAMP DEFAULT now(),
          updated_at TIMESTAMP DEFAULT now()
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS coach_overrides (
          event_key TEXT,
          match_number INTEGER,
          team INTEGER,
          scope TEXT DEFAULT 'total',
          delta DOUBLE,
          reason TEXT,
          created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    views_sql = (HERE/"views.sql").read_text(encoding="utf-8")
    con.execute(views_sql)
    if has_schedule:
        con.execute("""
        CREATE OR REPLACE VIEW v_match_preview AS
        SELECT
          s.match_number AS match_num,
          s.red1, s.red2, s.red3,
          s.blue1, s.blue2, s.blue3
        FROM schedule s
        WHERE s.match_number IS NOT NULL
        ORDER BY s.match_number;
        """)
    # helpful index-ish ordering
    con.execute("""CREATE OR REPLACE VIEW v_team_list AS
    SELECT DISTINCT team FROM raw WHERE team IS NOT NULL ORDER BY team;""")
    table_names = set(con.execute("SHOW TABLES").df()["name"].tolist())
    has_tba_tables = {"match_meta_alliance", "match_teams"}.issubset(table_names)
    if has_tba_tables:
        con.execute(
            """
            CREATE OR REPLACE VIEW v_picklist_2026 AS
            WITH tba_rp AS (
              SELECT
                mt.team AS team,
                AVG(COALESCE(mma.rp, 0.0)) AS team_rp_avg
              FROM match_teams mt
              JOIN match_meta_alliance mma
                ON mt.match_key = mma.match_key AND mt.alliance = mma.alliance
              GROUP BY mt.team
            ),
            rp_bounds AS (
              SELECT
                QUANTILE_CONT(team_rp_avg, 0.70) AS rp_top30_cut,
                QUANTILE_CONT(team_rp_avg, 0.30) AS rp_bot30_cut
              FROM tba_rp
            ),
            model_bounds AS (
              SELECT
                QUANTILE_CONT(RPScore, 0.70) AS model_top30_cut,
                QUANTILE_CONT(RPScore, 0.30) AS model_bot30_cut
              FROM v_team_scores_2026
            )
            SELECT
              s.team,
              s.matches_played,
              s.PointsScore,
              s.RPScore,
              s.QualsScore,
              t.team_rp_avg,
              CASE
                WHEN t.team_rp_avg >= rb.rp_top30_cut AND s.RPScore <= mb.model_bot30_cut THEN 1
                ELSE 0
              END AS carried_risk,
              s.avg_auto_score,
              s.avg_active_shooting_est,
              s.avg_active_cycles,
              s.avg_inactive_ferry,
              s.prep_rate,
              s.avg_endgame_score
            FROM v_team_scores_2026 s
            LEFT JOIN tba_rp t ON t.team = s.team
            CROSS JOIN rp_bounds rb
            CROSS JOIN model_bounds mb
            """
        )
    else:
        con.execute(
            """
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
            FROM v_team_scores_2026
            """
        )

def ingest_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def ingest_pit_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # If headers look like data (e.g., first header is a team number), re-read without header
    bad_header = False
    if len(df.columns) > 0:
        first = str(df.columns[0]).strip()
        if first.isdigit() or first.startswith("Pit Scouting_Images"):
            bad_header = True
    if bad_header:
        df = pd.read_csv(path, header=None)
        cols = ["team"] + [f"pit_col_{i}" for i in range(1, df.shape[1])]
        df.columns = cols
    return df

def ingest_super_scout_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validated = validate_appsheet_csv(df, "SuperScoutAuto")
    for w in validated.warnings:
        print(f"⚠ {w}")
    out = validated.df.copy()
    out["alliance"] = out["alliance"].astype(str).str.strip().str.lower()
    out["match_number"] = pd.to_numeric(out["match_number"], errors="coerce").astype("Int64")
    out["team"] = pd.to_numeric(out["team"], errors="coerce").astype("Int64")
    return out

def build(args):
    con = connect_db(Path(args.out))
    try:
        ensure_ingest_tables(con)
        payloads = [("raw", args.raw), ("schedule", args.schedule), ("pit", args.pit), ("super", args.super_scout)]
        summary = {"matchrobot": 0, "superscout": 0, "pit": 0, "schedule": 0, "skipped": 0}
        warnings = []
        for _, path_str in payloads:
            if not path_str:
                continue
            path = Path(path_str)
            local = ingest_upload_bytes(
                con,
                path.read_bytes(),
                path.name,
                default_event_key=args.event_key or "2026TEST",
            )
            for k in summary:
                summary[k] += int(local.get(k, 0))
            warnings.extend(local.get("warnings", []))

        refresh_compat_views(con)
        has_schedule = summary["schedule"] > 0 or con.execute("SELECT COUNT(*) FROM schedule").fetchone()[0] > 0
        create_views(con, has_schedule=has_schedule)
        for w in warnings[:8]:
            print(f"⚠ {w}")

        # basic quick audit counts
        teams = con.execute("SELECT COUNT(DISTINCT team) FROM raw").fetchone()[0]
        rows  = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
        print(f"✅ Built {args.out}")
        print(f"   raw rows: {rows:,} | teams: {teams:,} | schedule: {has_schedule}")
        print(
            "   ingested: "
            f"MatchRobot={summary['matchrobot']}, Super Scout={summary['superscout']}, "
            f"Pit={summary['pit']}, Schedule={summary['schedule']}, Skipped={summary['skipped']}"
        )
    finally:
        con.close()

def prefetch_tba(args):
    # writes TBA matches into the same duckdb file so you can be offline at the event
    out_path = Path(args.out)
    con = connect_db(out_path)
    try:
        schema_sql = (HERE/"tba_schema.sql").read_text(encoding="utf-8")
        con.execute(schema_sql)
        key = args.tba_key or os.environ.get("TBA_AUTH_KEY", "")
        if not key:
            raise SystemExit("Missing TBA key. Provide --tba-key or set TBA_AUTH_KEY.")
        event = args.event.strip()
        url = f"https://www.thebluealliance.com/api/v3/event/{event}/matches"
        r = requests.get(url, headers={"X-TBA-Auth-Key": key}, timeout=30)
        if r.status_code != 200:
            raise SystemExit(f"TBA error {r.status_code}: {r.text[:200]}")
        matches = r.json()
        rows = []
        for m in matches:
            rows.append({
                "key": m.get("key"),
                "comp_level": m.get("comp_level"),
                "match_number": m.get("match_number"),
                "set_number": m.get("set_number"),
                "alliances_json": json.dumps(m.get("alliances", {})),
                "event_key": event
            })
        df = pd.DataFrame(rows)
        con.register("df", df)
        con.execute("INSERT INTO tba_matches SELECT * FROM df")
        print(f"✅ Cached {len(df):,} TBA matches into {out_path.name}")
    finally:
        con.close()

def _prompt(text: str, default: Optional[str] = None) -> str:
    if default is not None and default != "":
        prompt = f"{text} [{default}]: "
    else:
        prompt = f"{text}: "
    val = input(prompt).strip()
    return val if val != "" else (default or "")

def interactive_menu(args):
    # Lazy imports so build/prefetch don't require plotting deps
    import numpy as np
    import matplotlib.pyplot as plt

    print("Venom Event Builder - Interactive Menu")
    print("1. Plot teleop pieces + algae (net) trends for a team")
    print("2. Plot auto pieces + algae (net) trends for a team")
    print("3. Plot auto vs teleop vs total match pieces (with algae) for a team")
    print("q. Quit")
    choice = _prompt("Select option", "1").lower()
    if choice in {"q", "quit", "exit"}:
        return
    if choice not in {"1", "2", "3"}:
        print("Unknown option.")
        return

    db_path = _prompt("DuckDB file", "event.duckdb")
    team_str = _prompt("Team number", "")
    if team_str == "":
        print("Team number is required.")
        return
    try:
        team = int(team_str)
    except ValueError:
        print("Invalid team number.")
        return

    start_match = _prompt("Start match (optional)", "")
    end_match = _prompt("End match (optional)", "")

    con = connect_db(Path(db_path))
    try:
        where = "team = ? AND match_num IS NOT NULL"
        params: List[object] = [team]
        if start_match != "":
            where += " AND match_num >= ?"
            params.append(int(start_match))
        if end_match != "":
            where += " AND match_num <= ?"
            params.append(int(end_match))

        if choice in {"1", "2"}:
            if choice == "1":
                coral_expr = "tele_l1 + tele_l2 + tele_l3 + tele_l4"
                algae_expr = "tele_net"
                title = "Teleop"
            else:
                coral_expr = "auto_l1 + auto_l2 + auto_l3 + auto_l4"
                algae_expr = "auto_net"
                title = "Auto"

            df = con.execute(f"""
            SELECT
              match_num,
              AVG({coral_expr}) AS avg_pieces,
              AVG({algae_expr}) AS avg_algae
            FROM raw
            WHERE {where}
            GROUP BY match_num
            ORDER BY match_num
            """, params).df()

            if df.empty:
                print("No data found for that selection.")
                return

            x = df["match_num"].to_numpy()
            y = df["avg_pieces"].to_numpy()
            y_algae = df["avg_algae"].to_numpy()

            plt.figure(figsize=(9, 5))
            plt.plot(x, y, marker="o", color="blue", label="Avg Pieces per Match")
            plt.plot(x, y_algae, marker="o", color="green", label="Avg Algae (Net) per Match")
            if len(x) >= 2:
                m, b = np.polyfit(x, y, 1)
                plt.plot(x, m*x + b, linestyle="--", color="blue", alpha=0.7, label="Pieces Trend")
                m2, b2 = np.polyfit(x, y_algae, 1)
                plt.plot(x, m2*x + b2, linestyle="--", color="green", alpha=0.7, label="Algae Trend")
            plt.title(f"Team {team} {title} Pieces + Algae Over Matches")
            plt.xlabel("Qual Match Number")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
        else:
            df = con.execute(f"""
            SELECT
              match_num,
              AVG(auto_l1 + auto_l2 + auto_l3 + auto_l4) AS avg_auto,
              AVG(tele_l1 + tele_l2 + tele_l3 + tele_l4) AS avg_tele,
              AVG(auto_net) AS avg_auto_algae,
              AVG(tele_net) AS avg_tele_algae
            FROM raw
            WHERE {where}
            GROUP BY match_num
            ORDER BY match_num
            """, params).df()

            if df.empty:
                print("No data found for that selection.")
                return

            df["avg_algae_total"] = df["avg_auto_algae"] + df["avg_tele_algae"]
            df["avg_combined"] = df["avg_auto"] + df["avg_tele"] + df["avg_algae_total"]

            x = df["match_num"].to_numpy()

            plt.figure(figsize=(10, 6))
            for col, color, label in [
                ("avg_auto", "green", "Auto Pieces"),
                ("avg_tele", "blue", "Tele Pieces"),
                ("avg_algae_total", "orange", "Total Algae (Net)"),
                ("avg_combined", "purple", "Total Match Pieces"),
            ]:
                y = df[col].to_numpy()
                plt.plot(x, y, marker="o", color=color, label=f"{label} Avg")
                if len(x) >= 2:
                    m, b = np.polyfit(x, y, 1)
                    plt.plot(x, m*x + b, linestyle="--", color=color, alpha=0.7, label=f"{label} Trend")

            plt.title(f"Team {team} Auto vs Teleop vs Combined Pieces")
            plt.xlabel("Qual Match Number")
            plt.ylabel("Pieces (L1+L2+L3+L4)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
    finally:
        con.close()

def main():
    p = argparse.ArgumentParser(prog="event_builder", description="Build offline DuckDB event file from scouting CSVs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build .duckdb from CSVs")
    b.add_argument("--raw", required=True, help="Path to raw scouting CSV (required)")
    b.add_argument("--schedule", default="", help="Path to schedule CSV (optional)")
    b.add_argument("--pit", default="", help="Path to pit CSV (optional)")
    b.add_argument("--super-scout", default="", help="Path to Super Scout auto CSV (optional)")
    b.add_argument("--out", required=True, help="Output .duckdb file")
    b.add_argument("--event-key", default="2026TEST", help="Default event key when source CSV omits one")
    b.add_argument("--mapping", default=str(HERE/"mapping.json"), help="Mapping json path")
    b.set_defaults(func=build)

    t = sub.add_parser("prefetch-tba", help="Cache TBA matches into the .duckdb for offline use")
    t.add_argument("--event", required=True, help="TBA event key, e.g. 2026miket")
    t.add_argument("--out", required=True, help="Existing .duckdb file to write into")
    t.add_argument("--tba-key", default="", help="TBA API key (or set env var TBA_AUTH_KEY)")
    t.set_defaults(func=prefetch_tba)

    m = sub.add_parser("menu", help="Interactive plotting menu")
    m.set_defaults(func=interactive_menu)

    args = p.parse_args()
    # normalize empty strings
    if hasattr(args, "schedule") and args.schedule.strip() == "":
        args.schedule = ""
    if hasattr(args, "pit") and args.pit.strip() == "":
        args.pit = ""
    if hasattr(args, "super_scout") and args.super_scout.strip() == "":
        args.super_scout = ""
    args.func(args)

if __name__ == "__main__":
    main()
