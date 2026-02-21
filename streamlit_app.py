import json
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
from datetime import datetime
from dotenv import load_dotenv
import streamlit.components.v1 as components
from PIL import Image
from schema_2026 import MATCHROBOT_REQUIRED
from ingest_backend import (
    ensure_ingest_tables,
    ingest_upload_bytes,
    refresh_compat_views,
    bootstrap_from_legacy_raw,
)
try:
    import streamlit.elements.image as st_image
    from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
    # Compatibility shim for streamlit-drawable-canvas, which calls image_to_url
    # with the old signature: (image, width:int, clamp, channels, output_format, image_id).
    class _LegacyLayoutConfig:
        def __init__(self, width):
            self.width = width

    def _compat_image_to_url(image, layout_or_width, clamp, channels, output_format, image_id):
        layout_config = (
            layout_or_width
            if hasattr(layout_or_width, "width")
            else _LegacyLayoutConfig(layout_or_width)
        )
        return _image_to_url(image, layout_config, clamp, channels, output_format, image_id)

    st_image.image_to_url = _compat_image_to_url
except Exception:
    pass
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("VENOM_DATA_DIR", "./data")).expanduser()
ACTIVE_DB_POINTER = DATA_DIR / ".venom_active_db"
EMPTY_DB_PATH = DATA_DIR / "empty.duckdb"
NOTES_PATH = DATA_DIR / "match_notes.json"
AUTO_PATH_DIR = DATA_DIR / "auto_paths"
AUTO_PATH_INDEX = AUTO_PATH_DIR / "index.json"
DEFAULT_FIELD_IMAGE = APP_DIR / "assets" / "field_map_2026.png"


def _resolve_active_db_path() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    explicit = os.environ.get("VENOM_DB_PATH", "").strip()
    if explicit:
        return explicit
    if ACTIVE_DB_POINTER.exists():
        p = ACTIVE_DB_POINTER.read_text(encoding="utf-8").strip()
        if p:
            return p
    return str(EMPTY_DB_PATH)


def _write_active_db_pointer(path: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_DB_POINTER.write_text(path.strip(), encoding="utf-8")


def load_teams(con: duckdb.DuckDBPyConnection) -> pd.Series:
    df = con.execute(
        "SELECT DISTINCT team FROM raw WHERE team IS NOT NULL ORDER BY team"
    ).df()
    return df["team"]

def load_team_matches(con: duckdb.DuckDBPyConnection, team: int) -> list:
    df = con.execute(
        """
        SELECT DISTINCT match_num
        FROM raw
        WHERE team = ? AND match_num IS NOT NULL
        ORDER BY match_num
        """,
        [int(team)],
    ).df()
    return df["match_num"].tolist()

def get_pit_team_col(con: duckdb.DuckDBPyConnection) -> str:
    try:
        cols = con.execute("PRAGMA table_info('pit')").df()["name"].tolist()
    except Exception:
        return ""
    candidates = ["Team Number", "team", "Team", "team_number", "team_num"]
    low = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in low:
            return low[key]
    return ""

def pit_field_map():
    # For pit CSV with no headers: team + pit_col_1..pit_col_12
    return {
        "team": "Team Number",
        "pit_col_1": "Robot Photo A",
        "pit_col_2": "Robot Photo B",
        "pit_col_3": "Robot Photo C",
        "pit_col_4": "Robot Photo D",
        "pit_col_5": "Auto Starting Location",
        "pit_col_6": "Starting Location",
        "pit_col_7": "Auto Routine (Movement/Scoring)",
        "pit_col_8": "Drive Base Type",
        "pit_col_9": "Robot Weight",
        "pit_col_10": "Climb Speed",
        "pit_col_11": "Extra Notes",
        "pit_col_12": "Robot Photo M",
    }

def load_auto_path_index() -> dict:
    if not AUTO_PATH_INDEX.exists():
        return {}
    try:
        return json.loads(AUTO_PATH_INDEX.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_auto_path_index(index: dict) -> None:
    AUTO_PATH_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_PATH_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")

def save_auto_path_image(image_data: np.ndarray, event_key: str, team_num: int, match_num: int) -> str:
    AUTO_PATH_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = AUTO_PATH_DIR / (event_key or "unknown_event")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"team_{int(team_num)}_match_{int(match_num)}.png"
    arr = np.asarray(image_data).astype("uint8")
    Image.fromarray(arr, mode="RGBA").save(out_path)
    return str(out_path)

def get_auto_paths_for_team(event_key: str, team_num: int) -> list:
    idx = load_auto_path_index()
    key = f"{event_key}|{int(team_num)}"
    return idx.get(key, [])

@st.cache_data(show_spinner=False)
def load_image_bytes_cached(path_str: str, mtime: float) -> bytes:
    # mtime is included so cache invalidates when file changes.
    return Path(path_str).read_bytes()

def detect_match_table(con: duckdb.DuckDBPyConnection) -> str:
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    for t in tables:
        try:
            cols = con.execute(f"PRAGMA table_info('{t}')").df()["name"].tolist()
        except Exception:
            continue
        low = {c.lower() for c in cols}
        if "team" in low and "match_num" in low:
            return t
    for t in tables:
        try:
            cols = con.execute(f"PRAGMA table_info('{t}')").df()["name"].tolist()
        except Exception:
            continue
        low = {c.lower() for c in cols}
        if "team" in low:
            return t
    return ""


def relation_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    try:
        return bool(
            con.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_name = ?
                LIMIT 1
                """,
                [name],
            ).fetchone()
        )
    except Exception:
        # Fallback for older catalogs.
        return name in set(con.execute("SHOW TABLES").df()["name"].tolist())

def parse_quick_query(q: str) -> list:
    import re

    q = q.lower().strip()
    if not q:
        return []

    # Normalize separators and phrasing.
    q = q.replace("teleop", "tele")
    q = q.replace("autonomous", "auto")
    q = q.replace("prepped", "prep")
    q = q.replace("prepping", "prep")
    q = q.replace("scoring", "score")
    q = q.replace("scpred", "scored")
    q = q.replace("average", "avg")
    q = re.sub(r"\s+", " ", q)

    alias_map = {
        "avg auto score": "avg_auto_score",
        "auto score": "avg_auto_score",
        "score in auto": "avg_auto_score",
        "scored in auto": "avg_auto_score",
        "auto points": "avg_auto_score",
        "active cycles": "avg_active_cycles",
        "cycles": "avg_active_cycles",
        "cycle": "avg_active_cycles",
        "active shooting": "avg_active_shooting_est",
        "shooting estimate": "avg_active_shooting_est",
        "active shots": "avg_active_shooting_est",
        "shooting": "avg_active_shooting_est",
        "shots": "avg_active_shooting_est",
        "inactive ferry": "avg_inactive_ferry",
        "ferry": "avg_inactive_ferry",
        "ferry estimate": "avg_inactive_ferry",
        "prep rate": "prep_rate",
        "prepped to shoot": "prep_rate",
        "prep to shoot": "prep_rate",
        "staged to shoot": "prep_rate",
        "endgame score": "avg_endgame_score",
        "endgame": "avg_endgame_score",
        "climb score": "avg_endgame_score",
        "climb": "avg_endgame_score",
        "climbed": "avg_endgame_score",
        "auto pickups": "avg_auto_pickups",
        "alliance zone pickups": "avg_auto_pickups",
        "pickups": "avg_auto_pickups",
        "leave alliance zone": "leave_rate",
        "left alliance zone": "leave_rate",
        "enter neutral zone": "neutral_enter_rate",
        "entered neutral zone": "neutral_enter_rate",
        "auto win": "auto_win_rate",
        "win auto": "auto_win_rate",
        "combined impact": "combined_impact",
        "impact": "combined_impact",
        "points score": "points_score",
        "rp score": "rp_score",
        "quals score": "quals_score",
    }

    cmp_tokens = [
        ("greater than or equal to", ">="),
        ("less than or equal to", "<="),
        ("at least", ">="),
        ("at most", "<="),
        ("greater than", ">"),
        ("higher than", ">"),
        ("more than", ">"),
        ("less than", "<"),
        ("under", "<"),
        ("over", ">"),
    ]

    def find_field(segment: str) -> str:
        # Prefer longer alias matches first.
        for alias, col in sorted(alias_map.items(), key=lambda x: len(x[0]), reverse=True):
            if alias in segment:
                return col
        return ""

    def parse_segment(segment: str):
        s = segment.strip()
        if not s:
            return None

        field = find_field(s)
        if not field and re.search(r"\b(?:score|scored|scores)\s+in\s+auto\b", s):
            field = "avg_auto_score"
        if not field and re.search(r"\b(?:climb|climbed)\b", s):
            field = "avg_endgame_score"
        if not field and re.search(r"\b(?:cycle|cycles)\b", s):
            field = "avg_active_cycles"
        if not field and re.search(r"\b(?:ferry)\b", s):
            field = "avg_inactive_ferry"
        if not field and re.search(r"\b(?:prep|prepped|staged)\b", s):
            field = "prep_rate"
        if not field:
            return None

        # Symbol comparator forms.
        m = re.search(r"([<>]=?|=)\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            return (field, m.group(1), m.group(2))

        # Word comparator forms.
        for token, op in cmp_tokens:
            if token in s:
                num = re.search(rf"{re.escape(token)}\s*(-?\d+(?:\.\d+)?)", s)
                if num:
                    return (field, op, num.group(1))

        # Binary presence style.
        if re.search(r"\b(no|without|none)\b", s):
            return (field, "=", "0")
        if re.search(r"\b(has|have|with|scored|score|does|did)\b", s):
            return (field, ">", "0")

        # Default fallback if field mentioned without comparator.
        return (field, ">", "0")

    # Split on common conjunctions and punctuation.
    segments = re.split(r",|;|\band\b|\bthen\b|&&", q)
    parsed = []
    for seg in segments:
        cond = parse_segment(seg)
        if cond:
            parsed.append(cond)

    # De-duplicate while preserving order.
    dedup = []
    seen = set()
    for c in parsed:
        key = (c[0], c[1], c[2])
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    return dedup

def compute_archetypes(con: duckdb.DuckDBPyConnection, match_table: str) -> pd.DataFrame:
    cols = set(
        c.lower() for c in con.execute(f"PRAGMA table_info('{match_table}')").df()["name"].tolist()
    )
    active_cycles_expr = "active_cycles" if "active_cycles" in cols else "0"
    if "active_shooting_est" in cols:
        active_shoot_expr = "active_shooting_est"
    elif "active_shoot_est_mid" in cols:
        active_shoot_expr = "active_shoot_est_mid"
    else:
        active_shoot_expr = "0"
    endgame_expr = "endgame_score" if "endgame_score" in cols else "0"
    if "active_defense_level" in cols and "inactive_defense_level" in cols:
        defense_expr = "(COALESCE(active_defense_level, 0) + COALESCE(inactive_defense_level, 0))"
    elif "active_defense_level" in cols:
        defense_expr = "COALESCE(active_defense_level, 0)"
    elif "inactive_defense_level" in cols:
        defense_expr = "COALESCE(inactive_defense_level, 0)"
    else:
        defense_expr = "0"
    if "inactive_ferry_est" in cols:
        ferry_expr = "COALESCE(inactive_ferry_est, 0)"
    elif "ferry_est_mid" in cols:
        ferry_expr = "COALESCE(ferry_est_mid, 0)"
    else:
        ferry_expr = "0"
    df = con.execute(
        f"""
        SELECT
          team,
          AVG({active_cycles_expr}) AS active_cycles,
          AVG({active_shoot_expr}) AS active_shooting,
          AVG({endgame_expr}) AS endgame,
          AVG({defense_expr}) AS defense_pressure,
          AVG({ferry_expr}) AS ferry
        FROM {match_table}
        WHERE team IS NOT NULL
        GROUP BY team
        """
    ).df()
    if df.empty:
        return df
    cycles_hi = df["active_cycles"].quantile(0.70)
    shoot_hi = df["active_shooting"].quantile(0.70)
    end_hi = df["endgame"].quantile(0.70)
    def_hi = df["defense_pressure"].quantile(0.70)
    ferry_hi = df["ferry"].quantile(0.70)
    shoot_ratio = df["active_shooting"] / df["active_cycles"].clip(lower=1)
    ratio_hi = shoot_ratio.quantile(0.70)
    ratio_lo = shoot_ratio.quantile(0.30)
    cycles_mid = df["active_cycles"].quantile(0.50)

    def classify(row):
        tags = []
        ratio = row["active_shooting"] / max(row["active_cycles"], 1)
        if row["active_cycles"] >= cycles_hi:
            tags.append("Volume Shooter")
        if ratio >= ratio_hi and row["active_shooting"] >= shoot_hi:
            tags.append("Accuracy Shooter")
        if row["active_cycles"] >= cycles_mid and ratio <= ratio_lo:
            tags.append("Under-Performing Shooter")
        if row["ferry"] >= ferry_hi:
            tags.append("Passer")
        if row["defense_pressure"] >= def_hi:
            tags.append("Defender")
        if row["endgame"] >= end_hi:
            tags.append("Endgame Player")
        if not tags:
            tags.append("Under-Performing Shooter")
        return ", ".join(tags)

    df["archetype"] = df.apply(classify, axis=1)
    return df[["team", "archetype"]]

def apply_match_filters(df: pd.DataFrame, start_match, end_match, exclude_matches: list) -> pd.DataFrame:
    if start_match is not None:
        df = df[df["match_num"] >= start_match]
    if end_match is not None:
        df = df[df["match_num"] <= end_match]
    if exclude_matches:
        df = df[~df["match_num"].isin(exclude_matches)]
    return df

def drop_lowest_n(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df
    n = min(n, len(df))
    return df.sort_values(col, ascending=True).iloc[n:]

def smooth_series(y: np.ndarray, window: int = 3) -> np.ndarray:
    if len(y) < window:
        return y
    s = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean()
    return s.to_numpy()

def ensure_prematch_schema(con: duckdb.DuckDBPyConnection) -> None:
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
    # Backward-compatible columns for older prematch_plan schema.
    con.execute("ALTER TABLE prematch_plan ADD COLUMN IF NOT EXISTS assignments_json TEXT")
    con.execute("ALTER TABLE prematch_plan ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP")
    con.execute("ALTER TABLE prematch_plan ADD COLUMN IF NOT EXISTS alliance TEXT")
    con.execute("ALTER TABLE prematch_plan ADD COLUMN IF NOT EXISTS red_roles TEXT")
    con.execute("ALTER TABLE prematch_plan ADD COLUMN IF NOT EXISTS blue_roles TEXT")
    con.execute("ALTER TABLE prematch_plan ADD COLUMN IF NOT EXISTS auto_notes TEXT")
    # Views are created by the builder from views.sql.
    # Avoid re-running full view DDL on every Streamlit rerun to prevent write-write conflicts.
    if not st.session_state.get("_prematch_views_checked", False):
        st.session_state["_prematch_views_checked"] = True

def import_schedule_csv_to_db(
    con: duckdb.DuckDBPyConnection,
    event_key: str,
    upload_bytes: bytes,
    filename: str,
) -> tuple[bool, str]:
    if not event_key.strip():
        return False, "Event key is required."
    try:
        df = pd.read_csv(BytesIO(upload_bytes))
    except Exception as exc:
        return False, f"Could not read CSV: {exc}"
    if df.empty:
        return False, "CSV is empty."

    cols = {c.lower().strip(): c for c in df.columns}
    long_needed = {"match_number", "alliance", "position", "team"}
    wide_needed = {"match_number", "red1", "red2", "red3", "blue1", "blue2", "blue3"}

    normalized = None
    if long_needed.issubset(set(cols.keys())):
        use = df.rename(columns={cols[k]: k for k in long_needed})
        use["alliance"] = use["alliance"].astype(str).str.strip().str.lower()
        use["position"] = pd.to_numeric(use["position"], errors="coerce").astype("Int64")
        use["team"] = pd.to_numeric(use["team"], errors="coerce").astype("Int64")
        use["match_number"] = pd.to_numeric(use["match_number"], errors="coerce").astype("Int64")
        rows = []
        for match_num, grp in use.groupby("match_number"):
            if pd.isna(match_num):
                continue
            row = {
                "event_key": event_key.strip(),
                "match_number": int(match_num),
                "comp_level": "qm",
                "red1": None,
                "red2": None,
                "red3": None,
                "blue1": None,
                "blue2": None,
                "blue3": None,
                "updated_at": pd.Timestamp.utcnow(),
            }
            for _, r in grp.iterrows():
                if pd.isna(r["team"]) or pd.isna(r["position"]):
                    continue
                al = str(r["alliance"]).lower()
                pos = int(r["position"])
                if al in {"red", "blue"} and pos in {1, 2, 3}:
                    row[f"{al}{pos}"] = int(r["team"])
            rows.append(row)
        normalized = pd.DataFrame(rows)
    elif wide_needed.issubset(set(cols.keys())):
        rename = {cols[k]: k for k in wide_needed}
        use = df.rename(columns=rename).copy()
        use["event_key"] = event_key.strip()
        use["comp_level"] = "qm"
        use["updated_at"] = pd.Timestamp.utcnow()
        for c in ["match_number", "red1", "red2", "red3", "blue1", "blue2", "blue3"]:
            use[c] = pd.to_numeric(use[c], errors="coerce").astype("Int64")
        normalized = use[
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
        ].dropna(subset=["match_number"])
    else:
        return False, (
            "Unsupported schedule CSV format. "
            "Use wide: match_number,red1,red2,red3,blue1,blue2,blue3 "
            "or long: match_number,alliance,position,team."
        )

    if normalized is None or normalized.empty:
        return False, "No schedule rows parsed from CSV."

    normalized["match_number"] = normalized["match_number"].astype(int)
    con.execute("DELETE FROM schedule WHERE event_key = ?", [event_key.strip()])
    con.register("sched_import_df", normalized)
    con.execute(
        """
        INSERT INTO schedule
        SELECT event_key, match_number, comp_level, red1, red2, red3, blue1, blue2, blue3, updated_at
        FROM sched_import_df
        """
    )
    return True, f"Imported {len(normalized)} matches from {filename}."


def check_tba_key(api_key: str, event_key: str) -> Tuple[bool, str]:
    if not api_key or not event_key:
        return False, "Missing API key or event key."
    url = f"https://www.thebluealliance.com/api/v3/event/{event_key}"
    try:
        r = requests.get(url, headers={"X-TBA-Auth-Key": api_key}, timeout=15)
        if r.status_code == 200:
            return True, "OK"
        return False, f"{r.status_code}: {r.text[:120]}"
    except Exception as exc:
        return False, str(exc)

def get_match_videos(api_key: str, event_key: str, match_num: int) -> list:
    if not api_key or not event_key or match_num is None:
        return []
    match_key = f"{event_key}_qm{int(match_num)}"
    url = f"https://www.thebluealliance.com/api/v3/match/{match_key}"
    try:
        r = requests.get(url, headers={"X-TBA-Auth-Key": api_key}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("videos", []) or []
    except Exception:
        return []

def get_team_key(api_key: str, team_num: int) -> str:
    if not api_key or not team_num:
        return ""
    team_key = f"frc{int(team_num)}"
    url = f"https://www.thebluealliance.com/api/v3/team/{team_key}"
    try:
        r = requests.get(url, headers={"X-TBA-Auth-Key": api_key}, timeout=15)
        if r.status_code != 200:
            return ""
        data = r.json()
        return data.get("key", team_key)
    except Exception:
        return ""

def get_event_teams(api_key: str, event_key: str) -> list:
    if not api_key or not event_key:
        return []
    url = f"https://www.thebluealliance.com/api/v3/event/{event_key}/teams/simple"
    try:
        r = requests.get(url, headers={"X-TBA-Auth-Key": api_key}, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        return sorted([t.get("team_number") for t in data if t.get("team_number") is not None])
    except Exception:
        return []

def year_from_event_key(event_key: str) -> str:
    if not event_key:
        return str(datetime.now().year)
    return event_key[:4] if len(event_key) >= 4 and event_key[:4].isdigit() else str(datetime.now().year)

def get_team_info(api_key: str, team_num: int) -> dict:
    if not api_key or not team_num:
        return {}
    team_key = f"frc{int(team_num)}"
    url = f"https://www.thebluealliance.com/api/v3/team/{team_key}"
    try:
        r = requests.get(url, headers={"X-TBA-Auth-Key": api_key}, timeout=15)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

def get_team_events(api_key: str, team_num: int, year: int) -> list:
    if not api_key or not team_num:
        return []
    team_key = f"frc{int(team_num)}"
    url = f"https://www.thebluealliance.com/api/v3/team/{team_key}/events/{year}/simple"
    try:
        r = requests.get(url, headers={"X-TBA-Auth-Key": api_key}, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        return [e.get("key") for e in data if e.get("key")]
    except Exception:
        return []

def autodetect_event_key(api_key: str, teams: list, year: int) -> str:
    # Sample first few teams to reduce API calls
    sample = teams[:5]
    counts = {}
    for t in sample:
        for ev in get_team_events(api_key, t, year):
            counts[ev] = counts.get(ev, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda x: x[1])[0]
def get_event_matches(api_key: str, event_key: str) -> list:
    if not api_key or not event_key:
        return []
    url = f"https://www.thebluealliance.com/api/v3/event/{event_key}/matches"
    try:
        r = requests.get(url, headers={"X-TBA-Auth-Key": api_key}, timeout=20)
        if r.status_code != 200:
            return []
        return r.json()
    except Exception:
        return []

def next_match_by_number(matches: list, team_key: str) -> dict:
    # Choose smallest upcoming qm match number for team
    team_matches = []
    for m in matches:
        if m.get("comp_level") != "qm":
            continue
        alliances = m.get("alliances", {})
        red = alliances.get("red", {}).get("team_keys", [])
        blue = alliances.get("blue", {}).get("team_keys", [])
        if team_key in red or team_key in blue:
            team_matches.append(m)
    if not team_matches:
        return {}
    team_matches.sort(key=lambda x: x.get("match_number", 0))
    return team_matches[0]

def main():
    if os.environ.get("DEV_MODE", "0") == "1":
        load_dotenv()
    st.set_page_config(page_title="Venom Analysis", layout="wide")
    st.title("Venom Analysis")
    # Quick search bar at top
    top_query = st.text_input(
        "Quick Search",
        placeholder="e.g., scored in auto, active cycles > 6, climbed",
        key="quick_query_top",
    )
    if top_query and top_query.strip():
        st.session_state["show_quick_tab"] = True
        # Only seed the tab query when it is empty, so users can edit it independently.
        if not st.session_state.get("quick_query", "").strip():
            st.session_state["quick_query"] = top_query.strip()

    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp {
            background: radial-gradient(circle at 20% 20%, #2b1055 0%, #0b0b0f 40%, #000000 100%);
            color: #f5f5f5;
        }
        h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stText, .stCaption {
            color: #f5f5f5 !important;
        }
        .stButton>button {
            background: #f1c40f;
            color: #000000;
            border: 1px solid #f1c40f;
        }
        .stButton>button:hover {
            background: #ffd84d;
            color: #000000;
        }
        .stRadio > div { color: #f5f5f5; }
        .sortable-container, .sortable-list, .sortable-item, .sortable-item * {
            background: #2b1055 !important;
            color: #f1c40f !important;
            border: 1px solid #4a2a88 !important;
        }
        .sortable-item {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 12px;
            padding: 10px 12px !important;
            border-radius: 6px !important;
            box-shadow: none !important;
        }
        .sortable-item:hover {
            background: #341767 !important;
        }
        div[data-baseweb="select"] {
            background: #3b3b3b !important;
            border: 2px solid #6a0dad !important;
        }
        div[data-baseweb="select"] > div {
            background: #3b3b3b !important;
            border-color: #6a0dad !important;
        }
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
            color: #f1c40f !important;
            background: #3b3b3b !important;
        }
        div[data-baseweb="select"] [role="listbox"],
        div[data-baseweb="select"] [role="option"] {
            background: #2b1055 !important;
            color: #f1c40f !important;
        }
        div[data-baseweb="select"] [role="option"][aria-selected="true"] {
            background: #4a2a88 !important;
            color: #f1c40f !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #f1c40f !important;
            border-bottom: 3px solid #f1c40f !important;
        }
        div[data-baseweb="select"] [role="option"]:hover,
        div[data-baseweb="select"] [role="option"][aria-selected="true"]:hover {
            background: #6a0dad !important;
            color: #f1c40f !important;
        }
        .stMarkdown span, .stTextInput > div > div > input, .stSelectbox > div > div {
            color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "db_path" not in st.session_state:
        st.session_state["db_path"] = _resolve_active_db_path()
    db_path = st.session_state.get("db_path", _resolve_active_db_path())
    matches_played = st.session_state.get("matches_played", 0)
    try:
        con = duckdb.connect(db_path)
    except Exception as exc:
        st.error(f"Failed to open DB: {exc}")
        return

    with con:
        ensure_prematch_schema(con)
        ensure_ingest_tables(con)
        bootstrap_from_legacy_raw(con)
        refresh_compat_views(con)
        try:
            con.execute((APP_DIR / "views.sql").read_text(encoding="utf-8"))
        except Exception:
            pass
        has_loaded_data = False
        try:
            has_loaded_data = int(con.execute("SELECT COUNT(*) FROM raw_match").fetchone()[0]) > 0
        except Exception:
            has_loaded_data = False
        if not has_loaded_data:
            st.info("No data loaded. Go to Admin → Unified Import and upload scouting CSV/ZIP.")
        teams = load_teams(con)
        if teams.empty:
            teams = pd.Series([0], name="team")

        team_ids = [int(teams.iloc[0])]
        team = team_ids[0]

        # Archetype guess will be shown inside Data and Comparison tab only

        tab_labels = ["Data and Comparison", "Team Info", "Pre-Match", "Auto Path", "Picklist", "Team Lookup", "Video Review", "Definitions"]
        if st.session_state.get("show_quick_tab", False):
            tab_labels.append("Quick Search")
        tabs = st.tabs(tab_labels)
        tab_data, tab_match, tab_prematch, tab_autopath, tab_pick, tab_team, tab_watch, tab_defs = tabs[:8]
        tab_quick = tabs[8] if len(tabs) > 8 else None

        def load_notes() -> list:
            if not NOTES_PATH.exists():
                return []
            try:
                return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []

        def save_notes(notes: list) -> None:
            NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
            NOTES_PATH.write_text(json.dumps(notes, indent=2), encoding="utf-8")

        with tab_data:
            st.subheader("Metrics")
            team_list_raw = st.text_input(
                "Teams (comma-separated, e.g. 8044, 245)", value=str(int(teams.iloc[0]))
            )
            team_ids = []
            for part in team_list_raw.split(","):
                part = part.strip()
                if part.isdigit():
                    team_ids.append(int(part))
            team_ids = list(dict.fromkeys(team_ids))
            if not team_ids:
                st.warning("Enter at least one valid team number. Using the first team as fallback.")
                team_ids = [int(teams.iloc[0])]
            st.session_state["team_ids_input"] = team_ids

            has_scores_view = relation_exists(con, "v_team_scores_2026")
            has_metric_catalog = relation_exists(con, "v_metric_catalog_2026")
            if not has_scores_view:
                st.warning("`v_team_scores_2026` not found. Rebuild DB.")
            else:
                if has_metric_catalog:
                    catalog = con.execute(
                        "SELECT metric_key, metric_label FROM v_metric_catalog_2026"
                    ).df()
                    metric_label_to_key = {
                        str(r["metric_label"]): str(r["metric_key"]) for _, r in catalog.iterrows()
                    }
                else:
                    metric_label_to_key = {
                        "Quals Score": "QualsScore",
                        "Points Score": "PointsScore",
                        "RP Score": "RPScore",
                        "Avg Active Cycles": "avg_active_cycles",
                        "Avg Active Shooting": "avg_active_shooting_est",
                        "Avg Endgame Score": "avg_endgame_score",
                        "Prep Rate": "prep_rate",
                        "Avg Inactive Ferry": "avg_inactive_ferry",
                        "Climb Success Rate": "climb_success_rate",
                        "Avg Auto Score": "avg_auto_score",
                        "Miss Heavy Rate": "miss_heavy_rate",
                    }

                metric_label = st.selectbox("Metric", list(metric_label_to_key.keys()), key="metric_key_2026")
                metric_key = metric_label_to_key[metric_label]
                st.caption("View: Per Match Trend")

                placeholders = ",".join(["?"] * len(team_ids))
                match_table = detect_match_table(con)
                arche_df = compute_archetypes(con, match_table) if match_table else pd.DataFrame()
                raw_cols = set(con.execute("PRAGMA table_info('raw')").df()["name"].str.lower().tolist())
                if "active_shooting_est" in raw_cols:
                    shoot_expr = "COALESCE(active_shooting_est, 0)"
                elif "active_shoot_est_mid" in raw_cols:
                    shoot_expr = "COALESCE(active_shoot_est_mid, 0)"
                else:
                    shoot_expr = "0"
                if "inactive_ferry_est" in raw_cols:
                    ferry_expr = "COALESCE(inactive_ferry_est, 0)"
                elif "ferry_est_mid" in raw_cols:
                    ferry_expr = "COALESCE(ferry_est_mid, 0)"
                else:
                    ferry_expr = "0"

                per_match_expr_map = {
                    "avg_auto_score": "COALESCE(auto_score, 0)",
                    "avg_active_cycles": "COALESCE(active_cycles, 0)",
                    "avg_active_shooting_est": shoot_expr,
                    "avg_endgame_score": "COALESCE(endgame_score, 0)",
                    "prep_rate": "COALESCE(inactive_prepped_to_shoot, 0)",
                    "avg_inactive_ferry": ferry_expr,
                    "climb_success_rate": "CASE WHEN COALESCE(climb_level, 0) > 0 AND (climb_failed = 0 OR climb_failed IS NULL) THEN 1 ELSE 0 END",
                    "miss_heavy_rate": "COALESCE(active_miss_heavy, 0)",
                    "RPScore": (
                        "100.0 * (0.30*LEAST(COALESCE(active_cycles,0)/12.0,1.0) + "
                        f"0.25*LEAST({shoot_expr}/5.0,1.0) + "
                        "0.25*LEAST(COALESCE(endgame_score,0)/30.0,1.0) + "
                        f"0.10*LEAST({ferry_expr}/3.0,1.0) + "
                        "0.10*COALESCE(inactive_prepped_to_shoot,0))"
                    ),
                    "PointsScore": (
                        "100.0 * (0.35*LEAST(COALESCE(active_cycles,0)/12.0,1.0) + "
                        f"0.35*LEAST({shoot_expr}/5.0,1.0) + "
                        "0.20*LEAST(COALESCE(endgame_score,0)/30.0,1.0) + "
                        f"0.10*LEAST({ferry_expr}/3.0,1.0))"
                    ),
                    "QualsScore": (
                        "(100.0 * (0.35*LEAST(COALESCE(active_cycles,0)/12.0,1.0) + "
                        f"0.35*LEAST({shoot_expr}/5.0,1.0) + "
                        "0.20*LEAST(COALESCE(endgame_score,0)/30.0,1.0) + "
                        f"0.10*LEAST({ferry_expr}/3.0,1.0))) + "
                        "0.30 * (100.0 * (0.30*LEAST(COALESCE(active_cycles,0)/12.0,1.0) + "
                        f"0.25*LEAST({shoot_expr}/5.0,1.0) + "
                        "0.25*LEAST(COALESCE(endgame_score,0)/30.0,1.0) + "
                        f"0.10*LEAST({ferry_expr}/3.0,1.0) + "
                        "0.10*COALESCE(inactive_prepped_to_shoot,0)))"
                    ),
                }
                expr = per_match_expr_map.get(metric_key, "")
                if not expr:
                    st.info("No per-match expression available for this metric.")
                else:
                    per_match_sql = f"""
                        SELECT
                          team,
                          match_num,
                          AVG({expr}) AS metric_value
                        FROM raw
                        WHERE team IN ({placeholders}) AND match_num IS NOT NULL
                        GROUP BY team, match_num
                        ORDER BY team, match_num
                        """
                    try:
                        per_match = con.execute(per_match_sql, team_ids).df()
                    except duckdb.BinderException as e:
                        # Some older event DBs don't have the new 2026 columns.
                        fallback_expr = expr
                        missing = str(e).lower()
                        if "active_shoot_est_mid" in missing:
                            fallback_expr = fallback_expr.replace("active_shoot_est_mid", "0")
                        if "ferry_est_mid" in missing:
                            fallback_expr = fallback_expr.replace("ferry_est_mid", "0")
                        if fallback_expr == expr:
                            raise
                        per_match_sql = f"""
                            SELECT
                              team,
                              match_num,
                              AVG({fallback_expr}) AS metric_value
                            FROM raw
                            WHERE team IN ({placeholders}) AND match_num IS NOT NULL
                            GROUP BY team, match_num
                            ORDER BY team, match_num
                            """
                        per_match = con.execute(per_match_sql, team_ids).df()
                    if per_match.empty:
                        st.warning("No per-match data found for selected teams.")
                    else:
                        fig, ax = plt.subplots(figsize=(7.5, 3.6))
                        for team_id in sorted(per_match["team"].unique()):
                            d = per_match[per_match["team"] == team_id]
                            ax.plot(
                                d["match_num"].to_numpy(),
                                d["metric_value"].to_numpy(),
                                marker="o",
                                label=f"Team {int(team_id)}",
                            )
                            if len(d) >= 2:
                                x = d["match_num"].to_numpy()
                                y = d["metric_value"].to_numpy()
                                m, b = np.polyfit(x, y, 1)
                                ax.plot(x, m * x + b, linestyle="--", alpha=0.6)
                        ax.set_title(f"{metric_label} (Per Match)")
                        ax.set_xlabel("Qual Match Number")
                        ax.set_ylabel(metric_label)
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        st.dataframe(per_match, use_container_width=True)

                with st.expander("Data Health", expanded=False):
                    raw_cols_now = set(con.execute("PRAGMA table_info('raw')").df()["name"].tolist())
                    missing_required = [c for c in MATCHROBOT_REQUIRED if c not in raw_cols_now]
                    present_required = [c for c in MATCHROBOT_REQUIRED if c in raw_cols_now]
                    st.caption(
                        f"Required columns present: {len(present_required)}/{len(MATCHROBOT_REQUIRED)}"
                    )
                    if missing_required:
                        st.warning(f"Missing required columns: {missing_required}")
                    else:
                        st.success("All required AppSheet MatchRobot columns are present.")

                    try:
                        health = con.execute(
                            """
                            SELECT
                              AVG(CASE WHEN COALESCE(ferried, FALSE) = TRUE
                                       AND COALESCE(TRIM(CAST(ferry_bucket AS VARCHAR)), '') = ''
                                       THEN 1 ELSE 0 END) AS pct_ferry_missing,
                              AVG(CASE WHEN COALESCE(played_defense, FALSE) = TRUE
                                       AND COALESCE(TRIM(CAST(defense_effective AS VARCHAR)), '') = ''
                                       THEN 1 ELSE 0 END) AS pct_def_missing
                            FROM raw
                            """
                        ).fetchone()
                        if health:
                            ferry_pct = float(health[0] or 0.0) * 100.0
                            def_pct = float(health[1] or 0.0) * 100.0
                            st.markdown(f"- `ferried=TRUE` but `ferry_bucket` missing: **{ferry_pct:.1f}%**")
                            st.markdown(f"- `played_defense=TRUE` but `defense_effective` missing: **{def_pct:.1f}%**")
                    except Exception as exc:
                        st.caption(f"Data health check skipped: {exc}")

                if not arche_df.empty:
                    sel = arche_df[arche_df["team"].isin(team_ids)]
                    if not sel.empty:
                        st.markdown(
                            f"<div style='font-size:20px; font-weight:700; margin-top:10px;'>Archetype guess: "
                            + ", ".join(f"{int(r.team)} → {r.archetype}" for r in sel.itertuples())
                            + "</div>",
                            unsafe_allow_html=True,
                        )

        with tab_match:
            st.subheader("Match-by-Match Table")
            table_team = st.selectbox("Team for table", teams, index=0, key="table_team")
            st.session_state["notes_team"] = int(table_team)
            raw_cols = con.execute("PRAGMA table_info('raw')").df()["name"].tolist()
            preferred_cols = [
                "team",
                "match_num",
                "start_position",
                "auto_preload_result",
                "auto_leave_alliance_zone",
                "auto_enter_neutral_zone",
                "auto_alliance_zone_pickups",
                "auto_score",
                "auto_win",
                "active_shoot_bucket",
                "active_shoot_est_mid",
                "active_shooting_est",
                "active_cycles",
                "active_miss_heavy",
                "active_defense_level",
                "ferried",
                "ferry_bucket",
                "ferry_est_mid",
                "inactive_ferry_est",
                "played_defense",
                "defense_effective",
                "inactive_defense_level",
                "inactive_prepped_to_shoot",
                "climb_level",
                "climb_failed",
                "climb_time_bucket",
                "endgame_kept_shooting",
                "endgame_score",
                "tele_score",
                "total_score",
                "notes",
            ]
            cols = [c for c in preferred_cols if c in raw_cols]

            table_where = "team = ? AND match_num IS NOT NULL"
            table_params = [int(table_team)]

            table_df = con.execute(
                f"""
                SELECT {", ".join(cols)}
                FROM raw
                WHERE {table_where}
                ORDER BY match_num
                """,
                table_params,
            ).df()
            st.dataframe(table_df, use_container_width=True)

            # Show notes for the same team below the table
            notes = load_notes()
            if notes:
                notes_df = pd.DataFrame(notes)
                notes_df = notes_df[notes_df["team"] == int(table_team)]
                if not notes_df.empty:
                    st.subheader("Notes for This Team")
                    st.dataframe(notes_df, use_container_width=True)

        with tab_pick:
            st.subheader("Picklist")
            if not relation_exists(con, "v_picklist_current"):
                st.info("No picklist available yet. Import scouting CSV and rebuild the DB in Admin.")
            else:
                pick_sql = """
                    SELECT
                      team,
                      matches_played,
                      ROUND(QualsScore, 2) AS QualsScore,
                      ROUND(PointsScore, 2) AS PointsScore,
                      ROUND(RPScore, 2) AS RPScore,
                      carried_risk,
                      ROUND(avg_auto_score, 2) AS avg_auto_score,
                      ROUND(avg_active_cycles, 2) AS avg_active_cycles,
                      ROUND(avg_active_shooting_est, 2) AS avg_active_shooting_est,
                      ROUND(avg_inactive_ferry, 2) AS avg_inactive_ferry,
                      ROUND(avg_endgame_score, 2) AS avg_endgame_score,
                      ROUND(prep_rate, 3) AS prep_rate
                    FROM v_picklist_current
                    ORDER BY QualsScore DESC, matches_played DESC, team
                    """
                try:
                    pick_2026 = con.execute(pick_sql).df()
                except duckdb.BinderException as exc:
                    if "contents of view were altered" in str(exc).lower():
                        con.execute((APP_DIR / "views.sql").read_text(encoding="utf-8"))
                        if relation_exists(con, "v_picklist_current"):
                            pick_2026 = con.execute(pick_sql).df()
                        else:
                            pick_2026 = pd.DataFrame()
                    else:
                        raise
                st.caption("2026 ranking using shrinkage-stabilized RP/Points scoring.")
                if pick_2026.empty:
                    st.info("No picklist rows found.")
                else:
                    reference_df = pick_2026.copy()
                    reference_df["risk"] = reference_df["carried_risk"].apply(
                        lambda v: "⚠ carried" if int(v) == 1 else ""
                    )
                    st.dataframe(
                        reference_df[
                            [
                                "team",
                                "matches_played",
                                "QualsScore",
                                "PointsScore",
                                "RPScore",
                                "risk",
                                "avg_auto_score",
                                "avg_active_cycles",
                                "avg_active_shooting_est",
                                "avg_inactive_ferry",
                                "avg_endgame_score",
                                "prep_rate",
                            ]
                        ],
                        use_container_width=True,
                    )
                    with st.expander("Score breakdown", expanded=False):
                        st.markdown(
                            "- `PointsScore`: scoring proxy (active cycles + active shooting + endgame + inactive ferry)\n"
                            "- `RPScore`: RP proxy (Points inputs + prep rate)\n"
                            "- `QualsScore = PointsScore + 0.30 * RPScore`"
                        )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader("Custom Picklist")
                        st.caption("Select teams from the bank, then reorder manually.")
                        if "custom_picklist" not in st.session_state:
                            st.session_state["custom_picklist"] = []

                        all_teams = [int(t) for t in reference_df["team"].tolist()]
                        available = [t for t in all_teams if t not in st.session_state["custom_picklist"]]

                        col_add, col_add_btn = st.columns([3, 1])
                        with col_add:
                            if available:
                                team_to_add = st.selectbox("Team bank", available, key="team_bank")
                            else:
                                team_to_add = None
                                st.caption("All teams already added.")
                        with col_add_btn:
                            if st.button("Add", disabled=(team_to_add is None)):
                                if team_to_add not in st.session_state["custom_picklist"]:
                                    st.session_state["custom_picklist"].append(int(team_to_add))

                        if st.session_state["custom_picklist"]:
                            if "custom_selected_value" not in st.session_state:
                                st.session_state["custom_selected_value"] = st.session_state["custom_picklist"][0]
                            if st.session_state["custom_selected_value"] not in st.session_state["custom_picklist"]:
                                st.session_state["custom_selected_value"] = st.session_state["custom_picklist"][0]
                            selected_idx = st.session_state["custom_picklist"].index(
                                st.session_state["custom_selected_value"]
                            )
                            selected = st.selectbox(
                                "Selected team",
                                st.session_state["custom_picklist"],
                                index=selected_idx,
                                key="custom_selected_widget",
                            )
                            st.session_state["custom_selected_value"] = selected
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                if st.button("Move Up"):
                                    idx = st.session_state["custom_picklist"].index(selected)
                                    if idx > 0:
                                        lst = st.session_state["custom_picklist"]
                                        lst[idx - 1], lst[idx] = lst[idx], lst[idx - 1]
                                        st.session_state["custom_selected_value"] = selected
                            with col_m2:
                                if st.button("Move Down"):
                                    idx = st.session_state["custom_picklist"].index(selected)
                                    if idx < len(st.session_state["custom_picklist"]) - 1:
                                        lst = st.session_state["custom_picklist"]
                                        lst[idx + 1], lst[idx] = lst[idx], lst[idx + 1]
                                        st.session_state["custom_selected_value"] = selected
                            with col_m3:
                                if st.button("Remove"):
                                    st.session_state["custom_picklist"] = [
                                        t for t in st.session_state["custom_picklist"] if t != selected
                                    ]
                        else:
                            st.info("Custom list is empty. Add teams from the bank.")

                        weighted = reference_df.set_index("team")
                        custom_rows = []
                        for idx, t in enumerate(st.session_state["custom_picklist"], start=1):
                            if t in weighted.index:
                                row = weighted.loc[t].to_dict()
                                row["custom_rank"] = idx
                                row["team"] = t
                                custom_rows.append(row)
                        if custom_rows:
                            custom_df = pd.DataFrame(custom_rows)
                            st.dataframe(
                                custom_df[
                                    [
                                        "custom_rank",
                                        "team",
                                        "QualsScore",
                                        "PointsScore",
                                        "RPScore",
                                        "avg_auto_score",
                                        "avg_active_cycles",
                                        "avg_active_shooting_est",
                                        "avg_inactive_ferry",
                                        "avg_endgame_score",
                                    ]
                                ],
                                use_container_width=True,
                            )

                    with col_b:
                        st.subheader("Reference (Auto Ranked)")
                        st.dataframe(reference_df, use_container_width=True)

        if tab_quick is not None:
            with tab_quick:
                st.subheader("Quick Search")
                st.caption(
                    "Try: 'active cycles > 6 and climbed', "
                    "'scored in auto and prep rate > 0.4', "
                    "'inactive ferry >= 1'"
                )
                query = st.text_input("Search query", key="quick_query")
                run_search = st.button("Run Search")
                if run_search or (top_query and top_query.strip() and not query.strip()):
                    match_table = detect_match_table(con)
                    if match_table == "":
                        st.warning("No match-level table detected.")
                    else:
                        # Prefer tab query when provided; otherwise fall back to top bar query.
                        use_query = query.strip() if query.strip() else top_query.strip()
                        conditions = parse_quick_query(use_query)
                        if not conditions:
                            st.info(
                                "No conditions detected. Try examples like: "
                                "'active cycles > 6', 'climbed', 'scored in auto', 'prep rate > 0.3'."
                            )
                        else:
                            where = " and ".join([f"{c[0]} {c[1]} {c[2]}" for c in conditions])
                            st.caption("Interpreted conditions:")
                            st.code("\n".join([f"- {c[0]} {c[1]} {c[2]}" for c in conditions]))
                            mt_cols = set(
                                c.lower()
                                for c in con.execute(f"PRAGMA table_info('{match_table}')").df()["name"].tolist()
                            )
                            if "active_shooting_est" in mt_cols:
                                shoot_avg_expr = "COALESCE(t.active_shooting_est, 0)"
                            elif "active_shoot_est_mid" in mt_cols:
                                shoot_avg_expr = "COALESCE(t.active_shoot_est_mid, 0)"
                            else:
                                shoot_avg_expr = "0"
                            if "inactive_ferry_est" in mt_cols:
                                ferry_avg_expr = "COALESCE(t.inactive_ferry_est, 0)"
                            elif "ferry_est_mid" in mt_cols:
                                ferry_avg_expr = "COALESCE(t.ferry_est_mid, 0)"
                            else:
                                ferry_avg_expr = "0"
                            stats = con.execute(
                                f"""
                                SELECT
                                  t.team AS team,
                                  AVG(t.auto_score) AS avg_auto_score,
                                  AVG({shoot_avg_expr}) AS avg_active_shooting_est,
                                  AVG(t.active_cycles) AS avg_active_cycles,
                                  AVG({ferry_avg_expr}) AS avg_inactive_ferry,
                                  AVG(t.inactive_prepped_to_shoot) AS prep_rate,
                                  AVG(t.endgame_score) AS avg_endgame_score,
                                  AVG(t.auto_alliance_zone_pickups) AS avg_auto_pickups,
                                  AVG(t.auto_leave_alliance_zone) AS leave_rate,
                                  AVG(t.auto_enter_neutral_zone) AS neutral_enter_rate,
                                  AVG(t.auto_win) AS auto_win_rate,
                                  AVG(t.auto_score + {shoot_avg_expr} + t.endgame_score) AS combined_impact,
                                  MAX(s.PointsScore) AS points_score,
                                  MAX(s.RPScore) AS rp_score,
                                  MAX(s.QualsScore) AS quals_score
                                FROM {match_table} t
                                LEFT JOIN v_scores_2026 s USING (team)
                                WHERE t.team IS NOT NULL
                                GROUP BY t.team
                                """
                            ).df()
                            if stats.empty:
                                st.warning("No data found.")
                            else:
                                try:
                                    result = stats.query(where)
                                except Exception as exc:
                                    st.error(f"Quick Search parse error: {exc}")
                                    result = pd.DataFrame()
                                st.caption(f"Applied conditions: {where}")
                                if result.empty:
                                    st.info("No teams matched those conditions.")
                                else:
                                    st.dataframe(result, use_container_width=True)
                    # Manual rank editing removed in favor of custom list

        with tab_team:
            st.subheader("Team Lookup")
            st.caption("Lookup teams at the current event key.")
            api_key = st.session_state.get("tba_key", "")
            event_key = st.session_state.get("tba_event", "")
            if not api_key:
                st.info("Set TBA API Key in the Admin sidebar.")
            if not event_key:
                st.info("Set TBA Event Key in the Admin sidebar.")

            teams_at_event = get_event_teams(api_key, event_key) if api_key and event_key else []
            if teams_at_event:
                team_num = st.selectbox("Team (from event)", teams_at_event, key="lookup_team")
            else:
                team_num = st.number_input("Team number", min_value=0, value=0, step=1, key="lookup_team")
                if not event_key:
                    st.info("No event key set. Team lookup will pull directly from TBA.")

            # Pit scouting data (only when event key is set)
            if event_key:
                st.subheader("Pit Scouting")
                pit_team_col = get_pit_team_col(con)
                if pit_team_col == "":
                    st.info("No pit table found in this DB. Rebuild with --pit to enable.")
                else:
                    try:
                        pit_df = con.execute(
                            f"SELECT * FROM pit WHERE {pit_team_col} = ?",
                            [int(team_num)],
                        ).df()
                    except Exception as exc:
                        st.error(f"Failed to load pit data: {exc}")
                        pit_df = pd.DataFrame()
                    if pit_df.empty:
                        st.info("No pit data for this team.")
                    else:
                        # Display fields vertically with labels
                        mapping = pit_field_map()
                        row = pit_df.iloc[0].to_dict()
                        fields = []
                        for key, label in mapping.items():
                            if key in {"pit_col_1", "pit_col_2", "pit_col_3", "pit_col_4"}:
                                continue
                            if key in row:
                                fields.append({"Field": label, "Value": row[key]})
                        pit_t = pd.DataFrame(fields)
                        st.dataframe(pit_t, use_container_width=True)

                        # Show photos if present
                        photo_keys = ["pit_col_1", "pit_col_2", "pit_col_3", "pit_col_4", "pit_col_12"]
                        photos = [row.get(k) for k in photo_keys if row.get(k)]
                        # Keep only URLs or existing local files
                        safe_photos = []
                        for p in photos:
                            p = str(p)
                            if p.startswith("http"):
                                safe_photos.append(p)
                            elif os.path.exists(p):
                                safe_photos.append(p)
                        if safe_photos:
                            st.subheader("Pit Photos")
                            st.image(safe_photos, width=220)

                        # Sanity check: auto routine vs actual auto scoring
                        routine = str(row.get("pit_col_7", "")).lower()
                        if "coral" in routine:
                            auto_actual = con.execute(
                                """
                                SELECT AVG(auto_l1 + auto_l2 + auto_l3 + auto_l4)
                                FROM raw
                                WHERE team = ?
                                """,
                                [int(team_num)],
                            ).fetchone()[0]
                            auto_actual = auto_actual or 0.0
                            if auto_actual > 0:
                                st.success("Sanity check: Auto coral claimed and data shows auto coral scored.")
                            else:
                                st.warning("Sanity check: Auto coral claimed but no auto coral scored in data.")

                saved_paths = get_auto_paths_for_team(event_key, int(team_num))
                if saved_paths:
                    st.subheader("Auto Path Drawings")
                    for entry in saved_paths:
                        st.markdown(f"**Match {entry.get('match')}**")
                        img_path = entry.get("image", "")
                        if os.path.exists(img_path):
                            st.image(img_path, width=220)
                        else:
                            st.caption(f"Missing image file: {img_path}")

            if st.button("Lookup Team"):
                if not api_key:
                    st.error("Missing TBA API Key.")
                elif team_num == 0:
                    st.error("Enter a team number.")
                else:
                    info = get_team_info(api_key, int(team_num))
                    if info:
                        st.session_state["tba_team_key"] = info.get("key", f"frc{int(team_num)}")
                        st.success(f"TBA team key: {st.session_state['tba_team_key']}")
                        year = year_from_event_key(event_key)
                        if year:
                            url = f"https://www.thebluealliance.com/team/{team_num}/{year}"
                            components.iframe(url, height=700, scrolling=True)
                        else:
                            st.info("Could not determine year from event key.")
                    else:
                        st.warning("Team not found or TBA error.")

        with tab_prematch:
            st.subheader("Pre-Match")
            event_key = st.session_state.get("tba_event", "").strip()
            st.caption(f"Using event key from Admin: `{event_key}`" if event_key else "Set Event Key in Admin sidebar.")
            if not event_key:
                st.info("Set Event Key in Admin sidebar (required for schedule use).")

            if event_key:
                match_df = con.execute(
                    """
                    SELECT DISTINCT match_number
                    FROM schedule
                    WHERE event_key = ? AND comp_level = 'qm'
                    ORDER BY match_number
                    """,
                    [event_key],
                ).df()
            else:
                match_df = pd.DataFrame()
            if match_df.empty:
                st.info("No schedule loaded for this event. Import schedule CSV first.")
            else:
                selected_match = st.selectbox(
                    "Match",
                    match_df["match_number"].astype(int).tolist(),
                    key="prematch_match_selector",
                )
                prematch_tabs = st.tabs(["Selection", "Coach Plan"])
                tab_selection, tab_coach = prematch_tabs

                sixpack = con.execute(
                    """
                    SELECT
                      event_key,
                      match_number,
                      alliance,
                      slot,
                      team,
                      COALESCE(QualsScore_adj, COALESCE(QualsScore, 50)) AS QualsScore_adj,
                      COALESCE(QualsScore, 50) AS QualsScore,
                      COALESCE(PointsScore, 50) AS PointsScore,
                      COALESCE(RPScore, 50) AS RPScore,
                      COALESCE(avg_active_cycles, 0) AS avg_active_cycles,
                      COALESCE(avg_active_shooting_est, 0) AS avg_active_shooting_est,
                      COALESCE(avg_endgame_score, 0) AS avg_endgame_score,
                      COALESCE(prep_rate, 0) AS prep_rate,
                      COALESCE(avg_inactive_ferry, 0) AS avg_inactive_ferry,
                      COALESCE(matches_played, 0) AS matches_played,
                      role_tag,
                      COALESCE(TeamConfidenceTier, 'Houston, We''ve Got a Problem') AS TeamConfidenceTier,
                      COALESCE(delta, 0.0) AS delta,
                      COALESCE(override_reason, '') AS override_reason
                    FROM v_prematch_sixpack_adjusted
                    WHERE event_key = ? AND match_number = ? AND comp_level = 'qm'
                    ORDER BY alliance, slot
                    """,
                    [event_key, int(selected_match)],
                ).df()
                mission = con.execute(
                    """
                    SELECT
                      MissionConfidenceTier,
                      confidence_score,
                      low_data_count,
                      volatile_count,
                      loose_fit_count,
                      override_active,
                      miss_risk_count,
                      endgame_risk_count
                    FROM v_match_confidence
                    WHERE event_key = ? AND match_number = ?
                    """,
                    [event_key, int(selected_match)],
                ).df()

                if sixpack.empty:
                    st.warning("No six-pack data for this match.")
                else:
                    with tab_selection:
                        if not mission.empty:
                            m = mission.iloc[0]
                            reasons = []
                            if int(m.get("low_data_count", 0)) > 0:
                                reasons.append(f"{int(m['low_data_count'])} low-data")
                            if int(m.get("volatile_count", 0)) > 0:
                                reasons.append(f"{int(m['volatile_count'])} volatile")
                            if int(m.get("override_active", 0)) > 0:
                                reasons.append("override active")
                            if int(m.get("miss_risk_count", 0)) > 0:
                                reasons.append(f"{int(m['miss_risk_count'])} miss-risk")
                            if int(m.get("endgame_risk_count", 0)) > 0:
                                reasons.append(f"{int(m['endgame_risk_count'])} endgame-risk")
                            reason_txt = " • ".join(reasons) if reasons else "no major flags"
                            st.markdown(
                                f"### Mission Confidence: **{m['MissionConfidenceTier']}** "
                                f"(score: {float(m['confidence_score']):.1f})"
                            )
                            st.caption(reason_txt)

                        red_df = sixpack[sixpack["alliance"] == "red"].sort_values("slot")
                        blue_df = sixpack[sixpack["alliance"] == "blue"].sort_values("slot")
                        c_red, c_blue = st.columns(2)
                        with c_red:
                            st.markdown("### Red Alliance")
                            for r in red_df.itertuples():
                                if pd.isna(r.team):
                                    continue
                                low_data = " ⚠ low data (n<3)" if int(r.matches_played) < 3 else ""
                                override_line = (
                                    f"  \nOverride: `{float(r.delta):.1f}` ({r.override_reason})"
                                    if float(r.delta) != 0
                                    else ""
                                )
                                st.markdown(
                                    f"""
**R{int(r.slot)} · Team {int(r.team)}**  
**QualsScore: {float(r.QualsScore_adj):.1f}**  
PointsScore: {float(r.PointsScore):.1f} | RPScore: {float(r.RPScore):.1f}  
Active Cycles: {float(r.avg_active_cycles):.2f} | Endgame: {float(r.avg_endgame_score):.2f}  
n: {int(r.matches_played)}{low_data}  
Confidence: `{r.TeamConfidenceTier}`  
Role: `{r.role_tag}`{override_line}
"""
                                )
                                st.divider()
                        with c_blue:
                            st.markdown("### Blue Alliance")
                            for r in blue_df.itertuples():
                                if pd.isna(r.team):
                                    continue
                                low_data = " ⚠ low data (n<3)" if int(r.matches_played) < 3 else ""
                                override_line = (
                                    f"  \nOverride: `{float(r.delta):.1f}` ({r.override_reason})"
                                    if float(r.delta) != 0
                                    else ""
                                )
                                st.markdown(
                                    f"""
**B{int(r.slot)} · Team {int(r.team)}**  
**QualsScore: {float(r.QualsScore_adj):.1f}**  
PointsScore: {float(r.PointsScore):.1f} | RPScore: {float(r.RPScore):.1f}  
Active Cycles: {float(r.avg_active_cycles):.2f} | Endgame: {float(r.avg_endgame_score):.2f}  
n: {int(r.matches_played)}{low_data}  
Confidence: `{r.TeamConfidenceTier}`  
Role: `{r.role_tag}`{override_line}
"""
                                )
                                st.divider()

                        team_options = sorted(
                            set(int(t) for t in sixpack["team"].tolist() if pd.notna(t))
                        )
                        compare_teams = st.multiselect(
                            "Quick compare (2-3 teams)",
                            team_options,
                            default=team_options[:2] if len(team_options) >= 2 else team_options,
                        )
                        if 2 <= len(compare_teams) <= 3:
                            cmp_df = sixpack[sixpack["team"].isin(compare_teams)][
                                [
                                    "team",
                                    "QualsScore_adj",
                                    "PointsScore",
                                    "RPScore",
                                    "avg_active_cycles",
                                    "avg_active_shooting_est",
                                    "avg_endgame_score",
                                    "prep_rate",
                                    "avg_inactive_ferry",
                                    "matches_played",
                                    "TeamConfidenceTier",
                                    "role_tag",
                                ]
                            ].sort_values("QualsScore_adj", ascending=False)
                            st.dataframe(cmp_df, use_container_width=True)

                    with tab_coach:
                        summary = sixpack.groupby("alliance", as_index=False).agg(
                            sum_quals_adj=("QualsScore_adj", "sum"),
                            avg_quals_adj=("QualsScore_adj", "mean"),
                            sum_points=("PointsScore", "sum"),
                            sum_rp=("RPScore", "sum"),
                            avg_endgame=("avg_endgame_score", "mean"),
                            avg_cycles=("avg_active_cycles", "mean"),
                            avg_support_ferry=("avg_inactive_ferry", "mean"),
                            avg_support_prep=("prep_rate", "mean"),
                        )
                        if not summary.empty:
                            summary["support_total"] = summary["avg_support_ferry"] + summary["avg_support_prep"]
                            st.dataframe(summary, use_container_width=True)

                            if len(summary) == 2:
                                s = summary.set_index("alliance")
                                red = s.loc["red"] if "red" in s.index else None
                                blue = s.loc["blue"] if "blue" in s.index else None
                                bullets = []
                                if red is not None and blue is not None:
                                    if float(red["avg_endgame"]) - float(blue["avg_endgame"]) > 3:
                                        bullets.append("Red has stronger endgame avg. Consider endgame race plan.")
                                    elif float(blue["avg_endgame"]) - float(red["avg_endgame"]) > 3:
                                        bullets.append("Blue has stronger endgame avg. Consider disruption/deny plan.")
                                    if float(red["avg_support_prep"] + red["avg_support_ferry"]) < 1.2:
                                        bullets.append("Red support flow is weak; simplify cycles and prioritize handoffs.")
                                    if float(blue["avg_support_prep"] + blue["avg_support_ferry"]) < 1.2:
                                        bullets.append("Blue support flow is weak; watch for stalled offense.")
                                    if float(red["avg_cycles"]) > float(blue["avg_cycles"]) + 1.2:
                                        bullets.append("Red cycle volume edge. Push pace.")
                                    elif float(blue["avg_cycles"]) > float(red["avg_cycles"]) + 1.2:
                                        bullets.append("Blue cycle volume edge. Protect your high-value cycle bot.")
                                if bullets:
                                    st.markdown("#### Win Conditions")
                                    for b in bullets[:4]:
                                        st.markdown(f"- {b}")

                        warn_msgs = []
                        for al in ["red", "blue"]:
                            d = sixpack[sixpack["alliance"] == al]
                            if d.empty:
                                continue
                            if float(d["avg_endgame_score"].sum()) < 15:
                                warn_msgs.append(f"{al.title()}: no reliable endgame.")
                            if float((d["avg_inactive_ferry"] + d["prep_rate"]).sum()) < 2.0:
                                warn_msgs.append(f"{al.title()}: low support (ferry+prep).")
                            scorers = int((d["role_tag"] == "Active Scorer").sum())
                            if scorers >= 3:
                                warn_msgs.append(f"{al.title()}: role imbalance (too many pure scorers).")
                        if warn_msgs:
                            for m in warn_msgs[:3]:
                                st.warning(m)
                        else:
                            st.success("No major synergy warnings.")

                        assignments_json = st.text_area(
                            "Assignments JSON",
                            value='{"red1":"scorer","red2":"support","red3":"flex","blue1":"scorer","blue2":"support","blue3":"flex"}',
                            key="prematch_assignments_json",
                        )
                        plan_text = st.text_area("Coach plan notes", key="prematch_plan_text")
                        if st.button("Save Plan"):
                            con.execute(
                                """
                                INSERT INTO prematch_plan
                                (event_key, match_number, alliance, plan_text, assignments_json, created_at, updated_at)
                                VALUES (?, ?, 'both', ?, ?, now(), now())
                                """,
                                [event_key, int(selected_match), plan_text, assignments_json],
                            )
                            st.success("Plan saved.")

                        o_col1, o_col2, o_col3 = st.columns([2, 1, 2])
                        with o_col1:
                            ov_team = st.selectbox(
                                "Override team",
                                sorted(set(int(t) for t in sixpack["team"].tolist() if pd.notna(t))),
                                key="prematch_override_team",
                            )
                        with o_col2:
                            ov_delta = st.slider("Delta", min_value=-20, max_value=20, value=0, step=1)
                        with o_col3:
                            ov_reason = st.selectbox(
                                "Reason",
                                ["Shooter dead", "Drive issues", "New driver", "Auto conflict", "Defense surprise", "Other"],
                                key="prematch_override_reason",
                            )
                        ov_other = ""
                        if ov_reason == "Other":
                            ov_other = st.text_input("Other reason")
                        if st.button("Save Override"):
                            reason_text = ov_other.strip() if ov_reason == "Other" else ov_reason
                            con.execute(
                                """
                                INSERT INTO coach_overrides
                                (event_key, match_number, team, scope, delta, reason, created_at)
                                VALUES (?, ?, ?, 'total', ?, ?, now())
                                """,
                                [event_key, int(selected_match), int(ov_team), float(ov_delta), reason_text],
                            )
                            st.success("Override saved.")
                            st.rerun()

                        saved = con.execute(
                            """
                            SELECT plan_text, assignments_json, created_at
                            FROM prematch_plan
                            WHERE event_key = ? AND match_number = ?
                            ORDER BY updated_at DESC, created_at DESC
                            LIMIT 1
                            """,
                            [event_key, int(selected_match)],
                        ).df()
                        if not saved.empty:
                            row = saved.iloc[0]
                            st.markdown("#### Latest Saved Plan")
                            st.markdown(f"Saved: `{row['created_at']}`")
                            st.markdown(f"**Assignments:** `{row['assignments_json']}`")
                            st.markdown(f"**Plan:** {row['plan_text']}")

        with tab_autopath:
            st.subheader("Auto Path")
            st.caption("Super Scout: draw autonomous paths per team/match. Saved paths are stored with team event data.")
            api_key = st.session_state.get("tba_key", "")
            event_key = st.session_state.get("tba_event", "")
            if st_canvas is None:
                st.warning("Canvas package missing. Install dependencies (`pip install -r requirements.txt`).")
            if st_canvas is not None:
                event_for_paths = event_key if event_key else "unknown_event"
                auto_team = 0
                auto_match = 0
                if api_key and event_key:
                    matches = get_event_matches(api_key, event_key)
                    qm_matches = [m for m in matches if m.get("comp_level") == "qm"]
                    match_nums = sorted(
                        list(dict.fromkeys([m.get("match_number") for m in qm_matches if m.get("match_number") is not None]))
                    )
                    if match_nums:
                        auto_match = st.selectbox("Match", match_nums, key="autopath_match")
                        selected = next((m for m in qm_matches if m.get("match_number") == int(auto_match)), None)
                        teams_in_match = []
                        if selected:
                            alliances = selected.get("alliances", {})
                            red = alliances.get("red", {}).get("team_keys", [])
                            blue = alliances.get("blue", {}).get("team_keys", [])
                            teams_in_match = [
                                int(t.replace("frc", ""))
                                for t in (red + blue)
                                if str(t).startswith("frc")
                            ]
                            teams_in_match = list(dict.fromkeys(teams_in_match))
                        if teams_in_match:
                            auto_team = st.selectbox("Team number", teams_in_match, key="autopath_team")
                        else:
                            st.warning("No teams found for this TBA match.")
                            auto_team = st.number_input(
                                "Team number", min_value=0, value=0, step=1, key="autopath_team"
                            )
                    else:
                        st.warning("No qualification matches returned from TBA.")
                        auto_team = st.number_input("Team number", min_value=0, value=0, step=1, key="autopath_team")
                        auto_match = st.number_input("Match", min_value=0, value=0, step=1, key="autopath_match")
                else:
                    st.info("Set TBA API key and Event key in Admin to auto-populate match -> team.")
                    auto_team = st.number_input("Team number", min_value=0, value=0, step=1, key="autopath_team")
                    auto_match = st.number_input("Match", min_value=0, value=0, step=1, key="autopath_match")
                routine_options = [
                    "Primary Auto",
                    "Alternate Auto",
                    "Safe Mobility",
                    "Defense Avoidance",
                    "Experimental",
                ]
                routine_name = st.selectbox("Routine", routine_options, key="autopath_routine")
                routine_color_map = {
                    "Primary Auto": "#00ff66",
                    "Alternate Auto": "#00a3ff",
                    "Safe Mobility": "#ffd400",
                    "Defense Avoidance": "#ff5a5a",
                    "Experimental": "#c77dff",
                }
                default_color = routine_color_map.get(routine_name, "#00ff66")
                stroke_color = st.color_picker("Path Color", value=default_color, key="autopath_color")

                bg_image = None
                if DEFAULT_FIELD_IMAGE.exists():
                    try:
                        b = load_image_bytes_cached(
                            str(DEFAULT_FIELD_IMAGE), DEFAULT_FIELD_IMAGE.stat().st_mtime
                        )
                        # Force a stable canvas background format/size to avoid black render issues.
                        bg_image = Image.open(BytesIO(b)).convert("RGB").resize((1000, 450))
                        st.caption(f"Using default field image: {DEFAULT_FIELD_IMAGE} ({bg_image.width}x{bg_image.height})")
                    except Exception as exc:
                        # Fallback to direct disk open if cache/bytes load fails.
                        try:
                            bg_image = Image.open(DEFAULT_FIELD_IMAGE).convert("RGB").resize((1000, 450))
                            st.caption(f"Using fallback field image load: {DEFAULT_FIELD_IMAGE}")
                        except Exception:
                            bg_image = None
                            st.warning(f"Failed loading default field image: {exc}")
                else:
                    st.caption(
                        f"Default field image not found at `{DEFAULT_FIELD_IMAGE}`. "
                        "Upload one below or add the file at that path."
                    )

                if bg_image is None:
                    field_img = st.file_uploader(
                        "Field map image (optional, png/jpg)",
                        type=["png", "jpg", "jpeg"],
                        key="autopath_field_image",
                    )
                    if field_img is not None:
                        uploaded_bytes = field_img.getvalue()
                        bg_image = Image.open(BytesIO(uploaded_bytes)).convert("RGB").resize((1000, 450))
                        if st.button("Set uploaded image as default"):
                            DEFAULT_FIELD_IMAGE.parent.mkdir(parents=True, exist_ok=True)
                            DEFAULT_FIELD_IMAGE.write_bytes(uploaded_bytes)
                            load_image_bytes_cached.clear()
                            st.success(f"Saved default field image to {DEFAULT_FIELD_IMAGE}")
                if bg_image is not None:
                    with st.expander("Show field map preview", expanded=False):
                        st.image(bg_image, width=420)
                else:
                    st.caption("No field map loaded. Canvas is using fallback background.")

                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 0, 0.3)",
                    stroke_width=4,
                    stroke_color=stroke_color,
                    background_color="#202020",
                    background_image=bg_image,
                    update_streamlit=True,
                    height=450,
                    width=1000,
                    drawing_mode="freedraw",
                    key=f"autopath_canvas_{int(auto_team)}_{int(auto_match)}",
                )

                if st.button("Save Auto Path"):
                    if canvas_result.image_data is None:
                        st.warning("No drawing to save.")
                    elif int(auto_team) == 0 or int(auto_match) == 0:
                        st.warning("Select valid team and match first.")
                    else:
                        saved_path = save_auto_path_image(
                            canvas_result.image_data, event_for_paths, int(auto_team), int(auto_match)
                        )
                        idx = load_auto_path_index()
                        idx_key = f"{event_for_paths}|{int(auto_team)}"
                        entries = idx.get(idx_key, [])
                        # Replace existing match entry if present
                        entries = [e for e in entries if int(e.get("match", -1)) != int(auto_match)]
                        entries.append(
                            {
                                "match": int(auto_match),
                                "image": saved_path,
                                "routine": routine_name,
                                "color": stroke_color,
                            }
                        )
                        entries.sort(key=lambda x: x.get("match", 0))
                        idx[idx_key] = entries
                        save_auto_path_index(idx)
                        st.success(f"Saved auto path for team {int(auto_team)} match {int(auto_match)}.")

                saved = get_auto_paths_for_team(event_for_paths, int(auto_team)) if int(auto_team) else []
                if saved:
                    st.subheader("Saved Paths")
                    for entry in saved:
                        routine_txt = entry.get("routine", "Unlabeled")
                        color_txt = entry.get("color", "#ffffff")
                        st.markdown(
                            f"**Match {entry.get('match')}** | Routine: `{routine_txt}` | "
                            f"Color: `{color_txt}`"
                        )
                        if os.path.exists(entry.get("image", "")):
                            st.image(entry["image"], width=320)
                        else:
                            st.caption(f"Missing image file: {entry.get('image')}")

        with tab_match:
            st.subheader("Match Notes")
            notes_team = st.selectbox("Team for notes", teams, index=0, key="notes_team")
            notes_event_key = st.session_state.get("tba_event", "").strip()
            notes_matches = load_team_matches(con, int(notes_team))
            if notes_matches:
                notes_match = st.selectbox(
                    "Match number", [int(m) for m in notes_matches], key="notes_match"
                )
            else:
                notes_match = 0
                st.info("No matches found for this team in the DB.")
            archetype_options = [
                "Volume Shooter",
                "Accuracy Shooter",
                "Under-Performing Shooter",
                "Passer",
                "Defender",
                "Endgame Player",
            ]
            archetype_list = st.multiselect(
                "Archetype(s)",
                archetype_options,
                key="notes_archetype",
            )
            note_text = st.text_area("Note", height=100, key="notes_text")

            if st.button("Save Note"):
                notes = load_notes()
                notes.append(
                    {
                        "team": int(notes_team),
                        "match_num": int(notes_match),
                        "archetype": ", ".join(archetype_list),
                        "note": note_text.strip(),
                    }
                )
                save_notes(notes)
                st.success("Note saved.")

            # Pit scouting notes/photos in Match Notes view
            pit_team_col = get_pit_team_col(con)
            if pit_team_col:
                try:
                    pit_df = con.execute(
                        f"SELECT * FROM pit WHERE {pit_team_col} = ?",
                        [int(notes_team)],
                    ).df()
                except Exception:
                    pit_df = pd.DataFrame()
                if not pit_df.empty:
                    row = pit_df.iloc[0].to_dict()
                    st.subheader("Pit Scouting Notes")
                    st.write(row.get("pit_col_11", ""))
                    photos = [row.get(k) for k in ["pit_col_1", "pit_col_2", "pit_col_3", "pit_col_4", "pit_col_12"] if row.get(k)]
                    safe_photos = []
                    for p in photos:
                        p = str(p)
                        if p.startswith("http"):
                            safe_photos.append(p)
                        elif os.path.exists(p):
                            safe_photos.append(p)
                    if safe_photos:
                        st.subheader("Pit Photos")
                        st.image(safe_photos, width=220)

                    # Show saved auto path drawings alongside pit scouting notes.
                    path_event_keys = [notes_event_key] if notes_event_key else []
                    if "unknown_event" not in path_event_keys:
                        path_event_keys.append("unknown_event")
                    saved_paths = []
                    for ek in path_event_keys:
                        for entry in get_auto_paths_for_team(ek, int(notes_team)):
                            merged = dict(entry)
                            merged["event_key"] = ek
                            saved_paths.append(merged)
                    if saved_paths:
                        st.subheader("Saved Auto Paths")
                        # Keep stable order and avoid duplicate rows.
                        seen = set()
                        for entry in sorted(saved_paths, key=lambda x: int(x.get("match", 0))):
                            image_path = str(entry.get("image", ""))
                            dedupe_key = (entry.get("event_key"), int(entry.get("match", 0)), image_path)
                            if dedupe_key in seen:
                                continue
                            seen.add(dedupe_key)
                            routine_txt = entry.get("routine", "Unlabeled")
                            color_txt = entry.get("color", "#ffffff")
                            st.markdown(
                                f"**Event `{entry.get('event_key', '')}` | Match {entry.get('match')}**  \n"
                                f"Routine: `{routine_txt}` | Color: `{color_txt}`"
                            )
                            if os.path.exists(image_path):
                                st.image(image_path, width=240)
                            else:
                                st.caption(f"Missing image file: {image_path}")

            notes = load_notes()
            if notes:
                notes_df = pd.DataFrame(notes)
                notes_df = notes_df[notes_df["team"] == int(notes_team)]
                if not notes_df.empty:
                    st.dataframe(notes_df, use_container_width=True)
                else:
                    st.info("No notes for this team yet.")
        with tab_watch:
            st.subheader("Video Review")
            watch_team = st.selectbox("Team for video", teams, index=0, key="watch_team")
            watch_matches = load_team_matches(con, int(watch_team))
            if watch_matches:
                watch_match = st.selectbox(
                    "Match number", [int(m) for m in watch_matches], key="watch_match"
                )
            else:
                watch_match = 0
                st.info("No matches found for this team in the DB.")

            api_key = st.session_state.get("tba_key", "")
            event_key = st.session_state.get("tba_event", "")
            if not api_key or not event_key:
                st.info("Set TBA API Key and Event Key in the Admin tab.")
            elif watch_match == 0:
                st.info("Select a match number to load video.")
            else:
                if st.button("Load Match Video"):
                    videos = get_match_videos(api_key, event_key, int(watch_match))
                    if not videos:
                        st.warning("No video found for that match.")
                    else:
                        yt = next((v for v in videos if v.get("type") == "youtube"), None)
                        if yt and yt.get("key"):
                            st.video(f"https://www.youtube.com/watch?v={yt['key']}")
                        else:
                            st.json(videos)

        with tab_defs:
            st.subheader("Definitions")
            with st.expander("Raw data scouts enter", expanded=True):
                st.markdown(
                    """
- **Auto:** start position, auto path (Super Scout), auto score (+ optional flags)
- **Tele (Active):** shooting est (0–5), cycles (int), miss-heavy (Y/N optional)
- **Tele (Inactive):** ferry est (0–3), defense (0–2), prepped-to-shoot (Y/N)
- **Endgame:** climb level (0–N), climb failed (Y/N optional), time bucket (optional)
"""
                )

            with st.expander("What Venom calculates", expanded=False):
                st.markdown(
                    """
- **Team averages:** auto, active cycles, active shooting, ferry, endgame
- **Rates:** prep rate, climb success rate
- **n:** matches played (confidence)
"""
                )

            with st.expander("Scores", expanded=True):
                st.markdown(
                    """
- **RPScore:** flow/ranking proxy (cycles + shooting + endgame + ferry + prep)
- **PointsScore:** strength proxy (cycles + shooting + endgame [+ small ferry])
- **QualsScore:** main sort = **PointsScore + 0.30 × RPScore**
"""
                )

            with st.expander("Notes", expanded=False):
                st.markdown(
                    """
- **Percentiles (0–1):** used to combine different metrics fairly
- **Shrinkage:** low n pulls scores toward ~50 (prevents early-match noise)
- **Interpretation:** high points = strong scorer; high RP = smart support/flow
"""
                )

        with st.sidebar:
            st.subheader("Admin")
            st.markdown("Configure data source and TBA access.")
            st.text_input("DuckDB file", db_path, disabled=True)
            st.subheader("Unified Import")
            imports = st.file_uploader(
                "Upload CSV or ZIP",
                type=["csv", "zip"],
                key="unified_import_upload",
                accept_multiple_files=True,
                help="Auto-detects MatchRobot, Schedule, Pit, and Super Scout files.",
            )
            if st.button("Run Unified Import", disabled=(not imports)):
                try:
                    event_key_for_file = st.session_state.get("tba_event", "").strip() or "2026TEST"
                    safe_event_key = "".join(ch for ch in event_key_for_file if ch.isalnum() or ch in ("_", "-")).strip("_-") or "event"
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    new_db_path = DATA_DIR / f"event_{safe_event_key}_{ts}.duckdb"
                    combined = {"matchrobot": 0, "superscout": 0, "pit": 0, "schedule": 0, "skipped": 0, "warnings": []}
                    with duckdb.connect(str(new_db_path)) as import_con:
                        ensure_prematch_schema(import_con)
                        ensure_ingest_tables(import_con)
                        for upl in imports or []:
                            s = ingest_upload_bytes(
                                import_con,
                                upl.getvalue(),
                                upl.name,
                                default_event_key=event_key_for_file,
                            )
                            for k in ["matchrobot", "superscout", "pit", "schedule", "skipped"]:
                                combined[k] += int(s.get(k, 0))
                            combined["warnings"].extend(s.get("warnings", []))
                        refresh_compat_views(import_con)
                        import_con.execute((APP_DIR / "views.sql").read_text(encoding="utf-8"))
                        gamepack_sql = APP_DIR / "gamepack.sql"
                        if gamepack_sql.exists():
                            import_con.execute(gamepack_sql.read_text(encoding="utf-8"))
                        if not relation_exists(import_con, "v_picklist_current"):
                            raise RuntimeError(
                                "Build failed: picklist view 'v_picklist_current' was not created. "
                                "Import succeeded, but SQL bootstrap did not complete."
                            )
                    _write_active_db_pointer(str(new_db_path))
                    st.session_state["db_path"] = str(new_db_path)
                    st.success(
                        "Ingested: "
                        f"MatchRobot ({combined['matchrobot']} rows), "
                        f"Schedule ({combined['schedule']} rows), "
                        f"Pit ({combined['pit']} rows), "
                        f"Super Scout ({combined['superscout']} rows)."
                    )
                    skipped_unknown = [w for w in combined["warnings"] if "unknown csv kind" in str(w).lower()]
                    for msg in skipped_unknown[:5]:
                        st.caption(f"Skipped: {msg}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import failed: {exc}")
            reset_confirm = st.checkbox("Confirm reset", key="confirm_reset_data")
            delete_files = st.checkbox("Delete imported DB files", key="delete_imported_dbs")
            if st.button("Reset Data", disabled=(not reset_confirm)):
                try:
                    if ACTIVE_DB_POINTER.exists():
                        ACTIVE_DB_POINTER.unlink()
                    if delete_files and DATA_DIR.exists():
                        for db_file in DATA_DIR.glob("event_*.duckdb"):
                            try:
                                db_file.unlink()
                            except Exception:
                                pass
                    st.session_state["db_path"] = str(EMPTY_DB_PATH)
                    st.success("Data reset. App returned to empty state.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Reset failed: {exc}")
            matches_played = st.number_input(
                "Matches played (optional reference)",
                min_value=0,
                value=int(matches_played),
                step=1,
            )
            st.session_state["matches_played"] = int(matches_played)
            st.divider()
            st.subheader("TBA API Check")
            api_key = st.text_input(
                "TBA API Key (uses TBA_AUTH_KEY env var if set)",
                type="password",
                value=os.environ.get("TBA_AUTH_KEY", ""),
            )
            event_key = st.text_input("Event Key (e.g., 2025miket)")
            st.session_state["tba_key"] = api_key.strip()
            st.session_state["tba_event"] = event_key.strip()
            if st.button("Validate Key"):
                ok, msg = check_tba_key(api_key.strip(), event_key.strip())
                if ok:
                    st.success("TBA key is valid and event exists.")
                else:
                    st.error(f"TBA check failed: {msg}")
            st.divider()
            if event_key.strip():
                sched_status = con.execute(
                    """
                    SELECT
                      COUNT(*) AS match_count,
                      MIN(match_number) AS first_match,
                      MAX(match_number) AS last_match,
                      MAX(updated_at) AS last_updated
                    FROM schedule
                    WHERE event_key = ?
                    """,
                    [event_key.strip()],
                ).df()
                if not sched_status.empty and int(sched_status.loc[0, "match_count"] or 0) > 0:
                    row = sched_status.iloc[0]
                    last_updated = row["last_updated"]
                    try:
                        last_updated_str = pd.to_datetime(last_updated).strftime("%Y-%m-%d %H:%M:%S UTC")
                    except Exception:
                        last_updated_str = str(last_updated)
                    st.caption(
                        "Schedule status: "
                        f"{int(row['match_count'])} matches "
                        f"(Q{int(row['first_match'])} to Q{int(row['last_match'])}) • "
                        f"Last import: {last_updated_str}"
                    )
                else:
                    st.caption("Schedule status: no schedule imported for this event key yet.")
            st.button("Sync from TBA (coming soon)", disabled=True, key="admin_tba_schedule_stub")
            st.divider()
            if st.button("Clear All Notes"):
                if NOTES_PATH.exists():
                    NOTES_PATH.write_text("[]", encoding="utf-8")
                st.success("All notes cleared.")


if __name__ == "__main__":
    main()
