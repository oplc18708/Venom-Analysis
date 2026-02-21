from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from file_detect import normalize_headers


BUCKET_VALUES = ["0-5", "6-12", "13-20", "21+"]
DEFENSE_EFFECTIVE_VALUES = ["Not Effective", "Somewhat", "Effective"]
AUTO_PRELOAD_VALUES = ["Made", "Missed", "None"]
AUTO_PICKUP_VALUES = ["0", "1", "2+"]
AUTO_PATH_TAG_VALUES = ["Stay Home", "Neutral Run", "Cross Heavy", "Other"]

BUCKET_TO_MID = {
    "0-5": 3,
    "6-12": 9,
    "13-20": 17,
    "21+": 24,
}

MATCHROBOT_REQUIRED = [
    "event_key",
    "match_number",
    "team",
    "active_cycles",
    "active_shoot_bucket",
    "ferried",
    "ferry_bucket",
    "inactive_prepped_to_shoot",
    "played_defense",
    "defense_effective",
    "climb_level",
]

MATCHROBOT_OPTIONAL = [
    "start_position",
    "auto_score",
    "auto_preload_result",
    "auto_leave_alliance_zone",
    "auto_enter_neutral_zone",
    "auto_alliance_zone_pickups",
    "auto_win",
    "climb_failed",
    "climb_time_bucket",
    "endgame_kept_shooting",
    "notes",
    "active_shooting_est",
    "inactive_ferry_est",
]

SUPERSCOUT_REQUIRED = [
    "event_key",
    "match_number",
    "alliance",
    "team",
    "auto_path_tag",
]

SUPERSCOUT_OPTIONAL = [
    "auto_path_image_ref",
    "auto_notes",
    "start_position_confirmed",
    "driver_skill",
    "defense_rating",
    "cycle_speed_rating",
    "want_on_alliance",
    "role_preference",
    "reliability_flag",
    "reliability_notes",
]


@dataclass
class ValidationResult:
    df: pd.DataFrame
    warnings: List[str]
    schema: str = "UNKNOWN"


def _normalize_bool_value(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def normalize_bool_series(s: pd.Series) -> pd.Series:
    if not isinstance(s, pd.Series):
        s = pd.Series([s])
    return s.apply(_normalize_bool_value).astype("boolean")


def bucket_mid(value: str) -> int:
    if value is None:
        return 0
    return BUCKET_TO_MID.get(str(value).strip(), 0)


def bucket_mid_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().map(BUCKET_TO_MID).fillna(0).astype("Int64")


def _validate_enums(df: pd.DataFrame, col: str, allowed: List[str]) -> List[str]:
    if col not in df.columns:
        return []
    vals = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    bad = sorted(set(v for v in vals if v != "" and v not in allowed))
    return bad


def standardize_colname(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = normalize_headers([standardize_colname(c) for c in out.columns])
    return out


def detect_matchrobot_schema(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "active_shoot_bucket" in cols:
        return "NEW"
    if "active_shooting_est" in cols or "active_shoot_est" in cols:
        return "LEGACY"
    return "MINIMAL"


def _bucket_from_numeric(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce").fillna(0)
    return pd.Series(
        pd.cut(
            n,
            bins=[-1e9, 5, 12, 20, 1e9],
            labels=["0-5", "6-12", "13-20", "21+"],
            include_lowest=True,
        ),
        index=s.index,
        dtype="string",
    ).fillna("0-5")


def normalize_matchrobot_df(df: pd.DataFrame, default_event_key: str = "2026TEST") -> pd.DataFrame:
    clean, _, _ = normalize_matchrobot_df_with_meta(df, default_event_key=default_event_key)
    return clean


def normalize_matchrobot_df_with_meta(
    df: pd.DataFrame, default_event_key: str = "2026TEST"
) -> tuple[pd.DataFrame, str, List[str]]:
    clean = standardize_columns(df)
    warnings: List[str] = []
    schema = detect_matchrobot_schema(clean)

    aliases = {
        "match": "match_number",
        "qual_number": "match_number",
        "team_number": "team",
        "defense": "played_defense",
        "active_shoot_est": "active_shooting_est",
        "ferry_count_est": "ferry_est_mid",
        "station": "slot",
    }
    for src, dst in aliases.items():
        if src in clean.columns and dst not in clean.columns:
            clean[dst] = clean[src]

    # base required/minimal fields
    if "event_key" not in clean.columns:
        clean["event_key"] = default_event_key
        warnings.append(f"event_key missing; defaulted to {default_event_key}.")
    if "match_number" not in clean.columns or "team" not in clean.columns:
        missing = [c for c in ["match_number", "team"] if c not in clean.columns]
        raise ValueError(
            "Cannot normalize MatchRobot CSV. Missing base required columns: "
            f"{missing}. Found columns: {list(clean.columns)}"
        )
    if "active_cycles" not in clean.columns:
        clean["active_cycles"] = 0
        warnings.append("active_cycles missing; defaulted to 0.")

    for bcol in [
        "ferried",
        "inactive_prepped_to_shoot",
        "played_defense",
        "auto_win",
        "climb_failed",
        "endgame_kept_shooting",
        "auto_leave_alliance_zone",
        "auto_enter_neutral_zone",
    ]:
        if bcol in clean.columns:
            clean[bcol] = normalize_bool_series(clean[bcol])
    if "ferried" not in clean.columns and "ferry_est_mid" in clean.columns:
        clean["ferried"] = pd.to_numeric(clean["ferry_est_mid"], errors="coerce").fillna(0).gt(0)

    # NEW schema path
    if schema == "NEW":
        clean["active_shoot_bucket"] = clean["active_shoot_bucket"].fillna("").astype(str).str.strip()
        clean["active_shoot_est_mid"] = bucket_mid_series(clean["active_shoot_bucket"])
        if "ferried" not in clean.columns:
            if "ferry_bucket" in clean.columns:
                clean["ferried"] = clean["ferry_bucket"].fillna("").astype(str).str.strip().ne("")
            else:
                clean["ferried"] = False
            warnings.append("ferried missing; derived from non-empty ferry_bucket.")
        clean["ferried"] = normalize_bool_series(clean["ferried"])
        if "ferry_bucket" not in clean.columns:
            clean["ferry_bucket"] = ""
        clean["ferry_bucket"] = clean["ferry_bucket"].fillna("").astype(str).str.strip()
        clean["ferry_est_mid"] = bucket_mid_series(clean["ferry_bucket"])
        clean.loc[~clean["ferried"].fillna(False), "ferry_bucket"] = ""
        clean.loc[~clean["ferried"].fillna(False), "ferry_est_mid"] = 0
        if "played_defense" not in clean.columns:
            clean["played_defense"] = False
        clean["played_defense"] = normalize_bool_series(clean["played_defense"])
        if "defense_effective" not in clean.columns:
            clean["defense_effective"] = ""
        clean["defense_effective"] = clean["defense_effective"].fillna("").astype(str).str.strip()
        clean.loc[~clean["played_defense"].fillna(False), "defense_effective"] = ""

    # LEGACY schema path
    elif schema == "LEGACY":
        if "active_shooting_est" not in clean.columns and "active_shoot_est" in clean.columns:
            clean["active_shooting_est"] = clean["active_shoot_est"]
        clean["active_shooting_est"] = pd.to_numeric(clean["active_shooting_est"], errors="coerce").fillna(0).astype("Int64")
        clean["active_shoot_est_mid"] = clean["active_shooting_est"].astype("Int64")
        clean["active_shoot_bucket"] = _bucket_from_numeric(clean["active_shoot_est_mid"])

        if "inactive_ferry_est" not in clean.columns and "ferry_count_est" in clean.columns:
            clean["inactive_ferry_est"] = clean["ferry_count_est"]
        if "inactive_ferry_est" in clean.columns:
            clean["inactive_ferry_est"] = pd.to_numeric(clean["inactive_ferry_est"], errors="coerce").fillna(0).astype("Int64")
            clean["ferry_est_mid"] = clean["inactive_ferry_est"].astype("Int64")
            clean["ferried"] = clean["ferry_est_mid"] > 0
            clean["ferry_bucket"] = _bucket_from_numeric(clean["ferry_est_mid"])
            clean.loc[~clean["ferried"].fillna(False), "ferry_bucket"] = ""
            clean.loc[~clean["ferried"].fillna(False), "ferry_est_mid"] = 0
        else:
            clean["ferried"] = False
            clean["ferry_bucket"] = ""
            clean["ferry_est_mid"] = 0
            warnings.append("inactive_ferry_est missing in legacy file; ferry defaults applied.")

        if "played_defense" not in clean.columns:
            if "defense_level" in clean.columns:
                clean["played_defense"] = pd.to_numeric(clean["defense_level"], errors="coerce").fillna(0) > 0
            else:
                clean["played_defense"] = False
        clean["played_defense"] = normalize_bool_series(clean["played_defense"])
        if "defense_effective" not in clean.columns:
            clean["defense_effective"] = ""
            clean.loc[clean["played_defense"].fillna(False), "defense_effective"] = "Somewhat"
        clean.loc[~clean["played_defense"].fillna(False), "defense_effective"] = ""

        if "inactive_prepped_to_shoot" not in clean.columns:
            clean["inactive_prepped_to_shoot"] = False
            warnings.append("inactive_prepped_to_shoot missing; defaulted FALSE.")

    # MINIMAL fallback
    else:
        raise ValueError(
            "Could not detect schema. Need at least one of: "
            "active_shoot_bucket (NEW) or active_shooting_est (LEGACY)."
        )

    # endgame_score fill
    if "endgame_score" not in clean.columns:
        if "climb_level" in clean.columns:
            c = pd.to_numeric(clean["climb_level"], errors="coerce").fillna(0)
            clean["endgame_score"] = c.map({0: 0, 1: 10, 2: 20, 3: 30}).fillna(c * 10).astype("Int64")
        else:
            clean["endgame_score"] = 0
            warnings.append("endgame_score/climb_level missing; defaulted endgame_score to 0.")

    if "ferried" in clean.columns and "ferry_bucket" in clean.columns:
        bad_mask = (~clean["ferried"].fillna(False)) & clean["ferry_bucket"].fillna("").astype(str).str.strip().ne("")
        if int(bad_mask.sum()) > 0:
            warnings.append("ferry_bucket present while ferried=FALSE; cleared.")
            clean.loc[bad_mask, "ferry_bucket"] = ""
            clean.loc[bad_mask, "ferry_est_mid"] = 0
    if "played_defense" in clean.columns and "defense_effective" in clean.columns:
        bad_mask = (~clean["played_defense"].fillna(False)) & clean["defense_effective"].fillna("").astype(str).str.strip().ne("")
        if int(bad_mask.sum()) > 0:
            warnings.append("defense_effective present while played_defense=FALSE; cleared.")
            clean.loc[bad_mask, "defense_effective"] = ""

    # Canonical aliases required by backend ingest contract.
    clean["active_shoot_est"] = pd.to_numeric(
        clean.get("active_shoot_est_mid", clean.get("active_shooting_est", 0)),
        errors="coerce",
    ).fillna(0).astype("Int64")
    clean["ferry_count_est"] = pd.to_numeric(
        clean.get("ferry_est_mid", clean.get("inactive_ferry_est", 0)),
        errors="coerce",
    ).fillna(0).astype("Int64")
    if "active_miss_heavy" in clean.columns:
        clean["miss_heavy"] = normalize_bool_series(clean["active_miss_heavy"])
    else:
        clean["miss_heavy"] = pd.Series([False] * len(clean), index=clean.index, dtype="boolean")
    clean.loc[~clean["ferried"].fillna(False), "ferry_count_est"] = 0
    clean.loc[~clean["played_defense"].fillna(False), "defense_effective"] = ""
    return clean, schema, warnings


def normalize_superscout_df(df: pd.DataFrame, default_event_key: str = "2026TEST") -> ValidationResult:
    clean = standardize_columns(df)
    warnings: List[str] = []
    aliases = {
        "match": "match_number",
        "team_number": "team",
        "auto_path_image": "auto_path_image_ref",
        "alliance_color": "alliance",
    }
    for src, dst in aliases.items():
        if src in clean.columns and dst not in clean.columns:
            clean[dst] = clean[src]
    if "event_key" not in clean.columns:
        clean["event_key"] = default_event_key
        warnings.append(f"event_key missing; defaulted to {default_event_key}.")
    required = ["event_key", "match_number", "team"]
    missing = [c for c in required if c not in clean.columns]
    if missing:
        raise ValueError(f"Super Scout missing required columns: {missing}")
    clean["match_number"] = pd.to_numeric(clean["match_number"], errors="coerce").astype("Int64")
    clean["team"] = pd.to_numeric(clean["team"], errors="coerce").astype("Int64")
    if "alliance" in clean.columns:
        clean["alliance"] = clean["alliance"].astype(str).str.strip().str.lower()
    else:
        clean["alliance"] = pd.Series([pd.NA] * len(clean), dtype="string")
        warnings.append("alliance missing; stored as NULL.")
    for bcol in ["want_on_alliance", "reliability_flag"]:
        if bcol in clean.columns:
            clean[bcol] = normalize_bool_series(clean[bcol])
    bad = _validate_enums(clean, "auto_path_tag", AUTO_PATH_TAG_VALUES)
    if bad:
        raise ValueError(f"Invalid auto_path_tag values: {bad}")
    return ValidationResult(df=clean, warnings=warnings, schema="SUPER_SCOUT")


def normalize_pit_df(df: pd.DataFrame, default_event_key: str = "2026TEST") -> ValidationResult:
    clean = standardize_columns(df)
    warnings: List[str] = []
    aliases = {"team_number": "team", "robot_weight": "weight"}
    for src, dst in aliases.items():
        if src in clean.columns and dst not in clean.columns:
            clean[dst] = clean[src]
    if "team" not in clean.columns:
        raise ValueError("Pit CSV missing required column: team")
    clean["team"] = pd.to_numeric(clean["team"], errors="coerce").astype("Int64")
    if "event_key" not in clean.columns:
        clean["event_key"] = pd.Series([pd.NA] * len(clean), dtype="string")
    for bcol in ["can_ferry", "can_play_defense"]:
        if bcol in clean.columns:
            clean[bcol] = normalize_bool_series(clean[bcol])
    if "updated_at" not in clean.columns:
        clean["updated_at"] = pd.Timestamp.utcnow()
    return ValidationResult(df=clean, warnings=warnings, schema="PIT")


def normalize_schedule_df(df: pd.DataFrame, default_event_key: str = "2026TEST") -> ValidationResult:
    clean = standardize_columns(df)
    warnings: List[str] = []
    req = {"match_number", "red1", "red2", "red3", "blue1", "blue2", "blue3"}
    long_req = {"match_number", "alliance", "position", "team"}
    if req.issubset(set(clean.columns)):
        pass
    elif long_req.issubset(set(clean.columns)):
        long_df = clean.copy()
        long_df["alliance"] = long_df["alliance"].astype(str).str.strip().str.lower()
        long_df["position"] = pd.to_numeric(long_df["position"], errors="coerce").astype("Int64")
        long_df["team"] = pd.to_numeric(long_df["team"], errors="coerce").astype("Int64")
        long_df["match_number"] = pd.to_numeric(long_df["match_number"], errors="coerce").astype("Int64")
        rows = []
        for match_num, grp in long_df.groupby("match_number"):
            if pd.isna(match_num):
                continue
            row = {
                "match_number": int(match_num),
                "red1": pd.NA,
                "red2": pd.NA,
                "red3": pd.NA,
                "blue1": pd.NA,
                "blue2": pd.NA,
                "blue3": pd.NA,
            }
            for _, r in grp.iterrows():
                al = str(r["alliance"]).lower()
                if al in {"red", "blue"} and int(r["position"]) in {1, 2, 3}:
                    row[f"{al}{int(r['position'])}"] = int(r["team"])
            rows.append(row)
        clean = pd.DataFrame(rows)
    else:
        missing = [c for c in req if c not in clean.columns]
        raise ValueError(f"Schedule CSV missing required columns: {missing} or long-format columns {sorted(long_req)}")
    if "event_key" not in clean.columns:
        clean["event_key"] = default_event_key
        warnings.append(f"event_key missing; defaulted to {default_event_key}.")
    for c in ["match_number", "red1", "red2", "red3", "blue1", "blue2", "blue3"]:
        clean[c] = pd.to_numeric(clean[c], errors="coerce").astype("Int64")
    if "updated_at" not in clean.columns:
        clean["updated_at"] = pd.Timestamp.utcnow()
    return ValidationResult(df=clean, warnings=warnings, schema="SCHEDULE")


def validate_appsheet_csv(df: pd.DataFrame, table_name: str) -> ValidationResult:
    table = table_name.strip().lower()
    if table in {"superscoutauto", "super_scout_auto", "super scout auto", "super scout"}:
        result = normalize_superscout_df(df)
        clean = result.df
        unknown = [c for c in clean.columns if c not in set(SUPERSCOUT_REQUIRED + SUPERSCOUT_OPTIONAL + ["alliance"])]
        warnings: List[str] = list(result.warnings)
        if unknown:
            warnings.append(f"Unknown columns (kept, non-blocking): {unknown}")
        bad = _validate_enums(clean, "auto_path_tag", AUTO_PATH_TAG_VALUES)
        if bad:
            raise ValueError(
                "AppSheet Super Scout CSV validation failed.\n"
                f"Invalid auto_path_tag values: {bad}\n"
                f"Example rows: {clean.head(5).to_dict(orient='records')}"
            )
        return ValidationResult(df=clean, warnings=warnings, schema="SUPER_SCOUT")

    clean, schema, warnings = normalize_matchrobot_df_with_meta(df)

    # Schema-aware required checks (soft for legacy).
    if schema == "NEW":
        required = ["match_number", "team", "active_cycles", "active_shoot_bucket", "inactive_prepped_to_shoot"]
    elif schema == "LEGACY":
        required = ["match_number", "team"]
        if "active_shooting_est" not in clean.columns and "active_shoot_bucket" not in clean.columns:
            required.append("active_shooting_est|active_shoot_bucket")
    else:
        required = ["match_number", "team"]
    missing = [c for c in required if c not in clean.columns and "|" not in c]

    bad_by_col: Dict[str, List[str]] = {}
    enum_checks = {
        "active_shoot_bucket": BUCKET_VALUES,
        "ferry_bucket": BUCKET_VALUES,
        "defense_effective": DEFENSE_EFFECTIVE_VALUES,
        "auto_preload_result": AUTO_PRELOAD_VALUES,
        "auto_alliance_zone_pickups": AUTO_PICKUP_VALUES,
    }
    for col, allowed in enum_checks.items():
        bad = _validate_enums(clean, col, allowed)
        if bad:
            bad_by_col[col] = bad

    if missing or bad_by_col:
        raise ValueError(
            "AppSheet MatchRobot CSV validation failed.\n"
            f"Detected schema: {schema}\n"
            f"Missing required columns: {missing}\n"
            f"Invalid enum values: {bad_by_col}\n"
            f"Example rows: {clean.head(5).to_dict(orient='records')}"
        )
    return ValidationResult(df=clean, warnings=warnings, schema=schema)
