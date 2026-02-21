from __future__ import annotations

import re
from typing import Iterable, List, Set


_HEADER_SYNONYMS = {
    "match": "match_number",
    "match_no": "match_number",
    "matchnum": "match_number",
    "qual_number": "match_number",
    "team_number": "team",
    "red_1": "red1",
    "red_2": "red2",
    "red_3": "red3",
    "blue_1": "blue1",
    "blue_2": "blue2",
    "blue_3": "blue3",
    "active_shooting_est": "active_shoot_est",
    "inactive_ferry_est": "ferry_count_est",
    "super_scout_auto_path_tag": "auto_path_tag",
}


def _clean_header(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return _HEADER_SYNONYMS.get(s, s)


def normalize_headers(headers: List[str]) -> List[str]:
    return [_clean_header(h) for h in headers]


def _score(cols: Set[str], signals: Iterable[str], weight: int = 1) -> int:
    return weight * sum(1 for s in signals if s in cols)


def detect_csv_kind(cols: Set[str]) -> str:
    c = {_clean_header(x) for x in cols}

    # Venom Scout export signatures
    match_export_required = {"composite_key", "event_code", "match_num", "alliance", "station", "team_num"}
    super_export_required = {"hub_status", "driver_ranking_1st"}
    has_super_team_cols = any(k in c for k in {"r1_team_num", "r2_team_num", "r3_team_num"})
    if match_export_required.issubset(c):
        return "matchrobot"
    if super_export_required.issubset(c) and has_super_team_cols:
        return "superscout"

    schedule_signals = {"match_number", "red1", "red2", "red3", "blue1", "blue2", "blue3"}
    schedule_long_signals = {"match_number", "alliance", "position", "team"}
    pit_signals = {"team", "drivebase", "intake", "climb_claim", "can_ferry", "preferred_role", "known_issues"}
    superscout_signals = {
        "team",
        "match_number",
        "auto_path_tag",
        "auto_path_image_ref",
        "want_on_alliance",
        "driver_skill",
        "reliability_flag",
    }
    matchrobot_signals = {
        "team",
        "match_number",
        "active_cycles",
        "active_shoot_est",
        "climb_level",
        "ferried",
        "ferry_count_est",
        "inactive_prepped_to_shoot",
    }

    sched_score = _score(c, schedule_signals, weight=2) + _score(c, schedule_long_signals, weight=2)
    pit_score = _score(c, pit_signals)
    ss_score = _score(c, superscout_signals)
    mr_score = _score(c, matchrobot_signals)

    # Strong schedule shape wins immediately (wide), or long without matchrobot signals.
    has_wide_schedule = schedule_signals.issubset(c)
    has_long_schedule = schedule_long_signals.issubset(c)
    if has_wide_schedule:
        return "schedule"
    if has_long_schedule and mr_score <= 2:
        return "schedule"
    if mr_score >= 6:
        return "matchrobot"

    # Tie-breaks and shape checks.
    if ss_score == mr_score and ss_score > 0:
        if "auto_path_tag" in c:
            return "superscout"
        return "matchrobot"
    if pit_score == mr_score and pit_score > 0:
        if "match_number" not in c:
            return "pit"
        return "matchrobot"

    ranked = sorted(
        [
            ("schedule", sched_score),
            ("pit", pit_score),
            ("superscout", ss_score),
            ("matchrobot", mr_score),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[0][0] if ranked[0][1] > 0 else "unknown"
