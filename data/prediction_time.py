"""Canonical game time helpers for live predictions.

Identity rule:
  - provider_game_id  → canonical game identity
  - start_time_utc    → canonical when the game occurs
  - game_date_pacific → derived display/filter date only (not identity)

game_date on the predictions table remains populated for backward
compatibility and is set to the Pacific calendar date when a UTC
timestamp is known.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from src.utils.timezone_utils import convert_utc_to_pst, utc_to_pst_fields

# Prediction lifecycle (belief / settlement)
PRED_UPCOMING = "UPCOMING"
PRED_ACTIVE = "ACTIVE"
PRED_SETTLED = "SETTLED"
PRED_VOID = "VOID"

# Game lifecycle (schedule reality)
GAME_SCHEDULED = "SCHEDULED"
GAME_LIVE = "LIVE"
GAME_FINAL = "FINAL"

SETTLED_OR_VOID = frozenset({PRED_SETTLED, PRED_VOID})


def parse_start_time_utc(value: Any) -> Optional[datetime]:
    """Parse provider timestamps into aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    text = str(value).strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        # Date-only → midnight UTC (unknown local tip-off)
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def enrich_prediction_times(row: Dict[str, Any]) -> Dict[str, Any]:
    """Attach start_time_utc / game_date_pacific; keep game_date as Pacific date.

    Accepts optional keys: start_time_utc, game_datetime, utc_date, game_date.
    Mutates and returns the same dict.

    Date-only ``game_date`` (YYYY-MM-DD) is treated as an already-local
    Pacific calendar date for display — it is NOT reinterpreted as UTC
    midnight (which would shift the calendar day for US evening games).
    """
    timed = row.get("start_time_utc") or row.get("game_datetime") or row.get("utc_date")
    date_only = str(row.get("game_date") or "").strip()
    # A bare YYYY-MM-DD in a "time" field carries no clock time; reading it as
    # UTC midnight would move US evening games to the previous Pacific day.
    if isinstance(timed, str) and len(timed.strip()) == 10 and timed.strip()[4] == "-":
        date_only = date_only or timed.strip()
        if len(date_only) < 10:
            date_only = timed.strip()
        timed = None

    if timed:
        start = parse_start_time_utc(timed)
        if start is None:
            start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        _, pacific_date = utc_to_pst_fields(start)
        row["start_time_utc"] = start.isoformat().replace("+00:00", "Z")
        row["game_date_pacific"] = pacific_date
        row["game_date"] = pacific_date
    elif len(date_only) >= 10:
        # Preserve caller-supplied calendar date as Pacific display date.
        pacific_date = date_only[:10]
        row["game_date_pacific"] = pacific_date
        row["game_date"] = pacific_date
        # Canonical tip-off unknown — store noon Pacific as a stable UTC anchor
        # so identity/time columns exist without shifting the calendar day.
        try:
            from zoneinfo import ZoneInfo

            local = datetime.fromisoformat(pacific_date).replace(
                hour=12, minute=0, second=0, tzinfo=ZoneInfo("America/Los_Angeles")
            )
            start = local.astimezone(timezone.utc)
            row["start_time_utc"] = start.isoformat().replace("+00:00", "Z")
        except Exception:
            row["start_time_utc"] = f"{pacific_date}T19:00:00Z"
    else:
        start = datetime.now(timezone.utc)
        _, pacific_date = utc_to_pst_fields(start)
        row["start_time_utc"] = start.isoformat().replace("+00:00", "Z")
        row["game_date_pacific"] = pacific_date
        row["game_date"] = pacific_date

    if not row.get("game_status"):
        row["game_status"] = GAME_SCHEDULED
    if not row.get("prediction_status"):
        row["prediction_status"] = PRED_UPCOMING
    return row


def pacific_today() -> str:
    return convert_utc_to_pst(datetime.now(timezone.utc)).date().isoformat()


def pacific_horizon(days: int = 7) -> str:
    base = convert_utc_to_pst(datetime.now(timezone.utc)).date()
    return (base + timedelta(days=days)).isoformat()


def display_window(days_ahead: int = 7) -> Tuple[str, str]:
    """Inclusive Pacific calendar window for upcoming dashboard cards."""
    return pacific_today(), pacific_horizon(days_ahead)
