"""Timezone utilities for converting UTC schedule timestamps to Los Angeles time."""

from __future__ import annotations

from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


LA_TIMEZONE = 'America/Los_Angeles'


def _parse_iso_utc(value: str | datetime) -> datetime:
    """Parse ISO-8601 string or datetime into an aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def convert_utc_to_timezone(value: str | datetime, tz_name: str = LA_TIMEZONE) -> datetime:
    """Convert UTC input into a target timezone-aware datetime."""
    utc_dt = _parse_iso_utc(value)
    if ZoneInfo is None:
        raise RuntimeError('zoneinfo is not available in this Python runtime')
    return utc_dt.astimezone(ZoneInfo(tz_name))


def convert_utc_to_pst(value: str | datetime) -> datetime:
    """Convert UTC input into America/Los_Angeles timezone."""
    return convert_utc_to_timezone(value, LA_TIMEZONE)


def utc_to_pst_fields(value: str | datetime) -> tuple[str, str]:
    """Return (pst_iso_datetime, pst_date_only) from UTC input."""
    pst_dt = convert_utc_to_pst(value)
    return pst_dt.isoformat(), pst_dt.date().isoformat()


def format_pst_military_time(value: str | datetime) -> str:
    """Return PST time in HH:MM (24-hour) format from UTC input."""
    pst_dt = convert_utc_to_pst(value)
    return pst_dt.strftime('%H:%M')
