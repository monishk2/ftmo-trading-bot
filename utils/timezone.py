"""
Timezone utilities for the FTMO trading bot.

Rule: ALL data is stored as UTC-aware datetimes. Convert to US/Eastern only
for session logic (London open, NY open, etc.). pytz handles DST automatically
via tz_convert — 03:00 Eastern is correct whether it's EST (UTC-5) or EDT (UTC-4).

Usage:
    from utils.timezone import to_eastern, to_utc, localize_utc, is_session_open

    dt_eastern = to_eastern(df["datetime"])          # Series or Timestamp
    midnight_cet = get_midnight_cet(dt_utc)          # for FTMO daily reset
"""

from datetime import datetime, time
from typing import Union

import pandas as pd
import pytz

UTC = pytz.utc
EASTERN = pytz.timezone("US/Eastern")
CET = pytz.timezone("Europe/Berlin")   # FTMO server time: CET/CEST


def to_eastern(dt: Union[pd.Series, pd.Timestamp, datetime]) -> Union[pd.Series, pd.Timestamp]:
    """Convert UTC-aware datetime(s) to US/Eastern, DST-aware.

    Input must already be UTC-aware. Raises if naive.
    """
    if isinstance(dt, pd.Series):
        if dt.dt.tz is None:
            raise ValueError("Series must be UTC-aware before calling to_eastern()")
        return dt.dt.tz_convert("US/Eastern")
    # Single Timestamp / datetime
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is None:
            raise ValueError("Timestamp must be UTC-aware before calling to_eastern()")
        return dt.tz_convert("US/Eastern")
    # stdlib datetime
    if dt.tzinfo is None:
        raise ValueError("datetime must be UTC-aware before calling to_eastern()")
    return dt.astimezone(EASTERN)


def to_utc(dt: Union[pd.Series, pd.Timestamp, datetime]) -> Union[pd.Series, pd.Timestamp]:
    """Convert timezone-aware datetime(s) to UTC."""
    if isinstance(dt, pd.Series):
        if dt.dt.tz is None:
            raise ValueError("Series must be tz-aware before calling to_utc()")
        return dt.dt.tz_convert("UTC")
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is None:
            raise ValueError("Timestamp must be tz-aware before calling to_utc()")
        return dt.tz_convert("UTC")
    if dt.tzinfo is None:
        raise ValueError("datetime must be tz-aware before calling to_utc()")
    return dt.astimezone(UTC)


def localize_utc(dt: Union[pd.Series, pd.Timestamp, datetime]) -> Union[pd.Series, pd.Timestamp]:
    """Attach UTC timezone to a naive datetime (does NOT shift time)."""
    if isinstance(dt, pd.Series):
        if dt.dt.tz is not None:
            raise ValueError("Series is already tz-aware; use to_utc() to convert instead")
        return dt.dt.tz_localize("UTC")
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is not None:
            raise ValueError("Timestamp is already tz-aware")
        return dt.tz_localize("UTC")
    if dt.tzinfo is not None:
        raise ValueError("datetime is already tz-aware")
    return UTC.localize(dt)


def unix_ms_to_utc(ms: Union[pd.Series, int, float]) -> Union[pd.Series, pd.Timestamp]:
    """Convert Unix millisecond timestamp(s) to UTC-aware datetime."""
    if isinstance(ms, pd.Series):
        return pd.to_datetime(ms, unit="ms", utc=True)
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def get_midnight_cet(dt_utc: Union[pd.Timestamp, datetime]) -> pd.Timestamp:
    """Return the most recent CET/CEST midnight before dt_utc.

    FTMO resets the daily loss limit at 00:00 CET server time. Guardian uses
    this to pin the reference balance for each trading day.
    """
    if isinstance(dt_utc, datetime) and not isinstance(dt_utc, pd.Timestamp):
        dt_utc = pd.Timestamp(dt_utc, tz="UTC")
    dt_cet = dt_utc.tz_convert(CET)
    midnight_cet = dt_cet.normalize()           # floor to midnight in CET
    return midnight_cet.tz_convert("UTC")       # return as UTC for arithmetic


def is_within_session_open_window(
    dt_utc: pd.Timestamp,
    window_minutes: int = 15,
) -> bool:
    """Return True if dt_utc falls within `window_minutes` of London (03:00 ET)
    or NY (09:30 ET) session opens. Used by the backtester for elevated slippage.
    """
    dt_et = dt_utc.tz_convert(EASTERN)
    t = dt_et.time()

    london_open = time(3, 0)
    ny_open = time(9, 30)

    def _within(session_time: time) -> bool:
        session_seconds = session_time.hour * 3600 + session_time.minute * 60
        t_seconds = t.hour * 3600 + t.minute * 60
        return 0 <= (t_seconds - session_seconds) < window_minutes * 60

    return _within(london_open) or _within(ny_open)


def eastern_time(dt_utc: pd.Timestamp) -> time:
    """Return the wall-clock time in US/Eastern for a UTC timestamp."""
    return dt_utc.tz_convert(EASTERN).time()
