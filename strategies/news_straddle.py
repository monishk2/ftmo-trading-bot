"""
strategies/news_straddle.py
============================
News Event Straddle and Fade strategies on M15 bars.

Two variants
------------
NewsStraddle:
  At event bar T, compute |close_T - anchor| where anchor = close of T-1 bar.
  If move > bracket (bracket_atr_mult × ATR14), enter in direction of move.
  SL = anchor ∓ bracket (opposite side of the bracket).
  TP = entry ± rr_ratio × |entry - sl|.
  Entry: T bar close (backtester fills at next-bar-open equivalent).

NewsFade:
  At event bar T, identify spike direction by event bar's close vs open.
  Enter OPPOSITE direction on T+1 bar open (immediate reversal fade).
  SL = spike extreme + sl_buffer_pips.
  TP = anchor (pre-event close — the "mean reversion" target).
  Only take fade if event bar range > min_spike_pips (filter for real spikes).

Instruments: EURUSD, GBPUSD, XAUUSD
Spread during news: normal_spread × news_spread_mult (default 2.0)

Position sizing (handled by caller, not in generate_signals):
  risk_dollars = balance × risk_pct/100
  lot_size     = risk_dollars / (pip_risk × pip_value_per_lot)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Instrument pip sizes and pip values ($/pip/lot)
INST_CFG = {
    "EURUSD": {"pip_size": 0.0001, "pip_value": 10.0},
    "GBPUSD": {"pip_size": 0.0001, "pip_value": 10.0},
    "XAUUSD": {"pip_size": 0.01,   "pip_value":  1.0},
}

# Normal spreads (half-spread applied at entry)
NORMAL_SPREAD_PIPS = {
    "EURUSD": 1.5,
    "GBPUSD": 2.0,
    "XAUUSD": 20.0,
}


@dataclass
class NewsSetup:
    """One actionable news event setup."""
    event_bar_idx:   int
    event_type:      str
    instrument:      str
    direction:       int          # +1 long / -1 short
    entry_price:     float        # entry after half-spread at news rate
    sl_price:        float
    tp_price:        float
    time_stop:       pd.Timestamp
    anchor:          float        # pre-event bar close
    atr:             float
    event_range_pips: float       # event bar range in pips
    variant:         str          # 'straddle' or 'fade'
    pip_size:        float
    pip_value:       float        # $/pip/lot


def _atr14(df: pd.DataFrame, idx: int) -> float:
    """14-period ATR using prior bars up to and including idx-1."""
    start = max(0, idx - 14)
    window = df.iloc[start:idx]
    if len(window) < 2:
        return (df.iloc[idx]["high"] - df.iloc[idx]["low"])
    tr = np.maximum(
        window["high"] - window["low"],
        np.maximum(
            np.abs(window["high"] - window["close"].shift(1)),
            np.abs(window["low"]  - window["close"].shift(1)),
        ),
    )
    return float(tr.mean())


def generate_straddle_setups(
    df: pd.DataFrame,
    news_df: pd.DataFrame,
    instrument: str,
    *,
    bracket_atr_mult: float = 0.5,
    rr_ratio: float = 2.0,
    time_stop_hours: float = 4.0,
    news_spread_mult: float = 2.0,
    filter_event_types: Optional[List[str]] = None,
) -> List[NewsSetup]:
    """
    Generate straddle setups for all news events in news_df.

    Parameters
    ----------
    df               : M15 OHLCV DataFrame indexed by tz-aware ET timestamps
    news_df          : output of data/news_calendar.build_calendar()
    instrument       : e.g. 'EURUSD'
    bracket_atr_mult : bracket half-width = this × ATR14
    rr_ratio         : TP distance = rr_ratio × SL distance
    time_stop_hours  : close trade N hours after event
    news_spread_mult : spread multiplier at news time
    filter_event_types : if set, only process these event types (e.g. ['NFP', 'CPI'])
    """
    cfg      = INST_CFG[instrument]
    pip_size = cfg["pip_size"]
    pip_val  = cfg["pip_value"]
    half_spread = NORMAL_SPREAD_PIPS[instrument] * news_spread_mult / 2.0 * pip_size

    idx_arr = df.index  # tz-aware ET
    setups: List[NewsSetup] = []

    filt = news_df if filter_event_types is None else news_df[news_df["event_type"].isin(filter_event_types)]

    for _, row in filt.iterrows():
        evt_ts = row["datetime"]
        if evt_ts.tzinfo is None:
            import pytz
            evt_ts = pytz.timezone("US/Eastern").localize(evt_ts)

        # Find event bar index (searchsorted for exact or next bar)
        pos = idx_arr.searchsorted(evt_ts)
        if pos >= len(idx_arr):
            continue
        # Verify bar is within 2 minutes of expected time (M15 bars start on :00/:15/:30/:45)
        bar_ts = idx_arr[pos]
        delta_min = abs((bar_ts - evt_ts).total_seconds()) / 60.0
        if delta_min > 16:
            continue  # no bar close to this event
        event_i = pos
        if event_i < 2:
            continue

        pre_i = event_i - 1          # T-1 bar (anchor)
        anchor = float(df.iloc[pre_i]["close"])
        atr    = _atr14(df, event_i)

        bracket = bracket_atr_mult * atr
        event_close = float(df.iloc[event_i]["close"])
        event_high  = float(df.iloc[event_i]["high"])
        event_low   = float(df.iloc[event_i]["low"])
        event_range_pips = (event_high - event_low) / pip_size

        move = event_close - anchor
        if abs(move) < bracket:
            continue  # not enough move — neutral event

        direction = 1 if move > 0 else -1
        entry = event_close + direction * half_spread
        sl    = anchor - direction * bracket
        risk  = abs(entry - sl)
        if risk < 1e-9:
            continue
        tp = entry + direction * rr_ratio * risk
        ts = bar_ts + pd.Timedelta(hours=time_stop_hours)

        setups.append(NewsSetup(
            event_bar_idx   = event_i,
            event_type      = row["event_type"],
            instrument      = instrument,
            direction       = direction,
            entry_price     = entry,
            sl_price        = sl,
            tp_price        = tp,
            time_stop       = ts,
            anchor          = anchor,
            atr             = atr,
            event_range_pips= event_range_pips,
            variant         = "straddle",
            pip_size        = pip_size,
            pip_value       = pip_val,
        ))

    return setups


def generate_fade_setups(
    df: pd.DataFrame,
    news_df: pd.DataFrame,
    instrument: str,
    *,
    min_spike_pips: float = 20.0,
    sl_buffer_pips: float = 10.0,
    rr_ratio: float = 2.0,
    time_stop_hours: float = 4.0,
    news_spread_mult: float = 2.0,
    filter_event_types: Optional[List[str]] = None,
) -> List[NewsSetup]:
    """
    Generate fade (mean-reversion) setups.

    Wait for large spike on event bar, enter OPPOSITE at T+1 bar open.
    TP = anchor (pre-event close). SL = spike extreme + sl_buffer_pips.
    """
    cfg      = INST_CFG[instrument]
    pip_size = cfg["pip_size"]
    pip_val  = cfg["pip_value"]
    half_spread = NORMAL_SPREAD_PIPS[instrument] * news_spread_mult / 2.0 * pip_size
    sl_buf  = sl_buffer_pips * pip_size

    idx_arr = df.index
    setups: List[NewsSetup] = []

    filt = news_df if filter_event_types is None else news_df[news_df["event_type"].isin(filter_event_types)]

    for _, row in filt.iterrows():
        evt_ts = row["datetime"]
        if evt_ts.tzinfo is None:
            import pytz
            evt_ts = pytz.timezone("US/Eastern").localize(evt_ts)

        pos = idx_arr.searchsorted(evt_ts)
        if pos >= len(idx_arr):
            continue
        bar_ts = idx_arr[pos]
        delta_min = abs((bar_ts - evt_ts).total_seconds()) / 60.0
        if delta_min > 16:
            continue
        event_i = pos
        if event_i < 2 or event_i + 1 >= len(df):
            continue

        pre_i  = event_i - 1
        anchor = float(df.iloc[pre_i]["close"])

        event_open  = float(df.iloc[event_i]["open"])
        event_high  = float(df.iloc[event_i]["high"])
        event_low   = float(df.iloc[event_i]["low"])
        event_range_pips = (event_high - event_low) / pip_size

        if event_range_pips < min_spike_pips:
            continue  # not a meaningful spike — skip

        # Spike direction = event bar open→close
        spike_dir = 1 if event_high - event_open > event_open - event_low else -1
        fade_dir  = -spike_dir

        # Entry: next bar open after event bar, with half-spread
        entry = float(df.iloc[event_i + 1]["open"]) + fade_dir * half_spread

        # SL: spike extreme + buffer
        if spike_dir == 1:
            sl = event_high + sl_buf
        else:
            sl = event_low - sl_buf

        risk = abs(entry - sl)
        if risk < 1e-9:
            continue

        # TP: anchor (pre-event close) OR rr_ratio × risk if anchor is closer
        tp_anchor = anchor
        tp_rr     = entry + fade_dir * rr_ratio * risk

        # Use whichever TP is closer to entry (more conservative)
        if fade_dir == 1:   # long fade: TP is above entry
            tp = min(tp_anchor, tp_rr) if tp_anchor > entry else tp_rr
        else:               # short fade: TP is below entry
            tp = max(tp_anchor, tp_rr) if tp_anchor < entry else tp_rr

        ts = bar_ts + pd.Timedelta(hours=time_stop_hours)
        atr = _atr14(df, event_i)

        setups.append(NewsSetup(
            event_bar_idx    = event_i + 1,   # entry at T+1
            event_type       = row["event_type"],
            instrument       = instrument,
            direction        = fade_dir,
            entry_price      = entry,
            sl_price         = sl,
            tp_price         = tp,
            time_stop        = ts,
            anchor           = anchor,
            atr              = atr,
            event_range_pips = event_range_pips,
            variant          = "fade",
            pip_size         = pip_size,
            pip_value        = pip_val,
        ))

    return setups
