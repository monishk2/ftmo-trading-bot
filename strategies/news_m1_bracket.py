"""
strategies/news_m1_bracket.py
==============================
Generate M1 bracket (OCO stop) orders placed 2 bars before each scheduled
news event.

Two variants
------------
Bracket (momentum):
  BUY STOP  above anchor + bracket_distance
  SELL STOP below anchor - bracket_distance
  → rides the news spike in whichever direction it breaks

Fade (mean-reversion):
  BUY LIMIT  below anchor - fade_distance  (buy the overshoot low)
  SELL LIMIT above anchor + fade_distance  (sell the overshoot high)
  → profits when price snaps back to anchor after spike
  TP = anchor price; SL = fade_distance × sl_mult beyond entry

Both variants place OCO pairs.

Instrument
----------
Primary: XAUUSD (M1 ATR is in $0.01 units = 1 pip per dollar move on gold)
Supports any instrument in INST_CFG.

Config keys (from strategy_params.json "news_m1_bracket")
-----------
  bracket_distance_atr_mult   float (default 0.5)  — M1 ATR(60) × mult
  fade_distance_atr_mult      float (default 0.5)
  rr_ratio                    float (default 2.0)
  time_stop_bars              int   (default 120)   — bars after fill
  expiry_bars                 int   (default 30)    — bars to stay pending
  news_spread_mult            float (default 2.0)
  news_window_bars            int   (default 5)     — bars around event
  min_atr_pips                float (default 0.0)   — skip if ATR too small
  risk_per_trade_pct          float (default 1.0)
  filter_event_types          list  (default all)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from backtesting.backtester import PendingOrder

# ── Instrument pip metadata ───────────────────────────────────────────────────
INST_CFG = {
    "EURUSD": {"pip_size": 0.0001},
    "GBPUSD": {"pip_size": 0.0001},
    "XAUUSD": {"pip_size": 0.01},
}


@dataclass
class BracketSetupMeta:
    """Per-event metadata stored alongside the PendingOrder pair."""
    event_type:     str
    event_bar:      int
    anchor:         float
    atr:            float           # M1 ATR(60) in price units at T-2
    bracket_dist:   float           # price distance used for bracket
    direction:      int             # which leg filled (+1 long / -1 short / 0 = unfilled)
    fill_price:     Optional[float] = None
    group_id:       int             = 0


def _m1_atr60(arr_h: np.ndarray, arr_l: np.ndarray,
              arr_c: np.ndarray, i: int) -> float:
    """Simple 60-bar ATR (Wilder) at bar i using prior 60 bars."""
    start = max(0, i - 60)
    if i - start < 2:
        return float(arr_h[i] - arr_l[i])
    h = arr_h[start:i]
    l = arr_l[start:i]
    c = arr_c[start - 1: i - 1] if start > 0 else arr_c[0:i - 1]
    c = c[:len(h)]
    tr = np.maximum(
        h - l,
        np.maximum(np.abs(h - c), np.abs(l - c))
    )
    return float(tr.mean())


def generate_bracket_orders(
    df_m1: pd.DataFrame,
    news_df: pd.DataFrame,
    instrument: str = "XAUUSD",
    *,
    bracket_distance_atr_mult: float = 0.5,
    rr_ratio: float = 2.0,
    time_stop_bars: int = 120,
    expiry_bars: int = 30,
    news_spread_mult: float = 2.0,
    news_window_bars: int = 5,
    filter_event_types: Optional[List[str]] = None,
    pre_event_bars: int = 2,
) -> tuple[List[PendingOrder], List[BracketSetupMeta]]:
    """
    Generate OCO bracket (buy-stop + sell-stop) orders for each news event.

    Returns
    -------
    orders  : flat list of PendingOrder objects ready for Backtester.run_bracket()
    metas   : BracketSetupMeta per event (for analysis / diagnostics)
    """
    pip_size = INST_CFG[instrument]["pip_size"]
    idx      = df_m1.index
    n        = len(df_m1)
    arr_h    = df_m1["high"].to_numpy(float)
    arr_l    = df_m1["low"].to_numpy(float)
    arr_c    = df_m1["close"].to_numpy(float)

    filt = (news_df if filter_event_types is None
            else news_df[news_df["event_type"].isin(filter_event_types)])

    orders: List[PendingOrder] = []
    metas:  List[BracketSetupMeta] = []
    group_counter = 0

    for _, row in filt.iterrows():
        evt_ts = row["datetime"]
        if evt_ts.tzinfo is None:
            import pytz
            evt_ts = pytz.timezone("US/Eastern").localize(evt_ts)

        # Find event bar
        pos = idx.searchsorted(evt_ts)
        if pos >= n:
            continue
        delta_min = abs((idx[pos] - evt_ts).total_seconds()) / 60.0
        if delta_min > 2:        # M1 bar must be within 2 min of event time
            continue
        event_bar = pos

        place_bar = event_bar - pre_event_bars
        if place_bar < 60:       # need 60 bars for ATR
            continue

        anchor = float(arr_c[place_bar])
        atr    = _m1_atr60(arr_h, arr_l, arr_c, place_bar)
        if atr <= 0:
            continue

        bracket_dist = bracket_distance_atr_mult * atr
        if bracket_dist < pip_size:
            continue

        group_id    = group_counter
        group_counter += 1
        expiry_bar  = place_bar + expiry_bars

        # SL for buy_stop = anchor - bracket_dist (mirror)
        # SL for sell_stop = anchor + bracket_dist (mirror)
        buy_entry  = anchor + bracket_dist
        buy_sl     = anchor - bracket_dist
        buy_risk   = buy_entry - buy_sl
        buy_tp     = buy_entry + rr_ratio * buy_risk

        sell_entry = anchor - bracket_dist
        sell_sl    = anchor + bracket_dist
        sell_risk  = sell_sl - sell_entry
        sell_tp    = sell_entry - rr_ratio * sell_risk

        orders.append(PendingOrder(
            order_type       = "buy_stop",
            entry_price      = buy_entry,
            sl_price         = buy_sl,
            tp_price         = buy_tp,
            place_bar        = place_bar,
            expiry_bar       = expiry_bar,
            group_id         = group_id,
            time_stop_bars   = time_stop_bars,
            news_spread_mult = news_spread_mult,
            event_bar        = event_bar,
            news_window_bars = news_window_bars,
            event_type       = row["event_type"],
        ))
        orders.append(PendingOrder(
            order_type       = "sell_stop",
            entry_price      = sell_entry,
            sl_price         = sell_sl,
            tp_price         = sell_tp,
            place_bar        = place_bar,
            expiry_bar       = expiry_bar,
            group_id         = group_id,
            time_stop_bars   = time_stop_bars,
            news_spread_mult = news_spread_mult,
            event_bar        = event_bar,
            news_window_bars = news_window_bars,
            event_type       = row["event_type"],
        ))

        metas.append(BracketSetupMeta(
            event_type   = row["event_type"],
            event_bar    = event_bar,
            anchor       = anchor,
            atr          = atr,
            bracket_dist = bracket_dist,
            direction    = 0,
            group_id     = group_id,
        ))

    return orders, metas


def generate_fade_orders(
    df_m1: pd.DataFrame,
    news_df: pd.DataFrame,
    instrument: str = "XAUUSD",
    *,
    fade_distance_atr_mult: float = 0.5,
    sl_mult: float = 1.5,
    rr_ratio: float = 1.5,
    time_stop_bars: int = 60,
    expiry_bars: int = 15,
    news_spread_mult: float = 2.0,
    news_window_bars: int = 5,
    filter_event_types: Optional[List[str]] = None,
    pre_event_bars: int = 2,
) -> tuple[List[PendingOrder], List[BracketSetupMeta]]:
    """
    Generate OCO fade (buy-limit + sell-limit) orders.

    Fade orders sit BELOW/ABOVE anchor and get filled if price overshoots.
    TP = anchor (mean reversion). SL = entry ∓ sl_mult × fade_distance.
    """
    pip_size = INST_CFG[instrument]["pip_size"]
    idx      = df_m1.index
    n        = len(df_m1)
    arr_h    = df_m1["high"].to_numpy(float)
    arr_l    = df_m1["low"].to_numpy(float)
    arr_c    = df_m1["close"].to_numpy(float)

    filt = (news_df if filter_event_types is None
            else news_df[news_df["event_type"].isin(filter_event_types)])

    orders: List[PendingOrder] = []
    metas:  List[BracketSetupMeta] = []
    group_counter = 0

    for _, row in filt.iterrows():
        evt_ts = row["datetime"]
        if evt_ts.tzinfo is None:
            import pytz
            evt_ts = pytz.timezone("US/Eastern").localize(evt_ts)

        pos = idx.searchsorted(evt_ts)
        if pos >= n:
            continue
        delta_min = abs((idx[pos] - evt_ts).total_seconds()) / 60.0
        if delta_min > 2:
            continue
        event_bar = pos
        place_bar = event_bar - pre_event_bars
        if place_bar < 60:
            continue

        anchor = float(arr_c[place_bar])
        atr    = _m1_atr60(arr_h, arr_l, arr_c, place_bar)
        if atr <= 0:
            continue

        fade_dist = fade_distance_atr_mult * atr
        if fade_dist < pip_size:
            continue

        group_id   = group_counter
        group_counter += 1
        expiry_bar = place_bar + expiry_bars

        # BUY LIMIT: price drops below anchor - fade_dist, then reverses to anchor
        buy_entry  = anchor - fade_dist
        buy_sl     = buy_entry - sl_mult * fade_dist
        buy_tp     = anchor   # revert to pre-event price

        # SELL LIMIT: price rises above anchor + fade_dist, then falls to anchor
        sell_entry = anchor + fade_dist
        sell_sl    = sell_entry + sl_mult * fade_dist
        sell_tp    = anchor

        # Validate RR (TP must be on correct side)
        if buy_tp <= buy_entry or sell_tp >= sell_entry:
            continue

        orders.append(PendingOrder(
            order_type       = "buy_limit",
            entry_price      = buy_entry,
            sl_price         = buy_sl,
            tp_price         = buy_tp,
            place_bar        = place_bar,
            expiry_bar       = expiry_bar,
            group_id         = group_id,
            time_stop_bars   = time_stop_bars,
            news_spread_mult = news_spread_mult,
            event_bar        = event_bar,
            news_window_bars = news_window_bars,
            event_type       = row["event_type"],
        ))
        orders.append(PendingOrder(
            order_type       = "sell_limit",
            entry_price      = sell_entry,
            sl_price         = sell_sl,
            tp_price         = sell_tp,
            place_bar        = place_bar,
            expiry_bar       = expiry_bar,
            group_id         = group_id,
            time_stop_bars   = time_stop_bars,
            news_spread_mult = news_spread_mult,
            event_bar        = event_bar,
            news_window_bars = news_window_bars,
            event_type       = row["event_type"],
        ))

        metas.append(BracketSetupMeta(
            event_type   = row["event_type"],
            event_bar    = event_bar,
            anchor       = anchor,
            atr          = atr,
            bracket_dist = fade_dist,
            direction    = 0,
            group_id     = group_id,
        ))

    return orders, metas
