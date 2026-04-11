"""
Strategy 9: Gold M1 Micro-Swing Scalper
=========================================

WHY IT DIFFERS FROM SWEEP REVERSAL
------------------------------------
Previous M1 sweep reversal used *daily* levels (PDH/PDL/session H&L) with a
2.5:1 RR target. Result: 23% win rate (needs 29% to break even). DEAD.

This strategy targets *intraday* M15 swing points — levels that price is
actively testing within the same session. Lower RR (1.5:1) means the WR
break-even drops to 40%, which is achievable for micro-swing fades.

LEADERBOARD PROFILE MATCH
---------------------------
Target: ~5 trades/day, ~55% WR, 1.5:1 RR → Expectancy = +0.325R/trade

LEVEL IDENTIFICATION (micro-swings)
--------------------------------------
M15 swing high: M15 bar high > prior 3 AND next 3 M15 highs (3-bar lookback/lookahead)
M15 swing low:  M15 bar low  < prior 3 AND next 3 M15 lows
Confirmed at:   bar i+3 (45-min lag — avoids look-ahead bias)
Active levels:  4 most recent confirmed swing highs + 4 swing lows
                + PDH, PDL, Asian session H/L (major levels)

ENTRY LOGIC (M1 bars)
-----------------------
Session windows: London 03:00–08:00 ET | NY 08:30–14:00 ET

LONG (fade low sweep):
  1. M1 bar.low < active_low_level - sweep_min_pips × pip_size
  2. bar.close > active_low_level (closed back above — wick sweep confirmed)
  3. bar.body = |close-open| > 0.2 × ATR(60 M1)
  4. No volume filter (too noisy on M1)

SHORT: mirror

SL = wick extreme + sl_buffer_pips, capped at sl_cap_pips
TP = 1.5 × SL distance (configurable rr_ratio)
Time stop: time_stop_minutes after entry

Max 6 trades/day | Max 1 position at a time
30-min refractory period per level (rounded to $1 resolution)

REGIME / TREND FILTERS
-----------------------
H1 ATR(14) regime: must be above 30th percentile of prior 60 H1 bars
H4 EMA20 slope (soft filter):
  slope > 0 → prefer longs; shorts only if SL ≤ 100 pips (tight only)
  slope < 0 → prefer shorts; longs only if SL ≤ 100 pips

RISK
-----
risk_per_trade_pct: 0.8% per trade (configurable)
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections import defaultdict
from datetime import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(h, m)


class GoldM1Scalper(BaseStrategy):
    """Gold (XAUUSD) M1 micro-swing scalper with M15 level tracking."""

    @property
    def name(self) -> str:
        return "GoldM1Scalper"

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        # Session windows
        self._london_start = _parse_time(config.get("london_entry_start", "03:00"))
        self._london_end   = _parse_time(config.get("london_entry_end",   "08:00"))
        self._ny_start     = _parse_time(config.get("ny_entry_start",     "08:30"))
        self._ny_end       = _parse_time(config.get("ny_entry_end",       "14:00"))

        # Sweep detection
        self._sweep_min_pips  = float(config.get("sweep_min_pips",    10.0))
        self._sl_buffer_pips  = float(config.get("sl_buffer_pips",     5.0))
        self._sl_cap_pips     = float(config.get("sl_cap_pips",       200.0))
        self._rr_ratio        = float(config.get("rr_ratio",            1.5))
        self._time_stop_min   = int(config.get("time_stop_minutes",    90))

        # Entry filters
        self._atr_period      = int(config.get("atr_period",           60))   # ~1h on M1
        self._body_atr_min    = float(config.get("candle_body_atr_min", 0.2))
        self._max_trades_day  = int(config.get("max_trades_per_day",    6))
        self._refractory_min  = int(config.get("refractory_minutes",   30))

        # M15 swing detection
        self._swing_lookback  = int(config.get("swing_lookback_bars",   3))    # each side
        self._max_swing_levels = int(config.get("max_swing_levels",     4))    # each side

        # H4 soft trend filter
        self._h4_ema_period   = int(config.get("h4_ema_period",        20))
        self._h4_slope_bars   = int(config.get("h4_slope_bars",         3))
        self._soft_sl_cap     = float(config.get("soft_filter_sl_cap", 100.0)) # pips

        # H1 regime filter
        self._h1_atr_period   = int(config.get("h1_atr_period",        14))
        self._h1_lookback     = int(config.get("h1_lookback_bars",     60))
        self._h1_percentile   = float(config.get("h1_atr_percentile",  30.0))

        self._no_friday       = bool(config.get("no_friday_trading",    True))
        self.risk_per_trade_pct = float(config["risk_per_trade_pct"])
        self._pip_size        = float(instrument_config["pip_size"])

        logger.info(
            "%s setup: sweep=%d  sl_cap=%d  RR=%.1f  t_stop=%dmin  "
            "body_atr=%.2f  sessions=03-08+08:30-14 ET",
            self.name, int(self._sweep_min_pips), int(self._sl_cap_pips),
            self._rr_ratio, self._time_stop_min, self._body_atr_min,
        )

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._init_signal_columns(df)
        df["level_type"] = ""
        df["sweep_pips"] = np.nan
        df["sl_pips"]    = np.nan

        ps = self._pip_size
        n  = len(df)
        idx = df.index

        # ── M1 ATR(60) for body displacement check ─────────────────────
        atr_m1 = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=self._atr_period, fillna=False,
        ).average_true_range().to_numpy()

        # ── M15 swing levels ───────────────────────────────────────────
        swing_levels_by_m1 = self._build_swing_level_map(df, idx)

        # ── H4 soft trend (array of +1/0/-1 aligned to M1) ────────────
        h4_trend = self._compute_h4_trend(df)

        # ── H1 regime gate (bool array aligned to M1) ─────────────────
        h1_ok = self._compute_h1_regime(df)

        # ── Raw arrays ────────────────────────────────────────────────
        open_arr  = df["open"].to_numpy()
        high_arr  = df["high"].to_numpy()
        low_arr   = df["low"].to_numpy()
        close_arr = df["close"].to_numpy()
        dates_arr = np.array(idx.date)

        # ── Entry window mask ─────────────────────────────────────────
        bar_hour = idx.hour + idx.minute / 60.0
        in_london = (
            (bar_hour >= (self._london_start.hour + self._london_start.minute / 60.0)) &
            (bar_hour <  (self._london_end.hour   + self._london_end.minute   / 60.0))
        )
        in_ny = (
            (bar_hour >= (self._ny_start.hour + self._ny_start.minute / 60.0)) &
            (bar_hour <  (self._ny_end.hour   + self._ny_end.minute   / 60.0))
        )
        in_window = in_london | in_ny

        # ── Output arrays ─────────────────────────────────────────────
        sig_arr = np.zeros(n, dtype=int)
        sl_arr  = np.full(n, np.nan)
        tp_arr  = np.full(n, np.nan)
        ts_arr  = np.full(n, None, dtype=object)
        lt_arr  = np.full(n, "", dtype=object)
        sp_arr  = np.full(n, np.nan)
        slp_arr = np.full(n, np.nan)

        sweep_min_price = self._sweep_min_pips * ps
        sl_buf_price    = self._sl_buffer_pips * ps
        sl_cap_price    = self._sl_cap_pips * ps
        soft_sl_cap_p   = self._soft_sl_cap * ps
        refractory_td   = _dt.timedelta(minutes=self._refractory_min)

        # ── Per-date scan ─────────────────────────────────────────────
        date_to_positions: Dict = defaultdict(list)
        for pi in range(n):
            date_to_positions[dates_arr[pi]].append(pi)

        for trade_date, positions in sorted(date_to_positions.items()):
            if self._no_friday and idx[positions[0]].dayofweek == 4:
                continue

            entry_positions = [p for p in positions if in_window[p] and h1_ok[p]]
            if not entry_positions:
                continue

            trades_today = 0
            level_last_used: Dict[int, pd.Timestamp] = {}  # level_key → last_entry_ts

            for pos in entry_positions:
                if trades_today >= self._max_trades_day:
                    break

                hi  = high_arr[pos]
                lo  = low_arr[pos]
                cl  = close_arr[pos]
                op  = open_arr[pos]
                atr = atr_m1[pos]
                ts  = idx[pos]
                trend = int(h4_trend[pos])

                if np.isnan(atr) or atr == 0:
                    continue

                body = abs(cl - op)
                if body < self._body_atr_min * atr:
                    continue   # candle body too small

                # Get active swing levels for this M1 bar
                levels = swing_levels_by_m1.get(pos, [])
                if not levels:
                    continue

                for level_price, level_type, is_high_level in levels:
                    # ── H4 soft filter ────────────────────────────────
                    # is_high_level=True → bearish (short) signal
                    # is_high_level=False → bullish (long) signal
                    if trend == 1 and is_high_level:    # bullish H4, short setup
                        continue  # against trend, skip
                    if trend == -1 and not is_high_level:  # bearish H4, long setup
                        continue  # against trend, skip
                    # Soft: both directions allowed in neutral trend

                    # ── Refractory period check ───────────────────────
                    level_key = int(round(level_price))
                    last_use  = level_last_used.get(level_key)
                    if last_use is not None and (ts - last_use) < refractory_td:
                        continue

                    if is_high_level:
                        # SHORT: sweep of swing high
                        sweep_ok = hi > level_price + sweep_min_price
                        close_ok = cl < level_price     # closed back below
                        if not (sweep_ok and close_ok):
                            continue

                        entry   = cl
                        sl_p    = hi + sl_buf_price
                        sl_dist = abs(sl_p - entry)
                        if sl_dist > sl_cap_price:
                            continue

                        tp_p = entry - self._rr_ratio * sl_dist
                        direction = -1

                    else:
                        # LONG: sweep of swing low
                        sweep_ok = lo < level_price - sweep_min_price
                        close_ok = cl > level_price     # closed back above
                        if not (sweep_ok and close_ok):
                            continue

                        entry   = cl
                        sl_p    = lo - sl_buf_price
                        sl_dist = abs(entry - sl_p)
                        if sl_dist > sl_cap_price:
                            continue

                        tp_p = entry + self._rr_ratio * sl_dist
                        direction = 1

                    # Time stop: entry_time + time_stop_minutes
                    tstop = ts + pd.Timedelta(minutes=self._time_stop_min)

                    sig_arr[pos]  = direction
                    sl_arr[pos]   = sl_p
                    tp_arr[pos]   = tp_p
                    ts_arr[pos]   = tstop
                    lt_arr[pos]   = level_type
                    sp_arr[pos]   = round(
                        (hi - level_price if is_high_level else level_price - lo) / ps, 1
                    )
                    slp_arr[pos]  = round(sl_dist / ps, 1)

                    level_last_used[level_key] = ts
                    trades_today += 1
                    break   # one signal per bar

        df["signal"]     = sig_arr
        df["sl_price"]   = sl_arr
        df["tp_price"]   = tp_arr
        df["time_stop"]  = ts_arr
        df["level_type"] = lt_arr
        df["sweep_pips"] = sp_arr
        df["sl_pips"]    = slp_arr

        n_sig = int((sig_arr != 0).sum())
        logger.info("%s: total signals=%d", self.name, n_sig)
        return df

    # ------------------------------------------------------------------ #
    # M15 swing level computation                                          #
    # ------------------------------------------------------------------ #

    def _build_swing_level_map(self, df_m1: pd.DataFrame, idx) -> Dict[int, List]:
        """
        For each M1 bar index (int), return a list of active (price, label, is_high)
        level tuples drawn from:
          - M15 confirmed swing highs/lows (last 4 of each)
          - PDH/PDL and Asian session H/L

        Confirmed = swing at M15 bar i is confirmed at bar i + lookback (i.e., 45 min later).
        Note: All internal operations use UTC to avoid DST-ambiguity issues with floor().
        """
        ps = self._pip_size
        lk = self._swing_lookback
        max_sw = self._max_swing_levels

        # ── Work in UTC to avoid DST-ambiguity in floor() ─────────────
        df_utc = df_m1.copy()
        if df_utc.index.tzinfo is not None:
            df_utc.index = df_utc.index.tz_convert("UTC")
        idx_utc = df_utc.index

        # ── Resample M1 → M15 (UTC) ────────────────────────────────────
        m15 = df_utc.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])

        m15_times  = m15.index
        m15_highs  = m15["high"].to_numpy()
        m15_lows   = m15["low"].to_numpy()
        nm15       = len(m15)

        # ── Find M15 swing highs/lows ──────────────────────────────────
        # confirmed_swings: list of (confirmed_at_ts, price, is_high)
        confirmed_swings: List[Tuple] = []
        for i in range(lk, nm15 - lk):
            if m15_highs[i] > m15_highs[i - lk:i].max() and \
               m15_highs[i] > m15_highs[i + 1:i + lk + 1].max():
                conf_ts = m15_times[i + lk]
                confirmed_swings.append((conf_ts, m15_highs[i], True))

            if m15_lows[i] < m15_lows[i - lk:i].min() and \
               m15_lows[i] < m15_lows[i + 1:i + lk + 1].min():
                conf_ts = m15_times[i + lk]
                confirmed_swings.append((conf_ts, m15_lows[i], False))

        confirmed_swings.sort(key=lambda x: x[0])

        # ── Build rolling active levels at each M15 boundary ──────────
        active_highs: List[float] = []   # prices of recent confirmed swing highs
        active_lows:  List[float] = []

        # M15 boundary timestamp → (high_levels, low_levels)
        m15_active: Dict = {}
        sw_idx = 0
        n_sw   = len(confirmed_swings)

        for m15_ts in m15_times:
            while sw_idx < n_sw and confirmed_swings[sw_idx][0] <= m15_ts:
                _, price, is_h = confirmed_swings[sw_idx]
                if is_h:
                    active_highs.append(price)
                    if len(active_highs) > max_sw:
                        active_highs = active_highs[-max_sw:]
                else:
                    active_lows.append(price)
                    if len(active_lows) > max_sw:
                        active_lows = active_lows[-max_sw:]
                sw_idx += 1
            m15_active[m15_ts] = (list(active_highs), list(active_lows))

        # ── Compute daily PDH/PDL and Asian levels (ET hours for session logic) ─
        # Use the original ET index for date/hour logic (session times are ET-based)
        m1_dates   = np.array(idx.date)     # ET dates
        m1_hours   = idx.hour               # ET hours (for Asian session 19:00-02:00 ET)
        unique_dates = sorted(set(m1_dates))
        date_pos_map: Dict = defaultdict(list)
        for pi in range(len(df_m1)):
            date_pos_map[m1_dates[pi]].append(pi)

        m1_high = df_m1["high"].to_numpy()
        m1_low  = df_m1["low"].to_numpy()
        date_set = set(unique_dates)

        daily_extra: Dict = {}   # date → list of (price, label, is_high)
        for trade_date in unique_dates:
            prev_date = None
            for delta in range(1, 8):
                cand = trade_date - _dt.timedelta(days=delta)
                if cand in date_set:
                    prev_date = cand; break
            if prev_date is None:
                daily_extra[trade_date] = []
                continue

            prev_pos  = date_pos_map[prev_date]
            today_pos = date_pos_map[trade_date]
            levels = []

            # PDH / PDL
            pdh = float(m1_high[prev_pos].max())
            pdl = float(m1_low[prev_pos].min())
            levels.append((pdh, "PDH", True))
            levels.append((pdl, "PDL", False))

            # Asian (prev evening ≥19:00 + today pre-session <02:00 ET)
            prev_eve  = [p for p in prev_pos  if m1_hours[p] >= 19]
            today_pre = [p for p in today_pos if m1_hours[p] < 2]
            asian_pos = prev_eve + today_pre
            if asian_pos:
                levels.append((float(m1_high[asian_pos].max()), "ASH", True))
                levels.append((float(m1_low[asian_pos].min()),  "ASL", False))

            daily_extra[trade_date] = levels

        # ── Map M15 boundary → M1 bar indices (use UTC to avoid DST issues) ──
        m1_floor_m15 = idx_utc.floor("15min")

        result: Dict[int, List] = {}
        for pi in range(len(df_m1)):
            m15_floor = m1_floor_m15[pi]
            m15_entry = m15_active.get(m15_floor)
            if m15_entry is None:
                # Use the most recent M15 boundary before this bar
                candidates = [t for t in m15_active if t <= m15_floor]
                if not candidates:
                    continue
                m15_entry = m15_active[max(candidates)]

            h_lvls, l_lvls = m15_entry
            all_levels: List[Tuple[float, str, bool]] = [
                (p, "SWH", True)  for p in h_lvls
            ] + [
                (p, "SWL", False) for p in l_lvls
            ] + daily_extra.get(m1_dates[pi], [])

            if all_levels:
                result[pi] = all_levels

        return result

    # ------------------------------------------------------------------ #
    # H4 trend (soft filter)                                              #
    # ------------------------------------------------------------------ #

    def _compute_h4_trend(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns int array aligned to df.index:
          +1 = H4 EMA slope positive (bullish bias)
          -1 = negative (bearish)
           0 = neutral / not enough data
        """
        h4_close = df["close"].resample("4h").last().dropna()
        if len(h4_close) < self._h4_ema_period + self._h4_slope_bars + 2:
            return np.zeros(len(df), dtype=int)

        ema = h4_close.ewm(span=self._h4_ema_period, adjust=False).mean()
        slope = ema - ema.shift(self._h4_slope_bars)

        h4_trend = pd.Series(0, index=h4_close.index, dtype=int)
        h4_trend[slope >  0] =  1
        h4_trend[slope <  0] = -1

        return h4_trend.reindex(df.index, method="ffill").fillna(0).astype(int).to_numpy()

    # ------------------------------------------------------------------ #
    # H1 ATR regime (hard gate)                                           #
    # ------------------------------------------------------------------ #

    def _compute_h1_regime(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns bool array aligned to df.index.
        True  → H1 ATR is above the {h1_percentile}th pct of the last {h1_lookback} H1 bars.
        """
        h1 = df.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna(subset=["open"])

        h1_atr_series = ta.volatility.AverageTrueRange(
            high=h1["high"], low=h1["low"], close=h1["close"],
            window=self._h1_atr_period, fillna=False,
        ).average_true_range()

        atr_vals = h1_atr_series.to_numpy()
        ok_arr   = np.ones(len(h1_atr_series), dtype=bool)
        lb       = self._h1_lookback
        pct      = self._h1_percentile

        for i in range(lb, len(atr_vals)):
            if np.isnan(atr_vals[i]):
                continue
            prior = atr_vals[max(0, i - lb):i]
            prior = prior[~np.isnan(prior)]
            if len(prior) < 10:
                continue
            threshold  = float(np.percentile(prior, pct))
            ok_arr[i]  = atr_vals[i] >= threshold

        h1_regime = pd.Series(ok_arr, index=h1_atr_series.index)
        return h1_regime.reindex(df.index, method="ffill").fillna(True).astype(bool).to_numpy()
