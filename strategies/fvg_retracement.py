"""
Strategy 2: Fair Value Gap (FVG) Retracement
=============================================

WHY IT WORKS
------------
When a large institutional order fires, price moves so fast that normal two-sided
order matching is skipped, leaving a "Fair Value Gap" — a price zone where no
real volume traded.  Other institutions (who missed the move or want to hedge)
place limit orders inside that gap expecting a fill at fair value.  The
resulting pull-back into the FVG is mechanical, not statistical.

The edge is structural:
  - Large orders physically cannot fill entirely in a single candle; the gap is
    the residue of forced, incomplete institutional execution.
  - Gaps act as liquidity pools: limit orders cluster there naturally.
  - This dynamic cannot be arbitraged away because it emerges from market
    microstructure, not a learnable price pattern.

EXACT RULES
-----------
1.  FVG DETECTION (15-min chart, 3-candle pattern):
      Bullish FVG: candle[i-2].high < candle[i].low
        → zone = [candle[i-2].high, candle[i].low]
      Bearish FVG: candle[i-2].low  > candle[i].high
        → zone = [candle[i].high,   candle[i-2].low]
      (Candle i is the third candle; candle i-1 is the impulse body.)

2.  SCAN WINDOW: Only detect FVGs whose *formation bar* (candle i) opens inside
      [fvg_scan_start, fvg_scan_end) Eastern (default 03:00–06:00).

3.  SIZE FILTER: gap_pips must be in [min_fvg_size_pips, max_fvg_size_pips].

4.  TREND FILTER (200 EMA on 1H):
      Resample 15-min data to 1H (OHLC forward-fill), compute EMA(200).
      Map each 15-min bar to the most recent 1H EMA value.
      Accept bullish FVG only if close > ema_200.
      Accept bearish FVG only if close < ema_200.

5.  ENTRY: Limit order at fvg_entry_level (0.5 = midpoint) of the FVG zone.
      Signal fires on the bar where price retraces INTO the zone during the
      entry window [entry_window_start, entry_window_end) Eastern.
      "Into the zone" = bar's low touches or enters the zone (bullish)
                        bar's high touches or enters the zone (bearish)

6.  STOP LOSS:
      Bullish: SL = fvg_zone_bottom - entry_buffer_pips × pip_size
      Bearish: SL = fvg_zone_top    + entry_buffer_pips × pip_size

7.  TAKE PROFIT: |entry - SL| × risk_reward_ratio, projected in trade direction.

8.  CANCEL (candle-based, primary): Cancel unfilled limit if price has not
      entered the zone within max_candles_until_cancel bars after the FVG
      formation bar.  (default 8 candles = 2 hours on 15-min data)

9.  CANCEL (time-based, backstop): If the limit is still unfilled at
      cancel_unfilled_hour Eastern, cancel regardless.

10. TIME STOP: Write time_stop = time_stop_hour Eastern on the signal bar.
      Backtester force-closes any open position at that time.

11. MAX 1 FVG TRADE PER DAY.

12. If multiple valid FVGs detected in the scan window, use the MOST RECENT
      one (highest candle index → closest to NY open).

13. TIME DECAY (Rule 13): FVGs are most valid immediately after formation.
      If price hasn't reached the limit within max_candles_until_cancel bars,
      institutional momentum has faded — cancel regardless of hour.

14. SKIP FRIDAYS when no_friday_trading is True.

SIGNAL COLUMN CONTRACT (Backtester)
-------------------------------------
  signal     int    1 = long, -1 = short, 0 = no trade
  sl_price   float  NaN if signal == 0
  tp_price   float  NaN if signal == 0
  lot_size   float  NaN → Backtester auto-sizes
  time_stop  object pd.Timestamp (Eastern) | None

  Diagnostic columns (always written for analysis):
  fvg_zone_top      float  NaN if no FVG active
  fvg_zone_bottom   float
  fvg_direction     int    1=bullish, -1=bearish, 0=none
  fvg_size_pips     float
  fvg_ema200        float  1H EMA200 value mapped to 15-min bar
  fvg_candles_open  int    bars since FVG formed (0 = formation bar)

LIMIT ORDER SIMULATION (Backtester note)
-----------------------------------------
Because the Backtester enters at bar close when signal==1|-1, the FVG strategy
emits the signal on the first bar where the entry zone is *touched*, which is
the exact bar the limit would have been filled.  The entry_price used by the
Backtester will therefore be close + spread/slippage on that bar.
sl_price / tp_price are written as absolute prices.
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(h, m)


def _bar_time(ts: pd.Timestamp) -> time:
    return ts.time()


# ---------------------------------------------------------------------------
# Internal FVG record
# ---------------------------------------------------------------------------

class _FVG(NamedTuple):
    direction:    int     # 1 = bullish, -1 = bearish
    zone_top:     float
    zone_bottom:  float
    size_pips:    float
    formation_ts: pd.Timestamp   # timestamp of candle[i] (third candle)
    formation_iloc: int          # position in the full DataFrame


class FVGRetracement(BaseStrategy):
    """FVG Retracement strategy (15-min OHLCV, US/Eastern DatetimeIndex)."""

    @property
    def name(self) -> str:
        return "FVGRetracement"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        self._scan_start  = _parse_time(config["fvg_scan_start"])       # 03:00
        self._scan_end    = _parse_time(config["fvg_scan_end"])         # 06:00
        self._entry_start = _parse_time(config["entry_window_start"])   # 09:30
        self._entry_end   = _parse_time(config["entry_window_end"])     # 12:00
        self._cancel_hour = _parse_time(config["cancel_unfilled_hour"]) # 12:00
        self._time_stop   = _parse_time(config["time_stop_hour"])       # 16:00

        self._min_fvg_pips   = float(config["min_fvg_size_pips"])       # 5
        self._max_fvg_pips   = float(config["max_fvg_size_pips"])       # 30
        self._entry_level    = float(config["fvg_entry_level"])         # 0.5
        self._entry_buffer   = float(config["entry_buffer_pips"])       # 3
        self._rr_ratio       = float(config["risk_reward_ratio"])       # 2.0
        self._max_candles    = int(config["max_candles_until_cancel"])  # 8
        self._ema_period     = int(config["ema_period"])                # 200
        self.risk_per_trade_pct = float(config["risk_per_trade_pct"])  # 0.5
        self._no_friday      = bool(config.get("no_friday_trading", True))

        self._pip_size = float(instrument_config["pip_size"])

        logger.info(
            "%s setup: scan=%s–%s entry=%s–%s fvg_pips=%.0f–%.0f cancel=%d candles rr=%.1f",
            self.name,
            self._scan_start, self._scan_end,
            self._entry_start, self._entry_end,
            self._min_fvg_pips, self._max_fvg_pips,
            self._max_candles, self._rr_ratio,
        )

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df must have a US/Eastern tz-aware DatetimeIndex (Backtester guarantees this).
        """
        df = self._init_signal_columns(df)

        # Diagnostic columns
        df["fvg_zone_top"]     = np.nan
        df["fvg_zone_bottom"]  = np.nan
        df["fvg_direction"]    = 0
        df["fvg_size_pips"]    = np.nan
        df["fvg_ema200"]       = np.nan
        df["fvg_candles_open"] = 0

        # 1. Compute 1H EMA(200) and map to 15-min bars
        ema_map = self._compute_ema200(df)
        df["fvg_ema200"] = df.index.map(lambda ts: ema_map.get(ts, np.nan))

        # 2. Build a timestamp→integer-position map (avoids get_loc returning
        #    slices when the index has duplicate timestamps).
        ts_to_pos: dict = {}
        for i, ts in enumerate(df.index):
            if ts not in ts_to_pos:          # keep first occurrence
                ts_to_pos[ts] = i

        # 3. Group by calendar day (Eastern)
        for date, day_df in df.groupby(df.index.date):
            self._process_day(df, day_df, date, ema_map, ts_to_pos)

        return df

    # ------------------------------------------------------------------
    # EMA computation
    # ------------------------------------------------------------------

    def _compute_ema200(self, df: pd.DataFrame) -> dict:
        """
        Resample 15-min OHLCV to 1H, compute EMA(200) on close,
        then return a dict mapping each 15-min bar's timestamp to the
        most recent 1H EMA value.

        Forward-fill: the EMA value at 09:00 ET applies to all 15-min
        bars from 09:00 to 09:45.
        """
        # Work in UTC to avoid DST-ambiguity errors during .floor()
        df_utc = df.copy()
        df_utc.index = df_utc.index.tz_convert("UTC")

        hourly = df_utc["close"].resample("1h").last().dropna()
        if len(hourly) < self._ema_period:
            # Not enough history — return empty mapping (all NaN)
            logger.warning(
                "%s: Only %d hourly bars available; need %d for EMA%d. "
                "Trend filter will be skipped (NaN EMA).",
                self.name, len(hourly), self._ema_period, self._ema_period,
            )
            return {}

        ema_series = hourly.ewm(span=self._ema_period, adjust=False).mean()

        # Map each original (ET) timestamp to the nearest preceding 1H EMA value.
        # Floor and lookup both happen in UTC — no DST ambiguity possible.
        ema_map: dict = {}
        for ts in df.index:
            ts_utc = ts.tz_convert("UTC").floor("1h")
            if ts_utc in ema_series.index:
                ema_map[ts] = ema_series[ts_utc]
            else:
                # Search backward for the last available hourly value
                candidates = ema_series.index[ema_series.index <= ts_utc]
                if len(candidates):
                    ema_map[ts] = float(ema_series[candidates[-1]])
        return ema_map

    # ------------------------------------------------------------------
    # Per-day logic
    # ------------------------------------------------------------------

    def _process_day(
        self,
        full_df: pd.DataFrame,
        day_df: pd.DataFrame,
        date,
        ema_map: dict,
        ts_to_pos: dict,
    ) -> None:
        # Skip Fridays
        if self._no_friday and day_df.index[0].weekday() == 4:
            return

        # Integer positions in full_df for each bar on this day.
        # Uses ts_to_pos (precomputed, always returns int) to avoid the
        # ambiguous-truth-value error that pd.Index.get_loc() triggers when
        # the index contains duplicate timestamps.
        day_positions = [ts_to_pos[ts] for ts in day_df.index if ts in ts_to_pos]
        if not day_positions:
            return

        # ── 1. Detect all valid FVGs in the scan window ──────────────
        fvgs: List[_FVG] = []

        for pos, ts in zip(day_positions, day_df.index):
            if not (self._scan_start <= _bar_time(ts) < self._scan_end):
                continue
            if pos < 2:
                continue  # need candle[i-2]

            fvg = self._detect_fvg(full_df, pos, ts, ema_map)
            if fvg is not None:
                fvgs.append(fvg)

        if not fvgs:
            return

        # ── 2. Use the MOST RECENT valid FVG (highest formation iloc) ─
        fvg = max(fvgs, key=lambda f: f.formation_iloc)

        logger.debug(
            "%s %s FVG detected @ %s | zone=[%.5f, %.5f] %.1f pips",
            self.name,
            "BULL" if fvg.direction == 1 else "BEAR",
            fvg.formation_ts, fvg.zone_bottom, fvg.zone_top, fvg.size_pips,
        )

        # Write FVG diagnostics for the whole day
        full_df.loc[day_df.index, "fvg_zone_top"]    = fvg.zone_top
        full_df.loc[day_df.index, "fvg_zone_bottom"] = fvg.zone_bottom
        full_df.loc[day_df.index, "fvg_direction"]   = fvg.direction
        full_df.loc[day_df.index, "fvg_size_pips"]   = fvg.size_pips

        # ── 3. Scan entry window for zone touch ───────────────────────
        # Pre-build limit price and stops
        entry_price = fvg.zone_bottom + self._entry_level * (fvg.zone_top - fvg.zone_bottom)
        if fvg.direction == 1:  # bullish
            sl_price = fvg.zone_bottom - self._entry_buffer * self._pip_size
        else:                   # bearish
            sl_price = fvg.zone_top + self._entry_buffer * self._pip_size

        sl_dist  = abs(entry_price - sl_price)
        tp_price = (entry_price + sl_dist * self._rr_ratio if fvg.direction == 1
                    else entry_price - sl_dist * self._rr_ratio)

        # Time-stop timestamp for this day
        ts_stop = pd.Timestamp(
            year=day_df.index[0].year,
            month=day_df.index[0].month,
            day=day_df.index[0].day,
            hour=self._time_stop.hour,
            minute=self._time_stop.minute,
            tz=day_df.index[0].tzinfo,
        )

        # Cancel-hour timestamp (hard backstop)
        ts_cancel = pd.Timestamp(
            year=day_df.index[0].year,
            month=day_df.index[0].month,
            day=day_df.index[0].day,
            hour=self._cancel_hour.hour,
            minute=self._cancel_hour.minute,
            tz=day_df.index[0].tzinfo,
        )

        candles_since_fvg = 0
        signal_fired = False

        for pos, ts in zip(day_positions, day_df.index):
            # Count candles since formation (including formation bar)
            if pos >= fvg.formation_iloc:
                candles_since_fvg = pos - fvg.formation_iloc
            full_df.at[ts, "fvg_candles_open"] = candles_since_fvg

            if signal_fired:
                continue

            # Must be in entry window
            if not (self._entry_start <= _bar_time(ts) < self._entry_end):
                continue

            # Rule 8/13: cancel if max_candles_until_cancel exceeded
            if candles_since_fvg > self._max_candles:
                logger.debug("%s FVG cancelled (candle decay) @ %s", self.name, ts)
                break

            # Rule 9: cancel at hard backstop hour
            if ts >= ts_cancel:
                logger.debug("%s FVG cancelled (hour backstop) @ %s", self.name, ts)
                break

            row = full_df.iloc[pos]

            # Check if price has retraced into the FVG zone
            touched = False
            if fvg.direction == 1:
                # Bullish: bar's low touches or enters the zone
                touched = row["low"] <= fvg.zone_top and row["high"] >= fvg.zone_bottom
            else:
                # Bearish: bar's high touches or enters the zone
                touched = row["high"] >= fvg.zone_bottom and row["low"] <= fvg.zone_top

            if touched:
                full_df.at[ts, "signal"]    = fvg.direction
                full_df.at[ts, "sl_price"]  = sl_price
                full_df.at[ts, "tp_price"]  = tp_price
                full_df.at[ts, "time_stop"] = ts_stop
                signal_fired = True

                logger.info(
                    "%s %s entry @ %s | limit=%.5f sl=%.5f tp=%.5f | "
                    "fvg=[%.5f–%.5f] %.1f pips | candles_open=%d",
                    self.name,
                    "LONG" if fvg.direction == 1 else "SHORT",
                    ts, entry_price, sl_price, tp_price,
                    fvg.zone_bottom, fvg.zone_top, fvg.size_pips,
                    candles_since_fvg,
                )

    # ------------------------------------------------------------------
    # FVG detection (single candle position)
    # ------------------------------------------------------------------

    def _detect_fvg(
        self,
        df: pd.DataFrame,
        pos: int,
        ts: pd.Timestamp,
        ema_map: dict,
    ) -> Optional[_FVG]:
        """
        Check whether the 3-candle pattern ending at `pos` forms a valid FVG.

        Returns an _FVG if valid, else None.
        """
        c0 = df.iloc[pos - 2]   # candle 1 (oldest)
        # c1 = df.iloc[pos - 1]  # candle 2 (impulse body) — not needed directly
        c2 = df.iloc[pos]       # candle 3 (newest = formation bar)

        # EMA trend filter — NaN means insufficient history, skip filter
        ema_val = ema_map.get(ts)
        if ema_val is not None and np.isnan(ema_val):
            ema_val = None

        direction: Optional[int] = None
        zone_top = zone_bottom = 0.0

        # Bullish FVG
        if c0["high"] < c2["low"]:
            zone_bottom = float(c0["high"])
            zone_top    = float(c2["low"])
            direction   = 1

        # Bearish FVG
        elif c0["low"] > c2["high"]:
            zone_top    = float(c0["low"])
            zone_bottom = float(c2["high"])
            direction   = -1

        if direction is None:
            return None

        size_pips = (zone_top - zone_bottom) / self._pip_size

        # Size filter
        if size_pips < self._min_fvg_pips or size_pips > self._max_fvg_pips:
            return None

        # Trend filter (skip if EMA not available)
        if ema_val is not None:
            close = float(c2["close"])
            if direction == 1 and close <= ema_val:
                logger.debug("Bullish FVG @ %s rejected: close %.5f <= EMA %.5f", ts, close, ema_val)
                return None
            if direction == -1 and close >= ema_val:
                logger.debug("Bearish FVG @ %s rejected: close %.5f >= EMA %.5f", ts, close, ema_val)
                return None

        return _FVG(
            direction=direction,
            zone_top=zone_top,
            zone_bottom=zone_bottom,
            size_pips=size_pips,
            formation_ts=ts,
            formation_iloc=pos,
        )
