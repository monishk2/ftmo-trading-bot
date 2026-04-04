"""
Strategy 1: London Open Breakout
=================================

WHY IT WORKS
------------
During the Asian session (00:00–02:45 US/Eastern), EUR/USD and GBP/USD trade in
a tight consolidation range because institutional order flow is sparse.  At
03:00 Eastern — the London open — European banks begin executing accumulated
client orders.  This mechanically forces price to break out of the Asian range.

The edge is *structural*, not statistical:
  - Banks MUST execute client orders at the open regardless of price direction.
  - High-volume institutional flow produces a directional impulse strong enough
    to sustain a 2:1 risk-reward trade.
  - The pattern cannot be arbitraged away because the underlying cause
    (mandatory order execution) cannot change.

EXACT RULES
-----------
1. Asian Range       — High/low of every 15-min candle whose bar OPEN falls
                       inside [asian_range_start, asian_range_end) Eastern.
2. Range filter      — Skip day if range_pips < min_asian_range_pips or
                       range_pips > max_asian_range_pips.
3. Entry window      — Only signal candles opening inside
                       [entry_window_start, entry_window_end) Eastern.
4. Long signal       — Bar CLOSE > asian_high + entry_buffer_pips × pip_size
5. Short signal      — Bar CLOSE < asian_low  - entry_buffer_pips × pip_size
6. SL (long)         — asian_low  - entry_buffer_pips × pip_size
   SL (short)        — asian_high + entry_buffer_pips × pip_size
7. TP                — entry_price ± sl_distance × risk_reward_ratio
                       (TP stored as absolute price; Backtester uses it directly)
8. Max 1 trade/day   — Once a signal fires, ignore remaining bars that day.
9. No Fridays        — Skip entire trading day if no_friday_trading is True.
10. Time stop        — Any open position at time_stop_hour is force-closed by
                       the Backtester (we write the time_stop column).

SIGNAL COLUMN CONTRACT (for Backtester)
----------------------------------------
  signal     int    1 = long, -1 = short, 0 = no trade
  sl_price   float  absolute SL price (NaN if signal == 0)
  tp_price   float  absolute TP price (NaN if signal == 0)
  lot_size   float  NaN → Backtester auto-sizes from risk_per_trade_pct
  time_stop  object pd.Timestamp (ET, tz-aware) when position must close;
                    None if no open position expected on this bar

  Diagnostic columns (always populated, useful for analysis):
  asian_high        float  NaN if no Asian range computed for this day
  asian_low         float
  asian_range_pips  float
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> time:
    """Parse 'HH:MM' into a datetime.time object."""
    h, m = map(int, s.split(":"))
    return time(h, m)


def _bar_open_time(ts: pd.Timestamp) -> time:
    """Wall-clock time of bar open (bar index = bar open for 15-min OHLCV)."""
    return ts.time()


class LondonOpenBreakout(BaseStrategy):
    """London Open Breakout strategy (15-min OHLCV, US/Eastern index)."""

    # ------------------------------------------------------------------ #
    # BaseStrategy interface                                               #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "LondonOpenBreakout"

    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        """Load parameters from strategy_params.json and instruments.json."""
        self._asian_start = _parse_time(config["asian_range_start"])   # 00:00
        self._asian_end   = _parse_time(config["asian_range_end"])     # 02:45
        self._entry_start = _parse_time(config["entry_window_start"])  # 03:00
        self._entry_end   = _parse_time(config["entry_window_end"])    # 05:30
        self._time_stop   = _parse_time(config["time_stop_hour"])      # 12:00

        self._min_range_pips  = float(config["min_asian_range_pips"])   # 20
        self._max_range_pips  = float(config["max_asian_range_pips"])   # 40
        self._entry_buffer    = float(config["entry_buffer_pips"])      # 2
        self._rr_ratio        = float(config["risk_reward_ratio"])      # 2.0
        self.risk_per_trade_pct = float(config["risk_per_trade_pct"])  # 0.75
        self._no_friday       = bool(config.get("no_friday_trading", True))

        self._pip_size = float(instrument_config["pip_size"])           # 0.0001

        logger.info(
            "%s setup: asian=%s–%s entry=%s–%s range=%.0f–%.0f pips rr=%.1f",
            self.name,
            self._asian_start, self._asian_end,
            self._entry_start, self._entry_end,
            self._min_range_pips, self._max_range_pips,
            self._rr_ratio,
        )

    # ------------------------------------------------------------------ #
    # Main signal generation                                               #
    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df :
            OHLCV DataFrame with **US/Eastern** tz-aware DatetimeIndex.
            (The Backtester guarantees this conversion before calling.)

        Returns
        -------
        pd.DataFrame
            Input df with signal columns appended.
        """
        df = self._init_signal_columns(df)

        # Diagnostic columns — populated regardless of trade activity
        df["asian_high"]        = np.nan
        df["asian_low"]         = np.nan
        df["asian_range_pips"]  = np.nan

        # Group by calendar date in Eastern time (the index is already Eastern)
        for date, day_df in df.groupby(df.index.date):
            self._process_day(df, day_df, date)

        return df

    # ------------------------------------------------------------------ #
    # Per-day logic                                                        #
    # ------------------------------------------------------------------ #

    def _process_day(
        self,
        full_df: pd.DataFrame,
        day_df: pd.DataFrame,
        date,
    ) -> None:
        """Compute Asian range + scan entry window for one trading day."""

        # Rule 9: skip Fridays
        # day_df.index[0] is a tz-aware Timestamp — .weekday() == 4 is Friday
        if self._no_friday and day_df.index[0].weekday() == 4:
            logger.debug("Skipping Friday %s", date)
            return

        # ── Step 1: Build Asian range ──────────────────────────────────
        asian_bars = day_df[day_df.index.map(
            lambda ts: self._asian_start <= _bar_open_time(ts) <= self._asian_end
        )]

        if asian_bars.empty:
            logger.debug("No Asian bars on %s — skipping", date)
            return

        asian_high = float(asian_bars["high"].max())
        asian_low  = float(asian_bars["low"].min())
        range_pips = (asian_high - asian_low) / self._pip_size

        # Write diagnostics onto every bar of this day
        full_df.loc[day_df.index, "asian_high"]       = asian_high
        full_df.loc[day_df.index, "asian_low"]        = asian_low
        full_df.loc[day_df.index, "asian_range_pips"] = round(range_pips, 1)

        # ── Step 2: Range filter ───────────────────────────────────────
        if range_pips < self._min_range_pips:
            logger.debug(
                "%s Asian range %.1f pips < min %.0f — skip",
                date, range_pips, self._min_range_pips,
            )
            return
        if range_pips > self._max_range_pips:
            logger.debug(
                "%s Asian range %.1f pips > max %.0f — skip",
                date, range_pips, self._max_range_pips,
            )
            return

        # ── Step 3: Pre-compute breakout levels ───────────────────────
        buffer_price  = self._entry_buffer * self._pip_size
        long_trigger  = asian_high + buffer_price   # close must exceed this
        short_trigger = asian_low  - buffer_price   # close must fall below this

        # SL levels (absolute prices)
        long_sl  = asian_low  - buffer_price
        short_sl = asian_high + buffer_price

        # ── Step 4: Compute time-stop timestamp for this day ──────────
        # time_stop_hour is "12:00" Eastern — build a tz-aware Timestamp
        ts_stop = pd.Timestamp(
            year=day_df.index[0].year,
            month=day_df.index[0].month,
            day=day_df.index[0].day,
            hour=self._time_stop.hour,
            minute=self._time_stop.minute,
            tz=day_df.index[0].tzinfo,
        )

        # ── Step 5: Scan entry window (one signal max) ─────────────────
        entry_bars = day_df[day_df.index.map(
            lambda ts: self._entry_start <= _bar_open_time(ts) < self._entry_end
        )]

        signal_fired = False

        for ts, row in entry_bars.iterrows():
            if signal_fired:
                break

            close = float(row["close"])

            signal:    int           = 0
            sl_price:  Optional[float] = None
            tp_price:  Optional[float] = None

            if close > long_trigger:
                signal   = 1
                sl_price = long_sl
                sl_dist  = abs(close - sl_price)
                tp_price = close + sl_dist * self._rr_ratio

            elif close < short_trigger:
                signal   = -1
                sl_price = short_sl
                sl_dist  = abs(close - sl_price)
                tp_price = close - sl_dist * self._rr_ratio

            if signal != 0:
                full_df.at[ts, "signal"]    = signal
                full_df.at[ts, "sl_price"]  = sl_price
                full_df.at[ts, "tp_price"]  = tp_price
                full_df.at[ts, "time_stop"] = ts_stop
                signal_fired = True

                logger.info(
                    "%s %s signal @ %s | close=%.5f sl=%.5f tp=%.5f | "
                    "asian_range=%.1f pips (%.5f–%.5f)",
                    self.name,
                    "LONG" if signal == 1 else "SHORT",
                    ts, close, sl_price, tp_price,
                    range_pips, asian_low, asian_high,
                )
