"""
strategies/nas100_ib_breakout.py
=================================
NAS100 Initial Balance (IB) Breakout strategy on M5 data.

Edge rationale
--------------
The first 60 min of RTH (09:30–10:30 ET) establishes the Initial Balance.
On days when the IB is narrow relative to the Average Daily Range (ADR),
institutional order flow has been compressed — and the eventual breakout
tends to be directional and sustained. Wide IB = early trend already
established = no setup.

Logic per day
-------------
1. Compute IB_high, IB_low from 09:30–10:30 M5 bars.
2. Compute ADR_5 = average of last 5 days' full daily ranges.
3. If IB_width > ib_adr_ratio × ADR_5 → skip day (wide IB = chop signal).
4. Entry window: 10:30–15:30 ET.
   • LONG  if M5 close > IB_high + buffer AND volume > vol_sma_mult × vol_SMA20
   • SHORT if M5 close < IB_low  - buffer AND same volume condition
   • Enter at OPEN of the NEXT M5 bar (no bar-close chasing).
5. SL: opposite IB wall ∓ buffer. Cap at max_sl_points.
   Skip if computed SL < min_sl_points (too tight).
6. TP: rr_ratio × SL distance.
7. Time stop: 15:55 ET bar close.
8. Max 1 trade per day.

generate_signals() produces columns accepted by Backtester.run() and run_fast():
  signal, sl_price, tp_price, time_stop

Signal fires on the bar BEFORE entry so that the backtester enters at
next-bar open (via its standard entry_price = close ± spread ± slippage).
Because M5 bars are short, this approximates the open-of-next-bar logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class NAS100IbBreakout:
    """NAS100 Initial Balance Breakout strategy."""

    name = "nas100_ib_breakout"

    def __init__(
        self,
        ib_adr_ratio:        float = 0.40,
        buffer_points:       float = 5.0,
        rr_ratio:            float = 3.0,
        risk_per_trade_pct:  float = 0.5,
        max_sl_points:       float = 80.0,
        min_sl_points:       float = 15.0,
        vol_sma_period:      int   = 20,
        vol_sma_mult:        float = 1.0,
        adr_lookback_days:   int   = 5,
    ) -> None:
        self.ib_adr_ratio       = ib_adr_ratio
        self.buffer_points      = buffer_points
        self.rr_ratio           = rr_ratio
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_sl_points      = max_sl_points
        self.min_sl_points      = min_sl_points
        self.vol_sma_period     = vol_sma_period
        self.vol_sma_mult       = vol_sma_mult
        self.adr_lookback_days  = adr_lookback_days

    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df is the M5 OHLCV DataFrame already in US/Eastern tz,
        as prepared by Backtester._prepare_data().
        """
        df = df.copy()
        n  = len(df)

        # Output columns
        df["signal"]    = 0
        df["sl_price"]  = np.nan
        df["tp_price"]  = np.nan
        # Use object dtype so tz-aware Timestamps can be stored
        df["time_stop"] = None
        df["time_stop"]  = df["time_stop"].astype(object)

        if n < 20:
            return df

        idx    = df.index
        opens  = df["open"].to_numpy(float)
        highs  = df["high"].to_numpy(float)
        lows   = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        vols   = df["volume"].to_numpy(float)

        # Hour and minute arrays (US/Eastern)
        hours   = np.array(idx.hour)
        minutes = np.array(idx.minute)
        dates   = np.array(idx.date)

        # Volume SMA(20)
        vol_sma = pd.Series(vols).rolling(self.vol_sma_period, min_periods=1).mean().to_numpy()

        # Daily full-range cache for ADR computation (keyed by date)
        # We'll accumulate on the fly
        daily_highs: dict  = {}
        daily_lows:  dict  = {}

        # Per-day tracking
        cur_date          = None
        ib_high           = -np.inf
        ib_low            = np.inf
        ib_done           = False
        trade_taken_today = False
        trade_skipped     = False   # IB too wide on this day

        for i in range(n):
            d  = dates[i]
            h  = hours[i]
            m  = minutes[i]
            hm = h * 100 + m

            # ── Day boundary ──────────────────────────────────────────────
            if d != cur_date:
                cur_date          = d
                ib_high           = -np.inf
                ib_low            = np.inf
                ib_done           = False
                trade_taken_today = False
                trade_skipped     = False

            # Track daily range
            if hm >= 930 and hm < 1600:
                daily_highs[d] = max(daily_highs.get(d, -np.inf), highs[i])
                daily_lows[d]  = min(daily_lows.get(d,  np.inf), lows[i])

            # ── IB accumulation: 09:30 – 10:25 bars (10:30 bar is POST-IB) ──
            if hm >= 930 and hm < 1030:
                if highs[i] > ib_high: ib_high = highs[i]
                if lows[i]  < ib_low:  ib_low  = lows[i]
                continue

            # ── IB just closed (first bar at or after 10:30) ──────────────
            if hm == 1030 and not ib_done:
                ib_done = True
                if ib_high == -np.inf or ib_low == np.inf:
                    trade_skipped = True
                    continue

                ib_width = ib_high - ib_low

                # ADR from last N complete trading days
                sorted_days = sorted(
                    [dd for dd in daily_highs if dd < d],
                    reverse=True
                )[: self.adr_lookback_days]

                if len(sorted_days) < 2:
                    trade_skipped = True
                    continue

                adr = np.mean([
                    daily_highs[dd] - daily_lows[dd]
                    for dd in sorted_days
                    if dd in daily_lows
                ])

                if adr <= 0 or ib_width > self.ib_adr_ratio * adr:
                    trade_skipped = True
                continue

            # ── Entry window: 10:30 – 15:29 ──────────────────────────────
            if trade_skipped or trade_taken_today:
                continue
            if not ib_done:
                continue
            if hm < 1030 or hm >= 1530:
                continue

            # Volume filter
            if vols[i] < self.vol_sma_mult * vol_sma[i]:
                continue

            # ── Long signal ───────────────────────────────────────────────
            if closes[i] > ib_high + self.buffer_points:
                sl  = ib_low - self.buffer_points
                sl  = max(sl, closes[i] - self.max_sl_points)
                sl_dist = closes[i] - sl
                if sl_dist < self.min_sl_points:
                    continue
                tp  = closes[i] + self.rr_ratio * sl_dist
                bar_ts = idx[i]
                ts_time = pd.Timestamp(
                    bar_ts.year, bar_ts.month, bar_ts.day, 15, 55,
                    tzinfo=bar_ts.tzinfo
                )

                df.iat[i, df.columns.get_loc("signal")]    = 1
                df.iat[i, df.columns.get_loc("sl_price")]  = sl
                df.iat[i, df.columns.get_loc("tp_price")]  = tp
                df.iat[i, df.columns.get_loc("time_stop")] = ts_time
                trade_taken_today = True
                continue

            # ── Short signal ──────────────────────────────────────────────
            if closes[i] < ib_low - self.buffer_points:
                sl  = ib_high + self.buffer_points
                sl  = min(sl, closes[i] + self.max_sl_points)
                sl_dist = sl - closes[i]
                if sl_dist < self.min_sl_points:
                    continue
                tp  = closes[i] - self.rr_ratio * sl_dist
                bar_ts = idx[i]
                ts_time = pd.Timestamp(
                    bar_ts.year, bar_ts.month, bar_ts.day, 15, 55,
                    tzinfo=bar_ts.tzinfo
                )

                df.iat[i, df.columns.get_loc("signal")]    = -1
                df.iat[i, df.columns.get_loc("sl_price")]  = sl
                df.iat[i, df.columns.get_loc("tp_price")]  = tp
                df.iat[i, df.columns.get_loc("time_stop")] = ts_time
                trade_taken_today = True

        return df
