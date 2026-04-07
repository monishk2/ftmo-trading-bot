"""
Strategy 5: H1 Trend Following
================================

WHY IT WORKS
------------
At H1 granularity, EMA alignment captures multi-hour institutional momentum
rather than M15 noise.  The H4 EMA50 slope filter ensures we trade WITH the
dominant trend of the prior 12–20 hours, giving structural directional bias.
The pullback-to-EMA20 entry avoids chasing extended moves by requiring price
to have tested support/resistance before committing.

EXACT RULES
-----------
Session (US/Eastern): 03:00 – 16:00  (London + NY combined)

Entry (LONG):
  1. H1 EMA(ema_fast=20) > H1 EMA(ema_slow=50)  (H1 uptrend)
  2. H4 EMA50 slope > 0 over last h4_slope_lookback=3 H4 bars
     (look-ahead safe: shifted by 1 completed H4 bar)
  3. Any of the last pullback_lookback_bars=3 H1 bars had
         low ≤ H1 EMA20  (price touched or crossed below the EMA)
  4. Current H1 bar closes ABOVE H1 EMA20  (recovery confirmed)
  5. H1 ATR(14) > 50th percentile of prior 20 H1 bars  (vol filter)

Entry (SHORT): mirror all conditions with reversed direction.

Exit:
  TP:        entry ± atr_tp_mult × H1 ATR(14)    [default 2.0×]
  SL:        entry ∓ atr_sl_mult × H1 ATR(14)    [default 1.5×]
  Time stop: 16:00 ET on the entry day
  NO break-even stop.

Session rules:
  - Max 1 trade per instrument per day (long OR short, first that fires).
  - No Friday trading if no_friday_trading = True.

SIGNAL COLUMN CONTRACT (for Backtester)
----------------------------------------
  signal     int    1=long, -1=short, 0=no trade
  sl_price   float  absolute SL price
  tp_price   float  absolute TP price
  time_stop  pd.Timestamp  16:00 ET (tz-aware)
  lot_size   float  NaN → Backtester auto-sizes from risk_per_trade_pct

Implementation note
-------------------
  The strategy receives M15 data from the Backtester.  Internally it:
    1. Resamples M15 → H1 and M15 → H4.
    2. Computes all indicators on resampled bars.
    3. Emits signals on the LAST M15 bar of each qualifying H1 period.
  This ensures the Backtester enters at the H1 bar close price (= the
  close of the last M15 bar in that H1 window).
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ta

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(h, m)


class H1TrendFollowing(BaseStrategy):
    """H1 trend-following strategy with H4 directional filter."""

    @property
    def name(self) -> str:
        return "H1TrendFollowing"

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        self._session_start   = _parse_time(config.get("session_start", "03:00"))
        self._session_end     = _parse_time(config.get("session_end",   "16:00"))
        self._ema_fast        = int(config.get("ema_fast", 20))
        self._ema_slow        = int(config.get("ema_slow", 50))
        self._h4_ema_period   = int(config.get("h4_ema_period",    50))
        self._h4_slope_lb     = int(config.get("h4_slope_lookback", 3))
        self._atr_period      = int(config.get("atr_period", 14))
        self._atr_tp_mult     = float(config.get("atr_tp_mult", 2.0))
        self._atr_sl_mult     = float(config.get("atr_sl_mult", 1.5))
        self._atr_pct_lb      = int(config.get("atr_percentile_lookback", 20))
        self._atr_pct_min     = float(config.get("atr_percentile_min", 50.0))
        self._pullback_lb     = int(config.get("pullback_lookback_bars", 3))
        self.risk_per_trade_pct  = float(config["risk_per_trade_pct"])
        self._no_friday          = bool(config.get("no_friday_trading", True))
        self._h4_filter_enabled  = bool(config.get("h4_filter_enabled", True))
        self._pip_size           = float(instrument_config.get("pip_size", 0.0001))

        logger.info(
            "%s setup: EMA%d/%d  H4-EMA%d-slope%d  ATR_TP=%.1f×  ATR_SL=%.1f×  "
            "ATR_pct≥%.0f  pullback=%dbars  session=%s-%s  risk=%.1f%%",
            self.name,
            self._ema_fast, self._ema_slow,
            self._h4_ema_period, self._h4_slope_lb,
            self._atr_tp_mult, self._atr_sl_mult,
            self._atr_pct_min, self._pullback_lb,
            self._session_start, self._session_end,
            self.risk_per_trade_pct,
        )

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df :
            M15 OHLCV DataFrame with US/Eastern tz-aware DatetimeIndex.
        """
        df = self._init_signal_columns(df)

        # ── Resample M15 → H1 OHLC and M15 → H4 close ─────────────────
        h1 = df.resample("1h").agg({
            "open":  "first",
            "high":  "max",
            "low":   "min",
            "close": "last",
        }).dropna(subset=["close"])

        h4_close = df["close"].resample("4h").last().dropna()

        if len(h1) < self._ema_slow + self._atr_pct_lb + 10:
            logger.warning("%s: insufficient H1 bars (%d), skipping signals.", self.name, len(h1))
            return df

        # ── H1 indicators (vectorised) ──────────────────────────────────
        h1_ema20 = ta.trend.EMAIndicator(
            close=h1["close"], window=self._ema_fast, fillna=False
        ).ema_indicator()

        h1_ema50 = ta.trend.EMAIndicator(
            close=h1["close"], window=self._ema_slow, fillna=False
        ).ema_indicator()

        h1_atr = ta.volatility.AverageTrueRange(
            high=h1["high"], low=h1["low"], close=h1["close"],
            window=self._atr_period, fillna=False,
        ).average_true_range()

        h1_atr_pct = h1_atr.rolling(self._atr_pct_lb + 1).apply(
            lambda x: float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1) * 100.0,
            raw=True,
        )

        # ── H4 EMA50 slope (look-ahead safe) ───────────────────────────
        # H4 bar at time T (covering T → T+3h45m) is complete at T+3h45m.
        # shift(1) moves each H4 bar's value to the NEXT H4 bar's timestamp,
        # so reindex+ffill gives each H1 bar the slope of the LAST COMPLETED
        # H4 bar (i.e. at least one full H4 period old).
        h4_ema50 = ta.trend.EMAIndicator(
            close=h4_close, window=self._h4_ema_period, fillna=False
        ).ema_indicator()
        h4_slope_raw     = h4_ema50 - h4_ema50.shift(self._h4_slope_lb)
        h4_slope_shifted = h4_slope_raw.shift(1)
        h4_slope_h1      = h4_slope_shifted.reindex(h1.index, method="ffill")

        # ── Convert to numpy for fast bar-level access ──────────────────
        ema20_arr    = h1_ema20.values
        ema50_arr    = h1_ema50.values
        atr_arr      = h1_atr.values
        atr_pct_arr  = h1_atr_pct.values
        h4_sl_arr    = h4_slope_h1.values
        cl_arr       = h1["close"].values
        lo_arr       = h1["low"].values
        hi_arr       = h1["high"].values
        h1_idx       = h1.index

        # ── Iterate H1 bars, collect signals ───────────────────────────
        # Contract: max 1 trade per calendar day (first qualifying bar wins).
        h1_sigs: List[Tuple] = []  # (h1_ts, signal, sl, tp, time_stop)

        current_date    = None
        signal_fired    = False

        for pos in range(len(h1_idx)):
            ts       = h1_idx[pos]
            bar_date = ts.date()
            bar_time = ts.time()

            # ── Reset daily counters ───────────────────────────────────
            if bar_date != current_date:
                current_date = bar_date
                signal_fired = False

            # ── Friday filter ──────────────────────────────────────────
            if self._no_friday and ts.weekday() == 4:
                continue

            # ── Session filter ─────────────────────────────────────────
            if not (self._session_start <= bar_time < self._session_end):
                continue

            # ── 1 trade per day cap ────────────────────────────────────
            if signal_fired:
                continue

            # ── NaN guard ──────────────────────────────────────────────
            e20   = ema20_arr[pos]
            e50   = ema50_arr[pos]
            atr_v = atr_arr[pos]
            atr_p = atr_pct_arr[pos]
            h4_sl = h4_sl_arr[pos]
            cl    = cl_arr[pos]

            if any(np.isnan(v) for v in [e20, e50, atr_v, atr_p, h4_sl]):
                continue

            # ── Volatility filter ──────────────────────────────────────
            if atr_p < self._atr_pct_min:
                continue

            # ── Pullback window (prior bars, not including current) ────
            pb_start = max(0, pos - self._pullback_lb)

            # ── H4 slope check (bypassed when h4_filter_enabled=False) ──
            h4_bull = (not self._h4_filter_enabled) or h4_sl > 0
            h4_bear = (not self._h4_filter_enabled) or h4_sl < 0

            # ── LONG ───────────────────────────────────────────────────
            if e20 > e50 and h4_bull:
                pullback = any(
                    not np.isnan(ema20_arr[j]) and lo_arr[j] <= ema20_arr[j]
                    for j in range(pb_start, pos)
                )
                if pullback and cl > e20:
                    sl_p = cl - self._atr_sl_mult * atr_v
                    tp_p = cl + self._atr_tp_mult * atr_v
                    if sl_p < cl:
                        tstop = pd.Timestamp(
                            year=ts.year, month=ts.month, day=ts.day,
                            hour=self._session_end.hour,
                            minute=self._session_end.minute,
                            tz=ts.tzinfo,
                        )
                        h1_sigs.append((ts, 1, sl_p, tp_p, tstop))
                        signal_fired = True
                        continue

            # ── SHORT ──────────────────────────────────────────────────
            if e20 < e50 and h4_bear:
                pullback = any(
                    not np.isnan(ema20_arr[j]) and hi_arr[j] >= ema20_arr[j]
                    for j in range(pb_start, pos)
                )
                if pullback and cl < e20:
                    sl_p = cl + self._atr_sl_mult * atr_v
                    tp_p = cl - self._atr_tp_mult * atr_v
                    if sl_p > cl:
                        tstop = pd.Timestamp(
                            year=ts.year, month=ts.month, day=ts.day,
                            hour=self._session_end.hour,
                            minute=self._session_end.minute,
                            tz=ts.tzinfo,
                        )
                        h1_sigs.append((ts, -1, sl_p, tp_p, tstop))
                        signal_fired = True

        # ── Map H1 signals → last M15 bar of each H1 period ────────────
        # Uses searchsorted for O(log n) lookup per signal.
        for h1_ts, sig, sl_p, tp_p, tstop in h1_sigs:
            period_end = h1_ts + pd.Timedelta(hours=1)
            lo = int(df.index.searchsorted(h1_ts,      side="left"))
            hi = int(df.index.searchsorted(period_end, side="left"))
            if lo >= hi:
                continue
            last_m15 = df.index[hi - 1]
            df.at[last_m15, "signal"]    = sig
            df.at[last_m15, "sl_price"]  = sl_p
            df.at[last_m15, "tp_price"]  = tp_p
            df.at[last_m15, "time_stop"] = tstop

        logger.info(
            "%s: %d signals from %d H1 bars (%d H4 bars)",
            self.name, len(h1_sigs), len(h1_idx), len(h4_close),
        )
        return df
