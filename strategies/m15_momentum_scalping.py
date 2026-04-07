"""
Strategy 4: M15 Momentum Scalping
===================================

WHY IT WORKS
------------
During active London and NY sessions, institutional order flow creates
sustained directional momentum lasting 1-3 hours.  EMA alignment
(fast > slow > trend) identifies this structural momentum, while the
pullback-to-EMA21 condition ensures we enter at value rather than
chasing extended moves.  ATR-based sizing dynamically adjusts to
volatility, keeping risk constant across changing market conditions.

EXACT RULES
-----------
Sessions (US/Eastern):
  London: 07:00 – 12:00  (European institutional flow)
  NY:     12:00 – 15:00  (pure US session, post-London)
  Note:  09:30–12:00 overlap is treated as London session.

Entry (LONG):
  1. EMA(fast) > EMA(slow) > EMA(50) on current bar (trend alignment)
  2. RSI(14) ∈ [rsi_long_min, rsi_long_max] (momentum without extreme)
  3. Any of the last `pullback_lookback_bars` bars had
         low ≤ EMA(slow) + pullback_atr_mult × ATR(14)
     (price pulled back toward EMA_slow support)
  4. Current bar closes > EMA(fast)  (pullback recovery confirmed)
  5. ATR(14) > atr_percentile_min-th percentile of prior
     atr_pct_lookback bars  (avoid dead/flat markets)
  6. Regime gate: daily ATR > 60th percentile AND H1 ADX > 25

Entry (SHORT): all conditions mirrored:
  EMA(fast) < EMA(slow) < EMA(50),  RSI ∈ [rsi_short_min, rsi_short_max],
  any recent bar had high ≥ EMA(slow) - pullback_atr_mult × ATR,
  current close < EMA(fast).

SL / TP:
  SL = entry ∓ atr_sl_mult × ATR(14)
  TP = entry ± atr_tp_mult × ATR(14)   (default 1:2 R:R)

Session rules:
  - Max 1 signal per session per day (no re-entry after stop hit).
  - time_stop = end of the session the trade was entered in.
  - No trading on Fridays if no_friday_trading = True.

SIGNAL COLUMN CONTRACT (for Backtester)
----------------------------------------
  signal     int    1=long, -1=short, 0=no trade
  sl_price   float  absolute SL price
  tp_price   float  absolute TP price
  time_stop  pd.Timestamp  session end (ET, tz-aware)
  lot_size   float  NaN → Backtester auto-sizes from risk_per_trade_pct

  Diagnostic columns:
  ema_fast_val, ema_slow_val, ema_trend_val, rsi_val, atr_val,
  atr_pct_rank, pullback_hit, session, regime_filtered
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import ta

from strategies.base_strategy import BaseStrategy
from strategies.regime_filter import _compute_atr, _compute_adx

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(h, m)


def _bar_time(ts: pd.Timestamp) -> time:
    return ts.time()


class M15MomentumScalping(BaseStrategy):
    """M15 momentum-pullback scalping strategy (London + NY sessions)."""

    @property
    def name(self) -> str:
        return "M15MomentumScalping"

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        self._london_start = _parse_time(config["london_session_start"])   # 07:00
        self._london_end   = _parse_time(config["london_session_end"])     # 12:00
        self._ny_start     = _parse_time(config["ny_session_start"])       # 12:00
        self._ny_end       = _parse_time(config["ny_session_end"])         # 15:00

        self._ema_fast     = int(config["ema_fast"])     # 9
        self._ema_slow     = int(config["ema_slow"])     # 21
        self._ema_trend    = int(config.get("ema_trend", 50))  # 50

        self._rsi_period   = int(config.get("rsi_period", 14))
        self._rsi_long_min = float(config["rsi_long_min"])    # 45
        self._rsi_long_max = float(config["rsi_long_max"])    # 65
        # Short RSI mirrors around 50
        self._rsi_short_min = float(config.get("rsi_short_min", 100.0 - self._rsi_long_max))
        self._rsi_short_max = float(config.get("rsi_short_max", 100.0 - self._rsi_long_min))

        self._atr_period   = int(config.get("atr_period", 14))
        self._atr_tp_mult  = float(config["atr_tp_mult"])     # 2.0
        self._atr_sl_mult  = float(config.get("atr_sl_mult", 1.0))

        self._pullback_mult     = float(config.get("pullback_atr_mult", 1.5))
        self._pullback_lookback = int(config.get("pullback_lookback_bars", 3))

        self._atr_pct_lookback  = int(config.get("atr_percentile_lookback", 20))
        self._atr_pct_min       = float(config.get("atr_percentile_min", 60.0))

        self.risk_per_trade_pct = float(config["risk_per_trade_pct"])
        self._no_friday         = bool(config.get("no_friday_trading", True))

        self._pip_size          = float(instrument_config["pip_size"])

        # Regime filter (same gate as London Breakout)
        self._regime_enabled  = bool(config.get("regime_filter_enabled", False))
        self._regime_atr_per  = int(config.get("regime_atr_period",     14))
        self._regime_lookback = int(config.get("regime_lookback_days",  60))
        self._regime_atr_pct  = float(config.get("regime_atr_percentile", 60.0))
        self._regime_adx_per  = int(config.get("regime_adx_period",     14))
        self._regime_adx_min  = float(config.get("regime_adx_h1_min",   25.0))

        logger.info(
            "%s setup: EMA%d/%d/%d RSI%.0f-%.0f ATR_TP=%.1f×  "
            "pullback=%.1f×(%.0fbars) regime=%s",
            self.name,
            self._ema_fast, self._ema_slow, self._ema_trend,
            self._rsi_long_min, self._rsi_long_max,
            self._atr_tp_mult, self._pullback_mult, self._pullback_lookback,
            "ON" if self._regime_enabled else "OFF",
        )

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df :
            OHLCV DataFrame with US/Eastern tz-aware DatetimeIndex.
        """
        df = self._init_signal_columns(df)

        # ── Diagnostic columns ─────────────────────────────────────────
        df["ema_fast_val"]    = np.nan
        df["ema_slow_val"]    = np.nan
        df["ema_trend_val"]   = np.nan
        df["rsi_val"]         = np.nan
        df["atr_val"]         = np.nan
        df["atr_pct_rank"]    = np.nan
        df["pullback_hit"]    = False
        df["session"]         = ""
        df["regime_filtered"] = False

        # ── Pre-compute indicators (vectorised) ────────────────────────
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]

        ema_fast  = ta.trend.EMAIndicator(close=close, window=self._ema_fast,  fillna=False).ema_indicator()
        ema_slow  = ta.trend.EMAIndicator(close=close, window=self._ema_slow,  fillna=False).ema_indicator()
        ema_trend = ta.trend.EMAIndicator(close=close, window=self._ema_trend, fillna=False).ema_indicator()
        rsi       = ta.momentum.RSIIndicator(close=close, window=self._rsi_period, fillna=False).rsi()
        atr       = ta.volatility.AverageTrueRange(
                        high=high, low=low, close=close,
                        window=self._atr_period, fillna=False
                    ).average_true_range()

        # ATR percentile rank: fraction of prior atr_pct_lookback values below current ATR
        # Rolling over ALL bars (not daily) — gives intraday volatility context
        def _rolling_pct_rank(series: pd.Series, window: int) -> pd.Series:
            rank = series.rolling(window + 1).apply(
                lambda x: float(np.sum(x[:-1] < x[-1])) / (len(x) - 1) * 100.0,
                raw=True,
            )
            return rank

        atr_pct_rank = _rolling_pct_rank(atr, self._atr_pct_lookback)

        # Write to df for diagnostics
        df["ema_fast_val"]  = ema_fast.values
        df["ema_slow_val"]  = ema_slow.values
        df["ema_trend_val"] = ema_trend.values
        df["rsi_val"]       = rsi.values
        df["atr_val"]       = atr.values
        df["atr_pct_rank"]  = atr_pct_rank.values

        # Convert to numpy for fast per-bar access
        ema_fast_arr  = ema_fast.values
        ema_slow_arr  = ema_slow.values
        ema_trend_arr = ema_trend.values
        rsi_arr       = rsi.values
        atr_arr       = atr.values
        atr_pct_arr   = atr_pct_rank.values
        close_arr     = close.values
        low_arr       = low.values
        high_arr      = high.values
        idx           = df.index  # DatetimeIndex

        # ── Regime gate (per-day) ──────────────────────────────────────
        regime_map: dict = {}
        if self._regime_enabled:
            regime_map = self._compute_regime_maps(df)

        n_regime_filtered = 0
        n_signals         = 0

        # ── Per-day processing ─────────────────────────────────────────
        for date, day_idx in self._iter_day_indices(idx):
            # Friday filter
            if self._no_friday and idx[day_idx[0]].weekday() == 4:
                continue

            # Regime gate
            if regime_map and date in regime_map and not regime_map[date]["pass"]:
                for i in day_idx:
                    df.iat[i, df.columns.get_loc("regime_filtered")] = True
                n_regime_filtered += 1
                continue

            # Session end timestamps (tz-aware)
            day_ts = idx[day_idx[0]]
            london_stop_ts = pd.Timestamp(
                year=day_ts.year, month=day_ts.month, day=day_ts.day,
                hour=self._london_end.hour, minute=self._london_end.minute,
                tz=day_ts.tzinfo,
            )
            ny_stop_ts = pd.Timestamp(
                year=day_ts.year, month=day_ts.month, day=day_ts.day,
                hour=self._ny_end.hour, minute=self._ny_end.minute,
                tz=day_ts.tzinfo,
            )

            # Partition bars into London and NY sessions
            # London: 07:00–12:00  |  NY: 12:00–15:00
            london_bars = [
                i for i in day_idx
                if self._london_start <= _bar_time(idx[i]) < self._london_end
            ]
            ny_bars = [
                i for i in day_idx
                if self._ny_start <= _bar_time(idx[i]) < self._ny_end
            ]

            # Max 1 signal per session
            for session_bars, session_stop_ts, session_label in [
                (london_bars, london_stop_ts, "london"),
                (ny_bars,     ny_stop_ts,     "ny"),
            ]:
                if not session_bars:
                    continue

                signal_fired = False
                for pos, i in enumerate(session_bars):
                    if signal_fired:
                        break

                    # Need enough history for all indicators
                    ef  = ema_fast_arr[i]
                    es  = ema_slow_arr[i]
                    et  = ema_trend_arr[i]
                    r   = rsi_arr[i]
                    atr_v = atr_arr[i]
                    atr_p = atr_pct_arr[i]
                    cl  = close_arr[i]

                    # Skip bars with NaN indicators
                    if any(np.isnan(x) for x in [ef, es, et, r, atr_v, atr_p]):
                        continue

                    # ── ATR percentile filter ──────────────────────────
                    if atr_p < self._atr_pct_min:
                        continue

                    # ── Pullback lookback window (preceding bars) ──────
                    # Look at the pullback_lookback bars BEFORE current bar
                    pb_start = max(0, i - self._pullback_lookback)
                    pb_end   = i  # exclusive

                    # ── LONG conditions ────────────────────────────────
                    if ef > es > et:
                        # Trend aligned bullish
                        if self._rsi_long_min <= r <= self._rsi_long_max:
                            # Pullback: any prior bar's low within 1.5×ATR of EMA_slow
                            pb_threshold = es + self._pullback_mult * atr_v
                            pullback_hit = any(
                                low_arr[j] <= pb_threshold
                                for j in range(pb_start, pb_end)
                                if not np.isnan(low_arr[j])
                            )
                            # Recovery: current bar closes above EMA_fast
                            if pullback_hit and cl > ef:
                                entry = cl
                                sl    = entry - self._atr_sl_mult * atr_v
                                tp    = entry + self._atr_tp_mult * atr_v

                                # SL sanity: must be below entry
                                if sl >= entry:
                                    continue

                                df.iat[i, df.columns.get_loc("signal")]    = 1
                                df.iat[i, df.columns.get_loc("sl_price")]  = sl
                                df.iat[i, df.columns.get_loc("tp_price")]  = tp
                                df.iat[i, df.columns.get_loc("time_stop")] = session_stop_ts
                                df.iat[i, df.columns.get_loc("session")]   = session_label
                                df.iat[i, df.columns.get_loc("pullback_hit")] = True
                                signal_fired = True
                                n_signals += 1
                                logger.debug(
                                    "%s LONG @ %s | entry=%.5f sl=%.5f tp=%.5f ATR=%.5f",
                                    self.name, idx[i], entry, sl, tp, atr_v,
                                )
                                break

                    # ── SHORT conditions ───────────────────────────────
                    elif ef < es < et:
                        # Trend aligned bearish
                        if self._rsi_short_min <= r <= self._rsi_short_max:
                            # Pullback: any prior bar's high within 1.5×ATR of EMA_slow (from below)
                            pb_threshold = es - self._pullback_mult * atr_v
                            pullback_hit = any(
                                high_arr[j] >= pb_threshold
                                for j in range(pb_start, pb_end)
                                if not np.isnan(high_arr[j])
                            )
                            # Recovery: current bar closes below EMA_fast
                            if pullback_hit and cl < ef:
                                entry = cl
                                sl    = entry + self._atr_sl_mult * atr_v
                                tp    = entry - self._atr_tp_mult * atr_v

                                # SL sanity: must be above entry
                                if sl <= entry:
                                    continue

                                df.iat[i, df.columns.get_loc("signal")]    = -1
                                df.iat[i, df.columns.get_loc("sl_price")]  = sl
                                df.iat[i, df.columns.get_loc("tp_price")]  = tp
                                df.iat[i, df.columns.get_loc("time_stop")] = session_stop_ts
                                df.iat[i, df.columns.get_loc("session")]   = session_label
                                df.iat[i, df.columns.get_loc("pullback_hit")] = True
                                signal_fired = True
                                n_signals += 1
                                logger.debug(
                                    "%s SHORT @ %s | entry=%.5f sl=%.5f tp=%.5f ATR=%.5f",
                                    self.name, idx[i], entry, sl, tp, atr_v,
                                )
                                break

        if self._regime_enabled:
            total_days = len(set(idx.date))
            logger.info(
                "%s: %d signals | regime filtered %d/%d days (%.0f%%)",
                self.name, n_signals,
                n_regime_filtered, total_days,
                100.0 * n_regime_filtered / total_days if total_days else 0,
            )

        return df

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _iter_day_indices(self, idx: pd.DatetimeIndex):
        """Yield (date, list_of_integer_positions) per calendar day."""
        dates = idx.date
        current_date = None
        current_group: list = []
        for pos, d in enumerate(dates):
            if d != current_date:
                if current_group:
                    yield current_date, current_group
                current_date = d
                current_group = [pos]
            else:
                current_group.append(pos)
        if current_group:
            yield current_date, current_group

    def _compute_regime_maps(self, df: pd.DataFrame) -> dict:
        """
        Identical regime gate as LondonOpenBreakout.
        Returns {date -> {"atr_pct": float, "adx": float, "pass": bool}}.
        """
        result: dict = {}

        df_utc = df.copy()
        df_utc.index = df_utc.index.tz_convert("UTC")

        daily = df_utc.resample("D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna(subset=["open", "close"])

        atr_series = _compute_atr(daily, self._regime_atr_per)
        daily_atr: dict = {
            ts.date(): float(v)
            for ts, v in atr_series.items()
            if not (isinstance(v, float) and np.isnan(v))
        }
        sorted_dates = sorted(daily_atr.keys())

        hourly_df = pd.DataFrame({
            "high":  df_utc["high"].resample("1h").max(),
            "low":   df_utc["low"].resample("1h").min(),
            "close": df_utc["close"].resample("1h").last(),
        }).dropna()

        adx_h1 = _compute_adx(hourly_df, self._regime_adx_per)

        # Find UTC timestamp of session open H1 bar (07:00 ET → UTC)
        open_utc_by_date: dict = {}
        for date, day_indices in self._iter_day_indices(df.index):
            session_bars = [
                df.index[i] for i in day_indices
                if _bar_time(df.index[i]) >= self._london_start
            ]
            if session_bars:
                open_utc_by_date[date] = session_bars[0].tz_convert("UTC").floor("1h")

        min_history = max(self._regime_atr_per, 10)

        for idx_pos, date in enumerate(sorted_dates):
            start_i   = max(0, idx_pos - self._regime_lookback)
            prior_atrs = [daily_atr[d] for d in sorted_dates[start_i:idx_pos]]

            if len(prior_atrs) < min_history:
                result[date] = {"atr_pct": float("nan"), "adx": float("nan"), "pass": True}
                continue

            today_atr   = daily_atr[date]
            atr_pct_val = float(np.sum(np.array(prior_atrs) < today_atr) / len(prior_atrs) * 100.0)
            atr_pass    = atr_pct_val >= self._regime_atr_pct

            adx_val:  Optional[float] = None
            adx_pass: bool            = True

            if adx_h1 is not None and date in open_utc_by_date:
                open_utc   = open_utc_by_date[date]
                candidates = adx_h1.index[adx_h1.index <= open_utc]
                if len(candidates) > 0:
                    raw = float(adx_h1.loc[candidates[-1]])
                    if not np.isnan(raw):
                        adx_val  = raw
                        adx_pass = adx_val >= self._regime_adx_min

            result[date] = {
                "atr_pct": round(atr_pct_val, 1),
                "adx":     adx_val if adx_val is not None else float("nan"),
                "pass":    atr_pass and adx_pass,
            }

        return result
