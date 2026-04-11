"""
Strategy 8: Gold M1 Liquidity Sweep Reversal (with H4 Trend Guard)
===================================================================

WHY THIS DIFFERS FROM THE M15 VERSION
--------------------------------------
The M15 GoldSweepReversal failed primarily due to a 2024 regime shift: gold
entered a structural uptrend driven by central-bank buying, making counter-trend
reversal fades against highs consistently unprofitable.

This M1 version adds a CRITICAL H4 TREND GUARD:
  - Compute H4 EMA50 on gold price
  - If EMA50 moved > trend_guard_threshold pips UP in last 10 H4 bars → ONLY
    allow bullish (long) sweeps (fading low sweeps); skip bearish fades
  - If EMA50 moved > trend_guard_threshold pips DOWN → ONLY allow bearish (short)
    sweeps; skip bullish fades
  - Neutral: allow both directions

M1 ADAPTATIONS vs M15
-----------------------
  - ATR(60) on M1 ≈ ATR(4) on H1 (1-hour equivalent of price range)
  - Volume SMA(60): 60-min rolling average of tick volume
  - Sweep detection: 30-pip threshold (tighter — M1 wicks more precise)
  - SL buffer: 5 pips beyond wick extreme (vs 10 on M15)
  - SL cap: 400 pips (vs 800 on M15)
  - Confirmation window: up to 3 M1 bars after sweep bar (3 min)
  - Entry in same direction as H4 EMA trend when trend is strong

SIGNAL COLUMN CONTRACT
-----------------------
  signal       int    1=long, -1=short, 0=no trade
  sl_price     float
  tp_price     float
  time_stop    pd.Timestamp
  level_type   str
  sweep_pips   float
  sl_pips      float
  h4_trend     int    H4 trend direction at signal bar: 1/0/-1
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
from strategies.regime_filter import _compute_atr

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(h, m)


_PDH = "PDH";  _PDL = "PDL"
_ASH = "ASH";  _ASL = "ASL"
_LSH = "LSH";  _LSL = "LSL"
_NSH = "NSH";  _NSL = "NSL"


class GoldM1SweepReversal(BaseStrategy):
    """Gold (XAUUSD) M1 liquidity sweep reversal with H4 trend guard."""

    @property
    def name(self) -> str:
        return "GoldM1SweepReversal"

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        # Session windows
        self._london_start = _parse_time(config.get("london_entry_start", "03:00"))
        self._london_end   = _parse_time(config.get("london_entry_end",   "07:00"))
        self._ny_start     = _parse_time(config.get("ny_entry_start",     "08:00"))
        self._ny_end       = _parse_time(config.get("ny_entry_end",       "11:00"))
        self._time_stop    = _parse_time(config.get("time_stop_hour",     "16:00"))

        # Sweep detection (M1-tuned)
        self._sweep_min_pips = float(config.get("sweep_min_pips",     30.0))
        self._sl_buffer_pips = float(config.get("sl_buffer_pips",      5.0))
        self._sl_cap_pips    = float(config.get("sl_cap_pips",        400.0))

        # Entry filters (M1-tuned periods)
        self._atr_period     = int(config.get("atr_period",            60))   # ~1 hour
        self._body_atr_min   = float(config.get("candle_body_atr_min",  0.3))  # looser for M1
        self._vol_mult       = float(config.get("volume_mult",           1.3))
        self._vol_sma_period = int(config.get("vol_sma_period",          60))  # 60-min SMA
        self._outer_frac     = float(config.get("outer_fraction",         0.33))
        self._confirm_bars   = int(config.get("confirm_bars",              3))  # M1 confirmation window

        # Exit
        self._rr_ratio       = float(config.get("rr_ratio",      2.5))
        self._no_friday      = bool(config.get("no_friday_trading", True))
        self._max_trades_day = int(config.get("max_trades_per_day", 2))

        # H4 trend guard
        self._trend_guard_enabled   = bool(config.get("trend_guard_enabled",    True))
        self._trend_guard_h4_ema    = int(config.get("trend_guard_h4_ema",      50))
        self._trend_guard_bars      = int(config.get("trend_guard_bars",        10))   # H4 bars lookback
        self._trend_guard_threshold = float(config.get("trend_guard_threshold", 500.0)) # pips

        # Position sizing
        self.risk_per_trade_pct = float(config["risk_per_trade_pct"])

        # Instrument
        self._pip_size = float(instrument_config["pip_size"])

        # Regime filter (ATR percentile only)
        self._regime_enabled  = bool(config.get("regime_filter_enabled",   True))
        self._regime_atr_per  = int(config.get("regime_atr_period",        14))
        self._regime_lookback = int(config.get("regime_lookback_days",     60))
        self._regime_atr_pct  = float(config.get("regime_atr_percentile",  40.0))

        logger.info(
            "%s setup: sweep_min=%d  sl_buf=%d  sl_cap=%d  RR=%.1f  "
            "body_atr=%.2f  trend_guard=%s(th=%.0fpips)  regime=%s",
            self.name,
            int(self._sweep_min_pips), int(self._sl_buffer_pips), int(self._sl_cap_pips),
            self._rr_ratio, self._body_atr_min,
            "ON" if self._trend_guard_enabled else "OFF", self._trend_guard_threshold,
            "ON" if self._regime_enabled else "OFF",
        )

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate M1 sweep reversal signals with H4 trend guard.

        Parameters
        ----------
        df : M1 OHLCV, US/Eastern tz-aware DatetimeIndex.
        """
        df = self._init_signal_columns(df)
        df["level_type"] = ""
        df["sweep_pips"] = np.nan
        df["sl_pips"]    = np.nan
        df["h4_trend"]   = 0

        ps = self._pip_size
        n  = len(df)
        idx = df.index

        # ── ATR on M1 (atr_period bars ≈ 1 hour) ─────────────────────
        atr_m1 = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=self._atr_period, fillna=False,
        ).average_true_range().to_numpy()

        # ── Volume SMA (60-min rolling, non-zero bars only) ────────────
        vol_raw  = df["volume"].to_numpy()
        non_zero = (vol_raw > 0).sum()
        vol_valid = non_zero > len(df) * 0.1
        if vol_valid:
            vol_sma = (
                pd.Series(vol_raw)
                .replace(0.0, np.nan)
                .rolling(self._vol_sma_period, min_periods=self._vol_sma_period // 2)
                .mean()
                .to_numpy()
            )
        else:
            vol_sma    = np.full(n, np.nan)
            vol_valid  = False

        # ── H4 Trend Guard ────────────────────────────────────────────
        trend_arr = np.zeros(n, dtype=int)
        if self._trend_guard_enabled:
            trend_arr = self._compute_h4_trend_guard(df)
            df["h4_trend"] = trend_arr

        # ── Regime gate ───────────────────────────────────────────────
        regime_pass: Dict = {}
        if self._regime_enabled:
            regime_pass = self._compute_regime(df)

        # ── Raw arrays ────────────────────────────────────────────────
        open_arr  = df["open"].to_numpy()
        high_arr  = df["high"].to_numpy()
        low_arr   = df["low"].to_numpy()
        close_arr = df["close"].to_numpy()

        dates_arr    = np.array(idx.date)
        unique_dates = sorted(set(dates_arr))

        date_pos_map: Dict = defaultdict(list)
        for pi in range(n):
            date_pos_map[dates_arr[pi]].append(pi)

        levels_by_date = self._compute_all_levels(
            df, unique_dates, dates_arr, date_pos_map, idx
        )

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

        # ── Per-date signal loop ───────────────────────────────────────
        for trade_date, positions in sorted(date_pos_map.items()):
            if self._no_friday and idx[positions[0]].dayofweek == 4:
                continue
            if regime_pass and not regime_pass.get(trade_date, True):
                continue

            levels = levels_by_date.get(trade_date, [])
            if not levels:
                continue

            consumed_levels: set = set()
            trades_today = 0
            entry_positions = [p for p in positions if in_window[p]]
            if not entry_positions:
                continue

            for pos in entry_positions:
                if trades_today >= self._max_trades_day:
                    break

                hi  = high_arr[pos]
                lo  = low_arr[pos]
                atr = atr_m1[pos]
                trend = int(trend_arr[pos])

                for level_price, level_type, is_high_level in levels:
                    if level_type in consumed_levels:
                        continue

                    # ── H4 trend guard ────────────────────────────────
                    # is_high_level=True → bearish (short) setup
                    # is_high_level=False → bullish (long) setup
                    if trend == 1 and is_high_level:
                        # Uptrend: skip bearish fades
                        continue
                    if trend == -1 and not is_high_level:
                        # Downtrend: skip bullish fades
                        continue

                    if is_high_level:
                        # ── HIGH SWEEP (bearish) ──────────────────────
                        sweep_ok = hi > level_price + sweep_min_price
                        if not sweep_ok:
                            continue

                        # Find confirmation bar (closes back below level)
                        conf_pos = None
                        for k in range(pos, min(pos + self._confirm_bars, n)):
                            if close_arr[k] < level_price and in_window[k]:
                                conf_pos = k
                                break
                        if conf_pos is None:
                            continue

                        # Filters on confirmation bar
                        c_hi  = high_arr[conf_pos]
                        c_lo  = low_arr[conf_pos]
                        c_cl  = close_arr[conf_pos]
                        c_op  = open_arr[conf_pos]
                        c_vol = vol_raw[conf_pos]
                        c_vs  = vol_sma[conf_pos]

                        bar_range = c_hi - c_lo
                        close_ok  = (bar_range > 0) and \
                                    (c_cl <= c_lo + self._outer_frac * bar_range)

                        body_size = abs(c_cl - c_op)
                        c_atr     = atr_m1[conf_pos]
                        disp_ok   = (
                            not np.isnan(c_atr) and c_atr > 0 and
                            body_size >= self._body_atr_min * c_atr
                        )

                        vol_ok = True
                        if vol_valid and not np.isnan(c_vs) and c_vs > 0:
                            vol_ok = c_vol >= self._vol_mult * c_vs

                        if not (close_ok and disp_ok and vol_ok):
                            continue

                        # SL/TP
                        entry   = c_cl
                        sl_p    = hi + sl_buf_price
                        sl_dist = abs(sl_p - entry)
                        if sl_dist > sl_cap_price:
                            continue

                        tp_p    = entry - self._rr_ratio * sl_dist

                        ts_bar = idx[conf_pos]
                        tstop  = pd.Timestamp(
                            year=ts_bar.year, month=ts_bar.month, day=ts_bar.day,
                            hour=self._time_stop.hour, minute=self._time_stop.minute,
                            tz=ts_bar.tzinfo,
                        )

                        sig_arr[conf_pos]  = -1
                        sl_arr[conf_pos]   = sl_p
                        tp_arr[conf_pos]   = tp_p
                        ts_arr[conf_pos]   = tstop
                        lt_arr[conf_pos]   = level_type
                        sp_arr[conf_pos]   = round((hi - level_price) / ps, 1)
                        slp_arr[conf_pos]  = round(sl_dist / ps, 1)

                        consumed_levels.add(level_type)
                        trades_today += 1
                        break

                    else:
                        # ── LOW SWEEP (bullish) ───────────────────────
                        sweep_ok = lo < level_price - sweep_min_price
                        if not sweep_ok:
                            continue

                        conf_pos = None
                        for k in range(pos, min(pos + self._confirm_bars, n)):
                            if close_arr[k] > level_price and in_window[k]:
                                conf_pos = k
                                break
                        if conf_pos is None:
                            continue

                        c_hi  = high_arr[conf_pos]
                        c_lo  = low_arr[conf_pos]
                        c_cl  = close_arr[conf_pos]
                        c_op  = open_arr[conf_pos]
                        c_vol = vol_raw[conf_pos]
                        c_vs  = vol_sma[conf_pos]

                        bar_range = c_hi - c_lo
                        close_ok  = (bar_range > 0) and \
                                    (c_cl >= c_lo + (1 - self._outer_frac) * bar_range)

                        body_size = abs(c_cl - c_op)
                        c_atr     = atr_m1[conf_pos]
                        disp_ok   = (
                            not np.isnan(c_atr) and c_atr > 0 and
                            body_size >= self._body_atr_min * c_atr
                        )

                        vol_ok = True
                        if vol_valid and not np.isnan(c_vs) and c_vs > 0:
                            vol_ok = c_vol >= self._vol_mult * c_vs

                        if not (close_ok and disp_ok and vol_ok):
                            continue

                        entry   = c_cl
                        sl_p    = lo - sl_buf_price
                        sl_dist = abs(entry - sl_p)
                        if sl_dist > sl_cap_price:
                            continue

                        tp_p = entry + self._rr_ratio * sl_dist

                        ts_bar = idx[conf_pos]
                        tstop  = pd.Timestamp(
                            year=ts_bar.year, month=ts_bar.month, day=ts_bar.day,
                            hour=self._time_stop.hour, minute=self._time_stop.minute,
                            tz=ts_bar.tzinfo,
                        )

                        sig_arr[conf_pos]  = 1
                        sl_arr[conf_pos]   = sl_p
                        tp_arr[conf_pos]   = tp_p
                        ts_arr[conf_pos]   = tstop
                        lt_arr[conf_pos]   = level_type
                        sp_arr[conf_pos]   = round((level_price - lo) / ps, 1)
                        slp_arr[conf_pos]  = round(sl_dist / ps, 1)

                        consumed_levels.add(level_type)
                        trades_today += 1
                        break

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
    # H4 Trend Guard                                                       #
    # ------------------------------------------------------------------ #

    def _compute_h4_trend_guard(self, df: pd.DataFrame) -> np.ndarray:
        """
        Resample M1 → H4, compute EMA50, check 10-bar displacement.

        Returns numpy int array aligned to df.index:
          +1 = strong uptrend (EMA50 rose > threshold pips in 10 H4 bars)
          -1 = strong downtrend
           0 = neutral
        """
        # H4 resample (anchor=midnight ET → bars at 00:00, 04:00, 08:00, …)
        h4_close = df["close"].resample("4h").last().dropna()

        if len(h4_close) < self._trend_guard_bars + self._trend_guard_h4_ema + 5:
            return np.zeros(len(df), dtype=int)

        ema50 = h4_close.ewm(span=self._trend_guard_h4_ema, adjust=False).mean()
        displacement_pips = (ema50 - ema50.shift(self._trend_guard_bars)) / self._pip_size

        h4_trend = pd.Series(0, index=h4_close.index, dtype=int)
        h4_trend[displacement_pips >  self._trend_guard_threshold]  =  1
        h4_trend[displacement_pips < -self._trend_guard_threshold]  = -1

        # Forward-fill H4 trend to M1 bars
        trend_m1 = h4_trend.reindex(df.index, method="ffill").fillna(0).astype(int)
        return trend_m1.to_numpy(dtype=int)

    # ------------------------------------------------------------------ #
    # Liquidity levels (same logic as M15 version)                        #
    # ------------------------------------------------------------------ #

    def _compute_all_levels(
        self,
        df: pd.DataFrame,
        unique_dates: list,
        dates_arr: np.ndarray,
        date_pos_map: dict,
        idx,
    ) -> Dict:
        high_arr  = df["high"].to_numpy()
        low_arr   = df["low"].to_numpy()
        hours_arr = np.array(idx.hour)

        levels_by_date: Dict = {}
        date_set = set(unique_dates)

        for trade_date in unique_dates:
            prev_date = None
            for delta in range(1, 8):
                cand = trade_date - _dt.timedelta(days=delta)
                if cand in date_set:
                    prev_date = cand
                    break

            levels: List[Tuple[float, str, bool]] = []

            if prev_date is None:
                levels_by_date[trade_date] = levels
                continue

            prev_pos  = date_pos_map.get(prev_date, [])
            today_pos = date_pos_map.get(trade_date, [])

            if not prev_pos:
                levels_by_date[trade_date] = levels
                continue

            # PDH / PDL
            pdh = float(high_arr[prev_pos].max())
            pdl = float(low_arr[prev_pos].min())
            levels.append((pdh, _PDH, True))
            levels.append((pdl, _PDL, False))

            # Asian (prev eve ≥19:00 + today pre <02:00 ET)
            prev_eve  = [p for p in prev_pos  if hours_arr[p] >= 19]
            today_pre = [p for p in today_pos if hours_arr[p] < 2]
            asian_pos = prev_eve + today_pre
            if asian_pos:
                levels.append((float(high_arr[asian_pos].max()), _ASH, True))
                levels.append((float(low_arr[asian_pos].min()),  _ASL, False))

            # Prev London (03:00–08:00 ET)
            lon_pos = [p for p in prev_pos if 3 <= hours_arr[p] < 8]
            if lon_pos:
                levels.append((float(high_arr[lon_pos].max()), _LSH, True))
                levels.append((float(low_arr[lon_pos].min()),  _LSL, False))

            # Prev NY (08:00–16:00 ET)
            ny_pos = [p for p in prev_pos if 8 <= hours_arr[p] < 16]
            if ny_pos:
                levels.append((float(high_arr[ny_pos].max()), _NSH, True))
                levels.append((float(low_arr[ny_pos].min()),  _NSL, False))

            levels_by_date[trade_date] = levels

        return levels_by_date

    # ------------------------------------------------------------------ #
    # Regime filter                                                        #
    # ------------------------------------------------------------------ #

    def _compute_regime(self, df: pd.DataFrame) -> Dict[object, bool]:
        result: Dict[object, bool] = {}
        df_utc = df.copy()
        df_utc.index = df_utc.index.tz_convert("UTC")

        daily = df_utc.resample("D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna(subset=["open", "close"])

        atr_series = _compute_atr(daily, self._regime_atr_per)
        daily_atr  = {
            ts.date(): float(v)
            for ts, v in atr_series.items()
            if not (isinstance(v, float) and np.isnan(v))
        }
        sorted_dates = sorted(daily_atr.keys())

        for idx_pos, date in enumerate(sorted_dates):
            start_i    = max(0, idx_pos - self._regime_lookback)
            prior_atrs = [daily_atr[d] for d in sorted_dates[start_i:idx_pos]]
            if len(prior_atrs) < 10:
                result[date] = True
                continue
            today_atr    = daily_atr[date]
            pct_val      = float(np.sum(np.array(prior_atrs) < today_atr) / len(prior_atrs) * 100.0)
            result[date] = pct_val >= self._regime_atr_pct

        return result
