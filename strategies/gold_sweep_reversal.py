"""
Strategy 7: Gold Liquidity Sweep Reversal
==========================================

WHY IT WORKS
------------
Gold's session opens (London 03:00 ET, NY 08:00 ET) are dominated by
stop hunts / liquidity sweeps.  Market makers push price above or below
obvious liquidity levels (PDH, PDL, Asian session high/low) to fill their
own large orders before reversing.  The edge is FADING these sweeps —
entering in the direction of the reversal after the sweep wick is confirmed.

This is the structural opposite of the session breakout strategy.  Where
breakout trades WITH the move, sweep reversal trades AGAINST the initial
move once the liquidity grab is confirmed.

EXACT RULES
-----------
Liquidity levels (recalculated daily at 17:00 ET):
  - Previous Day High (PDH) / Previous Day Low (PDL)
  - Asian session high/low (19:00–02:00 ET, spans midnight)
  - Previous London session high/low (03:00–08:00 ET prior day)
  - Previous NY session high/low (08:00–16:00 ET prior day)

Sweep detection (M15 bars):
  For HIGH sweep (bearish setup):
    bar.high > level + sweep_min_pips × pip_size
    AND (bar.close < level  OR  next_bar.close < level)
  For LOW sweep (bullish setup):
    bar.low < level - sweep_min_pips × pip_size
    AND (bar.close > level  OR  next_bar.close > level)

Entry conditions (all required on confirmation bar):
  1. Sweep confirmed as above
  2. Confirmation candle outer-33%:
       Bullish: close > low + 0.67 × (high - low)
       Bearish: close < low + 0.33 × (high - low)
  3. Displacement: body size > candle_body_atr_min × ATR(14) M15
  4. Volume: tick volume > volume_mult × SMA(20) of non-zero volume
     (skipped automatically if volume data is all-zero)

Session windows:
  London: 03:00–07:00 ET
  NY:     08:00–11:00 ET

Max 2 trades/day.  Max 1 trade per liquidity level.

Exit:
  SL: sweep wick extreme + sl_buffer_pips (capped at sl_cap_pips)
  TP-A: entry ± sl_dist × rr_ratio
  TP-B: opposing liquidity level, min 1.5 × sl_dist
  No break-even.  Time stop: 16:00 ET.

Regime filter:
  Daily ATR(14) > atr_percentile_th percentile of prior 60 days
  NO ADX filter (reversal strategy — ADX irrelevant)

SIGNAL COLUMN CONTRACT
-----------------------
  signal       int    1=long, -1=short, 0=no trade
  sl_price     float
  tp_price     float  (TP-A or TP-B per config)
  time_stop    pd.Timestamp
  level_type   str    which level was swept (diagnostic)
  sweep_pips   float  how far the wick went past the level
  sl_pips      float  SL distance in pips (diagnostic)
"""

from __future__ import annotations

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


# Level type constants
_PDH = "PDH"
_PDL = "PDL"
_ASH = "ASH"  # Asian session high
_ASL = "ASL"  # Asian session low
_LSH = "LSH"  # London session high (prior)
_LSL = "LSL"  # London session low (prior)
_NSH = "NSH"  # NY session high (prior)
_NSL = "NSL"  # NY session low (prior)


class GoldSweepReversal(BaseStrategy):
    """Gold (XAUUSD) liquidity sweep reversal strategy."""

    @property
    def name(self) -> str:
        return "GoldSweepReversal"

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

        # Sweep detection
        self._sweep_min_pips = float(config.get("sweep_min_pips",    20.0))
        self._sl_buffer_pips = float(config.get("sl_buffer_pips",    10.0))
        self._sl_cap_pips    = float(config.get("sl_cap_pips",       800.0))

        # Entry filters
        self._body_atr_min   = float(config.get("candle_body_atr_min", 0.5))
        self._vol_mult       = float(config.get("volume_mult",          1.3))
        self._vol_sma_period = int(config.get("vpa_sma_period",         20))
        self._outer_frac     = float(config.get("outer_fraction",       0.33))

        # Exit
        self._rr_ratio      = float(config.get("rr_ratio",      2.5))
        self._tp_mode       = str(config.get("tp_mode",         "A"))   # "A" or "B"
        self._tp_b_min_rr   = float(config.get("tp_b_min_rr",   1.5))
        self._no_friday     = bool(config.get("no_friday_trading", True))
        self._max_trades_day = int(config.get("max_trades_per_day", 2))

        # Position sizing
        self.risk_per_trade_pct = float(config["risk_per_trade_pct"])

        # Instrument
        self._pip_size = float(instrument_config["pip_size"])

        # Regime filter (ATR percentile only — no ADX for reversal)
        self._regime_enabled  = bool(config.get("regime_filter_enabled",  True))
        self._regime_atr_per  = int(config.get("regime_atr_period",       14))
        self._regime_lookback = int(config.get("regime_lookback_days",    60))
        self._regime_atr_pct  = float(config.get("regime_atr_percentile", 40.0))

        logger.info(
            "%s setup: sweep_min=%d  sl_buf=%d  sl_cap=%d  RR=%.1f  "
            "body_atr=%.1f  vol_mult=%.1f  regime=%s",
            self.name,
            int(self._sweep_min_pips), int(self._sl_buffer_pips), int(self._sl_cap_pips),
            self._rr_ratio, self._body_atr_min, self._vol_mult,
            "ON" if self._regime_enabled else "OFF",
        )

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full vectorised liquidity-sweep reversal signal generation.

        Parameters
        ----------
        df : M15 OHLCV, US/Eastern tz-aware DatetimeIndex.
        """
        df = self._init_signal_columns(df)
        # Extra diagnostic columns
        df["level_type"]  = ""
        df["sweep_pips"]  = np.nan
        df["sl_pips"]     = np.nan

        ps = self._pip_size

        # ── ATR(14) on M15 for displacement check ─────────────────────
        atr_m15 = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=14, fillna=False,
        ).average_true_range().to_numpy()

        # ── Volume SMA (non-zero bars) ─────────────────────────────────
        vol_raw   = df["volume"].to_numpy()
        vol_valid = (vol_raw > 0).sum() > len(df) * 0.1   # >10% non-zero
        if vol_valid:
            vol_sma = (
                pd.Series(vol_raw)
                .replace(0.0, np.nan)
                .rolling(self._vol_sma_period * 4, min_periods=self._vol_sma_period)
                .mean()
                .to_numpy()
            )
        else:
            vol_sma    = np.full(len(df), np.nan)
            vol_valid  = False   # skip volume filter

        # ── Regime gate ────────────────────────────────────────────────
        regime_pass: Dict[object, bool] = {}
        if self._regime_enabled:
            regime_pass = self._compute_regime(df)

        # ── Raw arrays for speed ───────────────────────────────────────
        idx        = df.index
        open_arr   = df["open"].to_numpy()
        high_arr   = df["high"].to_numpy()
        low_arr    = df["low"].to_numpy()
        close_arr  = df["close"].to_numpy()
        n          = len(df)

        # ── Compute daily liquidity levels (vectorized) ───────────────
        # We use calendar date in ET as the "trade date"
        dates_arr = np.array(idx.date)
        unique_dates = sorted(set(dates_arr))

        # Pre-build fast lookup: {date: slice of positions}
        date_to_positions_all: Dict = defaultdict(list)
        for pos_i in range(n):
            date_to_positions_all[dates_arr[pos_i]].append(pos_i)

        levels_by_date: Dict = self._compute_all_levels(
            df, unique_dates, dates_arr, date_to_positions_all, idx
        )

        # ── Entry window mask ──────────────────────────────────────────
        bar_hour  = idx.hour + idx.minute / 60.0
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
        sig_arr   = np.zeros(n, dtype=int)
        sl_arr    = np.full(n, np.nan)
        tp_arr    = np.full(n, np.nan)
        ts_arr    = np.full(n, None, dtype=object)
        lt_arr    = np.full(n, "", dtype=object)
        sp_arr    = np.full(n, np.nan)
        slp_arr   = np.full(n, np.nan)

        # ── Per-date processing ────────────────────────────────────────
        # Group bar positions by date
        date_positions: Dict = defaultdict(list)
        for pos in range(n):
            date_positions[dates_arr[pos]].append(pos)

        sweep_min_price = self._sweep_min_pips * ps
        sl_buf_price    = self._sl_buffer_pips * ps
        sl_cap_price    = self._sl_cap_pips * ps

        for trade_date, positions in sorted(date_positions.items()):
            # Friday filter
            if self._no_friday and idx[positions[0]].dayofweek == 4:
                continue

            # Regime gate
            if regime_pass and not regime_pass.get(trade_date, True):
                continue

            levels = levels_by_date.get(trade_date, [])
            if not levels:
                continue

            # Track which levels have been consumed and daily trade count
            consumed_levels: set = set()
            trades_today = 0

            # Get entry-window positions for this date
            entry_positions = [p for p in positions if in_window[p]]
            if not entry_positions:
                continue

            # Sweep detection: iterate entry-window bars
            # For each bar, check if it sweeps any active level
            for pi, pos in enumerate(entry_positions):
                if trades_today >= self._max_trades_day:
                    break

                hi  = high_arr[pos]
                lo  = low_arr[pos]
                cl  = close_arr[pos]
                vol = vol_raw[pos] if vol_valid else np.nan
                vs  = vol_sma[pos] if vol_valid else np.nan
                atr = atr_m15[pos]

                # Next bar for confirmation
                next_pos = pos + 1 if pos + 1 < n else None

                for level_price, level_type, is_high_level in levels:
                    if level_type in consumed_levels:
                        continue

                    if is_high_level:
                        # HIGH sweep (bearish setup)
                        sweep_ok = hi > level_price + sweep_min_price
                        if not sweep_ok:
                            continue

                        # Confirmation: close back below level (same bar or next)
                        conf_pos = None
                        if cl < level_price:
                            conf_pos = pos
                        elif next_pos is not None and close_arr[next_pos] < level_price:
                            conf_pos = next_pos

                        if conf_pos is None:
                            continue

                        # Entry filters on confirmation bar
                        c_hi  = high_arr[conf_pos]
                        c_lo  = low_arr[conf_pos]
                        c_cl  = close_arr[conf_pos]
                        c_op  = open_arr[conf_pos]
                        c_vol = vol_raw[conf_pos]
                        c_vs  = vol_sma[conf_pos]
                        c_atr = atr_m15[conf_pos]

                        # 1. Outer 33% bearish close
                        bar_range = c_hi - c_lo
                        close_ok  = (bar_range > 0) and (c_cl <= c_lo + self._outer_frac * bar_range)

                        # 2. Displacement: body size
                        body_size  = abs(c_cl - c_op)
                        disp_ok    = (not np.isnan(c_atr)) and (c_atr > 0) and \
                                     (body_size >= self._body_atr_min * c_atr)

                        # 3. Volume filter
                        if vol_valid and not np.isnan(c_vs) and c_vs > 0:
                            vol_ok = c_vol >= self._vol_mult * c_vs
                        else:
                            vol_ok = True   # skip if no volume data

                        if not (close_ok and disp_ok and vol_ok):
                            continue

                        # Check confirmation bar is still in entry window
                        if not in_window[conf_pos]:
                            continue

                        # Build trade
                        entry   = c_cl
                        sl_p    = hi + sl_buf_price         # beyond sweep wick
                        sl_dist = abs(sl_p - entry)

                        if sl_dist > sl_cap_price:
                            continue   # SL too wide

                        # TP-A: fixed RR
                        # TP-B: opposing level (PDL if sweeping PDH, etc.)
                        if self._tp_mode == "B":
                            opp   = self._opposing_level(level_type, levels)
                            tp_d  = abs(entry - opp) if opp is not None else 0.0
                            if tp_d < self._tp_b_min_rr * sl_dist:
                                tp_d = self._tp_b_min_rr * sl_dist
                        else:
                            tp_d = self._rr_ratio * sl_dist

                        tp_p = entry - tp_d   # SHORT

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

                        logger.debug(
                            "%s SHORT @ %s  level=%s lv=%.2f  entry=%.2f sl=%.2f tp=%.2f",
                            self.name, ts_bar, level_type, level_price, entry, sl_p, tp_p,
                        )
                        break   # one level per bar scan

                    else:
                        # LOW sweep (bullish setup)
                        sweep_ok = lo < level_price - sweep_min_price
                        if not sweep_ok:
                            continue

                        # Confirmation: close back above level
                        conf_pos = None
                        if cl > level_price:
                            conf_pos = pos
                        elif next_pos is not None and close_arr[next_pos] > level_price:
                            conf_pos = next_pos

                        if conf_pos is None:
                            continue

                        c_hi  = high_arr[conf_pos]
                        c_lo  = low_arr[conf_pos]
                        c_cl  = close_arr[conf_pos]
                        c_op  = open_arr[conf_pos]
                        c_vol = vol_raw[conf_pos]
                        c_vs  = vol_sma[conf_pos]
                        c_atr = atr_m15[conf_pos]

                        bar_range = c_hi - c_lo
                        close_ok  = (bar_range > 0) and \
                                    (c_cl >= c_lo + (1 - self._outer_frac) * bar_range)

                        body_size = abs(c_cl - c_op)
                        disp_ok   = (not np.isnan(c_atr)) and (c_atr > 0) and \
                                    (body_size >= self._body_atr_min * c_atr)

                        if vol_valid and not np.isnan(c_vs) and c_vs > 0:
                            vol_ok = c_vol >= self._vol_mult * c_vs
                        else:
                            vol_ok = True

                        if not (close_ok and disp_ok and vol_ok):
                            continue

                        if not in_window[conf_pos]:
                            continue

                        entry   = c_cl
                        sl_p    = lo - sl_buf_price
                        sl_dist = abs(entry - sl_p)

                        if sl_dist > sl_cap_price:
                            continue

                        if self._tp_mode == "B":
                            opp  = self._opposing_level(level_type, levels)
                            tp_d = abs(entry - opp) if opp is not None else 0.0
                            if tp_d < self._tp_b_min_rr * sl_dist:
                                tp_d = self._tp_b_min_rr * sl_dist
                        else:
                            tp_d = self._rr_ratio * sl_dist

                        tp_p = entry + tp_d

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

                        logger.debug(
                            "%s LONG @ %s  level=%s lv=%.2f  entry=%.2f sl=%.2f tp=%.2f",
                            self.name, ts_bar, level_type, level_price, entry, sl_p, tp_p,
                        )
                        break

        # Write back
        df["signal"]     = sig_arr
        df["sl_price"]   = sl_arr
        df["tp_price"]   = tp_arr
        df["time_stop"]  = ts_arr
        df["level_type"] = lt_arr
        df["sweep_pips"] = sp_arr
        df["sl_pips"]    = slp_arr

        n_trades = int((sig_arr != 0).sum())
        logger.info("%s: total signals=%d", self.name, n_trades)
        return df

    # ------------------------------------------------------------------ #
    # Liquidity level computation                                          #
    # ------------------------------------------------------------------ #

    def _compute_all_levels(
        self,
        df: pd.DataFrame,
        unique_dates: list,
        dates_arr: np.ndarray,
        date_pos_map: dict,
        idx,
    ) -> Dict:
        """
        Vectorized: compute all daily liquidity levels for every trade date
        in one pass, using pre-built date→position maps.
        """
        import datetime as _dt

        # Per-date OHLC arrays (pre-built for fast lookup)
        high_arr  = df["high"].to_numpy()
        low_arr   = df["low"].to_numpy()
        hours_arr = np.array(idx.hour)

        levels_by_date: Dict = {}
        date_set = set(unique_dates)

        for trade_date in unique_dates:
            # Find previous trading day (skip gaps / weekends)
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

            prev_pos = date_pos_map.get(prev_date, [])
            today_pos = date_pos_map.get(trade_date, [])

            if not prev_pos:
                levels_by_date[trade_date] = levels
                continue

            # ── PDH / PDL ─────────────────────────────────────────────
            ph = prev_pos  # all positions of prev day
            pdh = float(high_arr[ph].max()) if len(ph) else np.nan
            pdl = float(low_arr[ph].min())  if len(ph) else np.nan
            if not np.isnan(pdh):
                levels.append((pdh, _PDH, True))
                levels.append((pdl, _PDL, False))

            # ── Asian session: prev eve (hour>=19) + today pre (hour<2) ─
            prev_eve = [p for p in prev_pos if hours_arr[p] >= 19]
            today_pre = [p for p in today_pos if hours_arr[p] < 2]
            asian_pos = prev_eve + today_pre
            if asian_pos:
                ash = float(high_arr[asian_pos].max())
                asl = float(low_arr[asian_pos].min())
                levels.append((ash, _ASH, True))
                levels.append((asl, _ASL, False))

            # ── Prev London (03:00–08:00 ET) ──────────────────────────
            lon_pos = [p for p in prev_pos if 3 <= hours_arr[p] < 8]
            if lon_pos:
                levels.append((float(high_arr[lon_pos].max()), _LSH, True))
                levels.append((float(low_arr[lon_pos].min()),  _LSL, False))

            # ── Prev NY (08:00–16:00 ET) ──────────────────────────────
            ny_pos = [p for p in prev_pos if 8 <= hours_arr[p] < 16]
            if ny_pos:
                levels.append((float(high_arr[ny_pos].max()), _NSH, True))
                levels.append((float(low_arr[ny_pos].min()),  _NSL, False))

            levels_by_date[trade_date] = levels

        return levels_by_date

    def _opposing_level(
        self, swept_type: str, levels: List[Tuple[float, str, bool]]
    ) -> Optional[float]:
        """Return the natural opposing level price for TP-B."""
        opp_map = {
            _PDH: _PDL, _PDL: _PDH,
            _ASH: _ASL, _ASL: _ASH,
            _LSH: _LSL, _LSL: _LSH,
            _NSH: _NSL, _NSL: _NSH,
        }
        opp_type = opp_map.get(swept_type)
        if opp_type is None:
            return None
        for price, lt, _ in levels:
            if lt == opp_type:
                return price
        return None

    # ------------------------------------------------------------------ #
    # Regime filter (ATR percentile, no ADX)                              #
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

            today_atr   = daily_atr[date]
            pct_val     = float(np.sum(np.array(prior_atrs) < today_atr) / len(prior_atrs) * 100.0)
            result[date] = pct_val >= self._regime_atr_pct

        return result
