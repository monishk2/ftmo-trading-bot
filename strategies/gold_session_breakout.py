"""
Strategy 6: Gold Session Breakout with VPA Filter
===================================================

WHY IT WORKS
------------
Gold (XAUUSD) consolidates during the Asian session (19:00–02:00 ET) as
physical gold trading in Asia winds down.  At London open (03:00 ET),
European and Middle Eastern institutional flows create directional impulses.
This mirrors the London Breakout structural edge but with gold-specific
parameters: wider ranges, wider stops, and a VPA confirmation filter.

The VPA (Volume Price Analysis) filter requires that the breakout candle
have above-average tick volume AND close strongly in the direction of the
break (outer 33% of its range).  This eliminates fakeouts — low-volume
"pokes" above the range that reverse quickly — while keeping high-conviction
institutional-driven breaks.

EXACT RULES
-----------
Asian range:
  Window: 19:00 → 02:00 ET (spans midnight — built from prior-day 19:00)
  Range = highest high - lowest low across all 15-min bars in window
  Minimum range: 500 pips ($5.00 — gold pip = $0.01)
  Maximum range: 2500 pips ($25.00)

Entry window: 03:00 → 11:00 ET

Breakout + VPA (2-stage):
  Stage 1: Bar close crosses asian_high + buffer  (long)
           or asian_low - buffer  (short)
  Stage 2 (VPA filter on the same 15-min bar):
    a) Bar tick volume > vpa_volume_mult × SMA(vpa_sma_period) of volume
    b) Bar closes in outer 33% of its H-L range:
         Long:  close > low + 0.67 × (high - low)
         Short: close < low + 0.33 × (high - low)
    Both must be met. If Stage 2 fails, keep checking subsequent bars
    within a 60-min timeout window (4 bars @ 15 min).

Exit:
  TP-A: entry ± sl_distance × rr_ratio  (2:1 default)
  TP-B: entry ± 1.5 × H1 ATR(14)       (tested separately via use_h1_atr_tp flag)
  SL:   opposite side of Asian range, capped at sl_cap_pips
        Long SL  = asian_low  - buffer  (capped at entry - sl_cap_pips × pip_size)
        Short SL = asian_high + buffer  (capped at entry + sl_cap_pips × pip_size)
  No break-even stop.
  Time stop: 16:00 ET.

Regime filter (relaxed vs forex):
  Daily ATR(14) > 50th percentile of prior 60 days
  H1 ADX(14)   > 20 (vs 25 for forex — gold ADX baseline is lower)

SIGNAL COLUMN CONTRACT (for Backtester)
----------------------------------------
  signal     int    1=long, -1=short, 0=no trade
  sl_price   float  absolute SL price
  tp_price   float  absolute TP price  (TP-A or TP-B depending on config)
  time_stop  pd.Timestamp  16:00 ET
  lot_size   float  NaN → Backtester auto-sizes

  Diagnostic columns:
  asian_high, asian_low, asian_range_pips
  vpa_accepted   bool   True = passed Stage 2 VPA check
  vpa_rejected   bool   True = Stage 1 fired but Stage 2 never confirmed
  regime_filtered bool
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


class GoldSessionBreakout(BaseStrategy):
    """Gold (XAUUSD) session breakout with VPA confirmation filter."""

    @property
    def name(self) -> str:
        return "GoldSessionBreakout"

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        # Session window
        self._asian_start   = _parse_time(config.get("asian_range_start",  "19:00"))
        self._asian_end     = _parse_time(config.get("asian_range_end",    "02:00"))
        self._entry_start   = _parse_time(config.get("entry_window_start", "03:00"))
        self._entry_end     = _parse_time(config.get("entry_window_end",   "11:00"))
        self._time_stop     = _parse_time(config.get("time_stop_hour",     "16:00"))

        # Range filter
        self._min_range_pips = float(config.get("min_range_pips", 500.0))
        self._max_range_pips = float(config.get("max_range_pips", 2500.0))
        self._buffer_pips    = float(config.get("buffer_pips",    50.0))
        self._sl_cap_pips    = float(config.get("sl_cap_pips",    1500.0))

        # Exit
        self._rr_ratio       = float(config.get("rr_ratio",          2.0))
        self._use_h1_atr_tp  = bool(config.get("use_h1_atr_tp",      False))
        self._h1_atr_tp_mult = float(config.get("h1_atr_tp_mult",    1.5))

        # VPA filter
        self._vpa_vol_mult   = float(config.get("vpa_volume_mult",    1.5))
        self._vpa_sma_period = int(config.get("vpa_sma_period",       20))
        self._vpa_outer_frac = float(config.get("vpa_outer_fraction", 0.33))
        self._vpa_timeout_bars = int(config.get("vpa_timeout_bars",   4))   # 4 × 15min = 60min

        # Position sizing
        self.risk_per_trade_pct = float(config["risk_per_trade_pct"])
        self._no_friday         = bool(config.get("no_friday_trading", True))

        # Instrument
        self._pip_size          = float(instrument_config["pip_size"])  # 0.01 for gold

        # Regime filter (relaxed vs forex)
        self._regime_enabled  = bool(config.get("regime_filter_enabled", True))
        self._regime_atr_per  = int(config.get("regime_atr_period",     14))
        self._regime_lookback = int(config.get("regime_lookback_days",  60))
        self._regime_atr_pct  = float(config.get("regime_atr_percentile", 50.0))
        self._regime_adx_per  = int(config.get("regime_adx_period",     14))
        self._regime_adx_min  = float(config.get("regime_adx_h1_min",   20.0))

        logger.info(
            "%s setup: range=%d–%d pips  buf=%d  sl_cap=%d  RR=%.1f  "
            "vpa_mult=%.1f  regime=%s",
            self.name,
            int(self._min_range_pips), int(self._max_range_pips),
            int(self._buffer_pips), int(self._sl_cap_pips),
            self._rr_ratio, self._vpa_vol_mult,
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
            M15 OHLCV with US/Eastern tz-aware DatetimeIndex.

        Vectorized Asian range computation: assigns each bar to its
        'trade date' (the ET calendar date of the session open that follows
        its Asian window) without per-day DataFrame concatenation.
        """
        df = self._init_signal_columns(df)
        df["asian_high"]       = np.nan
        df["asian_low"]        = np.nan
        df["asian_range_pips"] = np.nan
        df["vpa_accepted"]     = False
        df["vpa_rejected"]     = False
        df["regime_filtered"]  = False

        # ── Pre-compute rolling volume SMA (non-zero bars only) ─────
        vol_nonzero = df["volume"].replace(0.0, np.nan)
        vol_sma     = vol_nonzero.rolling(
            self._vpa_sma_period * 4, min_periods=self._vpa_sma_period
        ).mean()

        # ── Vectorized Asian range grouping ──────────────────────────
        # Each bar belongs to the Asian window of the trade date defined as:
        #   bar time in [19:00, 24:00) → trade_date = next calendar day
        #   bar time in [00:00, 02:00) → trade_date = current calendar day
        #   otherwise                  → not in any Asian window (NaT)
        idx         = df.index
        bar_hour    = idx.hour + idx.minute / 60.0
        bar_dates   = np.array(idx.date)          # calendar date of each bar
        trade_dates = np.full(len(df), None, dtype=object)

        import datetime as _dt
        eve_mask  = bar_hour >= 19.0
        pre_mask  = bar_hour < (self._asian_end.hour + self._asian_end.minute / 60.0)

        for pos in np.where(eve_mask)[0]:
            trade_dates[pos] = bar_dates[pos] + _dt.timedelta(days=1)
        for pos in np.where(pre_mask)[0]:
            if trade_dates[pos] is None:   # don't overwrite evening bars
                trade_dates[pos] = bar_dates[pos]

        # ── Compute Asian high/low per trade date (vectorized) ────────
        df["_td"] = trade_dates
        asian_df  = df.dropna(subset=["_td"])
        asian_df  = asian_df[asian_df["_td"].apply(lambda d: d is not None)]
        asian_hl  = (
            asian_df.groupby("_td")
            .agg(asian_high=("high", "max"), asian_low=("low", "min"))
        )
        df.drop(columns=["_td"], inplace=True)

        # ── Regime gate ───────────────────────────────────────────────
        regime_map: dict = self._compute_regime_maps(df) if self._regime_enabled else {}

        # ── Pre-compute H1 ATR for TP-B ──────────────────────────────
        h1_atr_map: dict = {}
        if self._use_h1_atr_tp:
            h1_atr_map = self._build_h1_atr(df)

        n_filtered = 0
        n_accepted = 0
        n_rejected = 0

        buf      = self._buffer_pips * self._pip_size
        cap_dist = self._sl_cap_pips * self._pip_size

        # ── Precompute entry-window boolean array ─────────────────────
        entry_hour = bar_hour
        entry_mask_arr = (
            (entry_hour >= (self._entry_start.hour + self._entry_start.minute / 60.0)) &
            (entry_hour <  (self._entry_end.hour   + self._entry_end.minute   / 60.0)) &
            (idx.dayofweek < 5)
        )

        # ── Convert arrays to numpy for fast per-bar access ──────────
        close_arr  = df["close"].to_numpy()
        high_arr   = df["high"].to_numpy()
        low_arr    = df["low"].to_numpy()
        vol_arr    = df["volume"].fillna(0.0).to_numpy()
        vsma_arr   = vol_sma.to_numpy()
        dates_arr  = bar_dates
        wkday_arr  = np.array(idx.dayofweek)

        # signal output arrays
        sig_arr    = np.zeros(len(df), dtype=int)
        sl_arr     = np.full(len(df), np.nan)
        tp_arr     = np.full(len(df), np.nan)
        ts_arr     = np.full(len(df), None, dtype=object)
        vacc_arr   = np.zeros(len(df), dtype=bool)
        vrej_arr   = np.zeros(len(df), dtype=bool)
        rf_arr     = np.zeros(len(df), dtype=bool)
        ahi_arr    = np.full(len(df), np.nan)
        alo_arr    = np.full(len(df), np.nan)
        arp_arr    = np.full(len(df), np.nan)

        # group positions by trade date (entry-window bars only)
        from collections import defaultdict
        date_positions: Dict = defaultdict(list)
        for pos in range(len(df)):
            if entry_mask_arr[pos]:
                date_positions[dates_arr[pos]].append(pos)

        # ── Per-day processing ────────────────────────────────────────
        for trade_date, entry_positions in sorted(date_positions.items()):
            if not entry_positions:
                continue

            # Friday filter
            if self._no_friday and wkday_arr[entry_positions[0]] == 4:
                continue

            # Regime gate
            if regime_map and trade_date in regime_map:
                if not regime_map[trade_date]["pass"]:
                    # Mark all bars on this calendar day as filtered
                    for pos in range(len(df)):
                        if dates_arr[pos] == trade_date:
                            rf_arr[pos] = True
                    n_filtered += 1
                    continue

            # Asian range from precomputed groupby
            if trade_date not in asian_hl.index:
                continue
            asian_high = float(asian_hl.loc[trade_date, "asian_high"])
            asian_low  = float(asian_hl.loc[trade_date, "asian_low"])
            range_pips = (asian_high - asian_low) / self._pip_size

            # Write diagnostics on all bars for this day
            for pos in range(len(df)):
                if dates_arr[pos] == trade_date:
                    ahi_arr[pos] = asian_high
                    alo_arr[pos] = asian_low
                    arp_arr[pos] = round(range_pips, 1)

            # Range filter
            if range_pips < self._min_range_pips or range_pips > self._max_range_pips:
                continue

            long_trigger  = asian_high + buf
            short_trigger = asian_low  - buf

            # ── Entry bar scan with VPA ────────────────────────────────
            signal_fired      = False
            stage1_long_pos   = None
            stage1_short_pos  = None

            for bar_pos_idx, pos in enumerate(entry_positions):
                if signal_fired:
                    break

                cl  = close_arr[pos]
                hi  = high_arr[pos]
                lo  = low_arr[pos]
                vol = vol_arr[pos]
                vs  = vsma_arr[pos]

                # Stage 1
                if stage1_long_pos  is None and cl > long_trigger:
                    stage1_long_pos  = bar_pos_idx
                if stage1_short_pos is None and cl < short_trigger:
                    stage1_short_pos = bar_pos_idx

                # Stage 2 for each pending Stage 1
                for direction, s1 in ((1, stage1_long_pos), (-1, stage1_short_pos)):
                    if s1 is None:
                        continue
                    bars_since = bar_pos_idx - s1

                    # Timeout
                    if bars_since > self._vpa_timeout_bars:
                        vrej_arr[entry_positions[s1]] = True
                        n_rejected += 1
                        if direction == 1:
                            stage1_long_pos  = None
                        else:
                            stage1_short_pos = None
                        continue

                    # VPA a: volume
                    vol_ok = (not np.isnan(vs)) and (vs > 0) and (vol >= self._vpa_vol_mult * vs)
                    # VPA b: close in outer 33%
                    rng    = hi - lo
                    if rng > 0:
                        close_ok = (cl >= lo + (1 - self._vpa_outer_frac) * rng) if direction == 1 \
                                   else (cl <= lo + self._vpa_outer_frac * rng)
                    else:
                        close_ok = False

                    if vol_ok and close_ok:
                        entry = cl
                        if direction == 1:
                            sl_p = max(asian_low - buf, entry - cap_dist)
                        else:
                            sl_p = min(asian_high + buf, entry + cap_dist)

                        sl_dist = abs(entry - sl_p)
                        if sl_dist < 1e-9:
                            continue

                        if self._use_h1_atr_tp:
                            h1a    = h1_atr_map.get(trade_date, sl_dist / self._h1_atr_tp_mult)
                            tp_d   = self._h1_atr_tp_mult * h1a
                        else:
                            tp_d   = sl_dist * self._rr_ratio

                        tp_p   = entry + tp_d if direction == 1 else entry - tp_d
                        ts_bar = idx[pos]
                        tstop  = pd.Timestamp(
                            year=ts_bar.year, month=ts_bar.month, day=ts_bar.day,
                            hour=self._time_stop.hour, minute=self._time_stop.minute,
                            tz=ts_bar.tzinfo,
                        )

                        sig_arr[pos]  = direction
                        sl_arr[pos]   = sl_p
                        tp_arr[pos]   = tp_p
                        ts_arr[pos]   = tstop
                        vacc_arr[pos] = True
                        signal_fired  = True
                        n_accepted   += 1

                        logger.debug(
                            "%s %s @ %s  entry=%.2f sl=%.2f tp=%.2f  range=%.0f",
                            self.name, "LONG" if direction == 1 else "SHORT",
                            ts_bar, entry, sl_p, tp_p, range_pips,
                        )
                        break

        # Write numpy arrays back to DataFrame
        df["signal"]           = sig_arr
        df["sl_price"]         = sl_arr
        df["tp_price"]         = tp_arr
        df["time_stop"]        = ts_arr
        df["vpa_accepted"]     = vacc_arr
        df["vpa_rejected"]     = vrej_arr
        df["regime_filtered"]  = rf_arr
        df["asian_high"]       = ahi_arr
        df["asian_low"]        = alo_arr
        df["asian_range_pips"] = arp_arr

        logger.info(
            "%s: accepted=%d  rejected=%d  regime_filtered=%d",
            self.name, n_accepted, n_rejected, n_filtered,
        )
        return df

    # ------------------------------------------------------------------ #
    # H1 ATR for TP-B                                                      #
    # ------------------------------------------------------------------ #

    def _build_h1_atr(self, df: pd.DataFrame) -> Dict:
        """Return {date -> H1 ATR(14)} computed at session open."""
        h1 = df.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        atr_series = ta.volatility.AverageTrueRange(
            high=h1["high"], low=h1["low"], close=h1["close"], window=14, fillna=False,
        ).average_true_range()

        result: dict = {}
        for date in set(df.index.date):
            # ATR at session open (03:00 ET bar)
            session_open = pd.Timestamp(
                year=date.year, month=date.month, day=date.day,
                hour=self._entry_start.hour, minute=self._entry_start.minute,
                tz=df.index.tz,
            )
            prior = atr_series.index[atr_series.index <= session_open]
            if len(prior) > 0:
                val = float(atr_series.loc[prior[-1]])
                if not np.isnan(val):
                    result[date] = val
        return result

    # ------------------------------------------------------------------ #
    # Regime gate (same framework as London, relaxed thresholds)           #
    # ------------------------------------------------------------------ #

    def _compute_regime_maps(self, df: pd.DataFrame) -> dict:
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

        open_utc_by_date: dict = {}
        for d, day_df in df.groupby(df.index.date):
            entry_bars = [
                ts for ts in day_df.index
                if self._entry_start <= ts.time() < self._entry_end
            ]
            if entry_bars:
                open_utc_by_date[d] = entry_bars[0].tz_convert("UTC").floor("1h")

        min_history = max(self._regime_atr_per, 10)

        for idx_pos, date in enumerate(sorted_dates):
            start_i    = max(0, idx_pos - self._regime_lookback)
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
