"""
Strategy 10: Gold ICT Sweep-to-FVG  (SMC / ICT Framework)
===========================================================

WHY THIS IS DIFFERENT FROM PREVIOUS GOLD STRATEGIES
-----------------------------------------------------
Previous approaches entered DIRECTLY on the sweep reversal bar (Prompt 21)
or on an M1 wick sweep (Prompts 22-23).  Both entered too early — before
the market structure confirmed the reversal — resulting in large SLs and
poor win rates.

This strategy requires a 4-step sequence before entry:

  Step 1  SWEEP        M1 wick pierces an M15 liquidity level ≥ 15 pips,
                       closes BACK inside the level (wick, not breakout).

  Step 2  DISPLACEMENT Within 5 M1 bars after sweep: a strong reversal
                       candle with body > 0.5 × ATR(60) and close in the
                       outer 25% of its range.

  Step 3  FVG          The 3-candle pattern [pre, displacement, post]
                       leaves a Fair Value Gap ≥ min_fvg_pips.

  Step 4  RETEST       Price pulls back INTO the FVG zone.
                       Entry at close of the first retest bar.
                       If price fills >75% of zone before entry → SKIP.
                       Timeout: 30 M1 bars from FVG confirmation.

EXIT LOGIC
----------
  SL:   Sweep wick extreme + sl_buffer_pips (capped at sl_cap_pips)
  TP1:  entry ± 1.5 × SL distance  (close 50% of position)
  TP2:  entry ± 3.0 × SL distance  (close remaining 50%)
  After TP1: SL moves to breakeven for the remaining 50%.
  Time stop: 16:00 ET.
  Blended TP for single-exit backtesting: 2.25 × SL distance.

CONFLUENCE SCORING (0-6)
------------------------
  +1  Sweep of MAJOR level (PDH/PDL/Asian H/L)
  +1  Displacement volume > 1.3 × SMA(20) volume
  +1  FVG size > 20 pips
  +1  MSS/CHoCH confirmed before retest
  +1  H4 EMA20 slope aligns with trade direction
  +1  Within London–NY overlap window (09:30–12:00 ET)

  Min to trade: min_confluence (default 3)
  Score ≥ 5: risk = risk_high_pct (default 1.2%)
  Score 3-4: risk = risk_base_pct (default 0.8%)

SESSIONS
--------
  London: 03:00–08:00 ET
  NY:     08:30–14:00 ET
  Max 3 trades/day.
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

from strategies.base_strategy import BaseStrategy
from utils.fvg_detector import detect_fvg_at
from utils.structure_detector import compute_swing_arrays, check_mss

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(h, m)


@dataclass
class SetupRecord:
    """One fully-qualified ICT setup (all 4 steps completed)."""
    sweep_bar:       int
    disp_bar:        int
    fvg_bar:         int        # bar i where FVG pattern completes (=i in [i-2,i-1,i])
    entry_bar:       int        # bar where FVG retest triggers signal
    direction:       int        # +1 long, -1 short

    entry_price:     float      # close of retest bar (backtester entry before spread)
    sl_price:        float      # sweep wick ± buffer
    tp1_price:       float      # partial-1 TP (1.5 × SL dist)
    tp2_price:       float      # partial-2 TP (3.0 × SL dist)
    tp_blended:      float      # 2.25 × SL dist (single-exit approximation)
    time_stop:       pd.Timestamp

    # Metrics for filtering in walk-forward
    sweep_size_pips: float
    fvg_size_pips:   float
    confluence:      int
    is_major:        bool

    # Confluence sub-flags
    high_vol:        bool
    large_fvg:       bool
    mss_conf:        bool
    h4_align:        bool
    in_overlap:      bool

    risk_pct:        float       # 0.8 or 1.2 depending on score


# ─────────────────────────────────────────────────────────────────────────────
# Strategy class
# ─────────────────────────────────────────────────────────────────────────────

class GoldIctSweepFvg(BaseStrategy):
    """Gold XAUUSD M1 ICT Sweep-to-FVG strategy."""

    @property
    def name(self) -> str:
        return "GoldIctSweepFvg"

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        self._london_start = _parse_time(config.get("london_entry_start", "03:00"))
        self._london_end   = _parse_time(config.get("london_entry_end",   "08:00"))
        self._ny_start     = _parse_time(config.get("ny_entry_start",     "08:30"))
        self._ny_end       = _parse_time(config.get("ny_entry_end",       "14:00"))
        self._overlap_start = _parse_time(config.get("overlap_start",     "09:30"))
        self._overlap_end   = _parse_time(config.get("overlap_end",       "12:00"))

        self._sweep_min_pips  = float(config.get("sweep_min_pips",     15.0))
        self._sl_buffer_pips  = float(config.get("sl_buffer_pips",      5.0))
        self._sl_cap_pips     = float(config.get("sl_cap_pips",        300.0))
        self._fvg_min_pips    = float(config.get("fvg_min_pips",        10.0))
        self._fvg_timeout     = int(config.get("fvg_retest_timeout",    30))
        self._disp_timeout    = int(config.get("disp_search_bars",       5))
        self._body_atr_min    = float(config.get("disp_body_atr_min",    0.5))
        self._outer_frac      = float(config.get("disp_outer_fraction",  0.75))
        self._vol_mult        = float(config.get("vol_mult",             1.3))
        self._vol_sma_period  = int(config.get("vol_sma_period",        20))
        self._large_fvg_pips  = float(config.get("large_fvg_pips",      20.0))
        self._rr_partial1     = float(config.get("rr_partial1",          1.5))
        self._rr_partial2     = float(config.get("rr_partial2",          3.0))
        self._min_confluence  = int(config.get("min_confluence",          3))
        self._max_trades_day  = int(config.get("max_trades_per_day",     3))
        self._no_friday       = bool(config.get("no_friday_trading",    True))

        self._risk_base_pct   = float(config.get("risk_base_pct",        0.8))
        self._risk_high_pct   = float(config.get("risk_high_pct",        1.2))
        self.risk_per_trade_pct = self._risk_base_pct  # default for backtester auto-sizing

        # H4 trend
        self._h4_ema_period   = int(config.get("h4_ema_period",         20))
        self._h4_slope_bars   = int(config.get("h4_slope_bars",          3))

        # M15 swing for level map
        self._swing_lookback  = int(config.get("swing_lookback_bars",    3))
        self._max_swing_levels = int(config.get("max_swing_levels",      4))

        # Regime: daily ATR(14) > X-th pct of prior 60 days
        self._regime_atr_pct  = float(config.get("regime_atr_percentile", 30.0))

        self._pip_size        = float(instrument_config["pip_size"])

        logger.info(
            "%s setup: sweep_min=%g  fvg_min=%g  min_conf=%d  max_T=%d/day",
            self.name, self._sweep_min_pips, self._fvg_min_pips,
            self._min_confluence, self._max_trades_day,
        )

    # ------------------------------------------------------------------ #
    # generate_signals — backtester interface                              #
    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard backtester interface using BLENDED 2.25R TP."""
        df = self._init_signal_columns(df)

        setups, _ = self.detect_setups(df)
        if not setups:
            return df

        n = len(df)

        # Extra diagnostic columns
        df["confluence"]    = 0
        df["fvg_size_pips"] = np.nan
        df["sl_pips"]       = np.nan

        for s in setups:
            if s.entry_bar >= n:
                continue
            df.at[df.index[s.entry_bar], "signal"]      = s.direction
            df.at[df.index[s.entry_bar], "sl_price"]    = s.sl_price
            df.at[df.index[s.entry_bar], "tp_price"]    = s.tp_blended
            df.at[df.index[s.entry_bar], "time_stop"]   = s.time_stop
            df.at[df.index[s.entry_bar], "confluence"]  = s.confluence
            df.at[df.index[s.entry_bar], "fvg_size_pips"] = s.fvg_size_pips
            sl_dist = abs(s.entry_price - s.sl_price) / self._pip_size
            df.at[df.index[s.entry_bar], "sl_pips"]    = sl_dist

        n_sig = int((df["signal"] != 0).sum())
        logger.info("%s: %d signals from %d setups", self.name, n_sig, len(setups))
        return df

    # ------------------------------------------------------------------ #
    # detect_setups — 4-step ICT funnel                                   #
    # ------------------------------------------------------------------ #

    def detect_setups(
        self,
        df: pd.DataFrame,
        min_sweep_pips:  Optional[float] = None,
        min_fvg_pips:    Optional[float] = None,
        min_confluence:  Optional[int]   = None,
    ) -> Tuple[List[SetupRecord], Dict]:
        """
        Scan the M1 bar array through the 4-step funnel.

        Parameters use instance defaults when None.
        Returns (list_of_setups, funnel_counts).

        funnel_counts = {
          "sweeps": N detected sweeps,
          "disp_fvg": N that produced displacement + FVG,
          "retests": N where FVG was retested (= potential trades),
          "traded": N that passed min_confluence filter,
        }
        """
        sweep_min_p = (min_sweep_pips or self._sweep_min_pips) * self._pip_size
        fvg_min_p   = (min_fvg_pips or self._fvg_min_pips)    * self._pip_size
        min_conf    = min_confluence if min_confluence is not None else self._min_confluence

        n   = len(df)
        idx = df.index

        # ── Raw arrays ────────────────────────────────────────────────
        high_a  = df["high"].to_numpy(dtype=float)
        low_a   = df["low"].to_numpy(dtype=float)
        close_a = df["close"].to_numpy(dtype=float)
        open_a  = df["open"].to_numpy(dtype=float)
        vol_a   = df["volume"].to_numpy(dtype=float)

        # ── M1 ATR(60) ────────────────────────────────────────────────
        atr_a = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=60, fillna=False,
        ).average_true_range().to_numpy(dtype=float)

        # ── Volume SMA(20) ────────────────────────────────────────────
        # Only use non-zero volume bars to compute SMA
        vol_series = pd.Series(vol_a, dtype=float)
        vol_sma_a  = vol_series.where(vol_series > 0).rolling(
            self._vol_sma_period, min_periods=5
        ).mean().to_numpy(dtype=float)

        # ── Level map (M15 swing levels + PDH/PDL + Asian H/L) ────────
        levels_by_bar = self._build_level_map(df)

        # ── H4 EMA trend (+1/0/-1 aligned to M1) ─────────────────────
        h4_trend = self._compute_h4_trend(df)

        # ── M1 swing arrays for MSS detection ─────────────────────────
        last_sh, last_sl = compute_swing_arrays(high_a, low_a, self._swing_lookback)

        # ── Regime: daily ATR percentile ──────────────────────────────
        regime_ok = self._compute_regime(df)

        # ── Session masks (US/Eastern) ────────────────────────────────
        bar_h   = np.array([t.hour + t.minute / 60.0 for t in idx])
        ls, le  = self._london_start.hour + self._london_start.minute / 60, \
                  self._london_end.hour   + self._london_end.minute   / 60
        ns, ne  = self._ny_start.hour     + self._ny_start.minute     / 60, \
                  self._ny_end.hour       + self._ny_end.minute       / 60
        os_, oe = self._overlap_start.hour + self._overlap_start.minute / 60, \
                  self._overlap_end.hour   + self._overlap_end.minute   / 60
        in_session = ((bar_h >= ls) & (bar_h < le)) | ((bar_h >= ns) & (bar_h < ne))
        in_overlap  = (bar_h >= os_) & (bar_h < oe)

        # ── State machine ─────────────────────────────────────────────
        setups: List[SetupRecord] = []
        funnel = {"sweeps": 0, "disp_fvg": 0, "retests": 0, "traded": 0}

        phase            = "idle"   # idle | sweep | retest
        sweep_bar        = -1
        sweep_dir        = 0
        sweep_wick       = 0.0
        sweep_level      = 0.0
        sweep_is_major   = False
        disp_bar_        = -1
        fvg_top_         = 0.0
        fvg_bottom_      = 0.0
        fvg_size_pips_   = 0.0
        fvg_bar_         = -1
        fvg_conf_bar     = -1    # bar at which we start watching for retest
        mss_confirmed    = False
        pre_sweep_sh     = 0.0
        pre_sweep_sl     = 0.0
        h4_at_sweep      = 0
        sweep_disp_vol   = 0.0
        sweep_size_pips_ = 0.0
        disp_timeout_bar = -1
        retest_timeout   = -1

        trades_today     = 0
        prev_date        = None

        for i in range(2, n - 1):
            cur_date = idx[i].date()
            if prev_date != cur_date:
                trades_today = 0
                prev_date    = cur_date

            if self._no_friday and idx[i].dayofweek == 4:
                if phase != "idle":
                    phase = "idle"
                continue

            # ── PHASE: IDLE — look for sweep ──────────────────────────
            if phase == "idle":
                if trades_today >= self._max_trades_day:
                    continue
                if not in_session[i] or not regime_ok[i]:
                    continue

                levels = levels_by_bar.get(i, [])
                for level_price, level_type, is_high in levels:
                    if is_high:
                        # BEARISH: wick above level, close below
                        if (high_a[i] > level_price + sweep_min_p and
                                close_a[i] < level_price):
                            phase          = "sweep"
                            sweep_bar      = i
                            sweep_dir      = -1
                            sweep_wick     = high_a[i]
                            sweep_level    = level_price
                            sweep_is_major = level_type in ("PDH", "PDL", "ASH", "ASL")
                            pre_sweep_sh   = float(last_sh[i])
                            pre_sweep_sl   = float(last_sl[i])
                            h4_at_sweep    = int(h4_trend[i])
                            mss_confirmed  = False
                            sweep_size_pips_ = (high_a[i] - level_price) / self._pip_size
                            disp_timeout_bar = i + self._disp_timeout + 1
                            funnel["sweeps"] += 1
                            break
                    else:
                        # BULLISH: wick below level, close above
                        if (low_a[i] < level_price - sweep_min_p and
                                close_a[i] > level_price):
                            phase          = "sweep"
                            sweep_bar      = i
                            sweep_dir      = +1
                            sweep_wick     = low_a[i]
                            sweep_level    = level_price
                            sweep_is_major = level_type in ("PDH", "PDL", "ASH", "ASL")
                            pre_sweep_sh   = float(last_sh[i])
                            pre_sweep_sl   = float(last_sl[i])
                            h4_at_sweep    = int(h4_trend[i])
                            mss_confirmed  = False
                            sweep_size_pips_ = (level_price - low_a[i]) / self._pip_size
                            disp_timeout_bar = i + self._disp_timeout + 1
                            funnel["sweeps"] += 1
                            break

            # ── PHASE: SWEEP — look for displacement + FVG ───────────
            elif phase == "sweep":
                if i > disp_timeout_bar:
                    phase = "idle"
                    continue

                # Candidate 3-bar pattern: [i-2, i-1, i]
                # Displacement is bar i-1 (middle)
                disp = i - 1
                pre  = i - 2
                if disp <= sweep_bar:
                    continue

                # ── Displacement check (bar i-1) ──────────────────────
                atr_disp = atr_a[disp]
                if np.isnan(atr_disp) or atr_disp == 0:
                    continue

                body = abs(close_a[disp] - open_a[disp])
                if body < self._body_atr_min * atr_disp:
                    continue

                rng = high_a[disp] - low_a[disp]
                if rng > 0:
                    if sweep_dir == 1:   # bullish disp: close in top 25%
                        outer_ok = close_a[disp] > low_a[disp] + self._outer_frac * rng
                    else:                # bearish disp: close in bottom 25%
                        outer_ok = close_a[disp] < high_a[disp] - self._outer_frac * rng
                else:
                    outer_ok = False

                if not outer_ok:
                    continue

                # ── Direction consistency ─────────────────────────────
                if sweep_dir == 1 and close_a[disp] < open_a[disp]:
                    continue  # expected up candle
                if sweep_dir == -1 and close_a[disp] > open_a[disp]:
                    continue  # expected down candle

                # ── FVG check at bar [pre, disp, i] ──────────────────
                fvg = detect_fvg_at(high_a, low_a, i, fvg_min_p, self._pip_size)
                if fvg is None or fvg.direction != sweep_dir:
                    continue

                # ── FVG confirmed! ────────────────────────────────────
                disp_bar_      = disp
                fvg_top_       = fvg.zone_top
                fvg_bottom_    = fvg.zone_bottom
                fvg_size_pips_ = fvg.size / self._pip_size
                fvg_bar_       = i
                fvg_conf_bar   = i          # start watching for retest on next bar
                retest_timeout = i + self._fvg_timeout
                sweep_disp_vol = vol_a[disp]
                phase          = "retest"
                funnel["disp_fvg"] += 1

            # ── PHASE: RETEST — wait for FVG touch ───────────────────
            elif phase == "retest":
                if i > retest_timeout or trades_today >= self._max_trades_day:
                    phase = "idle"
                    continue

                # Update MSS
                if not mss_confirmed:
                    mss_confirmed = check_mss(
                        close_a[i], high_a[i], low_a[i],
                        sweep_dir, pre_sweep_sh, pre_sweep_sl,
                    )

                # Retest detection
                if sweep_dir == 1:   # bullish FVG: price comes back DOWN
                    in_zone = low_a[i] <= fvg_top_
                    # >75% consumed = close below zone_bottom + 0.25 × size
                    threshold = fvg_bottom_ + 0.25 * (fvg_top_ - fvg_bottom_)
                    not_overfilled = close_a[i] >= threshold
                else:                # bearish FVG: price comes back UP
                    in_zone = high_a[i] >= fvg_bottom_
                    threshold = fvg_top_ - 0.25 * (fvg_top_ - fvg_bottom_)
                    not_overfilled = close_a[i] <= threshold

                if not (in_zone and not_overfilled):
                    continue

                funnel["retests"] += 1

                # ── Compute SL / TP ───────────────────────────────────
                entry = close_a[i]
                sl_buf = self._sl_buffer_pips * self._pip_size
                if sweep_dir == 1:
                    sl_price = sweep_wick - sl_buf
                    sl_dist  = entry - sl_price
                else:
                    sl_price = sweep_wick + sl_buf
                    sl_dist  = sl_price - entry

                if sl_dist <= 0 or sl_dist > self._sl_cap_pips * self._pip_size:
                    phase = "idle"
                    continue

                # SL must be on correct side
                if sweep_dir == 1 and sl_price >= entry:
                    phase = "idle"
                    continue
                if sweep_dir == -1 and sl_price <= entry:
                    phase = "idle"
                    continue

                tp1      = entry + sweep_dir * self._rr_partial1 * sl_dist
                tp2      = entry + sweep_dir * self._rr_partial2 * sl_dist
                blended_r = (self._rr_partial1 + self._rr_partial2) / 2.0
                tp_blend = entry + sweep_dir * blended_r * sl_dist

                # ── Confluence scoring ────────────────────────────────
                high_vol_f = (
                    not np.isnan(vol_sma_a[disp_bar_])
                    and vol_sma_a[disp_bar_] > 0
                    and vol_a[disp_bar_] > self._vol_mult * vol_sma_a[disp_bar_]
                )
                large_fvg_f = fvg_size_pips_ >= self._large_fvg_pips
                h4_align_f  = (h4_at_sweep == sweep_dir)
                overlap_f   = bool(in_overlap[i])

                score = (int(sweep_is_major) + int(high_vol_f) + int(large_fvg_f)
                         + int(mss_confirmed) + int(h4_align_f) + int(overlap_f))

                if score < min_conf:
                    phase = "idle"
                    continue

                risk_pct = self._risk_high_pct if score >= 5 else self._risk_base_pct

                # ── Time stop: 16:00 ET on entry day ─────────────────
                try:
                    ts = idx[i].replace(hour=16, minute=0, second=0, microsecond=0)
                except Exception:
                    ts = idx[i] + pd.Timedelta(hours=2)

                rec = SetupRecord(
                    sweep_bar=sweep_bar,
                    disp_bar=disp_bar_,
                    fvg_bar=fvg_bar_,
                    entry_bar=i,
                    direction=sweep_dir,
                    entry_price=entry,
                    sl_price=sl_price,
                    tp1_price=tp1,
                    tp2_price=tp2,
                    tp_blended=tp_blend,
                    time_stop=ts,
                    sweep_size_pips=sweep_size_pips_,
                    fvg_size_pips=fvg_size_pips_,
                    confluence=score,
                    is_major=sweep_is_major,
                    high_vol=high_vol_f,
                    large_fvg=large_fvg_f,
                    mss_conf=mss_confirmed,
                    h4_align=h4_align_f,
                    in_overlap=overlap_f,
                    risk_pct=risk_pct,
                )
                setups.append(rec)
                funnel["traded"] += 1
                trades_today += 1
                phase = "idle"

        logger.info("%s funnel: %s", self.name, funnel)
        return setups, funnel

    # ------------------------------------------------------------------ #
    # M15 Level Map (PDH/PDL + Asian H/L + M15 swings)                    #
    # ------------------------------------------------------------------ #

    def _build_level_map(self, df_m1: pd.DataFrame) -> Dict[int, List]:
        """
        Returns {m1_bar_index: [(price, label, is_high), ...]} for active
        liquidity levels at each bar.  UTC-based to avoid DST floor() issues.
        """
        ps     = self._pip_size
        lk     = self._swing_lookback
        max_sw = self._max_swing_levels

        # Convert to UTC for safe floor/resample ops
        df_utc = df_m1.copy()
        if df_utc.index.tzinfo is not None:
            df_utc.index = df_utc.index.tz_convert("UTC")
        idx_utc = df_utc.index

        m1_high = df_m1["high"].to_numpy(dtype=float)
        m1_low  = df_m1["low"].to_numpy(dtype=float)
        # Use original ET index for session-based hour logic
        idx_et  = df_m1.index
        m1_dates = np.array(idx_et.date)
        m1_hours = idx_et.hour

        # ── M15 swing levels ──────────────────────────────────────────
        m15 = df_utc.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])
        m15_times = m15.index
        m15_highs = m15["high"].to_numpy(dtype=float)
        m15_lows  = m15["low"].to_numpy(dtype=float)
        nm15      = len(m15)

        confirmed_swings: List[Tuple] = []
        for ii in range(lk, nm15 - lk):
            if (m15_highs[ii] > m15_highs[max(0,ii-lk):ii].max() and
                    m15_highs[ii] > m15_highs[ii+1:ii+lk+1].max()):
                confirmed_swings.append((m15_times[ii + lk], m15_highs[ii], True))
            if (m15_lows[ii] < m15_lows[max(0,ii-lk):ii].min() and
                    m15_lows[ii] < m15_lows[ii+1:ii+lk+1].min()):
                confirmed_swings.append((m15_times[ii + lk], m15_lows[ii], False))
        confirmed_swings.sort(key=lambda x: x[0])

        active_highs: List[float] = []
        active_lows:  List[float] = []
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

        # ── PDH/PDL and Asian H/L ──────────────────────────────────────
        unique_dates = sorted(set(m1_dates))
        date_set = set(unique_dates)
        date_pos_map: Dict = defaultdict(list)
        for pi in range(len(df_m1)):
            date_pos_map[m1_dates[pi]].append(pi)

        daily_extra: Dict = {}
        for trade_date in unique_dates:
            prev_date = None
            for delta in range(1, 8):
                cand = trade_date - _dt.timedelta(days=delta)
                if cand in date_set:
                    prev_date = cand
                    break
            if prev_date is None:
                daily_extra[trade_date] = []
                continue
            prev_pos  = date_pos_map[prev_date]
            today_pos = date_pos_map[trade_date]
            pdh = float(m1_high[prev_pos].max())
            pdl = float(m1_low[prev_pos].min())
            lvls = [(pdh, "PDH", True), (pdl, "PDL", False)]
            prev_eve  = [p for p in prev_pos  if m1_hours[p] >= 19]
            today_pre = [p for p in today_pos if m1_hours[p] < 2]
            asian_pos = prev_eve + today_pre
            if asian_pos:
                lvls += [(float(m1_high[asian_pos].max()), "ASH", True),
                         (float(m1_low[asian_pos].min()),  "ASL", False)]
            daily_extra[trade_date] = lvls

        # ── Map to M1 bar indices ──────────────────────────────────────
        m1_floor = idx_utc.floor("15min")
        result: Dict[int, List] = {}
        for pi in range(len(df_m1)):
            fl = m1_floor[pi]
            m15_entry = m15_active.get(fl)
            if m15_entry is None:
                cands = [t for t in m15_active if t <= fl]
                if not cands:
                    continue
                m15_entry = m15_active[max(cands)]
            h_lvls, l_lvls = m15_entry
            all_lvls = (
                [(p, "SWH", True)  for p in h_lvls] +
                [(p, "SWL", False) for p in l_lvls] +
                daily_extra.get(m1_dates[pi], [])
            )
            if all_lvls:
                result[pi] = all_lvls
        return result

    # ------------------------------------------------------------------ #
    # H4 trend                                                             #
    # ------------------------------------------------------------------ #

    def _compute_h4_trend(self, df: pd.DataFrame) -> np.ndarray:
        h4_close = df["close"].resample("4h").last().dropna()
        if len(h4_close) < self._h4_ema_period + self._h4_slope_bars + 2:
            return np.zeros(len(df), dtype=int)
        ema   = h4_close.ewm(span=self._h4_ema_period, adjust=False).mean()
        slope = ema - ema.shift(self._h4_slope_bars)
        h4_trend = pd.Series(0, index=h4_close.index, dtype=int)
        h4_trend[slope > 0] =  1
        h4_trend[slope < 0] = -1
        return (h4_trend.reindex(df.index, method="ffill")
                .fillna(0).astype(int).to_numpy())

    # ------------------------------------------------------------------ #
    # Daily regime: ATR(14) above percentile                              #
    # ------------------------------------------------------------------ #

    def _compute_regime(self, df: pd.DataFrame) -> np.ndarray:
        """True where daily ATR(14) >= X-th percentile of prior 60 days."""
        daily_ohlc = df.resample("D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna(subset=["open"])
        d_atr = ta.volatility.AverageTrueRange(
            high=daily_ohlc["high"], low=daily_ohlc["low"],
            close=daily_ohlc["close"], window=14, fillna=False,
        ).average_true_range().to_numpy(dtype=float)

        ok_arr = np.ones(len(daily_ohlc), dtype=bool)
        lb = 60
        pct = self._regime_atr_pct
        for ii in range(lb, len(d_atr)):
            if np.isnan(d_atr[ii]):
                continue
            prior = d_atr[max(0, ii - lb) : ii]
            prior = prior[~np.isnan(prior)]
            if len(prior) < 10:
                continue
            ok_arr[ii] = d_atr[ii] >= float(np.percentile(prior, pct))

        d_regime = pd.Series(ok_arr, index=daily_ohlc.index)
        return (d_regime.reindex(df.index, method="ffill")
                .fillna(True).astype(bool).to_numpy())
