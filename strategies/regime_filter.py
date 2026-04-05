"""
Regime Filter
=============

Purpose
-------
Classify current market conditions into one of four regimes and return
which strategy names should be active for a given trading day.  This
prevents running strategies in conditions where they structurally
underperform.

Regime logic (all thresholds from strategy_params.json → regime_filter)
------------------------------------------------------------------------
  HIGH_VOL   ATR%ile > high_vol_percentile (70)
             Enable LondonOpenBreakout only.
             FVG disabled: high-vol gaps are blown through before
             price can retrace to the entry zone.

  NORMAL     high_vol_percentile >= ATR%ile >= low_vol_percentile (30-70)
             Enable BOTH strategies.

  LOW_VOL    ATR%ile < low_vol_percentile (30)
             Disable LondonOpenBreakout: Asian ranges are too tight for
             meaningful breakouts → high false-signal rate.
             Enable FVGRetracement only.

  DEAD       ATR%ile < dead_percentile_threshold (15)
             AND ADX < adx_ranging_threshold (15 / 20)
             Disable ALL.  Price is not moving enough for either strategy.

ATR / ADX
---------
Both are calculated on *daily* candles resampled from 15-min data.

  Daily OHLCV: resample 15-min open/high/low/close/volume to calendar day.
  ATR(14):     Wilder's True Range average.
  ADX(14):     Wilder's Directional Movement average.

ATR percentile
--------------
  Compare today's ATR against the rolling distribution of the previous
  `atr_lookback_days` (60) trading days.

Public API
----------
  RegimeFilter.setup(config)
  RegimeFilter.get_regime(daily_df, date)   -> str
  RegimeFilter.get_active_strategies(daily_df, date) -> list[str]

Where `daily_df` is an OHLCV DataFrame with a DatetimeIndex at daily
frequency (already resampled, or a 15-min DataFrame which will be
resampled internally).
"""

from __future__ import annotations

import json
import logging
from datetime import date as Date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"

# Canonical strategy name strings (match strategy.name properties)
LONDON_BREAKOUT = "LondonOpenBreakout"
FVG_RETRACEMENT = "FVGRetracement"

REGIME_HIGH_VOL = "high_vol"
REGIME_NORMAL   = "normal"
REGIME_LOW_VOL  = "low_vol"
REGIME_DEAD     = "dead"


class RegimeFilter:
    """
    Classifies daily market regimes and gates strategy activation.

    Usage
    -----
    rf = RegimeFilter()
    rf.setup(config)                          # pass regime_filter sub-dict

    # Pass either a 15-min or already-daily OHLCV DataFrame
    regime = rf.get_regime(df_15min, today)
    strategies = rf.get_active_strategies(df_15min, today)
    """

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, config: Dict[str, Any]) -> None:
        self._atr_period:          int   = int(config["atr_period"])           # 14
        self._atr_lookback_days:   int   = int(config["atr_lookback_days"])    # 60
        self._high_vol_pct:        float = float(config["high_vol_percentile"]) # 70
        self._low_vol_pct:         float = float(config["low_vol_percentile"])  # 30
        # Dead-market threshold: ATR%ile below this AND ADX below adx threshold
        self._dead_atr_pct:        float = float(config.get("dead_percentile_threshold", 15.0))
        self._adx_period:          int   = int(config["adx_period"])           # 14
        self._adx_trending:        float = float(config["adx_trending_threshold"]) # 25
        self._adx_ranging:         float = float(config["adx_ranging_threshold"])  # 20

        logger.info(
            "RegimeFilter setup: ATR%ile hi=%.0f lo=%.0f dead=%.0f | ADX ranging=%.0f trending=%.0f",
            self._high_vol_pct, self._low_vol_pct, self._dead_atr_pct,
            self._adx_ranging, self._adx_trending,
        )

    @classmethod
    def from_config_file(cls) -> "RegimeFilter":
        """Load directly from strategy_params.json (convenience for main.py)."""
        rf = cls()
        with open(_CONFIG_DIR / "strategy_params.json") as fh:
            rf.setup(json.load(fh)["regime_filter"])
        return rf

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_regime(self, df: pd.DataFrame, query_date: Date) -> str:
        """
        Classify market conditions on `query_date`.

        Parameters
        ----------
        df :
            OHLCV DataFrame.  Can be 15-min (will be resampled to daily)
            or already at daily frequency.  Must contain at least
            atr_lookback_days + atr_period prior trading days.
        query_date :
            The calendar date to classify.

        Returns
        -------
        str : one of REGIME_HIGH_VOL, REGIME_NORMAL, REGIME_LOW_VOL, REGIME_DEAD
        """
        daily = self._ensure_daily(df)
        atr_pct, adx_val = self._compute_indicators(daily, query_date)

        if atr_pct is None:
            logger.warning("Insufficient history for regime on %s — defaulting to NORMAL", query_date)
            return REGIME_NORMAL

        regime = self._classify(atr_pct, adx_val)
        logger.debug(
            "Regime %s | ATR%%ile=%.1f ADX=%.1f | date=%s",
            regime, atr_pct, adx_val if adx_val is not None else float("nan"), query_date,
        )
        return regime

    def get_active_strategies(self, df: pd.DataFrame, query_date: Date) -> List[str]:
        """
        Return list of strategy names that should run on `query_date`.

        Possible return values
        ----------------------
        [LONDON_BREAKOUT, FVG_RETRACEMENT]  — NORMAL
        [LONDON_BREAKOUT]                   — HIGH_VOL
        [FVG_RETRACEMENT]                   — LOW_VOL
        []                                  — DEAD
        """
        regime = self.get_regime(df, query_date)
        active = _regime_to_strategies(regime)
        logger.info("Regime=%s → active strategies: %s | date=%s", regime, active, query_date)
        return active

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If `df` has sub-daily frequency (≤ 1h bars), resample to daily OHLCV.
        Otherwise return as-is (assumed already daily).
        """
        if df.empty:
            return df

        # Detect frequency by checking median gap between consecutive bars
        if len(df) < 2:
            return df

        median_gap = pd.Series(df.index).diff().median()
        if median_gap < pd.Timedelta(hours=20):
            # Sub-daily → resample to calendar day
            daily = df.resample("D").agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna(subset=["open", "close"])
            return daily
        return df

    def _compute_indicators(
        self,
        daily: pd.DataFrame,
        query_date: Date,
    ):
        """
        Return (atr_percentile, adx_value) for `query_date`.
        Returns (None, None) if insufficient history.
        """
        # Restrict to data up to and including query_date
        if daily.index.tz is not None:
            cutoff = pd.Timestamp(query_date, tz=daily.index.tz)
        else:
            cutoff = pd.Timestamp(query_date)

        hist = daily[daily.index <= cutoff].copy()

        min_bars = self._atr_period + self._atr_lookback_days
        if len(hist) < min_bars:
            return None, None

        atr_series = _compute_atr(hist, self._atr_period)
        adx_series = _compute_adx(hist, self._adx_period)

        # Find the row corresponding to query_date
        date_mask = hist.index.date == query_date
        if not date_mask.any():
            return None, None

        today_atr = float(atr_series[date_mask].iloc[-1])

        # ATR percentile: compare today's ATR to the last N trading days
        lookback = atr_series[~date_mask].dropna().tail(self._atr_lookback_days)
        if lookback.empty:
            return None, None
        atr_pct = float(np.sum(lookback < today_atr) / len(lookback) * 100.0)

        # ADX for today (None if not enough history)
        adx_val: Optional[float] = None
        if adx_series is not None and date_mask.any():
            adx_today = adx_series[date_mask]
            if not adx_today.empty and not np.isnan(adx_today.iloc[-1]):
                adx_val = float(adx_today.iloc[-1])

        return atr_pct, adx_val

    def _classify(self, atr_pct: float, adx_val: Optional[float]) -> str:
        """Map (atr_percentile, adx) → regime string."""
        # Dead market: very low volatility AND very low ADX (price not moving)
        if atr_pct < self._dead_atr_pct:
            if adx_val is None or adx_val < self._adx_ranging:
                return REGIME_DEAD

        if atr_pct > self._high_vol_pct:
            return REGIME_HIGH_VOL

        if atr_pct < self._low_vol_pct:
            return REGIME_LOW_VOL

        return REGIME_NORMAL


# ---------------------------------------------------------------------------
# Strategy gate
# ---------------------------------------------------------------------------

def _regime_to_strategies(regime: str) -> List[str]:
    return {
        REGIME_HIGH_VOL: [LONDON_BREAKOUT],
        REGIME_NORMAL:   [LONDON_BREAKOUT, FVG_RETRACEMENT],
        REGIME_LOW_VOL:  [FVG_RETRACEMENT],
        REGIME_DEAD:     [],
    }.get(regime, [LONDON_BREAKOUT, FVG_RETRACEMENT])


# ---------------------------------------------------------------------------
# ATR (Wilder's smoothed)
# ---------------------------------------------------------------------------

def _compute_atr(daily: pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder's ATR:
      TR  = max(high-low, |high-prev_close|, |low-prev_close|)
      ATR = EMA(TR, span=period) with adjust=False (Wilder's smoothing)
    """
    high  = daily["high"]
    low   = daily["low"]
    close = daily["close"]
    prev  = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr


# ---------------------------------------------------------------------------
# ADX (Wilder's smoothed)
# ---------------------------------------------------------------------------

def _compute_adx(daily: pd.DataFrame, period: int) -> Optional[pd.Series]:
    """
    Wilder's ADX:
      +DM = max(high - prev_high, 0) if > |low - prev_low| else 0
      -DM = max(prev_low - low, 0)  if > |high - prev_high| else 0
      ATR14 (Wilder), smoothed +DM14, smoothed -DM14
      +DI = 100 × +DM14 / ATR14
      -DI = 100 × -DM14 / ATR14
      DX  = 100 × |+DI - -DI| / (+DI + -DI)
      ADX = Wilder EMA of DX
    """
    if len(daily) < period * 2:
        return None

    high  = daily["high"]
    low   = daily["low"]
    close = daily["close"]

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr14      = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    plus_dm14  = pd.Series(plus_dm,  index=daily.index).ewm(
        span=period, adjust=False, min_periods=period).mean()
    minus_dm14 = pd.Series(minus_dm, index=daily.index).ewm(
        span=period, adjust=False, min_periods=period).mean()

    plus_di  = 100 * plus_dm14  / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm14 / atr14.replace(0, np.nan)

    di_sum  = (plus_di + minus_di).replace(0, np.nan)
    dx      = 100 * (plus_di - minus_di).abs() / di_sum

    adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()
    return adx
