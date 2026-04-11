"""
Fair Value Gap (FVG) Detector
==============================

A Fair Value Gap is the price imbalance left by a strong displacement candle.

3-candle pattern:
  Bullish FVG at bar i:  high[i-2] < low[i]    → gap left by upward displacement at bar i-1
  Bearish FVG at bar i:  low[i-2]  > high[i]   → gap left by downward displacement at bar i-1

Zone definitions:
  Bullish: zone_bottom = high[i-2],  zone_top = low[i]
  Bearish: zone_top    = low[i-2],   zone_bottom = high[i]

An FVG is "filled" when price trades through more than 50% of the zone.
Stale after max_age_bars M1 bars (configurable, default 40).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class FVG:
    """Represents a single Fair Value Gap."""
    direction:     int              # +1 = bullish (buy zone), -1 = bearish (sell zone)
    zone_top:      float
    zone_bottom:   float
    creation_bar:  int              # bar index where FVG was confirmed (bar i in pattern)
    creation_time: pd.Timestamp
    filled:        bool = False

    @property
    def midpoint(self) -> float:
        return (self.zone_top + self.zone_bottom) / 2.0

    @property
    def size(self) -> float:
        return self.zone_top - self.zone_bottom

    def fill_pct(self, current_low: float, current_high: float) -> float:
        """Fraction of zone consumed by current bar's range (0–1)."""
        if self.direction == 1:   # bullish: fill = how far below zone_top
            consumed = max(0.0, self.zone_top - current_low)
        else:                     # bearish: fill = how far above zone_bottom
            consumed = max(0.0, current_high - self.zone_bottom)
        return min(1.0, consumed / max(self.size, 1e-9))


def detect_fvg_at(
    high: np.ndarray,
    low: np.ndarray,
    i: int,
    min_size: float,
    pip_size: float,
) -> "FVG | None":
    """
    Check if bars [i-2, i-1, i] form a valid FVG.
    Bar i-1 is the displacement candle (middle of pattern).

    Parameters
    ----------
    high, low : full bar arrays
    i         : index of the third bar of the pattern (must be >= 2)
    min_size  : minimum gap size in price units (min_pips × pip_size)
    pip_size  : instrument pip size

    Returns FVG object or None.
    """
    if i < 2:
        return None

    pre  = i - 2   # bar before displacement
    post = i       # bar after displacement

    # Bullish FVG
    gap_bull = low[post] - high[pre]
    if gap_bull >= min_size:
        return FVG(
            direction=1,
            zone_top=low[post],
            zone_bottom=high[pre],
            creation_bar=i,
            creation_time=pd.NaT,
        )

    # Bearish FVG
    gap_bear = low[pre] - high[post]
    if gap_bear >= min_size:
        return FVG(
            direction=-1,
            zone_top=low[pre],
            zone_bottom=high[post],
            creation_bar=i,
            creation_time=pd.NaT,
        )

    return None


def vectorized_fvg_scan(
    high: np.ndarray,
    low: np.ndarray,
    min_size: float,
) -> dict:
    """
    Scan all bars and return dict {bar_i: FVG} for every FVG found.
    Used for pre-computing active FVGs across the full dataset.
    """
    n = len(high)
    result = {}
    for i in range(2, n):
        gap_bull = low[i] - high[i - 2]
        if gap_bull >= min_size:
            result[i] = FVG(direction=1, zone_top=low[i], zone_bottom=high[i - 2],
                            creation_bar=i, creation_time=pd.NaT)
            continue
        gap_bear = low[i - 2] - high[i]
        if gap_bear >= min_size:
            result[i] = FVG(direction=-1, zone_top=low[i - 2], zone_bottom=high[i],
                            creation_bar=i, creation_time=pd.NaT)
    return result
