"""
Market Structure Shift (MSS) / Change of Character (CHoCH) Detection
======================================================================

Tracks M1 swing highs and lows (confirmed by N bars on each side).

Bullish MSS: price breaks ABOVE the most recent confirmed swing high
             after having made a lower low (shift from down to up structure)
Bearish MSS: price breaks BELOW the most recent confirmed swing low
             after having made a higher high (shift from up to down)

Returns precomputed arrays aligned to the bar index:
  last_sh[i] = price of most recent confirmed swing high as of bar i
  last_sl[i] = price of most recent confirmed swing low as of bar i
"""

from __future__ import annotations

import numpy as np


def compute_swing_arrays(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute rolling last-confirmed-swing-high and last-confirmed-swing-low
    for each bar in the series.

    A swing high at bar k is confirmed at bar k + lookback:
      high[k] > max(high[k-lookback : k]) AND high[k] > max(high[k+1 : k+lookback+1])

    Returns
    -------
    last_sh : ndarray, shape (n,) — most recent confirmed swing high price
    last_sl : ndarray, shape (n,) — most recent confirmed swing low price
    (NaN before any swing is confirmed)
    """
    n = len(high)
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)

    cur_sh = np.nan
    cur_sl = np.nan

    for i in range(lookback, n - lookback):
        # Swing candidate is at bar i; confirmed when we reach bar i + lookback
        k = i - lookback    # the candidate bar
        if k >= lookback:
            pre_h  = high[k - lookback : k]
            post_h = high[k + 1 : k + lookback + 1]
            if len(pre_h) > 0 and len(post_h) > 0:
                if high[k] > pre_h.max() and high[k] > post_h.max():
                    cur_sh = high[k]

            pre_l  = low[k - lookback : k]
            post_l = low[k + 1 : k + lookback + 1]
            if len(pre_l) > 0 and len(post_l) > 0:
                if low[k] < pre_l.min() and low[k] < post_l.min():
                    cur_sl = low[k]

        last_sh[i] = cur_sh
        last_sl[i] = cur_sl

    # Forward-fill remaining tail
    for i in range(n - lookback, n):
        last_sh[i] = cur_sh
        last_sl[i] = cur_sl

    return last_sh, last_sl


def check_mss(
    close_price: float,
    high_price: float,
    low_price: float,
    direction: int,
    pre_sweep_sh: float,
    pre_sweep_sl: float,
) -> bool:
    """
    Check if a Market Structure Shift has occurred for the current bar.

    Parameters
    ----------
    direction     : +1 (long setup) or -1 (short setup)
    pre_sweep_sh  : last confirmed swing high price before the sweep
    pre_sweep_sl  : last confirmed swing low price before the sweep

    Returns True if MSS confirmed.
    """
    if direction == 1:
        # Bullish setup: MSS = price breaks ABOVE pre-sweep swing high
        if not np.isnan(pre_sweep_sh) and high_price > pre_sweep_sh:
            return True
    else:
        # Bearish setup: MSS = price breaks BELOW pre-sweep swing low
        if not np.isnan(pre_sweep_sl) and low_price < pre_sweep_sl:
            return True
    return False
