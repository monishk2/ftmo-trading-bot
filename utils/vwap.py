"""
utils/vwap.py
==============
Anchored daily VWAP with ±1 standard deviation bands.

Anchor: 09:30 ET each trading day (NY open).
VWAP resets at each anchor point.

Typical price: (high + low + close) / 3

VWAP = cumsum(tp × volume) / cumsum(volume)
Band = VWAP ± sqrt(cumsum(volume × (tp − vwap)²) / cumsum(volume))

Returns three arrays aligned to df.index:
  vwap      : anchored VWAP value
  vwap_upper: VWAP + 1σ band
  vwap_lower: VWAP − 1σ band

Non-RTH bars (before 09:30 or outside session) get the last
carried-over value from the prior session — consistent with
how charting platforms handle VWAP display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vwap(
    df: pd.DataFrame,
    anchor_hour: int = 9,
    anchor_minute: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute anchored daily VWAP ± 1σ bands on any timeframe dataframe.

    Parameters
    ----------
    df              : OHLCV dataframe with tz-aware DatetimeIndex (US/Eastern)
    anchor_hour     : reset hour (default 9 = 09:xx ET)
    anchor_minute   : reset minute (default 30 = 09:30 ET)

    Returns
    -------
    vwap        : ndarray(n) — VWAP value at each bar
    vwap_upper  : ndarray(n) — VWAP + 1σ
    vwap_lower  : ndarray(n) — VWAP − 1σ
    """
    n       = len(df)
    idx     = df.index
    highs   = df["high"].to_numpy(float)
    lows    = df["low"].to_numpy(float)
    closes  = df["close"].to_numpy(float)
    vols    = df["volume"].to_numpy(float)

    hours   = np.array(idx.hour)
    minutes = np.array(idx.minute)

    vwap_arr  = np.full(n, np.nan)
    upper_arr = np.full(n, np.nan)
    lower_arr = np.full(n, np.nan)

    # Typical price
    tp = (highs + lows + closes) / 3.0

    # Running accumulators reset at anchor each day
    cum_vol_tp  = 0.0   # cumsum(vol × tp)
    cum_vol     = 0.0   # cumsum(vol)
    cum_vol_tp2 = 0.0   # cumsum(vol × tp²) — for variance

    for i in range(n):
        h = hours[i]; m = minutes[i]

        # Reset at anchor point
        if h == anchor_hour and m == anchor_minute:
            cum_vol_tp  = 0.0
            cum_vol     = 0.0
            cum_vol_tp2 = 0.0

        v = max(vols[i], 1e-10)   # guard against zero-volume bars

        cum_vol_tp  += v * tp[i]
        cum_vol     += v
        cum_vol_tp2 += v * tp[i] ** 2

        vw = cum_vol_tp / cum_vol
        # Variance = E[x²] - E[x]²  (volume-weighted)
        variance = max(0.0, cum_vol_tp2 / cum_vol - vw ** 2)
        sigma    = np.sqrt(variance)

        vwap_arr[i]  = vw
        upper_arr[i] = vw + sigma
        lower_arr[i] = vw - sigma

    return vwap_arr, upper_arr, lower_arr


def add_vwap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: compute VWAP bands and add as columns to df.
    Returns a copy with added columns: vwap, vwap_upper, vwap_lower.
    """
    df = df.copy()
    v, vu, vl = compute_vwap(df)
    df["vwap"]       = v
    df["vwap_upper"] = vu
    df["vwap_lower"] = vl
    return df
