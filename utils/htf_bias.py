"""
utils/htf_bias.py
==================
Higher-timeframe (M15) directional bias for NAS100 ICT strategy.

Two sources, in priority order:
  1. Asian-session liquidity sweep  (strongest — overrides FVG bias)
  2. M15 FVG / IFVG zone interaction (secondary)

Asian session: 18:00 ET (previous RTH evening) → 09:30 ET (next RTH open)

Bias encoding
-------------
  +1 = bullish
  -1 = bearish
   0 = no bias

Bias is "sticky" within a trading day: once a sweep fires, that direction
holds for the rest of RTH.  FVG bias is re-evaluated bar by bar.

Usage
-----
  bias_series = compute_htf_bias(df_m15, fvg_min_size=5.0)
  # Series aligned to df_m15.index, values in {-1, 0, +1}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.fvg_detector import vectorized_fvg_scan


# ---------------------------------------------------------------------------
# Asian-session level extraction
# ---------------------------------------------------------------------------

def _asian_levels(
    dates: np.ndarray,
    hm: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    rth_dates: list,
) -> tuple[dict, dict]:
    """
    For each RTH date, compute Asian session high/low.

    Asian bars = bars from previous RTH date's 18:00 through
    current RTH date's 09:25 (exclusive of 09:30 bar itself).
    """
    asian_high: dict = {}
    asian_low:  dict = {}

    for j, d in enumerate(rth_dates):
        if j == 0:
            asian_high[d] = np.nan
            asian_low[d]  = np.nan
            continue

        prev_d = rth_dates[j - 1]

        # Evening of previous RTH day (≥18:00)
        mask_eve = (dates == prev_d) & (hm >= 1800)
        # Overnight / early morning of current date (<09:30)
        mask_morn = (dates == d) & (hm < 930)
        mask = mask_eve | mask_morn

        if not mask.any():
            asian_high[d] = np.nan
            asian_low[d]  = np.nan
        else:
            asian_high[d] = float(highs[mask].max())
            asian_low[d]  = float(lows[mask].min())

    return asian_high, asian_low


# ---------------------------------------------------------------------------
# Main bias computation
# ---------------------------------------------------------------------------

def compute_htf_bias(
    df_m15: pd.DataFrame,
    fvg_min_size: float = 5.0,
) -> pd.Series:
    """
    Compute M15 HTF bias aligned to df_m15.index.

    Parameters
    ----------
    df_m15       : M15 OHLCV dataframe, tz-aware US/Eastern index
    fvg_min_size : minimum M15 FVG gap size in price points

    Returns
    -------
    pd.Series[int]  index = df_m15.index, values in {-1, 0, +1}
    """
    idx  = df_m15.index
    n    = len(df_m15)

    highs  = df_m15["high"].to_numpy(float)
    lows   = df_m15["low"].to_numpy(float)
    closes = df_m15["close"].to_numpy(float)

    hours   = np.array(idx.hour)
    minutes = np.array(idx.minute)
    hm      = hours * 100 + minutes
    dates   = np.array(idx.date)

    # ── Identify RTH dates (dates that have a 09:30 bar) ─────────────────────
    rth_dates = sorted({
        dates[i] for i in range(n) if hm[i] == 930
    })

    # ── Asian session levels ──────────────────────────────────────────────────
    a_high, a_low = _asian_levels(dates, hm, highs, lows, rth_dates)

    # ── Sweep bias (sticky within RTH day) ────────────────────────────────────
    sweep_arr = np.zeros(n, dtype=int)
    day_sweep: dict = {}   # date → current sticky sweep bias

    for i in range(n):
        d   = dates[i]
        h_m = hm[i]

        # Only look for sweeps during RTH (09:30 – 16:00)
        if h_m < 930 or h_m >= 1600:
            continue

        if d not in day_sweep:
            day_sweep[d] = 0

        # Reset carried bias at RTH open
        if h_m == 930:
            day_sweep[d] = 0

        ah = a_high.get(d, np.nan)
        al = a_low.get(d, np.nan)

        if not (np.isnan(ah) or np.isnan(al)):
            # New sweep event (only update if not already set)
            if day_sweep[d] == 0:
                if lows[i] < al and closes[i] > al:
                    day_sweep[d] = 1    # bullish sweep
                elif highs[i] > ah and closes[i] < ah:
                    day_sweep[d] = -1   # bearish sweep

        sweep_arr[i] = day_sweep[d]

    # ── FVG / IFVG bias (bar-by-bar, secondary) ───────────────────────────────
    raw_fvgs = vectorized_fvg_scan(highs, lows, min_size=fvg_min_size)

    fvg_bias_arr = np.zeros(n, dtype=int)
    active_fvgs: list[dict] = []   # each: {dir, zt, zb, inverted, age}

    for i in range(n):
        # Register new FVG confirmed at bar i
        if i in raw_fvgs:
            fvg = raw_fvgs[i]
            active_fvgs.append({
                "dir": fvg.direction,
                "zt":  fvg.zone_top,
                "zb":  fvg.zone_bottom,
                "inverted": False,
                "age": 0,
            })

        still_active: list[dict] = []
        fb = 0

        for g in active_fvgs:
            g["age"] += 1
            if g["age"] > 100:       # discard very stale FVGs (100 M15 bars ≈ 25 hrs)
                continue

            zt, zb = g["zt"], g["zb"]
            zone_size = max(zt - zb, 1e-9)

            # Check fill degree
            if not g["inverted"]:
                if g["dir"] == 1:     # bullish FVG: filled by price trading below zone_top
                    fill = max(0.0, zt - lows[i]) / zone_size
                else:                  # bearish FVG: filled by trading above zone_bottom
                    fill = max(0.0, highs[i] - zb) / zone_size

                if fill >= 1.0:
                    # 100% fill → becomes IFVG with flipped direction
                    g["inverted"] = True
                    g["dir"] = -g["dir"]
                    still_active.append(g)
                    continue

            # Price in zone? → contribute bias
            in_zone = (lows[i] <= zt) and (highs[i] >= zb)
            if in_zone:
                if g["dir"] == 1:
                    fb = max(fb, 1)
                else:
                    fb = min(fb, -1)

            still_active.append(g)

        active_fvgs = still_active[-50:]    # cap list size
        fvg_bias_arr[i] = fb

    # ── Combine: sweep overrides FVG ─────────────────────────────────────────
    final = np.zeros(n, dtype=int)
    for i in range(n):
        if sweep_arr[i] != 0:
            final[i] = sweep_arr[i]
        elif fvg_bias_arr[i] != 0:
            final[i] = fvg_bias_arr[i]

    return pd.Series(final, index=idx, name="htf_bias", dtype=int)


# ---------------------------------------------------------------------------
# Convenience: forward-fill M15 bias into a finer-grained index
# ---------------------------------------------------------------------------

def align_bias_to_ltf(
    bias_m15: pd.Series,
    ltf_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Reindex M15 bias into ltf_index with forward-fill.
    Bars before the first M15 bar get bias=0.
    """
    aligned = bias_m15.reindex(ltf_index, method="ffill").fillna(0).astype(int)
    return aligned
