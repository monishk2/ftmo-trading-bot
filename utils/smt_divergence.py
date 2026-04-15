"""
utils/smt_divergence.py
========================
SMT (Smart Money Technique) divergence between two correlated indices.

Primary use case: NQ (NAS100) vs ES (SP500) on M5 bars.

Bullish SMT (long bias):
  ES makes a LOWER swing low than its most recent prior swing low
  BUT NQ makes a HIGHER swing low → NQ diverges bullish vs ES

Bearish SMT (short bias):
  ES makes a HIGHER swing high than its most recent prior swing high
  BUT NQ makes a LOWER swing high → NQ diverges bearish vs ES

Implementation
--------------
Swing points confirmed by N bars on each side (default N=3).
Computes a precomputed signal array aligned to the bar index of the
PRIMARY instrument (NQ/NAS100).

A new SMT signal is emitted only when:
  1. ES makes a new swing high/low that breaks its prior swing
  2. NQ's most recent swing of the same type DOES NOT confirm (diverges)
  3. Signal stays active for smt_decay_bars (default 12 bars = 1 hour M5)
     so the IB breakout strategy can still reference it

Returns a numpy array smt_signal aligned to the primary df index:
  +1 = bullish SMT active
  -1 = bearish SMT active
   0 = no signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Swing detection ───────────────────────────────────────────────────────────

def _swing_highs_lows(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect confirmed swing highs and lows.

    A swing high at bar k is confirmed at bar k+lookback when:
      high[k] > all highs in [k-lookback, k) AND (k, k+lookback]

    Returns
    -------
    sh_price : ndarray(n) — confirmed swing high price, NaN otherwise
    sl_price : ndarray(n) — confirmed swing low  price, NaN otherwise
    Confirmation happens at bar k+lookback (the index where we know the swing).
    """
    n = len(high)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)

    for confirm_bar in range(2 * lookback, n):
        k = confirm_bar - lookback   # candidate swing bar
        if k < lookback:
            continue

        pre_h  = high[k - lookback : k]
        post_h = high[k + 1 : confirm_bar + 1]
        if len(pre_h) == 0 or len(post_h) == 0:
            continue
        if high[k] > pre_h.max() and high[k] > post_h.max():
            sh[confirm_bar] = high[k]

        pre_l  = low[k - lookback : k]
        post_l = low[k + 1 : confirm_bar + 1]
        if len(pre_l) == 0 or len(post_l) == 0:
            continue
        if low[k] < pre_l.min() and low[k] < post_l.min():
            sl[confirm_bar] = low[k]

    return sh, sl


# ── SMT divergence computation ───────────────────────────────────────────────

def compute_smt(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    lookback: int = 3,
    smt_decay_bars: int = 12,
) -> np.ndarray:
    """
    Compute SMT divergence: primary = NAS100, secondary = SP500.

    Both DataFrames must be aligned on the same timestamps (inner-join).
    Backtester._prepare_data converts to ET — pass ET-indexed frames.

    Parameters
    ----------
    df_primary   : NAS100 M5 OHLCV, ET index
    df_secondary : SP500  M5 OHLCV, ET index
    lookback     : bars each side for swing confirmation
    smt_decay_bars: how long a SMT signal stays active after emission

    Returns
    -------
    smt_signal : ndarray(n) aligned to df_primary.index
      +1 = bullish SMT (buy bias)
      -1 = bearish SMT (sell bias)
       0 = no signal
    """
    # Align on common timestamps
    common_idx = df_primary.index.intersection(df_secondary.index)
    nq = df_primary.loc[common_idx]
    es = df_secondary.loc[common_idx]
    n  = len(common_idx)

    if n < 2 * lookback + 2:
        return np.zeros(len(df_primary), dtype=int)

    nq_h = nq["high"].to_numpy(float)
    nq_l = nq["low"].to_numpy(float)
    es_h = es["high"].to_numpy(float)
    es_l = es["low"].to_numpy(float)

    # Confirmed swings on both
    nq_sh, nq_sl = _swing_highs_lows(nq_h, nq_l, lookback)
    es_sh, es_sl = _swing_highs_lows(es_h, es_l, lookback)

    # Running tracking
    smt_signal_common = np.zeros(n, dtype=int)
    last_es_sh     = np.nan
    last_es_sl     = np.nan
    last_nq_sh     = np.nan
    last_nq_sl     = np.nan
    active_signal  = 0
    decay_counter  = 0

    for i in range(n):
        # Update last confirmed swings
        if not np.isnan(es_sh[i]):
            # New ES swing high: check for bearish SMT
            if (not np.isnan(last_es_sh)          # prior ES SH exists
                    and es_sh[i] > last_es_sh      # ES makes HIGHER high (potential bearish SMT from NQ side)
                    and not np.isnan(last_nq_sh)
                    and nq_sh[i] < last_nq_sh):    # NQ makes LOWER high → divergence
                active_signal = -1
                decay_counter = smt_decay_bars
            last_es_sh = es_sh[i]

        if not np.isnan(es_sl[i]):
            # New ES swing low: check for bullish SMT
            if (not np.isnan(last_es_sl)
                    and es_sl[i] < last_es_sl      # ES makes LOWER low
                    and not np.isnan(last_nq_sl)
                    and nq_sl[i] > last_nq_sl):    # NQ makes HIGHER low → divergence
                active_signal = +1
                decay_counter = smt_decay_bars
            last_es_sl = es_sl[i]

        # Also update NQ swing tracking
        if not np.isnan(nq_sh[i]):
            last_nq_sh = nq_sh[i]
        if not np.isnan(nq_sl[i]):
            last_nq_sl = nq_sl[i]

        # Decay
        if decay_counter > 0:
            smt_signal_common[i] = active_signal
            decay_counter -= 1
        else:
            active_signal = 0

    # Map back to primary df index (bars not in common_idx get 0)
    result = np.zeros(len(df_primary), dtype=int)
    common_positions = df_primary.index.get_indexer(common_idx)
    valid = common_positions >= 0
    result[common_positions[valid]] = smt_signal_common[valid]
    return result


def precompute_smt_array(
    df_nas: pd.DataFrame,
    df_sp: pd.DataFrame,
    lookback: int = 3,
    smt_decay_bars: int = 12,
) -> pd.Series:
    """
    Convenience wrapper that returns a pd.Series aligned to df_nas.index.
    """
    arr = compute_smt(df_nas, df_sp, lookback=lookback,
                      smt_decay_bars=smt_decay_bars)
    return pd.Series(arr, index=df_nas.index, name="smt_signal")
