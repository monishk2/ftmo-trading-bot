"""
Position Sizer
==============

Pure function that converts a risk-percentage into a lot size.

Formula
-------
    dollar_risk  = balance × risk_pct / 100
    raw_lots     = dollar_risk / (sl_distance_pips × pip_value_per_lot)
    lot_size     = floor(raw_lots, 2)   ← always rounds DOWN to nearest 0.01

Rounding
--------
Rounding DOWN (never up) ensures we never exceed the intended risk percentage.
A lot size of 0.0 is returned if the balance cannot fund even one micro-lot
(0.01 lots) at the given risk; the caller should skip the trade.

pip_value_per_lot
-----------------
For USD-quoted pairs (EURUSD, GBPUSD): 1 standard lot × 1 pip = $10.
This is the default.  Override by adding "pip_value_per_lot" to
instruments.json → instrument entry.

Public API
----------
    calculate_lot_size(balance, risk_pct, sl_distance_pips, instrument_config)
        -> float   (0.00 to N.NN in increments of 0.01)
"""

from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)

# USD per pip per standard lot (100,000 units) for USD-quoted pairs.
_DEFAULT_PIP_VALUE = 10.0
_MICRO_LOT         = 0.01     # smallest tradeable increment
_MAX_LOTS          = 100.0    # sanity cap — never return more than this


def calculate_lot_size(
    balance: float,
    risk_pct: float,
    sl_distance_pips: float,
    instrument_config: dict,
) -> float:
    """
    Calculate the maximum lot size that keeps risk within risk_pct of balance.

    Parameters
    ----------
    balance :
        Current account balance in USD (realised, closed-trade-only).
    risk_pct :
        Percentage of balance to risk on this trade (e.g. 0.75 for 0.75 %).
    sl_distance_pips :
        Stop-loss distance in pips (positive, direction-agnostic).
    instrument_config :
        Dict from instruments.json for this symbol.  Must contain
        ``pip_size``.  Optionally contains ``pip_value_per_lot`` (default 10.0).

    Returns
    -------
    float
        Lot size rounded down to nearest 0.01.  Returns 0.0 if the position
        would require less than one micro-lot within the risk budget.

    Raises
    ------
    ValueError
        If balance <= 0, risk_pct <= 0, or sl_distance_pips <= 0.
    """
    if balance <= 0:
        raise ValueError(f"balance must be positive, got {balance}")
    if risk_pct <= 0:
        raise ValueError(f"risk_pct must be positive, got {risk_pct}")
    if sl_distance_pips <= 0:
        raise ValueError(f"sl_distance_pips must be positive, got {sl_distance_pips}")

    pip_value = float(instrument_config.get("pip_value_per_lot", _DEFAULT_PIP_VALUE))

    dollar_risk = balance * risk_pct / 100.0
    raw_lots    = dollar_risk / (sl_distance_pips * pip_value)

    # Round DOWN — never risk more than intended
    lot_size = math.floor(raw_lots * 100) / 100.0

    # Clamp to valid range
    lot_size = min(lot_size, _MAX_LOTS)

    if lot_size < _MICRO_LOT:
        logger.warning(
            "Calculated lot size %.4f < minimum %.2f for balance=%.2f "
            "risk_pct=%.2f%% sl=%.1fpips — returning 0.0 (skip trade)",
            lot_size, _MICRO_LOT, balance, risk_pct, sl_distance_pips,
        )
        return 0.0

    logger.debug(
        "Lot size: %.2f | balance=%.2f risk=%.2f%% "
        "dollar_risk=%.2f sl=%.1fpips pip_val=%.2f",
        lot_size, balance, risk_pct, dollar_risk, sl_distance_pips, pip_value,
    )
    return lot_size
