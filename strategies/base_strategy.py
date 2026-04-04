"""
Abstract base class for all trading strategies.

Interface contract
------------------
Every strategy must implement:
    setup(config, instrument_config)
        Load parameters from the JSON config dicts.
        Called once before generate_signals().

    generate_signals(df) -> pd.DataFrame
        Receives the full OHLCV DataFrame with a **US/Eastern tz-aware**
        DatetimeIndex (already converted by the Backtester).
        Returns the same DataFrame with these columns appended/overwritten:

            signal      int     1 = long, -1 = short, 0 = no trade
            sl_price    float   stop-loss price (NaN if no signal)
            tp_price    float   take-profit price (NaN if no signal)
            lot_size    float   (optional) override auto-sizing in Backtester
            time_stop   object  (optional) pd.Timestamp for forced exit

        Additional diagnostic columns are allowed and encouraged
        (e.g. asian_high, asian_low, asian_range_pips).

    name : str  (property or class attribute)

Backtester compatibility
------------------------
  - The Backtester converts its input to US/Eastern *before* calling
    generate_signals().  Strategies must NOT re-convert the index.
  - Use utils.timezone helpers for any intra-day time comparisons.
  - Do NOT hard-code numeric session boundaries — read from config.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_CONFIG_DIR = Path(__file__).parent.parent / "config"


class BaseStrategy(ABC):
    """Abstract base — all strategies inherit from this."""

    # ------------------------------------------------------------------
    # Class-level defaults — override in setup()
    # ------------------------------------------------------------------
    risk_per_trade_pct: float = 1.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier used in Trade records."""
        ...

    @abstractmethod
    def setup(self, config: Dict[str, Any], instrument_config: Dict[str, Any]) -> None:
        """Load strategy parameters from config dicts.

        Parameters
        ----------
        config :
            The strategy-specific sub-dict from strategy_params.json
            (e.g. ftmo_rules["london_open_breakout"]).
        instrument_config :
            The instrument sub-dict from instruments.json
            (e.g. instruments["EURUSD"]).
        """
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry signals for a full OHLCV history.

        Parameters
        ----------
        df :
            OHLCV DataFrame with a US/Eastern tz-aware DatetimeIndex.
            Columns: open, high, low, close, volume (lower-case).

        Returns
        -------
        pd.DataFrame
            Same DataFrame with at minimum these columns populated:
                signal    (int):   1 / -1 / 0
                sl_price  (float): NaN where signal == 0
                tp_price  (float): NaN where signal == 0
        """
        ...

    # ------------------------------------------------------------------
    # Convenience class methods for loading configs from disk
    # ------------------------------------------------------------------

    @classmethod
    def load_strategy_config(cls, strategy_key: str) -> Dict[str, Any]:
        """Return the strategy sub-dict from strategy_params.json."""
        with open(_CONFIG_DIR / "strategy_params.json") as fh:
            return json.load(fh)[strategy_key]

    @classmethod
    def load_instrument_config(cls, instrument: str) -> Dict[str, Any]:
        """Return the instrument sub-dict from instruments.json."""
        with open(_CONFIG_DIR / "instruments.json") as fh:
            return json.load(fh)[instrument]

    # ------------------------------------------------------------------
    # Helper shared by all strategies: ensure required output columns
    # ------------------------------------------------------------------

    @staticmethod
    def _init_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Initialise the mandatory output columns to neutral values."""
        df = df.copy()
        df["signal"]    = 0
        df["sl_price"]  = float("nan")
        df["tp_price"]  = float("nan")
        df["lot_size"]  = float("nan")
        df["time_stop"] = None          # object dtype — accepts tz-aware Timestamps
        return df
