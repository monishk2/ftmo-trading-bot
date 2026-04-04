"""
Vectorized event-driven backtesting engine.

Architecture
------------
Signal generation is vectorised (strategy processes the full DataFrame at once).
Execution is simulated bar-by-bar to correctly model:
  - Position state (only one position at a time)
  - FTMO daily loss limit (intraday equity check at every bar)
  - FTMO total drawdown halt

Entry model
-----------
  Signal fires at bar close (t).
  entry_price = close ± spread/2 ± slippage
    Long:  buy at ask  → close + spread_adj + slippage_adj
    Short: sell at bid → close - spread_adj - slippage_adj
  Slippage is drawn from a session-aware uniform distribution:
    - Within session_open_window_minutes of 03:00 or 09:30 Eastern → session_open_pips max
    - Otherwise → normal_pips max
  SL/TP checked from bar t+1 onward using bar high/low.
  If both SL and TP hit in the same bar → exit at SL (worst case).

P&L model
---------
  pnl_pips    = direction × (exit_price − entry_price) / pip_size
  pnl_dollars = pnl_pips × PIP_VALUE_PER_LOT × lot_size − commission
  equity      = balance (realised) + floating (open position marked to close)

FTMO daily loss
---------------
  Reference balance = account BALANCE (realised only) at midnight US/Eastern.
  Check = (equity − midnight_balance) / midnight_balance × 100
  Trigger at -daily_loss_trigger_pct% (default 4%) → block trading + close position.

Strategy interface (duck-typed)
--------------------------------
  strategy.name                  : str
  strategy.risk_per_trade_pct    : float   (used for auto lot sizing)
  strategy.generate_signals(df)  : pd.DataFrame
      Adds columns to df:
        signal    int    1=long, -1=short, 0=no trade
        sl_price  float  stop-loss price
        tp_price  float  take-profit price
        lot_size  float  (optional) overrides auto-sizing
        time_stop pd.Timestamp  (optional) force-close at this bar
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.timezone import is_within_session_open_window

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"

# USD per pip per standard lot for USD-quoted pairs (EURUSD, GBPUSD).
# Based on 100,000 unit contract: pip_value = pip_size × 100,000 / quote ≈ $10.
PIP_VALUE_PER_LOT = 10.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time:    pd.Timestamp
    exit_time:     pd.Timestamp
    direction:     int          # 1 = long, -1 = short
    entry_price:   float
    exit_price:    float
    sl:            float
    tp:            float
    lot_size:      float
    pnl_pips:      float
    pnl_dollars:   float
    pnl_pct:       float        # relative to balance at entry
    strategy_name: str
    instrument:    str
    exit_reason:   str          # sl | tp | time_stop | ftmo_daily | ftmo_total | end_of_data


@dataclass
class BacktestResult:
    trades:          List[Trade]
    equity_curve:    pd.Series   # equity (balance + floating) at every bar close
    daily_pnl:       pd.Series   # keyed by date (US/Eastern)
    config:          Dict[str, Any]
    initial_balance: float
    final_balance:   float
    ftmo_halt_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Backtester:
    """
    Bar-by-bar backtesting engine with realistic cost simulation
    and FTMO rule enforcement.

    Parameters
    ----------
    strategy :
        Object implementing the strategy interface (see module docstring).
    df :
        OHLCV DataFrame. Index must be tz-aware UTC datetimes (or a 'datetime'
        column). Columns: open, high, low, close, volume.
        The engine converts the index to US/Eastern internally.
    instrument :
        Key in instruments.json (e.g. "EURUSD").
    initial_balance :
        Starting account balance in USD.
    phase :
        "challenge" | "verification" | "funded" — selects FTMO rule set.
    seed :
        RNG seed for reproducible slippage draws.
    _override_instrument_config :
        Inject a custom instrument config dict instead of loading from file.
        Intended for unit tests only.
    _override_ftmo_rules :
        Inject custom FTMO rules dict instead of loading from file.
        Intended for unit tests only.
    """

    def __init__(
        self,
        strategy,
        df: pd.DataFrame,
        instrument: str,
        initial_balance: float = 10_000.0,
        phase: str = "challenge",
        seed: int = 42,
        _override_instrument_config: Optional[Dict] = None,
        _override_ftmo_rules: Optional[Dict] = None,
    ) -> None:
        self.strategy = strategy
        self.instrument = instrument
        self.initial_balance = initial_balance
        self.phase = phase
        self.rng = np.random.default_rng(seed)

        self._load_configs(_override_instrument_config, _override_ftmo_rules)
        self.df = self._prepare_data(df)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_configs(
        self,
        override_instrument: Optional[Dict],
        override_ftmo: Optional[Dict],
    ) -> None:
        if override_ftmo is not None:
            self.ftmo_rules = override_ftmo
        else:
            with open(_CONFIG_DIR / "ftmo_rules.json") as fh:
                self.ftmo_rules = json.load(fh)

        if override_instrument is not None:
            inst = override_instrument
        else:
            with open(_CONFIG_DIR / "instruments.json") as fh:
                inst = json.load(fh)[self.instrument]

        self.pip_size: float            = inst["pip_size"]
        self.spread_pips: float         = inst["typical_spread_pips"]
        self.slippage_model: Dict       = inst["slippage_model"]
        self.commission_per_lot: float  = inst["commission_per_lot_round_trip"]

        buffers = self.ftmo_rules["safety_buffers"]
        self.daily_loss_trigger_pct: float = buffers["daily_loss_trigger_pct"]
        self.total_loss_trigger_pct: float = buffers["total_loss_trigger_pct"]

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise to US/Eastern tz-aware DatetimeIndex with lower-case columns."""
        df = df.copy()
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("US/Eastern")
        df.columns = [c.lower() for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Cost modelling
    # ------------------------------------------------------------------

    def _slippage_pips(self, ts: pd.Timestamp) -> float:
        """Draw random slippage in pips, elevated near session opens."""
        ts_utc = ts.tz_convert("UTC")
        window = self.slippage_model["session_open_window_minutes"]
        if is_within_session_open_window(ts_utc, window):
            max_slip = self.slippage_model["session_open_pips"]
        else:
            max_slip = self.slippage_model["normal_pips"]
        return float(self.rng.uniform(0.0, max_slip))

    def _calc_entry_price(self, close: float, direction: int, ts: pd.Timestamp) -> float:
        """Apply half-spread + slippage to the signal-bar close price."""
        spread_adj = (self.spread_pips / 2.0) * self.pip_size
        slip_adj   = self._slippage_pips(ts) * self.pip_size
        if direction == 1:   # long: buy at ask
            return close + spread_adj + slip_adj
        return close - spread_adj - slip_adj  # short: sell at bid

    def _auto_lot_size(self, balance: float, entry: float, sl: float) -> float:
        """Size position so that an SL hit risks risk_per_trade_pct of balance."""
        risk_pct = getattr(self.strategy, "risk_per_trade_pct", 0.75)
        risk_usd = balance * risk_pct / 100.0
        sl_pips  = abs(entry - sl) / self.pip_size
        if sl_pips < 0.01:
            return 0.01
        lot = risk_usd / (sl_pips * PIP_VALUE_PER_LOT)
        return max(0.01, round(lot, 2))

    # ------------------------------------------------------------------
    # Trade lifecycle helpers
    # ------------------------------------------------------------------

    def _floating_pnl(self, pos: Dict, close: float) -> float:
        """Unrealised P&L in USD (commission excluded, charged only on close)."""
        pnl_pips = pos["direction"] * (close - pos["entry_price"]) / self.pip_size
        return pnl_pips * PIP_VALUE_PER_LOT * pos["lot_size"]

    def _close_position(
        self,
        pos: Dict,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
    ) -> Trade:
        """Build a closed Trade, deducting round-trip commission."""
        direction  = pos["direction"]
        pnl_pips   = direction * (exit_price - pos["entry_price"]) / self.pip_size
        gross_usd  = pnl_pips * PIP_VALUE_PER_LOT * pos["lot_size"]
        commission = self.commission_per_lot * pos["lot_size"]
        pnl_usd    = gross_usd - commission
        pnl_pct    = pnl_usd / pos["balance_at_entry"] * 100.0

        logger.debug(
            "%s %s exit=%s reason=%s pnl_pips=%.1f pnl_usd=%.2f commission=%.2f",
            self.instrument,
            "LONG" if direction == 1 else "SHORT",
            exit_time, exit_reason, pnl_pips, pnl_usd, commission,
        )

        return Trade(
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            direction=direction,
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            sl=pos["sl"],
            tp=pos["tp"],
            lot_size=pos["lot_size"],
            pnl_pips=pnl_pips,
            pnl_dollars=pnl_usd,
            pnl_pct=pnl_pct,
            strategy_name=self.strategy.name,
            instrument=self.instrument,
            exit_reason=exit_reason,
        )

    # ------------------------------------------------------------------
    # Parsing helpers (handle NaN / NaT / missing columns gracefully)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_signal(row) -> int:
        val = row.get("signal", 0)
        return 0 if pd.isna(val) else int(val)

    @staticmethod
    def _get_float(row, key: str) -> float:
        val = row.get(key, np.nan)
        return np.nan if pd.isna(val) else float(val)

    @staticmethod
    def _get_timestamp(row, key: str) -> Optional[pd.Timestamp]:
        val = row.get(key, None)
        return None if (val is None or pd.isna(val)) else val

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest. Returns a BacktestResult."""
        df = self.strategy.generate_signals(self.df.copy())

        balance   = self.initial_balance
        trades:   List[Trade] = []
        eq_curve: Dict[pd.Timestamp, float] = {}
        day_pnl:  Dict = {}

        open_pos:       Optional[Dict] = None
        prev_date                      = None
        midnight_balance               = balance
        daily_blocked                  = False
        halt_reason: Optional[str]     = None

        for ts, row in df.iterrows():
            current_date = ts.date()

            # ── Day boundary ──────────────────────────────────────────────
            if current_date != prev_date:
                if prev_date is not None:
                    day_pnl[prev_date] = balance - midnight_balance
                # Snapshot REALISED balance at midnight Eastern
                midnight_balance = balance
                daily_blocked    = False
                prev_date        = current_date

            # ── Equity = realised balance + floating unrealised ───────────
            equity = balance + (self._floating_pnl(open_pos, row["close"]) if open_pos else 0.0)

            # ── FTMO daily loss check ────────────────────────────────────
            daily_dd_pct = (equity - midnight_balance) / midnight_balance * 100.0

            if (not daily_blocked) and daily_dd_pct <= -self.daily_loss_trigger_pct:
                daily_blocked = True
                logger.warning(
                    "FTMO daily-loss triggered at %s | dd=%.2f%% | threshold=%.1f%%",
                    ts, daily_dd_pct, -self.daily_loss_trigger_pct,
                )
                if open_pos:
                    trade = self._close_position(open_pos, row["close"], ts, "ftmo_daily")
                    trades.append(trade)
                    balance  += trade.pnl_dollars
                    equity    = balance
                    open_pos  = None

            # ── FTMO total drawdown check ────────────────────────────────
            total_dd_pct = (equity - self.initial_balance) / self.initial_balance * 100.0

            if total_dd_pct <= -self.total_loss_trigger_pct:
                if open_pos:
                    trade = self._close_position(open_pos, row["close"], ts, "ftmo_total")
                    trades.append(trade)
                    balance  += trade.pnl_dollars
                    open_pos  = None
                halt_reason = (
                    f"Total drawdown {total_dd_pct:.2f}% reached safety buffer "
                    f"(-{self.total_loss_trigger_pct}%) at {ts}"
                )
                logger.error(halt_reason)
                eq_curve[ts] = balance
                break

            # ── Check open position for SL / TP / time-stop ─────────────
            if open_pos:
                direction = open_pos["direction"]
                sl, tp    = open_pos["sl"], open_pos["tp"]

                exit_price:  Optional[float] = None
                exit_reason: Optional[str]   = None

                if direction == 1:   # long
                    if row["low"] <= sl:
                        exit_price, exit_reason = sl, "sl"
                    elif row["high"] >= tp:
                        exit_price, exit_reason = tp, "tp"
                else:                # short
                    if row["high"] >= sl:
                        exit_price, exit_reason = sl, "sl"
                    elif row["low"] <= tp:
                        exit_price, exit_reason = tp, "tp"

                # Time stop: force-close at bar close if timestamp reached
                if exit_price is None:
                    time_stop = open_pos.get("time_stop")
                    if time_stop is not None and ts >= time_stop:
                        exit_price, exit_reason = row["close"], "time_stop"

                if exit_price is not None:
                    trade = self._close_position(open_pos, exit_price, ts, exit_reason)
                    trades.append(trade)
                    balance  += trade.pnl_dollars
                    equity    = balance
                    open_pos  = None

            # ── Enter new position ───────────────────────────────────────
            signal = self._get_signal(row)

            if open_pos is None and not daily_blocked and signal != 0:
                sl_price = self._get_float(row, "sl_price")
                tp_price = self._get_float(row, "tp_price")

                if not (np.isnan(sl_price) or np.isnan(tp_price)):
                    entry_price = self._calc_entry_price(row["close"], signal, ts)

                    raw_lot  = self._get_float(row, "lot_size")
                    lot_size = (
                        self._auto_lot_size(balance, entry_price, sl_price)
                        if np.isnan(raw_lot)
                        else raw_lot
                    )

                    open_pos = {
                        "entry_time":      ts,
                        "direction":       signal,
                        "entry_price":     entry_price,
                        "sl":              sl_price,
                        "tp":              tp_price,
                        "lot_size":        lot_size,
                        "balance_at_entry": balance,
                        "time_stop":       self._get_timestamp(row, "time_stop"),
                    }
                    logger.debug(
                        "%s %s | entry=%.5f sl=%.5f tp=%.5f lots=%.2f @ %s",
                        self.instrument, "LONG" if signal == 1 else "SHORT",
                        entry_price, sl_price, tp_price, lot_size, ts,
                    )

            # Record equity (floating included)
            eq_curve[ts] = balance + (self._floating_pnl(open_pos, row["close"]) if open_pos else 0.0)

        # ── End-of-data: close any still-open position ────────────────────
        if open_pos:
            last_ts    = df.index[-1]
            last_close = df.iloc[-1]["close"]
            trade = self._close_position(open_pos, last_close, last_ts, "end_of_data")
            trades.append(trade)
            balance += trade.pnl_dollars

        # Final daily P&L bucket
        if prev_date is not None:
            day_pnl[prev_date] = balance - midnight_balance

        config = {
            "instrument":    self.instrument,
            "initial_balance": self.initial_balance,
            "phase":         self.phase,
            "strategy_name": self.strategy.name,
            "ftmo_rules":    self.ftmo_rules,
        }

        return BacktestResult(
            trades=trades,
            equity_curve=pd.Series(eq_curve),
            daily_pnl=pd.Series(day_pnl),
            config=config,
            initial_balance=self.initial_balance,
            final_balance=balance,
            ftmo_halt_reason=halt_reason,
        )
