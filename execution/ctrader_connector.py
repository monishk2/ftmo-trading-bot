"""
cTrader Connector
=================

Abstracts all broker communication behind a clean interface.  Two
implementations ship here:

  CTraderConnector   (abstract base — defines the interface)
  MockCTraderConnector (paper-trading simulation — no real connection)

The mock implementation is intentionally realistic:
  • Orders fill at entry_price ± a configurable slippage
  • Positions track unrealised P&L against the last price update
  • SL/TP are stored on the position record for order_manager to check
  • Account equity is recalculated from balance + sum of unrealised P&L

Live cTrader Open API integration
-----------------------------------
When you are ready to go live, subclass CTraderConnector and implement
each method using the ``ctrader-open-api`` SDK (pip install ctrader-open-api).
The mock and live versions are drop-in replacements — order_manager only
ever calls the abstract interface.

Typical usage (paper trading / backtesting harness)
----------------------------------------------------
    conn = MockCTraderConnector(initial_balance=10_000.0)
    conn.connect()

    pos = conn.place_order(
        instrument  = "EURUSD",
        direction   = 1,
        lot_size    = 0.10,
        entry_price = 1.10000,
        sl_price    = 1.09800,
        tp_price    = 1.10400,
        comment     = "LondonOpenBreakout",
    )

    conn.update_prices({"EURUSD": {"bid": 1.10250, "ask": 1.10260}})
    info = conn.get_account_info()   # equity updated from unrealised P&L

    result = conn.close_position(pos["position_id"], reason="tp")
    # result["pnl_dollars"] is now available for guardian.record_trade_result()
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PIP_VALUE_PER_LOT = 10.0   # USD per pip per standard lot (USD-quoted pairs)


# ---------------------------------------------------------------------------
# Position container
# ---------------------------------------------------------------------------

@dataclass
class MockPosition:
    """An open simulated position."""
    position_id:   str
    instrument:    str
    direction:     int        # 1 = long, -1 = short
    lot_size:      float
    entry_price:   float      # actual fill price (includes slippage)
    sl_price:      float
    tp_price:      float
    pip_size:      float      # e.g. 0.0001 for EURUSD
    pip_value:     float      # pip_value_per_lot, e.g. 10.0
    open_time:     pd.Timestamp
    comment:       str = ""

    def unrealised_pnl(self, current_price: float) -> float:
        """Floating P&L at ``current_price`` (direction-aware, no commission)."""
        pnl_pips = self.direction * (current_price - self.entry_price) / self.pip_size
        return pnl_pips * self.pip_value * self.lot_size


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class CTraderConnector(ABC):
    """
    Abstract base class — defines the contract every connector must satisfy.
    order_manager only calls these methods; it never touches internals.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish broker connection. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully close the connection."""

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Return account snapshot.

        Keys: balance, equity, margin_used, free_margin, currency, leverage
        """

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Return list of open positions.

        Each dict has: position_id, instrument, direction, lot_size,
        entry_price, sl_price, tp_price, unrealised_pnl, open_time, comment
        """

    @abstractmethod
    def get_symbol_price(self, symbol: str) -> Dict[str, float]:
        """Return {"bid": float, "ask": float} for ``symbol``."""

    @abstractmethod
    def place_order(
        self,
        instrument:  str,
        direction:   int,
        lot_size:    float,
        entry_price: float,
        sl_price:    float,
        tp_price:    float,
        comment:     str = "",
        slippage_pips: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Returns a dict with at minimum: position_id, instrument, direction,
        lot_size, entry_price (actual fill), sl_price, tp_price, open_time,
        status ("filled" | "rejected").
        """

    @abstractmethod
    def modify_order(
        self,
        position_id: str,
        sl_price:    Optional[float] = None,
        tp_price:    Optional[float] = None,
    ) -> bool:
        """Modify SL and/or TP of an open position. Returns True on success."""

    @abstractmethod
    def close_position(
        self,
        position_id: str,
        exit_price:  Optional[float] = None,
        reason:      str = "",
    ) -> Dict[str, Any]:
        """
        Close a single open position.

        Returns dict with: position_id, exit_price, pnl_pips, pnl_dollars,
        exit_time, reason.
        """

    @abstractmethod
    def close_all(self, reason: str = "") -> List[Dict[str, Any]]:
        """Close every open position. Returns list of close-result dicts."""


# ---------------------------------------------------------------------------
# Mock / paper-trading implementation
# ---------------------------------------------------------------------------

class MockCTraderConnector(CTraderConnector):
    """
    Paper-trading simulation — no real broker connection.

    Tracks positions in memory.  Prices must be pushed via
    ``update_prices()`` so P&L remains current.  SL/TP hits are NOT
    auto-triggered; order_manager is responsible for polling and closing.

    Parameters
    ----------
    initial_balance :
        Starting account balance in USD.
    default_pip_size :
        Fallback pip size used when instrument_config is not supplied.
    leverage :
        Notional leverage (for margin display only — not enforced in mock).
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        default_pip_size: float = 0.0001,
        leverage: int = 100,
    ) -> None:
        self._initial_balance  = float(initial_balance)
        self._balance:   float = float(initial_balance)   # realised only
        self._leverage:  int   = leverage
        self._default_pip_size = default_pip_size

        self._positions:  Dict[str, MockPosition] = {}
        self._prices:     Dict[str, Dict[str, float]] = {}
        self._connected:  bool = False

    # ------------------------------------------------------------------ #
    # CTraderConnector interface                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        self._connected = True
        logger.info("[MOCK] Connected to paper-trading account (balance=%.2f)", self._balance)
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("[MOCK] Disconnected from paper-trading account")

    def get_account_info(self) -> Dict[str, Any]:
        unrealised = self._total_unrealised()
        equity = self._balance + unrealised
        return {
            "balance":      round(self._balance, 2),
            "equity":       round(equity, 2),
            "unrealised":   round(unrealised, 2),
            "margin_used":  self._margin_used(),
            "free_margin":  round(equity - self._margin_used(), 2),
            "currency":     "USD",
            "leverage":     self._leverage,
            "open_positions": len(self._positions),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        result = []
        for pos in self._positions.values():
            mid = self._mid_price(pos.instrument, pos.entry_price)
            result.append({
                "position_id":    pos.position_id,
                "instrument":     pos.instrument,
                "direction":      pos.direction,
                "lot_size":       pos.lot_size,
                "entry_price":    pos.entry_price,
                "sl_price":       pos.sl_price,
                "tp_price":       pos.tp_price,
                "unrealised_pnl": round(pos.unrealised_pnl(mid), 2),
                "open_time":      pos.open_time,
                "comment":        pos.comment,
            })
        return result

    def get_symbol_price(self, symbol: str) -> Dict[str, float]:
        if symbol not in self._prices:
            logger.warning("[MOCK] No price set for %s — returning zeros", symbol)
            return {"bid": 0.0, "ask": 0.0}
        return dict(self._prices[symbol])

    def place_order(
        self,
        instrument:    str,
        direction:     int,
        lot_size:      float,
        entry_price:   float,
        sl_price:      float,
        tp_price:      float,
        comment:       str = "",
        slippage_pips: float = 0.0,
        pip_size:      float = 0.0001,
        pip_value:     float = _PIP_VALUE_PER_LOT,
    ) -> Dict[str, Any]:
        if not self._connected:
            logger.error("[MOCK] place_order called before connect()")
            return {"status": "rejected", "reason": "not connected"}

        # Apply slippage (always adverse to the trader)
        slip_price = slippage_pips * pip_size
        fill_price = (
            entry_price + slip_price if direction == 1
            else entry_price - slip_price
        )

        pos_id = str(uuid.uuid4())[:8]
        pos = MockPosition(
            position_id=pos_id,
            instrument=instrument,
            direction=direction,
            lot_size=lot_size,
            entry_price=fill_price,
            sl_price=sl_price,
            tp_price=tp_price,
            pip_size=pip_size,
            pip_value=pip_value,
            open_time=pd.Timestamp.now(tz="UTC"),
            comment=comment,
        )
        self._positions[pos_id] = pos

        direction_str = "LONG" if direction == 1 else "SHORT"
        logger.info(
            "[MOCK] ORDER FILLED | %s %s %.2f lots @ %.5f | "
            "SL=%.5f TP=%.5f | id=%s",
            direction_str, instrument, lot_size, fill_price,
            sl_price, tp_price, pos_id,
        )

        return {
            "position_id": pos_id,
            "instrument":  instrument,
            "direction":   direction,
            "lot_size":    lot_size,
            "entry_price": fill_price,
            "sl_price":    sl_price,
            "tp_price":    tp_price,
            "open_time":   pos.open_time,
            "status":      "filled",
            "comment":     comment,
        }

    def modify_order(
        self,
        position_id: str,
        sl_price:    Optional[float] = None,
        tp_price:    Optional[float] = None,
    ) -> bool:
        if position_id not in self._positions:
            logger.warning("[MOCK] modify_order: position %s not found", position_id)
            return False
        pos = self._positions[position_id]
        if sl_price is not None:
            pos.sl_price = sl_price
        if tp_price is not None:
            pos.tp_price = tp_price
        logger.info(
            "[MOCK] MODIFIED | id=%s SL=%.5f TP=%.5f",
            position_id,
            pos.sl_price, pos.tp_price,
        )
        return True

    def close_position(
        self,
        position_id: str,
        exit_price:  Optional[float] = None,
        reason:      str = "",
    ) -> Dict[str, Any]:
        if position_id not in self._positions:
            logger.warning("[MOCK] close_position: id %s not found", position_id)
            return {"status": "not_found", "position_id": position_id}

        pos = self._positions.pop(position_id)

        if exit_price is None:
            exit_price = self._mid_price(pos.instrument, pos.entry_price)

        pnl_pips    = pos.direction * (exit_price - pos.entry_price) / pos.pip_size
        pnl_dollars = pnl_pips * pos.pip_value * pos.lot_size
        self._balance += pnl_dollars

        exit_time = pd.Timestamp.now(tz="UTC")
        logger.info(
            "[MOCK] CLOSED | id=%s %s %s %.2f lots | "
            "entry=%.5f exit=%.5f pnl=%.1fpips $%.2f | reason=%s",
            position_id, pos.instrument,
            "LONG" if pos.direction == 1 else "SHORT",
            pos.lot_size, pos.entry_price, exit_price,
            pnl_pips, pnl_dollars, reason or "manual",
        )

        return {
            "position_id": position_id,
            "instrument":  pos.instrument,
            "direction":   pos.direction,
            "lot_size":    pos.lot_size,
            "entry_price": pos.entry_price,
            "exit_price":  exit_price,
            "pnl_pips":    round(pnl_pips, 1),
            "pnl_dollars": round(pnl_dollars, 2),
            "exit_time":   exit_time,
            "reason":      reason,
        }

    def close_all(self, reason: str = "") -> List[Dict[str, Any]]:
        ids = list(self._positions.keys())
        results = []
        for pid in ids:
            r = self.close_position(pid, reason=reason)
            results.append(r)
        logger.info("[MOCK] close_all: closed %d positions | reason=%s", len(results), reason)
        return results

    # ------------------------------------------------------------------ #
    # Mock-specific helpers                                                #
    # ------------------------------------------------------------------ #

    def update_prices(self, prices: Dict[str, Dict[str, float]]) -> None:
        """
        Push current market prices.

        prices = {"EURUSD": {"bid": 1.1000, "ask": 1.1001}, ...}
        Must be called before get_symbol_price / P&L calculations.
        """
        self._prices.update(prices)

    def set_balance(self, balance: float) -> None:
        """Override balance directly (useful for test setup)."""
        self._balance = float(balance)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _mid_price(self, instrument: str, fallback: float) -> float:
        if instrument in self._prices:
            p = self._prices[instrument]
            return (p.get("bid", fallback) + p.get("ask", fallback)) / 2.0
        return fallback

    def _total_unrealised(self) -> float:
        total = 0.0
        for pos in self._positions.values():
            mid = self._mid_price(pos.instrument, pos.entry_price)
            total += pos.unrealised_pnl(mid)
        return total

    def _margin_used(self) -> float:
        """Approximate margin: notional value / leverage."""
        total = 0.0
        for pos in self._positions.values():
            notional = pos.lot_size * 100_000 * pos.entry_price
            total += notional / self._leverage
        return round(total, 2)
