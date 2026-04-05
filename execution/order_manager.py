"""
Order Manager
=============

The ONLY module that talks to the connector.  Enforces the complete
execution pipeline:

    Signal → Guardian veto → Position sizer → Connector → Journal

Veto is non-negotiable: every trade attempt calls
``guardian.approve_trade()`` first.  A rejection is logged and the
function returns ``False`` without touching the connector.

Signal dict contract (from strategy or live tick handler)
---------------------------------------------------------
{
    "instrument":  str,           e.g. "EURUSD"
    "direction":   int,           1=long, -1=short
    "entry_price": float,
    "sl_price":    float,
    "tp_price":    float,
    "lot_size":    float | None,  None → auto-size via position_sizer
    "risk_pct":    float,         used when lot_size is None
    "time_stop":   pd.Timestamp | None,   UTC-aware force-close time
    "timestamp":   pd.Timestamp,  UTC-aware signal time (used by guardian)
}

Position metadata (internal _positions dict)
--------------------------------------------
{
    "position_id":  str,
    "instrument":   str,
    "direction":    int,
    "lot_size":     float,
    "entry_price":  float,
    "sl_price":     float,
    "tp_price":     float,
    "initial_sl":   float,      original SL for breakeven math
    "time_stop":    Timestamp | None,
    "breakeven_moved": bool,
    "strategy_name": str,
    "open_time":    Timestamp,
}

Breakeven rule
--------------
When unrealised profit >= 1R (original risk distance in price), the SL
is moved to entry price so a winning trade can never become a full loss.
This is applied once per position.

manage_open_positions(current_time, current_prices)
---------------------------------------------------
Called by the live monitor every tick / bar:
  1. Update connector prices
  2. For each tracked position, in order:
     a. Time-stop expired?  → close
     b. SL hit?             → close
     c. TP hit?             → close
     d. Breakeven trigger?  → modify SL
  3. After closing, call guardian.record_trade_result()

In live trading via real cTrader, SL/TP are handled server-side.  Replace
the mock price-checking block with a `get_positions()` diff against the
last-known set to detect broker-initiated closures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from execution.ctrader_connector import CTraderConnector
from execution.ftmo_guardian import FTMOGuardian, TradeRecord, TradeRequest
from execution.position_sizer import calculate_lot_size

logger = logging.getLogger(__name__)

_PIP_SIZE_DEFAULT  = 0.0001
_PIP_VALUE_DEFAULT = 10.0


class OrderManager:
    """
    Orchestrates the full trade-execution pipeline.

    Parameters
    ----------
    guardian :
        FTMOGuardian instance.  approve_trade() is called before EVERY order.
    connector :
        CTraderConnector (real or mock).
    instruments_config :
        Parsed instruments.json dict keyed by symbol.
    journal :
        Optional trade journal (any object with a ``log_trade(dict)`` method).
        If None, trades are only written to the standard logger.
    """

    def __init__(
        self,
        guardian:           FTMOGuardian,
        connector:          CTraderConnector,
        instruments_config: Dict[str, Any],
        journal:            Optional[Any] = None,
    ) -> None:
        self._guardian    = guardian
        self._connector   = connector
        self._instruments = instruments_config
        self._journal     = journal

        # position_id → metadata dict (see module docstring)
        self._positions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def execute_trade(self, signal: dict, strategy_name: str) -> bool:
        """
        Attempt to place a trade for the given signal.

        Returns True if the order was placed, False if rejected or errored.
        """
        instrument = signal["instrument"]
        direction  = int(signal["direction"])
        entry      = float(signal["entry_price"])
        sl         = float(signal["sl_price"])
        tp         = float(signal["tp_price"])
        timestamp  = signal["timestamp"]
        time_stop  = signal.get("time_stop")

        instr_cfg = self._instruments.get(instrument, {})
        pip_size  = float(instr_cfg.get("pip_size", _PIP_SIZE_DEFAULT))
        pip_value = float(instr_cfg.get("pip_value_per_lot", _PIP_VALUE_DEFAULT))

        # ── Lot size ──────────────────────────────────────────────────
        lot_size = signal.get("lot_size")
        if lot_size is None:
            risk_pct     = float(signal.get("risk_pct", 1.0))
            sl_pips      = abs(entry - sl) / pip_size
            account_info = self._connector.get_account_info()
            balance      = account_info.get("balance", self._guardian._balance)
            lot_size     = calculate_lot_size(balance, risk_pct, sl_pips, instr_cfg)

        if lot_size <= 0:
            logger.warning(
                "Skipping trade on %s — lot_size=0 (balance too small for 1 micro-lot)",
                instrument,
            )
            return False

        sl_pips = abs(entry - sl) / pip_size

        # ── Guardian veto (MUST happen before every order) ───────────
        req = TradeRequest(
            instrument=instrument,
            direction=direction,
            lot_size=lot_size,
            sl_pips=sl_pips,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            timestamp=timestamp,
        )
        approval = self._guardian.approve_trade(req)
        if not approval.approved:
            logger.warning(
                "Trade REJECTED by guardian | %s %s | reason: %s",
                instrument, strategy_name, approval.reason,
            )
            return False

        # ── Place order ───────────────────────────────────────────────
        try:
            result = self._connector.place_order(
                instrument=instrument,
                direction=direction,
                lot_size=lot_size,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                comment=strategy_name,
                pip_size=pip_size,
                pip_value=pip_value,
            )
        except Exception as exc:
            logger.error("Connector error placing order on %s: %s", instrument, exc)
            # Guardian approved but connector failed — decrement guardian counter
            self._guardian._open_positions = max(
                0, self._guardian._open_positions - 1
            )
            self._guardian._save_state()
            return False

        if result.get("status") != "filled":
            logger.error("Order not filled: %s", result)
            self._guardian._open_positions = max(
                0, self._guardian._open_positions - 1
            )
            self._guardian._save_state()
            return False

        pos_id = result["position_id"]
        self._positions[pos_id] = {
            "position_id":    pos_id,
            "instrument":     instrument,
            "direction":      direction,
            "lot_size":       lot_size,
            "entry_price":    result["entry_price"],
            "sl_price":       sl,
            "tp_price":       tp,
            "initial_sl":     sl,         # kept for breakeven calculation
            "time_stop":      time_stop,
            "breakeven_moved": False,
            "strategy_name":  strategy_name,
            "open_time":      result.get("open_time", timestamp),
            "pip_size":       pip_size,
            "pip_value":      pip_value,
        }

        self._log_trade_open(result, strategy_name)
        return True

    def manage_open_positions(
        self,
        current_time:   pd.Timestamp,
        current_prices: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Poll all tracked positions and take action where needed.

        Call this every ~60 seconds from the live monitor (or every tick
        in paper-trading mode).

        Parameters
        ----------
        current_time :
            UTC-aware timestamp of the current bar / tick.
        current_prices :
            {"EURUSD": {"bid": 1.1000, "ask": 1.1001}, ...}
        """
        # Push prices to connector so P&L is fresh
        if hasattr(self._connector, "update_prices"):
            self._connector.update_prices(current_prices)

        closed_ids = []

        for pos_id, meta in list(self._positions.items()):
            instrument = meta["instrument"]
            direction  = meta["direction"]
            entry      = meta["entry_price"]
            sl         = meta["sl_price"]
            tp         = meta["tp_price"]
            pip_size   = meta.get("pip_size", _PIP_SIZE_DEFAULT)

            price_info = current_prices.get(instrument, {})
            bid = price_info.get("bid", entry)
            ask = price_info.get("ask", entry)
            # Use adverse fill price for exit checks
            exit_price_long  = bid   # long position sold at bid
            exit_price_short = ask   # short position bought back at ask

            current_exit_price = (
                exit_price_long if direction == 1 else exit_price_short
            )

            close_reason: Optional[str] = None
            close_price:  Optional[float] = None

            # 1. Time stop
            if meta.get("time_stop") and current_time >= meta["time_stop"]:
                close_reason = "time_stop"
                close_price  = current_exit_price

            # 2. SL hit (skip if already closing)
            elif direction == 1 and bid <= sl:
                close_reason = "sl"
                close_price  = sl
            elif direction == -1 and ask >= sl:
                close_reason = "sl"
                close_price  = sl

            # 3. TP hit
            elif direction == 1 and bid >= tp:
                close_reason = "tp"
                close_price  = tp
            elif direction == -1 and ask <= tp:
                close_reason = "tp"
                close_price  = tp

            if close_reason:
                self._close_and_record(pos_id, meta, close_price, close_reason)
                closed_ids.append(pos_id)
                continue

            # 4. Breakeven move (only once per position)
            if not meta["breakeven_moved"]:
                initial_risk = abs(entry - meta["initial_sl"])
                moved_in_favor = direction * (current_exit_price - entry)
                if moved_in_favor >= initial_risk and initial_risk > 0:
                    new_sl = entry  # move SL to breakeven
                    ok = self._connector.modify_order(pos_id, sl_price=new_sl)
                    if ok:
                        meta["sl_price"]       = new_sl
                        meta["breakeven_moved"] = True
                        logger.info(
                            "Breakeven: %s %s SL moved to entry %.5f",
                            instrument, pos_id[:6], entry,
                        )

        for pid in closed_ids:
            self._positions.pop(pid, None)

    def close_all_positions(self, reason: str = "manual") -> None:
        """
        Force-close every open position and notify the guardian.

        Called on emergency halt, end-of-day, or guardian force_close_all.
        """
        self._guardian.force_close_all(reason)
        results = self._connector.close_all(reason=reason)
        for r in results:
            pos_id = r.get("position_id")
            pnl    = r.get("pnl_dollars", 0.0)
            meta   = self._positions.pop(pos_id, {})
            self._guardian.record_trade_result(
                TradeRecord(
                    pnl_dollars=pnl,
                    closed_at=r.get("exit_time", pd.Timestamp.now(tz="UTC")),
                )
            )
            self._log_trade_close(r, meta, reason)
        logger.info("close_all_positions: %d positions closed | reason=%s",
                    len(results), reason)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _close_and_record(
        self,
        pos_id:       str,
        meta:         Dict[str, Any],
        exit_price:   float,
        reason:       str,
    ) -> None:
        result = self._connector.close_position(
            position_id=pos_id,
            exit_price=exit_price,
            reason=reason,
        )
        pnl = result.get("pnl_dollars", 0.0)
        self._guardian.record_trade_result(
            TradeRecord(
                pnl_dollars=pnl,
                closed_at=result.get("exit_time", pd.Timestamp.now(tz="UTC")),
            )
        )
        self._log_trade_close(result, meta, reason)

    def _log_trade_open(self, result: Dict[str, Any], strategy: str) -> None:
        msg = {
            "event":       "trade_open",
            "strategy":    strategy,
            "position_id": result.get("position_id"),
            "instrument":  result.get("instrument"),
            "direction":   "LONG" if result.get("direction") == 1 else "SHORT",
            "lot_size":    result.get("lot_size"),
            "entry_price": result.get("entry_price"),
            "sl_price":    result.get("sl_price"),
            "tp_price":    result.get("tp_price"),
            "open_time":   str(result.get("open_time")),
        }
        logger.info("TRADE_OPEN %s", msg)
        if self._journal and hasattr(self._journal, "log_trade"):
            self._journal.log_trade(msg)

    def _log_trade_close(
        self,
        result: Dict[str, Any],
        meta:   Dict[str, Any],
        reason: str,
    ) -> None:
        msg = {
            "event":       "trade_close",
            "strategy":    meta.get("strategy_name", ""),
            "position_id": result.get("position_id"),
            "instrument":  result.get("instrument", meta.get("instrument")),
            "direction":   "LONG" if result.get("direction", 1) == 1 else "SHORT",
            "lot_size":    result.get("lot_size", meta.get("lot_size")),
            "entry_price": result.get("entry_price", meta.get("entry_price")),
            "exit_price":  result.get("exit_price"),
            "pnl_pips":    result.get("pnl_pips"),
            "pnl_dollars": result.get("pnl_dollars"),
            "exit_reason": reason,
            "exit_time":   str(result.get("exit_time")),
        }
        logger.info("TRADE_CLOSE %s", msg)
        if self._journal and hasattr(self._journal, "log_trade"):
            self._journal.log_trade(msg)

    # ------------------------------------------------------------------ #
    # Read-only status                                                     #
    # ------------------------------------------------------------------ #

    @property
    def open_position_count(self) -> int:
        return len(self._positions)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return a copy of the internal position metadata list."""
        return list(self._positions.values())
