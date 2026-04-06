"""
Trade Journal
=============

Persists every trade and daily summary to CSV files so the full record
survives process restarts and can be loaded into pandas for ad-hoc
analysis or included in reports.

Files written
-------------
  logs/trades.csv       — one row per completed trade
  logs/daily_log.csv    — one row per calendar day (written at daily close)

Both files are append-only: existing rows are never modified.  If the file
does not exist it is created with a header on first write.

Public API
----------
  journal = TradeJournal()
  journal = TradeJournal(trades_path="logs/trades.csv",
                         daily_path="logs/daily_log.csv")

  # Called by order_manager via log_trade(dict):
  journal.log_trade(msg_dict)     # handles both trade_open and trade_close events

  # Called at end of each trading day:
  journal.log_daily_summary(guardian_status, regime, strategies_active)

  # Load historical records for analysis:
  df = journal.load_trades()
  df = journal.load_daily_log()
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_TRADE_COLS = [
    "timestamp",           # ISO UTC — time of trade close
    "position_id",
    "instrument",
    "direction",           # LONG | SHORT
    "lot_size",
    "entry_price",
    "exit_price",
    "sl_price",
    "tp_price",
    "pnl_pips",
    "pnl_dollars",
    "pnl_pct",             # pnl_dollars / balance_at_entry × 100
    "strategy",
    "regime",              # regime label at trade open (if available)
    "exit_reason",         # sl | tp | time_stop | daily_halt | manual | …
    "duration_minutes",
    "guardian_status",     # daily_halted | permanently_halted | ok
]

_DAILY_COLS = [
    "date",
    "start_balance",
    "end_balance",
    "daily_pnl",
    "daily_pnl_pct",
    "num_trades",
    "wins",
    "losses",
    "max_drawdown_today_pct",
    "regime",
    "strategies_active",
]


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

class TradeJournal:
    """
    Append-only CSV trade and daily-summary logger.

    Parameters
    ----------
    trades_path :
        Path to the per-trade CSV.  Created on first write.
    daily_path :
        Path to the daily-summary CSV.  Created on first write.
    """

    def __init__(
        self,
        trades_path: str | Path = "logs/trades.csv",
        daily_path:  str | Path = "logs/daily_log.csv",
    ) -> None:
        self._trades_path = Path(trades_path)
        self._daily_path  = Path(daily_path)
        self._trades_path.parent.mkdir(parents=True, exist_ok=True)
        self._daily_path.parent.mkdir(parents=True, exist_ok=True)

        # In-flight open trades: position_id → open metadata
        self._open_trades: Dict[str, Dict[str, Any]] = {}

        # Track today's closed trades for daily summary
        self._today_trades: List[Dict[str, Any]] = []
        self._daily_min_equity: float = float("inf")

        self._ensure_headers()
        logger.info(
            "TradeJournal initialised | trades=%s daily=%s",
            self._trades_path, self._daily_path,
        )

    # ------------------------------------------------------------------ #
    # Public: called by order_manager                                      #
    # ------------------------------------------------------------------ #

    def log_trade(self, msg: Dict[str, Any]) -> None:
        """
        Entry point called by order_manager for both open and close events.

        Expected keys (from order_manager._log_trade_open):
            event="trade_open", position_id, instrument, direction,
            lot_size, entry_price, sl_price, tp_price, open_time, strategy

        Expected keys (from order_manager._log_trade_close):
            event="trade_close", position_id, instrument, direction,
            lot_size, entry_price, exit_price, pnl_pips, pnl_dollars,
            exit_reason, exit_time, strategy
        """
        event = msg.get("event", "")
        if event == "trade_open":
            self._handle_open(msg)
        elif event == "trade_close":
            self._handle_close(msg)
        else:
            logger.debug("TradeJournal: unknown event %r — ignored", event)

    def log_daily_summary(
        self,
        guardian_status:    Dict[str, Any],
        regime:             str = "",
        strategies_active:  List[str] | None = None,
    ) -> None:
        """
        Write one row to daily_log.csv.  Call this at 16:00 ET (or on
        shutdown) after all positions have been closed.

        Parameters
        ----------
        guardian_status :
            The dict returned by FTMOGuardian.get_status().
        regime :
            Today's regime string (from RegimeFilter).
        strategies_active :
            List of strategy name strings active today.
        """
        start_balance = guardian_status.get("midnight_balance",
                        guardian_status.get("initial_balance", 0.0))
        end_balance   = guardian_status.get("current_balance", 0.0)
        daily_pnl     = guardian_status.get("daily_pnl", 0.0)
        daily_pnl_pct = (
            daily_pnl / start_balance * 100.0 if start_balance else 0.0
        )

        wins   = sum(1 for t in self._today_trades if t.get("pnl_dollars", 0) > 0)
        losses = sum(1 for t in self._today_trades if t.get("pnl_dollars", 0) <= 0)

        max_dd = guardian_status.get("daily_drawdown_pct", 0.0)
        if self._daily_min_equity != float("inf") and start_balance:
            max_dd = max(
                max_dd,
                (start_balance - self._daily_min_equity) / start_balance * 100.0,
            )

        row = {
            "date":                 pd.Timestamp.now(tz="UTC").date().isoformat(),
            "start_balance":        round(start_balance, 2),
            "end_balance":          round(end_balance, 2),
            "daily_pnl":            round(daily_pnl, 2),
            "daily_pnl_pct":        round(daily_pnl_pct, 4),
            "num_trades":           len(self._today_trades),
            "wins":                 wins,
            "losses":               losses,
            "max_drawdown_today_pct": round(max_dd, 4),
            "regime":               regime,
            "strategies_active":    ",".join(strategies_active or []),
        }
        self._append_row(self._daily_path, _DAILY_COLS, row)
        logger.info("Daily summary logged: %s", row)

        # Reset intra-day tracking
        self._today_trades.clear()
        self._daily_min_equity = float("inf")

    def update_equity(self, current_equity: float) -> None:
        """
        Call every time the guardian calls update_equity so the journal
        can track intra-day drawdown for the daily summary.
        """
        if current_equity < self._daily_min_equity:
            self._daily_min_equity = current_equity

    def load_trades(self) -> pd.DataFrame:
        """Return all closed trades as a DataFrame. Empty if file missing."""
        if not self._trades_path.exists():
            return pd.DataFrame(columns=_TRADE_COLS)
        return pd.read_csv(self._trades_path, parse_dates=["timestamp"])

    def load_daily_log(self) -> pd.DataFrame:
        """Return all daily summaries as a DataFrame. Empty if file missing."""
        if not self._daily_path.exists():
            return pd.DataFrame(columns=_DAILY_COLS)
        return pd.read_csv(self._daily_path, parse_dates=["date"])

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _handle_open(self, msg: Dict[str, Any]) -> None:
        pos_id = msg.get("position_id")
        if pos_id:
            self._open_trades[pos_id] = {
                "entry_price":  msg.get("entry_price"),
                "sl_price":     msg.get("sl_price"),
                "tp_price":     msg.get("tp_price"),
                "open_time":    msg.get("open_time"),
                "strategy":     msg.get("strategy", ""),
            }
        logger.debug("Journal: trade open recorded for %s", pos_id)

    def _handle_close(self, msg: Dict[str, Any]) -> None:
        pos_id   = msg.get("position_id", "")
        open_meta = self._open_trades.pop(pos_id, {})

        entry_price = msg.get("entry_price") or open_meta.get("entry_price")
        open_time_raw = open_meta.get("open_time") or msg.get("exit_time")
        try:
            open_ts  = pd.Timestamp(open_time_raw)
            close_ts = pd.Timestamp(msg.get("exit_time", pd.Timestamp.now(tz="UTC")))
            duration = (close_ts - open_ts).total_seconds() / 60.0
        except Exception:
            duration = None

        pnl_dollars = msg.get("pnl_dollars", 0.0) or 0.0

        row: Dict[str, Any] = {
            "timestamp":       pd.Timestamp.now(tz="UTC").isoformat(),
            "position_id":     pos_id,
            "instrument":      msg.get("instrument", ""),
            "direction":       msg.get("direction", ""),
            "lot_size":        msg.get("lot_size"),
            "entry_price":     entry_price,
            "exit_price":      msg.get("exit_price"),
            "sl_price":        open_meta.get("sl_price"),
            "tp_price":        open_meta.get("tp_price"),
            "pnl_pips":        msg.get("pnl_pips"),
            "pnl_dollars":     round(float(pnl_dollars), 2),
            "pnl_pct":         None,       # filled below if possible
            "strategy":        msg.get("strategy", open_meta.get("strategy", "")),
            "regime":          "",         # set externally via log_trade(regime=...)
            "exit_reason":     msg.get("exit_reason", ""),
            "duration_minutes": round(duration, 1) if duration is not None else None,
            "guardian_status": msg.get("guardian_status", "ok"),
        }

        self._append_row(self._trades_path, _TRADE_COLS, row)
        self._today_trades.append(row)
        logger.debug("Journal: trade close recorded for %s pnl=$%.2f", pos_id, pnl_dollars)

    def _ensure_headers(self) -> None:
        """Write CSV headers if files don't exist yet."""
        for path, cols in [
            (self._trades_path, _TRADE_COLS),
            (self._daily_path,  _DAILY_COLS),
        ]:
            if not path.exists():
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(cols)

    @staticmethod
    def _append_row(path: Path, cols: List[str], row: Dict[str, Any]) -> None:
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writerow(row)
