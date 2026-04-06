"""
Alert System
============

Sends alerts for critical trading events: trade entries/exits, guardian
blocks, daily summaries, approaching FTMO limits, and errors.

Design for extensibility
------------------------
All alerts flow through ``AlertManager.send()``.  Right now it prints to
the terminal and appends to a log file.  To add Telegram, Slack, email,
or any other channel, subclass ``AlertManager`` and override ``_deliver()``,
or register a callable with ``register_handler(fn)``.

Alert levels
------------
  INFO     — routine events (trade entry, daily summary)
  WARNING  — approaching limits (70%+ of daily/total DD used)
  CRITICAL — guardian blocks, permanent halts, errors

Public API
----------
  alerts = AlertManager()
  alerts = AlertManager(log_path="logs/alerts.log",
                        daily_limit_warn_pct=70.0)

  alerts.on_trade_entry(signal_dict, guardian_status)
  alerts.on_trade_exit(close_result_dict, guardian_status)
  alerts.on_daily_summary(guardian_status, regime, strategies_active)
  alerts.on_guardian_block(reason, guardian_status)
  alerts.on_approaching_limit(limit_type, used_pct, limit_pct)
  alerts.on_error(context, exception)
  alerts.check_limits(guardian_status)   — call every equity check cycle

  # Register a custom delivery channel (Telegram, Slack, etc.)
  alerts.register_handler(fn)  # fn(level, title, body) -> None
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

_EASTERN = pytz.timezone("US/Eastern")

# Threshold (% of limit used) at which a WARNING fires
_DEFAULT_WARN_THRESHOLD = 70.0


class AlertManager:
    """
    Central alert dispatcher for all monitoring events.

    Parameters
    ----------
    log_path :
        File where every alert is appended (in addition to stdout).
    daily_limit_warn_pct :
        Percentage of the daily/total drawdown limit at which a WARNING fires.
        Default: 70 % (i.e. warn when 2.8% of the 4% daily limit is used).
    """

    def __init__(
        self,
        log_path:              str | Path = "logs/alerts.log",
        daily_limit_warn_pct:  float      = _DEFAULT_WARN_THRESHOLD,
    ) -> None:
        self._log_path    = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._warn_thresh = float(daily_limit_warn_pct)

        # Extra delivery handlers (e.g. Telegram, Slack)
        self._handlers: List[Callable[[str, str, str], None]] = []

        # Deduplication: track which limit warnings have already fired today
        self._warned_today: set = set()

        logger.info(
            "AlertManager initialised | log=%s warn_at=%.0f%%",
            self._log_path, self._warn_thresh,
        )

    # ------------------------------------------------------------------ #
    # Handler registration                                                 #
    # ------------------------------------------------------------------ #

    def register_handler(
        self, fn: Callable[[str, str, str], None]
    ) -> None:
        """
        Register an external delivery function.

        fn(level: str, title: str, body: str) -> None

        Example — Telegram stub:
            def send_telegram(level, title, body):
                message = f"[{level}] {title}\n{body}"
                requests.post(TELEGRAM_URL, json={"text": message})
            alerts.register_handler(send_telegram)
        """
        self._handlers.append(fn)
        logger.debug("Alert handler registered: %s", fn)

    def reset_daily_warnings(self) -> None:
        """Call at midnight reset to allow limit warnings to fire again."""
        self._warned_today.clear()

    # ------------------------------------------------------------------ #
    # Typed alert methods                                                  #
    # ------------------------------------------------------------------ #

    def on_trade_entry(
        self,
        signal: Dict[str, Any],
        guardian_status: Dict[str, Any],
    ) -> None:
        """Fire when a trade is placed (approved + filled)."""
        direction = "LONG" if signal.get("direction") == 1 else "SHORT"
        title = f"TRADE ENTRY — {signal.get('instrument')} {direction}"
        body = (
            f"Strategy:    {signal.get('strategy_name', signal.get('strategy', ''))}\n"
            f"Lot size:    {signal.get('lot_size')}\n"
            f"Entry price: {signal.get('entry_price')}\n"
            f"SL:          {signal.get('sl_price')}  "
            f"TP: {signal.get('tp_price')}\n"
            f"Balance:     ${guardian_status.get('current_balance', 0):,.2f}  "
            f"Equity: ${guardian_status.get('current_equity', 0):,.2f}\n"
            f"Daily DD:    {guardian_status.get('daily_drawdown_pct', 0):.2f}% used  "
            f"({guardian_status.get('daily_drawdown_remaining_pct', 0):.2f}% remaining)"
        )
        self.send("INFO", title, body)

    def on_trade_exit(
        self,
        result: Dict[str, Any],
        guardian_status: Dict[str, Any],
    ) -> None:
        """Fire when a position is closed."""
        pnl   = result.get("pnl_dollars", 0.0) or 0.0
        pips  = result.get("pnl_pips", 0.0) or 0.0
        emoji = "✓" if pnl >= 0 else "✗"
        title = (
            f"TRADE EXIT {emoji} — {result.get('instrument')} "
            f"${pnl:+.2f}"
        )
        body = (
            f"Strategy:    {result.get('strategy', '')}\n"
            f"Exit reason: {result.get('exit_reason', result.get('reason', ''))}\n"
            f"P&L:         {pips:+.1f} pips  /  ${pnl:+.2f}\n"
            f"Entry:       {result.get('entry_price')}  "
            f"Exit: {result.get('exit_price')}\n"
            f"Balance now: ${guardian_status.get('current_balance', 0):,.2f}  "
            f"Equity: ${guardian_status.get('current_equity', 0):,.2f}\n"
            f"Daily P&L:   ${guardian_status.get('daily_pnl', 0):+.2f}"
        )
        self.send("INFO", title, body)

    def on_daily_summary(
        self,
        guardian_status:   Dict[str, Any],
        regime:            str = "",
        strategies_active: Optional[List[str]] = None,
    ) -> None:
        """Fire once per day at the daily close."""
        pnl    = guardian_status.get("daily_pnl", 0.0)
        pnl_pct = (
            pnl / guardian_status.get("midnight_balance", 1.0) * 100.0
            if guardian_status.get("midnight_balance") else 0.0
        )
        status_str = (
            "HALTED" if guardian_status.get("permanently_halted") else
            "daily-halted" if guardian_status.get("daily_halted") else
            "OK"
        )
        title = f"DAILY SUMMARY — ${pnl:+.2f}  ({pnl_pct:+.2f}%)"
        body = (
            f"Balance:         ${guardian_status.get('current_balance', 0):,.2f}\n"
            f"Equity:          ${guardian_status.get('current_equity', 0):,.2f}\n"
            f"Daily P&L:       ${pnl:+.2f}  ({pnl_pct:+.2f}%)\n"
            f"Daily DD used:   {guardian_status.get('daily_drawdown_pct', 0):.2f}%  "
            f"(limit {guardian_status.get('daily_drawdown_limit_pct', 4):.1f}%)\n"
            f"Total DD:        {guardian_status.get('total_drawdown_pct', 0):.2f}%  "
            f"(limit {guardian_status.get('total_drawdown_limit_pct', 9):.1f}%)\n"
            f"Regime:          {regime}\n"
            f"Active strats:   {', '.join(strategies_active or [])}\n"
            f"Guardian status: {status_str}"
        )
        level = "CRITICAL" if guardian_status.get("permanently_halted") else "INFO"
        self.send(level, title, body)

    def on_guardian_block(
        self,
        reason: str,
        guardian_status: Dict[str, Any],
    ) -> None:
        """Fire when guardian rejects a trade or triggers a halt."""
        is_permanent = guardian_status.get("permanently_halted", False)
        level = "CRITICAL" if is_permanent else "WARNING"
        title = (
            "PERMANENT HALT — ALL TRADING STOPPED"
            if is_permanent else
            "GUARDIAN BLOCK — trade rejected"
        )
        body = (
            f"Reason: {reason}\n"
            f"Balance:   ${guardian_status.get('current_balance', 0):,.2f}\n"
            f"Equity:    ${guardian_status.get('current_equity', 0):,.2f}\n"
            f"Daily DD:  {guardian_status.get('daily_drawdown_pct', 0):.2f}%\n"
            f"Total DD:  {guardian_status.get('total_drawdown_pct', 0):.2f}%"
        )
        self.send(level, title, body)

    def on_approaching_limit(
        self,
        limit_type: str,    # "daily_loss" | "total_drawdown"
        used_pct:   float,  # percentage of the limit already consumed
        limit_pct:  float,  # the limit itself (e.g. 4.0 or 9.0)
    ) -> None:
        """
        Fire when used_pct/limit_pct × 100 >= daily_limit_warn_pct.
        Deduplicates: fires at most once per limit_type per day.
        """
        key = f"{limit_type}_{int(used_pct)}"
        if key in self._warned_today:
            return
        self._warned_today.add(key)

        fraction = used_pct / limit_pct * 100.0 if limit_pct else 0.0
        title = f"APPROACHING {limit_type.upper().replace('_', ' ')} LIMIT"
        body = (
            f"{used_pct:.2f}% of {limit_pct:.1f}% limit used  "
            f"({fraction:.0f}% consumed)\n"
            f"Only {limit_pct - used_pct:.2f}% remaining before halt."
        )
        self.send("WARNING", title, body)

    def on_error(self, context: str, exc: Exception) -> None:
        """Fire on unexpected exceptions in the live loop."""
        title = f"ERROR — {context}"
        body  = (
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        self.send("CRITICAL", title, body)

    def check_limits(self, guardian_status: Dict[str, Any]) -> None:
        """
        Inspect guardian status and fire approaching-limit warnings if needed.

        Call this every equity-check cycle (~60 s).
        """
        daily_used  = guardian_status.get("daily_drawdown_pct", 0.0)
        daily_limit = guardian_status.get("daily_drawdown_limit_pct", 4.0)
        total_used  = guardian_status.get("total_drawdown_pct", 0.0)
        total_limit = guardian_status.get("total_drawdown_limit_pct", 9.0)

        if daily_limit and (daily_used / daily_limit * 100.0) >= self._warn_thresh:
            self.on_approaching_limit("daily_loss", daily_used, daily_limit)

        if total_limit and (total_used / total_limit * 100.0) >= self._warn_thresh:
            self.on_approaching_limit("total_drawdown", total_used, total_limit)

        if guardian_status.get("permanently_halted"):
            self.on_guardian_block(
                guardian_status.get("halt_reason", "unknown"),
                guardian_status,
            )

    # ------------------------------------------------------------------ #
    # Core delivery                                                        #
    # ------------------------------------------------------------------ #

    def send(self, level: str, title: str, body: str) -> None:
        """
        Deliver an alert through all registered channels.

        Override ``_deliver()`` in a subclass to change the default
        (terminal + file) behaviour without touching the typed methods.
        """
        self._deliver(level, title, body)
        for fn in self._handlers:
            try:
                fn(level, title, body)
            except Exception as exc:
                logger.error("Alert handler %s failed: %s", fn, exc)

    # ------------------------------------------------------------------ #
    # Default delivery: terminal + log file                                #
    # ------------------------------------------------------------------ #

    def _deliver(self, level: str, title: str, body: str) -> None:
        now    = datetime.now(_EASTERN).strftime("%Y-%m-%d %H:%M:%S ET")
        border = "=" * 60 if level == "CRITICAL" else "-" * 50

        lines = [
            border,
            f"[{level}]  {now}",
            f"  {title}",
        ]
        for line in body.splitlines():
            lines.append(f"  {line}")
        lines.append(border)

        output = "\n".join(lines)

        # Terminal — use appropriate Python logger level
        log_fn = {
            "CRITICAL": logger.critical,
            "WARNING":  logger.warning,
        }.get(level, logger.info)
        log_fn("%s: %s", title, body.replace("\n", " | "))

        # Also print directly so it's visible even when log level is high
        print(output, flush=True)

        # Append to alert log file
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(output + "\n")
        except OSError as exc:
            logger.error("Failed to write alert log: %s", exc)
