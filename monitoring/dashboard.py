"""
Terminal Dashboard
==================

A print-based, self-refreshing status display for live trading mode.
No curses or external UI libraries — just ANSI clear-screen + print,
so it works everywhere (SSH, tmux, VSCode terminal, CI logs).

Refresh cycle
-------------
  In live mode call ``Dashboard.refresh()`` every 30 seconds.
  The dashboard renders the full screen each time and is safe to call
  from any thread (it acquires a lock).

Public API
----------
  dash = Dashboard(guardian, connector, order_manager)
  dash.set_regime(regime_str)
  dash.set_active_strategies(["london_breakout", "fvg_retracement"])
  dash.set_next_action("09:30 ET — NY open / FVG entries")
  dash.set_today_trades(list_of_trade_dicts)   # optional enrichment

  dash.refresh()   # render one frame now

  # Run a blocking 30-second refresh loop:
  dash.run(interval_seconds=30)   # blocks; Ctrl+C to stop
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

_EASTERN = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# ANSI helpers (graceful fallback when terminal doesn't support colour)
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()

_RESET  = "\033[0m"   if _COLOR else ""
_BOLD   = "\033[1m"   if _COLOR else ""
_GREEN  = "\033[32m"  if _COLOR else ""
_RED    = "\033[31m"  if _COLOR else ""
_YELLOW = "\033[33m"  if _COLOR else ""
_CYAN   = "\033[36m"  if _COLOR else ""
_DIM    = "\033[2m"   if _COLOR else ""


def _clr(text: str, *codes: str) -> str:
    if not codes:
        return text
    return "".join(codes) + text + _RESET


def _pct_color(value: float, good_positive: bool = True) -> str:
    """Colour a percentage: green = good, red = bad."""
    if good_positive:
        return _clr(f"{value:+.2f}%", _GREEN if value >= 0 else _RED)
    else:
        return _clr(f"{value:.2f}%", _RED if value > 0 else _GREEN)


def _bar(used: float, total: float, width: int = 20, invert: bool = False) -> str:
    """ASCII progress bar.  invert=True → full bar is bad (drawdown)."""
    frac  = min(max(used / total, 0.0), 1.0) if total else 0.0
    filled = int(frac * width)
    empty  = width - filled
    color  = (_RED if invert else _GREEN) if frac < 0.7 else _YELLOW
    if frac >= 0.9 and invert:
        color = _RED
    bar = _clr("█" * filled, color) + _clr("░" * empty, _DIM)
    return f"[{bar}] {frac*100:.0f}%"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    """
    Single-screen terminal dashboard for live trading status.

    Parameters
    ----------
    guardian :
        FTMOGuardian instance — ``get_status()`` is called on refresh.
    connector :
        CTraderConnector — ``get_account_info()`` and ``get_positions()``
        called on refresh.
    order_manager :
        OrderManager — ``open_position_count`` read on refresh.
    """

    def __init__(
        self,
        guardian,
        connector,
        order_manager,
    ) -> None:
        self._guardian     = guardian
        self._connector    = connector
        self._order_mgr    = order_manager
        self._lock         = threading.Lock()

        # State set externally between refreshes
        self._regime:             str        = "—"
        self._active_strategies:  List[str]  = []
        self._next_action:        str        = "—"
        self._today_trades:       List[Dict] = []
        self._running:            bool       = False

    # ------------------------------------------------------------------ #
    # Setters (thread-safe)                                                #
    # ------------------------------------------------------------------ #

    def set_regime(self, regime: str) -> None:
        with self._lock:
            self._regime = regime

    def set_active_strategies(self, strategies: List[str]) -> None:
        with self._lock:
            self._active_strategies = list(strategies)

    def set_next_action(self, description: str) -> None:
        with self._lock:
            self._next_action = description

    def set_today_trades(self, trades: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._today_trades = list(trades)

    # ------------------------------------------------------------------ #
    # Refresh                                                              #
    # ------------------------------------------------------------------ #

    def refresh(self) -> None:
        """Render one complete screen frame."""
        with self._lock:
            try:
                self._render()
            except Exception as exc:
                # Dashboard must never crash the live loop
                print(f"[Dashboard render error: {exc}]", flush=True)

    def run(self, interval_seconds: int = 30) -> None:
        """Block and refresh every interval_seconds until Ctrl+C."""
        self._running = True
        try:
            while self._running:
                self.refresh()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def _render(self) -> None:
        # Gather data
        try:
            status     = self._guardian.get_status()
        except Exception:
            status = {}

        try:
            acct       = self._connector.get_account_info()
        except Exception:
            acct = {}

        try:
            positions  = self._connector.get_positions()
        except Exception:
            positions = []

        now_et  = pd.Timestamp.now(tz="US/Eastern")
        now_str = now_et.strftime("%Y-%m-%d %H:%M:%S ET")

        # Clear screen
        if _COLOR:
            print("\033[2J\033[H", end="", flush=True)
        else:
            print("\n" * 3, flush=True)

        W = 64   # display width

        def rule(char="─"):
            print(char * W)

        def header(text: str) -> None:
            print(_clr(f"  {text}", _BOLD, _CYAN))

        def kv(label: str, value: str, pad: int = 22) -> None:
            print(f"  {label:<{pad}} {value}")

        # ── Title bar ─────────────────────────────────────────────────
        rule("═")
        title = "  FTMO TRADING BOT  —  LIVE DASHBOARD"
        mode_label = f"[{'SIMULATION' if not getattr(self._connector, '_connected', True) else status.get('mode','').upper()}]"
        print(_clr(f"{title:<{W-len(mode_label)}}{mode_label}", _BOLD))
        print(_clr(f"  {now_str}", _DIM))
        rule("═")

        # ── Account ───────────────────────────────────────────────────
        balance   = acct.get("balance",  status.get("current_balance",  0.0))
        equity    = acct.get("equity",   status.get("current_equity",   0.0))
        daily_pnl = status.get("daily_pnl", 0.0)
        init_bal  = status.get("initial_balance", balance)
        daily_pnl_pct = daily_pnl / status.get("midnight_balance", max(init_bal, 1)) * 100.0

        header("ACCOUNT")
        kv("Balance:",      f"${balance:>12,.2f}")
        kv("Equity:",       f"${equity:>12,.2f}")
        kv("Daily P&L:",    f"${daily_pnl:>+12,.2f}  {_pct_color(daily_pnl_pct)}")
        kv("Initial bal.:", f"${init_bal:>12,.2f}")
        print()

        # ── FTMO Limits ───────────────────────────────────────────────
        daily_dd      = status.get("daily_drawdown_pct",      0.0)
        daily_limit   = status.get("daily_drawdown_limit_pct", 4.0)
        daily_remain  = status.get("daily_drawdown_remaining_pct", daily_limit)
        total_dd      = status.get("total_drawdown_pct",      0.0)
        total_limit   = status.get("total_drawdown_limit_pct", 9.0)
        total_remain  = status.get("total_drawdown_remaining_pct", total_limit)

        header("FTMO LIMITS")
        daily_bar = _bar(daily_dd, daily_limit, invert=True)
        total_bar = _bar(total_dd, total_limit, invert=True)
        kv("Daily loss:",
           f"{_pct_color(daily_dd, good_positive=False)} / {daily_limit:.1f}%  "
           f"(remaining: {daily_remain:.2f}%)")
        print(f"    {daily_bar}")
        kv("Total DD:",
           f"{_pct_color(total_dd, good_positive=False)} / {total_limit:.1f}%  "
           f"(remaining: {total_remain:.2f}%)")
        print(f"    {total_bar}")

        halt_color = _RED if status.get("daily_halted") else _GREEN
        halt_label = (
            "PERMANENTLY HALTED" if status.get("permanently_halted") else
            "DAILY HALTED"       if status.get("daily_halted") else
            "OK"
        )
        kv("Guardian:",     _clr(halt_label, halt_color, _BOLD))
        if status.get("halt_reason"):
            kv("Halt reason:", status.get("halt_reason", "")[:W-24])
        print()

        # ── Open positions ────────────────────────────────────────────
        n_open = len(positions)
        header(f"OPEN POSITIONS  ({n_open})")
        if positions:
            for p in positions:
                dir_str  = "LONG " if p.get("direction") == 1 else "SHORT"
                pnl_val  = p.get("unrealised_pnl", 0.0)
                pnl_str  = _pct_color(pnl_val, good_positive=True)
                print(
                    f"  {p.get('instrument',''):<8} {dir_str}  "
                    f"{p.get('lot_size'):.2f} lots  "
                    f"@ {p.get('entry_price'):.5f}  "
                    f"PnL: ${pnl_val:+.2f}"
                )
                print(
                    f"    SL: {p.get('sl_price'):.5f}  "
                    f"TP: {p.get('tp_price'):.5f}"
                )
        else:
            print("  (none)")
        print()

        # ── Today's trades ────────────────────────────────────────────
        header(f"TODAY'S CLOSED TRADES  ({len(self._today_trades)})")
        if self._today_trades:
            wins   = sum(1 for t in self._today_trades if (t.get("pnl_dollars") or 0) > 0)
            losses = len(self._today_trades) - wins
            tot_pnl = sum((t.get("pnl_dollars") or 0) for t in self._today_trades)
            print(
                f"  Wins: {wins}  Losses: {losses}  "
                f"Total P&L: {_pct_color(tot_pnl, good_positive=True)}"
            )
            for t in self._today_trades[-5:]:   # show last 5
                pnl_d = t.get("pnl_dollars", 0.0) or 0.0
                print(
                    f"  {t.get('instrument',''):<8} "
                    f"{'LONG ' if t.get('direction','')=='LONG' else 'SHORT'} "
                    f"{t.get('strategy',''):<20}  "
                    f"${pnl_d:>+7.2f}  [{t.get('exit_reason','')}]"
                )
        else:
            print("  (no closed trades yet today)")
        print()

        # ── Regime & strategy ─────────────────────────────────────────
        header("MARKET CONDITIONS")
        kv("Regime:",          self._regime)
        kv("Active strats:",   ", ".join(self._active_strategies) or "—")
        kv("Next action:",     self._next_action)
        print()

        # ── Footer ────────────────────────────────────────────────────
        rule()
        print(
            _clr(
                f"  Refreshes every 30s  |  Ctrl+C to exit  |  {now_str}",
                _DIM,
            )
        )
        rule()
        sys.stdout.flush()
