"""
FTMO Guardian
=============

The single most critical module in the entire project. Has ABSOLUTE VETO
POWER over all trades. order_manager MUST call approve_trade() before placing
ANY order — there is no bypass.

Rules enforced
--------------
1.  DAILY LOSS
    At midnight US/Eastern, the guardian pins a reference balance:
      • use_higher_of_balance_equity = False  → reference = midnight_balance
                                               (FTMO standard)
      • use_higher_of_balance_equity = True   → reference =
                                               max(midnight_balance, midnight_equity)
    Throughout the day:
      daily_drawdown = reference − current_equity
      If daily_drawdown / reference × 100 >= daily_loss_trigger_pct (4 %)
        → block all new trades AND close all open positions immediately.
    Note: midnight_balance = closed-trade-only account balance at 00:00 ET.
          It does NOT move if you profit intra-day; it only changes at the
          next midnight reset.

2.  TOTAL DRAWDOWN
    If equity ≤ (1 − total_loss_trigger_pct/100) × initial_balance (9 %)
    → block ALL trades PERMANENTLY, close positions, log CRITICAL alert.
    This halt can never be reversed (survives daily_reset calls).

3.  POSITION SIZE
    Reject any trade where:
      risk_dollars = lot_size × sl_pips × pip_value_per_lot
    exceeds max_risk_per_trade_pct% × current_balance.

4.  WEEKEND
    Friday after 16:00 US/Eastern → block new trades, close all positions.
    Saturday / Sunday → same.

5.  MAX OPEN POSITIONS
    Never more than max_open_positions (2) simultaneously.

6.  CONSECUTIVE LOSSES
    After max_consecutive_losses (3) consecutive losses in a day → stop
    for rest of day. Counter resets on a winning trade or at midnight reset.

7.  NEWS FILTER
    Placeholder is_news_window() always returns False. Wire to a live
    news-calendar API (ForexFactory / Myfxbook) when ready.

State persistence
-----------------
    State is saved to JSON after every mutation so the guardian survives
    process crashes and restarts without losing its counters.

Public API
----------
    guardian = FTMOGuardian(initial_balance, config, mode, state_path)
    guardian.approve_trade(req)          -> ApprovalResult
    guardian.update_equity(equity, pnl)  -> None   (called every ~60 s)
    guardian.record_trade_result(trade)  -> None   (called at trade close)
    guardian.daily_reset(balance, equity)-> None   (called at midnight ET)
    guardian.get_status()                -> dict
    guardian.force_close_all(reason)     -> None
    guardian.save_state(path)            -> None
    guardian.load_state(path)            -> None
    guardian.is_news_window(timestamp)   -> bool   (placeholder → False)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"

# USD per pip per standard lot for USD-quoted pairs (EURUSD, GBPUSD).
# pip_size = 0.0001 ; contract = 100,000 units ; pip_value ≈ $10.
PIP_VALUE_PER_LOT = 10.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TradeRequest:
    """Proposed trade submitted to approve_trade()."""
    instrument: str
    direction: int             # 1 = long, -1 = short
    lot_size: float
    sl_pips: float             # stop-loss distance in pips (positive value)
    entry_price: float
    sl_price: float
    tp_price: float
    timestamp: pd.Timestamp    # UTC-aware; used for weekend / session checks


@dataclass
class ApprovalResult:
    """Return value of approve_trade()."""
    approved: bool
    reason: str


@dataclass
class TradeRecord:
    """Outcome of a completed trade, passed to record_trade_result()."""
    pnl_dollars: float         # negative = loss
    closed_at: pd.Timestamp    # UTC-aware


# ---------------------------------------------------------------------------
# Guardian
# ---------------------------------------------------------------------------

class FTMOGuardian:
    """
    FTMO risk-rule enforcer with absolute veto power over all trades.

    Parameters
    ----------
    initial_balance :
        Account balance at the very start of the challenge / funded phase.
        Used as the fixed denominator for total-drawdown calculations.
        Never changes once set.
    config :
        Parsed ftmo_rules.json (top-level dict) or a compatible dict with
        "safety_buffers" and "guardian" sub-keys.
        If None, loaded automatically from config/ftmo_rules.json.
    mode :
        "challenge", "verification", or "funded".
    state_path :
        If provided, state is auto-saved to this JSON file after every
        mutation and auto-loaded on construction (crash recovery).
    """

    # ------------------------------------------------------------------ #
    # Construction                                                          #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        initial_balance: float,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "challenge",
        state_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if config is None:
            with open(_CONFIG_DIR / "ftmo_rules.json") as fh:
                config = json.load(fh)

        buffers     = config.get("safety_buffers", config)
        guardian_cfg = config.get("guardian", {})

        self.mode = mode
        self._initial_balance: float = float(initial_balance)

        # --- Thresholds (all read from config, never hard-coded here) ---
        self._daily_trigger: float = float(
            buffers.get("daily_loss_trigger_pct", 4.0))
        self._total_trigger: float = float(
            buffers.get("total_loss_trigger_pct", 9.0))
        self._max_positions: int = int(
            guardian_cfg.get("max_open_positions", 2))
        self._max_consec_losses: int = int(
            guardian_cfg.get("max_consecutive_losses", 3))
        self._max_risk_pct: float = float(
            guardian_cfg.get("max_risk_per_trade_pct", 1.0))
        self._weekend_close_hour: int = int(
            guardian_cfg.get("weekend_close_hour", 16))
        self._use_higher: bool = bool(
            guardian_cfg.get("use_higher_of_balance_equity", True))
        self._pip_value: float = float(
            guardian_cfg.get("pip_value_per_lot", PIP_VALUE_PER_LOT))

        # --- Mutable state -------------------------------------------------
        self._balance: float            = float(initial_balance)
        self._equity: float             = float(initial_balance)
        self._midnight_balance: float   = float(initial_balance)
        self._midnight_equity: float    = float(initial_balance)
        self._open_positions: int       = 0
        self._consecutive_losses: int   = 0
        self._daily_pnl: float          = 0.0
        self._daily_halted: bool        = False
        self._permanently_halted: bool  = False
        self._halt_reason: str          = ""
        self._trading_day: str          = ""    # YYYY-MM-DD Eastern

        # --- State persistence ---------------------------------------------
        self._state_path: Optional[Path] = (
            Path(state_path).expanduser().resolve() if state_path else None
        )
        if self._state_path and self._state_path.exists():
            self._load_state_from_file()
            logger.info("Guardian state restored from %s", self._state_path)

        logger.info(
            "FTMOGuardian ready | mode=%s initial=%.2f "
            "daily_trigger=%.1f%% total_trigger=%.1f%% "
            "use_higher=%s",
            mode, initial_balance, self._daily_trigger, self._total_trigger,
            self._use_higher,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def approve_trade(self, req: TradeRequest) -> ApprovalResult:
        """
        Evaluate a proposed trade against all FTMO rules.

        Checks are applied in this order (fail-fast):
          1. Permanent halt
          2. Daily halt  (daily-loss or consecutive-loss already triggered)
          3. News window (placeholder)
          4. Weekend / session cutoff
          5. Max open positions
          6. Position size / risk-per-trade
          7. Re-evaluate daily drawdown (defence-in-depth double-check)

        On approval: increments open_positions and saves state.
        On rejection: logs warning with the specific rule violated.
        """
        # 1. Permanent halt
        if self._permanently_halted:
            return self._reject("PERMANENTLY HALTED", self._halt_reason, "perm_halt")

        # 2. Daily halt
        if self._daily_halted:
            return self._reject("DAILY HALT", self._halt_reason, "daily_halt")

        # 3. News window
        if self.is_news_window(req.timestamp):
            return self._reject(
                "NEWS WINDOW",
                "no trading during high-impact news events",
                "news",
            )

        # 4. Weekend / cutoff
        blocked, w_reason = self._check_weekend(req.timestamp)
        if blocked:
            return self._reject("WEEKEND RULE", w_reason, "weekend")

        # 5. Max open positions
        if self._open_positions >= self._max_positions:
            return self._reject(
                "MAX POSITIONS",
                f"{self._open_positions} positions open (limit {self._max_positions})",
                "max_positions",
            )

        # 6. Position size
        ok, r_reason = self._check_position_size(req)
        if not ok:
            return self._reject("RISK TOO HIGH", r_reason, "risk")

        # 7. Daily drawdown double-check
        dd_ok, dd_reason = self._evaluate_daily_drawdown()
        if not dd_ok:
            self._trigger_daily_halt(dd_reason)
            return self._reject("DAILY LOSS", dd_reason, "daily_dd")

        # ── All checks passed ─────────────────────────────────────────
        self._open_positions += 1
        self._save_state()
        logger.info(
            "Trade APPROVED | %s %s %.2f lots sl=%.1f pips | open=%d",
            "LONG" if req.direction == 1 else "SHORT",
            req.instrument, req.lot_size, req.sl_pips, self._open_positions,
        )
        return ApprovalResult(True, "All FTMO rules satisfied")

    def update_equity(
        self,
        current_equity: float,
        open_pnl: float = 0.0,
        open_positions_count: Optional[int] = None,
    ) -> None:
        """
        Called every ~60 seconds by the live position monitor.

        Parameters
        ----------
        current_equity :
            Broker-reported equity: closed-trade balance + unrealised P&L.
        open_pnl :
            Sum of unrealised P&L on all open positions.  Used to
            back-calculate the closed-trade-only balance.
        open_positions_count :
            If provided, syncs the internal position counter with the
            broker's actual count (prevents drift).
        """
        self._equity  = float(current_equity)
        self._balance = self._equity - float(open_pnl)

        if open_positions_count is not None:
            self._open_positions = int(open_positions_count)

        # Check total drawdown first (permanent)
        total_dd_pct = (
            (self._initial_balance - self._equity) / self._initial_balance * 100.0
        )
        if total_dd_pct >= self._total_trigger and not self._permanently_halted:
            reason = (
                f"Total drawdown {total_dd_pct:.2f}% >= "
                f"trigger {self._total_trigger:.1f}% "
                f"(equity={self._equity:.2f}, initial={self._initial_balance:.2f})"
            )
            self._trigger_permanent_halt(reason)

        # Check daily drawdown (daily, resets at midnight)
        if not self._daily_halted and not self._permanently_halted:
            ok, reason = self._evaluate_daily_drawdown()
            if not ok:
                self._trigger_daily_halt(reason)

        self._save_state()

    def record_trade_result(self, trade: TradeRecord) -> None:
        """
        Called after each trade closes (filled or force-closed).

        Updates balance, daily P&L, consecutive loss counter, and
        open-position count.  Triggers daily halt if max consecutive
        losses is reached.
        """
        pnl = float(trade.pnl_dollars)
        self._balance            += pnl
        self._equity              = self._balance   # approximate until next update_equity
        self._daily_pnl          += pnl
        self._open_positions      = max(0, self._open_positions - 1)

        if pnl < 0.0:
            self._consecutive_losses += 1
            logger.info(
                "Trade result: LOSS %.2f | consecutive_losses=%d/%d",
                pnl, self._consecutive_losses, self._max_consec_losses,
            )
            if (self._consecutive_losses >= self._max_consec_losses
                    and not self._daily_halted
                    and not self._permanently_halted):
                self._trigger_daily_halt(
                    f"{self._consecutive_losses} consecutive losses "
                    f"(limit {self._max_consec_losses})"
                )
        else:
            logger.info(
                "Trade result: WIN %.2f | consecutive_losses reset 0",
                pnl,
            )
            self._consecutive_losses = 0

        self._save_state()

    def daily_reset(
        self,
        current_balance: float,
        current_equity: float,
    ) -> None:
        """
        Call at midnight US/Eastern (00:00 ET).

        Pins the new reference balance/equity for today's drawdown
        calculations, then resets all daily counters.

        Parameters
        ----------
        current_balance :
            Closed-trade-only account balance at midnight (from broker).
        current_equity :
            Balance + unrealised at midnight (from broker).
        """
        prev_day = self._trading_day
        now_et   = pd.Timestamp.now(tz="US/Eastern")
        self._trading_day = now_et.strftime("%Y-%m-%d")

        self._midnight_balance = float(current_balance)
        self._midnight_equity  = float(current_equity)

        # Reset daily state — permanent halt is NEVER reversed
        self._daily_pnl          = 0.0
        self._consecutive_losses = 0
        if not self._permanently_halted:
            self._daily_halted = False
            self._halt_reason  = ""

        self._save_state()
        logger.info(
            "Daily reset | %s → %s | midnight_balance=%.2f midnight_equity=%.2f",
            prev_day, self._trading_day,
            self._midnight_balance, self._midnight_equity,
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Return a complete snapshot of the guardian's state and limits.

        Intended for dashboard display, health-check endpoints, and logs.
        """
        ref = self._daily_reference()
        daily_dd_pct = (ref - self._equity) / ref * 100.0 if ref > 0 else 0.0
        total_dd_pct = (
            (self._initial_balance - self._equity) / self._initial_balance * 100.0
        )
        return {
            "mode":                          self.mode,
            "initial_balance":               self._initial_balance,
            "current_balance":               round(self._balance, 2),
            "current_equity":                round(self._equity, 2),
            "midnight_balance":              round(self._midnight_balance, 2),
            "midnight_equity":               round(self._midnight_equity, 2),
            "daily_reference":               round(ref, 2),
            "use_higher_of_balance_equity":  self._use_higher,
            "daily_drawdown_pct":            round(daily_dd_pct, 4),
            "daily_drawdown_limit_pct":      self._daily_trigger,
            "daily_drawdown_remaining_pct":  round(
                max(0.0, self._daily_trigger - daily_dd_pct), 4),
            "total_drawdown_pct":            round(total_dd_pct, 4),
            "total_drawdown_limit_pct":      self._total_trigger,
            "total_drawdown_remaining_pct":  round(
                max(0.0, self._total_trigger - total_dd_pct), 4),
            "open_positions":                self._open_positions,
            "max_open_positions":            self._max_positions,
            "consecutive_losses":            self._consecutive_losses,
            "max_consecutive_losses":        self._max_consec_losses,
            "daily_pnl":                     round(self._daily_pnl, 2),
            "daily_halted":                  self._daily_halted,
            "permanently_halted":            self._permanently_halted,
            "halt_reason":                   self._halt_reason,
            "trading_day":                   self._trading_day,
        }

    def force_close_all(self, reason: str) -> None:
        """
        Immediately invalidate all open positions and halt trading.

        The actual order cancellation is the responsibility of order_manager
        (which monitors the guardian's state).  This method resets the
        position counter and triggers a daily halt so no new trades can be
        approved after the close.
        """
        logger.critical("FORCE CLOSE ALL: %s", reason)
        self._open_positions = 0
        if not self._permanently_halted:
            self._trigger_daily_halt(f"Force close: {reason}")
        self._save_state()

    def is_news_window(
        self, timestamp: Optional[pd.Timestamp] = None
    ) -> bool:
        """
        News filter placeholder — always returns False.

        To activate: replace with a real calendar check, e.g.
          return _news_calendar.is_high_impact_within_30min(timestamp)
        """
        return False

    def save_state(self, filepath: Union[str, Path]) -> None:
        """Explicitly persist guardian state to a JSON file."""
        path = Path(filepath).expanduser().resolve()
        self._write_state(path)
        logger.debug("State saved to %s", path)

    def load_state(self, filepath: Union[str, Path]) -> None:
        """Restore guardian state from a previously saved JSON file."""
        path = Path(filepath).expanduser().resolve()
        self._state_path = path
        self._load_state_from_file()
        logger.info("State loaded from %s", path)

    # ------------------------------------------------------------------ #
    # Internal — rule evaluation                                           #
    # ------------------------------------------------------------------ #

    def _daily_reference(self) -> float:
        """
        Reference balance for daily drawdown.

        use_higher_of_balance_equity = True  → max(midnight_balance, midnight_equity)
        use_higher_of_balance_equity = False → midnight_balance only
        """
        if self._use_higher:
            return max(self._midnight_balance, self._midnight_equity)
        return self._midnight_balance

    def _evaluate_daily_drawdown(self):
        """
        Returns (within_limit: bool, reason: str).
        Triggered when daily_drawdown / reference >= daily_loss_trigger_pct.
        """
        ref = self._daily_reference()
        if ref <= 0:
            return True, ""
        dd_pct = (ref - self._equity) / ref * 100.0
        if dd_pct >= self._daily_trigger:
            reason = (
                f"Daily drawdown {dd_pct:.2f}% >= trigger {self._daily_trigger:.1f}% "
                f"(equity={self._equity:.2f}, reference={ref:.2f})"
            )
            return False, reason
        return True, ""

    def _check_weekend(self, timestamp: pd.Timestamp):
        """Return (blocked: bool, reason: str)."""
        et      = timestamp.tz_convert("US/Eastern")
        weekday = et.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun

        if weekday == 4 and et.hour >= self._weekend_close_hour:
            return True, (
                f"Friday {et.strftime('%H:%M')} ET — "
                f"no new trades after {self._weekend_close_hour:02d}:00"
            )
        if weekday >= 5:
            return True, f"Weekend ({et.strftime('%A')}) — market closed"
        return False, ""

    def _check_position_size(self, req: TradeRequest):
        """Return (ok: bool, reason: str)."""
        risk_dollars = req.lot_size * req.sl_pips * self._pip_value
        max_risk     = self._max_risk_pct / 100.0 * self._balance
        if risk_dollars > max_risk:
            reason = (
                f"risk=${risk_dollars:.2f} "
                f"(lot={req.lot_size:.3f} × sl={req.sl_pips:.1f}pips "
                f"× pip_val=${self._pip_value:.0f}) "
                f"> max=${max_risk:.2f} "
                f"({self._max_risk_pct:.1f}% of balance ${self._balance:.2f})"
            )
            return False, reason
        return True, ""

    # ------------------------------------------------------------------ #
    # Internal — halt triggers                                             #
    # ------------------------------------------------------------------ #

    def _trigger_daily_halt(self, reason: str) -> None:
        """Set the daily-halt flag. Does NOT affect the permanent-halt flag."""
        self._daily_halted    = True
        self._halt_reason     = reason
        self._open_positions  = 0
        logger.warning("DAILY HALT: %s", reason)

    def _trigger_permanent_halt(self, reason: str) -> None:
        """Set the permanent-halt flag. Irreversible — survives daily resets."""
        self._permanently_halted = True
        self._daily_halted       = True
        self._halt_reason        = reason
        self._open_positions     = 0
        logger.critical("PERMANENT HALT: %s", reason)

    # ------------------------------------------------------------------ #
    # Internal — helpers                                                   #
    # ------------------------------------------------------------------ #

    def _reject(self, rule: str, detail: str, tag: str) -> ApprovalResult:
        reason = f"{rule}: {detail}"
        logger.warning("Trade REJECTED [%s] %s", tag, reason)
        return ApprovalResult(False, reason)

    # ------------------------------------------------------------------ #
    # State persistence                                                    #
    # ------------------------------------------------------------------ #

    def _state_dict(self) -> Dict[str, Any]:
        return {
            "mode":               self.mode,
            "initial_balance":    self._initial_balance,
            "balance":            self._balance,
            "equity":             self._equity,
            "midnight_balance":   self._midnight_balance,
            "midnight_equity":    self._midnight_equity,
            "open_positions":     self._open_positions,
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl":          self._daily_pnl,
            "daily_halted":       self._daily_halted,
            "permanently_halted": self._permanently_halted,
            "halt_reason":        self._halt_reason,
            "trading_day":        self._trading_day,
        }

    def _apply_state_dict(self, d: Dict[str, Any]) -> None:
        self.mode                = d.get("mode",               self.mode)
        self._balance            = float(d.get("balance",            self._balance))
        self._equity             = float(d.get("equity",             self._equity))
        self._midnight_balance   = float(d.get("midnight_balance",   self._midnight_balance))
        self._midnight_equity    = float(d.get("midnight_equity",    self._midnight_equity))
        self._open_positions     = int(d.get("open_positions",       self._open_positions))
        self._consecutive_losses = int(d.get("consecutive_losses",   self._consecutive_losses))
        self._daily_pnl          = float(d.get("daily_pnl",          self._daily_pnl))
        self._daily_halted       = bool(d.get("daily_halted",        self._daily_halted))
        self._permanently_halted = bool(d.get("permanently_halted",  self._permanently_halted))
        self._halt_reason        = str(d.get("halt_reason",          self._halt_reason))
        self._trading_day        = str(d.get("trading_day",          self._trading_day))

    def _write_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state_dict(), indent=2), encoding="utf-8")

    def _load_state_from_file(self) -> None:
        if self._state_path and self._state_path.exists():
            raw = self._state_path.read_text(encoding="utf-8")
            self._apply_state_dict(json.loads(raw))

    def _save_state(self) -> None:
        if self._state_path:
            self._write_state(self._state_path)
