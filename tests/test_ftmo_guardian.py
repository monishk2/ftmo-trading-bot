"""
Tests for execution/ftmo_guardian.py

Test plan (35 tests across 15 classes)
---------------------------------------
Daily-loss rule
  1.  Daily loss blocks at exactly 4 % (update_equity path)
  2.  Daily loss blocks at > 4 %
  3.  Daily loss at 3.99 % does NOT block
  4.  Approval double-check also triggers daily halt
  5.  use_higher_of_balance_equity = True uses max(balance, equity) as reference
  6.  use_higher_of_balance_equity = False uses midnight_balance only

Total-drawdown rule
  7.  Total drawdown blocks at exactly 9 %
  8.  Total drawdown blocks at > 9 %
  9.  Total drawdown at 8.99 % does NOT block
  10. Permanent halt survives daily_reset

Weekend rule
  11. Friday at exactly 16:00 ET → blocked
  12. Friday at 15:59 ET → allowed
  13. Saturday → blocked
  14. Wednesday mid-day → allowed

Consecutive-loss rule
  15. 3rd consecutive loss triggers daily halt
  16. Win resets consecutive loss counter → 3 losses after win does NOT halt prematurely
  17. Consecutive loss counter resets at daily_reset

Max-positions rule
  18. 3rd position rejected when limit is 2
  19. 2nd position approved when limit is 2
  20. Approved trade increments open_positions counter

Position-size rule
  21. Risk exceeds 1 % of balance → rejected
  22. Risk exactly at 1 % limit → approved
  23. Risk below limit → approved

Approval reason strings
  24. Each rejection category returns a non-empty descriptive reason

State persistence
  25. save_state / load_state round-trip: daily halt, permanent halt, consecutive losses
  26. Auto-save via state_path: every approve_trade writes to disk
  27. State survives a simulated restart (new FTMOGuardian with same state_path)

daily_reset
  28. Resets daily_halted and consecutive_losses
  29. Does NOT reset permanent halt
  30. Pins new midnight_balance / midnight_equity

get_status
  31. All expected keys present
  32. Remaining-room values decrease as drawdown increases

force_close_all
  33. Resets open_positions to 0
  34. Sets daily_halted

News filter
  35. is_news_window always returns False (placeholder)
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
import pytz

from execution.ftmo_guardian import (
    ApprovalResult,
    FTMOGuardian,
    TradeRecord,
    TradeRequest,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EASTERN = pytz.timezone("US/Eastern")


def _cfg(
    daily_trigger: float = 4.0,
    total_trigger: float = 9.0,
    max_positions: int = 2,
    max_consec: int = 3,
    max_risk_pct: float = 1.0,
    weekend_hour: int = 16,
    use_higher: bool = False,
    pip_value: float = 10.0,
) -> dict:
    """Minimal config dict for FTMOGuardian (no file I/O required)."""
    return {
        "safety_buffers": {
            "daily_loss_trigger_pct": daily_trigger,
            "total_loss_trigger_pct": total_trigger,
        },
        "guardian": {
            "max_open_positions":         max_positions,
            "max_consecutive_losses":     max_consec,
            "max_risk_per_trade_pct":     max_risk_pct,
            "weekend_close_hour":         weekend_hour,
            "use_higher_of_balance_equity": use_higher,
            "pip_value_per_lot":          pip_value,
        },
    }


def _guardian(
    initial: float = 10_000.0,
    mode: str = "challenge",
    use_higher: bool = False,
    state_path: Optional[Path] = None,
    **kw,
) -> FTMOGuardian:
    return FTMOGuardian(
        initial_balance=initial,
        config=_cfg(use_higher=use_higher, **kw),
        mode=mode,
        state_path=state_path,
    )


def _ts(
    weekday_iso: int,
    hour: int,
    minute: int = 0,
    year: int = 2024,
) -> pd.Timestamp:
    """
    Create a UTC-aware Timestamp for a given ISO weekday (1=Mon … 7=Sun),
    at the specified hour:minute in US/Eastern.

    2024-01-01 is Monday (ISO 1). We offset from that.
    """
    # Monday 2024-01-01 in Eastern
    monday = _EASTERN.localize(datetime(2024, 1, 1, hour, minute))
    offset  = weekday_iso - 1
    target  = monday.replace(day=monday.day + offset)
    return pd.Timestamp(target).tz_convert("UTC")


def _friday(hour: int = 12, minute: int = 0) -> pd.Timestamp:
    return _ts(5, hour, minute)


def _wednesday(hour: int = 10, minute: int = 0) -> pd.Timestamp:
    return _ts(3, hour, minute)


def _saturday(hour: int = 10) -> pd.Timestamp:
    return _ts(6, hour)


def _req(
    lot_size: float = 0.01,
    sl_pips: float = 10.0,
    timestamp: Optional[pd.Timestamp] = None,
    instrument: str = "EURUSD",
) -> TradeRequest:
    """Minimal valid TradeRequest for a Wednesday trade."""
    if timestamp is None:
        timestamp = _wednesday()
    return TradeRequest(
        instrument=instrument,
        direction=1,
        lot_size=lot_size,
        sl_pips=sl_pips,
        entry_price=1.1000,
        sl_price=1.0990,
        tp_price=1.1020,
        timestamp=timestamp,
    )


def _loss(pnl: float = -100.0) -> TradeRecord:
    return TradeRecord(pnl_dollars=pnl, closed_at=_wednesday())


def _win(pnl: float = 200.0) -> TradeRecord:
    return TradeRecord(pnl_dollars=pnl, closed_at=_wednesday())


# ---------------------------------------------------------------------------
# 1–6  Daily loss rule
# ---------------------------------------------------------------------------

class TestDailyLossRule:
    """Rule 1: daily_drawdown / midnight_reference >= 4% → block."""

    def test_blocks_at_exactly_4pct(self):
        g = _guardian(initial=10_000.0)
        # midnight_balance = 10,000 → 4% drawdown = equity 9,600
        g.update_equity(9_600.0, open_pnl=0.0)
        assert g._daily_halted

    def test_blocks_above_4pct(self):
        g = _guardian(initial=10_000.0)
        g.update_equity(9_500.0)   # 5 % drawdown
        assert g._daily_halted

    def test_allows_below_4pct(self):
        g = _guardian(initial=10_000.0)
        # 3.99 % drawdown: equity = 10_000 × (1 - 0.0399) = 9_601
        g.update_equity(9_601.0)
        assert not g._daily_halted

    def test_approve_trade_double_checks_daily_dd(self):
        """approve_trade re-evaluates drawdown even if update_equity was not called."""
        g = _guardian(initial=10_000.0)
        # Manually set equity below threshold without calling update_equity
        g._equity = 9_600.0
        result = g.approve_trade(_req())
        assert not result.approved
        assert "DAILY LOSS" in result.reason or "DAILY HALT" in result.reason

    def test_use_higher_true_uses_equity_when_higher(self):
        """
        use_higher=True: midnight_equity=10,500 > midnight_balance=10,000
        → reference = 10,500.  4% of 10,500 = 420.
        Equity must drop to 10,500 - 420 = 10,080 to trigger.
        """
        g = _guardian(initial=10_000.0, use_higher=True)
        g._midnight_balance = 10_000.0
        g._midnight_equity  = 10_500.0
        # equity = 10,080 → dd = 420 / 10,500 = 4.0 % exactly → trigger
        g.update_equity(10_080.0)
        assert g._daily_halted

    def test_use_higher_false_ignores_midnight_equity(self):
        """
        use_higher=False: reference = midnight_balance only (10,000).
        equity = 9,700 → dd = 3% → NOT triggered.
        """
        g = _guardian(initial=10_000.0, use_higher=False)
        g._midnight_balance = 10_000.0
        g._midnight_equity  = 10_500.0  # higher but ignored
        g.update_equity(9_700.0)        # 3% of 10,000 → ok
        assert not g._daily_halted

    def test_daily_halt_blocks_subsequent_approvals(self):
        g = _guardian(initial=10_000.0)
        g.update_equity(9_600.0)     # triggers halt
        result = g.approve_trade(_req())
        assert not result.approved
        assert "DAILY HALT" in result.reason

    def test_halt_reason_mentions_drawdown_pct(self):
        g = _guardian(initial=10_000.0)
        g.update_equity(9_600.0)
        assert "4." in g._halt_reason or "4%" in g._halt_reason or "4.00" in g._halt_reason


# ---------------------------------------------------------------------------
# 7–10  Total drawdown rule
# ---------------------------------------------------------------------------

class TestTotalDrawdownRule:
    """Rule 2: (initial - equity) / initial >= 9% → permanent halt."""

    def test_blocks_at_exactly_9pct(self):
        g = _guardian(initial=10_000.0)
        # 9% drawdown: equity = 9,100
        g.update_equity(9_100.0)
        assert g._permanently_halted

    def test_blocks_above_9pct(self):
        g = _guardian(initial=10_000.0)
        g.update_equity(8_000.0)   # 20% drawdown
        assert g._permanently_halted

    def test_allows_below_9pct(self):
        g = _guardian(initial=10_000.0)
        # 8.99%: equity = 10,000 × (1 - 0.0899) = 9,101
        g.update_equity(9_101.0)
        assert not g._permanently_halted

    def test_permanent_halt_survives_daily_reset(self):
        g = _guardian(initial=10_000.0)
        g.update_equity(9_100.0)           # total-dd halt
        assert g._permanently_halted
        g.daily_reset(9_100.0, 9_100.0)   # midnight reset
        assert g._permanently_halted       # still halted

    def test_permanent_halt_blocks_approval(self):
        g = _guardian(initial=10_000.0)
        g.update_equity(9_100.0)
        result = g.approve_trade(_req())
        assert not result.approved
        assert "PERMANENTLY HALTED" in result.reason

    def test_halt_reason_preserved_after_reset(self):
        g = _guardian(initial=10_000.0)
        g.update_equity(9_100.0)
        original_reason = g._halt_reason
        g.daily_reset(9_100.0, 9_100.0)
        assert g._halt_reason == original_reason


# ---------------------------------------------------------------------------
# 11–14  Weekend rule
# ---------------------------------------------------------------------------

class TestWeekendRule:
    """Rule 4: Friday >= 16:00 ET → block. Saturday/Sunday → block."""

    def test_friday_at_16_blocked(self):
        g = _guardian()
        result = g.approve_trade(_req(timestamp=_friday(16, 0)))
        assert not result.approved
        assert "WEEKEND" in result.reason

    def test_friday_at_1601_blocked(self):
        g = _guardian()
        result = g.approve_trade(_req(timestamp=_friday(16, 1)))
        assert not result.approved

    def test_friday_at_1559_allowed(self):
        g = _guardian()
        result = g.approve_trade(_req(timestamp=_friday(15, 59)))
        assert result.approved

    def test_saturday_blocked(self):
        g = _guardian()
        result = g.approve_trade(_req(timestamp=_saturday()))
        assert not result.approved
        assert "WEEKEND" in result.reason

    def test_wednesday_allowed(self):
        g = _guardian()
        result = g.approve_trade(_req(timestamp=_wednesday()))
        assert result.approved

    def test_weekend_reason_contains_day(self):
        g = _guardian()
        result = g.approve_trade(_req(timestamp=_saturday()))
        assert "Saturday" in result.reason or "Weekend" in result.reason


# ---------------------------------------------------------------------------
# 15–17  Consecutive loss rule
# ---------------------------------------------------------------------------

class TestConsecutiveLossRule:
    """Rule 6: 3 consecutive losses → daily halt."""

    def test_three_losses_triggers_halt(self):
        g = _guardian()
        g.record_trade_result(_loss(-50))
        g.record_trade_result(_loss(-50))
        assert not g._daily_halted   # 2 losses → still ok
        g.record_trade_result(_loss(-50))
        assert g._daily_halted       # 3rd loss → halted

    def test_win_resets_counter(self):
        g = _guardian()
        g.record_trade_result(_loss(-50))
        g.record_trade_result(_loss(-50))
        g.record_trade_result(_win(100))    # win resets counter to 0
        assert g._consecutive_losses == 0
        g.record_trade_result(_loss(-50))
        g.record_trade_result(_loss(-50))
        g.record_trade_result(_loss(-50))   # 3 losses post-win → halt
        assert g._daily_halted

    def test_counter_resets_at_daily_reset(self):
        g = _guardian()
        g.record_trade_result(_loss(-50))
        g.record_trade_result(_loss(-50))
        assert g._consecutive_losses == 2
        g.daily_reset(10_000.0, 10_000.0)
        assert g._consecutive_losses == 0
        assert not g._daily_halted

    def test_consecutive_losses_blocks_approval(self):
        g = _guardian()
        for _ in range(3):
            g.record_trade_result(_loss(-50))
        result = g.approve_trade(_req())
        assert not result.approved
        assert "DAILY HALT" in result.reason

    def test_two_losses_and_approve_still_works(self):
        g = _guardian()
        g.record_trade_result(_loss(-50))
        g.record_trade_result(_loss(-50))
        result = g.approve_trade(_req())
        assert result.approved


# ---------------------------------------------------------------------------
# 18–20  Max open positions rule
# ---------------------------------------------------------------------------

class TestMaxPositionsRule:
    """Rule 5: never more than max_open_positions (2) simultaneously."""

    def test_third_position_rejected(self):
        g = _guardian(max_positions=2)
        g.approve_trade(_req())    # open 1
        g.approve_trade(_req())    # open 2
        result = g.approve_trade(_req())
        assert not result.approved
        assert "MAX POSITIONS" in result.reason

    def test_second_position_approved(self):
        g = _guardian(max_positions=2)
        g.approve_trade(_req())
        result = g.approve_trade(_req())
        assert result.approved

    def test_approved_trade_increments_counter(self):
        g = _guardian()
        assert g._open_positions == 0
        g.approve_trade(_req())
        assert g._open_positions == 1
        g.approve_trade(_req())
        assert g._open_positions == 2

    def test_record_result_decrements_counter(self):
        g = _guardian()
        g.approve_trade(_req())
        assert g._open_positions == 1
        g.record_trade_result(_win())
        assert g._open_positions == 0

    def test_positions_never_go_negative(self):
        g = _guardian()
        g.record_trade_result(_win())   # without an approve first
        assert g._open_positions == 0


# ---------------------------------------------------------------------------
# 21–23  Position size / risk rule
# ---------------------------------------------------------------------------

class TestPositionSizeRule:
    """Rule 3: lot_size × sl_pips × pip_value > max_risk_pct × balance → reject."""

    def test_risk_exceeds_limit_rejected(self):
        """
        balance = 10,000 | max_risk = 1% = 100
        risk = 1.1 lots × 10 pips × $10 = 110 > 100 → reject
        """
        g = _guardian(initial=10_000.0, max_risk_pct=1.0)
        result = g.approve_trade(_req(lot_size=1.1, sl_pips=10.0))
        assert not result.approved
        assert "RISK TOO HIGH" in result.reason

    def test_risk_exactly_at_limit_approved(self):
        """
        balance = 10,000 | max_risk = 1% = 100
        risk = 1.0 lot × 10 pips × $10 = 100 == 100 → approved
        """
        g = _guardian(initial=10_000.0, max_risk_pct=1.0)
        result = g.approve_trade(_req(lot_size=1.0, sl_pips=10.0))
        assert result.approved

    def test_risk_below_limit_approved(self):
        """risk = 0.01 lots × 10 pips × $10 = $1 << $100 → approved"""
        g = _guardian(initial=10_000.0)
        result = g.approve_trade(_req(lot_size=0.01, sl_pips=10.0))
        assert result.approved

    def test_risk_reason_contains_amounts(self):
        g = _guardian(initial=10_000.0, max_risk_pct=1.0)
        result = g.approve_trade(_req(lot_size=1.1, sl_pips=10.0))
        assert "$" in result.reason or "risk" in result.reason.lower()


# ---------------------------------------------------------------------------
# 24  Rejection reason strings
# ---------------------------------------------------------------------------

class TestRejectionReasons:
    """Every rejection must return an ApprovalResult with a clear reason."""

    def test_perm_halt_reason_not_empty(self):
        g = _guardian()
        g.update_equity(9_100.0)
        r = g.approve_trade(_req())
        assert r.reason and len(r.reason) > 5

    def test_daily_halt_reason_not_empty(self):
        g = _guardian()
        g.update_equity(9_600.0)
        r = g.approve_trade(_req())
        assert r.reason and len(r.reason) > 5

    def test_weekend_reason_not_empty(self):
        g = _guardian()
        r = g.approve_trade(_req(timestamp=_friday(17)))
        assert r.reason and len(r.reason) > 5

    def test_max_positions_reason_not_empty(self):
        g = _guardian(max_positions=1)
        g.approve_trade(_req())
        r = g.approve_trade(_req())
        assert r.reason and len(r.reason) > 5

    def test_risk_reason_not_empty(self):
        g = _guardian(max_risk_pct=0.001)   # very tight limit
        r = g.approve_trade(_req(lot_size=1.0, sl_pips=100.0))
        assert r.reason and len(r.reason) > 5

    def test_all_approvals_return_true(self):
        g = _guardian()
        r = g.approve_trade(_req())
        assert r.approved
        assert r.reason


# ---------------------------------------------------------------------------
# 25–27  State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    """save_state / load_state round-trip and auto-save."""

    def test_save_load_roundtrip_halt_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian.json"
            g1 = _guardian(initial=10_000.0, state_path=path)
            # Trigger total drawdown halt
            g1.update_equity(9_100.0)
            assert g1._permanently_halted

            # New instance, same state_path — should reload halted state
            g2 = _guardian(initial=10_000.0, state_path=path)
            assert g2._permanently_halted

    def test_save_load_consecutive_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian.json"
            g1 = _guardian(state_path=path)
            g1.record_trade_result(_loss(-50))
            g1.record_trade_result(_loss(-50))
            assert g1._consecutive_losses == 2

            g2 = _guardian(state_path=path)
            assert g2._consecutive_losses == 2

    def test_save_load_open_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian.json"
            g1 = _guardian(state_path=path)
            g1.approve_trade(_req())
            assert g1._open_positions == 1

            g2 = _guardian(state_path=path)
            assert g2._open_positions == 1

    def test_auto_save_on_approve(self):
        """Every approve_trade must write state to disk."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian.json"
            g = _guardian(state_path=path)
            assert not path.exists()   # nothing written yet
            g.approve_trade(_req())
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["open_positions"] == 1

    def test_explicit_save_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = _guardian()
            g._consecutive_losses = 2
            save_path = Path(tmp) / "out.json"
            g.save_state(save_path)
            data = json.loads(save_path.read_text())
            assert data["consecutive_losses"] == 2

    def test_state_json_keys_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            g = _guardian(state_path=path)
            g.approve_trade(_req())
            data = json.loads(path.read_text())
            expected_keys = {
                "mode", "initial_balance", "balance", "equity",
                "midnight_balance", "midnight_equity",
                "open_positions", "consecutive_losses", "daily_pnl",
                "daily_halted", "permanently_halted", "halt_reason",
                "trading_day",
            }
            assert expected_keys.issubset(data.keys())

    def test_simulated_crash_recovery(self):
        """Process A saves state; process B (new guardian) continues seamlessly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crash.json"

            # "Process A": approve two trades, take one loss
            g_a = _guardian(initial=10_000.0, state_path=path)
            g_a.approve_trade(_req())
            g_a.approve_trade(_req())
            g_a.record_trade_result(_loss(-50))

            # "Process B": new guardian loads recovered state
            g_b = _guardian(initial=10_000.0, state_path=path)
            assert g_b._open_positions     == 1
            assert g_b._consecutive_losses == 1
            assert not g_b._daily_halted


# ---------------------------------------------------------------------------
# 28–30  daily_reset
# ---------------------------------------------------------------------------

class TestDailyReset:
    """Midnight reset must unblock daily halt but never touch permanent halt."""

    def test_reset_clears_daily_halt(self):
        g = _guardian()
        g.update_equity(9_600.0)     # daily dd halt
        assert g._daily_halted
        g.daily_reset(9_600.0, 9_600.0)
        assert not g._daily_halted

    def test_reset_clears_consecutive_losses(self):
        g = _guardian()
        g.record_trade_result(_loss(-50))
        g.record_trade_result(_loss(-50))
        g.daily_reset(10_000.0, 10_000.0)
        assert g._consecutive_losses == 0

    def test_reset_does_not_clear_permanent_halt(self):
        g = _guardian()
        g.update_equity(9_100.0)     # total-dd → permanent halt
        g.daily_reset(9_100.0, 9_100.0)
        assert g._permanently_halted
        result = g.approve_trade(_req())
        assert not result.approved

    def test_reset_pins_new_midnight_balance(self):
        g = _guardian(initial=10_000.0)
        g.daily_reset(10_200.0, 10_200.0)
        assert g._midnight_balance == 10_200.0

    def test_reset_pins_new_midnight_equity(self):
        g = _guardian(initial=10_000.0)
        g.daily_reset(10_200.0, 10_300.0)
        assert g._midnight_equity == 10_300.0

    def test_reset_clears_daily_pnl(self):
        g = _guardian()
        g.record_trade_result(_win(500))
        assert g._daily_pnl > 0
        g.daily_reset(10_500.0, 10_500.0)
        assert g._daily_pnl == 0.0


# ---------------------------------------------------------------------------
# 31–32  get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    _REQUIRED_KEYS = {
        "mode", "initial_balance", "current_balance", "current_equity",
        "midnight_balance", "midnight_equity", "daily_reference",
        "use_higher_of_balance_equity",
        "daily_drawdown_pct", "daily_drawdown_limit_pct",
        "daily_drawdown_remaining_pct",
        "total_drawdown_pct", "total_drawdown_limit_pct",
        "total_drawdown_remaining_pct",
        "open_positions", "max_open_positions",
        "consecutive_losses", "max_consecutive_losses",
        "daily_pnl", "daily_halted", "permanently_halted",
        "halt_reason", "trading_day",
    }

    def test_all_keys_present(self):
        g = _guardian()
        status = g.get_status()
        assert self._REQUIRED_KEYS.issubset(status.keys())

    def test_remaining_room_decreases_with_drawdown(self):
        g = _guardian(initial=10_000.0)
        before = g.get_status()["daily_drawdown_remaining_pct"]
        g.update_equity(9_800.0)   # 2% drawdown
        after = g.get_status()["daily_drawdown_remaining_pct"]
        assert after < before

    def test_mode_reflected(self):
        g = _guardian(mode="funded")
        assert g.get_status()["mode"] == "funded"

    def test_initial_balance_correct(self):
        g = _guardian(initial=25_000.0)
        assert g.get_status()["initial_balance"] == 25_000.0


# ---------------------------------------------------------------------------
# 33–34  force_close_all
# ---------------------------------------------------------------------------

class TestForceCloseAll:
    def test_resets_open_positions(self):
        g = _guardian()
        g.approve_trade(_req())
        g.approve_trade(_req())
        assert g._open_positions == 2
        g.force_close_all("test reason")
        assert g._open_positions == 0

    def test_sets_daily_halted(self):
        g = _guardian()
        g.force_close_all("manual stop")
        assert g._daily_halted

    def test_does_not_override_permanent_halt(self):
        g = _guardian()
        g.update_equity(9_100.0)   # permanent halt
        g.force_close_all("manual")
        assert g._permanently_halted  # unchanged


# ---------------------------------------------------------------------------
# 35  News filter placeholder
# ---------------------------------------------------------------------------

class TestNewsFilter:
    def test_is_news_window_returns_false(self):
        g = _guardian()
        assert g.is_news_window() is False

    def test_is_news_window_with_timestamp_returns_false(self):
        g = _guardian()
        ts = _wednesday()
        assert g.is_news_window(ts) is False

    def test_news_window_does_not_block_trade(self):
        """Since is_news_window always returns False, trades proceed normally."""
        g = _guardian()
        result = g.approve_trade(_req())
        assert result.approved
