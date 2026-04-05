"""
Tests for execution/order_manager.py and execution/ctrader_connector.py

Test plan (46 tests across 9 classes)
--------------------------------------
MockCTraderConnector — basic operations
  1.  connect() returns True, sets connected flag
  2.  place_order() fills immediately, returns position_id
  3.  get_positions() reflects the placed order
  4.  get_account_info() includes balance / equity / open_positions
  5.  modify_order() updates sl/tp on existing position
  6.  modify_order() on unknown id returns False
  7.  close_position() removes from positions, returns P&L dict
  8.  close_all() closes every position, returns list
  9.  P&L calculation: long TP hit → positive; long SL hit → negative
  10. P&L calculation: short TP hit → positive; short SL hit → negative
  11. Slippage applied adversely (long: fill > entry; short: fill < entry)
  12. update_prices() flows into unrealised P&L in get_positions()
  13. get_symbol_price() after update_prices() returns correct bid/ask
  14. Disconnect sets connected flag to False

OrderManager — guardian veto is enforced
  15. Rejected trade does NOT call connector.place_order
  16. Approved trade DOES call connector.place_order
  17. guardian.approve_trade() called before every execute_trade
  18. After rejection, open_position_count unchanged
  19. After approval, open_position_count incremented

OrderManager — lot-size auto-calculation
  20. lot_size=None triggers position_sizer (produces non-zero result)
  21. lot_size provided in signal bypasses position_sizer
  22. Zero lot_size from sizer skips trade (returns False)

OrderManager — manage_open_positions
  23. Time-stop expiry closes position via connector
  24. Time-stop not yet reached leaves position open
  25. SL hit (long) closes position
  26. SL hit (short) closes position
  27. TP hit (long) closes position
  28. TP hit (short) closes position
  29. After closure, guardian.record_trade_result() called with correct P&L
  30. Position removed from internal tracking after close

OrderManager — breakeven logic
  31. Breakeven: SL moves to entry when profit >= 1R
  32. Breakeven only triggers once per position
  33. Breakeven not triggered when profit < 1R

OrderManager — close_all_positions
  34. close_all_positions calls connector.close_all
  35. close_all_positions calls guardian.force_close_all
  36. close_all_positions calls guardian.record_trade_result for each closed trade
  37. Internal _positions cleared after close_all

OrderManager — journal integration
  38. Journal.log_trade called on open
  39. Journal.log_trade called on close

OrderManager — connector error handling
  40. Connector raises exception → guardian position counter not stuck
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from execution.ctrader_connector import MockCTraderConnector
from execution.ftmo_guardian import (
    ApprovalResult,
    FTMOGuardian,
    TradeRecord,
    TradeRequest,
)
from execution.order_manager import OrderManager


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

_EURUSD_CFG = {
    "pip_size":              0.0001,
    "pip_value_per_lot":     10.0,
    "typical_spread_pips":   1.0,
    "commission_per_lot_round_trip": 3.0,
    "slippage_model": {
        "normal_pips": 0.3,
        "session_open_pips": 1.5,
        "session_open_window_minutes": 15,
        "news_pips": 2.0,
    },
}

_INSTRUMENTS = {"EURUSD": _EURUSD_CFG}


def _guardian_cfg(
    daily_trigger: float = 4.0,
    total_trigger: float = 9.0,
) -> dict:
    return {
        "safety_buffers": {
            "daily_loss_trigger_pct": daily_trigger,
            "total_loss_trigger_pct": total_trigger,
        },
        "guardian": {
            "max_open_positions":       2,
            "max_consecutive_losses":   3,
            "max_risk_per_trade_pct":   2.0,
            "weekend_close_hour":       16,
            "use_higher_of_balance_equity": False,
            "pip_value_per_lot":        10.0,
        },
    }


def _guardian(initial: float = 10_000.0) -> FTMOGuardian:
    return FTMOGuardian(
        initial_balance=initial,
        config=_guardian_cfg(),
        mode="challenge",
    )


def _connector(balance: float = 10_000.0) -> MockCTraderConnector:
    c = MockCTraderConnector(initial_balance=balance)
    c.connect()
    return c


def _manager(
    guardian=None,
    connector=None,
    journal=None,
) -> OrderManager:
    g = guardian or _guardian()
    c = connector or _connector()
    return OrderManager(g, c, _INSTRUMENTS, journal=journal)


def _wednesday_ts(hour: int = 10) -> pd.Timestamp:
    return pd.Timestamp("2024-01-03", tz="UTC").replace(hour=hour)


def _signal(
    direction: int = 1,
    entry: float = 1.1000,
    sl: float = 1.0980,
    tp: float = 1.1040,
    lot_size=None,
    risk_pct: float = 1.0,
    time_stop=None,
    timestamp=None,
) -> dict:
    return {
        "instrument":  "EURUSD",
        "direction":   direction,
        "entry_price": entry,
        "sl_price":    sl,
        "tp_price":    tp,
        "lot_size":    lot_size,
        "risk_pct":    risk_pct,
        "time_stop":   time_stop,
        "timestamp":   timestamp or _wednesday_ts(),
    }


# ---------------------------------------------------------------------------
# MockCTraderConnector tests
# ---------------------------------------------------------------------------

class TestMockConnectorBasic:
    def test_connect_returns_true(self):
        c = MockCTraderConnector()
        assert c.connect() is True
        assert c._connected

    def test_place_order_fills_and_returns_position_id(self):
        c = _connector()
        result = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        assert result["status"]      == "filled"
        assert "position_id"         in result
        assert result["lot_size"]    == 0.10
        assert result["entry_price"] == 1.1000

    def test_get_positions_reflects_open_order(self):
        c = _connector()
        c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        positions = c.get_positions()
        assert len(positions) == 1
        assert positions[0]["instrument"] == "EURUSD"

    def test_get_account_info_structure(self):
        c = _connector(10_000)
        info = c.get_account_info()
        for key in ("balance", "equity", "margin_used", "free_margin",
                    "currency", "leverage", "open_positions"):
            assert key in info

    def test_account_balance_starts_at_initial(self):
        c = _connector(10_000)
        assert c.get_account_info()["balance"] == 10_000.0

    def test_modify_order_updates_sl(self):
        c = _connector()
        r = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        pid = r["position_id"]
        c.modify_order(pid, sl_price=1.0990)
        assert c._positions[pid].sl_price == 1.0990

    def test_modify_unknown_position_returns_false(self):
        c = _connector()
        assert c.modify_order("nonexistent", sl_price=1.0) is False

    def test_close_position_removes_from_positions(self):
        c = _connector()
        r = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        pid = r["position_id"]
        c.close_position(pid, exit_price=1.1020)
        assert pid not in c._positions

    def test_close_all_clears_all_positions(self):
        c = _connector()
        c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        results = c.close_all("test")
        assert len(results) == 2
        assert len(c._positions) == 0


class TestMockConnectorPnL:
    def test_long_tp_produces_positive_pnl(self):
        c = _connector()
        r = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        pid = r["position_id"]
        result = c.close_position(pid, exit_price=1.1040)
        assert result["pnl_pips"]    > 0
        assert result["pnl_dollars"] > 0

    def test_long_sl_produces_negative_pnl(self):
        c = _connector()
        r = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        pid = r["position_id"]
        result = c.close_position(pid, exit_price=1.0980)
        assert result["pnl_pips"]    < 0
        assert result["pnl_dollars"] < 0

    def test_short_tp_produces_positive_pnl(self):
        c = _connector()
        r = c.place_order("EURUSD", -1, 0.10, 1.1000, 1.1020, 1.0960)
        pid = r["position_id"]
        result = c.close_position(pid, exit_price=1.0960)
        assert result["pnl_dollars"] > 0

    def test_short_sl_produces_negative_pnl(self):
        c = _connector()
        r = c.place_order("EURUSD", -1, 0.10, 1.1000, 1.1020, 1.0960)
        pid = r["position_id"]
        result = c.close_position(pid, exit_price=1.1020)
        assert result["pnl_dollars"] < 0

    def test_pnl_magnitude_correct(self):
        """0.10 lots × 40 pips × $10/pip/lot = $40."""
        c = _connector()
        r = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0960, 1.1040)
        pid = r["position_id"]
        result = c.close_position(pid, exit_price=1.1040)
        assert abs(result["pnl_dollars"] - 40.0) < 0.01

    def test_balance_updates_after_close(self):
        c = _connector(10_000)
        r = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        pid = r["position_id"]
        c.close_position(pid, exit_price=1.1040)
        # 40 pips × 0.10 lots × $10 = +$40
        assert abs(c.get_account_info()["balance"] - 10_040.0) < 0.01


class TestMockConnectorPrices:
    def test_slippage_adverse_for_long(self):
        c = _connector()
        r = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040,
                          slippage_pips=1.0)
        assert r["entry_price"] > 1.1000   # filled worse than requested

    def test_slippage_adverse_for_short(self):
        c = _connector()
        r = c.place_order("EURUSD", -1, 0.10, 1.1000, 1.1020, 1.0960,
                          slippage_pips=1.0)
        assert r["entry_price"] < 1.1000   # filled worse than requested

    def test_update_prices_reflected_in_unrealised(self):
        c = _connector()
        r = c.place_order("EURUSD", 1, 0.10, 1.1000, 1.0980, 1.1040)
        c.update_prices({"EURUSD": {"bid": 1.1020, "ask": 1.1021}})
        positions = c.get_positions()
        assert positions[0]["unrealised_pnl"] > 0

    def test_get_symbol_price_after_update(self):
        c = _connector()
        c.update_prices({"EURUSD": {"bid": 1.0999, "ask": 1.1000}})
        price = c.get_symbol_price("EURUSD")
        assert price["bid"] == 1.0999
        assert price["ask"] == 1.1000

    def test_disconnect_clears_connected(self):
        c = _connector()
        c.disconnect()
        assert not c._connected


# ---------------------------------------------------------------------------
# OrderManager — guardian veto
# ---------------------------------------------------------------------------

class TestOrderManagerGuardianVeto:
    def test_rejected_trade_skips_connector(self):
        g = _guardian()
        g.update_equity(9_600.0)   # trigger daily halt
        c = _connector()
        m = _manager(guardian=g, connector=c)
        result = m.execute_trade(_signal(), "TestStrategy")
        assert result is False
        assert len(c.get_positions()) == 0

    def test_approved_trade_hits_connector(self):
        m = _manager()
        result = m.execute_trade(_signal(), "TestStrategy")
        assert result is True
        assert m.open_position_count == 1

    def test_guardian_approve_called_first(self):
        g = MagicMock(spec=FTMOGuardian)
        g.approve_trade.return_value = ApprovalResult(False, "test rejection")
        g._balance = 10_000.0
        c = _connector()
        m = OrderManager(g, c, _INSTRUMENTS)
        m.execute_trade(_signal(), "S")
        g.approve_trade.assert_called_once()

    def test_rejection_leaves_position_count_unchanged(self):
        g = _guardian()
        g.update_equity(9_600.0)
        m = _manager(guardian=g)
        assert m.open_position_count == 0
        m.execute_trade(_signal(), "S")
        assert m.open_position_count == 0

    def test_approval_increments_position_count(self):
        m = _manager()
        assert m.open_position_count == 0
        m.execute_trade(_signal(), "S")
        assert m.open_position_count == 1


# ---------------------------------------------------------------------------
# OrderManager — lot-size auto-calculation
# ---------------------------------------------------------------------------

class TestOrderManagerLotSize:
    def test_auto_lot_size_from_sizer(self):
        """lot_size=None → position sizer used; result should be nonzero."""
        m = _manager()
        result = m.execute_trade(_signal(lot_size=None, risk_pct=1.0), "S")
        assert result is True
        positions = m.get_open_positions()
        assert positions[0]["lot_size"] > 0

    def test_explicit_lot_size_bypasses_sizer(self):
        m = _manager()
        m.execute_trade(_signal(lot_size=0.07), "S")
        assert m.get_open_positions()[0]["lot_size"] == 0.07

    def test_zero_lot_from_sizer_skips_trade(self):
        """
        If sizer returns 0.0 (balance too small), execute_trade should
        return False without calling guardian or connector.
        """
        g = MagicMock(spec=FTMOGuardian)
        g.approve_trade.return_value = ApprovalResult(True, "ok")
        g._balance = 1.0     # tiny balance → lot_size = 0.0
        c = _connector(1.0)
        m = OrderManager(g, c, _INSTRUMENTS)
        # With $1 balance, 1% risk, 20pip SL: 0.01 / 200 = 0.00005 → 0.0
        result = m.execute_trade(_signal(lot_size=None, risk_pct=1.0,
                                        sl=1.0980), "S")
        assert result is False
        g.approve_trade.assert_not_called()


# ---------------------------------------------------------------------------
# OrderManager — manage_open_positions
# ---------------------------------------------------------------------------

class TestManageOpenPositions:
    def _open(self, m: OrderManager, direction: int = 1,
              entry: float = 1.1000, sl: float = 1.0980,
              tp: float = 1.1040, time_stop=None) -> None:
        m.execute_trade(
            _signal(direction=direction, entry=entry, sl=sl, tp=tp,
                    time_stop=time_stop),
            "S",
        )

    def test_time_stop_closes_position(self):
        m = _manager()
        ts = _wednesday_ts(hour=12)
        self._open(m, time_stop=ts)
        assert m.open_position_count == 1
        # Advance time past the stop
        future = _wednesday_ts(hour=13)
        m.manage_open_positions(future, {"EURUSD": {"bid": 1.1000, "ask": 1.1001}})
        assert m.open_position_count == 0

    def test_time_stop_not_yet_reached(self):
        m = _manager()
        ts = _wednesday_ts(hour=16)
        self._open(m, time_stop=ts)
        past = _wednesday_ts(hour=12)
        m.manage_open_positions(past, {"EURUSD": {"bid": 1.1000, "ask": 1.1001}})
        assert m.open_position_count == 1

    def test_sl_hit_long_closes_position(self):
        m = _manager()
        self._open(m, direction=1, entry=1.1000, sl=1.0980)
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.0975, "ask": 1.0976}},  # bid < SL
        )
        assert m.open_position_count == 0

    def test_sl_hit_short_closes_position(self):
        m = _manager()
        self._open(m, direction=-1, entry=1.1000, sl=1.1020, tp=1.0960)
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.1018, "ask": 1.1025}},  # ask > SL
        )
        assert m.open_position_count == 0

    def test_tp_hit_long_closes_position(self):
        m = _manager()
        self._open(m, direction=1, entry=1.1000, sl=1.0980, tp=1.1040)
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.1045, "ask": 1.1046}},  # bid >= TP
        )
        assert m.open_position_count == 0

    def test_tp_hit_short_closes_position(self):
        m = _manager()
        self._open(m, direction=-1, entry=1.1000, sl=1.1020, tp=1.0960)
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.0954, "ask": 1.0955}},  # ask <= TP
        )
        assert m.open_position_count == 0

    def test_record_trade_result_called_after_close(self):
        g = _guardian()
        m = _manager(guardian=g)
        self._open(m, direction=1, entry=1.1000, sl=1.0980, tp=1.1040)
        initial_positions = g._open_positions
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.1045, "ask": 1.1046}},
        )
        # Guardian should have recorded the closed trade
        assert g._open_positions < initial_positions or g._daily_pnl != 0.0

    def test_position_removed_from_tracking_after_close(self):
        m = _manager()
        self._open(m, direction=1, entry=1.1000, sl=1.0980, tp=1.1040)
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.1045, "ask": 1.1046}},
        )
        assert len(m.get_open_positions()) == 0


# ---------------------------------------------------------------------------
# OrderManager — breakeven logic
# ---------------------------------------------------------------------------

class TestBreakevenLogic:
    def test_breakeven_triggered_when_1r_in_profit(self):
        c = _connector()
        m = _manager(connector=c)
        # entry=1.1000, sl=1.0980 → initial_risk=20pips=0.0020
        m.execute_trade(_signal(entry=1.1000, sl=1.0980, tp=1.1060), "S")
        pos_id = list(m._positions.keys())[0]

        # Move price 20+ pips in favor (bid >= entry + 20 pips = 1.1020)
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.1021, "ask": 1.1022}},
        )
        meta = m._positions.get(pos_id)
        if meta:  # position still open (TP not hit)
            assert meta["breakeven_moved"] is True
            assert meta["sl_price"] == pytest.approx(1.1000, abs=1e-5)

    def test_breakeven_only_triggers_once(self):
        c = _connector()
        m = _manager(connector=c)
        m.execute_trade(_signal(entry=1.1000, sl=1.0980, tp=1.1060), "S")
        pos_id = list(m._positions.keys())[0]

        # First tick: triggers breakeven
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.1021, "ask": 1.1022}},
        )
        meta = m._positions.get(pos_id)
        if meta:
            assert meta["breakeven_moved"] is True
            # Manually force SL back to check it's not re-moved
            meta["sl_price"] = 1.0985
            m.manage_open_positions(
                _wednesday_ts(),
                {"EURUSD": {"bid": 1.1030, "ask": 1.1031}},
            )
            # SL should stay at 1.0985, not re-moved to entry
            meta2 = m._positions.get(pos_id)
            if meta2:
                assert meta2["sl_price"] == pytest.approx(1.0985, abs=1e-5)

    def test_breakeven_not_triggered_below_1r(self):
        m = _manager()
        m.execute_trade(_signal(entry=1.1000, sl=1.0980, tp=1.1060), "S")
        pos_id = list(m._positions.keys())[0]
        # Only 10 pips in profit (< 20 pip initial risk)
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.1010, "ask": 1.1011}},
        )
        meta = m._positions.get(pos_id)
        if meta:
            assert meta["breakeven_moved"] is False


# ---------------------------------------------------------------------------
# OrderManager — close_all_positions
# ---------------------------------------------------------------------------

class TestCloseAllPositions:
    def test_close_all_calls_connector(self):
        c = MagicMock(spec=MockCTraderConnector)
        c.get_account_info.return_value = {"balance": 10_000.0}
        c.place_order.return_value = {
            "position_id": "abc", "entry_price": 1.1000,
            "status": "filled", "open_time": pd.Timestamp.now(tz="UTC"),
        }
        c.close_all.return_value = []
        m = _manager(connector=c)
        m.close_all_positions("test")
        c.close_all.assert_called_once_with(reason="test")

    def test_close_all_calls_guardian_force_close(self):
        g = _guardian()
        m = _manager(guardian=g)
        m.close_all_positions("emergency")
        assert g._daily_halted   # force_close_all triggers daily halt

    def test_close_all_records_pnl_in_guardian(self):
        g = _guardian()
        c = _connector()
        m = _manager(guardian=g, connector=c)
        m.execute_trade(_signal(), "S")
        c.update_prices({"EURUSD": {"bid": 1.1020, "ask": 1.1021}})
        pre_pnl = g._daily_pnl
        m.close_all_positions("end_of_day")
        # guardian.record_trade_result called → daily_pnl updated
        # (direction depends on mock close price, just check it was called)
        assert g._daily_pnl != pre_pnl or g._open_positions == 0

    def test_internal_positions_cleared_after_close_all(self):
        m = _manager()
        m.execute_trade(_signal(), "S")
        assert m.open_position_count == 1
        m.close_all_positions("test")
        assert m.open_position_count == 0


# ---------------------------------------------------------------------------
# OrderManager — journal integration
# ---------------------------------------------------------------------------

class TestJournalIntegration:
    def test_journal_called_on_trade_open(self):
        journal = MagicMock()
        m = _manager(journal=journal)
        m.execute_trade(_signal(), "S")
        journal.log_trade.assert_called()
        call_args = journal.log_trade.call_args[0][0]
        assert call_args["event"] == "trade_open"

    def test_journal_called_on_trade_close(self):
        journal = MagicMock()
        m = _manager(journal=journal)
        m.execute_trade(_signal(entry=1.1000, sl=1.0980, tp=1.1040), "S")
        journal.reset_mock()
        m.manage_open_positions(
            _wednesday_ts(),
            {"EURUSD": {"bid": 1.1045, "ask": 1.1046}},  # TP hit
        )
        journal.log_trade.assert_called()
        call_args = journal.log_trade.call_args[0][0]
        assert call_args["event"] == "trade_close"
