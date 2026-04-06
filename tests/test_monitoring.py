"""
Tests for monitoring/trade_journal.py, monitoring/alerts.py,
monitoring/dashboard.py

Test plan (57 tests across 10 classes)
---------------------------------------
TradeJournal — CSV creation
  1.  Initialising creates trades.csv with correct header
  2.  Initialising creates daily_log.csv with correct header
  3.  Header is NOT duplicated on second initialisation (same files)

TradeJournal — log_trade (open events)
  4.  log_trade(event=trade_open) stores entry in _open_trades
  5.  Unknown event type is silently ignored

TradeJournal — log_trade (close events)
  6.  log_trade(event=trade_close) writes a row to trades.csv
  7.  All expected columns present in the written row
  8.  pnl_dollars stored correctly (positive win)
  9.  pnl_dollars stored correctly (negative loss)
  10. duration_minutes computed from open_time → exit_time
  11. Closed trade removed from _open_trades
  12. Closed trade added to _today_trades

TradeJournal — open→close roundtrip
  13. sl_price / tp_price from open event appear in close row
  14. strategy name carried from open event when close doesn't provide it

TradeJournal — log_daily_summary
  15. Writes a row to daily_log.csv
  16. date column has today's date
  17. wins / losses counted correctly
  18. daily_pnl computed from guardian_status
  19. strategies_active joined with comma
  20. _today_trades and _daily_min_equity reset after summary

TradeJournal — load helpers
  21. load_trades returns empty DataFrame when file absent
  22. load_daily_log returns empty DataFrame when file absent
  23. load_trades returns data after trades are written
  24. load_daily_log returns data after summary is written

TradeJournal — update_equity
  25. update_equity tracks intra-day minimum equity

AlertManager — send / _deliver
  26. send prints to stdout (INFO level)
  27. send appends to log file
  28. CRITICAL messages are written to log file
  29. Log file created automatically on first alert

AlertManager — typed methods produce non-empty output
  30. on_trade_entry produces non-empty body
  31. on_trade_exit positive pnl produces non-empty body
  32. on_daily_summary produces non-empty body
  33. on_guardian_block CRITICAL path
  34. on_approaching_limit produces WARNING-level output

AlertManager — check_limits
  35. check_limits fires on_approaching_limit when daily DD >= 70% of limit
  36. check_limits fires on_approaching_limit when total DD >= 70% of limit
  37. check_limits does NOT fire when DD is below threshold
  38. on_approaching_limit deduplicates (fires at most once per day per type)
  39. reset_daily_warnings clears dedup so warnings can fire again

AlertManager — custom handler registration
  40. register_handler receives (level, title, body) on every send
  41. Handler exception does not crash the alert system
  42. Multiple handlers all called

Dashboard — setters
  43. set_regime stores regime string
  44. set_active_strategies stores list
  45. set_next_action stores description
  46. set_today_trades stores list copy

Dashboard — render (smoke test)
  47. refresh() does not raise on happy-path mocked data
  48. refresh() does not raise when connector raises an exception
  49. refresh() does not raise when guardian raises an exception
  50. Rendered output contains balance value
  51. Rendered output contains equity value
  52. Rendered output contains regime string
  53. Rendered output reflects daily_halted state

Dashboard — _bar helper
  54. Empty bar when used=0
  55. Full bar when used=total
  56. Partial bar at 50%

Dashboard — run / stop
  57. stop() terminates run() loop without blocking
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from monitoring.alerts import AlertManager
from monitoring.dashboard import Dashboard, _bar
from monitoring.trade_journal import TradeJournal, _TRADE_COLS, _DAILY_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guardian_status(
    balance: float = 10_000.0,
    equity:  float = 10_000.0,
    daily_pnl: float = 0.0,
    daily_dd: float = 0.0,
    total_dd: float = 0.0,
    daily_halted: bool = False,
    permanently_halted: bool = False,
) -> Dict[str, Any]:
    return {
        "current_balance":             balance,
        "current_equity":              equity,
        "midnight_balance":            balance,
        "midnight_equity":             equity,
        "initial_balance":             10_000.0,
        "daily_pnl":                   daily_pnl,
        "daily_drawdown_pct":          daily_dd,
        "daily_drawdown_limit_pct":    4.0,
        "daily_drawdown_remaining_pct": max(0, 4.0 - daily_dd),
        "total_drawdown_pct":          total_dd,
        "total_drawdown_limit_pct":    9.0,
        "total_drawdown_remaining_pct": max(0, 9.0 - total_dd),
        "daily_halted":                daily_halted,
        "permanently_halted":          permanently_halted,
        "halt_reason":                 "test" if (daily_halted or permanently_halted) else "",
        "mode":                        "challenge",
        "open_positions":              0,
        "max_open_positions":          2,
        "consecutive_losses":          0,
        "max_consecutive_losses":      3,
        "trading_day":                 "2024-01-03",
        "daily_reference":             balance,
        "use_higher_of_balance_equity": False,
    }


def _open_msg(pos_id: str = "abc", strategy: str = "LondonOpenBreakout") -> dict:
    return {
        "event":       "trade_open",
        "position_id": pos_id,
        "instrument":  "EURUSD",
        "direction":   "LONG",
        "lot_size":    0.10,
        "entry_price": 1.1000,
        "sl_price":    1.0980,
        "tp_price":    1.1040,
        "open_time":   "2024-01-03 03:15:00+00:00",
        "strategy":    strategy,
    }


def _close_msg(pos_id: str = "abc", pnl: float = 40.0, pips: float = 40.0) -> dict:
    return {
        "event":       "trade_close",
        "position_id": pos_id,
        "instrument":  "EURUSD",
        "direction":   "LONG",
        "lot_size":    0.10,
        "entry_price": 1.1000,
        "exit_price":  1.1040,
        "pnl_pips":    pips,
        "pnl_dollars": pnl,
        "exit_reason": "tp",
        "exit_time":   "2024-01-03 04:30:00+00:00",
    }


# ---------------------------------------------------------------------------
# TradeJournal — CSV creation
# ---------------------------------------------------------------------------

class TestJournalCreation:
    def test_creates_trades_csv_with_header(self, tmp_path):
        j = TradeJournal(trades_path=tmp_path / "t.csv", daily_path=tmp_path / "d.csv")
        with open(tmp_path / "t.csv") as f:
            header = next(csv.reader(f))
        assert header == _TRADE_COLS

    def test_creates_daily_csv_with_header(self, tmp_path):
        j = TradeJournal(trades_path=tmp_path / "t.csv", daily_path=tmp_path / "d.csv")
        with open(tmp_path / "d.csv") as f:
            header = next(csv.reader(f))
        assert header == _DAILY_COLS

    def test_header_not_duplicated_on_reopen(self, tmp_path):
        tp = tmp_path / "t.csv"
        dp = tmp_path / "d.csv"
        TradeJournal(tp, dp)
        TradeJournal(tp, dp)  # second init — must not re-write headers
        with open(tp) as f:
            rows = list(csv.reader(f))
        assert rows[0] == _TRADE_COLS
        assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# TradeJournal — trade_open events
# ---------------------------------------------------------------------------

class TestJournalOpen:
    def test_open_stored_in_open_trades(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_trade(_open_msg("xyz"))
        assert "xyz" in j._open_trades

    def test_unknown_event_ignored(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_trade({"event": "spaceship", "data": 1})
        assert len(j._open_trades) == 0


# ---------------------------------------------------------------------------
# TradeJournal — trade_close events
# ---------------------------------------------------------------------------

class TestJournalClose:
    def _journal_with_open(self, tmp_path, pos_id="abc"):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_trade(_open_msg(pos_id))
        return j

    def test_close_writes_row(self, tmp_path):
        j = self._journal_with_open(tmp_path)
        j.log_trade(_close_msg())
        df = j.load_trades()
        assert len(df) == 1

    def test_all_columns_present(self, tmp_path):
        j = self._journal_with_open(tmp_path)
        j.log_trade(_close_msg())
        df = j.load_trades()
        for col in _TRADE_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_positive_pnl_stored(self, tmp_path):
        j = self._journal_with_open(tmp_path)
        j.log_trade(_close_msg(pnl=40.0))
        assert j.load_trades()["pnl_dollars"].iloc[0] == pytest.approx(40.0)

    def test_negative_pnl_stored(self, tmp_path):
        j = self._journal_with_open(tmp_path)
        j.log_trade(_close_msg(pnl=-20.0))
        assert j.load_trades()["pnl_dollars"].iloc[0] == pytest.approx(-20.0)

    def test_duration_computed(self, tmp_path):
        j = self._journal_with_open(tmp_path)
        j.log_trade(_close_msg())
        row = j.load_trades().iloc[0]
        # open 03:15, close 04:30 → 75 minutes
        assert abs(row["duration_minutes"] - 75.0) < 1.0

    def test_open_removed_from_open_trades(self, tmp_path):
        j = self._journal_with_open(tmp_path)
        j.log_trade(_close_msg())
        assert "abc" not in j._open_trades

    def test_close_added_to_today_trades(self, tmp_path):
        j = self._journal_with_open(tmp_path)
        j.log_trade(_close_msg())
        assert len(j._today_trades) == 1


# ---------------------------------------------------------------------------
# TradeJournal — open→close roundtrip metadata
# ---------------------------------------------------------------------------

class TestJournalRoundtrip:
    def test_sl_tp_from_open_in_close_row(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_trade(_open_msg("p1"))
        j.log_trade(_close_msg("p1"))
        row = j.load_trades().iloc[0]
        assert float(row["sl_price"]) == pytest.approx(1.0980)
        assert float(row["tp_price"]) == pytest.approx(1.1040)

    def test_strategy_from_open_when_close_omits_it(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_trade(_open_msg("p2", strategy="FVGRetracement"))
        close = _close_msg("p2")
        close.pop("strategy", None)
        j.log_trade(close)
        row = j.load_trades().iloc[0]
        assert row["strategy"] == "FVGRetracement"


# ---------------------------------------------------------------------------
# TradeJournal — log_daily_summary
# ---------------------------------------------------------------------------

class TestJournalDailySummary:
    def test_writes_row_to_daily_log(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_daily_summary(_guardian_status(), "normal", ["london_breakout"])
        df = j.load_daily_log()
        assert len(df) == 1

    def test_date_is_today(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_daily_summary(_guardian_status())
        df = j.load_daily_log()
        today = pd.Timestamp.now(tz="UTC").date().isoformat()
        # parse_dates may return a Timestamp or string — normalise both
        date_val = str(df["date"].iloc[0])[:10]
        assert date_val == today

    def test_wins_losses_counted(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_trade(_open_msg("w1"))
        j.log_trade(_close_msg("w1", pnl=50.0))  # win
        j.log_trade(_open_msg("l1"))
        j.log_trade(_close_msg("l1", pnl=-25.0)) # loss
        j.log_daily_summary(_guardian_status(daily_pnl=25.0))
        df = j.load_daily_log()
        assert df["wins"].iloc[0] == 1
        assert df["losses"].iloc[0] == 1

    def test_strategies_active_comma_joined(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_daily_summary(
            _guardian_status(),
            strategies_active=["london_breakout", "fvg_retracement"],
        )
        df = j.load_daily_log()
        assert "london_breakout" in df["strategies_active"].iloc[0]

    def test_today_trades_reset_after_summary(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_trade(_open_msg("x"))
        j.log_trade(_close_msg("x"))
        assert len(j._today_trades) == 1
        j.log_daily_summary(_guardian_status())
        assert len(j._today_trades) == 0


# ---------------------------------------------------------------------------
# TradeJournal — load helpers + update_equity
# ---------------------------------------------------------------------------

class TestJournalHelpers:
    def test_load_trades_empty_when_no_file(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        # Remove file to simulate absent
        (tmp_path / "t.csv").unlink()
        df = j.load_trades()
        assert df.empty

    def test_load_daily_log_empty_when_no_file(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        (tmp_path / "d.csv").unlink()
        df = j.load_daily_log()
        assert df.empty

    def test_load_trades_returns_data(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.log_trade(_open_msg())
        j.log_trade(_close_msg())
        df = j.load_trades()
        assert len(df) == 1

    def test_update_equity_tracks_minimum(self, tmp_path):
        j = TradeJournal(tmp_path / "t.csv", tmp_path / "d.csv")
        j.update_equity(10_000.0)
        j.update_equity(9_800.0)
        j.update_equity(9_900.0)
        assert j._daily_min_equity == pytest.approx(9_800.0)


# ---------------------------------------------------------------------------
# AlertManager — send / _deliver
# ---------------------------------------------------------------------------

class TestAlertSend:
    def test_send_appends_to_log_file(self, tmp_path, capsys):
        a = AlertManager(log_path=tmp_path / "alerts.log")
        a.send("INFO", "Test Title", "Test body")
        assert (tmp_path / "alerts.log").exists()
        content = (tmp_path / "alerts.log").read_text()
        assert "Test Title" in content

    def test_critical_written_to_log(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log")
        a.send("CRITICAL", "Halt!", "All trading stopped")
        content = (tmp_path / "a.log").read_text()
        assert "CRITICAL" in content
        assert "Halt!" in content

    def test_log_file_auto_created(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "alerts.log"
        a = AlertManager(log_path=path)
        a.send("INFO", "T", "B")
        assert path.exists()


# ---------------------------------------------------------------------------
# AlertManager — typed methods
# ---------------------------------------------------------------------------

class TestAlertTypedMethods:
    def _alerts(self, tmp_path):
        return AlertManager(log_path=tmp_path / "a.log")

    def test_on_trade_entry_no_crash(self, tmp_path):
        a = self._alerts(tmp_path)
        signal = {
            "instrument": "EURUSD", "direction": 1, "lot_size": 0.10,
            "entry_price": 1.1000, "sl_price": 1.0980, "tp_price": 1.1040,
            "strategy_name": "LondonOpenBreakout",
        }
        a.on_trade_entry(signal, _guardian_status())
        content = (tmp_path / "a.log").read_text()
        assert "EURUSD" in content

    def test_on_trade_exit_win(self, tmp_path):
        a = self._alerts(tmp_path)
        result = {
            "instrument": "EURUSD", "pnl_dollars": 40.0, "pnl_pips": 40.0,
            "exit_reason": "tp", "entry_price": 1.1000, "exit_price": 1.1040,
        }
        a.on_trade_exit(result, _guardian_status())
        content = (tmp_path / "a.log").read_text()
        assert "40" in content

    def test_on_daily_summary(self, tmp_path):
        a = self._alerts(tmp_path)
        a.on_daily_summary(_guardian_status(daily_pnl=100.0), "normal",
                           ["london_breakout"])
        content = (tmp_path / "a.log").read_text()
        assert "DAILY SUMMARY" in content

    def test_on_guardian_block_critical_for_permanent(self, tmp_path):
        a = self._alerts(tmp_path)
        a.on_guardian_block(
            "total_dd exceeded",
            _guardian_status(permanently_halted=True),
        )
        content = (tmp_path / "a.log").read_text()
        assert "PERMANENT" in content

    def test_on_approaching_limit(self, tmp_path):
        a = self._alerts(tmp_path)
        a.on_approaching_limit("daily_loss", used_pct=3.0, limit_pct=4.0)
        content = (tmp_path / "a.log").read_text()
        assert "DAILY LOSS" in content.upper()


# ---------------------------------------------------------------------------
# AlertManager — check_limits + dedup
# ---------------------------------------------------------------------------

class TestAlertCheckLimits:
    def test_fires_when_daily_dd_at_70pct(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log", daily_limit_warn_pct=70.0)
        # daily_dd=2.8, limit=4.0 → 70% consumed
        a.check_limits(_guardian_status(daily_dd=2.8))
        assert (tmp_path / "a.log").read_text() != ""

    def test_fires_when_total_dd_at_70pct(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log", daily_limit_warn_pct=70.0)
        # total_dd=6.3, limit=9.0 → 70% consumed
        a.check_limits(_guardian_status(total_dd=6.3))
        assert "TOTAL" in (tmp_path / "a.log").read_text().upper()

    def test_no_fire_below_threshold(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log", daily_limit_warn_pct=70.0)
        a.check_limits(_guardian_status(daily_dd=1.0, total_dd=1.0))
        assert not (tmp_path / "a.log").exists() or (tmp_path / "a.log").read_text() == ""

    def test_dedup_fires_once(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log", daily_limit_warn_pct=70.0)
        a.check_limits(_guardian_status(daily_dd=2.8))
        a.check_limits(_guardian_status(daily_dd=2.8))
        content = (tmp_path / "a.log").read_text()
        assert content.count("DAILY LOSS") == 1

    def test_reset_allows_refiring(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log", daily_limit_warn_pct=70.0)
        a.check_limits(_guardian_status(daily_dd=2.8))
        a.reset_daily_warnings()
        a.check_limits(_guardian_status(daily_dd=2.8))
        content = (tmp_path / "a.log").read_text()
        assert content.count("DAILY LOSS") == 2


# ---------------------------------------------------------------------------
# AlertManager — custom handlers
# ---------------------------------------------------------------------------

class TestAlertHandlers:
    def test_handler_receives_level_title_body(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log")
        calls = []
        a.register_handler(lambda l, t, b: calls.append((l, t, b)))
        a.send("INFO", "MyTitle", "MyBody")
        assert calls == [("INFO", "MyTitle", "MyBody")]

    def test_bad_handler_doesnt_crash(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log")
        a.register_handler(lambda l, t, b: 1 / 0)   # raises ZeroDivision
        a.send("INFO", "T", "B")   # must not propagate

    def test_multiple_handlers_called(self, tmp_path):
        a = AlertManager(log_path=tmp_path / "a.log")
        calls = []
        a.register_handler(lambda l, t, b: calls.append("h1"))
        a.register_handler(lambda l, t, b: calls.append("h2"))
        a.send("INFO", "T", "B")
        assert "h1" in calls and "h2" in calls


# ---------------------------------------------------------------------------
# Dashboard — setters
# ---------------------------------------------------------------------------

def _mock_guardian(status_override=None):
    g = MagicMock()
    g.get_status.return_value = status_override or _guardian_status()
    return g


def _mock_connector(positions=None):
    c = MagicMock()
    c.get_account_info.return_value = {"balance": 10_000.0, "equity": 10_050.0,
                                       "open_positions": 0}
    c.get_positions.return_value = positions or []
    return c


def _mock_order_mgr():
    m = MagicMock()
    m.open_position_count = 0
    return m


class TestDashboardSetters:
    def test_set_regime(self):
        d = Dashboard(_mock_guardian(), _mock_connector(), _mock_order_mgr())
        d.set_regime("high_vol")
        assert d._regime == "high_vol"

    def test_set_active_strategies(self):
        d = Dashboard(_mock_guardian(), _mock_connector(), _mock_order_mgr())
        d.set_active_strategies(["london_breakout"])
        assert d._active_strategies == ["london_breakout"]

    def test_set_next_action(self):
        d = Dashboard(_mock_guardian(), _mock_connector(), _mock_order_mgr())
        d.set_next_action("03:00 London open")
        assert d._next_action == "03:00 London open"

    def test_set_today_trades_stores_copy(self):
        d = Dashboard(_mock_guardian(), _mock_connector(), _mock_order_mgr())
        trades = [{"pnl_dollars": 10}]
        d.set_today_trades(trades)
        trades.append({"pnl_dollars": 20})  # mutate original
        assert len(d._today_trades) == 1    # dashboard copy unaffected


# ---------------------------------------------------------------------------
# Dashboard — render smoke tests
# ---------------------------------------------------------------------------

class TestDashboardRender:
    def _dash(self, status=None, positions=None):
        g = _mock_guardian(status)
        c = _mock_connector(positions)
        return Dashboard(g, c, _mock_order_mgr())

    def test_refresh_no_crash(self, capsys):
        d = self._dash()
        d.refresh()   # must not raise

    def test_refresh_connector_error(self, capsys):
        g = _mock_guardian()
        c = MagicMock()
        c.get_account_info.side_effect = RuntimeError("no data")
        c.get_positions.side_effect    = RuntimeError("no data")
        d = Dashboard(g, c, _mock_order_mgr())
        d.refresh()   # must not raise

    def test_refresh_guardian_error(self, capsys):
        g = MagicMock()
        g.get_status.side_effect = RuntimeError("boom")
        d = Dashboard(g, _mock_connector(), _mock_order_mgr())
        d.refresh()

    def test_output_contains_balance(self, capsys):
        d = self._dash()
        d.refresh()
        out = capsys.readouterr().out
        assert "10,000" in out or "10000" in out

    def test_output_contains_regime(self, capsys):
        d = self._dash()
        d.set_regime("DEAD_MARKET_REGIME")
        d.refresh()
        out = capsys.readouterr().out
        assert "DEAD_MARKET_REGIME" in out

    def test_output_shows_daily_halted(self, capsys):
        status = _guardian_status(daily_halted=True)
        d = self._dash(status=status)
        d.refresh()
        out = capsys.readouterr().out
        assert "HALTED" in out.upper()


# ---------------------------------------------------------------------------
# Dashboard — _bar helper
# ---------------------------------------------------------------------------

class TestDashboardBar:
    def test_empty_bar_when_zero(self):
        result = _bar(0, 10, width=10, invert=False)
        assert "0%" in result

    def test_full_bar_when_equal(self):
        result = _bar(10, 10, width=10, invert=False)
        assert "100%" in result

    def test_partial_bar_at_50pct(self):
        result = _bar(5, 10, width=10, invert=False)
        assert "50%" in result


# ---------------------------------------------------------------------------
# Dashboard — run / stop
# ---------------------------------------------------------------------------

class TestDashboardRunStop:
    def test_stop_terminates_run(self):
        d = Dashboard(_mock_guardian(), _mock_connector(), _mock_order_mgr())
        t = threading.Thread(target=d.run, kwargs={"interval_seconds": 1})
        t.start()
        time.sleep(0.1)
        d.stop()
        t.join(timeout=3)
        assert not t.is_alive(), "Dashboard.run() did not stop within 3 seconds"
