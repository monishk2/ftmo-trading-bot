"""
Tests for backtesting/backtester.py

Test scenarios
--------------
1.  Known long trade → TP hit  : exact P&L (no spread, no slippage, no commission)
2.  Known short trade → TP hit : direction-correct P&L
3.  Spread applied correctly   : entry_price = close + spread/2 × pip_size (long)
4.  Commission deducted        : pnl_dollars = gross - commission_per_lot × lots
5.  SL exit                    : trade exits at stop-loss price, negative P&L
6.  Time-stop exit             : position force-closed at specified timestamp
7.  FTMO daily loss → halts    : no new trades accepted after daily trigger fires
8.  FTMO total drawdown → halt : backtest terminates, halt_reason is set
9.  Auto lot sizing            : lot derived from risk_per_trade_pct and SL distance
10. Session-open slippage      : entry at 03:00 ET uses session_open_pips budget
"""

import numpy as np
import pandas as pd
import pytest

from backtesting.backtester import Backtester, PIP_VALUE_PER_LOT, Trade


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

EASTERN = "US/Eastern"
PIP = 0.0001  # EURUSD pip size


def _ts(date_str: str) -> pd.Timestamp:
    """Create a US/Eastern timestamp from 'YYYY-MM-DD HH:MM' string."""
    return pd.Timestamp(date_str, tz=EASTERN)


def _make_df(rows: list[tuple]) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame with a US/Eastern DatetimeIndex.

    Each row is: (ts_string, open, high, low, close)
    volume defaults to 0.
    """
    index = pd.DatetimeIndex([_ts(r[0]) for r in rows], name="datetime")
    data  = [
        {"open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": 0}
        for r in rows
    ]
    return pd.DataFrame(data, index=index)


def _clean_instrument(
    spread_pips: float = 0.0,
    commission: float = 0.0,
    normal_slip: float = 0.0,
    session_slip: float = 0.0,
    window_minutes: int = 15,
) -> dict:
    """Instrument config with fully controllable cost parameters."""
    return {
        "pip_size": PIP,
        "typical_spread_pips": spread_pips,
        "commission_per_lot_round_trip": commission,
        "slippage_model": {
            "normal_pips": normal_slip,
            "session_open_pips": session_slip,
            "session_open_window_minutes": window_minutes,
            "news_pips": 2.0,
        },
        "trading_sessions": {},
    }


def _clean_ftmo(
    daily_trigger: float = 4.0,
    total_trigger: float = 9.0,
) -> dict:
    return {
        "challenge": {
            "profit_target_pct": 10.0,
            "max_daily_loss_pct": 5.0,
            "max_total_loss_pct": 10.0,
        },
        "safety_buffers": {
            "daily_loss_trigger_pct": daily_trigger,
            "total_loss_trigger_pct": total_trigger,
        },
    }


class MockStrategy:
    """
    Strategy stub that injects pre-defined signals at specified bar positions
    (by integer location, i.e. ``iloc``).
    """

    def __init__(self, name: str = "mock", risk_per_trade_pct: float = 1.0):
        self.name = name
        self.risk_per_trade_pct = risk_per_trade_pct
        self._signals: list[dict] = []

    def add_signal(
        self,
        iloc: int,
        direction: int,
        sl_price: float,
        tp_price: float,
        lot_size: float = 1.0,
        time_stop: pd.Timestamp | None = None,
    ) -> "MockStrategy":
        self._signals.append(dict(
            iloc=iloc, direction=direction,
            sl_price=sl_price, tp_price=tp_price,
            lot_size=lot_size, time_stop=time_stop,
        ))
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"]    = 0
        df["sl_price"]  = np.nan
        df["tp_price"]  = np.nan
        df["lot_size"]  = np.nan
        # Use object dtype so tz-aware Timestamps can be stored without dtype conflict
        df["time_stop"] = None

        for sig in self._signals:
            i = sig["iloc"]
            df.iloc[i, df.columns.get_loc("signal")]   = sig["direction"]
            df.iloc[i, df.columns.get_loc("sl_price")] = sig["sl_price"]
            df.iloc[i, df.columns.get_loc("tp_price")] = sig["tp_price"]
            df.iloc[i, df.columns.get_loc("lot_size")] = sig["lot_size"]
            if sig["time_stop"] is not None:
                df.iloc[i, df.columns.get_loc("time_stop")] = sig["time_stop"]
        return df


# ---------------------------------------------------------------------------
# 1. Known long trade — TP hit, zero costs
# ---------------------------------------------------------------------------

class TestKnownLongTradePnL:
    """
    Bar 0: long signal, close=1.10000, sl=1.09900 (10 pips), tp=1.10200 (20 pips)
    Bar 1: high=1.10250 → TP hit at 1.10200

    Expected (no spread / slippage / commission):
      entry_price   = 1.10000
      exit_price    = 1.10200
      pnl_pips      = (1.10200 - 1.10000) / 0.0001 = 20
      pnl_dollars   = 20 × 10 × 1.0 = 200
    """

    def setup_method(self):
        df = _make_df([
            ("2023-01-02 04:00", 1.10000, 1.10050, 1.09950, 1.10000),  # bar 0: signal
            ("2023-01-02 04:15", 1.10000, 1.10250, 1.09950, 1.10100),  # bar 1: TP hit
        ])
        strat = MockStrategy().add_signal(
            iloc=0, direction=1, sl_price=1.09900, tp_price=1.10200, lot_size=1.0
        )
        self.result = Backtester(
            strategy=strat,
            df=df,
            instrument="EURUSD",
            initial_balance=10_000.0,
            _override_instrument_config=_clean_instrument(),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()

    def test_one_trade_produced(self):
        assert len(self.result.trades) == 1

    def test_exit_reason_tp(self):
        assert self.result.trades[0].exit_reason == "tp"

    def test_entry_price(self):
        assert self.result.trades[0].entry_price == pytest.approx(1.10000, abs=1e-8)

    def test_exit_price(self):
        assert self.result.trades[0].exit_price == pytest.approx(1.10200, abs=1e-8)

    def test_pnl_pips(self):
        assert self.result.trades[0].pnl_pips == pytest.approx(20.0, abs=0.01)

    def test_pnl_dollars(self):
        assert self.result.trades[0].pnl_dollars == pytest.approx(200.0, abs=0.01)

    def test_final_balance(self):
        assert self.result.final_balance == pytest.approx(10_200.0, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Known short trade — TP hit, zero costs
# ---------------------------------------------------------------------------

class TestKnownShortTradePnL:
    """
    Bar 0: short signal, close=1.10000, sl=1.10100 (10 pips), tp=1.09800 (20 pips)
    Bar 1: low=1.09750 → TP hit at 1.09800

    Expected:
      entry_price = 1.10000
      pnl_pips    = -1 × (1.09800 - 1.10000) / 0.0001 = 20
      pnl_dollars = 200
    """

    def setup_method(self):
        df = _make_df([
            ("2023-01-02 04:00", 1.10000, 1.10050, 1.09950, 1.10000),
            ("2023-01-02 04:15", 1.09900, 1.10020, 1.09750, 1.09850),
        ])
        strat = MockStrategy().add_signal(
            iloc=0, direction=-1, sl_price=1.10100, tp_price=1.09800, lot_size=1.0
        )
        self.result = Backtester(
            strategy=strat, df=df, instrument="EURUSD",
            _override_instrument_config=_clean_instrument(),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()

    def test_direction(self):
        assert self.result.trades[0].direction == -1

    def test_exit_reason_tp(self):
        assert self.result.trades[0].exit_reason == "tp"

    def test_pnl_pips(self):
        assert self.result.trades[0].pnl_pips == pytest.approx(20.0, abs=0.01)

    def test_pnl_dollars(self):
        assert self.result.trades[0].pnl_dollars == pytest.approx(200.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Spread applied correctly on entry
# ---------------------------------------------------------------------------

class TestSpreadApplied:
    """
    spread_pips=1.0 → spread_adj = 0.5 × 0.0001 = 0.00005
    Long entry at close=1.10000 → entry_price = 1.10005 (+ 0 slippage)
    Short entry at close=1.10000 → entry_price = 1.09995
    """

    def _run(self, direction: int) -> Trade:
        # Bar 1 high/low chosen so TP is hit for both directions
        if direction == 1:
            df = _make_df([
                ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),
                ("2023-01-02 04:15", 1.10, 1.11, 1.09, 1.10500),
            ])
            sl, tp = 1.09900, 1.10500
        else:
            df = _make_df([
                ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),
                ("2023-01-02 04:15", 1.09, 1.10, 1.09, 1.09500),
            ])
            sl, tp = 1.10100, 1.09500

        strat = MockStrategy().add_signal(
            iloc=0, direction=direction, sl_price=sl, tp_price=tp, lot_size=1.0
        )
        result = Backtester(
            strategy=strat, df=df, instrument="EURUSD",
            _override_instrument_config=_clean_instrument(spread_pips=1.0),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()
        return result.trades[0]

    def test_long_entry_price_includes_spread(self):
        trade = self._run(1)
        expected = 1.10000 + (1.0 / 2) * PIP
        assert trade.entry_price == pytest.approx(expected, abs=1e-8)

    def test_short_entry_price_includes_spread(self):
        trade = self._run(-1)
        expected = 1.10000 - (1.0 / 2) * PIP
        assert trade.entry_price == pytest.approx(expected, abs=1e-8)

    def test_spread_reduces_long_pnl(self):
        """Long entry above close → fewer pips than zero-cost case."""
        zero_cost = Backtester(
            strategy=MockStrategy().add_signal(
                iloc=0, direction=1, sl_price=1.09900, tp_price=1.10500, lot_size=1.0
            ),
            df=_make_df([
                ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),
                ("2023-01-02 04:15", 1.10, 1.11, 1.09, 1.10500),
            ]),
            instrument="EURUSD",
            _override_instrument_config=_clean_instrument(spread_pips=0.0),
            _override_ftmo_rules=_clean_ftmo(),
        ).run().trades[0]

        with_spread = self._run(1)
        assert with_spread.pnl_dollars < zero_cost.pnl_dollars


# ---------------------------------------------------------------------------
# 4. Commission deducted from P&L
# ---------------------------------------------------------------------------

class TestCommissionDeducted:
    """
    commission_per_lot = 3.0, lot_size = 1.0
    Long TP at +20 pips → gross = 200, pnl_dollars = 200 − 3 = 197
    """

    def setup_method(self):
        df = _make_df([
            ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),
            ("2023-01-02 04:15", 1.10, 1.10300, 1.09950, 1.10200),
        ])
        strat = MockStrategy().add_signal(
            iloc=0, direction=1, sl_price=1.09900, tp_price=1.10200, lot_size=1.0
        )
        self.result = Backtester(
            strategy=strat, df=df, instrument="EURUSD",
            _override_instrument_config=_clean_instrument(commission=3.0),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()

    def test_commission_reduces_pnl(self):
        assert self.result.trades[0].pnl_dollars == pytest.approx(197.0, abs=0.01)

    def test_final_balance_reflects_commission(self):
        assert self.result.final_balance == pytest.approx(10_197.0, abs=0.01)


# ---------------------------------------------------------------------------
# 5. Stop-loss exit — trade closes at SL, negative P&L
# ---------------------------------------------------------------------------

class TestStopLossExit:
    """
    Long: entry=1.10000, sl=1.09900
    Bar 1 low=1.09850 → SL hit at 1.09900
    pnl_pips = -10, pnl_dollars = -100
    """

    def setup_method(self):
        df = _make_df([
            ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),
            ("2023-01-02 04:15", 1.10, 1.10050, 1.09850, 1.09920),
        ])
        strat = MockStrategy().add_signal(
            iloc=0, direction=1, sl_price=1.09900, tp_price=1.11000, lot_size=1.0
        )
        self.result = Backtester(
            strategy=strat, df=df, instrument="EURUSD",
            _override_instrument_config=_clean_instrument(),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()

    def test_exit_reason_sl(self):
        assert self.result.trades[0].exit_reason == "sl"

    def test_exit_price_at_sl(self):
        assert self.result.trades[0].exit_price == pytest.approx(1.09900, abs=1e-8)

    def test_pnl_pips_negative(self):
        assert self.result.trades[0].pnl_pips == pytest.approx(-10.0, abs=0.01)

    def test_pnl_dollars_negative(self):
        assert self.result.trades[0].pnl_dollars == pytest.approx(-100.0, abs=0.01)

    def test_balance_reduced(self):
        assert self.result.final_balance == pytest.approx(9_900.0, abs=0.01)


# ---------------------------------------------------------------------------
# 6. Time-stop exit
# ---------------------------------------------------------------------------

class TestTimeStopExit:
    """
    Bar 0: long signal with time_stop = bar 3 timestamp, sl=1.09500, tp=1.11000
    Bar 1, 2: price stays between SL and TP
    Bar 3: time_stop fires → exit at close=1.10100

    pnl_pips = (1.10100 - 1.10000) / 0.0001 = 10
    pnl_dollars = 100
    """

    def setup_method(self):
        ts3 = _ts("2023-01-02 04:45")
        df = _make_df([
            ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),   # 0 signal
            ("2023-01-02 04:15", 1.10, 1.10300, 1.09800, 1.10150),  # 1 no exit
            ("2023-01-02 04:30", 1.10, 1.10200, 1.09700, 1.10050),  # 2 no exit
            ("2023-01-02 04:45", 1.10, 1.10150, 1.09600, 1.10100),  # 3 time stop
        ])
        strat = MockStrategy().add_signal(
            iloc=0, direction=1,
            sl_price=1.09500, tp_price=1.11000,
            lot_size=1.0, time_stop=ts3,
        )
        self.result = Backtester(
            strategy=strat, df=df, instrument="EURUSD",
            _override_instrument_config=_clean_instrument(),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()

    def test_one_trade(self):
        assert len(self.result.trades) == 1

    def test_exit_reason_time_stop(self):
        assert self.result.trades[0].exit_reason == "time_stop"

    def test_exit_price_at_bar3_close(self):
        assert self.result.trades[0].exit_price == pytest.approx(1.10100, abs=1e-8)

    def test_pnl_dollars(self):
        assert self.result.trades[0].pnl_dollars == pytest.approx(100.0, abs=0.01)

    def test_exit_time_matches_bar3(self):
        assert self.result.trades[0].exit_time == _ts("2023-01-02 04:45")


# ---------------------------------------------------------------------------
# 7. FTMO daily loss → blocks further trading that day
# ---------------------------------------------------------------------------

class TestFtmoDailyLossBlocking:
    """
    Setup
    -----
    Balance = 10,000  |  daily_trigger = 4%  →  threshold equity = 9,600

    Bar 0 (04:00 ET, day 1): long signal, sl=1.09500, tp=1.11000, lots=1
    Bar 1 (04:15 ET, day 1): close=1.09590
      floating = (1.09590-1.10000)/0.0001 × $10 × 1 = -$410
      equity   = $9,590 < $9,600 → daily trigger fires
      position closed at 1.09590, balance = $9,590
    Bar 2 (04:30 ET, day 1): another signal → BLOCKED (daily_blocked=True)
    Bar 3 (04:00 ET, day 2): signal → should be accepted (new day resets block)
    """

    def setup_method(self):
        # Day 2 starts at midnight → 00:00 ET Jan 3.  Use 04:00 to keep
        # trades inside a realistic session window.
        df = _make_df([
            ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),      # 0  long signal
            ("2023-01-02 04:15", 1.10, 1.10, 1.09, 1.09590),      # 1  triggers daily loss
            ("2023-01-02 04:30", 1.10, 1.10, 1.09, 1.09700),      # 2  blocked signal
            ("2023-01-03 04:00", 1.10, 1.11, 1.10, 1.10000),      # 3  new day signal
            ("2023-01-03 04:15", 1.10, 1.11, 1.10, 1.10200),      # 4  TP hit
        ])
        strat = (
            MockStrategy()
            .add_signal(iloc=0, direction=1, sl_price=1.09500, tp_price=1.11000, lot_size=1.0)
            .add_signal(iloc=2, direction=1, sl_price=1.09500, tp_price=1.11000, lot_size=1.0)
            .add_signal(iloc=3, direction=1, sl_price=1.09500, tp_price=1.10200, lot_size=1.0)
        )
        self.result = Backtester(
            strategy=strat, df=df, instrument="EURUSD",
            initial_balance=10_000.0,
            _override_instrument_config=_clean_instrument(),
            _override_ftmo_rules=_clean_ftmo(daily_trigger=4.0),
        ).run()

    def test_bar2_signal_skipped(self):
        """Bar 2 signal must be blocked — only 2 trades total (bar0 + bar3)."""
        assert len(self.result.trades) == 2

    def test_first_trade_exit_reason(self):
        assert self.result.trades[0].exit_reason == "ftmo_daily"

    def test_first_trade_exit_price(self):
        """Position closed at bar-1 close when daily trigger fires."""
        assert self.result.trades[0].exit_price == pytest.approx(1.09590, abs=1e-8)

    def test_second_trade_accepted_next_day(self):
        """Bar 3 (new day) should produce a trade."""
        assert self.result.trades[1].exit_reason == "tp"

    def test_no_halt(self):
        """Daily loss should NOT halt the whole backtest."""
        assert self.result.ftmo_halt_reason is None


# ---------------------------------------------------------------------------
# 8. FTMO total drawdown → terminates backtest
# ---------------------------------------------------------------------------

class TestFtmoTotalDrawdownHalt:
    """
    initial = 10,000  |  total_trigger = 9%  →  halt when equity < $9,100
                       |  daily_trigger = 50% →  effectively disabled

    Day 1
        Bar 0 (04:00): long, SL=1.09500, TP=1.11000, lots=1
        Bar 1 (04:15): low=1.09500 → SL hit → pnl = -$500, balance = $9,500

    Day 2 (balance = $9,500, midnight_balance = $9,500)
        Bar 2 (04:00): long, SL=1.09000, TP=1.11000, lots=1, entry=1.10000
        Bar 3 (04:15): close=1.09580
          floating = (1.09580-1.10000)/0.0001 × 10 × 1 = -$420
          equity   = 9,500 - 420 = $9,080
          daily_dd = (9,080 - 9,500) / 9,500 = -4.4%  → below 50% trigger — no fire
          total_dd = (9,080 - 10,000) / 10,000 = -9.2% → exceeds 9% → halt!
    """

    def setup_method(self):
        df = _make_df([
            ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),    # 0 signal day 1
            ("2023-01-02 04:15", 1.09, 1.10, 1.09, 1.09500),    # 1 SL hit → -$500
            ("2023-01-03 04:00", 1.10, 1.10, 1.10, 1.10000),    # 2 signal day 2
            ("2023-01-03 04:15", 1.09, 1.10, 1.09, 1.09580),    # 3 total dd halt
        ])
        strat = (
            MockStrategy()
            .add_signal(iloc=0, direction=1, sl_price=1.09500, tp_price=1.11000, lot_size=1.0)
            .add_signal(iloc=2, direction=1, sl_price=1.09000, tp_price=1.11000, lot_size=1.0)
        )
        self.result = Backtester(
            strategy=strat, df=df, instrument="EURUSD",
            initial_balance=10_000.0,
            _override_instrument_config=_clean_instrument(),
            _override_ftmo_rules=_clean_ftmo(daily_trigger=50.0, total_trigger=9.0),
        ).run()

    def test_halt_reason_set(self):
        assert self.result.ftmo_halt_reason is not None
        assert "Total drawdown" in self.result.ftmo_halt_reason

    def test_second_trade_exit_reason_ftmo_total(self):
        assert len(self.result.trades) == 2
        assert self.result.trades[1].exit_reason == "ftmo_total"

    def test_equity_curve_ends_early(self):
        """Equity curve should stop at bar 3, not run to the end of the DataFrame."""
        assert len(self.result.equity_curve) == 4  # bars 0-3

    def test_backtest_halted_below_initial_minus_9pct(self):
        """Final equity must be below the 9% total-loss trigger."""
        total_dd = (self.result.final_balance - 10_000.0) / 10_000.0 * 100
        assert total_dd < -9.0


# ---------------------------------------------------------------------------
# 9. Auto lot sizing from risk_per_trade_pct
# ---------------------------------------------------------------------------

class TestAutoLotSizing:
    """
    risk_per_trade_pct=1.0, balance=10,000 → risk=$100
    SL distance = 10 pips → lot = 100 / (10 × 10) = 1.0

    No explicit lot_size provided → backtester computes it.
    """

    def setup_method(self):
        df = _make_df([
            ("2023-01-02 04:00", 1.10, 1.10, 1.10, 1.10000),
            ("2023-01-02 04:15", 1.10, 1.10250, 1.09950, 1.10200),
        ])

        class AutoStrat:
            name = "auto"
            risk_per_trade_pct = 1.0

            def generate_signals(self, df):
                df["signal"]    = 0
                df["sl_price"]  = np.nan
                df["tp_price"]  = np.nan
                df["lot_size"]  = np.nan  # intentionally omit
                df["time_stop"] = pd.NaT
                df.iloc[0, df.columns.get_loc("signal")]   = 1
                df.iloc[0, df.columns.get_loc("sl_price")] = 1.09900   # 10 pips
                df.iloc[0, df.columns.get_loc("tp_price")] = 1.10200
                return df

        self.result = Backtester(
            strategy=AutoStrat(), df=df, instrument="EURUSD",
            initial_balance=10_000.0,
            _override_instrument_config=_clean_instrument(),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()

    def test_lot_size_computed(self):
        # risk=100, sl=10 pips → lot = 100/(10×10) = 1.0
        assert self.result.trades[0].lot_size == pytest.approx(1.0, abs=0.01)

    def test_trade_executed(self):
        assert len(self.result.trades) == 1


# ---------------------------------------------------------------------------
# 10. Session-open slippage budget is higher near 03:00 ET
# ---------------------------------------------------------------------------

class TestSessionOpenSlippage:
    """
    normal_pips=0.0, session_open_pips=1.5, window=15 min
    Entry at 03:00 ET (inside window) → slippage drawn from [0, 1.5] pips.
    Entry at 10:00 ET (outside window) → slippage drawn from [0, 0.0] pips = 0.

    With seed=42, the 03:00 draw should be > 0.
    """

    def _run_at_time(self, time_str: str) -> float:
        """Return entry_price for a long at the given Eastern time."""
        df = _make_df([
            (f"2023-01-03 {time_str}", 1.10, 1.11, 1.09, 1.10000),
            (f"2023-01-03 {time_str[:5]}:15" if ":" in time_str else time_str,
             1.10, 1.11, 1.09, 1.11000),
        ])
        # Rebuild to guarantee two distinct rows
        ts0 = _ts(f"2023-01-03 {time_str}")
        ts1 = ts0 + pd.Timedelta(minutes=15)
        df = pd.DataFrame(
            [
                {"open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10000, "volume": 0},
                {"open": 1.10, "high": 1.11, "low": 1.09, "close": 1.11000, "volume": 0},
            ],
            index=pd.DatetimeIndex([ts0, ts1]),
        )
        strat = MockStrategy().add_signal(
            iloc=0, direction=1, sl_price=1.09000, tp_price=1.12000, lot_size=1.0
        )
        result = Backtester(
            strategy=strat, df=df, instrument="EURUSD", seed=42,
            _override_instrument_config=_clean_instrument(
                spread_pips=0.0, normal_slip=0.0, session_slip=1.5, window_minutes=15
            ),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()
        return result.trades[0].entry_price

    def test_session_open_slippage_nonzero(self):
        """At 03:00 ET, session_open_pips budget → entry > close."""
        entry = self._run_at_time("03:00")
        # seed=42, uniform(0, 1.5 × 0.0001) > 0 with overwhelming probability
        assert entry > 1.10000

    def test_outside_window_no_slippage(self):
        """At 10:00 ET, normal_pips=0 → no slippage, entry == close."""
        entry = self._run_at_time("10:00")
        assert entry == pytest.approx(1.10000, abs=1e-8)


# ---------------------------------------------------------------------------
# Metrics smoke test (requires a result with trades)
# ---------------------------------------------------------------------------

class TestMetricsSmoke:
    """Ensure calculate_metrics runs without error and returns expected keys."""

    def setup_method(self):
        # 4 winning trades over two days to have enough data for all metrics
        rows = []
        for day in ["2023-01-02", "2023-01-03"]:
            for hh in ["04:00", "06:00"]:
                rows.append((f"{day} {hh}", 1.10, 1.10, 1.10, 1.10000))
                rows.append((f"{day} {hh[:5]}:15"
                              if hh != "06:00" else f"{day} 06:15",
                              1.10, 1.10250, 1.09950, 1.10200))
        # Re-build with unique timestamps
        timestamps = pd.date_range("2023-01-02 04:00", periods=8, freq="15min", tz=EASTERN)
        closes     = [1.10000, 1.10200, 1.10000, 1.10200,
                      1.10000, 1.10200, 1.10000, 1.10200]
        df = pd.DataFrame({
            "open":   closes,
            "high":   [c + 0.0025 for c in closes],
            "low":    [c - 0.0005 for c in closes],
            "close":  closes,
            "volume": [0] * 8,
        }, index=timestamps)

        strat = MockStrategy()
        for i in [0, 2, 4, 6]:
            strat.add_signal(iloc=i, direction=1,
                             sl_price=1.09900, tp_price=1.10200, lot_size=1.0)

        self.result = Backtester(
            strategy=strat, df=df, instrument="EURUSD",
            initial_balance=10_000.0,
            _override_instrument_config=_clean_instrument(),
            _override_ftmo_rules=_clean_ftmo(),
        ).run()

    def test_metrics_run(self):
        from backtesting.metrics import calculate_metrics
        m = calculate_metrics(self.result)
        assert "error" not in m

    def test_required_keys_present(self):
        from backtesting.metrics import calculate_metrics
        m = calculate_metrics(self.result)
        required = [
            "total_return_pct", "cagr_pct", "win_rate_pct",
            "profit_factor", "max_drawdown_pct", "sharpe_ratio",
            "sortino_ratio", "monthly_returns", "ftmo_pass_rate_pct",
            "max_consecutive_wins", "max_consecutive_losses",
        ]
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_all_trades_won(self):
        from backtesting.metrics import calculate_metrics
        m = calculate_metrics(self.result)
        assert m["win_rate_pct"] == pytest.approx(100.0)

    def test_profit_factor_infinite_on_all_wins(self):
        from backtesting.metrics import calculate_metrics
        m = calculate_metrics(self.result)
        assert m["profit_factor"] == float("inf")

    def test_positive_total_return(self):
        from backtesting.metrics import calculate_metrics
        m = calculate_metrics(self.result)
        assert m["total_return_pct"] > 0
