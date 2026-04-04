"""
Tests for strategies/london_open_breakout.py

All synthetic DataFrames use 15-min bars with a US/Eastern tz-aware index,
exactly matching what the Backtester delivers to generate_signals().

Test matrix
-----------
1.  Asian range correctly computed from 00:00–02:45 ET bars
2.  Long signal when close > asian_high + buffer (entry window)
3.  Short signal when close < asian_low  - buffer (entry window)
4.  NO signal when range < min_asian_range_pips (too tight)
5.  NO signal when range > max_asian_range_pips (too wide)
6.  NO signal on Friday
7.  Only 1 signal per day (max_trades_per_day = 1)
8.  NO signal when breakout bar is outside entry window
9.  SL and TP prices are geometrically correct (long)
10. SL and TP prices are geometrically correct (short)
11. time_stop column populated only on signal bar
12. Diagnostic columns (asian_high, asian_low, asian_range_pips) always written
13. DST correctness — 03:00 Eastern is correct in both EST and EDT
14. No signal when close is inside range (no breakout)
"""

import numpy as np
import pandas as pd
import pytest

from strategies.london_open_breakout import LondonOpenBreakout

# ---------------------------------------------------------------------------
# Config fixtures (mirror real strategy_params.json / instruments.json values)
# ---------------------------------------------------------------------------

STRATEGY_CFG = {
    "asian_range_start":   "00:00",
    "asian_range_end":     "02:45",
    "entry_window_start":  "03:00",
    "entry_window_end":    "05:30",
    "time_stop_hour":      "12:00",
    "min_asian_range_pips": 20,
    "max_asian_range_pips": 40,
    "entry_buffer_pips":    2,
    "risk_reward_ratio":    2.0,
    "risk_per_trade_pct":   0.75,
    "no_friday_trading":    True,
}

INSTRUMENT_CFG = {
    "pip_size": 0.0001,
    "typical_spread_pips": 1.0,
    "commission_per_lot_round_trip": 3.0,
    "slippage_model": {
        "normal_pips": 0.3,
        "session_open_pips": 1.5,
        "session_open_window_minutes": 15,
        "news_pips": 2.0,
    },
    "trading_sessions": {},
}

PIP = 0.0001
EASTERN = "US/Eastern"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strat() -> LondonOpenBreakout:
    s = LondonOpenBreakout()
    s.setup(STRATEGY_CFG, INSTRUMENT_CFG)
    return s


def _bar(date: str, hhmm: str, o: float, h: float, lo: float, c: float) -> dict:
    return {"ts": pd.Timestamp(f"{date} {hhmm}", tz=EASTERN),
            "open": o, "high": h, "low": lo, "close": c, "volume": 0}


def _df(bars: list[dict]) -> pd.DataFrame:
    rows = [{"open": b["open"], "high": b["high"], "low": b["low"],
             "close": b["close"], "volume": b["volume"]} for b in bars]
    idx  = pd.DatetimeIndex([b["ts"] for b in bars])
    return pd.DataFrame(rows, index=idx)


def _standard_asian_bars(date: str, asian_high: float, asian_low: float) -> list[dict]:
    """
    Produce a block of Asian-session 15-min bars (00:00–02:45 ET) that span
    exactly [asian_low, asian_high].  Range = asian_high − asian_low.
    """
    times = ["00:00", "00:15", "00:30", "00:45",
             "01:00", "01:15", "01:30", "01:45",
             "02:00", "02:15", "02:30", "02:45"]
    mid   = (asian_high + asian_low) / 2
    bars  = []
    for t in times:
        bars.append(_bar(date, t, mid, asian_high, asian_low, mid))
    return bars


# ---------------------------------------------------------------------------
# 1. Asian range correctly computed
# ---------------------------------------------------------------------------

class TestAsianRangeComputation:
    """Diagnostic columns must reflect bar high/low across 00:00–02:45 ET."""

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        # One entry-window bar (no breakout — close stays inside range)
        entry_bar = _bar("2023-01-03", "03:00", 1.10100, 1.10200, 1.10050, 1.10150)
        self.result = _strat().generate_signals(_df(asian_bars + [entry_bar]))

    def test_asian_high_correct(self):
        assert self.result["asian_high"].iloc[0] == pytest.approx(1.10300, abs=1e-8)

    def test_asian_low_correct(self):
        assert self.result["asian_low"].iloc[0] == pytest.approx(1.10000, abs=1e-8)

    def test_asian_range_pips(self):
        # 1.10300 - 1.10000 = 0.00300 → 30 pips
        assert self.result["asian_range_pips"].iloc[0] == pytest.approx(30.0, abs=0.1)

    def test_all_bars_share_same_diagnostic(self):
        assert self.result["asian_high"].nunique() == 1


# ---------------------------------------------------------------------------
# 2. Long signal generated correctly
# ---------------------------------------------------------------------------

class TestLongSignal:
    """
    Asian range: 1.10000–1.10300 = 30 pips  (within 20–40 filter)
    buffer = 2 pips → long_trigger = 1.10300 + 0.0002 = 1.10320
    Entry bar close = 1.10330 > 1.10320 → LONG signal
    """

    def setup_method(self):
        asian_bars  = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        # bar at 03:00 breaks above trigger
        entry_bar   = _bar("2023-01-03", "03:00", 1.10300, 1.10340, 1.10290, 1.10330)
        self.df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))
        self.sig_row = self.df_out[self.df_out["signal"] != 0].iloc[0]

    def test_signal_is_long(self):
        assert self.sig_row["signal"] == 1

    def test_exactly_one_signal(self):
        assert (self.df_out["signal"] != 0).sum() == 1

    def test_signal_on_entry_bar(self):
        assert self.sig_row.name.hour == 3
        assert self.sig_row.name.minute == 0

    def test_sl_price_is_asian_low_minus_buffer(self):
        expected_sl = 1.10000 - 2 * PIP  # 1.09980
        assert self.sig_row["sl_price"] == pytest.approx(expected_sl, abs=1e-8)

    def test_tp_is_2r(self):
        sl_dist = abs(self.sig_row["sl_price"] - 1.10330)
        expected_tp = 1.10330 + sl_dist * 2.0
        assert self.sig_row["tp_price"] == pytest.approx(expected_tp, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Short signal generated correctly
# ---------------------------------------------------------------------------

class TestShortSignal:
    """
    Asian range: 1.10000–1.10300 = 30 pips
    buffer = 2 pips → short_trigger = 1.10000 - 0.0002 = 1.09980
    Entry bar close = 1.09970 < 1.09980 → SHORT signal
    """

    def setup_method(self):
        asian_bars  = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        entry_bar   = _bar("2023-01-03", "03:00", 1.10000, 1.10020, 1.09960, 1.09970)
        self.df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))
        self.sig_row = self.df_out[self.df_out["signal"] != 0].iloc[0]

    def test_signal_is_short(self):
        assert self.sig_row["signal"] == -1

    def test_sl_price_is_asian_high_plus_buffer(self):
        expected_sl = 1.10300 + 2 * PIP  # 1.10320
        assert self.sig_row["sl_price"] == pytest.approx(expected_sl, abs=1e-8)

    def test_tp_is_below_entry_by_2r(self):
        sl_dist = abs(self.sig_row["sl_price"] - 1.09970)
        expected_tp = 1.09970 - sl_dist * 2.0
        assert self.sig_row["tp_price"] == pytest.approx(expected_tp, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. No signal when range too tight (< 20 pips)
# ---------------------------------------------------------------------------

class TestRangeTooTight:
    """
    Asian high=1.10150, low=1.10000 → range=15 pips < 20 → skip day
    Even if price breaks out, no signal must appear.
    """

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10150, 1.10000)
        # Close well above trigger
        entry_bar  = _bar("2023-01-03", "03:00", 1.10200, 1.10300, 1.10150, 1.10260)
        self.df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))

    def test_no_signal(self):
        assert (self.df_out["signal"] != 0).sum() == 0

    def test_diagnostics_still_written(self):
        """Asian range columns should still be populated for analysis."""
        assert not self.df_out["asian_high"].isna().all()
        assert self.df_out["asian_range_pips"].iloc[0] == pytest.approx(15.0, abs=0.1)


# ---------------------------------------------------------------------------
# 5. No signal when range too wide (> 40 pips)
# ---------------------------------------------------------------------------

class TestRangeTooWide:
    """
    Asian high=1.10500, low=1.10000 → range=50 pips > 40 → skip day
    """

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10500, 1.10000)
        entry_bar  = _bar("2023-01-03", "03:00", 1.10500, 1.10600, 1.10450, 1.10560)
        self.df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))

    def test_no_signal(self):
        assert (self.df_out["signal"] != 0).sum() == 0

    def test_range_pips_recorded(self):
        assert self.df_out["asian_range_pips"].iloc[0] == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# 6. No signal on Friday
# ---------------------------------------------------------------------------

class TestNoFridayTrading:
    """
    2023-01-06 is a Friday.  Strategy must skip entirely.
    """

    def setup_method(self):
        date = "2023-01-06"   # Friday
        asian_bars = _standard_asian_bars(date, 1.10300, 1.10000)
        entry_bar  = _bar(date, "03:00", 1.10300, 1.10400, 1.10280, 1.10380)
        self.df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))

    def test_no_signal_on_friday(self):
        assert (self.df_out["signal"] != 0).sum() == 0

    def test_weekday_is_friday(self):
        assert self.df_out.index[0].weekday() == 4  # sanity check


# ---------------------------------------------------------------------------
# 7. Max one signal per day
# ---------------------------------------------------------------------------

class TestMaxOneSignalPerDay:
    """
    Two breakout bars back-to-back in the entry window — only first fires.
    """

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        # Both bars close above trigger
        bar1 = _bar("2023-01-03", "03:00", 1.10300, 1.10400, 1.10290, 1.10350)
        bar2 = _bar("2023-01-03", "03:15", 1.10350, 1.10500, 1.10330, 1.10420)
        self.df_out = _strat().generate_signals(_df(asian_bars + [bar1, bar2]))

    def test_only_one_signal(self):
        assert (self.df_out["signal"] != 0).sum() == 1

    def test_first_bar_has_signal(self):
        sig_idx = self.df_out[self.df_out["signal"] != 0].index[0]
        assert sig_idx.hour == 3 and sig_idx.minute == 0

    def test_second_bar_no_signal(self):
        sig_idx = self.df_out[self.df_out["signal"] != 0].index[0]
        # Find the bar at 03:15
        bar2_idx = self.df_out.index[
            (self.df_out.index.hour == 3) & (self.df_out.index.minute == 15)
        ]
        assert len(bar2_idx) == 1
        assert self.df_out.at[bar2_idx[0], "signal"] == 0


# ---------------------------------------------------------------------------
# 8. No signal when breakout bar is outside entry window
# ---------------------------------------------------------------------------

class TestNoSignalOutsideEntryWindow:
    """
    Breakout candle at 06:00 ET — outside entry_window_end of 05:30 → skip.
    """

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        # Bar at 05:45 — one 15-min candle past the 05:30 cutoff
        late_bar = _bar("2023-01-03", "05:45", 1.10300, 1.10500, 1.10280, 1.10450)
        self.df_out = _strat().generate_signals(_df(asian_bars + [late_bar]))

    def test_no_signal_outside_window(self):
        assert (self.df_out["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 9. SL/TP geometry — long
# ---------------------------------------------------------------------------

class TestLongGeometry:
    """
    Precise arithmetic check on SL and TP for a long trade.

    asian_high = 1.10300, asian_low = 1.10000
    buffer     = 2 pips → 0.0002
    long_trigger = 1.10320
    entry close  = 1.10350

    sl_price = asian_low - buffer = 1.09980
    sl_dist  = |1.10350 - 1.09980| = 0.00370  (37 pips)
    tp_price = 1.10350 + 0.00370 × 2 = 1.11090
    """

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        entry_bar  = _bar("2023-01-03", "03:00", 1.10300, 1.10400, 1.10290, 1.10350)
        df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))
        self.row = df_out[df_out["signal"] != 0].iloc[0]

    def test_sl_price(self):
        assert self.row["sl_price"] == pytest.approx(1.09980, abs=1e-8)

    def test_tp_price(self):
        sl_dist = abs(1.10350 - 1.09980)
        assert self.row["tp_price"] == pytest.approx(1.10350 + sl_dist * 2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. SL/TP geometry — short
# ---------------------------------------------------------------------------

class TestShortGeometry:
    """
    asian_high=1.10300, asian_low=1.10000, buffer=2 pips
    short_trigger = 1.09980
    entry close   = 1.09960

    sl_price = 1.10300 + 0.0002 = 1.10320
    sl_dist  = |1.09960 - 1.10320| = 0.00360
    tp_price = 1.09960 - 0.00360 × 2 = 1.09240
    """

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        entry_bar  = _bar("2023-01-03", "03:00", 1.10000, 1.10020, 1.09950, 1.09960)
        df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))
        self.row = df_out[df_out["signal"] != 0].iloc[0]

    def test_sl_price(self):
        assert self.row["sl_price"] == pytest.approx(1.10320, abs=1e-8)

    def test_tp_price(self):
        sl_dist = abs(1.09960 - 1.10320)
        assert self.row["tp_price"] == pytest.approx(1.09960 - sl_dist * 2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 11. time_stop column populated on signal bar only
# ---------------------------------------------------------------------------

class TestTimeStop:
    """
    time_stop on the signal bar must be 12:00 ET of the same trading day.
    All other bars must have time_stop == None.
    """

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        entry_bar  = _bar("2023-01-03", "03:00", 1.10300, 1.10380, 1.10290, 1.10360)
        non_signal = _bar("2023-01-03", "04:00", 1.10300, 1.10380, 1.10290, 1.10200)
        self.df_out = _strat().generate_signals(_df(asian_bars + [entry_bar, non_signal]))

    def test_time_stop_on_signal_bar(self):
        sig_idx = self.df_out[self.df_out["signal"] != 0].index[0]
        ts = self.df_out.at[sig_idx, "time_stop"]
        assert ts is not None
        assert isinstance(ts, pd.Timestamp)
        assert ts.hour == 12
        assert ts.minute == 0
        assert ts.tzinfo is not None

    def test_time_stop_none_on_non_signal_bars(self):
        non_sig = self.df_out[self.df_out["signal"] == 0]
        # Every non-signal bar must have time_stop = None
        assert all(v is None for v in non_sig["time_stop"])


# ---------------------------------------------------------------------------
# 12. Diagnostic columns always populated when Asian bars exist
# ---------------------------------------------------------------------------

class TestDiagnosticColumns:
    """asian_high / asian_low / asian_range_pips written even when no trade taken."""

    def setup_method(self):
        # Tight range (no trade) — diagnostics must still be written
        asian_bars = _standard_asian_bars("2023-01-03", 1.10100, 1.10000)  # 10 pips
        entry_bar  = _bar("2023-01-03", "03:00", 1.10100, 1.10200, 1.10080, 1.10180)
        self.df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))

    def test_asian_high_written(self):
        assert not self.df_out["asian_high"].isna().all()

    def test_asian_low_written(self):
        assert not self.df_out["asian_low"].isna().all()

    def test_asian_range_pips_written(self):
        assert not self.df_out["asian_range_pips"].isna().all()

    def test_range_value_correct(self):
        assert self.df_out["asian_range_pips"].iloc[0] == pytest.approx(10.0, abs=0.1)


# ---------------------------------------------------------------------------
# 13. DST correctness — 03:00 ET works in both winter (EST) and summer (EDT)
# ---------------------------------------------------------------------------

class TestDSTCorrectness:
    """
    Winter date (EST = UTC-5): 2023-01-03 03:00 ET = 08:00 UTC
    Summer date (EDT = UTC-4): 2023-06-06 03:00 ET = 07:00 UTC

    In both cases the strategy must see the bar as 03:00 Eastern and fire
    a signal (assuming a valid breakout).  The UTC offset must not shift
    the effective session boundaries.
    """

    def _run(self, date: str) -> pd.DataFrame:
        asian_bars = _standard_asian_bars(date, 1.10300, 1.10000)
        entry_bar  = _bar(date, "03:00", 1.10300, 1.10400, 1.10280, 1.10360)
        return _strat().generate_signals(_df(asian_bars + [entry_bar]))

    def test_winter_signal_at_03_ET(self):
        df = self._run("2023-01-03")   # EST — UTC offset -05:00
        signals = df[df["signal"] != 0]
        assert len(signals) == 1
        assert signals.index[0].hour == 3

    def test_summer_signal_at_03_ET(self):
        df = self._run("2023-06-06")   # EDT — UTC offset -04:00
        signals = df[df["signal"] != 0]
        assert len(signals) == 1
        assert signals.index[0].hour == 3

    def test_winter_utc_offset_is_minus5(self):
        """Sanity-check: winter timestamp is UTC-5."""
        ts = pd.Timestamp("2023-01-03 03:00", tz="US/Eastern")
        assert ts.utcoffset().total_seconds() == -5 * 3600

    def test_summer_utc_offset_is_minus4(self):
        """Sanity-check: summer timestamp is UTC-4."""
        ts = pd.Timestamp("2023-06-06 03:00", tz="US/Eastern")
        assert ts.utcoffset().total_seconds() == -4 * 3600


# ---------------------------------------------------------------------------
# 14. No signal when close is inside range (no breakout confirmation)
# ---------------------------------------------------------------------------

class TestNoSignalInsideRange:
    """
    Bar closes at 1.10250 — inside Asian range 1.10000–1.10300.
    Must NOT trigger a signal even if it's in the entry window.
    """

    def setup_method(self):
        asian_bars = _standard_asian_bars("2023-01-03", 1.10300, 1.10000)
        # Close is 1.10250 — inside the range, does not breach trigger of 1.10320
        entry_bar  = _bar("2023-01-03", "03:00", 1.10200, 1.10350, 1.10180, 1.10250)
        self.df_out = _strat().generate_signals(_df(asian_bars + [entry_bar]))

    def test_no_signal(self):
        assert (self.df_out["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 15. Multi-day: each day evaluated independently
# ---------------------------------------------------------------------------

class TestMultiDayIndependence:
    """
    Day 1 (Mon): range OK, breakout → signal
    Day 2 (Tue): range too tight → no signal
    Day 3 (Wed): range OK, breakout → signal

    Exactly 2 signals across 3 days.
    """

    def setup_method(self):
        bars = []
        # Monday 2023-01-02: 30-pip range, breakout
        bars += _standard_asian_bars("2023-01-02", 1.10300, 1.10000)
        bars.append(_bar("2023-01-02", "03:00", 1.10300, 1.10400, 1.10280, 1.10360))
        # Tuesday 2023-01-03: 10-pip range (tight) — no signal
        bars += _standard_asian_bars("2023-01-03", 1.10100, 1.10000)
        bars.append(_bar("2023-01-03", "03:00", 1.10100, 1.10200, 1.10080, 1.10180))
        # Wednesday 2023-01-04: 25-pip range, breakout
        bars += _standard_asian_bars("2023-01-04", 1.10250, 1.10000)
        bars.append(_bar("2023-01-04", "03:00", 1.10250, 1.10320, 1.10230, 1.10295))

        self.df_out = _strat().generate_signals(_df(bars))

    def test_exactly_two_signals(self):
        assert (self.df_out["signal"] != 0).sum() == 2

    def test_no_signal_on_tuesday(self):
        tuesday = self.df_out[self.df_out.index.date == pd.Timestamp("2023-01-03").date()]
        assert (tuesday["signal"] != 0).sum() == 0

    def test_signal_on_monday(self):
        monday = self.df_out[self.df_out.index.date == pd.Timestamp("2023-01-02").date()]
        assert (monday["signal"] != 0).sum() == 1

    def test_signal_on_wednesday(self):
        wednesday = self.df_out[self.df_out.index.date == pd.Timestamp("2023-01-04").date()]
        assert (wednesday["signal"] != 0).sum() == 1
