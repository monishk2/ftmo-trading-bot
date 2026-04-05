"""
Tests for strategies/fvg_retracement.py

Test matrix
-----------
1.  Bullish FVG detected on exact 3-candle pattern
2.  Bearish FVG detected on exact 3-candle pattern
3.  No FVG when candles overlap (not a gap)
4.  No FVG when gap too small (< min_fvg_size_pips)
5.  No FVG when gap too large (> max_fvg_size_pips)
6.  Trend filter: bullish FVG rejected when close < EMA200
7.  Trend filter: bearish FVG rejected when close > EMA200
8.  Trend filter bypassed when EMA history is insufficient
9.  Limit order placed at midpoint (fvg_entry_level=0.5)
10. Signal fires on first bar where price retraces into zone
11. No signal when price never retraces (candle-based cancel fires)
12. No signal after cancel_unfilled_hour (time backstop)
13. Only 1 signal per day (most-recent FVG used)
14. No signal on Friday
15. FVG outside scan window is ignored
16. SL/TP geometry — bullish
17. SL/TP geometry — bearish
18. time_stop written at correct hour
19. Diagnostic columns written for all bars
20. DST: signal fires at 03:00 Eastern in both winter and summer
"""

import numpy as np
import pandas as pd
import pytest

from strategies.fvg_retracement import FVGRetracement

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

STRATEGY_CFG = {
    "fvg_scan_start":        "03:00",
    "fvg_scan_end":          "06:00",
    "entry_window_start":    "09:30",
    "entry_window_end":      "12:00",
    "cancel_unfilled_hour":  "12:00",
    "time_stop_hour":        "16:00",
    "min_fvg_size_pips":     5,
    "max_fvg_size_pips":     30,
    "fvg_entry_level":       0.5,
    "entry_buffer_pips":     3,
    "risk_reward_ratio":     2.0,
    "risk_per_trade_pct":    0.5,
    "max_candles_until_cancel": 8,
    "ema_period":            200,
    "no_friday_trading":     True,
}

INSTRUMENT_CFG = {
    "pip_size": 0.0001,
    "typical_spread_pips": 1.0,
    "commission_per_lot_round_trip": 3.0,
    "slippage_model": {
        "normal_pips": 0.3, "session_open_pips": 1.5,
        "session_open_window_minutes": 15, "news_pips": 2.0,
    },
    "trading_sessions": {},
}

PIP = 0.0001
EASTERN = "US/Eastern"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strat(ema_period: int = 200, max_candles: int = 8) -> FVGRetracement:
    cfg = {**STRATEGY_CFG, "ema_period": ema_period, "max_candles_until_cancel": max_candles}
    s = FVGRetracement()
    s.setup(cfg, INSTRUMENT_CFG)
    return s


def _ts(date: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{date} {hhmm}", tz=EASTERN)


def _bar(ts: pd.Timestamp, o: float, h: float, lo: float, c: float) -> dict:
    return {"ts": ts, "open": o, "high": h, "low": lo, "close": c, "volume": 0}


def _df(bars: list[dict]) -> pd.DataFrame:
    rows = [{"open": b["open"], "high": b["high"], "low": b["low"],
             "close": b["close"], "volume": b["volume"]} for b in bars]
    return pd.DataFrame(rows, index=pd.DatetimeIndex([b["ts"] for b in bars]))


def _make_bull_fvg_bars(
    date: str,
    scan_time: str = "03:15",   # candle[i] timestamp
    c0_high: float = 1.10100,   # candle[i-2] high
    c2_low:  float = 1.10200,   # candle[i].low  (gap above c0_high)
    close:   float = 1.10250,
) -> list[dict]:
    """
    Build 3 candles that form a bullish FVG:
      candle[i-2]: high = c0_high  (and low = c0_high - 10 pips)
      candle[i-1]: impulse body    (spans whole gap)
      candle[i]:   low = c2_low > c0_high

    scan_time is the timestamp assigned to candle[i].
    The two preceding bars get times scan_time-30m and scan_time-15m.
    """
    ts2 = pd.Timestamp(f"{date} {scan_time}", tz=EASTERN)
    ts1 = ts2 - pd.Timedelta(minutes=15)
    ts0 = ts2 - pd.Timedelta(minutes=30)

    return [
        _bar(ts0, c0_high - 0.0010, c0_high, c0_high - 0.0010, c0_high - 0.0005),
        _bar(ts1, c0_high, c2_low + 0.0020, c0_high - 0.0005, c2_low + 0.0015),
        _bar(ts2, c2_low + 0.0010, c2_low + 0.0030, c2_low, close),
    ]


def _make_bear_fvg_bars(
    date: str,
    scan_time: str = "03:15",
    c0_low:  float = 1.10200,   # candle[i-2].low
    c2_high: float = 1.10100,   # candle[i].high  (gap below c0_low)
    close:   float = 1.10050,
) -> list[dict]:
    """
    3-candle bearish FVG: candle[i-2].low > candle[i].high
    """
    ts2 = pd.Timestamp(f"{date} {scan_time}", tz=EASTERN)
    ts1 = ts2 - pd.Timedelta(minutes=15)
    ts0 = ts2 - pd.Timedelta(minutes=30)

    return [
        _bar(ts0, c0_low + 0.0005, c0_low + 0.0010, c0_low, c0_low + 0.0005),
        _bar(ts1, c0_low, c0_low + 0.0005, c2_high - 0.0020, c2_high - 0.0015),
        _bar(ts2, c2_high - 0.0010, c2_high, c2_high - 0.0020, close),
    ]


def _entry_bar(date: str, hhmm: str, low: float, high: float,
               close: float | None = None) -> dict:
    """A bar in the entry window that retraces into the FVG zone."""
    c = close if close is not None else (low + high) / 2
    return _bar(_ts(date, hhmm), low, high, low, c)


def _filler_bars(date: str, n: int, start_hhmm: str,
                 price: float = 1.10200) -> list[dict]:
    """N consecutive 15-min bars at a neutral price (outside any FVG zone)."""
    bars = []
    ts = pd.Timestamp(f"{date} {start_hhmm}", tz=EASTERN)
    for _ in range(n):
        bars.append(_bar(ts, price, price + 0.0002, price - 0.0002, price))
        ts += pd.Timedelta(minutes=15)
    return bars


def _enough_history_prefix(
    n: int = 820,
    price: float = 1.10000,
    start: str = "2022-09-01 00:00",
) -> list[dict]:
    """
    Generate `n` 15-min bars before the Jan 2023 test dates.

    EMA(200) on 1H requires 200 hourly bars = 800 × 15-min bars.
    Default n=820 gives a comfortable margin.
    Default start 2022-09-01 ensures 820 bars end well before 2022-12-31,
    leaving a gap before any 2023-01-03 FVG fixture bar.
    """
    ts = pd.Timestamp(start, tz=EASTERN)
    bars = []
    for _ in range(n):
        bars.append(_bar(ts, price, price, price, price))
        ts += pd.Timedelta(minutes=15)
    return bars


# ---------------------------------------------------------------------------
# 1. Bullish FVG detected on exact 3-candle pattern
# ---------------------------------------------------------------------------

class TestBullishFVGDetection:
    """
    c0.high=1.10100, c2.low=1.10200 → gap=10 pips, bullish
    With ema_period=3 (so EMA computes on tiny history) and close > ema.
    """

    def setup_method(self):
        history = _enough_history_prefix(n=205, price=1.09900)
        fvg_bars = _make_bull_fvg_bars("2023-01-03", scan_time="03:15",
                                       c0_high=1.10100, c2_low=1.10200, close=1.10250)
        # Entry bar in NY window: low touches the zone [1.10100, 1.10200]
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10110, high=1.10280)
        self.df = _strat(ema_period=3).generate_signals(_df(history + fvg_bars + [e_bar]))

    def test_signal_generated(self):
        sigs = self.df[self.df["signal"] != 0]
        assert len(sigs) == 1

    def test_signal_is_long(self):
        sigs = self.df[self.df["signal"] != 0]
        assert sigs.iloc[0]["signal"] == 1

    def test_fvg_zone_top(self):
        sigs = self.df[self.df["signal"] != 0]
        assert sigs.iloc[0]["fvg_zone_top"] == pytest.approx(1.10200, abs=1e-8)

    def test_fvg_zone_bottom(self):
        sigs = self.df[self.df["signal"] != 0]
        assert sigs.iloc[0]["fvg_zone_bottom"] == pytest.approx(1.10100, abs=1e-8)

    def test_fvg_size_pips(self):
        sigs = self.df[self.df["signal"] != 0]
        assert sigs.iloc[0]["fvg_size_pips"] == pytest.approx(10.0, abs=0.1)


# ---------------------------------------------------------------------------
# 2. Bearish FVG detected on exact 3-candle pattern
# ---------------------------------------------------------------------------

class TestBearishFVGDetection:
    """
    c0.low=1.10200, c2.high=1.10100 → gap=10 pips, bearish
    """

    def setup_method(self):
        history = _enough_history_prefix(n=205, price=1.10500)
        fvg_bars = _make_bear_fvg_bars("2023-01-03", scan_time="03:15",
                                       c0_low=1.10200, c2_high=1.10100, close=1.10050)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10050, high=1.10150)
        self.df = _strat(ema_period=3).generate_signals(_df(history + fvg_bars + [e_bar]))

    def test_signal_is_short(self):
        sigs = self.df[self.df["signal"] != 0]
        assert len(sigs) == 1
        assert sigs.iloc[0]["signal"] == -1

    def test_fvg_zone_top(self):
        sigs = self.df[self.df["signal"] != 0]
        assert sigs.iloc[0]["fvg_zone_top"] == pytest.approx(1.10200, abs=1e-8)

    def test_fvg_zone_bottom(self):
        sigs = self.df[self.df["signal"] != 0]
        assert sigs.iloc[0]["fvg_zone_bottom"] == pytest.approx(1.10100, abs=1e-8)


# ---------------------------------------------------------------------------
# 3. No FVG when candles overlap (not a gap)
# ---------------------------------------------------------------------------

class TestNoFVGWhenCandlesOverlap:
    """
    c0.high=1.10200, c2.low=1.10150 → c2.low < c0.high → overlap, NOT a gap.
    """

    def setup_method(self):
        date = "2023-01-03"
        ts2 = _ts(date, "03:15")
        ts1 = ts2 - pd.Timedelta(minutes=15)
        ts0 = ts2 - pd.Timedelta(minutes=30)
        bars = [
            _bar(ts0, 1.10100, 1.10200, 1.10080, 1.10190),   # high=1.10200
            _bar(ts1, 1.10200, 1.10350, 1.10180, 1.10300),
            _bar(ts2, 1.10250, 1.10380, 1.10150, 1.10350),   # low=1.10150 < c0.high
        ]
        e_bar = _entry_bar(date, "09:30", low=1.10200, high=1.10350)
        self.df = _strat(ema_period=3).generate_signals(_df(bars + [e_bar]))

    def test_no_signal(self):
        assert (self.df["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 4. No FVG when gap too small
# ---------------------------------------------------------------------------

class TestFVGTooSmall:
    """Gap = 3 pips < min_fvg_size_pips=5 → rejected."""

    def setup_method(self):
        # c0.high=1.10100, c2.low=1.10130 → gap=3 pips
        fvg_bars = _make_bull_fvg_bars("2023-01-03", scan_time="03:15",
                                       c0_high=1.10100, c2_low=1.10130, close=1.10160)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10105, high=1.10135)
        self.df = _strat(ema_period=3).generate_signals(_df(fvg_bars + [e_bar]))

    def test_no_signal(self):
        assert (self.df["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 5. No FVG when gap too large
# ---------------------------------------------------------------------------

class TestFVGTooLarge:
    """Gap = 35 pips > max_fvg_size_pips=30 → rejected."""

    def setup_method(self):
        # c0.high=1.10000, c2.low=1.10350 → gap=35 pips
        fvg_bars = _make_bull_fvg_bars("2023-01-03", scan_time="03:15",
                                       c0_high=1.10000, c2_low=1.10350, close=1.10380)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10050, high=1.10300)
        self.df = _strat(ema_period=3).generate_signals(_df(fvg_bars + [e_bar]))

    def test_no_signal(self):
        assert (self.df["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 6. Trend filter: bullish FVG rejected when close < EMA200
# ---------------------------------------------------------------------------

class TestBullishFVGRejectedByTrend:
    """
    Seed EMA with high prices so EMA200 >> close of FVG formation bar.
    Bullish FVG must be rejected.
    """

    def setup_method(self):
        # EMA history: 820 bars at 1.11000 → EMA ≈ 1.11000
        history = _enough_history_prefix(n=820, price=1.11000)
        # FVG formation: close=1.10250 < 1.11000 → trend filter rejects bullish
        fvg_bars = _make_bull_fvg_bars("2023-01-03", c0_high=1.10100,
                                       c2_low=1.10200, close=1.10250)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10110, high=1.10210)
        self.df = _strat(ema_period=200).generate_signals(_df(history + fvg_bars + [e_bar]))

    def test_no_signal(self):
        assert (self.df["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 7. Trend filter: bearish FVG rejected when close > EMA200
# ---------------------------------------------------------------------------

class TestBearishFVGRejectedByTrend:
    """
    EMA history at 1.09000, bearish FVG close=1.10050 > EMA → rejected.
    """

    def setup_method(self):
        history = _enough_history_prefix(n=820, price=1.09000)
        fvg_bars = _make_bear_fvg_bars("2023-01-03", c0_low=1.10200,
                                       c2_high=1.10100, close=1.10050)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10050, high=1.10150)
        self.df = _strat(ema_period=200).generate_signals(_df(history + fvg_bars + [e_bar]))

    def test_no_signal(self):
        assert (self.df["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 8. Trend filter bypassed when EMA history insufficient
# ---------------------------------------------------------------------------

class TestTrendFilterBypassedOnInsufficientHistory:
    """
    Only 5 bars of history → ema_period=200 cannot compute → filter skipped →
    valid FVG still produces a signal.
    """

    def setup_method(self):
        fvg_bars = _make_bull_fvg_bars("2023-01-03", c0_high=1.10100,
                                       c2_low=1.10200, close=1.10250)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10110, high=1.10230)
        self.df = _strat(ema_period=200).generate_signals(_df(fvg_bars + [e_bar]))

    def test_signal_still_fires(self):
        """Filter bypassed → signal should be generated."""
        assert (self.df["signal"] != 0).sum() == 1


# ---------------------------------------------------------------------------
# 9. Limit order entry at midpoint of FVG zone
# ---------------------------------------------------------------------------

class TestEntryAtMidpoint:
    """
    zone=[1.10100, 1.10200], entry_level=0.5 → entry=1.10150
    SL = zone_bottom - 3 pips = 1.10070
    TP = entry + |entry - SL| × 2 = 1.10150 + 0.00080 × 2 = 1.10310

    The Backtester uses sl_price and tp_price directly, so we only check
    those (spread/slippage are applied by the Backtester on top).
    """

    def setup_method(self):
        fvg_bars = _make_bull_fvg_bars("2023-01-03", c0_high=1.10100,
                                       c2_low=1.10200, close=1.10250)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10110, high=1.10230)
        self.row = _strat(ema_period=3).generate_signals(
            _df(fvg_bars + [e_bar])
        ).loc[_ts("2023-01-03", "09:30")]

    def test_sl_price(self):
        # zone_bottom=1.10100, buffer=3 pips
        assert self.row["sl_price"] == pytest.approx(1.10100 - 3 * PIP, abs=1e-8)

    def test_tp_price(self):
        entry     = 1.10150             # midpoint of zone
        sl        = 1.10100 - 3 * PIP  # 1.10070
        sl_dist   = abs(entry - sl)
        expected_tp = entry + sl_dist * 2.0
        assert self.row["tp_price"] == pytest.approx(expected_tp, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. Signal fires on first retrace bar (not before)
# ---------------------------------------------------------------------------

class TestSignalOnFirstRetraceBar:
    """
    3 bars in the entry window: first two don't touch zone, third does.
    Signal must appear only on the third.
    """

    def setup_method(self):
        # zone=[1.10100, 1.10200]
        fvg_bars = _make_bull_fvg_bars("2023-01-03", c0_high=1.10100,
                                       c2_low=1.10200, close=1.10250)
        # Two bars above zone
        miss1 = _entry_bar("2023-01-03", "09:30", low=1.10220, high=1.10350)
        miss2 = _entry_bar("2023-01-03", "09:45", low=1.10210, high=1.10330)
        # Third bar: low=1.10150 → enters zone
        hit   = _entry_bar("2023-01-03", "10:00", low=1.10150, high=1.10300)
        self.df = _strat(ema_period=3).generate_signals(
            _df(fvg_bars + [miss1, miss2, hit])
        )

    def test_only_one_signal(self):
        assert (self.df["signal"] != 0).sum() == 1

    def test_signal_on_third_bar(self):
        sig_ts = self.df[self.df["signal"] != 0].index[0]
        assert sig_ts == _ts("2023-01-03", "10:00")

    def test_no_signal_on_miss_bars(self):
        for hhmm in ["09:30", "09:45"]:
            assert self.df.at[_ts("2023-01-03", hhmm), "signal"] == 0


# ---------------------------------------------------------------------------
# 11. Candle-based cancel: no signal when price never retraces in time
# ---------------------------------------------------------------------------

class TestCandleBasedCancel:
    """
    FVG forms at 03:15.  max_candles=3.
    Entry window: 09:30–12:00.
    By the time the entry window starts, 25+ candles have passed → cancelled.
    """

    def setup_method(self):
        # FVG at 03:15 — many bars later the entry window opens
        fvg_bars = _make_bull_fvg_bars("2023-01-03", scan_time="03:15",
                                       c0_high=1.10100, c2_low=1.10200)
        # Neutral bars from 03:30 through 09:30 (lots of them, > 3)
        fillers = _filler_bars("2023-01-03", n=26, start_hhmm="03:30", price=1.10300)
        e_bar   = _entry_bar("2023-01-03", "10:00", low=1.10110, high=1.10210)
        self.df = _strat(ema_period=3, max_candles=3).generate_signals(
            _df(fvg_bars + fillers + [e_bar])
        )

    def test_no_signal_after_candle_cancel(self):
        assert (self.df["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 12. Hour-based cancel backstop
# ---------------------------------------------------------------------------

class TestHourBackstopCancel:
    """
    FVG forms in scan window.  cancel_unfilled_hour=12:00.
    Retrace bar is at 12:00 → already at or past backstop → no signal.

    Use max_candles=9999 to disable candle-based cancel.
    """

    def setup_method(self):
        cfg = {**STRATEGY_CFG, "ema_period": 3, "max_candles_until_cancel": 9999,
               "cancel_unfilled_hour": "12:00", "entry_window_end": "13:00"}
        s = FVGRetracement()
        s.setup(cfg, INSTRUMENT_CFG)

        fvg_bars = _make_bull_fvg_bars("2023-01-03", c0_high=1.10100, c2_low=1.10200)
        # Retrace bar exactly at cancel hour
        late_bar = _entry_bar("2023-01-03", "12:00", low=1.10110, high=1.10210)
        self.df = s.generate_signals(_df(fvg_bars + [late_bar]))

    def test_no_signal_at_cancel_hour(self):
        assert (self.df["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 13. Max 1 trade/day — most recent FVG wins
# ---------------------------------------------------------------------------

class TestMostRecentFVGUsed:
    """
    Two bullish FVGs on the same day — most recent wins.

      FVG-1 forms at 03:15: zone=[1.10100, 1.10200]
        candles: 02:45, 03:00, 03:15

      FVG-2 forms at 05:15: zone=[1.10300, 1.10420]
        candles: 04:45, 05:00, 05:15
        (Chosen so no timestamps overlap with FVG-1 or the fillers between them.)

      Fillers between FVGs:  03:30–04:30  (5 bars)
      Fillers before NY:     05:30–09:15  (16 bars, ends exactly at 09:15)
      Entry bar:             09:30        (unique, touches FVG-2's zone)

    Signal must use FVG-2 (most recent) → zone_bottom=1.10300, zone_top=1.10420.
    """

    def setup_method(self):
        # FVG-1: timestamps 02:45, 03:00, 03:15
        fvg1_bars = _make_bull_fvg_bars("2023-01-03", scan_time="03:15",
                                        c0_high=1.10100, c2_low=1.10200, close=1.10250)

        # Fillers 03:30–04:30 (5 bars) — no overlap with fvg1 or fvg2
        fillers1 = _filler_bars("2023-01-03", n=5, start_hhmm="03:30", price=1.10280)

        # FVG-2: timestamps 04:45, 05:00, 05:15  (no overlap with fillers1)
        fvg2_bars = _make_bull_fvg_bars("2023-01-03", scan_time="05:15",
                                        c0_high=1.10300, c2_low=1.10420, close=1.10460)

        # Fillers 05:30–09:15 (16 bars = 4h, last bar at 09:15)
        fillers2 = _filler_bars("2023-01-03", n=16, start_hhmm="05:30", price=1.10500)

        # Entry bar 09:30 — unique, touches FVG-2 zone [1.10300, 1.10420]
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10310, high=1.10440)

        # max_candles=9999 disables candle-based cancel so we can isolate
        # the "most recent FVG wins" selection logic independently.
        self.df = _strat(ema_period=3, max_candles=9999).generate_signals(
            _df(fvg1_bars + fillers1 + fvg2_bars + fillers2 + [e_bar])
        )

    def test_exactly_one_signal(self):
        assert (self.df["signal"] != 0).sum() == 1

    def test_signal_uses_fvg2_zone(self):
        row = self.df[self.df["signal"] != 0].iloc[0]
        # FVG-2: zone_bottom = c0.high = 1.10300, zone_top = c2.low = 1.10420
        assert row["fvg_zone_bottom"] == pytest.approx(1.10300, abs=1e-8)
        assert row["fvg_zone_top"]    == pytest.approx(1.10420, abs=1e-8)


# ---------------------------------------------------------------------------
# 14. No signal on Friday
# ---------------------------------------------------------------------------

class TestNoFridaySignal:
    """2023-01-06 is a Friday."""

    def setup_method(self):
        fvg_bars = _make_bull_fvg_bars("2023-01-06", c0_high=1.10100, c2_low=1.10200)
        e_bar    = _entry_bar("2023-01-06", "09:30", low=1.10110, high=1.10210)
        self.df  = _strat(ema_period=3).generate_signals(_df(fvg_bars + [e_bar]))

    def test_no_friday_signal(self):
        assert (self.df["signal"] != 0).sum() == 0

    def test_weekday_is_friday(self):
        assert self.df.index[0].weekday() == 4


# ---------------------------------------------------------------------------
# 15. FVG formed outside scan window is ignored
# ---------------------------------------------------------------------------

class TestFVGOutsideScanWindow:
    """FVG formation bar at 07:00 is outside fvg_scan_end=06:00."""

    def setup_method(self):
        fvg_bars = _make_bull_fvg_bars("2023-01-03", scan_time="07:00",
                                       c0_high=1.10100, c2_low=1.10200)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10110, high=1.10210)
        self.df = _strat(ema_period=3).generate_signals(_df(fvg_bars + [e_bar]))

    def test_no_signal(self):
        assert (self.df["signal"] != 0).sum() == 0


# ---------------------------------------------------------------------------
# 16. SL/TP geometry — bullish
# ---------------------------------------------------------------------------

class TestBullishGeometry:
    """
    zone=[1.10100, 1.10200], entry_level=0.5, buffer=3 pips, rr=2.0

    entry   = 1.10100 + 0.5 × 0.00100 = 1.10150
    sl      = 1.10100 - 3 × 0.0001    = 1.10070
    sl_dist = 1.10150 - 1.10070       = 0.00080
    tp      = 1.10150 + 0.00080 × 2   = 1.10310
    """

    def setup_method(self):
        fvg_bars = _make_bull_fvg_bars("2023-01-03", c0_high=1.10100,
                                       c2_low=1.10200, close=1.10250)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10120, high=1.10240)
        row = _strat(ema_period=3).generate_signals(_df(fvg_bars + [e_bar]))
        self.row = row[row["signal"] != 0].iloc[0]

    def test_sl(self):
        assert self.row["sl_price"] == pytest.approx(1.10070, abs=1e-8)

    def test_tp(self):
        assert self.row["tp_price"] == pytest.approx(1.10310, abs=1e-6)


# ---------------------------------------------------------------------------
# 17. SL/TP geometry — bearish
# ---------------------------------------------------------------------------

class TestBearishGeometry:
    """
    zone=[1.10100, 1.10200], entry_level=0.5, buffer=3 pips, rr=2.0

    entry   = 1.10100 + 0.5 × 0.00100 = 1.10150
    sl      = 1.10200 + 3 × 0.0001    = 1.10230
    sl_dist = 1.10230 - 1.10150       = 0.00080
    tp      = 1.10150 - 0.00080 × 2   = 1.09990
    """

    def setup_method(self):
        fvg_bars = _make_bear_fvg_bars("2023-01-03", c0_low=1.10200,
                                       c2_high=1.10100, close=1.10050)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10050, high=1.10180)
        # EMA must be below close for bearish — seed with low prices
        history = _enough_history_prefix(n=205, price=1.09000)
        row = _strat(ema_period=200).generate_signals(_df(history + fvg_bars + [e_bar]))
        self.row = row[row["signal"] != 0].iloc[0]

    def test_sl(self):
        assert self.row["sl_price"] == pytest.approx(1.10230, abs=1e-8)

    def test_tp(self):
        assert self.row["tp_price"] == pytest.approx(1.09990, abs=1e-6)


# ---------------------------------------------------------------------------
# 18. time_stop written at correct hour
# ---------------------------------------------------------------------------

class TestTimeStopColumn:
    """time_stop on signal bar = 16:00 Eastern of the same day."""

    def setup_method(self):
        fvg_bars = _make_bull_fvg_bars("2023-01-03", c0_high=1.10100, c2_low=1.10200)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10110, high=1.10230)
        df = _strat(ema_period=3).generate_signals(_df(fvg_bars + [e_bar]))
        self.sig_row = df[df["signal"] != 0].iloc[0]
        self.full_df = df

    def test_time_stop_hour(self):
        ts = self.sig_row["time_stop"]
        assert isinstance(ts, pd.Timestamp)
        assert ts.hour == 16
        assert ts.minute == 0

    def test_time_stop_tz_aware(self):
        assert self.sig_row["time_stop"].tzinfo is not None

    def test_non_signal_bars_have_no_time_stop(self):
        non_sig = self.full_df[self.full_df["signal"] == 0]
        assert all(v is None for v in non_sig["time_stop"])


# ---------------------------------------------------------------------------
# 19. Diagnostic columns written
# ---------------------------------------------------------------------------

class TestDiagnosticColumns:
    """All FVG diagnostic columns present and non-NaN on signal day."""

    def setup_method(self):
        fvg_bars = _make_bull_fvg_bars("2023-01-03", c0_high=1.10100, c2_low=1.10200)
        e_bar = _entry_bar("2023-01-03", "09:30", low=1.10110, high=1.10230)
        self.df = _strat(ema_period=3).generate_signals(_df(fvg_bars + [e_bar]))

    def test_fvg_zone_top_written(self):
        # At least some bars must have a zone_top (from the day the FVG was detected)
        day_bars = self.df[self.df.index.date == pd.Timestamp("2023-01-03").date()]
        assert not day_bars["fvg_zone_top"].isna().all()

    def test_fvg_zone_bottom_written(self):
        day_bars = self.df[self.df.index.date == pd.Timestamp("2023-01-03").date()]
        assert not day_bars["fvg_zone_bottom"].isna().all()

    def test_fvg_direction_written(self):
        day_bars = self.df[self.df.index.date == pd.Timestamp("2023-01-03").date()]
        assert (day_bars["fvg_direction"] != 0).any()

    def test_fvg_size_pips_written(self):
        day_bars = self.df[self.df.index.date == pd.Timestamp("2023-01-03").date()]
        assert not day_bars["fvg_size_pips"].isna().all()


# ---------------------------------------------------------------------------
# 20. DST correctness — 03:00 ET in both winter (EST) and summer (EDT)
# ---------------------------------------------------------------------------

class TestDSTCorrectness:
    """FVG scan window at 03:15 Eastern fires correctly in winter and summer."""

    def _run(self, date: str) -> pd.DataFrame:
        fvg_bars = _make_bull_fvg_bars(date, scan_time="03:15",
                                       c0_high=1.10100, c2_low=1.10200, close=1.10250)
        e_bar = _entry_bar(date, "09:30", low=1.10110, high=1.10230)
        return _strat(ema_period=3).generate_signals(_df(fvg_bars + [e_bar]))

    def test_winter_fvg_detected(self):
        df = self._run("2023-01-03")   # EST
        assert (df["signal"] != 0).sum() == 1

    def test_summer_fvg_detected(self):
        df = self._run("2023-06-06")   # EDT
        assert (df["signal"] != 0).sum() == 1

    def test_winter_scan_bar_is_at_03_ET(self):
        df = self._run("2023-01-03")
        sig_ts = df[df["signal"] != 0].index[0]
        assert sig_ts.hour == 9  # signal fires at 09:30
        # Verify FVG direction written (FVG at 03:15 was scanned correctly)
        assert df.at[sig_ts, "fvg_direction"] == 1

    def test_summer_scan_bar_is_at_03_ET(self):
        df = self._run("2023-06-06")
        sig_ts = df[df["signal"] != 0].index[0]
        assert sig_ts.hour == 9
